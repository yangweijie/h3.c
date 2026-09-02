#include "h3_weights.h"

#include <dirent.h>
#include <fcntl.h>
#include <stdarg.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

struct h3_weight_store {
    h3_st_header *headers;
    size_t count;
};

static void fail(char *error, size_t error_size, const char *format, ...) {
    if (!error || !error_size) return;
    va_list arguments;
    va_start(arguments, format);
    vsnprintf(error, error_size, format, arguments);
    va_end(arguments);
}

static int safetensors_name(const char *name) {
    static const char suffix[] = ".safetensors";
    size_t length = strlen(name);
    return length > sizeof(suffix) - 1 &&
           strcmp(name + length - (sizeof(suffix) - 1), suffix) == 0;
}

static int compare_paths(const void *left, const void *right) {
    const char *const *a = left;
    const char *const *b = right;
    return strcmp(*a, *b);
}

static void free_paths(char **paths, size_t count) {
    if (!paths) return;
    for (size_t index = 0; index < count; index++) free(paths[index]);
    free(paths);
}

h3_weight_store *h3_weight_store_open(const char *directory,
                                      char *error, size_t error_size) {
    if (!directory || !*directory) {
        fail(error, error_size, "weight directory is required");
        return NULL;
    }
    DIR *stream = opendir(directory);
    if (!stream) {
        fail(error, error_size, "cannot open weight directory: %s", directory);
        return NULL;
    }
    char **paths = NULL;
    size_t count = 0;
    size_t capacity = 0;
    struct dirent *entry;
    while ((entry = readdir(stream)) != NULL) {
        if (!safetensors_name(entry->d_name)) continue;
        if (count == capacity) {
            size_t next = capacity ? capacity * 2 : 8;
            char **grown = realloc(paths, next * sizeof(*grown));
            if (!grown) {
                fail(error, error_size, "out of memory listing weight shards");
                closedir(stream);
                free_paths(paths, count);
                return NULL;
            }
            paths = grown;
            capacity = next;
        }
        size_t length = strlen(directory) + strlen(entry->d_name) + 2;
        paths[count] = malloc(length);
        if (!paths[count]) {
            fail(error, error_size, "out of memory resolving a weight shard");
            closedir(stream);
            free_paths(paths, count);
            return NULL;
        }
        snprintf(paths[count], length, "%s/%s", directory, entry->d_name);
        count++;
    }
    closedir(stream);
    if (!count) {
        fail(error, error_size, "no safetensors shards in %s", directory);
        free(paths);
        return NULL;
    }
    qsort(paths, count, sizeof(*paths), compare_paths);
    h3_weight_store *store = calloc(1, sizeof(*store));
    if (!store) {
        fail(error, error_size, "out of memory creating weight store");
        free_paths(paths, count);
        return NULL;
    }
    store->headers = calloc(count, sizeof(*store->headers));
    if (!store->headers) {
        fail(error, error_size, "out of memory allocating weight headers");
        free(store);
        free_paths(paths, count);
        return NULL;
    }
    store->count = count;
    for (size_t index = 0; index < count; index++) {
        char detail[384];
        if (!h3_st_read_header(paths[index], &store->headers[index], detail,
                               sizeof(detail))) {
            fail(error, error_size, "%s", detail);
            free_paths(paths, count);
            h3_weight_store_free(store);
            return NULL;
        }
    }
    free_paths(paths, count);
    return store;
}

void h3_weight_store_free(h3_weight_store *store) {
    if (!store) return;
    for (size_t index = 0; index < store->count; index++) {
        h3_st_free_header(&store->headers[index]);
    }
    free(store->headers);
    free(store);
}

size_t h3_weight_store_shards(const h3_weight_store *store) {
    return store ? store->count : 0;
}

const h3_st_tensor *h3_weight_find(const h3_weight_store *store,
                                   const char *name,
                                   const h3_st_header **header) {
    if (header) *header = NULL;
    if (!store || !name) return NULL;
    for (size_t index = 0; index < store->count; index++) {
        const h3_st_tensor *tensor = h3_st_find(&store->headers[index], name);
        if (tensor) {
            if (header) *header = &store->headers[index];
            return tensor;
        }
    }
    return NULL;
}

/* --- P13: INT8 (ConvRot) -> BF16/F32 dequantisation --------------------- */
/* Quantized MiniMax checkpoints store the big matmuls as I8 plus a
 * per-output-channel F32 scale `{name}_scale` of shape [rows, 1], and the
 * weights are stored pre-rotated by the ConvRot transform.  Callers that ask
 * for BF16/F32 (video VAE, ClipProj text encoder) previously just failed the
 * dtype check; we now dequantise on the CPU and undo the rotation so those
 * callers can consume the quantized files directly.
 *   dequant: w = i8 * scale[row]
 *   unrotate: radix-4 butterfly over the input dim in blocks of 256, x1/16
 * The butterfly mirrors h3_dit.c convrot_unrotate_cpu (stages 1,4,16,64).
 * H3_INT8_UNROTATE=0 skips the butterfly for plain (non-ConvRot) INT8 files. */
#define H3_INT8_BLOCK 256

static int int8_unrotate_enabled(void) {
    const char *flag = getenv("H3_INT8_UNROTATE");
    return !(flag && *flag && strcmp(flag, "0") == 0);
}

static void convrot_unrotate_row(float *values) {
    for (int stride = 1; stride < H3_INT8_BLOCK; stride *= 4) {
        int span = stride * 4;
        for (int base = 0; base < H3_INT8_BLOCK; base += span)
            for (int lane = 0; lane < stride; lane++) {
                int i0 = base + lane, i1 = i0 + stride,
                    i2 = i0 + 2 * stride, i3 = i0 + 3 * stride;
                float a = values[i0], b = values[i1],
                      c = values[i2], d = values[i3];
                values[i0] = a + b + c - d;
                values[i1] = a + b - c + d;
                values[i2] = a - b + c + d;
                values[i3] = -a + b + c + d;
            }
    }
    for (int k = 0; k < H3_INT8_BLOCK; k++)
        values[k] *= (1.0f / 16.0f);
}

static uint16_t f32_to_bf16_u16(float value) {
    uint32_t bits;
    memcpy(&bits, &value, sizeof(bits));
    bits += ((bits >> 16) & 1u) + 0x7fffu; /* round to nearest even */
    return (uint16_t)(bits >> 16);
}

static int pread_bytes(const char *path, uint64_t offset, void *destination,
                       size_t bytes) {
    int fd = open(path, O_RDONLY);
    if (fd < 0) return 0;
    size_t done = 0;
    while (done < bytes) {
        ssize_t count = pread(fd, (char *)destination + done, bytes - done,
                              (off_t)(offset + done));
        if (count <= 0) { close(fd); return 0; }
        done += (size_t)count;
    }
    close(fd);
    return 1;
}

static h3_gpu_tensor *load_int8_dequantized(const h3_weight_store *store,
                                            h3_gpu *gpu, const char *name,
                                            const h3_st_tensor *tensor,
                                            const h3_st_header *header,
                                            size_t elements, h3_dtype dtype,
                                            char *error, size_t error_size) {
    char scale_name[512];
    int written = snprintf(scale_name, sizeof(scale_name), "%s_scale", name);
    if (written < 0 || (size_t)written >= sizeof(scale_name)) {
        fail(error, error_size, "weight name is too long: %s", name);
        return NULL;
    }
    const h3_st_header *scale_header = NULL;
    const h3_st_tensor *scale = h3_weight_find(store, scale_name, &scale_header);
    if (!scale) {
        fail(error, error_size, "int8 weight %s is missing %s", name,
             scale_name);
        return NULL;
    }
    if (scale->dtype != H3_DTYPE_F32 || scale->shape[1] != 1) {
        fail(error, error_size,
             "int8 weight %s: %s must be F32 [rows,1], got %s [%llu,%llu]",
             name, scale_name, h3_dtype_name(scale->dtype),
             (unsigned long long)scale->shape[0],
             (unsigned long long)scale->shape[1]);
        return NULL;
    }
    size_t rows = (size_t)scale->shape[0];
    if (rows == 0 || elements % rows) {
        fail(error, error_size,
             "int8 weight %s: %zu scale rows do not divide %zu elements",
             name, rows, elements);
        return NULL;
    }
    size_t columns = elements / rows;
    int unrotate = int8_unrotate_enabled();
    if (unrotate && columns % H3_INT8_BLOCK) {
        fail(error, error_size,
             "int8 weight %s: input dim %zu is not a multiple of %d",
             name, columns, H3_INT8_BLOCK);
        return NULL;
    }

    int8_t *quantized = malloc(elements);
    float *scales = malloc(rows * sizeof(float));
    float *values = malloc(elements * sizeof(float));
    if (!quantized || !scales || !values) {
        free(quantized); free(scales); free(values);
        fail(error, error_size, "out of memory dequantizing %s", name);
        return NULL;
    }
    if (!pread_bytes(header->path, tensor->file_offset, quantized, elements) ||
        !pread_bytes(scale_header->path, scale->file_offset, scales,
                     rows * sizeof(float))) {
        free(quantized); free(scales); free(values);
        fail(error, error_size, "cannot read int8 payload for %s", name);
        return NULL;
    }

    for (size_t row = 0; row < rows; row++) {
        const int8_t *source = quantized + row * columns;
        float *target = values + row * columns;
        float scale_value = scales[row];
        for (size_t block = 0; block < columns; block += H3_INT8_BLOCK) {
            for (int k = 0; k < H3_INT8_BLOCK; k++)
                target[block + k] = (float)source[block + k] * scale_value;
            if (unrotate) convrot_unrotate_row(target + block);
        }
    }
    free(quantized);
    free(scales);

    h3_gpu_tensor *result = NULL;
    if (dtype == H3_DTYPE_BF16) {
        uint16_t *packed = malloc(elements * sizeof(*packed));
        if (!packed) {
            free(values);
            fail(error, error_size, "out of memory packing %s", name);
            return NULL;
        }
        for (size_t index = 0; index < elements; index++)
            packed[index] = f32_to_bf16_u16(values[index]);
        result = h3_gpu_tensor_from_bf16(gpu, packed, elements);
        free(packed);
    } else {
        result = h3_gpu_tensor_from_f32(gpu, values, elements);
    }
    free(values);
    if (!result)
        fail(error, error_size, "cannot upload dequantized %s: %s", name,
             h3_gpu_error(gpu));
    return result;
}
/* FP16 -> F32/BF16 widening.  The video VAE shipped in the INT8-ConvRot bundle
 * is FP16 (4.85 GiB, exactly half the F32 original) rather than INT8, so it
 * needs a widen-on-load path instead of the dequantize path above. */
static float f16_to_f32(uint16_t half) {
    uint32_t sign = (uint32_t)(half & 0x8000u) << 16;
    uint32_t exponent = (uint32_t)(half >> 10) & 0x1fu;
    uint32_t mantissa = (uint32_t)(half & 0x3ffu);
    uint32_t bits;
    if (exponent == 0u) {
        if (mantissa == 0u) {
            bits = sign;                        /* +-0 */
        } else {                                /* subnormal: renormalise */
            exponent = 127u - 15u + 1u;
            while ((mantissa & 0x400u) == 0u) { mantissa <<= 1; exponent--; }
            mantissa &= 0x3ffu;
            bits = sign | (exponent << 23) | (mantissa << 13);
        }
    } else if (exponent == 31u) {
        bits = sign | 0x7f800000u | (mantissa << 13);   /* inf / nan */
    } else {
        bits = sign | ((exponent + 127u - 15u) << 23) | (mantissa << 13);
    }
    float value;
    memcpy(&value, &bits, sizeof(value));
    return value;
}

static h3_gpu_tensor *load_f16_as_float(h3_gpu *gpu, const char *name,
                                        const h3_st_tensor *tensor,
                                        const h3_st_header *header,
                                        size_t elements, h3_dtype dtype,
                                        char *error, size_t error_size) {
    uint16_t *halves = malloc(elements * sizeof(*halves));
    float *values = malloc(elements * sizeof(*values));
    if (!halves || !values) {
        free(halves); free(values);
        fail(error, error_size, "out of memory widening fp16 weight %s", name);
        return NULL;
    }
    if (!pread_bytes(header->path, tensor->file_offset, halves,
                     elements * sizeof(*halves))) {
        free(halves); free(values);
        fail(error, error_size, "cannot read fp16 payload for %s", name);
        return NULL;
    }
    for (size_t index = 0; index < elements; index++)
        values[index] = f16_to_f32(halves[index]);
    free(halves);

    h3_gpu_tensor *result = NULL;
    if (dtype == H3_DTYPE_BF16) {
        uint16_t *packed = malloc(elements * sizeof(*packed));
        if (!packed) {
            free(values);
            fail(error, error_size, "out of memory packing %s", name);
            return NULL;
        }
        for (size_t index = 0; index < elements; index++)
            packed[index] = f32_to_bf16_u16(values[index]);
        result = h3_gpu_tensor_from_bf16(gpu, packed, elements);
        free(packed);
    } else {
        result = h3_gpu_tensor_from_f32(gpu, values, elements);
    }
    free(values);
    if (!result)
        fail(error, error_size, "cannot upload widened fp16 weight %s: %s",
             name, h3_gpu_error(gpu));
    return result;
}
/* --- end P13 ------------------------------------------------------------ */

static h3_gpu_tensor *load_tensor(const h3_weight_store *store, h3_gpu *gpu,
                                  const char *name, int ndim,
                                  const uint64_t *shape, h3_dtype dtype,
                                  char *error, size_t error_size) {
    const h3_st_header *header = NULL;
    const h3_st_tensor *tensor = h3_weight_find(store, name, &header);
    if (!tensor) {
        fail(error, error_size, "required weight is absent: %s", name);
        return NULL;
    }
    /* P13: INT8 weights can be dequantized and FP16 weights widened on the fly
     * for callers that ask for BF16/F32. */
    int int8_dequant = tensor->dtype == H3_DTYPE_I8 &&
                       (dtype == H3_DTYPE_F32 || dtype == H3_DTYPE_BF16);
    int f16_widen = tensor->dtype == H3_DTYPE_F16 &&
                    (dtype == H3_DTYPE_F32 || dtype == H3_DTYPE_BF16);
    if (!int8_dequant && !f16_widen &&
        (tensor->dtype != dtype || tensor->ndim != ndim)) {
        fail(error, error_size, "weight %s has dtype/rank %s/%d, expected %s/%d",
             name, h3_dtype_name(tensor->dtype), tensor->ndim,
             h3_dtype_name(dtype), ndim);
        return NULL;
    }
    uint64_t elements = 1;
    for (int dimension = 0; dimension < ndim; dimension++) {
        if (tensor->shape[dimension] != shape[dimension]) {
            fail(error, error_size, "weight %s shape mismatch at dimension %d",
                 name, dimension);
            return NULL;
        }
        if (shape[dimension] && elements > UINT64_MAX / shape[dimension]) {
            fail(error, error_size, "weight %s shape overflows", name);
            return NULL;
        }
        elements *= shape[dimension];
    }
    if (elements > SIZE_MAX) {
        fail(error, error_size, "weight %s is too large for this process", name);
        return NULL;
    }
    /* P13: INT8 -> BF16/F32 (dequantize + undo the ConvRot rotation). */
    if (int8_dequant)
        return load_int8_dequantized(store, gpu, name, tensor, header,
                                     (size_t)elements, dtype, error,
                                     error_size);
    /* P13: FP16 -> F32/BF16 (widen on load). */
    if (f16_widen)
        return load_f16_as_float(gpu, name, tensor, header, (size_t)elements,
                                 dtype, error, error_size);
    h3_gpu_tensor *result;
    if (dtype == H3_DTYPE_BF16)
        result = h3_gpu_tensor_load_bf16(gpu, header->path, tensor->file_offset,
                                        (size_t)elements);
    else if (dtype == H3_DTYPE_I8)
        result = h3_gpu_tensor_load_i8(gpu, header->path, tensor->file_offset,
                                       (size_t)elements);
    else
        result = h3_gpu_tensor_load_f32(gpu, header->path, tensor->file_offset,
                                        (size_t)elements);
    if (!result) {
        fail(error, error_size, "cannot load %s: %s", name, h3_gpu_error(gpu));
    }
    return result;
}

h3_gpu_tensor *h3_weight_load_bf16(const h3_weight_store *store, h3_gpu *gpu,
                                   const char *name, int ndim,
                                   const uint64_t *shape,
                                   char *error, size_t error_size) {
    return load_tensor(store, gpu, name, ndim, shape, H3_DTYPE_BF16,
                       error, error_size);
}

h3_gpu_tensor *h3_weight_load_f32(const h3_weight_store *store, h3_gpu *gpu,
                                  const char *name, int ndim,
                                  const uint64_t *shape,
                                  char *error, size_t error_size) {
    return load_tensor(store, gpu, name, ndim, shape, H3_DTYPE_F32,
                       error, error_size);
}

h3_gpu_tensor *h3_weight_load_i8(const h3_weight_store *store, h3_gpu *gpu,
                                const char *name, int ndim, const uint64_t *shape,
                                char *error, size_t error_size) {
    return load_tensor(store, gpu, name, ndim, shape, H3_DTYPE_I8,
                       error, error_size);
}
