#include "h3_lora.h"

#include "h3_safetensors.h"

#include <math.h>
#include <stdarg.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

struct h3_lora {
    h3_st_header header;
    float scale;   /* alpha / rank */
    size_t rank;
    char adapter[64];   /* PEFT adapter name parsed from the lora_A keys */
};

static float bf16_to_f32(uint16_t value) {
    uint32_t bits = (uint32_t)value << 16;
    float result;
    memcpy(&result, &bits, sizeof(result));
    return result;
}

static uint16_t f32_to_bf16(float value) {
    uint32_t bits;
    memcpy(&bits, &value, sizeof(bits));
    bits += 0x7fffu + ((bits >> 16) & 1u);
    bits &= UINT32_C(0xffff0000);
    uint16_t result = (uint16_t)(bits >> 16);
    return result;
}

static void scale_bf16_buffer(uint16_t *values, size_t count, float scale) {
    if (scale == 1.0f) return;
    for (size_t index = 0; index < count; index++) {
        values[index] = f32_to_bf16(bf16_to_f32(values[index]) * scale);
    }
}

static void fail(char *error, size_t error_size, const char *format, ...) {
    if (!error || !error_size) return;
    va_list arguments;
    va_start(arguments, format);
    vsnprintf(error, error_size, format, arguments);
    va_end(arguments);
}

h3_lora *h3_lora_open(const char *path, char *error, size_t error_size) {
    if (!path || !*path) {
        fail(error, error_size, "empty LoRA path");
        return NULL;
    }
    h3_lora *lora = calloc(1, sizeof(*lora));
    if (!lora) {
        fail(error, error_size, "cannot allocate LoRA state");
        return NULL;
    }
    char detail[512];
    if (!h3_st_read_header(path, &lora->header, detail, sizeof(detail))) {
        fail(error, error_size, "cannot read LoRA %s: %s", path, detail);
        h3_lora_close(lora);
        return NULL;
    }
    /* Determine rank from the first lora_A factor. */
    lora->rank = 0;
    for (size_t index = 0; index < lora->header.tensor_count; index++) {
        const h3_st_tensor *tensor = &lora->header.tensors[index];
        if (tensor->ndim == 2 && strstr(tensor->name, "lora_A") != NULL) {
            lora->rank = (size_t)tensor->shape[0];
            break;
        }
    }
    if (!lora->rank) {
        fail(error, error_size, "LoRA %s has no lora_A factors", path);
        h3_lora_close(lora);
        return NULL;
    }
    /* Parse the adapter name out of the first lora_A key
     * ("...lora_A.<adapter>.weight"); "default" is the PEFT fallback. */
    for (size_t index = 0; index < lora->header.tensor_count; index++) {
        const h3_st_tensor *tensor = &lora->header.tensors[index];
        if (tensor->ndim == 2 && strstr(tensor->name, "lora_A") != NULL) {
            const char *start = strstr(tensor->name, "lora_A.");
            if (start) {
                start += 7;
                const char *end = strstr(start, ".weight");
                size_t length = end && end > start ?
                    (size_t)(end - start) : 0;
                if (length && length < sizeof(lora->adapter)) {
                    memcpy(lora->adapter, start, length);
                    lora->adapter[length] = '\0';
                }
            }
            break;
        }
    }
    if (!lora->adapter[0]) snprintf(lora->adapter, sizeof(lora->adapter),
                                    "default");
    /* Read alpha from the raw JSON header (the tensor parser skips
     * __metadata__). Diffusers writes it as a JSON string, e.g. "128". */
    float alpha = 1.0f;
    FILE *file = fopen(path, "rb");
    if (file) {
        uint64_t header_length = 0;
        if (fread(&header_length, sizeof(header_length), 1, file) == 1 &&
            header_length > 0 && header_length <= (1u << 24)) {
            char *json = malloc((size_t)header_length + 1);
            if (json &&
                fread(json, 1, (size_t)header_length, file) ==
                    (size_t)header_length) {
                json[header_length] = '\0';
                const char *alpha_key = strstr(json, "\"alpha\"");
                if (alpha_key) {
                    alpha_key = strchr(alpha_key + 7, ':');
                    if (alpha_key) {
                        while (*alpha_key == ':' || *alpha_key == ' ' ||
                               *alpha_key == '"' || *alpha_key == '\t')
                            alpha_key++;
                        alpha = (float)strtod(alpha_key, NULL);
                        if (!(alpha > 0.0f)) alpha = 1.0f;
                    }
                }
            }
            free(json);
        }
        fclose(file);
    }
    lora->scale = alpha / (float)lora->rank;
    return lora;
}

static int lora_merge(h3_gpu *gpu, h3_lora *lora, h3_gpu_tensor *weight,
                      const char *lora_prefix, const char *target,
                      const char *adapter, size_t row0, size_t rows,
                      size_t in_dim, int blocking,
                      char *error, size_t error_size);

int h3_lora_merge_blocking(h3_gpu *gpu, h3_lora *lora, h3_gpu_tensor *weight,
                           const char *lora_prefix, const char *target,
                           size_t row0, size_t rows, size_t in_dim,
                           char *error, size_t error_size) {
    return lora_merge(gpu, lora, weight, lora_prefix, target, NULL, row0,
                      rows, in_dim, 1, error, error_size);
}

void h3_lora_close(h3_lora *lora) {
    if (!lora) return;
    h3_st_free_header(&lora->header);
    free(lora);
}

int h3_lora_matches(h3_lora *lora, const char *lora_prefix,
                    const char *target, size_t rows, size_t in_dim) {
    (void)rows;   /* the lora_A factor constrains only the input width */
    if (!lora || !target) return 0;
    char key[256];
    int written = lora_prefix && lora_prefix[0]
        ? snprintf(key, sizeof(key), "%s.%s.lora_A.%s.weight",
                   lora_prefix, target, lora->adapter)
        : snprintf(key, sizeof(key), "%s.lora_A.%s.weight",
                   target, lora->adapter);
    if (written >= (int)sizeof(key)) return 0;
    const h3_st_tensor *a = h3_st_find(&lora->header, key);
    if (!a || a->dtype != H3_DTYPE_BF16 || a->ndim != 2) return 0;
    return (size_t)a->shape[1] == in_dim;
}

int h3_lora_apply(h3_gpu *gpu, h3_lora *lora, h3_gpu_tensor *weight,
                  const char *lora_prefix, const char *target,
                  size_t row0, size_t rows, size_t in_dim,
                  char *error, size_t error_size) {
    /* NULL falls back to the adapter name this file actually carries, matching
     * h3_lora_matches() and h3_lora_merge_blocking(). A hard-coded "default"
     * silently skipped turbo-only adapters (whose keys end in .turbo). */
    return lora_merge(gpu, lora, weight, lora_prefix, target, NULL, row0,
                      rows, in_dim, 0, error, error_size);
}

int h3_lora_apply_named(h3_gpu *gpu, h3_lora *lora, h3_gpu_tensor *weight,
                        const char *lora_prefix, const char *target,
                        const char *adapter, size_t row0, size_t rows,
                        size_t in_dim, char *error, size_t error_size) {
    return lora_merge(gpu, lora, weight, lora_prefix, target, adapter, row0,
                      rows, in_dim, 0, error, error_size);
}

/* Shared merge body. `blocking` runs the delta GEMM on a private command
 * buffer so the SSD prefetch thread can merge without touching the main
 * thread's open command buffer. */
static int lora_merge(h3_gpu *gpu, h3_lora *lora, h3_gpu_tensor *weight,
                      const char *lora_prefix, const char *target,
                      const char *adapter, size_t row0, size_t rows,
                      size_t in_dim, int blocking,
                      char *error, size_t error_size) {
    if (!gpu || !lora || !weight || !lora_prefix || !target) {
        fail(error, error_size, "invalid LoRA apply arguments");
        return 0;
    }
    if (!rows) return 1;
    if (!adapter || !*adapter) adapter = lora->adapter;
    char key_a[256], key_b[256];
    int written_a, written_b;
    if (lora_prefix[0])
        written_a = snprintf(key_a, sizeof(key_a), "%s.%s.lora_A.%s.weight",
                             lora_prefix, target, adapter);
    else
        written_a = snprintf(key_a, sizeof(key_a), "%s.lora_A.%s.weight",
                             target, adapter);
    if (lora_prefix[0])
        written_b = snprintf(key_b, sizeof(key_b), "%s.%s.lora_B.%s.weight",
                             lora_prefix, target, adapter);
    else
        written_b = snprintf(key_b, sizeof(key_b), "%s.lora_B.%s.weight",
                             target, adapter);
    if (written_a >= (int)sizeof(key_a) || written_b >= (int)sizeof(key_b)) {
        fail(error, error_size, "LoRA key overflow");
        return 0;
    }
    const h3_st_tensor *a = h3_st_find(&lora->header, key_a);
    const h3_st_tensor *b = h3_st_find(&lora->header, key_b);
    if (!a || !b) return 1;   /* target not covered by this LoRA */
    if (a->dtype != H3_DTYPE_BF16 || b->dtype != H3_DTYPE_BF16 ||
        a->ndim != 2 || b->ndim != 2) {
        fail(error, error_size, "LoRA %s is not a 2-D BF16 factor pair", key_a);
        return 0;
    }
    size_t rank = (size_t)a->shape[0];
    if (!rank || (size_t)a->shape[1] != in_dim ||
        (size_t)b->shape[0] != rows || (size_t)b->shape[1] != rank) {
        fail(error, error_size,
             "LoRA %s shape mismatch: A=[%llu,%llu] B=[%llu,%llu], "
             "expected A=[%zu,%zu] B=[%zu,%zu]",
             key_a, (unsigned long long)a->shape[0],
             (unsigned long long)a->shape[1],
             (unsigned long long)b->shape[0],
             (unsigned long long)b->shape[1], rank, in_dim, rows, rank);
        return 0;
    }
    char detail[512];
    /* rank/in_dim/rows come from the file's declared shapes: reject any product
     * whose BF16 byte count would wrap size_t before allocating. */
    if (!in_dim || !rows ||
        rank > (SIZE_MAX / sizeof(uint16_t)) / in_dim ||
        rank > (SIZE_MAX / sizeof(uint16_t)) / rows) {
        fail(error, error_size, "LoRA factor dimensions overflow");
        return 0;
    }
    uint16_t *a_bf16 = malloc(rank * in_dim * sizeof(uint16_t));
    uint16_t *b_bf16 = malloc(rows * rank * sizeof(uint16_t));
    uint16_t *at_bf16 = malloc(in_dim * rank * sizeof(uint16_t));
    if (!a_bf16 || !b_bf16 || !at_bf16) {
        fail(error, error_size, "cannot allocate LoRA factor buffers");
        free(a_bf16); free(b_bf16); free(at_bf16);
        return 0;
    }
    if (!h3_st_read_data(&lora->header, a, a_bf16,
                         rank * in_dim * sizeof(uint16_t),
                         detail, sizeof(detail)) ||
        !h3_st_read_data(&lora->header, b, b_bf16,
                         rows * rank * sizeof(uint16_t),
                         detail, sizeof(detail))) {
        fail(error, error_size, "cannot read LoRA factors: %s", detail);
        free(a_bf16); free(b_bf16); free(at_bf16);
        return 0;
    }
    /* Apply scale to B up front (delta = scale * B @ A). */
    scale_bf16_buffer(b_bf16, rows * rank, lora->scale);
    /* Transpose A into [in_dim, rank] so the GPU linear layer can compute
     * delta = B @ A as x @ W^T with x = B [rows, rank], W = A^T. */
    for (size_t row = 0; row < rank; row++) {
        for (size_t column = 0; column < in_dim; column++) {
            at_bf16[column * rank + row] = a_bf16[row * in_dim + column];
        }
    }
    h3_gpu_tensor *b_t = h3_gpu_tensor_from_bf16(gpu, b_bf16, rows * rank);
    h3_gpu_tensor *at_t =
        h3_gpu_tensor_from_bf16(gpu, at_bf16, in_dim * rank);
    int ok = b_t && at_t;
    /* One fused kernel: weight += scale * B @ A, all on the GPU. The old path
     * read the delta back to the host, added it there and wrote it back, which
     * dominated the SSD streaming loop (about 1 GiB of host traffic per block
     * per adapter, every step). */
    if (ok && blocking) {
        ok = h3_gpu_blocking_lora_geam_bf16(gpu, weight, row0, b_t, at_t,
                                            (uint32_t)rows, (uint32_t)rank,
                                            (uint32_t)in_dim);
    } else if (ok) {
        /* Both GEAM entry points own a private, immediately-waited command
         * buffer, so opening the caller's buffer here would only add a failure
         * mode (h3_gpu_begin fails when an encoder is already open) and, on
         * failure, leak an unsubmitted gpu.command that breaks every later GPU
         * stage. */
        ok = h3_gpu_lora_geam_bf16(gpu, weight, row0, b_t, at_t,
                                   (uint32_t)rows, (uint32_t)rank,
                                   (uint32_t)in_dim);
    }
    h3_gpu_tensor_free(b_t);
    h3_gpu_tensor_free(at_t);
    free(a_bf16);
    free(b_bf16);
    free(at_bf16);
    if (!ok && error && error_size && !error[0])
        fail(error, error_size, "LoRA merge failed: %s",
             h3_gpu_error(gpu));
    return ok;
}
