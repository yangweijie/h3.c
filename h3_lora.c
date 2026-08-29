#include "h3_lora.h"

#include "h3_safetensors.h"

#include <math.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

struct h3_lora {
    h3_st_header header;
    float scale;   /* alpha / rank */
    size_t rank;
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

void h3_lora_close(h3_lora *lora) {
    if (!lora) return;
    h3_st_free_header(&lora->header);
    free(lora);
}

int h3_lora_apply(h3_gpu *gpu, h3_lora *lora, h3_gpu_tensor *weight,
                  const char *lora_prefix, const char *target,
                  size_t row0, size_t rows, size_t in_dim,
                  char *error, size_t error_size) {
    if (!gpu || !lora || !weight || !lora_prefix || !target) {
        fail(error, error_size, "invalid LoRA apply arguments");
        return 0;
    }
    if (!rows) return 1;
    char key_a[256], key_b[256];
    if (snprintf(key_a, sizeof(key_a), "%s.%s.lora_A.default.weight",
                 lora_prefix, target) >= (int)sizeof(key_a) ||
        snprintf(key_b, sizeof(key_b), "%s.%s.lora_B.default.weight",
                 lora_prefix, target) >= (int)sizeof(key_b)) {
        fail(error, error_size, "LoRA key overflow");
        return 0;
    }
    const h3_st_tensor *a = h3_st_find(&lora->header, key_a);
    const h3_st_tensor *b = h3_st_find(&lora->header, key_b);
    if (!a || !b) return 1;   /* target not covered by this LoRA */
    size_t rank = (size_t)a->shape[0];
    if (a->dtype != H3_DTYPE_BF16 || b->dtype != H3_DTYPE_BF16 ||
        a->ndim != 2 || b->ndim != 2 ||
        (size_t)a->shape[1] != in_dim ||
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
    h3_gpu_tensor *delta = h3_gpu_tensor_new_bf16(gpu, rows * in_dim);
    int ok = b_t && at_t && delta;
    if (ok) ok = h3_gpu_begin(gpu);
    if (ok) {
        ok = h3_gpu_linear_bf16(gpu, delta, b_t, at_t, NULL,
                                (uint32_t)rows, (uint32_t)rank,
                                (uint32_t)in_dim);
    }
    if (ok) ok = h3_gpu_submit(gpu);
    if (ok) {
        /* Update the weight band: current + delta (both BF16). */
        size_t band = rows * in_dim;
        uint16_t *delta_bf16 = malloc(band * sizeof(uint16_t));
        uint16_t *current_bf16 = malloc(band * sizeof(uint16_t));
        uint16_t *merged_bf16 = malloc(band * sizeof(uint16_t));
        if (!delta_bf16 || !current_bf16 || !merged_bf16) {
            fail(error, error_size, "cannot allocate LoRA merge buffers");
            ok = 0;
        }
        if (ok) ok = h3_gpu_tensor_read_bf16(delta, delta_bf16, band);
        if (ok) ok = h3_gpu_tensor_read_bf16_range(
            weight, row0 * in_dim, current_bf16, band);
        if (ok) {
            for (size_t index = 0; index < band; index++) {
                merged_bf16[index] = f32_to_bf16(
                    bf16_to_f32(current_bf16[index]) +
                    bf16_to_f32(delta_bf16[index]));
            }
            ok = h3_gpu_tensor_write_bf16_range(
                weight, row0 * in_dim, merged_bf16, band);
        }
        free(delta_bf16);
        free(current_bf16);
        free(merged_bf16);
    }
    h3_gpu_tensor_free(b_t);
    h3_gpu_tensor_free(at_t);
    h3_gpu_tensor_free(delta);
    free(a_bf16);
    free(b_bf16);
    free(at_bf16);
    if (!ok && error && error_size && !error[0])
        fail(error, error_size, "LoRA merge failed: %s",
             h3_gpu_error(gpu));
    return ok;
}
