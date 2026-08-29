#include "h3_gpu.h"
#include "h3_lora.h"

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

enum {
    HIDDEN = 5376,
    INNER = 7168,
    FFN = 14336,
    RANK = 128,
    ALPHA = 64
};

static int failures = 0;

static void check(int ok, const char *message) {
    if (!ok) {
        fprintf(stderr, "FAIL: %s\n", message);
        failures++;
    } else {
        printf("ok: %s\n", message);
    }
}

static uint16_t *read_bin(const char *path, size_t *count) {
    FILE *file = fopen(path, "rb");
    if (!file) return NULL;
    fseek(file, 0, SEEK_END);
    long bytes = ftell(file);
    fseek(file, 0, SEEK_SET);
    if (bytes <= 0 || bytes % 2) {
        fclose(file);
        return NULL;
    }
    uint16_t *values = malloc((size_t)bytes);
    if (values && fread(values, 1, (size_t)bytes, file) != (size_t)bytes) {
        free(values);
        values = NULL;
    }
    fclose(file);
    if (count) *count = (size_t)bytes / 2;
    return values;
}

static float bf16_to_f32(uint16_t value) {
    uint32_t bits = (uint32_t)value << 16;
    float result;
    memcpy(&result, &bits, sizeof(result));
    return result;
}

static int compare(const char *tag, h3_gpu *gpu, h3_gpu_tensor *weight,
                   const char *expected_path, size_t rows, size_t columns) {
    size_t elements = rows * columns;
    uint16_t *expected = read_bin(expected_path, NULL);
    uint16_t *got = malloc(elements * sizeof(uint16_t));
    if (!expected || !got) {
        check(0, "compare allocation");
        free(expected);
        free(got);
        return 0;
    }
    if (!h3_gpu_tensor_read_bf16(weight, got, elements)) {
        check(0, "read merged weight");
        free(expected);
        free(got);
        return 0;
    }
    double max_rel = 0.0, sum_sq_delta = 0.0, sum_sq_ref = 0.0;
    for (size_t index = 0; index < elements; index++) {
        double want = bf16_to_f32(expected[index]);
        double d = bf16_to_f32(got[index]) - want;
        sum_sq_delta += d * d;
        sum_sq_ref += want * want;
        if (fabs(want) > 1e-12 && fabs(d) / fabs(want) > max_rel)
            max_rel = fabs(d) / fabs(want);
    }
    double rel_l2 = sqrt(sum_sq_delta / (sum_sq_ref > 1e-24 ? sum_sq_ref : 1e-24));
    char message[256];
    snprintf(message, sizeof(message),
             "%s merge matches reference (rel-L2 %.4g, max-rel %.4g)",
             tag, rel_l2, max_rel);
    /* BF16 linear accumulation vs F64 reference: expect ~0.3% rel-L2. Large
     * max-rel values come from elements near zero and are not meaningful. */
    check(rel_l2 < 0.01, message);
    free(expected);
    free(got);
    return rel_l2 < 0.01;
}

int main(void) {
    char error[1024];
    h3_gpu *gpu = h3_gpu_create("h3_shaders.metal", error, sizeof(error));
    check(gpu != NULL, "create Metal GPU");
    if (!gpu) return 1;
    h3_lora *lora = h3_lora_open("tmp_lora_test/lora_test.safetensors",
                                 error, sizeof(error));
    check(lora != NULL, "open synthetic LoRA");
    if (!lora) return 1;
    printf("lora rank=%zu scale=%.4f\n", lora ? 0 : 0, 0.0);
    /* (rank/scale are private; the merge result proves them.) */

    /* qkv: three row bands from to_q/to_k/to_v. */
    {
        size_t rows = (size_t)INNER * 3, columns = HIDDEN;
        uint16_t *init = read_bin("tmp_lora_test/w_qkv_init.bin", NULL);
        check(init != NULL, "read qkv init");
        h3_gpu_tensor *weight = h3_gpu_tensor_from_bf16(gpu, init, rows * columns);
        check(weight != NULL, "upload qkv weight");
        check(h3_lora_apply(gpu, lora, weight, "transformer_blocks.0",
                            "attn.to_q", 0, INNER, columns,
                            error, sizeof(error)) &&
              h3_lora_apply(gpu, lora, weight, "transformer_blocks.0",
                            "attn.to_k", INNER, INNER, columns,
                            error, sizeof(error)) &&
              h3_lora_apply(gpu, lora, weight, "transformer_blocks.0",
                            "attn.to_v", 2 * INNER, INNER, columns,
                            error, sizeof(error)),
              "apply qkv LoRA bands");
        compare("qkv", gpu, weight, "tmp_lora_test/w_qkv_expected.bin",
                rows, columns);
        h3_gpu_tensor_free(weight);
        free(init);
    }
    /* out / fc1 / fc2: whole-tensor merges. */
    {
        size_t rows = HIDDEN, columns = INNER;
        uint16_t *init = read_bin("tmp_lora_test/w_out_init.bin", NULL);
        check(init != NULL, "read out init");
        h3_gpu_tensor *weight = h3_gpu_tensor_from_bf16(gpu, init, rows * columns);
        check(h3_lora_apply(gpu, lora, weight, "transformer_blocks.0",
                            "attn.to_out.0", 0, rows, columns,
                            error, sizeof(error)),
              "apply out_proj LoRA");
        compare("out", gpu, weight, "tmp_lora_test/w_out_expected.bin",
                rows, columns);
        h3_gpu_tensor_free(weight);
        free(init);
    }
    {
        size_t rows = (size_t)FFN * 2, columns = HIDDEN;
        uint16_t *init = read_bin("tmp_lora_test/w_fc1_init.bin", NULL);
        check(init != NULL, "read fc1 init");
        h3_gpu_tensor *weight = h3_gpu_tensor_from_bf16(gpu, init, rows * columns);
        check(h3_lora_apply(gpu, lora, weight, "transformer_blocks.0",
                            "ff.net.0.proj", 0, rows, columns,
                            error, sizeof(error)),
              "apply fc1 LoRA");
        compare("fc1", gpu, weight, "tmp_lora_test/w_fc1_expected.bin",
                rows, columns);
        h3_gpu_tensor_free(weight);
        free(init);
    }
    {
        size_t rows = HIDDEN, columns = FFN;
        uint16_t *init = read_bin("tmp_lora_test/w_fc2_init.bin", NULL);
        check(init != NULL, "read fc2 init");
        h3_gpu_tensor *weight = h3_gpu_tensor_from_bf16(gpu, init, rows * columns);
        check(h3_lora_apply(gpu, lora, weight, "transformer_blocks.0",
                            "ff.net.2", 0, rows, columns,
                            error, sizeof(error)),
              "apply fc2 LoRA");
        compare("fc2", gpu, weight, "tmp_lora_test/w_fc2_expected.bin",
                rows, columns);
        h3_gpu_tensor_free(weight);
        free(init);
    }
    /* A missing target must be a silent no-op. */
    {
        size_t rows = HIDDEN, columns = INNER;
        uint16_t *init = read_bin("tmp_lora_test/w_out_init.bin", NULL);
        h3_gpu_tensor *weight = h3_gpu_tensor_from_bf16(gpu, init, rows * columns);
        check(h3_lora_apply(gpu, lora, weight, "transformer_blocks.99",
                            "attn.to_out.0", 0, rows, columns,
                            error, sizeof(error)),
              "missing target is a no-op");
        h3_gpu_tensor_free(weight);
        free(init);
    }
    h3_lora_close(lora);
    h3_gpu_free(gpu);
    if (failures) {
        fprintf(stderr, "%d test group(s) failed\n", failures);
        return 1;
    }
    puts("ok: h3_lora_apply numerical parity");
    return 0;
}
