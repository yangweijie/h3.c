/* Verify the packed group-quantized weight loader (h3_weights.c).

 * The format is written by fastvideo_qad/scripts/export_h3_int6_native.py:
 *   {name}.weight        U8   [rows, columns*bits/8]   packed unsigned codes
 *   {name}.weight_scale  F16  [rows, columns/group]
 *   {name}.weight_bias   F16  [rows, columns/group]
 *   dequant: w = code*scale + bias, then the ConvRot un-rotation.
 *
 * This test loads four matrices of one block through the real engine entry point
 * (h3_weight_load_bf16, which is what both the resident and the streaming loaders
 * feed) and compares them against goldens produced independently in Python.
 *
 * Tolerance is on the float values, not bit-exact: the engine may contract
 * `code * scale + bias` into an FMA while the golden script evaluates mul-then-add,
 * and one float32 ulp can flip a BF16 tie.
 *
 * Usage (from the repo root):
 *     python fastvideo_qad/scripts/gen_grouped_golden.py --native <dir> --out <golden>
 *     ./h3_grouped_tests <native transformer dir> <golden dir>
 */

#include "h3_gpu.h"
#include "h3_safetensors.h"
#include "h3_weights.h"

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

enum {
    HIDDEN = 5376,
    INNER = 7168,
    FFN = 14336
};

typedef struct {
    const char *name;
    const char *golden;
    uint64_t rows;
    uint64_t columns;
} fixture;

static const fixture FIXTURES[] = {
    {"blocks.0.attn.qkv_proj.weight",   "golden_attn_qkv_proj.bin",  21504, HIDDEN},
    {"blocks.0.attn.out_proj.weight",   "golden_attn_out_proj.bin",   5376, INNER},
    {"blocks.0.mlp.fc1.weight",         "golden_mlp_fc1.bin",         FFN * 2, HIDDEN},
    {"blocks.0.mlp.fc2.weight",         "golden_mlp_fc2.bin",         HIDDEN, FFN}
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

static float bf16_to_f32(uint16_t value) {
    uint32_t bits = (uint32_t)value << 16;
    float result;
    memcpy(&result, &bits, sizeof(result));
    return result;
}

static void *read_file(const char *path, size_t bytes) {
    FILE *file = fopen(path, "rb");
    if (!file) return NULL;
    void *buffer = malloc(bytes ? bytes : 1);
    if (!buffer) { fclose(file); return NULL; }
    if (fread(buffer, 1, bytes, file) != bytes) {
        free(buffer);
        fclose(file);
        return NULL;
    }
    fclose(file);
    return buffer;
}

int main(int argc, char **argv) {
    if (argc < 3) {
        fprintf(stderr, "usage: %s <native transformer dir> <golden dir>\n",
                argv[0]);
        return 2;
    }
    char error[512] = "";
    h3_weight_store *store = h3_weight_store_open(argv[1], error, sizeof(error));
    if (!store) {
        fprintf(stderr, "cannot open %s: %s\n", argv[1], error);
        return 2;
    }
    h3_gpu *gpu = h3_gpu_create("h3_shaders.metal", error, sizeof(error));
    if (!gpu) {
        fprintf(stderr, "cannot create a GPU context: %s\n", error);
        h3_weight_store_free(store);
        return 2;
    }

    /* Structural check first: the loader must derive bits/group from shapes. */
    for (size_t index = 0; index < sizeof(FIXTURES) / sizeof(FIXTURES[0]);
         index++) {
        const fixture *item = &FIXTURES[index];
        h3_grouped_spec spec;
        int status = h3_weight_grouped_spec(store, item->name,
                                            item->rows * item->columns, &spec,
                                            error, sizeof(error));
        char message[256];
        snprintf(message, sizeof(message), "%s: derived spec", item->name);
        check(status == 1, message);
        if (status != 1) continue;
        snprintf(message, sizeof(message),
                 "%s: bits=%u group=%u rows=%u columns=%u %s accumulators",
                 item->name, spec.bits, spec.group, spec.rows, spec.columns,
                 spec.f16_accumulators ? "F16" : "F32");
        check(spec.bits == 6 && spec.group == 128 && spec.rows == item->rows &&
              spec.columns == item->columns && spec.f16_accumulators, message);
    }

    for (size_t index = 0; index < sizeof(FIXTURES) / sizeof(FIXTURES[0]);
         index++) {
        const fixture *item = &FIXTURES[index];
        uint64_t shape[2] = {item->rows, item->columns};
        size_t elements = (size_t)(item->rows * item->columns);
        h3_gpu_tensor *loaded = h3_weight_load_bf16(store, gpu, item->name, 2,
                                                    shape, error, sizeof(error));
        char message[256];
        snprintf(message, sizeof(message), "%s: loaded through the engine",
                 item->name);
        check(loaded != NULL, message);
        if (!loaded) {
            fprintf(stderr, "      %s\n", error);
            continue;
        }
        uint16_t *produced = malloc(elements * sizeof(uint16_t));
        char path[1024];
        snprintf(path, sizeof(path), "%s/%s", argv[2], item->golden);
        uint16_t *golden = read_file(path, elements * sizeof(uint16_t));
        int ok = produced && golden &&
                 h3_gpu_tensor_read_bf16(loaded, produced, elements);
        snprintf(message, sizeof(message), "%s: host readback and golden present",
                 item->name);
        check(ok, message);
        if (ok) {
            double error_squared = 0.0;
            double reference_squared = 0.0;
            for (size_t i = 0; i < elements; i++) {
                double a = bf16_to_f32(produced[i]);
                double b = bf16_to_f32(golden[i]);
                error_squared += (a - b) * (a - b);
                reference_squared += b * b;
            }
            double relative = sqrt(error_squared / (reference_squared + 1e-30));
            snprintf(message, sizeof(message),
                     "%s: relRMS %.3e against the Python golden (tolerance 1e-3)",
                     item->name, relative);
            check(relative < 1e-3, message);
        }
        free(produced);
        free(golden);
        h3_gpu_tensor_free(loaded);
    }

    h3_gpu_free(gpu);
    h3_weight_store_free(store);
    printf("\n%s: %d failure(s)\n", failures ? "FAILED" : "PASSED", failures);
    return failures ? 1 : 0;
}
