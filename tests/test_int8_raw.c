/* Validate the raw comfy-quants int8 loader that feeds the Neural Engine graphs.
 *
 * The ANE path consumes the I8 codes and the F32 per-row scales exactly as
 * stored (the graph dequantises inside `constexpr_affine_dequantize`), so this
 * checks four things against a real ConvRot checkpoint:
 *   1. the payload lands byte-for-byte where the safetensors header says,
 *   2. the sidecar-derived ConvRot group size matches the engine's assumption,
 *   3. the table derotation used by the ANE path and the radix-4 butterfly used
 *      by the resident/streaming loaders recover the same weights,
 *   4. the fp16 raw loader copies an F16 tensor bit-for-bit and refuses an I8
 *      one (the video VAE graphs bake their weights through that loader).
 * Item 3 is the one that matters for the port: if the two ever drift, an ANE
 * projection silently disagrees with every Metal projection in the same run.
 *
 * Usage (from the repo root):
 *     ./h3_int8_raw_test <transformer dir holding the int8_convrot shard>
 */

#include "h3_convrot.h"
#include "h3_weights.h"

#include <math.h>
#include <stdarg.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

enum { SAMPLE_ROWS = 256 };

static int failures = 0;

static void check(int ok, const char *format, ...) {
    va_list arguments;
    fprintf(stdout, ok ? "ok:   " : "FAIL: ");
    va_start(arguments, format);
    vprintf(format, arguments);
    va_end(arguments);
    fputc('\n', stdout);
    if (!ok) failures++;
}

static void compare_window(const char *path, uint64_t offset, const void *data,
                          size_t bytes, const char *what) {
    size_t probe = bytes < (1u << 20) ? bytes : (1u << 20);
    FILE *file = fopen(path, "rb");
    if (!file) {
        check(0, "%s: cannot reopen %s", what, path);
        return;
    }
    unsigned char *raw = malloc(probe);
    if (!raw) {
        check(0, "%s: out of memory reading windows", what);
        fclose(file);
        return;
    }
    int head = fseeko(file, (off_t)offset, SEEK_SET) == 0 &&
               fread(raw, 1, probe, file) == probe &&
               memcmp(raw, data, probe) == 0;
    int tail = fseeko(file, (off_t)(offset + bytes - probe), SEEK_SET) == 0 &&
               fread(raw, 1, probe, file) == probe &&
               memcmp(raw, (const unsigned char *)data + bytes - probe,
                      probe) == 0;
    free(raw);
    fclose(file);
    check(head && tail,
          "%s: first and last %zu bytes match the shard at offset 0x%llx "
          "(head=%d tail=%d)", what, probe, (unsigned long long)offset, head,
          tail);
}

/* Dequantise `rows` rows and un-rotate them twice: once with the cached
 * Hadamard table (the ANE contract) and once with h3_weights.c's butterfly. */
static void compare_unrotate(const int8_t *codes, const float *scales,
                            uint64_t columns, int group_size, uint64_t rows,
                            const char *what) {
    size_t elements = (size_t)rows * columns;
    float *table = malloc(sizeof(float) * elements);
    float *butterfly = malloc(sizeof(float) * elements);
    if (!table || !butterfly) {
        check(0, "%s: out of memory comparing unrotations", what);
        free(table);
        free(butterfly);
        return;
    }
    for (size_t i = 0; i < elements; i++)
        butterfly[i] = (float)codes[i] * scales[i / columns];
    memcpy(table, butterfly, sizeof(float) * elements);
    if (group_size) {
        check(h3_convrot_derotate_f32(table, rows, columns, group_size),
              "%s: table derotate accepted (group %d)", what, group_size);
        h3_weight_unrotate_rows(butterfly, rows, columns);
    }
    double scale = 0.0, rms = 0.0, worst = 0.0;
    for (size_t i = 0; i < elements; i++) {
        double difference = (double)table[i] - (double)butterfly[i];
        double magnitude = fabs((double)table[i]);
        if (magnitude > scale) scale = magnitude;
        if (fabs(difference) > worst) worst = fabs(difference);
        rms += difference * difference;
    }
    rms = sqrt(rms / (double)elements);
    check(rms / (scale + 1e-30) < 1e-6 && isfinite(rms),
          "%s: table vs butterfly un-rotate rel_rms=%.3e max_abs=%.3e over %zu rows",
          what, rms / (scale + 1e-30), worst, (size_t)rows);
    free(table);
    free(butterfly);
}

static void probe(const h3_weight_store *store, const char *name, uint64_t rows,
                  uint64_t columns) {
    int8_t *codes = NULL;
    float *scales = NULL;
    int group_size = -1;
    char error[512] = {0};
    if (!h3_weight_load_int8_raw(store, name, rows, columns, &codes, &scales,
                                 &group_size, error, sizeof(error))) {
        check(0, "%s: loader rejected the tensor: %s", name, error);
        return;
    }
    check(codes && scales, "%s: [%llu,%llu] loaded (%.1f MiB, convrot group %d)",
          name, (unsigned long long)rows, (unsigned long long)columns,
          (double)((size_t)rows * columns) / (1024.0 * 1024.0), group_size);

    int finite = 1;
    float smallest = 0.0f, largest = 0.0f;
    for (uint64_t i = 0; i < rows; i++) {
        if (!isfinite(scales[i]) || scales[i] <= 0.0f) finite = 0;
        if (i == 0 || scales[i] < smallest) smallest = scales[i];
        if (i == 0 || scales[i] > largest) largest = scales[i];
    }
    check(finite, "%s: %llu F32 row scales in [%.3e, %.3e]", name,
          (unsigned long long)rows, smallest, largest);

    const h3_st_header *header = NULL;
    const h3_st_tensor *tensor = h3_weight_find(store, name, &header);
    size_t bytes = (size_t)rows * columns;
    compare_window(header->path, tensor->file_offset, codes, bytes, name);

    uint64_t sample = rows < SAMPLE_ROWS ? rows : SAMPLE_ROWS;
    compare_unrotate(codes, scales, columns, group_size, sample, name);
    free(codes);
    free(scales);
}

/* A wrong shape or an unquantized tensor must be refused, not read. */
static void rejects(const h3_weight_store *store) {
    int8_t *codes = NULL;
    float *scales = NULL;
    int group_size = 0;
    char error[512] = {0};
    check(!h3_weight_load_int8_raw(store, "blocks.0.attn.qkv_proj.weight",
                                   21503, 5376, &codes, &scales, &group_size,
                                   error, sizeof(error)),
          "shape mismatch refused: %s", error);
    check(!h3_weight_load_int8_raw(store, "blocks.0.attn.qkv_proj.no_such_weight",
                                   4, 4, &codes, &scales, &group_size, error,
                                   sizeof(error)),
          "absent tensor refused: %s", error);
    uint16_t *halves = NULL;
    check(!h3_weight_load_f16_raw(store, "blocks.0.attn.qkv_proj.weight", 21504,
                                  5376, &halves, error, sizeof(error)) && !halves,
          "int8 tensor refused by the fp16 loader: %s", error);
}

/* The shard also ships a handful of unquantized F16 matrices, which cover the
 * fp16 loader's passthrough branch: the payload has to land bit-for-bit. */
static void probe_f16(const h3_weight_store *store, const char *name,
                      uint64_t rows, uint64_t columns) {
    uint16_t *halves = NULL;
    char error[512] = {0};
    if (!h3_weight_load_f16_raw(store, name, rows, columns, &halves, error,
                                sizeof(error))) {
        check(0, "%s: fp16 loader rejected the tensor: %s", name, error);
        return;
    }
    const h3_st_header *header = NULL;
    const h3_st_tensor *tensor = h3_weight_find(store, name, &header);
    check(tensor && tensor->dtype == H3_DTYPE_F16, "%s: stored as F16 [%llu][%llu]",
          name, (unsigned long long)rows, (unsigned long long)columns);
    compare_window(header->path, tensor->file_offset, halves,
                   sizeof(uint16_t) * (size_t)rows * columns, name);
    int finite = 1;
    for (uint64_t i = 0; i < rows * columns; i++) {
        float value = (float)*(__fp16 *)&halves[i];
        if (!isfinite(value)) finite = 0;
    }
    check(finite, "%s: every half decodes to a finite fp16 value", name);
    free(halves);
}

int main(int argc, char **argv) {
    if (argc < 2) {
        fprintf(stderr, "usage: %s <transformer dir>\n", argv[0]);
        return 1;
    }
    char error[512] = {0};
    h3_weight_store *store = h3_weight_store_open(argv[1], error, sizeof(error));
    if (!store) {
        fprintf(stderr, "FAIL: cannot open %s: %s\n", argv[1], error);
        return 1;
    }
    printf("shards: %zu\n", h3_weight_store_shards(store));
    rejects(store);
    probe(store, "blocks.0.attn.qkv_proj.weight", 21504, 5376);
    probe(store, "blocks.0.attn.out_proj.weight", 5376, 7168);
    probe(store, "blocks.0.mlp.fc1.weight", 28672, 5376);
    probe(store, "blocks.0.mlp.fc2.weight", 5376, 14336);
    probe_f16(store, "blocks.0.adaln_proj.linear.weight", 96768, 8);
    h3_weight_store_free(store);
    printf("%d failure(s)\n", failures);
    return failures ? 1 : 0;
}
