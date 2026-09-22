/* Same-process A/B of the two BF16 DiT MLP paths, so cross-run noise cannot be
 * blamed for the difference.
 *
 * The DiT per-op profile (H3_DIT_OP_PROFILE) showed the fused fc1->SwiGLU->fc2
 * graph costing 8-10% more than the two separate GEMMs, but the two numbers came
 * from two processes, and the same QKV op measured 18% apart across those two
 * runs. Here both paths run in one process on one set of buffers, alternating
 * round by round, so the spread within a column is the real noise floor.
 *
 * Shapes are the released DiT ones; the row counts are the two resolutions the
 * profile measured (864x480 and 576x320 at 2 seconds).
 *
 * Usage (from the repo root):
 *     ./h3_mlp_fusion_bench [ROUNDS] [ROWS] [ROWS...]
 */

#include "h3_gpu.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

enum { HIDDEN = 5376, FFN = 14336 };

typedef struct {
    h3_gpu_tensor *input;
    h3_gpu_tensor *output;
    h3_gpu_tensor *fc1_weight;
    h3_gpu_tensor *fc2_weight;
    h3_gpu_tensor *fc1_result;   /* only the unfused path materialises this */
    h3_gpu_tensor *activated;
} buffers;

static double now_seconds(void) {
    struct timespec stamp;
    clock_gettime(CLOCK_MONOTONIC, &stamp);
    return (double)stamp.tv_sec + (double)stamp.tv_nsec * 1e-9;
}

static int compare(const void *left, const void *right) {
    double a = *(const double *)left, b = *(const double *)right;
    return a < b ? -1 : (a > b ? 1 : 0);
}

static double sorted_median(double *values, int count) {
    qsort(values, (size_t)count, sizeof(*values), compare);
    return values[count / 2];
}

static h3_gpu_tensor *zeros(h3_gpu *gpu, size_t elements) {
    h3_gpu_tensor *tensor = h3_gpu_tensor_new_bf16(gpu, elements);
    uint16_t *storage = tensor ? h3_gpu_tensor_bf16_storage(tensor, NULL) : NULL;
    if (!tensor || !storage) {
        fprintf(stderr, "FAIL: cannot allocate %zu bf16 elements\n", elements);
        return NULL;
    }
    memset(storage, 0, elements * sizeof(*storage));
    return tensor;
}

static void buffers_free(buffers *data) {
    h3_gpu_tensor_free(data->input);
    h3_gpu_tensor_free(data->output);
    h3_gpu_tensor_free(data->fc1_weight);
    h3_gpu_tensor_free(data->fc2_weight);
    h3_gpu_tensor_free(data->fc1_result);
    h3_gpu_tensor_free(data->activated);
    memset(data, 0, sizeof(*data));
}

static int allocate(h3_gpu *gpu, uint32_t rows, buffers *data) {
    memset(data, 0, sizeof(*data));
    data->input = zeros(gpu, (size_t)rows * HIDDEN);
    data->output = zeros(gpu, (size_t)rows * HIDDEN);
    data->fc1_weight = zeros(gpu, (size_t)FFN * 2 * HIDDEN);
    data->fc2_weight = zeros(gpu, (size_t)HIDDEN * FFN);
    data->fc1_result = zeros(gpu, (size_t)rows * FFN * 2);
    data->activated = zeros(gpu, (size_t)rows * FFN);
    return data->input && data->output && data->fc1_weight &&
           data->fc2_weight && data->fc1_result && data->activated;
}

static int run_fused(h3_gpu *gpu, const buffers *data, uint32_t rows) {
    return h3_gpu_mlp_bf16(gpu, data->output, data->input, data->fc1_weight,
                           data->fc2_weight, rows, HIDDEN, FFN, HIDDEN);
}

static int run_unfused(h3_gpu *gpu, const buffers *data, uint32_t rows) {
    return h3_gpu_linear_bf16(gpu, data->fc1_result, data->input,
                              data->fc1_weight, NULL, rows, HIDDEN, FFN * 2) &&
           h3_gpu_swiglu_bf16(gpu, data->activated, data->fc1_result, rows, FFN)
               &&
           h3_gpu_linear_bf16(gpu, data->output, data->activated,
                              data->fc2_weight, NULL, rows, FFN, HIDDEN);
}

/* One timed window. The chain is opened outside the clock and closed by the
 * commit+wait inside it, so the measured span holds only the variant's encode,
 * submit and drain, and the chain is left closed for the next call. */
static int measure(h3_gpu *gpu, const buffers *data, uint32_t rows, int fused,
                   double *seconds) {
    if (!h3_gpu_begin(gpu)) {
        fprintf(stderr, "FAIL: cannot begin the chain: %s\n",
                h3_gpu_error(gpu));
        return 0;
    }
    double started = now_seconds();
    int ok = fused ? run_fused(gpu, data, rows) : run_unfused(gpu, data, rows);
    if (!h3_gpu_submit(gpu)) {
        fprintf(stderr, "FAIL: submit failed: %s\n", h3_gpu_error(gpu));
        return 0;
    }
    *seconds = now_seconds() - started;
    if (!ok) {
        fprintf(stderr, "FAIL: %s MLP: %s\n", fused ? "fused" : "unfused",
                h3_gpu_error(gpu));
        return 0;
    }
    return 1;
}

static int sweep(h3_gpu *gpu, uint32_t rows, int rounds) {
    buffers data;
    if (!allocate(gpu, rows, &data)) return 0;
    double one = 0.0;
    if (!measure(gpu, &data, rows, 1, &one) ||
        !measure(gpu, &data, rows, 0, &one)) {
        buffers_free(&data);
        return 0;
    }
    double *fused = malloc((size_t)rounds * sizeof(*fused));
    double *unfused = malloc((size_t)rounds * sizeof(*unfused));
    if (!fused || !unfused) {
        fprintf(stderr, "FAIL: out of memory\n");
        buffers_free(&data);
        return 0;
    }
    printf("rows %u: round-by-round ms (alternating, order flips each round)\n",
           rows);
    for (int round = 0; round < rounds; round++) {
        int first_fused = round % 2 == 0;
        double value = 0.0;
        if (!measure(gpu, &data, rows, first_fused, &value) ||
            !measure(gpu, &data, rows, !first_fused, &one)) {
            free(fused);
            free(unfused);
            buffers_free(&data);
            return 0;
        }
        if (first_fused) { fused[round] = value; unfused[round] = one; }
        else { fused[round] = one; unfused[round] = value; }
        printf("  round %d  fused %8.3f  unfused %8.3f  ratio %6.3f\n",
               round + 1, fused[round] * 1e3, unfused[round] * 1e3,
               fused[round] / unfused[round]);
    }
    double fused_median = sorted_median(fused, rounds) * 1e3;
    double unfused_median = sorted_median(unfused, rounds) * 1e3;
    printf("rows %u medians: fused %.3f ms  unfused %.3f ms  "
           "fused/unfused %.3f  (spread fused %.3f..%.3f, unfused %.3f..%.3f)\n",
           rows, fused_median, unfused_median, fused_median / unfused_median,
           fused[0] * 1e3, fused[rounds - 1] * 1e3, unfused[0] * 1e3,
           unfused[rounds - 1] * 1e3);
    free(fused);
    free(unfused);
    buffers_free(&data);
    return 1;
}

int main(int argc, char **argv) {
    int rounds = argc > 1 ? atoi(argv[1]) : 5;
    if (rounds < 1) rounds = 1;
    char error[512] = {0};
    h3_gpu *gpu = h3_gpu_create("h3_shaders.metal", error, sizeof(error));
    if (!gpu) {
        fprintf(stderr, "FAIL: %s\n", error);
        return 1;
    }
    printf("MLP fusion A/B: HIDDEN %d, FFN %d, rounds %d, int8 %s\n", HIDDEN,
           FFN, rounds, h3_gpu_has_int8_mlp(gpu) ? "available" : "off");
    int ok = 1;
    if (argc > 2) {
        for (int index = 2; index < argc && ok; index++)
            ok = sweep(gpu, (uint32_t)atoi(argv[index]), rounds);
    } else {
        ok = sweep(gpu, 3249, rounds) && sweep(gpu, 7074, rounds);
    }
    h3_gpu_free(gpu);
    printf("%s\n", ok ? "PASS" : "FAIL");
    return ok ? 0 : 1;
}
