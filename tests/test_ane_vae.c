/* Video-VAE projection gate: h3.c's fp32 Metal linear against the same
 * projection on the Neural Engine, both bare and as the engine would run it
 * (Metal pack -> graph -> Metal unpack with the channel bias).
 *
 * usage: h3_ane_vae_test VIDEO_VAE_DIR [ROWS]
 *
 * The decoder is 36 plain ViT blocks (hidden 2048, ffn 8192, weights stored
 * F32) that run entirely in fp32 today, and its weights are re-read from SSD on
 * every decode pass. A Neural Engine graph bakes the weight in as an fp16
 * constexpr, so a block is read once and reused across passes -- which also
 * means one compile per (block, rows) has to amortize. This gate measures both
 * sides of that trade: the speed ratio against the fp32 kernel, and the error
 * of rounding weights and activations down to fp16.
 *
 * Nothing here touches the decoder itself. The bare graph comparison leaves bias
 * out of both sides so it isolates the reduction; the end-to-end comparison
 * carries it, because that is the job the fp32 kernel does today. */

#include "h3_ane_linear.h"
#include "h3_gpu.h"
#include "h3_safetensors.h"
#include "h3_weights.h"

#include <Accelerate/Accelerate.h>
#include <math.h>
#include <stdarg.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

enum {
    HIDDEN = 2048,
    HEAD_DIM = 64,
    HEADS = 32,
    INNER = HEADS * HEAD_DIM,
    FFN = 8192,
    DEFAULT_ROWS = 1797 /* 256-pixel tile: 7 latent frames x lh x lw + 5 */
};

#define PROJECTIONS 4
/* Real decoder block-0 shapes, stored as [output][input]. */
static const char *const WEIGHT_NAMES[PROJECTIONS] = {
    "decoder.transformer_blocks.0.attn.to_qkv.weight",
    "decoder.transformer_blocks.0.attn.to_out.weight",
    "decoder.transformer_blocks.0.ff.w1.weight",
    "decoder.transformer_blocks.0.ff.w2.weight"
};
static const char *const SHORT_NAMES[PROJECTIONS] = {"qkv", "out", "w1", "w2"};
static const uint32_t OUTPUT_DIMS[PROJECTIONS] = {INNER * 3, HIDDEN, FFN * 2, HIDDEN};
static const uint32_t INPUT_DIMS[PROJECTIONS] = {HIDDEN, INNER, HIDDEN, FFN};

static double now_seconds(void) {
    struct timespec now;
    clock_gettime(CLOCK_MONOTONIC, &now);
    return (double)now.tv_sec + (double)now.tv_nsec * 1e-9;
}

static uint64_t rng_state = 0x9E3779B97F4A7C15ull;
static float rng_next(void) {
    rng_state ^= rng_state >> 12;
    rng_state ^= rng_state << 25;
    rng_state ^= rng_state >> 27;
    return (float)((double)rng_state / 18446744073709551616.0) * 4.0f - 2.0f;
}

static int failures;
static void check(int ok, const char *label, const char *format, ...) {
    va_list args;
    printf("%s %s: ", ok ? "ok:  " : "FAIL:", label);
    va_start(args, format);
    vprintf(format, args);
    va_end(args);
    fputc('\n', stdout);
    fflush(stdout);
    if (!ok) failures++;
}

/* The decoder weights enter the graph as fp16 constexprs. Read the stored
 * payload as-is when it already is fp16, otherwise round it and report how far
 * the fp32 values sit from the fp16 range. */
static int load_projection(const h3_weight_store *store, const char *name,
                           uint32_t rows, uint32_t columns,
                           uint16_t **stored, float **widened,
                           double *weight_max, uint64_t *weight_overflow,
                           char *error, size_t error_size) {
    const h3_st_header *header = NULL;
    const h3_st_tensor *tensor = h3_weight_find(store, name, &header);
    if (!tensor) {
        snprintf(error, error_size, "missing weight %s", name);
        return 0;
    }
    if (tensor->ndim != 2 || tensor->shape[0] != rows ||
        tensor->shape[1] != columns) {
        snprintf(error, error_size, "%s is not [%u][%u]", name, rows, columns);
        return 0;
    }
    size_t elements = (size_t)rows * columns;
    *widened = malloc(sizeof(float) * elements);
    if (tensor->dtype == H3_DTYPE_F16) {
        uint16_t *raw = malloc(sizeof(uint16_t) * elements);
        if (!raw || !*widened) {
            snprintf(error, error_size, "oom reading %s", name);
            free(raw);
            return 0;
        }
        if (!h3_st_read_data(header, tensor, raw, sizeof(uint16_t) * elements,
                             error, error_size)) {
            free(raw);
            return 0;
        }
        for (size_t i = 0; i < elements; i++) {
            float value = (float)*(__fp16 *)&raw[i];
            (*widened)[i] = value;
            if (fabs((double)value) > *weight_max)
                *weight_max = fabs((double)value);
        }
        *stored = raw;
        return 1;
    }
    if (tensor->dtype == H3_DTYPE_F32) {
        uint16_t *half = malloc(sizeof(uint16_t) * elements);
        if (!half || !*widened) {
            snprintf(error, error_size, "oom reading %s", name);
            free(half);
            return 0;
        }
        if (!h3_st_read_data(header, tensor, *widened, sizeof(float) * elements,
                             error, error_size)) {
            free(half);
            return 0;
        }
        for (size_t i = 0; i < elements; i++) {
            double value = fabs((double)(*widened)[i]);
            if (value > *weight_max) *weight_max = value;
            if (value > 65504.0) (*weight_overflow)++;
            __fp16 rounded = (__fp16)(*widened)[i];
            memcpy(&half[i], &rounded, sizeof(rounded));
        }
        *stored = half;
        return 1;
    }
    snprintf(error, error_size, "%s has unsupported dtype %s", name,
             h3_dtype_name(tensor->dtype));
    return 0;
}

/* The decoder's projections all carry an F32 channel bias, which the Neural
 * Engine graph leaves to the unpack kernel. */
static int load_bias(const h3_weight_store *store, const char *weight_name,
                     uint32_t count, float **values,
                     char *error, size_t error_size) {
    const char *suffix = strrchr(weight_name, '.');
    char bias_name[192];
    if (!suffix) {
        snprintf(error, error_size, "%s has no suffix", weight_name);
        return 0;
    }
    snprintf(bias_name, sizeof(bias_name), "%.*s.bias",
             (int)(suffix - weight_name), weight_name);
    const h3_st_header *header = NULL;
    const h3_st_tensor *tensor = h3_weight_find(store, bias_name, &header);
    if (!tensor || tensor->ndim != 1 || tensor->shape[0] != count ||
        tensor->dtype != H3_DTYPE_F32) {
        snprintf(error, error_size, "%s is not [%u] F32", bias_name, count);
        return 0;
    }
    float *bias = malloc(sizeof(float) * count);
    if (!bias) {
        snprintf(error, error_size, "oom reading %s", bias_name);
        return 0;
    }
    if (!h3_st_read_data(header, tensor, bias, sizeof(float) * count,
                         error, error_size)) {
        free(bias);
        return 0;
    }
    *values = bias;
    return 1;
}

/* out[row][n] = sum_k weight[n][k] * x[row][k], fp32 through the same BLAS call
 * the engine's fp32 kernels sit on. */
static void reference_gemm(const float *weight, const float *x, float *out,
                           uint32_t rows, uint32_t input_dim,
                           uint32_t output_dim) {
    const float one = 1.0f, zero = 0.0f;
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, (int)rows,
                (int)output_dim, (int)input_dim, one, x, (int)input_dim,
                weight, (int)input_dim, zero, out, (int)output_dim);
}

int main(int argc, char **argv) {
    if (argc < 2) {
        fprintf(stderr, "usage: %s VIDEO_VAE_DIR [ROWS]\n", argv[0]);
        return 2;
    }
    uint32_t rows = argc > 2 ? (uint32_t)atoi(argv[2]) : DEFAULT_ROWS;
    if (!h3_ane_linear_available()) {
        fprintf(stderr, "the Neural Engine bridge is unavailable\n");
        return 2;
    }
    char error[512] = {0};
    h3_weight_store *store = h3_weight_store_open(argv[1], error, sizeof(error));
    if (!store) { fprintf(stderr, "%s\n", error); return 2; }
    h3_gpu *gpu = h3_gpu_create("h3_shaders.metal", error, sizeof(error));
    if (!gpu) {
        fprintf(stderr, "cannot create the Metal context: %s\n", error);
        h3_weight_store_free(store);
        return 2;
    }

    /* One buffer sized for the widest input (w2 reads FFN columns); every
     * projection strides through it with its own input_dim. */
    float *x = malloc(sizeof(float) * (size_t)rows * FFN);
    if (!x) { fprintf(stderr, "oom activations\n"); return 2; }
    for (size_t i = 0; i < (size_t)rows * FFN; i++) x[i] = rng_next();

    printf("video VAE block 0, rows=%u (weights stored F32, rounded to fp16 "
           "for the graph)\n", rows);
    double compile_total = 0.0, weight_total = 0.0;
    double gemm_metal = 0.0, gemm_ane = 0.0;
    double e2e_metal = 0.0, e2e_ane = 0.0;
    int all_pass = 1;
    for (int p = 0; p < PROJECTIONS; p++) {
        uint32_t input_dim = INPUT_DIMS[p], output_dim = OUTPUT_DIMS[p];
        uint16_t *stored = NULL;
        float *widened = NULL;
        double weight_max = 0.0;
        uint64_t weight_overflow = 0;
        if (!load_projection(store, WEIGHT_NAMES[p], output_dim, input_dim,
                             &stored, &widened, &weight_max, &weight_overflow,
                             error, sizeof(error))) {
            fprintf(stderr, "%s\n", error);
            return 2;
        }
        /* The wiring reads the same tensors through h3_weights; a drift here
         * would put different numbers in the graph than this gate measured. */
        uint16_t *loader = NULL;
        if (h3_weight_load_f16_raw(store, WEIGHT_NAMES[p], output_dim,
                                   input_dim, &loader, error, sizeof(error))) {
            size_t elements = (size_t)output_dim * input_dim;
            check(loader && memcmp(loader, stored, sizeof(uint16_t) * elements) == 0,
                  SHORT_NAMES[p], "h3_weight_load_f16_raw matches this gate's "
                  "fp16 payload (%llu halves)",
                  (unsigned long long)elements);
        } else {
            check(0, SHORT_NAMES[p], "h3_weight_load_f16_raw: %s", error);
        }
        free(loader);
        h3_gpu_tensor *weight = h3_gpu_tensor_from_f32(gpu, widened,
            (size_t)output_dim * input_dim);
        h3_gpu_tensor *input = h3_gpu_tensor_from_f32(gpu, x,
            (size_t)rows * input_dim);
        h3_gpu_tensor *metal_out = h3_gpu_tensor_new_f32(gpu,
            (size_t)rows * output_dim);
        if (!weight || !input || !metal_out) {
            fprintf(stderr, "%s: cannot build Metal tensors: %s\n", SHORT_NAMES[p],
                    h3_gpu_error(gpu));
            return 2;
        }

        /* fp32 references: one through the engine's own kernel, one through
         * BLAS on the host, so a disagreement can be attributed. */
        float *want = malloc(sizeof(float) * (size_t)rows * output_dim);
        if (!want) { fprintf(stderr, "oom reference\n"); return 2; }
        reference_gemm(widened, x, want, (int)rows, (int)input_dim,
                       (int)output_dim);
        double peak = 0.0;
        for (size_t i = 0; i < (size_t)rows * output_dim; i++)
            if (fabs((double)want[i]) > peak) peak = fabs((double)want[i]);

        if (!h3_gpu_begin(gpu) ||
            !h3_gpu_linear_f32(gpu, metal_out, input, weight, NULL, rows,
                               input_dim, output_dim) || !h3_gpu_submit(gpu)) {
            fprintf(stderr, "%s: metal linear failed: %s\n", SHORT_NAMES[p],
                    h3_gpu_error(gpu));
            return 2;
        }
        float *got = malloc(sizeof(float) * (size_t)rows * output_dim);
        if (!got || !h3_gpu_tensor_read_f32(metal_out, got,
                                            (size_t)rows * output_dim)) {
            fprintf(stderr, "%s: cannot read back the Metal output\n",
                    SHORT_NAMES[p]);
            return 2;
        }
        double metal_rel = 0.0, metal_max = 0.0, energy = 0.0;
        for (size_t i = 0; i < (size_t)rows * output_dim; i++) {
            double d = (double)got[i] - (double)want[i];
            energy += (double)want[i] * want[i];
            metal_rel += d * d;
            if (fabs(d) > metal_max) metal_max = fabs(d);
        }
        metal_rel = sqrt(metal_rel / (energy + 1e-30));
        check(metal_rel <= 1e-5, SHORT_NAMES[p],
              "metal fp32 matches BLAS rel_l2=%.3e max_abs=%.3e", metal_rel,
              metal_max);

        /* The Neural Engine graph. */
        uint32_t kc = h3_ane_linear_default_chunk(input_dim);
        h3_ane_linear *linear = h3_ane_linear_create(SHORT_NAMES[p], stored,
            H3_ANE_W_F16, input_dim, output_dim, rows, kc, error, sizeof(error));
        if (!linear) {
            fprintf(stderr, "%s: ANE create failed: %s\n", SHORT_NAMES[p], error);
            return 2;
        }
        uint32_t chunks = h3_ane_linear_chunks(linear);
        uint32_t chunk_dim = h3_ane_linear_chunk_dim(linear);
        uint32_t plane_rows = h3_ane_linear_plane_rows(linear);
        double fill_seconds = 0.0, eval_best = 1e30, eval_total = 0.0;
        int repeats = 5;
        for (int r = 0; r < repeats; r++) {
            double started = now_seconds();
            for (uint32_t c = 0; c < chunks; c++) {
                float *plane = h3_ane_linear_input(linear, c);
                uint32_t base = c * chunk_dim;
                uint32_t span = input_dim > base ?
                    (input_dim - base < chunk_dim ? input_dim - base : chunk_dim) : 0;
                for (uint32_t k = 0; k < span; k++) {
                    /* Plane row k holds feature (base+k) of every activation
                     * row, so the fill gathers across the row-major stride. */
                    float *row = plane + (size_t)k * plane_rows;
                    for (uint32_t row_index = 0; row_index < rows; row_index++)
                        row[row_index] =
                            x[(size_t)row_index * input_dim + base + k];
                }
                for (uint32_t k = span; k < chunk_dim; k++)
                    memset(plane + (size_t)k * plane_rows, 0,
                           sizeof(float) * plane_rows);
            }
            fill_seconds += now_seconds() - started;
            started = now_seconds();
            if (!h3_ane_linear_eval(linear, error, sizeof(error))) {
                fprintf(stderr, "%s: ANE eval failed: %s\n", SHORT_NAMES[p], error);
                return 2;
            }
            double elapsed = now_seconds() - started;
            eval_total += elapsed;
            if (elapsed < eval_best) eval_best = elapsed;
        }
        /* Metal fp32 timing for the same projection. */
        double metal_best = 1e30, metal_total = 0.0;
        for (int r = 0; r < repeats; r++) {
            double started = now_seconds();
            if (!h3_gpu_begin(gpu) ||
                !h3_gpu_linear_f32(gpu, metal_out, input, weight, NULL, rows,
                                   input_dim, output_dim) ||
                !h3_gpu_submit(gpu)) {
                fprintf(stderr, "%s: metal timing failed: %s\n", SHORT_NAMES[p],
                        h3_gpu_error(gpu));
                return 2;
            }
            double elapsed = now_seconds() - started;
            metal_total += elapsed;
            if (elapsed < metal_best) metal_best = elapsed;
        }

        /* Compare the graph output against the fp32 reference. */
        const float *result = h3_ane_linear_output(linear);
        double dot = 0.0, left = 0.0, right = 0.0, rel = 0.0, worst = 0.0;
        size_t nonfinite = 0, overflow = 0;
        for (uint32_t n = 0; n < output_dim; n++)
            for (uint32_t row = 0; row < rows; row++) {
                double g = result[(size_t)n * plane_rows + row];
                double w = want[(size_t)row * output_dim + n];
                if (!isfinite(g)) { nonfinite++; continue; }
                if (fabs(w) > 65504.0) overflow++;
                dot += g * w; left += g * g; right += w * w;
                rel += (g - w) * (g - w);
                if (fabs(g - w) > worst) worst = fabs(g - w);
            }
        double cosine = dot / (sqrt(left) * sqrt(right) + 1e-30);
        double relative = sqrt(rel / (right + 1e-30));
        double compile = h3_ane_linear_compile_seconds(linear);
        double weights = (double)h3_ane_linear_weight_bytes(linear);
        compile_total += compile;
        weight_total += weights;
        gemm_metal += metal_best;
        gemm_ane += eval_best;
        int pass = relative <= 1e-3 && nonfinite == 0 && weight_overflow == 0;
        all_pass = all_pass && pass;
        printf("  %-4s K=%u N=%u kc=%u/%u cos=%.6f rel_l2=%.3e max_abs=%.3e\n"
               "        ref peak=%.3e | weight max=%.3e fp16-unrepresentable="
               "%llu out-above-fp16=%zu nonfinite=%zu\n"
               "        metal fp32 best=%.2f ms mean=%.2f ms | "
               "ane best=%.2f ms mean=%.2f ms ratio=%.2fx | cpu plane fill="
               "%.2f ms (harness only)\n"
               "        compile=%.2fs blob=%.1f MiB %s\n",
               SHORT_NAMES[p], input_dim, output_dim, chunk_dim, chunks,
               cosine, relative, worst,
               peak, weight_max, (unsigned long long)weight_overflow,
               overflow, nonfinite,
               metal_best * 1e3, metal_total / repeats * 1e3,
               eval_best * 1e3, eval_total / repeats * 1e3,
               metal_best / eval_best, fill_seconds / repeats * 1e3,
               compile, weights / (1024.0 * 1024.0), pass ? "PASS" : "FAIL");
        fflush(stdout);

        /* --- the path the engine would actually take: staging included -----
         * Bias rides on the unpack, so the fp32 comparison has to carry it too
         * or the two sides are not doing the same job.
         *
         * The bare graph is released first: the projection rebuilds an identical
         * MIL + weight set, and the bridge keys its staging directory on the
         * model's content identifier, so two live handles would share one. */
        h3_ane_linear_free(linear);
        linear = NULL;
        float *bias_values = NULL;
        if (!load_bias(store, WEIGHT_NAMES[p], output_dim, &bias_values,
                       error, sizeof(error))) {
            fprintf(stderr, "%s\n", error);
            return 2;
        }
        h3_gpu_tensor *bias = h3_gpu_tensor_from_f32(gpu, bias_values,
            output_dim);
        h3_ane_projection *projection = h3_ane_projection_create_f16(gpu,
            SHORT_NAMES[p], stored, input_dim, output_dim, rows, kc,
            error, sizeof(error));
        if (!bias || !projection) {
            fprintf(stderr, "%s: cannot build the end-to-end path: %s%s\n",
                    SHORT_NAMES[p], error, h3_gpu_error(gpu));
            return 2;
        }
        double metal_bias_best = 1e30;
        for (int r = 0; r < repeats; r++) {
            double started = now_seconds();
            if (!h3_gpu_begin(gpu) ||
                !h3_gpu_linear_f32(gpu, metal_out, input, weight, bias, rows,
                                   input_dim, output_dim) ||
                !h3_gpu_submit(gpu)) {
                fprintf(stderr, "%s: metal+bias timing failed: %s\n",
                        SHORT_NAMES[p], h3_gpu_error(gpu));
                return 2;
            }
            double elapsed = now_seconds() - started;
            if (elapsed < metal_bias_best) metal_bias_best = elapsed;
        }
        double e2e_best = 1e30;
        for (int r = 0; r < repeats; r++) {
            double started = now_seconds();
            if (!h3_gpu_begin(gpu) ||
                !h3_ane_projection_apply_f32(projection, gpu, metal_out, input,
                                            bias, error, sizeof(error)) ||
                !h3_gpu_submit(gpu)) {
                fprintf(stderr, "%s: end-to-end eval failed: %s\n",
                        SHORT_NAMES[p], error);
                return 2;
            }
            double elapsed = now_seconds() - started;
            if (elapsed < e2e_best) e2e_best = elapsed;
        }
        if (!h3_gpu_tensor_read_f32(metal_out, got,
                                    (size_t)rows * output_dim)) {
            fprintf(stderr, "%s: cannot read back the end-to-end output\n",
                    SHORT_NAMES[p]);
            return 2;
        }
        double e2e_rel = 0.0, e2e_max = 0.0, e2e_energy = 0.0;
        for (uint32_t row = 0; row < rows; row++)
            for (uint32_t n = 0; n < output_dim; n++) {
                double want_biased =
                    (double)want[(size_t)row * output_dim + n] + bias_values[n];
                double difference =
                    (double)got[(size_t)row * output_dim + n] - want_biased;
                e2e_energy += want_biased * want_biased;
                e2e_rel += difference * difference;
                if (fabs(difference) > e2e_max) e2e_max = fabs(difference);
            }
        e2e_rel = sqrt(e2e_rel / (e2e_energy + 1e-30));
        double e2e_ratio = metal_bias_best / e2e_best;
        /* The per-projection bar is "no regression": one small projection sitting
         * below the block threshold is still a win in absolute terms, and the
         * block total is what decides whether the decoder goes to the Neural
         * Engine at all. */
        int e2e_pass = e2e_ratio >= 1.2 && e2e_rel <= 1e-3;
        all_pass = all_pass && e2e_pass;
        e2e_metal += metal_bias_best;
        e2e_ane += e2e_best;
        printf("  %-4s e2e: metal fp32+bias best=%.2f ms | pack+ane+unpack+bias "
               "best=%.2f ms ratio=%.2fx (staging %.2f ms) | rel_l2=%.3e "
               "max_abs=%.3e %s\n",
               SHORT_NAMES[p], metal_bias_best * 1e3, e2e_best * 1e3, e2e_ratio,
               (e2e_best - eval_best) * 1e3, e2e_rel, e2e_max,
               e2e_pass ? "ok" : "REGRESSION");
        fflush(stdout);

        h3_ane_projection_free(projection);
        h3_gpu_tensor_free(bias);
        free(bias_values);
        free(got);
        free(want);
        free(stored);
        free(widened);
        h3_gpu_tensor_free(weight);
        h3_gpu_tensor_free(input);
        h3_gpu_tensor_free(metal_out);
    }
    printf("block total: metal fp32=%.1f ms ane=%.1f ms (%.2fx)\n"
           "        one decode pass of 36 blocks: %.2f s -> %.2f s\n"
           "        compile=%.1fs cached bytes=%.0f MiB -> 36 blocks = %.1f GiB "
           "per rows-shape\n",
           gemm_metal * 1e3, gemm_ane * 1e3, gemm_metal / gemm_ane,
           gemm_metal * 36.0, gemm_ane * 36.0,
           compile_total, weight_total / (1024.0 * 1024.0),
           weight_total * 36.0 / (1024.0 * 1024.0 * 1024.0));
    /* The gate that matters: the engine path with its staging and bias, against
     * the fp32 kernel that does the same job today. */
    double e2e_ratio = e2e_metal / e2e_ane;
    check(e2e_ratio >= 2.0, "block e2e",
          "metal fp32+bias=%.1f ms ane path=%.1f ms ratio=%.2fx -> one pass of "
          "36 blocks %.2f s -> %.2f s", e2e_metal * 1e3, e2e_ane * 1e3,
          e2e_ratio, e2e_metal * 36.0, e2e_ane * 36.0);
    free(x);
    h3_gpu_free(gpu);
    h3_weight_store_free(store);
    return all_pass && !failures ? 0 : 1;
}
