/* INT8 ConvRot gates for the Neural Engine projection module.
 *
 * usage: h3_ane_int8_test [CHECKPOINT_DIR]
 *
 * Synthetic gates mint int8_tensorwise payloads with the reference formula in
 * double precision and compare the ANE result against a double replay of the
 * stored contract (fp16-rounded scale, dequantize, derotate, matmul). With a
 * checkpoint directory, the full block-0 qkv projection runs on real weights. */

#include "h3_ane_linear.h"
#include "h3_convrot.h"
#include "h3_weights.h"
#include "h3_safetensors.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

static unsigned failures;

static uint64_t rng_state = 0x9E3779B97F4A7C15ull;
static double rng_uniform(void) {
    rng_state ^= rng_state >> 12;
    rng_state ^= rng_state << 25;
    rng_state ^= rng_state >> 27;
    return ((double)((rng_state * 2685821657736338717ull) >> 11) /
            9007199254740992.0) - 0.5;
}

/* Double replay of the stored contract for one projection. */
static double *reference(const int8_t *quantized, const float *scales,
                         int group_size, uint32_t input_dim,
                         uint32_t output_dim, const float *x, uint32_t rows) {
    const float *table = group_size ? h3_convrot_hadamard(group_size) : NULL;
    double *weights = malloc(sizeof(double) * output_dim * input_dim);
    double *result = malloc(sizeof(double) * output_dim * rows);
    if (!weights || !result) { free(weights); free(result); return NULL; }
    for (uint32_t n = 0; n < output_dim; n++) {
        double scale = (double)(__fp16)scales[n];
        for (uint32_t k = 0; k < input_dim; k++) {
            if (!group_size) {
                weights[(size_t)n * input_dim + k] =
                    (double)quantized[(size_t)n * input_dim + k] * scale;
                continue;
            }
            uint32_t group = k / (uint32_t)group_size;
            uint32_t j = k % (uint32_t)group_size;
            double acc = 0;
            for (int i = 0; i < group_size; i++)
                acc += (double)quantized[(size_t)n * input_dim +
                                         group * group_size + i] * scale *
                       (double)table[i * group_size + (int)j];
            weights[(size_t)n * input_dim + k] = acc;
        }
    }
    for (uint32_t n = 0; n < output_dim; n++)
        for (uint32_t s = 0; s < rows; s++) {
            double acc = 0;
            for (uint32_t k = 0; k < input_dim; k++)
                acc += weights[(size_t)n * input_dim + k] *
                       (double)x[(size_t)k * rows + s];
            result[(size_t)n * rows + s] = acc;
        }
    free(weights);
    return result;
}

static void gate(const char *name, const int8_t *quantized,
                 const float *scales, int group_size, uint32_t input_dim,
                 uint32_t output_dim, uint32_t rows, uint32_t kc) {
    char error[256] = {0};
    h3_ane_linear *linear = h3_ane_linear_create_int8(
        name, quantized, scales, group_size, input_dim, output_dim, rows, kc,
        error, sizeof(error));
    if (!linear) {
        printf("FAIL %s create: %s\n", name, error);
        failures++;
        return;
    }
    uint32_t chunks = h3_ane_linear_chunks(linear);
    uint32_t chunk_dim = h3_ane_linear_chunk_dim(linear);
    uint32_t plane_rows = h3_ane_linear_plane_rows(linear);
    float *x = malloc(sizeof(float) * input_dim * rows);
    for (size_t i = 0; i < (size_t)input_dim * rows; i++)
        x[i] = (float)(rng_uniform() * 4.0);
    for (uint32_t c = 0; c < chunks; c++) {
        float *plane = h3_ane_linear_input(linear, c);
        memset(plane, 0, h3_ane_linear_input_bytes(linear));
        uint32_t base = c * chunk_dim;
        uint32_t span = input_dim > base ?
            (input_dim - base < chunk_dim ? input_dim - base : chunk_dim) : 0;
        for (uint32_t k = 0; k < span; k++)
            memcpy(plane + (size_t)k * plane_rows,
                   x + (size_t)(base + k) * rows, sizeof(float) * rows);
    }
    if (!h3_ane_linear_eval(linear, error, sizeof(error))) {
        printf("FAIL %s eval: %s\n", name, error);
        failures++;
        h3_ane_linear_free(linear);
        free(x);
        return;
    }
    double *want = reference(quantized, scales, group_size, input_dim,
                             output_dim, x, rows);
    const float *got = h3_ane_linear_output(linear);
    double dot = 0, got_norm = 0, want_norm = 0, error_sum = 0, scale_sum = 0;
    double worst = 0;
    size_t nonfinite = 0;
    for (uint32_t n = 0; n < output_dim; n++)
        for (uint32_t s = 0; s < rows; s++) {
            double a = got[(size_t)n * plane_rows + s];
            double b = want[(size_t)n * rows + s];
            if (!isfinite(a)) { nonfinite++; continue; }
            dot += a * b;
            got_norm += a * a;
            want_norm += b * b;
            double difference = a - b;
            error_sum += difference * difference;
            scale_sum += b * b;
            if (fabs(difference) > worst) worst = fabs(difference);
        }
    double cosine = dot / (sqrt(got_norm) * sqrt(want_norm) + 1e-30);
    double relative = sqrt(error_sum / (scale_sum + 1e-30));
    int pass = cosine >= 0.9999 && relative <= 2e-3 && nonfinite == 0;
    printf("%-12s K=%u N=%u kc=%u rows=%u gs=%d: cos=%.7f rel_l2=%.3e "
           "max_abs=%.3e nonfinite=%zu weights=%.1f MiB compile=%.2fs cache=%d %s\n",
           name, input_dim, output_dim, kc, rows, group_size, cosine, relative,
           worst, nonfinite,
           (double)h3_ane_linear_weight_bytes(linear) / (1024.0 * 1024.0),
           h3_ane_linear_compile_seconds(linear),
           h3_ane_linear_cache_hit(linear) ? 1 : 0,
           pass ? "PASS" : "FAIL");
    if (!pass) failures++;
    h3_ane_linear_free(linear);
    free(x);
    free(want);
}

static void synthetic(const char *name, uint32_t input_dim,
                      uint32_t output_dim, uint32_t rows, uint32_t kc,
                      int group_size) {
    const float *table = group_size ? h3_convrot_hadamard(group_size) : NULL;
    double *original = malloc(sizeof(double) * output_dim * input_dim);
    double *stored = malloc(sizeof(double) * output_dim * input_dim);
    int8_t *quantized = malloc((size_t)output_dim * input_dim);
    float *scales = malloc(sizeof(float) * output_dim);
    for (size_t i = 0; i < (size_t)output_dim * input_dim; i++)
        original[i] = rng_uniform();
    for (uint32_t n = 0; n < output_dim; n++)
        for (uint32_t k = 0; k < input_dim; k++) {
            if (!group_size) {
                stored[(size_t)n * input_dim + k] =
                    original[(size_t)n * input_dim + k];
                continue;
            }
            uint32_t group = k / (uint32_t)group_size;
            uint32_t i = k % (uint32_t)group_size;
            double acc = 0;
            for (int j = 0; j < group_size; j++)
                acc += original[(size_t)n * input_dim + group * group_size + j] *
                       (double)table[(int)i * group_size + j];
            stored[(size_t)n * input_dim + k] = acc;
        }
    for (uint32_t n = 0; n < output_dim; n++) {
        double amax = 0;
        for (uint32_t k = 0; k < input_dim; k++)
            amax = fmax(amax, fabs(stored[(size_t)n * input_dim + k]));
        scales[n] = (float)(amax / 127.0);
        for (uint32_t k = 0; k < input_dim; k++) {
            double q = round(stored[(size_t)n * input_dim + k] /
                             (double)scales[n]);
            quantized[(size_t)n * input_dim + k] =
                (int8_t)fmax(-128.0, fmin(127.0, q));
        }
    }
    gate(name, quantized, scales, group_size, input_dim, output_dim, rows, kc);
    free(original);
    free(stored);
    free(quantized);
    free(scales);
}

/* Load into a fresh linear, eval a fixed input, return the raw output copy. */
static float *cache_run(const char *name, const int8_t *quantized,
                        const float *scales, uint32_t input_dim,
                        uint32_t output_dim, uint32_t rows, bool *hit,
                        bool remove_dir, double *seconds) {
    char error[256] = {0};
    h3_ane_linear *linear = h3_ane_linear_create_int8(
        name, quantized, scales, 256, input_dim, output_dim, rows,
        input_dim / 2, error, sizeof(error));
    if (!linear) {
        printf("FAIL %s create: %s\n", name, error);
        failures++;
        return NULL;
    }
    *hit = h3_ane_linear_cache_hit(linear);
    *seconds = h3_ane_linear_compile_seconds(linear);
    uint32_t plane_rows = h3_ane_linear_plane_rows(linear);
    uint32_t chunk_dim = h3_ane_linear_chunk_dim(linear);
    for (uint32_t chunk = 0; chunk < h3_ane_linear_chunks(linear); chunk++) {
        float *plane = h3_ane_linear_input(linear, chunk);
        memset(plane, 0, h3_ane_linear_input_bytes(linear));
        for (uint32_t j = 0; j < chunk_dim; j++) {
            uint32_t k = chunk * chunk_dim + j;
            if (k >= input_dim) break;
            for (uint32_t s = 0; s < rows; s++)
                plane[(size_t)j * plane_rows + s] =
                    (float)((int)((k * 131 + s * 17) % 97) - 48) / 24.0f;
        }
    }
    float *output = NULL;
    if (!h3_ane_linear_eval(linear, error, sizeof(error))) {
        printf("FAIL %s eval: %s\n", name, error);
        failures++;
    } else {
        size_t bytes = h3_ane_linear_output_bytes(linear);
        output = malloc(bytes);
        memcpy(output, h3_ane_linear_output(linear), bytes);
    }
    if (remove_dir) setenv("H3_ANE_CACHE", "0", 1);
    h3_ane_linear_free(linear);
    if (remove_dir) setenv("H3_ANE_CACHE", "1", 1);
    return output;
}

/* Compile-cache gates: a recreate must hit and reproduce the first run
 * bit-exactly; same-shape different weights must MISS (the content hash
 * covers the weights) and still match their own reference. */
static void cache_gates(void) {
    enum { K = 1024, N = 256, ROWS = 16 };
    /* These gates need the cache on, but the caller's choice must survive the
     * function: leaving it set makes the real-weight gates below report a
     * cross-process cache hit and keeps ~130 MiB of artifacts per shape that
     * `H3_ANE_CACHE=0` promises not to write. */
    const char *requested = getenv("H3_ANE_CACHE");
    char prior[16];
    int had_prior = requested != NULL;
    snprintf(prior, sizeof(prior), "%s", requested ? requested : "");
    setenv("H3_ANE_CACHE", "1", 1);
    int8_t *quantized = malloc((size_t)N * K);
    float *scales = malloc(sizeof(float) * N);
    uint32_t salt = (uint32_t)time(NULL);
    for (size_t i = 0; i < (size_t)N * K; i++)
        quantized[i] = (int8_t)(((i * 2654435761u + salt) >> 16) % 255 - 127);
    for (uint32_t n = 0; n < N; n++) scales[n] = 0.01f + 1e-5f * (float)n;
    /* The reference gate compiles fresh, passes numerics, and its cache-on
     * free seeds the entry, so both recreates below must load from cache. */
    gate("cache-shape", quantized, scales, 256, K, N, ROWS, K / 2);
    bool hit1 = false, hit2 = false, hit3 = true, scratch = false;
    double first = 0, second = 0, third = 0, ignored = 0;
    float *a = cache_run("cache-a", quantized, scales, K, N, ROWS, &hit1,
                         false, &first);
    float *b = cache_run("cache-b", quantized, scales, K, N, ROWS, &hit2,
                         false, &second);
    float kept = scales[0];
    scales[0] *= 2.0f;
    float *c = cache_run("cache-c", quantized, scales, K, N, ROWS, &hit3,
                         false, &third);
    /* Cleanup loads: a cache-off free evicts each content entry. */
    free(cache_run("cache-rm-c", quantized, scales, K, N, ROWS, &scratch,
                   true, &ignored));
    scales[0] = kept;
    free(cache_run("cache-rm-a", quantized, scales, K, N, ROWS, &scratch,
                   true, &ignored));
    int bitexact = a && b &&
        !memcmp(a, b, sizeof(float) * N * ((ROWS + 15) / 16 * 16));
    /* c must differ from a: a collision would silently reuse a's model. */
    int distinct = a && c && memcmp(a, c, sizeof(float) * N * ROWS) != 0;
    int pass = a && b && c && hit1 && hit2 && !hit3 && bitexact && distinct;
    printf("cache        hit=%.3fs rehit=%.3fs miss=%.3fs hits=%d/%d/%d "
           "bitexact=%d distinct=%d %s\n",
           first, second, third, hit1, hit2, hit3, bitexact, distinct,
           pass ? "PASS" : "FAIL");
    if (!pass) failures++;
    if (had_prior) setenv("H3_ANE_CACHE", prior, 1);
    else unsetenv("H3_ANE_CACHE");
    free(a);
    free(b);
    free(c);
    free(quantized);
    free(scales);
}

/* y = row . R for every row, using the grouped Hadamard the graph applies to
 * the activations. Used both to build the stored rotated weights and, for the
 * LoRA A factor, to pre-rotate it so the bypass can share those activations. */
static void rotate_rows(const double *src, double *dst, uint32_t n_rows,
                        uint32_t input_dim, int group_size) {
    const float *table = group_size ? h3_convrot_hadamard(group_size) : NULL;
    for (uint32_t n = 0; n < n_rows; n++)
        for (uint32_t k = 0; k < input_dim; k++) {
            if (!group_size) {
                dst[(size_t)n * input_dim + k] = src[(size_t)n * input_dim + k];
                continue;
            }
            uint32_t group = k / (uint32_t)group_size;
            uint32_t j = k % (uint32_t)group_size;
            double acc = 0;
            for (int i = 0; i < group_size; i++)
                acc += src[(size_t)n * input_dim + group * group_size + i] *
                       (double)table[i * group_size + j];
            dst[(size_t)n * input_dim + k] = acc;
        }
}

/* comfy-quants int8_tensorwise: per-row amax / 127. */
static void quantize_rows(const double *src, int8_t *dst, float *scales,
                          uint32_t n_rows, uint32_t cols) {
    for (uint32_t n = 0; n < n_rows; n++) {
        double amax = 0;
        for (uint32_t k = 0; k < cols; k++)
            amax = fmax(amax, fabs(src[(size_t)n * cols + k]));
        scales[n] = (float)(amax / 127.0);
        for (uint32_t k = 0; k < cols; k++)
            dst[(size_t)n * cols + k] = (int8_t)fmax(-128.0, fmin(127.0,
                round(src[(size_t)n * cols + k] / (double)scales[n])));
    }
}

/* Gate the fused bypass branch: the graph only ever sees R x, so A is fed
 * pre-rotated (A_rot = A R) and the result must still be
 *
 *     W x  +  scale . B (A x)
 *
 * lora_scale is deliberately large: this gate proves the branch is wired up and
 * shares the rotated activations, NOT that a 0.2% delta survives fp16. */
static void bypass_gate(const char *name, uint32_t input_dim,
                        uint32_t output_dim, uint32_t rows, uint32_t kc,
                        int group_size, uint32_t rank, double lora_scale,
                        int mode) {
    size_t wn = (size_t)output_dim * input_dim;
    double *original = malloc(sizeof(double) * wn);
    double *stored = malloc(sizeof(double) * wn);
    double *lora_a = malloc(sizeof(double) * rank * input_dim);
    double *lora_a_rot = malloc(sizeof(double) * rank * input_dim);
    double *lora_b = malloc(sizeof(double) * output_dim * rank);
    int8_t *quantized = malloc(wn);
    float *scales = malloc(sizeof(float) * output_dim);
    int8_t *a_rot_q = malloc((size_t)rank * input_dim);
    int8_t *b_q = malloc((size_t)output_dim * rank);
    float *a_rot_s = malloc(sizeof(float) * rank);
    float *b_s = malloc(sizeof(float) * output_dim);
    double *a_recon = malloc(sizeof(double) * rank * input_dim);
    double *a_eff = malloc(sizeof(double) * rank * input_dim);
    double *b_recon = malloc(sizeof(double) * output_dim * rank);
    float *x = malloc(sizeof(float) * input_dim * rows);
    if (!original || !stored || !lora_a || !lora_a_rot || !lora_b ||
        !quantized || !scales || !a_rot_q || !b_q || !a_rot_s || !b_s ||
        !a_recon || !a_eff || !b_recon || !x) {
        printf("FAIL %s out of memory\n", name);
        failures++;
        return;
    }
    for (size_t i = 0; i < wn; i++) original[i] = rng_uniform();
    for (size_t i = 0; i < (size_t)rank * input_dim; i++)
        lora_a[i] = rng_uniform();
    /* Folded into B, not applied to the reference: the graph sees only the
     * factors, so lora_scale sets how big the bypass term is and nothing else. */
    for (size_t i = 0; i < (size_t)output_dim * rank; i++)
        lora_b[i] = rng_uniform() * lora_scale;
    for (size_t i = 0; i < (size_t)input_dim * rows; i++)
        x[i] = (float)(rng_uniform() * 4.0);
    rotate_rows(original, stored, output_dim, input_dim, group_size);
    rotate_rows(lora_a, lora_a_rot, rank, input_dim, group_size);
    for (uint32_t n = 0; n < output_dim; n++) {
        double amax = 0;
        for (uint32_t k = 0; k < input_dim; k++)
            amax = fmax(amax, fabs(stored[(size_t)n * input_dim + k]));
        scales[n] = (float)(amax / 127.0);
        for (uint32_t k = 0; k < input_dim; k++)
            quantized[(size_t)n * input_dim + k] =
                (int8_t)fmax(-128.0, fmin(127.0,
                    round(stored[(size_t)n * input_dim + k] / (double)scales[n])));
    }
    quantize_rows(lora_a_rot, a_rot_q, a_rot_s, rank, input_dim);
    quantize_rows(lora_b, b_q, b_s, output_dim, rank);
    /* mode 1 zeroes B and mode 2 zeroes A, so the bypass contributes nothing
     * and the output must collapse to the plain main convolution. */
    if (mode == 1) memset(b_q, 0, (size_t)output_dim * rank);
    if (mode == 2) memset(a_rot_q, 0, (size_t)rank * input_dim);
    /* The reference must see what the graph sees: rebuild the quantized A_rot
     * and derotate it back, since R is symmetric and the graph eats R x. */
    for (uint32_t r = 0; r < rank; r++)
        for (uint32_t k = 0; k < input_dim; k++)
            a_recon[(size_t)r * input_dim + k] =
                (double)a_rot_q[(size_t)r * input_dim + k] *
                (double)(__fp16)a_rot_s[r];
    rotate_rows(a_recon, a_eff, rank, input_dim, group_size);
    for (uint32_t n = 0; n < output_dim; n++)
        for (uint32_t r = 0; r < rank; r++)
            b_recon[(size_t)n * rank + r] =
                (double)b_q[(size_t)n * rank + r] * (double)(__fp16)b_s[n];

    char error[256] = {0};
    h3_ane_linear *linear = h3_ane_linear_create_int8_bypass(
        name, quantized, scales, group_size, a_rot_q, a_rot_s, b_q, b_s, rank,
        input_dim, output_dim, rows, kc, error, sizeof(error));
    if (!linear) {
        printf("FAIL %s create: %s\n", name, error);
        failures++;
        return;
    }
    uint32_t chunks = h3_ane_linear_chunks(linear);
    uint32_t chunk_dim = h3_ane_linear_chunk_dim(linear);
    uint32_t plane_rows = h3_ane_linear_plane_rows(linear);
    for (uint32_t c = 0; c < chunks; c++) {
        float *plane = h3_ane_linear_input(linear, c);
        memset(plane, 0, h3_ane_linear_input_bytes(linear));
        uint32_t base = c * chunk_dim;
        uint32_t span = input_dim > base ?
            (input_dim - base < chunk_dim ? input_dim - base : chunk_dim) : 0;
        for (uint32_t k = 0; k < span; k++)
            memcpy(plane + (size_t)k * plane_rows,
                   x + (size_t)(base + k) * rows, sizeof(float) * rows);
    }
    if (!h3_ane_linear_eval(linear, error, sizeof(error))) {
        printf("FAIL %s eval: %s\n", name, error);
        failures++;
        h3_ane_linear_free(linear);
        return;
    }
    /* want = W x + scale . B (A x); W comes back through the derotate. */
    double *weights = malloc(sizeof(double) * wn);
    double *want = malloc(sizeof(double) * output_dim * rows);
    double *ax = malloc(sizeof(double) * rank * rows);
    if (!weights || !want || !ax) {
        printf("FAIL %s out of memory\n", name);
        failures++;
        h3_ane_linear_free(linear);
        return;
    }
    const float *dtable = group_size ? h3_convrot_hadamard(group_size) : NULL;
    for (uint32_t n = 0; n < output_dim; n++) {
        double scale = (double)(__fp16)scales[n];
        for (uint32_t k = 0; k < input_dim; k++) {
            if (!group_size) {
                weights[(size_t)n * input_dim + k] =
                    (double)quantized[(size_t)n * input_dim + k] * scale;
                continue;
            }
            uint32_t group = k / (uint32_t)group_size;
            uint32_t j = k % (uint32_t)group_size;
            double acc = 0;
            for (int i = 0; i < group_size; i++)
                acc += (double)quantized[(size_t)n * input_dim +
                                         group * group_size + i] * scale *
                       (double)dtable[i * group_size + j];
            weights[(size_t)n * input_dim + k] = acc;
        }
    }
    for (uint32_t r = 0; r < rank; r++)
        for (uint32_t s = 0; s < rows; s++) {
            double acc = 0;
            for (uint32_t k = 0; k < input_dim; k++)
                acc += a_eff[(size_t)r * input_dim + k] *
                       (double)x[(size_t)k * rows + s];
            ax[(size_t)r * rows + s] = acc;
        }
    double main_norm = 0, bypass_norm = 0;
    for (uint32_t n = 0; n < output_dim; n++)
        for (uint32_t s = 0; s < rows; s++) {
            double acc = 0;
            for (uint32_t k = 0; k < input_dim; k++)
                acc += weights[(size_t)n * input_dim + k] *
                       (double)x[(size_t)k * rows + s];
            double lora = 0;
            for (uint32_t r = 0; r < rank; r++)
                lora += b_recon[(size_t)n * rank + r] * ax[(size_t)r * rows + s];
            main_norm += acc * acc;
            bypass_norm += lora * lora;
            want[(size_t)n * rows + s] = acc + lora;
        }
    const float *got = h3_ane_linear_output(linear);
    double dot = 0, got_norm = 0, want_norm = 0, error_sum = 0, scale_sum = 0;
    double worst = 0;
    size_t nonfinite = 0;
    for (uint32_t n = 0; n < output_dim; n++)
        for (uint32_t s = 0; s < rows; s++) {
            double a = got[(size_t)n * plane_rows + s];
            double b = want[(size_t)n * rows + s];
            if (!isfinite(a)) { nonfinite++; continue; }
            dot += a * b;
            got_norm += a * a;
            want_norm += b * b;
            double difference = a - b;
            error_sum += difference * difference;
            scale_sum += b * b;
            if (fabs(difference) > worst) worst = fabs(difference);
        }
    double cosine = dot / (sqrt(got_norm) * sqrt(want_norm) + 1e-30);
    double relative = sqrt(error_sum / (scale_sum + 1e-30));
    double share = sqrt(bypass_norm / (main_norm + 1e-30));
    int pass = cosine >= 0.9999 && relative <= 2e-3 && nonfinite == 0;
    printf("%-12s K=%u N=%u kc=%u rows=%u gs=%d r=%u: cos=%.7f rel_l2=%.3e "
           "max_abs=%.3e bypass/main=%.3f nonfinite=%zu weights=%.1f MiB %s\n",
           name, input_dim, output_dim, kc, rows, group_size, rank, cosine,
           relative, worst, share, nonfinite,
           (double)h3_ane_linear_weight_bytes(linear) / (1024.0 * 1024.0),
           pass ? "PASS" : "FAIL");
    if (!pass) failures++;
    h3_ane_linear_free(linear);
    free(original); free(stored); free(lora_a); free(lora_a_rot);
    free(lora_b); free(quantized); free(scales); free(a_rot_q); free(b_q);
    free(a_rot_s); free(b_s); free(a_recon); free(a_eff); free(b_recon);
    free(x); free(weights); free(want); free(ax);
}

/* Several bands, the shape qkv needs: three diffusers modules packed
 * module-major into one projection, each driving its own slice of output rows.
 * Ranks differ per band so a mistake in per-band offsetting cannot cancel out. */
static void bands_gate(const char *name, uint32_t input_dim,
                       uint32_t output_dim, uint32_t rows, uint32_t kc,
                       int group_size, const uint32_t *ranks,
                       uint32_t band_count, double lora_scale) {
    uint32_t band_rows = output_dim / band_count;
    size_t wn = (size_t)output_dim * input_dim;
    double *original = malloc(sizeof(double) * wn);
    double *stored = malloc(sizeof(double) * wn);
    int8_t *quantized = malloc(wn);
    float *scales = malloc(sizeof(float) * output_dim);
    float *x = malloc(sizeof(float) * input_dim * rows);
    h3_ane_bypass_band *bands = calloc(band_count, sizeof(*bands));
    double **a_eff = calloc(band_count, sizeof(*a_eff));
    double **b_recon = calloc(band_count, sizeof(*b_recon));
    double **ax = calloc(band_count, sizeof(*ax));
    if (!original || !stored || !quantized || !scales || !x || !bands ||
        !a_eff || !b_recon || !ax) {
        printf("FAIL %s out of memory\n", name);
        failures++;
        return;
    }
    for (size_t i = 0; i < wn; i++) original[i] = rng_uniform();
    for (size_t i = 0; i < (size_t)input_dim * rows; i++)
        x[i] = (float)(rng_uniform() * 4.0);
    rotate_rows(original, stored, output_dim, input_dim, group_size);
    for (uint32_t n = 0; n < output_dim; n++) {
        double amax = 0;
        for (uint32_t k = 0; k < input_dim; k++)
            amax = fmax(amax, fabs(stored[(size_t)n * input_dim + k]));
        scales[n] = (float)(amax / 127.0);
        for (uint32_t k = 0; k < input_dim; k++)
            quantized[(size_t)n * input_dim + k] = (int8_t)fmax(-128.0,
                fmin(127.0, round(stored[(size_t)n * input_dim + k] /
                                  (double)scales[n])));
    }
    for (uint32_t i = 0; i < band_count; i++) {
        uint32_t rank = ranks[i];
        double *a = malloc(sizeof(double) * rank * input_dim);
        double *a_rot = malloc(sizeof(double) * rank * input_dim);
        double *a_recon = malloc(sizeof(double) * rank * input_dim);
        double *b = malloc(sizeof(double) * band_rows * rank);
        int8_t *aq = malloc((size_t)rank * input_dim);
        int8_t *bq = malloc((size_t)band_rows * rank);
        float *as = malloc(sizeof(float) * rank);
        float *bs = malloc(sizeof(float) * band_rows);
        a_eff[i] = malloc(sizeof(double) * rank * input_dim);
        b_recon[i] = malloc(sizeof(double) * band_rows * rank);
        ax[i] = malloc(sizeof(double) * rank * rows);
        if (!a || !a_rot || !a_recon || !b || !aq || !bq || !as || !bs ||
            !a_eff[i] || !b_recon[i] || !ax[i]) {
            printf("FAIL %s out of memory\n", name);
            failures++;
            return;
        }
        for (size_t j = 0; j < (size_t)rank * input_dim; j++)
            a[j] = rng_uniform();
        for (size_t j = 0; j < (size_t)band_rows * rank; j++)
            b[j] = rng_uniform() * lora_scale;
        rotate_rows(a, a_rot, rank, input_dim, group_size);
        quantize_rows(a_rot, aq, as, rank, input_dim);
        quantize_rows(b, bq, bs, band_rows, rank);
        for (uint32_t r = 0; r < rank; r++)
            for (uint32_t k = 0; k < input_dim; k++)
                a_recon[(size_t)r * input_dim + k] =
                    (double)aq[(size_t)r * input_dim + k] *
                    (double)(__fp16)as[r];
        rotate_rows(a_recon, a_eff[i], rank, input_dim, group_size);
        for (uint32_t n = 0; n < band_rows; n++)
            for (uint32_t r = 0; r < rank; r++)
                b_recon[i][(size_t)n * rank + r] =
                    (double)bq[(size_t)n * rank + r] * (double)(__fp16)bs[n];
        for (uint32_t r = 0; r < rank; r++)
            for (uint32_t s = 0; s < rows; s++) {
                double acc = 0;
                for (uint32_t k = 0; k < input_dim; k++)
                    acc += a_eff[i][(size_t)r * input_dim + k] *
                           (double)x[(size_t)k * rows + s];
                ax[i][(size_t)r * rows + s] = acc;
            }
        bands[i].a = aq;
        bands[i].a_scales = as;
        bands[i].b = bq;
        bands[i].b_scales = bs;
        bands[i].rank = rank;
        bands[i].row0 = i * band_rows;
        bands[i].rows = band_rows;
        free(a); free(a_rot); free(a_recon); free(b);
    }
    char error[256] = {0};
    h3_ane_linear *linear = h3_ane_linear_create_int8_bands(
        name, quantized, scales, group_size, bands, band_count, input_dim,
        output_dim, rows, kc, error, sizeof(error));
    if (!linear) {
        printf("FAIL %s create: %s\n", name, error);
        failures++;
        return;
    }
    uint32_t chunks = h3_ane_linear_chunks(linear);
    uint32_t chunk_dim = h3_ane_linear_chunk_dim(linear);
    uint32_t plane_rows = h3_ane_linear_plane_rows(linear);
    for (uint32_t c = 0; c < chunks; c++) {
        float *plane = h3_ane_linear_input(linear, c);
        memset(plane, 0, h3_ane_linear_input_bytes(linear));
        uint32_t base = c * chunk_dim;
        uint32_t span = input_dim > base ?
            (input_dim - base < chunk_dim ? input_dim - base : chunk_dim) : 0;
        for (uint32_t k = 0; k < span; k++)
            memcpy(plane + (size_t)k * plane_rows,
                   x + (size_t)(base + k) * rows, sizeof(float) * rows);
    }
    if (!h3_ane_linear_eval(linear, error, sizeof(error))) {
        printf("FAIL %s eval: %s\n", name, error);
        failures++;
        h3_ane_linear_free(linear);
        return;
    }
    double *weights = malloc(sizeof(double) * wn);
    double *want = malloc(sizeof(double) * output_dim * rows);
    const float *dtable = group_size ? h3_convrot_hadamard(group_size) : NULL;
    for (uint32_t n = 0; n < output_dim; n++) {
        double scale = (double)(__fp16)scales[n];
        for (uint32_t k = 0; k < input_dim; k++) {
            if (!group_size) {
                weights[(size_t)n * input_dim + k] =
                    (double)quantized[(size_t)n * input_dim + k] * scale;
                continue;
            }
            uint32_t group = k / (uint32_t)group_size;
            uint32_t j = k % (uint32_t)group_size;
            double acc = 0;
            for (int i = 0; i < group_size; i++)
                acc += (double)quantized[(size_t)n * input_dim +
                                         group * group_size + i] * scale *
                       (double)dtable[i * group_size + j];
            weights[(size_t)n * input_dim + k] = acc;
        }
    }
    double main_norm = 0, bypass_norm = 0;
    for (uint32_t n = 0; n < output_dim; n++) {
        uint32_t band = n / band_rows;
        uint32_t local = n % band_rows;
        uint32_t rank = ranks[band];
        for (uint32_t s = 0; s < rows; s++) {
            double acc = 0;
            for (uint32_t k = 0; k < input_dim; k++)
                acc += weights[(size_t)n * input_dim + k] *
                       (double)x[(size_t)k * rows + s];
            double delta = 0;
            for (uint32_t r = 0; r < rank; r++)
                delta += b_recon[band][(size_t)local * rank + r] *
                         ax[band][(size_t)r * rows + s];
            main_norm += acc * acc;
            bypass_norm += delta * delta;
            want[(size_t)n * rows + s] = acc + delta;
        }
    }
    const float *got = h3_ane_linear_output(linear);
    double dot = 0, got_norm = 0, want_norm = 0, error_sum = 0, scale_sum = 0;
    double worst = 0;
    size_t nonfinite = 0;
    for (uint32_t n = 0; n < output_dim; n++)
        for (uint32_t s = 0; s < rows; s++) {
            double a = got[(size_t)n * plane_rows + s];
            double b = want[(size_t)n * rows + s];
            if (!isfinite(a)) { nonfinite++; continue; }
            dot += a * b;
            got_norm += a * a;
            want_norm += b * b;
            double difference = a - b;
            error_sum += difference * difference;
            scale_sum += b * b;
            if (fabs(difference) > worst) worst = fabs(difference);
        }
    double cosine = dot / (sqrt(got_norm) * sqrt(want_norm) + 1e-30);
    double relative = sqrt(error_sum / (scale_sum + 1e-30));
    double share = sqrt(bypass_norm / (main_norm + 1e-30));
    int pass = cosine >= 0.9999 && relative <= 2e-3 && nonfinite == 0;
    printf("%-12s K=%u N=%u kc=%u rows=%u gs=%d bands=%u: cos=%.7f "
           "rel_l2=%.3e max_abs=%.3e bypass/main=%.3f nonfinite=%zu %s\n",
           name, input_dim, output_dim, kc, rows, group_size, band_count,
           cosine, relative, worst, share, nonfinite,
           pass ? "PASS" : "FAIL");
    if (!pass) failures++;
    h3_ane_linear_free(linear);
    free(original); free(stored); free(quantized); free(scales); free(x);
    free(weights); free(want);
    for (uint32_t i = 0; i < band_count; i++) {
        free((void *)bands[i].a);
        free((void *)bands[i].a_scales);
        free((void *)bands[i].b);
        free((void *)bands[i].b_scales);
        free(a_eff[i]); free(b_recon[i]); free(ax[i]);
    }
    free(bands); free(a_eff); free(b_recon); free(ax);
}

/* --- Real LoRA, block 0, through the actual graph -------------------------
 * Read one combined .safetensors (bf16 base weights + diffusers LoRA factors,
 * exactly the layout the earlier scan confirmed). Quantize the bf16 qkv
 * ourselves, build the three real LoRA bands (to_q/to_k/to_v), run them
 * through h3_ane_linear_create_int8_bands, and compare to a C reference that
 * derotates the main weight and adds B(A x) per band. This is the end-to-end
 * proof the synthetic/bands fixtures can't give: real LoRA weights. */

static float bf16_to_f32(uint16_t value) {
    uint32_t bits = (uint32_t)value << 16;
    float result;
    memcpy(&result, &bits, sizeof(result));
    return result;
}

static float *read_bf16_f32(const h3_st_header *header, const char *name,
                           size_t *count) {
    const h3_st_tensor *t = h3_st_find(header, name);
    if (!t || t->dtype != H3_DTYPE_BF16) return NULL;
    size_t n = (size_t)h3_st_tensor_elements(t);
    uint16_t *bf = malloc(n * 2);
    char err[256] = {0};
    if (!h3_st_read_data(header, t, bf, n * 2, err, sizeof(err))) {
        free(bf);
        return NULL;
    }
    float *out = malloc(n * sizeof(float));
    for (size_t i = 0; i < n; i++)
        out[i] = bf16_to_f32(bf[i]);
    free(bf);
    if (count) *count = n;
    return out;
}

static void quantize_rows_f32(const float *src, int8_t *dst, float *scales,
                              uint32_t n_rows, uint32_t cols) {
    for (uint32_t n = 0; n < n_rows; n++) {
        float amax = 0;
        for (uint32_t k = 0; k < cols; k++)
            amax = fmaxf(amax, fabsf(src[(size_t)n * cols + k]));
        scales[n] = amax / 127.0f;
        for (uint32_t k = 0; k < cols; k++)
            dst[(size_t)n * cols + k] = (int8_t)fmaxf(-128.0f, fminf(127.0f,
                roundf(src[(size_t)n * cols + k] / scales[n])));
    }
}

/* Parse a string-valued metadata key ("key":"value") returning the numeric
 * value, or -1 if absent. */
static double meta_double(const char *json, const char *key) {
    char pat[40];
    snprintf(pat, sizeof(pat), "\"%s\"", key);
    const char *p = strstr(json, pat);
    if (!p) return -1;
    p += strlen(pat);
    p = strchr(p, ':');
    if (!p) return -1;
    p++;
    while (*p && (*p == ' ' || *p == '"')) p++;   /* skip space + quote */
    return strtod(p, NULL);
}

/* alpha/rank from the safetensors __metadata__, so the LoRA delta is scaled
 * exactly as diffusers applies it. */
static double lora_scale_from_file(const char *path) {
    FILE *f = fopen(path, "rb");
    if (!f) return 1.0;
    uint8_t head[8];
    if (fread(head, 1, 8, f) != 8) { fclose(f); return 1.0; }
    uint64_t hs = 0;
    for (int i = 0; i < 8; i++) hs |= (uint64_t)head[i] << (8 * i);
    char *buf = malloc(hs + 1);
    if (fread(buf, 1, hs, f) != hs) { free(buf); fclose(f); return 1.0; }
    buf[hs] = 0;
    fclose(f);
    char *meta = strstr(buf, "__metadata__");
    double alpha = meta ? meta_double(meta, "alpha") : -1;
    double rank = meta ? meta_double(meta, "rank") : -1;
    if (alpha < 0) alpha = 128.0;
    if (rank < 0) rank = 128.0;
    free(buf);
    return alpha / rank;
}

static void real_lora_gate(const int8_t *main_q, const float *main_s,
                           int group_size, uint32_t in_dim, uint32_t out_dim,
                           const char *lora_path, unsigned block) {
    h3_st_header header;
    char error[512] = {0};
    if (!h3_st_read_header(lora_path, &header, error, sizeof(error))) {
        printf("FAIL %s read: %s\n", lora_path, error);
        failures++;
        return;
    }
    uint32_t band = out_dim / 3;          /* module-major: q|k|v */
    double scale = lora_scale_from_file(lora_path);

    const char *mods[3] = {"attn.to_q", "attn.to_k", "attn.to_v"};
    h3_ane_bypass_band bands[3];
    double *a_eff[3] = {NULL, NULL, NULL};
    double *b_recon[3] = {NULL, NULL, NULL};
    double *ax[3] = {NULL, NULL, NULL};
    int ok = 1;
    uint32_t rows = 32;
    uint32_t kc = h3_ane_linear_default_chunk(in_dim);
    for (uint32_t b = 0; b < 3; b++) {
        char an[160], bn[160];
        snprintf(an, sizeof(an),
                 "transformer_blocks.%u.%s.lora_A.default.weight", block,
                 mods[b]);
        snprintf(bn, sizeof(bn),
                 "transformer_blocks.%u.%s.lora_B.default.weight", block,
                 mods[b]);
        float *A = read_bf16_f32(&header, an, NULL);
        float *B = read_bf16_f32(&header, bn, NULL);
        const h3_st_tensor *at = h3_st_find(&header, an);
        const h3_st_tensor *bt = h3_st_find(&header, bn);
        if (!A || !B || !at || !bt) {
            printf("FAIL %s: missing %s LoRA factor\n", lora_path, mods[b]);
            ok = 0; break;
        }
        uint32_t rank = (uint32_t)at->shape[0];
        uint32_t bd = (uint32_t)bt->shape[0];   /* == band */
        /* A must be derotated once, matching the convrot the main weight uses.
         * The matrix is symmetric, so derotate == rotate here. */
        h3_convrot_derotate_f32(A, rank, in_dim, group_size);
        int8_t *Aq = malloc((size_t)rank * in_dim);
        float *As = malloc((size_t)rank * sizeof(float));
        quantize_rows_f32(A, Aq, As, rank, in_dim);
        for (size_t n = 0; n < (size_t)bd * rank; n++)
            B[n] *= (float)scale;
        int8_t *Bq = malloc((size_t)bd * rank);
        float *Bs = malloc((size_t)bd * sizeof(float));
        quantize_rows_f32(B, Bq, Bs, bd, rank);
        bands[b].a = Aq; bands[b].a_scales = As;
        bands[b].b = Bq; bands[b].b_scales = Bs;
        bands[b].rank = rank; bands[b].row0 = b * band; bands[b].rows = bd;
        a_eff[b] = malloc((size_t)rank * in_dim * sizeof(double));
        b_recon[b] = malloc((size_t)bd * rank * sizeof(double));
        ax[b] = malloc((size_t)rank * rows * sizeof(double));
        for (uint32_t r = 0; r < rank; r++)
            for (uint32_t k = 0; k < in_dim; k++)
                a_eff[b][(size_t)r * in_dim + k] =
                    (double)Aq[(size_t)r * in_dim + k] * (double)(__fp16)As[r];
        /* a_eff holds dequant(A_rot) = A R. The graph applies A_rot to the
         * rotated activation R x, i.e. A R (R x) = A x, so rotate a_eff once
         * more (R is a symmetric involution) to recover the true A the
         * reference multiplies by the raw x. */
        {
            double *a_true = malloc((size_t)rank * in_dim * sizeof(double));
            rotate_rows(a_eff[b], a_true, rank, in_dim, group_size);
            memcpy(a_eff[b], a_true, (size_t)rank * in_dim * sizeof(double));
            free(a_true);
        }
        for (uint32_t n = 0; n < bd; n++)
            for (uint32_t r = 0; r < rank; r++)
                b_recon[b][(size_t)n * rank + r] =
                    (double)Bq[(size_t)n * rank + r] * (double)(__fp16)Bs[n];
        free(A); free(B);
    }
    if (!ok) {
        for (uint32_t b = 0; b < 3; b++) {
            free((void *)bands[b].a); free((void *)bands[b].a_scales);
            free((void *)bands[b].b); free((void *)bands[b].b_scales);
            free(a_eff[b]); free(b_recon[b]); free(ax[b]);
        }
        h3_st_free_header(&header);
        failures++;
        return;
    }

    h3_ane_linear *linear = h3_ane_linear_create_int8_bands(
        "real-lora-qkv", main_q, main_s, group_size, bands, 3, in_dim,
        out_dim, rows, kc, error, sizeof(error));
    if (!linear) {
        printf("FAIL %s create: %s\n", lora_path, error);
        failures++;
        for (uint32_t b = 0; b < 3; b++) {
            free((void *)bands[b].a); free((void *)bands[b].a_scales);
            free((void *)bands[b].b); free((void *)bands[b].b_scales);
            free(a_eff[b]); free(b_recon[b]); free(ax[b]);
        }
        h3_st_free_header(&header);
        return;
    }
    uint32_t chunks = h3_ane_linear_chunks(linear);
    uint32_t chunk_dim = h3_ane_linear_chunk_dim(linear);
    uint32_t plane_rows = h3_ane_linear_plane_rows(linear);
    float *x = malloc(sizeof(float) * in_dim * rows);
    for (size_t i = 0; i < (size_t)in_dim * rows; i++)
        x[i] = (float)(rng_uniform() * 4.0);
    for (uint32_t c = 0; c < chunks; c++) {
        float *plane = h3_ane_linear_input(linear, c);
        memset(plane, 0, h3_ane_linear_input_bytes(linear));
        uint32_t base = c * chunk_dim;
        uint32_t span = in_dim > base ?
            (in_dim - base < chunk_dim ? in_dim - base : chunk_dim) : 0;
        for (uint32_t k = 0; k < span; k++)
            memcpy(plane + (size_t)k * plane_rows,
                   x + (size_t)(base + k) * rows, sizeof(float) * rows);
    }
    if (!h3_ane_linear_eval(linear, error, sizeof(error))) {
        printf("FAIL %s eval: %s\n", lora_path, error);
        failures++;
        h3_ane_linear_free(linear);
        for (uint32_t b = 0; b < 3; b++) {
            free((void *)bands[b].a); free((void *)bands[b].a_scales);
            free((void *)bands[b].b); free((void *)bands[b].b_scales);
            free(a_eff[b]); free(b_recon[b]); free(ax[b]);
        }
        free(x);
        h3_st_free_header(&header);
        return;
    }
    /* reference: derotated main W . x  +  per-band B (A x) */
    double *main_out = reference(main_q, main_s, group_size, in_dim, out_dim,
                                x, rows);
    for (uint32_t b = 0; b < 3; b++) {
        uint32_t rank = bands[b].rank;
        for (uint32_t r = 0; r < rank; r++)
            for (uint32_t s = 0; s < rows; s++) {
                double acc = 0;
                for (uint32_t k = 0; k < in_dim; k++)
                    acc += a_eff[b][(size_t)r * in_dim + k] *
                           (double)x[(size_t)k * rows + s];
                ax[b][(size_t)r * rows + s] = acc;
            }
    }
    double *want = malloc(sizeof(double) * out_dim * rows);
    for (uint32_t n = 0; n < out_dim; n++)
        for (uint32_t s = 0; s < rows; s++) {
            uint32_t bb = n / band, local = n % band;
            double delta = 0;
            uint32_t rank = bands[bb].rank;
            for (uint32_t r = 0; r < rank; r++)
                delta += b_recon[bb][(size_t)local * rank + r] *
                         ax[bb][(size_t)r * rows + s];
            want[(size_t)n * rows + s] = main_out[(size_t)n * rows + s] + delta;
        }
    const float *got = h3_ane_linear_output(linear);
    double dot = 0, got_norm = 0, want_norm = 0, error_sum = 0, scale_sum = 0;
    double worst = 0;
    size_t nonfinite = 0;
    for (uint32_t n = 0; n < out_dim; n++)
        for (uint32_t s = 0; s < rows; s++) {
            double a = got[(size_t)n * plane_rows + s];
            double bb = want[(size_t)n * rows + s];
            if (!isfinite(a)) { nonfinite++; continue; }
            dot += a * bb;
            got_norm += a * a;
            want_norm += bb * bb;
            double diff = a - bb;
            error_sum += diff * diff;
            scale_sum += bb * bb;
            if (fabs(diff) > worst) worst = fabs(diff);
        }
    double cosine = dot / (sqrt(got_norm) * sqrt(want_norm) + 1e-30);
    double relative = sqrt(error_sum / (scale_sum + 1e-30));
    int pass = cosine >= 0.9999 && relative <= 2e-3 && nonfinite == 0;
    printf("%-14s K=%u N=%u kc=%u rows=%u gs=%d: cos=%.7f rel_l2=%.3e "
           "max_abs=%.3e nonfinite=%zu compile=%.2fs cache=%d %s\n",
           "real-lora", in_dim, out_dim, kc, rows, group_size, cosine,
           relative, worst, nonfinite,
           h3_ane_linear_compile_seconds(linear),
           h3_ane_linear_cache_hit(linear) ? 1 : 0,
           pass ? "PASS" : "FAIL");
    if (!pass) failures++;
    h3_ane_linear_free(linear);
    for (uint32_t b = 0; b < 3; b++) {
        free((void *)bands[b].a); free((void *)bands[b].a_scales);
        free((void *)bands[b].b); free((void *)bands[b].b_scales);
        free(a_eff[b]); free(b_recon[b]); free(ax[b]);
    }
    free(x); free(main_out); free(want);
    h3_st_free_header(&header);
}

int main(int argc, char **argv) {
    if (!h3_ane_linear_available()) {
        printf("skip: the Neural Engine bridge is unavailable\n");
        return 0;
    }
    /* Two structurally different synthetic cases: a padded multi-chunk
     * rotated projection and an unrotated kc=2048 projection. */
    synthetic("rot-padded", 1280, 384, 32, 1024, 256);
    synthetic("plain-2048", 4096, 256, 64, 2048, 0);
    /* Same shape as rot-padded plus the fused bypass branch. Rank is swept:
     * the Neural Engine may reject small convolution channel counts. */
    bypass_gate("bypass-r32", 1280, 384, 32, 1024, 256, 32, 0.2, 0);
    bypass_gate("bypass-r128", 1280, 384, 32, 1024, 256, 128, 0.2, 0);
    /* Zeroed factors: the bypass must vanish and leave the main convolution. */
    bypass_gate("bypass-zeroB", 1280, 384, 32, 1024, 256, 32, 0.2, 1);
    bypass_gate("bypass-zeroA", 1280, 384, 32, 1024, 256, 32, 0.2, 2);
    /* One chunk, so the bypass has no K padding and no add of its own. */
    bypass_gate("bypass-1chunk", 1024, 384, 32, 1024, 256, 128, 0.2, 0);
    {
        uint32_t ranks[3] = {32, 64, 32};
        bands_gate("bands-qkv", 1280, 384, 32, 1024, 256, ranks, 3, 0.2);
    }
    cache_gates();
    if (argc > 1) {
        char error[512] = {0};
        h3_weight_store *store = h3_weight_store_open(argv[1], error,
                                                      sizeof(error));
        if (!store) {
            printf("FAIL store: %s\n", error);
            return 1;
        }
        int8_t *quantized = NULL;
        float *scales = NULL;
        int group_size = 0;
        if (!h3_weight_load_int8_raw(store, "blocks.0.attn.qkv_proj.weight",
                                     21504, 5376, &quantized, &scales,
                                     &group_size, error, sizeof(error))) {
            printf("FAIL raw: %s\n", error);
            return 1;
        }
        gate("real-qkv", quantized, scales, group_size, 5376, 21504, 32,
             h3_ane_linear_default_chunk(5376));
        /* argv[2] is the real LoRA file: build its three bands on top of this
         * exact int8 base and run them through the graph, end-to-end. */
        if (argc > 2 && h3_ane_linear_available())
            real_lora_gate(quantized, scales, group_size, 5376, 21504, argv[2],
                           0);
        free(quantized);
        free(scales);
        h3_weight_store_free(store);
    }
    printf("%u failure(s)\n", failures);
    return failures ? 1 : 0;
}
