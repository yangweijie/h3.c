/* Test the Tiled windowed softmax attention kernel. */

#include <float.h>
#include <math.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "h3_gpu.h"

#ifndef max
#define max(a, b) ((a) > (b) ? (a) : (b))
#endif
#ifndef min
#define min(a, b) ((a) < (b) ? (a) : (b))
#endif

typedef struct {
    h3_gpu *gpu;
    const char *label;
    int failures;
} test_context;

static int fail(test_context *test, const char *fmt, ...) {
    va_list args;
    va_start(args, fmt);
    fprintf(stderr, "FAIL %s: ", test->label);
    vfprintf(stderr, fmt, args);
    fprintf(stderr, "\n");
    va_end(args);
    test->failures++;
    return 0;
}

#define REQUIRE(cond, ...) do { if (!(cond)) { fail(test, __VA_ARGS__); return; } } while (0)

static float bf16_to_f32(uint16_t v) {
    uint32_t bits = (uint32_t)v << 16;
    float result;
    memcpy(&result, &bits, sizeof(result));
    return result;
}

static uint16_t f32_to_bf16(float v) {
    uint32_t bits;
    memcpy(&bits, &v, sizeof(bits));
    bits += 0x7fffu + ((bits >> 16) & 1u);
    return bits >> 16;
}

static h3_gpu_tensor *new_bf16(test_context *test, size_t elements) {
    h3_gpu_tensor *t = h3_gpu_tensor_new_bf16(test->gpu, elements);
    if (!t) fail(test, "cannot allocate bf16 tensor (%zu)", elements);
    return t;
}

static int upload_bf16(test_context *test, h3_gpu_tensor *tensor,
                       const uint16_t *values, size_t count) {
    int ok = h3_gpu_begin(test->gpu) &&
             h3_gpu_tensor_write_bf16(tensor, values, count) &&
             h3_gpu_submit(test->gpu);
    if (!ok) fail(test, "cannot upload bf16 data");
    return ok;
}

static int readback_bf16(test_context *test, h3_gpu_tensor *tensor,
                         uint16_t *values, size_t count) {
    int ok = h3_gpu_tensor_read_bf16(tensor, values, count);
    if (!ok) fail(test, "cannot read bf16 data");
    return ok;
}

/* Reference tiled windowed attention */
static void reference_tiled_windowed(const float *q, const float *k, const float *v,
                                     float *out, uint32_t seq, uint32_t heads,
                                     uint32_t head_dim, float scale,
                                     uint32_t window_radius, uint32_t num_frames,
                                     uint32_t tokens_per_frame, uint32_t video_start,
                                     uint32_t text_rows, uint32_t audio_rows) {
    uint32_t video_end = video_start + num_frames * tokens_per_frame;
    uint32_t audio_start = video_end;

    for (uint32_t h = 0; h < heads; h++) {
        for (uint32_t t = 0; t < seq; t++) {
            /* Determine key ranges */
            float max_val = -INFINITY;

            /* First pass: find max score */
            for (uint32_t s = 0; s < seq; s++) {
                int keep = 0;
                if (s > t) {
                    keep = 0;  /* causal */
                } else if (t < text_rows) {
                    keep = (s < text_rows);  /* text sees text */
                } else if (t >= video_start && t < video_end) {
                    if (s < text_rows) keep = 1;  /* video sees text */
                    else if (s >= audio_start) keep = 1;  /* video sees audio */
                    else {
                        uint32_t tf = (t - video_start) / tokens_per_frame;
                        uint32_t sf = (s - video_start) / tokens_per_frame;
                        int diff = (int)tf - (int)sf;
                        keep = (diff >= -(int)window_radius && diff <= (int)window_radius);
                    }
                } else {
                    keep = 1;  /* audio sees all */
                }

                if (!keep) continue;

                float score = 0;
                for (uint32_t d = 0; d < head_dim; d++) {
                    score += q[((size_t)t * heads + h) * head_dim + d] *
                             k[((size_t)s * heads + h) * head_dim + d];
                }
                score *= scale;
                if (score > max_val) max_val = score;
            }

            /* Second pass: softmax and weighted sum */
            float sum_exp = 0;
            float scores[256];
            for (uint32_t s = 0; s < seq; s++) {
                int keep = 0;
                if (s > t) {
                    keep = 0;
                } else if (t < text_rows) {
                    keep = (s < text_rows);
                } else if (t >= video_start && t < video_end) {
                    if (s < text_rows) keep = 1;
                    else if (s >= audio_start) keep = 1;
                    else {
                        uint32_t tf = (t - video_start) / tokens_per_frame;
                        uint32_t sf = (s - video_start) / tokens_per_frame;
                        int diff = (int)tf - (int)sf;
                        keep = (diff >= -(int)window_radius && diff <= (int)window_radius);
                    }
                } else {
                    keep = 1;
                }

                if (!keep) {
                    scores[s] = 0;
                    continue;
                }

                float score = 0;
                for (uint32_t d = 0; d < head_dim; d++) {
                    score += q[((size_t)t * heads + h) * head_dim + d] *
                             k[((size_t)s * heads + h) * head_dim + d];
                }
                score *= scale;
                scores[s] = exp(score - max_val);
                sum_exp += scores[s];
            }

            /* Third pass: weighted sum */
            for (uint32_t d = 0; d < head_dim; d++) {
                float acc = 0;
                for (uint32_t s = 0; s < seq; s++) {
                    if (scores[s] > 0) {
                        float w = scores[s] / sum_exp;
                        acc += w * v[((size_t)s * heads + h) * head_dim + d];
                    }
                }
                out[((size_t)t * heads + h) * head_dim + d] = acc;
            }
        }
    }
}

/* Test: tiled windowed attention */
static void test_tiled_windowed(test_context *test) {
    /* F=4 frames, S=4 tokens/frame => 16 video tokens.
     * total = 4 text + 16 video + 4 audio = 24 rows. */
    uint32_t num_frames = 4, tokens_per_frame = 4;
    uint32_t text_rows = 4, audio_rows = 4;
    uint32_t video_start = text_rows;
    uint32_t seq = text_rows + num_frames * tokens_per_frame + audio_rows;
    uint32_t heads = 2, head_dim = 16;
    uint32_t window_radius = 2;
    float scale = 1.0f / sqrtf((float)head_dim);
    size_t count = (size_t)seq * heads * head_dim;

    float *q = calloc(count, sizeof(float));
    float *k = calloc(count, sizeof(float));
    float *v = calloc(count, sizeof(float));
    float *out_ref = calloc(count, sizeof(float));
    REQUIRE(q && k && v && out_ref, "cannot allocate test data");

    for (size_t i = 0; i < count; i++) {
        q[i] = (float)(i % 7) * 0.05f;
        k[i] = (float)(i % 5) * 0.05f;
        v[i] = (float)(i % 3) * 0.1f;
    }

    reference_tiled_windowed(q, k, v, out_ref, seq, heads, head_dim, scale,
                             window_radius, num_frames, tokens_per_frame,
                             video_start, text_rows, audio_rows);

    uint16_t *q_bf = malloc(count * sizeof(uint16_t));
    uint16_t *k_bf = malloc(count * sizeof(uint16_t));
    uint16_t *v_bf = malloc(count * sizeof(uint16_t));
    for (size_t i = 0; i < count; i++) {
        q_bf[i] = f32_to_bf16(q[i]);
        k_bf[i] = f32_to_bf16(k[i]);
        v_bf[i] = f32_to_bf16(v[i]);
    }

    h3_gpu_tensor *q_t = new_bf16(test, count);
    h3_gpu_tensor *k_t = new_bf16(test, count);
    h3_gpu_tensor *v_t = new_bf16(test, count);
    h3_gpu_tensor *out_t = new_bf16(test, count);
    if (!q_t || !k_t || !v_t || !out_t) goto cleanup;

    REQUIRE(upload_bf16(test, q_t, q_bf, count), "upload q");
    REQUIRE(upload_bf16(test, k_t, k_bf, count), "upload k");
    REQUIRE(upload_bf16(test, v_t, v_bf, count), "upload v");

    int ok = h3_gpu_begin(test->gpu) &&
             h3_gpu_flash_attn_bf16(test->gpu, out_t, q_t, k_t, v_t,
                 seq, heads, head_dim, scale, window_radius, num_frames,
                 tokens_per_frame, video_start, text_rows, audio_rows, 1, 0) &&
             h3_gpu_submit(test->gpu);
    REQUIRE(ok, "tiled_windowed dispatch failed");

    uint16_t *out_bf = malloc(count * sizeof(uint16_t));
    REQUIRE(out_bf, "cannot allocate readback");
    REQUIRE(readback_bf16(test, out_t, out_bf, count), "read output");

    float max_err = 0;
    for (size_t i = 0; i < count; i++) {
        float gpu_v = bf16_to_f32(out_bf[i]);
        float ref_v = out_ref[i];
        float err = fabsf(gpu_v - ref_v);
        if (err > max_err) max_err = err;
    }
    REQUIRE(max_err < 5e-2f, "tiled_windowed mismatch max=%g", max_err);
    printf("  tiled_windowed: OK (seq=%u,F=%u,S=%u,r=%u) max_err=%.3g\n",
           seq, num_frames, tokens_per_frame, window_radius, max_err);

cleanup:
    free(q); free(k); free(v); free(out_ref);
    free(q_bf); free(k_bf); free(v_bf); free(out_bf);
    if (q_t) h3_gpu_tensor_free(q_t);
    if (k_t) h3_gpu_tensor_free(k_t);
    if (v_t) h3_gpu_tensor_free(v_t);
    if (out_t) h3_gpu_tensor_free(out_t);
}

int main(void) {
    test_context test = { .label = "tiled_windowed", .failures = 0 };
    char error[256] = "";
    test.gpu = h3_gpu_create("h3_shaders.metal", error, sizeof(error));
    if (!test.gpu) { fprintf(stderr, "FAIL: cannot create GPU: %s\n", error); return 1; }

    printf("Testing Tiled windowed softmax kernel...\n");
    test_tiled_windowed(&test);

    h3_gpu_free(test.gpu);

    if (test.failures) {
        printf("\n%d FAILURES\n", test.failures);
        return 1;
    }
    printf("\nAll tiled-windowed tests passed.\n");
    return 0;
}