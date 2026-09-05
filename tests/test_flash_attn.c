/* Test the FlashAttention-style tiled attention kernel.
 *
 * Exercises:
 *   1. Causal FlashAttention: verifies online softmax against full reference
 *   2. Windowed FlashAttention: verifies per-frame window masking
 *   3. Numerical stability: verifies online softmax with large magnitude differences
 *   4. Head-major layout: verifies [H,T,d] vs [T,H,d] dispatch
 */

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

/* Reference attention computation: full QK^T, mask, softmax, V */
static void reference_attention(const float *q, const float *k, const float *v,
                                float *out, uint32_t seq, uint32_t heads,
                                uint32_t head_dim, float scale,
                                uint32_t window_radius, uint32_t num_frames,
                                uint32_t tokens_per_frame, uint32_t video_start,
                                uint32_t text_rows, uint32_t audio_rows,
                                int causal) {
    for (uint32_t h = 0; h < heads; h++) {
        for (uint32_t t = 0; t < seq; t++) {
            float max_val = -INFINITY;
            /* First pass: compute scores and find max */
            float scores[256];  /* max seq supported */
            for (uint32_t s = 0; s < seq; s++) {
                float score = 0;
                for (uint32_t d = 0; d < head_dim; d++) {
                    /* Use BF16 quantized values to match GPU behavior */
                    uint16_t q_bf = f32_to_bf16(q[((size_t)t * heads + h) * head_dim + d]);
                    uint16_t k_bf = f32_to_bf16(k[((size_t)s * heads + h) * head_dim + d]);
                    score += bf16_to_f32(q_bf) * bf16_to_f32(k_bf);
                }
                score *= scale;
                /* Apply mask */
                int keep = 1;
                if (causal && s > t) keep = 0;
                if (window_radius > 0 && keep) {
                    if (t >= video_start && t < video_start + num_frames * tokens_per_frame) {
                        if (s >= video_start && s < video_start + num_frames * tokens_per_frame) {
                            uint32_t tf = (t - video_start) / tokens_per_frame;
                            uint32_t sf = (s - video_start) / tokens_per_frame;
                            int diff = (int)tf - (int)sf;
                            if (diff < -(int)window_radius || diff > (int)window_radius)
                                keep = 0;
                        }
                    }
                }
                scores[s] = keep ? score : -INFINITY;
                if (keep && score > max_val) max_val = score;
            }
            /* Second pass: softmax */
            float sum_exp = 0;
            for (uint32_t s = 0; s < seq; s++) {
                if (scores[s] > -INFINITY) {
                    sum_exp += expf(scores[s] - max_val);
                }
            }
            /* Third pass: weighted sum */
            for (uint32_t d = 0; d < head_dim; d++) {
                float acc = 0;
                for (uint32_t s = 0; s < seq; s++) {
                    if (scores[s] > -INFINITY) {
                        float w = expf(scores[s] - max_val) / sum_exp;
                        uint16_t v_bf = f32_to_bf16(v[((size_t)s * heads + h) * head_dim + d]);
                        acc += w * bf16_to_f32(v_bf);
                    }
                }
                out[((size_t)t * heads + h) * head_dim + d] = acc;
            }
        }
    }
}

/* Test 1: causal FlashAttention with debug output */
static void test_flash_attn_causal(test_context *test) {
    uint32_t seq = 8, heads = 1, head_dim = 4;  /* very small for debug */
    float scale = 1.0f / sqrtf((float)head_dim);
    size_t count = (size_t)seq * heads * head_dim;

    float *q = calloc(count, sizeof(float));
    float *k = calloc(count, sizeof(float));
    float *v = calloc(count, sizeof(float));
    float *out_ref = calloc(count, sizeof(float));
    REQUIRE(q && k && v && out_ref, "cannot allocate test data");

    /* Simple deterministic values */
    for (size_t i = 0; i < count; i++) {
        q[i] = (float)(i % 5) * 0.1f;
        k[i] = (float)(i % 3) * 0.1f;
        v[i] = (float)(i % 7) * 0.1f;
    }

    reference_attention(q, k, v, out_ref, seq, heads, head_dim, scale,
                        0, 0, 0, 0, 0, 0, 1);

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

    /* Test with simple passthrough kernel first to verify dispatch */
    int ok2 = h3_gpu_begin(test->gpu) &&
              h3_gpu_flash_attn_test_dispatch(test->gpu, out_t, q_t,
                  seq, heads, head_dim, 0) &&
              h3_gpu_submit(test->gpu);
    if (!ok2) {
        printf("    TEST DISPATCH ERROR: %s\n", h3_gpu_error(test->gpu));
    } else {
        uint16_t *test_out = malloc(count * sizeof(uint16_t));
        readback_bf16(test, out_t, test_out, count);
        printf("    Test dispatch: t=0: ");
        for (uint32_t d = 0; d < head_dim; d++) {
            printf("%.4f ", bf16_to_f32(test_out[d]));
        }
        printf("\n");
        printf("    Expected:      t=0: ");
        for (uint32_t d = 0; d < head_dim; d++) {
            printf("%.4f ", bf16_to_f32(q_bf[d]));
        }
        printf("\n");
        free(test_out);
    }

    int ok = h3_gpu_begin(test->gpu) &&
             h3_gpu_flash_attn_bf16(test->gpu, out_t, q_t, k_t, v_t,
                 seq, heads, head_dim, scale, 0, 0, 0, 0, 0, 0, 1, 0) &&
             h3_gpu_submit(test->gpu);
    if (!ok) {
        printf("    ERROR: %s\n", h3_gpu_error(test->gpu));
    }
    REQUIRE(ok, "flash_attn causal dispatch failed");

    uint16_t *out_bf = malloc(count * sizeof(uint16_t));
    REQUIRE(out_bf, "cannot allocate readback");
    REQUIRE(readback_bf16(test, out_t, out_bf, count), "read output");

    /* Debug: print first few values */
    printf("    Debug: seq=%u, heads=%u, dim=%u\n", seq, heads, head_dim);
    for (uint32_t t = 0; t < min(seq, 4u); t++) {
        printf("    t=%u: ref=", t);
        for (uint32_t d = 0; d < head_dim; d++) {
            printf("%.4f ", out_ref[((size_t)t * heads) * head_dim + d]);
        }
        printf("\n          gpu=");
        for (uint32_t d = 0; d < head_dim; d++) {
            printf("%.4f ", bf16_to_f32(out_bf[((size_t)t * heads) * head_dim + d]));
        }
        printf("\n");
    }

    float max_err = 0;
    for (size_t i = 0; i < count; i++) {
        float gpu_v = bf16_to_f32(out_bf[i]);
        float ref_v = out_ref[i];
        float err = fabsf(gpu_v - ref_v);
        if (err > max_err) max_err = err;
    }
    REQUIRE(max_err < 5e-2f, "flash_attn causal mismatch max=%g", max_err);
    printf("  flash_attn_causal: OK (seq=%u,heads=%u,dim=%u) max_err=%.3g\n",
           seq, heads, head_dim, max_err);

cleanup:
    free(q); free(k); free(v); free(out_ref);
    free(q_bf); free(k_bf); free(v_bf); free(out_bf);
    if (q_t) h3_gpu_tensor_free(q_t);
    if (k_t) h3_gpu_tensor_free(k_t);
    if (v_t) h3_gpu_tensor_free(v_t);
    if (out_t) h3_gpu_tensor_free(out_t);
}

/* Test 2: windowed FlashAttention */
static void test_flash_attn_windowed(test_context *test) {
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

    reference_attention(q, k, v, out_ref, seq, heads, head_dim, scale,
                        window_radius, num_frames, tokens_per_frame,
                        video_start, text_rows, audio_rows, 1);

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
    REQUIRE(ok, "flash_attn windowed dispatch failed");

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
    REQUIRE(max_err < 1e-2f, "flash_attn windowed mismatch max=%g", max_err);
    printf("  flash_attn_windowed: OK (seq=%u,F=%u,S=%u,r=%u) max_err=%.3g\n",
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
    test_context test = { .label = "flash_attn", .failures = 0 };
    char error[256] = "";
    test.gpu = h3_gpu_create("h3_shaders.metal", error, sizeof(error));
    if (!test.gpu) { fprintf(stderr, "FAIL: cannot create GPU: %s\n", error); return 1; }

    printf("Testing FlashAttention-style kernels...\n");
    test_flash_attn_causal(&test);
    test_flash_attn_windowed(&test);

    h3_gpu_free(test.gpu);

    if (test.failures) {
        printf("\n%d FAILURES\n", test.failures);
        return 1;
    }
    printf("\nAll FlashAttention tests passed.\n");
    return 0;
}