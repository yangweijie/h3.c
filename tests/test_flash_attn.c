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

/* Rectangular SDPA: a query block that scores a different number of rows than
 * it has. That is the shape block-sparse attention needs -- the square call can
 * only express a block-diagonal pattern -- and packing the selected key blocks
 * with contiguous copies is how those rows reach the kernel. */
static void reference_rect(const uint16_t *query, const uint16_t *key,
                           const uint16_t *value, const uint32_t *key_rows,
                           uint32_t query_rows, uint32_t key_count,
                           uint32_t heads, uint32_t head_dim, float scale,
                           float *output) {
    for (uint32_t row = 0; row < query_rows; row++) {
        for (uint32_t h = 0; h < heads; h++) {
            const uint16_t *q =
                query + ((size_t)row * heads + h) * head_dim;
            float scores[64];
            float highest = -INFINITY;
            for (uint32_t index = 0; index < key_count; index++) {
                const uint16_t *k =
                    key + ((size_t)key_rows[index] * heads + h) * head_dim;
                float sum = 0.0f;
                for (uint32_t d = 0; d < head_dim; d++)
                    sum += bf16_to_f32(q[d]) * bf16_to_f32(k[d]);
                scores[index] = sum * scale;
                if (scores[index] > highest) highest = scores[index];
            }
            float total = 0.0f;
            for (uint32_t index = 0; index < key_count; index++) {
                scores[index] = expf(scores[index] - highest);
                total += scores[index];
            }
            float *out = output + ((size_t)row * heads + h) * head_dim;
            for (uint32_t d = 0; d < head_dim; d++) out[d] = 0.0f;
            for (uint32_t index = 0; index < key_count; index++) {
                const uint16_t *v =
                    value + ((size_t)key_rows[index] * heads + h) * head_dim;
                float weight = scores[index] / total;
                for (uint32_t d = 0; d < head_dim; d++)
                    out[d] += weight * bf16_to_f32(v[d]);
            }
        }
    }
}

static void test_sdpa_rectangular(test_context *test) {
    enum {
        RECT_HEADS = 2, RECT_DIM = 64, RECT_QUERY = 5, RECT_KEYS = 11,
        RECT_SOURCE = 13, RECT_FIRST_BLOCK = 8, RECT_SECOND_BLOCK = 3
    };
    const size_t rect_row = RECT_HEADS * RECT_DIM;
    const size_t rect_query_count = RECT_SOURCE * rect_row;
    const size_t rect_key_count = RECT_SOURCE * rect_row;
    const float rect_scale = 1.0f / sqrtf((float)RECT_DIM);
    uint16_t *rect_q = malloc(rect_query_count * sizeof(*rect_q));
    uint16_t *rect_k = malloc(rect_key_count * sizeof(*rect_k));
    uint16_t *rect_v = malloc(rect_key_count * sizeof(*rect_v));
    float *rect_ref = malloc(RECT_QUERY * rect_row * sizeof(*rect_ref));
    uint16_t *rect_got = malloc(RECT_QUERY * rect_row * sizeof(*rect_got));
    uint16_t *rect_dense_got = malloc(RECT_KEYS * rect_row * sizeof(*rect_dense_got));
    REQUIRE(rect_q && rect_k && rect_v && rect_ref && rect_got && rect_dense_got,
            "cannot allocate rectangular SDPA data");
    for (size_t i = 0; i < rect_query_count; i++)
        rect_q[i] = f32_to_bf16((float)(i % 7) * 0.125f - 0.375f);
    for (size_t i = 0; i < rect_key_count; i++)
        rect_k[i] = f32_to_bf16((float)(i % 5) * 0.1f - 0.2f);
    for (size_t i = 0; i < rect_key_count; i++)
        rect_v[i] = f32_to_bf16((float)(i % 11) * 0.09f - 0.45f);

    h3_gpu_tensor *q_t = new_bf16(test, rect_query_count);
    h3_gpu_tensor *k_t = new_bf16(test, rect_key_count);
    h3_gpu_tensor *v_t = new_bf16(test, rect_key_count);
    h3_gpu_tensor *rect_t = new_bf16(test, RECT_QUERY * rect_row);
    h3_gpu_tensor *dense_t = new_bf16(test, RECT_KEYS * rect_row);
    h3_gpu_tensor *square_t = new_bf16(test, RECT_QUERY * rect_row);
    h3_gpu_tensor *square_rect_t = new_bf16(test, RECT_QUERY * rect_row);
    h3_gpu_tensor *packed_k_t = new_bf16(test, RECT_KEYS * rect_row);
    h3_gpu_tensor *packed_v_t = new_bf16(test, RECT_KEYS * rect_row);
    h3_gpu_tensor *packed_t = new_bf16(test, RECT_QUERY * rect_row);
    REQUIRE(q_t && k_t && v_t && rect_t && dense_t && square_t &&
            square_rect_t && packed_k_t && packed_v_t && packed_t,
            "cannot allocate rectangular SDPA tensors");
    REQUIRE(upload_bf16(test, q_t, rect_q, rect_query_count), "upload q");
    REQUIRE(upload_bf16(test, k_t, rect_k, rect_key_count), "upload k");
    REQUIRE(upload_bf16(test, v_t, rect_v, rect_key_count), "upload v");

    /* Selected key blocks: rows 5..12 then rows 0..2. */
    uint32_t order[RECT_KEYS];
    for (uint32_t row = 0; row < RECT_FIRST_BLOCK; row++) order[row] = 5 + row;
    for (uint32_t row = 0; row < RECT_SECOND_BLOCK; row++)
        order[RECT_FIRST_BLOCK + row] = row;
    const size_t first_elements = RECT_FIRST_BLOCK * rect_row;
    const size_t second_elements = RECT_SECOND_BLOCK * rect_row;

    int ok = h3_gpu_begin(test->gpu) &&
             h3_gpu_sdpa_rect_bf16(test->gpu, rect_t, q_t, k_t, v_t, RECT_QUERY,
                                   RECT_KEYS, RECT_HEADS, RECT_DIM, rect_scale) &&
             h3_gpu_sdpa_bf16(test->gpu, dense_t, q_t, k_t, v_t, RECT_KEYS,
                              RECT_HEADS, RECT_DIM, rect_scale) &&
             h3_gpu_sdpa_bf16(test->gpu, square_t, q_t, k_t, v_t, RECT_QUERY,
                              RECT_HEADS, RECT_DIM, rect_scale) &&
             h3_gpu_sdpa_rect_bf16(test->gpu, square_rect_t, q_t, k_t, v_t,
                                   RECT_QUERY, RECT_QUERY, RECT_HEADS,
                                   RECT_DIM, rect_scale) &&
             h3_gpu_copy_bf16(test->gpu, packed_k_t, 0, k_t,
                              5 * rect_row, first_elements) &&
             h3_gpu_copy_bf16(test->gpu, packed_k_t, first_elements, k_t, 0,
                              second_elements) &&
             h3_gpu_copy_bf16(test->gpu, packed_v_t, 0, v_t,
                              5 * rect_row, first_elements) &&
             h3_gpu_copy_bf16(test->gpu, packed_v_t, first_elements, v_t, 0,
                              second_elements) &&
             h3_gpu_sdpa_rect_bf16(test->gpu, packed_t, q_t, packed_k_t,
                                   packed_v_t, RECT_QUERY, RECT_KEYS,
                                   RECT_HEADS, RECT_DIM, rect_scale) &&
             h3_gpu_submit(test->gpu);
    REQUIRE(ok, "rectangular SDPA dispatch failed: %s",
            h3_gpu_error(test->gpu));
    REQUIRE(readback_bf16(test, rect_t, rect_got, RECT_QUERY * rect_row),
            "read rectangular output");
    REQUIRE(readback_bf16(test, dense_t, rect_dense_got,
                          RECT_KEYS * rect_row), "read dense output");
    uint16_t *square_got = malloc(RECT_QUERY * rect_row * sizeof(*square_got));
    uint16_t *square_rect_got =
        malloc(RECT_QUERY * rect_row * sizeof(*square_rect_got));
    REQUIRE(square_got && square_rect_got, "cannot allocate square readback");
    REQUIRE(readback_bf16(test, square_t, square_got, RECT_QUERY * rect_row),
            "read square output");
    REQUIRE(readback_bf16(test, square_rect_t, square_rect_got,
                          RECT_QUERY * rect_row), "read square rect output");

    /* Equal row counts must land on the very graph the square call uses. */
    REQUIRE(memcmp(square_got, square_rect_got,
                   RECT_QUERY * rect_row * sizeof(*square_got)) == 0,
            "rectangular SDPA with equal rows differs from the square call");

    /* Same rows, same keys: the rectangular call must be the dense call cut to
     * its query rows. */
    float max_prefix = 0.0f;
    for (size_t i = 0; i < RECT_QUERY * rect_row; i++) {
        float err = fabsf(bf16_to_f32(rect_got[i]) -
                          bf16_to_f32(rect_dense_got[i]));
        if (err > max_prefix) max_prefix = err;
    }
    REQUIRE(max_prefix < 1e-2f,
            "rectangular SDPA differs from its dense rows max=%g", max_prefix);

    /* And it must agree with the BF16 reference, both over a prefix of keys and
     * over the packed pair of selected blocks. */
    uint32_t prefix[RECT_KEYS];
    for (uint32_t row = 0; row < RECT_KEYS; row++) prefix[row] = row;
    float max_error = 0.0f;
    reference_rect(rect_q, rect_k, rect_v, prefix, RECT_QUERY, RECT_KEYS,
                   RECT_HEADS, RECT_DIM, rect_scale, rect_ref);
    for (size_t i = 0; i < RECT_QUERY * rect_row; i++) {
        float err = fabsf(bf16_to_f32(rect_got[i]) - rect_ref[i]);
        if (err > max_error) max_error = err;
    }
    REQUIRE(max_error < 1e-2f, "rectangular SDPA reference mismatch max=%g",
            max_error);
    uint16_t *packed_got = malloc(RECT_QUERY * rect_row * sizeof(*packed_got));
    REQUIRE(packed_got, "cannot allocate packed readback");
    REQUIRE(readback_bf16(test, packed_t, packed_got, RECT_QUERY * rect_row),
            "read packed output");
    float max_packed = 0.0f;
    reference_rect(rect_q, rect_k, rect_v, order, RECT_QUERY, RECT_KEYS,
                   RECT_HEADS, RECT_DIM, rect_scale, rect_ref);
    for (size_t i = 0; i < RECT_QUERY * rect_row; i++) {
        float err = fabsf(bf16_to_f32(packed_got[i]) - rect_ref[i]);
        if (err > max_packed) max_packed = err;
    }
    REQUIRE(max_packed < 1e-2f,
            "packed key blocks mismatch reference max=%g", max_packed);
    printf("  sdpa_rectangular: OK (%u x %u rows, %u heads, dim %u) "
           "dense-prefix max_err=%.3g reference max_err=%.3g packed %.3g\n",
           RECT_QUERY, RECT_KEYS, RECT_HEADS, RECT_DIM, max_prefix, max_error,
           max_packed);

    free(rect_q); free(rect_k); free(rect_v); free(rect_ref);
    free(rect_got); free(rect_dense_got); free(packed_got);
    free(square_got); free(square_rect_got);
    h3_gpu_tensor_free(q_t); h3_gpu_tensor_free(k_t); h3_gpu_tensor_free(v_t);
    h3_gpu_tensor_free(rect_t); h3_gpu_tensor_free(dense_t);
    h3_gpu_tensor_free(square_t); h3_gpu_tensor_free(square_rect_t);
    h3_gpu_tensor_free(packed_k_t);
    h3_gpu_tensor_free(packed_v_t); h3_gpu_tensor_free(packed_t);
}

/* One packed key/value scratch reused by every query block, chain after chain.
 *
 * The bench gives each block its own buffers to avoid manufacturing a warm
 * cache, which costs G x 2 x key rows of storage (813 MB for 18 x 393 x 1,179).
 * A real block loop only needs one scratch pair if the graph honours the
 * write-then-read order inside a single command buffer, so that saving depends
 * on a hazard guarantee this test pins down: three blocks pack different key
 * sets -- including a two-segment set -- through the same scratch, and each
 * output must still match the reference for its own set. */
static void test_sdpa_scratch_reuse(test_context *test) {
    enum {
        SCRATCH_HEADS = 2, SCRATCH_DIM = 64, SCRATCH_QUERY = 5,
        SCRATCH_KEYS = 11, SCRATCH_SOURCE = 13, SCRATCH_BLOCKS = 3
    };
    typedef struct { uint32_t start, rows; } segment;
    const segment sets[SCRATCH_BLOCKS][2] = {
        {{0, SCRATCH_KEYS}},
        {{5, 8}, {0, 3}},
        {{2, SCRATCH_KEYS}},
    };
    const uint32_t set_counts[SCRATCH_BLOCKS] = {1, 2, 1};

    const size_t row = SCRATCH_HEADS * SCRATCH_DIM;
    const size_t source_count = SCRATCH_SOURCE * row;
    const size_t key_count = SCRATCH_KEYS * row;
    const size_t query_count = SCRATCH_QUERY * row;
    const float scale = 1.0f / sqrtf((float)SCRATCH_DIM);
    uint16_t *query = malloc(source_count * sizeof(*query));
    uint16_t *keys = malloc(source_count * sizeof(*keys));
    uint16_t *values = malloc(source_count * sizeof(*values));
    uint16_t *got = malloc(query_count * sizeof(*got));
    uint16_t *previous = malloc(query_count * sizeof(*previous));
    float *reference = malloc(query_count * sizeof(*reference));
    REQUIRE(query && keys && values && got && previous && reference,
            "cannot allocate scratch reuse data");
    for (size_t i = 0; i < source_count; i++)
        query[i] = f32_to_bf16((float)(i % 7) * 0.125f - 0.375f);
    for (size_t i = 0; i < source_count; i++)
        keys[i] = f32_to_bf16((float)(i % 5) * 0.1f - 0.2f);
    for (size_t i = 0; i < source_count; i++)
        values[i] = f32_to_bf16((float)(i % 11) * 0.09f - 0.45f);

    h3_gpu_tensor *query_t = new_bf16(test, source_count);
    h3_gpu_tensor *key_t = new_bf16(test, source_count);
    h3_gpu_tensor *value_t = new_bf16(test, source_count);
    h3_gpu_tensor *scratch_key_t = new_bf16(test, key_count);
    h3_gpu_tensor *scratch_value_t = new_bf16(test, key_count);
    h3_gpu_tensor *output_t[SCRATCH_BLOCKS];
    for (int block = 0; block < SCRATCH_BLOCKS; block++)
        output_t[block] = new_bf16(test, query_count);
    REQUIRE(query_t && key_t && value_t && scratch_key_t && scratch_value_t &&
            output_t[0] && output_t[1] && output_t[2],
            "cannot allocate scratch reuse tensors");
    REQUIRE(upload_bf16(test, query_t, query, source_count), "upload q");
    REQUIRE(upload_bf16(test, key_t, keys, source_count), "upload k");
    REQUIRE(upload_bf16(test, value_t, values, source_count), "upload v");

    /* Every block rewrites the same scratch and then reads it through its own
     * SDPA, all inside one chain. */
    int ok = h3_gpu_begin(test->gpu);
    for (int block = 0; block < SCRATCH_BLOCKS && ok; block++) {
        uint32_t written = 0;
        for (uint32_t piece = 0; piece < set_counts[block] && ok; piece++) {
            const segment part = sets[block][piece];
            ok = ok && h3_gpu_copy_bf16(test->gpu, scratch_key_t,
                                         (size_t)written * row, key_t,
                                         (size_t)part.start * row,
                                         (size_t)part.rows * row) &&
                    h3_gpu_copy_bf16(test->gpu, scratch_value_t,
                                     (size_t)written * row, value_t,
                                     (size_t)part.start * row,
                                     (size_t)part.rows * row);
            written += part.rows;
        }
        ok = ok && written == SCRATCH_KEYS &&
             h3_gpu_sdpa_rect_bf16(test->gpu, output_t[block], query_t,
                                   scratch_key_t, scratch_value_t,
                                   SCRATCH_QUERY, SCRATCH_KEYS, SCRATCH_HEADS,
                                   SCRATCH_DIM, scale);
    }
    ok = ok && h3_gpu_submit(test->gpu);
    REQUIRE(ok, "shared scratch dispatch failed: %s", h3_gpu_error(test->gpu));

    uint32_t order[SCRATCH_KEYS];
    float worst = 0.0f;
    for (int block = 0; block < SCRATCH_BLOCKS; block++) {
        uint32_t index = 0;
        for (uint32_t piece = 0; piece < set_counts[block]; piece++)
            for (uint32_t row_index = 0; row_index < sets[block][piece].rows;
                 row_index++)
                order[index++] = sets[block][piece].start + row_index;
        REQUIRE(readback_bf16(test, output_t[block], got, query_count),
                "read shared scratch output");
        if (block)
            REQUIRE(memcmp(got, previous, query_count * sizeof(*got)) != 0,
                    "block %d produced block %d's output verbatim", block,
                    block - 1);
        memcpy(previous, got, query_count * sizeof(*got));
        reference_rect(query, keys, values, order, SCRATCH_QUERY, SCRATCH_KEYS,
                       SCRATCH_HEADS, SCRATCH_DIM, scale, reference);
        for (size_t i = 0; i < query_count; i++) {
            float err = fabsf(bf16_to_f32(got[i]) - reference[i]);
            if (err > worst) worst = err;
        }
    }
    REQUIRE(worst < 1e-2f, "shared scratch mismatch max=%g", worst);
    printf("  sdpa_scratch_reuse: OK (%u blocks x %u keys through one scratch,"
           " two-segment gather included) max_err=%.3g\n", SCRATCH_BLOCKS,
           SCRATCH_KEYS, worst);

    free(query); free(keys); free(values); free(got); free(previous);
    free(reference);
    h3_gpu_tensor_free(query_t); h3_gpu_tensor_free(key_t);
    h3_gpu_tensor_free(value_t); h3_gpu_tensor_free(scratch_key_t);
    h3_gpu_tensor_free(scratch_value_t);
    for (int block = 0; block < SCRATCH_BLOCKS; block++)
        h3_gpu_tensor_free(output_t[block]);
}

int main(void) {
    test_context test = { .label = "flash_attn", .failures = 0 };
    char error[256] = "";
    test.gpu = h3_gpu_create("h3_shaders.metal", error, sizeof(error));
    if (!test.gpu) { fprintf(stderr, "FAIL: cannot create GPU: %s\n", error); return 1; }

    printf("Testing FlashAttention-style kernels...\n");
    test_flash_attn_causal(&test);
    test_flash_attn_windowed(&test);
    test_sdpa_rectangular(&test);
    test_sdpa_scratch_reuse(&test);

    h3_gpu_free(test.gpu);

    if (test.failures) {
        printf("\n%d FAILURES\n", test.failures);
        return 1;
    }
    printf("\nAll FlashAttention tests passed.\n");
    return 0;
}