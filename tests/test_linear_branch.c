/* Test the windowed SDPA mask and linear far-branch attention kernels. */

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

static h3_gpu_tensor *new_f32(test_context *test, size_t elements) {
    h3_gpu_tensor *t = h3_gpu_tensor_new_f32(test->gpu, elements);
    if (!t) fail(test, "cannot allocate f32 tensor (%zu)", elements);
    return t;
}

static h3_gpu_tensor *new_bf16(test_context *test, size_t elements) {
    h3_gpu_tensor *t = h3_gpu_tensor_new_bf16(test->gpu, elements);
    if (!t) fail(test, "cannot allocate bf16 tensor (%zu)", elements);
    return t;
}

static float bf16_to_f32(uint16_t v) {
    uint32_t bits = (uint32_t)v << 16;
    float result;
    memcpy(&result, &bits, sizeof(result));
    return result;
}

static uint16_t f32_to_bf16(float v) {
    uint32_t bits = (union { float f; uint32_t i; }){.f = v}.i;
    bits += 0x7fffu + ((bits >> 16) & 1u);
    return bits >> 16;
}

static int upload_f32(test_context *test, h3_gpu_tensor *tensor,
                      const float *values, size_t count) {
    int ok = h3_gpu_begin(test->gpu) &&
             h3_gpu_tensor_write_f32(tensor, values, count) &&
             h3_gpu_submit(test->gpu);
    if (!ok) fail(test, "cannot upload f32 data");
    return ok;
}

static int upload_bf16(test_context *test, h3_gpu_tensor *tensor,
                       const uint16_t *values, size_t count) {
    int ok = h3_gpu_begin(test->gpu) &&
             h3_gpu_tensor_write_bf16(tensor, values, count) &&
             h3_gpu_submit(test->gpu);
    if (!ok) fail(test, "cannot upload bf16 data");
    return ok;
}

static int readback_f32(test_context *test, h3_gpu_tensor *tensor,
                        float *values, size_t count) {
    int ok = h3_gpu_tensor_read_f32(tensor, values, count);
    if (!ok) fail(test, "cannot read f32 data");
    return ok;
}

static int readback_bf16(test_context *test, h3_gpu_tensor *tensor,
                         uint16_t *values, size_t count) {
    int ok = h3_gpu_tensor_read_bf16(tensor, values, count);
    if (!ok) fail(test, "cannot read bf16 data");
    return ok;
}

/* Test 1: window mask geometry */
static void test_window_mask(test_context *test) {
    uint32_t num_frames = 8, tokens_per_frame = 4;
    uint32_t text_rows = 4, audio_rows = 4;
    uint32_t video_start = text_rows;
    uint32_t total_rows = text_rows + num_frames * tokens_per_frame + audio_rows;
    uint32_t window_radius = 2;
    size_t count = (size_t)total_rows * total_rows;
    h3_gpu_tensor *mask = new_f32(test, count);
    if (!mask) return;

    int ok = h3_gpu_begin(test->gpu) &&
             h3_gpu_sdpa_window_mask_bf16(test->gpu, mask, window_radius,
                 num_frames, tokens_per_frame, video_start, total_rows,
                 text_rows, audio_rows) &&
             h3_gpu_submit(test->gpu);
    REQUIRE(ok, "window mask dispatch failed");

    float *values = calloc(count, sizeof(float));
    REQUIRE(values, "cannot allocate mask readback");
    ok = readback_f32(test, mask, values, count);
    if (!ok) { free(values); h3_gpu_tensor_free(mask); return; }

    int failed = 0;
    for (uint32_t q = 0; q < total_rows && !failed; q++) {
        for (uint32_t k = 0; k < total_rows; k++) {
            float v = values[(size_t)q * total_rows + k];
            int allowed;
            if (k > q) {
                allowed = 0;
            } else if (q < text_rows) {
                allowed = 1;
            } else if (q >= video_start && q < video_start + num_frames * tokens_per_frame) {
                if (k < text_rows) allowed = 1;
                else if (k >= video_start + num_frames * tokens_per_frame) allowed = 1;
                else {
                    uint32_t qf = (q - video_start) / tokens_per_frame;
                    uint32_t kf = (k - video_start) / tokens_per_frame;
                    int diff = (int)qf - (int)kf;
                    allowed = (diff >= -(int)window_radius && diff <= (int)window_radius);
                }
            } else {
                allowed = 1;
            }
            float expected = allowed ? 0.0f : -INFINITY;
            if (v != expected) {
                fail(test, "mask[%u,%u] = %f, expected %f", q, k, v, expected);
                failed = 1;
                break;
            }
        }
    }
    if (!failed) printf("  window_mask: OK (%ux%u, F=%u, S=%u, r=%u)\n",
                        total_rows, total_rows, num_frames, tokens_per_frame, window_radius);

    free(values);
    h3_gpu_tensor_free(mask);
}

/* Test 2: frame statistics */
static void test_frame_stats(test_context *test) {
    uint32_t F = 2, S = 3, H = 2, d = 4;
    size_t kv_count = (size_t)F * H * S * d;
    size_t beta_count = (size_t)F * H * S;
    size_t stats_count = (size_t)F * H * d * d;

    float *k = calloc(kv_count, sizeof(float));
    float *v = calloc(kv_count, sizeof(float));
    float *beta = calloc(beta_count, sizeof(float));
    float *A_ref = calloc(stats_count, sizeof(float));
    float *B_ref = calloc(stats_count, sizeof(float));
    REQUIRE(k && v && beta && A_ref && B_ref, "cannot allocate test data");

    for (size_t i = 0; i < kv_count; i++) { k[i] = (float)(i % 7) * 0.1f; v[i] = (float)(i % 5) * 0.2f; }
    for (size_t i = 0; i < beta_count; i++) { beta[i] = (float)(i % 3) * 0.3f + 0.1f; }

    for (uint32_t f = 0; f < F; f++)
        for (uint32_t h = 0; h < H; h++)
            for (uint32_t di = 0; di < d; di++)
                for (uint32_t dj = 0; dj < d; dj++) {
                    float a = 0, b = 0;
                    for (uint32_t s = 0; s < S; s++) {
                        uint32_t bi = f * H * S + h * S + s;
                        uint32_t ki = ((f * H + h) * S + s) * d;
                        a += beta[bi] * k[ki + di] * k[ki + dj];
                        b += beta[bi] * v[ki + di] * k[ki + dj];
                    }
                    A_ref[((f * H + h) * d + di) * d + dj] = a;
                    B_ref[((f * H + h) * d + di) * d + dj] = b;
                }

    uint16_t *k_bf = malloc(kv_count * sizeof(uint16_t));
    uint16_t *v_bf = malloc(kv_count * sizeof(uint16_t));
    for (size_t i = 0; i < kv_count; i++) { k_bf[i] = f32_to_bf16(k[i]); v_bf[i] = f32_to_bf16(v[i]); }

    h3_gpu_tensor *k_t = new_bf16(test, kv_count);
    h3_gpu_tensor *v_t = new_bf16(test, kv_count);
    h3_gpu_tensor *beta_t = new_f32(test, beta_count);
    h3_gpu_tensor *A_t = new_f32(test, stats_count);
    h3_gpu_tensor *B_t = new_f32(test, stats_count);
    if (!k_t || !v_t || !beta_t || !A_t || !B_t) goto cleanup;

    REQUIRE(upload_bf16(test, k_t, k_bf, kv_count), "upload k");
    REQUIRE(upload_bf16(test, v_t, v_bf, kv_count), "upload v");
    REQUIRE(upload_f32(test, beta_t, beta, beta_count), "upload beta");

    int ok = h3_gpu_begin(test->gpu) &&
             h3_gpu_linear_branch_frame_stats(test->gpu, A_t, B_t, k_t, v_t, beta_t,
                 F, S, H, d) &&
             h3_gpu_submit(test->gpu);
    REQUIRE(ok, "frame_stats dispatch failed");

    float *A_gpu = calloc(stats_count, sizeof(float));
    float *B_gpu = calloc(stats_count, sizeof(float));
    REQUIRE(A_gpu && B_gpu, "cannot allocate readback");
    REQUIRE(readback_f32(test, A_t, A_gpu, stats_count), "read A");
    REQUIRE(readback_f32(test, B_t, B_gpu, stats_count), "read B");

    float max_a = 0, max_b = 0;
    for (size_t i = 0; i < stats_count; i++) {
        float ea = fabsf(A_gpu[i] - A_ref[i]);
        float eb = fabsf(B_gpu[i] - B_ref[i]);
        if (ea > max_a) max_a = ea;
        if (eb > max_b) max_b = eb;
    }
    REQUIRE(max_a < 1e-2f, "A mismatch max=%g", max_a);
    REQUIRE(max_b < 1e-2f, "B mismatch max=%g", max_b);
    printf("  frame_stats: OK (F=%u,S=%u,H=%u,d=%u) max_err A=%.3g B=%.3g\n",
           F, S, H, d, max_a, max_b);

cleanup:
    free(k); free(v); free(beta); free(A_ref); free(B_ref);
    free(k_bf); free(v_bf); free(A_gpu); free(B_gpu);
    if (k_t) h3_gpu_tensor_free(k_t);
    if (v_t) h3_gpu_tensor_free(v_t);
    if (beta_t) h3_gpu_tensor_free(beta_t);
    if (A_t) h3_gpu_tensor_free(A_t);
    if (B_t) h3_gpu_tensor_free(B_t);
}

/* Test 3: factor_apply */
static void test_factor_apply(test_context *test) {
    uint32_t F = 2, S = 4, H = 2, d = 4;
    size_t stats_count = (size_t)F * H * d * d;
    size_t alpha_count = (size_t)F * H * d;

    float *A = calloc(stats_count, sizeof(float));
    float *B = calloc(stats_count, sizeof(float));
    float *alpha = calloc(alpha_count, sizeof(float));
    REQUIRE(A && B && alpha, "cannot allocate test data");

    for (size_t i = 0; i < stats_count; i++) { A[i] = (float)(i % 5) * 0.01f; B[i] = (float)(i % 7) * 0.02f; }
    for (size_t i = 0; i < alpha_count; i++) { alpha[i] = 0.9f + (float)(i % 3) * 0.01f; }

    h3_gpu_tensor *A_t = new_f32(test, stats_count);
    h3_gpu_tensor *B_t = new_f32(test, stats_count);
    h3_gpu_tensor *alpha_t = new_f32(test, alpha_count);
    h3_gpu_tensor *T_t = new_f32(test, stats_count);
    h3_gpu_tensor *J_t = new_f32(test, stats_count);
    if (!A_t || !B_t || !alpha_t || !T_t || !J_t) goto cleanup;

    REQUIRE(upload_f32(test, A_t, A, stats_count), "upload A");
    REQUIRE(upload_f32(test, B_t, B, stats_count), "upload B");
    REQUIRE(upload_f32(test, alpha_t, alpha, alpha_count), "upload alpha");

    int ok = h3_gpu_begin(test->gpu) &&
             h3_gpu_linear_branch_factor_apply(test->gpu, T_t, J_t, A_t, B_t, alpha_t,
                 F, S, H, d) &&
             h3_gpu_submit(test->gpu);
    REQUIRE(ok, "factor_apply dispatch failed");

    float *T_gpu = calloc(stats_count, sizeof(float));
    float *J_gpu = calloc(stats_count, sizeof(float));
    REQUIRE(T_gpu && J_gpu, "cannot allocate readback");
    REQUIRE(readback_f32(test, T_t, T_gpu, stats_count), "read T");
    REQUIRE(readback_f32(test, J_t, J_gpu, stats_count), "read J");

    float max_t = 0, max_j = 0;
    float c2 = 1.0f / (float)S;
    float c = sqrtf(c2);
    for (uint32_t f = 0; f < F; f++)
        for (uint32_t h = 0; h < H; h++)
            for (uint32_t di = 0; di < d; di++)
                for (uint32_t dj = 0; dj < d; dj++) {
                    size_t idx = ((f * H + h) * d + di) * d + dj;
                    float a_ij = A[idx];
                    float alpha_j = alpha[f * H * d + h * d + dj];
                    float t_ref = alpha_j * ((di == dj ? 1.0f : 0.0f) - c2 * a_ij);
                    float j_ref = c * B[idx];
                    float et = fabsf(T_gpu[idx] - t_ref);
                    float ej = fabsf(J_gpu[idx] - j_ref);
                    if (et > max_t) max_t = et;
                    if (ej > max_j) max_j = ej;
                }
    REQUIRE(max_t < 1e-5f, "T mismatch max=%g", max_t);
    REQUIRE(max_j < 1e-5f, "J mismatch max=%g", max_j);
    printf("  factor_apply: OK (F=%u,S=%u,H=%u,d=%u) max_err T=%.3g J=%.3g\n",
           F, S, H, d, max_t, max_j);

cleanup:
    free(A); free(B); free(alpha); free(T_gpu); free(J_gpu);
    if (A_t) h3_gpu_tensor_free(A_t);
    if (B_t) h3_gpu_tensor_free(B_t);
    if (alpha_t) h3_gpu_tensor_free(alpha_t);
    if (T_t) h3_gpu_tensor_free(T_t);
    if (J_t) h3_gpu_tensor_free(J_t);
}

/* Test 4: log-alpha prefix sums */
static void test_log_alpha_prefix(test_context *test) {
    uint32_t F = 4, H = 2, d = 4;
    size_t alpha_count = (size_t)F * H * d;
    size_t prefix_count = (size_t)(F + 1) * H * d;

    float *alpha = calloc(alpha_count, sizeof(float));
    REQUIRE(alpha, "cannot allocate test data");
    for (size_t i = 0; i < alpha_count; i++) { alpha[i] = 0.5f + (float)(i % 5) * 0.1f; }

    h3_gpu_tensor *alpha_t = new_f32(test, alpha_count);
    h3_gpu_tensor *prefix_t = new_f32(test, prefix_count);
    if (!alpha_t || !prefix_t) goto cleanup;

    REQUIRE(upload_f32(test, alpha_t, alpha, alpha_count), "upload alpha");

    int ok = h3_gpu_begin(test->gpu) &&
             h3_gpu_linear_branch_log_alpha_prefix(test->gpu, prefix_t, alpha_t,
                 F, H, d) &&
             h3_gpu_submit(test->gpu);
    REQUIRE(ok, "log_alpha_prefix dispatch failed");

    float *prefix_gpu = calloc(prefix_count, sizeof(float));
    REQUIRE(prefix_gpu, "cannot allocate readback");
    REQUIRE(readback_f32(test, prefix_t, prefix_gpu, prefix_count), "read prefix");

    float max_err = 0;
    for (uint32_t h = 0; h < H; h++)
        for (uint32_t dj = 0; dj < d; dj++) {
            float sum = 0.0f;
            size_t p0 = h * d + dj;
            if (prefix_gpu[p0] != 0.0f) { max_err = 1e6f; }
            for (uint32_t f = 0; f < F; f++) {
                sum += logf(alpha[f * H * d + h * d + dj]);
                size_t idx = (f + 1) * H * d + h * d + dj;
                float err = fabsf(prefix_gpu[idx] - sum);
                if (err > max_err) max_err = err;
            }
        }
    REQUIRE(max_err < 1e-4f, "prefix mismatch max=%g", max_err);
    printf("  log_alpha_prefix: OK (F=%u,H=%u,d=%u) max_err=%.3g\n",
           F, H, d, max_err);

cleanup:
    free(alpha); free(prefix_gpu);
    if (alpha_t) h3_gpu_tensor_free(alpha_t);
    if (prefix_t) h3_gpu_tensor_free(prefix_t);
}

/* Test 5: scan frame (recurrence) */
static void test_scan_frame(test_context *test) {
    uint32_t H = 2, d = 4;
    size_t state_count = (size_t)H * d * d;

    float *state_in = calloc(state_count, sizeof(float));
    float *T = calloc(state_count, sizeof(float));
    float *J = calloc(state_count, sizeof(float));
    REQUIRE(state_in && T && J, "cannot allocate test data");

    for (size_t i = 0; i < state_count; i++) {
        state_in[i] = (float)(i % 5) * 0.1f;
        T[i] = (float)(i % 7) * 0.05f;
        J[i] = (float)(i % 3) * 0.02f;
    }

    h3_gpu_tensor *si_t = new_f32(test, state_count);
    h3_gpu_tensor *T_t = new_f32(test, state_count);
    h3_gpu_tensor *J_t = new_f32(test, state_count);
    h3_gpu_tensor *so_t = new_f32(test, state_count);
    if (!si_t || !T_t || !J_t || !so_t) goto cleanup;

    REQUIRE(upload_f32(test, si_t, state_in, state_count), "upload state_in");
    REQUIRE(upload_f32(test, T_t, T, state_count), "upload T");
    REQUIRE(upload_f32(test, J_t, J, state_count), "upload J");

    int ok = h3_gpu_begin(test->gpu) &&
             h3_gpu_linear_branch_scan_frame(test->gpu, so_t, si_t, T_t, J_t,
                 H, d) &&
             h3_gpu_submit(test->gpu);
    REQUIRE(ok, "scan_frame dispatch failed");

    float *so_gpu = calloc(state_count, sizeof(float));
    REQUIRE(so_gpu, "cannot allocate readback");
    REQUIRE(readback_f32(test, so_t, so_gpu, state_count), "read state_out");

    float max_err = 0;
    for (uint32_t h = 0; h < H; h++)
        for (uint32_t c = 0; c < d; c++)
            for (uint32_t r = 0; r < d; r++) {
                float sum = J[(h * d + c) * d + r];
                for (uint32_t k = 0; k < d; k++) {
                    sum += state_in[(h * d + c) * d + k] * T[(h * d + k) * d + r];
                }
                size_t idx = (h * d + c) * d + r;
                float err = fabsf(so_gpu[idx] - sum);
                if (err > max_err) max_err = err;
            }
    REQUIRE(max_err < 1e-3f, "scan_frame mismatch max=%g", max_err);
    printf("  scan_frame: OK (H=%u,d=%u) max_err=%.3g\n", H, d, max_err);

cleanup:
    free(state_in); free(T); free(J); free(so_gpu);
    if (si_t) h3_gpu_tensor_free(si_t);
    if (T_t) h3_gpu_tensor_free(T_t);
    if (J_t) h3_gpu_tensor_free(J_t);
    if (so_t) h3_gpu_tensor_free(so_t);
}

/* Test 6: gather (directional combination) */
static void test_gather(test_context *test) {
    uint32_t F = 4, H = 2, d = 3;
    uint32_t S = 2;
    size_t state_count = (size_t)F * H * d * d;
    size_t alpha_count = (size_t)F * H * d;
    size_t text_count = (size_t)H * d * d;
    size_t bounds_count = (size_t)F * 2;
    size_t prefix_count = (size_t)(F + 1) * H * d;

    float *prefix = calloc(state_count, sizeof(float));
    float *suffix = calloc(state_count, sizeof(float));
    float *alpha = calloc(alpha_count, sizeof(float));
    float *text = calloc(text_count, sizeof(float));
    float *log_pref = calloc(prefix_count, sizeof(float));
    uint32_t *bounds = calloc(bounds_count, sizeof(uint32_t));
    REQUIRE(prefix && suffix && alpha && text && log_pref && bounds, "cannot allocate");

    for (size_t i = 0; i < state_count; i++) { prefix[i] = (float)(i % 5) * 0.1f; suffix[i] = (float)(i % 7) * 0.1f; }
    for (size_t i = 0; i < alpha_count; i++) { alpha[i] = 0.9f + (float)(i % 3) * 0.01f; }
    for (size_t i = 0; i < text_count; i++) { text[i] = (float)(i % 4) * 0.05f; }
    for (uint32_t f = 0; f < F; f++) { bounds[f * 2 + 0] = f; bounds[f * 2 + 1] = f; }

    for (uint32_t h = 0; h < H; h++)
        for (uint32_t dj = 0; dj < d; dj++) {
            float sum = 0.0f;
            log_pref[h * d + dj] = 0.0f;
            for (uint32_t f = 0; f < F; f++) {
                sum += logf(alpha[f * H * d + h * d + dj]);
                log_pref[(f + 1) * H * d + h * d + dj] = sum;
            }
        }

    h3_gpu_tensor *prefix_t = new_f32(test, state_count);
    h3_gpu_tensor *suffix_t = new_f32(test, state_count);
    h3_gpu_tensor *alpha_t = new_f32(test, alpha_count);
    h3_gpu_tensor *bounds_t = h3_gpu_tensor_from_u32(test->gpu, bounds, bounds_count);
    h3_gpu_tensor *text_t = new_f32(test, text_count);
    h3_gpu_tensor *log_pref_t = new_f32(test, prefix_count);
    h3_gpu_tensor *gathered_t = new_f32(test, state_count);
    if (!prefix_t || !suffix_t || !alpha_t || !bounds_t || !text_t || !log_pref_t || !gathered_t)
        goto cleanup;

    REQUIRE(upload_f32(test, prefix_t, prefix, state_count), "upload prefix");
    REQUIRE(upload_f32(test, suffix_t, suffix, state_count), "upload suffix");
    REQUIRE(upload_f32(test, alpha_t, alpha, alpha_count), "upload alpha");
    REQUIRE(upload_f32(test, text_t, text, text_count), "upload text");
    REQUIRE(upload_f32(test, log_pref_t, log_pref, prefix_count), "upload log_pref");

    int ok = h3_gpu_begin(test->gpu) &&
             h3_gpu_linear_branch_gather(test->gpu, gathered_t, prefix_t, suffix_t,
                 alpha_t, bounds_t, text_t, log_pref_t, F, S, H, d) &&
             h3_gpu_submit(test->gpu);
    REQUIRE(ok, "gather dispatch failed");

    float *g_gpu = calloc(state_count, sizeof(float));
    REQUIRE(g_gpu, "cannot allocate readback");
    REQUIRE(readback_f32(test, gathered_t, g_gpu, state_count), "read gathered");

    float max_err = 0;
    for (uint32_t f = 0; f < F; f++) {
        uint32_t lo = f, hi = f;
        int before_idx = (int)lo - 1;
        int after_idx = (int)hi + 1;
        int has_before = before_idx >= 0;
        int has_after = after_idx < (int)F;
        before_idx = (int)max(before_idx, 0);
        after_idx = (int)min(after_idx, (int)F - 1);

        for (uint32_t h = 0; h < H; h++)
            for (uint32_t di = 0; di < d; di++)
                for (uint32_t dj = 0; dj < d; dj++) {
                    float ab = 1.0f, aa = 1.0f;
                    if (has_before && f > 0) {
                        int a = before_idx + 1, b = f;
                        if (a <= b) {
                            float ls = log_pref[(b + 1) * H * d + h * d + dj] -
                                       log_pref[a * H * d + h * d + dj];
                            ab = expf(ls);
                        }
                    }
                    if (has_after && f < F - 1) {
                        int a = f, b = after_idx;
                        if (a <= b) {
                            float ls = log_pref[(b + 1) * H * d + h * d + dj] -
                                       log_pref[a * H * d + h * d + dj];
                            aa = expf(ls);
                        }
                    }
                    float sb = has_before ? prefix[((before_idx) * H + h) * d * d + di * d + dj]
                                          : text[h * d * d + di * d + dj];
                    float sa = has_after ? suffix[((after_idx) * H + h) * d * d + di * d + dj]
                                         : text[h * d * d + di * d + dj];
                    float ref = sb * ab + sa * aa;
                    size_t idx = ((f * H + h) * d + di) * d + dj;
                    float err = fabsf(g_gpu[idx] - ref);
                    if (err > max_err) max_err = err;
                }
    }
    REQUIRE(max_err < 1e-3f, "gather mismatch max=%g", max_err);
    printf("  gather: OK (F=%u,H=%u,d=%u) max_err=%.3g\n", F, H, d, max_err);

cleanup:
    free(prefix); free(suffix); free(alpha); free(text);
    free(log_pref); free(bounds); free(g_gpu);
    if (prefix_t) h3_gpu_tensor_free(prefix_t);
    if (suffix_t) h3_gpu_tensor_free(suffix_t);
    if (alpha_t) h3_gpu_tensor_free(alpha_t);
    if (bounds_t) h3_gpu_tensor_free(bounds_t);
    if (text_t) h3_gpu_tensor_free(text_t);
    if (log_pref_t) h3_gpu_tensor_free(log_pref_t);
    if (gathered_t) h3_gpu_tensor_free(gathered_t);
}

/* Test 7: output projection */
static void test_output(test_context *test) {
    uint32_t F = 1, H = 1, S = 2, d = 4;
    size_t state_count = (size_t)F * H * d * d;
    size_t q_count = (size_t)F * H * S * d;
    size_t gate_count = (size_t)F * H * S * d;
    size_t norm_count = d;

    float *gathered = calloc(state_count, sizeof(float));
    float *q = calloc(q_count, sizeof(float));
    float *gate = calloc(gate_count, sizeof(float));
    float *q_norm_w = calloc(norm_count, sizeof(float));
    REQUIRE(gathered && q && gate && q_norm_w, "cannot allocate test data");

    for (size_t i = 0; i < state_count; i++) { gathered[i] = (float)(i % 5) * 0.1f; }
    for (size_t i = 0; i < q_count; i++) { q[i] = (float)(i % 7) * 0.1f; }
    for (size_t i = 0; i < gate_count; i++) { gate[i] = 0.5f + (float)(i % 3) * 0.1f; }
    for (size_t i = 0; i < norm_count; i++) { q_norm_w[i] = 1.0f; }

    uint16_t *q_bf = malloc(q_count * sizeof(uint16_t));
    uint16_t *gate_bf = malloc(gate_count * sizeof(uint16_t));
    uint16_t *qnw_bf = malloc(norm_count * sizeof(uint16_t));
    for (size_t i = 0; i < q_count; i++) { q_bf[i] = f32_to_bf16(q[i]); }
    for (size_t i = 0; i < gate_count; i++) { gate_bf[i] = f32_to_bf16(gate[i]); }
    for (size_t i = 0; i < norm_count; i++) { qnw_bf[i] = f32_to_bf16(q_norm_w[i]); }

    h3_gpu_tensor *g_t = new_f32(test, state_count);
    h3_gpu_tensor *q_t = new_bf16(test, q_count);
    h3_gpu_tensor *gate_t = new_bf16(test, gate_count);
    h3_gpu_tensor *qnw_t = new_bf16(test, norm_count);
    h3_gpu_tensor *out_t = new_bf16(test, q_count);
    if (!g_t || !q_t || !gate_t || !qnw_t || !out_t) goto cleanup;

    REQUIRE(upload_f32(test, g_t, gathered, state_count), "upload gathered");
    REQUIRE(upload_bf16(test, q_t, q_bf, q_count), "upload q");
    REQUIRE(upload_bf16(test, gate_t, gate_bf, gate_count), "upload gate");
    REQUIRE(upload_bf16(test, qnw_t, qnw_bf, norm_count), "upload q_norm_w");

    int ok = h3_gpu_begin(test->gpu) &&
             h3_gpu_linear_branch_output(test->gpu, out_t, g_t, q_t, gate_t, qnw_t,
                 F, S, H, d, 1e-5f) &&
             h3_gpu_submit(test->gpu);
    REQUIRE(ok, "output dispatch failed");

    uint16_t *out_bf = malloc(q_count * sizeof(uint16_t));
    REQUIRE(out_bf, "cannot allocate readback");
    REQUIRE(readback_bf16(test, out_t, out_bf, q_count), "read output");

    float max_err = 0;
    for (uint32_t f = 0; f < F; f++)
        for (uint32_t h = 0; h < H; h++)
            for (uint32_t s = 0; s < S; s++) {
                size_t q_base = ((f * H + h) * S + s) * d;
                float sum_sq = 0;
                for (uint32_t i = 0; i < d; i++) sum_sq += q[q_base + i] * q[q_base + i];
                float inv_r2 = 1.0f / sqrtf(sum_sq / (float)d + 1e-5f);
                size_t g_base = (f * H + h) * d * d;
                for (uint32_t j = 0; j < d; j++) {
                    float dot = 0;
                    for (uint32_t k = 0; k < d; k++) {
                        float qk = q[q_base + k] * inv_r2 * q_norm_w[k];
                        dot += gathered[g_base + j * d + k] * qk;
                    }
                    float ref = gate[q_base + j] * dot;
                    float gpu_v = bf16_to_f32(out_bf[q_base + j]);
                    float err = fabsf(gpu_v - ref);
                    if (err > max_err) max_err = err;
                }
            }
    REQUIRE(max_err < 5e-2f, "output mismatch max=%g (bf16 precision)", max_err);
    printf("  output: OK (F=%u,H=%u,S=%u,d=%u) max_err=%.3g\n", F, H, S, d, max_err);

cleanup:
    free(gathered); free(q); free(gate); free(q_norm_w);
    free(q_bf); free(gate_bf); free(qnw_bf); free(out_bf);
    if (g_t) h3_gpu_tensor_free(g_t);
    if (q_t) h3_gpu_tensor_free(q_t);
    if (gate_t) h3_gpu_tensor_free(gate_t);
    if (qnw_t) h3_gpu_tensor_free(qnw_t);
    if (out_t) h3_gpu_tensor_free(out_t);
}

int main(void) {
    test_context test = { .label = "linear_branch", .failures = 0 };
    char error[256] = "";
    test.gpu = h3_gpu_create("h3_shaders.metal", error, sizeof(error));
    if (!test.gpu) { fprintf(stderr, "FAIL: cannot create GPU: %s\n", error); return 1; }

    printf("Testing linear far-branch kernels...\n");
    test_window_mask(&test);
    test_frame_stats(&test);
    test_factor_apply(&test);
    test_log_alpha_prefix(&test);
    test_scan_frame(&test);
    test_gather(&test);
    test_output(&test);

    h3_gpu_free(test.gpu);

    if (test.failures) {
        printf("\n%d FAILURES\n", test.failures);
        return 1;
    }
    printf("\nAll linear-branch tests passed.\n");
    return 0;
}
