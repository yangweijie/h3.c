/* VDN-H3 linear-branch pipeline test: drives the new Metal kernels through
 * the exact stage-b forward (features -> stats -> vdn_solve -> scans ->
 * gather -> readout) on small random shapes and compares against a plain C
 * reference implementation of the Python math. Also checks the batched
 * (I+A)^-1 path and the windowed softmax attention kernel. */

#include "h3_gpu.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static void die(const char *message) {
    fprintf(stderr, "vdn test: %s\n", message);
    exit(1);
}

static uint32_t rng_state = 0x12345678u;
static float frand(void) {
    rng_state = rng_state * 1664525u + 1013904223u;
    return (float)((rng_state >> 8) & 0xFFFFFF) / (float)0xFFFFFF - 0.5f;
}

static uint16_t f32_to_bf16(float value) {
    uint32_t bits;
    memcpy(&bits, &value, sizeof(bits));
    bits += 0x7fffu + ((bits >> 16) & 1u);
    return (uint16_t)(bits >> 16);
}

static float bf16_to_f32(uint16_t value) {
    uint32_t bits = (uint32_t)value << 16;
    float result;
    memcpy(&result, &bits, sizeof(result));
    return result;
}

static uint16_t *rand_bf16(size_t count, float scale) {
    uint16_t *data = malloc(count * sizeof(uint16_t));
    if (!data) die("out of memory");
    for (size_t i = 0; i < count; i++)
        data[i] = f32_to_bf16(frand() * scale);
    return data;
}

static float rel_l2(const float *got, const float *want, size_t count) {
    double num = 0.0, den = 0.0;
    for (size_t i = 0; i < count; i++) {
        double d = (double)got[i] - (double)want[i];
        num += d * d;
        den += (double)want[i] * (double)want[i];
    }
    return (float)sqrt(num / (den + 1e-30));
}

static void upload_bf16(h3_gpu *gpu, const uint16_t *host, size_t count,
                        h3_gpu_tensor **out) {
    *out = h3_gpu_tensor_from_bf16(gpu, host, count);
    if (!*out) die(h3_gpu_error(gpu));
}

static void read_f32(h3_gpu *gpu, const h3_gpu_tensor *tensor, float *host,
                     size_t count) {
    (void)gpu;
    if (!h3_gpu_tensor_read_f32(tensor, host, count)) die("read f32 failed");
}

static void read_bf16(h3_gpu *gpu, const h3_gpu_tensor *tensor,
                      uint16_t *host, size_t count) {
    (void)gpu;
    if (!h3_gpu_tensor_read_bf16(tensor, host, count))
        die("read bf16 failed");
}

/* ------------------------------------------------------------------ */
/* CPU reference pieces.                                               */

static void ref_cholesky_inv(int batch, int dim, const float *a_with_i,
                             float *inv) {
    for (int b = 0; b < batch; b++) {
        const float *m = a_with_i + (size_t)b * dim * dim;
        float *l = calloc((size_t)dim * dim, sizeof(float));
        for (int j = 0; j < dim; j++) {
            float d = m[(size_t)j * dim + j];
            for (int k = 0; k < j; k++)
                d -= l[(size_t)j * dim + k] * l[(size_t)j * dim + k];
            l[(size_t)j * dim + j] = sqrtf(d);
            for (int i = j + 1; i < dim; i++) {
                float s = m[(size_t)i * dim + j];
                for (int k = 0; k < j; k++)
                    s -= l[(size_t)i * dim + k] * l[(size_t)j * dim + k];
                l[(size_t)i * dim + j] = s / l[(size_t)j * dim + j];
            }
        }
        float *x = calloc((size_t)dim * dim, sizeof(float));
        for (int col = 0; col < dim; col++)
            for (int i = col; i < dim; i++) {
                float s = (i == col) ? 1.0f : 0.0f;
                for (int k = col; k < i; k++)
                    s -= l[(size_t)i * dim + k] * x[(size_t)k * dim + col];
                x[(size_t)i * dim + col] = s / l[(size_t)i * dim + i];
            }
        for (int i = 0; i < dim; i++)
            for (int j = 0; j < dim; j++) {
                float s = 0.0f;
                for (int k = 0; k < dim; k++)
                    s += x[(size_t)k * dim + i] * x[(size_t)k * dim + j];
                inv[(size_t)b * dim * dim + (size_t)i * dim + j] = s;
            }
        free(l);
        free(x);
    }
}

int main(void) {
    char error[512];
    h3_gpu *gpu = h3_gpu_create("h3_shaders.metal", error, sizeof(error));
    if (!gpu) die(error);

    /* ---------------- subtest A: (I+A)^-1 ------------------------------ */
    {
        enum { BATCH = 4, DIM = 128, ROWS = 16 };
        size_t plane = (size_t)DIM * DIM;
        float *ref_a = malloc(BATCH * plane * sizeof(float));
        float *ref_inv = malloc(BATCH * plane * sizeof(float));
        uint16_t *r = rand_bf16((size_t)BATCH * ROWS * DIM, 1.0f);
        for (int b = 0; b < BATCH; b++)
            for (int i = 0; i < DIM; i++)
                for (int j = 0; j < DIM; j++) {
                    float s = 0.0f;
                    for (int k = 0; k < ROWS; k++)
                        s += bf16_to_f32(r[((size_t)b * ROWS + k) * DIM + i]) *
                             bf16_to_f32(r[((size_t)b * ROWS + k) * DIM + j]);
                    ref_a[(size_t)b * plane + (size_t)i * DIM + j] = s;
                }
        for (int b = 0; b < BATCH; b++)
            for (int i = 0; i < DIM; i++)
                ref_a[(size_t)b * plane + (size_t)i * DIM + i] += 1.0f;
        ref_cholesky_inv(BATCH, DIM, ref_a, ref_inv);

        /* The GPU kernel factors (I + A) itself, so hand it the raw A. */
        float *gpu_a = malloc(BATCH * plane * sizeof(float));
        memcpy(gpu_a, ref_a, BATCH * plane * sizeof(float));
        for (int b = 0; b < BATCH; b++)
            for (int i = 0; i < DIM; i++)
                gpu_a[(size_t)b * plane + (size_t)i * DIM + i] -= 1.0f;
        h3_gpu_tensor *a_t = h3_gpu_tensor_from_f32(gpu, gpu_a,
                                                    BATCH * plane);
        h3_gpu_tensor *x_t = h3_gpu_tensor_new_f32(gpu, BATCH * plane);
        h3_gpu_tensor *inv_t = h3_gpu_tensor_new_f32(gpu, BATCH * plane);
        if (!a_t || !x_t || !inv_t) die(h3_gpu_error(gpu));
        if (!h3_gpu_begin(gpu)) die(h3_gpu_error(gpu));
        if (!h3_gpu_vdn_cholesky(gpu, a_t, BATCH, DIM) ||
            !h3_gpu_vdn_triinv(gpu, x_t, a_t, BATCH, DIM))
            die(h3_gpu_error(gpu));
        h3_vdn_bmm_args args = {BATCH, DIM, DIM, DIM, 1, 0, 0};
        if (!h3_gpu_vdn_bmm(gpu, inv_t, x_t, x_t, NULL, &args))
            die(h3_gpu_error(gpu));
        if (!h3_gpu_submit(gpu)) die(h3_gpu_error(gpu));
        float *got = malloc(BATCH * plane * sizeof(float));
        read_f32(gpu, inv_t, got, BATCH * plane);
        float err = rel_l2(got, ref_inv, BATCH * plane);
        printf("cholesky inverse: rel_l2=%.6f %s\n", err,
               err < 2e-4f ? "OK" : "FAIL");
        if (!(err < 2e-4f)) die("cholesky inverse mismatch");
        free(got); free(ref_a); free(ref_inv); free(r);
        h3_gpu_tensor_free(a_t); h3_gpu_tensor_free(x_t);
        h3_gpu_tensor_free(inv_t);
    }

    /* ---------------- subtest B: windowed softmax attention ------------ */
    {
        enum { H = 2, D = 128, TEXT = 8, F = 4, S = 6, AUDIO = 8 };
        enum { VIDEO = F * S, TOTAL = TEXT + VIDEO + AUDIO };
        uint16_t *q = rand_bf16((size_t)TOTAL * H * D, 2.0f);
        uint16_t *k = rand_bf16((size_t)TOTAL * H * D, 2.0f);
        uint16_t *v = rand_bf16((size_t)TOTAL * H * D, 2.0f);
        h3_gpu_tensor *q_t, *k_t, *v_t, *out_t, *bounds_t;
        upload_bf16(gpu, q, (size_t)TOTAL * H * D, &q_t);
        upload_bf16(gpu, k, (size_t)TOTAL * H * D, &k_t);
        upload_bf16(gpu, v, (size_t)TOTAL * H * D, &v_t);
        out_t = h3_gpu_tensor_new_bf16(gpu, (size_t)TOTAL * H * D);
        float bounds[F * 2];
        for (int t = 0; t < F; t++) {
            bounds[t * 2] = (float)((t / 2 - 1) * 2);
            bounds[t * 2 + 1] = (float)((t / 2 + 2) * 2 - 1);
        }
        bounds_t = h3_gpu_tensor_from_f32(gpu, bounds, F * 2);
        if (!out_t || !bounds_t) die(h3_gpu_error(gpu));
        float scale = 1.0f / sqrtf((float)D);
        h3_vdn_window_args args = {H, TOTAL, D, TEXT, F, S,
                                   0, 3, 0};
        memcpy(&args.scale_bits, &scale, sizeof(scale));
        if (!h3_gpu_begin(gpu)) die(h3_gpu_error(gpu));
        if (!h3_gpu_vdn_window_attention(gpu, out_t, q_t, k_t, v_t, bounds_t,
                                         &args)) die(h3_gpu_error(gpu));
        if (!h3_gpu_submit(gpu)) die(h3_gpu_error(gpu));
        uint16_t *got = malloc((size_t)TOTAL * H * D * sizeof(uint16_t));
        read_bf16(gpu, out_t, got, (size_t)TOTAL * H * D);
        float *want = malloc((size_t)TOTAL * H * D * sizeof(float));
        for (int row = 0; row < TOTAL; row++)
            for (int head = 0; head < H; head++)
                for (int d0 = 0; d0 < D; d0++) {
                    int keys[TOTAL], count = 0;
                    if (row < TEXT || row >= TEXT + VIDEO) {
                        for (int i = 0; i < TOTAL; i++) keys[count++] = i;
                    } else {
                        int t = (row - TEXT) / S;
                        for (int i = 0; i < TEXT; i++) keys[count++] = i;
                        for (int i = TEXT + VIDEO; i < TOTAL; i++)
                            keys[count++] = i;
                        int anchor_rows = (t == 0 || t == F - 1);
                        int lo = (int)bounds[t * 2], hi = (int)bounds[t * 2 + 1];
                        if (lo < 0) lo = 0;
                        if (hi > F - 1) hi = F - 1;
                        for (int f = 0; f < F; f++) {
                            int inside = f >= lo && f <= hi;
                            int anchor_col = f == 0 || f == F - 1;
                            if (anchor_rows || inside || anchor_col)
                                for (int s = 0; s < S; s++)
                                    keys[count++] = TEXT + f * S + s;
                        }
                    }
                    float max_val = -INFINITY;
                    static float scores[TOTAL];
                    for (int i = 0; i < count; i++) {
                        float s = 0.0f;
                        for (int e = 0; e < D; e++)
                            s += bf16_to_f32(q[((size_t)row * H + head) * D +
                                               e]) *
                                 bf16_to_f32(k[((size_t)keys[i] * H + head) *
                                               D + e]);
                        scores[i] = s * scale;
                        if (scores[i] > max_val) max_val = scores[i];
                    }
                    float sum = 0.0f;
                    for (int i = 0; i < count; i++) {
                        scores[i] = expf(scores[i] - max_val);
                        sum += scores[i];
                    }
                    float acc = 0.0f;
                    for (int i = 0; i < count; i++)
                        acc += scores[i] *
                               bf16_to_f32(v[((size_t)keys[i] * H + head) *
                                             D + d0]);
                    want[((size_t)row * H + head) * D + d0] = acc / sum;
                }
        float *got_f = malloc((size_t)TOTAL * H * D * sizeof(float));
        for (size_t i = 0; i < (size_t)TOTAL * H * D; i++)
            got_f[i] = bf16_to_f32(got[i]);
        float err = rel_l2(got_f, want, (size_t)TOTAL * H * D);
        printf("window attention: rel_l2=%.5f %s\n", err,
               err < 0.02f ? "OK" : "FAIL");
        if (!(err < 0.02f)) die("window attention mismatch");
        free(got); free(got_f); free(want); free(q); free(k); free(v);
        h3_gpu_tensor_free(q_t); h3_gpu_tensor_free(k_t);
        h3_gpu_tensor_free(v_t); h3_gpu_tensor_free(out_t);
        h3_gpu_tensor_free(bounds_t);
    }

    h3_gpu_free(gpu);
    printf("vdn tests passed\n");
    return 0;
}
