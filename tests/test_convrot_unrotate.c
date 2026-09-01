#include "h3_gpu.h"

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* End-to-end check of the ConvRot int8 recovery path:
 *   W_true -> rotate by normalized block-Hadamard (W_rot = W_true @ R)
 *          -> symmetric per-row int8 quantize -> dequant + unrotate kernel
 *          -> should recover W_true (up to int8 + bf16 precision).
 * This validates the H_256 generation, the Metal kernel indexing, and the C
 * wrapper. Pass => H is orthonormal (H@H == I) and the unrotation is correct. */
#define ROWS 64
#define COLS 512 /* must be divisible by 256 */
#define BLOCK 256

static uint16_t f32_to_bf16(float v) {
    uint32_t b;
    memcpy(&b, &v, sizeof(b));
    return (uint16_t)(b >> 16);
}

static float bf16_to_f32(uint16_t v) {
    uint32_t b = (uint32_t)v << 16;
    float f;
    memcpy(&f, &b, sizeof(f));
    return f;
}

/* Build the ConvRot rotation matrix of size n: the official implementation
 * runs a radix-4 butterfly over the identity (stages stride=1,4,16,...; each
 * quad (a,b,c,d) -> (a+b+c-d, a+b-c+d, a-b+c+d, -a+b+c+d)), then scales by
 * 1/sqrt(n). This is NOT the natural-order Sylvester Hadamard. */
static void build_hadamard_bf16(uint16_t *out, int n) {
    float *h = malloc((size_t)n * n * sizeof(float));
    for (int i = 0; i < n; i++)
        for (int j = 0; j < n; j++)
            h[i * n + j] = (i == j) ? 1.0f : 0.0f;
    for (int stride = 1; stride < n; stride *= 4) {
        int span = stride * 4;
        for (int base = 0; base < n; base += span)
            for (int lane = 0; lane < stride; lane++)
                for (int row = 0; row < n; row++) {
                    float *r = h + (size_t)row * n;
                    int i0 = base + lane, i1 = i0 + stride,
                        i2 = i0 + 2 * stride, i3 = i0 + 3 * stride;
                    float a = r[i0], b = r[i1], c = r[i2], d = r[i3];
                    r[i0] = a + b + c - d;
                    r[i1] = a + b - c + d;
                    r[i2] = a - b + c + d;
                    r[i3] = -a + b + c + d;
                }
    }
    float norm = 1.0f / sqrtf((float)n);
    for (int i = 0; i < n * n; i++)
        out[i] = f32_to_bf16(h[i] * norm);
    free(h);
}

/* Mirror of h3_dit.c convrot_unrotate_cpu: radix-4 ConvRot butterfly on the
 * dequantized int8 weight (exact in f32 up to 2^24), then scale * 1/16. */
static void fwht_unrotate_cpu(const int8_t *weight, const float *scales,
                              uint16_t *out, int rows, int columns) {
    float values[BLOCK];
    for (int row = 0; row < rows; row++) {
        const int8_t *source = weight + (size_t)row * columns;
        uint16_t *target = out + (size_t)row * columns;
        float scale = scales[row];
        for (int blk = 0; blk < columns; blk += BLOCK) {
            for (int k = 0; k < BLOCK; k++) values[k] = (float)source[blk + k];
            for (int stride = 1; stride < BLOCK; stride *= 4) {
                int span = stride * 4;
                for (int base = 0; base < BLOCK; base += span)
                    for (int lane = 0; lane < stride; lane++) {
                        int i0 = base + lane, i1 = i0 + stride,
                            i2 = i0 + 2 * stride, i3 = i0 + 3 * stride;
                        float a = values[i0], b = values[i1],
                              c = values[i2], d = values[i3];
                        values[i0] = a + b + c - d;
                        values[i1] = a + b - c + d;
                        values[i2] = a - b + c + d;
                        values[i3] = -a + b + c + d;
                    }
            }
            for (int k = 0; k < BLOCK; k++)
                target[blk + k] = f32_to_bf16(
                    scale * values[k] * (1.0f / 16.0f));
        }
    }
}

int main(void) {
    char error[256];
    h3_gpu *gpu = h3_gpu_create("h3_shaders.metal", error, sizeof(error));
    if (!gpu) {
        fprintf(stderr, "FAIL: cannot create gpu: %s\n", error);
        return 1;
    }

    uint16_t *h = malloc(sizeof(uint16_t) * BLOCK * BLOCK);
    build_hadamard_bf16(h, BLOCK);

    float *wtrue = malloc(sizeof(float) * ROWS * COLS);
    int8_t *qi8 = malloc(sizeof(int8_t) * ROWS * COLS);
    float *scale = malloc(sizeof(float) * ROWS);
    float *wrot = calloc((size_t)ROWS * COLS, sizeof(float));
    uint16_t *got = malloc(sizeof(uint16_t) * ROWS * COLS);
    if (!h || !wtrue || !qi8 || !scale || !wrot || !got) {
        fprintf(stderr, "FAIL: allocation\n");
        return 1;
    }

    srand(12345);
    float amax_global = 0.0f;
    for (int i = 0; i < ROWS * COLS; i++) {
        wtrue[i] = ((float)rand() / RAND_MAX) * 2.0f - 1.0f;
        float a = fabsf(wtrue[i]);
        if (a > amax_global) amax_global = a;
    }

    /* W_rot = W_true @ R (block-diagonal normalized Hadamard along columns). */
    for (int i = 0; i < ROWS; i++)
        for (int o = 0; o < COLS; o++) {
            int block = o / BLOCK, local = o % BLOCK;
            float acc = 0.0f;
            for (int k = 0; k < BLOCK; k++)
                acc += wtrue[i * COLS + block * BLOCK + k] *
                       bf16_to_f32(h[local * BLOCK + k]);
            wrot[i * COLS + o] = acc;
        }

    /* Symmetric per-row int8 quantization (matches h3's own convrot-style scale). */
    for (int i = 0; i < ROWS; i++) {
        float amax = 0.0f;
        for (int k = 0; k < COLS; k++) {
            float a = fabsf(wrot[i * COLS + k]);
            if (a > amax) amax = a;
        }
        float s = amax > 1e-12f ? amax / 127.0f : 1.0f;
        scale[i] = s;
        for (int k = 0; k < COLS; k++) {
            long q = lroundf(wrot[i * COLS + k] / s);
            if (q > 127) q = 127;
            if (q < -128) q = -128;
            qi8[i * COLS + k] = (int8_t)q;
        }
    }

    FILE *f = fopen("/tmp/convrot_i8.bin", "wb");
    if (!f || fwrite(qi8, 1, (size_t)ROWS * COLS, f) != (size_t)ROWS * COLS) {
        fprintf(stderr, "FAIL: cannot write temp int8\n");
        return 1;
    }
    fclose(f);

    /* CPU reference: replicate the kernel math exactly (int8 + bf16 H) to isolate
     * any GPU-side bug. out_ref[i,o] = scale[i] * sum_k H[local,k] int8[i, blk*256+k]. */
    float *ref = malloc(sizeof(float) * ROWS * COLS);
    float *exact = malloc(sizeof(float) * ROWS * COLS);
    double ref_max = 0.0, exact_max = 0.0;
    for (int i = 0; i < ROWS; i++)
        for (int o = 0; o < COLS; o++) {
            int blk = o / BLOCK, local = o % BLOCK;
            float acc = 0.0f, accx = 0.0f;
            for (int k = 0; k < BLOCK; k++) {
                acc += bf16_to_f32(h[local * BLOCK + k]) *
                       (float)qi8[i * COLS + blk * BLOCK + k];
                accx += bf16_to_f32(h[local * BLOCK + k]) *
                        wrot[i * COLS + blk * BLOCK + k];
            }
            ref[i * COLS + o] = scale[i] * acc;
            exact[i * COLS + o] = accx;
            double d = fabs((double)ref[i * COLS + o] - (double)wtrue[i * COLS + o]);
            if (d > ref_max) ref_max = d;
            double dx = fabs((double)exact[i * COLS + o] - (double)wtrue[i * COLS + o]);
            if (dx > exact_max) exact_max = dx;
        }
    printf("cpu-ref(int8) vs wtrue: max_abs=%.5f\n", ref_max);
    printf("cpu exact(wrot) unrotate vs wtrue: max_abs=%.5f\n", exact_max);
    /* verify H@H == I */
    double hh = 0.0;
    for (int a = 0; a < BLOCK; a++)
        for (int b = 0; b < BLOCK; b++) {
            double s = 0.0;
            for (int k = 0; k < BLOCK; k++)
                s += (double)bf16_to_f32(h[a * BLOCK + k]) *
                      (double)bf16_to_f32(h[b * BLOCK + k]);
            double e = fabs(s - (a == b ? 1.0 : 0.0));
            if (e > hh) hh = e;
        }
    printf("H@H vs I: max_abs=%.5f\n", hh);
    printf("samples: wtrue[0]=%.4f wrot[0]=%.4f int8[0]=%d scale[0]=%.4f "
           "H[0,0]=%.4f H[0,1]=%.4f ref[0]=%.4f exact[0]=%.4f\n",
           wtrue[0], wrot[0], (int)qi8[0], scale[0],
           bf16_to_f32(h[0]), bf16_to_f32(h[1]), ref[0], exact[0]);

    h3_gpu_tensor *weight =
        h3_gpu_tensor_load_i8(gpu, "/tmp/convrot_i8.bin", 0, (size_t)ROWS * COLS);
    h3_gpu_tensor *sc = h3_gpu_tensor_from_f32(gpu, scale, (size_t)ROWS);
    h3_gpu_tensor *had = h3_gpu_tensor_from_bf16(gpu, h, (size_t)BLOCK * BLOCK);
    h3_gpu_tensor *out = h3_gpu_tensor_new_bf16(gpu, (size_t)ROWS * COLS);
    if (!weight || !sc || !had || !out) {
        fprintf(stderr, "FAIL: tensor allocation: %s\n", h3_gpu_error(gpu));
        return 1;
    }

    if (!h3_gpu_begin(gpu)) {
        fprintf(stderr, "FAIL: begin: %s\n", h3_gpu_error(gpu));
        return 1;
    }
    if (!h3_gpu_weight_dequant_unrotate_int8(gpu, out, weight, sc, had, ROWS,
                                            COLS, 0, 0, 0)) {
        fprintf(stderr, "FAIL: dequant+unrotate: %s\n", h3_gpu_error(gpu));
        return 1;
    }
    if (!h3_gpu_submit(gpu)) {
        fprintf(stderr, "FAIL: submit: %s\n", h3_gpu_error(gpu));
        return 1;
    }
    if (!h3_gpu_tensor_read_bf16(out, got, (size_t)ROWS * COLS)) {
        fprintf(stderr, "FAIL: read back\n");
        return 1;
    }

    double max_abs = 0.0, max_vs_ref = 0.0;
    for (int i = 0; i < ROWS * COLS; i++) {
        double d = fabs((double)bf16_to_f32(got[i]) - (double)wtrue[i]);
        if (d > max_abs) max_abs = d;
        double dr = fabs((double)bf16_to_f32(got[i]) - (double)ref[i]);
        if (dr > max_vs_ref) max_vs_ref = dr;
    }
    printf("convrot unrotate: max_abs=%.5f amax_global=%.5f ratio=%.4f "
           "gpu_vs_cpu_ref=%.5f\n",
           max_abs, (double)amax_global, max_abs / (double)amax_global, max_vs_ref);
    printf("gpu[0]=%.4f ref[0]=%.4f wtrue[0]=%.4f\n",
           bf16_to_f32(got[0]), ref[0], wtrue[0]);

    /* CPU FWHT path (used by SSD streaming): must match wtrue and the GPU. */
    uint16_t *cpu_out = malloc(sizeof(uint16_t) * (size_t)ROWS * COLS);
    if (!cpu_out) {
        fprintf(stderr, "FAIL: allocation\n");
        return 1;
    }
    fwht_unrotate_cpu(qi8, scale, cpu_out, ROWS, COLS);
    double cpu_max = 0.0, cpu_vs_gpu = 0.0;
    for (int i = 0; i < ROWS * COLS; i++) {
        double d = fabs((double)bf16_to_f32(cpu_out[i]) - (double)wtrue[i]);
        if (d > cpu_max) cpu_max = d;
        double dg = fabs((double)bf16_to_f32(cpu_out[i]) -
                         (double)bf16_to_f32(got[i]));
        if (dg > cpu_vs_gpu) cpu_vs_gpu = dg;
    }
    printf("cpu-fwht unrotate vs wtrue: max_abs=%.5f  vs gpu: %.5f\n",
           cpu_max, cpu_vs_gpu);

    int ok = max_abs < 0.05 * (double)amax_global &&
             cpu_max < 0.05 * (double)amax_global;
    h3_gpu_tensor_free(weight);
    h3_gpu_tensor_free(sc);
    h3_gpu_tensor_free(had);
    h3_gpu_tensor_free(out);
    h3_gpu_free(gpu);
    free(h);
    free(wtrue);
    free(qi8);
    free(scale);
    free(wrot);
    free(got);
    if (!ok) {
        fprintf(stderr, "FAIL: recovered weight error too large\n");
        return 1;
    }
    printf("PASS tests/test_convrot_unrotate.c\n");
    return 0;
}
