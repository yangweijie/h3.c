/* Debug FlashAttention kernel step by step. */

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
    uint32_t bits = (union { float f; uint32_t i; }){.f = v}.i;
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

/* Test 1: Simple constant write to verify basic kernel works */
static void test_flash_attn_step1(test_context *test) {
    printf("Step 1: Testing constant write kernel...\n");

    uint32_t seq = 4, heads = 1, head_dim = 4;
    float scale = 1.0f / sqrtf((float)head_dim);
    size_t count = (size_t)seq * heads * head_dim;

    uint16_t *q_bf = malloc(count * sizeof(uint16_t));
    memset(q_bf, 0, count * sizeof(uint16_t));

    h3_gpu_tensor *q_t = new_bf16(test, count);
    h3_gpu_tensor *out_t = new_bf16(test, count);
    if (!q_t || !out_t) goto cleanup;

    REQUIRE(upload_bf16(test, q_t, q_bf, count), "upload q");

    int ok = h3_gpu_begin(test->gpu) &&
             h3_gpu_flash_attn_test_dispatch(test->gpu, out_t, q_t, seq, heads, head_dim, 0) &&
             h3_gpu_submit(test->gpu);
    if (!ok) {
        printf("  ERROR: %s\n", h3_gpu_error(test->gpu));
    }
    REQUIRE(ok, "test dispatch failed");

    uint16_t *out_bf = malloc(count * sizeof(uint16_t));
    REQUIRE(out_bf, "cannot allocate readback");
    REQUIRE(readback_bf16(test, out_t, out_bf, count), "read output");

    printf("  Output: ");
    for (uint32_t i = 0; i < count; i++) {
        printf("%.4f ", bf16_to_f32(out_bf[i]));
    }
    printf("\n");

    /* Check if output is all zeros (expected since input is all zeros) */
    int all_zeros = 1;
    for (uint32_t i = 0; i < count; i++) {
        if (bf16_to_f32(out_bf[i]) != 0.0f) {
            all_zeros = 0;
            break;
        }
    }
    if (all_zeros) {
        printf("  All zeros (expected for zero input)\n");
    } else {
        printf("  NOT all zeros (unexpected)\n");
    }

cleanup:
    free(q_bf); free(out_bf);
    if (q_t) h3_gpu_tensor_free(q_t);
    if (out_t) h3_gpu_tensor_free(out_t);
}

/* Test 2: Simple copy to verify memory access works */
static void test_flash_attn_step2(test_context *test) {
    printf("Step 2: Testing copy kernel...\n");

    uint32_t seq = 4, heads = 1, head_dim = 4;
    float scale = 1.0f / sqrtf((float)head_dim);
    size_t count = (size_t)seq * heads * head_dim;

    uint16_t *q_bf = malloc(count * sizeof(uint16_t));
    for (uint32_t i = 0; i < count; i++) {
        q_bf[i] = f32_to_bf16((float)i * 0.1f);
    }

    h3_gpu_tensor *q_t = new_bf16(test, count);
    h3_gpu_tensor *out_t = new_bf16(test, count);
    if (!q_t || !out_t) goto cleanup;

    REQUIRE(upload_bf16(test, q_t, q_bf, count), "upload q");

    int ok = h3_gpu_begin(test->gpu) &&
             h3_gpu_flash_attn_test_dispatch(test->gpu, out_t, q_t, seq, heads, head_dim, 0) &&
             h3_gpu_submit(test->gpu);
    if (!ok) {
        printf("  ERROR: %s\n", h3_gpu_error(test->gpu));
    }
    REQUIRE(ok, "test dispatch failed");

    uint16_t *out_bf = malloc(count * sizeof(uint16_t));
    REQUIRE(out_bf, "cannot allocate readback");
    REQUIRE(readback_bf16(test, out_t, out_bf, count), "read output");

    printf("  Input:  ");
    for (uint32_t i = 0; i < count; i++) {
        printf("%.4f ", bf16_to_f32(q_bf[i]));
    }
    printf("\n");

    printf("  Output: ");
    for (uint32_t i = 0; i < count; i++) {
        printf("%.4f ", bf16_to_f32(out_bf[i]));
    }
    printf("\n");

    /* Check if output matches input */
    int matches = 1;
    for (uint32_t i = 0; i < count; i++) {
        if (bf16_to_f32(out_bf[i]) != bf16_to_f32(q_bf[i])) {
            matches = 0;
            break;
        }
    }
    if (matches) {
        printf("  PASS: Output matches input\n");
    } else {
        printf("  FAIL: Output does not match input\n");
    }

cleanup:
    free(q_bf); free(out_bf);
    if (q_t) h3_gpu_tensor_free(q_t);
    if (out_t) h3_gpu_tensor_free(out_t);
}

/* Test 3: Minimal kernel that writes constant to verify it runs */
static void test_flash_attn_step3(test_context *test) {
    printf("Step 3: Testing minimal attention kernel (writes constant)...\n");

    uint32_t seq = 2, heads = 1, head_dim = 2;
    float scale = 1.0f / sqrtf((float)head_dim);
    size_t count = (size_t)seq * heads * head_dim;

    float q[4] = {1, 0, 0, 1};
    float k[4] = {1, 0, 0, 1};
    float v[4] = {1, 0, 0, 1};

    uint16_t *q_bf = malloc(count * sizeof(uint16_t));
    uint16_t *k_bf = malloc(count * sizeof(uint16_t));
    uint16_t *v_bf = malloc(count * sizeof(uint16_t));
    for (uint32_t i = 0; i < count; i++) {
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
             h3_gpu_flash_attn_minimal(test->gpu, out_t, q_t, k_t, v_t,
                 seq, heads, head_dim, 0) &&
             h3_gpu_submit(test->gpu);
    if (!ok) {
        printf("  ERROR: %s\n", h3_gpu_error(test->gpu));
    }
    REQUIRE(ok, "minimal dispatch failed");

    uint16_t *out_bf = malloc(count * sizeof(uint16_t));
    REQUIRE(out_bf, "cannot allocate readback");
    REQUIRE(readback_bf16(test, out_t, out_bf, count), "read output");

    printf("  Output (raw): ");
    for (uint32_t i = 0; i < count; i++) printf("0x%04X ", out_bf[i]);
    printf("\n");

    printf("  Output:  ");
    for (uint32_t i = 0; i < count; i++) printf("%.4f ", bf16_to_f32(out_bf[i]));
    printf("\n");
    printf("  Expected: QK^T scores for q[0]=[1,0] and q[1]=[0,1]\n");
    printf("    q[0]: scores = [1.0, 0.0] (q[0]·k[0]=1, q[0]·k[1]=0)\n");
    printf("    q[1]: scores = [0.0, 1.0] (q[1]·k[0]=0, q[1]·k[1]=1)\n");

cleanup:
    free(q_bf); free(k_bf); free(v_bf); free(out_bf);
    if (q_t) h3_gpu_tensor_free(q_t);
    if (k_t) h3_gpu_tensor_free(k_t);
    if (v_t) h3_gpu_tensor_free(v_t);
    if (out_t) h3_gpu_tensor_free(out_t);
}

/* Test 4: Compute attention manually for a simple case */
static void test_flash_attn_step4(test_context *test) {
    printf("Step 4: Testing simple attention computation...\n");

    uint32_t seq = 2, heads = 1, head_dim = 2;
    float scale = 1.0f / sqrtf((float)head_dim);
    size_t count = (size_t)seq * heads * head_dim;

    /* Simple input: q[0] = [1, 0], q[1] = [0, 1] */
    /*               k[0] = [1, 0], k[1] = [0, 1] */
    /*               v[0] = [1, 0], v[1] = [0, 1] */
    float q[4] = {1, 0, 0, 1};
    float k[4] = {1, 0, 0, 1};
    float v[4] = {1, 0, 0, 1};

    uint16_t *q_bf = malloc(count * sizeof(uint16_t));
    uint16_t *k_bf = malloc(count * sizeof(uint16_t));
    uint16_t *v_bf = malloc(count * sizeof(uint16_t));
    for (uint32_t i = 0; i < count; i++) {
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
                 seq, heads, head_dim, scale, 0, 0, 0, 0, 0, 0, 1, 0) &&
             h3_gpu_submit(test->gpu);
    if (!ok) {
        printf("  ERROR: %s\n", h3_gpu_error(test->gpu));
    }
    REQUIRE(ok, "flash_attn dispatch failed");

    uint16_t *out_bf = malloc(count * sizeof(uint16_t));
    REQUIRE(out_bf, "cannot allocate readback");
    REQUIRE(readback_bf16(test, out_t, out_bf, count), "read output");

    printf("  Input q: ");
    for (uint32_t i = 0; i < count; i++) printf("%.4f ", bf16_to_f32(q_bf[i]));
    printf("\n");

    printf("  Input k: ");
    for (uint32_t i = 0; i < count; i++) printf("%.4f ", bf16_to_f32(k_bf[i]));
    printf("\n");

    printf("  Input v: ");
    for (uint32_t i = 0; i < count; i++) printf("%.4f ", bf16_to_f32(v_bf[i]));
    printf("\n");

    printf("  Output:  ");
    for (uint32_t i = 0; i < count; i++) printf("%.4f ", bf16_to_f32(out_bf[i]));
    printf("\n");

    /* Expected:
     * For q[0] = [1, 0]:
     *   scores: q[0]·k[0] = 1, q[0]·k[1] = 0 (but causal, so only k[0])
     *   softmax: [1.0] (only one valid key)
     *   output: 1.0 * v[0] = [1, 0]
     * For q[1] = [0, 1]:
     *   scores: q[1]·k[0] = 0, q[1]·k[1] = 1
     *   softmax: [exp(0)/Z, exp(1)/Z] where Z = exp(0) + exp(1)
     *   output: softmax[0] * v[0] + softmax[1] * v[1]
     */
    printf("  Expected for q[0]: [1.0000, 0.0000]\n");
    printf("  Expected for q[1]: [0.2689, 0.7311] (softmax of [0, 1])\n");

cleanup:
    free(q_bf); free(k_bf); free(v_bf); free(out_bf);
    if (q_t) h3_gpu_tensor_free(q_t);
    if (k_t) h3_gpu_tensor_free(k_t);
    if (v_t) h3_gpu_tensor_free(v_t);
    if (out_t) h3_gpu_tensor_free(out_t);
}

int main(void) {
    test_context test = { .label = "flash_attn_debug", .failures = 0 };
    char error[256] = "";
    test.gpu = h3_gpu_create("h3_shaders.metal", error, sizeof(error));
    if (!test.gpu) { fprintf(stderr, "FAIL: cannot create GPU: %s\n", error); return 1; }

    printf("=== FlashAttention Kernel Debug ===\n\n");
    test_flash_attn_step1(&test);
    printf("\n");
    test_flash_attn_step2(&test);
    printf("\n");
    test_flash_attn_step3(&test);
    printf("\n");
    test_flash_attn_step4(&test);

    h3_gpu_free(test.gpu);

    if (test.failures) {
        printf("\n%d FAILURES\n", test.failures);
        return 1;
    }
    printf("\nAll debug steps completed.\n");
    return 0;
}
