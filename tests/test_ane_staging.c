/* Metal round-trip for the Neural Engine activation staging.
 *
 * h3_gpu_pack_ane_input_bf16() turns a row-major BF16 activation into the
 * channel-major F32 plane an ANE graph reads ([1, chunk_dim, 1, plane_rows],
 * token axis innermost, padded tail zero); h3_gpu_unpack_ane_output_bf16()
 * does the inverse for one output slice.  The ANE gates never touch these two
 * kernels -- the host test writes planes directly -- so they are only covered
 * here, before any DiT projection is routed through the Neural Engine.
 *
 * Values are chosen to be exact in BF16 so every comparison below is equality,
 * not a tolerance.
 */

#include "h3_gpu.h"

#include <stdarg.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

enum {
    ROWS = 40,           /* tokens, deliberately not a multiple of 16 */
    PLANE_ROWS = 48,     /* ROWS padded up to the ANE's 16-row plane */
    IN_DIM = 1280,
    CHUNK = 512,         /* K per graph input */
    OUT_DIM = 384
};

static int failures = 0;

static uint16_t f32_to_bf16(float value) {
    uint32_t bits;
    memcpy(&bits, &value, sizeof(bits));
    return (uint16_t)(bits >> 16);
}

static float bf16_to_f32(uint16_t value) {
    uint32_t bits = (uint32_t)value << 16;
    float result;
    memcpy(&result, &bits, sizeof(result));
    return result;
}

static void check(int ok, const char *format, ...) {
    va_list arguments;
    fprintf(stdout, ok ? "ok:   " : "FAIL: ");
    va_start(arguments, format);
    vprintf(format, arguments);
    va_end(arguments);
    fputc('\n', stdout);
    if (!ok) failures++;
}

int main(void) {
    char error[256] = {0};
    h3_gpu *gpu = h3_gpu_create("h3_shaders.metal", error, sizeof(error));
    if (!gpu) {
        fprintf(stderr, "FAIL: cannot create gpu: %s\n", error);
        return 1;
    }
    uint16_t *input = malloc(sizeof(uint16_t) * ROWS * IN_DIM);
    float *plane = malloc(sizeof(float) * PLANE_ROWS * CHUNK);
    uint16_t *unpacked = malloc(sizeof(uint16_t) * ROWS * OUT_DIM);
    h3_gpu_tensor *activation = h3_gpu_tensor_new_bf16(gpu,
                                                      (size_t)ROWS * IN_DIM);
    h3_gpu_tensor *staging = h3_gpu_tensor_new_f32(gpu,
                                                  (size_t)PLANE_ROWS * CHUNK);
    h3_gpu_tensor *result = h3_gpu_tensor_new_bf16(gpu,
                                                  (size_t)ROWS * OUT_DIM);
    h3_gpu_tensor *source = h3_gpu_tensor_new_f32(gpu,
                                                 (size_t)PLANE_ROWS * OUT_DIM);
    if (!input || !plane || !unpacked || !activation || !staging ||
        !result || !source) {
        fprintf(stderr, "FAIL: allocation\n");
        return 1;
    }
    /* Exact-in-BF16 activations: half-integers scaled by 1/8. */
    for (size_t i = 0; i < (size_t)ROWS * IN_DIM; i++)
        input[i] = f32_to_bf16((float)((int)(i % 511) - 255) / 8.0f);
    if (!h3_gpu_tensor_write_bf16(activation, input, (size_t)ROWS * IN_DIM)) {
        fprintf(stderr, "FAIL: upload activation: %s\n", h3_gpu_error(gpu));
        return 1;
    }

    if (!h3_gpu_begin(gpu)) {
        fprintf(stderr, "FAIL: begin: %s\n", h3_gpu_error(gpu));
        return 1;
    }
    for (uint32_t chunk = 0; chunk * CHUNK < IN_DIM; chunk++) {
        uint32_t base = chunk * CHUNK;
        if (!h3_gpu_pack_ane_input_bf16(gpu, staging, activation, ROWS, IN_DIM,
                                        base, CHUNK, PLANE_ROWS)) {
            fprintf(stderr, "FAIL: pack chunk %u: %s\n", chunk,
                    h3_gpu_error(gpu));
            return 1;
        }
        if (!h3_gpu_submit(gpu) || !h3_gpu_begin(gpu)) {
            fprintf(stderr, "FAIL: submit pack %u: %s\n", chunk,
                    h3_gpu_error(gpu));
            return 1;
        }
        if (!h3_gpu_tensor_read_f32(staging, plane,
                                    (size_t)PLANE_ROWS * CHUNK)) {
            fprintf(stderr, "FAIL: read plane %u\n", chunk);
            return 1;
        }
        uint32_t span = IN_DIM - base < CHUNK ? IN_DIM - base : CHUNK;
        size_t misplaced = 0, nonzero_pad = 0;
        for (uint32_t c = 0; c < CHUNK; c++)
            for (uint32_t r = 0; r < PLANE_ROWS; r++) {
                float got = plane[(size_t)c * PLANE_ROWS + r];
                float want = r < ROWS && c < span ?
                    bf16_to_f32(input[(size_t)r * IN_DIM + base + c]) : 0.0f;
                if (got != want) {
                    if (want == 0.0f) nonzero_pad++;
                    else misplaced++;
                }
            }
        check(misplaced == 0 && nonzero_pad == 0,
              "pack chunk %u (base %u, span %u): %zu misplaced, %zu unpadded "
              "tail entries", chunk, base, span, misplaced, nonzero_pad);
    }
    if (!h3_gpu_submit(gpu)) {
        fprintf(stderr, "FAIL: final submit: %s\n", h3_gpu_error(gpu));
        return 1;
    }

    /* Unpack: the plane holds [output_dim][plane_rows], the tensor is
     * [rows][output_dim] BF16, and the padded token tail must be dropped. */
    for (size_t i = 0; i < (size_t)PLANE_ROWS * OUT_DIM; i++)
        plane[i] = (float)((int)(i % 253) - 126) / 8.0f;
    if (!h3_gpu_tensor_write_f32(source, plane, (size_t)PLANE_ROWS * OUT_DIM) ||
        !h3_gpu_begin(gpu)) {
        fprintf(stderr, "FAIL: upload plane: %s\n", h3_gpu_error(gpu));
        return 1;
    }
    /* A too-small destination must be refused rather than scribbled past.
     * Checked inside the same command window, so the only reason to fail here
     * is the element guard and not an inactive command buffer. */
    check(!h3_gpu_unpack_ane_output_bf16(gpu, result, source, ROWS + 1, OUT_DIM,
                                         PLANE_ROWS),
          "unpack into an undersized output is refused: %s", h3_gpu_error(gpu));
    if (!h3_gpu_unpack_ane_output_bf16(gpu, result, source, ROWS, OUT_DIM,
                                       PLANE_ROWS)) {
        fprintf(stderr, "FAIL: unpack: %s\n", h3_gpu_error(gpu));
        return 1;
    }
    if (!h3_gpu_submit(gpu) ||
        !h3_gpu_tensor_read_bf16(result, unpacked, (size_t)ROWS * OUT_DIM)) {
        fprintf(stderr, "FAIL: read unpacked: %s\n", h3_gpu_error(gpu));
        return 1;
    }
    size_t wrong = 0;
    for (uint32_t r = 0; r < ROWS; r++)
        for (uint32_t n = 0; n < OUT_DIM; n++)
            if (unpacked[(size_t)r * OUT_DIM + n] !=
                f32_to_bf16(plane[(size_t)n * PLANE_ROWS + r])) wrong++;
    check(wrong == 0, "unpack %ux%u from the padded plane: %zu mismatches",
          ROWS, OUT_DIM, wrong);

    free(input); free(plane); free(unpacked);
    h3_gpu_tensor_free(activation);
    h3_gpu_tensor_free(staging);
    h3_gpu_tensor_free(result);
    h3_gpu_tensor_free(source);
    h3_gpu_free(gpu);
    printf("%d failure(s)\n", failures);
    return failures ? 1 : 0;
}
