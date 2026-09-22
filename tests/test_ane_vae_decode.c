/* A/B the wired video VAE decoder: the same latent decoded with the Neural
 * Engine off and on, compared pixel by pixel and timed.
 *
 * H3_ANE_VAE is a runtime switch inside h3_video_vae.c, so the only way to see
 * whether the graphs agree with the fp32 Metal path is to run both in one
 * process on one input. The Metal pass runs first: it is the reference, and if
 * the ANE pass then fails we still have the baseline on screen.
 *
 * A 16x16 latent tile is 1797 rows, the shape the residency gate measured, and
 * latent_time 12 makes two chunks share one walk of the block stack, so this
 * also covers the cross-chunk path.
 *
 * Usage (from the repo root):
 *     ./h3_ane_vae_decode_test VIDEO_VAE_SOURCE_DIR [LATENT_T] [LATENT_HW] [REPEAT]
 *
 * The Neural Engine pass builds its graphs inside the timed region, so REPEAT
 * decodes (>= 1) report the first pass next to the steady-state one: a win that
 * only exists after the build cost is paid cannot be seen otherwise.
 *
 * H3_ANE_VAE_MAX_BLOCKS and the other ANE knobs are inherited from the
 * environment.
 */

#include "h3_video_vae.h"

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#define CHANNELS 24

static uint64_t rng_state = 0x9E3779B97F4A7C15ull;

static float rng_next(void) {
    rng_state ^= rng_state >> 12;
    rng_state ^= rng_state << 25;
    rng_state ^= rng_state >> 27;
    return (float)((double)rng_state / 18446744073709551616.0) - 0.5f;
}

static double now_seconds(void) {
    struct timespec stamp;
    clock_gettime(CLOCK_MONOTONIC, &stamp);
    return (double)stamp.tv_sec + (double)stamp.tv_nsec * 1e-9;
}

typedef struct {
    float *rgb;
    size_t count;
    int frames, height, width;
    double seconds;
} decoded;

static void decode_once(const char *weights, const float *latent, int latent_t,
                        int latent_hw, const char *label, decoded *result) {
    char error[512] = {0};
    h3_video_frames frames;
    memset(&frames, 0, sizeof(frames));
    memset(result, 0, sizeof(*result));
    double started = now_seconds();
    if (!h3_video_vae_decode(weights, "h3_shaders.metal", latent, latent_t,
                             latent_hw, latent_hw, NULL, NULL, 1, &frames,
                             NULL, NULL, error, sizeof(error))) {
        fprintf(stderr, "FAIL: %s decode failed: %s\n", label, error);
        exit(1);
    }
    result->seconds = now_seconds() - started;
    result->rgb = frames.rgb;
    result->count = (size_t)frames.frames * (size_t)frames.height *
                    (size_t)frames.width * 3;
    result->frames = frames.frames;
    result->height = frames.height;
    result->width = frames.width;
}

int main(int argc, char **argv) {
    if (argc < 2) {
        fprintf(stderr,
                "usage: %s VIDEO_VAE_SOURCE_DIR [LATENT_T] [LATENT_HW]\n",
                argv[0]);
        return 2;
    }
    int latent_t = argc > 2 ? atoi(argv[2]) : 12;
    int latent_hw = argc > 3 ? atoi(argv[3]) : 16;
    int repeat = argc > 4 ? atoi(argv[4]) : 1;
    if (repeat < 1) repeat = 1;
    if (latent_t < 7 || (latent_t - 2) % 5 || latent_hw < 1) {
        fprintf(stderr,
                "latent_time must be 7 or 5k+2 and the tile edge >= 1\n");
        return 2;
    }
    size_t latent_count = (size_t)CHANNELS * (size_t)latent_t *
                          (size_t)latent_hw * (size_t)latent_hw;
    float *latent = malloc(latent_count * sizeof(*latent));
    if (!latent) {
        fprintf(stderr, "FAIL: out of memory for the latent\n");
        return 1;
    }
    for (size_t index = 0; index < latent_count; index++)
        latent[index] = rng_next();
    printf("video VAE A/B: latent %dx%dx%d, %u rows per tile, streaming=1\n",
           latent_t, latent_hw, latent_hw,
           (unsigned)(7 * latent_hw * latent_hw + 5));
    fflush(stdout);

    decoded metal, ane = {0};
    unsetenv("H3_ANE_VAE");
    decode_once(argv[1], latent, latent_t, latent_hw, "metal", &metal);
    printf("  metal fp32: %d frames %dx%d in %.2f s\n", metal.frames,
           metal.height, metal.width, metal.seconds);
    fflush(stdout);

    setenv("H3_ANE_VAE", "1", 1);
    for (int pass = 1; pass <= repeat; pass++) {
        free(ane.rgb);
        decode_once(argv[1], latent, latent_t, latent_hw, "ane", &ane);
        printf("  neural engine pass %d: %d frames %dx%d in %.2f s "
               "(%.2fx of metal)\n", pass, ane.frames, ane.height, ane.width,
               ane.seconds, metal.seconds / ane.seconds);
        fflush(stdout);
    }

    if (ane.frames != metal.frames || ane.height != metal.height ||
        ane.width != metal.width || ane.count != metal.count) {
        fprintf(stderr, "FAIL: the two paths returned different shapes\n");
        return 1;
    }
    double dot = 0.0, square_a = 0.0, square_b = 0.0, largest = 0.0;
    size_t nonfinite = 0;
    for (size_t index = 0; index < ane.count; index++) {
        double left = metal.rgb[index], right = ane.rgb[index];
        if (!isfinite(left) || !isfinite(right)) nonfinite++;
        dot += left * right;
        square_a += left * left;
        square_b += right * right;
        if (fabs(left - right) > largest) largest = fabs(left - right);
    }
    double cosine = dot / sqrt(square_a * square_b + 1e-30);
    double relative_l2 = sqrt((square_a - 2.0 * dot + square_b) /
                              ((square_a > square_b ? square_a : square_b) +
                               1e-30));
    printf("  cosine %.6f  rel_l2 %.3e  max_abs %.3e  nonfinite=%zu\n", cosine,
           relative_l2, largest, nonfinite);
    free(metal.rgb);
    free(ane.rgb);
    free(latent);
    int ok = !nonfinite && cosine >= 0.999 && relative_l2 <= 0.02;
    printf("%s\n", ok ? "PASS" : "FAIL");
    return ok ? 0 : 1;
}
