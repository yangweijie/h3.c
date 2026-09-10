/* End-to-end AudioVAE decode against the released weights, without the MLX
 * fixture. This complements tests/test_real_audio_vae.c: that one pins the
 * waveform against the reference oracle, this one can still run on a machine
 * that only has the checkpoint and covers the parts that do not need an oracle
 * (stage arithmetic, output shape, finiteness, determinism across runs,
 * dispatch structure, shortest legal latent).
 *
 * Usage: ./h3_real_audio_vae_e2e_test [MODEL_ROOT]
 *        MODEL_ROOT defaults to MiniMax-H3 and must contain
 *        FL2VA/audio_vae/model.safetensors. Run from the repository root: the
 *        Metal source is loaded from the relative path h3_shaders.metal. */
#include "h3_audio_vae.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

enum { CHANNELS = 32, STEREO = 2, HOP_LENGTH = 800, MAIN_LENGTH = 37 };

static void die(const char *message) {
    fprintf(stderr, "FAIL tests/test_real_audio_vae_e2e.c: %s\n", message);
    exit(1);
}

static void progress(int completed, int total, void *opaque) {
    (void)opaque;
    fprintf(stderr, "native AudioVAE stage: %d/%d\n", completed, total);
}

/* Same deterministic LCG for every length, so the two MAIN_LENGTH decodes below
 * consume an identical latent and can be compared byte for byte. */
static float *make_latent(int length) {
    size_t count = (size_t)CHANNELS * STEREO * (size_t)length;
    float *latent = malloc(count * sizeof(*latent));
    if (!latent) die("out of memory building the synthetic latent");
    unsigned state = 12345u;
    for (size_t index = 0; index < count; index++) {
        state = state * 1664525u + 1013904223u;
        float unit = ((float)(state >> 8) / (float)(1u << 24)) * 2.0f - 1.0f;
        latent[index] = unit * 0.5f;
    }
    return latent;
}

static void decode(const char *weights, const float *latent, int length,
                   h3_audio_waveform *out, char *error, size_t error_size) {
    if (!h3_audio_vae_decode(weights, "h3_shaders.metal", latent, length,
                             progress, NULL, out, error, error_size)) {
        fprintf(stderr, "FAIL tests/test_real_audio_vae_e2e.c: %s\n", error);
        exit(1);
    }
}

/* Decode one length and require the documented output geometry plus only finite
 * samples. */
static size_t decode_checked(const char *weights, int length,
                             h3_audio_waveform *out, char *error,
                             size_t error_size) {
    float *latent = make_latent(length);
    decode(weights, latent, length, out, error, error_size);
    free(latent);
    if (out->channels != STEREO || out->samples != length * HOP_LENGTH ||
        out->sample_rate != 32000)
        die("native AudioVAE returned the wrong shape");
    size_t count = (size_t)STEREO * (size_t)length * HOP_LENGTH;
    for (size_t index = 0; index < count; index++)
        if (!isfinite(out->pcm[index]))
            die("native AudioVAE returned non-finite PCM");
    return count;
}

int main(int argc, char **argv) {
    const char *model_root = argc > 1 ? argv[1] : "MiniMax-H3";
    char weights[1024];
    if (snprintf(weights, sizeof(weights), "%s/FL2VA/audio_vae", model_root) >=
        (int)sizeof(weights))
        die("model root path is too long");
    char error[512];

    h3_audio_waveform first;
    size_t count = decode_checked(weights, MAIN_LENGTH, &first, error,
                                  sizeof(error));
    double peak = 0.0, square = 0.0;
    for (size_t index = 0; index < count; index++) {
        double magnitude = fabs((double)first.pcm[index]);
        if (magnitude > peak) peak = magnitude;
        square += (double)first.pcm[index] * (double)first.pcm[index];
    }
    printf("AudioVAE waveform: %d channels, %d samples @ %d Hz, peak %.6g, "
           "rms %.6g\n", first.channels, first.samples, first.sample_rate,
           peak, sqrt(square / (double)count));
    printf("AudioVAE: %.3f GiB allocated, %.3f GPU seconds, "
           "%llu MPS convolutions, %llu submissions\n",
           (double)first.gpu_stats.allocated_bytes /
               (1024.0 * 1024.0 * 1024.0),
           first.gpu_stats.gpu_seconds,
           (unsigned long long)first.gpu_stats.mps_conv_dispatches,
           (unsigned long long)first.gpu_stats.submissions);

    /* 16 baseline submissions = 1 input + 7 stage-normalization + 7 stage +
     * 1 output. run_stage additionally submits right after the upsample (and
     * drops the previous hidden state there) to lower peak memory, which adds
     * one submission per stage: STAGES (7) + 16 = 23. Changing that structure
     * knowingly means updating both numbers. */
    if (first.gpu_stats.mps_conv_dispatches != 136 ||
        first.gpu_stats.submissions != 23)
        die("native AudioVAE dispatch structure changed unexpectedly");

    h3_audio_waveform second;
    size_t second_count = decode_checked(weights, MAIN_LENGTH, &second, error,
                                         sizeof(error));
    if (second_count != count ||
        memcmp(first.pcm, second.pcm, count * sizeof(float)) != 0)
        die("two identical AudioVAE decodes differ");
    puts("AudioVAE determinism: two identical decodes are byte-identical");
    h3_audio_waveform_free(&first);
    h3_audio_waveform_free(&second);

    /* One frame is the shortest legal latent: input_length - 1 == 0 exercises
     * the stage length arithmetic at its lower bound. */
    for (int length = 1; length <= 2; length++) {
        h3_audio_waveform small;
        size_t small_count = decode_checked(weights, length, &small, error,
                                           sizeof(error));
        printf("AudioVAE boundary: latent %d -> %d samples\n", length,
               small.samples);
        if (small_count != (size_t)STEREO * (size_t)length * HOP_LENGTH)
            die("native AudioVAE boundary length mismatch");
        h3_audio_waveform_free(&small);
    }
    puts("ok: native Metal AudioVAE decodes end to end on the released weights");
    return 0;
}
