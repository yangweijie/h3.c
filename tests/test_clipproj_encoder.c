/* Validates the in-engine ClipProj text encoder (h3_text_encode_clipproj_bf16):
 * runs Qwen3-VL-4B truncated to the tapped layer on Metal, lifts the hidden to
 * the 5120-dim H3 conditioning space with the ClipProj MLP, and checks the
 * result is a sane 5120-dim per-token embedding. */
#include "h3_safetensors.h"
#include "h3_text_encoder.h"
#include "h3_tokenizer.h"

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

static void fail(const char *message) {
    fprintf(stderr, "FAIL tests/test_clipproj_encoder.c: %s\n", message);
    exit(1);
}

static char *path_join(const char *root, const char *suffix) {
    size_t length = strlen(root) + strlen(suffix) + 2;
    char *result = malloc(length);
    if (result) snprintf(result, length, "%s/%s", root, suffix);
    return result;
}

static double seconds(void) {
    struct timespec value;
    clock_gettime(CLOCK_MONOTONIC, &value);
    return (double)value.tv_sec + (double)value.tv_nsec / 1e9;
}

static void progress(int completed, int total, void *opaque) {
    (void)opaque;
    if (completed == 1 || completed == total || completed % 5 == 0)
        fprintf(stderr, "ClipProj Qwen4B encode: %d/%d layers\n", completed, total);
}

static float bf16_to_f32(uint16_t value) {
    uint32_t bits = (uint32_t)value << 16;
    float result;
    memcpy(&result, &bits, sizeof(result));
    return result;
}

static int is_finite(float v) { return isfinite(v); }

int main(int argc, char **argv) {
    if (argc < 2 || argc > 5) {
        fprintf(stderr,
                "usage: %s MODEL_ROOT [QWEN4B_DIR] [PROJ_DIR] [PROMPT]\n",
                argv[0]);
        return 2;
    }
    const char *prompt = argc >= 5 ? argv[4] :
        "A red fox walking through snow";
    const char *qwen4b = argc >= 3 ? argv[2] :
        "/Volumes/data/.lmstudio/models/Qwen3-VL-4B-Instruct";
    const char *proj = argc >= 4 ? argv[3] :
        "/Volumes/data/.lmstudio/models/ClipProj-MiniMax-H3";

    char *tokenizer_path = path_join(argv[1], "FL2VA/tokenizer/tokenizer.json");
    if (!tokenizer_path) fail("path allocation failed");
    char error[512];
    h3_tokenizer *tokenizer = h3_tokenizer_load(tokenizer_path, error,
                                                 sizeof(error));
    if (!tokenizer) fail(error);
    uint32_t *ids = NULL;
    size_t token_count = 0;
    if (!h3_tokenizer_encode(tokenizer, prompt, 1, &ids, &token_count,
                             error, sizeof(error))) fail(error);
    printf("prompt: %s\ntokens: %zu", prompt, token_count);
    for (size_t i = 0; i < token_count; i++) printf(" %u", ids[i]);
    putchar('\n');

    h3_text_embedding embedding;
    double start = seconds();
    if (!h3_text_encode_clipproj_bf16(qwen4b, proj, "h3_shaders.metal", ids,
                                      token_count, progress, NULL, &embedding,
                                      error, sizeof(error))) fail(error);
    double elapsed = seconds() - start;

    if (embedding.tokens != token_count)
        fail("token count mismatch");
    if (embedding.width != H3_TEXT_HIDDEN_SIZE)
        fail("width is not 5120");
    size_t elements = embedding.tokens * embedding.width;

    double sum = 0, sumsq = 0, minv = 1e30, maxv = -1e30;
    size_t nanc = 0;
    for (size_t i = 0; i < elements; i++) {
        float v = bf16_to_f32(embedding.values[i]);
        if (!is_finite(v)) nanc++;
        sum += v; sumsq += v * v;
        if (v < minv) minv = v;
        if (v > maxv) maxv = v;
    }
    double mean = sum / (double)elements;
    double std = sqrt(sumsq / (double)elements - mean * mean);
    printf("clipproj embedding: %zux%zu BF16, %.3f s\n",
           embedding.tokens, embedding.width, elapsed);
    printf("  mean %.5f  std %.5f  min %.5f  max %.5f  nonfinite %zu\n",
           mean, std, minv, maxv, nanc);
    printf("GPU: %.3f GiB allocated, %llu linear dispatches, %llu submissions, "
           "%.3f GPU s\n",
           (double)embedding.gpu_stats.allocated_bytes / (1024.0*1024.0*1024.0),
           (unsigned long long)embedding.gpu_stats.mps_linear_dispatches,
           (unsigned long long)embedding.gpu_stats.submissions,
           embedding.gpu_stats.gpu_seconds);

    if (nanc != 0) fail("embedding contains NaN/Inf");
    if (!(std > 1e-4)) fail("embedding is degenerate (no variance)");
    if (embedding.gpu_stats.submissions < 20)
        fail("forward did not dispatch enough GPU work");

    /* Optional dump for golden fidelity comparison (make clipproj-golden):
       writes [seq u32][dim u32][seq*dim float32] so a Python script can
       compare B's embedding against A_local's harness .npz reference. */
    const char *dump_path = getenv("H3_CLIPPROJ_DUMP");
    if (dump_path) {
        FILE *df = fopen(dump_path, "wb");
        if (!df) fail("cannot open H3_CLIPPROJ_DUMP");
        uint32_t seq = (uint32_t)embedding.tokens;
        uint32_t dim = (uint32_t)embedding.width;
        if (fwrite(&seq, sizeof(seq), 1, df) != 1 ||
            fwrite(&dim, sizeof(dim), 1, df) != 1)
            fail("cannot write dump header");
        for (size_t i = 0; i < elements; i++) {
            float v = bf16_to_f32(embedding.values[i]);
            if (fwrite(&v, sizeof(v), 1, df) != 1)
                fail("cannot write dump data");
        }
        fclose(df);
    }

    h3_text_embedding_free(&embedding);
    h3_tokenizer_ids_free(ids);
    h3_tokenizer_free(tokenizer);
    free(tokenizer_path);
    puts("ok: ClipProj in-engine text encoder completed");
    return 0;
}
