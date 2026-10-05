/* Gate for the AdaLN cache's LoRA and conditioning decisions, on a fixture small
 * enough to build in a second (tests/gen_adaln_cache_fixture.py).
 *
 * The cache's schedule key is deliberately full of -12345.0, so every case that
 * gets *past* the LoRA and conditioning gates stops at the prefix check. Reaching
 * "different sigma schedule" is therefore the proof that a gate accepted: it is
 * downstream of them and nothing before it runs the network or loads a tensor.
 *
 *   ./h3_adaln_cache_lora_gate_test tmp_adaln_cache_fixture
 */
#include "h3_dit_schedule.h"
#include "h3_lora.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int failures = 0;

static void report(int condition, const char *what, const char *detail) {
    fprintf(stderr, "%s %s\n", condition ? "  ok  " : "  FAIL", what);
    if (!condition) {
        fprintf(stderr, "       got: %s\n", detail);
        failures++;
    }
}

/* Run one precompute and judge the error string. `needle` NULL means the call
 * must succeed, which on this fixture it cannot: the schedule key never matches.
 */
static void expect(h3_weight_store *store, h3_gpu *gpu,
                  const h3_sigma_schedule *sigmas, int visual, int audio,
                  h3_lora **loras, int lora_count, const char *needle,
                  const char *what) {
    char error[512] = "";
    h3_dit_schedule *schedule = h3_dit_schedule_precompute(
        store, gpu, sigmas, visual, audio, loras, lora_count, NULL, NULL, error,
        sizeof(error));
    if (schedule) {
        h3_dit_schedule_free(schedule);
        report(0, what, "precompute unexpectedly succeeded");
        return;
    }
    if (!needle) {
        report(1, what, error);
        return;
    }
    report(strstr(error, needle) != NULL, what, error);
}

static h3_lora *open_adapter(const char *path) {
    char error[256] = "";
    h3_lora *lora = h3_lora_open(path, error, sizeof(error));
    if (!lora) {
        fprintf(stderr, "cannot open %s: %s\n", path, error);
        exit(1);
    }
    return lora;
}

int main(int argc, char **argv) {
    const char *fixture = argc > 1 ? argv[1] : "tmp_adaln_cache_fixture";
    char path[1024];
    char error[512];
    h3_weight_store *with_meta = NULL;
    h3_weight_store *without_meta = NULL;
    snprintf(path, sizeof(path), "%s/with_meta", fixture);
    with_meta = h3_weight_store_open(path, error, sizeof(error));
    if (!with_meta) {
        fprintf(stderr, "cannot open %s: %s\n", path, error);
        return 1;
    }
    snprintf(path, sizeof(path), "%s/without_meta", fixture);
    without_meta = h3_weight_store_open(path, error, sizeof(error));
    if (!without_meta) {
        fprintf(stderr, "cannot open %s: %s\n", path, error);
        return 1;
    }
    h3_gpu *gpu = h3_gpu_create("h3_shaders.metal", error, sizeof(error));
    if (!gpu) {
        fprintf(stderr, "cannot create GPU: %s\n", error);
        return 1;
    }
    h3_sigma_schedule sigmas;
    if (!h3_schedule_build(1, &sigmas)) {
        fprintf(stderr, "cannot build a 1-step schedule\n");
        return 1;
    }
    snprintf(path, sizeof(path), "%s/adapters/adapter_attn_only.safetensors",
             fixture);
    h3_lora *attn_only = open_adapter(path);
    snprintf(path, sizeof(path), "%s/adapters/adapter_adaln.safetensors",
             fixture);
    h3_lora *adaln = open_adapter(path);
    snprintf(path, sizeof(path), "%s/adapters/adapter_norm_out.safetensors",
             fixture);
    h3_lora *norm_out = open_adapter(path);
    h3_lora *one_attn[] = {attn_only};
    h3_lora *one_adaln[] = {adaln};
    h3_lora *one_norm[] = {norm_out};
    h3_lora *both[] = {attn_only, adaln};

    const char *past_the_gates = "different sigma schedule";

    /* The cache path is reached at all, with nothing merged into AdaLN. */
    expect(with_meta, gpu, &sigmas, 1, 1, NULL, 0, past_the_gates,
           "no adapters: the cache is adopted");
    /* P1: an adapter with no AdaLN factor must not be refused wholesale. */
    expect(with_meta, gpu, &sigmas, 1, 1, one_attn, 1, past_the_gates,
           "attention-only adapter is accepted by a width-declaring cache");
    /* P1: the two AdaLN sites each still conflict. */
    expect(with_meta, gpu, &sigmas, 1, 1, one_adaln, 1,
           "carries an AdaLN factor",
           "block adaln_proj.linear adapter is refused");
    expect(with_meta, gpu, &sigmas, 1, 1, one_norm, 1,
           "carries an AdaLN factor",
           "norm_out.linear adapter is refused");
    /* A mix still refuses, on the one adapter that conflicts. */
    expect(with_meta, gpu, &sigmas, 1, 1, both, 2, "carries an AdaLN factor",
           "a conflicting adapter among several still refuses");
    /* P1: without a declared width nothing can be proven, so all are refused --
     * the pre-existing rule, and what the shipped caches still get. */
    expect(without_meta, gpu, &sigmas, 1, 1, one_attn, 1,
           "has no adaln_cache_meta_s1 width key",
           "an undeclared width refuses even a harmless adapter");
    expect(without_meta, gpu, &sigmas, 1, 1, NULL, 0, past_the_gates,
           "an undeclared width still loads with no adapters");
    /* P3: the one conditioning mode no cache covers, named for its real reason
     * instead of advising a rebuild that could never satisfy it. */
    expect(with_meta, gpu, &sigmas, 0, 1, NULL, 0, "audio-only reference",
           "audio-only reference is refused with an actionable reason");

    h3_lora_close(norm_out);
    h3_lora_close(adaln);
    h3_lora_close(attn_only);
    h3_gpu_free(gpu);
    h3_weight_store_free(with_meta);
    h3_weight_store_free(without_meta);
    if (failures) {
        fprintf(stderr, "FAIL tests/test_adaln_cache_lora_gate.c: %d check(s)\n",
                failures);
        return 1;
    }
    puts("ok: the AdaLN cache gate accepts what it can and names what it cannot");
    return 0;
}
