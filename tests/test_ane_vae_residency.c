/* Video-VAE residency probe (Phase D2a): how many Neural Engine graphs the
 * decoder can keep wired at once, and what rotation costs when it cannot.
 *
 * usage: h3_ane_vae_residency_test VIDEO_VAE_DIR [MODE] [ROWS] [BLOCKS] [WINDOW]
 *   MODE    capacity (default) | reload | pass | window
 *   ROWS    activation rows per graph (D2 buckets these to multiples of 256)
 *   BLOCKS  decoder blocks to take, 1..36
 *   WINDOW  graphs wired at once in window mode (0 = all of them)
 *
 * The decoder is 36 identical-shape blocks x 4 projections = 144 graphs, so the
 * wiring choice depends entirely on two numbers this probe produces: the resident
 * cost per graph, and the unload/reload milliseconds. A graph carries the baked
 * fp16 weight AND a set of activation planes, and the planes do not go away on
 * unload, which is why the two are reported separately per line and summed
 * separately at the end: every block repeats the same four plane shapes, so if
 * planes dominate the bill the fix is sharing one plane set across blocks rather
 * than rotating graphs.
 *
 * Nothing here runs the fp32 Metal path -- D1 already measured the per-block
 * ratio. This is only about residency, so it never touches h3_video_vae.c.
 *
 * Guards, because this runs on a 16 GiB machine: it stops at the first failure
 * and at H3_ANE_RESIDENCY_LIMIT_MIB of task footprint (default 10240).
 *
 * Env: H3_ANE_RESIDENCY_SHARE=1 builds the graphs on pooled (shared) planes,
 * which is what a resident decoder needs; H3_ANE_RESIDENCY_REPEATS and
 * H3_ANE_ROW_BUCKET work as elsewhere. */

#include "h3_ane_linear.h"
#include "h3_gpu.h"
#include "h3_safetensors.h"
#include "h3_weights.h"

#include <mach/mach.h>
#include <mach/mach_host.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

enum { PROJECTIONS = 4, BLOCKS_TOTAL = 36, DEFAULT_ROWS = 2048,
       DEFAULT_BLOCKS = 4, HIDDEN = 2048, INNER = 2048, FFN = 8192 };

static const char *const WEIGHT_SUFFIX[PROJECTIONS] = {
    "attn.to_qkv.weight", "attn.to_out.weight", "ff.w1.weight", "ff.w2.weight"
};
static const char *const SHORT_NAME[PROJECTIONS] = {"qkv", "out", "w1", "w2"};
static const uint32_t OUTPUT_DIM[PROJECTIONS] = {INNER * 3, HIDDEN, FFN * 2, HIDDEN};
static const uint32_t INPUT_DIM[PROJECTIONS] = {HIDDEN, INNER, HIDDEN, FFN};

static double now_seconds(void) {
    struct timespec now;
    clock_gettime(CLOCK_MONOTONIC, &now);
    return (double)now.tv_sec + (double)now.tv_nsec * 1e-9;
}

static uint64_t mib(uint64_t bytes) { return bytes / (1024ULL * 1024ULL); }

/* phys_footprint is what the machine actually has to hold for us; `internal`
 * strips the file-backed pages the compiled artifacts map in. */
static uint64_t task_footprint(void) {
    task_vm_info_data_t info;
    mach_msg_type_number_t count = TASK_VM_INFO_COUNT;
    if (task_info(mach_task_self(), TASK_VM_INFO, (task_info_t)&info, &count)
        != KERN_SUCCESS)
        return 0;
    return info.phys_footprint;
}

static uint64_t task_internal(void) {
    task_vm_info_data_t info;
    mach_msg_type_number_t count = TASK_VM_INFO_COUNT;
    if (task_info(mach_task_self(), TASK_VM_INFO, (task_info_t)&info, &count)
        != KERN_SUCCESS)
        return 0;
    return info.internal;
}

static uint64_t available_memory(void) {
    vm_statistics64_data_t vm;
    mach_msg_type_number_t count = HOST_VM_INFO64_COUNT;
    if (host_statistics64(mach_host_self(), HOST_VM_INFO64, (host_info64_t)&vm,
                          &count) != KERN_SUCCESS)
        return 0;
    uint64_t pages = (uint64_t)vm.free_count + vm.inactive_count +
        vm.speculative_count;
    return pages * (uint64_t)vm_page_size;
}

static double environment_seconds(const char *name, double fallback) {
    const char *text = getenv(name);
    return text && *text ? atof(text) : fallback;
}

/* The decoder ships its weights as F32; the graph wants row-major fp16. */
static int read_weight_f16(const h3_weight_store *store, uint32_t block,
                           uint32_t projection, uint16_t **weights,
                           char *error, size_t error_size) {
    char name[192];
    if (snprintf(name, sizeof(name), "decoder.transformer_blocks.%u.%s", block,
                 WEIGHT_SUFFIX[projection]) >= (int)sizeof(name)) {
        snprintf(error, error_size, "weight name overflow for block %u", block);
        return 0;
    }
    return h3_weight_load_f16_raw(store, name, OUTPUT_DIM[projection],
                                  INPUT_DIM[projection], weights, error,
                                  error_size);
}

typedef struct {
    h3_ane_projection *projection;
    uint64_t weight_bytes, plane_bytes;
    double create_seconds;
    double compile_seconds;
    bool wired;         /* loaded on the Neural Engine right now */
    uint64_t last_used; /* for the window's least-recently-used eviction */
} resident_graph;

/* Creates graphs one at a time and keeps every one of them alive, printing the
 * per-graph cost against the process footprint. `park` unloads each graph as
 * soon as it exists, which is what the wiring would do to hold more handles
 * than the Neural Engine lets it keep wired. Returns the count that made it. */
static uint32_t build_resident_set(const h3_weight_store *store, h3_gpu *gpu,
                                   uint32_t rows, uint32_t wanted, bool park,
                                   resident_graph *graphs,
                                   char *error, size_t error_size) {
    const uint64_t limit =
        (uint64_t)environment_seconds("H3_ANE_RESIDENCY_LIMIT_MIB", 10240.0) *
        1024ULL * 1024ULL;
    uint32_t count = 0;
    for (uint32_t block = 0; block < BLOCKS_TOTAL && count < wanted; block++) {
        for (uint32_t p = 0; p < PROJECTIONS && count < wanted; p++) {
            uint16_t *weights = NULL;
            if (!read_weight_f16(store, block, p, &weights, error, error_size)) {
                fprintf(stderr, "%s\n", error);
                return count;
            }
            char name[32];
            snprintf(name, sizeof(name), "b%u_%s", block, SHORT_NAME[p]);
            double started = now_seconds();
            h3_ane_projection *projection = h3_ane_projection_create_f16(
                gpu, name, weights, INPUT_DIM[p], OUTPUT_DIM[p], rows,
                h3_ane_linear_default_chunk(INPUT_DIM[p]), error, error_size);
            free(weights);
            if (!projection) {
                fprintf(stderr, "STOP %s after %u graphs: %s "
                        "(footprint=%llu MiB, available=%llu MiB)\n", name,
                        (unsigned)count, error,
                        (unsigned long long)mib(task_footprint()),
                        (unsigned long long)mib(available_memory()));
                fflush(stderr);
                return count;
            }
            uint64_t footprint = task_footprint();
            if (footprint > limit) {
                h3_ane_projection_free(projection);
                fprintf(stderr, "STOP %s: footprint guard %llu MiB after %u "
                        "graphs\n", name, (unsigned long long)mib(footprint),
                        count);
                return count;
            }
            graphs[count].projection = projection;
            graphs[count].weight_bytes = h3_ane_projection_weight_bytes(projection);
            graphs[count].plane_bytes = h3_ane_projection_plane_bytes(projection);
            graphs[count].create_seconds = now_seconds() - started;
            graphs[count].compile_seconds =
                h3_ane_projection_compile_seconds(projection);
            graphs[count].wired = true;
            if (park && !h3_ane_projection_unload(projection, error, error_size)) {
                fprintf(stderr, "STOP parking %s: %s\n", name, error);
                return count;
            }
            if (park) graphs[count].wired = false;
            printf("  b%-2u %-4s weight=%5llu MiB planes=%5llu MiB "
                   "create=%6.0f ms compile=%5.2f s %s%s | footprint=%6llu MiB "
                   "internal=%6llu MiB available=%5llu MiB\n",
                   block, SHORT_NAME[p],
                   (unsigned long long)mib(graphs[count].weight_bytes),
                   (unsigned long long)mib(graphs[count].plane_bytes),
                   graphs[count].create_seconds * 1e3,
                   graphs[count].compile_seconds,
                   h3_ane_projection_cache_hit(projection) ? "cache" : "compile",
                   park ? "+park" : "",
                   (unsigned long long)mib(footprint),
                   (unsigned long long)mib(task_internal()),
                   (unsigned long long)mib(available_memory()));
            fflush(stdout);
            count++;
        }
    }
    return count;
}

static void summarize(const resident_graph *graphs, uint32_t count,
                      uint32_t rows) {
    uint64_t weights = 0, planes = 0, pooled = 0;
    for (uint32_t i = 0; i < count; i++) {
        weights += graphs[i].weight_bytes;
        planes += graphs[i].plane_bytes;
    }
    /* One block's four plane sets is all a shared-pool wiring would ever need,
     * because every decoder block has the same shapes. */
    for (uint32_t p = 0; p < PROJECTIONS; p++) {
        uint32_t plane_rows = (rows + 15) / 16 * 16;
        pooled += (uint64_t)INPUT_DIM[p] * plane_rows * sizeof(float) +
            (uint64_t)OUTPUT_DIM[p] * plane_rows * sizeof(float);
    }
    printf("resident=%u graphs: weights=%llu MiB planes=%llu MiB "
           "(one pooled plane set=%llu MiB, pool now=%llu MiB) "
           "footprint=%llu MiB\n",
           (unsigned)count, (unsigned long long)mib(weights),
           (unsigned long long)mib(planes), (unsigned long long)mib(pooled),
           (unsigned long long)mib(h3_ane_planes_bytes()),
           (unsigned long long)mib(task_footprint()));
    fflush(stdout);
}

/* One input wide enough for the widest projection (w2 reads FFN columns) and
 * one output wide enough for w1's; every graph strides its own dims. */
static int prepare_pass_io(h3_gpu *gpu, uint32_t rows, h3_gpu_tensor **input,
                           h3_gpu_tensor **output, char *error,
                           size_t error_size) {
    float *scratch = malloc(sizeof(float) * (size_t)rows * FFN);
    if (!scratch) { snprintf(error, error_size, "oom for pass scratch"); return 0; }
    for (size_t i = 0; i < (size_t)rows * FFN; i++)
        scratch[i] = (float)((i % 977) - 488) / 512.0f;
    *input = h3_gpu_tensor_from_f32(gpu, scratch, (size_t)rows * FFN);
    free(scratch);
    *output = h3_gpu_tensor_new_f32(gpu, (size_t)rows * OUTPUT_DIM[2]);
    if (!*input || !*output) {
        snprintf(error, error_size, "oom for pass tensors: %s", h3_gpu_error(gpu));
        return 0;
    }
    return 1;
}

/* One full pass with everything wired, then one pass per graph parked and
 * rewired every pass. The second number bounds any window schedule: it is the
 * worst case, because each graph pays one unload plus one reload per pass. */
static int run_passes(h3_gpu *gpu, const resident_graph *graphs, uint32_t count,
                      uint32_t rows, char *error, size_t error_size) {
    h3_gpu_tensor *input = NULL, *output = NULL;
    if (!prepare_pass_io(gpu, rows, &input, &output, error, error_size))
        return 0;
    const int repeats = (int)environment_seconds("H3_ANE_RESIDENCY_REPEATS", 3.0);

    double best_warm = 1e30;
    for (int r = 0; r < repeats; r++) {
        double started = now_seconds();
        for (uint32_t i = 0; i < count; i++) {
            if (!h3_gpu_begin(gpu) ||
                !h3_ane_projection_apply_f32(graphs[i].projection, gpu, output,
                                             input, NULL, error, error_size) ||
                !h3_gpu_submit(gpu)) {
                snprintf(error, error_size, "pass graph %u failed: %s", i, error);
                return 0;
            }
        }
        double elapsed = now_seconds() - started;
        if (elapsed < best_warm) best_warm = elapsed;
        printf("  pass %d all-resident: %.1f ms (%.2f ms/graph)\n", r,
               elapsed * 1e3, elapsed * 1e3 / count);
        fflush(stdout);
    }

    double best_rotate = 1e30, unload_total = 0.0, reload_total = 0.0;
    /* Park everything first so every rotating pass, including the first, pays
     * exactly one reload and one unload per graph. */
    for (uint32_t i = 0; i < count; i++)
        if (!h3_ane_projection_unload(graphs[i].projection, error, error_size)) {
            snprintf(error, error_size, "parking graph %u failed: %s", i, error);
            return 0;
        }
    for (int r = 0; r < repeats; r++) {
        double started = now_seconds();
        for (uint32_t i = 0; i < count; i++) {
            double mark = now_seconds();
            if (!h3_ane_projection_reload(graphs[i].projection, error,
                                          error_size)) {
                snprintf(error, error_size, "reload graph %u failed: %s", i, error);
                return 0;
            }
            reload_total += now_seconds() - mark;
            if (!h3_gpu_begin(gpu) ||
                !h3_ane_projection_apply_f32(graphs[i].projection, gpu, output,
                                             input, NULL, error, error_size) ||
                !h3_gpu_submit(gpu)) {
                snprintf(error, error_size, "rotating pass graph %u failed: %s",
                         i, error);
                return 0;
            }
            mark = now_seconds();
            if (!h3_ane_projection_unload(graphs[i].projection, error,
                                          error_size)) {
                snprintf(error, error_size, "unload graph %u failed: %s", i, error);
                return 0;
            }
            unload_total += now_seconds() - mark;
        }
        double elapsed = now_seconds() - started;
        if (elapsed < best_rotate) best_rotate = elapsed;
        printf("  pass %d rotate-every-graph: %.1f ms\n", r, elapsed * 1e3);
        fflush(stdout);
    }
    printf("pass=%u graphs: all-resident=%.1f ms | rotating=%.1f ms | "
           "reload=%.2f ms/graph unload=%.2f ms/graph\n",
           (unsigned)count, best_warm * 1e3, best_rotate * 1e3,
           reload_total / repeats / count * 1e3,
           unload_total / repeats / count * 1e3);
    h3_gpu_tensor_free(input);
    h3_gpu_tensor_free(output);
    return 1;
}

/* The shape the wiring would actually live with: every handle stays alive, and
 * at most `window` of them stay wired, least-recently-used evicting the rest.
 * Reports each pass split into rewiring and everything else, because the split
 * is what decides whether a window costs more than the Neural Engine saves. */
static int run_window(h3_gpu *gpu, resident_graph *graphs, uint32_t count,
                      uint32_t rows, uint32_t window, char *error,
                      size_t error_size) {
    h3_gpu_tensor *input = NULL, *output = NULL;
    if (!prepare_pass_io(gpu, rows, &input, &output, error, error_size))
        return 0;
    const int repeats = (int)environment_seconds("H3_ANE_RESIDENCY_REPEATS", 3.0);
    uint64_t stamp = 0, wired = 0;
    double best_total = 1e30, best_reload = 1e30;
    for (int r = 0; r < repeats; r++) {
        double started = now_seconds(), reload = 0.0;
        uint32_t evictions = 0;
        for (uint32_t i = 0; i < count; i++) {
            if (!graphs[i].wired) {
                double mark = now_seconds();
                if (!h3_ane_projection_reload(graphs[i].projection, error,
                                              error_size)) {
                    snprintf(error, error_size, "reload graph %u failed: %s",
                             i, error);
                    return 0;
                }
                reload += now_seconds() - mark;
                graphs[i].wired = true;
                wired++;
            }
            if (!h3_gpu_begin(gpu) ||
                !h3_ane_projection_apply_f32(graphs[i].projection, gpu, output,
                                             input, NULL, error, error_size) ||
                !h3_gpu_submit(gpu)) {
                snprintf(error, error_size, "window pass graph %u failed: %s",
                         i, error);
                return 0;
            }
            graphs[i].last_used = ++stamp;
            while (wired > window) {
                uint32_t victim = count;
                for (uint32_t j = 0; j < count; j++)
                    if (graphs[j].wired && j != i &&
                        (victim == count ||
                         graphs[j].last_used < graphs[victim].last_used))
                        victim = j;
                if (victim == count) break;
                if (!h3_ane_projection_unload(graphs[victim].projection, error,
                                              error_size)) {
                    snprintf(error, error_size, "evict graph %u failed: %s",
                             victim, error);
                    return 0;
                }
                graphs[victim].wired = false;
                wired--;
                evictions++;
            }
        }
        double total = now_seconds() - started;
        if (total < best_total) best_total = total;
        if (reload < best_reload) best_reload = reload;
        printf("  pass %d window=%u: total=%.1f ms rewiring=%.1f ms "
               "(%.2f ms/reload) other=%.1f ms evictions=%u\n",
               r, (unsigned)window, total * 1e3, reload * 1e3,
               reload * 1e3 / (evictions ? evictions : 1),
               (total - reload) * 1e3, (unsigned)evictions);
        fflush(stdout);
    }
    printf("window=%u over %u graphs: best pass=%.1f ms, of which rewiring="
           "%.1f ms\n", (unsigned)window, (unsigned)count, best_total * 1e3,
           best_reload * 1e3);
    fflush(stdout);
    h3_gpu_tensor_free(input);
    h3_gpu_tensor_free(output);
    return 1;
}

/* Park and rewire a known set without evaluating, to separate the driver cost
 * from the pass timing. */
static int run_rotation_cost(const resident_graph *graphs, uint32_t count,
                             char *error, size_t error_size) {
    const int repeats = (int)environment_seconds("H3_ANE_RESIDENCY_REPEATS", 5.0);
    double best_unload = 1e30, best_reload = 1e30;
    uint64_t weights = 0;
    for (uint32_t i = 0; i < count; i++) weights += graphs[i].weight_bytes;
    for (int r = 0; r < repeats; r++) {
        double unload = 0.0, reload = 0.0;
        for (uint32_t i = 0; i < count; i++) {
            double started = now_seconds();
            if (!h3_ane_projection_unload(graphs[i].projection, error, error_size)) {
                snprintf(error, error_size, "unload %u failed: %s", i, error);
                return 0;
            }
            unload += now_seconds() - started;
            started = now_seconds();
            if (!h3_ane_projection_reload(graphs[i].projection, error, error_size)) {
                snprintf(error, error_size, "reload %u failed: %s", i, error);
                return 0;
            }
            reload += now_seconds() - started;
        }
        if (unload < best_unload) best_unload = unload;
        if (reload < best_reload) best_reload = reload;
        printf("  round %d: unload=%.1f ms reload=%.1f ms over %u graphs "
               "(%.2f / %.2f ms each)\n", r, unload * 1e3, reload * 1e3,
               (unsigned)count, unload / count * 1e3, reload / count * 1e3);
        fflush(stdout);
    }
    printf("rotation for %u graphs / %llu MiB of weights: unload best=%.2f ms, "
           "reload best=%.2f ms per graph\n", (unsigned)count,
           (unsigned long long)mib(weights), best_unload / count * 1e3,
           best_reload / count * 1e3);
    fflush(stdout);
    return 1;
}

int main(int argc, char **argv) {
    if (argc < 2) {
        fprintf(stderr, "usage: %s VIDEO_VAE_DIR [MODE] [ROWS] [BLOCKS] "
                "[WINDOW]\n", argv[0]);
        return 2;
    }
    const char *mode = argc > 2 ? argv[2] : "capacity";
    uint32_t rows = argc > 3 ? (uint32_t)atoi(argv[3]) : DEFAULT_ROWS;
    uint32_t blocks = argc > 4 ? (uint32_t)atoi(argv[4]) : DEFAULT_BLOCKS;
    uint32_t window = argc > 5 ? (uint32_t)atoi(argv[5]) : 8;
    if (blocks > BLOCKS_TOTAL) blocks = BLOCKS_TOTAL;
    /* D2's rounding policy, applied here so a run at 1797 and one at 2048
     * prove they really share the same compiled artifact set. */
    const uint32_t requested_rows = rows;
    rows = h3_ane_rows_bucket(rows);
    const bool park = strcmp(mode, "window") == 0;
    /* D2a showed planes, not weights, are what a resident decoder runs out of,
     * so the same probe has to be able to measure the pooled form too. */
    const int share_planes = getenv("H3_ANE_RESIDENCY_SHARE") ?
        atoi(getenv("H3_ANE_RESIDENCY_SHARE")) : 0;
    h3_ane_planes_share(share_planes);
    if (!h3_ane_linear_available()) {
        fprintf(stderr, "the Neural Engine bridge is unavailable\n");
        return 2;
    }
    char error[512] = {0};
    h3_weight_store *store = h3_weight_store_open(argv[1], error, sizeof(error));
    if (!store) { fprintf(stderr, "%s\n", error); return 2; }
    h3_gpu *gpu = h3_gpu_create("h3_shaders.metal", error, sizeof(error));
    if (!gpu) {
        fprintf(stderr, "cannot create the Metal context: %s\n", error);
        h3_weight_store_free(store);
        return 2;
    }
    uint32_t wanted = blocks * PROJECTIONS;
    if (park && !window) window = wanted;
    resident_graph *graphs = calloc(wanted, sizeof(*graphs));
    if (!graphs) { fprintf(stderr, "oom for the graph table\n"); return 2; }
    printf("video VAE residency: mode=%s rows=%u (bucket %u) "
           "blocks=%u (%u graphs) window=%u, footprint at start=%llu MiB, "
           "available=%llu MiB\n",
           mode, requested_rows, rows, blocks,
           (unsigned)wanted, (unsigned)window,
           (unsigned long long)mib(task_footprint()),
           (unsigned long long)mib(available_memory()));
    fflush(stdout);

    uint32_t count = build_resident_set(store, gpu, rows, wanted, park, graphs,
                                        error, sizeof(error));
    int ok = count > 0;
    if (!count) fprintf(stderr, "no graph became resident: %s\n", error);
    summarize(graphs, count, rows);
    if (ok && strcmp(mode, "reload") == 0)
        ok = run_rotation_cost(graphs, count, error, sizeof(error));
    else if (ok && strcmp(mode, "pass") == 0)
        ok = run_passes(gpu, graphs, count, rows, error, sizeof(error));
    else if (ok && park)
        ok = run_window(gpu, graphs, count, rows,
                        window ? window : count, error, sizeof(error));
    else if (ok && strcmp(mode, "capacity") != 0)
        fprintf(stderr, "unknown mode %s\n", mode);

    for (uint32_t i = 0; i < count; i++)
        h3_ane_projection_free(graphs[i].projection);
    printf("after freeing %u graphs: pool still holds=%llu MiB, "
           "footprint=%llu MiB\n", (unsigned)count,
           (unsigned long long)mib(h3_ane_planes_bytes()),
           (unsigned long long)mib(task_footprint()));
    h3_ane_planes_clear();
    free(graphs);
    h3_gpu_free(gpu);
    h3_weight_store_free(store);
    printf("%s\n", ok ? "probe finished" : "probe FAILED");
    return ok ? 0 : 1;
}
