/* Same-process A/B of the dense MPSGraph SDPA the DiT actually uses against the
 * two ways of getting block-sparse attention.
 *
 * The DiT per-op profile (F24) puts full attention at 20.5% of a block at
 * 864x480, so shrinking the visible window looked like -10% end to end. That
 * quote assumed a sparse variant costs the same per token pair as dense SDPA.
 * F26 measured the library's own windowed kernel at 70-78x the per-pair cost of
 * dense, so the remaining cheap route is to score fewer pairs with the dense
 * kernel itself:
 *
 *   dense     h3_gpu_sdpa_bf16 over the whole sequence (what ships)
 *   split     G independent dense SDPA calls over sequence/G rows each, which
 *             keeps 1/G of the pairs and needs no new kernel
 *   rect      G h3_gpu_sdpa_rect_bf16 calls over sequence/G query rows that each
 *             score KEY_ROWS key rows, so any block pattern is expressible once
 *             the selected rows are packed
 *   rectg     same as rect but each call first gathers its KEY_ROWS rows out of
 *             the full key/value tensors, stitched from SEGMENTS discontinuous
 *             pieces, which is the price of the whole path
 *   windowed  h3_gpu_flash_attn_bf16 with a frame radius (library kernel, kept
 *             here only as the F26 control; off by default)
 *
 * Each variant gets one timing window with one commit plus wait, and the pair
 * fraction is reported so a time ratio can be divided by the work ratio.
 *
 * Usage (from the repo root):
 *     ./h3_attention_bench [ROUNDS] [VARIANTS] [MODE] [MODE...]
 * VARIANTS is a subset of d/w/s/r/g (default "ds"). MODE is
 * SEQUENCE:TEXT:FRAMES:TOKENS_PER_FRAME:RADIUS[:GROUPS[:KEY_ROWS[:SEGMENTS]]];
 * radius 0 drops the windowed probe, groups 1 drops the split probe, key rows are
 * the rows each rectangular call scores (0 drops it), and segments is how many
 * discontinuous pieces the gather stitches together -- a real pattern needs its
 * frame window plus the text rows, so more than one. With no MODE argument the
 * two profiled resolutions run at groups 3 and 6.
 *
 * Every mode also prints its first-use wall and the process resident size: a
 * per-layer keep table means dozens of distinct graph shapes, so compile cost and
 * cache residency are part of the price rather than an assumption.
 */

#include "h3_gpu.h"

#include <mach/mach.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

/* #define rather than enum: these appear in double arithmetic and the compiler
 * warns on enum/float mixing. */
#define HEADS 56
#define HEAD_DIM 128
static const double head_pair_flops = 4.0 * HEADS * HEAD_DIM;

typedef enum {
    VARIANT_DENSE, VARIANT_WINDOWED, VARIANT_SPLIT, VARIANT_RECT,
    VARIANT_RECT_GATHER, VARIANTS
} variant;

static const char *variant_names[VARIANTS] = {"dense ", "window", "split ",
                                              "rect  ", "rectg "};

typedef struct {
    uint32_t sequence, text_rows, frames, tokens_per_frame, radius, groups;
    /* Rows each rectangular call scores; 0 drops the rectangular probe. */
    uint32_t key_rows;
    /* Discontinuous pieces the gather stitches those rows out of. */
    uint32_t segments;
} geometry;

typedef struct {
    h3_gpu_tensor *query, *key, *value, *output;
} tensors;

/* The dense and windowed probes share one full-size set; the sparse probes get
 * one smaller set per block so the G calls touch distinct buffers the way a
 * real per-block slice would. */
typedef struct {
    tensors full;
    tensors *slice;
} pool;

typedef struct {
    int wanted[VARIANTS];
    geometry geo;
} run_spec;

static double now_seconds(void) {
    struct timespec stamp;
    clock_gettime(CLOCK_MONOTONIC, &stamp);
    return (double)stamp.tv_sec + (double)stamp.tv_nsec * 1e-9;
}

/* Compiled MPSGraphs are the one thing the timing windows cannot see, so track
 * the process footprint around each shape's first use. */
static double resident_mb(void) {
    mach_task_basic_info_data_t info;
    mach_msg_type_number_t count = MACH_TASK_BASIC_INFO_COUNT;
    kern_return_t status = task_info(mach_task_self(), MACH_TASK_BASIC_INFO,
                                     (task_info_t)&info, &count);
    if (status != KERN_SUCCESS) return 0.0;
    return (double)info.resident_size / (1024.0 * 1024.0);
}

/* Piece `index` of a KEY_ROWS window split into `segments` pieces. */
static uint32_t piece_rows(uint32_t key_rows, uint32_t segments,
                           uint32_t index) {
    uint32_t base = key_rows / segments;
    uint32_t extra = key_rows % segments;
    return base + (index < extra ? 1u : 0u);
}

static int compare(const void *left, const void *right) {
    double a = *(const double *)left, b = *(const double *)right;
    return a < b ? -1 : (a > b ? 1 : 0);
}

static double sorted_median(double *values, int count) {
    qsort(values, (size_t)count, sizeof(*values), compare);
    return values[count / 2];
}

static void tensors_free(tensors *data) {
    h3_gpu_tensor_free(data->query);
    h3_gpu_tensor_free(data->key);
    h3_gpu_tensor_free(data->value);
    h3_gpu_tensor_free(data->output);
    memset(data, 0, sizeof(*data));
}

static int tensors_alloc_shape(h3_gpu *gpu, uint32_t query_rows,
                               uint32_t key_rows, tensors *data) {
    size_t query_count = (size_t)query_rows * HEADS * HEAD_DIM;
    size_t key_count = (size_t)key_rows * HEADS * HEAD_DIM;
    memset(data, 0, sizeof(*data));
    h3_gpu_tensor **slot[4] = {&data->query, &data->key, &data->value,
                               &data->output};
    size_t count[4] = {query_count, key_count, key_count, query_count};
    for (int index = 0; index < 4; index++) {
        h3_gpu_tensor *tensor = h3_gpu_tensor_new_bf16(gpu, count[index]);
        uint16_t *storage =
            tensor ? h3_gpu_tensor_bf16_storage(tensor, NULL) : NULL;
        if (!tensor || !storage) {
            fprintf(stderr, "FAIL: cannot allocate %zu bf16 elements\n",
                    count[index]);
            tensors_free(data);
            return 0;
        }
        memset(storage, 0, count[index] * sizeof(*storage));
        *slot[index] = tensor;
    }
    return 1;
}

static int tensors_alloc(h3_gpu *gpu, uint32_t rows, tensors *data) {
    return tensors_alloc_shape(gpu, rows, rows, data);
}

static void pool_free(pool *buffers, uint32_t groups) {
    tensors_free(&buffers->full);
    for (uint32_t group = 0; group < groups; group++)
        tensors_free(&buffers->slice[group]);
    free(buffers->slice);
    memset(buffers, 0, sizeof(*buffers));
}

/* The number of per-block buffer sets the sparse probes need. */
static uint32_t slice_count(const geometry *geo, const run_spec *spec) {
    if (!geo->groups || geo->groups == 1) return 0;
    if (spec->wanted[VARIANT_SPLIT] || spec->wanted[VARIANT_RECT] ||
        spec->wanted[VARIANT_RECT_GATHER])
        return geo->groups;
    return 0;
}

static int pool_alloc(h3_gpu *gpu, const run_spec *spec, pool *buffers) {
    const geometry *geo = &spec->geo;
    uint32_t slices = slice_count(geo, spec);
    memset(buffers, 0, sizeof(*buffers));
    if (spec->wanted[VARIANT_DENSE] || spec->wanted[VARIANT_WINDOWED] ||
        spec->wanted[VARIANT_RECT_GATHER]) {
        if (!tensors_alloc(gpu, geo->sequence, &buffers->full)) {
            pool_free(buffers, 0);
            return 0;
        }
    }
    if (!slices) return 1;
    buffers->slice = calloc(slices, sizeof(*buffers->slice));
    if (!buffers->slice) {
        fprintf(stderr, "FAIL: out of memory\n");
        pool_free(buffers, 0);
        return 0;
    }
    uint32_t rows = geo->sequence / geo->groups;
    uint32_t wanted_keys = geo->key_rows ? geo->key_rows : rows;
    /* One set of slices serves both sparse probes, so the key/value buffers are
     * as tall as the taller call needs; SDPA only reads the row count its graph
     * was built with. */
    uint32_t key_rows = wanted_keys > rows ? wanted_keys : rows;
    for (uint32_t slice = 0; slice < slices; slice++) {
        if (!tensors_alloc_shape(gpu, rows, key_rows,
                                 &buffers->slice[slice])) {
            pool_free(buffers, slice + 1);
            return 0;
        }
    }
    return 1;
}

static int run(h3_gpu *gpu, const pool *buffers, const geometry *geo,
               variant which) {
    float scale = 1.0f / sqrtf((float)HEAD_DIM);
    if (which == VARIANT_DENSE)
        return h3_gpu_sdpa_bf16(gpu, buffers->full.output, buffers->full.query,
                                buffers->full.key, buffers->full.value,
                                geo->sequence, HEADS, HEAD_DIM, scale);
    if (which == VARIANT_WINDOWED)
        return h3_gpu_flash_attn_bf16(gpu, buffers->full.output,
                                      buffers->full.query, buffers->full.key,
                                      buffers->full.value, geo->sequence,
                                      HEADS, HEAD_DIM, scale, geo->radius,
                                      geo->frames, geo->tokens_per_frame,
                                      geo->text_rows, geo->text_rows,
                                      geo->sequence - geo->text_rows -
                                          geo->frames * geo->tokens_per_frame,
                                      0, 0);
    uint32_t rows = geo->sequence / geo->groups;
    int gather = which == VARIANT_RECT_GATHER;
    if (gather || which == VARIANT_RECT) {
        uint32_t key_rows = geo->key_rows;
        uint32_t max_offset = geo->sequence - key_rows;
        for (uint32_t group = 0; group < geo->groups; group++) {
            const tensors *slice = &buffers->slice[group];
            if (gather) {
                /* A window that walks the sequence so no call reads past the end.
                 * Content does not matter here; only the bytes moved and the
                 * number of copy operations do. */
                uint32_t offset = (uint32_t)((double)group * max_offset /
                                             (double)(geo->groups - 1));
                uint32_t done = 0;
                for (uint32_t piece = 0; piece < geo->segments; piece++) {
                    uint32_t piece_count = piece_rows(key_rows, geo->segments,
                                                      piece);
                    size_t elements = (size_t)piece_count * HEADS * HEAD_DIM;
                    size_t source = (size_t)(offset + done) * HEADS * HEAD_DIM;
                    size_t destination = (size_t)done * HEADS * HEAD_DIM;
                    if (!h3_gpu_copy_bf16(gpu, slice->key, destination,
                                          buffers->full.key, source,
                                          elements) ||
                        !h3_gpu_copy_bf16(gpu, slice->value, destination,
                                          buffers->full.value, source, elements))
                        return 0;
                    done += piece_count;
                }
            }
            if (!h3_gpu_sdpa_rect_bf16(gpu, slice->output, slice->query,
                                       slice->key, slice->value, rows, key_rows,
                                       HEADS, HEAD_DIM, scale))
                return 0;
        }
        return 1;
    }
    for (uint32_t group = 0; group < geo->groups; group++) {
        const tensors *slice = &buffers->slice[group];
        if (!h3_gpu_sdpa_bf16(gpu, slice->output, slice->query, slice->key,
                              slice->value, rows, HEADS, HEAD_DIM, scale))
            return 0;
    }
    return 1;
}

/* Token pairs each variant actually scores, so a time ratio can be divided by
 * the work ratio to get the per-pair cost. */
static double pair_fraction(const geometry *geo, variant which) {
    if (which == VARIANT_DENSE) return 1.0;
    double dense = (double)geo->sequence * (double)geo->sequence;
    if (which == VARIANT_SPLIT) {
        uint32_t rows = geo->sequence / geo->groups;
        return ((double)geo->groups * rows * rows) / dense;
    }
    if (which == VARIANT_RECT || which == VARIANT_RECT_GATHER) {
        uint32_t rows = geo->sequence / geo->groups;
        return ((double)geo->groups * rows * (double)geo->key_rows) / dense;
    }
    uint32_t video_rows = geo->frames * geo->tokens_per_frame;
    double pairs = (double)geo->text_rows * geo->text_rows;
    for (uint32_t frame = 0; frame < geo->frames; frame++) {
        int low = (int)frame - (int)geo->radius;
        int high = (int)frame + (int)geo->radius;
        if (low < 0) low = 0;
        if (high > (int)geo->frames - 1) high = (int)geo->frames - 1;
        pairs += (double)geo->tokens_per_frame *
                 ((double)geo->text_rows + (high - low + 1) *
                                                geo->tokens_per_frame);
    }
    pairs += (double)(geo->sequence - geo->text_rows - video_rows) *
             geo->sequence;
    return pairs / dense;
}

/* One timed window: the chain opens outside the clock and the commit plus wait
 * inside it, so each variant pays exactly one drain and the chain is closed on
 * return. Encode and wait are split out because the split variant encodes G
 * graphs on the CPU inside the same window. */
static int measure(h3_gpu *gpu, const pool *buffers, const geometry *geo,
                   variant which, double *seconds, double *encode,
                   double *wait) {
    h3_gpu_stats before;
    if (!h3_gpu_get_stats(gpu, &before)) {
        fprintf(stderr, "FAIL: cannot read stats\n");
        return 0;
    }
    if (!h3_gpu_begin(gpu)) {
        fprintf(stderr, "FAIL: cannot begin the chain: %s\n",
                h3_gpu_error(gpu));
        return 0;
    }
    double started = now_seconds();
    int ok = run(gpu, buffers, geo, which);
    if (!h3_gpu_submit(gpu)) {
        fprintf(stderr, "FAIL: submit failed: %s\n", h3_gpu_error(gpu));
        return 0;
    }
    *seconds = now_seconds() - started;
    if (!ok) {
        fprintf(stderr, "FAIL: %s attention: %s\n", variant_names[which],
                h3_gpu_error(gpu));
        return 0;
    }
    h3_gpu_stats after;
    if (!h3_gpu_get_stats(gpu, &after)) {
        fprintf(stderr, "FAIL: cannot read stats\n");
        return 0;
    }
    *encode = (after.command_encode_seconds - before.command_encode_seconds) * 1e3;
    *wait = (after.command_wait_seconds - before.command_wait_seconds) * 1e3;
    return 1;
}

static int applicable(const run_spec *spec, variant which) {
    if (!spec->wanted[which]) return 0;
    if (which == VARIANT_WINDOWED) return spec->geo.radius != 0;
    if (which == VARIANT_SPLIT) return spec->geo.groups > 1;
    if (which == VARIANT_RECT || which == VARIANT_RECT_GATHER)
        return spec->geo.groups > 1 && spec->geo.key_rows != 0;
    return 1;
}

static int sweep_body(h3_gpu *gpu, const run_spec *spec, int rounds,
                      pool *buffers, const variant *order, int count);

/* One timing pass. `given` lets the caller keep a pool alive across modes that
 * share sequence and groups, so per-shape residency is not polluted by allocating
 * and freeing hundreds of megabytes per sweep. */
static int sweep(h3_gpu *gpu, const run_spec *spec, int rounds, pool *given) {
    const geometry *geo = &spec->geo;
    pool local;
    pool *buffers = given;
    int owned = 0;
    variant order[VARIANTS];
    int count = 0;
    for (int index = 0; index < VARIANTS; index++)
        if (applicable(spec, (variant)index)) order[count++] = (variant)index;
    if (count < 2) {
        fprintf(stderr, "FAIL: need a dense reference plus one sparse variant\n");
        return 0;
    }
    if (!buffers) {
        buffers = &local;
        owned = 1;
        if (!pool_alloc(gpu, spec, buffers)) return 0;
    }
    int result = sweep_body(gpu, spec, rounds, buffers, order, count);
    if (owned) pool_free(buffers, geo->groups);
    return result;
}

static int sweep_body(h3_gpu *gpu, const run_spec *spec, int rounds,
                      pool *buffers, const variant *order, int count) {
    const geometry *geo = &spec->geo;

    uint32_t video_rows = geo->frames * geo->tokens_per_frame;
    printf("sequence %u (text %u, %u frames x %u, tail %u): radius %u, %u calls"
           " x %u rows x %u keys x %u segments\n", geo->sequence,
           geo->text_rows, geo->frames, geo->tokens_per_frame,
           geo->sequence - geo->text_rows - video_rows, geo->radius,
           geo->groups, geo->sequence / geo->groups,
           geo->key_rows ? geo->key_rows : geo->sequence / geo->groups,
           geo->segments);
    /* Each shape compiles its own MPSGraph on first use; warm every variant
     * before the clock runs. The per-call wall alone cannot see that compile
     * (MPSGraph may schedule it outside the command buffer we wait on), so the
     * cumulative clock over the whole warm phase is the number to read: it moves
     * by the compile cost of each new shape. */
    double warm_started = now_seconds();
    for (int index = 0; index < count; index++) {
        double warm = 0.0, encode = 0.0, wait = 0.0;
        h3_gpu_stats before;
        if (!h3_gpu_get_stats(gpu, &before)) {
            fprintf(stderr, "FAIL: cannot read stats\n");
            return 0;
        }
        if (!measure(gpu, buffers, geo, order[index], &warm, &encode, &wait)) {
            return 0;
        }
        h3_gpu_stats after;
        if (!h3_gpu_get_stats(gpu, &after)) {
            fprintf(stderr, "FAIL: cannot read stats\n");
            return 0;
        }
        printf("  first use %s wall %8.3f ms (enc %5.1f wait %8.1f)"
               "  cum %9.3f ms  resident %.0f MB  engine tensors %.0f/%.0f MB\n",
               variant_names[order[index]], warm * 1e3, encode, wait,
               (now_seconds() - warm_started) * 1e3, resident_mb(),
               (double)after.live_bytes / (1024.0 * 1024.0),
               (double)after.peak_live_bytes / (1024.0 * 1024.0));
    }
    double *values[VARIANTS] = {0};
    for (int index = 0; index < count; index++) {
        values[order[index]] = malloc((size_t)rounds * sizeof(*values[index]));
        if (!values[order[index]]) {
            fprintf(stderr, "FAIL: out of memory\n");
            for (int free_index = 0; free_index < VARIANTS; free_index++)
                free(values[free_index]);
            return 0;
        }
    }
    int ok = 1;
    for (int round = 0; round < rounds && ok; round++) {
        /* Rotate the order so a monotone drift cannot favour one variant. */
        printf("  round %2d", round + 1);
        for (int slot = 0; slot < count; slot++) {
            variant which = order[(slot + round) % count];
            double seconds = 0.0, encode = 0.0, wait = 0.0;
            if (!measure(gpu, buffers, geo, which, &seconds, &encode, &wait)) {
                ok = 0;
                break;
            }
            values[which][round] = seconds * 1e3;
            printf("  %s %8.3f (enc %5.1f wait %7.1f)", variant_names[which],
                   seconds * 1e3, encode, wait);
        }
        printf("\n");
    }
    double dense_median = 0.0;
    double dense_tflops = 0.0;
    /* The dense reference is the same graph within one sequence length, so it
     * doubles as the probe for this machine window: per-pair ratios do not
     * survive a slowdown, because a small call loses more to contention than a
     * full-sequence one. Tracked per sequence, since that is the only thing that
     * makes two sweeps' dense figures comparable. */
    struct dense_probe {
        uint32_t sequence;
        double fastest;
    };
    enum { DENSE_PROBES = 8 };
    static struct dense_probe probes[DENSE_PROBES];
    static int probe_count;
    for (int index = 0; index < count && ok; index++) {
        variant which = order[index];
        double median = sorted_median(values[which], rounds);
        double fraction = pair_fraction(geo, which);
        double tflops = head_pair_flops * (double)geo->sequence *
                        (double)geo->sequence * fraction / median / 1e9;
        if (which == VARIANT_DENSE) {
            dense_median = median;
            dense_tflops = tflops;
            printf("  median %s %9.3f ms  %.2f TFLOPS  (reference)\n",
                   variant_names[which], median, tflops);
            struct dense_probe *probe = NULL;
            for (int seen = 0; seen < probe_count; seen++) {
                if (probes[seen].sequence == geo->sequence) {
                    probe = &probes[seen];
                    break;
                }
            }
            if (!probe && probe_count < DENSE_PROBES) {
                probe = &probes[probe_count++];
                probe->sequence = geo->sequence;
                probe->fastest = median;
            }
            if (median < probe->fastest) {
                probe->fastest = median;
            } else if (median > probe->fastest * 1.1) {
                printf("  NOTE: dense is %.2fx the fastest dense at sequence %u"
                       " (%.3f ms) -- this machine window is degraded, discard"
                       " the ratios above\n",
                       median / probe->fastest, geo->sequence, probe->fastest);
            }
            continue;
        }

        printf("  median %s %9.3f ms  %.2f TFLOPS  time x%.3f  pairs x%.3f"
               "  per-pair x%.1f  vs dense efficiency x%.3f\n",
               variant_names[which], median, tflops, median / dense_median,
               fraction, median / dense_median / fraction, tflops / dense_tflops);
    }
    for (int index = 0; index < VARIANTS; index++) free(values[index]);
    return ok;
}

static int parse_mode(char *text, run_spec *spec) {
    unsigned values[8] = {0, 0, 0, 0, 0, 1, 0, 1};
    int fields = sscanf(text, "%u:%u:%u:%u:%u:%u:%u:%u", &values[0], &values[1],
                        &values[2], &values[3], &values[4], &values[5],
                        &values[6], &values[7]);
    if (fields < 5) {
        fprintf(stderr, "FAIL: bad MODE '%s'\n", text);
        return 0;
    }
    spec->geo.sequence = values[0];
    spec->geo.text_rows = values[1];
    spec->geo.frames = values[2];
    spec->geo.tokens_per_frame = values[3];
    spec->geo.radius = values[4];
    spec->geo.groups = values[5];
    spec->geo.key_rows = values[6];
    spec->geo.segments = values[7] ? values[7] : 1u;
    uint32_t video_rows = spec->geo.frames * spec->geo.tokens_per_frame;
    /* Rows per call floor-divides, so a non-dividing group count simply scores a
     * few fewer pairs; pair_fraction uses the same rounding. */
    if (!spec->geo.sequence || !spec->geo.groups ||
        video_rows > spec->geo.sequence || !spec->geo.tokens_per_frame ||
        (spec->geo.groups > 1 &&
         spec->geo.sequence / spec->geo.groups < 1)) {
        fprintf(stderr, "FAIL: inconsistent MODE '%s'\n", text);
        return 0;
    }
    if (spec->geo.radius &&
        spec->geo.text_rows + video_rows > spec->geo.sequence) {
        fprintf(stderr, "FAIL: window geometry overruns sequence in '%s'\n",
                text);
        return 0;
    }
    /* The gather probe reads a whole window out of the full tensors. */
    if (spec->geo.key_rows > spec->geo.sequence) {
        fprintf(stderr, "FAIL: key rows %u exceed sequence %u in '%s'\n",
                spec->geo.key_rows, spec->geo.sequence, text);
        return 0;
    }
    if (spec->geo.key_rows && spec->geo.segments > spec->geo.key_rows) {
        fprintf(stderr, "FAIL: %u segments cannot cover %u key rows in '%s'\n",
                spec->geo.segments, spec->geo.key_rows, text);
        return 0;
    }
    return 1;
}

static int parse_variants(const char *text, run_spec *spec) {
    for (int index = 0; index < VARIANTS; index++) spec->wanted[index] = 0;
    for (const char *cursor = text; *cursor; cursor++) {
        int found = -1;
        if (*cursor == 'd') found = VARIANT_DENSE;
        else if (*cursor == 'w') found = VARIANT_WINDOWED;
        else if (*cursor == 's') found = VARIANT_SPLIT;
        else if (*cursor == 'r') found = VARIANT_RECT;
        else if (*cursor == 'g') found = VARIANT_RECT_GATHER;
        if (found < 0) {
            fprintf(stderr, "FAIL: bad VARIANTS '%c' (use d/w/s/r/g)\n",
                    *cursor);
            return 0;
        }
        spec->wanted[found] = 1;
    }
    if (!spec->wanted[VARIANT_DENSE]) {
        fprintf(stderr, "FAIL: variant d is the reference and always required\n");
        return 0;
    }
    return 1;
}

int main(int argc, char **argv) {
    int rounds = argc > 1 ? atoi(argv[1]) : 3;
    if (rounds < 1) rounds = 1;
    char error[512] = {0};
    h3_gpu *gpu = h3_gpu_create("h3_shaders.metal", error, sizeof(error));
    if (!gpu) {
        fprintf(stderr, "FAIL: %s\n", error);
        return 1;
    }
    printf("attention A/B: heads %d, head_dim %d, rounds %d\n", HEADS,
           HEAD_DIM, rounds);
    /* 864x480 and 576x320 at 2 seconds: 17 latent frames of 405 / 180 tokens
     * plus 189 text+audio rows, which is how both profiled sequences split. */
    static const geometry defaults[] = {
        {7074, 189, 17, 405, 0, 3, 0, 1},
        {7074, 189, 17, 405, 0, 6, 0, 1},
        {3249, 189, 17, 180, 0, 3, 0, 1},
        {3249, 189, 17, 180, 0, 6, 0, 1},
    };
    enum { DEFAULTS = (int)(sizeof(defaults) / sizeof(*defaults)) };
    run_spec base;
    memset(&base, 0, sizeof(base));
    if (argc > 2) {
        if (!parse_variants(argv[2], &base)) return 1;
    } else {
        base.wanted[VARIANT_DENSE] = 1;
        base.wanted[VARIANT_SPLIT] = 1;
    }
    int mode_count = argc > 3 ? argc - 3 : DEFAULTS;
    run_spec *specs = malloc((size_t)mode_count * sizeof(*specs));
    if (!specs) {
        fprintf(stderr, "FAIL: out of memory\n");
        h3_gpu_free(gpu);
        return 1;
    }
    for (int index = 0; index < mode_count; index++) {
        specs[index] = base;
        if (argc > 3) {
            if (!parse_mode(argv[3 + index], &specs[index])) {
                free(specs);
                h3_gpu_free(gpu);
                return 1;
            }
        } else {
            specs[index].geo = defaults[index];
        }
    }

    /* A keep sweep runs dozens of shapes that differ only in their key rows.
     * Reusing one pool across them is what makes the resident growth readable as
     * graph-cache cost: allocating and freeing hundreds of megabytes per mode
     * moves the same amount of memory and drowns the signal (F28 measured that
     * artefact as a ~4 GB ramp with a *single* repeated shape). */
    int shared = mode_count > 1;
    uint32_t max_keys = 0;
    for (int index = 0; index < mode_count; index++) {
        const geometry *geo = &specs[index].geo;
        uint32_t rows = geo->sequence / geo->groups;
        uint32_t keys = geo->key_rows ? geo->key_rows : rows;
        if (keys > max_keys) max_keys = keys;
        if (geo->sequence != specs[0].geo.sequence ||
            geo->groups != specs[0].geo.groups)
            shared = 0;
    }
    pool buffers;
    int hold = 0;
    memset(&buffers, 0, sizeof(buffers));
    if (shared) {
        run_spec pool_spec = specs[0];
        pool_spec.geo.key_rows = max_keys;
        hold = pool_alloc(gpu, &pool_spec, &buffers);
        if (!hold) {
            free(specs);
            h3_gpu_free(gpu);
            return 1;
        }
        printf("one pool reused by %d modes (up to %u key rows per call),"
               " resident %.0f MB\n", mode_count, max_keys, resident_mb());
    }
    int ok = 1;
    for (int index = 0; index < mode_count && ok; index++)
        ok = sweep(gpu, &specs[index], rounds, hold ? &buffers : NULL);
    if (hold) pool_free(&buffers, specs[0].geo.groups);
    free(specs);
    h3_gpu_free(gpu);
    printf("%s\n", ok ? "PASS" : "FAIL");
    return ok ? 0 : 1;
}
