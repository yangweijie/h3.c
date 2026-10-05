#include "h3_dit_schedule.h"

#include "h3_lora.h"

#include <errno.h>
#include <math.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

enum {
    TIME_INPUT = 256,
    TIME_HIDDEN = 5376,
    BLOCK_OUTPUT = H3_ADALN_CACHE_BLOCK_OUTPUT,
    FINAL_OUTPUT = H3_ADALN_CACHE_FINAL_OUTPUT
};

struct h3_dit_schedule {
    h3_gpu *gpu;
    int steps;
    uint32_t time_rows;
    uint32_t *video_rows;
    uint32_t *audio_rows;
    uint32_t *visual_condition_rows;
    uint32_t *audio_condition_rows;
    h3_gpu_tensor *blocks[H3_DIT_BLOCKS];
    h3_gpu_tensor *final;
};

static void fail(char *error, size_t error_size, const char *format, ...) {
    if (!error || !error_size) return;
    va_list arguments;
    va_start(arguments, format);
    vsnprintf(error, error_size, format, arguments);
    va_end(arguments);
}

static int gpu_op(h3_gpu *gpu, int ok, char *error, size_t error_size,
                  const char *operation) {
    if (ok) return 1;
    fail(error, error_size, "%s: %s", operation, h3_gpu_error(gpu));
    return 0;
}

static h3_gpu_tensor *weight_f32_1d(const h3_weight_store *store, h3_gpu *gpu,
                                    const char *name, uint64_t width,
                                    char *error, size_t error_size) {
    uint64_t shape[] = {width};
    return h3_weight_load_f32(store, gpu, name, 1, shape, error, error_size);
}

static h3_gpu_tensor *weight_f32_2d(const h3_weight_store *store, h3_gpu *gpu,
                                    const char *name, uint64_t rows,
                                    uint64_t columns, char *error,
                                    size_t error_size) {
    uint64_t shape[] = {rows, columns};
    return h3_weight_load_f32(store, gpu, name, 2, shape, error, error_size);
}

static float schedule_f16_to_f32(uint16_t value) {
    uint32_t sign = (uint32_t)(value & 0x8000u) << 16;
    uint32_t exponent = (value >> 10) & 0x1fu;
    uint32_t mantissa = value & 0x3ffu;
    uint32_t bits;
    if (exponent == 0x1f) {
        bits = sign | 0x7f800000u | (mantissa << 13);
    } else if (exponent == 0) {
        if (mantissa == 0) {
            bits = sign;
        } else {
            uint32_t e = 113;
            do { e--; mantissa <<= 1; } while (!(mantissa & 0x400u));
            mantissa &= 0x3ffu;
            bits = sign | (e << 23) | (mantissa << 13);
        }
    } else {
        bits = sign | ((exponent + 112u) << 23) | (mantissa << 13);
    }
    float result;
    memcpy(&result, &bits, sizeof(result));
    return result;
}

static uint16_t schedule_f32_to_bf16(float value) {
    uint32_t bits;
    memcpy(&bits, &value, sizeof(bits));
    uint16_t high = (uint16_t)(bits >> 16);
    if ((bits & 0xffffu) >= 0x8000u) high = (uint16_t)(high + 1);
    return high;
}

/* ConvRot checkpoints keep the tiny AdaLN projections in F16 (the patch heads
 * are F32); accept BF16 natively and convert F16/F32 to BF16 on the host. */
static h3_gpu_tensor *weight_bf16_any(const h3_weight_store *store, h3_gpu *gpu,
                                      const char *name, int ndim,
                                      const uint64_t *shape,
                                      char *error, size_t error_size) {
    const h3_st_header *header = NULL;
    const h3_st_tensor *tensor = h3_weight_find(store, name, &header);
    if (!tensor) {
        fail(error, error_size, "required weight is absent: %s", name);
        return NULL;
    }
    if (tensor->dtype == H3_DTYPE_BF16)
        return h3_weight_load_bf16(store, gpu, name, ndim, shape,
                                   error, error_size);
    if (tensor->dtype != H3_DTYPE_F16 && tensor->dtype != H3_DTYPE_F32) {
        fail(error, error_size, "weight %s has unsupported dtype %s",
             name, h3_dtype_name(tensor->dtype));
        return NULL;
    }
    if (tensor->ndim != ndim) {
        fail(error, error_size, "weight %s has rank %d, expected %d",
             name, tensor->ndim, ndim);
        return NULL;
    }
    uint64_t elements = 1;
    for (int dimension = 0; dimension < ndim; dimension++) {
        if (tensor->shape[dimension] != shape[dimension]) {
            fail(error, error_size, "weight %s shape mismatch at dimension %d",
                 name, dimension);
            return NULL;
        }
        if (shape[dimension] && elements > UINT64_MAX / shape[dimension]) {
            fail(error, error_size, "weight %s shape overflows", name);
            return NULL;
        }
        elements *= shape[dimension];
    }
    if (elements > SIZE_MAX / sizeof(float)) {
        fail(error, error_size, "weight %s is too large to convert", name);
        return NULL;
    }
    size_t raw_size = (size_t)elements *
        (tensor->dtype == H3_DTYPE_F16 ? sizeof(uint16_t) : sizeof(float));
    void *raw = malloc(raw_size);
    uint16_t *bf16 = malloc((size_t)elements * sizeof(uint16_t));
    if (!raw || !bf16) {
        free(raw);
        free(bf16);
        fail(error, error_size, "cannot allocate conversion buffer for %s",
             name);
        return NULL;
    }
    if (!h3_st_read_data(header, tensor, raw, raw_size, error, error_size)) {
        free(raw);
        free(bf16);
        return NULL;
    }
    if (tensor->dtype == H3_DTYPE_F16) {
        const uint16_t *values = raw;
        for (uint64_t index = 0; index < elements; index++)
            bf16[index] =
                schedule_f32_to_bf16(schedule_f16_to_f32(values[index]));
    } else {
        const float *values = raw;
        for (uint64_t index = 0; index < elements; index++)
            bf16[index] = schedule_f32_to_bf16(values[index]);
    }
    free(raw);
    h3_gpu_tensor *result = h3_gpu_tensor_from_bf16(gpu, bf16,
                                                    (size_t)elements);
    free(bf16);
    if (!result)
        fail(error, error_size, "cannot upload %s: %s", name,
             h3_gpu_error(gpu));
    return result;
}

static h3_gpu_tensor *weight_bf16_any_1d(const h3_weight_store *store,
                                         h3_gpu *gpu, const char *name,
                                         uint64_t width, char *error,
                                         size_t error_size) {
    uint64_t shape[] = {width};
    return weight_bf16_any(store, gpu, name, 1, shape, error, error_size);
}

static h3_gpu_tensor *weight_bf16_any_2d(const h3_weight_store *store,
                                         h3_gpu *gpu, const char *name,
                                         uint64_t rows, uint64_t columns,
                                         char *error, size_t error_size) {
    uint64_t shape[] = {rows, columns};
    return weight_bf16_any(store, gpu, name, 2, shape, error, error_size);
}

static void free_tensor(h3_gpu_tensor **tensor) {
    h3_gpu_tensor_free(*tensor);
    *tensor = NULL;
}

/* Row map plus the per-row timestep values. Reads no weight and touches no GPU,
 * so the AdaLN cache path can validate a cached schedule before committing to
 * it, and so a cache miss can still fall back to the full projection. */
static int prepare_times(h3_dit_schedule *schedule,
                         const h3_sigma_schedule *sigmas,
                         int visual_condition, int audio_condition,
                         float **times_out, char *error, size_t error_size) {
    schedule->steps = sigmas->steps;
    schedule->video_rows = calloc((size_t)sigmas->steps,
                                  sizeof(*schedule->video_rows));
    schedule->audio_rows = calloc((size_t)sigmas->steps,
                                  sizeof(*schedule->audio_rows));
    if (visual_condition)
        schedule->visual_condition_rows = calloc(
            (size_t)sigmas->steps, sizeof(*schedule->visual_condition_rows));
    if (audio_condition)
        schedule->audio_condition_rows = calloc(
            (size_t)sigmas->steps, sizeof(*schedule->audio_condition_rows));
    if (!schedule->video_rows || !schedule->audio_rows ||
        (visual_condition && !schedule->visual_condition_rows) ||
        (audio_condition && !schedule->audio_condition_rows)) {
        fail(error, error_size, "out of memory allocating timestep row maps");
        return 0;
    }
    uint32_t count = 0;
    for (int step = 0; step < sigmas->steps; step++) {
        float video = 1.0f - sigmas->video[step];
        float audio = 1.0f - sigmas->audio[step];
        if (video == audio) {
            schedule->video_rows[step] = count;
            schedule->audio_rows[step] = count++;
        } else if (video < audio) {
            schedule->video_rows[step] = count++;
            schedule->audio_rows[step] = count++;
        } else {
            schedule->audio_rows[step] = count++;
            schedule->video_rows[step] = count++;
        }
    }
    uint32_t visual_condition_row = UINT32_MAX;
    uint32_t audio_condition_row = UINT32_MAX;
    if (visual_condition) visual_condition_row = count++;
    if (audio_condition) audio_condition_row = count++;
    for (int step = 0; step < sigmas->steps; step++) {
        float video = 1.0f - sigmas->video[step];
        float audio = 1.0f - sigmas->audio[step];
        if (visual_condition)
            schedule->visual_condition_rows[step] = video >= 0.999f ?
                schedule->video_rows[step] : visual_condition_row;
        if (audio_condition)
            schedule->audio_condition_rows[step] = audio >= 0.999f ?
                schedule->audio_rows[step] : audio_condition_row;
    }
    schedule->time_rows = count;
    if (!count || count > UINT32_MAX / TIME_INPUT) {
        fail(error, error_size, "invalid number of timestep rows");
        return 0;
    }
    float *times = calloc(count, sizeof(*times));
    if (!times) {
        fail(error, error_size, "out of memory allocating timestep values");
        return 0;
    }
    for (int step = 0; step < sigmas->steps; step++) {
        times[schedule->video_rows[step]] = 1.0f - sigmas->video[step];
        times[schedule->audio_rows[step]] = 1.0f - sigmas->audio[step];
    }
    if (visual_condition) times[visual_condition_row] = 0.999f;
    if (audio_condition) times[audio_condition_row] = 1.0f;
    if (getenv("H3_DEBUG_CONVROT")) {
        fprintf(stderr, "convrot debug: count=%u times=[", count);
        for (uint32_t i = 0; i < count; i++) fprintf(stderr, " %.4f", times[i]);
        fprintf(stderr, " ] video_rows=");
        for (int s = 0; s < sigmas->steps; s++) fprintf(stderr, " %u", schedule->video_rows[s]);
        fprintf(stderr, "\n");
    }
    *times_out = times;
    return 1;
}

/* AdaLN input features for every row: either the raw adaln_t_table row (ConvRot
 * pruned checkpoints) or the sinusoidal embedding the time MLP consumes. */
static int prepare_features(const float *times, uint32_t count,
                            uint32_t feature_dim, const float *table,
                            uint32_t table_rows, float **features_out,
                            char *error, size_t error_size) {
    /* Each branch below fills one fixed row stride, and its consumer reads
     * exactly that stride: the table path is consumed as [count, feature_dim],
     * the sinusoidal path as [count, TIME_INPUT]. */
    uint32_t width = table ? feature_dim : TIME_INPUT;
    if (!count || !width || count > SIZE_MAX / width / sizeof(float)) {
        fail(error, error_size, "invalid number of timestep features");
        return 0;
    }
    float *features = malloc((size_t)count * width * sizeof(*features));
    if (!features) {
        fail(error, error_size, "out of memory allocating timestep features");
        return 0;
    }
    if (table) {
        /* ConvRot pruned checkpoints replace the time-embedding MLP with a
         * lookup table of raw AdaLN features per discrete timestep.
         * Matches ComfyUI interpolate_curve_table: linear interp in [0,1]. */
        uint32_t debug_index = 0;
        float span = (float)(table_rows - 1);
        for (uint32_t row = 0; row < count; row++) {
            float pos = times[row] * span;
            if (pos < 0.0f) pos = 0.0f;
            if (pos > span) pos = span;
            uint32_t lower = (uint32_t)floorf(pos);
            if (lower > table_rows - 2) lower = table_rows - 2;
            float w = pos - (float)lower;
            if (row == 0) debug_index = lower;
            for (uint32_t k = 0; k < feature_dim; k++) {
                float a = table[(size_t)lower * feature_dim + k];
                float b = table[(size_t)(lower + 1) * feature_dim + k];
                features[(size_t)row * feature_dim + k] = a + w * (b - a);
            }
        }
        if (getenv("H3_DEBUG_CONVROT")) {
            fprintf(stderr,
                    "convrot debug: time_rows=%u dim=%u row0 index=%u "
                    "features=", count, feature_dim, debug_index);
            for (uint32_t k = 0; k < feature_dim; k++)
                fprintf(stderr, " %.6f", features[k]);
            fprintf(stderr, "\n");
        }
    } else {
        for (uint32_t row = 0; row < count; row++) {
            for (uint32_t index = 0; index < TIME_INPUT / 2; index++) {
                float frequency = expf(-logf(10000.0f) *
                                       (float)index / (float)(TIME_INPUT / 2));
                float angle = times[row] * frequency;
                features[(size_t)row * TIME_INPUT + index] = cosf(angle);
                features[(size_t)row * TIME_INPUT + TIME_INPUT / 2 + index] =
                    sinf(angle);
            }
        }
    }
    *features_out = features;
    return 1;
}

static h3_gpu_tensor *time_embeddings(const h3_weight_store *weights,
                                      h3_gpu *gpu, uint32_t rows,
                                      const float *features, char *error,
                                      size_t error_size) {
    h3_gpu_tensor *input = h3_gpu_tensor_from_f32(
        gpu, features, (size_t)rows * TIME_INPUT);
    h3_gpu_tensor *in_w = weight_f32_2d(weights, gpu,
        "time_embedder.proj_in.weight", TIME_HIDDEN, TIME_INPUT,
        error, error_size);
    h3_gpu_tensor *in_b = weight_f32_1d(weights, gpu,
        "time_embedder.proj_in.bias", TIME_HIDDEN, error, error_size);
    h3_gpu_tensor *out_w = weight_f32_2d(weights, gpu,
        "time_embedder.proj_out.weight", H3_DIT_TIME_DIM, TIME_HIDDEN,
        error, error_size);
    h3_gpu_tensor *out_b = weight_f32_1d(weights, gpu,
        "time_embedder.proj_out.bias", H3_DIT_TIME_DIM, error, error_size);
    h3_gpu_tensor *hidden = h3_gpu_tensor_new_f32(
        gpu, (size_t)rows * TIME_HIDDEN);
    h3_gpu_tensor *activated = h3_gpu_tensor_new_f32(
        gpu, (size_t)rows * TIME_HIDDEN);
    h3_gpu_tensor *output = h3_gpu_tensor_new_f32(
        gpu, (size_t)rows * H3_DIT_TIME_DIM);
    h3_gpu_tensor *bf16 = h3_gpu_tensor_new_bf16(
        gpu, (size_t)rows * H3_DIT_TIME_DIM);
    h3_gpu_tensor *silu = h3_gpu_tensor_new_bf16(
        gpu, (size_t)rows * H3_DIT_TIME_DIM);
    h3_gpu_tensor *result = NULL;
    h3_gpu_tensor *all[] = {input, in_w, in_b, out_w, out_b, hidden,
                            activated, output, bf16, silu};
    for (size_t index = 0; index < sizeof(all) / sizeof(*all); index++) {
        if (!all[index]) {
            if (!error || !*error)
                fail(error, error_size, "cannot allocate timestep tensors: %s",
                     h3_gpu_error(gpu));
            goto cleanup;
        }
    }
    if (!gpu_op(gpu, h3_gpu_begin(gpu), error, error_size,
                "begin timestep embedding") ||
        !gpu_op(gpu, h3_gpu_linear_f32(gpu, hidden, input, in_w, in_b, rows,
                                       TIME_INPUT, TIME_HIDDEN),
                error, error_size, "timestep input projection") ||
        !gpu_op(gpu, h3_gpu_silu_f32(gpu, activated, hidden,
                                     rows * TIME_HIDDEN),
                error, error_size, "timestep SiLU") ||
        !gpu_op(gpu, h3_gpu_linear_f32(gpu, output, activated, out_w, out_b,
                                       rows, TIME_HIDDEN, H3_DIT_TIME_DIM),
                error, error_size, "timestep output projection") ||
        !gpu_op(gpu, h3_gpu_cast_f32_to_bf16(
                        gpu, bf16, output, rows * H3_DIT_TIME_DIM),
                error, error_size, "timestep BF16 cast") ||
        !gpu_op(gpu, h3_gpu_silu_bf16(gpu, silu, bf16,
                                      rows * H3_DIT_TIME_DIM),
                error, error_size, "timestep AdaLN SiLU") ||
        !gpu_op(gpu, h3_gpu_submit(gpu), error, error_size,
                "submit timestep embedding")) {
        goto cleanup;
    }
    result = silu;
    silu = NULL;
cleanup:
    free_tensor(&input);
    free_tensor(&in_w);
    free_tensor(&in_b);
    free_tensor(&out_w);
    free_tensor(&out_b);
    free_tensor(&hidden);
    free_tensor(&activated);
    free_tensor(&output);
    free_tensor(&bf16);
    free_tensor(&silu);
    return result;
}

/* Table mode: ConvRot pruned checkpoints serve the AdaLN input straight from
 * adaln_t_table (one feature row per discrete timestep); just cast it. */
static h3_gpu_tensor *time_table_embeddings(h3_gpu *gpu, uint32_t rows,
                                            uint32_t dim,
                                            const float *features,
                                            char *error, size_t error_size) {
    h3_gpu_tensor *input = h3_gpu_tensor_from_f32(gpu, features,
                                                  (size_t)rows * dim);
    h3_gpu_tensor *bf16 = h3_gpu_tensor_new_bf16(gpu, (size_t)rows * dim);
    if (!input || !bf16 ||
        !gpu_op(gpu, h3_gpu_begin(gpu), error, error_size,
                "begin timestep table") ||
        !gpu_op(gpu, h3_gpu_cast_f32_to_bf16(gpu, bf16, input,
                                             (size_t)rows * dim),
                error, error_size, "timestep table BF16 cast") ||
        !gpu_op(gpu, h3_gpu_submit(gpu), error, error_size,
                "submit timestep table")) {
        h3_gpu_tensor_free(input);
        h3_gpu_tensor_free(bf16);
        if (!error || !*error)
            fail(error, error_size, "cannot build table AdaLN input: %s",
                 h3_gpu_error(gpu));
        return NULL;
    }
    h3_gpu_tensor_free(input);
    return bf16;
}

/* Merge LoRA adapters into an AdaLN projection weight before the per-step
 * precompute consumes it. Adapters that do not fit the base layout (pruned
 * ConvRot checkpoints shrink the AdaLN input) are skipped with a warning
 * instead of failing the load. Returns 0 on hard failure. */
static int merge_adaln_loras(h3_lora **loras, int lora_count, h3_gpu *gpu,
                             h3_gpu_tensor *weight, const char *lora_prefix,
                             const char *target, size_t rows, size_t in_dim,
                             char *error, size_t error_size) {
    if (!loras || !lora_count) return 1;
    for (int index = 0; index < lora_count; index++) {
        h3_lora *lora = loras[index];
        if (!h3_lora_matches(lora, lora_prefix, target, rows, in_dim)) {
            fprintf(stderr,
                    "warning: LoRA adapter skipped for %s%s (%zu-wide AdaLN "
                    "input does not fit this checkpoint)\n",
                    lora_prefix ? lora_prefix : "", target, in_dim);
            continue;
        }
        if (!h3_lora_apply(gpu, lora, weight, lora_prefix, target, 0, rows,
                           in_dim, error, error_size))
            return 0;
    }
    return 1;
}

/* The AdaLN input width the cache was exported from, which its optional
 * `adaln_cache_meta_s{steps}` key carries. *width_out stays 0 when the cache has
 * no such key: an export from before that key existed cannot say which adapters
 * would have touched AdaLN. Returns 0 only for a malformed key. */
static int adaln_cache_time_dim(const h3_weight_store *weights, int steps,
                                uint32_t *width_out, char *error,
                                size_t error_size) {
    *width_out = 0;
    char name[H3_ADALN_CACHE_NAME_MAX];
    const h3_st_header *header = NULL;
    snprintf(name, sizeof(name), H3_ADALN_CACHE_META_FORMAT, steps);
    const h3_st_tensor *meta = h3_weight_find(weights, name, &header);
    if (!meta) return 1;
    if (meta->dtype != H3_DTYPE_U32 || meta->ndim != 1 || meta->shape[0] != 1) {
        fail(error, error_size, "AdaLN cache meta %s is not U32 [1]", name);
        return 0;
    }
    return h3_st_read_data(header, meta, width_out, sizeof(*width_out), error,
                           error_size) && *width_out;
}

/* Decide whether the adapters this run carries can coexist with a cached
 * modulation. An adapter that targets AdaLN cannot: the weights it merges into
 * are what the cache replaced. `merge_adaln_loras` used to skip such an adapter
 * with a warning once per block; here there is nothing to skip into, so the
 * combination is named and refused. Without a declared width the loader cannot
 * tell which adapters those are, so every adapter is refused -- which is the
 * conservative rule pre-meta exports have always had. */
static int adaln_cache_accepts_adapters(h3_lora **loras, int lora_count,
                                        uint32_t time_dim, int steps,
                                        char *error, size_t error_size) {
    if (!lora_count) return 1;
    if (!time_dim) {
        char name[H3_ADALN_CACHE_NAME_MAX];
        snprintf(name, sizeof(name), H3_ADALN_CACHE_META_FORMAT, steps);
        fail(error, error_size,
             "the AdaLN cache in this checkpoint has no %s width key, so it "
             "cannot tell whether these %d LoRA adapter(s) merge into AdaLN; "
             "re-export the cache with export_h3_adaln_cache.py from a "
             "checkpoint that still declares that width, or run without "
             "adapters", name, lora_count);
        return 0;
    }
    /* Only block 0 needs probing: h3_dit_schedule_precompute applies the same
     * target names to every block, and the final layer's target is the only
     * other AdaLN site. */
    for (int index = 0; index < lora_count; index++) {
        if (h3_lora_matches(loras[index], "transformer_blocks.0",
                            "adaln_proj.linear", BLOCK_OUTPUT, time_dim) ||
            h3_lora_matches(loras[index], "", "norm_out.linear", FINAL_OUTPUT,
                            time_dim)) {
            fail(error, error_size,
                 "LoRA adapter %d of %d carries an AdaLN factor for a %u-wide "
                 "input, which a cached checkpoint no longer projects; use a "
                 "checkpoint that keeps the adaln_proj weights",
                 index + 1, lora_count, time_dim);
            return 0;
        }
    }
    fprintf(stderr,
            "note: none of the %d LoRA adapter(s) target AdaLN, so the cached "
            "AdaLN modulation for %d steps is used unmodified\n",
            lora_count, steps);
    return 1;
}

/* Adopt the modulation a checkpoint ships instead of projecting it. The cached
 * schedule key must have this run's row values as a prefix, bit for bit: a
 * cache built for a different `--steps` would otherwise apply the wrong
 * modulation at every step while looking perfectly healthy, and a cache built
 * without a first frame is one row short of what an image-to-video run needs. */
static int load_cached_adaln(h3_dit_schedule *schedule,
                             const h3_weight_store *weights, h3_gpu *gpu,
                             const float *times, uint64_t cache_rows, int steps,
                             char *error, size_t error_size) {
    if (cache_rows < schedule->time_rows) {
        fail(error, error_size,
             "AdaLN cache for %d steps holds %llu timestep rows but this run "
             "needs %u; rebuild the cache with H3_DIT_ADALN_CACHE_DUMP",
             steps, (unsigned long long)cache_rows, schedule->time_rows);
        return 0;
    }
    char name[H3_ADALN_CACHE_NAME_MAX];
    const h3_st_header *header = NULL;
    snprintf(name, sizeof(name), H3_ADALN_CACHE_TIMES_FORMAT, steps);
    const h3_st_tensor *key = h3_weight_find(weights, name, &header);
    if (!key || key->dtype != H3_DTYPE_F32 || key->ndim != 1 ||
        key->shape[0] != cache_rows) {
        fail(error, error_size, "AdaLN cache is missing its %s schedule key",
             name);
        return 0;
    }
    size_t key_bytes = (size_t)cache_rows * sizeof(float);
    float *cached = malloc(key_bytes);
    if (!cached) {
        fail(error, error_size, "out of memory reading the AdaLN cache key");
        return 0;
    }
    if (!h3_st_read_data(header, key, cached, key_bytes, error, error_size)) {
        free(cached);
        return 0;
    }
    /* Only the prefix this run actually indexes has to agree; the tail is the
     * condition rows it does not use. */
    for (uint32_t row = 0; row < schedule->time_rows; row++) {
        if (memcmp(&cached[row], &times[row], sizeof(float)) == 0) continue;
        fail(error, error_size,
             "AdaLN cache for %d steps was built for a different sigma "
             "schedule (row %u: cached %.6f, this run %.6f); rebuild the cache "
             "at these steps", steps, row, cached[row], times[row]);
        free(cached);
        return 0;
    }
    free(cached);
    uint64_t block_shape[] = {cache_rows, BLOCK_OUTPUT};
    uint64_t final_shape[] = {cache_rows, FINAL_OUTPUT};
    for (unsigned block = 0; block < H3_DIT_BLOCKS; block++) {
        snprintf(name, sizeof(name), H3_ADALN_CACHE_BLOCK_FORMAT, block, steps);
        schedule->blocks[block] = h3_weight_load_bf16(
            weights, gpu, name, 2, block_shape, error, error_size);
        if (!schedule->blocks[block]) return 0;
    }
    snprintf(name, sizeof(name), H3_ADALN_CACHE_FINAL_FORMAT, steps);
    schedule->final = h3_weight_load_bf16(
        weights, gpu, name, 2, final_shape, error, error_size);
    return schedule->final != NULL;
}

/* H3_DIT_ADALN_CACHE_DUMP=<path> exports the freshly computed modulation as a
 * raw blob that fastvideo_qad/scripts/export_h3_adaln_cache.py repackages into
 * adaln_cache_s<steps>.safetensors. See H3_ADALN_CACHE_MAGIC in the header for
 * the on-disk layout and the test that pins both sides to it.
 *
 * Only the modulation is written. For the released 8-step schedule that is
 * ~157 MiB, against the 24.3 GiB of AdaLN weights it replaces. */
_Static_assert(sizeof(H3_ADALN_CACHE_MAGIC) - 1 == H3_ADALN_CACHE_MAGIC_BYTES,
               "the AdaLN cache magic must be exactly the bytes it writes");

/* One counted payload write; records why the stream failed. */
static int emit_dump_bytes(FILE *file, const void *data, size_t bytes,
                           int *reason) {
    if (fwrite(data, 1, bytes, file) == bytes) return 1;
    *reason = errno;
    return 0;
}

static int dump_adaln_cache(const h3_dit_schedule *schedule,
                            const float *times, const char *path,
                            char *error, size_t error_size) {
    uint32_t fields[H3_ADALN_CACHE_HEADER_FIELDS] = {
        [H3_ADALN_CACHE_FIELD_STEPS] = (uint32_t)schedule->steps,
        [H3_ADALN_CACHE_FIELD_TIME_ROWS] = schedule->time_rows,
        [H3_ADALN_CACHE_FIELD_BLOCKS] = H3_DIT_BLOCKS,
        [H3_ADALN_CACHE_FIELD_BLOCK_OUTPUT] = BLOCK_OUTPUT,
        [H3_ADALN_CACHE_FIELD_FINAL_OUTPUT] = FINAL_OUTPUT,
        [H3_ADALN_CACHE_FIELD_RESERVED] = 0u};
    /* Fill a sibling name and rename into place, so an interrupted export
     * cannot leave a half-written blob under the path the caller asked for. */
    char temp_name[4096];
    if (snprintf(temp_name, sizeof(temp_name), "%s.tmp", path) >=
        (int)sizeof(temp_name)) {
        fail(error, error_size, "AdaLN cache dump path too long: %s", path);
        return 0;
    }
    FILE *file = fopen(temp_name, "wb");
    if (!file) {
        fail(error, error_size, "cannot open AdaLN cache dump %s: %s", temp_name,
             strerror(errno));
        return 0;
    }
    size_t block_elements = (size_t)schedule->time_rows * BLOCK_OUTPUT;
    size_t final_elements = (size_t)schedule->time_rows * FINAL_OUTPUT;
    size_t capacity = block_elements > final_elements ?
        block_elements : final_elements;
    uint16_t *values = malloc(capacity * sizeof(*values));
    int reason = 0;
    int ok = values != NULL;
    if (!ok)
        fail(error, error_size, "out of memory dumping the AdaLN cache");
    if (ok && (!emit_dump_bytes(file, H3_ADALN_CACHE_MAGIC,
                                H3_ADALN_CACHE_MAGIC_BYTES, &reason) ||
               !emit_dump_bytes(file, fields, sizeof(fields), &reason) ||
               !emit_dump_bytes(file, times,
                                (size_t)schedule->time_rows * sizeof(*times),
                                &reason)))
        ok = 0;
    for (unsigned block = 0; ok && block < H3_DIT_BLOCKS; block++) {
        ok = h3_gpu_tensor_read_bf16(schedule->blocks[block], values,
                                     block_elements) &&
            emit_dump_bytes(file, values, block_elements * sizeof(*values),
                            &reason);
    }
    if (ok)
        ok = h3_gpu_tensor_read_bf16(schedule->final, values, final_elements) &&
            emit_dump_bytes(file, values, final_elements * sizeof(*values),
                            &reason);
    if (fclose(file) != 0 && ok) {
        ok = 0;
        reason = errno;
    }
    free(values);
    if (!ok) {
        remove(temp_name);
        if (!error || !*error)
            fail(error, error_size, "cannot write AdaLN cache dump %s: %s", path,
                 reason ? strerror(reason) :
                          "the modulation could not be read back");
        return 0;
    }
    if (rename(temp_name, path) != 0) {
        fail(error, error_size, "cannot install AdaLN cache dump %s: %s", path,
             strerror(errno));
        remove(temp_name);
        return 0;
    }
    return 1;
}

h3_dit_schedule *h3_dit_schedule_precompute(
    const h3_weight_store *weights, h3_gpu *gpu,
    const h3_sigma_schedule *sigmas, int visual_condition,
    int audio_condition, h3_lora **loras, int lora_count,
    h3_dit_schedule_progress progress, void *progress_opaque,
    char *error, size_t error_size) {
    if (error && error_size) error[0] = '\0';
    if (!weights || !gpu || !sigmas || sigmas->steps < 1 ||
        sigmas->steps > H3_MAX_STEPS) {
        fail(error, error_size, "invalid AdaLN schedule arguments");
        return NULL;
    }
    h3_dit_schedule *schedule = calloc(1, sizeof(*schedule));
    if (!schedule) {
        fail(error, error_size, "out of memory creating AdaLN schedule");
        return NULL;
    }
    schedule->gpu = gpu;
    /* Read once and use the same test everywhere: an empty value means "not
     * set", and treating it as set would force two extra condition rows into
     * the schedule while exporting nothing. */
    const char *dump_path = getenv("H3_DIT_ADALN_CACHE_DUMP");
    int dumping = dump_path && *dump_path;
    float *features = NULL;
    float *times = NULL;
    /* Declared here, not at first use: every failure below jumps to `failed`,
     * which frees it. */
    h3_gpu_tensor *time = NULL;
    /* ConvRot pruned checkpoints shrink the AdaLN input from the 2688-dim
     * time embedding to a few raw features served by adaln_t_table. */
    uint32_t time_dim = H3_DIT_TIME_DIM;
    float *table = NULL;
    uint32_t table_rows = 0;
    char cache_name[H3_ADALN_CACHE_NAME_MAX];
    snprintf(cache_name, sizeof(cache_name), H3_ADALN_CACHE_BLOCK_FORMAT, 0u,
             sigmas->steps);
    const h3_st_tensor *cache_tensor = h3_weight_find(weights, cache_name, NULL);
    const h3_st_tensor *adaln_probe = h3_weight_find(
        weights, "blocks.0.adaln_proj.linear.weight", NULL);
    if (cache_tensor) {
        /* A checkpoint that ships the modulation need not ship the weights it
         * came from, so this probe is what selects the cache path. */
        if (cache_tensor->dtype != H3_DTYPE_BF16 || cache_tensor->ndim != 2 ||
            cache_tensor->shape[1] != BLOCK_OUTPUT ||
            !cache_tensor->shape[0]) {
            fail(error, error_size,
                 "AdaLN cache tensor %s is not BF16 [rows, %u]", cache_name,
                 (unsigned)BLOCK_OUTPUT);
            goto failed;
        }
    } else if (adaln_probe && adaln_probe->ndim == 2 &&
               adaln_probe->shape[1] != H3_DIT_TIME_DIM) {
        time_dim = (uint32_t)adaln_probe->shape[1];
        const h3_st_header *table_header = NULL;
        const h3_st_tensor *table_tensor =
            h3_weight_find(weights, "adaln_t_table", &table_header);
        if (!table_tensor || table_tensor->dtype != H3_DTYPE_F32 ||
            table_tensor->ndim != 2 ||
            table_tensor->shape[1] != time_dim || !table_tensor->shape[0]) {
            fail(error, error_size,
                 "checkpoint uses a %llu-wide AdaLN input but adaln_t_table "
                 "is missing or incompatible",
                 (unsigned long long)adaln_probe->shape[1]);
            goto failed;
        }
        table_rows = (uint32_t)table_tensor->shape[0];
        table = malloc((size_t)table_rows * time_dim * sizeof(*table));
        if (!table) {
            fail(error, error_size, "out of memory loading adaln_t_table");
            goto failed;
        }
        if (!h3_st_read_data(table_header, table_tensor, table,
                             (size_t)table_rows * time_dim * sizeof(*table),
                             error, error_size))
            goto failed;
    }

    /* A checkpoint that ships the modulation drops the weights it came from, so
     * a step count with no cache has nothing left to project from. Failing here
     * names the real problem; letting it through would surface much later as a
     * bare "required weight is absent: blocks.0.adaln_proj.linear.weight". */
    if (!cache_tensor && !adaln_probe) {
        fail(error, error_size,
             "this checkpoint ships AdaLN caches but none for %d steps, and it "
             "has no adaln_proj weights to fall back on; export a cache at "
             "these steps with H3_DIT_ADALN_CACHE_DUMP", sigmas->steps);
        goto failed;
    }

    /* An exported cache has to cover every conditioning mode that will consume
     * it, and the modes differ only in the condition rows appended after the
     * step rows, in the fixed order visual then audio:
     *
     *   text-to-video       2*steps - 1 rows   (no condition row)
     *   first/last frame    2*steps     rows   (visual row)
     *   reference image     2*steps + 1 rows   (visual row, audio row)
     *
     * Materializing both rows during an export therefore makes one cache serve
     * all three -- the shorter modes match it as a prefix -- at the cost of two
     * unused rows (~20 MiB). Forcing only the visual row would leave reference
     * images one row short and refused, with "rebuild the cache" advice that
     * rebuilding could never satisfy. Only the export run is affected. */
    if (dumping) {
        visual_condition = 1;
        audio_condition = 1;
    }
    if (!prepare_times(schedule, sigmas, visual_condition, audio_condition,
                       &times, error, error_size)) goto failed;
    if (cache_tensor) {
        /* One conditioning mode no cache can serve: its single condition row
         * lands where every export puts the *visual* row, so the prefix check
         * below would report it as a mismatched sigma schedule and advise
         * rebuilding -- which no step count could ever satisfy. */
        if (audio_condition && !visual_condition) {
            fail(error, error_size,
                 "an AdaLN cache cannot serve an audio-only reference: exports "
                 "put the visual condition row first, so this run's audio row "
                 "falls on a different schedule value; use a reference image or "
                 "a checkpoint that keeps the adaln_proj weights");
            goto failed;
        }
        uint32_t cached_time_dim = 0;
        if (!adaln_cache_time_dim(weights, sigmas->steps, &cached_time_dim,
                                 error, error_size))
            goto failed;
        /* A cached checkpoint has no adaln_proj weights for an adapter to merge
         * into, and the cached modulation was computed without one. */
        if (!adaln_cache_accepts_adapters(loras, lora_count,
                                          cached_time_dim, sigmas->steps,
                                          error, error_size))
            goto failed;
        if (!load_cached_adaln(schedule, weights, gpu, times,
                               cache_tensor->shape[0], sigmas->steps, error,
                               error_size))
            goto failed;
        if (dumping)
            fprintf(stderr,
                    "warning: H3_DIT_ADALN_CACHE_DUMP ignored; this checkpoint "
                    "already ships an AdaLN cache\n");
        /* No per-block projections run, so report the phase as complete rather
         * than leaving it stalled at 0/50. */
        if (progress)
            progress((int)H3_DIT_BLOCKS, (int)H3_DIT_BLOCKS, progress_opaque);
        free(times);
        return schedule;
    }
    if (!prepare_features(times, schedule->time_rows, time_dim, table,
                          table_rows, &features, error, error_size))
        goto failed;
    time = table ?
        time_table_embeddings(gpu, schedule->time_rows, time_dim, features,
                              error, error_size) :
        time_embeddings(weights, gpu, schedule->time_rows, features,
                        error, error_size);
    free(features);
    features = NULL;
    free(table);
    table = NULL;
    if (!time) goto failed;

    for (unsigned block = 0; block < H3_DIT_BLOCKS; block++) {
        char weight_name[128], bias_name[128], operation[128];
        snprintf(weight_name, sizeof(weight_name),
                 "blocks.%u.adaln_proj.linear.weight", block);
        snprintf(bias_name, sizeof(bias_name),
                 "blocks.%u.adaln_proj.linear.bias", block);
        h3_gpu_tensor *weight = weight_bf16_any_2d(
            weights, gpu, weight_name, BLOCK_OUTPUT, time_dim,
            error, error_size);
        /* Load the bias only after the weight succeeded: h3_st_read_data
         * clears the shared error buffer on entry, which would otherwise wipe
         * out the weight's failure message. */
        h3_gpu_tensor *bias = NULL;
        if (weight)
            bias = weight_bf16_any_1d(weights, gpu, bias_name, BLOCK_OUTPUT,
                                      error, error_size);
        schedule->blocks[block] = h3_gpu_tensor_new_bf16(
            gpu, (size_t)schedule->time_rows * BLOCK_OUTPUT);
        if (!weight || !bias || !schedule->blocks[block]) {
            if (!error || !*error)
                fail(error, error_size,
                     "cannot allocate AdaLN block %u: weight=%d bias=%d out=%d (%s)",
                     block, weight != NULL, bias != NULL,
                     schedule->blocks[block] != NULL, h3_gpu_error(gpu));
            free_tensor(&weight);
            free_tensor(&bias);
            h3_gpu_tensor_free(time);
            time = NULL;
            goto failed;
        }
        snprintf(operation, sizeof(operation), "AdaLN block %u", block);
        char lora_prefix[64];
        snprintf(lora_prefix, sizeof(lora_prefix), "transformer_blocks.%u",
                 block);
        int ok = merge_adaln_loras(loras, lora_count, gpu, weight,
                                   lora_prefix, "adaln_proj.linear",
                                   BLOCK_OUTPUT, time_dim, error, error_size) &&
            gpu_op(gpu, h3_gpu_begin(gpu), error, error_size, operation) &&
            gpu_op(gpu, h3_gpu_linear_bf16(
                gpu, schedule->blocks[block], time, weight, bias,
                schedule->time_rows, time_dim, BLOCK_OUTPUT),
                error, error_size, operation) &&
            gpu_op(gpu, h3_gpu_submit(gpu), error, error_size, operation);
        free_tensor(&weight);
        free_tensor(&bias);
        if (!ok) {
            h3_gpu_tensor_free(time);
            time = NULL;
            goto failed;
        }
        if (block == 0 && getenv("H3_DEBUG_CONVROT")) {
            uint16_t probe[8];
            if (h3_gpu_tensor_read_bf16(schedule->blocks[0], probe, 8)) {
                fprintf(stderr, "convrot debug: mod0[0..7] =");
                for (int k = 0; k < 8; k++) {
                    uint32_t bits = (uint32_t)probe[k] << 16;
                    float v;
                    memcpy(&v, &bits, sizeof(v));
                    fprintf(stderr, " %.6f", v);
                }
                fprintf(stderr, "\n");
            }
        }
        if (progress) progress((int)block + 1, (int)H3_DIT_BLOCKS,
                               progress_opaque);
    }

    h3_gpu_tensor *final_w = weight_bf16_any_2d(
        weights, gpu, "final_layer.adaln_proj.linear.weight",
        FINAL_OUTPUT, time_dim, error, error_size);
    h3_gpu_tensor *final_b = weight_bf16_any_1d(
        weights, gpu, "final_layer.adaln_proj.linear.bias",
        FINAL_OUTPUT, error, error_size);
    schedule->final = h3_gpu_tensor_new_bf16(
        gpu, (size_t)schedule->time_rows * FINAL_OUTPUT);
    if (!final_w || !final_b || !schedule->final ||
        !merge_adaln_loras(loras, lora_count, gpu, final_w, "",
                           "norm_out.linear", FINAL_OUTPUT, time_dim,
                           error, error_size) ||
        !gpu_op(gpu, h3_gpu_begin(gpu), error, error_size,
                "begin final AdaLN") ||
        !gpu_op(gpu, h3_gpu_linear_bf16(
            gpu, schedule->final, time, final_w, final_b, schedule->time_rows,
            time_dim, FINAL_OUTPUT), error, error_size,
            "final AdaLN projection") ||
        !gpu_op(gpu, h3_gpu_submit(gpu), error, error_size,
                "submit final AdaLN")) {
        if ((!error || !*error) && (!final_w || !final_b || !schedule->final))
            fail(error, error_size, "cannot allocate final AdaLN tensors: %s",
                 h3_gpu_error(gpu));
        free_tensor(&final_w);
        free_tensor(&final_b);
        h3_gpu_tensor_free(time);
        time = NULL;
        goto failed;
    }
    free_tensor(&final_w);
    free_tensor(&final_b);
    h3_gpu_tensor_free(time);
    time = NULL;
    if (dumping &&
        !dump_adaln_cache(schedule, times, dump_path, error, error_size))
        goto failed;
    free(times);
    return schedule;

failed:
    free(table);
    free(features);
    free(times);
    h3_gpu_tensor_free(time);
    h3_dit_schedule_free(schedule);
    return NULL;
}

void h3_dit_schedule_free(h3_dit_schedule *schedule) {
    if (!schedule) return;
    for (unsigned block = 0; block < H3_DIT_BLOCKS; block++)
        h3_gpu_tensor_free(schedule->blocks[block]);
    h3_gpu_tensor_free(schedule->final);
    free(schedule->video_rows);
    free(schedule->audio_rows);
    free(schedule->visual_condition_rows);
    free(schedule->audio_condition_rows);
    free(schedule);
}

int h3_dit_schedule_steps(const h3_dit_schedule *schedule) {
    return schedule ? schedule->steps : 0;
}

uint32_t h3_dit_schedule_time_rows(const h3_dit_schedule *schedule) {
    return schedule ? schedule->time_rows : 0;
}

uint32_t h3_dit_schedule_video_row(const h3_dit_schedule *schedule, int step) {
    return schedule && step >= 0 && step < schedule->steps ?
        schedule->video_rows[step] : UINT32_MAX;
}

uint32_t h3_dit_schedule_audio_row(const h3_dit_schedule *schedule, int step) {
    return schedule && step >= 0 && step < schedule->steps ?
        schedule->audio_rows[step] : UINT32_MAX;
}

uint32_t h3_dit_schedule_visual_condition_row(
    const h3_dit_schedule *schedule, int step) {
    return schedule && schedule->visual_condition_rows && step >= 0 &&
        step < schedule->steps ? schedule->visual_condition_rows[step] :
        UINT32_MAX;
}

uint32_t h3_dit_schedule_audio_condition_row(
    const h3_dit_schedule *schedule, int step) {
    return schedule && schedule->audio_condition_rows && step >= 0 &&
        step < schedule->steps ? schedule->audio_condition_rows[step] :
        UINT32_MAX;
}

const h3_gpu_tensor *h3_dit_schedule_block(const h3_dit_schedule *schedule,
                                           unsigned block) {
    return schedule && block < H3_DIT_BLOCKS ? schedule->blocks[block] : NULL;
}

int h3_dit_schedule_gate_scores(const h3_dit_schedule *schedule,
                                unsigned first, unsigned count, double *out) {
    if (!schedule || !out || !count) return 0;
    if (first >= H3_DIT_BLOCKS || count > H3_DIT_BLOCKS - first) return 0;
    size_t elements = (size_t)schedule->time_rows * BLOCK_OUTPUT;
    if (!elements || elements > SIZE_MAX / sizeof(uint16_t)) return 0;
    /* One readback buffer is reused for every block: ranking the whole stack
     * would otherwise allocate (and free) a fresh buffer per block. */
    uint16_t *values = malloc(elements * sizeof(*values));
    if (!values) return 0;
    int ok = 1;
    for (unsigned index = 0; index < count; index++) {
        unsigned block = first + index;
        if (!schedule->blocks[block] ||
            !h3_gpu_tensor_read_bf16(schedule->blocks[block], values,
                                     elements)) {
            ok = 0;
            break;
        }
        double total = 0.0;
        size_t samples = 0;
        for (uint32_t row = 0; row < schedule->time_rows; row++)
            for (uint32_t modality = 0; modality < H3_DIT_MODALITIES; modality++)
                for (uint32_t slot = 2; slot <= 5; slot += 3) {
                    size_t base = ((size_t)row * H3_DIT_MODALITIES *
                                   H3_DIT_ADALN_SLOTS +
                                   (size_t)modality * H3_DIT_ADALN_SLOTS + slot) *
                                  H3_DIT_HIDDEN;
                    for (uint32_t column = 0; column < H3_DIT_HIDDEN; column++) {
                        uint32_t bits = (uint32_t)values[base + column] << 16;
                        float value;
                        memcpy(&value, &bits, sizeof(value));
                        total += fabs((double)value);
                    }
                    samples += H3_DIT_HIDDEN;
                }
        out[index] = samples ? total / (double)samples : -1.0;
    }
    free(values);
    return ok;
}

double h3_dit_schedule_gate_score(const h3_dit_schedule *schedule,
                                  unsigned block) {
    double score = -1.0;
    if (!schedule || block >= H3_DIT_BLOCKS) return -1.0;
    if (!h3_dit_schedule_gate_scores(schedule, block, 1, &score)) return -1.0;
    return score;
}

void h3_dit_schedule_prune(h3_dit_schedule *schedule,
                           const uint8_t *active_blocks, size_t count) {
    if (!schedule || !active_blocks || count != H3_DIT_BLOCKS) return;
    for (unsigned block = 0; block < H3_DIT_BLOCKS; block++) {
        if (active_blocks[block]) continue;
        h3_gpu_tensor_free(schedule->blocks[block]);
        schedule->blocks[block] = NULL;
    }
}

const h3_gpu_tensor *h3_dit_schedule_final(const h3_dit_schedule *schedule) {
    return schedule ? schedule->final : NULL;
}

int h3_dit_schedule_row_map(const h3_dit_schedule *schedule, int step,
                            const h3_layout *layout,
                            const uint8_t *text_tags, size_t text_tag_count,
                            uint32_t *rows, size_t row_count) {
    if (!schedule || step < 0 || step >= schedule->steps || !layout || !rows ||
        row_count != layout->seq_len || !layout->segments ||
        (text_tags && text_tag_count != (size_t)layout->signature[0])) return 0;
    size_t text_index = 0;
    for (size_t seg_index = 0; seg_index < layout->segment_count; seg_index++) {
        const h3_segment *segment = &layout->segments[seg_index];
        if (segment->start > segment->stop || segment->stop > row_count)
            return 0;
        uint32_t time_row;
        uint32_t tag;
        switch (segment->kind) {
        case H3_SEG_TEXT:
            time_row = schedule->video_rows[step];
            for (size_t row = segment->start; row < segment->stop; row++) {
                uint32_t text_tag = text_tags ? text_tags[text_index] : 1u;
                if (text_tag >= H3_DIT_MODALITIES) return 0;
                rows[row] = time_row * H3_DIT_MODALITIES + text_tag;
                text_index++;
            }
            continue;
        case H3_SEG_COND:
        case H3_SEG_REF_IMAGE:
            if (!schedule->visual_condition_rows) return 0;
            time_row = schedule->visual_condition_rows[step];
            tag = 0;
            break;
        case H3_SEG_REF_AUDIO:
            if (!schedule->audio_condition_rows) return 0;
            time_row = schedule->audio_condition_rows[step];
            tag = 2;
            break;
        case H3_SEG_AUDIO:
            time_row = schedule->audio_rows[step];
            tag = 2;
            break;
        case H3_SEG_VIDEO:
            time_row = schedule->video_rows[step];
            tag = 0;
            break;
        default:
            return 0;
        }
        uint32_t modulation = time_row * H3_DIT_MODALITIES + tag;
        for (size_t row = segment->start; row < segment->stop; row++)
            rows[row] = modulation;
    }
    return text_index == (size_t)layout->signature[0];
}
