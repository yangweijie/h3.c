#include "h3_text_encoder.h"

#include "h3_weights.h"

#include <math.h>
#include <pthread.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

enum {
    TEXT_LAYERS = 50,
    TEXT_VOCAB = 151936,
    TEXT_HIDDEN = 5120,
    TEXT_INTERMEDIATE = 25600,
    TEXT_QUERY_HEADS = 64,
    TEXT_KV_HEADS = 8,
    TEXT_HEAD_DIM = 128,
    TEXT_QUERY_DIM = TEXT_QUERY_HEADS * TEXT_HEAD_DIM,
    TEXT_KV_DIM = TEXT_KV_HEADS * TEXT_HEAD_DIM,
    TEXT_ROPE_HALF = TEXT_HEAD_DIM / 2,
    TEXT_DEFERRED_WEIGHTS = 1 + TEXT_LAYERS * 11
};

static const float TEXT_RMS_EPSILON = 1e-6f;
static const float TEXT_ROPE_THETA = 5000000.0f;

static float round_bf16(float value) {
    uint32_t bits;
    memcpy(&bits, &value, sizeof(bits));
    bits += 0x7fffu + ((bits >> 16) & 1u);
    bits &= UINT32_C(0xffff0000);
    memcpy(&value, &bits, sizeof(value));
    return value;
}

typedef struct {
    h3_gpu_tensor *input_norm;
    h3_gpu_tensor *query;
    h3_gpu_tensor *key;
    h3_gpu_tensor *value;
    h3_gpu_tensor *query_norm;
    h3_gpu_tensor *key_norm;
    h3_gpu_tensor *attention_output;
    h3_gpu_tensor *post_norm;
    h3_gpu_tensor *gate;
    h3_gpu_tensor *up;
    h3_gpu_tensor *down;
} text_layer_weights;

typedef struct {
    const h3_weight_store *store;
    h3_gpu *gpu;
    h3_gpu_tensor *deferred[TEXT_DEFERRED_WEIGHTS];
    size_t deferred_count;
    char *error;
    size_t error_size;
} load_context;

typedef struct {
    const h3_weight_store *store;
    int layer;
    text_layer_weights *weights;
    int lane;
    int lanes;
    int ok;
    char error[512];
} text_layer_prefetch;

typedef struct {
    int occupied;
    int active;
    int layer;
    int lanes;
    load_context load;
    text_layer_weights weights;
    text_layer_prefetch prefetch[8];
    pthread_t threads[8];
    int started[8];
} text_prefetch_slot;

static void fail(char *error, size_t error_size, const char *format, ...) {
    if (!error || !error_size) return;
    va_list arguments;
    va_start(arguments, format);
    vsnprintf(error, error_size, format, arguments);
    va_end(arguments);
}

static h3_gpu_tensor *defer(load_context *load, h3_gpu_tensor *tensor) {
    if (!tensor) return NULL;
    if (load->deferred_count >= TEXT_DEFERRED_WEIGHTS) {
        fail(load->error, load->error_size, "Qwen deferred weight registry overflow");
        h3_gpu_tensor_free(tensor);
        return NULL;
    }
    load->deferred[load->deferred_count++] = tensor;
    return tensor;
}

static void retire_deferred(load_context *load) {
    for (size_t index = 0; index < load->deferred_count; index++)
        h3_gpu_tensor_free(load->deferred[index]);
    load->deferred_count = 0;
}

static h3_gpu_tensor *load_1d(load_context *load, const char *name,
                              uint64_t width) {
    uint64_t shape[] = {width};
    return defer(load, h3_weight_load_bf16(load->store, load->gpu, name, 1,
                                            shape, load->error,
                                            load->error_size));
}

static h3_gpu_tensor *load_2d(load_context *load, const char *name,
                              uint64_t rows, uint64_t columns) {
    uint64_t shape[] = {rows, columns};
    return defer(load, h3_weight_load_bf16(load->store, load->gpu, name, 2,
                                            shape, load->error,
                                            load->error_size));
}

static int text_prefetch_threads(void) {
    const char *value = getenv("H3_QWEN_PREFETCH");
    if (!value || !*value) return 8;
    if (!strcmp(value, "0")) return 0;
    char *tail = NULL;
    long threads = strtol(value, &tail, 10);
    if (!tail || *tail || threads < 1) threads = 1;
    if (threads > 8) threads = 8;
    return (int)threads;
}

static int text_prefetch_depth(const h3_gpu *gpu) {
    const char *value = getenv("H3_QWEN_PREFETCH_DEPTH");
    if (!value || !*value) return h3_gpu_is_m5(gpu) ? 3 : 2;
    char *tail = NULL;
    long depth = strtol(value, &tail, 10);
    if (!tail || *tail || depth < 1) depth = 1;
    if (depth > 6) depth = 6;
    return (int)depth;
}

static int layer_weights_load(load_context *load, int layer,
                              text_layer_weights *weights) {
    char prefix[96];
    int length = snprintf(prefix, sizeof(prefix),
                          "model.language_model.layers.%d.", layer);
    if (length < 0 || (size_t)length >= sizeof(prefix)) {
        fail(load->error, load->error_size, "cannot format Qwen layer name");
        return 0;
    }
#define LOAD_1D(field, suffix, width) do {                                      \
    char name[192];                                                             \
    snprintf(name, sizeof(name), "%s%s", prefix, suffix);                     \
    weights->field = load_1d(load, name, width);                                \
    if (!weights->field) return 0;                                              \
} while (0)
#define LOAD_2D(field, suffix, rows, columns) do {                              \
    char name[192];                                                             \
    snprintf(name, sizeof(name), "%s%s", prefix, suffix);                     \
    weights->field = load_2d(load, name, rows, columns);                        \
    if (!weights->field) return 0;                                              \
} while (0)
    LOAD_1D(input_norm, "input_layernorm.weight", TEXT_HIDDEN);
    LOAD_2D(query, "self_attn.q_proj.weight", TEXT_QUERY_DIM, TEXT_HIDDEN);
    LOAD_2D(key, "self_attn.k_proj.weight", TEXT_KV_DIM, TEXT_HIDDEN);
    LOAD_2D(value, "self_attn.v_proj.weight", TEXT_KV_DIM, TEXT_HIDDEN);
    LOAD_1D(query_norm, "self_attn.q_norm.weight", TEXT_HEAD_DIM);
    LOAD_1D(key_norm, "self_attn.k_norm.weight", TEXT_HEAD_DIM);
    LOAD_2D(attention_output, "self_attn.o_proj.weight", TEXT_HIDDEN,
            TEXT_QUERY_DIM);
    LOAD_1D(post_norm, "post_attention_layernorm.weight", TEXT_HIDDEN);
    LOAD_2D(gate, "mlp.gate_proj.weight", TEXT_INTERMEDIATE, TEXT_HIDDEN);
    LOAD_2D(up, "mlp.up_proj.weight", TEXT_INTERMEDIATE, TEXT_HIDDEN);
    LOAD_2D(down, "mlp.down_proj.weight", TEXT_HIDDEN, TEXT_INTERMEDIATE);
#undef LOAD_1D
#undef LOAD_2D
    return 1;
}

static h3_gpu_tensor *allocate_bf16(load_context *load, size_t elements) {
    h3_gpu_tensor *tensor = defer(
        load, h3_gpu_tensor_new_bf16(load->gpu, elements));
    if (!tensor) {
        fail(load->error, load->error_size,
             "cannot allocate prefetched Qwen weight: %s",
             h3_gpu_error(load->gpu));
    }
    return tensor;
}

static int layer_weights_allocate(load_context *load,
                                  text_layer_weights *weights) {
#define ALLOCATE(field, elements) do {                                         \
    weights->field = allocate_bf16(load, (elements));                          \
    if (!weights->field) return 0;                                              \
} while (0)
    ALLOCATE(input_norm, TEXT_HIDDEN);
    ALLOCATE(query, (size_t)TEXT_QUERY_DIM * TEXT_HIDDEN);
    ALLOCATE(key, (size_t)TEXT_KV_DIM * TEXT_HIDDEN);
    ALLOCATE(value, (size_t)TEXT_KV_DIM * TEXT_HIDDEN);
    ALLOCATE(query_norm, TEXT_HEAD_DIM);
    ALLOCATE(key_norm, TEXT_HEAD_DIM);
    ALLOCATE(attention_output, (size_t)TEXT_HIDDEN * TEXT_QUERY_DIM);
    ALLOCATE(post_norm, TEXT_HIDDEN);
    ALLOCATE(gate, (size_t)TEXT_INTERMEDIATE * TEXT_HIDDEN);
    ALLOCATE(up, (size_t)TEXT_INTERMEDIATE * TEXT_HIDDEN);
    ALLOCATE(down, (size_t)TEXT_HIDDEN * TEXT_INTERMEDIATE);
#undef ALLOCATE
    return 1;
}

static int read_weight_bf16(const h3_weight_store *store, const char *name,
                            int ndim, const uint64_t *shape,
                            h3_gpu_tensor *destination,
                            char *error, size_t error_size) {
    const h3_st_header *header = NULL;
    const h3_st_tensor *tensor = h3_weight_find(store, name, &header);
    if (!tensor) {
        fail(error, error_size, "required weight is absent: %s", name);
        return 0;
    }
    if (tensor->dtype != H3_DTYPE_BF16 || tensor->ndim != ndim) {
        fail(error, error_size,
             "weight %s has dtype/rank %s/%d, expected BF16/%d", name,
             h3_dtype_name(tensor->dtype), tensor->ndim, ndim);
        return 0;
    }
    uint64_t elements = 1;
    for (int dimension = 0; dimension < ndim; dimension++) {
        if (tensor->shape[dimension] != shape[dimension]) {
            fail(error, error_size,
                 "weight %s shape mismatch at dimension %d", name,
                 dimension);
            return 0;
        }
        if (shape[dimension] && elements > UINT64_MAX / shape[dimension]) {
            fail(error, error_size, "weight %s shape overflows", name);
            return 0;
        }
        elements *= shape[dimension];
    }
    if (elements > SIZE_MAX ||
        !h3_gpu_tensor_read_file_bf16(destination, header->path,
                                      tensor->file_offset, (size_t)elements,
                                      error, error_size)) {
        if (error && error_size && !error[0])
            fail(error, error_size, "cannot prefetch %s", name);
        return 0;
    }
    return 1;
}

static int layer_weights_read_lane(const h3_weight_store *store, int layer,
                                   text_layer_weights *weights,
                                   int lane, int lanes,
                                   char *error, size_t error_size) {
    char prefix[96];
    int length = snprintf(prefix, sizeof(prefix),
                          "model.language_model.layers.%d.", layer);
    if (length < 0 || (size_t)length >= sizeof(prefix)) {
        fail(error, error_size, "cannot format Qwen layer name");
        return 0;
    }
#define READ_1D(field, suffix, width) do {                                     \
    int selected = item++ % lanes == lane;                                     \
    char name[192];                                                             \
    uint64_t shape[] = {width};                                                 \
    snprintf(name, sizeof(name), "%s%s", prefix, suffix);                    \
    if (selected && !read_weight_bf16(                                         \
                          store, name, 1, shape, weights->field,               \
                          error, error_size)) return 0;                         \
} while (0)
#define READ_2D(field, suffix, rows, columns) do {                             \
    int selected = item++ % lanes == lane;                                     \
    char name[192];                                                             \
    uint64_t shape[] = {rows, columns};                                         \
    snprintf(name, sizeof(name), "%s%s", prefix, suffix);                    \
    if (selected && !read_weight_bf16(                                         \
                          store, name, 2, shape, weights->field,               \
                          error, error_size)) return 0;                         \
} while (0)
    int item = 0;
    READ_1D(input_norm, "input_layernorm.weight", TEXT_HIDDEN);
    READ_2D(query, "self_attn.q_proj.weight", TEXT_QUERY_DIM, TEXT_HIDDEN);
    READ_2D(key, "self_attn.k_proj.weight", TEXT_KV_DIM, TEXT_HIDDEN);
    READ_2D(value, "self_attn.v_proj.weight", TEXT_KV_DIM, TEXT_HIDDEN);
    READ_1D(query_norm, "self_attn.q_norm.weight", TEXT_HEAD_DIM);
    READ_1D(key_norm, "self_attn.k_norm.weight", TEXT_HEAD_DIM);
    READ_2D(attention_output, "self_attn.o_proj.weight", TEXT_HIDDEN,
            TEXT_QUERY_DIM);
    READ_1D(post_norm, "post_attention_layernorm.weight", TEXT_HIDDEN);
    READ_2D(gate, "mlp.gate_proj.weight", TEXT_INTERMEDIATE, TEXT_HIDDEN);
    READ_2D(up, "mlp.up_proj.weight", TEXT_INTERMEDIATE, TEXT_HIDDEN);
    READ_2D(down, "mlp.down_proj.weight", TEXT_HIDDEN, TEXT_INTERMEDIATE);
#undef READ_1D
#undef READ_2D
    return 1;
}

static void *layer_prefetch_main(void *opaque) {
    text_layer_prefetch *prefetch = opaque;
    prefetch->ok = layer_weights_read_lane(
        prefetch->store, prefetch->layer, prefetch->weights,
        prefetch->lane, prefetch->lanes,
        prefetch->error, sizeof(prefetch->error));
    return NULL;
}

static void prefetch_slot_retire(text_prefetch_slot *slot) {
    if (!slot || !slot->occupied) return;
    if (slot->active) {
        for (int lane = 0; lane < slot->lanes; lane++) {
            if (slot->started[lane])
                pthread_join(slot->threads[lane], NULL);
        }
    }
    retire_deferred(&slot->load);
    memset(slot, 0, sizeof(*slot));
}

static int prefetch_slot_start(text_prefetch_slot *slot,
                               const h3_weight_store *store, h3_gpu *gpu,
                               int layer, int lanes,
                               char *error, size_t error_size) {
    if (!slot || slot->occupied || lanes < 1 || lanes > 8) return 0;
    memset(slot, 0, sizeof(*slot));
    slot->occupied = 1;
    slot->active = 1;
    slot->layer = layer;
    slot->lanes = lanes;
    slot->load.store = store;
    slot->load.gpu = gpu;
    slot->load.error = error;
    slot->load.error_size = error_size;
    if (!layer_weights_allocate(&slot->load, &slot->weights)) {
        prefetch_slot_retire(slot);
        return 0;
    }
    for (int lane = 0; lane < lanes; lane++) {
        text_layer_prefetch *prefetch = &slot->prefetch[lane];
        prefetch->store = store;
        prefetch->layer = layer;
        prefetch->weights = &slot->weights;
        prefetch->lane = lane;
        prefetch->lanes = lanes;
        if (pthread_create(&slot->threads[lane], NULL,
                           layer_prefetch_main, prefetch) == 0) {
            slot->started[lane] = 1;
        } else {
            prefetch->ok = layer_weights_read_lane(
                store, layer, &slot->weights, lane, lanes,
                prefetch->error, sizeof(prefetch->error));
        }
    }
    return 1;
}

static int prefetch_slot_take(text_prefetch_slot *slot,
                              load_context *load,
                              text_layer_weights *weights,
                              char *error, size_t error_size) {
    if (!slot || !slot->occupied || !load || !weights) return 0;
    if (slot->active) {
        for (int lane = 0; lane < slot->lanes; lane++) {
            if (slot->started[lane])
                pthread_join(slot->threads[lane], NULL);
        }
        slot->active = 0;
    }
    for (int lane = 0; lane < slot->lanes; lane++) {
        if (!slot->prefetch[lane].ok) {
            fail(error, error_size,
                 "Qwen layer %d prefetch lane %d failed: %s",
                 slot->layer, lane,
                 slot->prefetch[lane].error[0] ?
                     slot->prefetch[lane].error : "unknown error");
            prefetch_slot_retire(slot);
            return 0;
        }
    }
    *load = slot->load;
    *weights = slot->weights;
    slot->load.deferred_count = 0;
    memset(slot, 0, sizeof(*slot));
    return 1;
}

static void prefetch_slots_retire(text_prefetch_slot *slots, int count) {
    for (int index = 0; index < count; index++)
        prefetch_slot_retire(&slots[index]);
}

static int gpu_operation(h3_gpu *gpu, int ok, char *error, size_t error_size,
                         const char *operation, int layer) {
    if (ok) return 1;
    if (layer >= 0) {
        fail(error, error_size, "Qwen layer %d %s failed: %s", layer,
             operation, h3_gpu_error(gpu));
    } else {
        fail(error, error_size, "Qwen %s failed: %s", operation,
             h3_gpu_error(gpu));
    }
    return 0;
}

static int encode_layer(h3_gpu *gpu, const text_layer_weights *weight,
                        uint32_t tokens, h3_gpu_tensor *hidden,
                        h3_gpu_tensor *norm, h3_gpu_tensor *query,
                        h3_gpu_tensor *key, h3_gpu_tensor *value,
                        h3_gpu_tensor *attention_heads,
                        h3_gpu_tensor *attention_output,
                        h3_gpu_tensor *gate, h3_gpu_tensor *up,
                        h3_gpu_tensor *mlp_output,
                        h3_gpu_tensor *rope_cos, h3_gpu_tensor *rope_sin,
                        int layer, char *error, size_t error_size) {
#define OP(call, label) do {                                                    \
    if (!gpu_operation(gpu, (call), error, error_size, label, layer)) return 0; \
} while (0)
    OP(h3_gpu_rms_norm_bf16(gpu, norm, hidden, weight->input_norm,
                             tokens, TEXT_HIDDEN, TEXT_RMS_EPSILON),
       "input RMSNorm");
    OP(h3_gpu_linear_bf16(gpu, query, norm, weight->query, NULL, tokens,
                           TEXT_HIDDEN, TEXT_QUERY_DIM), "query projection");
    OP(h3_gpu_linear_bf16(gpu, key, norm, weight->key, NULL, tokens,
                           TEXT_HIDDEN, TEXT_KV_DIM), "key projection");
    OP(h3_gpu_linear_bf16(gpu, value, norm, weight->value, NULL, tokens,
                           TEXT_HIDDEN, TEXT_KV_DIM), "value projection");
    OP(h3_gpu_head_rms_norm_bf16(gpu, query, weight->query_norm, tokens,
                                  TEXT_QUERY_HEADS, TEXT_HEAD_DIM,
                                  TEXT_RMS_EPSILON), "query RMSNorm");
    OP(h3_gpu_head_rms_norm_bf16(gpu, key, weight->key_norm, tokens,
                                  TEXT_KV_HEADS, TEXT_HEAD_DIM,
                                  TEXT_RMS_EPSILON), "key RMSNorm");
    OP(h3_gpu_rope_text_bf16(gpu, query, key, rope_cos, rope_sin, tokens,
                              TEXT_QUERY_HEADS, TEXT_KV_HEADS, TEXT_HEAD_DIM),
       "RoPE");
    OP(h3_gpu_gqa_causal_bf16(gpu, attention_heads, query, key, value, tokens,
                               TEXT_QUERY_HEADS, TEXT_KV_HEADS, TEXT_HEAD_DIM,
                               1.0f / sqrtf((float)TEXT_HEAD_DIM)),
       "causal GQA");
    OP(h3_gpu_linear_bf16(gpu, attention_output, attention_heads,
                           weight->attention_output, NULL, tokens,
                           TEXT_QUERY_DIM, TEXT_HIDDEN),
       "attention output projection");
    OP(h3_gpu_add_bf16(gpu, hidden, hidden, attention_output,
                        tokens * TEXT_HIDDEN), "attention residual");
    OP(h3_gpu_rms_norm_bf16(gpu, norm, hidden, weight->post_norm, tokens,
                             TEXT_HIDDEN, TEXT_RMS_EPSILON),
       "post-attention RMSNorm");
    OP(h3_gpu_linear_bf16(gpu, gate, norm, weight->gate, NULL, tokens,
                           TEXT_HIDDEN, TEXT_INTERMEDIATE), "MLP gate");
    OP(h3_gpu_linear_bf16(gpu, up, norm, weight->up, NULL, tokens,
                           TEXT_HIDDEN, TEXT_INTERMEDIATE), "MLP up");
    OP(h3_gpu_silu_mul_bf16(gpu, gate, gate, up,
                             tokens * TEXT_INTERMEDIATE), "fused SwiGLU");
    OP(h3_gpu_linear_bf16(gpu, mlp_output, gate, weight->down, NULL, tokens,
                           TEXT_INTERMEDIATE, TEXT_HIDDEN), "MLP down");
    OP(h3_gpu_add_bf16(gpu, hidden, hidden, mlp_output,
                        tokens * TEXT_HIDDEN), "MLP residual");
#undef OP
    return 1;
}

void h3_text_embedding_free(h3_text_embedding *embedding) {
    if (!embedding) return;
    free(embedding->values);
    free(embedding->tags);
    memset(embedding, 0, sizeof(*embedding));
}

static int text_encode_bf16_impl(
                        const char *weight_directory,
                        const char *shader_source_path,
                        const uint32_t *token_ids, size_t token_count,
                        const h3_text_vision_span *spans, size_t span_count,
                        const uint32_t *position_ids, const uint8_t *tags,
                        int layer_count,
                        h3_text_progress progress, void *progress_opaque,
                        h3_text_embedding *output,
                        char *error, size_t error_size) {
    if (output) memset(output, 0, sizeof(*output));
    if (!weight_directory || !shader_source_path || !token_ids || !token_count ||
        !output || token_count > UINT32_MAX ||
        layer_count < 1 || layer_count > TEXT_LAYERS ||
        token_count > UINT32_MAX / TEXT_HIDDEN ||
        token_count > UINT32_MAX / TEXT_INTERMEDIATE) {
        fail(error, error_size, "invalid Qwen text encoder arguments");
        return 0;
    }
    for (size_t index = 0; index < token_count; index++) {
        if (token_ids[index] >= TEXT_VOCAB) {
            fail(error, error_size, "Qwen token ID %u is outside the vocabulary",
                 token_ids[index]);
            return 0;
        }
        if (tags && tags[index] > 2) {
            fail(error, error_size, "Qwen presentation tag is invalid");
            return 0;
        }
    }
    size_t span_cursor = 0;
    for (size_t index = 0; index < span_count; index++) {
        const h3_text_vision_span *span = &spans[index];
        if (!span->tokens || span->start < span_cursor ||
            span->start > token_count || span->tokens > token_count - span->start ||
            !span->embeddings || !span->deepstack[0] || !span->deepstack[1] ||
            !span->deepstack[2]) {
            fail(error, error_size, "invalid Qwen vision presentation span");
            return 0;
        }
        span_cursor = span->start + span->tokens;
    }

    h3_weight_store *store = h3_weight_store_open(weight_directory, error,
                                                   error_size);
    if (!store) return 0;
    h3_gpu *gpu = h3_gpu_create(shader_source_path, error, error_size);
    if (!gpu) {
        h3_weight_store_free(store);
        return 0;
    }
    h3_gpu_profile_set_label(gpu, "Qwen text encoder");
    load_context load = {store, gpu, {NULL}, 0, error, error_size};
    uint32_t tokens = (uint32_t)token_count;
    size_t hidden_count = token_count * TEXT_HIDDEN;
    size_t query_count = token_count * TEXT_QUERY_DIM;
    size_t kv_count = token_count * TEXT_KV_DIM;
    size_t intermediate_count = token_count * TEXT_INTERMEDIATE;

    float *cosines = malloc(token_count * TEXT_ROPE_HALF * sizeof(*cosines));
    float *sines = malloc(token_count * TEXT_ROPE_HALF * sizeof(*sines));
    if (!cosines || !sines) {
        fail(error, error_size, "out of memory allocating Qwen RoPE tables");
        free(cosines);
        free(sines);
        h3_gpu_free(gpu);
        h3_weight_store_free(store);
        return 0;
    }
    float inverse_frequency[TEXT_ROPE_HALF];
    for (size_t index = 0; index < TEXT_ROPE_HALF; index++) {
        inverse_frequency[index] = 1.0f /
            powf(TEXT_ROPE_THETA,
                 (float)(index * 2) / (float)TEXT_HEAD_DIM);
    }
    for (size_t position = 0; position < token_count; position++) {
        for (size_t index = 0; index < TEXT_ROPE_HALF; index++) {
            size_t axis = 0;
            if (position_ids && index < 60 && index % 3 == 1) axis = 1;
            else if (position_ids && index < 60 && index % 3 == 2) axis = 2;
            float coordinate = position_ids ?
                (float)position_ids[axis * token_count + position] :
                (float)position;
            float angle = coordinate * inverse_frequency[index];
            float cosine = cosf(angle);
            float sine = sinf(angle);
            /* The explicit Qwen mRoPE path casts its tables to the BF16
             * embedding dtype. Keep F32 storage for the fused Metal kernel,
             * but pin the values to that same operation boundary. */
            cosines[position * TEXT_ROPE_HALF + index] =
                position_ids ? round_bf16(cosine) : cosine;
            sines[position * TEXT_ROPE_HALF + index] =
                position_ids ? round_bf16(sine) : sine;
        }
    }

    h3_gpu_tensor *ids = h3_gpu_tensor_from_u32(gpu, token_ids, token_count);
    h3_gpu_tensor *rope_cos = h3_gpu_tensor_from_f32(
        gpu, cosines, token_count * TEXT_ROPE_HALF);
    h3_gpu_tensor *rope_sin = h3_gpu_tensor_from_f32(
        gpu, sines, token_count * TEXT_ROPE_HALF);
    free(cosines);
    free(sines);
    h3_gpu_tensor *hidden = h3_gpu_tensor_new_bf16(gpu, hidden_count);
    h3_gpu_tensor *norm = h3_gpu_tensor_new_bf16(gpu, hidden_count);
    h3_gpu_tensor *query = h3_gpu_tensor_new_bf16(gpu, query_count);
    h3_gpu_tensor *key = h3_gpu_tensor_new_bf16(gpu, kv_count);
    h3_gpu_tensor *value = h3_gpu_tensor_new_bf16(gpu, kv_count);
    h3_gpu_tensor *attention_heads = h3_gpu_tensor_new_bf16(gpu, query_count);
    h3_gpu_tensor *attention_output = h3_gpu_tensor_new_bf16(gpu, hidden_count);
    h3_gpu_tensor *gate = h3_gpu_tensor_new_bf16(gpu, intermediate_count);
    h3_gpu_tensor *up = h3_gpu_tensor_new_bf16(gpu, intermediate_count);
    h3_gpu_tensor *mlp_output = h3_gpu_tensor_new_bf16(gpu, hidden_count);
    h3_gpu_tensor *deepstack[3] = {NULL, NULL, NULL};
    if (span_count) {
        uint16_t *values = calloc(hidden_count, sizeof(*values));
        if (!values) {
            fail(error, error_size, "out of memory packing Qwen deepstack rows");
            goto early_cleanup;
        }
        for (size_t layer = 0; layer < 3; layer++) {
            memset(values, 0, hidden_count * sizeof(*values));
            for (size_t index = 0; index < span_count; index++) {
                const h3_text_vision_span *span = &spans[index];
                memcpy(values + span->start * TEXT_HIDDEN,
                       span->deepstack[layer],
                       span->tokens * TEXT_HIDDEN * sizeof(*values));
            }
            deepstack[layer] = h3_gpu_tensor_from_bf16(gpu, values,
                                                        hidden_count);
        }
        free(values);
    }
    h3_gpu_tensor *activations[] = {
        ids, rope_cos, rope_sin, hidden, norm, query, key, value,
        attention_heads, attention_output, gate, up, mlp_output,
        deepstack[0], deepstack[1], deepstack[2]
    };
    int ok = 1;
    for (size_t index = 0; index < sizeof(activations) / sizeof(*activations);
         index++) {
        if (!activations[index] && (index < 13 || span_count)) ok = 0;
    }
    if (!ok) {
        fail(error, error_size, "cannot allocate Qwen activations: %s",
             h3_gpu_error(gpu));
        goto cleanup;
    }

    h3_gpu_tensor *embedding_weight = load_2d(
        &load, "model.language_model.embed_tokens.weight", TEXT_VOCAB,
        TEXT_HIDDEN);
    if (!embedding_weight) goto cleanup;
    if (!gpu_operation(gpu, h3_gpu_begin(gpu), error, error_size,
                       "command stream begin", -1) ||
        !gpu_operation(gpu, h3_gpu_embedding_bf16(
                                 gpu, hidden, embedding_weight, ids, tokens,
                                 TEXT_VOCAB, TEXT_HIDDEN),
                       error, error_size, "embedding lookup", -1) ||
        !gpu_operation(gpu, h3_gpu_submit(gpu), error, error_size,
                       "embedding stream submit", -1)) {
        goto cleanup;
    }
    retire_deferred(&load);
    for (size_t index = 0; index < span_count; index++) {
        const h3_text_vision_span *span = &spans[index];
        if (!h3_gpu_tensor_write_bf16_range(
                hidden, span->start * TEXT_HIDDEN, span->embeddings,
                span->tokens * TEXT_HIDDEN)) {
            fail(error, error_size, "cannot splice Qwen vision embeddings");
            goto cleanup;
        }
    }

    int prefetch_threads = text_prefetch_threads();
    int prefetch_layers = prefetch_threads > 0 && layer_count > 1;
    int prefetch_depth = prefetch_layers ? text_prefetch_depth(gpu) : 0;
    text_prefetch_slot slots[6];
    memset(slots, 0, sizeof(slots));
    text_layer_weights weights;
    memset(&weights, 0, sizeof(weights));
    if (!layer_weights_load(&load, 0, &weights)) goto cleanup;
    int next_prefetch_layer = 1;
    for (int index = 0;
         index < prefetch_depth && next_prefetch_layer < layer_count;
         index++, next_prefetch_layer++) {
        if (!prefetch_slot_start(&slots[index], store, gpu,
                                 next_prefetch_layer, prefetch_threads,
                                 error, error_size)) {
            prefetch_slots_retire(slots, prefetch_depth);
            goto cleanup;
        }
    }
    for (int layer = 0; layer < layer_count; layer++) {
        int layer_ok =
            gpu_operation(gpu, h3_gpu_begin(gpu), error, error_size,
                          "layer stream begin", layer) &&
            encode_layer(gpu, &weights, tokens, hidden, norm, query, key,
                         value, attention_heads, attention_output, gate, up,
                         mlp_output, rope_cos, rope_sin, layer,
                         error, error_size) &&
            (!(layer < 3 && span_count) ||
             gpu_operation(gpu, h3_gpu_add_bf16(
                 gpu, hidden, hidden, deepstack[layer],
                 tokens * TEXT_HIDDEN), error, error_size,
                 "deepstack residual", layer)) &&
            gpu_operation(gpu, h3_gpu_submit(gpu), error, error_size,
                          "layer stream submit", layer);

        retire_deferred(&load);
        if (!layer_ok) {
            prefetch_slots_retire(slots, prefetch_depth);
            goto cleanup;
        }
        if (progress) progress(layer + 1, TEXT_LAYERS, progress_opaque);
        if (layer + 1 >= layer_count) continue;

        memset(&weights, 0, sizeof(weights));
        if (prefetch_layers) {
            text_prefetch_slot *next = NULL;
            for (int index = 0; index < prefetch_depth; index++) {
                if (slots[index].occupied &&
                    slots[index].layer == layer + 1) {
                    next = &slots[index];
                    break;
                }
            }
            if (!next || !prefetch_slot_take(next, &load, &weights,
                                              error, error_size)) {
                if (!next)
                    fail(error, error_size,
                         "Qwen layer %d is absent from the prefetch ring",
                         layer + 1);
                prefetch_slots_retire(slots, prefetch_depth);
                goto cleanup;
            }
            if (next_prefetch_layer < layer_count) {
                if (!prefetch_slot_start(
                        next, store, gpu, next_prefetch_layer,
                        prefetch_threads, error, error_size)) {
                    prefetch_slots_retire(slots, prefetch_depth);
                    goto cleanup;
                }
                next_prefetch_layer++;
            }
        } else if (!layer_weights_load(&load, layer + 1, &weights)) {
            goto cleanup;
        }
    }

    output->values = malloc(hidden_count * sizeof(*output->values));
    if (!output->values) {
        fail(error, error_size, "out of memory reading Qwen output");
        goto cleanup;
    }
    if (!h3_gpu_tensor_read_bf16(hidden, output->values, hidden_count) ||
        !h3_gpu_get_stats(gpu, &output->gpu_stats)) {
        h3_text_embedding_free(output);
        fail(error, error_size, "cannot read completed Qwen output");
        goto cleanup;
    }
    output->tokens = token_count;
    output->width = TEXT_HIDDEN;
    if (tags) {
        output->tags = malloc(token_count * sizeof(*output->tags));
        if (!output->tags) {
            h3_text_embedding_free(output);
            fail(error, error_size, "out of memory reading Qwen tags");
            goto cleanup;
        }
        memcpy(output->tags, tags, token_count * sizeof(*output->tags));
    }
    ok = 1;
    goto finished;

cleanup:
    ok = 0;
finished:
    retire_deferred(&load);
    for (size_t index = 0; index < sizeof(activations) / sizeof(*activations);
         index++) {
        h3_gpu_tensor_free(activations[index]);
    }
    h3_gpu_free(gpu);
    h3_weight_store_free(store);
    return ok;

early_cleanup:
    h3_gpu_tensor_free(ids);
    h3_gpu_tensor_free(rope_cos);
    h3_gpu_tensor_free(rope_sin);
    h3_gpu_tensor_free(hidden);
    h3_gpu_tensor_free(norm);
    h3_gpu_tensor_free(query);
    h3_gpu_tensor_free(key);
    h3_gpu_tensor_free(value);
    h3_gpu_tensor_free(attention_heads);
    h3_gpu_tensor_free(attention_output);
    h3_gpu_tensor_free(gate);
    h3_gpu_tensor_free(up);
    h3_gpu_tensor_free(mlp_output);
    for (size_t index = 0; index < 3; index++)
        h3_gpu_tensor_free(deepstack[index]);
    retire_deferred(&load);
    h3_gpu_free(gpu);
    h3_weight_store_free(store);
    return 0;
}

/* ---- ClipProj in-engine text encoder -------------------------------------
 * Runs Qwen3-VL-4B (truncated to tap+1 layers) on Metal reusing the same
 * gpu ops as the 50-layer encoder, then lifts the tapped 2560-dim hidden to
 * the 5120-dim H3 conditioning space with the ClipProj MLP. Dimensions are
 * read from the 4B weights so the function stays robust to config changes. */

#define CP_TAP_LAYERS 25          /* harness tap=24 -> hidden_states[25] */
#define CP_HEAD_DIM 128
#define CP_ROPE_HALF (CP_HEAD_DIM / 2)
#define CP_ROPE_THETA 5000000.0f
#define CP_MLP_HIDDEN 32768       /* ClipProj inner dim (2560 -> 32768 -> 5120) */
#define CP_OUT_DIM 5120
#define CP_RMS_EPS 1e-6f
#define CP_VOCAB 151936

static float cp_bf16_to_f32(uint16_t h) {
    uint32_t bits = ((uint32_t)h) << 16;
    float f;
    memcpy(&f, &bits, sizeof(f));
    return f;
}

static uint16_t cp_f32_to_bf16(float value) {
    uint32_t bits;
    memcpy(&bits, &value, sizeof(bits));
    bits += 0x7fffu + ((bits >> 16) & 1u);
    bits &= UINT32_C(0xffff0000);
    return (uint16_t)(bits >> 16);
}

static float cp_f16_to_f32(uint16_t h) {
    uint32_t sign = (uint32_t)(h >> 15) & 1u;
    uint32_t exp = (uint32_t)(h >> 10) & 0x1fu;
    uint32_t mant = (uint32_t)(h & 0x3ffu);
    if (exp == 0) {
        if (mant == 0) return sign ? -0.0f : 0.0f;
        float f = (float)mant * 5.9604644775390625e-8f;  /* 2^-24 */
        return sign ? -f : f;
    }
    if (exp == 0x1fu) {
        uint32_t r = (sign << 31) | 0x7f800000u | (mant ? 0x7fffffu : 0u);
        float f;
        memcpy(&f, &r, sizeof(f));
        return f;
    }
    uint32_t r = (sign << 31) | ((exp - 15u + 127u) << 23) | (mant << 13);
    float f;
    memcpy(&f, &r, sizeof(f));
    return f;
}

static h3_gpu_tensor *cp_load_2d(h3_gpu *gpu, const h3_weight_store *store,
                                 const char *name, uint64_t rows, uint64_t cols,
                                 char *error, size_t error_size) {
    uint64_t shape[] = {rows, cols};
    return h3_weight_load_bf16(store, gpu, name, 2, shape, error, error_size);
}

static h3_gpu_tensor *cp_load_1d(h3_gpu *gpu, const h3_weight_store *store,
                                 const char *name, uint64_t dim,
                                 char *error, size_t error_size) {
    uint64_t shape[] = {dim};
    return h3_weight_load_bf16(store, gpu, name, 1, shape, error, error_size);
}

/* Read a tensor from a weight store, always upcast to host float32 (the
 * ClipProj projection file stores F16 weights; the Python harness upcasts too). */
static int cp_load_f32(const h3_weight_store *store, const char *name,
                       float **out, size_t *count,
                       char *error, size_t error_size) {
    const h3_st_header *header = NULL;
    const h3_st_tensor *tensor = h3_weight_find(store, name, &header);
    if (!tensor) {
        fail(error, error_size, "ClipProj projection tensor absent: %s", name);
        return 0;
    }
    size_t n = (size_t)h3_st_tensor_elements(tensor);
    float *buf = malloc(n * sizeof(float));
    if (!buf) {
        fail(error, error_size, "out of memory reading ClipProj %s", name);
        return 0;
    }
    if (tensor->dtype == H3_DTYPE_F32) {
        if (!h3_st_read_data(header, tensor, buf, n * sizeof(float),
                             error, error_size)) { free(buf); return 0; }
    } else if (tensor->dtype == H3_DTYPE_F16) {
        uint16_t *raw = malloc(n * sizeof(uint16_t));
        if (!raw) { free(buf);
                    fail(error, error_size, "oom reading ClipProj %s", name);
                    return 0; }
        if (!h3_st_read_data(header, tensor, raw, n * sizeof(uint16_t),
                             error, error_size)) { free(raw); free(buf); return 0; }
        for (size_t i = 0; i < n; i++) buf[i] = cp_f16_to_f32(raw[i]);
        free(raw);
    } else {
        fail(error, error_size, "ClipProj tensor %s is %s, expected F32/F16",
             name, h3_dtype_name(tensor->dtype));
        free(buf);
        return 0;
    }
    *out = buf;
    *count = n;
    return 1;
}

static float cp_gelu(float x) {
    return 0.5f * x * (1.0f + erff(x / (float)M_SQRT2));
}

static int cp_encode_layer(h3_gpu *gpu, const text_layer_weights *w,
                           uint32_t tokens, h3_gpu_tensor *hidden,
                           h3_gpu_tensor *norm, h3_gpu_tensor *query,
                           h3_gpu_tensor *key, h3_gpu_tensor *value,
                           h3_gpu_tensor *attention_heads,
                           h3_gpu_tensor *attention_output,
                           h3_gpu_tensor *gate, h3_gpu_tensor *up,
                           h3_gpu_tensor *mlp_output,
                           h3_gpu_tensor *rope_cos, h3_gpu_tensor *rope_sin,
                           uint64_t H, uint64_t I, uint64_t QD, uint64_t KVD,
                           int QH, int KVH, char *error, size_t error_size) {
#define CP_OP(call, label) do {                                            \
    if (!gpu_operation(gpu, (call), error, error_size, label, -1)) return 0; \
} while (0)
    CP_OP(h3_gpu_rms_norm_bf16(gpu, norm, hidden, w->input_norm,
                                tokens, (int)H, CP_RMS_EPS), "input RMSNorm");
    CP_OP(h3_gpu_linear_bf16(gpu, query, norm, w->query, NULL, tokens,
                             (int)H, (int)QD), "query projection");
    CP_OP(h3_gpu_linear_bf16(gpu, key, norm, w->key, NULL, tokens,
                             (int)H, (int)KVD), "key projection");
    CP_OP(h3_gpu_linear_bf16(gpu, value, norm, w->value, NULL, tokens,
                             (int)H, (int)KVD), "value projection");
    CP_OP(h3_gpu_head_rms_norm_bf16(gpu, query, w->query_norm, tokens,
                                    QH, CP_HEAD_DIM, CP_RMS_EPS), "query RMSNorm");
    CP_OP(h3_gpu_head_rms_norm_bf16(gpu, key, w->key_norm, tokens,
                                    KVH, CP_HEAD_DIM, CP_RMS_EPS), "key RMSNorm");
    CP_OP(h3_gpu_rope_text_bf16(gpu, query, key, rope_cos, rope_sin, tokens,
                                 QH, KVH, CP_HEAD_DIM), "RoPE");
    {
        const char *qd = getenv("H3_CLIPPROJ_DUMP_QROPE");
        if (qd) {
            uint16_t *qb = malloc(tokens * (size_t)QD * sizeof(uint16_t));
            if (qb && h3_gpu_tensor_read_bf16(query, qb, tokens * (size_t)QD)) {
                FILE *qf = fopen(qd, "wb");
                if (qf) {
                    uint32_t seq = (uint32_t)tokens, dim = (uint32_t)QD;
                    fwrite(&seq, sizeof(seq), 1, qf);
                    fwrite(&dim, sizeof(dim), 1, qf);
                    for (size_t i = 0; i < tokens * (size_t)QD; i++) {
                        float v = cp_bf16_to_f32(qb[i]);
                        fwrite(&v, sizeof(v), 1, qf);
                    }
                    fclose(qf);
                }
                free(qb);
            }
        }
    }
    CP_OP(h3_gpu_gqa_causal_bf16(gpu, attention_heads, query, key, value,
                                  tokens, QH, KVH, CP_HEAD_DIM,
                                  1.0f / (float)sqrt((float)CP_HEAD_DIM)),
         "causal GQA");
    CP_OP(h3_gpu_linear_bf16(gpu, attention_output, attention_heads,
                             w->attention_output, NULL, tokens, (int)QD, (int)H),
         "attention output projection");
    CP_OP(h3_gpu_add_bf16(gpu, hidden, hidden, attention_output,
                          tokens * (int)H), "attention residual");
    {
        const char *ad = getenv("H3_CLIPPROJ_DUMP_ATTN");
        if (ad) {
            uint16_t *ah = malloc(tokens * (size_t)H * sizeof(uint16_t));
            if (ah && h3_gpu_tensor_read_bf16(hidden, ah, tokens * (size_t)H)) {
                FILE *af = fopen(ad, "wb");
                if (af) {
                    uint32_t seq = (uint32_t)tokens, hd = (uint32_t)H;
                    fwrite(&seq, sizeof(seq), 1, af);
                    fwrite(&hd, sizeof(hd), 1, af);
                    for (size_t i = 0; i < tokens * (size_t)H; i++) {
                        float v = cp_bf16_to_f32(ah[i]);
                        fwrite(&v, sizeof(v), 1, af);
                    }
                    fclose(af);
                }
                free(ah);
            }
        }
    }
    CP_OP(h3_gpu_rms_norm_bf16(gpu, norm, hidden, w->post_norm, tokens,
                               (int)H, CP_RMS_EPS), "post-attention RMSNorm");
    CP_OP(h3_gpu_linear_bf16(gpu, gate, norm, w->gate, NULL, tokens,
                             (int)H, (int)I), "MLP gate");
    CP_OP(h3_gpu_linear_bf16(gpu, up, norm, w->up, NULL, tokens,
                             (int)H, (int)I), "MLP up");
    CP_OP(h3_gpu_silu_mul_bf16(gpu, gate, gate, up, tokens * (int)I),
         "fused SwiGLU");
    CP_OP(h3_gpu_linear_bf16(gpu, mlp_output, gate, w->down, NULL, tokens,
                             (int)I, (int)H), "MLP down");
    CP_OP(h3_gpu_add_bf16(gpu, hidden, hidden, mlp_output, tokens * (int)H),
         "MLP residual");
#undef CP_OP
    return 1;
}

int h3_text_encode_clipproj_bf16(const char *qwen4b_directory,
                                 const char *projection_directory,
                                 const char *shader_source_path,
                                 const uint32_t *token_ids, size_t token_count,
                                 h3_text_progress progress, void *progress_opaque,
                                 h3_text_embedding *output,
                                 char *error, size_t error_size) {
    if (output) memset(output, 0, sizeof(*output));
    if (!qwen4b_directory || !projection_directory || !shader_source_path ||
        !token_ids || !token_count || token_count > UINT32_MAX) {
        fail(error, error_size, "invalid ClipProj text encoder arguments");
        return 0;
    }
    for (size_t i = 0; i < token_count; i++)
        if (token_ids[i] >= CP_VOCAB) {
            fail(error, error_size, "ClipProj token ID %u out of vocabulary",
                 token_ids[i]);
            return 0;
        }
    int tap_layers = CP_TAP_LAYERS;
    const char *tl = getenv("H3_CLIPPROJ_LAYERS");
    if (tl && atoi(tl) > 0) tap_layers = atoi(tl);

    h3_weight_store *store = h3_weight_store_open(qwen4b_directory, error,
                                                 error_size);
    if (!store) return 0;
    h3_weight_store *pstore = h3_weight_store_open(projection_directory, error,
                                                  error_size);
    if (!pstore) { h3_weight_store_free(store); return 0; }

    /* Derive dims from the 4B weights (robust to config changes). */
    const h3_st_header *h = NULL;
    const h3_st_tensor *t;
    uint64_t H = 0, I = 0, QD = 0, KVD = 0;
    int ok = 1;
    if (!(t = h3_weight_find(store, "model.language_model.embed_tokens.weight",
                             &h)) ||
        t->ndim != 2) { fail(error, error_size, "4B embed_tokens missing"); ok = 0; }
    else H = t->shape[1];
    if (ok && (t = h3_weight_find(store,
            "model.language_model.layers.0.self_attn.q_proj.weight", &h)) &&
        t->ndim == 2) QD = t->shape[0];
    else { fail(error, error_size, "4B q_proj missing"); ok = 0; }
    if (ok && (t = h3_weight_find(store,
            "model.language_model.layers.0.self_attn.k_proj.weight", &h)) &&
        t->ndim == 2) KVD = t->shape[0];
    else { fail(error, error_size, "4B k_proj missing"); ok = 0; }
    if (ok && (t = h3_weight_find(store,
            "model.language_model.layers.0.mlp.gate_proj.weight", &h)) &&
        t->ndim == 2) I = t->shape[0];
    else { fail(error, error_size, "4B gate_proj missing"); ok = 0; }
    int QH = (int)(QD / CP_HEAD_DIM), KVH = (int)(KVD / CP_HEAD_DIM);
    if (!ok || H == 0 || I == 0 || QD == 0 || KVD == 0 || QH < 1 || KVH < 1) {
        h3_weight_store_free(pstore);
        h3_weight_store_free(store);
        return 0;
    }

    /* Load ClipProj F32 projection tensors. */
    float *mean_in = NULL, *std_in = NULL, *mean_out = NULL, *std_out = NULL;
    float *sink_out = NULL, *w0 = NULL, *b0 = NULL, *w2 = NULL, *b2 = NULL;
    size_t n_unused;
    ok = cp_load_f32(pstore, "mean_in", &mean_in, &n_unused, error, error_size) &&
         cp_load_f32(pstore, "std_in", &std_in, &n_unused, error, error_size) &&
         cp_load_f32(pstore, "mean_out", &mean_out, &n_unused, error, error_size) &&
         cp_load_f32(pstore, "std_out", &std_out, &n_unused, error, error_size) &&
         cp_load_f32(pstore, "sink_out", &sink_out, &n_unused, error, error_size) &&
         cp_load_f32(pstore, "mlp.0.weight", &w0, &n_unused, error, error_size) &&
         cp_load_f32(pstore, "mlp.0.bias", &b0, &n_unused, error, error_size) &&
         cp_load_f32(pstore, "mlp.2.weight", &w2, &n_unused, error, error_size) &&
         cp_load_f32(pstore, "mlp.2.bias", &b2, &n_unused, error, error_size);
    if (!ok) {
        free(mean_in); free(std_in); free(mean_out); free(std_out);
        free(sink_out); free(w0); free(b0); free(w2); free(b2);
        h3_weight_store_free(pstore);
        h3_weight_store_free(store);
        return 0;
    }

    h3_gpu *gpu = h3_gpu_create(shader_source_path, error, error_size);
    if (!gpu) {
        free(mean_in); free(std_in); free(mean_out); free(std_out);
        free(sink_out); free(w0); free(b0); free(w2); free(b2);
        h3_weight_store_free(pstore);
        h3_weight_store_free(store);
        return 0;
    }

    uint32_t tokens = (uint32_t)token_count;
    size_t hidden_count = token_count * H;
    size_t q_count = token_count * QD;
    size_t kv_count = token_count * KVD;
    size_t inter_count = token_count * I;

    /* RoPE tables (1D positions, text-only mRoPE). */
    float *cosines = malloc(token_count * CP_ROPE_HALF * sizeof(float));
    float *sines = malloc(token_count * CP_ROPE_HALF * sizeof(float));
    float inv_freq[CP_ROPE_HALF];
    for (size_t i = 0; i < CP_ROPE_HALF; i++)
        inv_freq[i] = 1.0f / powf(CP_ROPE_THETA,
                                   (float)(i * 2) / (float)CP_HEAD_DIM);
    for (size_t pos = 0; pos < token_count; pos++)
        for (size_t i = 0; i < CP_ROPE_HALF; i++) {
            float angle = (float)pos * inv_freq[i];
            cosines[pos * CP_ROPE_HALF + i] = cosf(angle);
            sines[pos * CP_ROPE_HALF + i] = sinf(angle);
        }

    int rc = 0;
    h3_gpu_tensor *ids = h3_gpu_tensor_from_u32(gpu, token_ids, token_count);
    h3_gpu_tensor *rope_cos = h3_gpu_tensor_from_f32(gpu, cosines,
                                                     token_count * CP_ROPE_HALF);
    h3_gpu_tensor *rope_sin = h3_gpu_tensor_from_f32(gpu, sines,
                                                     token_count * CP_ROPE_HALF);
    h3_gpu_tensor *hidden = h3_gpu_tensor_new_bf16(gpu, hidden_count);
    h3_gpu_tensor *norm = h3_gpu_tensor_new_bf16(gpu, hidden_count);
    h3_gpu_tensor *query = h3_gpu_tensor_new_bf16(gpu, q_count);
    h3_gpu_tensor *key = h3_gpu_tensor_new_bf16(gpu, kv_count);
    h3_gpu_tensor *value = h3_gpu_tensor_new_bf16(gpu, kv_count);
    h3_gpu_tensor *attention_heads = h3_gpu_tensor_new_bf16(gpu, q_count);
    h3_gpu_tensor *attention_output = h3_gpu_tensor_new_bf16(gpu, hidden_count);
    h3_gpu_tensor *gate = h3_gpu_tensor_new_bf16(gpu, inter_count);
    h3_gpu_tensor *up = h3_gpu_tensor_new_bf16(gpu, inter_count);
    h3_gpu_tensor *mlp_output = h3_gpu_tensor_new_bf16(gpu, hidden_count);
    h3_gpu_tensor *activations[] = {ids, rope_cos, rope_sin, hidden, norm,
        query, key, value, attention_heads, attention_output, gate, up,
        mlp_output};
    for (size_t i = 0; i < sizeof(activations)/sizeof(*activations); i++)
        if (!activations[i]) { fail(error, error_size, "ClipProj alloc failed");
                               goto cp_cleanup; }

    h3_gpu_tensor *embed = cp_load_2d(gpu, store,
        "model.language_model.embed_tokens.weight", CP_VOCAB, H,
        error, error_size);
    if (!embed) goto cp_cleanup;
    if (!gpu_operation(gpu, h3_gpu_begin(gpu), error, error_size,
                       "stream begin", -1) ||
        !gpu_operation(gpu, h3_gpu_embedding_bf16(gpu, hidden, embed, ids,
                         tokens, (int)CP_VOCAB, (int)H), error, error_size,
                       "embedding lookup", -1) ||
        !gpu_operation(gpu, h3_gpu_submit(gpu), error, error_size,
                       "embedding submit", -1)) goto cp_cleanup;
    h3_gpu_tensor_free(embed);

    /* Optional dump of the post-embedding hidden (pre-layer) for fidelity
       debugging (compares against harness hidden_states[0]). */
    const char *embed_dump = getenv("H3_CLIPPROJ_DUMP_EMBED");
    if (embed_dump) {
        uint16_t *he = malloc(hidden_count * sizeof(uint16_t));
        if (he && h3_gpu_tensor_read_bf16(hidden, he, hidden_count)) {
            FILE *ef = fopen(embed_dump, "wb");
            if (ef) {
                uint32_t seq = (uint32_t)token_count, hd = (uint32_t)H;
                fwrite(&seq, sizeof(seq), 1, ef);
                fwrite(&hd, sizeof(hd), 1, ef);
                for (size_t i = 0; i < hidden_count; i++) {
                    float v = cp_bf16_to_f32(he[i]);
                    fwrite(&v, sizeof(v), 1, ef);
                }
                fclose(ef);
            }
            free(he);
        }
    }

    text_layer_weights w;
    memset(&w, 0, sizeof(w));
    char prefix[96];
    for (int layer = 0; layer < tap_layers; layer++) {
        snprintf(prefix, sizeof(prefix),
                 "model.language_model.layers.%d.", layer);
#define CP_L(name, field, rows, cols) do {                                  \
        char _n[192];                                                       \
        snprintf(_n, sizeof(_n), "%s%s", prefix, name);                     \
        w.field = cp_load_2d(gpu, store, _n, rows, cols,                    \
                             error, error_size);                            \
        if (!w.field) goto cp_cleanup;                                      \
    } while (0)
#define CP_L1(name, field, dim) do {                                         \
        char _n[192];                                                       \
        snprintf(_n, sizeof(_n), "%s%s", prefix, name);                     \
        w.field = cp_load_1d(gpu, store, _n, dim, error, error_size);        \
        if (!w.field) goto cp_cleanup;                                      \
    } while (0)
        CP_L1("input_layernorm.weight", input_norm, H);
        CP_L("self_attn.q_proj.weight", query, QD, H);
        CP_L("self_attn.k_proj.weight", key, KVD, H);
        CP_L("self_attn.v_proj.weight", value, KVD, H);
        CP_L1("self_attn.q_norm.weight", query_norm, CP_HEAD_DIM);
        CP_L1("self_attn.k_norm.weight", key_norm, CP_HEAD_DIM);
        CP_L("self_attn.o_proj.weight", attention_output, H, QD);
        CP_L1("post_attention_layernorm.weight", post_norm, H);
        CP_L("mlp.gate_proj.weight", gate, I, H);
        CP_L("mlp.up_proj.weight", up, I, H);
        CP_L("mlp.down_proj.weight", down, H, I);
#undef CP_L1
#undef CP_L
        if (!gpu_operation(gpu, h3_gpu_begin(gpu), error, error_size,
                           "layer begin", layer) ||
            !cp_encode_layer(gpu, &w, tokens, hidden, norm, query, key, value,
                             attention_heads, attention_output, gate, up,
                             mlp_output, rope_cos, rope_sin, H, I, QD, KVD,
                             QH, KVH, error, error_size) ||
            !gpu_operation(gpu, h3_gpu_submit(gpu), error, error_size,
                           "layer submit", layer)) goto cp_cleanup;
        h3_gpu_tensor_free(w.input_norm); h3_gpu_tensor_free(w.query);
        h3_gpu_tensor_free(w.key); h3_gpu_tensor_free(w.value);
        h3_gpu_tensor_free(w.query_norm); h3_gpu_tensor_free(w.key_norm);
        h3_gpu_tensor_free(w.attention_output); h3_gpu_tensor_free(w.post_norm);
        h3_gpu_tensor_free(w.gate); h3_gpu_tensor_free(w.up);
        h3_gpu_tensor_free(w.down);
        memset(&w, 0, sizeof(w));
        if (progress) progress(layer + 1, tap_layers, progress_opaque);
    }

    /* Apply the model's final RMSNorm before tapping. HF / ComfyUI clip the
       normed hidden (hidden_states[tap+1]); without it token 0's residual
       stream is still in its ~4700-magnitude exploded state and every token's
       direction drifts, collapsing the cosine vs the reference to ~0.7. */
    h3_gpu_tensor *final_norm = cp_load_1d(gpu, store,
        "model.language_model.norm.weight", H, error, error_size);
    if (!final_norm) goto cp_cleanup;
    if (!gpu_operation(gpu, h3_gpu_begin(gpu), error, error_size,
                       "final-norm begin", -1) ||
        !gpu_operation(gpu, h3_gpu_rms_norm_bf16(gpu, hidden, hidden,
                         final_norm, tokens, (int)H, CP_RMS_EPS), error,
                       error_size, "final RMSNorm", -1) ||
        !gpu_operation(gpu, h3_gpu_submit(gpu), error, error_size,
                       "final-norm submit", -1)) {
        h3_gpu_tensor_free(final_norm);
        goto cp_cleanup;
    }
    h3_gpu_tensor_free(final_norm);

    /* Read tapped 2560-dim hidden and run ClipProj MLP on CPU. */
    uint16_t *host_hidden = malloc(hidden_count * sizeof(uint16_t));
    if (!host_hidden) { fail(error, error_size, "out of memory (host hidden)");
                       goto cp_cleanup; }
    if (!h3_gpu_tensor_read_bf16(hidden, host_hidden, hidden_count)) {
        fail(error, error_size, "cannot read ClipProj tapped hidden");
        free(host_hidden);
        goto cp_cleanup;
    }
    /* Optional dump of the tapped 2560-dim hidden (pre-MLP) for fidelity
       debugging: writes [seq u32][dim u32][seq*dim float32]. */
    const char *hidden_dump = getenv("H3_CLIPPROJ_DUMP_HIDDEN");
    if (hidden_dump) {
        FILE *hf = fopen(hidden_dump, "wb");
        if (hf) {
            uint32_t seq = (uint32_t)token_count, hd = (uint32_t)H;
            fwrite(&seq, sizeof(seq), 1, hf);
            fwrite(&hd, sizeof(hd), 1, hf);
            for (size_t i = 0; i < hidden_count; i++) {
                float v = cp_bf16_to_f32(host_hidden[i]);
                fwrite(&v, sizeof(v), 1, hf);
            }
            fclose(hf);
        }
    }
    size_t total_out = token_count * CP_OUT_DIM;
    uint16_t *outv = malloc(total_out * sizeof(uint16_t));
    if (!outv) { fail(error, error_size, "out of memory (output)");
                free(host_hidden); goto cp_cleanup; }
    float *xn = malloc(H * sizeof(float));
    float *xm = malloc(CP_MLP_HIDDEN * sizeof(float));
    float *cond = malloc(CP_OUT_DIM * sizeof(float));
    if (!xn || !xm || !cond) {
        fail(error, error_size, "out of memory (mlp buffers)");
        free(host_hidden); free(outv); free(xn); free(xm); free(cond);
        goto cp_cleanup;
    }
    for (size_t tok = 0; tok < token_count; tok++) {
        const uint16_t *row = host_hidden + tok * H;
        for (uint64_t c = 0; c < H; c++)
            xn[c] = (cp_bf16_to_f32(row[c]) - mean_in[c]) / std_in[c];
        for (size_t r = 0; r < CP_MLP_HIDDEN; r++) {
            const float *wr = w0 + r * H;
            float acc = b0[r];
            for (uint64_t c = 0; c < H; c++) acc += xn[c] * wr[c];
            xm[r] = cp_gelu(acc);
        }
        for (size_t r = 0; r < CP_OUT_DIM; r++) {
            const float *wr = w2 + r * CP_MLP_HIDDEN;
            float acc = b2[r];
            for (uint64_t c = 0; c < CP_MLP_HIDDEN; c++) acc += xm[c] * wr[c];
            cond[r] = acc * std_out[r] + mean_out[r];
        }
        if (sink_out && tok == 0)
            for (size_t r = 0; r < CP_OUT_DIM; r++) cond[r] = sink_out[r];
        for (size_t r = 0; r < CP_OUT_DIM; r++)
            outv[tok * CP_OUT_DIM + r] = cp_f32_to_bf16(cond[r]);
    }
    free(xn); free(xm); free(cond); free(host_hidden);

    output->values = outv;
    output->tokens = token_count;
    output->width = CP_OUT_DIM;
    if (h3_gpu_get_stats(gpu, &output->gpu_stats)) rc = 1;

cp_cleanup:
    free(mean_in); free(std_in); free(mean_out); free(std_out);
    free(sink_out); free(w0); free(b0); free(w2); free(b2);
    for (size_t i = 0; i < sizeof(activations)/sizeof(*activations); i++)
        h3_gpu_tensor_free(activations[i]);
    h3_gpu_free(gpu);
    h3_weight_store_free(pstore);
    h3_weight_store_free(store);
    if (!rc && output->values) { free(output->values); output->values = NULL; }
    return rc;
}

int h3_text_encode_bf16(const char *weight_directory,
                        const char *shader_source_path,
                        const uint32_t *token_ids, size_t token_count,
                        h3_text_progress progress, void *progress_opaque,
                        h3_text_embedding *output,
                        char *error, size_t error_size) {
    return text_encode_bf16_impl(
        weight_directory, shader_source_path, token_ids, token_count,
        NULL, 0, NULL, NULL, TEXT_LAYERS, progress, progress_opaque,
        output, error, error_size);
}

int h3_text_encode_multimodal_bf16(
                        const char *weight_directory,
                        const char *shader_source_path,
                        const uint32_t *token_ids, size_t token_count,
                        const h3_text_vision_span *spans, size_t span_count,
                        const uint32_t *position_ids, const uint8_t *tags,
                        h3_text_progress progress, void *progress_opaque,
                        h3_text_embedding *output,
                        char *error, size_t error_size) {
    if (!spans || !span_count || !position_ids || !tags) {
        if (output) memset(output, 0, sizeof(*output));
        fail(error, error_size, "multimodal Qwen presentation is incomplete");
        return 0;
    }
    return text_encode_bf16_impl(
        weight_directory, shader_source_path, token_ids, token_count,
        spans, span_count, position_ids, tags, TEXT_LAYERS,
        progress, progress_opaque,
        output, error, error_size);
}

int h3_text_encode_multimodal_layers_bf16(
                        const char *weight_directory,
                        const char *shader_source_path,
                        const uint32_t *token_ids, size_t token_count,
                        const h3_text_vision_span *spans, size_t span_count,
                        const uint32_t *position_ids, const uint8_t *tags,
                        int layer_count,
                        h3_text_progress progress, void *progress_opaque,
                        h3_text_embedding *output,
                        char *error, size_t error_size) {
    if (!spans || !span_count || !position_ids || !tags) {
        if (output) memset(output, 0, sizeof(*output));
        fail(error, error_size, "multimodal Qwen presentation is incomplete");
        return 0;
    }
    return text_encode_bf16_impl(
        weight_directory, shader_source_path, token_ids, token_count,
        spans, span_count, position_ids, tags, layer_count,
        progress, progress_opaque, output, error, error_size);
}
