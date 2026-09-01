#ifndef H3_TEXT_ENCODER_H
#define H3_TEXT_ENCODER_H

#include "h3_gpu.h"

#include <stddef.h>
#include <stdint.h>

#define H3_TEXT_HIDDEN_SIZE 5120u

typedef struct {
    size_t tokens;
    size_t width;
    uint16_t *values;
    h3_gpu_stats gpu_stats;
    /* Optional DiT modality tag per presentation row: 1 for language and 0
     * for the Qwen vision span including its boundary tokens. */
    uint8_t *tags;
} h3_text_embedding;

typedef void (*h3_text_progress)(int completed_layers, int total_layers,
                                 void *opaque);

typedef struct {
    size_t start;
    size_t tokens;
    const uint16_t *embeddings;
    const uint16_t *deepstack[3];
} h3_text_vision_span;

/* Run the released first 50 Qwen3-VL language layers. The caller owns the
 * returned BF16 values and releases them with h3_text_embedding_free(). */
int h3_text_encode_bf16(const char *weight_directory,
                        const char *shader_source_path,
                        const uint32_t *token_ids, size_t token_count,
                        h3_text_progress progress, void *progress_opaque,
                        h3_text_embedding *output,
                        char *error, size_t error_size);

/* Run the same 50 decoder layers with Qwen3-VL presentation spans. The base
 * token embedding at every span is replaced by vision embeddings; deepstack
 * rows are added after language layers 0, 1, and 2. position_ids is axis-major
 * [3,tokens], and tags carries the DiT modality tag for every text row. */
int h3_text_encode_multimodal_bf16(
                        const char *weight_directory,
                        const char *shader_source_path,
                        const uint32_t *token_ids, size_t token_count,
                        const h3_text_vision_span *spans, size_t span_count,
                        const uint32_t *position_ids, const uint8_t *tags,
                        h3_text_progress progress, void *progress_opaque,
                        h3_text_embedding *output,
                        char *error, size_t error_size);

/* Prefix form used by parity tooling to localize a multimodal mismatch. */
int h3_text_encode_multimodal_layers_bf16(
                        const char *weight_directory,
                        const char *shader_source_path,
                        const uint32_t *token_ids, size_t token_count,
                        const h3_text_vision_span *spans, size_t span_count,
                        const uint32_t *position_ids, const uint8_t *tags,
                        int layer_count,
                        h3_text_progress progress, void *progress_opaque,
                        h3_text_embedding *output,
                        char *error, size_t error_size);
void h3_text_embedding_free(h3_text_embedding *embedding);

/* ClipProj path: run Qwen3-VL-4B (truncated to the tapped layer) as the text
 * encoder and lift its 2560-dim tapped hidden to the 5120-dim H3 conditioning
 * space with the ClipProj MLP (mean/std normalization + 2-layer projection +
 * attention-sink replacement). Produces an h3_text_embedding fully compatible
 * with the DiT (width == TEXT_DIM == 5120), so the DiT consumes it unchanged.
 *
 * This is the in-engine equivalent of clipproj_harness.py: it removes the need
 * to read the ~62 GiB 50-layer encoder, replacing it with a ~5.5 GiB 4B model
 * plus a tiny projection. Enable by pointing H3_CLIPPROJ_DIR at a BF16
 * Qwen3-VL-4B-Instruct directory (H3_CLIPPROJ_PROJ overrides the projection dir). */
int h3_text_encode_clipproj_bf16(
                        const char *qwen4b_directory,
                        const char *projection_directory,
                        const char *shader_source_path,
                        const uint32_t *token_ids, size_t token_count,
                        h3_text_progress progress, void *progress_opaque,
                        h3_text_embedding *output,
                        char *error, size_t error_size);

#endif
