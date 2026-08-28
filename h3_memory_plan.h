#ifndef H3_MEMORY_PLAN_H
#define H3_MEMORY_PLAN_H

#include <stdint.h>
#include "h3.h"

/*
 * Automatic memory-tier planner.
 *
 * Ported in spirit from ds4/DwarfStar's SSD-streaming cache planner
 * (ds4_ssd_auto_cache_plan / ds4_streaming_manual_cache_safe_bytes): given the
 * device's recommended working set and the model's resident weight footprint,
 * decide whether to enable SSD streaming and/or int8, and how large a streaming
 * cache budget to reserve. This lets small-RAM Macs (16/24 GB) run MiniMax-H3
 * without manual --ssd-streaming / --int8 tuning.
 *
 * The plan is advisory: callers may override any field by setting the matching
 * h3_params entry explicitly before generate().
 */

typedef struct {
    /* Whether to stream DiT transformer blocks from disk (keeps only 2 BF16
     * blocks resident at a time). No longer mutually exclusive with int8: the
     * two are orthogonal (streaming=where weights live, int8=how they're
     * compressed), mirroring ds4's decoupled routing/expert-cache design. */
    int ssd_streaming;
    /* Use int8 row-FC2 quantization to halve resident DiT/MLP weights. May be
     * suggested together with ssd_streaming (decoupled from streaming). */
    int use_int8_row_fc2;
    /* Number of active DiT blocks; may be lowered toward H3_MIN_DIT_LAYERS on
     * very tight budgets. 0 means "leave at h3_params default". */
    int dit_layers;
    /* Stream the video VAE decoder weights (currently resident-only; this is
     * the single largest fixed footprint driver and the main reason 32 GB is
     * the practical floor). Suggested when budget is tight. */
    int video_vae_streaming;
    /* Stream the text/image encoder (Qwen3-VL first 50 layers) once condition
     * building is done; freed before denoise. Suggested on tight budgets. */
    int encoder_streaming;
    /* Bytes to reserve for the streaming weight cache (resident hot set). */
    uint64_t cache_budget_bytes;
    /* Plain-text rationale for logging / --verbose. */
    char reason[256];
} h3_memory_plan;

/*
 * total_weight_bytes:         sum of h3_component_info.bytes across text_encoder,
 *                             fl2va_transformer, ref2va_transformer, video_vae,
 *                             audio_vae (BF16 resident footprint if nothing streamed).
 * streamed_resident_bytes:    the footprint *after* streaming is applied (DiT
 *                             keeps ~2 blocks, VAE decoder ~1 block, encoders are
 *                             freed per call). Used for the actual fit decision so
 *                             the planner does not chronically overestimate memory.
 * activation_bytes:           rough peak activation buffer (latents, VAE tiles, PCM).
 * device:                     probed h3_device_info.
 * Returns 0 on success.
 */
int h3_memory_plan_auto(const h3_device_info *device,
                        uint64_t total_weight_bytes,
                        uint64_t streamed_resident_bytes,
                        uint64_t activation_bytes,
                        h3_memory_plan *out);

/* Reserve budget for a streaming cache: recommended_working_set * 7/8,
 * minus the steady-state model + activation footprint. Mirrors ds4's
 * ds4_streaming_manual_cache_safe_bytes. Returns a GiB-aligned byte count
 * (at least 1 GiB when positive). */
uint64_t h3_memory_cache_budget_bytes(const h3_device_info *device,
                                      uint64_t steady_state_bytes);

#endif /* H3_MEMORY_PLAN_H */
