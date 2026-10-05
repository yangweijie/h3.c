#ifndef H3_DIT_SCHEDULE_H
#define H3_DIT_SCHEDULE_H

#include "h3_gpu.h"
#include "h3_host.h"
#include "h3_weights.h"

#include <stddef.h>
#include <stdint.h>

#define H3_DIT_BLOCKS 50u
#define H3_DIT_HIDDEN 5376u
#define H3_DIT_TIME_DIM 2688u
#define H3_DIT_MODALITIES 3u
#define H3_DIT_ADALN_SLOTS 6u

typedef struct h3_dit_schedule h3_dit_schedule;

typedef struct h3_lora h3_lora;

/* --- Precomputed AdaLN modulation cache ---------------------------------
 * The AdaLN modulation is a pure function of the sigma schedule, and
 * `h3_dit_schedule_precompute` materializes it for every step before denoising
 * starts, so the 51 `adaln_proj.linear` matrices (24.3 GiB on the released
 * architecture, which is the single largest tensor group in the checkpoint) are
 * read exactly once and then freed. A checkpoint that ships the modulation
 * itself can therefore drop them entirely.
 *
 * One cache per step count, keyed by `--steps` so several can coexist:
 *
 *   adaln_cache_times_s{steps}       F32  [rows]
 *   blocks.N.adaln_cache_s{steps}    BF16 [rows, 96768]
 *   final_layer.adaln_cache_s{steps} BF16 [rows, 10752]
 *   adaln_cache_meta_s{steps}        U32  [1]   optional, see below
 *
 * `adaln_cache_times_s{steps}` pins the schedule the cache was built from, and
 * the loader requires this run's row values to be a **prefix** of it. The
 * prefix rule is what lets one cache serve several conditioning modes: the
 * condition rows are appended after the step rows, in the order visual then
 * audio, so a run without a first frame -- or with only a first frame -- uses
 * exactly the leading rows of a reference-image run. The exporter therefore
 * materializes both condition rows, which makes a single cache cover
 * text-to-video (2*steps - 1 rows), first/last frame (2*steps) and reference
 * image (2*steps + 1). A cache that is too short, or whose values disagree, is
 * refused rather than silently misapplied.
 *
 * The row values depend only on `--steps` and the conditioning mode, not on
 * resolution, frame count or seed, so one cache covers every run at that step
 * count. Build one with H3_DIT_ADALN_CACHE_DUMP and
 * fastvideo_qad/scripts/export_h3_adaln_cache.py.
 *
 * LoRA adapters merge into the AdaLN weights a cached checkpoint no longer
 * carries, so an adapter that targets AdaLN cannot be used here. Which adapters
 * do is decided by `adaln_cache_meta_s{steps}`, an optional U32 [1] holding the
 * AdaLN input width of the checkpoint the cache was exported from: with it, an
 * adapter that carries no AdaLN factor (an attention-only adapter, for instance)
 * loads normally and only the conflicting ones are refused. Without it the width
 * is unknown -- a pruned checkpoint shrinks that input, so the loader cannot
 * tell which adapters would have touched AdaLN -- and every adapter is refused.
 * Caches exported before that key existed behave exactly as before. */
#define H3_ADALN_CACHE_TIMES_FORMAT "adaln_cache_times_s%d"
#define H3_ADALN_CACHE_BLOCK_FORMAT "blocks.%u.adaln_cache_s%d"
#define H3_ADALN_CACHE_FINAL_FORMAT "final_layer.adaln_cache_s%d"
#define H3_ADALN_CACHE_META_FORMAT "adaln_cache_meta_s%d"
#define H3_ADALN_CACHE_NAME_MAX 64

/* --- On-disk contract of the H3_DIT_ADALN_CACHE_DUMP blob -----------------
 * Written by dump_adaln_cache() in h3_dit_schedule.c, read by
 * fastvideo_qad/scripts/export_h3_adaln_cache.py. Both sides must agree on
 * every number here; tests/test_adaln_cache_codec.py compiles a probe against
 * these macros and pins the Python module's copy against it, so drift fails at
 * `make test` instead of surfacing as a wrong render N hours later.
 *
 *   magic   char[8]  "H3ADALN2"
 *   fields  uint32[6] little-endian, in order:
 *           steps, time_rows, blocks, block_output, final_output, reserved
 *   times   float[time_rows]
 *   blocks  uint16[blocks][time_rows * block_output]   BF16
 *   final   uint16[time_rows * final_output]           BF16
 *
 * `rows` below is the time_rows the export run produced, which for a cache
 * covering all three conditioning modes is 2*steps + 1. */
#define H3_ADALN_CACHE_MAGIC "H3ADALN2"
#define H3_ADALN_CACHE_MAGIC_BYTES 8
#define H3_ADALN_CACHE_HEADER_FIELDS 6
#define H3_ADALN_CACHE_HEADER_BYTES \
    (H3_ADALN_CACHE_MAGIC_BYTES + H3_ADALN_CACHE_HEADER_FIELDS * 4)
#define H3_ADALN_CACHE_FIELD_STEPS 0
#define H3_ADALN_CACHE_FIELD_TIME_ROWS 1
#define H3_ADALN_CACHE_FIELD_BLOCKS 2
#define H3_ADALN_CACHE_FIELD_BLOCK_OUTPUT 3
#define H3_ADALN_CACHE_FIELD_FINAL_OUTPUT 4
#define H3_ADALN_CACHE_FIELD_RESERVED 5
#define H3_ADALN_CACHE_BLOCK_ROWS H3_DIT_BLOCKS
#define H3_ADALN_CACHE_BLOCK_OUTPUT \
    (H3_DIT_MODALITIES * H3_DIT_ADALN_SLOTS * H3_DIT_HIDDEN)
#define H3_ADALN_CACHE_FINAL_OUTPUT (2 * H3_DIT_HIDDEN)
#define H3_ADALN_CACHE_DUMP_BYTES(rows) \
    ((size_t)H3_ADALN_CACHE_HEADER_BYTES + (size_t)(rows) * 4u + \
     (size_t)H3_ADALN_CACHE_BLOCK_ROWS * (size_t)(rows) * \
         (size_t)H3_ADALN_CACHE_BLOCK_OUTPUT * 2u + \
     (size_t)(rows) * (size_t)H3_ADALN_CACHE_FINAL_OUTPUT * 2u)

typedef void (*h3_dit_schedule_progress)(int completed_blocks,
                                         int total_blocks, void *opaque);

/* Materialize every per-step AdaLN value. This intentionally submits one
 * projection at a time, so a 498 MiB block projection is released before the
 * next is loaded. `loras` (optional, count up to 4) merge into the
 * blocks.N.adaln_proj.linear / final norm_out.linear weights before the
 * per-step projections are computed; adapters whose rank/input width does not
 * match the checkpoint (e.g. a diffusers-shaped turbo LoRA over a pruned
 * ConvRot base) are skipped with a warning. */
h3_dit_schedule *h3_dit_schedule_precompute(
    const h3_weight_store *weights, h3_gpu *gpu,
    const h3_sigma_schedule *sigmas, int visual_condition,
    int audio_condition, h3_lora **loras, int lora_count,
    h3_dit_schedule_progress progress, void *progress_opaque,
    char *error, size_t error_size);
void h3_dit_schedule_free(h3_dit_schedule *schedule);

int h3_dit_schedule_steps(const h3_dit_schedule *schedule);
uint32_t h3_dit_schedule_time_rows(const h3_dit_schedule *schedule);
uint32_t h3_dit_schedule_video_row(const h3_dit_schedule *schedule, int step);
uint32_t h3_dit_schedule_audio_row(const h3_dit_schedule *schedule, int step);
uint32_t h3_dit_schedule_visual_condition_row(
    const h3_dit_schedule *schedule, int step);
uint32_t h3_dit_schedule_audio_condition_row(
    const h3_dit_schedule *schedule, int step);
const h3_gpu_tensor *h3_dit_schedule_block(const h3_dit_schedule *schedule,
                                           unsigned block);
double h3_dit_schedule_gate_score(const h3_dit_schedule *schedule,
                                  unsigned block);
/* Batched variant: fills out[0..count-1] with the gate scores of blocks
 * first..first+count-1 while reusing a single readback buffer, so ranking every
 * block costs one allocation instead of one per block. Returns 0 on any failure
 * (out is then only partially written). */
int h3_dit_schedule_gate_scores(const h3_dit_schedule *schedule,
                                unsigned first, unsigned count, double *out);
void h3_dit_schedule_prune(h3_dit_schedule *schedule,
                           const uint8_t *active_blocks, size_t count);
const h3_gpu_tensor *h3_dit_schedule_final(const h3_dit_schedule *schedule);

/* Build the row map consumed by the fused AdaLN/gate kernels. text_tags may be
 * NULL (all tag 1), or one tag per text row. Qwen vision presentation spans use
 * tag 0. Segment kinds select target/condition timesteps and modality tags. */
int h3_dit_schedule_row_map(const h3_dit_schedule *schedule, int step,
                            const h3_layout *layout,
                            const uint8_t *text_tags, size_t text_tag_count,
                            uint32_t *rows, size_t row_count);

#endif
