#include "h3_memory_plan.h"

#include "h3_dit.h"

#include <stdio.h>
#include <string.h>

#define H3_GIB (1024ull * 1024ull * 1024ull)
/* Physical RAM kept for the OS / window server so a greedy long clip cannot
 * starve the system into a panic. */
#define H3_OS_RESERVE_GIB 4ull

uint64_t h3_memory_plan_budget_bytes(const h3_device_info *device) {
    if (!device || device->recommended_working_set == 0) return 0;
    /* Cap the budget by Metal's recommendation *and* by physical RAM minus an
     * OS reserve: on a unified-memory Mac the GPU's wired allocations cannot be
     * swapped, so trusting Metal's optimistic recommendation alone lets a long
     * clip overrun physical RAM and panic the system. */
    uint64_t target = (device->recommended_working_set * 80ull) / 100ull;
    const uint64_t physical = device->physical_memory;
    const uint64_t reserve = H3_OS_RESERVE_GIB * H3_GIB;
    if (physical > reserve) {
        const uint64_t phys_target = ((physical - reserve) * 85ull) / 100ull;
        if (phys_target < target) target = phys_target;
    }
    return target;
}

int h3_memory_plan_frames_within(uint64_t available_bytes, int width, int height,
                                 size_t text_rows_upper_bound,
                                 size_t condition_count,
                                 size_t reference_count) {
    /* Aligned clip lengths are 22 + 17k: h3_align_frame_count() leaves those
     * fixed, and the video VAE is trained on whole 22-frame chunks. Searching
     * that ladder rather than raw frame counts means the answer is a length the
     * caller can actually request. */
    size_t low = 0;
    size_t high = (size_t)((H3_PLAN_FRAMES_CEILING - 22) / 17);
    if (h3_dit_plan_bytes(width, height, 22, text_rows_upper_bound,
                          condition_count, reference_count) > available_bytes)
        return 0;
    while (low < high) {
        size_t middle = low + (high - low + 1) / 2;
        uint64_t needed = h3_dit_plan_bytes(
            width, height, (int)(22 + 17 * middle), text_rows_upper_bound,
            condition_count, reference_count);
        if (needed <= available_bytes) low = middle;
        else high = middle - 1;
    }
    return (int)(22 + 17 * low);
}

/*
 * Decide an automatic memory plan from the device's recommended working set and
 * the model's resident weight footprint.
 *
 * Strategy (adapted from ds4's streaming cache planner):
 *   - total_weight_bytes is the *fully resident* footprint (all DiT blocks,
 *     full VAE decoder, encoders loaded). If it fits within the working set *
 *     0.8, run fully resident and prefer int8 row-FC2 when supported.
 *   - Otherwise streaming is enabled. Crucially, the decision then uses
 *     streamed_resident_bytes (the footprint *after* streaming, not the full
 *     total) so the planner does not chronically overestimate memory: with
 *     streaming the DiT holds only 2 blocks resident and the VAE decoder only
 *     1 block, and the encoders are freed per call. int8 is suggested *in
 *     addition* to streaming (the two are orthogonal, per ds4's decoupled
 *     expert-cache design), not forced off.
 *   - On extreme budgets (< ~4 GiB headroom after streaming) also drop
 *     dit_layers toward H3_MIN_DIT_LAYERS.
 */
int h3_memory_plan_auto(const h3_device_info *device,
                        uint64_t total_weight_bytes,
                        uint64_t streamed_resident_bytes,
                        uint64_t activation_bytes,
                        h3_memory_plan *out) {
    if (!out || !device) return 1;
    memset(out, 0, sizeof(*out));
    if (device->recommended_working_set == 0) {
        snprintf(out->reason, sizeof(out->reason),
                 "no device working-set info; leaving defaults");
        return 0;
    }

    const uint64_t rec = device->recommended_working_set;
    /* One definition of the ceiling, shared with the inverse query in
     * h3_memory_plan_frames_within() so the two cannot drift apart. */
    const uint64_t target = h3_memory_plan_budget_bytes(device);
    const uint64_t steady = total_weight_bytes + activation_bytes;
    const uint64_t steady_streamed =
        streamed_resident_bytes + activation_bytes;

    if (steady <= target) {
        out->ssd_streaming = 0;
        out->use_int8_row_fc2 = 0; /* suggest; caller checks metal4 */
        out->dit_layers = 0;       /* keep default (full) */
        out->video_vae_streaming = 0;
        snprintf(out->reason, sizeof(out->reason),
                 "model %.1f GiB + activations %.1f GiB fit in %.1f GiB "
                 "working set; full resident",
                 (double)total_weight_bytes / H3_GIB,
                 (double)activation_bytes / H3_GIB,
                 (double)target / H3_GIB);
        return 0;
    }

    /* Does not fit resident: enable streaming. Decide using the *streamed*
     * resident footprint, not the full total. */
    out->ssd_streaming = 1;
    out->use_int8_row_fc2 = 1; /* suggest; caller checks metal4 */
    out->video_vae_streaming = 1;

    /* Extreme budget after streaming: also trim DiT depth toward the validated
     * minimum. */
    const uint64_t free_after_stream =
        rec > steady_streamed ? rec - steady_streamed : 0;
    if (free_after_stream < 4ull * H3_GIB) {
        out->dit_layers = H3_MIN_DIT_LAYERS;
        snprintf(out->reason, sizeof(out->reason),
                 "model %.1f GiB exceeds %.1f GiB working set; after streaming "
                 "%.1f GiB stay resident, SSD+VAE streaming on, int8 on, "
                 "DiT layers -> %d",
                 (double)total_weight_bytes / H3_GIB,
                 (double)target / H3_GIB,
                 (double)steady_streamed / H3_GIB, H3_MIN_DIT_LAYERS);
    } else {
        out->dit_layers = 0;
        snprintf(out->reason, sizeof(out->reason),
                 "model %.1f GiB exceeds %.1f GiB working set; after streaming "
                 "%.1f GiB stay resident, SSD+VAE streaming on, int8 on",
                 (double)total_weight_bytes / H3_GIB,
                 (double)target / H3_GIB,
                 (double)steady_streamed / H3_GIB);
    }
    return 0;
}