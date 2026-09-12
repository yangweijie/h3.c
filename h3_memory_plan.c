#include "h3_memory_plan.h"

#include <stdio.h>
#include <string.h>

#define H3_GIB (1024ull * 1024ull * 1024ull)

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
    const uint64_t target = (rec * 80ull) / 100ull; /* 80% headroom cap */
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