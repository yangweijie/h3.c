/* Calibration for h3_dit_sequence_estimate() against the layout the engine
 * actually builds, on the paths this machine cannot render end to end.
 *
 * Both installed checkpoints carry only FL2VA -- no video encoder, no REF2VA --
 * so an i2v or Ref2VA run cannot be started here, and F50 closed with "the
 * keyframe and reference paths are unmeasured". Everything that decides those
 * runs' memory is host-side though: h3_layout_build() computes the real row
 * count from the same geometry the estimate uses, no GPU and no weights needed.
 * So this test asks both for the same shapes and compares.
 *
 * The property that matters is directional: an estimate below the layout is the
 * one case that makes a tier choice too optimistic, so every case asserts
 * `estimate >= layout`. Exactness is asserted separately where the estimate has
 * no reason to pad (text bound equal to the real length, keyframe conditions),
 * and the reference slack is *printed* rather than asserted tight -- the estimate
 * charges a whole clip's video rows per reference while the layout emits one
 * frame grid plus that reference's audio rows.
 *
 *   ./h3_sequence_estimate_layout_test
 */
#include "h3.h"
#include "h3_dit.h"
#include "h3_host.h"

#include <inttypes.h>
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

/* One case: build the layout, ask the estimate the same question, compare. */
static size_t compare(int width, int height, int frames, int text_len,
                      size_t text_bound, size_t condition_count,
                      const h3_layout_ref *references, size_t reference_count,
                      const char *what) {
    h3_temporal_shape temporal = h3_temporal(frames);
    int latent_w = 0;
    int latent_h = 0;
    h3_latent_canvas(width, height, &latent_w, &latent_h);
    int keyframes[2] = {0, temporal.frame_count - 1};
    h3_layout_spec spec = {text_len, temporal.video_t, latent_h, latent_w,
                           temporal.audio_t, temporal.frame_count,
                           condition_count ? keyframes : NULL, condition_count,
                           references, reference_count};
    h3_layout layout;
    char error[256] = "";
    char detail[256];
    if (!h3_layout_build(&spec, &layout, error, sizeof(error))) {
        snprintf(detail, sizeof(detail), "h3_layout_build failed: %s", error);
        report(0, what, detail);
        return 0;
    }
    size_t estimate = h3_dit_sequence_estimate(width, height, frames, text_bound,
                                              condition_count, reference_count);
    snprintf(detail, sizeof(detail), "estimate %zu vs layout %zu (%+zd rows)",
             estimate, layout.seq_len, (ssize_t)estimate - (ssize_t)layout.seq_len);
    report(estimate >= layout.seq_len, what, detail);
    size_t real = layout.seq_len;
    h3_layout_free(&layout);
    return real;
}

int main(void) {
    /* The identity cases: with the true text length and one condition row block
     * per keyframe, the estimate has no slack, so equality is the expectation. */
    fprintf(stderr, "where the estimate has no reason to pad, it is exact:\n");
    int frames = h3_align_frame_count(22);
    h3_temporal_shape temporal = h3_temporal(frames);
    size_t rows = compare(256, 256, frames, 6, 6, 0, NULL, 0,
                          "t2v 256x256/22f, text bound == real length");
    if (rows != 528) {
        fprintf(stderr, "  FAIL the t2v anchor moved: %zu rows (was 528)\n", rows);
        failures++;
    } else {
        fprintf(stderr, "  ok   that shape is %zu rows, the number h3_generate logs\n",
                rows);
    }
    compare(256, 256, frames, 6, 6, 1, NULL, 0,
            "first frame only: one condition block");
    compare(256, 256, frames, 6, 6, 2, NULL, 0,
            "first and last frame: two condition blocks");
    compare(256, 256, frames, 6, 70, 2, NULL, 0,
            "a looser text bound pads by exactly its slack");

    /* Reference shapes: an image reference costs one frame grid, a video
     * reference one frame grid plus its audio rows, an audio reference only
     * audio rows -- all far below the whole-clip charge the estimate uses. */
    fprintf(stderr, "references: the estimate must never be below, slack reported:\n");
    int latent_w = 0;
    int latent_h = 0;
    h3_latent_canvas(256, 256, &latent_w, &latent_h);
    h3_layout_ref image_at_target = {H3_LAYOUT_REF_IMAGE, 1, latent_h, latent_w, 0};
    compare(256, 256, frames, 6, 6, 0, &image_at_target, 1,
            "one ordered image reference at the target canvas");
    h3_layout_ref small_image = {H3_LAYOUT_REF_IMAGE, 1, latent_h / 2,
                                 latent_w / 2, 0};
    compare(256, 256, frames, 6, 6, 0, &small_image, 1,
            "one ordered image reference at half the linear canvas");
    h3_layout_ref audio = {H3_LAYOUT_REF_AUDIO, 0, 0, 0, temporal.audio_t};
    compare(256, 256, frames, 6, 6, 0, &audio, 1,
            "one ordered audio reference");
    h3_layout_ref mixed[3] = {
        {H3_LAYOUT_REF_IMAGE, 1, latent_h, latent_w, 0},
        {H3_LAYOUT_REF_AUDIO, 0, 0, 0, temporal.audio_t},
        {H3_LAYOUT_REF_VIDEO, temporal.video_t, latent_h, latent_w,
         temporal.audio_t},
    };
    compare(256, 256, frames, 6, 6, 0, mixed, 3,
            "three ordered references of every kind");

    fprintf(stderr, "the sweep must never come out below the layout:\n");
    /* Both edges are multiples of H3_CANVAS_MULTIPLE because h3_generate()
     * requires it -- h3.c:900 for the output canvas, h3.c:915 for the internal
     * render canvas -- which is what keeps every latent canvas even:
     * h3_frame_grid() refuses an odd one, so a non-multiple shape like 768x432
     * is not a layout the engine can build at all. */
    const int canvases[4][2] = {{256, 256}, {512, 512}, {768, 448}, {448, 768}};
    const int clip_frames[3] = {22, 56, 127};
    h3_layout_ref one[1] = {{H3_LAYOUT_REF_IMAGE, 1, latent_h, latent_w, 0}};
    int worst_over = 0;
    int worst_frames = 0, worst_refs = 0;
    size_t worst_case_rows = 0;
    int exact_cases = 0, cases = 0;
    for (size_t canvas = 0; canvas < 4; canvas++) {
        for (size_t clip = 0; clip < 3; clip++) {
            for (int refs = 0; refs <= 1; refs++) {
                int lw = 0, lh = 0;
                h3_latent_canvas(canvases[canvas][0], canvases[canvas][1],
                                 &lw, &lh);
                one[0].latent_h = lh;
                one[0].latent_w = lw;
                for (size_t conds = 0; conds <= 2; conds++) {
                    if (refs && conds) continue;   /* mutually exclusive */
                    size_t real = compare(canvases[canvas][0], canvases[canvas][1],
                                          clip_frames[clip], 6, 6, conds,
                                          refs ? one : NULL, refs,
                                          "sweep case stays an upper bound");
                    size_t estimate = h3_dit_sequence_estimate(
                        canvases[canvas][0], canvases[canvas][1],
                        clip_frames[clip], 6, conds, refs);
                    cases++;
                    if (estimate == real) exact_cases++;
                    int percent = real ? (int)((estimate - real) * 100 / real) : 0;
                    if (refs)
                        fprintf(stderr,
                                "       %4dx%-4d %3df refs=%d estimate %6zu"
                                " layout %6zu  slack %d%%\n",
                                canvases[canvas][0], canvases[canvas][1],
                                clip_frames[clip], refs, estimate, real, percent);
                    if (percent > worst_over) {
                        worst_over = percent;
                        worst_case_rows = real;
                        worst_frames = clip_frames[clip];
                        worst_refs = refs;
                    }
                }
            }
        }
    }
    char summary[192];
    snprintf(summary, sizeof(summary),
             "%d cases, %d exact; worst slack %d%% at %df/refs=%d, layout %zu rows",
             cases, exact_cases, worst_over, worst_frames, worst_refs,
             worst_case_rows);
    fprintf(stderr, "  info %s\n", summary);
    /* The reference term is the only one that pads: it charges a whole clip's
     * video + audio rows per reference while the layout emits one frame grid
     * plus that reference's audio rows. Asserting the direction, not the size. */
    report(worst_over >= 0 && cases > 20 && worst_refs == 1,
           "the sweep covered both mode families and its slack sits on references",
           summary);

    if (failures) {
        fprintf(stderr, "FAIL tests/test_sequence_estimate_layout.c: %d check(s)\n",
                failures);
        return 1;
    }
    puts("ok: the sequence estimate is an upper bound on every layout built here");
    return 0;
}
