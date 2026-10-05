/* Table test for the inverse memory query: "how long a clip fits this budget".
 *
 * The point of the inverse function is that it must agree with the forward one,
 * so every case here asks the question and then re-checks the answer with
 * h3_dit_plan_bytes() -- a fit it reports has to actually fit, and the next
 * length on the aligned ladder has to not fit. It also pins the answer to the
 * ladder h3_align_frame_count() accepts, so a caller can request what it is
 * told, and pins the ceiling so "at least this much" cannot be read as a limit.
 *
 *   ./h3_memory_plan_inverse_test
 */
#include "h3.h"
#include "h3_dit.h"
#include "h3_host.h"
#include "h3_memory_plan.h"

#include <inttypes.h>
#include <stdio.h>
#include <string.h>

#define GIB (1024ull * 1024ull * 1024ull)

static int failures = 0;

static void check(int condition, const char *what) {
    fprintf(stderr, "%s %s\n", condition ? "  ok  " : "  FAIL", what);
    if (!condition) failures++;
}

static h3_device_info device(uint64_t physical, uint64_t recommended) {
    h3_device_info info;
    memset(&info, 0, sizeof(info));
    info.physical_memory = physical;
    info.recommended_working_set = recommended;
    return info;
}

/* No reference, no keyframe condition, 64 prompt bytes as the token bound. */
static uint64_t plan(int frames, int width, int height) {
    return h3_dit_plan_bytes(width, height, frames, 64, 0, 0);
}

static void expect_exact_inverse(uint64_t budget, int width, int height,
                                 const char *what) {
    int frames = h3_memory_plan_frames_within(budget, width, height, 64, 0, 0);
    int ok = 1;
    if (frames) {
        ok = plan(frames, width, height) <= budget;
        /* The next length on the 22 + 17k ladder must overshoot, unless the
         * search already ran into the ceiling. */
        if (ok && frames < H3_PLAN_FRAMES_CEILING)
            ok = plan(frames + 17, width, height) > budget;
        /* A length that h3_align_frame_count() would rewrite is unrequestable. */
        if (ok) ok = h3_align_frame_count(frames) == frames;
    } else {
        ok = plan(22, width, height) > budget;
    }
    fprintf(stderr, "  ---- %-46s -> %d frames\n", what, frames);
    check(ok, what);
}

int main(void) {
    fprintf(stderr, "the ceiling the plan compares against:\n");
    h3_device_info m4 = device(16 * GIB, 20 * GIB);
    uint64_t expected = ((16 * GIB - 4 * GIB) * 85ull) / 100ull;
    check(h3_memory_plan_budget_bytes(&m4) == expected,
          "16 GiB RAM / 20 GiB recommendation is bound by physical RAM minus the "
          "OS reserve, not by the recommendation");
    h3_device_info none = device(0, 0);
    check(h3_memory_plan_budget_bytes(&none) == 0,
          "no device info yields no ceiling rather than a guess");
    h3_device_info small = device(16 * GIB, 2 * GIB);
    check(h3_memory_plan_budget_bytes(&small) == (2 * GIB * 80ull) / 100ull,
          "a recommendation below the physical bound wins");

    fprintf(stderr, "inverse and forward must agree:\n");
    expect_exact_inverse(plan(22, 256, 256) - 1, 256, 256,
                         "one byte short of a 22-frame chunk fits nothing");
    expect_exact_inverse(plan(22, 256, 256), 256, 256,
                         "exactly a 22-frame chunk returns 22");
    expect_exact_inverse(plan(39, 256, 256) + 1, 256, 256,
                         "one byte past 39 frames still returns 39");
    expect_exact_inverse(8 * GIB, 256, 256, "an 8 GiB budget at 256x256");
    expect_exact_inverse(8 * GIB, 768, 432, "an 8 GiB budget at 768x432");
    check(h3_align_frame_count(H3_PLAN_FRAMES_CEILING) == H3_PLAN_FRAMES_CEILING,
          "the search ceiling is itself on the requestable ladder, so "
          "\"at least this much\" can actually be reached");
    expect_exact_inverse(64 * GIB, 256, 256, "a 64 GiB budget hits the ceiling");

    fprintf(stderr, "monotonicity:\n");
    int short_clip = h3_memory_plan_frames_within(4 * GIB, 256, 256, 64, 0, 0);
    int long_clip = h3_memory_plan_frames_within(12 * GIB, 256, 256, 64, 0, 0);
    check(short_clip <= long_clip, "more bytes never buys a shorter clip");
    int wide = h3_memory_plan_frames_within(8 * GIB, 512, 512, 64, 0, 0);
    int narrow = h3_memory_plan_frames_within(8 * GIB, 256, 256, 64, 0, 0);
    check(wide <= narrow, "a bigger canvas never buys a longer clip");
    int conditions = h3_memory_plan_frames_within(8 * GIB, 256, 256, 64, 2, 0);
    int plain = h3_memory_plan_frames_within(8 * GIB, 256, 256, 64, 0, 0);
    check(conditions <= plain, "two keyframe conditions never lengthen the clip");
    int referenced =
        h3_memory_plan_frames_within(8 * GIB, 256, 256, 64, 0, 3);
    check(referenced <= plain, "ordered references never lengthen the clip");

    if (failures) {
        fprintf(stderr, "FAIL tests/test_memory_plan_inverse.c: %d check(s)\n",
                failures);
        return 1;
    }
    puts("ok: the inverse query answers in lengths the run can actually request");
    return 0;
}
