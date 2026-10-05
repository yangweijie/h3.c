/* Table test for the tier decision in h3_memory_plan_auto(): which knobs the
 * planner flips for a given device and footprint.
 *
 * The discriminating case is a Mac where Metal's recommendation is much larger
 * than the ceiling the plan actually enforces (physical RAM minus the OS
 * reserve, discounted). If the "extreme budget" headroom is measured against the
 * recommendation instead of against that ceiling, it reports "plenty of room" on
 * exactly the machines that need DiT depth trimmed -- so the tight case below
 * asserts its own premise: the two headrooms must land on opposite sides of the
 * 4 GiB threshold, or the case proves nothing.
 *
 *   ./h3_memory_plan_tiers_test
 */
#include "h3.h"
#include "h3_memory_plan.h"

#include <inttypes.h>
#include <stdio.h>
#include <string.h>

#define GIB (1024ull * 1024ull * 1024ull)
#define THRESHOLD_GIB (4ull * GIB)

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

/* Whole-and-half GiB, as the planner's own arithmetic would see them. */
static uint64_t gib(double value) { return (uint64_t)(value * (double)GIB); }

static h3_memory_plan plan(const h3_device_info *device, double total_weight,
                           double streamed_resident, double activation) {
    h3_memory_plan out;
    memset(&out, 0, sizeof(out));
    int rc = h3_memory_plan_auto(device, gib(total_weight),
                                gib(streamed_resident), gib(activation), &out);
    if (rc != 0) fprintf(stderr, "  (h3_memory_plan_auto returned %d)\n", rc);
    return out;
}

int main(void) {
    /* 16 GiB of RAM under a 20 GiB recommendation: the physical bound is what
     * bites, and it is the ordinary small-Mac case. */
    h3_device_info m4 = device(16 * GIB, 20 * GIB);
    uint64_t target = h3_memory_plan_budget_bytes(&m4);
    fprintf(stderr, "clamped ceiling: %" PRIu64 " GiB-sized bytes (16 GiB RAM, "
                    "20 GiB recommendation)\n", target);
    check(target < m4.recommended_working_set,
          "this device's ceiling is below its recommendation, so the two can "
          "disagree about headroom");

    fprintf(stderr, "the extreme-budget headroom is measured against the ceiling:\n");
    /* 6.5 GiB stays resident after streaming: 13.5 GiB of "room" by the
     * recommendation, 3.7 GiB by the ceiling the plan enforces. */
    const double streamed = 6.0, activation = 0.5;
    uint64_t steady_streamed = gib(streamed + activation);
    uint64_t room_by_rec = m4.recommended_working_set > steady_streamed
        ? m4.recommended_working_set - steady_streamed : 0;
    uint64_t room_by_target = target > steady_streamed ? target - steady_streamed : 0;
    check(room_by_rec >= THRESHOLD_GIB && room_by_target < THRESHOLD_GIB,
          "the tight case really does straddle the 4 GiB threshold (otherwise it "
          "cannot tell the two rules apart)");
    h3_memory_plan tight = plan(&m4, 30.0, streamed, activation);
    check(tight.ssd_streaming == 1 && tight.use_int8_row_fc2 == 1 &&
              tight.video_vae_streaming == 1,
          "a model that will not fit resident turns on streaming and int8");
    check(tight.dit_layers == H3_MIN_DIT_LAYERS,
          "tight against the enforced ceiling trims DiT depth (this is what "
          "measuring against the raw recommendation used to miss)");

    /* 5.5 GiB resident: 4.7 GiB of room under the same ceiling, so depth stays. */
    h3_memory_plan loose = plan(&m4, 30.0, 5.0, activation);
    check(loose.ssd_streaming == 1,
          "the loose case still takes the streaming branch");
    check(loose.dit_layers == 0,
          "4.7 GiB of room under the ceiling keeps the full DiT depth");

    fprintf(stderr, "the other branches are unchanged:\n");
    /* 8.5 GiB total fits the 10.2 GiB ceiling: full resident, nothing streamed. */
    h3_memory_plan resident = plan(&m4, 8.0, 5.0, activation);
    check(resident.ssd_streaming == 0 && resident.use_int8_row_fc2 == 0 &&
              resident.video_vae_streaming == 0 && resident.dit_layers == 0,
          "what fits the ceiling runs fully resident with no knobs touched");
    check(strstr(resident.reason, "full resident") != NULL,
          "the rationale says so");

    h3_device_info unknown = device(16 * GIB, 0);
    h3_memory_plan left = plan(&unknown, 30.0, 5.0, activation);
    check(left.ssd_streaming == 0 && left.dit_layers == 0,
          "no working-set info leaves the defaults alone");
    check(strstr(left.reason, "no device working-set info") != NULL,
          "and names why");

    fprintf(stderr, "the printed ceiling is the one the decision used:\n");
    /* The rationale quotes the ceiling; pin it to the shared budget function so a
     * future edit cannot print one number and compare against another. */
    char budget_text[32];
    snprintf(budget_text, sizeof(budget_text), "%.1f GiB working set",
             (double)target / (double)GIB);
    check(strstr(tight.reason, budget_text) != NULL,
          "the tight rationale quotes h3_memory_plan_budget_bytes()'s number");
    check(strstr(loose.reason, budget_text) != NULL,
          "so does the loose one");

    /* A recommendation smaller than the physical bound wins, in that direction
     * too: 2 GiB recommended, 1.6 GiB ceiling, 1.0 GiB resident => 0.6 GiB room. */
    h3_device_info tiny = device(16 * GIB, 2 * GIB);
    h3_memory_plan small = plan(&tiny, 8.0, 0.5, activation);
    check(small.ssd_streaming == 1 && small.dit_layers == H3_MIN_DIT_LAYERS,
          "a recommendation below the physical bound still trims depth");

    if (failures) {
        fprintf(stderr, "FAIL tests/test_memory_plan_tiers.c: %d check(s)\n",
                failures);
        return 1;
    }
    puts("ok: tier decisions follow the ceiling the plan enforces");
    return 0;
}
