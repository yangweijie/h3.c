/* Table test for the ClipProj encoder-selection rule (F44 §1.2).
 *
 * `h3.c` defines the rule privately, so this test includes it and drives the
 * pure classifier -- the whole point of F44 §1.2 was that the load path, the
 * generate path and the conditioning cache key each re-derived the same
 * condition by hand and got three different answers. Only the classifier is
 * tested here; the two callers are one line each and are covered by the
 * resolver being the only remaining reader of the environment.
 *
 * The hazard this pins: with `H3_CLIPPROJ_DIR` unset, the generate path used to
 * fall back to a built-in ClipProj directory while the disk key recorded
 * `clipmodel=none`, which means "50-layer encoder" to every reader. One
 * encoder's conditioning could be handed back as the other's.
 *
 *   cc -std=c11 -I. tests/test_clipproj_selection.c h3.c \
 *      -o /tmp/t -undefined dynamic_lookup && /tmp/t
 */
#include "h3.c"

#include <stdio.h>
#include <string.h>

static int failures = 0;

static void expect_mode(const char *directory, const char *projection,
                        h3_clipproj_mode want, const char *what) {
    const char *out_directory = NULL;
    const char *out_projection = NULL;
    h3_clipproj_mode got = h3_clipproj_classify(directory, projection,
                                                &out_directory,
                                                &out_projection);
    int ok = got == want;
    /* Only the ClipProj mode may hand back both directories. */
    if (ok && want == H3_CLIPPROJ_CLIPPROJ)
        ok = out_directory == directory && out_projection == projection;
    else if (ok)
        ok = !out_directory && !out_projection;
    fprintf(stderr, "%s %s\n", ok ? "  ok  " : "  FAIL", what);
    if (!ok) {
        fprintf(stderr, "       got mode=%d directory=%s projection=%s\n", got,
                out_directory ? out_directory : "(null)",
                out_projection ? out_projection : "(null)");
        failures++;
    }
}

/* The identity a disk key records for a given environment, through the same
 * resolver the generate path reads. */
static void expect_disk_identity(const char *directory, const char *projection,
                                 const char *want_directory,
                                 const char *want_projection, const char *what) {
    if (directory) setenv("H3_CLIPPROJ_DIR", directory, 1);
    else unsetenv("H3_CLIPPROJ_DIR");
    if (projection) setenv("H3_CLIPPROJ_PROJ", projection, 1);
    else unsetenv("H3_CLIPPROJ_PROJ");
    const char *recorded_directory = NULL;
    const char *recorded_projection = NULL;
    h3_clipproj_disk_identity(&recorded_directory, &recorded_projection);
    int ok = strcmp(recorded_directory, want_directory) == 0 &&
        strcmp(recorded_projection, want_projection) == 0;
    fprintf(stderr, "%s %s\n", ok ? "  ok  " : "  FAIL", what);
    if (!ok) {
        fprintf(stderr, "       recorded %s / %s\n", recorded_directory,
                recorded_projection);
        failures++;
    }
    unsetenv("H3_CLIPPROJ_DIR");
    unsetenv("H3_CLIPPROJ_PROJ");
}

int main(void) {
    unsetenv("H3_CLIPPROJ_DIR");
    unsetenv("H3_CLIPPROJ_PROJ");

    fprintf(stderr, "the four ways the environment can read:\n");
    expect_mode(NULL, NULL, H3_CLIPPROJ_50_LAYER,
                "unset directory: the 50-layer encoder, no built-in fallback");
    expect_mode("", NULL, H3_CLIPPROJ_50_LAYER,
                "empty directory is not a request for ClipProj");
    expect_mode("0", "/proj", H3_CLIPPROJ_50_LAYER,
                "=0 falls back even with a projection set");
    expect_mode("off", "/proj", H3_CLIPPROJ_50_LAYER,
                "=off falls back even with a projection set");
    expect_mode("/qwen4b", "/proj", H3_CLIPPROJ_CLIPPROJ,
                "both set: ClipProj with exactly those directories");

    fprintf(stderr, "the half-configured request must refuse, not guess:\n");
    expect_mode("/qwen4b", NULL, H3_CLIPPROJ_INCOMPLETE,
                "directory without a projection");
    expect_mode("/qwen4b", "", H3_CLIPPROJ_INCOMPLETE,
                "directory with an empty projection");

    fprintf(stderr, "what the conditioning cache key records:\n");
    expect_disk_identity(NULL, NULL, "none", "none",
                         "unset keys as the 50-layer encoder");
    expect_disk_identity("/qwen4b", "/proj", "/qwen4b", "/proj",
                         "a full request keys as that exact encoder pair");
    expect_disk_identity("/qwen4b", NULL, "none", "none",
                         "a half request keys as the fallback it will not run");

    if (failures) {
        fprintf(stderr, "FAIL tests/test_clipproj_selection.c: %d check(s)\n",
                failures);
        return 1;
    }
    puts("ok: one ClipProj rule, and the cache key agrees with the encoder used");
    return 0;
}
