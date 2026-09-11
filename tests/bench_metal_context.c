/* Measures the cost of bringing up a Metal context: the runtime compile of
 * h3_shaders.metal plus its compute pipeline states. Run from the repository
 * root, where the default relative shader path resolves:
 *
 *     ./h3_metal_bench               # five contexts
 *     ./h3_metal_bench 8             # eight contexts
 *     H3_PROFILE=1 ./h3_metal_bench  # per-stage breakdown per context
 *
 * One process is the right unit because the engine builds a separate context
 * for the DiT, the Qwen text encoder, and each VAE, so the same shader source
 * is currently compiled once per component.
 */
#include "h3_gpu.h"

#include <stdio.h>
#include <stdlib.h>
#include <time.h>

static double seconds(void) {
    struct timespec value;
    if (clock_gettime(CLOCK_MONOTONIC, &value) != 0) return 0.0;
    return (double)value.tv_sec + (double)value.tv_nsec * 1e-9;
}

int main(int argc, char **argv) {
    const char *shader = "h3_shaders.metal";
    int count = 5;
    if (argc > 1) count = atoi(argv[1]);
    if (argc > 2) shader = argv[2];
    if (count < 1) count = 1;

    char error[512] = "";
    double total = 0.0;
    for (int index = 0; index < count; index++) {
        double start = seconds();
        h3_gpu *gpu = h3_gpu_create(shader, error, sizeof(error));
        double elapsed = seconds() - start;
        if (!gpu) {
            fprintf(stderr, "h3_metal_bench: context %d failed: %s\n",
                    index, error);
            return 1;
        }
        h3_gpu_free(gpu);
        total += elapsed;
        printf("context %2d: %8.3f s\n", index, elapsed);
        fflush(stdout);
    }
    printf("total     : %8.3f s  (%.3f s per context, %d contexts)\n",
           total, total / (double)count, count);
    return 0;
}
