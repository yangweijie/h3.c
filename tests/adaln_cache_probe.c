/* Probe for the AdaLN cache on-disk contract, compiled by
 * tests/test_adaln_cache_codec.py against the same header the loader reads.
 *
 * The point is that every number below comes from the macros
 * `h3_dit_schedule_precompute` and `dump_adaln_cache` actually use, so the
 * Python exporter's hand-copied constants are pinned to this file rather than
 * to a second reading of the same source.
 *
 *   print              contract constants and the dump size at 1/3/9/17/41 rows
 *   parse <path>       decode a dump header, positionally, as the loader does
 *   keys <steps>       the tensor names the loader looks up for that step count
 */
#include "h3_dit_schedule.h"

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static void print_contract(void) {
    printf("magic=%s\n", H3_ADALN_CACHE_MAGIC);
    printf("magic_bytes=%u\n", (unsigned)H3_ADALN_CACHE_MAGIC_BYTES);
    printf("header_fields=%u\n", (unsigned)H3_ADALN_CACHE_HEADER_FIELDS);
    printf("header_bytes=%u\n", (unsigned)H3_ADALN_CACHE_HEADER_BYTES);
    printf("blocks=%u\n", (unsigned)H3_ADALN_CACHE_BLOCK_ROWS);
    printf("block_output=%u\n", (unsigned)H3_ADALN_CACHE_BLOCK_OUTPUT);
    printf("final_output=%u\n", (unsigned)H3_ADALN_CACHE_FINAL_OUTPUT);
    /* Field slots hold distinct sentinels so a swapped pair cannot cancel out. */
    printf("field_steps=%u\n", (unsigned)H3_ADALN_CACHE_FIELD_STEPS);
    printf("field_time_rows=%u\n", (unsigned)H3_ADALN_CACHE_FIELD_TIME_ROWS);
    printf("field_blocks=%u\n", (unsigned)H3_ADALN_CACHE_FIELD_BLOCKS);
    printf("field_block_output=%u\n",
           (unsigned)H3_ADALN_CACHE_FIELD_BLOCK_OUTPUT);
    printf("field_final_output=%u\n",
           (unsigned)H3_ADALN_CACHE_FIELD_FINAL_OUTPUT);
    printf("field_reserved=%u\n", (unsigned)H3_ADALN_CACHE_FIELD_RESERVED);
    for (unsigned rows = 1; rows <= 41; rows += rows == 1 ? 2u : 1u) {
        if (rows != 1u && rows != 3u && rows != 9u && rows != 17u &&
            rows != 41u)
            continue;
        printf("dump_bytes_%u=%llu\n", rows,
               (unsigned long long)H3_ADALN_CACHE_DUMP_BYTES(rows));
    }
}

static int parse_header(const char *path) {
    FILE *file = fopen(path, "rb");
    if (!file) {
        fprintf(stderr, "probe: cannot open %s\n", path);
        return 2;
    }
    unsigned char header[H3_ADALN_CACHE_HEADER_BYTES];
    if (fread(header, 1, sizeof(header), file) != sizeof(header)) {
        fprintf(stderr, "probe: %s is shorter than the %u-byte header\n", path,
                (unsigned)H3_ADALN_CACHE_HEADER_BYTES);
        fclose(file);
        return 2;
    }
    fclose(file);
    if (memcmp(header, H3_ADALN_CACHE_MAGIC, H3_ADALN_CACHE_MAGIC_BYTES) != 0) {
        fprintf(stderr, "probe: %s does not start with %s\n", path,
                H3_ADALN_CACHE_MAGIC);
        return 2;
    }
    uint32_t fields[H3_ADALN_CACHE_HEADER_FIELDS];
    memcpy(fields, header + H3_ADALN_CACHE_MAGIC_BYTES, sizeof(fields));
    /* Little-endian on disk; this box is, and the exporter assumes the same. */
    printf("steps=%u\n", fields[H3_ADALN_CACHE_FIELD_STEPS]);
    printf("time_rows=%u\n", fields[H3_ADALN_CACHE_FIELD_TIME_ROWS]);
    printf("blocks=%u\n", fields[H3_ADALN_CACHE_FIELD_BLOCKS]);
    printf("block_output=%u\n", fields[H3_ADALN_CACHE_FIELD_BLOCK_OUTPUT]);
    printf("final_output=%u\n", fields[H3_ADALN_CACHE_FIELD_FINAL_OUTPUT]);
    printf("reserved=%u\n", fields[H3_ADALN_CACHE_FIELD_RESERVED]);
    return 0;
}

static void print_keys(int steps) {
    char name[H3_ADALN_CACHE_NAME_MAX];
    snprintf(name, sizeof(name), H3_ADALN_CACHE_TIMES_FORMAT, steps);
    puts(name);
    for (unsigned block = 0; block < H3_ADALN_CACHE_BLOCK_ROWS; block++) {
        snprintf(name, sizeof(name), H3_ADALN_CACHE_BLOCK_FORMAT, block, steps);
        puts(name);
    }
    snprintf(name, sizeof(name), H3_ADALN_CACHE_FINAL_FORMAT, steps);
    puts(name);
    snprintf(name, sizeof(name), H3_ADALN_CACHE_META_FORMAT, steps);
    puts(name);
}

int main(int argc, char **argv) {
    if (argc > 1 && strcmp(argv[1], "print") == 0) {
        print_contract();
        return 0;
    }
    if (argc > 2 && strcmp(argv[1], "parse") == 0) return parse_header(argv[2]);
    if (argc > 2 && strcmp(argv[1], "keys") == 0) {
        print_keys(atoi(argv[2]));
        return 0;
    }
    fprintf(stderr, "usage: probe print | parse <dump> | keys <steps>\n");
    return 2;
}
