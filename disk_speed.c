/*
 * disk_speed.c — 顺序读写速度测试
 *
 * 用法:
 *   ./disk_speed <path> [mode] [chunk_mib] [total_mib]
 *     path        测试文件路径（会创建/截断该文件）
 *     mode        read|write|rw  (默认 read)
 *     chunk_mib   每次读写的块大小 MiB (默认 64)
 *     total_mib   总读写量 MiB (默认 2048)
 *
 * 用途: 对比不同 Type-C 口 / 不同盘的顺序读写带宽。
 * 写模式会覆盖目标文件，注意选择路径。
 */
#include <fcntl.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>

static double now_seconds(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec / 1e9;
}

static size_t parse_mib(const char *text) {
    char *tail = NULL;
    long value = strtol(text, &tail, 10);
    if (!tail || *tail || value <= 0) {
        fprintf(stderr, "bad size: %s\n", text);
        exit(2);
    }
    return (size_t)value;
}

int main(int argc, char **argv) {
    if (argc < 2) {
        fprintf(stderr,
                "usage: %s <path> [read|write|rw] [chunk_mib] [total_mib]\n",
                argv[0]);
        return 2;
    }
    const char *path = argv[1];
    const char *mode = argc >= 3 ? argv[2] : "read";
    size_t chunk_mib = argc >= 4 ? parse_mib(argv[3]) : 64;
    size_t total_mib = argc >= 5 ? parse_mib(argv[4]) : 2048;
    int do_write = strcmp(mode, "write") == 0 || strcmp(mode, "rw") == 0;
    int do_read = strcmp(mode, "read") == 0 || strcmp(mode, "rw") == 0;
    if (!do_write && !do_read) {
        fprintf(stderr, "mode must be read, write or rw\n");
        return 2;
    }
    size_t chunk_bytes = chunk_mib * 1024 * 1024;
    size_t total_bytes = total_mib * 1024 * 1024;
    if (total_bytes < chunk_bytes) total_bytes = chunk_bytes;

    /* 写测试先把文件截断成目标大小，模拟真实大文件。 */
    if (do_write) {
        int fd = open(path, O_WRONLY | O_CREAT | O_TRUNC, 0644);
        if (fd < 0) { perror("open (write)"); return 1; }
        if (ftruncate(fd, (off_t)total_bytes) != 0) { perror("ftruncate"); return 1; }
        /* 写满 0 让文件实际占盘，避免稀疏文件虚报速度 */
        unsigned char *pattern = malloc(chunk_bytes);
        memset(pattern, 0xAB, chunk_bytes);
        double start = now_seconds();
        size_t written = 0;
        while (written < total_bytes) {
            size_t now = total_bytes - written < chunk_bytes ?
                             total_bytes - written : chunk_bytes;
            ssize_t n = write(fd, pattern, now);
            if (n <= 0) { perror("write"); return 1; }
            written += (size_t)n;
        }
        double elapsed = now_seconds() - start;
        printf("%-12s path=%-32s size=%.0f MiB  WRITE  %.1f MiB/s  (%.1f s)\n",
               mode, path, (double)total_mib,
               (double)total_bytes / elapsed / (1024.0 * 1024.0), elapsed);
        close(fd);
        free(pattern);
    }
    if (do_read) {
        int fd = open(path, O_RDONLY);
        if (fd < 0) { perror("open (read)"); return 1; }
        unsigned char *buffer = malloc(chunk_bytes);
        double start = now_seconds();
        size_t read_bytes = 0;
        while (read_bytes < total_bytes) {
            ssize_t n = read(fd, buffer, chunk_bytes);
            if (n <= 0) break;
            read_bytes += (size_t)n;
        }
        double elapsed = now_seconds() - start;
        printf("%-12s path=%-32s size=%.0f MiB  READ   %.1f MiB/s  (%.1f s)\n",
               mode, path, (double)total_mib,
               (double)read_bytes / elapsed / (1024.0 * 1024.0), elapsed);
        close(fd);
        free(buffer);
    }
    return 0;
}
