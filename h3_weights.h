#ifndef H3_WEIGHTS_H
#define H3_WEIGHTS_H

#include "h3_gpu.h"
#include "h3_safetensors.h"

#include <stddef.h>
#include <stdint.h>

typedef struct h3_weight_store h3_weight_store;

/* Open every safetensors header in a component directory without reading
 * tensor payloads. */
h3_weight_store *h3_weight_store_open(const char *directory,
                                      char *error, size_t error_size);
/* Open a single safetensors file as a one-shard weight store. */
h3_weight_store *h3_weight_store_open_file(const char *path,
                                          char *error, size_t error_size);
void h3_weight_store_free(h3_weight_store *store);
size_t h3_weight_store_shards(const h3_weight_store *store);

const h3_st_tensor *h3_weight_find(const h3_weight_store *store,
                                   const char *name,
                                   const h3_st_header **header);

/* Validate an exact BF16 shape, allocate a shared Metal buffer, and read the
 * payload directly into that buffer with no intermediate host allocation. */
h3_gpu_tensor *h3_weight_load_bf16(const h3_weight_store *store, h3_gpu *gpu,
                                   const char *name, int ndim,
                                   const uint64_t *shape,
                                   char *error, size_t error_size);
h3_gpu_tensor *h3_weight_load_f32(const h3_weight_store *store, h3_gpu *gpu,
                                  const char *name, int ndim,
                                  const uint64_t *shape,
                                  char *error, size_t error_size);
h3_gpu_tensor *h3_weight_load_i8(const h3_weight_store *store, h3_gpu *gpu,
                                 const char *name, int ndim, const uint64_t *shape,
                                 char *error, size_t error_size);

/* --- Packed group-quantized weights -------------------------------------
 * A second quantized representation, written by
 * fastvideo_qad/scripts/export_h3_int6_native.py, that stores the codes
 * bit-packed instead of one byte per weight:
 *
 *   {name}.weight        U8   [rows, columns * bits / 8]   packed unsigned codes
 *   {name}.weight_scale  F16 or F32  [rows, columns / group]
 *   {name}.weight_bias   F16 or F32  [rows, columns / group]
 *   dequant:  w = code * scale + bias            (single FMA per weight)
 *
 * Nothing is stored in metadata: the loader derives
 *   bits  = packed_bytes_per_row * 8 / columns
 *   group = columns / scale_columns
 * and requires `group` to be a multiple of 4 so every group starts on a byte
 * boundary.  Codes are little-endian; with bits == 6 four codes occupy three
 * bytes (c0 | c1<<6 | c2<<12 | c3<<18), with bits == 4 two per byte, and with
 * bits == 8 one per byte.  Weights stay in ConvRot-rotated space, and the group
 * scale applies *before* the un-rotation because group boundaries (e.g. 128)
 * are not a multiple of the rotation block (256). */
#define H3_GROUPED_MAX_BITS 8

typedef struct {
    const h3_st_tensor *weight;
    const h3_st_header *weight_header;
    const h3_st_tensor *scale;
    const h3_st_header *scale_header;
    const h3_st_tensor *bias;
    const h3_st_header *bias_header;
    unsigned bits;
    unsigned group;
    unsigned rows;
    unsigned columns;
    uint64_t packed_row_bytes;
    int f16_accumulators;
} h3_grouped_spec;

/* Derive and validate the spec for `name` given the caller's element count.
 * Returns 0 when the tensor is not in the packed group-quantized format (so the
 * caller can fall back), and -1 on a malformed packed tensor (error is set). */
int h3_weight_grouped_spec(const h3_weight_store *store, const char *name,
                           size_t elements, h3_grouped_spec *out,
                           char *error, size_t error_size);

/* Unpack and dequantize `rows` x `columns` codes into `values` (host F32).
 * `scales`/`biases` point at the first accumulator of this row window, i.e.
 * rows * (columns / group) entries each.  `widened` says how to read them: 0
 * means the raw on-disk F16/F32 arrays (the resident loader), 1 means the
 * reader already converted them to host float (the streaming consumer stages
 * thread-private copies).  Reading a widened array as F16 silently produces
 * garbage scales, so the two cases must not be confused. */
int h3_weight_dequantize_grouped(const h3_grouped_spec *spec,
                                 const uint8_t *packed, const void *scales,
                                 const void *biases, size_t rows, int widened,
                                 float *values, char *error, size_t error_size);

/* Undo the ConvRot rotation in place on already-dequantized rows, using the same
 * radix-4 butterfly as the INT8 loaders. Shared with the SSD streaming consumer
 * so the butterfly keeps a single implementation per call site. */
void h3_weight_unrotate_rows(float *values, size_t rows, size_t columns);

/* Read a packed weight's accumulators for the whole matrix, widened to float and
 * laid out as all-scales-then-all-biases: 2 * rows * (columns / group) floats in
 * a single allocation.  The streaming reader stages each chunk's slice from it
 * and hands the two halves to the dequantizer as separate pointers. */
int h3_weight_load_grouped_accumulators(const h3_grouped_spec *spec, float **out,
                                        char *error, size_t error_size);

#endif
