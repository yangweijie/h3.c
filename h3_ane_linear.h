#ifndef H3_ANE_LINEAR_H
#define H3_ANE_LINEAR_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

/* One LoRA factor pair driving a contiguous band of output rows. A projection
 * that packs several diffusers modules (qkv = to_q/to_k/to_v, stored
 * module-major) needs one band per module; the others need exactly one. */
typedef struct {
    const int8_t *a;        /* [rank][input_dim] int8, already A_rot = A . R */
    const float *a_scales;  /* [rank] f32 */
    const int8_t *b;        /* [rows][rank] int8 */
    const float *b_scales;  /* [rows] f32 */
    uint32_t rank;
    uint32_t row0;          /* first output row this band drives */
    uint32_t rows;          /* band height; the bands must tile output_dim */
} h3_ane_bypass_band;

#define H3_ANE_MAX_BANDS 4

/* Neural Engine replacement for one DiT projection. The graph is a reduction
 * split: the K axis is cut into chunks of kc, every chunk is a 1x1 convolution
 * over a [1, kc, 1, rows] activation, and the partial products are summed.
 * Request input binding silently misbinds above eight inputs, so kc must keep
 * the chunk count at or below eight. */

typedef struct h3_ane_linear h3_ane_linear;

typedef enum {
    H3_ANE_W_F16 = 0,
    H3_ANE_W_BF16 = 1
} h3_ane_weight_dtype;

int h3_ane_linear_available(void);

/* Row-major [output_dim][input_dim] weights. */
h3_ane_linear *h3_ane_linear_create(const char *name, const void *weights,
                                    h3_ane_weight_dtype dtype,
                                    uint32_t input_dim, uint32_t output_dim,
                                    uint32_t rows, uint32_t kc,
                                    char *error, size_t error_size);

/* comfy-quants int8_tensorwise payload: row-major [output_dim][input_dim]
 * int8 with per-row f32 scales, kept int8 on the Neural Engine through
 * constexpr dequantization. A positive convrot_group_size adds the in-graph
 * grouped Hadamard activation rotation that the stored rows expect. */
h3_ane_linear *h3_ane_linear_create_int8(const char *name,
                                    const int8_t *quantized,
                                    const float *scales,
                                    int convrot_group_size,
                                    uint32_t input_dim, uint32_t output_dim,
                                    uint32_t rows, uint32_t kc,
                                    char *error, size_t error_size);

/* As above, plus a rank-r bypass branch folded into the SAME graph:
 *
 *     out = W_rot (R x)  +  B (A_rot (R x))
 *
 * The bypass reuses the rotated activations r{i} the main convolution already
 * computes, so it adds no input binding and costs no extra evaluation. A_rot
 * conv shares the main chunking (its K is the full input_dim); B is a single
 * [output_dim][rank] convolution, so its K needs no padding.
 *
 * The factors are comfy-quants int8_tensorwise like the main weight, and A
 * MUST already be rotated (A_rot = A . R with the grouped Hadamard of
 * convrot_group_size), because the graph feeds it r{i}, not h{i}. Fold
 * alpha/rank into either factor before calling.
 *
 * The factors go through constexpr_affine_dequantize, not a plain fp16 const:
 * the Neural Engine rejects a non-grouped convolution whose weight is an
 * ordinary const, which is why the only fp16 const convolution in the graph
 * (the Hadamard rotation) is a grouped one.
 *
 * Each band's convolution produces only its own rows and the bands are
 * concatenated along channels, so a band never pays for the other bands'
 * ranks --- unlike one block-diagonal B, which would cost (bands x rank) per
 * output row. */
h3_ane_linear *h3_ane_linear_create_int8_bands(const char *name,
                                    const int8_t *quantized,
                                    const float *scales,
                                    int convrot_group_size,
                                    const h3_ane_bypass_band *bands,
                                    uint32_t band_count,
                                    uint32_t input_dim, uint32_t output_dim,
                                    uint32_t rows, uint32_t kc,
                                    char *error, size_t error_size);

/* Single-band convenience wrapper: one factor pair drives every output row. */
h3_ane_linear *h3_ane_linear_create_int8_bypass(const char *name,
                                    const int8_t *quantized,
                                    const float *scales,
                                    int convrot_group_size,
                                    const int8_t *bypass_a,
                                    const float *bypass_a_scales,
                                    const int8_t *bypass_b,
                                    const float *bypass_b_scales,
                                    uint32_t rank,
                                    uint32_t input_dim, uint32_t output_dim,
                                    uint32_t rows, uint32_t kc,
                                    char *error, size_t error_size);

/* Payload that is already chunks x [output_dim][kc] fp16 with the K padding
 * zeroed. Used by the fixture gate. */
h3_ane_linear *h3_ane_linear_create_chunked(const char *name,
                                    const void *chunks, size_t chunk_bytes,
                                    uint32_t input_dim, uint32_t output_dim,
                                    uint32_t rows, uint32_t kc,
                                    char *error, size_t error_size);

void h3_ane_linear_free(h3_ane_linear *linear);

uint32_t h3_ane_linear_chunks(const h3_ane_linear *linear);
uint32_t h3_ane_linear_chunk_dim(const h3_ane_linear *linear);
uint32_t h3_ane_linear_rows(const h3_ane_linear *linear);
/* Row count the graph really runs: rows rounded up to a multiple of sixteen. */
uint32_t h3_ane_linear_plane_rows(const h3_ane_linear *linear);
uint32_t h3_ane_linear_output_dim(const h3_ane_linear *linear);

/* F32 [kc][rows] input plane for one chunk, and the F32 [output_dim][rows]
 * result. Both are IOSurface backed so a Metal kernel can write and read them
 * without a host copy. */
float *h3_ane_linear_input(h3_ane_linear *linear, uint32_t chunk);
float *h3_ane_linear_output(h3_ane_linear *linear);
size_t h3_ane_linear_input_bytes(const h3_ane_linear *linear);
size_t h3_ane_linear_output_bytes(const h3_ane_linear *linear);

int h3_ane_linear_eval(h3_ane_linear *linear, char *error, size_t error_size);

/* Residency rotation: unload drops the wired compiled net from the Neural
 * Engine but keeps the handle, the planes and the request binding, so reload
 * re-wires from the compile cache instead of recompiling. */
int h3_ane_linear_unload(h3_ane_linear *linear, char *error, size_t error_size);
int h3_ane_linear_reload(h3_ane_linear *linear, char *error, size_t error_size);

double h3_ane_linear_compile_seconds(const h3_ane_linear *linear);
uint64_t h3_ane_linear_weight_bytes(const h3_ane_linear *linear);
bool h3_ane_linear_cache_hit(const h3_ane_linear *linear);

/* One spliced DiT projection: the ANE graph plus the Metal staging that moves
 * activations between h3.c's row-major BF16 rows and the channel-major F32
 * planes the Neural Engine reads. */

#include "h3_gpu.h"
#include "h3_weights.h"

typedef struct h3_ane_projection h3_ane_projection;

/* Largest reduction tile that keeps the graph at eight inputs or fewer. */
uint32_t h3_ane_linear_default_chunk(uint32_t input_dim);

/* Activation rows rounded up to a compiled-shape bucket, so every resolution
 * reuses one artifact set instead of paying a fresh compile and a fresh
 * multiple-GiB cache per tile height. H3_ANE_ROW_BUCKET overrides the bucket
 * (0 turns rounding off). */
uint32_t h3_ane_rows_bucket(uint32_t rows);

/* Graphs created while sharing is on borrow their activation planes from a
 * per-shape pool instead of owning a set each, which is how a many-module
 * graph stays resident (a video VAE decoder block costs 320 MiB of planes at
 * rows 2048 and the decoder has 36 of them with identical shapes). Sharing
 * means same-shaped graphs alias the same memory, so only sequential use is
 * safe; the pool must outlive them. */
void h3_ane_planes_share(int enable);
uint64_t h3_ane_planes_bytes(void);
void h3_ane_planes_clear(void);

h3_ane_projection *h3_ane_projection_create(h3_gpu *gpu, const char *name,
                                    const h3_gpu_tensor *weight,
                                    uint32_t input_dim, uint32_t output_dim,
                                    uint32_t rows, uint32_t kc,
                                    char *error, size_t error_size);

/* As above from an fp16 payload laid out row-major [output_dim][input_dim],
 * for modules whose weights round straight to fp16 (the video VAE). */
h3_ane_projection *h3_ane_projection_create_f16(h3_gpu *gpu, const char *name,
                                    const uint16_t *weights,
                                    uint32_t input_dim, uint32_t output_dim,
                                    uint32_t rows, uint32_t kc,
                                    char *error, size_t error_size);

/* Build the projection straight from a stored int8_tensorwise weight; the
 * raw payload is read, packed into the graph, and released. */
h3_ane_projection *h3_ane_projection_create_int8(h3_gpu *gpu, const char *name,
                                    const h3_weight_store *store,
                                    const char *weight_name,
                                    uint32_t input_dim, uint32_t output_dim,
                                    uint32_t rows, uint32_t kc,
                                    char *error, size_t error_size);

/* As above from an in-memory int8 payload. */
h3_ane_projection *h3_ane_projection_create_int8_raw(h3_gpu *gpu,
                                    const char *name,
                                    const int8_t *quantized,
                                    const float *scales,
                                    int convrot_group_size,
                                    uint32_t input_dim, uint32_t output_dim,
                                    uint32_t rows, uint32_t kc,
                                    char *error, size_t error_size);

/* As above plus the multi-band LoRA bypass (see h3_ane_linear_create_int8_bands):
 * qkv needs three bands (to_q/to_k/to_v, module-major); the rest need one. */
h3_ane_projection *h3_ane_projection_create_int8_bands_raw(h3_gpu *gpu,
                                    const char *name,
                                    const int8_t *quantized,
                                    const float *scales,
                                    int convrot_group_size,
                                    const h3_ane_bypass_band *bands,
                                    uint32_t band_count,
                                    uint32_t input_dim, uint32_t output_dim,
                                    uint32_t rows, uint32_t kc,
                                    char *error, size_t error_size);
void h3_ane_projection_free(h3_ane_projection *projection);

/* Consumes the open command buffer, runs the Neural Engine, and leaves a new
 * command buffer open with the unpack encoded. */
int h3_ane_projection_apply(h3_ane_projection *projection, h3_gpu *gpu,
                            h3_gpu_tensor *output, const h3_gpu_tensor *input,
                            char *error, size_t error_size);

/* F32 staging variant: row-major F32 activations in and out, with the channel
 * bias added on the unpack side because the graph models only the reduction.
 * `bias` may be NULL. */
int h3_ane_projection_apply_f32(h3_ane_projection *projection, h3_gpu *gpu,
                                h3_gpu_tensor *output,
                                const h3_gpu_tensor *input,
                                const h3_gpu_tensor *bias,
                                char *error, size_t error_size);

/* Narrow how many rows the staging reads and writes, for callers whose
 * activation count sits below the compiled bucket. The artifact keeps its padded
 * height; a 1x1 convolution is position-wise, so rows past this count neither
 * read the activations nor land in the output. Returns 0 if `rows` exceeds the
 * graph's plane height. */
int h3_ane_projection_set_activation_rows(h3_ane_projection *projection,
                                          uint32_t rows);

void h3_ane_projection_timings(const h3_ane_projection *projection,
                               double *pack_seconds, double *eval_seconds,
                               uint64_t *calls);
/* pack_seconds here also covers draining the Metal work already queued ahead of
 * the pack, since without stage profiling there is no sync to separate it. Read
 * it as "time until the Neural Engine could start", not as staging cost. */
/* With H3_ANE_PROFILE_STAGES set, apply uses separate command buffers so sync,
 * pack and unpack can be measured without attributing prior Metal work to the
 * pack. This is a diagnostic mode and intentionally adds synchronization. */
void h3_ane_projection_stage_timings(const h3_ane_projection *projection,
                                     double *sync_seconds,
                                     double *pack_seconds,
                                     double *eval_seconds,
                                     double *unpack_seconds,
                                     uint64_t *calls);
uint64_t h3_ane_projection_weight_bytes(const h3_ane_projection *projection);
double h3_ane_projection_compile_seconds(const h3_ane_projection *projection);
bool h3_ane_projection_cache_hit(const h3_ane_projection *projection);

/* Activation planes stay allocated across unload/reload, so a residency budget
 * has to count them next to the compiled constants. */
size_t h3_ane_projection_plane_bytes(const h3_ane_projection *projection);

/* Rotation for the spliced form: the planes and Metal-side tensors keep their
 * memory, only the Neural Engine handle is parked and rewired. */
int h3_ane_projection_unload(h3_ane_projection *projection, char *error,
                             size_t error_size);
int h3_ane_projection_reload(h3_ane_projection *projection, char *error,
                             size_t error_size);

#endif
