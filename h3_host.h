#ifndef H3_HOST_H
#define H3_HOST_H

#include <stddef.h>
#include <stdint.h>

#define H3_CANVAS_MULTIPLE 32
#define H3_MAX_PIXELS (768 * 1344)
#define H3_FPS 24
#define H3_AUDIO_LATENT_FPS 40
#define H3_VAE_SPATIAL_RATIO 16
#define H3_VIDEO_SIGMA_SHIFT 12.0
#define H3_AUDIO_SIGMA_SHIFT 3.0
#define H3_MAX_STEPS 1000

typedef struct {
    int frame_count;
    int video_t;
    int audio_t;
} h3_temporal_shape;

typedef struct {
    double t;
    double h;
    double w;
} h3_position;

typedef enum {
    H3_SEG_TEXT,
    H3_SEG_COND,
    H3_SEG_REF_IMAGE,
    H3_SEG_REF_AUDIO,
    H3_SEG_AUDIO,
    H3_SEG_VIDEO
} h3_segment_kind;

typedef struct {
    size_t start;
    size_t stop;
    h3_segment_kind kind;
} h3_segment;

typedef enum {
    H3_LAYOUT_REF_IMAGE,
    H3_LAYOUT_REF_AUDIO,
    H3_LAYOUT_REF_VIDEO
} h3_layout_ref_kind;

typedef struct {
    h3_layout_ref_kind kind;
    int latent_t;
    int latent_h;
    int latent_w;
    int audio_t;
} h3_layout_ref;

typedef struct {
    int text_len;
    int latent_t;
    int latent_h;
    int latent_w;
    int audio_t;
    int frame_count;
    const int *keyframes;
    size_t keyframe_count;
    const h3_layout_ref *references;
    size_t reference_count;
} h3_layout_spec;

typedef struct {
    size_t seq_len;
    h3_segment *segments;
    size_t segment_count;
    h3_position *positions;
    size_t img_cond_rows;
    size_t img_target_rows;
    size_t audio_cond_rows;
    size_t audio_target_rows;
    int signature[5];
} h3_layout;

typedef struct {
    int steps;
    float video[H3_MAX_STEPS + 1];
    float audio[H3_MAX_STEPS + 1];
} h3_sigma_schedule;

typedef struct {
    uint64_t state;
    uint64_t increment;
    float spare;
    int has_spare;
} h3_rng;

int h3_align_frame_count(int requested);
int h3_video_latent_t(int frame_count);
int h3_video_encoder_latent_t(int frame_count);
h3_temporal_shape h3_temporal(int requested_frames);
void h3_latent_canvas(int width, int height, int *latent_w, int *latent_h);
int h3_adapt_canvas(int width, int height, int *adapted_w, int *adapted_h);
/* Ref2VA image sizing is down-only and aspect preserving. max_short_edge=0
 * matches the output pixel area; a positive value selects a short-edge cap. */
int h3_reference_image_canvas(int width, int height,
                              int target_width, int target_height,
                              int max_short_edge,
                              int *adapted_w, int *adapted_h);
/* Ref2VA video references use the normal target-style canvas, except that a
 * smaller source is never enlarged. */
int h3_reference_video_canvas(int width, int height,
                              int *adapted_w, int *adapted_h);

double h3_time_shift_sigma(double sigma, double from_shift, double to_shift);
double h3_time_shift_slope(double sigma, double from_shift, double to_shift);
int h3_schedule_build(int steps, h3_sigma_schedule *schedule);
/* Released linear base grid: evaluations model forwards plus terminal zero. */
int h3_serving_schedule_build(int evaluations, h3_sigma_schedule *schedule);
/* Same shifted curve as h3_serving_schedule_build, but starting at
 * start_sigma instead of 1.0. Used to refine a clean latent (img2img /
 * hires-fix after a latent upscale): the caller re-noises the latent to
 * start_sigma and then runs `evaluations` passes to zero. */
int h3_refine_schedule_build(float start_sigma, int evaluations,
                             h3_sigma_schedule *schedule);

int h3_layout_build(const h3_layout_spec *spec, h3_layout *layout,
                    char *error, size_t error_size);
void h3_layout_free(h3_layout *layout);
const char *h3_segment_name(h3_segment_kind kind);

void h3_rng_seed(h3_rng *rng, uint64_t seed);
uint32_t h3_rng_u32(h3_rng *rng);
float h3_rng_normal(h3_rng *rng);
void h3_rng_fill_normal(h3_rng *rng, float *values, size_t count);

/* Resize interleaved RGB24 frames with Accelerate/vImage high-quality
 * resampling. The caller owns *output. Identity geometry still returns an
 * independent copy. */
int h3_resize_rgb24_high_quality(const uint8_t *input, int frames,
                                 int input_width, int input_height,
                                 int output_width, int output_height,
                                 uint8_t **output);

/* Bilinear spatial resize of an NCDHW float volume, one plane per
 * (channel, frame). Matches PyTorch's
 * F.interpolate(mode="trilinear", align_corners=False) for the case the H3
 * latent upscaler uses: the temporal extent is left untouched, so the
 * trilinear kernel degenerates to bilinear per frame. Returns 0 on invalid
 * geometry. */
int h3_resize_bilinear_f32(const float *source, int channels, int frames,
                           int source_height, int source_width,
                           float *destination, int destination_height,
                           int destination_width);

int h3_res_step(float *output, const float *sample, const float *denoised,
                const float *old_denoised, size_t count,
                const float *sigmas, int step, int total_steps);
int h3_euler_velocity_step(float *sample, const float *velocity, size_t count,
                           float sigma, float sigma_next);

/* Query currently available physical memory (free + inactive + purgeable) in
 * bytes. Returns 0 if the platform does not support the query. */
uint64_t h3_host_available_memory(void);

/* Runtime memory guard: returns 1 while available physical memory stays above
 * the safety floor, or 0 (setting error) once it drops below. Called between
 * denoise steps and VAE chunks so a small-RAM Mac errors out gracefully
 * instead of thrashing into a system panic (GPU wired memory cannot be
 * swapped). The floor is H3_MEM_GUARD_DEFAULT_MB, overridable with
 * H3_MEM_GUARD_MB (0 disables the guard). */
#ifndef H3_MEM_GUARD_DEFAULT_MB
#define H3_MEM_GUARD_DEFAULT_MB 1024
#endif
/* System headroom kept below physical RAM for the footprint check. */
#ifndef H3_MEM_HEADROOM_DEFAULT_MB
#define H3_MEM_HEADROOM_DEFAULT_MB 2560
#endif
int h3_host_memory_guard(const char *phase, char *error, size_t error_size);

/* Physical footprint of the current process (resident + GPU wired +
 * compressed), in bytes. The most direct measure of how much RAM we actually
 * hold. Returns 0 if unavailable. */
uint64_t h3_host_footprint(void);

/* Estimate how many DiT blocks can be kept resident given the available memory
 * headroom. per_block_cost is the resident cost of one block (bytes).
 * Returns a value in [0, active_blocks - 2] (at least 2 blocks must stay in the
 * streaming ring). */
unsigned h3_dit_resident_budget(uint64_t available, uint64_t per_block_cost,
                                unsigned active_blocks);

#endif
