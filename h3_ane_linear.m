#import "h3_ane_linear.h"

#import "h3_ane_bridge.h"
#import "h3_convrot.h"
#import "h3_weights.h"

#import <Foundation/Foundation.h>
#import <IOSurface/IOSurface.h>

#include <string.h>
#include <sys/stat.h>
#include <time.h>

#define H3_ANE_MAX_CHUNKS 8
/* The graph reads [1, chunk, 1, rows]. A row count that is not a multiple of
 * sixteen is either rejected at evaluation or, in the three counts below each
 * multiple, silently reduced over a truncated K. Pad instead. */
#define H3_ANE_ROW_MULTIPLE 16u

struct h3_ane_linear {
    uint32_t input_dim;
    uint32_t padded_dim;
    uint32_t output_dim;
    uint32_t rows;
    uint32_t plane_rows;
    uint32_t chunk_dim;
    uint32_t chunks;
    size_t input_bytes;
    size_t output_bytes;
    uint64_t weight_bytes;
    IOSurfaceRef input_surface[H3_ANE_MAX_CHUNKS];
    IOSurfaceRef output_surface;
    float *input_base[H3_ANE_MAX_CHUNKS];
    float *output_base;
    h3_ane_model *model;
};

static void ane_fail(char *error, size_t error_size, const char *format, ...) {
    if (!error || !error_size) return;
    va_list arguments;
    va_start(arguments, format);
    vsnprintf(error, error_size, format, arguments);
    va_end(arguments);
}

static double ane_seconds(void) {
    struct timespec now;
    clock_gettime(CLOCK_MONOTONIC, &now);
    return (double)now.tv_sec + (double)now.tv_nsec * 1e-9;
}

int h3_ane_linear_available(void) {
    return h3_ane_bridge_available();
}

static NSString *ane_program(uint32_t chunk_dim, uint32_t output_dim,
                             uint32_t chunks, uint32_t rows) {
    NSMutableString *text = [NSMutableString string];
    [text appendString:@"program(1.3)\n[buildInfo = dict<string, string>({{\""
        "coremlc-component-MIL\", \"3510.2.1\"}, {\"coremlc-version\", "
        "\"3505.4.1\"}, {\"coremltools-component-milinternal\", \"\"}, "
        "{\"coremltools-version\", \"9.0\"}})]\n{\n    func main<ios18>("];
    for (uint32_t i = 0; i < chunks; i++)
        [text appendFormat:@"tensor<fp32, [1, %u, 1, %u]> x%u%s",
            chunk_dim, rows, i, i + 1 == chunks ? "" : ", "];
    [text appendString:@") {\n"
        "            string pt = const()[name = string(\"pt\"), val = string(\"valid\")];\n"
        "            tensor<int32, [2]> st = const()[name = string(\"st\"), val = tensor<int32, [2]>([1, 1])];\n"
        "            tensor<int32, [4]> pd = const()[name = string(\"pd\"), val = tensor<int32, [4]>([0, 0, 0, 0])];\n"
        "            tensor<int32, [2]> dl = const()[name = string(\"dl\"), val = tensor<int32, [2]>([1, 1])];\n"
        "            int32 gr = const()[name = string(\"gr\"), val = int32(1)];\n"
        "            string f16 = const()[name = string(\"f16\"), val = string(\"fp16\")];\n"];
    unsigned long stride = 64ul + (unsigned long)chunk_dim * output_dim * 2ul;
    for (uint32_t i = 0; i < chunks; i++) {
        [text appendFormat:@"            tensor<fp16, [1, %u, 1, %u]> h%u = "
            "cast(dtype = f16, x = x%u)[name = string(\"ci%u\")];\n",
            chunk_dim, rows, i, i, i];
        [text appendFormat:@"            tensor<fp16, [%u, %u, 1, 1]> W%u = "
            "const()[name = string(\"W%u\"), val = tensor<fp16, [%u, %u, 1, 1]>("
            "BLOBFILE(path = string(\"@model_path/weights/weight.bin\"), "
            "offset = uint64(%lu)))];\n",
            output_dim, chunk_dim, i, i, output_dim, chunk_dim,
            64ul + (unsigned long)i * stride];
        [text appendFormat:@"            tensor<fp16, [1, %u, 1, %u]> p%u = "
            "conv(dilations = dl, groups = gr, pad = pd, pad_type = pt, "
            "strides = st, weight = W%u, x = h%u)[name = string(\"p%u\")];\n",
            output_dim, rows, i, i, i, i];
    }
    NSString *previous = @"p0";
    for (uint32_t i = 1; i < chunks; i++) {
        NSString *sum = [NSString stringWithFormat:@"s%u", i];
        [text appendFormat:@"            tensor<fp16, [1, %u, 1, %u]> %@ = "
            "add(x = %@, y = p%u)[name = string(\"%@\")];\n",
            output_dim, rows, sum, previous, i, sum];
        previous = sum;
    }
    [text appendString:@"            string f32 = const()[name = string(\"f32\"), "
        "val = string(\"fp32\")];\n"];
    [text appendFormat:@"            tensor<fp32, [1, %u, 1, %u]> c = "
        "cast(dtype = f32, x = %@)[name = string(\"co\")];\n",
        output_dim, rows, previous];
    [text appendString:@"        } -> (c);\n}\n"];
    return text;
}

typedef struct {
    uint32_t group_size;
    unsigned long tile_stride;
    unsigned long scale_offset;
    unsigned long rotation_offset;
    /* Zero band_count means no bypass branch and the plain int8 program. */
    uint32_t band_count;
    uint32_t band_rank[H3_ANE_MAX_BANDS];
    uint32_t band_rows[H3_ANE_MAX_BANDS];
    unsigned long band_a_stride[H3_ANE_MAX_BANDS];
    unsigned long band_a_offset[H3_ANE_MAX_BANDS];
    unsigned long band_a_scale[H3_ANE_MAX_BANDS];
    unsigned long band_b_offset[H3_ANE_MAX_BANDS];
    unsigned long band_b_scale[H3_ANE_MAX_BANDS];
} ane_int8_layout;

/* INT8 program: per K chunk the fp16 activations are optionally rotated by a
 * grouped 1x1 convolution against the shared ConvRot Hadamard block, then
 * convolved with constexpr-dequantized int8 weights. The per-row scale is a
 * rank-1 fp16 tensor; ANECCompile rejects the [N,1,1,1] spelling. */
static NSString *ane_program_int8(uint32_t chunk_dim, uint32_t output_dim,
                                  uint32_t chunks, uint32_t rows,
                                  const ane_int8_layout *layout) {
    uint32_t group_size = layout->group_size;
    unsigned long tile_stride = layout->tile_stride;
    unsigned long scale_offset = layout->scale_offset;
    unsigned long rotation_offset = layout->rotation_offset;
    uint32_t band_count = layout->band_count;
    NSMutableString *text = [NSMutableString string];
    [text appendString:@"program(1.3)\n[buildInfo = dict<string, string>({{\""
        "coremlc-component-MIL\", \"3510.2.1\"}, {\"coremlc-version\", "
        "\"3505.4.1\"}, {\"coremltools-component-milinternal\", \"\"}, "
        "{\"coremltools-version\", \"9.0\"}})]\n{\n    func main<ios18>("];
    for (uint32_t i = 0; i < chunks; i++)
        [text appendFormat:@"tensor<fp32, [1, %u, 1, %u]> x%u%s",
            chunk_dim, rows, i, i + 1 == chunks ? "" : ", "];
    [text appendString:@") {\n"
        "            string pt = const()[name = string(\"pt\"), val = string(\"valid\")];\n"
        "            tensor<int32, [2]> st = const()[name = string(\"st\"), val = tensor<int32, [2]>([1, 1])];\n"
        "            tensor<int32, [4]> pd = const()[name = string(\"pd\"), val = tensor<int32, [4]>([0, 0, 0, 0])];\n"
        "            tensor<int32, [2]> dl = const()[name = string(\"dl\"), val = tensor<int32, [2]>([1, 1])];\n"
        "            int32 gr = const()[name = string(\"gr\"), val = int32(1)];\n"
        "            string f16 = const()[name = string(\"f16\"), val = string(\"fp16\")];\n"];
    if (group_size)
        [text appendFormat:@"            int32 hg = const()[name = string(\"hg\"), val = int32(%u)];\n"
            "            tensor<fp16, [%u, %u, 1, 1]> HR = const()[name = string(\"HR\"), "
            "val = tensor<fp16, [%u, %u, 1, 1]>(BLOBFILE("
            "path = string(\"@model_path/weights/weight.bin\"), "
            "offset = uint64(%lu)))];\n",
            chunk_dim / group_size, chunk_dim, group_size, chunk_dim,
            group_size, rotation_offset];
    for (uint32_t i = 0; i < chunks; i++) {
        [text appendFormat:@"            tensor<fp16, [1, %u, 1, %u]> h%u = "
            "cast(dtype = f16, x = x%u)[name = string(\"ci%u\")];\n",
            chunk_dim, rows, i, i, i];
        NSString *feed = [NSString stringWithFormat:@"h%u", i];
        if (group_size) {
            [text appendFormat:@"            tensor<fp16, [1, %u, 1, %u]> r%u = "
                "conv(dilations = dl, groups = hg, pad = pd, pad_type = pt, "
                "strides = st, weight = HR, x = %@)[name = string(\"r%u\")];\n",
                chunk_dim, rows, i, feed, i];
            feed = [NSString stringWithFormat:@"r%u", i];
        }
        [text appendFormat:@"            tensor<fp16, [%u, %u, 1, 1]> W%u = "
            "constexpr_affine_dequantize()[axis = int32(0), "
            "name = string(\"W%u\"), quantized_data = tensor<int8, "
            "[%u, %u, 1, 1]>(BLOBFILE(path = string(\"@model_path/weights/"
            "weight.bin\"), offset = uint64(%lu))), scale = tensor<fp16, "
            "[%u]>(BLOBFILE(path = string(\"@model_path/weights/weight.bin\"), "
            "offset = uint64(%lu))), zero_point = int8(0)];\n",
            output_dim, chunk_dim, i, i, output_dim, chunk_dim,
            64ul + (unsigned long)i * tile_stride, output_dim, scale_offset];
        [text appendFormat:@"            tensor<fp16, [1, %u, 1, %u]> p%u = "
            "conv(dilations = dl, groups = gr, pad = pd, pad_type = pt, "
            "strides = st, weight = W%u, x = %@)[name = string(\"p%u\")];\n",
            output_dim, rows, i, i, feed, i];
    }
    NSString *previous = @"p0";
    for (uint32_t i = 1; i < chunks; i++) {
        NSString *sum = [NSString stringWithFormat:@"s%u", i];
        [text appendFormat:@"            tensor<fp16, [1, %u, 1, %u]> %@ = "
            "add(x = %@, y = p%u)[name = string(\"%@\")];\n",
            output_dim, rows, sum, previous, i, sum];
        previous = sum;
    }
    /* Bypass bands: the same rotated activations feed a rank-r projection per
     * band; each band yields only its own rows and the bands are concatenated
     * along channels, so the whole thing stays one eval. */
    if (band_count) {
        NSString *feed = group_size ? @"r" : @"h";
        if (band_count > 1)
            [text appendString:
                @"            int32 ch_axis = const()[name = string(\"ch_axis\"),"
                " val = int32(1)];\n"
                "            bool noil = const()[name = string(\"noil\"), "
                "val = bool(false)];\n"];
        NSMutableString *parts = [NSMutableString string];
        for (uint32_t band = 0; band < band_count; band++) {
            uint32_t rank = layout->band_rank[band];
            uint32_t b_rows = layout->band_rows[band];
            for (uint32_t i = 0; i < chunks; i++) {
                [text appendFormat:@"            tensor<fp16, [%u, %u, 1, 1]> "
                    "WA%u_%u = constexpr_affine_dequantize()[axis = int32(0), "
                    "name = string(\"WA%u_%u\"), quantized_data = tensor<int8, "
                    "[%u, %u, 1, 1]>(BLOBFILE(path = string(\"@model_path/weights/"
                    "weight.bin\"), offset = uint64(%lu))), scale = tensor<fp16, "
                    "[%u]>(BLOBFILE(path = string(\"@model_path/weights/"
                    "weight.bin\"), offset = uint64(%lu))), "
                    "zero_point = int8(0)];\n",
                    rank, chunk_dim, band, i, band, i, rank, chunk_dim,
                    layout->band_a_offset[band] +
                        (unsigned long)i * layout->band_a_stride[band],
                    rank, layout->band_a_scale[band]];
                [text appendFormat:@"            tensor<fp16, [1, %u, 1, %u]> "
                    "q%u_%u = conv(dilations = dl, groups = gr, pad = pd, "
                    "pad_type = pt, strides = st, weight = WA%u_%u, "
                    "x = %@%u)[name = string(\"q%u_%u\")];\n",
                    rank, rows, band, i, band, i, feed, i, band, i];
            }
            NSString *rank_sum = [NSString stringWithFormat:@"q%u_0", band];
            for (uint32_t i = 1; i < chunks; i++) {
                NSString *sum = [NSString stringWithFormat:@"t%u_%u", band, i];
                [text appendFormat:@"            tensor<fp16, [1, %u, 1, %u]> %@ = "
                    "add(x = %@, y = q%u_%u)[name = string(\"%@\")];\n",
                    rank, rows, sum, rank_sum, band, i, sum];
                rank_sum = sum;
            }
            [text appendFormat:@"            tensor<fp16, [%u, %u, 1, 1]> "
                "WB%u = constexpr_affine_dequantize()[axis = int32(0), "
                "name = string(\"WB%u\"), quantized_data = tensor<int8, "
                "[%u, %u, 1, 1]>(BLOBFILE(path = string(\"@model_path/weights/"
                "weight.bin\"), offset = uint64(%lu))), scale = tensor<fp16, "
                "[%u]>(BLOBFILE(path = string(\"@model_path/weights/weight.bin\"), "
                "offset = uint64(%lu))), zero_point = int8(0)];\n",
                b_rows, rank, band, band, b_rows, rank,
                layout->band_b_offset[band], b_rows, layout->band_b_scale[band]];
            [text appendFormat:@"            tensor<fp16, [1, %u, 1, %u]> "
                "bp%u = conv(dilations = dl, groups = gr, pad = pd, "
                "pad_type = pt, strides = st, weight = WB%u, x = %@)"
                "[name = string(\"bp%u\")];\n",
                b_rows, rows, band, band, rank_sum, band];
            [parts appendFormat:@"%@bp%u", band ? @", " : @"", band];
        }
        NSString *combined = [NSString stringWithFormat:@"bp%u", band_count - 1];
        if (band_count > 1) {
            combined = @"bcat";
            [text appendFormat:@"            tensor<fp16, [1, %u, 1, %u]> bcat = "
                "concat(axis = ch_axis, interleave = noil, values = (%@))"
                "[name = string(\"bcat\")];\n",
                output_dim, rows, parts];
        }
        [text appendFormat:@"            tensor<fp16, [1, %u, 1, %u]> o = "
            "add(x = %@, y = %@)[name = string(\"o\")];\n",
            output_dim, rows, previous, combined];
        previous = @"o";
    }
    [text appendString:@"            string f32 = const()[name = string(\"f32\"), "
        "val = string(\"fp32\")];\n"];
    [text appendFormat:@"            tensor<fp32, [1, %u, 1, %u]> c = "
        "cast(dtype = f32, x = %@)[name = string(\"co\")];\n",
        output_dim, rows, previous];
    [text appendString:@"        } -> (c);\n}\n"];
    return text;
}

/* 64-byte file header, then one 64-byte chunk header per weight tile. The
 * chunk header carries the payload size at +8 and the payload offset from the
 * start of the file at +16. */
static uint8_t *ane_weight_blob(const void *weights, h3_ane_weight_dtype dtype,
                                uint32_t input_dim, uint32_t output_dim,
                                uint32_t chunk_dim, uint32_t chunks,
                                size_t *out_bytes) {
    size_t tile = (size_t)output_dim * chunk_dim * 2;
    size_t stride = 64 + tile;
    size_t total = 64 + stride * chunks;
    uint8_t *blob = calloc(total, 1);
    if (!blob) return NULL;
    blob[0] = 0x01;
    blob[4] = 0x02;
    for (uint32_t c = 0; c < chunks; c++) {
        uint8_t *header = blob + 64 + (size_t)c * stride;
        header[0] = 0xEF; header[1] = 0xBE; header[2] = 0xAD; header[3] = 0xDE;
        header[4] = 0x01;
        uint32_t payload = (uint32_t)tile;
        uint32_t offset = (uint32_t)(64 + (size_t)c * stride + 64);
        memcpy(header + 8, &payload, sizeof(payload));
        memcpy(header + 16, &offset, sizeof(offset));
        __fp16 *destination = (__fp16 *)(void *)(header + 64);
        uint32_t base = c * chunk_dim;
        uint32_t span = input_dim > base ?
            (input_dim - base < chunk_dim ? input_dim - base : chunk_dim) : 0;
        for (uint32_t n = 0; n < output_dim; n++) {
            const uint16_t *source =
                (const uint16_t *)weights + (size_t)n * input_dim + base;
            __fp16 *row = destination + (size_t)n * chunk_dim;
            if (dtype == H3_ANE_W_F16) {
                memcpy(row, source, (size_t)span * 2);
            } else {
                for (uint32_t k = 0; k < span; k++) {
                    uint32_t wide = (uint32_t)source[k] << 16;
                    float value;
                    memcpy(&value, &wide, sizeof(value));
                    row[k] = (__fp16)value;
                }
            }
        }
    }
    *out_bytes = total;
    return blob;
}

static uint8_t *ane_weight_blob_chunked(const void *chunked, size_t bytes,
                                        uint32_t output_dim,
                                        uint32_t chunk_dim, uint32_t chunks,
                                        size_t *out_bytes) {
    size_t tile = (size_t)output_dim * chunk_dim * 2;
    if (bytes != tile * chunks) return NULL;
    size_t stride = 64 + tile;
    size_t total = 64 + stride * chunks;
    uint8_t *blob = calloc(total, 1);
    if (!blob) return NULL;
    blob[0] = 0x01;
    blob[4] = 0x02;
    for (uint32_t c = 0; c < chunks; c++) {
        uint8_t *header = blob + 64 + (size_t)c * stride;
        header[0] = 0xEF; header[1] = 0xBE; header[2] = 0xAD; header[3] = 0xDE;
        header[4] = 0x01;
        uint32_t payload = (uint32_t)tile;
        uint32_t offset = (uint32_t)(64 + (size_t)c * stride + 64);
        memcpy(header + 8, &payload, sizeof(payload));
        memcpy(header + 16, &offset, sizeof(offset));
        memcpy(header + 64, (const uint8_t *)chunked + (size_t)c * tile, tile);
    }
    *out_bytes = total;
    return blob;
}

static void ane_chunk_header(uint8_t *blob, size_t header_offset,
                             size_t payload) {
    uint8_t *header = blob + header_offset;
    header[0] = 0xEF; header[1] = 0xBE; header[2] = 0xAD; header[3] = 0xDE;
    header[4] = 0x01;
    uint32_t size = (uint32_t)payload;
    uint32_t offset = (uint32_t)(header_offset + 64);
    memcpy(header + 8, &size, sizeof(size));
    memcpy(header + 16, &offset, sizeof(offset));
}

/* Blob layout: T int8 tiles [N][kc] (zero K padding), one fp16 scale chunk
 * [N], when rotated one fp16 Hadamard chunk [kc][gs], then, per bypass band,
 * T int8 A tiles [rank][kc], an fp16 scale chunk [rank], one int8 B tile
 * [rows][rank] and an fp16 scale chunk [rows]. */
static uint8_t *ane_weight_blob_int8(const int8_t *quantized,
                                     const float *scales,
                                     uint32_t group_size,
                                     const h3_ane_bypass_band *bands,
                                     uint32_t band_count,
                                     uint32_t input_dim, uint32_t output_dim,
                                     uint32_t chunk_dim, uint32_t chunks,
                                     size_t *out_bytes,
                                     ane_int8_layout *layout) {
    size_t tile = (size_t)output_dim * chunk_dim;
    size_t stride = 64 + ((tile + 63) & ~(size_t)63);
    size_t scale_bytes = (size_t)output_dim * 2;
    size_t rotation_bytes = group_size ?
        (size_t)chunk_dim * group_size * 2 : 0;
    size_t scale_header = 64 + stride * chunks;
    size_t rotation_header = scale_header + 64 +
        ((scale_bytes + 63) & ~(size_t)63);
    size_t total = group_size ?
        rotation_header + 64 + rotation_bytes : rotation_header;
    size_t a_stride[H3_ANE_MAX_BANDS] = {0};
    for (uint32_t i = 0; i < band_count; i++) {
        uint32_t rank = bands[i].rank;
        size_t a_tile = (size_t)rank * chunk_dim;
        a_stride[i] = 64 + ((a_tile + 63) & ~(size_t)63);
        size_t b_bytes = (size_t)bands[i].rows * rank;
        size_t a_header = ((total + 63) & ~(size_t)63);
        size_t a_scale_header = ((a_header + a_stride[i] * chunks + 63) &
                                 ~(size_t)63);
        size_t b_header = ((a_scale_header + 64 + (size_t)rank * 2 + 63) &
                           ~(size_t)63);
        size_t b_scale_header = ((b_header + 64 + b_bytes + 63) & ~(size_t)63);
        total = b_scale_header + 64 + (size_t)bands[i].rows * 2;
        layout->band_rank[i] = rank;
        layout->band_rows[i] = bands[i].rows;
        layout->band_a_stride[i] = (unsigned long)a_stride[i];
        layout->band_a_offset[i] = (unsigned long)a_header;
        layout->band_a_scale[i] = (unsigned long)a_scale_header;
        layout->band_b_offset[i] = (unsigned long)b_header;
        layout->band_b_scale[i] = (unsigned long)b_scale_header;
    }
    uint8_t *blob = calloc(total, 1);
    if (!blob) return NULL;
    blob[0] = 0x01;
    blob[4] = 0x02;
    for (uint32_t c = 0; c < chunks; c++) {
        ane_chunk_header(blob, 64 + (size_t)c * stride, tile);
        int8_t *destination = (int8_t *)(blob + 64 + (size_t)c * stride + 64);
        uint32_t base = c * chunk_dim;
        uint32_t span = input_dim > base ?
            (input_dim - base < chunk_dim ? input_dim - base : chunk_dim) : 0;
        for (uint32_t n = 0; n < output_dim; n++)
            memcpy(destination + (size_t)n * chunk_dim,
                   quantized + (size_t)n * input_dim + base, span);
    }
    ane_chunk_header(blob, scale_header, scale_bytes);
    __fp16 *scale_data = (__fp16 *)(void *)(blob + scale_header + 64);
    for (uint32_t n = 0; n < output_dim; n++)
        scale_data[n] = (__fp16)scales[n];
    if (group_size) {
        ane_chunk_header(blob, rotation_header, rotation_bytes);
        const float *table = h3_convrot_hadamard((int)group_size);
        __fp16 *rotation = (__fp16 *)(void *)(blob + rotation_header + 64);
        for (uint32_t o = 0; o < chunk_dim; o++)
            for (uint32_t j = 0; j < group_size; j++)
                rotation[(size_t)o * group_size + j] =
                    (__fp16)table[j * group_size + o % group_size];
    }
    for (uint32_t i = 0; i < band_count; i++) {
        uint32_t rank = bands[i].rank;
        size_t a_header = layout->band_a_offset[i];
        size_t a_tile = (size_t)rank * chunk_dim;
        for (uint32_t c = 0; c < chunks; c++) {
            ane_chunk_header(blob, a_header + (size_t)c * a_stride[i], a_tile);
            int8_t *destination = (int8_t *)(void *)(blob + a_header +
                (size_t)c * a_stride[i] + 64);
            uint32_t base = c * chunk_dim;
            uint32_t span = input_dim > base ?
                (input_dim - base < chunk_dim ? input_dim - base : chunk_dim)
                : 0;
            for (uint32_t r = 0; r < rank; r++)
                memcpy(destination + (size_t)r * chunk_dim,
                       bands[i].a + (size_t)r * input_dim + base, span);
        }
        ane_chunk_header(blob, layout->band_a_scale[i], (size_t)rank * 2);
        __fp16 *a_scale =
            (__fp16 *)(void *)(blob + layout->band_a_scale[i] + 64);
        for (uint32_t r = 0; r < rank; r++)
            a_scale[r] = (__fp16)bands[i].a_scales[r];
        size_t b_bytes = (size_t)bands[i].rows * rank;
        ane_chunk_header(blob, layout->band_b_offset[i], b_bytes);
        memcpy(blob + layout->band_b_offset[i] + 64, bands[i].b, b_bytes);
        ane_chunk_header(blob, layout->band_b_scale[i],
                         (size_t)bands[i].rows * 2);
        __fp16 *b_scale =
            (__fp16 *)(void *)(blob + layout->band_b_scale[i] + 64);
        for (uint32_t n = 0; n < bands[i].rows; n++)
            b_scale[n] = (__fp16)bands[i].b_scales[n];
    }
    *out_bytes = total;
    layout->tile_stride = (unsigned long)stride;
    layout->scale_offset = (unsigned long)scale_header;
    layout->rotation_offset = group_size ? (unsigned long)rotation_header : 0;
    return blob;
}

/* Activation planes are the part of a graph that scales with the module count:
 * at rows 2048 a video VAE block owns 320 MiB of them and the decoder has 36
 * blocks with the same four shapes. Sharing one set per shape turns 11.5 GiB
 * into 320 MiB, which is the difference between a resident decoder and none. */
typedef struct {
    size_t input_bytes;
    size_t output_bytes;
    uint32_t chunks;
    IOSurfaceRef inputs[H3_ANE_MAX_CHUNKS];
    IOSurfaceRef output;
} ane_plane_set;

static ane_plane_set *ane_planes;
static uint32_t ane_plane_count;
static int ane_planes_shared;

static uint64_t ane_plane_bytes(const ane_plane_set *set) {
    return (uint64_t)set->input_bytes * set->chunks + set->output_bytes;
}

static void ane_plane_drop(uint32_t index) {
    ane_plane_set *set = &ane_planes[index];
    for (uint32_t c = 0; c < H3_ANE_MAX_CHUNKS; c++)
        if (set->inputs[c]) CFRelease(set->inputs[c]);
    if (set->output) CFRelease(set->output);
    memset(set, 0, sizeof(*set));
    for (uint32_t i = index; i + 1 < ane_plane_count; i++)
        ane_planes[i] = ane_planes[i + 1];
    ane_plane_count--;
}

static int ane_plane_reserve(char *error, size_t error_size) {
    ane_plane_set *grown = realloc(ane_planes,
        (ane_plane_count + 1) * sizeof(*ane_planes));
    if (!grown) {
        ane_fail(error, error_size, "out of memory growing the ANE plane pool");
        return 0;
    }
    ane_planes = grown;
    memset(&ane_planes[ane_plane_count], 0, sizeof(ane_plane_set));
    return 1;
}

/* Borrow the planes for one shape, creating them on first use. The caller gets
 * its own reference, so freeing a graph never pulls the memory out from under
 * another one; the pool keeps its reference until h3_ane_planes_clear(). */
static int ane_planes_acquire(size_t input_bytes, size_t output_bytes,
                              uint32_t chunks, int shared,
                              IOSurfaceRef *inputs, IOSurfaceRef *output,
                              char *error, size_t error_size) {
    if (!shared) {
        for (uint32_t c = 0; c < chunks; c++) {
            inputs[c] = h3_ane_bridge_surface(input_bytes);
            if (!inputs[c]) {
                ane_fail(error, error_size, "ANE input surface failed");
                return 0;
            }
            memset(IOSurfaceGetBaseAddress(inputs[c]), 0, input_bytes);
        }
        *output = h3_ane_bridge_surface(output_bytes);
        if (!*output) {
            ane_fail(error, error_size, "ANE output surface failed");
            return 0;
        }
        memset(IOSurfaceGetBaseAddress(*output), 0, output_bytes);
        return 1;
    }
    for (uint32_t i = 0; i < ane_plane_count; i++) {
        const ane_plane_set *set = &ane_planes[i];
        if (set->chunks != chunks || set->input_bytes != input_bytes ||
            set->output_bytes != output_bytes) continue;
        for (uint32_t c = 0; c < chunks; c++)
            inputs[c] = (IOSurfaceRef)(void *)CFRetain(set->inputs[c]);
        *output = (IOSurfaceRef)(void *)CFRetain(set->output);
        return 1;
    }
    if (!ane_plane_reserve(error, error_size)) return 0;
    ane_plane_set *set = &ane_planes[ane_plane_count];
    set->input_bytes = input_bytes;
    set->output_bytes = output_bytes;
    set->chunks = chunks;
    for (uint32_t c = 0; c < chunks; c++) {
        set->inputs[c] = h3_ane_bridge_surface(input_bytes);
        if (!set->inputs[c] ||
            !(inputs[c] = (IOSurfaceRef)(void *)CFRetain(set->inputs[c]))) {
            ane_fail(error, error_size, "ANE input surface failed");
            ane_plane_drop(ane_plane_count);
            return 0;
        }
        memset(IOSurfaceGetBaseAddress(inputs[c]), 0, input_bytes);
    }
    set->output = h3_ane_bridge_surface(output_bytes);
    if (!set->output ||
        !(*output = (IOSurfaceRef)(void *)CFRetain(set->output))) {
        ane_fail(error, error_size, "ANE output surface failed");
        ane_plane_drop(ane_plane_count);
        return 0;
    }
    memset(IOSurfaceGetBaseAddress(*output), 0, output_bytes);
    ane_plane_count++;
    return 1;
}

void h3_ane_planes_share(int enable) {
    ane_planes_shared = enable ? 1 : 0;
}

uint64_t h3_ane_planes_bytes(void) {
    uint64_t total = 0;
    for (uint32_t i = 0; i < ane_plane_count; i++)
        total += ane_plane_bytes(&ane_planes[i]);
    return total;
}

/* Only call this once the graphs built from the pool are gone; a graph that is
 * still alive keeps its own reference, so the memory would linger anyway. */
void h3_ane_planes_clear(void) {
    while (ane_plane_count) ane_plane_drop(ane_plane_count - 1);
    free(ane_planes);
    ane_planes = NULL;
}

static h3_ane_linear *ane_build(const char *name, uint8_t *blob,
                                size_t blob_bytes, uint32_t input_dim,
                                uint32_t output_dim, uint32_t rows,
                                uint32_t chunk_dim, uint32_t chunks,
                                const ane_int8_layout *int8_layout,
                                char *error, size_t error_size) {
    h3_ane_linear *linear = calloc(1, sizeof(*linear));
    if (!linear) {
        free(blob);
        ane_fail(error, error_size, "out of memory creating ANE %s", name);
        return NULL;
    }
    linear->input_dim = input_dim;
    linear->padded_dim = chunk_dim * chunks;
    linear->output_dim = output_dim;
    linear->rows = rows;
    linear->plane_rows = (rows + H3_ANE_ROW_MULTIPLE - 1) /
        H3_ANE_ROW_MULTIPLE * H3_ANE_ROW_MULTIPLE;
    linear->chunk_dim = chunk_dim;
    linear->chunks = chunks;
    linear->input_bytes = (size_t)chunk_dim * linear->plane_rows * sizeof(float);
    linear->output_bytes = (size_t)output_dim * linear->plane_rows *
        sizeof(float);
    linear->weight_bytes = blob_bytes;

    if (!ane_planes_acquire(linear->input_bytes, linear->output_bytes, chunks,
                            ane_planes_shared, linear->input_surface,
                            &linear->output_surface, error, error_size)) {
        free(blob);
        h3_ane_linear_free(linear);
        return NULL;
    }
    for (uint32_t c = 0; c < chunks; c++)
        linear->input_base[c] =
            IOSurfaceGetBaseAddress(linear->input_surface[c]);
    linear->output_base = IOSurfaceGetBaseAddress(linear->output_surface);
    @autoreleasepool {
        NSString *text = int8_layout ?
            ane_program_int8(chunk_dim, output_dim, chunks,
                             linear->plane_rows, int8_layout) :
            ane_program(chunk_dim, output_dim, chunks, linear->plane_rows);
        if (getenv("H3_ANE_DUMP_MIL")) {
            char dir[256], path[320];
            snprintf(dir, sizeof(dir), "/tmp/ane_dump_%s", name);
            snprintf(path, sizeof(path), "%s/weights", dir);
            mkdir(dir, 0755);
            mkdir(path, 0755);
            snprintf(path, sizeof(path), "%s/model.mil", dir);
            FILE *dump = fopen(path, "w");
            if (dump) { fputs(text.UTF8String, dump); fclose(dump); }
            snprintf(path, sizeof(path), "%s/weights/weight.bin", dir);
            dump = fopen(path, "w");
            if (dump) { fwrite(blob, 1, blob_bytes, dump); fclose(dump); }
            fprintf(stderr, "dumped %s\n", dir);
        }
        linear->model = h3_ane_model_create(name, text.UTF8String, blob,
                                            blob_bytes, linear->input_surface,
                                            chunks, linear->output_surface,
                                            error, error_size);
    }
    if (!linear->model) {
        h3_ane_linear_free(linear);
        return NULL;
    }
    return linear;
}

static int ane_shape(const char *name, uint32_t input_dim, uint32_t output_dim,
                     uint32_t rows, uint32_t kc, uint32_t *chunk_dim,
                     uint32_t *chunks, char *error, size_t error_size) {
    if (!input_dim || !output_dim || !rows || !kc) {
        ane_fail(error, error_size, "ANE %s has an empty dimension", name);
        return 0;
    }
    if (!h3_ane_linear_available()) {
        ane_fail(error, error_size, "the Neural Engine bridge is unavailable");
        return 0;
    }
    uint32_t count = (input_dim + kc - 1) / kc;
    if (count > H3_ANE_MAX_CHUNKS) {
        ane_fail(error, error_size,
                 "ANE %s needs %u inputs; request binding is only reliable "
                 "through %d", name, count, H3_ANE_MAX_CHUNKS);
        return 0;
    }
    *chunk_dim = kc;
    *chunks = count;
    return 1;
}

h3_ane_linear *h3_ane_linear_create(const char *name, const void *weights,
                                    h3_ane_weight_dtype dtype,
                                    uint32_t input_dim, uint32_t output_dim,
                                    uint32_t rows, uint32_t kc,
                                    char *error, size_t error_size) {
    uint32_t chunk_dim = 0, chunks = 0;
    if (!weights || !ane_shape(name, input_dim, output_dim, rows, kc,
                               &chunk_dim, &chunks, error, error_size))
        return NULL;
    size_t blob_bytes = 0;
    uint8_t *blob = ane_weight_blob(weights, dtype, input_dim, output_dim,
                                    chunk_dim, chunks, &blob_bytes);
    if (!blob) {
        ane_fail(error, error_size, "out of memory building ANE %s weights",
                 name);
        return NULL;
    }
    return ane_build(name, blob, blob_bytes, input_dim, output_dim, rows,
                     chunk_dim, chunks, NULL, error, error_size);
}

h3_ane_linear *h3_ane_linear_create_int8(const char *name,
                                    const int8_t *quantized,
                                    const float *scales,
                                    int convrot_group_size,
                                    uint32_t input_dim, uint32_t output_dim,
                                    uint32_t rows, uint32_t kc,
                                    char *error, size_t error_size) {
    uint32_t chunk_dim = 0, chunks = 0;
    if (!quantized || !scales ||
        !ane_shape(name, input_dim, output_dim, rows, kc, &chunk_dim, &chunks,
                   error, error_size))
        return NULL;
    uint32_t group_size = convrot_group_size > 0 ?
        (uint32_t)convrot_group_size : 0;
    if (group_size &&
        (input_dim % group_size || chunk_dim % group_size ||
         !h3_convrot_hadamard((int)group_size))) {
        ane_fail(error, error_size,
                 "ANE %s cannot rotate: dims %u/%u are not multiples of the "
                 "convrot group %u", name, input_dim, chunk_dim, group_size);
        return NULL;
    }
    size_t blob_bytes = 0;
    ane_int8_layout layout = { .group_size = group_size };
    uint8_t *blob = ane_weight_blob_int8(quantized, scales, group_size, NULL, 0,
                                         input_dim, output_dim, chunk_dim,
                                         chunks, &blob_bytes, &layout);
    if (!blob) {
        ane_fail(error, error_size, "out of memory building ANE %s int8 "
                 "weights", name);
        return NULL;
    }
    return ane_build(name, blob, blob_bytes, input_dim, output_dim, rows,
                     chunk_dim, chunks, &layout, error, error_size);
}

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
                                    char *error, size_t error_size) {
    h3_ane_bypass_band band = { .a = bypass_a, .a_scales = bypass_a_scales,
                                .b = bypass_b, .b_scales = bypass_b_scales,
                                .rank = rank, .row0 = 0,
                                .rows = output_dim };
    if (!bypass_a || !bypass_a_scales || !bypass_b || !bypass_b_scales || !rank) {
        ane_fail(error, error_size, "ANE %s bypass needs both factors", name);
        return NULL;
    }
    return h3_ane_linear_create_int8_bands(name, quantized, scales,
                                           convrot_group_size, &band, 1,
                                           input_dim, output_dim, rows, kc,
                                           error, error_size);
}

h3_ane_linear *h3_ane_linear_create_int8_bands(const char *name,
                                    const int8_t *quantized,
                                    const float *scales,
                                    int convrot_group_size,
                                    const h3_ane_bypass_band *bands,
                                    uint32_t band_count,
                                    uint32_t input_dim, uint32_t output_dim,
                                    uint32_t rows, uint32_t kc,
                                    char *error, size_t error_size) {
    uint32_t chunk_dim = 0, chunks = 0;
    if (!quantized || !scales ||
        !ane_shape(name, input_dim, output_dim, rows, kc, &chunk_dim, &chunks,
                   error, error_size))
        return NULL;
    if (!bands || !band_count || band_count > H3_ANE_MAX_BANDS) {
        ane_fail(error, error_size, "ANE %s needs 1..%d bypass bands, got %u",
                 name, H3_ANE_MAX_BANDS, band_count);
        return NULL;
    }
    uint32_t covered = 0;
    for (uint32_t i = 0; i < band_count; i++) {
        if (!bands[i].a || !bands[i].a_scales || !bands[i].b ||
            !bands[i].b_scales || !bands[i].rank || !bands[i].rows) {
            ane_fail(error, error_size, "ANE %s bypass band %u is incomplete",
                     name, i);
            return NULL;
        }
        covered += bands[i].rows;
    }
    if (covered != output_dim) {
        ane_fail(error, error_size,
                 "ANE %s bypass bands cover %u of %u output rows", name,
                 covered, output_dim);
        return NULL;
    }
    uint32_t group_size = convrot_group_size > 0 ?
        (uint32_t)convrot_group_size : 0;
    if (group_size &&
        (input_dim % group_size || chunk_dim % group_size ||
         !h3_convrot_hadamard((int)group_size))) {
        ane_fail(error, error_size,
                 "ANE %s cannot rotate: dims %u/%u are not multiples of the "
                 "convrot group %u", name, input_dim, chunk_dim, group_size);
        return NULL;
    }
    size_t blob_bytes = 0;
    ane_int8_layout layout = { .group_size = group_size,
                               .band_count = band_count };
    uint8_t *blob = ane_weight_blob_int8(quantized, scales, group_size, bands,
                                         band_count, input_dim, output_dim,
                                         chunk_dim, chunks, &blob_bytes,
                                         &layout);
    if (!blob) {
        ane_fail(error, error_size, "out of memory building ANE %s int8 "
                 "bypass weights", name);
        return NULL;
    }
    return ane_build(name, blob, blob_bytes, input_dim, output_dim, rows,
                     chunk_dim, chunks, &layout, error, error_size);
}

h3_ane_linear *h3_ane_linear_create_chunked(const char *name,
                                    const void *chunks_data, size_t chunk_bytes,
                                    uint32_t input_dim, uint32_t output_dim,
                                    uint32_t rows, uint32_t kc,
                                    char *error, size_t error_size) {
    uint32_t chunk_dim = 0, chunks = 0;
    if (!chunks_data || !ane_shape(name, input_dim, output_dim, rows, kc,
                                   &chunk_dim, &chunks, error, error_size))
        return NULL;
    size_t blob_bytes = 0;
    uint8_t *blob = ane_weight_blob_chunked(chunks_data, chunk_bytes,
                                            output_dim, chunk_dim, chunks,
                                            &blob_bytes);
    if (!blob) {
        ane_fail(error, error_size,
                 "ANE %s chunk payload is %zu bytes, expected %zu", name,
                 chunk_bytes, (size_t)output_dim * chunk_dim * 2 * chunks);
        return NULL;
    }
    return ane_build(name, blob, blob_bytes, input_dim, output_dim, rows,
                     chunk_dim, chunks, NULL, error, error_size);
}

void h3_ane_linear_free(h3_ane_linear *linear) {
    if (!linear) return;
    h3_ane_model_free(linear->model);
    for (uint32_t c = 0; c < linear->chunks; c++)
        if (linear->input_surface[c]) CFRelease(linear->input_surface[c]);
    if (linear->output_surface) CFRelease(linear->output_surface);
    free(linear);
}

uint32_t h3_ane_linear_chunks(const h3_ane_linear *linear) {
    return linear ? linear->chunks : 0;
}

uint32_t h3_ane_linear_chunk_dim(const h3_ane_linear *linear) {
    return linear ? linear->chunk_dim : 0;
}

uint32_t h3_ane_linear_rows(const h3_ane_linear *linear) {
    return linear ? linear->rows : 0;
}

uint32_t h3_ane_linear_plane_rows(const h3_ane_linear *linear) {
    return linear ? linear->plane_rows : 0;
}

uint32_t h3_ane_linear_output_dim(const h3_ane_linear *linear) {
    return linear ? linear->output_dim : 0;
}

float *h3_ane_linear_input(h3_ane_linear *linear, uint32_t chunk) {
    if (!linear || chunk >= linear->chunks) return NULL;
    return linear->input_base[chunk];
}

float *h3_ane_linear_output(h3_ane_linear *linear) {
    return linear ? linear->output_base : NULL;
}

size_t h3_ane_linear_input_bytes(const h3_ane_linear *linear) {
    return linear ? linear->input_bytes : 0;
}

size_t h3_ane_linear_output_bytes(const h3_ane_linear *linear) {
    return linear ? linear->output_bytes : 0;
}

double h3_ane_linear_compile_seconds(const h3_ane_linear *linear) {
    return linear ? h3_ane_model_compile_seconds(linear->model) : 0.0;
}

bool h3_ane_linear_cache_hit(const h3_ane_linear *linear) {
    return linear ? h3_ane_model_cache_hit(linear->model) : false;
}

uint64_t h3_ane_linear_weight_bytes(const h3_ane_linear *linear) {
    return linear ? linear->weight_bytes : 0;
}

int h3_ane_linear_eval(h3_ane_linear *linear, char *error, size_t error_size) {
    if (!linear) {
        ane_fail(error, error_size, "the ANE projection is not loaded");
        return 0;
    }
    return h3_ane_model_eval(linear->model, error, error_size);
}

int h3_ane_linear_unload(h3_ane_linear *linear, char *error, size_t error_size) {
    if (!linear) {
        ane_fail(error, error_size, "the ANE projection is not loaded");
        return 0;
    }
    return h3_ane_model_unload(linear->model, error, error_size);
}

int h3_ane_linear_reload(h3_ane_linear *linear, char *error, size_t error_size) {
    if (!linear) {
        ane_fail(error, error_size, "the ANE projection is not loaded");
        return 0;
    }
    return h3_ane_model_reload(linear->model, error, error_size);
}

struct h3_ane_projection {
    h3_ane_linear *linear;
    h3_gpu_tensor *plane[H3_ANE_MAX_CHUNKS];
    h3_gpu_tensor *result;
    uint32_t input_dim;
    uint32_t output_dim;
    uint32_t rows;
    /* Rows the activations really carry. `rows` sizes the compiled artifact (a
     * bucket, so many shapes share one), while the conv is 1x1 and therefore
     * position-wise: staging may read and write fewer rows than the artifact
     * can hold. Always equal to `rows` unless the caller narrows it. */
    uint32_t activation_rows;
    double sync_seconds;
    double pack_seconds;
    double eval_seconds;
    double unpack_seconds;
    uint64_t calls;
};

uint32_t h3_ane_linear_default_chunk(uint32_t input_dim) {
    uint32_t chunk = 1024;
    while ((input_dim + chunk - 1) / chunk > H3_ANE_MAX_CHUNKS) chunk += 1024;
    return chunk;
}

uint32_t h3_ane_rows_bucket(uint32_t rows) {
    const char *env = getenv("H3_ANE_ROW_BUCKET");
    long step = env ? atol(env) : 256;
    if (step <= 0 || !rows) return rows;
    uint32_t rounded = (rows + (uint32_t)step - 1) / (uint32_t)step *
                       (uint32_t)step;
    return rounded < rows ? rows : rounded;
}

static h3_ane_projection *ane_projection_wrap(h3_gpu *gpu, const char *name,
                                    h3_ane_linear *linear,
                                    uint32_t input_dim, uint32_t output_dim,
                                    uint32_t rows,
                                    char *error, size_t error_size);

h3_ane_projection *h3_ane_projection_create(h3_gpu *gpu, const char *name,
                                    const h3_gpu_tensor *weight,
                                    uint32_t input_dim, uint32_t output_dim,
                                    uint32_t rows, uint32_t kc,
                                    char *error, size_t error_size) {
    const void *values = h3_gpu_tensor_host_pointer(weight);
    if (!values) {
        ane_fail(error, error_size, "ANE %s has no host weight", name);
        return NULL;
    }
    if (h3_gpu_tensor_elements(weight) != (size_t)input_dim * output_dim ||
        h3_gpu_tensor_dtype(weight) != H3_GPU_BF16) {
        ane_fail(error, error_size, "ANE %s weight is not [%u][%u] BF16", name,
                 output_dim, input_dim);
        return NULL;
    }
    h3_ane_linear *linear = h3_ane_linear_create(name, values, H3_ANE_W_BF16,
                                                 input_dim, output_dim, rows,
                                                 kc, error, error_size);
    return ane_projection_wrap(gpu, name, linear, input_dim, output_dim, rows,
                               error, error_size);
}

h3_ane_projection *h3_ane_projection_create_f16(h3_gpu *gpu, const char *name,
                                    const uint16_t *weights,
                                    uint32_t input_dim, uint32_t output_dim,
                                    uint32_t rows, uint32_t kc,
                                    char *error, size_t error_size) {
    h3_ane_linear *linear = h3_ane_linear_create(name, weights, H3_ANE_W_F16,
                                                 input_dim, output_dim, rows,
                                                 kc, error, error_size);
    return ane_projection_wrap(gpu, name, linear, input_dim, output_dim, rows,
                               error, error_size);
}

h3_ane_projection *h3_ane_projection_create_int8_raw(h3_gpu *gpu,
                                    const char *name,
                                    const int8_t *quantized,
                                    const float *scales,
                                    int convrot_group_size,
                                    uint32_t input_dim, uint32_t output_dim,
                                    uint32_t rows, uint32_t kc,
                                    char *error, size_t error_size) {
    h3_ane_linear *linear = h3_ane_linear_create_int8(
        name, quantized, scales, convrot_group_size, input_dim, output_dim,
        rows, kc, error, error_size);
    return ane_projection_wrap(gpu, name, linear, input_dim, output_dim, rows,
                               error, error_size);
}

h3_ane_projection *h3_ane_projection_create_int8(h3_gpu *gpu, const char *name,
                                    const h3_weight_store *store,
                                    const char *weight_name,
                                    uint32_t input_dim, uint32_t output_dim,
                                    uint32_t rows, uint32_t kc,
                                    char *error, size_t error_size) {
    int8_t *quantized = NULL;
    float *scales = NULL;
    int group_size = 0;
    if (!h3_weight_load_int8_raw(store, weight_name, output_dim, input_dim,
                                 &quantized, &scales, &group_size,
                                 error, error_size))
        return NULL;
    h3_ane_projection *projection = h3_ane_projection_create_int8_raw(
        gpu, name, quantized, scales, group_size, input_dim, output_dim, rows,
        kc, error, error_size);
    free(quantized);
    free(scales);
    return projection;
}

h3_ane_projection *h3_ane_projection_create_int8_bands_raw(h3_gpu *gpu,
                                    const char *name,
                                    const int8_t *quantized,
                                    const float *scales,
                                    int convrot_group_size,
                                    const h3_ane_bypass_band *bands,
                                    uint32_t band_count,
                                    uint32_t input_dim, uint32_t output_dim,
                                    uint32_t rows, uint32_t kc,
                                    char *error, size_t error_size) {
    h3_ane_linear *linear = h3_ane_linear_create_int8_bands(
        name, quantized, scales, convrot_group_size, bands, band_count,
        input_dim, output_dim, rows, kc, error, error_size);
    if (!linear) return NULL;
    return ane_projection_wrap(gpu, name, linear, input_dim, output_dim, rows,
                               error, error_size);
}

static h3_ane_projection *ane_projection_wrap(h3_gpu *gpu, const char *name,
                                    h3_ane_linear *linear,
                                    uint32_t input_dim, uint32_t output_dim,
                                    uint32_t rows,
                                    char *error, size_t error_size) {
    if (!linear) return NULL;
    h3_ane_projection *projection = calloc(1, sizeof(*projection));
    if (!projection) {
        h3_ane_linear_free(linear);
        ane_fail(error, error_size, "out of memory creating ANE %s", name);
        return NULL;
    }
    projection->input_dim = input_dim;
    projection->output_dim = output_dim;
    projection->rows = rows;
    projection->activation_rows = rows;
    projection->linear = linear;
    uint32_t chunks = h3_ane_linear_chunks(projection->linear);
    uint32_t chunk_dim = h3_ane_linear_chunk_dim(projection->linear);
    for (uint32_t c = 0; c < chunks; c++) {
        projection->plane[c] = h3_gpu_tensor_wrap_f32(
            gpu, h3_ane_linear_input(projection->linear, c),
            (size_t)chunk_dim * h3_ane_linear_plane_rows(projection->linear));
        if (!projection->plane[c]) {
            ane_fail(error, error_size, "cannot share ANE %s plane %u", name, c);
            h3_ane_projection_free(projection);
            return NULL;
        }
    }
    projection->result = h3_gpu_tensor_wrap_f32(
        gpu, h3_ane_linear_output(projection->linear),
        (size_t)output_dim * h3_ane_linear_plane_rows(projection->linear));
    if (!projection->result) {
        ane_fail(error, error_size, "cannot share ANE %s result", name);
        h3_ane_projection_free(projection);
        return NULL;
    }
    return projection;
}

void h3_ane_projection_free(h3_ane_projection *projection) {
    if (!projection) return;
    for (uint32_t c = 0; c < H3_ANE_MAX_CHUNKS; c++)
        if (projection->plane[c]) h3_gpu_tensor_free(projection->plane[c]);
    if (projection->result) h3_gpu_tensor_free(projection->result);
    h3_ane_linear_free(projection->linear);
    free(projection);
}

static int ane_projection_apply(h3_ane_projection *projection, h3_gpu *gpu,
                               h3_gpu_tensor *output,
                               const h3_gpu_tensor *input,
                               const h3_gpu_tensor *bias, int f32_staging,
                               char *error, size_t error_size) {
    if (!projection || !gpu) {
        ane_fail(error, error_size, "the ANE projection is missing");
        return 0;
    }
    uint32_t chunks = h3_ane_linear_chunks(projection->linear);
    uint32_t chunk_dim = h3_ane_linear_chunk_dim(projection->linear);
    int profile_stages = getenv("H3_ANE_PROFILE_STAGES") != NULL;
    double started = ane_seconds();
    if (profile_stages) {
        if (!h3_gpu_submit(gpu)) {
            ane_fail(error, error_size, "ANE pre-pack sync failed: %s",
                     h3_gpu_error(gpu));
            return 0;
        }
        projection->sync_seconds += ane_seconds() - started;
        if (!h3_gpu_begin(gpu)) {
            ane_fail(error, error_size, "cannot begin ANE pack: %s",
                     h3_gpu_error(gpu));
            return 0;
        }
        started = ane_seconds();
    }
    for (uint32_t c = 0; c < chunks; c++) {
        int packed_plane = f32_staging ?
            h3_gpu_pack_ane_input_f32(gpu, projection->plane[c], input,
                                      projection->activation_rows,
                                      projection->input_dim,
                                      c * chunk_dim, chunk_dim,
                                      h3_ane_linear_plane_rows(
                                          projection->linear)) :
            h3_gpu_pack_ane_input_bf16(gpu, projection->plane[c], input,
                                      projection->activation_rows,
                                      projection->input_dim,
                                      c * chunk_dim, chunk_dim,
                                      h3_ane_linear_plane_rows(
                                          projection->linear));
        if (!packed_plane) {
            ane_fail(error, error_size, "ANE pack failed: %s",
                     h3_gpu_error(gpu));
            return 0;
        }
    }
    if (!h3_gpu_submit(gpu)) {
        ane_fail(error, error_size, "ANE pack submit failed: %s",
                 h3_gpu_error(gpu));
        return 0;
    }
    double packed = ane_seconds();
    if (!h3_ane_linear_eval(projection->linear, error, error_size)) return 0;
    double evaluated = ane_seconds();
    if (!h3_gpu_begin(gpu)) {
        ane_fail(error, error_size, "cannot resume after the ANE: %s",
                 h3_gpu_error(gpu));
        return 0;
    }
    int unpacked = f32_staging ?
        h3_gpu_unpack_ane_output_f32(gpu, output, projection->result, bias,
                                     projection->activation_rows,
                                     projection->output_dim,
                                     h3_ane_linear_plane_rows(
                                         projection->linear)) :
        h3_gpu_unpack_ane_output_bf16(gpu, output, projection->result,
                                      projection->activation_rows,
                                      projection->output_dim,
                                      h3_ane_linear_plane_rows(
                                          projection->linear));
    if (!unpacked) {
        ane_fail(error, error_size, "ANE unpack failed: %s", h3_gpu_error(gpu));
        return 0;
    }
    if (profile_stages) {
        double unpack_started = ane_seconds();
        if (!h3_gpu_submit(gpu)) {
            ane_fail(error, error_size, "ANE unpack submit failed: %s",
                     h3_gpu_error(gpu));
            return 0;
        }
        projection->unpack_seconds += ane_seconds() - unpack_started;
        if (!h3_gpu_begin(gpu)) {
            ane_fail(error, error_size, "cannot resume after ANE profiling: %s",
                     h3_gpu_error(gpu));
            return 0;
        }
    }
    projection->pack_seconds += packed - started;
    projection->eval_seconds += evaluated - packed;
    projection->calls++;
    return 1;
}

int h3_ane_projection_apply(h3_ane_projection *projection, h3_gpu *gpu,
                            h3_gpu_tensor *output, const h3_gpu_tensor *input,
                            char *error, size_t error_size) {
    return ane_projection_apply(projection, gpu, output, input, NULL, 0,
                                error, error_size);
}

int h3_ane_projection_apply_f32(h3_ane_projection *projection, h3_gpu *gpu,
                                h3_gpu_tensor *output,
                                const h3_gpu_tensor *input,
                                const h3_gpu_tensor *bias,
                                char *error, size_t error_size) {
    return ane_projection_apply(projection, gpu, output, input, bias, 1,
                                error, error_size);
}

void h3_ane_projection_timings(const h3_ane_projection *projection,
                               double *pack_seconds, double *eval_seconds,
                               uint64_t *calls) {
    if (pack_seconds) *pack_seconds = projection ? projection->pack_seconds : 0.0;
    if (eval_seconds) *eval_seconds = projection ? projection->eval_seconds : 0.0;
    if (calls) *calls = projection ? projection->calls : 0;
}

void h3_ane_projection_stage_timings(const h3_ane_projection *projection,
                                     double *sync_seconds,
                                     double *pack_seconds,
                                     double *eval_seconds,
                                     double *unpack_seconds,
                                     uint64_t *calls) {
    if (sync_seconds)
        *sync_seconds = projection ? projection->sync_seconds : 0.0;
    if (pack_seconds)
        *pack_seconds = projection ? projection->pack_seconds : 0.0;
    if (eval_seconds)
        *eval_seconds = projection ? projection->eval_seconds : 0.0;
    if (unpack_seconds)
        *unpack_seconds = projection ? projection->unpack_seconds : 0.0;
    if (calls) *calls = projection ? projection->calls : 0;
}

uint64_t h3_ane_projection_weight_bytes(const h3_ane_projection *projection) {
    return projection ? h3_ane_linear_weight_bytes(projection->linear) : 0;
}

double h3_ane_projection_compile_seconds(const h3_ane_projection *projection) {
    return projection ? h3_ane_linear_compile_seconds(projection->linear) : 0.0;
}

bool h3_ane_projection_cache_hit(const h3_ane_projection *projection) {
    return projection ? h3_ane_linear_cache_hit(projection->linear) : false;
}

size_t h3_ane_projection_plane_bytes(const h3_ane_projection *projection) {
    if (!projection) return 0;
    return h3_ane_linear_input_bytes(projection->linear) *
               h3_ane_linear_chunks(projection->linear) +
           h3_ane_linear_output_bytes(projection->linear);
}

int h3_ane_projection_set_activation_rows(h3_ane_projection *projection,
                                          uint32_t rows) {
    if (!projection || rows < 1 ||
        rows > h3_ane_linear_plane_rows(projection->linear))
        return 0;
    projection->activation_rows = rows;
    return 1;
}

int h3_ane_projection_unload(h3_ane_projection *projection, char *error,
                             size_t error_size) {
    if (!projection) {
        ane_fail(error, error_size, "the ANE projection is not loaded");
        return 0;
    }
    return h3_ane_linear_unload(projection->linear, error, error_size);
}

int h3_ane_projection_reload(h3_ane_projection *projection, char *error,
                             size_t error_size) {
    if (!projection) {
        ane_fail(error, error_size, "the ANE projection is not loaded");
        return 0;
    }
    return h3_ane_linear_reload(projection->linear, error, error_size);
}
