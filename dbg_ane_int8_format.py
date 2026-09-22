"""Offline check: is the on-disk ConvRot int8 payload already the format the
Apple Neural Engine wants?

h3_ane_block.m feeds weights through

    constexpr_affine_dequantize(axis=0, quantized_data=int8[N,C],
                                scale=fp16[N], zero_point=int8(0))

so the payload must be (a) one scale per OUTPUT row, (b) a zero point of 0,
(c) scales that survive fp16, (d) rows laid out the way the graph slices them.

Run with the anaconda python (system python3 has no safetensors):

    /Volumes/data/Application/anaconda3/bin/python dbg_ane_int8_format.py \
        /Users/jay/h3_sys/MiniMax-H3-Convrot/FL2VA/transformer
"""

import glob
import os
import sys

import numpy as np
from safetensors import safe_open

BLOCK = "blocks.0."
PROJECTIONS = [
    ("qkv", BLOCK + "attn.qkv_proj.weight", 7168 * 3, 5376),
    ("out", BLOCK + "attn.out_proj.weight", 5376, 7168),
    ("fc1", BLOCK + "mlp.fc1.weight", 14336 * 2, 5376),
    ("fc2", BLOCK + "mlp.fc2.weight", 5376, 14336),
]

FP16_MIN_NORMAL = float(np.finfo(np.float16).tiny)      # 6.10e-5
FP16_MIN_SUB = float(np.finfo(np.float16).smallest_subnormal)  # 5.96e-8


def open_stores(directory):
    files = sorted(glob.glob(os.path.join(directory, "*.safetensors")))
    if not files:
        sys.exit("no safetensors under %s" % directory)
    return [(path, safe_open(path, framework="numpy")) for path in files]


def find(store, name):
    for path, handle in store:
        try:
            return path, handle.get_slice(name)
        except Exception:
            continue
    return None, None


def main(directory):
    stores = open_stores(directory)
    for label, name, rows, columns in PROJECTIONS:
        path, slc = find(stores, name)
        if slc is None:
            print("%-4s MISSING %s" % (label, name))
            continue
        dtype = slc.get_dtype()
        shape = slc.get_shape()
        print("\n%-4s %s" % (label, name))
        print("     file   %s" % os.path.basename(path))
        print("     weight dtype=%s shape=%s  (expect [%d, %d])"
              % (dtype, list(shape), rows, columns))
        if list(shape) != [rows, columns]:
            print("     !! shape mismatch: graph constants assume [out,in]")

        # Scales live next to the weight as <name-without-.weight>.weight_scale
        stem = name[: -len(".weight")]
        _, scale_slc = find(stores, stem + ".weight_scale")
        _, zero_slc = find(stores, stem + ".zero")
        _, bias_slc = find(stores, stem + ".bias")

        if scale_slc is None:
            print("     !! no weight_scale; ANE needs one fp16 scale per row")
            continue
        scales = np.asarray(scale_slc[:], dtype=np.float64).reshape(-1)
        print("     scale  dtype=%s shape=%s -> %d values"
              % (scale_slc.get_dtype(), list(scale_slc.get_shape()),
                 scales.size))
        if scales.size != rows:
            print("     !! scale count %d != output rows %d: NOT per-row"
                  % (scales.size, rows))
        finite = scales[np.isfinite(scales)]
        print("     scale  min=%.3e max=%.3e mean=%.3e"
              % (finite.min(), finite.max(), finite.mean()))
        sub = int((np.abs(finite) < FP16_MIN_SUB).sum())
        tiny = int((np.abs(finite) < FP16_MIN_NORMAL).sum())
        print("     scale  fp16: %d below %0.1e (flush to zero), %d subnormal"
              % (sub, FP16_MIN_SUB, tiny - sub))
        if finite.size:
            rel = np.abs(finite.astype(np.float16).astype(np.float64)
                         - finite) / np.abs(finite)
            print("     scale  fp16 round-trip max relative error %.3e"
                  % rel.max())

        if zero_slc is not None:
            zeros = np.asarray(zero_slc[:]).reshape(-1)
            print("     zero   dtype=%s shape=%s min=%s max=%s unique=%s"
                  % (zero_slc.get_dtype(), list(zero_slc.get_shape()),
                     zeros.min(), zeros.max(),
                     np.unique(zeros).size if zeros.size < 64 else ">64"))
            if np.any(zeros != 0):
                print("     !! nonzero zero_point: ANE graph hardcodes "
                      "zero_point=int8(0); codes must be re-centred")
            else:
                print("     zero   all zero -> affine dequant with "
                      "zero_point=0 is exact")
        else:
            print("     zero   absent (symmetric)")

        if bias_slc is not None:
            print("     bias   present: MIL conv has no bias operand, needs "
                  "a separate add")
        else:
            print("     bias   absent -> conv is sufficient")

        codes = np.asarray(slc[0:1][:], dtype=np.int64) if dtype == "I8" \
            else np.asarray(slc[0:1][:])
        print("     codes  first row min=%s max=%s"
              % (codes.min(), codes.max()))
        if dtype == "I8" and codes.min() <= -128:
            print("     !! uses -128; ANE int8 range is fine but the "
                  "symmetric dequant loses the asymmetric code")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    main(sys.argv[1])
