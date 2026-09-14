#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Phase 1a of the int4 plan: sweep quantization schemes on the real DiT weights.

Pure offline weight-error measurement -- no export, no engine run, no side effects.
The reference for every scheme is what the engine runs today: the source
checkpoint's own per-row symmetric INT8 dequantization (`w = i8 * weight_scale`),
in its ConvRot-rotated space.  This is the same yardstick `mlx_int8_to_h3.py
--verify` uses, so numbers are comparable across phases.

Reported per scheme:
    relRMS   root-mean-square error / rms(reference), pooled over the sampled
             matrices and blocks
    B/param  bits/8 + sizeof(scale+zero)/group   (the real storage cost)
    GiB       projected full-DiT size for the 19.27e9 streamed parameters

Sample size is controlled by --layers; the error turned out to be very uniform
across matrices/blocks (see task_plan "Phase 1"), so a handful of blocks is
enough and the full sweep stays in the seconds-to-minutes range.

Usage:
    python quant_scheme_sweep.py --base /Users/jay/h3_sys/MiniMax-H3-Convrot/FL2VA/transformer
    python quant_scheme_sweep.py --base <dir> --bits 4,6,8 --groups 32,64,128 --layers 5
"""

from __future__ import annotations

import argparse
import json

from pathlib import Path

import numpy as np
import safetensors.torch as st

# Streamed DiT parameters (4 matrices x 50 blocks, excluding the unused VSA gate).
STREAMED_PARAMS = 385_415_424 * 50

# matrices the engine streams, and the fraction of streamed bytes each owns
KINDS = [("attn.qkv_proj", 21504, 5376),
         ("attn.out_proj", 5376, 7168),
         ("mlp.fc1", 28672, 5376),
         ("mlp.fc2", 5376, 14336)]


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--base", type=Path, required=True,
                   help="ConvRot transformer dir (or the single .safetensors file)")
    p.add_argument("--bits", default="4,6,8")
    p.add_argument("--groups", default="32,64,128")
    p.add_argument("--layers", type=int, default=3,
                   help="how many DiT blocks to sample (default 3)")
    p.add_argument("--accumulator", default="f32", choices=["f32", "f16"],
                   help="dtype of the scale/zero arrays (default f32)")
    return p


def open_base(path: Path):
    files = [path] if path.is_file() else sorted(path.glob("*.safetensors"))
    if not files:
        raise SystemExit(f"no .safetensors under {path}")
    mapping = {}
    for f in files:
        with st.safe_open(str(f), framework="pt", device="cpu") as fh:
            for k in fh.keys():
                mapping[k] = f
    return files, mapping


def reference_matrix(file_map, key) -> np.ndarray:
    """Source per-row INT8 dequant, exactly as h3.c reconstructs it."""
    with st.safe_open(str(file_map[key]), framework="pt", device="cpu") as fh:
        w = fh.get_tensor(key).numpy().astype(np.float32)
        s = fh.get_tensor(key + "_scale").numpy().astype(np.float32)
    return w * s.reshape(-1, 1)


def group_view(w: np.ndarray, group: int) -> np.ndarray:
    rows, columns = w.shape
    if columns % group:
        raise SystemExit(f"columns {columns} not divisible by group {group}")
    return w.reshape(rows, columns // group, group)


def quantize(w: np.ndarray, bits: int, group: int, symmetric: bool):
    """Return (dequantized, scale, zero) for one of the two affine forms."""
    g = group_view(w, group)
    if symmetric:
        qmax = float(2 ** (bits - 1) - 1)
        qmin = -float(2 ** (bits - 1))
        scale = np.max(np.abs(g), axis=2, keepdims=True) / qmax
        np.maximum(scale, 1e-12, out=scale)
        q = np.clip(np.rint(g / scale), qmin, qmax)
        zero = np.zeros_like(scale)
        dequantized = q * scale
    else:
        levels = float(2 ** bits - 1)
        low = g.min(axis=2, keepdims=True)
        high = g.max(axis=2, keepdims=True)
        scale = (high - low) / levels
        np.maximum(scale, 1e-12, out=scale)
        zero = np.rint(-low / scale)
        q = np.clip(np.rint(g / scale) + zero, 0.0, levels)
        dequantized = (q - zero) * scale
    return dequantized.reshape(w.shape), scale[..., 0], zero[..., 0]


def bytes_per_param(bits: int, group: int, accumulator: str) -> float:
    width = 4 if accumulator == "f32" else 2
    return bits / 8.0 + 2 * width / group


def main() -> int:
    args = build_parser().parse_args()
    bits_list = [int(x) for x in args.bits.split(",")]
    groups = [int(x) for x in args.groups.split(",")]
    files, file_map = open_base(args.base)
    print(f"[base] {len(files)} file(s), {len(file_map)} tensors; "
          f"sampling {args.layers} block(s), streamed params {STREAMED_PARAMS / 1e9:.2f}e9")

    sources = []
    for layer in range(args.layers):
        for suffix, rows, columns in KINDS:
            key = f"blocks.{layer}.{suffix}.weight"
            if key not in file_map:
                raise SystemExit(f"missing {key}")
            sources.append((key, reference_matrix(file_map, key)))

    print(f"\n{'scheme':24s} {'B/param':>8s} {'GiB':>7s} {'vs source':>10s} "
          f"{'best':>8s} {'worst':>8s}  per-matrix relRMS (mean)")
    rows_out = []
    for bits in bits_list:
        for group in groups:
            for symmetric in (True, False):
                for accumulator in (args.accumulator,):
                    errors = []
                    per_kind = {s: [] for s, _, _ in KINDS}
                    for key, reference in sources:
                        dequantized, scale, zero = quantize(reference, bits, group,
                                                            symmetric)
                        err = float(np.sqrt(np.mean((dequantized - reference) ** 2))
                                    / (np.sqrt(np.mean(reference ** 2)) + 1e-12))
                        errors.append(err)
                        kind = key.split(".", 2)[2].rsplit(".", 1)[0]
                        per_kind[kind].append(err)
                    bpp = bytes_per_param(bits, group, accumulator)
                    label = (f"int{bits}-g{group}-"
                             f"{'sym' if symmetric else 'asym'}-{accumulator}")
                    mean = float(np.mean(errors))
                    detail = " ".join(f"{np.mean(per_kind[s]):.4f}" for s, _, _ in KINDS)
                    print(f"{label:24s} {bpp:8.4f} {bpp * STREAMED_PARAMS / 2**30:7.2f} "
                          f"{mean:10.5f} {min(errors):8.5f} {max(errors):8.5f}  {detail}")
                    rows_out.append({"scheme": label, "bits": bits, "group": group,
                                     "symmetric": symmetric, "bytes_per_param": bpp,
                                     "projected_gib": bpp * STREAMED_PARAMS / 2**30,
                                     "rel_rms_mean": mean, "rel_rms_max": float(max(errors))})

    print("\nreference: source per-row symmetric INT8 dequant (what the engine runs now)")
    print(f"source size for the same {STREAMED_PARAMS / 1e9:.2f}e9 params: "
          f"{STREAMED_PARAMS / 2**30:.2f} GiB at 1 B/param")
    print("S2 target: full DiT <= 14 GiB")
    print("\nper-matrix column order: " + " ".join(s for s, _, _ in KINDS))
    out = Path("/tmp/h3_quant_sweep.json")
    out.write_text(json.dumps(rows_out, indent=2))
    print(f"json: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
