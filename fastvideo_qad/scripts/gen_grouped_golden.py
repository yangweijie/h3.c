#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Phase 2 verification: write golden dequantizations for tests/test_grouped_weights.c.

Re-implements the packed format from the *written checkpoint* (so the F16
rounding of the accumulators is included, exactly as the engine reads it) and
emits, per tensor, the dequantized + un-rotated weight as raw BF16.

The C test loads the same tensors through h3_weight_load_bf16 and compares.
Tolerance is applied on the float values, because the engine may contract
`code * scale + bias` into an FMA while this script evaluates it as mul-then-add;
the difference is one float32 ulp, which can flip a BF16 tie.

Usage:
    python gen_grouped_golden.py --native /Volumes/data/work/h3_qad/h3_int6g128_native \
        --out /Volumes/data/work/h3_qad/golden_int6g128 [--block 0]
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import safetensors.torch as st
import torch

from quant_calib_analysis import hadamard, rotate_rows

KINDS = {"attn.qkv_proj": 5376, "attn.out_proj": 7168,
         "mlp.fc1": 5376, "mlp.fc2": 14336}


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--native", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--block", type=int, default=0)
    return p


def main() -> int:
    args = build_parser().parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    files = {}
    for shard in sorted(args.native.glob("*.safetensors")):
        with st.safe_open(str(shard), framework="pt", device="cpu") as handle:
            for key in handle.keys():
                files[key] = str(shard)
    rotation = hadamard()

    for suffix, columns in KINDS.items():
        base = f"blocks.{args.block}.{suffix}.weight"
        with st.safe_open(files[base], framework="pt", device="cpu") as handle:
            packed = handle.get_tensor(base).numpy()
        with st.safe_open(files[base + "_scale"], framework="pt", device="cpu") as h:
            scale = h.get_tensor(base + "_scale").numpy().astype(np.float32)
        with st.safe_open(files[base + "_bias"], framework="pt", device="cpu") as h:
            bias = h.get_tensor(base + "_bias").numpy().astype(np.float32)

        rows = scale.shape[0]
        group = columns // scale.shape[1]
        assert group == 128, group
        assert packed.shape == (rows, columns * 6 // 8), (packed.shape, rows, columns)

        octets = packed.reshape(rows, -1, 3).astype(np.uint32)
        word = octets[:, :, 0] | (octets[:, :, 1] << 8) | (octets[:, :, 2] << 16)
        codes = np.stack([word & 63, (word >> 6) & 63, (word >> 12) & 63,
                          (word >> 18) & 63], axis=2).reshape(rows, columns)
        dequantized = (codes.astype(np.float32).reshape(rows, -1, group)
                       * scale[:, :, None] + bias[:, :, None]).reshape(rows, columns)
        plain = rotate_rows(dequantized, rotation)
        golden = torch.from_numpy(np.ascontiguousarray(plain)).to(torch.bfloat16)

        path = args.out / f"golden_{suffix.replace('.', '_')}.bin"
        # numpy has no bfloat16, so reinterpret the storage as uint16.
        path.write_bytes(np.asarray(golden.view(torch.uint16)).tobytes())
        print(f"[golden] {base}: {rows}x{columns} bits=6 group={group} "
              f"rms={float(np.sqrt(np.mean(plain.astype(np.float64) ** 2))):.6f} "
              f"-> {path.name} ({path.stat().st_size / 2**20:.1f} MiB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
