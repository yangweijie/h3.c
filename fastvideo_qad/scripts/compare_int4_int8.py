#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Compare INT4 vs INT8 checkpoint quality using numpy dequantization.

Handles MLX's INT4 packed storage (2 values per byte) and per-group
affine quantization.

Usage:
    python scripts/compare_int4_int8.py \
        --int4 /Volumes/data/work/h3_qad/h3_mlx/int4 \
        --int8 /Volumes/data/work/h3_qad/h3_mlx/int8
"""

from __future__ import annotations

import argparse
import json
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import safetensors.torch as st
import torch


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--int4", type=Path, required=True)
    parser.add_argument("--int8", type=Path, required=True)
    return parser


def load_tensor(path: Path, key: str):
    with st.safe_open(path, framework="pt", device="cpu") as f:
        return f.get_tensor(key)


def _unpack_uint32(weight: np.ndarray, bits: int) -> np.ndarray:
    """Unpack MLX quantized weights stored as uint32.

    INT4: each uint32 = 8 values
    INT8: each uint32 = 4 values
    """
    rows, cols_packed = weight.shape
    values_per_word = 32 // bits  # 8 for INT4, 4 for INT8

    # Convert uint32 to individual values
    w = weight.astype(np.uint32).reshape(-1)
    unpacked = np.zeros(len(w) * values_per_word, dtype=np.float32)

    for i in range(values_per_word):
        # Extract bits [i*bits : (i+1)*bits]
        shift = i * bits
        if bits == 8:
            mask = 0xFF
        else:  # bits == 4
            mask = 0x0F
        unpacked[i::values_per_word] = ((w >> shift) & mask).astype(np.float32)

    return unpacked.reshape(rows, cols_packed * values_per_word)


def dequantize(weight: np.ndarray, scales: np.ndarray,
               biases: np.ndarray | None, group_size: int, bits: int) -> np.ndarray:
    """Dequantize MLX affine-quantized weight."""
    # Unpack uint32 → float values
    unpacked = _unpack_uint32(weight, bits)
    rows, cols = unpacked.shape

    # Apply per-group affine: w = (w - bias) * scale
    w = unpacked.reshape(rows, cols // group_size, group_size)
    s = scales.astype(np.float32)[:, :, np.newaxis]
    if biases is not None:
        b = biases.astype(np.float32)[:, :, np.newaxis]
        w = w - b
    return (w * s).reshape(rows, cols)


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    int4_dir = args.int4
    int8_dir = args.int8

    m4 = json.loads((int4_dir / "manifest.json").read_text())
    m8 = json.loads((int8_dir / "manifest.json").read_text())

    def build_key_map(d: Path) -> dict[str, Path]:
        km = {}
        for sf in sorted(d.glob("shard-*.safetensors")):
            with st.safe_open(sf, framework="pt", device="cpu") as f:
                for k in f.keys():
                    km[k] = sf
        return km

    print("[index] Building key maps...")
    km4 = build_key_map(int4_dir)
    km8 = build_key_map(int8_dir)

    shared = sorted(set(m4["quantized_keys"].keys()) & set(m8["quantized_keys"].keys()))
    print(f"[info] {len(shared)} shared quantized weights")

    results = []
    t0 = time.perf_counter()

    for i, base_key in enumerate(shared):
        try:
            # INT4
            w4 = load_tensor(km4[base_key], base_key).numpy()
            s4 = load_tensor(km4[base_key], base_key + ".scales").numpy()
            b4 = None
            try:
                b4 = load_tensor(km4[base_key], base_key + ".biases").numpy()
            except Exception:
                pass

            # INT8
            w8 = load_tensor(km8[base_key], base_key).numpy()
            s8 = load_tensor(km8[base_key], base_key + ".scales").numpy()
            b8 = None
            try:
                b8 = load_tensor(km8[base_key], base_key + ".biases").numpy()
            except Exception:
                pass

            gs = m4["quantization"]["group_size"]
            dq4 = dequantize(w4, s4, b4, gs, m4["quantization"]["bits"])
            dq8 = dequantize(w8, s8, b8, gs, m8["quantization"]["bits"])

            if dq4.shape != dq8.shape:
                print(f"  [skip] {base_key}: shape {dq4.shape} vs {dq8.shape}")
                continue

            a, b = dq8, dq4
            abs_err = np.abs(a - b)
            denom = np.abs(a) + 1e-8
            rel_err = float(np.mean(abs_err / denom))
            mse = float(np.mean((a - b) ** 2))
            psnr = float(10 * np.log10(np.max(a ** 2) / (mse + 1e-12))) if mse > 0 else 100.0
            results.append((base_key, rel_err, psnr, list(dq8.shape)))

            if (i + 1) % 50 == 0:
                print(f"  [{i + 1}/{len(shared)}] {base_key}: err={rel_err:.4f} PSNR={psnr:.1f}dB")

        except Exception as e:
            print(f"  [error] {base_key}: {e}")
            continue

    dt = time.perf_counter() - t0
    if not results:
        print("[done] No comparable weights")
        return

    rel_errs = [r[1] for r in results]
    psnrs = [r[2] for r in results]

    print(f"\n{'=' * 60}")
    print(f"[summary] {len(results)} weights compared in {dt:.1f}s")
    print(f"  Relative error: mean={np.mean(rel_errs):.4f}, max={np.max(rel_errs):.4f}")
    print(f"  PSNR: mean={np.mean(psnrs):.1f} dB, min={np.min(psnrs):.1f} dB")

    mean_psnr = np.mean(psnrs)
    if mean_psnr > 50:
        quality = "Excellent (imperceptible)"
    elif mean_psnr > 40:
        quality = "Good (minor, acceptable)"
    elif mean_psnr > 30:
        quality = "Fair (noticeable)"
    else:
        quality = "Poor (significant loss)"
    print(f"  Quality: {quality}")

    worst = sorted(results, key=lambda x: x[1], reverse=True)[:5]
    print(f"\n  Highest error:")
    for key, err, psnr, shape in worst:
        print(f"    {key:50s} err={err:.4f}  PSNR={psnr:.1f}dB")


if __name__ == "__main__":
    main()
