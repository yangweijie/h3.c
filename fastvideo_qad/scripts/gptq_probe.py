#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Phase 3b': single-layer GPTQ probe -- the decisive test for the second-order family.

Context: AWQ-style per-channel scaling measurably *hurts* on this model
(`quant_calib_analysis.py`: 0.891x) because ConvRot's fixed Hadamard already
equalises the input diagonals by 5-7 orders of magnitude.  A one-shot whitening
transform is not a valid oracle either (Sigma is far too ill-conditioned, the
Sigma^(1/2)/Sigma^(-1/2) pair amplifies low-energy error without bound -- measured
550x).  That leaves exactly one member of the family untested: the *sequential*
GPTQ update, which handles the conditioning by construction.

This runs real GPTQ (Frantar-style, damped, per-group scales fixed at the group's
original weights) on ONE matrix and compares the layer output error
    err = ||(What - W) X^T||_F / ||W X^T||_F
against plain group quantization.  Cost measured on this machine: 3-5 min/layer
for d_in=5376, so ~30-50 h for all 200 layers -- this probe exists to decide
whether that run is worth starting.

Usage:
    python gptq_probe.py --calib /Volumes/data/work/h3_calib \
        --base <src transformer dir> --block 0 --field qkv --bits 4 --group 64
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np

from mlx_int8_to_h3 import BaseStore
from quant_scheme_sweep import quantize
from quant_calib_analysis import load_dump, hadamard, rotate_rows, FIELD_KEYS


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--calib", type=Path, required=True)
    p.add_argument("--base", type=Path, required=True)
    p.add_argument("--block", type=int, default=0)
    p.add_argument("--field", default="qkv", choices=list(FIELD_KEYS))
    p.add_argument("--bits", type=int, default=4)
    p.add_argument("--group", type=int, default=64)
    p.add_argument("--damp", type=float, default=0.01,
                   help="Hessian damping as a fraction of the mean diagonal")
    p.add_argument("--max-columns", type=int, default=0,
                   help="stop after N columns (0 = all); use for a quick smoke test")
    return p


def column_stats(w: np.ndarray) -> tuple[float, float]:
    levels = float(2 ** 4 - 1)
    low = float(w.min())
    high = float(w.max())
    scale = max((high - low) / levels, 1e-12)
    zero = round(-low / scale)
    return scale, zero


def gptq(w: np.ndarray, x: np.ndarray, bits: int, group: int, damp: float,
         max_columns: int = 0, log_every: int = 512) -> np.ndarray:
    """Sequential error-compensating quantization along the input dim.

    w: [d_out, d_in] weights in the quantization domain (ConvRot-rotated).
    x: [n, d_in] matching inputs (also rotated).
    """
    d_out, d_in = w.shape
    levels = float(2 ** bits - 1)
    started = time.perf_counter()

    hessian = (x.T @ x).astype(np.float32)
    mean_diagonal = float(np.mean(np.diag(hessian)))
    hessian += (damp * mean_diagonal) * np.eye(d_in, dtype=np.float32)
    # GPTQ needs the *upper Cholesky factor of H^-1*, not the full inverse: that
    # factor encodes the sequential elimination, so column j is only compensated
    # against the columns that are still unquantized.  Using the full inverse
    # (an earlier version of this probe did) pushes the compensated values far
    # outside their group's quantization range and measures worse than plain.
    hessian64 = hessian.astype(np.float64)
    lower = np.linalg.cholesky(hessian64)          # H = L L^T
    inverse_lower = np.linalg.solve(lower, np.eye(d_in))   # L^-1
    inverse = inverse_lower.T @ inverse_lower      # H^-1 = L^-T L^-1
    inverse = np.linalg.cholesky(inverse).T        # upper factor U, H^-1 = U^T U
    inverse = inverse.astype(np.float32)

    work = w.copy()
    quantized = np.empty_like(w)
    limit = d_in if max_columns <= 0 else min(max_columns, d_in)
    scale = zero = 0.0
    for j in range(limit):
        if j % group == 0:
            block = w[:, j:min(j + group, d_in)]
            low = float(block.min())
            high = float(block.max())
            scale = max((high - low) / levels, 1e-12)
            zero = round(-low / scale)
        column = work[:, j]
        q = np.clip(np.rint(column / scale) + zero, 0.0, levels)
        quantized[:, j] = (q - zero) * scale
        error = (column - quantized[:, j]) / float(inverse[j, j])
        if j + 1 < d_in:
            work[:, j + 1:] -= error[:, None] * inverse[j, j + 1:][None, :]
        if log_every and (j + 1) % log_every == 0:
            elapsed = time.perf_counter() - started
            print(f"  [{j + 1}/{d_in}] {elapsed:.0f}s "
                  f"(eta {elapsed / (j + 1) * (d_in - j - 1):.0f}s)", flush=True)
    if limit < d_in:
        quantized[:, limit:] = w[:, limit:]
    return quantized


def main() -> int:
    args = build_parser().parse_args()
    suffix, _ = FIELD_KEYS[args.field]
    columns = {"qkv": 5376, "out": 7168, "fc1": 5376, "fc2": 14336}[args.field]
    key = f"blocks.{args.block}.{suffix}.weight"

    x_plain = load_dump(args.calib / f"{args.field}.{args.block:02d}.bin", columns)
    if x_plain is None:
        raise SystemExit(f"no dump for {key}")
    x_rot = rotate_rows(x_plain, hadamard())
    base = BaseStore(args.base)
    w = base.raw(key).astype(np.float32)
    print(f"[probe] {key}  W{w.shape}  X{x_rot.shape}  "
          f"int{args.bits}-g{args.group}  damp={args.damp}")

    def output_error(what: np.ndarray) -> float:
        return float(np.linalg.norm((what - w) @ x_rot.T)
                     / (np.linalg.norm(w @ x_rot.T) + 1e-12))

    plain = quantize(w, args.bits, args.group, False)[0]
    err_plain = output_error(plain)
    print(f"[plain]  output error {err_plain:.5f}  "
          f"weight relRMS {np.linalg.norm(plain - w) / np.linalg.norm(w):.5f}")

    started = time.perf_counter()
    compensated = gptq(w, x_rot, args.bits, args.group, args.damp,
                       args.max_columns)
    elapsed = time.perf_counter() - started
    err_gptq = output_error(compensated)
    print(f"[gptq]   output error {err_gptq:.5f}  "
          f"weight relRMS {np.linalg.norm(compensated - w) / np.linalg.norm(w):.5f}"
          f"   ({elapsed:.0f}s)")
    if args.max_columns:
        print(f"[smoke]  only the first {args.max_columns} columns were quantized; "
              f"the rest pass through, so the numbers are not a full result")
    else:
        print(f"[gain]   plain/gptq = {err_plain / (err_gptq + 1e-12):.3f}x "
              f"reduction in layer output error")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
