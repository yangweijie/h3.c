#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Phase 3b: activation-aware quantizer evaluation on real calibration data.

Uses the activation dump produced by the engine (`H3_DUMP_ACT=<dir>`, Phase 3a)
to answer the question that decides whether the whole GPTQ/AWQ family has any
headroom on this model:

    is the *quantization-domain* input covariance already isotropic?

h3.c quantizes the ConvRot-rotated weights and un-rotates at load time, so the
quantizer sees W_rot = W_plain @ H and, by orthogonality, the matching input is
x_rot = x_plain @ H.  The dumped activations are plain (they are what the matmul
consumes), so this script applies the same block-Hadamard rotation before
building the second-moment matrix.

Metric -- this is the important part.  Weight relRMS is *not* monotonic with
end-to-end quality (measured: int6-g128 has a worse weight relRMS than int6-g64
but a better end-to-end verdict).  What actually matters is the layer output
error weighted by the real input distribution:

    err = || (What - W) X^T ||_F / || W X^T ||_F

so every scheme is scored with that, plus the plain weight relRMS for reference.

Schemes:
    plain      group-N affine quantization of W_rot
    awq        quantize W_rot @ diag(s), s_j = sqrt(E[x_j^2]) of x_rot, then
               divide back -- the implementable activation-aware scaling
    whitened   quantize W_rot @ Sigma^(1/2), divide back -- ORACLE upper bound on
               the whole second-order family (needs a dense per-layer transform,
               so it is only here to size the remaining headroom)

Usage:
    python quant_calib_analysis.py --calib /Volumes/data/work/h3_calib \
        --base <src transformer dir> --bits 4 --group 64 --blocks 0,1,2,25,49
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import safetensors.torch as st

from mlx_int8_to_h3 import BaseStore, NUM_HEADS, HEAD_DIM
from quant_scheme_sweep import KINDS, quantize, bytes_per_param

BLOCK = 256
FIELD_KEYS = {  # dump file prefix -> checkpoint key suffix
    "qkv": ("attn.qkv_proj", 21504),
    "out": ("attn.out_proj", 5376),
    "fc1": ("mlp.fc1", 28672),
    "fc2": ("mlp.fc2", 5376),
}


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--calib", type=Path, required=True)
    p.add_argument("--base", type=Path, required=True)
    p.add_argument("--bits", type=int, default=4)
    p.add_argument("--group", type=int, default=64)
    p.add_argument("--blocks", default="0,1,2,25,49")
    p.add_argument("--aniso", action="store_true",
                   help="also report eigenvalue-based anisotropy (O(d^3))")
    p.add_argument("--whiten", action="store_true",
                   help="also evaluate the dense-whitening oracle (expensive)")
    return p


def hadamard(n: int = BLOCK) -> np.ndarray:
    h = np.eye(n, dtype=np.float32)
    stride = 1
    while stride < n:
        span = stride * 4
        for base in range(0, n, span):
            for lane in range(stride):
                for row in range(n):
                    i0 = base + lane
                    i1, i2, i3 = i0 + stride, i0 + 2 * stride, i0 + 3 * stride
                    a, b, c, d = h[row, i0], h[row, i1], h[row, i2], h[row, i3]
                    h[row, i0] = a + b + c - d
                    h[row, i1] = a + b - c + d
                    h[row, i2] = a - b + c + d
                    h[row, i3] = -a + b + c + d
        stride *= 4
    return h / 16.0


def rotate_rows(x: np.ndarray, h: np.ndarray) -> np.ndarray:
    """x_plain @ H, block-wise along the input dim (same op as apply_convrot)."""
    out = np.empty_like(x)
    for start in range(0, x.shape[1], BLOCK):
        out[:, start:start + BLOCK] = x[:, start:start + BLOCK] @ h.T
    return out


def load_dump(path: Path, columns: int) -> np.ndarray | None:
    if not path.is_file():
        return None
    raw = np.fromfile(path, dtype=np.uint16)
    rows = raw.size // columns
    if rows == 0:
        return None
    bits = raw[:rows * columns].astype(np.uint32) << 16
    return bits.view(np.float32).reshape(rows, columns)


def anisotropy(cov: np.ndarray) -> float:
    """max/min eigenvalue ratio of the (already PSD) second-moment matrix.

    Expensive: O(d^3).  Only used with --aniso / --whiten.
    """
    eigenvalues = np.linalg.eigvalsh(cov.astype(np.float64))
    positive = eigenvalues[eigenvalues > eigenvalues.max() * 1e-9]
    return float(positive.max() / positive.min()) if positive.size else float("inf")


def diagonal_ratio(x: np.ndarray) -> float:
    """max/min per-channel second moment -- the cheap proxy for anisotropy, and
    exactly the quantity activation-aware scaling can normalise."""
    power = np.mean(x.astype(np.float64) ** 2, axis=0)
    positive = power[power > 0]
    return float(positive.max() / positive.min()) if positive.size else float("inf")


def main() -> int:
    args = build_parser().parse_args()
    blocks = [int(x) for x in args.blocks.split(",")]
    h = hadamard()
    base = BaseStore(args.base)

    print(f"[scheme] int{args.bits} affine group-{args.group}; "
          f"{len(blocks)} blocks; metric = ||(What-W)X^T||_F / ||WX^T||_F")
    header = (f"{'layer':26s} {'n':>5s} {'diag_plain':>10s} {'diag_rot':>9s} "
              f"{'plain':>9s} {'awq':>9s} {'w relRMS':>9s}")
    if args.aniso:
        header += f" {'eig_plain':>10s} {'eig_rot':>9s}"
    if args.whiten:
        header += f" {'whiten':>9s}"
    print("\n" + header)
    rows_out = []
    for block in blocks:
        for field, (suffix, out_rows) in FIELD_KEYS.items():
            dump = args.calib / f"{field}.{block:02d}.bin"
            key = f"blocks.{block}.{suffix}.weight"
            columns = {"qkv": 5376, "out": 7168, "fc1": 5376, "fc2": 14336}[field]
            x_plain = load_dump(dump, columns)
            if x_plain is None:
                print(f"{key:26s} (no dump)")
                continue
            w = base.raw(key).astype(np.float32)          # rotated, as stored
            x_rot = rotate_rows(x_plain, h)

            # --- the decisive diagnostic -------------------------------- #
            diag_plain = diagonal_ratio(x_plain)
            diag_rot = diagonal_ratio(x_rot)

            # --- output-error metric ------------------------------------ #
            def output_error(what: np.ndarray) -> float:
                reference = w @ x_rot.T
                return float(np.linalg.norm((what - w) @ x_rot.T)
                             / (np.linalg.norm(reference) + 1e-12))

            def weight_error(what: np.ndarray) -> float:
                return float(np.linalg.norm(what - w)
                             / (np.linalg.norm(w) + 1e-12))

            # quantize() returns (dequantized, scale, zero).
            plain = quantize(w, args.bits, args.group, False)[0]
            err_plain = output_error(plain)

            # AWQ: scale columns by the input channel RMS, then undo.
            sigma = np.sqrt(np.mean(x_rot.astype(np.float64) ** 2, axis=0))
            sigma = np.maximum(sigma, 1e-8)
            scaled = w * sigma.reshape(1, -1)
            awq = quantize(scaled, args.bits, args.group, False)[0] / sigma.reshape(1, -1)
            err_awq = output_error(awq)

            line = (f"{key:26s} {x_rot.shape[0]:5d} {diag_plain:10.1f} "
                    f"{diag_rot:9.1f} {err_plain:9.5f} {err_awq:9.5f} "
                    f"{weight_error(plain):9.5f}")
            entries = {"key": key, "n": int(x_rot.shape[0]),
                       "diag_plain": diag_plain, "diag_rot": diag_rot,
                       "err_plain": err_plain, "err_awq": err_awq,
                       "weight_relrms": weight_error(plain)}
            cov = (x_rot.T @ x_rot) / x_rot.shape[0]
            if args.aniso:
                entries["eig_plain"] = anisotropy(
                    (x_plain.T @ x_plain) / x_plain.shape[0])
                entries["eig_rot"] = anisotropy(cov)
                line += (f" {entries['eig_plain']:10.1f} {entries['eig_rot']:9.1f}")
            if args.whiten:
                eigenvalues, eigenvectors = np.linalg.eigh(cov.astype(np.float64))
                root = (eigenvectors * np.sqrt(np.maximum(eigenvalues, 0))
                        ).astype(np.float32)
                inverse_root = (eigenvectors * (1.0 / np.sqrt(
                    np.maximum(eigenvalues, eigenvalues.max() * 1e-9)))
                ).astype(np.float32)
                whitened = quantize(w @ root, args.bits, args.group,
                                    False)[0] @ inverse_root
                entries["err_whitened"] = output_error(whitened)
                line += f" {entries['err_whitened']:9.5f}"
            print(line)
            rows_out.append(entries)

    if rows_out:
        def mean(key):
            values = [r[key] for r in rows_out if key in r]
            return float(np.mean(values)) if values else float("nan")
        print(f"\n[mean] diag_plain={mean('diag_plain'):.1f} "
              f"diag_rot={mean('diag_rot'):.1f} | plain={mean('err_plain'):.5f} "
              f"awq={mean('err_awq'):.5f} "
              + (f"whitened={mean('err_whitened'):.5f} " if args.whiten else "")
              + f"| weight_relRMS={mean('weight_relrms'):.5f}")
        print(f"[bytes] {bytes_per_param(args.bits, args.group, 'f32'):.4f} B/param")
        print(f"[awq gain] {mean('err_plain') / (mean('err_awq') + 1e-12):.3f}x "
              f"reduction in layer output error")
        if args.whiten:
            print(f"[whiten gain] {mean('err_plain') / (mean('err_whitened') + 1e-12):.3f}x "
                  f"(oracle upper bound of the second-order family)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
