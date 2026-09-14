#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Phase 1 of the int4 plan: build an h3.c checkpoint with an arbitrary weight
quantization scheme, directly from the source ConvRot checkpoint.

Why a "proxy" checkpoint: h3.c currently has no int4/int6 load path, so a scheme
under evaluation is delivered through the existing per-row INT8 path --
quantize with the candidate scheme, dequantize, then re-quantize per-row INT8 and
write a normal h3 checkpoint.  The extra per-row INT8 step contributes a known
~0.5% noise floor (measured: int8-group64 passes through it still scoring
`preserved` in the A/B), which is far below the effects being compared.

Everything except the four streamed matrices per block is copied byte-for-byte
from the source, so the A/B remains a single-variable comparison.  This also
keeps the ConvRot rotation untouched: the weights stay in rotated space, which is
what h3.c expects, and no permutation is applied (the source stores qkv_proj in
module-major order -- see mlx_int8_to_h3.py note 5).

Usage:
    python quantize_h3_proxy.py --base <src transformer dir> \
        --output /Volumes/data/work/h3_qad/h3_int6g64 \
        --bits 6 --group 64
    # per-matrix override, e.g. keep qkv at 6 bits and push the MLP to 4
    python quantize_h3_proxy.py --base <dir> --output <dir> \
        --bits 4 --group 64 --override mlp.fc1=6:64 --override mlp.fc2=6:64
"""

from __future__ import annotations

import argparse
import json
import shutil
import time
from pathlib import Path

import numpy as np
import safetensors.torch as st
import torch

from mlx_int8_to_h3 import BaseStore, NUM_LAYERS, SHARD_BYTES, requantize_per_row

KINDS = ["attn.qkv_proj", "attn.out_proj", "mlp.fc1", "mlp.fc2"]


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--base", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--bits", type=int, required=True, choices=[4, 6, 8])
    p.add_argument("--group", type=int, default=64)
    p.add_argument("--symmetric", action="store_true",
                   help="drop the zero point (default: asymmetric, like MLX affine)")
    p.add_argument("--override", action="append", default=[], metavar="KIND=BITS:GROUP",
                   help="per-matrix override, KIND is one of " + ", ".join(KINDS))
    p.add_argument("--layers", type=int, default=NUM_LAYERS)
    return p


def parse_overrides(items: list[str]) -> dict[str, tuple[int, int]]:
    out = {}
    for item in items:
        kind, _, spec = item.partition("=")
        bits, _, group = spec.partition(":")
        if kind not in KINDS:
            raise SystemExit(f"unknown --override kind {kind!r}; want one of {KINDS}")
        out[kind] = (int(bits), int(group or 64))
    return out


def quantize(w: np.ndarray, bits: int, group: int, symmetric: bool) -> np.ndarray:
    """Affine group quantization, same math as quant_scheme_sweep.py."""
    rows, columns = w.shape
    view = w.reshape(rows, columns // group, group)
    if symmetric:
        qmax = float(2 ** (bits - 1) - 1)
        scale = np.max(np.abs(view), axis=2, keepdims=True) / qmax
        np.maximum(scale, 1e-12, out=scale)
        quantized = np.clip(np.rint(view / scale), -float(2 ** (bits - 1)), qmax)
        return (quantized * scale).reshape(w.shape)
    levels = float(2 ** bits - 1)
    low = view.min(axis=2, keepdims=True)
    high = view.max(axis=2, keepdims=True)
    scale = (high - low) / levels
    np.maximum(scale, 1e-12, out=scale)
    zero = np.rint(-low / scale)
    quantized = np.clip(np.rint(view / scale) + zero, 0.0, levels)
    return ((quantized - zero) * scale).reshape(w.shape)


def main() -> int:
    args = build_parser().parse_args()
    overrides = parse_overrides(args.override)
    base = BaseStore(args.base)
    output: Path = args.output
    if output.exists():
        raise SystemExit(f"output already exists: {output} (remove it first)")
    output.mkdir(parents=True)

    targets = {}
    for n in range(args.layers):
        for kind in KINDS:
            targets[f"blocks.{n}.{kind}.weight"] = kind

    buffer: dict[str, torch.Tensor] = {}
    buffer_bytes = 0
    shard = 0
    index_map: dict[str, str] = {}
    stats: list[tuple[str, float]] = []
    started = time.perf_counter()

    def flush():
        nonlocal buffer, buffer_bytes, shard
        if not buffer:
            return
        name = f"model-{shard + 1:05d}.safetensors"
        st.save_file(buffer, str(output / name))
        for key in buffer:
            index_map[key] = name
        print(f"  [shard {shard}] {name}: {len(buffer)} tensors, "
              f"{buffer_bytes / 2**30:.2f} GiB ({time.perf_counter() - started:.0f}s)")
        shard += 1
        buffer = {}
        buffer_bytes = 0

    def emit(key: str, tensor: torch.Tensor):
        nonlocal buffer_bytes
        buffer[key] = tensor
        buffer_bytes += tensor.numel() * tensor.element_size()
        if buffer_bytes >= SHARD_BYTES:
            flush()

    replaced = 0
    for key in base.order:
        kind = targets.get(key)
        if kind is None:
            emit(key, base.tensor(key))
            continue

        bits, group = overrides.get(kind, (args.bits, args.group))
        reference = base.raw(key)
        dequantized = quantize(reference, bits, group, args.symmetric)
        quantized, scale = requantize_per_row(dequantized)
        stats.append((key, float(np.sqrt(np.mean((quantized.astype(np.float32)
                                                  * scale.reshape(-1, 1)
                                                  - reference) ** 2))
                                 / (np.sqrt(np.mean(reference ** 2)) + 1e-12))))
        emit(key, torch.from_numpy(quantized))
        emit(key + "_scale", torch.from_numpy(scale.reshape(-1, 1)))
        replaced += 1

    flush()
    (output / "model.safetensors.index.json").write_text(
        json.dumps({"metadata": {}, "weight_map": index_map}, indent=2))
    config = base.files[0].parent / "config.json"
    if config.is_file():
        shutil.copy2(config, output / "config.json")

    errors = [e for _, e in stats]
    scheme = (f"int{args.bits}-g{args.group}-"
              f"{'sym' if args.symmetric else 'asym'}")
    if overrides:
        scheme += " + " + ", ".join(f"{k}=int{b}g{g}" for k, (b, g) in overrides.items())
    print(f"\n[done] scheme {scheme}: {replaced} matrices rebuilt, {shard} shards, "
          f"{time.perf_counter() - started:.0f}s")
    print(f"       relRMS vs source: mean={np.mean(errors):.5f} max={np.max(errors):.5f}")
    print(f"       output: {output} "
          f"({sum(f.stat().st_size for f in output.glob('*.safetensors')) / 2**30:.1f} GiB)")
    print("       NOTE: size shown is the proxy INT8 file, not the scheme's native size; "
          "native bytes come from quant_scheme_sweep.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
