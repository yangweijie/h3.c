#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Phase 2 export: a *native* group-quantized checkpoint h3.c can load directly.

Unlike quantize_h3_proxy.py -- which re-quantizes to per-row INT8 so the existing
engine path can run it -- this writes the real packed representation, so the byte
saving is actually realised on disk.

Format (self-describing; the engine derives everything from dtype + shapes):
    {name}.weight        U8   [rows, cols*6/8]       6-bit dense, 4 values per 3 bytes
    {name}.weight_scale  F16  [rows, cols/group]     per-group scale
    {name}.weight_bias   F16  [rows, cols/group]     per-group bias
    dequant:  w = u * scale + bias,   u in [0, 2^bits - 1]
    small-endian bit order; the first value of a chunk occupies the low 6 bits.

Derivation used by the loader (documented in h3_weights.c):
    bits  = stored_bytes_per_row * 8 / columns
    group = columns / scale_columns
`group` must be a multiple of 4 so every group starts on a byte boundary (128 ->
96 bytes per group).  Weights stay in ConvRot-rotated space and qkv_proj stays
module-major, exactly as the source checkpoint stores them, so no permutation and
no rotation is applied here.

Usage:
    python export_h3_int6_native.py --base <src transformer dir> \
        --output /Volumes/data/work/h3_qad/h3_int6g128_native \
        --bits 6 --group 128 --accumulator f16
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

from mlx_int8_to_h3 import BaseStore, NUM_LAYERS, SHARD_BYTES

KINDS = ["attn.qkv_proj", "attn.out_proj", "mlp.fc1", "mlp.fc2"]


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--base", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--bits", type=int, default=6, choices=[4, 6, 8])
    p.add_argument("--group", type=int, default=128)
    p.add_argument("--accumulator", default="f16", choices=["f16", "f32"],
                   help="dtype of the scale/bias arrays")
    p.add_argument("--layers", type=int, default=NUM_LAYERS)
    return p


def quantize_grouped(w: np.ndarray, bits: int, group: int):
    """Return (u, scale, bias) with u in [0, 2^bits-1] and w ~= u*scale + bias."""
    rows, columns = w.shape
    if group % 4 or columns % group:
        raise SystemExit(f"group {group} must divide {columns} and be a multiple of 4")
    view = w.reshape(rows, columns // group, group).astype(np.float32)
    levels = float(2 ** bits - 1)
    low = view.min(axis=2)
    high = view.max(axis=2)
    scale = np.maximum((high - low) / levels, 1e-12)
    zero = np.rint(-low / scale)
    # u = round(w/scale + zero)  =>  w = (u - zero) * scale = u*scale - zero*scale,
    # so the stored bias is -zero*scale, giving the engine a single FMA.
    bias = -zero * scale
    # The stored code is the *unsigned* index u = round(w/scale + zero); biasing
    # by `zero` and then not adding it back is the classic off-by-zero mistake
    # (it clamps the whole group into code 0 and looks like a huge error).
    u = np.clip(np.rint(view / scale[:, :, None]) + zero[:, :, None],
                0.0, levels).astype(np.uint8)
    return u.reshape(rows, columns), scale.astype(np.float32), bias.astype(np.float32)


def pack_6bit(u: np.ndarray) -> np.ndarray:
    """[rows, columns] uint8 values in [0,63] -> [rows, columns*6/8] packed bytes.

    Four 6-bit values become three little-endian bytes, in order.
    """
    rows, columns = u.shape
    if columns % 4:
        raise SystemExit("columns must be a multiple of 4 for 6-bit packing")
    chunks = u.reshape(rows, columns // 4, 4).astype(np.uint32)
    word = (chunks[:, :, 0] | (chunks[:, :, 1] << 6) |
            (chunks[:, :, 2] << 12) | (chunks[:, :, 3] << 18)).astype("<u4")
    # Each 24-bit word is written as three little-endian bytes; the top byte of
    # the uint32 is always zero and is dropped.
    octets = word.view(np.uint8).reshape(rows, columns // 4, 4)
    return np.ascontiguousarray(octets[:, :, :3]).reshape(rows, columns // 4 * 3)


def pack_4bit(u: np.ndarray) -> np.ndarray:
    """[rows, columns] values in [0,15] -> two per byte, low nibble first."""
    rows, columns = u.shape
    pairs = u.reshape(rows, columns // 2, 2).astype(np.uint8)
    return (pairs[:, :, 0] | (pairs[:, :, 1] << 4)).reshape(rows, columns // 2)


def main() -> int:
    args = build_parser().parse_args()
    base = BaseStore(args.base)
    output: Path = args.output
    if output.exists():
        raise SystemExit(f"output already exists: {output} (remove it first)")
    output.mkdir(parents=True)

    accumulator = torch.float16 if args.accumulator == "f16" else torch.float32
    targets = {f"blocks.{n}.{kind}.weight": kind
               for n in range(args.layers) for kind in KINDS}
    # The source checkpoint carries its own per-row `*.weight_scale` (F32 [rows,1])
    # next to every streamed matrix.  Those are NOT targets, so the pass-through
    # branch below would copy them over the group accumulators just written --
    # and since the dictionary is keyed by name, the stale per-row scale wins.
    superseded = {key + "_scale" for key in targets}

    buffer: dict[str, torch.Tensor] = {}
    buffer_bytes = 0
    shard = 0
    index_map: dict[str, str] = {}
    stats: list[float] = []
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

    quantized_count = 0
    for key in base.order:
        if key in superseded:
            continue                      # replaced by the group accumulators
        if key not in targets:
            emit(key, base.tensor(key))
            continue
        reference = base.raw(key)               # rotated space, as stored
        u, scale, bias = quantize_grouped(reference, args.bits, args.group)
        dequantized = (u.astype(np.float32).reshape(reference.shape[0], -1, args.group)
                       * scale[:, :, None] + bias[:, :, None]).reshape(reference.shape)
        stats.append(float(np.sqrt(np.mean((dequantized - reference) ** 2))
                           / (np.sqrt(np.mean(reference ** 2)) + 1e-12)))

        packed = pack_6bit(u) if args.bits == 6 else (
            pack_4bit(u) if args.bits == 4 else u)
        emit(key, torch.from_numpy(packed))
        # F16 rounding of the accumulator is part of the tested format, so it is
        # applied before the error above would be recomputed; the number reported
        # is the F32-accumulator value and is a slight lower bound.
        emit(key + "_scale", torch.from_numpy(
            scale.reshape(reference.shape[0], -1)).to(accumulator))
        emit(key + "_bias", torch.from_numpy(
            bias.reshape(reference.shape[0], -1)).to(accumulator))
        quantized_count += 1

    flush()
    (output / "model.safetensors.index.json").write_text(
        json.dumps({"metadata": {}, "weight_map": index_map}, indent=2))
    config = base.files[0].parent / "config.json"
    if config.is_file():
        shutil.copy2(config, output / "config.json")

    # Re-read a sample of what was actually written.  The in-memory error above
    # cannot see a name collision in the output (the per-row `*_scale` of the
    # source once silently overwrote the group accumulators), so derive the
    # format back from the file exactly as the engine does.
    print("\n[verify] deriving the format back from the written shards")
    reader = BaseStore(output)
    sample = [f"blocks.{n}.{kind}.weight"
              for n in (0, 1, args.layers // 2, args.layers - 1) for kind in KINDS]
    for key in sample:
        if key not in targets and key not in reader.order:
            continue
        packed = reader.tensor(key).numpy()
        scale = reader.tensor(key + "_scale").numpy().astype(np.float32)
        bias = reader.tensor(key + "_bias").numpy().astype(np.float32)
        reference = base.raw(key)
        rows, columns = reference.shape
        packed_columns = packed.shape[1]
        derived_bits = packed_columns * 8 // columns
        derived_group = columns // scale.shape[1]
        octets = packed.reshape(rows, -1, 3).astype(np.uint32)
        word = octets[:, :, 0] | (octets[:, :, 1] << 8) | (octets[:, :, 2] << 16)
        codes = np.stack([word & 63, (word >> 6) & 63, (word >> 12) & 63,
                          (word >> 18) & 63], axis=2).reshape(rows, columns)
        dequantized = (codes.astype(np.float32).reshape(rows, -1, derived_group)
                       * scale[:, :, None] + bias[:, :, None]).reshape(rows, columns)
        relative = float(np.sqrt(np.mean((dequantized - reference) ** 2))
                         / np.sqrt(np.mean(reference ** 2)))
        flag = "" if (derived_bits == args.bits and derived_group == args.group) \
            else "   <-- FORMAT MISMATCH"
        print(f"[verify] {key}: bits={derived_bits} group={derived_group} "
              f"[{rows},{columns}] relRMS={relative:.5f}{flag}")

    total = sum(f.stat().st_size for f in output.glob("*.safetensors"))
    streamed_bits = args.bits / 8.0 + 2 * (2 if args.accumulator == "f16" else 4) / args.group
    print(f"\n[done] int{args.bits} affine group-{args.group} ({args.accumulator} "
          f"accumulators): {quantized_count} matrices packed, {shard} shards, "
          f"{time.perf_counter() - started:.0f}s")
    print(f"       relRMS vs source (F32 accumulators): "
          f"mean={np.mean(stats):.5f} max={np.max(stats):.5f}")
    print(f"       on-disk {total / 2**30:.2f} GiB; streamed-weight model says "
          f"{streamed_bits:.4f} B/param -> "
          f"{streamed_bits * 385_415_424 * 50 / 2**30:.2f} GiB")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
