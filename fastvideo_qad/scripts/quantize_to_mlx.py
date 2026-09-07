#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Memory-efficient quantization: Diffusers BF16 -> MLX INT4/INT6/INT8.

Processes one tensor at a time, writes incrementally to avoid OOM.
Optionally drops VSA gate projections (not needed for dense inference).

Usage:
    # INT4, drop VSA (~10 GB, dense inference only)
    python scripts/quantize_to_mlx.py \
        --input /Volumes/data/work/h3_qad/h3_diffusers/transformer \
        --output /Volumes/data/work/h3_qad/h3_mlx/int4 \
        --bits 4 --group-size 64 --drop-vsa

    # INT4, keep VSA (~12 GB, enables sparse attention)
    python scripts/quantize_to_mlx.py \
        --input /Volumes/data/work/h3_qad/h3_diffusers/transformer \
        --output /Volumes/data/work/h3_qad/h3_mlx/int4-vsa \
        --bits 4 --group-size 64
"""

from __future__ import annotations

import argparse
import json
import time
from collections import defaultdict
from pathlib import Path

import mlx.core as mx
import numpy as np
import safetensors.torch as st
import torch


# VSA gate projections to drop for dense-only inference
_VSA_GATE_SUFFIX = "attn.to_gate_compress.weight"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input", type=Path, required=True,
                        help="Diffusers BF16 transformer/ directory")
    parser.add_argument("--output", type=Path, required=True,
                        help="Output directory for MLX quantized checkpoint")
    parser.add_argument("--bits", type=int, default=8, choices=[4, 6, 8])
    parser.add_argument("--group-size", type=int, default=64)
    parser.add_argument("--drop-vsa", action="store_true",
                        help="Drop VSA gate projections (saves ~3.6 GB, disables VSA)")
    return parser


def quantize_numpy(weight_np: np.ndarray, bits: int, group_size: int):
    """Quantize a weight matrix using MLX affine quantization.

    Returns (q_int8, scales, biases) or None if not quantizable.
    """
    if len(weight_np.shape) < 2:
        return None
    rows, cols = weight_np.shape
    if cols % group_size != 0:
        return None
    w = mx.array(weight_np)
    q = mx.quantize(w, group_size=group_size, bits=bits, mode="affine")
    mx.eval(q[0], q[1])
    biases = None
    if len(q) == 3 and q[2] is not None:
        mx.eval(q[2])
        biases = np.array(q[2])
    return np.array(q[0]), np.array(q[1]).astype(np.float32), biases


def should_quantize(key: str, drop_vsa: bool) -> bool:
    """Return True if this key should be processed (and not skipped)."""
    # Skip metadata
    if key.endswith(".scale") or key.endswith(".bias"):
        return False
    # Skip VSA gates if requested
    if drop_vsa and key.endswith(_VSA_GATE_SUFFIX):
        return False
    return True


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    in_dir: Path = args.input
    out_dir: Path = args.output
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load index
    index_path = in_dir / "model.safetensors.index.json"
    index = json.loads(index_path.read_text())
    weight_map = index["weight_map"]

    shard_keys: dict[str, list[str]] = defaultdict(list)
    for key, shard in weight_map.items():
        if should_quantize(key, args.drop_vsa):
            shard_keys[shard].append(key)

    # Incremental shard writing
    buf: dict[str, torch.Tensor] = {}
    buf_bytes = 0
    shard_idx = 0
    max_buf_bytes = 2 * 1024**3  # 2 GB buffer
    manifest_keys = {}
    total = quantized = skipped = dropped = 0
    t0 = time.perf_counter()

    def flush():
        nonlocal buf, buf_bytes, shard_idx
        if not buf:
            return
        name = f"shard-{shard_idx:05d}.safetensors"
        st.save_file(buf, str(out_dir / name))
        print(f"  [shard {shard_idx}] {name} ({len(buf)} arrays)")
        shard_idx += 1
        buf = {}
        buf_bytes = 0

    def emit(key: str, tensor: torch.Tensor):
        nonlocal buf_bytes
        buf[key] = tensor
        buf_bytes += tensor.numel() * tensor.element_size()
        if buf_bytes >= max_buf_bytes:
            flush()

    for shard_name, keys in sorted(shard_keys.items()):
        shard_path = in_dir / shard_name
        print(f"[load] {shard_name} ({len(keys)} keys)")
        with st.safe_open(shard_path, framework="pt", device="cpu") as f:
            for key in keys:
                total += 1
                tensor = f.get_tensor(key)
                w_np = tensor.float().numpy()

                result = quantize_numpy(w_np, args.bits, args.group_size)
                if result is not None:
                    q_w, q_scales, q_biases = result
                    emit(key, torch.from_numpy(q_w))
                    emit(key + ".scales", torch.from_numpy(q_scales))
                    if q_biases is not None:
                        emit(key + ".biases", torch.from_numpy(q_biases))
                    manifest_keys[key] = {
                        "bits": args.bits,
                        "group_size": args.group_size,
                        "dequantized_dtype": "bf16",
                        "has_biases": q_biases is not None,
                    }
                    quantized += 1
                else:
                    # 1D or not divisible: keep as-is
                    emit(key, tensor)
                    skipped += 1

                del tensor, w_np

    # Count dropped VSA keys
    for key in weight_map:
        if args.drop_vsa and key.endswith(_VSA_GATE_SUFFIX):
            dropped += 1

    flush()

    # Manifest
    manifest = {
        "format_version": 1,
        "quantization": {"mode": "affine", "bits": args.bits, "group_size": args.group_size},
        "quantized_keys": manifest_keys,
        "num_blocks": 50,
        "vsa_capable": not args.drop_vsa,
        "dropped_vsa_keys": dropped,
        "config": {
            "num_layers": 50,
            "hidden_size": 5376,
            "num_attention_heads": 56,
            "attention_head_dim": 128,
            "ffn_dim": 14336,
            "in_channels": 24,
        },
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))

    dt = time.perf_counter() - t0
    size_gb = sum(f.stat().st_size for f in out_dir.glob("*.safetensors")) / (1024**3)
    print(f"\n[done] {args.bits}-bit, group_size={args.group_size}")
    print(f"  {total} tensors: {quantized} quantized, {skipped} kept as-is, {dropped} VSA dropped")
    print(f"  {dt:.1f}s, {shard_idx} shards -> {out_dir} ({size_gb:.1f} GB)")


if __name__ == "__main__":
    main()
