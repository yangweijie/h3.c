#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Convert MLX INT8 checkpoint to h3.c ConvRot INT8 format.

h3.c expects:
  - int8 weight [rows, columns] (ConvRot Hadamard already applied)
  - float32 scale [rows] (per-row)
  - Key names: blocks.N.attn.qkv_proj.weight (combined QKV)
               blocks.N.attn.out_proj.weight
               blocks.N.mlp.fc1/fc2.weight

MLX INT8 provides:
  - uint32 packed weight [rows, cols_packed] (4 INT8 per uint32)
  - float32 scales [rows, cols // group_size] (per-group, affine)
  - float32 biases [rows, cols // group_size]
  - Key names: transformer_blocks.N.attn.to_q/to_k/to_v.weight
               transformer_blocks.N.attn.to_out.0.weight
               transformer_blocks.N.ff.net.0.proj / net.2.weight

Conversion:
  1. Unpack uint32 -> INT8 values
  2. Dequantize per-group -> BF16
  3. Remap keys (transformer_blocks -> blocks, merge QKV, rename sub-modules)
  4. Apply ConvRot Hadamard rotation
  5. Re-quantize per-row -> INT8 + scale

Usage:
    python scripts/mlx_int8_to_h3.py \
        --input /Volumes/data/work/h3_qad/h3_mlx/int8 \
        --output /Volumes/data/work/h3_qad/h3_int8
"""

from __future__ import annotations

import argparse
import json
import re
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import safetensors.torch as st
import torch


CONVROT_BLOCK = 256
NUM_HEADS = 56
HEAD_DIM = 128


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def build_hadamard_matrix() -> np.ndarray:
    """Build the 256x256 ConvRot Hadamard matrix (identical to h3.c)."""
    n = CONVROT_BLOCK
    h = np.eye(n, dtype=np.float32)
    stride = 1
    while stride < n:
        span = stride * 4
        for base in range(0, n, span):
            for lane in range(stride):
                for row in range(n):
                    i0 = base + lane
                    i1 = i0 + stride
                    i2 = i0 + 2 * stride
                    i3 = i0 + 3 * stride
                    a, b, c, d = h[row, i0], h[row, i1], h[row, i2], h[row, i3]
                    h[row, i0] = a + b + c - d
                    h[row, i1] = a + b - c + d
                    h[row, i2] = a - b + c + d
                    h[row, i3] = -a + b + c + d
        stride *= 4
    return h / 16.0


def unpack_uint32_int8(weight: np.ndarray) -> np.ndarray:
    """Unpack MLX INT8 (4 values per uint32) -> int8 values as float32."""
    rows, cols_packed = weight.shape
    w = weight.view(np.uint32).reshape(-1)
    unpacked = np.zeros(len(w) * 4, dtype=np.float32)
    for i in range(4):
        shift = i * 8
        unpacked[i::4] = ((w >> shift) & 0xFF).astype(np.int8).astype(np.float32)
    return unpacked.reshape(rows, cols_packed * 4)


def dequantize_mlx_int8(weight: np.ndarray, scales: np.ndarray,
                        biases: np.ndarray | None, group_size: int) -> np.ndarray:
    """Dequantize MLX INT8 -> BF16."""
    w = unpack_uint32_int8(weight)
    rows, cols = w.shape
    w = w.reshape(rows, cols // group_size, group_size)
    s = scales.astype(np.float32)[:, :, np.newaxis]
    if biases is not None:
        b = biases.astype(np.float32)[:, :, np.newaxis]
        w = w - b
    return (w * s).reshape(rows, cols)


def apply_convrot(weight: np.ndarray, hadamard: np.ndarray) -> np.ndarray:
    """Apply ConvRot: W_rot = W @ H^T."""
    rows, cols = weight.shape
    result = np.zeros_like(weight)
    for block_start in range(0, cols, CONVROT_BLOCK):
        block_end = min(block_start + CONVROT_BLOCK, cols)
        if block_end - block_start == CONVROT_BLOCK:
            result[:, block_start:block_end] = weight[:, block_start:block_end] @ hadamard.T
        else:
            result[:, block_start:block_end] = weight[:, block_start:block_end]
    return result


def interleave_qkv_separated(weight: np.ndarray) -> np.ndarray:
    """Re-interleave a [3*H*D, C] q/k/v SEPARATED weight into the INTERLEAVED
    layout h3.c expects for qkv_proj.

    h3.c's de-rotation kernel (h3_gpu_weight_dequant_unrotate_int8 with is_qkv)
    undoes this exact permutation, so the stored weight must be interleaved the
    same way the original ConvRot checkpoint is. The inverse of
    convrot_to_diffusers.convrot_qkv_deinterleave:
        dst = (3*(row//D % H) + (row//D // H)) * D + (row % D)
    maps interleaved row `row` -> separated row `dst`; we invert it here.
    """
    h, d = NUM_HEADS, HEAD_DIM
    out = np.empty_like(weight)
    for row in range(3 * h * d):
        slot = row // d
        b = slot % h
        hd = slot // h
        dim = row % d
        dst = (3 * b + hd) * d + dim
        out[row] = weight[dst]
    return out


def quantize_per_row(weight: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Per-row INT8 quantization."""
    scale = np.max(np.abs(weight), axis=1, keepdims=True) / 127.0
    scale = np.maximum(scale, 1e-8)
    q = np.clip(np.round(weight / scale), -128, 127).astype(np.int8)
    return q, scale.flatten().astype(np.float32)


def remap_key(key: str) -> str:
    """Map MLX key name to h3.c key name."""
    # transformer_blocks.N.attn.to_q/to_k/to_v -> handled separately (QKV merge)
    # transformer_blocks.N.attn.to_out.0.weight -> blocks.N.attn.out_proj.weight
    # transformer_blocks.N.ff.net.0.proj.weight -> blocks.N.mlp.fc1.weight
    # transformer_blocks.N.ff.net.2.weight -> blocks.N.mlp.fc2.weight

    m = re.match(r"^transformer_blocks\.(\d+)\.(.+)$", key)
    if m:
        idx, sub = m.group(1), m.group(2)
        if sub.startswith("attn.to_out.0."):
            return f"blocks.{idx}.attn.out_proj.{sub[len('attn.to_out.0'):]}"
        if sub.startswith("ff.net.0.proj."):
            return f"blocks.{idx}.mlp.fc1.{sub[len('ff.net.0.proj'):]}"
        if sub.startswith("ff.net.2."):
            return f"blocks.{idx}.mlp.fc2.{sub[len('ff.net.2'):]}"
        if sub.startswith("adaln_proj.") or sub.startswith("norm"):
            return f"blocks.{idx}.{sub}"
        # Skip to_q/to_k/to_v (handled by QKV merge)
        if sub.startswith("attn.to_") and sub.endswith(".weight"):
            return ""  # signal: skip, handled separately
        return f"blocks.{idx}.{sub}"

    # Refiner blocks
    m = re.match(r"^token_refiner\.refiner_blocks\.(\d+)\.(.+)$", key)
    if m:
        idx, sub = m.group(1), m.group(2)
        if sub.startswith("attn.to_out.0."):
            return f"token_refiner.blocks.{idx}.attn.out_proj.{sub[len('attn.to_out.0'):]}"
        if sub.startswith("ff.net.0.proj."):
            return f"token_refiner.blocks.{idx}.mlp.fc1.{sub[len('ff.net.0.proj'):]}"
        if sub.startswith("ff.net.2."):
            return f"token_refiner.blocks.{idx}.mlp.fc2.{sub[len('ff.net.2'):]}"
        if sub.startswith("attn.to_") and sub.endswith(".weight"):
            return ""  # skip
        return f"token_refiner.blocks.{idx}.{sub}"

    return key


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    in_dir: Path = args.input
    out_dir: Path = args.output
    out_dir.mkdir(parents=True, exist_ok=True)

    m = json.loads((in_dir / "manifest.json").read_text())
    group_size = m["quantization"]["group_size"]

    print("[build] Hadamard matrix...")
    hadamard = build_hadamard_matrix()

    # Load all tensors from all shards
    print("[load] Reading all shards...")
    all_tensors: dict[str, torch.Tensor] = {}
    for sf in sorted(in_dir.glob("shard-*.safetensors")):
        all_tensors.update(st.load_file(str(sf), device="cpu"))

    # Group QKV by block
    qkv_groups: dict[str, dict[str, torch.Tensor]] = {}  # key_prefix -> {q, k, v, scales, biases}
    non_qkv_keys = []

    for key in sorted(all_tensors.keys()):
        if key.endswith(".scales") or key.endswith(".biases"):
            continue
        # Check if it's a QKV component
        m_qkv = re.match(r"^(transformer_blocks\.\d+\.attn)\.to_(q|k|v)\.weight$", key)
        if m_qkv:
            prefix = m_qkv.group(1)
            qkv_type = m_qkv.group(2)
            qkv_groups.setdefault(prefix, {})[qkv_type] = key
            continue
        # Check refiner QKV
        m_qkv = re.match(r"^(token_refiner\.refiner_blocks\.\d+\.attn)\.to_(q|k|v)\.weight$", key)
        if m_qkv:
            prefix = m_qkv.group(1)
            qkv_type = m_qkv.group(2)
            qkv_groups.setdefault(prefix, {})[qkv_type] = key
            continue
        non_qkv_keys.append(key)

    # Process
    print(f"[info] {len(qkv_groups)} QKV groups, {len(non_qkv_keys)} other weights")

    buf: dict[str, torch.Tensor] = {}
    buf_bytes = 0
    shard_idx = 0
    max_buf_bytes = 2 * 1024**3

    def flush():
        nonlocal buf, buf_bytes, shard_idx
        if not buf:
            return
        name = f"model-{shard_idx + 1:05d}.safetensors"
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

    t0 = time.perf_counter()

    # 1. Merge QKV groups
    for prefix, qkv in sorted(qkv_groups.items()):
        # Load Q, K, V
        q_w = all_tensors[qkv["q"]].numpy()
        s_q = all_tensors[qkv["q"] + ".scales"].numpy()
        b_q = all_tensors.get(qkv["q"] + ".biases")
        b_q = b_q.numpy() if b_q is not None else None

        k_w = all_tensors[qkv["k"]].numpy()
        s_k = all_tensors[qkv["k"] + ".scales"].numpy()
        b_k = all_tensors.get(qkv["k"] + ".biases")
        b_k = b_k.numpy() if b_k is not None else None

        v_w = all_tensors[qkv["v"]].numpy()
        s_v = all_tensors[qkv["v"] + ".scales"].numpy()
        b_v = all_tensors.get(qkv["v"] + ".biases")
        b_v = b_v.numpy() if b_v is not None else None

        # Dequantize each
        q_bf16 = dequantize_mlx_int8(q_w, s_q, b_q, group_size)
        k_bf16 = dequantize_mlx_int8(k_w, s_k, b_k, group_size)
        v_bf16 = dequantize_mlx_int8(v_w, s_v, b_v, group_size)

        # Concatenate (separated [q, k, v]) then re-interleave to the layout
        # h3.c expects for qkv_proj (q0,k0,v0,q1,k1,v1,...). ConvRot acts on the
        # column dim so interleave (rows) and rotation commute.
        merged_sep = np.concatenate([q_bf16, k_bf16, v_bf16], axis=0)
        merged = interleave_qkv_separated(merged_sep)

        # Apply ConvRot
        rotated = apply_convrot(merged, hadamard)

        # Quantize per-row
        q_int8, scale = quantize_per_row(rotated)

        # Map key name: transformer_blocks.N.attn -> blocks.N.attn.qkv_proj
        new_key = re.sub(r"transformer_blocks\.(\d+)\.attn",
                         r"blocks.\1.attn.qkv_proj", prefix)
        new_key = re.sub(r"token_refiner\.refiner_blocks\.(\d+)\.attn",
                         r"token_refiner.blocks.\1.attn.qkv_proj", new_key)

        emit(new_key + ".weight", torch.from_numpy(q_int8))
        emit(new_key + ".weight_scale", torch.from_numpy(scale))

    # 2. Process non-QKV weights
    for key in non_qkv_keys:
        new_key = remap_key(key)
        if not new_key:
            continue  # skip QKV components (already handled)

        tensor = all_tensors[key]

        # Check if it has scales (quantized)
        if key + ".scales" in all_tensors and new_key in [remap_key(k) for k in m["quantized_keys"]]:
            # Dequantize -> ConvRot -> requantize
            w = tensor.numpy()
            s = all_tensors[key + ".scales"].numpy()
            b = all_tensors.get(key + ".biases")
            b = b.numpy() if b is not None else None

            bf16 = dequantize_mlx_int8(w, s, b, group_size)
            rotated = apply_convrot(bf16, hadamard)
            q_int8, scale = quantize_per_row(rotated)
            emit(new_key + ".weight", torch.from_numpy(q_int8))
            emit(new_key + ".weight_scale", torch.from_numpy(scale))
        else:
            # Non-quantized: pass through
            emit(new_key, tensor)

    flush()

    # Write index
    weight_map = {}
    for sf in sorted(out_dir.glob("model-*.safetensors")):
        with st.safe_open(sf, framework="pt", device="cpu") as f:
            for k in f.keys():
                weight_map[k] = sf.name

    (out_dir / "model.safetensors.index.json").write_text(
        json.dumps({"metadata": {}, "weight_map": weight_map}, indent=2))

    dt = time.perf_counter() - t0
    size_gb = sum(f.stat().st_size for f in out_dir.glob("*.safetensors")) / (1024**3)
    print(f"[done] {dt:.1f}s -> {out_dir} ({size_gb:.1f} GB)")


if __name__ == "__main__":
    main()
