#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Convert MiniMax H3 ConvRot BF16 checkpoint to standard Diffusers format.

ConvRot BF16 weights only have q/k/v interleaving (no Hadamard rotation,
which is int8-only). This script:
  1. De-interleaves qkv_proj -> separated Q/K/V
  2. Renames blocks.N.* -> transformer_blocks.N.*
  3. Renames time_embedder.proj_in/proj_out -> linear_1/linear_2
  4. Maps condition_proj -> context_embedder
  5. Maps final_layer.adaln_proj -> norm_out.linear
  6. Renames token_refiner.blocks.N -> token_refiner.refiner_blocks.N
  7. Saves as standard safetensors for FastVideo MLX conversion

Usage:
    python scripts/convrot_to_diffusers.py \\
        --input /Users/jay/h3_sys/MiniMax-H3-Convrot/FL2VA/transformer \\
        --output /Volumes/data/work/h3_qad/h3_diffusers/transformer
"""

from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path

import numpy as np
import safetensors.torch as st
import torch


# H3 architecture constants (must match h3.c / FastVideo)
NUM_HEADS = 56
HEAD_DIM = 128
NUM_LAYERS = 50
NUM_REFINER_LAYERS = 2
HIDDEN_SIZE = 5376
FFN_DIM = 14336
IN_CHANNELS = 24
AUDIO_CHANNELS = 32


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--input", type=Path, required=True,
                        help="ConvRot transformer/ directory")
    parser.add_argument("--output", type=Path, required=True,
                        help="Output directory (SSD recommended)")
    parser.add_argument("--dtype", choices=("bf16", "fp16", "fp32"), default="bf16")
    return parser


def convrot_qkv_deinterleave(weight: np.ndarray) -> np.ndarray:
    """Undo ConvRot q/k/v row interleaving: interleaved -> separated layout."""
    rows, cols = weight.shape
    out = np.zeros_like(weight)
    for row in range(rows):
        slot = row // HEAD_DIM
        dim = row % HEAD_DIM
        b = slot % NUM_HEADS
        h = slot // NUM_HEADS
        dst_row = (3 * b + h) * HEAD_DIM + dim
        out[dst_row] = weight[row]
    return out


def split_qkv(weight: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Split deinterleaved [3*HEADS*HEAD_DIM, HIDDEN] into Q, K, V."""
    n = NUM_HEADS * HEAD_DIM
    return weight[0:n].copy(), weight[n:2*n].copy(), weight[2*n:3*n].copy()


# Main block sub-path mapping
_BLOCK_SUB_MAP = {
    "attn.qkv_proj.weight": None,  # special: split
    "attn.out_proj.weight": "attn.to_out.0.weight",
    "mlp.fc1.weight": "ff.net.0.proj.weight",
    "mlp.fc2.weight": "ff.net.2.weight",
}

# Refiner block sub-path mapping
_REFINER_SUB_MAP = {
    "attn.qkv_proj.weight": None,  # special: split
    "attn.out_proj.weight": "attn.to_out.0.weight",
    "mlp.fc1.weight": "ff.net.0.proj.weight",
    "mlp.fc2.weight": "ff.net.2.weight",
}

# Top-level key mapping
_TOP_LEVEL_MAP = {
    "time_embedder.proj_in.weight": "time_embedder.linear_1.weight",
    "time_embedder.proj_in.bias": "time_embedder.linear_1.bias",
    "time_embedder.proj_out.weight": "time_embedder.linear_2.weight",
    "time_embedder.proj_out.bias": "time_embedder.linear_2.bias",
    "condition_proj.weight": "context_embedder.weight",
    "condition_proj.bias": "context_embedder.bias",
    "final_layer.adaln_proj.linear.weight": "norm_out.linear.weight",
    "final_layer.adaln_proj.linear.bias": "norm_out.linear.bias",
    "video_patch_proj.weight": "video_patch_proj.weight",
    "video_patch_proj.bias": "video_patch_proj.bias",
    "audio_patch_proj.weight": "audio_patch_proj.weight",
    "audio_patch_proj.bias": "audio_patch_proj.bias",
    "final_layer.norm.weight": "final_layer.norm.weight",
    "final_layer.video_out.weight": "final_layer.video_out.weight",
    "final_layer.video_out.bias": "final_layer.video_out.bias",
    "final_layer.audio_out.weight": "final_layer.audio_out.weight",
    "final_layer.audio_out.bias": "final_layer.audio_out.bias",
    "token_refiner.final_norm.weight": "token_refiner.final_norm.weight",
}

_SKIP_KEYS = {"rope.inv_freq"}


def convert_key(key: str) -> list[tuple[str, str]]:
    """Map ConvRot key -> list of (new_key, qkv_suffix)."""
    if key in _SKIP_KEYS:
        return []

    # Main blocks: blocks.N.<sub> -> transformer_blocks.N.<sub>
    m = re.match(r"^blocks\.(\d+)\.(.+)$", key)
    if m:
        idx, sub = m.group(1), m.group(2)
        if sub == "attn.qkv_proj.weight":
            base = f"transformer_blocks.{idx}.attn"
            return [(f"{base}.to_q.weight", "q"),
                    (f"{base}.to_k.weight", "k"),
                    (f"{base}.to_v.weight", "v")]
        new_sub = _BLOCK_SUB_MAP.get(sub, sub)
        return [(f"transformer_blocks.{idx}.{new_sub}", "")]

    # Refiner: token_refiner.blocks.N.<sub> -> token_refiner.refiner_blocks.N.<sub>
    m = re.match(r"^token_refiner\.blocks\.(\d+)\.(.+)$", key)
    if m:
        idx, sub = m.group(1), m.group(2)
        if sub == "attn.qkv_proj.weight":
            base = f"token_refiner.refiner_blocks.{idx}.attn"
            return [(f"{base}.to_q.weight", "q"),
                    (f"{base}.to_k.weight", "k"),
                    (f"{base}.to_v.weight", "v")]
        new_sub = _REFINER_SUB_MAP.get(sub, sub)
        return [(f"token_refiner.refiner_blocks.{idx}.{new_sub}", "")]

    # Top-level
    return [(_TOP_LEVEL_MAP.get(key, key), "")]


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    in_dir: Path = args.input
    out_dir: Path = args.output
    out_dir.mkdir(parents=True, exist_ok=True)
    target_dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[args.dtype]

    # Load model index
    index_path = in_dir / "model.safetensors.index.json"
    single_shard_mode = not index_path.exists()

    if not single_shard_mode:
        index = json.loads(index_path.read_text())
        weight_map = index.get("weight_map", {})
        shard_keys: dict[str, list[str]] = {}
        for key, shard in weight_map.items():
            shard_keys.setdefault(shard, []).append(key)
        source_shards = [(in_dir / s, keys) for s, keys in shard_keys.items()]
    else:
        source_shards = []
        for sf in sorted(in_dir.glob("*.safetensors")):
            with st.safe_open(sf, framework="pt") as f:
                keys = list(f.keys())
            if keys:
                source_shards.append((sf, keys))

    if not source_shards:
        raise FileNotFoundError(f"No safetensors found in {in_dir}")

    # Convert
    output_weights: dict[str, torch.Tensor] = {}
    total_keys = 0
    t0 = time.perf_counter()

    for shard_file, keys in source_shards:
        print(f"[load] {shard_file.name}")
        tensors = st.load_file(str(shard_file), device="cpu")
        for key in keys:
            total_keys += 1
            w = tensors[key]
            # BFloat16 has no numpy() support; cast via fp32
            if w.dtype == torch.bfloat16:
                w = w.to(torch.float32)
            weight = w.numpy()
            for new_key, qkv_suffix in convert_key(key):
                if qkv_suffix:
                    separated = convrot_qkv_deinterleave(weight)
                    q, k, v = split_qkv(separated)
                    out_weight = {"q": q, "k": k, "v": v}[qkv_suffix]
                else:
                    out_weight = weight
                output_weights[new_key] = torch.from_numpy(out_weight.copy()).to(target_dtype)

    # Write shards (~4 GB each)
    max_shard_bytes = 4 * 1024**3
    shard_idx = 0
    current: dict[str, torch.Tensor] = {}
    current_bytes = 0

    for key, tensor in output_weights.items():
        current[key] = tensor
        current_bytes += tensor.numel() * tensor.element_size()
        if current_bytes >= max_shard_bytes:
            name = f"diffusion_pytorch_model-{shard_idx + 1:05d}.safetensors"
            st.save_file(current, str(out_dir / name))
            shard_idx += 1
            current = {}
            current_bytes = 0

    if current:
        name = f"diffusion_pytorch_model-{shard_idx + 1:05d}.safetensors"
        st.save_file(current, str(out_dir / name))
        shard_idx += 1

    # model_index.json
    weight_map = {}
    for sf in sorted(out_dir.glob("diffusion_pytorch_model-*.safetensors")):
        with st.safe_open(sf, framework="pt") as f:
            for key in f.keys():
                weight_map[key] = sf.name

    index_out = {
        "metadata": {"total_size": sum(t.numel() * t.element_size() for t in output_weights.values())},
        "weight_map": weight_map,
    }
    (out_dir / "model.safetensors.index.json").write_text(json.dumps(index_out, indent=2))

    # config.json
    config = {
        "_class_name": "MiniMaxH3Transformer3DModel",
        "_diffusers_version": "0.30.0",
        "num_layers": NUM_LAYERS,
        "num_refiner_layers": NUM_REFINER_LAYERS,
        "hidden_size": HIDDEN_SIZE,
        "num_attention_heads": NUM_HEADS,
        "attention_head_dim": HEAD_DIM,
        "ffn_dim": FFN_DIM,
        "in_channels": IN_CHANNELS,
        "audio_in_channels": AUDIO_CHANNELS,
        "patch_size": [1, 2, 2],
        "freq_dim": 256,
        "time_embed_dim": 2688,
        "rope_freq_dim": 16,
        "rope_theta": 10000.0,
        "text_dim": 5120,
        "norm_eps": 1e-5,
        "qk_norm_eps": 1e-5,
        "final_norm_eps": 1e-5,
    }
    (out_dir / "config.json").write_text(json.dumps(config, indent=2))

    dt = time.perf_counter() - t0
    size_gb = sum(t.numel() * t.element_size() for t in output_weights.values()) / (1024**3)
    print(f"[done] {total_keys} keys -> {len(output_weights)} weights, {shard_idx} shards in {dt:.1f}s")
    print(f"[done] Output: {out_dir} ({size_gb:.1f} GB)")


if __name__ == "__main__":
    main()
