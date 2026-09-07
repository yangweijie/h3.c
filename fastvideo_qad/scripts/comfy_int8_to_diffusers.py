#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Convert ComfyUI INT8 FastH3 model to standard Diffusers format.

Memory-efficient: uses safetensors safe_open (mmap) to avoid loading the
entire 21 GB file into RAM. Processes one tensor at a time.

Usage:
    python scripts/comfy_int8_to_diffusers.py \\
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
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--dtype", choices=("bf16", "fp16", "fp32"), default="bf16")
    parser.add_argument("--deinterleave", action="store_true",
                        help="Deinterleave QKV (ConvRot layout)")
    parser.add_argument("--no-deinterleave", action="store_true",
                        help="QKV already separated")
    return parser


def deinterleave_qkv(weight: np.ndarray) -> np.ndarray:
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
    n = NUM_HEADS * HEAD_DIM
    return weight[0:n].copy(), weight[n:2*n].copy(), weight[2*n:3*n].copy()


_BLOCK_SUB_MAP = {
    "attn.qkv_proj": None,
    "attn.out_proj": "attn.to_out.0",
    "attn.to_gate_compress": "attn.to_gate_compress",
    "mlp.fc1": "ff.net.0.proj",
    "mlp.fc2": "ff.net.2",
}

_REFINER_SUB_MAP = {
    "attn.qkv_proj": None,
    "attn.out_proj": "attn.to_out.0",
    "mlp.fc1": "ff.net.0.proj",
    "mlp.fc2": "ff.net.2",
}

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

_SKIP_SUFFIXES = (".comfy_quant", ".weight_scale")
_SKIP_EXACT = {"adaln_t_table", "rope.inv_freq"}


def convert_key(key: str) -> list[tuple[str, str]]:
    if key in _SKIP_EXACT:
        return []
    if any(key.endswith(s) for s in _SKIP_SUFFIXES):
        return []
    if ".weight_scale" in key:
        return []

    m = re.match(r"^blocks\.(\d+)\.(.+)$", key)
    if m:
        idx, sub = m.group(1), m.group(2)
        # Strip .weight suffix to get the module path
        mod_path = sub[:-7] if sub.endswith(".weight") else sub
        if mod_path == "attn.qkv_proj":
            base = f"transformer_blocks.{idx}.attn"
            return [(f"{base}.to_q.weight", "q"),
                    (f"{base}.to_k.weight", "k"),
                    (f"{base}.to_v.weight", "v")]
        new_sub = _BLOCK_SUB_MAP.get(mod_path, mod_path)
        if new_sub is None:
            new_sub = mod_path
        # Add .weight back
        new_key = f"transformer_blocks.{idx}.{new_sub}.weight" if not new_sub.endswith(".weight") else f"transformer_blocks.{idx}.{new_sub}"
        return [(new_key, "")]

    m = re.match(r"^token_refiner\.blocks\.(\d+)\.(.+)$", key)
    if m:
        idx, sub = m.group(1), m.group(2)
        mod_path = sub[:-7] if sub.endswith(".weight") else sub
        if mod_path == "attn.qkv_proj":
            base = f"token_refiner.refiner_blocks.{idx}.attn"
            return [(f"{base}.to_q.weight", "q"),
                    (f"{base}.to_k.weight", "k"),
                    (f"{base}.to_v.weight", "v")]
        new_sub = _REFINER_SUB_MAP.get(mod_path, mod_path)
        if new_sub is None:
            new_sub = mod_path
        new_key = f"token_refiner.refiner_blocks.{idx}.{new_sub}.weight" if not new_sub.endswith(".weight") else f"token_refiner.refiner_blocks.{idx}.{new_sub}"
        return [(new_key, "")]

    if key in _TOP_LEVEL_MAP:
        return [(_TOP_LEVEL_MAP[key], "")]

    print(f"[warn] Unmapped: {key}")
    return [(key, "")]


def should_process(key: str) -> bool:
    """Return True if this key holds a weight to convert (not metadata/scale)."""
    if key in _SKIP_EXACT:
        return False
    if any(key.endswith(s) for s in _SKIP_SUFFIXES):
        return False
    if ".weight_scale" in key:
        return False
    if ".comfy_quant" in key:
        return False
    return True


def to_numpy(tensor: torch.Tensor, target_dtype: torch.dtype) -> np.ndarray:
    """Convert tensor to numpy, handling BF16 and dequantization."""
    if tensor.dtype == torch.int8:
        # INT8 weight without scale - just cast
        return tensor.float().to(target_dtype).numpy()
    if tensor.dtype == torch.bfloat16:
        return tensor.float().to(target_dtype).numpy()
    return tensor.to(target_dtype).numpy()


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    in_dir: Path = args.input
    out_dir: Path = args.output
    out_dir.mkdir(parents=True, exist_ok=True)
    target_dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[args.dtype]
    deinterleave = args.deinterleave and not args.no_deinterleave

    # Collect all safetensors files
    safetensors_files = sorted(in_dir.glob("*.safetensors"))
    if not safetensors_files:
        raise FileNotFoundError(f"No safetensors in {in_dir}")

    # Build the list of (file, key) pairs to process, filtering metadata
    tasks: list[tuple[Path, str]] = []
    for sf in safetensors_files:
        with st.safe_open(sf, framework="pt", device="cpu") as f:
            for key in f.keys():
                if should_process(key):
                    tasks.append((sf, key))

    print(f"[info] {len(tasks)} weight tensors to convert")

    # Process and write incrementally (avoid accumulating all in RAM)
    t0 = time.perf_counter()

    from collections import defaultdict
    file_keys: dict[Path, list[str]] = defaultdict(list)
    for sf, key in tasks:
        file_keys[sf].append(key)

    # Shard writing state
    shard_buf: dict[str, torch.Tensor] = {}
    shard_bytes = 0
    shard_idx = 0
    max_shard_bytes = 4 * 1024**3  # ~4 GB per shard
    total_written = 0

    def flush_shard():
        nonlocal shard_buf, shard_bytes, shard_idx, total_written
        if not shard_buf:
            return
        name = f"diffusion_pytorch_model-{shard_idx + 1:05d}.safetensors"
        st.save_file(shard_buf, str(out_dir / name))
        total_written += len(shard_buf)
        print(f"  [shard {shard_idx + 1}] {name} ({len(shard_buf)} keys)")
        shard_idx += 1
        shard_buf = {}
        shard_bytes = 0

    def emit(key: str, np_weight: np.ndarray):
        nonlocal shard_bytes
        tensor = torch.from_numpy(np_weight.copy()).to(target_dtype)
        shard_buf[key] = tensor
        shard_bytes += tensor.numel() * tensor.element_size()
        if shard_bytes >= max_shard_bytes:
            flush_shard()

    for sf, keys in file_keys.items():
        print(f"[load] {sf.name} ({len(keys)} keys)")
        with st.safe_open(sf, framework="pt", device="cpu") as f:
            for key in keys:
                tensor = f.get_tensor(key)
                scale_tensor = None
                if key.endswith(".weight"):
                    scale_key = key + "_scale"
                    try:
                        scale_tensor = f.get_tensor(scale_key)
                    except Exception:
                        pass

                # Dequantize / cast to fp32
                if tensor.dtype == torch.int8 and scale_tensor is not None:
                    w = (tensor.float() * scale_tensor.float())
                else:
                    w = tensor.float()

                np_weight = w.numpy()

                for new_key, qkv_suffix in convert_key(key):
                    if qkv_suffix and deinterleave:
                        separated = deinterleave_qkv(np_weight)
                        q, k, v = split_qkv(separated)
                        out_weight = {"q": q, "k": k, "v": v}[qkv_suffix]
                    elif qkv_suffix:
                        n = NUM_HEADS * HEAD_DIM
                        if qkv_suffix == "q":
                            out_weight = np_weight[0:n, :]
                        elif qkv_suffix == "k":
                            out_weight = np_weight[n:2*n, :]
                        else:
                            out_weight = np_weight[2*n:3*n, :]
                    else:
                        out_weight = np_weight

                    emit(new_key, out_weight)

                del tensor, w, np_weight

    flush_shard()  # final shard

    # model_index.json
    weight_map = {}
    total_size = 0
    for sf in sorted(out_dir.glob("diffusion_pytorch_model-*.safetensors")):
        with st.safe_open(sf, framework="pt", device="cpu") as f:
            for key in f.keys():
                weight_map[key] = sf.name
                t = f.get_tensor(key)
                total_size += t.numel() * t.element_size()

    index_out = {
        "metadata": {"total_size": total_size},
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
    size_gb = total_size / (1024**3)
    print(f"[done] {len(tasks)} keys -> {total_written} weights, {shard_idx} shards in {dt:.1f}s")
    print(f"[done] Output: {out_dir} ({size_gb:.1f} GB)")


if __name__ == "__main__":
    main()
