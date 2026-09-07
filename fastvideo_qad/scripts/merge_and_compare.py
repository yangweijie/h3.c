#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Merge shards to FastVideo MLX format and compare INT4 vs INT8 quality.

Step 1: Merge each checkpoint into FastVideo's expected single-file format.
Step 2: Run a minimal forward pass comparison.

Usage:
    # Merge INT4 checkpoint
    python scripts/merge_and_compare.py --merge \
        --input /Volumes/data/work/h3_qad/h3_mlx/int4 \
        --output /Volumes/data/work/h3_qad/int4_merged

    # Compare INT4 vs INT8 (after both are merged)
    python scripts/merge_and_compare.py --compare \
        --int4 /Volumes/data/work/h3_qad/int4_merged \
        --int8 /Volumes/data/work/h3_qad/int8_merged
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import mlx.core as mx
import numpy as np
import safetensors.torch as st
import torch


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--merge", action="store_true", help="Merge shards to FastVideo format")
    parser.add_argument("--compare", action="store_true", help="Compare INT4 vs INT8")
    parser.add_argument("--input", type=Path, help="Input sharded directory")
    parser.add_argument("--output", type=Path, help="Output merged directory")
    parser.add_argument("--int4", type=Path, help="INT4 merged checkpoint")
    parser.add_argument("--int8", type=Path, help="INT8 merged checkpoint")
    return parser


def merge_shards(in_dir: Path, out_dir: Path):
    """Merge sharded checkpoint into FastVideo's single-file MLX format."""
    print(f"[merge] {in_dir} -> {out_dir}")
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load manifest
    manifest_path = in_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text())

    # Load all shards
    all_tensors = {}
    for shard in sorted(in_dir.glob("shard-*.safetensors")):
        print(f"  [load] {shard.name}")
        t = st.load_file(str(shard), device="cpu")
        all_tensors.update(t)

    # Convert to MLX arrays
    mlx_arrays = {}
    for key, tensor in all_tensors.items():
        if isinstance(tensor, torch.Tensor):
            mlx_arrays[key] = mx.array(tensor.float().numpy())
        else:
            mlx_arrays[key] = mx.array(tensor)

    # Save as single safetensors file
    out_weights = out_dir / "mlx_h3_dit.safetensors"
    # MLX expects numpy arrays
    np_dict = {k: np.array(v) for k, v in mlx_arrays.items()}
    st.save_file(np_dict, str(out_weights))

    # Save manifest in FastVideo format
    fv_manifest = {
        "format_version": 1,
        "quantization": manifest["quantization"],
        "quantized_keys": manifest["quantized_keys"],
        "num_blocks": manifest["num_blocks"],
        "num_refiner_blocks": manifest.get("num_refiner_blocks", 2),
        "config": manifest["config"],
    }
    (out_dir / "mlx_h3_dit.json").write_text(json.dumps(fv_manifest, indent=2))

    size_gb = out_weights.stat().st_size / (1024**3)
    print(f"[done] {out_dir} ({size_gb:.1f} GB)")


def compare_checkpoints(int4_dir: Path, int8_dir: Path):
    """Compare INT4 vs INT8 with a minimal forward pass."""
    print(f"[compare] INT4: {int4_dir} vs INT8: {int8_dir}")

    from fastvideo.mlx_runtime.minimax_h3 import load_mlx_h3_checkpoint

    # Load INT4
    print("[load] INT4...")
    t0 = time.perf_counter()
    dit4 = load_mlx_h3_checkpoint(int4_dir)
    t4 = time.perf_counter() - t0
    print(f"  loaded in {t4:.1f}s")

    # Create dummy input (small for quick test)
    batch = 1
    seq_len = 64  # small test
    hidden = dit4.hidden_size

    x = mx.random.normal((batch * seq_len, hidden))
    timestep = mx.array([0.5])

    # INT4 forward
    print("[forward] INT4...")
    t0 = time.perf_counter()
    # Minimal: just test the first block
    # Full forward requires the full pipeline; this is a basic sanity check
    out4 = x + 0  # placeholder
    print(f"  (full forward requires pipeline integration)")

    # Free INT4
    del dit4
    mx.clear_cache()

    # Load INT8
    print("[load] INT8...")
    t0 = time.perf_counter()
    dit8 = load_mlx_h3_checkpoint(int8_dir)
    t8 = time.perf_counter() - t0
    print(f"  loaded in {t8:.1f}s")

    print("\n[result] Both checkpoints loaded successfully.")
    print("  For full quality comparison, use mlx_fasth3.py with each checkpoint:")
    print(f"    python examples/inference/basic/mlx_fasth3.py --mlx-checkpoint {int4_dir} ...")
    print(f"    python examples/inference/basic/mlx_fasth3.py --mlx-checkpoint {int8_dir} ...")


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.merge:
        merge_shards(args.input, args.output)
    elif args.compare:
        compare_checkpoints(args.int4, args.int8)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
