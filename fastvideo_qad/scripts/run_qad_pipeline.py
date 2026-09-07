#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Full pipeline: ConvRot BF16 -> Diffusers BF16 -> FastVideo MLX INT8.

Writes intermediate BF16 to SSD, runs FastVideo quantization, then deletes
the intermediate. Only the final INT8 MLX checkpoint remains.

Usage:
    python scripts/run_qad_pipeline.py \\
        --input /Users/jay/h3_sys/MiniMax-H3-Convrot/FL2VA/transformer \\
        --ssd-work /Volumes/data/work/h3_qad \\
        --fastvideo-root /Volumes/data/git/python/FastVideo \\
        --formats int8

Output:
    --ssd-work/h3_diffusers/transformer/   (intermediate, auto-deleted)
    --ssd-work/h3_mlx/int8/                (final INT8 checkpoint)
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import time
from pathlib import Path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--input", type=Path, required=True,
        help="ConvRot transformer/ directory",
    )
    parser.add_argument(
        "--ssd-work", type=Path, required=True,
        help="SSD working directory (must have enough space: ~70 GB)",
    )
    parser.add_argument(
        "--fastvideo-root", type=Path, required=True,
        help="FastVideo repository root",
    )
    parser.add_argument(
        "--formats", default="int8",
        help="Comma-separated MLX formats (int8,int6,int4)",
    )
    parser.add_argument(
        "--keep-intermediate", action="store_true",
        help="Keep the intermediate Diffusers BF16 (default: delete after quantization)",
    )
    return parser


def dir_size_gb(path: Path) -> float:
    total = 0
    for f in path.rglob("*"):
        if f.is_file():
            total += f.stat().st_size
    return total / (1024 ** 3)


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    work: Path = args.ssd_work
    work.mkdir(parents=True, exist_ok=True)

    diffusers_dir = work / "h3_diffusers" / "transformer"
    mlx_dir = work / "h3_mlx"
    script_dir = Path(__file__).parent

    t0 = time.perf_counter()

    # ---- Step 1: ConvRot -> Diffusers BF16 ----
    if diffusers_dir.exists():
        print(f"[skip] Diffusers output already exists: {diffusers_dir}")
    else:
        print(f"[step 1/2] ConvRot BF16 -> Diffusers BF16")
        print(f"  Input:  {args.input}")
        print(f"  Output: {diffusers_dir}")
        diffusers_dir.parent.mkdir(parents=True, exist_ok=True)

        cmd = [
            sys.executable, str(script_dir / "convrot_to_diffusers.py"),
            "--input", str(args.input),
            "--output", str(diffusers_dir),
            "--dtype", "bf16",
        ]
        r = subprocess.run(cmd)
        if r.returncode != 0:
            print(f"[fail] convrot_to_diffusers.py returned {r.returncode}")
            sys.exit(1)

        size_gb = dir_size_gb(diffusers_dir)
        print(f"[done] Diffusers BF16: {size_gb:.1f} GB")

    # ---- Step 2: FastVideo MLX quantization ----
    print(f"[step 2/2] Diffusers BF16 -> MLX INT8")
    print(f"  FastVideo: {args.fastvideo_root}")
    print(f"  Output:    {mlx_dir}")

    convert_script = args.fastvideo_root / "scripts" / "checkpoint_conversion" / "convert_minimax_h3_mlx.py"
    if not convert_script.exists():
        print(f"[fail] FastVideo convert script not found: {convert_script}")
        sys.exit(1)

    cmd = [
        sys.executable, str(convert_script),
        "--model-root", str(diffusers_dir),
        "--out", str(mlx_dir),
        "--formats", args.formats,
    ]
    r = subprocess.run(cmd)
    if r.returncode != 0:
        print(f"[fail] convert_minimax_h3_mlx.py returned {r.returncode}")
        sys.exit(1)

    # ---- Cleanup ----
    if not args.keep_intermediate and diffusers_dir.exists():
        print(f"[cleanup] Deleting intermediate Diffusers BF16: {diffusers_dir}")
        shutil.rmtree(diffusers_dir.parent)  # remove h3_diffusers/
        print("[cleanup] Done")

    mlx_size_gb = dir_size_gb(mlx_dir)
    dt = time.perf_counter() - t0
    print(f"\n[pipeline done] {dt:.0f}s total")
    print(f"[final] MLX INT8 checkpoint: {mlx_dir} ({mlx_size_gb:.1f} GB)")


if __name__ == "__main__":
    main()
