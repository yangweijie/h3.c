#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""End-to-end quality A/B for quantized h3.c checkpoints.

Runs the engine once per variant with identical parameters (same seed, same
resolution, same step count, same non-transformer components) and scores every
output frame against a chosen reference.

The point of the check is to separate "the weights changed" from "the engine
changed": every variant directory differs only in FL2VA/transformer.

Reading the numbers
-------------------
SSIM / PSNR / L2 / cosine compare two *different weight sets* at the same step
count, so they measure **trajectory divergence**, not quality.  Diffusion
sampling amplifies any weight perturbation as steps increase, so a variant whose
output is visually identical to the reference still loses SSIM as you raise
--steps (measured: int8-group64 goes 0.891 at 2 steps -> 0.739 at 4 steps while
its texture and saturation stay level with the reference).  Never read a falling
SSIM as "worse quality" without the proxies below.

The quality proxies are absolute, reference-independent frame statistics:
    detail      mean |d(luma)/dx| -- amount of high-frequency structure
    saturation  mean (max-min) over RGB -- colourfulness
    luma_std    contrast
A variant that preserves quality keeps all three close to the reference.  A
degraded one typically loses saturation and either smears detail away or replaces
it with graininess (detail above the reference *with* saturation below it is
noise, not texture -- this is exactly how the int4-group64 arm shows up).

Guard against a degenerate reference: if the reference itself is flat (luma_std
near zero, e.g. an under-sampled single-step blob) the proxies are meaningless
and the comparison cannot be trusted.

Usage:
    python ab_quant_quality.py \
        --variant base=/Volumes/data/work/h3_qad/ab/base \
        --variant int8g64=/Volumes/data/work/h3_qad/ab/int8g64 \
        --variant int4g64=/Volumes/data/work/h3_qad/ab/int4g64 \
        --reference base --out /tmp/h3_ab \
        --width 256 --height 256 --seconds 0.5 --steps 2 --seed 42
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

H3_REPO = Path("/Volumes/data/git/c/h3c")
CLIPPROJ_DIR = "/Volumes/data/.lmstudio/models/Qwen3-VL-4B-Instruct-int8-convrot"
CLIPPROJ_PROJ = "/Volumes/data/.lmstudio/models/ClipProj-MiniMax-H3"

sys.path.insert(0, str(H3_REPO / "benchmark"))
import benchmark as h3bench  # noqa: E402  (reuse the reference SSIM/PSNR/L2 code)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--variant", action="append", required=True,
                   metavar="NAME=MODEL_DIR",
                   help="repeatable; NAME must be a plain identifier")
    p.add_argument("--reference", required=True, help="variant name to score against")
    p.add_argument("--out", type=Path, default=Path("/tmp/h3_ab"))
    p.add_argument("--prompt", default="A red fox walks through fresh snow in a pine forest.")
    p.add_argument("--width", type=int, default=256)
    p.add_argument("--height", type=int, default=256)
    p.add_argument("--seconds", type=float, default=0.5)
    p.add_argument("--steps", type=int, default=2)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--extra", default="--ssd-streaming",
                   help="extra engine flags (default: --ssd-streaming)")
    p.add_argument("--skip-run", action="store_true",
                   help="reuse the mp4s already in --out")
    return p


def parse_variants(pairs: list[str]) -> dict[str, Path]:
    out = {}
    for item in pairs:
        name, _, path = item.partition("=")
        if not name or not path:
            raise SystemExit(f"bad --variant {item!r}, expected NAME=MODEL_DIR")
        out[name] = Path(path)
    return out


def run_variant(name: str, model_dir: Path, out_dir: Path, args) -> dict:
    mp4 = out_dir / f"{name}.mp4"
    if args.skip_run and mp4.exists():
        return {"name": name, "ok": True, "wall_s": None, "reused": True}
    cmd = [str(H3_REPO / "h3"), "-d", str(model_dir), "-p", args.prompt,
           "--width", str(args.width), "--height", str(args.height),
           "--seconds", str(args.seconds), "--steps", str(args.steps),
           "--seed", str(args.seed), "-o", str(mp4)]
    cmd += args.extra.split()
    env = os.environ.copy()
    env["H3_CLIPPROJ_DIR"] = CLIPPROJ_DIR
    env["H3_CLIPPROJ_PROJ"] = CLIPPROJ_PROJ
    print(f"[run] {name}: {' '.join(cmd[1:])}")
    started = time.time()
    proc = subprocess.run(cmd, capture_output=True, text=True,
                          cwd=str(H3_REPO), env=env, timeout=3600)
    wall = time.time() - started
    tail = (proc.stdout + proc.stderr).strip().splitlines()[-3:]
    print(f"[run] {name}: rc={proc.returncode} wall={wall:.1f}s mp4={mp4.exists()}")
    for line in tail:
        print(f"      | {line}")
    return {"name": name, "ok": proc.returncode == 0 and mp4.exists(),
            "wall_s": wall, "reused": False}


def probe_size(mp4: Path) -> tuple[int, int, int]:
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0",
         "-show_entries", "stream=width,height,nb_frames",
         "-of", "csv=p=0", str(mp4)],
        capture_output=True, text=True, timeout=60).stdout.strip().split(",")
    width, height = int(out[0]), int(out[1])
    frames = int(out[2]) if len(out) > 2 and out[2].isdigit() else 0
    return width, height, frames


def extract_frames(mp4: Path) -> np.ndarray:
    """All video frames as float32 [T,H,W,3] in [0,1]."""
    width, height, _ = probe_size(mp4)
    raw = subprocess.run(
        ["ffmpeg", "-v", "error", "-i", str(mp4), "-f", "rawvideo",
         "-pix_fmt", "rgb24", "-"],
        capture_output=True, timeout=300).stdout
    per_frame = width * height * 3
    count = len(raw) // per_frame
    frames = np.frombuffer(raw[:count * per_frame], dtype=np.uint8)
    return frames.reshape(count, height, width, 3).astype(np.float32) / 255.0


def md5(path: Path) -> str:
    digest = hashlib.md5()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def score(reference: np.ndarray, test: np.ndarray) -> dict:
    count = min(len(reference), len(test))
    ssim, psnr, l2, cos, ccos = [], [], [], [], []
    for index in range(count):
        a, b = reference[index], test[index]
        ssim.append(h3bench.compute_ssim(a, b))
        psnr.append(h3bench.compute_psnr(a, b))
        l2.append(h3bench.compute_l2(a, b))
        fa, fb = a.ravel().astype(np.float64), b.ravel().astype(np.float64)
        cos.append(float(fa @ fb / (np.linalg.norm(fa) * np.linalg.norm(fb) + 1e-12)))
        ca, cb = fa - fa.mean(), fb - fb.mean()
        ccos.append(float(ca @ cb / (np.linalg.norm(ca) * np.linalg.norm(cb) + 1e-12)))
    return {
        "frames": count,
        "ssim_mean": float(np.mean(ssim)), "ssim_min": float(np.min(ssim)),
        "psnr_mean": float(np.mean(psnr)), "psnr_min": float(np.min(psnr)),
        "l2_mean": float(np.mean(l2)),
        "cosine_mean": float(np.mean(cos)), "cosine_min": float(np.min(cos)),
        "centered_cosine_mean": float(np.mean(ccos)),
        "centered_cosine_min": float(np.min(ccos)),
    }


def luma(frame: np.ndarray) -> np.ndarray:
    """Luminance, averaged over RGB (matches benchmark.compute_ssim's convention)."""
    return frame.mean(axis=2) if frame.ndim == 3 else frame


def quality_proxies(frames: np.ndarray) -> dict:
    """Reference-independent quality statistics, averaged over all frames."""
    detail, saturation, luma_std = [], [], []
    for frame in frames:
        gray = luma(frame)
        detail.append(float(np.mean(np.abs(np.diff(gray, axis=1)))))
        saturation.append(float(np.mean(frame.max(axis=2) - frame.min(axis=2))))
        luma_std.append(float(np.std(gray)))
    return {"detail": float(np.mean(detail)),
            "saturation": float(np.mean(saturation)),
            "luma_std": float(np.mean(luma_std))}


# Heuristic gate. `detail` above the reference *combined with* `saturation` not
# above it is graininess, not texture, so saturation is checked first and the
# detail window is deliberately a little loose on the high side.
SATURATION_FLOOR = 0.95
DETAIL_LOW, DETAIL_HIGH = 0.80, 1.25


def verdict(proxies: dict, reference: dict) -> str:
    if reference["luma_std"] < 0.05:
        return "reference too flat"
    saturation = proxies["saturation"] / (reference["saturation"] + 1e-12)
    detail = proxies["detail"] / (reference["detail"] + 1e-12)
    if saturation < SATURATION_FLOOR:
        return "degrades (desaturated)"
    if detail > DETAIL_HIGH:
        # Extra high-frequency energy without extra colour is noise, not texture.
        return "degrades (grainy)" if saturation < 1.02 else "degrades (detail)"
    if detail < DETAIL_LOW:
        return "degrades (smoothed)"
    return "preserved"


def main() -> int:
    args = build_parser().parse_args()
    variants = parse_variants(args.variant)
    if args.reference not in variants:
        raise SystemExit(f"--reference {args.reference!r} is not one of {list(variants)}")
    args.out.mkdir(parents=True, exist_ok=True)

    runs = []
    for name, model_dir in variants.items():
        runs.append(run_variant(name, model_dir, args.out, args))

    frames, hashes = {}, {}
    for run in runs:
        mp4 = args.out / f"{run['name']}.mp4"
        if not mp4.exists():
            print(f"[warn] {run['name']}: no output, skipped")
            continue
        frames[run["name"]] = extract_frames(mp4)
        hashes[run["name"]] = md5(mp4)
        print(f"[frames] {run['name']}: {len(frames[run['name']])} frames, "
              f"md5={hashes[run['name']]}")

    if args.reference not in frames:
        raise SystemExit("reference variant produced no video")

    reference = frames[args.reference]
    ref_proxies = quality_proxies(reference)
    report = {"params": vars(args) | {"out": str(args.out)},
              "hashes": hashes, "runs": runs, "scores": {}, "proxies": {},
              "reference_proxies": ref_proxies}
    print(f"\n[divergence vs {args.reference}]  -- grows with --steps, not a quality score")
    print(f"{'variant':12s} {'wall':>7s} {'frames':>7s} {'SSIM':>7s} {'SSIMmin':>8s} "
          f"{'PSNR':>7s} {'L2':>7s} {'cos':>7s} {'ccos':>7s}")
    for name, data in frames.items():
        result = score(reference, data)
        report["scores"][name] = result
        wall = next((r["wall_s"] for r in runs if r["name"] == name), None)
        print(f"{name:12s} {('%.1fs' % wall) if wall else '-':>7s} "
              f"{result['frames']:7d} {result['ssim_mean']:7.4f} "
              f"{result['ssim_min']:8.4f} {result['psnr_mean']:7.2f} "
              f"{result['l2_mean']:7.4f} {result['cosine_mean']:7.5f} "
              f"{result['centered_cosine_mean']:7.5f}")

    print(f"\n[quality proxies]  -- absolute; compare each variant against "
          f"{args.reference}, not against SSIM")
    print(f"{'variant':12s} {'detail':>9s} {'ratio':>7s} {'saturation':>11s} {'ratio':>7s} "
          f"{'luma_std':>9s} {'ratio':>7s}  {'verdict':<20s}")
    for name, data in frames.items():
        proxies = quality_proxies(data)
        report["proxies"][name] = proxies | {
            "detail_ratio": proxies["detail"] / (ref_proxies["detail"] + 1e-12),
            "saturation_ratio": proxies["saturation"] / (ref_proxies["saturation"] + 1e-12),
            "luma_std_ratio": proxies["luma_std"] / (ref_proxies["luma_std"] + 1e-12),
        }
        marked = report["proxies"][name]
        note = "reference" if name == args.reference else verdict(proxies, ref_proxies)
        print(f"{name:12s} {proxies['detail']:9.5f} {marked['detail_ratio']:7.3f} "
              f"{proxies['saturation']:11.4f} {marked['saturation_ratio']:7.3f} "
              f"{proxies['luma_std']:9.4f} {marked['luma_std_ratio']:7.3f}  {note:<20s}")

    if ref_proxies["luma_std"] < 0.05:
        print("\n[warn] reference is nearly flat -- under-sampled output makes every "
              "metric here unreliable; raise --steps before drawing conclusions.")
    print(f"\nreference = {args.reference}   (1.0000 SSIM / 0.0000 L2 = identical)")
    print("verdict rule: preserved iff saturation >= 0.95x and detail within "
          "[0.80, 1.25]x of the reference")

    report_path = args.out / "report.json"
    report_path.write_text(json.dumps(report, indent=2, default=str))
    print(f"report: {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
