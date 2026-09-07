#!/usr/bin/env python3
"""h3 benchmark: parameter sweep + quality scoring + HTML report.

Runs h3 with a matrix of speed/quality parameters, scores each output against a
high-quality reference using SSIM/PSNR/L2, and writes a self-contained HTML report
with embedded frame comparisons.

Usage:
    cd /Volumes/data/git/c/h3.c/benchmark
    python3 benchmark.py                # run full sweep + report
    python3 benchmark.py --skip-gen     # reuse existing videos, just re-score
"""
import argparse
import base64
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #
H3_DIR = "/Volumes/data/git/c/h3.c"
H3_BIN = os.path.join(H3_DIR, "h3")
MODEL_DIR = "/Users/jay/h3_sys/MiniMax-H3-Convrot"
# Text encoder is external (not inside FL2VA/); pointed to via env vars.
CLIPPROJ_DIR = "/Volumes/data/.lmstudio/models/Qwen3-VL-4B-Instruct-int8-convrot"
CLIPPROJ_PROJ = "/Volumes/data/.lmstudio/models/ClipProj-MiniMax-H3"
PROMPT = "A red fox"
SEED = 42
WIDTH, HEIGHT = 256, 256
SECONDS = 1
FRAME_TIME = 0.8          # extract frame at t=0.8s for comparison
OUTPUT_DIR = "/tmp/h3_benchmark"

# Reference: highest quality we will compare against.
REF_PARAMS = {"steps": 20, "layers": 50, "reuse": 1, "render": None}

# Test matrix ORDERED by expected peak memory (low to high).
# Memory grows with: more steps, more layers, less reuse, larger render.
# We run lowest-memory tests first and stop if any exceeds MEMORY_LIMIT_GB.
# Round 1 (original) + Round 2 (core-reuse / token-reduction, step=4 layer=40).
TEST_MATRIX = [
    # --- Round 1 (existing videos, reused via --skip-gen) ---
    {"label": "s4-l40-r3-R192", "steps": 4, "layers": 40, "reuse": 3, "render": 192},
    {"label": "s4-l40-r3",      "steps": 4, "layers": 40, "reuse": 3, "render": None},
    {"label": "s4-l45-r2-R192", "steps": 4, "layers": 45, "reuse": 2, "render": 192},
    {"label": "s4-l45-r2",      "steps": 4, "layers": 45, "reuse": 2, "render": None},
    {"label": "s4-l50-r2",      "steps": 4, "layers": 50, "reuse": 2, "render": None},
    {"label": "s7-l45-r2",      "steps": 7, "layers": 45, "reuse": 2, "render": None},
    {"label": "s10-l45-r2",     "steps": 10, "layers": 45, "reuse": 2, "render": None},
    # --- Round 2: step=4, layer=40, vary core-reuse / token-reduction / render ---
    {"label": "s4-l40-cr2",         "steps": 4, "layers": 40, "core_reuse": 2, "render": None},
    {"label": "s4-l40_cr4",         "steps": 4, "layers": 40, "core_reuse": 4, "render": None},
    {"label": "s4-l40_cr6",         "steps": 4, "layers": 40, "core_reuse": 6, "render": None},
    {"label": "s4-l40_cr4-tr",      "steps": 4, "layers": 40, "core_reuse": 4,
                                  "token_reduction": True, "render": None},
    {"label": "s4-l40_cr4-tr-R192", "steps": 4, "layers": 40, "core_reuse": 4,
                                  "token_reduction": True, "render": 192},
    {"label": "s4-l40_cr6-tr",      "steps": 4, "layers": 40, "core_reuse": 6,
                                  "token_reduction": True, "render": None},
    # --- Round 3: step=7 variants (append to table) ---
    {"label": "s7-l40-r3-R192",      "steps": 7, "layers": 40, "reuse": 3, "render": 192},
    {"label": "s7-l40_cr6-tr",       "steps": 7, "layers": 40, "core_reuse": 6,
                                   "token_reduction": True, "render": None},
    {"label": "s7-l40_cr4-tr",       "steps": 7, "layers": 40, "core_reuse": 4,
                                   "token_reduction": True, "render": None},
    {"label": "s7-l40-r3",           "steps": 7, "layers": 40, "reuse": 3, "render": None},
    {"label": "s7-l40_cr6",          "steps": 7, "layers": 40, "core_reuse": 6, "render": None},
    {"label": "s7-l40_cr4",          "steps": 7, "layers": 40, "core_reuse": 4, "render": None},
    {"label": "s7-l45-r2-R192",      "steps": 7, "layers": 45, "reuse": 2, "render": 192},
    {"label": "s7-l40_cr2",          "steps": 7, "layers": 40, "core_reuse": 2, "render": None},
    {"label": "s7-l50-r2",           "steps": 7, "layers": 50, "reuse": 2, "render": None},
    {"label": "s7-l45-r2",           "steps": 7, "layers": 45, "reuse": 2, "render": None},
]

MEMORY_LIMIT_GB = 13.0

# --------------------------------------------------------------------------- #
# h3 execution
# --------------------------------------------------------------------------- #
def run_h3(params, output_path):
    """Run h3 with the given params. Returns (ok, wall_seconds, log_text)."""
    cmd = [
        H3_BIN, "-d", MODEL_DIR, "-p", PROMPT,
        "--width", str(WIDTH), "--height", str(HEIGHT),
        "--seconds", str(SECONDS), "--seed", str(SEED),
        "--ssd-streaming",
        "-o", str(output_path),
    ]
    if "turbo" in params:
        cmd += ["--linear-branch", params["turbo"]]
    cmd += ["--steps", str(params["steps"])]
    cmd += ["--layers", str(params["layers"])]
    if "core_reuse" in params and params["core_reuse"]:
        cmd += ["--core-reuse", str(params["core_reuse"])]
    else:
        cmd += ["--reuse", str(params["reuse"])]
    if "token_reduction" in params and params["token_reduction"]:
        cmd += ["--token-reduction"]
    if params.get("render"):
        cmd += ["--render-width", str(params["render"]),
                "--render-height", str(params["render"])]

    t0 = time.time()
    env = os.environ.copy()
    env["H3_CLIPPROJ_DIR"] = CLIPPROJ_DIR
    env["H3_CLIPPROJ_PROJ"] = CLIPPROJ_PROJ
    # Wrap with /usr/bin/time -l to capture peak RSS (macOS).
    time_cmd = ["/usr/bin/time", "-l"] + cmd
    try:
        proc = subprocess.run(time_cmd, capture_output=True, text=True, timeout=2400,
                              cwd=H3_DIR, env=env)
        wall = time.time() - t0
        log = proc.stdout + "\n" + proc.stderr
        ok = proc.returncode == 0 and output_path.exists()
        # Parse peak RSS from "/usr/bin/time -l" output. macOS uses either
        #   "  12345678  peak memory footprint"       (newer)
        #   "  12345678  maximum resident set size"   (older)
        peak_bytes = 0
        for line in (proc.stderr + proc.stdout).splitlines():
            low = line.lower()
            if "peak memory footprint" in low or "maximum resident" in low:
                try:
                    peak_bytes = int(line.strip().split()[0])
                except (ValueError, IndexError):
                    pass
        peak_gb = peak_bytes / (1024 ** 3) if peak_bytes else 0.0
        return ok, wall, log, peak_gb
    except subprocess.TimeoutExpired:
        return False, time.time() - t0, "TIMEOUT", 0.0
    except Exception as e:
        return False, time.time() - t0, str(e), 0.0


# --------------------------------------------------------------------------- #
# Frame extraction & scoring
# --------------------------------------------------------------------------- #
def extract_frame(video_path, t=FRAME_TIME):
    """Extract a single frame at time t as numpy (H,W,3) float32 [0,1]."""
    cmd = [
        "ffmpeg", "-y", "-v", "error",
        "-ss", str(t), "-i", str(video_path),
        "-frames:v", "1", "-f", "rawvideo", "-pix_fmt", "rgb24", "-",
    ]
    try:
        proc = subprocess.run(cmd, capture_output=True, timeout=60)
        if proc.returncode != 0:
            return None
        raw = proc.stdout
        expected = WIDTH * HEIGHT * 3
        if len(raw) < expected:
            return None
        img = np.frombuffer(raw[:expected], dtype=np.uint8).reshape(HEIGHT, WIDTH, 3)
        return img.astype(np.float32) / 255.0
    except Exception:
        return None


def _gaussian_kernel_1d(size, sigma):
    x = np.arange(size, dtype=np.float32) - size // 2
    g = np.exp(-(x * x) / (2 * sigma * sigma))
    return g / g.sum()


def _filter_separable(img, kernel):
    """Separable Gaussian filter on a 2D float32 image."""
    k = kernel
    pad = len(k) // 2
    # horizontal
    img_pad = np.pad(img, ((0, 0), (pad, pad)), mode="reflect")
    tmp = np.zeros_like(img, dtype=np.float32)
    for i in range(len(k)):
        tmp += k[i] * img_pad[:, i:i + img.shape[1]]
    # vertical
    tmp_pad = np.pad(tmp, ((pad, pad), (0, 0)), mode="reflect")
    out = np.zeros_like(img, dtype=np.float32)
    for i in range(len(k)):
        out += k[i] * tmp_pad[i:i + img.shape[0], :]
    return out


def compute_ssim(img1, img2):
    """SSIM between two float32 [0,1] images. Color -> luminance first."""
    if img1.ndim == 3:
        img1 = np.mean(img1, axis=2)
    if img2.ndim == 3:
        img2 = np.mean(img2, axis=2)
    kernel = _gaussian_kernel_1d(11, 1.5)
    mu1 = _filter_separable(img1, kernel)
    mu2 = _filter_separable(img2, kernel)
    s1 = _filter_separable(img1 * img1, kernel) - mu1 * mu1
    s2 = _filter_separable(img2 * img2, kernel) - mu2 * mu2
    s12 = _filter_separable(img1 * img2, kernel) - mu1 * mu2
    C1 = 0.01 ** 2
    C2 = 0.03 ** 2
    ssim_map = ((2 * mu1 * mu2 + C1) * (2 * s12 + C2)) / \
               ((mu1 * mu1 + mu2 * mu2 + C1) * (s1 + s2 + C2))
    return float(np.mean(ssim_map))


def compute_psnr(img1, img2):
    """PSNR (dB) between two float32 [0,1] images."""
    mse = np.mean((img1 - img2) ** 2)
    if mse < 1e-12:
        return 100.0
    return float(10 * np.log10(1.0 / mse))


def compute_l2(img1, img2):
    """Normalized L2 distance (0 = identical, 1 = max difference)."""
    return float(np.sqrt(np.mean((img1 - img2) ** 2)))


def score_frame(test_frame, ref_frame):
    """Return dict of quality metrics."""
    return {
        "ssim": compute_ssim(test_frame, ref_frame),
        "psnr": compute_psnr(test_frame, ref_frame),
        "l2":   compute_l2(test_frame, ref_frame),
    }


# --------------------------------------------------------------------------- #
# HTML report
# --------------------------------------------------------------------------- #
def _img_to_base64(img_float):
    """Convert float32 [0,1] (H,W,3) to base64 PNG."""
    from io import BytesIO
    arr = (np.clip(img_float, 0, 1) * 255).astype(np.uint8)
    # manual PNG encode via ppm -> ffmpeg (no PIL)
    h, w = arr.shape[:2]
    raw = arr.tobytes()
    cmd = ["ffmpeg", "-y", "-v", "error",
           "-f", "rawvideo", "-pix_fmt", "rgb24",
           "-s", f"{w}x{h}", "-i", "-",
           "-frames:v", "1", "-f", "image2pipe", "-vcodec", "png", "-"]
    try:
        proc = subprocess.run(cmd, input=raw, capture_output=True, timeout=30)
        if proc.returncode == 0:
            return base64.b64encode(proc.stdout).decode("ascii")
    except Exception:
        pass
    return ""


def generate_html(results, output_path):
    """Write a self-contained HTML report."""
    ref = results["reference"]
    tests = results["tests"]

    # Sort tests by efficiency (ssim / time), descending.
    for t in tests:
        t["efficiency"] = t["ssim"] / max(t["time"], 0.01)
    tests_sorted = sorted(tests, key=lambda x: x["efficiency"], reverse=True)

    # Pre-encode frames as base64.
    ref_b64 = _img_to_base64(ref["frame"]) if ref.get("frame") is not None else ""
    for t in tests_sorted:
        t["b64"] = _img_to_base64(t["frame"]) if t.get("frame") is not None else ""
        if not t.get("error"):
            t["efficiency"] = t["ssim"] / max(t["time"], 0.01)

    # Build HTML.
    rows = ""
    for i, t in enumerate(tests_sorted, 1):
        badge = ""
        if t["ssim"] > 0.85:
            badge = '<span class="badge good">Excellent</span>'
        elif t["ssim"] > 0.70:
            badge = '<span class="badge ok">Good</span>'
        elif t["ssim"] > 0.50:
            badge = '<span class="badge warn">Fair</span>'
        else:
            badge = '<span class="badge bad">Poor</span>'
        rows += f"""
        <tr>
            <td>{i}</td>
            <td><strong>{t["label"]}</strong><br><small>{t["param_str"]}</small></td>
            <td>{t["time"]:.1f}s</td>
            <td>{t.get("peak_gb", 0):.2f} GB</td>
            <td>{t["ssim"]:.4f}</td>
            <td>{t["psnr"]:.1f}</td>
            <td>{t["l2"]:.4f}</td>
            <td>{t["efficiency"]:.4f}</td>
            <td>{badge}</td>
            <td><img src="data:image/png;base64,{t["b64"]}" width="128" height="128" loading="lazy"/></td>
        </tr>"""

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8"/>
<title>h3 Benchmark Report</title>
<style>
:root {{ --bg:#0d1117; --card:#161b22; --fg:#c9d1d9; --accent:#58a6ff; --border:#30363d; }}
body {{ font-family:-apple-system,BlinkMacSystemFont,Segoe UI,Helvetica,Arial,sans-serif;
       background:var(--bg); color:var(--fg); margin:0; padding:2rem; }}
h1 {{ color:var(--accent); margin-top:0; }}
h2 {{ color:var(--fg); border-bottom:1px solid var(--border); padding-bottom:.5rem; }}
.meta {{ color:#8b949e; font-size:.9rem; margin-bottom:2rem; }}
table {{ border-collapse:collapse; width:100%; margin:1rem 0; background:var(--card);
        border-radius:6px; overflow:hidden; }}
th, td {{ padding:.6rem .8rem; text-align:left; border-bottom:1px solid var(--border); }}
th {{ background:#21262d; color:var(--accent); font-weight:600; }}
td small {{ color:#8b949e; }}
img {{ border-radius:4px; border:1px solid var(--border); image-rendering:auto; }}
.badge {{ padding:.15rem .5rem; border-radius:10px; font-size:.75rem; font-weight:600; }}
.badge.good {{ background:#1a7f3733; color:#3fb950; }}
.badge.ok {{ background:#9e6a0333; color:#d29922; }}
.badge.warn {{ background:#6e400033; color:#db6d28; }}
.badge.bad {{ background:#da363333; color:#f85149; }}
.ref {{ display:flex; gap:2rem; align-items:flex-start; flex-wrap:wrap; }}
.ref img {{ border:2px solid var(--accent); }}
.kv {{ display:grid; grid-template-columns:auto 1fr; gap:.3rem 1rem; }}
.kv dt {{ color:#8b949e; }}
.kv dd {{ margin:0; }}
</style>
</head>
<body>
<h1>h3 Benchmark Report</h1>
<div class="meta">
  Generated: {results["timestamp"]} &nbsp;|&nbsp;
  Model: MiniMax-H3-Convrot &nbsp;|&nbsp;
  Prompt: "{PROMPT}" &nbsp;|&nbsp;
  Resolution: {WIDTH}x{HEIGHT} &nbsp;|&nbsp;
  Seed: {SEED} &nbsp;|&nbsp;
  Reference: steps={REF_PARAMS["steps"]}, layers={REF_PARAMS["layers"]}, reuse={REF_PARAMS["reuse"]}
</div>

<h2>Reference (highest quality)</h2>
<div class="ref">
  <img src="data:image/png;base64,{ref_b64}" width="256" height="256"/>
  <div class="kv">
    <dt>Wall time</dt><dd>{ref["time"]:.1f}s</dd>
    <dt>Peak memory</dt><dd>{ref.get("peak_gb", 0):.2f} GB</dd>
    <dt>Steps</dt><dd>{REF_PARAMS["steps"]}</dd>
    <dt>Layers</dt><dd>{REF_PARAMS["layers"]}</dd>
    <dt>Reuse</dt><dd>{REF_PARAMS["reuse"]}</dd>
  </div>
</div>

<h2>Test Results (sorted by efficiency = SSIM / time)</h2>
<table>
<thead><tr>
  <th>#</th><th>Parameters</th><th>Time</th><th>Peak Mem</th><th>SSIM</th><th>PSNR</th><th>L2</th>
  <th>Efficiency</th><th>Quality</th><th>Frame</th>
</tr></thead>
<tbody>{rows}</tbody>
</table>

<h2>How to read this report</h2>
<ul>
  <li><strong>SSIM</strong> (Structural Similarity): 0–1, higher is better. &gt;0.85 is excellent,
      &gt;0.70 good, &gt;0.50 acceptable for preview.</li>
  <li><strong>PSNR</strong> (Peak Signal-to-Noise Ratio): higher is better, dB scale.</li>
  <li><strong>L2</strong> normalized pixel distance: 0 = identical, lower is better.</li>
  <li><strong>Efficiency</strong> = SSIM / time: quality per second. Higher means better trade-off.</li>
  <li>All comparisons are against the reference frame at t={FRAME_TIME}s.</li>
</ul>
</body>
</html>"""

    with open(output_path, "w") as f:
        f.write(html)
    return output_path


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main():
    parser = argparse.ArgumentParser(description="h3 benchmark")
    parser.add_argument("--skip-gen", action="store_true",
                        help="Reuse existing videos, only re-score + report")
    parser.add_argument("--report-only", action="store_true",
                        help="Reuse existing JSON + videos, only re-generate HTML report")
    parser.add_argument("--output-dir", default=OUTPUT_DIR)
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    videos_dir = out_dir / "videos"
    videos_dir.mkdir(exist_ok=True)

    # --- Report-only mode: load existing JSON, regenerate HTML. ---
    if args.report_only:
        results_path = out_dir / "results.json"
        if not results_path.exists():
            print(f"No existing results.json at {results_path}")
            sys.exit(1)
        with open(results_path) as f:
            data = json.load(f)
        # Rebuild results structure with frames for HTML generation.
        results = {
            "timestamp": data.get("timestamp", time.strftime("%Y-%m-%d %H:%M:%S")),
            "reference": {"time": data.get("reference_time", 0),
                          "peak_gb": data.get("reference_peak_gb", 0)},
            "tests": [],
        }
        ref_path = videos_dir / "reference.mp4"
        if ref_path.exists():
            results["reference"]["frame"] = extract_frame(ref_path)
        for t in data.get("tests", []):
            label = t["label"]
            vpath = videos_dir / f"test_{label}.mp4"
            entry = dict(t)
            entry["frame"] = extract_frame(vpath) if vpath.exists() else None
            results["tests"].append(entry)
        report_path = out_dir / "report.html"
        generate_html(results, report_path)
        print(f"HTML report regenerated: {report_path}")
        print(f"Open: file://{report_path}")
        return

    results = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "reference": {},
        "tests": [],
    }

    # --- Reference ---
    ref_path = videos_dir / "reference.mp4"
    if not args.skip_gen or not ref_path.exists():
        print(f"[ref] Generating reference: steps={REF_PARAMS['steps']}, "
              f"layers={REF_PARAMS['layers']}, reuse={REF_PARAMS['reuse']} ...")
        ok, wall, log, peak_gb = run_h3(REF_PARAMS, ref_path)
        if not ok:
            print(f"[ref] FAILED:\n{log[-2000:]}")
            sys.exit(1)
        print(f"[ref] Done in {wall:.1f}s, peak {peak_gb:.2f} GB -> {ref_path}")
    else:
        print(f"[ref] Reusing {ref_path}")
        wall, peak_gb = 0, 0.0

    if peak_gb > MEMORY_LIMIT_GB:
        print(f"[ref] ABORT: peak memory {peak_gb:.2f} GB exceeds "
              f"{MEMORY_LIMIT_GB:.0f} GB limit.")
        sys.exit(1)

    ref_frame = extract_frame(ref_path)
    if ref_frame is None:
        print("[ref] Failed to extract reference frame")
        results["reference"] = {"time": wall, "peak_gb": peak_gb, "frame": None,
                                "error": "reference generation failed"}
    else:
        results["reference"] = {"time": wall, "peak_gb": peak_gb, "frame": ref_frame}

    # --- Tests (ordered low -> high memory; stop if limit exceeded) ---
    stopped_mem = False
    for i, params in enumerate(TEST_MATRIX, 1):
        label = params.get("label", f"test-{i}")
        vpath = videos_dir / f"test_{label}.mp4"

        param_parts = [f"steps={params['steps']}", f"layers={params['layers']}"]
        if params.get("core_reuse"):
            param_parts.append(f"core-reuse={params['core_reuse']}")
        else:
            param_parts.append(f"reuse={params['reuse']}")
        if params.get("render"):
            param_parts.append(f"render={params['render']}")
        if params.get("token_reduction"):
            param_parts.append("token-reduction")
        param_str = ", ".join(param_parts)

        if not args.skip_gen or not vpath.exists():
            print(f"[{i}/{len(TEST_MATRIX)}] {label}: {param_str} ...")
            ok, wall, log, peak_gb = run_h3(params, vpath)
            if not ok:
                print(f"  FAILED (skipping):\n{log[-800:]}")
                results["tests"].append({
                    "label": label, "param_str": param_str,
                    "time": 0, "peak_gb": peak_gb, "ssim": 0, "psnr": 0, "l2": 1,
                    "frame": None, "error": True,
                })
                continue
            print(f"  Done in {wall:.1f}s, peak {peak_gb:.2f} GB")
        else:
            print(f"[{i}/{len(TEST_MATRIX)}] {label}: reusing {vpath}")
            wall, peak_gb = 0, 0.0

        if peak_gb > MEMORY_LIMIT_GB:
            print(f"  ABORT: peak {peak_gb:.2f} GB exceeds {MEMORY_LIMIT_GB:.0f} GB "
                  f"limit. Stopping sweep.")
            stopped_mem = True
            break

        frame = extract_frame(vpath)
        if frame is None:
            print(f"  Failed to extract frame, skipping score")
            results["tests"].append({
                "label": label, "param_str": param_str,
                "time": wall, "peak_gb": peak_gb, "ssim": 0, "psnr": 0, "l2": 1,
                "frame": None, "error": True,
            })
            continue

        if ref_frame is None:
            print(f"  No reference frame, skipping score")
            results["tests"].append({
                "label": label, "param_str": param_str,
                "time": wall, "peak_gb": peak_gb, "ssim": 0, "psnr": 0, "l2": 1,
                "frame": frame, "error": True,
            })
            continue

        scores = score_frame(frame, ref_frame)
        print(f"  SSIM={scores['ssim']:.4f}  PSNR={scores['psnr']:.1f}  L2={scores['l2']:.4f}")
        results["tests"].append({
            "label": label, "param_str": param_str,
            "time": wall, "peak_gb": peak_gb, "frame": frame, **scores,
        })

    if stopped_mem:
        print(f"\n*** Sweep stopped: a test exceeded {MEMORY_LIMIT_GB:.0f} GB "
              f"peak memory limit. ***")

    # --- Save raw results ---
    results_path = out_dir / "results.json"
    serializable = {
        "timestamp": results["timestamp"],
        "reference_time": results["reference"]["time"],
        "reference_peak_gb": results["reference"].get("peak_gb", 0),
        "memory_limit_gb": MEMORY_LIMIT_GB,
        "tests": [
            {k: v for k, v in t.items() if k != "frame"}
            for t in results["tests"]
        ],
    }
    with open(results_path, "w") as f:
        json.dump(serializable, f, indent=2)
    print(f"\nResults JSON: {results_path}")

    # --- HTML report ---
    report_path = out_dir / "report.html"
    generate_html(results, report_path)
    print(f"HTML report: {report_path}")
    print(f"Open: file://{report_path}")


if __name__ == "__main__":
    main()
