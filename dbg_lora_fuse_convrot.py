"""""Is a diffusers-keyed LoRA foldable into the on-disk ConvRot int8 payload?

ANSWER: NO.  Verdict from `effective delta vs D` below --- after folding, the
delta the network actually sees has relRMS ~1.0 and cos ~0.02-0.2 against the
true delta, i.e. it is buried in the int8 quantisation noise.  The delta is
only ~0.2% of ||W||, which is exactly the size of the per-row int8 error.
Folding looks harmless if you only compare the WHOLE weight before/after
(relRMS stays ~0.002 either way) --- that comparison is the trap.

The LoRA therefore has to run as a high-precision low-rank bypass.

The fold we evaluated, kept here as the measurement harness:

    A_rot = A . R          (R = radix-4 ConvRot butterfly / sqrt(256))
    D_rot = B @ A_rot
    q_new = quantize(q_disk * s_disk + D_rot)

Correct because R is orthonormal and symmetric:
    (W + D)_rot . (R x) = (W + D) x        and   D . R = B (A R)

The open question this script answers: the disk payload carries ONE FP32
SCALE PER OUTPUT ROW, so adding D_rot perturbs each row's amax. Do we

    (a) recompute the per-row scale from the new amax   (two passes, and
        every code in the row loses precision by the factor amax'/amax), or
    (b) keep the old scale and clip the overflowing codes (one pass, the
        scale blob does not change at all, but values past +-127 saturate)?

Reports relRMS against the exact f32 target for both.

    /Volumes/data/Application/anaconda3/bin/python dbg_lora_fuse_convrot.py \
        /Users/jay/h3_sys/MiniMax-H3-Convrot/FL2VA/transformer \
        /Volumes/data/.lmstudio/models/LightX2V/MiniMax-H3-Turbo/\
minimax_h3_fl2v_turbo_4step_v1.1_768p_bf16.safetensors \
        [--block 0] [--self-test]
"""

import argparse
import glob
import os
import sys

import numpy as np
from safetensors import safe_open

BLOCK = 256

# (label, diffusers module on the LoRA side, disk weight name, band count)
# qkv is stored module-major: rows [0,7168) = q, [7168,14336) = k, rest = v.
PROJECTIONS = [
    ("qkv", ["attn.to_q", "attn.to_k", "attn.to_v"],
     "blocks.{b}.attn.qkv_proj.weight"),
    ("out", ["attn.to_out.0"], "blocks.{b}.attn.out_proj.weight"),
    ("fc1", ["ff.net.0.proj"], "blocks.{b}.mlp.fc1.weight"),
    ("fc2", ["ff.net.2"], "blocks.{b}.mlp.fc2.weight"),
]


def fwht_radix4(x):
    """In-place radix-4 ConvRot butterfly along the last axis, /sqrt(256).

    Mirrors build_hadamard_bf16 / fwht_unrotate_cpu in
    tests/test_convrot_unrotate.c, minus the per-row scale.
    """
    out = np.array(x, dtype=np.float32, copy=True)
    n = out.shape[-1]
    if n % BLOCK:
        sys.exit("last axis %d is not a multiple of %d" % (n, BLOCK))
    for blk in range(0, n, BLOCK):
        v = out[..., blk:blk + BLOCK]
        stride = 1
        while stride < BLOCK:
            span = stride * 4
            for base in range(0, BLOCK, span):
                i0 = base + np.arange(stride)
                i1 = i0 + stride
                i2 = i0 + 2 * stride
                i3 = i0 + 3 * stride
                a = v[..., i0].copy()
                b = v[..., i1].copy()
                c = v[..., i2].copy()
                d = v[..., i3].copy()
                v[..., i0] = a + b + c - d
                v[..., i1] = a + b - c + d
                v[..., i2] = a - b + c + d
                v[..., i3] = -a + b + c + d
            stride *= 4
    return out / 16.0


def self_test():
    """R must be orthonormal and symmetric, or the whole fold is wrong."""
    ident = np.eye(BLOCK)
    r = fwht_radix4(ident)
    ortho = np.abs(r @ r.T - np.eye(BLOCK)).max()
    sym = np.abs(r - r.T).max()
    print("self-test  R orthonormal err %.3e  symmetric err %.3e" % (ortho, sym))
    ok = ortho < 1e-9 and sym < 1e-9
    print("self-test  %s" % ("PASS" if ok else "FAIL - do not trust the fold"))
    return ok


def open_stores(directory):
    files = sorted(glob.glob(os.path.join(directory, "*.safetensors")))
    if not files:
        sys.exit("no safetensors under %s" % directory)
    return [safe_open(p, framework="pt") for p in files]


def fetch(store, name):
    """Read a tensor as f32. The numpy framework cannot decode bfloat16 at
    all (get_tensor and get_slice both raise), so every store is opened with
    framework="pt" and converted here."""
    for handle in store:
        try:
            return handle.get_tensor(name).float().numpy()
        except Exception:
            continue
    return None


def module_tensors(lora, block, module, adapters):
    """Return (A [r,in], B [out,r]) for one diffusers module."""
    prefix = "transformer_blocks.%d.%s" % (block, module)
    a = fetch(lora, prefix + ".lora_A.%s.weight" % adapters[0])
    b = fetch(lora, prefix + ".lora_B.%s.weight" % adapters[0])
    if a is None or b is None:
        sys.exit("LoRA is missing %s (keys look like %s...)"
                 % (prefix, prefix + ".lora_A.default.weight"))
    return np.asarray(a, dtype=np.float32), np.asarray(b, dtype=np.float32)


def fetch_shape(store, name):
    """Shape only --- no payload read, so a full 50-block sweep is instant."""
    for handle in store:
        try:
            return list(handle.get_slice(name).get_shape())
        except Exception:
            continue
    return None


def lora_shape(lora, key):
    try:
        return list(lora.get_slice(key).get_shape())
    except Exception:
        return None


def infer_blocks(lora, adapter):
    """Highest transformer_blocks.N the LoRA actually carries."""
    count = 0
    while any(lora_shape(lora, "transformer_blocks.%d.%s.lora_A.%s.weight"
                         % (count, m, adapter))
              for _, modules, _ in PROJECTIONS for m in modules):
        count += 1
    return count


def fetch_slice(store, name, row0, row1):
    """Few rows only --- keeps the 50-block sweep off the disk hot path."""
    for handle in store:
        try:
            return handle.get_slice(name)[row0:row1].float().numpy()
        except Exception:
            continue
    return None


def bypass_check(lora, adapter, block):
    """The bypass can reuse the ALREADY-ROTATED activation R x that the main
    conv consumes, provided A is pre-rotated once at pack time:

        A_rot = A R        (R symmetric => R x = fwht(x), A R = fwht_rows(A))
        B (A_rot (R x)) == B (A x)

    If this holds, the bypass needs no activation rotate of its own, costs only
    the two thin convs, and its output lands in the same space as the main conv
    output (both module-major on disk), so no permutation is needed either.
    """
    rng = np.random.default_rng(0)
    for _, modules, _ in PROJECTIONS:
        for module in modules:
            a, b = module_tensors(lora, block, module, [adapter])
            a = a.astype(np.float64)
            b = b.astype(np.float64)
            x = rng.standard_normal(a.shape[1])
            rx = fwht_radix4(x.astype(np.float32)).astype(np.float64)
            a_rot = fwht_radix4(a.astype(np.float32)).astype(np.float64)
            lhs = b @ (a_rot @ rx)
            rhs = b @ (a @ x)
            print("bypass  %-16s B(A_rot(Rx)) vs B(Ax)  relRMS %.3e"
                  % (module, rel_rms(lhs, rhs)))


def survey_snr(store, lora, adapter, alpha, blocks, rows_per=32, head=64):
    """Sweep every block for the ONE number that decides fold-vs-bypass.

    No full GEMM: per block we fold a sample of output rows and compare
    RMS(delta_row) against that row's int8 step s.  Rounding noise has
    RMS = s/sqrt(12), so

        SNR = RMS(D) * sqrt(12) / s        SNR < 1  => delta buried
        cos(effective_delta, D) ~ SNR / sqrt(1 + SNR^2)

    This is a per-element criterion, so a row sample decides the whole block.
    """
    print("survey  SNR = RMS(delta)/RMS(round noise)   cos = SNR/sqrt(1+SNR^2)")
    print("survey  %4s %8s %8s %8s %8s %7s" %
          ("blk", "qkv", "out", "fc1", "fc2", "worst"))
    worst_overall = 1e9
    snr_by_proj = {label: [] for label, _, _ in PROJECTIONS}
    for b in range(blocks):
        row = []
        for label, modules, pattern in PROJECTIONS:
            name = pattern.format(b=b)
            shape = fetch_shape(store, name)
            scales = fetch(store, name[: -len(".weight")] + ".weight_scale")
            if shape is None or scales is None:
                print("survey  block %d %s MISSING" % (b, label))
                continue
            rows, cols = shape[0], shape[1]
            scales = scales.reshape(-1).astype(np.float64)
            band = rows // len(modules)

            # typical |W_row| from a small head slice, for the D/W ratio.
            codes = fetch_slice(store, name, 0, head)
            rms_w = float(np.sqrt(
                ((codes.astype(np.float64) * scales[:head, None]) ** 2).mean()))

            snrs, ratios = [], []
            for index, module in enumerate(modules):
                a, bb = module_tensors(lora, b, module, [adapter])
                scale = alpha / a.shape[0]
                a_rot = fwht_radix4(a)
                idx = np.linspace(0, band - 1, rows_per).astype(int)
                d = scale * (bb[idx] @ a_rot.astype(np.float64))
                s = scales[index * band + idx]
                rms_d = np.sqrt((d ** 2).mean(axis=1))
                snrs.append(float(np.mean(rms_d * np.sqrt(12.0) / s)))
                ratios.append(float(np.mean(rms_d) / rms_w))
            snr_by_proj.setdefault(label, []).append(min(snrs))
            row.append(min(snrs))
        worst = min(row)
        worst_overall = min(worst_overall, worst)
        print("survey  %4d %8.4f %8.4f %8.4f %8.4f %7.3f" %
              (b, row[0], row[1], row[2], row[3],
               worst / np.sqrt(1.0 + worst ** 2)))
    print("\nsurvey  worst SNR over all %d blocks: %.4f  (cos %.3f)"
          % (blocks, worst_overall,
             worst_overall / np.sqrt(1.0 + worst_overall ** 2)))
    for label, _, _ in PROJECTIONS:
        v = snr_by_proj.get(label) or []
        if v:
            print("survey  %-4s SNR min %.4f  mean %.4f  max %.4f"
                  % (label, min(v), float(np.mean(v)), max(v)))
    print("survey  VERDICT: %s" % (
        "fold is viable - delta clears the int8 noise floor"
        if worst_overall > 2.0 else
        "fold is NOT viable - delta is at or below the int8 noise floor"))
    return worst_overall


def scan_coverage(store, lora, adapter):
    """Reproduce h3_lora_matches() over every block and report anything that
    would silently be skipped.

    h3_lora_matches only tests that lora_A exists and that its INPUT WIDTH
    equals in_dim (h3_lora.c:156).  We additionally demand B's rows equal the
    band width and rank(A) == rank(B), because the fold needs those too.
    """
    blocks = infer_blocks(lora, adapter)
    print("scan  LoRA carries transformer_blocks.0..%d" % (blocks - 1))
    problems = []
    checked = 0
    for b in range(blocks):
        for label, modules, pattern in PROJECTIONS:
            name = pattern.format(b=b)
            shape = fetch_shape(store, name)
            if shape is None:
                problems.append("block %d %s: disk tensor %s absent"
                                % (b, label, name))
                continue
            if fetch_shape(store, name[: -len(".weight")] + ".weight_scale") \
                    is None:
                problems.append("block %d %s: weight_scale absent" % (b, label))
            rows, cols = shape[0], shape[1]
            band = rows // len(modules)
            if band * len(modules) != rows:
                problems.append("block %d %s: disk rows %d not a multiple of %d"
                                % (b, label, rows, len(modules)))
            for index, module in enumerate(modules):
                checked += 1
                prefix = "transformer_blocks.%d.%s" % (b, module)
                a = lora_shape(lora, prefix + ".lora_A.%s.weight" % adapter)
                bs = lora_shape(lora, prefix + ".lora_B.%s.weight" % adapter)
                if a is None or bs is None:
                    problems.append("block %d %s: missing %s.%s factor"
                                    % (b, label, module,
                                       "lora_A" if a is None else "lora_B"))
                    continue
                # the exact h3_lora_matches test
                if a[-1] != cols:
                    problems.append("block %d %s: %s lora_A width %d != disk "
                                    "in_dim %d  (h3_lora_matches skips this)"
                                    % (b, label, module, a[-1], cols))
                if bs[0] != band:
                    problems.append("block %d %s: %s lora_B rows %d != band %d"
                                    % (b, label, module, bs[0], band))
                if a[0] != bs[-1]:
                    problems.append("block %d %s: %s rank %d != %d"
                                    % (b, label, module, a[0], bs[-1]))
    print("scan  %d (block, module) factors checked" % checked)
    if problems:
        print("scan  %d PROBLEM(S):" % len(problems))
        for p in problems[:40]:
            print("      %s" % p)
        if len(problems) > 40:
            print("      ... %d more" % (len(problems) - 40))
    else:
        print("scan  PASS - no block/module would be skipped; every A width "
              "fits the disk in_dim and every B row count fits its band")
    return blocks, not problems


def rel_rms(approx, target):
    diff = (approx.astype(np.float64) - target.astype(np.float64))
    ref = target.astype(np.float64)
    return float(np.sqrt((diff ** 2).mean()) / np.sqrt((ref ** 2).mean()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("transformer_dir")
    ap.add_argument("lora_path")
    ap.add_argument("--blocks", default="0",
                    help="comma-separated blocks for the accuracy run")
    ap.add_argument("--scan", action="store_true",
                    help="metadata sweep over every block, no GEMM")
    ap.add_argument("--survey", action="store_true",
                    help="per-block delta-vs-quantisation-step sweep")
    ap.add_argument("--bypass-check", action="store_true",
                    help="verify B(A_rot(Rx)) == B(Ax) for the bypass path")
    ap.add_argument("--adapter", default="default")
    ap.add_argument("--self-test", action="store_true")
    args = ap.parse_args()

    if args.self_test and not self_test():
        return 1

    store = open_stores(args.transformer_dir)
    lora = [safe_open(args.lora_path, framework="pt")]
    alpha = float(lora[0].metadata().get("alpha", "128"))

    if args.scan:
        blocks, ok = scan_coverage(store, lora[0], args.adapter)
        if not ok:
            return 1
        print()
    else:
        blocks = infer_blocks(lora[0], args.adapter)

    if args.bypass_check:
        bypass_check(lora, args.adapter, 0)
        return 0

    if args.survey:
        survey_snr(store, lora, args.adapter, alpha, blocks)
        print()
        return 0

    for b in [int(t) for t in args.blocks.split(",") if t.strip()]:
        print("=== block %d ===" % b)
        measure_block(store, lora, args.adapter, alpha, b)
    return 0


def measure_block(store, lora, adapter, alpha, block):
    """Fold one block and report (a) rescale vs (b) clip, plus how much of the
    delta survives the int8 codes."""
    for label, modules, pattern in PROJECTIONS:
        name = pattern.format(b=block)
        codes = fetch(store, name)
        scales = fetch(store, name[: -len(".weight")] + ".weight_scale")
        if codes is None or scales is None:
            print("%-4s MISSING %s" % (label, name))
            continue
        codes = codes.astype(np.float64)
        scales = scales.reshape(-1).astype(np.float64)
        rows, cols = codes.shape

        # Fold every band of the projection into the matching LoRA module.
        band = rows // len(modules)
        delta = np.zeros_like(codes)
        for index, module in enumerate(modules):
            a, b = module_tensors(lora, block, module, [adapter])
            rank = a.shape[0]
            if a.shape[1] != cols or b.shape[0] != band:
                sys.exit("%s: A%s B%s does not fit disk [%d,%d]"
                         % (module, a.shape, b.shape, band, cols))
            scale = alpha / rank
            a_rot = fwht_radix4(a)
            delta[index * band:(index + 1) * band] = scale * (b @ a_rot)

        w_rot = codes * scales[:, None]
        base_rms = rel_rms(w_rot, w_rot)          # exact f32 target is w_rot+delta
        target = w_rot + delta

        # (a) recompute the per-row scale from the new amax.
        amax_new = np.abs(target).max(axis=1)
        scale_new = np.maximum(amax_new / 127.0, 1e-12)
        q_a = np.clip(np.round(target / scale_new[:, None]), -127, 127)
        err_a = rel_rms(q_a * scale_new[:, None], target)

        # (b) keep the disk scale, clip whatever overflows.
        q_b = np.clip(np.round(target / scales[:, None]), -127, 127)
        err_b = rel_rms(q_b * scales[:, None], target)

        growth = amax_new / np.abs(w_rot).max(axis=1)
        clipped = float((np.abs(target / scales[:, None]) > 127).mean())

        print("\n%-4s [%d,%d]  rank=%d  alpha/rank=%.3f"
              % (label, rows, cols, rank, scale))
        print("     ||D||/||W_rot||        %.4f"
              % (np.sqrt((delta ** 2).mean()) / np.sqrt((w_rot ** 2).mean())))
        print("     amax growth  mean %.4f  max %.4f"
              % (growth.mean(), growth.max()))
        print("     (a) new scale   relRMS %.5f" % err_a)
        print("     (b) old scale   relRMS %.5f   clipped codes %.4f%%"
              % (err_b, 100.0 * clipped))
        print("     winner: %s" % ("(b) clip" if err_b <= err_a else "(a) rescale"))

        # The number that actually decides fold-vs-bypass: how much of D
        # survives being pressed into the int8 codes?  delta_eff is what the
        # network will really see.  relRMS near 1.0 means the int8 quantisation
        # noise is as large as the delta itself and the LoRA is buried.
        for tag, q in (("a", q_a), ("b", q_b)):
            delta_eff = q * scales[:, None] - w_rot
            print("     (%s) effective delta vs D  relRMS %.3f  "
                  "cos %.3f   [1.0 / cos~0 => LoRA destroyed]"
                  % (tag, rel_rms(delta_eff, delta),
                     float((delta_eff * delta).sum()
                           / (np.sqrt((delta_eff ** 2).sum())
                              * np.sqrt((delta ** 2).sum())))))


if __name__ == "__main__":
    sys.exit(main())
