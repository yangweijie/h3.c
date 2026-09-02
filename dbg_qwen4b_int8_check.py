#!/usr/bin/env python3
"""P13.1 — does the Qwen3-VL-4B INT8 checkpoint need a ConvRot un-rotation?

The checkpoint (qwen3vl_4b_int8_convrot.safetensors) stores quantised matmuls as
I8 weights plus a per-output-channel F32 `{key}_scale` of shape [out, 1], i.e.
exactly the layout the ConvRot DiT uses.  But "convrot" in the name does not
prove the rotation was actually applied to *this* file, so we settle it
empirically by comparing against the BF16 original:

  * cos(dequant, bf16) ~ 1        -> plain INT8, no rotation needed
  * cos(dequant, bf16) ~ 0        -> rotated; then cos(unrotate, bf16) ~ 1
                                     confirms the radix-4 butterfly is right

Only the byte ranges we need are read, so the multi-GB files are never loaded.
"""
import gc
import json
import struct
import sys

import numpy as np

INT8 = ("/Volumes/data/.lmstudio/models/Qwen3-VL-4B-Instruct-int8-convrot/"
        "qwen3vl_4b_int8_convrot.safetensors")
BF16_DIR = "/Volumes/data/.lmstudio/models/Qwen3-VL-4B-Instruct/"
ROT = 256

# representative quantised matmuls: one attention proj and one MLP proj
TARGETS = [
    "model.language_model.layers.0.self_attn.q_proj.weight",
    "model.language_model.layers.0.self_attn.o_proj.weight",
    "model.language_model.layers.0.mlp.gate_proj.weight",
    "model.language_model.layers.0.mlp.down_proj.weight",
]


def load_header(path):
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        hdr = json.loads(f.read(n))
    hdr.pop("__metadata__", None)
    return hdr, 8 + n


def read_tensor(path, hdr, base_off, name):
    info = hdr[name]
    dt, shape = info["dtype"], info["shape"]
    s, e = info["data_offsets"]
    with open(path, "rb") as f:
        f.seek(base_off + s)
        buf = f.read(e - s)
    if dt == "I8":
        return np.frombuffer(buf, dtype=np.int8).reshape(shape).astype(np.float32)
    if dt == "F32":
        return np.frombuffer(buf, dtype="<f4").reshape(shape)
    if dt == "F16":
        return np.frombuffer(buf, dtype="<f2").reshape(shape).astype(np.float32)
    if dt == "BF16":
        u = np.frombuffer(buf, dtype="<u2").reshape(shape)
        return (u.astype(np.uint32) << 16).view(np.float32)
    raise ValueError("unhandled dtype %s" % dt)


def butterfly_rows(seg):
    """Radix-4 ConvRot butterfly over every row of seg [rows, 256] (x 1/16)."""
    stride = 1
    while stride < ROT:
        span = stride * 4
        for base in range(0, ROT, span):
            for lane in range(stride):
                i0 = base + lane
                i1 = i0 + stride
                i2 = i0 + 2 * stride
                i3 = i0 + 3 * stride
                a = seg[:, i0].copy()
                b = seg[:, i1].copy()
                c = seg[:, i2].copy()
                d = seg[:, i3].copy()
                seg[:, i0] = a + b + c - d
                seg[:, i1] = a + b - c + d
                seg[:, i2] = a - b + c + d
                seg[:, i3] = -a + b + c + d
        stride *= 4
    return seg * (1.0 / 16.0)


def unrotate_last(W):
    """W . R : butterfly along the last (K / input) dim, blocks of 256."""
    K = W.shape[-1]
    if K % ROT:
        return None
    flat = W.reshape(-1, K)
    out = np.empty_like(flat)
    for blk in range(0, K, ROT):
        out[:, blk:blk + ROT] = butterfly_rows(flat[:, blk:blk + ROT].copy())
    return out.reshape(W.shape)


def cos(a, b):
    a = a.ravel().astype(np.float64)
    b = b.ravel().astype(np.float64)
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-30))


def main():
    ihdr, ioff = load_header(INT8)
    index = json.load(open(BF16_DIR + "model.safetensors.index.json"))
    wmap = index["weight_map"]

    print("int8 tensors: %d, bf16 tensors: %d" % (len(ihdr), len(wmap)))
    verdict = {}
    for name in TARGETS:
        if name not in ihdr:
            print("== %s\n   MISSING in int8 checkpoint" % name)
            continue
        shard = wmap.get(name)
        if not shard:
            print("== %s\n   MISSING in bf16 index" % name)
            continue
        bhdr, boff = load_header(BF16_DIR + shard)

        wq = read_tensor(INT8, ihdr, ioff, name)
        sc = read_tensor(INT8, ihdr, ioff, name + "_scale")
        wb = read_tensor(BF16_DIR + shard, bhdr, boff, name)
        print("=" * 66)
        print("%s" % name)
        print("   int8 %s %s | scale %s | bf16 %s (%s)"
              % (ihdr[name]["dtype"], wq.shape, sc.shape, wb.shape, shard))

        wd = wq * sc  # dequant, scale broadcasts over [out, 1]
        c_raw = cos(wd, wb)
        print("   cos(dequant only, bf16)   = %+.6f" % c_raw)
        best = ("dequant-only", c_raw)
        wu = unrotate_last(wd)
        if wu is not None:
            c_rot = cos(wu, wb)
            print("   cos(unrotate(K), bf16)    = %+.6f" % c_rot)
            if abs(c_rot) > abs(best[1]):
                best = ("unrotate-last", c_rot)
        print("   --> %s (cos %+.6f)" % best)
        verdict[name] = best
        del wq, sc, wb, wd, wu
        gc.collect()

    print("=" * 66)
    print("VERDICT:")
    for name, (hyp, c) in verdict.items():
        print("  %-58s %-14s cos %+.6f" % (name.split(".layers.0.")[-1], hyp, c))
    needs = any(h == "unrotate-last" for h, _ in verdict.values())
    print("\n=> ConvRot un-rotation %s for the 4B INT8 text encoder."
          % ("REQUIRED" if needs else "NOT required (plain INT8)"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
