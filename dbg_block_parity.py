#!/usr/bin/env python3
"""P11.1 — single-DiT-block numerical parity between ConvRot INT8 and base BF16.

For each of block 0's four quantised matmuls (qkv_proj / out_proj / fc1 / fc2),
dequantise the ConvRot checkpoint (i8 * per-output-channel scale) and compare it
against the base BF16 tensor under several rotation hypotheses, to pin down:

  1. the R convention: is the stored weight  W . R_K  (rotate along the K /
     input dim, i.e. "rot last")  or  R_out . W  (rotate along the output dim,
     i.e. "rot first")?
  2. the qkv row permutation (P8.1 "formula B"):
     dst_slot = 3*(src_slot % heads) + (src_slot // heads), heads=56, head_dim=128

Whichever hypothesis gives the highest cosine against the base BF16 truth wins.
Only the small tensor byte-ranges are read via seek(), so the 20 GB and 5 GB
files are never loaded whole.
"""
import gc
import json
import struct
import sys

import numpy as np

CONVROT = ("/Users/jay/h3_sys/MiniMax-H3-Convrot/FL2VA/transformer/"
           "MiniMax_H3_FL2VA_pruned_int8_convrot.safetensors")
BASE = ("/Volumes/data/.lmstudio/models/MiniMax-H3/FL2VA/transformer/"
        "model-00001-of-00013.safetensors")

HEADS = 56
HEAD_DIM = 128
ROT = 256

PAIRS = [
    ("qkv", "blocks.0.attn.qkv_proj.weight"),
    ("out", "blocks.0.attn.out_proj.weight"),
    ("fc1", "blocks.0.mlp.fc1.weight"),
    ("fc2", "blocks.0.mlp.fc2.weight"),
]


def read_header(path):
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        return json.loads(f.read(n)), 8 + n


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


def _butterfly_rows(seg):
    """Radix-4 ConvRot butterfly over every row of seg [rows, 256].

    Mirrors h3_dit.c convrot_unrotate_cpu exactly: stages stride = 1,4,16,64,
    each quad (a,b,c,d) -> (a+b+c-d, a+b-c+d, a-b+c+d, -a+b+c+d), then the
    whole block is scaled by 1/sqrt(256) = 1/16.
    """
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


def _butterfly_cols(seg):
    """Radix-4 butterfly over the row axis of seg [256, cols] (i.e. R . W)."""
    stride = 1
    while stride < ROT:
        span = stride * 4
        for base in range(0, ROT, span):
            for lane in range(stride):
                i0 = base + lane
                i1 = i0 + stride
                i2 = i0 + 2 * stride
                i3 = i0 + 3 * stride
                a = seg[i0, :].copy()
                b = seg[i1, :].copy()
                c = seg[i2, :].copy()
                d = seg[i3, :].copy()
                seg[i0, :] = a + b + c - d
                seg[i1, :] = a + b - c + d
                seg[i2, :] = a - b + c + d
                seg[i3, :] = -a + b + c + d
        stride *= 4
    return seg * (1.0 / 16.0)


def rot_last(W):
    """W . R : butterfly along the last (K / input) dim in blocks of 256.

    NOTE: this is the radix-4 ConvRot butterfly, deliberately NOT the
    natural-order Sylvester Hadamard -- h3_dit.c warns that the Sylvester /
    popcount form is a different matrix and corrupts the weights.
    """
    K = W.shape[-1]
    if K % ROT:
        return None
    flat = W.reshape(-1, K)
    out = np.empty_like(flat)
    for blk in range(0, K, ROT):
        out[:, blk:blk + ROT] = _butterfly_rows(flat[:, blk:blk + ROT].copy())
    return out.reshape(W.shape)


def rot_first(W):
    """R . W : butterfly along the first (output) dim in blocks of 256."""
    N = W.shape[0]
    if N % ROT:
        return None
    out = W.copy()
    for blk in range(0, N, ROT):
        out[blk:blk + ROT, :] = _butterfly_cols(out[blk:blk + ROT, :].copy())
    return out


def perm_qkv_formula_B(W):
    """src slot s -> dst slot 3*(s % heads) + s // heads."""
    rows = W.shape[0]
    nslots = rows // HEAD_DIM
    idx = np.empty(rows, dtype=np.int64)
    for s in range(nslots):
        dst = 3 * (s % HEADS) + (s // HEADS)
        idx[dst * HEAD_DIM:(dst + 1) * HEAD_DIM] = \
            np.arange(s * HEAD_DIM, (s + 1) * HEAD_DIM)
    return W[idx]


def cos(a, b):
    a = a.ravel().astype(np.float64)
    b = b.ravel().astype(np.float64)
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-30))


def main():
    chdr, coff = read_header(CONVROT)
    bhdr, boff = read_header(BASE)
    results = {}

    for tag, name in PAIRS:
        print("=" * 66)
        Wq = read_tensor(CONVROT, chdr, coff, name)
        sc = read_tensor(CONVROT, chdr, coff, name + "_scale")
        Wb = read_tensor(BASE, bhdr, boff, name)
        print("%-4s %s" % (tag, name))
        print("     convrot i8 %s  scale %s   base bf16 %s"
              % (Wq.shape, sc.shape, Wb.shape))

        # base may be stored transposed relative to the quantised file
        if Wb.shape != Wq.shape and Wb.shape == tuple(reversed(Wq.shape)):
            Wb = Wb.T
            print("     (base tensor transposed to %s)" % (Wb.shape,))

        Wd = Wq * sc  # dequant, scale broadcasts over [out, 1]
        c_raw = cos(Wd, Wb)
        print("     cos(dequant only)          = %+.6f" % c_raw)

        best = ("none", c_raw)
        Wk = rot_last(Wd)
        if Wk is not None:
            c = cos(Wk, Wb)
            print("     cos(rot K   / last dim)    = %+.6f" % c)
            if abs(c) > abs(best[1]):
                best = ("rot_last", c)
        Wo = rot_first(Wd)
        if Wo is not None:
            c = cos(Wo, Wb)
            print("     cos(rot out / first dim)   = %+.6f" % c)
            if abs(c) > abs(best[1]):
                best = ("rot_first", c)

        if tag == "qkv":
            c = cos(perm_qkv_formula_B(Wd), Wb)
            print("     cos(dequant + permB)       = %+.6f" % c)
            if Wk is not None:
                c = cos(perm_qkv_formula_B(Wk), Wb)
                print("     cos(rot K  + permB)        = %+.6f" % c)
                if abs(c) > abs(best[1]):
                    best = ("rot_last+permB", c)
            if Wo is not None:
                c = cos(perm_qkv_formula_B(Wo), Wb)
                print("     cos(rot out+ permB)        = %+.6f" % c)
                if abs(c) > abs(best[1]):
                    best = ("rot_first+permB", c)

        print("     --> best hypothesis: %s (cos %+.6f)" % best)
        results[tag] = best
        del Wq, sc, Wb, Wd, Wk, Wo
        gc.collect()

    print("=" * 66)
    print("SUMMARY (block 0, best hypothesis per tensor):")
    for tag, (hyp, c) in results.items():
        print("  %-4s %-16s cos %+.6f" % (tag, hyp, c))
    return 0


if __name__ == "__main__":
    sys.exit(main())
