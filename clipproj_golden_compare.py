#!/usr/bin/env python3
"""Compare the in-engine ClipProj embedding (B) against A_local's harness golden.

A_local (clipproj_harness.py) writes one [seq,5120] float32 array per prompt into
a .npz keyed by the prompt string. The in-engine test (test_clipproj_encoder.c)
dumps [seq u32][dim u32][seq*dim float32] to a flat binary via H3_CLIPPROJ_DUMP.

Usage: clipproj_golden_compare.py GOLDEN.npz ENGINE.bin PROMPT [THRESHOLD]
Exits 0 if mean token-wise cosine >= THRESHOLD (default 0.999), else 1.
"""
import sys
import struct
import numpy as np


def cosine(a, b):
    a = a / (np.linalg.norm(a, axis=-1, keepdims=True) + 1e-12)
    b = b / (np.linalg.norm(b, axis=-1, keepdims=True) + 1e-12)
    return float(np.mean(np.sum(a * b, axis=-1)))


def main():
    if len(sys.argv) < 4:
        print("usage: clipproj_golden_compare.py GOLDEN.npz ENGINE.bin PROMPT "
              "[THRESHOLD]")
        sys.exit(2)
    golden, engine, prompt = sys.argv[1], sys.argv[2], sys.argv[3]
    thr = float(sys.argv[4]) if len(sys.argv) > 4 else 0.999

    g = np.load(golden)
    if prompt not in g.files:
        print(f"FAIL: prompt {prompt!r} not in golden npz "
              f"(have {list(g.files)})")
        sys.exit(1)
    A = g[prompt].astype(np.float32)  # [seq, 5120]

    with open(engine, "rb") as f:
        seq, dim = struct.unpack("<II", f.read(8))
        B = np.frombuffer(f.read(), dtype=np.float32).reshape(seq, dim)

    print(f"[compare] A_local seq={A.shape}  B seq={B.shape}")
    if A.shape != B.shape:
        # Token sequences disagree -> tokenizers / tap layer likely mismatch,
        # making the comparison invalid rather than just imprecise.
        print(f"FAIL: shape mismatch (A={A.shape} B={B.shape}); "
              f"check tokenizer / tap-layer agreement")
        sys.exit(1)
    c = cosine(A, B)
    print(f"[compare] cosine(B vs A_local) = {c:.6f}  (threshold {thr})")
    sys.exit(0 if c >= thr else 1)


if __name__ == "__main__":
    main()
