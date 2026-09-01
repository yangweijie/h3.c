#!/usr/bin/env python3
"""Dump HF Qwen3-VL hidden_states[0..N] for a prompt, to localize depth-compounding
divergence in the in-engine ClipProj text encoder. Reuses dbg_compare.load_model."""
import sys
import numpy as np
import torch
from dbg_compare import load_model

MODEL = "/Volumes/data/.lmstudio/models/Qwen3-VL-4B-Instruct"
PROMPT = sys.argv[1] if len(sys.argv) > 1 else "A red fox walking through snow"
OUT = sys.argv[2] if len(sys.argv) > 2 else "/tmp/all_hs.npz"


def main():
    model, tok = load_model(MODEL, "float16", 25)
    ids = tok(PROMPT, return_tensors="pt", add_special_tokens=True).input_ids
    mask = (ids != tok.pad_token_id)
    with torch.no_grad():
        out = model(input_ids=ids, attention_mask=mask, output_hidden_states=True)
    hs = out.hidden_states  # list of [1, seq, 2560]
    d = {}
    for i, h in enumerate(hs):
        d[f"hs{i}"] = h[0].detach().cpu().numpy().astype(np.float32)
    np.savez(OUT, **d)
    print(f"[all_hs] dumped {len(d)} layers -> {OUT}  (seq={hs[0].shape[1]})")


if __name__ == "__main__":
    main()
