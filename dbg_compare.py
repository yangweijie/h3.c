#!/usr/bin/env python3
"""Bisect the in-engine ClipProj forward (B) vs the HF reference (A_local).

Runs the 4B model for a single layer with forward hooks, dumps intermediate
activations, and compares them against the engine's intermediate dumps
(produced with H3_CLIPPROJ_DUMP_EMBED / _QROPE / _ATTN / _HIDDEN env vars).
"""
import sys
import re
import glob
import struct
import numpy as np
import torch
import torch.nn as nn
from transformers import AutoConfig, AutoTokenizer, Qwen3VLModel
from safetensors import safe_open
from transformers.models.qwen3_vl.modeling_qwen3_vl import (
    Qwen3VLTextRotaryEmbedding, ROPE_INIT_FUNCTIONS)

MODEL = "/Volumes/data/.lmstudio/models/Qwen3-VL-4B-Instruct"
PROMPT = "A red fox walking through snow"
THETA = 5e6
HEAD_DIM = 128


def cos(a, b):
    a = np.asarray(a, dtype=np.float32)
    b = np.asarray(b, dtype=np.float32)
    a = a / (np.linalg.norm(a, axis=-1, keepdims=True) + 1e-12)
    b = b / (np.linalg.norm(b, axis=-1, keepdims=True) + 1e-12)
    return float(np.mean(np.sum(a * b, axis=-1)))


def apply_rope(x):
    seq, heads, hd = x.shape
    half = hd // 2
    inv = 1.0 / (THETA ** (np.arange(0, hd, 2) / hd))
    ang = np.arange(seq)[:, None] * inv[None, :]
    c = np.cos(ang)[:, None, :]
    s = np.sin(ang)[:, None, :]
    first = x[..., :half]
    second = x[..., half:]
    o0 = first * c - second * s
    o1 = second * c + first * s
    return np.concatenate([o0, o1], axis=-1)


def load_b(path):
    with open(path, "rb") as f:
        seq, dim = struct.unpack("<II", f.read(8))
        return np.frombuffer(f.read(), dtype=np.float32).reshape(seq, dim)


def load_model(model_dir, enc_dtype="float16", max_layers=1):
    dtype = {"float16": torch.float16, "bfloat16": torch.bfloat16,
             "float32": torch.float32}[enc_dtype]
    cfg = AutoConfig.from_pretrained(model_dir, trust_remote_code=True)
    full = int(cfg.text_config.num_hidden_layers)
    truncated = bool(max_layers and max_layers < full)
    if truncated:
        cfg.text_config.num_hidden_layers = max_layers
    keep = re.compile(
        r"^model\.language_model\.(embed_tokens\.weight|layers\.(\d+)\..*|norm\.weight)$")

    def want(name):
        if truncated and name.startswith("model.visual."):
            return False
        if truncated and not keep.match(name):
            return False
        if truncated and (mm := keep.match(name)) and mm.group(2) is not None \
                and int(mm.group(2)) >= max_layers:
            return False
        return True

    state_dict = {}
    for shard in sorted(glob.glob(model_dir + "/model-*.safetensors")):
        with safe_open(shard, framework="pt") as f:
            for name in f.keys():
                if want(name):
                    state_dict[name.removeprefix("model.")] = f.get_tensor(name).to(dtype)
    tok = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
    with torch.device("meta"):
        model = Qwen3VLModel(cfg)
    model.load_state_dict(state_dict, assign=True, strict=False)
    model.to(dtype)
    for module in model.modules():
        if isinstance(module, Qwen3VLTextRotaryEmbedding):
            if module.rope_type != "default":
                inv_freq, attn_scaling = ROPE_INIT_FUNCTIONS[module.rope_type](module.config)
            else:
                inv_freq, attn_scaling = module.compute_default_rope_parameters(module.config)
            module.inv_freq = nn.Buffer(inv_freq, persistent=False)
            module.original_inv_freq = nn.Buffer(inv_freq.clone(), persistent=False)
            module.attention_scaling = attn_scaling
    model.eval()
    return model, tok


def main():
    b_embed = load_b(sys.argv[1])
    b_qrope = load_b(sys.argv[2])
    b_attn = load_b(sys.argv[3])
    b_layer = load_b(sys.argv[4])

    model, tok = load_model(MODEL, "float16", 1)
    cap = {}
    layer = model.language_model.layers[0]

    def _cap(key):
        def _h(m, i, o):
            cap.setdefault(key, o.detach().cpu().numpy().astype(np.float32))
        return _h
    layer.register_forward_hook(_cap("layer_out"))
    def _attn_hook(m, i, o):
        cap.setdefault("attn_out", o[0].detach().cpu().numpy().astype(np.float32))
    layer.self_attn.register_forward_hook(_attn_hook)
    layer.mlp.register_forward_hook(_cap("mlp_out"))
    layer.self_attn.q_norm.register_forward_hook(_cap("q_norm_out"))

    ids = tok(PROMPT, return_tensors="pt", add_special_tokens=True).input_ids
    mask = (ids != tok.pad_token_id)
    with torch.no_grad():
        out = model(input_ids=ids, attention_mask=mask, output_hidden_states=True)
    hf_embed = out.hidden_states[0].detach().cpu().numpy().astype(np.float32).reshape(-1, 2560)
    S = hf_embed.shape[0]
    qh = b_qrope.shape[1] // HEAD_DIM
    b_qrope_r = b_qrope.reshape(S, qh, HEAD_DIM)
    hf_qnorm = cap["q_norm_out"].reshape(S, qh, HEAD_DIM)
    hf_qrope = apply_rope(hf_qnorm)
    attn_out = cap["attn_out"].reshape(S, 2560)
    mlp_out = cap["mlp_out"].reshape(S, 2560)
    layer_out = cap["layer_out"].reshape(S, 2560)

    print(f"embed  B vs HF                  : {cos(b_embed, hf_embed):.6f}")
    print(f"qrope  B vs HF(rope(q_norm))    : {cos(b_qrope_r, hf_qrope):.6f}")
    print(f"attn  B vs HF(embed+attn_out)   : {cos(b_attn, hf_embed + attn_out):.6f}")
    print(f"mlp   B vs HF(layer-attn)       : {cos(b_layer - b_attn, mlp_out):.6f}")
    print(f"layer B vs HF                   : {cos(b_layer, layer_out):.6f}")


if __name__ == "__main__":
    main()
