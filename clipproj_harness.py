#!/usr/bin/env python3
"""ClipProj harness: Qwen3-VL-4B (layer t) -> ClipProj embedding (5120-dim).

Reproduces nicolab28/ComfyUI-ClipProj forward (clipproj_projection.py /
clipproj_nodes.py):

    xn      = (h - mean_in) / std_in
    y       = MLP(xn)            # Linear(2560->32768) -> GELU -> Linear(32768->5120)
    cond    = y * std_out + mean_out
    cond[0] = sink_out           # replace attention-sink token (token 0)

where h is the RAW hidden state of 4B layer `tap` (no final layer norm;
clipproj_nodes sets layer_norm_hidden_state=False).

NOTE on layer indexing:
    ComfyUI sets sm.layer = [tap] (tap=24 from metadata). In HF transformers,
    model(..., output_hidden_states=True).hidden_states is a tuple where
    hidden_states[0] = embedding output and hidden_states[k] = output of the
    (k-1)-th transformer block. So "layer tap" == hidden_states[tap+1].
    This is the DEFAULT (--hs-index = tap+1). If the real ComfyUI path indexes
    the raw per-layer list (index 0 = layer 0), use --hs-index = tap. Verify by
    matching the article's reported cos_test ~0.81.

Run with python3.13 (torch 2.12 + transformers 5.15 live there):
    python3.13 clipproj_harness.py --prompts prompts.txt --out cand.npz
    python3.13 clipproj_harness.py --inline "a cat,a dog" --out cand.npz
    python3.13 clipproj_harness.py --prompts prompts.txt --reference ref32b.npz
"""
import argparse, json, struct, sys
import numpy as np

_HADAMARD_CACHE = {}


def _build_hadamard(size, device="cpu", dtype=None):
    """Regular Hadamard matrix (size = power of 4), normalized by 1/sqrt(size).

    Mirrors comfy_kitchen.backends.eager.convrot_w4a4._build_hadamard: the
    4x4 seed is Kronecker-multiplied until it reaches `size`, then normalized.
    Normalized Hadamard is symmetric & orthogonal (H@H = I).
    """
    import torch
    key = (size, str(device), str(dtype))
    if key in _HADAMARD_CACHE:
        return _HADAMARD_CACHE[key]
    if size < 4 or (size & (size - 1)) != 0:
        raise ValueError(f"Hadamard size must be a power of 4, got {size}")
    dt = dtype if dtype is not None else torch.float32
    h4 = torch.tensor([[1, 1, 1, -1], [1, 1, -1, 1], [1, -1, 1, 1], [-1, 1, 1, 1]],
                      dtype=dt, device=device)
    h = h4
    current = 4
    while current < size:
        h = torch.kron(h, h4)
        current *= 4
    h = h / (size ** 0.5)
    _HADAMARD_CACHE[key] = h
    return h


def convrot_unrotate(w, group_size=256):
    """Undo the ConvRot Hadamard rotation: stored W = W_orig @ H^T (per group).

    w: [out, in] dequantized-but-still-rotated weight. Returns W_orig.
    Since normalized Hadamard is symmetric (H^T = H) and orthogonal, applying
    H^T again recovers the original basis (W_orig = W_rot @ H^T).
    """
    import torch
    out_f, in_f = w.shape
    h = _build_hadamard(group_size, device=w.device, dtype=torch.float32).to(w.dtype)
    n_groups = in_f // group_size
    grouped = w.reshape(out_f, n_groups, group_size)
    return torch.matmul(grouped, h.T).reshape(out_f, in_f)


# ---------- minimal safetensors loader (no external deps) ----------
def _bf16_to_f32(u16):
    return (u16.astype(np.uint32) << 16).view(np.float32)


def load_safetensors(path):
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        header = json.loads(f.read(n))
        data = f.read()
    out, meta = {}, header.get("__metadata__", {})
    dtmap = {"F16": np.float16, "F32": np.float32, "F64": np.float64,
             "I64": np.int64, "I32": np.int32}
    for name, info in header.items():
        if name == "__metadata__":
            continue
        dt = info["dtype"]; shape = info["shape"]
        s, e = info["data_offsets"]
        raw = data[s:e]
        if dt == "BF16":
            arr = _bf16_to_f32(np.frombuffer(raw, dtype=np.uint16).reshape(shape))
        else:
            arr = np.frombuffer(raw, dtype=dtmap[dt]).reshape(shape)
        out[name] = arr.astype(np.float32)
    return out, meta


# ---------- GELU (exact erf, matching torch.nn.GELU default) ----------
try:
    from scipy.special import erf
    def gelu(x): return 0.5 * x * (1.0 + erf(x / np.sqrt(2.0)))
except Exception:
    def gelu(x):
        return x * 0.5 * (1.0 + np.tanh(0.7978845608028654 * (x + 0.044715 * x * x * x)))


class ClipProj:
    def __init__(self, t):
        self.mean_in = t["mean_in"]; self.std_in = t["std_in"]
        self.mean_out = t["mean_out"]; self.std_out = t["std_out"]
        self.sink_out = t.get("sink_out")
        self.w0 = t["mlp.0.weight"].T   # [2560, 32768]
        self.b0 = t["mlp.0.bias"]      # [32768]
        self.w2 = t["mlp.2.weight"].T  # [32768, 5120]
        self.b2 = t["mlp.2.bias"]      # [5120]

    def __call__(self, h):
        # h: [seq, 2560] float32
        xn = (h - self.mean_in) / self.std_in
        x = xn @ self.w0 + self.b0
        x = gelu(x)
        y = x @ self.w2 + self.b2
        cond = y * self.std_out + self.mean_out
        if self.sink_out is not None and cond.shape[0] > 0:
            cond[0] = self.sink_out
        return cond  # [seq, 5120]


def cosine_matrix(a, b):
    a = a / (np.linalg.norm(a, axis=-1, keepdims=True) + 1e-12)
    b = b / (np.linalg.norm(b, axis=-1, keepdims=True) + 1e-12)
    return float(np.mean(np.sum(a * b, axis=-1)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", default="/Users/jay/.lmstudio/models/Qwen3-VL-4B-Instruct")
    ap.add_argument("--projection", default="/Users/jay/.lmstudio/models/ClipProj-MiniMax-H3/mmh3-4b-ClipProj-v3-mlp.safetensors")
    ap.add_argument("--prompts", required=True, help="txt file (one prompt/line) or, with --inline, comma-sep string")
    ap.add_argument("--inline", action="store_true", help="treat --prompts as inline comma-separated list")
    ap.add_argument("--out", default="candidate.npz")
    ap.add_argument("--reference", default=None, help="npz of 32B embeddings: prompt -> [seq,5120]")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--enc-dtype", default="float16", choices=["float16", "bfloat16", "float32"])
    ap.add_argument("--weights-file", default=None,
                    help="single safetensors file with weight-only int8 weights "
                         "(I8 'weight' + F32 'weight_scale' [out,1], e.g. qwen3vl_4b_int8_convrot). "
                         "Weights are dequantized w = i8 * scale. Overrides --model-dir shards.")
    ap.add_argument("--max-layers", type=int, default=25,
                    help="only load language-model layers 0..max_layers-1 (default 25 = embed + tap=24), "
                         "skipping the rest of the 36 layers + vision tower. Cuts fp16 weight load from "
                         "8.9GB to ~5.5GB so this 16GB box can run without OOM.")
    ap.add_argument("--quantize", default="none", choices=["none", "int8"],
                    help="weight-only int8 quantization after load (torch.ao.quantization.quantize_dynamic, "
                         "CPU; per the model card, a matrix calibrated on bf16 applied to an int8 encoder "
                         "differs by only ~0.0023 cosine, so this path is expected to match cos_test closely)")
    ap.add_argument("--tap", type=int, default=None, help="override projection tap (default: from metadata)")
    ap.add_argument("--hs-index", type=int, default=None, help="HF hidden_states index; default tap+1")
    ap.add_argument("--chat", action="store_true", help="apply Qwen3-VL chat template (user role)")
    ap.add_argument("--max-prompts", type=int, default=0)
    ap.add_argument("--dump-hidden", default=None,
                    help="npz to dump per-prompt tapped hidden [seq,2560] "
                         "for fidelity debugging (keyed by prompt)")
    args = ap.parse_args()

    import torch
    from torch import nn
    from transformers import AutoTokenizer, Qwen3VLModel

    t, meta = load_safetensors(args.projection)
    tap = args.tap if args.tap is not None else int(meta.get("tap", 24))
    hs_index = args.hs_index if args.hs_index is not None else (tap + 1)
    print(f"[proj] tap={tap} hs_index={hs_index} meta={ {k: meta[k] for k in ('d_in','d_out','cos_test','version','mlp_hidden') if k in meta} }", file=sys.stderr)

    dtype = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}[args.enc_dtype]
    print(f"[model] loading {args.model_dir} ({args.enc_dtype}, max_layers={args.max_layers})...", file=sys.stderr)

    from transformers import AutoConfig, AutoTokenizer, Qwen3VLModel

    # Truncated load: only read the language-model layers we actually need
    # (embed + layers 0..max_layers-1). The vision tower and trailing LM layers
    # are never read from disk, which keeps the fp16 load ~5.5GB instead of
    # ~8.9GB and avoids the OOM that killed this 16GB box before.
    import glob, re
    from safetensors import safe_open
    import torch.ao.quantization as aoq

    cfg = AutoConfig.from_pretrained(args.model_dir, trust_remote_code=True)
    full_text_layers = int(cfg.text_config.num_hidden_layers)
    if args.max_layers and args.max_layers < full_text_layers:
        cfg.text_config.num_hidden_layers = args.max_layers
        truncated = True
    else:
        truncated = False

    # which tensors to load (truncated LM stack only)
    keep = re.compile(r"^model\.language_model\.(embed_tokens\.weight|layers\.(\d+)\..*|norm\.weight)$")
    def want(name):
        if truncated and name.startswith("model.visual."):
            return False
        if truncated and not keep.match(name):
            return False
        if truncated and (mm := keep.match(name)) and mm.group(2) is not None \
                and int(mm.group(2)) >= args.max_layers:
            return False
        return True

    if args.weights_file:
        # Single weight-only-int8 file (I8 'weight' + F32 'weight_scale' [out,1],
        # BF16 bias/norm/embed). Dequantize w = i8 * scale (per out-channel).
        # If the layer carries a comfy_quant descriptor with "convrot": true the
        # stored weight is Hadamard-rotated (W_stored = W_orig @ H^T per 256-wide
        # group); undo that rotation or the network output is garbage (huge
        # activations -> NaN).
        # Skip the *_scale / *.comfy_quant companions (handled with their weight).
        state_dict = {}
        with safe_open(args.weights_file, framework="pt") as f:
            names = [n for n in f.keys()
                     if want(n) and not n.endswith("_scale") and not n.endswith("comfy_quant")]
            for name in names:
                base = name.removeprefix("model.")
                if name.endswith(".weight") and (name + "_scale") in f.keys() \
                        and f.get_slice(name).get_dtype() == "I8":
                    w = f.get_tensor(name).to(torch.float32)              # [out, in]
                    s = f.get_tensor(name + "_scale").to(torch.float32)   # [out, 1]
                    w = w * s
                    # ConvRot? the comfy_quant sidecar JSON says format/convrot.
                    # Sidecar key is "<base>.comfy_quant" (NOT ".weight.comfy_quant").
                    qname = name.removesuffix(".weight") + ".comfy_quant"
                    if qname in f.keys():
                        qraw = f.get_tensor(qname)
                        try:
                            qmeta = json.loads(bytes(qraw).decode("utf-8", errors="replace"))
                            if qmeta.get("convrot"):
                                gs = int(qmeta.get("convrot_groupsize", 256))
                                w = convrot_unrotate(w, gs)
                        except Exception:
                            pass
                    state_dict[base] = w.to(dtype)
                else:
                    state_dict[base] = f.get_tensor(name).to(dtype)
        print(f"[model] int8-file dequant: {len(state_dict)} tensors "
              f"({sum(v.numel() for v in state_dict.values())/1e9:.2f}B params)",
              file=sys.stderr)
    else:
        shards = sorted(glob.glob(args.model_dir + "/model-*.safetensors"))
        state_dict = {}
        for shard in shards:
            with safe_open(shard, framework="pt") as f:
                for name in f.keys():
                    if want(name):
                        # strip the "model." prefix safetensors uses; the nn.Module
                        # instance (Qwen3VLModel(cfg)) expects "language_model.*"
                        state_dict[name.removeprefix("model.")] = f.get_tensor(name).to(dtype)
        del shards
        print(f"[model] loaded {len(state_dict)} tensors ({sum(v.numel() for v in state_dict.values())/1e9:.2f}B params)",
              file=sys.stderr)

    tok = AutoTokenizer.from_pretrained(args.model_dir, trust_remote_code=True)
    # transformers >= 5.15 forbids passing state_dict with a model name, so
    # build the (truncated) model from config on meta device, then load tensors
    # with assign=True (replaces meta tensors in place, no extra copy).
    with torch.device("meta"):
        model = Qwen3VLModel(cfg)
    # strict=False: the vision tower is intentionally not loaded (unused for
    # text-only encoding) and stays as meta-placeholder; trailing LM layers are
    # already removed from the config.
    model.load_state_dict(state_dict, assign=True, strict=False)
    model.to(dtype)
    del state_dict
    # Rebuild meta buffers. The model is constructed on meta device so the
    # vision tower and trailing LM layers never allocate; buffers created there
    # (RoPE inv_freq) are meta and must be recomputed for the truncated stack.
    from transformers.models.qwen3_vl.modeling_qwen3_vl import (
        Qwen3VLTextRotaryEmbedding, ROPE_INIT_FUNCTIONS)
    for module in model.modules():
        if isinstance(module, Qwen3VLTextRotaryEmbedding):
            if module.rope_type != "default":
                inv_freq, attention_scaling = ROPE_INIT_FUNCTIONS[module.rope_type](module.config)
            else:
                inv_freq, attention_scaling = module.compute_default_rope_parameters(module.config)
            module.inv_freq = nn.Buffer(inv_freq, persistent=False)
            module.original_inv_freq = nn.Buffer(inv_freq.clone(), persistent=False)
            module.attention_scaling = attention_scaling
    if args.quantize == "int8":
        # Weight-only dynamic int8 on Linear layers: ~2x memory cut,
        # runs on CPU without extra deps.
        model = aoq.quantize_dynamic(model, {torch.nn.Linear}, dtype=torch.qint8)
        qbytes = sum(t.numel() * t.element_size() for t in model.state_dict().values())
        print(f"[model] quantized int8, weight bytes ≈ {qbytes/1e9:.2f} GB", file=sys.stderr)
    model.eval()
    proj = ClipProj(t)

    if args.inline:
        prompts = [p.strip() for p in args.prompts.split(",") if p.strip()]
    else:
        with open(args.prompts) as f:
            prompts = [l.strip() for l in f if l.strip()]
    if args.max_prompts:
        prompts = prompts[:args.max_prompts]

    ref = None
    if args.reference:
        ref = np.load(args.reference)
        print(f"[ref] loaded 32B reference with {len(ref.files)} prompts", file=sys.stderr)

    results = {}
    hidden_results = {}
    cosines = []
    for i, p in enumerate(prompts):
        if args.chat:
            ids = tok.apply_chat_template([{"role": "user", "content": p}],
                                          tokenize=True, add_generation_prompt=False,
                                          return_tensors="pt")
        else:
            ids = tok(p, return_tensors="pt", add_special_tokens=True).input_ids
        mask = (ids != tok.pad_token_id)
        with torch.no_grad():
            out = model(input_ids=ids.to(args.device),
                        attention_mask=mask.to(args.device),
                        output_hidden_states=True)
        h = out.hidden_states[hs_index][0].detach().cpu().numpy().astype(np.float32)  # [seq,2560]
        if args.dump_hidden:
            hidden_results[p] = h
        cond = proj(h)  # [seq,5120]
        results[p] = cond
        print(f"[{i}] seq={cond.shape[0]} mean_norm={np.linalg.norm(cond, axis=-1).mean():.3f} :: {p[:60]}", file=sys.stderr)

        if ref is not None and p in ref.files:
            c = cosine_matrix(cond, ref[p])
            cosines.append(c)
            print(f"    cosine_vs_32B={c:.4f}", file=sys.stderr)

    np.savez(args.out, **results)
    if args.dump_hidden:
        np.savez(args.dump_hidden, **hidden_results)
    print(f"[done] saved {len(results)} embeddings -> {args.out}", file=sys.stderr)
    if cosines:
        arr = np.array(cosines)
        frac = float(np.mean(arr >= 0.7))
        print(f"[cosine vs 32B] mean={arr.mean():.4f} min={arr.min():.4f} max={arr.max():.4f} frac>=0.7={frac:.2%}", file=sys.stderr)


if __name__ == "__main__":
    main()
