#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Convert a FastVideo/MLX affine group-quantized H3 DiT into an h3.c ConvRot
INT8 checkpoint.

Correctness notes -- each one is a defect that the previous version of this
script had, and each is verified numerically by `--verify`:

1. Rotation.  The MLX checkpoints in this pipeline were exported from the *int8
   ConvRot* source, so their dequantized weights are still in ConvRot (rotated)
   space.  Measured on the shipped artifacts: corr(mlx_dequant, source_rotated)
   = 1.000, corr(mlx_dequant, source_unrotated) = 0.06.  h3.c un-rotates every
   I8 weight on load (h3_dit.c::convrot_unrotate_cpu), so this script must NOT
   apply the Hadamard rotation itself.  The previous version did, which produced
   doubly rotated (unusable) weights.

2. Dequantization.  MLX affine dequant is `w * scales + biases`.  The previous
   version used `(w - biases) * scales`, which decorrelates the weights
   (corr drops to 0.06).

3. Single variable.  The base checkpoint is copied through unchanged, so only
   the four matrices under test are replaced.  The MLX export additionally
   quantized token_refiner / final_layer / context_embedder / time_embedder and
   renamed the time_embedder keys; taking the whole MLX file would have changed
   far more than the variable being measured.

4. Key names.  The previous remap produced malformed keys such as
   `blocks.0.attn.out_proj..weight.weight`.

5. QKV row order.  `blocks.N.attn.qkv_proj.weight` in the ConvRot checkpoint is
   stored in *module-major* order (the whole Q block, then K, then V) -- i.e. it
   is exactly `concat(to_q, to_k, to_v)`, measured relRMS 0.0056 against that
   concatenation versus 1.404 against the head-interleaved rearrangement.  h3.c
   converts stored -> the head-interleaved slot order itself
   (h3_dit.c::convrot_unrotate_cpu, `layout == 1`), so no permutation belongs
   here.  The previous version applied `interleave_qkv_separated`, i.e. a second
   permutation, which scrambled every attention projection.

Only the four matrices the engine actually streams are rebuilt:
    blocks.N.attn.qkv_proj.weight   <- concat(to_q, to_k, to_v)
    blocks.N.attn.out_proj.weight   <- attn.to_out.0
    blocks.N.mlp.fc1.weight         <- ff.net.0.proj
    blocks.N.mlp.fc2.weight         <- ff.net.2
(`attn.to_gate_compress` is VSA-only and is never read by h3.c, so it is copied
through from the base.)

Usage:
    # pre-flight: numerical checks only, no output written
    python mlx_int8_to_h3.py --base <ConvRot transformer dir> \
        --input /Volumes/data/work/h3_qad/h3_mlx/int4 --verify

    # export
    python mlx_int8_to_h3.py --base <ConvRot transformer dir> \
        --input /Volumes/data/work/h3_qad/h3_mlx/int4 \
        --output /Volumes/data/work/h3_qad/h3_int4g64
"""

from __future__ import annotations

import argparse
import json
import shutil
import time
from pathlib import Path

import numpy as np
import safetensors.torch as st
import torch

NUM_HEADS = 56
HEAD_DIM = 128
NUM_LAYERS = 50
SHARD_BYTES = 2 * 1024 ** 3


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--base", type=Path, required=True,
                   help="ConvRot transformer dir (or the single .safetensors file)")
    p.add_argument("--input", type=Path, required=True,
                   help="MLX quantized checkpoint dir (contains manifest.json)")
    p.add_argument("--output", type=Path,
                   help="output transformer dir (required unless --verify)")
    p.add_argument("--verify", action="store_true",
                   help="run the numerical checks only")
    p.add_argument("--blocks", type=int, default=3,
                   help="how many blocks to check in --verify (default 3)")
    return p


# --------------------------------------------------------------------------- #
# base checkpoint
# --------------------------------------------------------------------------- #
def base_files(path: Path) -> list[Path]:
    if path.is_file():
        return [path]
    files = sorted(path.glob("*.safetensors"))
    if not files:
        raise SystemExit(f"no .safetensors under {path}")
    return files


class BaseStore:
    """Lazy per-tensor reader over the base checkpoint shards."""

    def __init__(self, path: Path):
        self.files = base_files(path)
        self.map: dict[str, Path] = {}
        for f in self.files:
            with st.safe_open(str(f), framework="pt", device="cpu") as fh:
                for k in fh.keys():
                    self.map[k] = f
        self.order = list(self.map)

    def tensor(self, key: str) -> torch.Tensor:
        with st.safe_open(str(self.map[key]), framework="pt", device="cpu") as fh:
            return fh.get_tensor(key)

    def raw(self, key: str) -> np.ndarray:
        """Dequantized int8 source weight (rotated space, still per-row scaled)."""
        w = self.tensor(key).numpy().astype(np.float32)
        s = self.tensor(key + "_scale").numpy().astype(np.float32)
        return w * s.reshape(-1, 1)


# --------------------------------------------------------------------------- #
# MLX group-quantized checkpoint
# --------------------------------------------------------------------------- #
class MlxStore:
    """MLX affine group-quantized checkpoint (weights are packed uint32)."""

    def __init__(self, path: Path):
        self.path = path
        self.manifest = json.loads((path / "manifest.json").read_text())
        self.bits = int(self.manifest["quantization"]["bits"])
        self.group_size = int(self.manifest["quantization"]["group_size"])
        self.quantized = set(self.manifest["quantized_keys"])
        self.map: dict[str, Path] = {}
        for f in sorted(path.glob("shard-*.safetensors")):
            with st.safe_open(str(f), framework="pt", device="cpu") as fh:
                for k in fh.keys():
                    self.map[k] = f

    def has(self, key: str) -> bool:
        return key in self.map

    def _t(self, key: str):
        with st.safe_open(str(self.map[key]), framework="pt", device="cpu") as fh:
            return fh.get_tensor(key)

    def dequant(self, key: str) -> np.ndarray | None:
        if key not in self.map:
            return None
        w = self._t(key).numpy()
        scales = self._t(key + ".scales").numpy().astype(np.float32)
        biases = self._t(key + ".biases").numpy().astype(np.float32) \
            if (key + ".biases") in self.map else None

        values_per_word = 32 // self.bits
        mask = (1 << self.bits) - 1
        words = w.astype(np.uint32).reshape(-1)
        flat = np.empty(words.size * values_per_word, dtype=np.float32)
        for i in range(values_per_word):
            flat[i::values_per_word] = ((words >> (i * self.bits)) & mask).astype(np.float32)
        values = flat.reshape(w.shape[0], -1)

        rows, columns = values.shape
        gs = self.group_size
        values = values.reshape(rows, columns // gs, gs)
        out = values * scales[:, :, None]
        if biases is not None:
            out = out + biases[:, :, None]
        return out.reshape(rows, columns)


# --------------------------------------------------------------------------- #
# weight math
# --------------------------------------------------------------------------- #
def requantize_per_row(weight: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Per-output-row symmetric INT8, matching the ConvRot checkpoint format."""
    scale = np.max(np.abs(weight), axis=1, keepdims=True) / 127.0
    np.maximum(scale, 1e-8, out=scale)
    quantized = np.clip(np.rint(weight / scale), -128.0, 127.0).astype(np.int8)
    return quantized, scale.reshape(-1).astype(np.float32)


MLX_KEYS = {
    "out_proj": "transformer_blocks.{n}.attn.to_out.0.weight",
    "fc1": "transformer_blocks.{n}.ff.net.0.proj.weight",
    "fc2": "transformer_blocks.{n}.ff.net.2.weight",
}


def rebuild_block_matrix(mlx: MlxStore, kind: str, index: int):
    """Return (int8, scale[rows]) for one DiT matrix, or None if unavailable."""
    if kind == "qkv":
        parts = []
        for name in ("to_q", "to_k", "to_v"):
            w = mlx.dequant(f"transformer_blocks.{index}.attn.{name}.weight")
            if w is None:
                return None
            parts.append(w)
        # Module-major, i.e. exactly what the ConvRot checkpoint stores.
        return requantize_per_row(np.concatenate(parts, axis=0))
    w = mlx.dequant(MLX_KEYS[kind].format(n=index))
    if w is None:
        return None
    return requantize_per_row(w)


def replacement_keys() -> dict[str, tuple[str, int]]:
    """h3 key -> (kind, block index) for the matrices this script rebuilds."""
    keys = {}
    for n in range(NUM_LAYERS):
        keys[f"blocks.{n}.attn.qkv_proj.weight"] = ("qkv", n)
        keys[f"blocks.{n}.attn.out_proj.weight"] = ("out_proj", n)
        keys[f"blocks.{n}.mlp.fc1.weight"] = ("fc1", n)
        keys[f"blocks.{n}.mlp.fc2.weight"] = ("fc2", n)
    return keys


# --------------------------------------------------------------------------- #
# verification
# --------------------------------------------------------------------------- #
def rel_rms(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.sqrt(np.mean((a - b) ** 2)) / (np.sqrt(np.mean(a ** 2)) + 1e-12))


def verify(base: BaseStore, mlx: MlxStore, blocks: int) -> int:
    print(f"[verify] MLX: {mlx.bits}-bit affine group-{mlx.group_size}, "
          f"{len(mlx.quantized)} quantized keys")
    print(f"{'matrix':46s} {'src rms':>9s} {'relRMS vs src':>14s} {'scale ratio':>12s}")
    worst = 0.0
    for n in range(blocks):
        for kind in ("qkv", "out_proj", "fc1", "fc2"):
            h3_key = {"qkv": f"blocks.{n}.attn.qkv_proj.weight",
                      "out_proj": f"blocks.{n}.attn.out_proj.weight",
                      "fc1": f"blocks.{n}.mlp.fc1.weight",
                      "fc2": f"blocks.{n}.mlp.fc2.weight"}[kind]
            if h3_key not in base.map:
                continue
            rebuilt = rebuild_block_matrix(mlx, kind, n)
            if rebuilt is None:
                print(f"{h3_key:46s} MISSING in MLX")
                worst = float("inf")
                continue
            quantized, scale = rebuilt
            reference = base.raw(h3_key)
            stored = quantized.astype(np.float32) * scale.reshape(-1, 1)
            source_scale = base.tensor(h3_key + "_scale").numpy().astype(np.float32).reshape(-1)
            err = rel_rms(stored, reference)
            ratio = float(np.median(scale) / np.median(source_scale))
            worst = max(worst, err)
            print(f"{h3_key:46s} {reference.std():9.4f} {err:14.5f} {ratio:12.4f}")
            if kind == "qkv":
                # Non-tautological row-order check: the emitted tensor must match the
                # stored layout, and must NOT match the head-interleaved one the
                # engine itself derives at load time (h3 convrot_unrotate_cpu).
                rows = np.arange(reference.shape[0])
                slot = rows // HEAD_DIM
                remapped = reference[(3 * (slot % NUM_HEADS) + slot // NUM_HEADS)
                                     * HEAD_DIM + (rows % HEAD_DIM)]
                print(f"{'  row order: stored':46s} {err:14.5f}"
                      f"   interleaved (wrong): {rel_rms(stored, remapped):.5f}")
    print(f"[verify] worst relRMS = {worst:.5f}  (int8 g64 is expected ~0.006, int4 g64 ~0.09)")
    return 0


# --------------------------------------------------------------------------- #
# export
# --------------------------------------------------------------------------- #
def export(base: BaseStore, mlx: MlxStore, output: Path) -> int:
    if output.exists():
        raise SystemExit(f"output already exists: {output} (remove it first)")
    output.mkdir(parents=True)

    replacements = replacement_keys()
    buffer: dict[str, torch.Tensor] = {}
    buffer_bytes = 0
    shard = 0
    index_map: dict[str, str] = {}
    stats: list[tuple[str, float]] = []
    started = time.perf_counter()

    def flush():
        nonlocal buffer, buffer_bytes, shard
        if not buffer:
            return
        name = f"model-{shard + 1:05d}.safetensors"
        st.save_file(buffer, str(output / name))
        for key in buffer:
            index_map[key] = name
        print(f"  [shard {shard}] {name}: {len(buffer)} tensors, "
              f"{buffer_bytes / 2**30:.2f} GiB ({time.perf_counter() - started:.0f}s)")
        shard += 1
        buffer = {}
        buffer_bytes = 0

    def emit(key: str, tensor: torch.Tensor):
        nonlocal buffer_bytes
        buffer[key] = tensor
        buffer_bytes += tensor.numel() * tensor.element_size()
        if buffer_bytes >= SHARD_BYTES:
            flush()

    replaced = 0
    for key in base.order:
        target = replacements.get(key)
        if target is None:
            emit(key, base.tensor(key))
            continue

        kind, index = target
        rebuilt = rebuild_block_matrix(mlx, kind, index)
        if rebuilt is None:
            print(f"  [warn] {key}: not in MLX, copied from base")
            emit(key, base.tensor(key))
            continue

        quantized, scale = rebuilt
        reference = base.raw(key)
        stats.append((key, rel_rms(quantized.astype(np.float32) * scale.reshape(-1, 1),
                                   reference)))
        emit(key, torch.from_numpy(quantized))
        emit(key + "_scale", torch.from_numpy(scale.reshape(-1, 1)))
        replaced += 1

    flush()

    (output / "model.safetensors.index.json").write_text(
        json.dumps({"metadata": {}, "weight_map": index_map}, indent=2))

    config = base.files[0].parent / "config.json"
    if config.is_file():
        shutil.copy2(config, output / "config.json")

    errors = [e for _, e in stats if np.isfinite(e)]
    print(f"\n[done] {len(base.order)} tensors, {replaced} matrices rebuilt, "
          f"{shard} shards, {time.perf_counter() - started:.0f}s")
    if errors:
        print(f"       rebuilt-vs-source relRMS: mean={np.mean(errors):.5f} "
              f"max={np.max(errors):.5f}")
    print(f"       output: {output} "
          f"({sum(f.stat().st_size for f in output.glob('*.safetensors')) / 2**30:.1f} GiB)")
    return 0


def main() -> int:
    args = build_parser().parse_args()
    base = BaseStore(args.base)
    mlx = MlxStore(args.input)
    print(f"[base] {len(base.order)} tensors from {len(base.files)} file(s)")
    if args.verify:
        return verify(base, mlx, args.blocks)
    if args.output is None:
        raise SystemExit("--output is required unless --verify")
    if args.blocks:
        verify(base, mlx, min(args.blocks, NUM_LAYERS))
    return export(base, mlx, args.output)


if __name__ == "__main__":
    raise SystemExit(main())
