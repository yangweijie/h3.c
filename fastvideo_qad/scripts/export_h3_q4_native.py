#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Convert a mere-run MLX affine INT4 H3 DiT into an h3.c-loadable native
group-quantized checkpoint.

Why this script exists
----------------------
`/Volumes/data/MODELS/h3-16gb-q4/transformer.safetensors` (mere.run, MLX affine
4-bit group-64, U32-packed) cannot be loaded by h3.c as-is:

  * h3.c's packed group format is U8 (2 codes/byte), not U32 (8 codes/word).
    `h3_weight_grouped_spec` rejects anything whose dtype is not U8.
  * h3.c discovers `FL2VA/transformer/model-*-of-*.safetensors` shards, not a
    single merged `transformer.safetensors`.
  * The q4 file omits every AdaLN weight (`cache_covered_weights_omitted`), so
    on its own it is not a complete DiT.

Rotation space -- the part that makes this lossy
------------------------------------------------
h3.c's packed group path is contractually defined to hold **ConvRot-rotated**
weights: the streaming loader unconditionally runs the radix-4 Hadamard
butterfly over each 256-wide column block (`h3_weight_unrotate_rows`, called
with no env guard in the grouped branch of `h3_dit.c`), and the shader
`h3_weight_dequant_unrotate_int8` documents the same contract for I8.  The q4
file, by contrast, was quantized straight from the **released BF16** shards
(`source_precision: released mixed BF16/F32`), which are *not* rotated -- h3.c
loads those through the plain BF16 branch with no Hadamard at all.

So the conversion has to rotate, and rotating an already-quantized matrix forces
a second affine fit: the output error is q4 error (+) requant error, not q4
error alone.  `--no-rotate` skips the rotation and is only correct together with
`H3_INT8_UNROTATE=0` on a build whose grouped streaming path honours that switch
(the stock one does not).

QKV row order
-------------
The q4 file stores `qkv_proj` as global slabs `[all-q; all-k; all-v]`
(`qkv_layout: global-qkv-slabs`), and so does h3.c's engine: the streaming
grouped path passes `layout = 1` for STREAM_QKV, which is
`dst = 3*(slot%heads) + slot/heads`, i.e. it *consumes* slabs and emits the
per-head interleaved rows the attention kernel indexes (`qkv[base + d]`,
`qkv[base + head_dim + d]`, `qkv[base + 2*head_dim + d]`).  Verified byte-exactly:
`mx.quantize(to_slabs(released_bf16), group_size=64, bits=4)` reproduces the q4
file's packed/scales/biases bit for bit.  Therefore **no permutation is
applied**; `--qkv-input` exists so the assumption can be re-tested.

The token refiner stays BF16
----------------------------
`token_refiner.blocks.{0,1}.*` is one of the 208 quantized matrices in the q4
file, but h3.c does not read it through the DiT streamer: the text-encoder
module loads it (`h3_text_encoder.c:227`) and hard-requires BF16, rejecting the
U8 packed dtype outright.  Those 8 matrices are therefore copied through from
the base checkpoint, which costs 1.44 GiB.

Why the output is bigger than the q4 file
-----------------------------------------
Measured on this checkpoint (`q4` 10.56 GiB vs output 35.93 GiB, 25.38 GiB more):

    +24.28 GiB  the 51 `adaln_proj.linear.weight` matrices
                (50 blocks + `final_layer`), each BF16 [96768, 2688] = 496 MiB.
                The q4 file contains *zero* `adaln` keys: mere-run declares
                `cache_covered_weights_omitted: true` and caches the AdaLN
                *outputs* instead of the weights, so it never needs them at
                inference.  h3.c has no such cache -- it loads the projection
                per block, applies it to the timestep embedding and frees it
                (`h3_dit_schedule.c:553`) -- so the weights must be on disk.
     +1.03 GiB  token_refiner kept BF16 (above).
     +0.06 GiB  the 4-bit matrices themselves: 10.15 GiB in the q4 file vs
                10.21 GiB here.  Identical scheme, identical 0.5625 B/param.

The 4-bit half is therefore the same size on both sides, and the 25 GiB gap is
entirely non-quantized AdaLN.  That gap is **not** reducible from this script:

  * F16 does not help -- BF16 and F16 are both 16 bits/param.
  * The engine's AdaLN loader (`h3_dit_schedule.c:94 weight_bf16_any`) accepts
    only BF16 / F16 / F32, so the projections cannot be stored packed-4bit
    without extending that loader to dequantize on the host.  Doing so would
    take them from 24.28 GiB to 6.95 GiB, but AdaLN produces the modulation
    applied to every activation, so 4-bit error there is far more damaging than
    in the attention/MLP matrices and would need measuring.
  * Matching mere-run exactly now means running
    `export_h3_adaln_cache.py`, which drops these 51 matrices outright (35.93
    GiB -> 11.65 GiB) in exchange for a ~150 MiB cache of the modulation they
    produce.  Run this script first, then that one.

Format written (derived by the engine from dtype + shapes alone):
    blocks.N.attn.qkv_proj.weight        U8   [rows, cols/2]
                                         4-bit dense, 2 codes/byte,
                                         first code in the low nibble
    blocks.N.attn.qkv_proj.weight_scale  F16  [rows, cols/group]
    blocks.N.attn.qkv_proj.weight_bias   F16  [rows, cols/group]
    dequant: w = code*scale + bias,  code in [0, 15]

The accumulators are `{full_matrix_key}_scale` / `_bias` -- i.e. `_scale` is
appended to a key that already ends in `.weight`, giving `...weight_scale`.
That is literally what `h3_weight_grouped_spec` probes (h3_weights.c:623-628).

Usage:
    # format + numerics smoke test, nothing written
    python export_h3_q4_native.py --limit-blocks 4 --verify

    # full export
    python export_h3_q4_native.py --output /Volumes/data/work/h3_qad/h3_q4g64_native
"""

from __future__ import annotations

import argparse
import json
import shutil
import struct
import time
from pathlib import Path

import numpy as np

DEFAULT_Q4 = Path("/Volumes/data/MODELS/h3-16gb-q4/transformer.safetensors")
DEFAULT_BASE = Path("/Volumes/data/MODELS/h3c-official/FL2VA/transformer")

NUM_HEADS, HEAD_DIM, NUM_LAYERS = 56, 128, 50
ROT_BLOCK = 256          # ConvRot Hadamard block (h3_weights.c::H3_INT8_BLOCK)
SHARD_BYTES = 2 * 1024 ** 3
SOURCE_GROUP = 64        # mere-run's MLX affine group size


def scale_key(key: str) -> str:
    """Accumulator tensor names, exactly as the engine probes them.

    `h3_weight_grouped_spec` (h3_weights.c:614) looks up `name + "_scale"` and
    `name + "_bias"` where `name` is the full matrix key including `.weight`, so
    `blocks.0.attn.qkv_proj.weight` gets `...weight_scale` / `...weight_bias`.
    The `-7`-character form in h3_dit.c:1243 is the same thing spelled as a
    printf pattern (`"%.*s.weight_scale"`), not a different convention.
    """
    return key + "_scale"


def bias_key(key: str) -> str:
    return key + "_bias"

_NP_OF = {
    "BF16": "<u2", "F16": "<f2", "F32": "<f4", "F64": "<f8",
    "U8": "u1", "I8": "i1", "U16": "<u2", "I16": "<i2",
    "U32": "<u4", "I32": "<i4", "I64": "<i8", "BOOL": "?",
}
_BYTES_OF = {
    "BF16": 2, "F16": 2, "F32": 4, "F64": 8, "U8": 1, "I8": 1,
    "U16": 2, "I16": 2, "U32": 4, "I32": 4, "I64": 8, "BOOL": 1,
}


# --------------------------------------------------------------------------- #
# raw safetensors I/O (numpy has no bfloat16, so bf16 stays as uint16)
# --------------------------------------------------------------------------- #
def st_header(path: Path):
    with path.open("rb") as handle:
        length = struct.unpack("<Q", handle.read(8))[0]
        return json.loads(handle.read(length)), 8 + length


def st_read(path: Path, name: str):
    """Return (array, dtype) reading raw bytes, bypassing safetensors' bf16 gap."""
    header, base = st_header(path)
    entry = header[name]
    begin, end = entry["data_offsets"]
    with path.open("rb") as handle:
        handle.seek(base + begin)
        raw = handle.read(end - begin)
    array = np.frombuffer(raw, dtype=_NP_OF[entry["dtype"]]).reshape(entry["shape"])
    return array, entry["dtype"]


def st_read_bytes(path: Path, name: str) -> bytes:
    header, base = st_header(path)
    begin, end = header[name]["data_offsets"]
    with path.open("rb") as handle:
        handle.seek(base + begin)
        return handle.read(end - begin)


def st_write(path: Path, entries) -> None:
    """entries: iterable of (key, dtype, shape, producer->bytes). Streamed."""
    entries = list(entries)
    header = {}
    offset = 0
    for key, dtype, shape, _ in entries:
        count = 1
        for dimension in shape:
            count *= dimension
        size = count * _BYTES_OF[dtype]
        header[key] = {"dtype": dtype, "shape": list(shape),
                       "data_offsets": [offset, offset + size]}
        offset += size
    blob = json.dumps(header, separators=(",", ":")).encode()
    padding = -(len(blob) + 8) % 8
    with path.open("wb") as handle:
        handle.write(struct.pack("<Q", len(blob) + padding))
        handle.write(blob + b" " * padding)
        for _, _, _, produce in entries:
            handle.write(produce())


def bf16_to_f32(raw_u16: np.ndarray) -> np.ndarray:
    return (raw_u16.astype(np.uint32) << 16).view(np.float32)


# --------------------------------------------------------------------------- #
# MLX affine dequant / h3.c affine quant
# --------------------------------------------------------------------------- #
def dequantize_mlx(packed_u32: np.ndarray, scales_u16: np.ndarray,
                   biases_u16: np.ndarray, bits: int) -> np.ndarray:
    """MLX affine: w = code*scale + bias, first code in the lowest bits."""
    rows, words_per_row = packed_u32.shape
    per_word = 32 // bits
    mask = (1 << bits) - 1
    words = packed_u32.astype(np.uint32)
    codes = np.empty((rows, words_per_row * per_word), dtype=np.uint8)
    for index in range(per_word):
        codes[:, index::per_word] = ((words >> (index * bits)) & mask).astype(np.uint8)

    columns = codes.shape[1]
    scale = bf16_to_f32(scales_u16).astype(np.float32)
    bias = bf16_to_f32(biases_u16).astype(np.float32)
    group = columns // scale.shape[1]
    values = codes.astype(np.float32).reshape(rows, columns // group, group)
    out = values * scale[:, :, None] + bias[:, :, None]
    return out.reshape(rows, columns)


def quantize_grouped(weight: np.ndarray, bits: int, group: int):
    """Per-group affine, matching export_h3_int6_native.py exactly."""
    rows, columns = weight.shape
    if group % 4 or columns % group:
        raise SystemExit(f"group {group} must divide {columns} and be a multiple of 4")
    view = weight.reshape(rows, columns // group, group).astype(np.float32)
    levels = float(2 ** bits - 1)
    low = view.min(axis=2)
    high = view.max(axis=2)
    scale = np.maximum((high - low) / levels, 1e-12)
    zero = np.rint(-low / scale)
    bias = -zero * scale
    codes = np.clip(np.rint(view / scale[:, :, None]) + zero[:, :, None],
                    0.0, levels).astype(np.uint8)
    return codes.reshape(rows, columns), scale.astype(np.float32), bias.astype(np.float32)


def dequantize_grouped(codes: np.ndarray, scale: np.ndarray,
                       bias: np.ndarray) -> np.ndarray:
    rows, columns = codes.shape
    group = columns // scale.shape[1]
    return (codes.astype(np.float32).reshape(rows, -1, group) * scale[:, :, None]
            + bias[:, :, None]).reshape(rows, columns)


def pack_4bit(codes: np.ndarray) -> np.ndarray:
    """[rows, columns] codes in [0,15] -> [rows, columns/2], low nibble first."""
    rows, columns = codes.shape
    if columns % 2:
        raise SystemExit("columns must be even for 4-bit packing")
    pairs = codes.reshape(rows, columns // 2, 2).astype(np.uint8)
    return np.ascontiguousarray(pairs[:, :, 0] | (pairs[:, :, 1] << 4))


def rotate_rows(weight: np.ndarray) -> np.ndarray:
    """ConvRot rotation: x @ H, H = normalized Hadamard, block-diagonal over 256.

    Mirrors h3_weights.c::convrot_unrotate_row.  H is symmetric and orthonormal,
    so this is its own inverse -- the engine applies the identical butterfly.
    """
    rows, columns = weight.shape
    if columns % ROT_BLOCK:
        raise SystemExit(f"input dim {columns} is not a multiple of {ROT_BLOCK}")
    # Explicit copy: ascontiguousarray returns the *same* object for an already
    # contiguous float32 input, which would rotate the caller's array in place.
    out = np.array(weight, dtype=np.float32, order="C", copy=True)
    view = out.reshape(-1, ROT_BLOCK)
    stride = 1
    while stride < ROT_BLOCK:
        span = stride * 4
        quads = view.reshape(-1, ROT_BLOCK // span, 4, stride)
        a = quads[:, :, 0, :].copy()
        b = quads[:, :, 1, :].copy()
        c = quads[:, :, 2, :].copy()
        d = quads[:, :, 3, :].copy()
        quads[:, :, 0, :] = a + b + c - d
        quads[:, :, 1, :] = a + b - c + d
        quads[:, :, 2, :] = a - b + c + d
        quads[:, :, 3, :] = -a + b + c + d
        stride *= 4
    view *= 1.0 / 16.0
    return out


def to_slabs(weight: np.ndarray, input_layout: str) -> np.ndarray:
    """Return qkv rows as global slabs [all-q; all-k; all-v].

    That is what the engine's streaming grouped path consumes: it passes
    layout=1 for STREAM_QKV, i.e. `dst = 3*(slot%heads) + slot/heads`, which only
    recovers the right rows if the stored slot order is `which*heads + head`.
    """
    expected = NUM_HEADS * 3 * HEAD_DIM
    if weight.shape[0] != expected:
        raise SystemExit(f"unexpected qkv rows {weight.shape[0]}, expected {expected}")
    if input_layout == "slabs":
        return weight
    blocks = weight.reshape(NUM_HEADS, 3, HEAD_DIM, -1)
    return np.ascontiguousarray(
        np.concatenate([blocks[:, index] for index in range(3)], axis=0)
    ).reshape(expected, -1)


def slabs_to_interleaved(weight: np.ndarray) -> np.ndarray:
    """slabs -> per-head interleaved, matching the released BF16 row order."""
    expected = NUM_HEADS * 3 * HEAD_DIM
    if weight.shape[0] != expected:
        raise SystemExit(f"unexpected qkv rows {weight.shape[0]}, expected {expected}")
    slabs = weight.reshape(3, NUM_HEADS, HEAD_DIM, -1)
    return np.ascontiguousarray(
        slabs.transpose(1, 0, 2, 3)).reshape(expected, -1)


def resolved_qkv_input(args) -> str:
    """Row layout of the *input* qkv_proj; the stored output is always slabs."""
    if args.qkv_input != "auto":
        return args.qkv_input
    return "slabs" if args.source == "q4" else "interleaved"


def engine_rows(weight: np.ndarray, key: str) -> np.ndarray:
    """The row order the engine ends up with for a matrix we stored as `weight`.

    The released BF16 checkpoint is per-head interleaved, so comparing a stored
    (slabs) qkv against BF16 requires undoing the engine's remap first.
    """
    return slabs_to_interleaved(weight) if "qkv_proj" in key else weight


def rel_rms(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.sqrt(np.mean((a - b) ** 2))
                 / (np.sqrt(np.mean(b ** 2)) + 1e-12))


# --------------------------------------------------------------------------- #
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--q4", type=Path, default=DEFAULT_Q4,
                        help="mere-run MLX affine q4 transformer.safetensors")
    parser.add_argument("--base", type=Path, default=DEFAULT_BASE,
                        help="released BF16 transformer dir supplying every "
                             "non-quantized key (incl. blocks.N.adaln_proj)")
    parser.add_argument("--output", type=Path, help="output transformer dir")
    parser.add_argument("--bits", type=int, default=4, choices=[4])
    parser.add_argument("--group", type=int, default=64)
    parser.add_argument("--accumulator", default="f16", choices=["f16", "f32"])
    parser.add_argument("--rotate", dest="rotate", action="store_true", default=True,
                        help="apply the ConvRot rotation (default; required by h3.c)")
    parser.add_argument("--no-rotate", dest="rotate", action="store_false",
                        help="store unrotated; needs H3_INT8_UNROTATE=0 on a patched build")
    parser.add_argument("--qkv-input", default="auto",
                        choices=["auto", "slabs", "interleaved"],
                        help="row layout of the *input* qkv_proj: the mere-run q4 file "
                             "stores global slabs, the released BF16 stores per-head "
                             "interleaved. The stored output is always slabs, because "
                             "the engine's grouped path remaps slabs->interleaved on load. "
                             "'auto' picks by --source")
    parser.add_argument("--source", default="q4", choices=["q4", "bf16"],
                        help="'q4' (default) requantizes the mere-run 4-bit file, which "
                             "double-quantizes because the rotation has to be re-applied; "
                             "'bf16' quantizes the released BF16 once, in rotated space, "
                             "for the same output size and strictly less error")
    parser.add_argument("--token-refiner", default="bf16", choices=["bf16", "quantize"],
                        help="the 2 token_refiner blocks are read by the text-encoder "
                             "module (h3_text_encoder.c:227), which hard-requires BF16 and "
                             "cannot read U8 packed weights, so they stay BF16 (1.44 GiB). "
                             "'quantize' writes them anyway and the engine will reject it")
    parser.add_argument("--limit-blocks", type=int, default=0,
                        help="convert only blocks [0,N); the rest copy through as BF16")
    parser.add_argument("--shard-bytes", type=int, default=SHARD_BYTES)
    parser.add_argument("--verify", action="store_true",
                        help="numerical checks before/while writing")
    return parser


def base_index(base: Path):
    index = json.loads((base / "model.safetensors.index.json").read_text())
    weight_map = index["weight_map"]
    order = sorted(weight_map)
    return {key: base / weight_map[key] for key in order}, order


def quantized_keys(q4: Path) -> list[str]:
    header, _ = st_header(q4)
    return sorted(key for key in header if key != "__metadata__"
                  and key.endswith(".weight")
                  and key.replace(".weight", ".scales") in header)


def targets(keys, limit_blocks: int, token_refiner: str) -> set[str]:
    selected = set()
    for key in keys:
        head = key[: -len(".weight")]
        if head.startswith("blocks."):
            if limit_blocks and int(head.split(".")[1]) >= limit_blocks:
                continue
            selected.add(key)
        elif head.startswith("token_refiner.blocks."):
            if limit_blocks or token_refiner == "bf16":
                continue
            selected.add(key)
    return selected


def source_reference(key: str, args) -> np.ndarray:
    """The float matrix the conversion starts from, per --source."""
    if args.source == "bf16":
        return bf16_to_f32(st_read(args.base_of[key], key)[0])
    packed, _ = st_read(args.q4, key)
    return dequantize_mlx(
        packed.astype(np.uint32),
        st_read(args.q4, key.replace(".weight", ".scales"))[0],
        st_read(args.q4, key.replace(".weight", ".biases"))[0], 4)


def convert_one(key: str, args):
    """Return (packed_bytes, scale_bytes, bias_bytes, fidelity dict)."""
    reference = source_reference(key, args)

    working = (to_slabs(reference, resolved_qkv_input(args))
               if "qkv_proj" in key else reference)
    rotated = rotate_rows(working) if args.rotate else working
    codes, scale, bias = quantize_grouped(rotated, args.bits, args.group)

    # What the engine will actually reconstruct: dequant then un-rotate.
    restored = dequantize_grouped(codes, scale, bias)
    if args.rotate:
        restored = rotate_rows(restored)
    # Two different yardsticks, and only the second is comparable across
    # --source: error against the conversion input, and cumulative error against
    # the released BF16 (the q4 input is itself ~0.1 away from BF16).
    truth = (reference if args.source == "bf16"
             else bf16_to_f32(st_read(args.base_of[key], key)[0]))
    fidelity = {"rows": reference.shape[0], "columns": reference.shape[1],
                # `working` is the input in the stored (slabs) row frame, so this
                # is the pure quantization error.
                "end_to_end": rel_rms(restored, working),
                "vs_bf16": rel_rms(engine_rows(restored, key), truth)}

    accum = np.float16 if args.accumulator == "f16" else np.float32
    return (pack_4bit(codes).tobytes(),
            scale.astype(accum).tobytes(),
            bias.astype(accum).tobytes(),
            fidelity)


def verify_written(output: Path, index_map: dict, base: Path, sample: list[str],
                   args) -> float:
    """Re-derive the format from disk exactly as the engine does."""
    print("\n[verify] re-deriving the written format from disk")
    accum_dtype = "F16" if args.accumulator == "f16" else "F32"
    worst = 0.0
    for key in sample:
        shard = output / index_map[key]
        header, _ = st_header(shard)
        packed, packed_dtype = st_read(shard, key)
        scale_raw, _ = st_read(shard, scale_key(key))
        bias_raw, _ = st_read(shard, bias_key(key))
        # The accumulators are written as np.float16 / np.float32, and `st_read`
        # already decodes those dtypes (F16 -> "<f2"), so this is a plain widen.
        # Do NOT route F16 through bf16_to_f32: it reinterprets the bit pattern
        # as a bf16 exponent/mantissa pair and produces garbage.
        scale = scale_raw.astype(np.float32)
        bias = bias_raw.astype(np.float32)

        rows, packed_columns = packed.shape
        codes = np.empty((rows, packed_columns * 2), dtype=np.uint8)
        codes[:, 0::2] = packed & 0x0F
        codes[:, 1::2] = packed >> 4
        # Same derivation the engine uses: bits = packed_row_bytes*8 / columns.
        derived_bits = packed_columns * 8 // codes.shape[1]
        derived_group = codes.shape[1] // scale.shape[1]
        restored = dequantize_grouped(codes, scale, bias)
        if args.rotate:
            restored = rotate_rows(restored)

        reference = source_reference(key, args)
        truth = bf16_to_f32(st_read(base, key)[0])
        error = rel_rms(restored, reference)
        worst = max(worst, error)
        dtype_ok = (packed_dtype == "U8"
                    and header[scale_key(key)]["dtype"] == accum_dtype
                    and header[bias_key(key)]["dtype"] == accum_dtype)
        print(f"[verify] {key:44s} bits={derived_bits} group={derived_group} "
              f"packed={packed_dtype}{packed.shape} "
              f"vs {args.source}={error:.5f} "
              f"vs bf16={rel_rms(engine_rows(restored, key), truth):.5f}"
              f"{'' if dtype_ok else '   <-- DTYPE MISMATCH'}")
    print(f"[verify] worst vs {args.source} = {worst:.5f}")
    return worst


def main() -> int:
    args = build_parser().parse_args()
    if not args.q4.is_file():
        raise SystemExit(f"q4 checkpoint not found: {args.q4}")
    if not (args.base / "model.safetensors.index.json").is_file():
        raise SystemExit(f"base transformer dir has no index: {args.base}")

    keys = quantized_keys(args.q4)
    selected = targets(keys, args.limit_blocks, args.token_refiner)
    print(f"[plan] {len(keys)} quantized matrices in {args.q4.name}; "
          f"converting {len(selected)}"
          + (f" (limit-blocks {args.limit_blocks})" if args.limit_blocks else ""))
    print(f"[plan] token_refiner={args.token_refiner} "
          f"source={args.source} rotation={'on' if args.rotate else 'OFF'} "
          f"qkv-input={resolved_qkv_input(args)} bits={args.bits} group={args.group} "
          f"accumulator={args.accumulator}")

    shard_of, order = base_index(args.base)
    args.base_of = shard_of
    missing = [key for key in sorted(selected) if key not in shard_of]
    if missing:
        raise SystemExit(f"{len(missing)} converted keys are absent from the base "
                         f"checkpoint, e.g. {missing[:3]}")

    fidelity: list[tuple[str, float]] = []
    started = time.perf_counter()

    if args.verify and args.output is None:
        print("\n[verify] rotation involution (R is symmetric and orthonormal, so "
              "R@R = I)")
        probe = np.random.default_rng(0).standard_normal((8, ROT_BLOCK)).astype(np.float32)
        print(f"  max|R(R(x)) - x| = {np.abs(rotate_rows(rotate_rows(probe)) - probe).max():.3e}")
        print("\n[verify] engine round-trip on a sample (nothing is written)")
        for key in sorted(selected)[:4]:
            _, _, _, stats = convert_one(key, args)
            print(f"  {key:44s} vs {args.source}={stats['end_to_end']:.5f} "
                  f"vs bf16={stats['vs_bf16']:.5f} "
                  f"[{stats['rows']},{stats['columns']}]")
        return 0

    if args.output is None:
        raise SystemExit("--output is required (or use --verify alone)")
    output: Path = args.output
    if output.exists():
        raise SystemExit(f"output already exists: {output} (remove it first)")
    output.mkdir(parents=True)

    # One entry per emitted tensor, so a shard boundary can never split a
    # matrix's (weight, scale, bias) triple.  Logical shapes come from the base
    # header, which is authoritative for both --source values.
    headers = {path: st_header(path)[0] for path in set(shard_of.values())}
    # Regression guard for the engine's probe convention.  A `_weight_scale`
    # suffix (i.e. appending to the module name instead of the matrix key)
    # produces a file that `--info` happily counts but that the DiT streamer
    # rejects at load time with "grouped streaming weight schema mismatch",
    # because `h3_weight_grouped_spec` returns 0 when the aux tensor is absent.
    assert scale_key("blocks.0.attn.qkv_proj.weight") == \
        "blocks.0.attn.qkv_proj.weight_scale", "engine probes {key}_scale"
    assert bias_key("blocks.0.attn.qkv_proj.weight") == \
        "blocks.0.attn.qkv_proj.weight_bias", "engine probes {key}_bias"

    expanded: list[tuple[str, str, tuple, tuple]] = []
    for key in order:
        if key in selected:
            rows, columns = headers[shard_of[key]][key]["shape"]
            accum_dtype = "F16" if args.accumulator == "f16" else "F32"
            accum_shape = (rows, columns // args.group)
            expanded.append((key, "U8", (rows, columns // 2), ("q4", key)))
            expanded.append((scale_key(key), accum_dtype, accum_shape,
                             ("q4", key)))
            expanded.append((bias_key(key), accum_dtype, accum_shape,
                             ("q4", key)))
        else:
            entry = headers[shard_of[key]][key]
            expanded.append((key, entry["dtype"], tuple(entry["shape"]),
                             ("base", shard_of[key], key)))

    cache: dict = {}

    def converted(key: str):
        if key not in cache:
            cache.clear()                     # one matrix resident at a time
            result = convert_one(key, args)
            cache[key] = result
            fidelity.append((key, result[3]["vs_bf16"]))
        return cache[key]

    def produce(source, key: str, shape):
        if source[0] == "base":
            return lambda: st_read_bytes(source[1], source[2])

        def emit():
            packed_bytes, scale_bytes, bias_bytes, _ = converted(source[1])
            if key.endswith(".weight"):
                return packed_bytes
            if key == scale_key(source[1]):
                return scale_bytes
            return bias_bytes

        return emit

    def size_of(dtype, shape):
        count = 1
        for dimension in shape:
            count *= dimension
        return count * _BYTES_OF[dtype]

    shard_entries: list = []
    shard_bytes = 0
    index_map: dict[str, str] = {}
    shard = 0

    def flush():
        nonlocal shard_entries, shard_bytes, shard
        if not shard_entries:
            return
        name = f"model-{shard + 1:05d}.safetensors"
        st_write(output / name, shard_entries)
        for entry_key, _, _, _ in shard_entries:
            index_map[entry_key] = name
        print(f"  [shard {shard}] {name}: {len(shard_entries)} tensors, "
              f"{shard_bytes / 2**30:.2f} GiB ({time.perf_counter() - started:.0f}s)")
        shard += 1
        shard_entries = []
        shard_bytes = 0

    for key, dtype, shape, source in expanded:
        shard_entries.append((key, dtype, shape, produce(source, key, shape)))
        shard_bytes += size_of(dtype, shape)
        # A converted matrix's triple is always adjacent, so only flush once the
        # bias has been queued -- never between a weight and its accumulators.
        if shard_bytes >= args.shard_bytes and (
                source[0] == "base" or key == bias_key(source[1])):
            flush()
    flush()

    (output / "model.safetensors.index.json").write_text(
        json.dumps({"metadata": {}, "weight_map": index_map}, indent=2))
    config = args.base / "config.json"
    if config.is_file():
        shutil.copy2(config, output / "config.json")

    if fidelity:
        values = [value for _, value in fidelity]
        print(f"\n[done] engine-roundtrip relRMS vs BF16: mean={np.mean(values):.5f} "
              f"max={np.max(values):.5f} over {len(values)} matrices")
    total = sum(file.stat().st_size for file in output.glob("*.safetensors"))
    print(f"[done] {len(index_map)} tensors, {shard} shards, "
          f"{total / 2**30:.2f} GiB, {time.perf_counter() - started:.0f}s")
    print(f"       output: {output}")
    print(f"       run with: ./h3 -d {output} --ssd-streaming ...")

    if args.verify:
        sample = sorted(selected)[:4]
        verify_written(output, index_map, args.base, sample, args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
