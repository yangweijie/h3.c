#!/usr/bin/env python3
"""Build the fixture h3_adaln_cache_lora_gate_test needs: a cache that selects the
cached AdaLN path, plus adapters that do and do not target AdaLN.

Nothing here has to be numerically meaningful. The gate under test runs *before*
`load_cached_adaln` reads the 51 modulation tensors, and the modulation itself is
only reached by the "gate accepted" case -- which is expected to stop at the
schedule-key prefix check because `adaln_cache_times_s1` is deliberately full of
-12345.0, a value no `1 - sigma` row can take. So one block tensor (< 1 MiB) is
enough, and the fixture never has to reproduce a real checkpoint.

    python3 tests/gen_adaln_cache_fixture.py <dir>

Writes, under the given directory:
    with_meta/adaln_cache_s1.safetensors     times + blocks.0 + final +
                                             meta U32[1] = 2688
    without_meta/adaln_cache_s1.safetensors  the same, no meta key
    adapters/adapter_attn_only.safetensors   attn.orig.to_q (no AdaLN target)
    adapters/adapter_adaln.safetensors       adaln_proj.linear (targets AdaLN)
    adapters/adapter_norm_out.safetensors    norm_out.linear (the final layer)
"""

import json
import struct
import sys
from pathlib import Path

BLOCK_OUTPUT = 96768
FINAL_OUTPUT = 10752
ROWS = 3          # steps=1: one shared step row, then the visual and audio rows
STEPS = 1
TIME_DIM = 2688   # what the meta key declares, and what the adapter must fit
RANK = 16
BAD_TIME = -12345.0


def write_safetensors(path: Path, tensors) -> None:
    """tensors: list of (key, dtype, shape, raw bytes)."""
    header, offset = {}, 0
    for key, dtype, shape, data in tensors:
        header[key] = {"dtype": dtype, "shape": list(shape),
                       "data_offsets": [offset, offset + len(data)]}
        offset += len(data)
    blob = json.dumps(header, separators=(",", ":")).encode()
    padding = -(len(blob) + 8) % 8
    with path.open("wb") as handle:
        handle.write(struct.pack("<Q", len(blob) + padding))
        handle.write(blob + b" " * padding)
        for _, _, _, data in tensors:
            handle.write(data)


def cache_tensors(meta: bool):
    bf16_rows = b"\x00" * (ROWS * BLOCK_OUTPUT * 2)
    tensors = [("adaln_cache_times_s%d" % STEPS, "F32", [ROWS],
                struct.pack("<%df" % ROWS, *[BAD_TIME] * ROWS)),
               ("blocks.0.adaln_cache_s%d" % STEPS, "BF16", [ROWS, BLOCK_OUTPUT],
                bf16_rows)]
    if meta:
        tensors.append(("adaln_cache_meta_s%d" % STEPS, "U32", [1],
                        struct.pack("<I", TIME_DIM)))
    tensors.append(("final_layer.adaln_cache_s%d" % STEPS, "BF16",
                    [ROWS, FINAL_OUTPUT], b"\x00" * (ROWS * FINAL_OUTPUT * 2)))
    return tensors


def adapter_tensors(target: str):
    """A rank-`RANK` two-factor adapter for `target`, with an `lora_A` wide enough
    to fit the width the meta key declares -- which is all `h3_lora_matches`
    inspects. Key shape mirrors what `merge_adaln_loras` asks for:
    `{prefix}.{target}.lora_A.{adapter}.weight`."""
    prefix = {"attn_only": "transformer_blocks.0.attn.orig.to_q",
              "adaln": "transformer_blocks.0.adaln_proj.linear",
              "norm_out": "norm_out.linear"}[target]
    return [(f"{prefix}.lora_A.default.weight", "BF16", [RANK, TIME_DIM],
             b"\x00" * (RANK * TIME_DIM * 2)),
            (f"{prefix}.lora_B.default.weight", "BF16", [7168, RANK],
             b"\x00" * (7168 * RANK * 2))]


def main() -> int:
    if len(sys.argv) != 2:
        print(__doc__)
        return 2
    out = Path(sys.argv[1])
    # Two directories rather than two files in one: both would otherwise declare
    # the same `blocks.0.adaln_cache_s1` key and the store lookup would be
    # order-dependent.
    for name, meta in (("with_meta", True), ("without_meta", False)):
        directory = out / name
        directory.mkdir(parents=True, exist_ok=True)
        write_safetensors(directory / "adaln_cache_s1.safetensors",
                          cache_tensors(meta))
    adapters = out / "adapters"
    adapters.mkdir(parents=True, exist_ok=True)
    for target in ("attn_only", "adaln", "norm_out"):
        write_safetensors(adapters / f"adapter_{target}.safetensors",
                          adapter_tensors(target))
    total = sum(path.stat().st_size for path in out.rglob("*.safetensors"))
    print(f"{out}: 5 shards, {total / 2**20:.1f} MiB "
          f"(meta width {TIME_DIM}, {ROWS} rows)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
