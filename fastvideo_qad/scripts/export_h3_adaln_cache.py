#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Install precomputed AdaLN modulation caches into an h3.c DiT checkpoint.

Why this script exists
----------------------
`h3_dit_schedule_precompute` materializes the AdaLN modulation for **every**
step before denoising starts, one block at a time, and frees each projection as
soon as it has been applied (`h3_dit_schedule.c`). The 51 `adaln_proj.linear`
matrices are therefore read exactly once per run and never touched again -- but
they still have to sit on disk, and on the released architecture they are the
single largest group in the checkpoint:

    blocks.N.adaln_proj.linear.weight   BF16 [96768, 2688] x 50  = 24.22 GiB
    final_layer.adaln_proj.linear.*     BF16 [10752, 2688]       =  0.05 GiB
    biases (50 + 1)                     BF16 [96768] / [10752]   =  0.01 GiB

The modulation they produce is a pure function of the sigma schedule, so a
checkpoint can ship the *output* instead of the weights. The same trade mere-run
makes with `cache_covered_weights_omitted: true`.

One cache per step count
------------------------
The modulation depends on `--steps`, so a cache is keyed by it and several can
live side by side in one checkpoint, letting a run switch step counts without
re-exporting:

    adaln_cache_s8.safetensors
      adaln_cache_times_s8       F32  [17]
      blocks.N.adaln_cache_s8    BF16 [17, 96768]
      final_layer.adaln_cache_s8 BF16 [17, 10752]
      adaln_cache_meta_s8        U32  [1]     AdaLN input width (optional)

The row values depend only on `--steps` and the conditioning mode -- not on
resolution, frame count or seed -- so one cache covers every run at that step
count. h3.c probes `blocks.0.adaln_cache_s{steps}` and refuses a cache that is
too short or whose values disagree, so a wrong step count fails loudly instead
of silently applying the wrong modulation.

One cache covers every conditioning mode
----------------------------------------
The conditioning modes differ only in the condition rows appended *after* the
step rows, in the fixed order visual then audio, so their row values nest:

    text-to-video      2*steps - 1 rows   no condition row
    first/last frame   2*steps     rows   visual row
    reference image    2*steps + 1 rows   visual row, audio row

h3.c forces both condition rows on while dumping, so a single cache serves all
three -- the shorter modes match it as a prefix -- for the cost of two unused
rows (~20 MiB). Materializing only the visual row would leave reference images
one row short and refused, with "rebuild the cache" advice that rebuilding
could never satisfy. Audio-only references (`H3_LAYOUT_REF_AUDIO` with no
image) are the one mode no cache covers, because their single condition row
would land where the visual row sits; h3.c rejects them rather than misapplying
it. They are not reachable from the CLI or REPL, only from the C API.

LoRA adapters merge into the AdaLN weights a cached checkpoint has dropped, so an
adapter that *targets* AdaLN cannot be used with one. Which adapters do that is
decided by `adaln_cache_meta_s{steps}`, written whenever this script can tell the
AdaLN input width: from the `--model` checkpoint's own `adaln_proj` shape, or else
from a width an already-installed cache recorded. An adapter with no AdaLN factor
(an attention-only one, like the VDN `.default` set) therefore loads against a
cached checkpoint unchanged, while the turbo set -- which carries
`norm_out.linear` -- is refused. If no width can be determined the key is omitted
and h3.c refuses every adapter, which is the rule caches exported before this key
existed have always had.

Usage
-----
    # 1) build one cache per step count, from a checkpoint that still has the
    #    weights (the exporter only needs to reach the AdaLN precompute)
    for steps in 4 8 20; do
        H3_DIT_ADALN_CACHE_DUMP=/tmp/adaln_s$steps.raw \\
            ./h3 -d <full model> --ssd-streaming ... --steps $steps
    done

    # 2) install them, writing a trimmed copy of the model next to it
    python export_h3_adaln_cache.py \\
        --dump /tmp/adaln_s4.raw --dump /tmp/adaln_s8.raw --dump /tmp/adaln_s20.raw \\
        --model /Volumes/data/MODELS/h3c-q4-native/FL2VA/transformer \\
        --trim-to /Volumes/data/MODELS/h3c-q4-adalncache/FL2VA/transformer

    # 3) check without writing anything
    python export_h3_adaln_cache.py --dump /tmp/adaln_s8.raw \\
        --model <trimmed dir> --verify

`--model` writes `adaln_cache_s<steps>.safetensors` in place (a new file,
nothing is rewritten). `--trim-to` writes a *new* directory, so the source model
is never modified; the AdaLN matrices are the only keys dropped.
"""

from __future__ import annotations

import argparse
import json
import shutil
import struct
import sys
import time
from pathlib import Path

import numpy as np

MAGIC = b"H3ADALN2"
HEADER_BYTES = 8 + 6 * 4  # magic + 6 uint32 fields
BLOCKS = 50
BLOCK_OUTPUT = 96768
FINAL_OUTPUT = 10752
CACHE_PREFIX = "adaln_cache_s"

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


def range_reader(path: Path, offset: int, length: int):
    """A producer that streams one byte range out of `path`."""
    def produce() -> bytes:
        with path.open("rb") as handle:
            handle.seek(offset)
            return handle.read(length)
    return produce


def is_cache_file(path: Path) -> bool:
    return path.name.startswith(CACHE_PREFIX)


def cache_steps_of(path: Path) -> int | None:
    """The step count a cache file name carries, or None if it is malformed."""
    name = path.name
    if not is_cache_file(path):
        return None
    tail = name[len(CACHE_PREFIX):]
    for split in (".", "_"):
        if split in tail:
            tail = tail.split(split, 1)[0]
            break
    return int(tail) if tail.isdigit() else None


META_PREFIX = "adaln_cache_meta_s"


def meta_key(steps: int) -> str:
    return f"{META_PREFIX}{steps}"


def read_tensor_bytes(path: Path, entry, base: int) -> bytes:
    begin, end = entry["data_offsets"]
    return range_reader(path, base + begin, end - begin)()


def weight_time_dim(model: Path) -> int | None:
    """The AdaLN input width the surviving `adaln_proj` weights declare."""
    for shard in sorted(model.glob("*.safetensors")):
        if is_cache_file(shard):
            continue
        header, _ = st_header(shard)
        entry = header.get("blocks.0.adaln_proj.linear.weight")
        if entry and len(entry["shape"]) == 2:
            return int(entry["shape"][1])
    return None


def source_time_dim(model: Path) -> int | None:
    """The AdaLN input width this directory declares, if it still declares one.

    A trimmed directory no longer has `adaln_proj` weights, so fall back to a
    width an already-installed cache recorded -- which is what keeps a later
    `--dump` for the same directory agreeing with the earlier ones."""
    width = weight_time_dim(model)
    if width:
        return width
    for shard in sorted(model.glob(f"{CACHE_PREFIX}*.safetensors")):
        header, base = st_header(shard)
        for key, entry in header.items():
            if key.startswith(META_PREFIX) and entry["dtype"] == "U32":
                return int(struct.unpack("<I", read_tensor_bytes(
                    shard, entry, base))[0])
    return None


# --------------------------------------------------------------------------- #
# the raw dump written by H3_DIT_ADALN_CACHE_DUMP
# --------------------------------------------------------------------------- #
class Dump:
    """Streams the modulation straight out of the raw blob into safetensors.

    Nothing is materialized: at `--steps 1000` the dump is ~9.7 GiB, so the
    entries below seek instead of holding the payload.
    """

    def __init__(self, path: Path):
        self.path = path
        size = path.stat().st_size
        with path.open("rb") as handle:
            magic = handle.read(8)
            if magic != MAGIC:
                hint = (" (H3ADALN1: an older build; re-dump with the current "
                        "h3)" if magic == b"H3ADALN1" else "")
                raise SystemExit(f"{path}: bad magic {magic!r}, expected "
                                 f"{MAGIC!r}{hint}")
            steps, time_rows, blocks, block_output, final_output, _ = \
                struct.unpack("<6I", handle.read(24))
        if blocks != BLOCKS:
            raise SystemExit(f"{path}: {blocks} blocks, expected {BLOCKS}")
        if block_output != BLOCK_OUTPUT or final_output != FINAL_OUTPUT:
            raise SystemExit(
                f"{path}: output widths {block_output}/{final_output}, expected "
                f"{BLOCK_OUTPUT}/{FINAL_OUTPUT}")
        if not steps or not time_rows:
            raise SystemExit(f"{path}: steps={steps} time_rows={time_rows}")
        self.steps = steps
        self.time_rows = time_rows
        self.blocks = blocks
        self.block_output = block_output
        self.final_output = final_output

        self.times_offset = HEADER_BYTES
        self.times_bytes = time_rows * 4
        self.block_bytes = time_rows * block_output * 2
        self.final_bytes = time_rows * final_output * 2
        self.blocks_offset = self.times_offset + self.times_bytes
        self.final_offset = self.blocks_offset + blocks * self.block_bytes
        expected = self.final_offset + self.final_bytes
        if size != expected:
            raise SystemExit(
                f"{path}: {size} bytes, but the header describes {expected}; "
                "the dump is truncated or the header is wrong")

    @property
    def cache_name(self) -> str:
        return f"{CACHE_PREFIX}{self.steps}.safetensors"

    def times(self) -> np.ndarray:
        with self.path.open("rb") as handle:
            handle.seek(self.times_offset)
            raw = handle.read(self.times_bytes)
        return np.frombuffer(raw, dtype="<f4").copy()

    def block_producer(self, block: int):
        return range_reader(self.path,
                            self.blocks_offset + block * self.block_bytes,
                            self.block_bytes)

    def final_producer(self):
        return range_reader(self.path, self.final_offset, self.final_bytes)

    def entries(self):
        yield (f"adaln_cache_times_s{self.steps}", "F32", (self.time_rows,),
               range_reader(self.path, self.times_offset, self.times_bytes))
        for block in range(self.blocks):
            yield (f"blocks.{block}.adaln_cache_s{self.steps}", "BF16",
                   (self.time_rows, self.block_output),
                   self.block_producer(block))
        yield (f"final_layer.adaln_cache_s{self.steps}", "BF16",
               (self.time_rows, self.final_output), self.final_producer())

    def describe(self) -> str:
        return (f"steps={self.steps}, {self.time_rows} rows, {self.blocks} "
                f"blocks [{self.time_rows}, {self.block_output}] + final "
                f"[{self.time_rows}, {self.final_output}]")


def install(dumps, model: Path, time_dim: int | None) -> list[Path]:
    written = []
    for dump in dumps:
        entries = list(dump.entries())
        if time_dim:
            entries.append((meta_key(dump.steps), "U32", (1,),
                            lambda width=time_dim: struct.pack("<I", width)))
        target = model / dump.cache_name
        started = time.time()
        st_write(target, entries)
        print(f"  wrote {target.name}  "
              f"({target.stat().st_size / 2**20:.1f} MiB, "
              f"{time.time() - started:.1f}s, "
              f"{'AdaLN width %d' % time_dim if time_dim else 'no width key'})")
        written.append(target)
    return written


# --------------------------------------------------------------------------- #
# trimming: drop the AdaLN matrices the caches replace
# --------------------------------------------------------------------------- #
def is_adaln_weight(key: str) -> bool:
    return ".adaln_proj.linear." in key


def trim(model: Path, out_dir: Path, dumps) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    shards = sorted(model.glob("*.safetensors"))
    if not shards:
        raise SystemExit(f"{model}: no *.safetensors shards")
    time_dim = source_time_dim(model)

    dropped_bytes = 0
    dropped_keys = 0
    kept_bytes = 0
    for shard in shards:
        if is_cache_file(shard):
            continue
        header, base = st_header(shard)
        entries = []
        for key, entry in header.items():
            count = 1
            for dimension in entry["shape"]:
                count *= dimension
            size = count * _BYTES_OF[entry["dtype"]]
            begin, end = entry["data_offsets"]
            assert end - begin == size, f"{shard}:{key} offset/size mismatch"
            if is_adaln_weight(key):
                dropped_bytes += size
                dropped_keys += 1
                continue
            kept_bytes += size
            entries.append((key, entry["dtype"], entry["shape"],
                            range_reader(shard, base + begin, size)))
        if not entries:
            print(f"  {shard.name}: every tensor was AdaLN, skipped")
            continue
        st_write(out_dir / shard.name, entries)

    for other in sorted(model.iterdir()):
        if other.is_file() and other.suffix != ".safetensors":
            shutil.copy2(other, out_dir / other.name)
    print(f"  dropped {dropped_keys} AdaLN tensors "
          f"({dropped_bytes / 2**30:.2f} GiB), kept "
          f"{kept_bytes / 2**30:.2f} GiB of weights")
    install(dumps, out_dir, time_dim)
    # One cache per step count is the design, so a directory trimmed with only
    # some of the --dump files would otherwise lose the step counts it already
    # carried -- and `--verify` only checks the dumps it was handed, so the loss
    # would go unreported.
    installing = {dump.steps for dump in dumps}
    carried = []
    for shard in sorted(model.glob(f"{CACHE_PREFIX}*.safetensors")):
        steps = cache_steps_of(shard)
        if steps is None or steps in installing:
            continue
        shutil.copy2(shard, out_dir / shard.name)
        carried.append(shard.name)
    if carried:
        print(f"  carried {len(carried)} existing cache shard(s) across: "
              f"{', '.join(carried)}")
    return {"dropped_bytes": dropped_bytes, "dropped_keys": dropped_keys,
            "kept_bytes": kept_bytes}


# --------------------------------------------------------------------------- #
# verification
# --------------------------------------------------------------------------- #
def verify(dumps, model: Path) -> int:
    failures = 0
    declared = weight_time_dim(model)

    def check(condition: bool, message: str) -> None:
        nonlocal failures
        print(("  ok   " if condition else "  FAIL ") + message)
        if not condition:
            failures += 1

    for dump in dumps:
        cache = model / dump.cache_name
        print(f"{cache.name}:")
        if not cache.exists():
            check(False, f"{cache} does not exist")
            continue
        header, base = st_header(cache)
        expected_keys = (
            {f"adaln_cache_times_s{dump.steps}",
             f"final_layer.adaln_cache_s{dump.steps}"} |
            {f"blocks.{b}.adaln_cache_s{dump.steps}" for b in range(BLOCKS)})
        meta = meta_key(dump.steps)
        extra = {meta} if meta in header else set()
        check(set(header) == expected_keys | extra,
              f"holds the {len(expected_keys)} required keys"
              + (f" plus {meta}" if extra else
                 " and no width key (so h3.c refuses every LoRA adapter here)")
              + f" (found {len(header)})")

        entry = header.get(f"adaln_cache_times_s{dump.steps}")
        check(entry is not None and entry["dtype"] == "F32" and
              entry["shape"] == [dump.time_rows],
              f"schedule key is F32 [{dump.time_rows}]")
        if entry is None:
            continue
        times_cached = np.frombuffer(
            range_reader(cache, base + entry["data_offsets"][0],
                         dump.times_bytes)(), dtype="<f4")
        check(np.array_equal(times_cached, dump.times()),
              "schedule key matches the dump bit-for-bit")

        for label, key, producer in (
                ("block 0", f"blocks.0.adaln_cache_s{dump.steps}",
                 dump.block_producer(0)),
                ("block 49", f"blocks.{BLOCKS - 1}.adaln_cache_s{dump.steps}",
                 dump.block_producer(BLOCKS - 1)),
                ("final", f"final_layer.adaln_cache_s{dump.steps}",
                 dump.final_producer())):
            entry = header.get(key)
            if entry is None:
                check(False, f"{label}: {key} is missing")
                continue
            begin, end = entry["data_offsets"]
            check(range_reader(cache, base + begin, end - begin)() == producer(),
                  f"{label}: {key} matches the dump byte-for-byte")

        entry = header.get(meta)
        if entry is None:
            continue
        well_formed = entry["dtype"] == "U32" and entry["shape"] == [1]
        width = (int(struct.unpack("<I", read_tensor_bytes(cache, entry,
                                                           base))[0])
                 if well_formed else 0)
        check(well_formed and width > 0, f"{meta} is a non-zero U32 [1]")
        # Only a surviving adaln_proj weight can contradict the width a cache
        # claims; a second cache would agree with it by construction.
        if width and declared and width != declared:
            check(False, f"{meta} says {width} but the checkpoint's "
                         f"adaln_proj declares {declared}")

    stale = []
    for shard in sorted(model.glob("*.safetensors")):
        if is_cache_file(shard):
            continue
        shard_header, _ = st_header(shard)
        stale.extend(key for key in shard_header if is_adaln_weight(key))
    check(not stale, f"no adaln_proj weights remain ({len(stale)} found)")

    caches = sorted(model.glob(f"{CACHE_PREFIX}*.safetensors"))
    weights = sum(path.stat().st_size for path in model.glob("*.safetensors")
                  if not is_cache_file(path))
    cached = sum(path.stat().st_size for path in caches)
    print(f"  info  {len(caches)} cache(s) "
          f"[{', '.join(c.name for c in caches)}] = {cached / 2**20:.1f} MiB")
    print(f"  info  transformer dir is {(weights + cached) / 2**30:.2f} GiB "
          f"({weights / 2**30:.2f} GiB weights + {cached / 2**20:.1f} MiB cache)")
    print("PASS" if not failures else f"{failures} check(s) failed")
    return 1 if failures else 0


# --------------------------------------------------------------------------- #
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dump", type=Path, action="append", required=True,
                        metavar="PATH",
                        help="raw blob written by H3_DIT_ADALN_CACHE_DUMP; "
                             "repeat once per step count")
    parser.add_argument("--model", type=Path,
                        help="transformer dir to install the caches into")
    parser.add_argument("--trim-to", type=Path,
                        help="write a copy of --model here with the 51 "
                             "adaln_proj matrices removed")
    parser.add_argument("--verify", action="store_true",
                        help="check the caches and model instead of writing")
    parser.add_argument("--dry-run", action="store_true",
                        help="report what would be written and stop")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    dumps = []
    for path in args.dump:
        if not path.exists():
            raise SystemExit(f"{path}: no such file")
        dumps.append(Dump(path))
    for dump in dumps:
        print(f"{dump.path.name}: {dump.path.stat().st_size / 2**20:.1f} MiB, "
              f"{dump.describe()}")
    steps_seen = [dump.steps for dump in dumps]
    if len(set(steps_seen)) != len(steps_seen):
        raise SystemExit(f"duplicate --steps among dumps: {steps_seen}")

    if args.verify:
        if not args.model:
            raise SystemExit("--verify needs --model")
        return verify(dumps, args.model)

    if args.dry_run:
        if args.model:
            shards = [s for s in args.model.glob("*.safetensors")
                      if not is_cache_file(s)]
            dropped = 0
            for shard in shards:
                header, _ = st_header(shard)
                for key, entry in header.items():
                    if not is_adaln_weight(key):
                        continue
                    count = 1
                    for dimension in entry["shape"]:
                        count *= dimension
                    dropped += count * _BYTES_OF[entry["dtype"]]
            total = sum(path.stat().st_size for path in shards)
            cache_total = sum(dump.final_offset + dump.final_bytes
                              for dump in dumps)
            print(f"would drop {dropped / 2**30:.2f} GiB of AdaLN weights and "
                  f"add {cache_total / 2**20:.1f} MiB of cache, taking "
                  f"{args.model} from {total / 2**30:.2f} GiB to "
                  f"{(total - dropped + cache_total) / 2**30:.2f} GiB")
        return 0

    if not args.model and not args.trim_to:
        raise SystemExit("nothing to do: pass --model and/or --trim-to")
    if args.trim_to and not args.model:
        raise SystemExit("--trim-to needs --model")

    if args.trim_to:
        print(f"trimming {args.model} -> {args.trim_to}")
        trim(args.model, args.trim_to, dumps)
        total = sum(path.stat().st_size
                    for path in args.trim_to.glob("*.safetensors"))
        print(f"{args.trim_to} is now {total / 2**30:.2f} GiB")
    elif args.model:
        install(dumps, args.model, source_time_dim(args.model))
    return 0


if __name__ == "__main__":
    sys.exit(main())
