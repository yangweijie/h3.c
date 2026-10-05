#!/usr/bin/env python3
"""Gate for `add_h3_adaln_cache_meta.py`, the tool that rewrites a user's model
files, so its promises are checked rather than argued.

The tool claims three things: it only grows the header (no payload byte moves), it
takes the width from a checkpoint that declares it instead of inventing one, and
it refuses rather than half-applying. Each claim is checkable on a fixture, and
each is the kind of property that breaks silently when someone edits a header
writer.

The width declared by the fixture's source checkpoint is **1344** -- deliberately
neither the engine's real 2688 nor the fixture generator's own `TIME_DIM`, so a
tool that hardcoded either would fail the first check instead of passing by
coincidence.

Run directly: `python3 tests/test_adaln_cache_meta_tool.py`
"""

from __future__ import annotations

import hashlib
import json
import struct
import subprocess
import sys
import tempfile
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
TOOL = ROOT / "fastvideo_qad" / "scripts" / "add_h3_adaln_cache_meta.py"
FIXTURE = ROOT / "tests" / "gen_adaln_cache_fixture.py"

DECLARED_WIDTH = 1344        # not the engine's 2688, not the fixture's TIME_DIM
META_KEY = "adaln_cache_meta_s1"

failures: list[str] = []


def check(condition: bool, message: str) -> bool:
    print(("  ok   " if condition else "  FAIL ") + message)
    if not condition:
        failures.append(message)
    return condition


def load_fixture():
    spec = spec_from_file_location("fx", FIXTURE)
    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def st_read(path: Path):
    """(header, data_start, raw bytes) the way the tool itself reads it."""
    raw = path.read_bytes()
    length = struct.unpack("<Q", raw[:8])[0]
    return json.loads(raw[8:8 + length]), 8 + length, raw


def run_tool(cache: Path, source: Path, backup: Path) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, str(TOOL), "--cache-dir", str(cache),
         "--source-model", str(source), "--backup-dir", str(backup)],
        capture_output=True, text=True)


def build(tmp: Path, names=("adaln_cache_s1.safetensors",),
          width: int | None = DECLARED_WIDTH, metadata_names=()):
    """A cache directory of pre-meta shards, plus a source checkpoint that may or
    may not still declare the AdaLN input width."""
    fx = load_fixture()
    cache = tmp / "transformer"
    cache.mkdir()
    for name in names:
        fx.write_safetensors(cache / name, fx.cache_tensors(meta=False))
        if name in metadata_names:
            header, base, raw = st_read(cache / name)
            header["__metadata__"] = {"format": "pt"}
            blob = json.dumps(header, separators=(",", ":")).encode()
            blob += b" " * (-(len(blob) + 8) % 8)
            (cache / name).write_bytes(struct.pack("<Q", len(blob)) + blob + raw[base:])
    source = tmp / "source"
    source.mkdir()
    tensors = ([("blocks.0.adaln_proj.linear.weight", "BF16", [8, width],
                 b"\x00" * (8 * width * 2))] if width else
               [("blocks.0.attn.qkv.weight", "BF16", [8, 8], b"\x00" * 128)])
    fx.write_safetensors(source / "model-00001.safetensors", tensors)
    return cache, source, tmp / "backup"


def main() -> int:
    print("the width comes from the checkpoint, not from the tool:")
    with tempfile.TemporaryDirectory() as tmpname:
        tmp = Path(tmpname)
        cache, source, backup = build(tmp)
        shard = cache / "adaln_cache_s1.safetensors"
        before_header, before_base, before_raw = st_read(shard)
        result = run_tool(cache, source, backup)
        check(result.returncode == 0,
              f"the tool exits 0 (stderr: {result.stderr.strip()[:100]})")
        after_header, after_base, after_raw = st_read(shard)
        entry = after_header.get(META_KEY)
        if check(entry is not None, f"{META_KEY} is present afterwards"):
            check(entry["dtype"] == "U32" and entry["shape"] == [1],
                  f"{META_KEY} is U32 [1], which is what the C loader requires")
            begin, end = entry["data_offsets"]
            width = struct.unpack(
                "<I", after_raw[after_base + begin:after_base + end])[0]
            check(width == DECLARED_WIDTH,
                  f"{META_KEY} reads {width} == the width --source-model declares "
                  f"(a hardcoded 2688 or TIME_DIM would read differently here)")
            check(begin == len(before_raw) - before_base,
                  f"the new tensor starts at the old payload's end ({begin})")

        print("no payload byte moves:")
        moved = [key for key in before_header
                 if before_header[key] != after_header.get(key)]
        check(not moved,
              f"all {len(before_header)} pre-existing tensors keep dtype, shape "
              f"and offsets")
        payload = len(before_raw) - before_base
        check(hashlib.sha256(after_raw[after_base:after_base + payload]).hexdigest()
              == hashlib.sha256(before_raw[before_base:]).hexdigest(),
              "the payload itself hashes identical, only its start moved")
        check(len(after_raw) == len(before_raw) + (after_base - before_base) + 4,
              f"file grew by exactly the header delta "
              f"{after_base - before_base} plus 4 data bytes")
        check((backup / shard.name).is_file(),
              "the original went to --backup-dir before the rename")

        print("re-running is a no-op:")
        digest_before = st_read(shard)[2]
        second = run_tool(cache, source, backup)
        check(second.returncode == 0,
              f"the second run exits 0 (stderr: {second.stderr.strip()[:90]})")
        check("already carries" in second.stdout,
              "and says the key is already there")
        check(st_read(shard)[2] == digest_before,
              "and leaves the bytes alone")

    print("it refuses instead of guessing or half-applying:")
    for label, kwargs, needle in (
            ("no declared width", dict(width=None), "adaln_proj"),
            ("name without a step count",
             dict(names=("adaln_cache_s1.safetensors",
                         "adaln_cache_sX.safetensors")), "no step count"),
            ("header carries __metadata__",
             dict(metadata_names=("adaln_cache_s1.safetensors",)),
             "__metadata__"),
            ("a *later* shard carries __metadata__",
             dict(names=("adaln_cache_s1.safetensors",
                         "adaln_cache_s8.safetensors"),
                  metadata_names=("adaln_cache_s8.safetensors",)),
             "__metadata__")):
        with tempfile.TemporaryDirectory() as tmpname:
            tmp = Path(tmpname)
            cache, source, backup = build(tmp, **kwargs)
            shard = cache / "adaln_cache_s1.safetensors"
            before = shard.read_bytes()
            result = run_tool(cache, source, backup)
            check(result.returncode != 0 and needle in result.stderr,
                  f"{label}: refused with {needle!r} named "
                  f"({result.stderr.strip()[:90]})")
            check(shard.read_bytes() == before,
                  f"{label}: the shard that would have been rewritten first is "
                  f"untouched")

    print("PASS" if not failures else f"{len(failures)} check(s) failed")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
