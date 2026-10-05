#!/usr/bin/env python3
"""Add the optional `adaln_cache_meta_s{steps}` width key to cache shards that
were installed before that key existed, without touching a single payload byte.

The key lets the engine tell whether a LoRA adapter merges into AdaLN, so a
cached checkpoint can carry adapters that do not (h3_dit_schedule.c's
adaln_cache_accepts_adapters). The width is read from a checkpoint tree that
still declares it in `blocks.0.adaln_proj.linear.weight` -- never hardcoded.

Only the header grows: the new tensor's 4 bytes go at the end of the data
section, so every existing tensor keeps its relative offset and its bytes.
Originals are copied to --backup-dir before the first rename.

  python3 add_h3_adaln_cache_meta.py --cache-dir DIR --source-model DIR \
      --backup-dir DIR [--dry-run]
"""
import argparse
import json
import os
import pathlib
import shutil
import struct
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from export_h3_adaln_cache import (  # noqa: E402  (same helpers, same repo)
    cache_steps_of, is_cache_file, meta_key, st_header, weight_time_dim)


def rewrite(shard: pathlib.Path, key: str, width: int, backup: pathlib.Path,
            dry: bool) -> None:
    size = shard.stat().st_size
    header, base = st_header(shard)
    if key in header:
        existing = struct.unpack(
            "<I", _read(shard, base + header[key]["data_offsets"][0], 4))[0]
        print(f"{shard.name}: already carries {key} = {existing}; left alone")
        return
    payload = size - base
    if header.get("__metadata__") is not None:
        raise SystemExit(f"{shard.name}: __metadata__ key present; this tool "
                         "does not reorder headers around it")

    entry = {"dtype": "U32", "shape": [1],
             "data_offsets": [payload, payload + 4]}
    header[key] = entry
    text = json.dumps(header, separators=(",", ":"),
                      ensure_ascii=False).encode("utf-8")
    pad = (-len(text)) % 8
    text += b" " * pad

    print(f"{shard.name}: header {base - 8} -> {len(text)} B, "
          f"data {payload} B (+4), width {width}")
    if dry:
        return

    backup.mkdir(parents=True, exist_ok=True)
    original = backup / shard.name
    if not original.exists():
        shutil.copy2(shard, original)
    before = _digest(original, 8 + (base - 8), payload)

    tmp = shard.with_name(shard.name + ".tmp")
    with open(original, "rb") as src, open(tmp, "wb") as dst:
        dst.write(struct.pack("<Q", len(text)))
        dst.write(text)
        src.seek(base)
        shutil.copyfileobj(src, dst, length=1 << 22)
        dst.write(struct.pack("<I", width))
        dst.flush()
        os.fsync(dst.fileno())
    tmp.replace(shard)

    after = _digest(shard, 8 + len(text), payload)
    if before != after:
        raise SystemExit(f"{shard.name}: PAYLOAD CHANGED -- original kept at "
                         f"{original}, the shard was replaced; do not use it")
    check = st_header(shard)
    got = struct.unpack("<I", _read(shard, check[1] +
                    check[0][key]["data_offsets"][0], 4))[0]
    print(f"{shard.name}: payload digest {before[:12]}.. unchanged, "
          f"{key} reads back {got}")
    if got != width:
        raise SystemExit(f"{shard.name}: {key} reads back {got}, expected {width}")


def _read(path: pathlib.Path, offset: int, length: int) -> bytes:
    with open(path, "rb") as handle:
        handle.seek(offset)
        return handle.read(length)


def _digest(path: pathlib.Path, offset: int, length: int) -> str:
    import hashlib
    hash_ = hashlib.sha256()
    with open(path, "rb") as handle:
        handle.seek(offset)
        remaining = length
        while remaining:
            chunk = handle.read(min(1 << 22, remaining))
            if not chunk:
                raise SystemExit(f"{path.name}: short read at {offset}")
            hash_.update(chunk)
            remaining -= len(chunk)
    return hash_.hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cache-dir", type=pathlib.Path, required=True,
                    help="transformer directory holding adaln_cache_s*.safetensors")
    ap.add_argument("--source-model", type=pathlib.Path, required=True,
                    help="checkpoint tree that still declares adaln_proj weights")
    ap.add_argument("--backup-dir", type=pathlib.Path, required=True)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    width = weight_time_dim(args.source_model)
    if not width:
        raise SystemExit(f"{args.source_model} declares no "
                         "blocks.0.adaln_proj.linear.weight; refusing to guess "
                         "the AdaLN input width")
    print(f"AdaLN input width from {args.source_model}: {width}")

    shards = sorted(p for p in args.cache_dir.glob("*.safetensors")
                    if is_cache_file(p))
    if not shards:
        raise SystemExit(f"no adaln_cache_s*.safetensors under {args.cache_dir}")
    for shard in shards:
        steps = cache_steps_of(shard)
        if steps is None:
            raise SystemExit(f"{shard.name}: name carries no step count")
        rewrite(shard, meta_key(steps), width, args.backup_dir, args.dry_run)
    return 0


if __name__ == "__main__":
    sys.exit(main())
