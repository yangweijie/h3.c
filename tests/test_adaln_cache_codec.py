#!/usr/bin/env python3
"""Pin the AdaLN modulation cache's on-disk contract to the C code that defines it.

`dump_adaln_cache()` in h3_dit_schedule.c writes a raw blob that
`export_h3_adaln_cache.py` repackages into `adaln_cache_s<steps>.safetensors`, and
`load_cached_adaln()` reads those back. The two sides each spell out the same five
numbers -- magic, header size, block count, block width, final width -- plus the
order of the header fields and the set of tensor key names. Nothing checked that,
so a one-line change on either side stayed invisible until a multi-hour render
came out wrong.

The anchor is external to the code under test on both sides:

* the C constants come from **compiling** `tests/adaln_cache_probe.c`, which only
  uses the macros `h3_dit_schedule.h` defines -- so it reports what the loader and
  the writer actually see, not a second transcription;
* the field **order** and the **size formula** are checked against a dump header
  built at a C-computed length, then decoded by the exporter's own `Dump` parser,
  and re-decoded by the probe.

Run directly: `python3 tests/test_adaln_cache_codec.py [--model <transformer dir>]`
"""

from __future__ import annotations

import argparse
import os
import struct
import subprocess
import sys
import tempfile
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
EXPORTER = ROOT / "fastvideo_qad" / "scripts" / "export_h3_adaln_cache.py"
PROBE_SOURCE = ROOT / "tests" / "adaln_cache_probe.c"

failures: list[str] = []


def check(condition: bool, message: str) -> bool:
    print(("  ok   " if condition else "  FAIL ") + message)
    if not condition:
        failures.append(message)
    return condition


def parse_dump(exporter, path: Path, label: str):
    """`Dump` refuses a malformed blob by raising, which is the right behaviour
    for the exporter but would abort this test instead of reporting it. A
    rejection is a detection; record it as a failed check."""
    try:
        return exporter.Dump(path)
    except SystemExit as error:
        check(False, f"{label}: the exporter rejected the blob ({error})")
    return None


def load_exporter():
    spec = spec_from_file_location("export_h3_adaln_cache", EXPORTER)
    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def build_probe(work: Path) -> Path:
    compiler = os.environ.get("CC", "cc").split()
    executable = work / "adaln_cache_probe"
    result = subprocess.run(
        compiler + ["-std=c11", "-Wall", "-Wextra", "-Wpedantic",
                    "-I", str(ROOT), str(PROBE_SOURCE), "-o", str(executable)],
        capture_output=True, text=True)
    if result.returncode:
        raise SystemExit("probe did not compile:\n" + result.stderr)
    return executable


def probe(exe: Path, *arguments: str) -> dict[str, int | str]:
    result = subprocess.run([str(exe), *arguments], capture_output=True,
                            text=True)
    if result.returncode:
        raise SystemExit(f"probe {' '.join(arguments)} failed: {result.stderr}")
    values: dict[str, int | str] = {}
    for line in result.stdout.splitlines():
        key, _, value = line.partition("=")
        values[key] = value if not value.isdigit() else int(value)
    return values


def sentinel_dump(exe: Path, work: Path, fields: list[int]) -> Path:
    """A dump whose header is laid out by the probe's own field indices, cut to
    the exact length the C size formula gives."""
    contract = probe(exe, "print")
    path = work / "sentinel.raw"
    packed = "<{}I".format(int(contract["header_fields"]))
    with path.open("wb") as handle:
        handle.write(contract["magic"].encode())
        handle.write(struct.pack(packed, *fields))
        handle.truncate(int(contract["dump_bytes_3"]))
    return path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path,
                        help="transformer dir whose installed cache files get "
                             "cross-checked against the C size formula")
    args = parser.parse_args()

    try:
        exporter = load_exporter()
    except ImportError as error:
        print(f"skip: the exporter needs numpy ({error})")
        return 0

    with tempfile.TemporaryDirectory() as directory:
        work = Path(directory)
        exe = build_probe(work)
        contract = probe(exe, "print")

        print("constants:")
        check(exporter.MAGIC == contract["magic"].encode(),
              f"magic {exporter.MAGIC!r} == C {contract['magic']!r}")
        check(len(exporter.MAGIC) == contract["magic_bytes"],
              f"magic length == {contract['magic_bytes']}")
        check(exporter.HEADER_BYTES == contract["header_bytes"],
              f"header is {exporter.HEADER_BYTES} bytes, C says "
              f"{contract['header_bytes']}")
        check(exporter.HEADER_BYTES ==
              contract["magic_bytes"] + contract["header_fields"] * 4,
              "header = magic + header_fields uint32")
        check(exporter.BLOCKS == contract["blocks"],
              f"block count {exporter.BLOCKS} == C {contract['blocks']}")
        check(exporter.BLOCK_OUTPUT == contract["block_output"],
              f"block width {exporter.BLOCK_OUTPUT} == C "
              f"{contract['block_output']}")
        check(exporter.FINAL_OUTPUT == contract["final_output"],
              f"final width {exporter.FINAL_OUTPUT} == C "
              f"{contract['final_output']}")

        print("header field order (distinct sentinels, so a swap cannot cancel):")
        # steps and time_rows deliberately differ; blocks/widths must stay at the
        # values the exporter's parser insists on.
        fields = [1, 3, contract["blocks"], contract["block_output"],
                  contract["final_output"], 0]
        order = ["steps", "time_rows", "blocks", "block_output",
                 "final_output", "reserved"]
        positions = [contract[f"field_{name}"] for name in order]
        check(positions == list(range(len(order))),
              f"C maps the names to slots {positions}")
        path = sentinel_dump(exe, work, fields)
        parsed = probe(exe, "parse", str(path))
        check([parsed[name] for name in order] == fields,
              f"probe reads slot order {order} back as {fields}")
        dump = parse_dump(exporter, path, "sentinel header")
        if dump is None:
            return finish()
        exported = [dump.steps, dump.time_rows, dump.blocks,
                    dump.block_output, dump.final_output]
        check(exported == fields[:5],
              f"the exporter's Dump decodes the same header as {exported}")

        print("size formula:")
        for rows in (1, 3, 9, 17, 41):
            size = int(contract[f"dump_bytes_{rows}"])
            reference = (exporter.HEADER_BYTES + rows * 4 +
                         exporter.BLOCKS * rows * exporter.BLOCK_OUTPUT * 2 +
                         rows * exporter.FINAL_OUTPUT * 2)
            check(size == reference,
                  f"{rows} rows: C {size} == Python's arithmetic {reference}")
        too_short = work / "short.raw"
        with too_short.open("wb") as handle:
            handle.write(path.read_bytes()[:exporter.HEADER_BYTES])
            handle.truncate(int(contract["dump_bytes_3"]) - 8)
        try:
            exporter.Dump(too_short)
            check(False, "a dump 8 bytes under the C length must be refused")
        except SystemExit as error:
            check("truncated" in str(error),
                  f"8 bytes short is refused: {str(error).splitlines()[-1]}")

        print("tensor key names:")
        steps = 8
        c_keys = subprocess.run([str(exe), "keys", str(steps)],
                                capture_output=True, text=True,
                                check=True).stdout.split()
        sentinel = work / f"s{steps}.raw"
        packed = "<{}I".format(int(contract["header_fields"]))
        with sentinel.open("wb") as handle:
            handle.write(exporter.MAGIC +
                         struct.pack(packed, steps, 2 * steps + 1,
                                     contract["blocks"],
                                     contract["block_output"],
                                     contract["final_output"], 0))
            handle.truncate(int(contract[f"dump_bytes_{2 * steps + 1}"]))
        big = parse_dump(exporter, sentinel, f"s{steps} blob")
        if big is None:
            return finish()
        check(list(key for key, _, _, _ in big.entries()) == c_keys[:-1],
              f"the {len(c_keys) - 1} keys the exporter writes are exactly the "
              f"names the loader looks up for s{steps}")
        check(c_keys[-1] == exporter.meta_key(steps),
              f"the optional width key is spelled the same on both sides "
              f"({c_keys[-1]})")

        if args.model:
            print(f"installed caches under {args.model}:")
            rows_by_steps = {4: 9, 8: 17, 20: 41}
            for cache in sorted(args.model.glob("adaln_cache_s*.safetensors")):
                header, base = exporter.st_header(cache)
                key = f"adaln_cache_times_s{cache.stem[len('adaln_cache_s'):]}"
                entry = header.get(key)
                if entry is None:
                    check(False, f"{cache.name} has no schedule key to size by")
                    continue
                rows = entry["shape"][0]
                payload = int(contract[f"dump_bytes_{rows}"]) - \
                    int(contract["header_bytes"])
                check(cache.stat().st_size == base + payload,
                      f"{cache.name}: {rows} rows -> {cache.stat().st_size} "
                      f"bytes == header {base} + payload {payload}")

    print("PASS" if not failures else f"{len(failures)} check(s) failed")
    return 1 if failures else 0


def finish() -> int:
    """Report and stop: a blob the exporter itself rejects already names the
    drift, so later checks would only restate it."""
    print(f"aborted after {len(failures)} check(s) failed")
    return 1


if __name__ == "__main__":
    sys.exit(main())
