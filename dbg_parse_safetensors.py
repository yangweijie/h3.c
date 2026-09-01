#!/usr/bin/env python3
"""Parse a safetensors *header* (tensor names / dtypes / shapes) without
reading the giant weight blob.

- Local file: opens and reads only the 8-byte length prefix + the JSON header
  (a few KB to a few MB), never the tensor data, so it is instant even on a
  30 GB file.
- Remote (--url): fetches just the first N MiB via HTTP Range (works only if
  the host honors Range requests) and parses the header from that prefix.

The point is to answer, *before* downloading or writing any C code, the
questions that decide whether h3.c can consume a quantized checkpoint:

  * which tensors are weights vs scales/zeros,
  * is there a Convrot / rotation matrix, or is rotation baked into weights,
  * int8 (I8) vs int4 (U8-packed) layout,
  * how many DiT blocks (to match H3_DIT_BLOCKS).

Usage:
  python3 dbg_parse_safetensors.py /path/to/model.safetensors
  python3 dbg_parse_safetensors.py --url https://host/path/model.safetensors
"""
import sys, json, struct, math, urllib.request

# safetensors dtypes -> bytes per element
DTYPE_SIZE = {
    "F64": 8, "F32": 4, "F16": 2, "BF16": 2, "I64": 8, "I32": 4,
    "I16": 2, "U16": 2, "I8": 1, "U8": 1, "BOOL": 1,
    "F8_E5M2": 1, "F8_E4M3": 1,
}


def read_prefix(path=None, url=None, range_mb=32):
    if url:
        req = urllib.request.Request(
            url, headers={"Range": "bytes=0-%d" % (range_mb * 1024 * 1024)})
        with urllib.request.urlopen(req) as r:
            return r.read()
    with open(path, "rb") as f:
        return f.read(8 + 64 * 1024 * 1024)  # header is tiny vs weights


def parse(buf):
    n = struct.unpack("<Q", buf[:8])[0]
    return json.loads(buf[8:8 + n])


def main():
    args = sys.argv[1:]
    url = path = None
    if "--url" in args:
        url = args[args.index("--url") + 1]
    elif args:
        path = args[0]
    if not path and not url:
        print("usage: dbg_parse_safetensors.py [--url URL] PATH")
        sys.exit(2)

    header = parse(read_prefix(path, url))
    meta = header.pop("__metadata__", {})
    tensors = header

    totals = {}
    for t in tensors.values():
        dt, shape = t["dtype"], t.get("shape", [])
        sz = DTYPE_SIZE.get(dt, 1) * (math.prod(shape) if shape else 1)
        totals[dt] = totals.get(dt, 0) + sz

    def hits(sub):
        return [k for k in tensors if sub.lower() in k.lower()]

    print("source: %s" % (path or url))
    print("tensors: %d" % len(tensors))
    if meta:
        print("metadata: %s" % json.dumps(meta)[:500])
    print("--- dtype totals ---")
    for dt, b in sorted(totals.items(), key=lambda x: -x[1]):
        print("  %-10s %14d bytes  (%.2f GiB)" % (dt, b, b / 1024 ** 3))
    print("--- suspicious / structural keys ---")
    for pat in ("scale", "zero", "rot", "convrot", "norm", "adaln",
                ".weight", ".bias", "blocks", "layers"):
        h = hits(pat)
        if h:
            print("  [%s] %d hits, e.g. %s" % (pat, len(h), h[0]))
    blocks = sorted({k for k in tensors if "blocks." in k or "layers." in k})
    if blocks:
        print("--- first/last block-like keys ---")
        print("  first: %s" % blocks[0])
        print("  last : %s" % blocks[-1])
    print("--- sample tensor names (first 25) ---")
    for k in list(tensors)[:25]:
        t = tensors[k]
        print("  %s  %s  %s" % (k, t["dtype"], t["shape"]))


if __name__ == "__main__":
    main()
