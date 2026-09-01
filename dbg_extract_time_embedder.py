#!/usr/bin/env python3
"""Extract the time_embedder tensors (dropped by the ConvRot pruning, which
quantizes only the per-block matmuls) from the BF16 FL2VA DiT shards into one
small safetensors file, so a hybrid model dir can combine the int8 ConvRot
checkpoint with the missing pieces (tensor names do not overlap, so both files
can live in the same FL2VA/transformer directory).

Usage:
  python3 dbg_extract_time_embedder.py <bf16_transformer_dir> <out.safetensors>
"""
import glob
import json
import struct
import sys

PREFIX = "time_embedder."


def main():
    if len(sys.argv) != 3:
        sys.exit("usage: dbg_extract_time_embedder.py <bf16_dir> <out.safetensors>")
    src_dir, out_path = sys.argv[1], sys.argv[2]
    found = {}
    for path in sorted(glob.glob(src_dir + "/*.safetensors")):
        with open(path, "rb") as f:
            n = struct.unpack("<Q", f.read(8))[0]
            header = json.loads(f.read(n))
            base = 8 + n
            for name, meta in header.items():
                if name == "__metadata__" or not name.startswith(PREFIX):
                    continue
                start, end = meta["data_offsets"]
                f.seek(base + start)
                found[name] = (meta["dtype"], meta["shape"], f.read(end - start))
    if not found:
        sys.exit("no %s* tensors found in %s" % (PREFIX, src_dir))
    header, blob, offset = {}, b"", 0
    for name in sorted(found):
        dtype, shape, data = found[name]
        header[name] = {"dtype": dtype, "shape": shape,
                        "data_offsets": [offset, offset + len(data)]}
        offset += len(data)
        blob += data
    meta = json.dumps(header).encode()
    with open(out_path, "wb") as f:
        f.write(struct.pack("<Q", len(meta)))
        f.write(meta)
        f.write(blob)
    print("wrote %s: %d tensors, %d data bytes" % (out_path, len(found), len(blob)))


if __name__ == "__main__":
    main()
