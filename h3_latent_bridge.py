#!/usr/bin/env python3
"""Move latents between h3c's on-disk bundle and ComfyUI's .latent format.

h3c's `--latent-out` writes a self-describing bundle (magic "H3LT"): a seven
word header, the video latent laid out as [C, T, H, W], then an optional audio
block. ComfyUI's LoadLatent/SaveLatent use safetensors holding `latent_tensor`
as [B, C, T, H, W] alongside the `latent_format_version_0` marker -- without
that marker ComfyUI rescales by 1/0.18215, which this model does not want.

Both ends store the same *normalized* latent. ComfyUI's video VAE applies
`z * latents_std + latents_mean` inside decode(); h3c's video VAE does the same
in prepare_input(), with byte-identical per-channel constants. So this bridge
only rearranges axes: it must never rescale values.

The upscaler's own `(s - mean) / std` is internal to that node and pairs with
its `out * std + mean` exit, so it is transparent here.

Examples:
  # h3c bundle -> a file LoadLatent can pick up
  h3_latent_bridge.py to-comfy /tmp/stage1.h3lt

  # what SaveLatent produced -> a bundle h3c can --latent-in
  h3_latent_bridge.py to-bundle out/latents/ComfyUI_00001_.latent /tmp/refine.h3lt
"""

import argparse
import os
import struct
import sys

MAGIC = 0x48334C54  # "H3LT"
VERSION_F32 = 2
VERSION_F16 = 3

DEFAULT_COMFYUI = "/Volumes/data/Documents/ComfyUI"


def _fail(message):
    sys.stderr.write(f"h3-latent-bridge: error: {message}\n")
    raise SystemExit(1)


def _read_exact(handle, count, what):
    data = handle.read(count)
    if len(data) != count:
        _fail(f"truncated {what}: wanted {count} bytes, got {len(data)}")
    return data


def _unpack(values, count, use_f16, what):
    """Decode `count` floats from raw little-endian bytes."""
    import numpy as np

    if use_f16:
        if len(values) != count * 2:
            _fail(f"truncated {what}")
        return np.frombuffer(values, dtype="<f2").astype(np.float32)
    if len(values) != count * 4:
        _fail(f"truncated {what}")
    return np.frombuffer(values, dtype="<f4").astype(np.float32)


def read_bundle(path):
    """Parse an H3LT bundle. Returns a dict; `video` is [C, T, H, W] float32."""
    if not os.path.isfile(path):
        _fail(f"no such bundle: {path}")
    with open(path, "rb") as handle:
        header = struct.unpack("<7I", _read_exact(handle, 28, "header"))
        magic, version, dtype, t, h, w, c = header
        if magic != MAGIC:
            _fail(f"{path} is not an H3LT bundle (magic 0x{magic:08x})")
        if version not in (VERSION_F32, VERSION_F16):
            _fail(f"{path} has unsupported version {version} (want 2 or 3)")
        if version == VERSION_F32:
            dtype = 0
        if dtype not in (0, 1):
            _fail(f"{path} declares unexpected dtype flag {dtype}")
        if min(t, h, w, c) < 1:
            _fail(f"{path} has degenerate shape {c}x{t}x{h}x{w}")
        use_f16 = dtype == 1
        count = c * t * h * w
        video = _unpack(
            _read_exact(handle, count * (2 if use_f16 else 4), "video latent"),
            count, use_f16, "video latent",
        )

        audio = None
        tail = handle.read(4)
        if len(tail) == 4:
            (present,) = struct.unpack("<i", tail)
            if present:
                ac, aplanes, at = struct.unpack(
                    "<3i", _read_exact(handle, 12, "audio header"))
                acount = ac * aplanes * at
                audio = {
                    "channels": ac,
                    "planes": aplanes,
                    "t": at,
                    "values": _unpack(
                        _read_exact(handle,
                                    acount * (2 if use_f16 else 4),
                                    "audio latent"),
                        acount, use_f16, "audio latent",
                    ),
                }

    return {
        "version": version,
        "use_f16": use_f16,
        "t": t, "h": h, "w": w, "c": c,
        "video": video.reshape(c, t, h, w),
        "audio": audio,
    }


def write_bundle(path, bundle, use_f16):
    """Serialise an H3LT bundle. `bundle["video"]` must be [C, T, H, W]."""
    import numpy as np

    c, t, h, w = bundle["video"].shape
    video = bundle["video"].astype(np.float32, copy=False)
    # h3c's own writer always stamps version 3 and carries the width in the
    # dtype flag, so float32 bundles are version 3 with dtype 0 rather than
    # the legacy version 2. Match that to keep round trips byte-identical.
    header = struct.pack("<7I", MAGIC, VERSION_F16,
                         1 if use_f16 else 0, t, h, w, c)
    payload = video.astype("<f2" if use_f16 else "<f4").tobytes(order="C")

    with open(path, "wb") as handle:
        handle.write(header)
        handle.write(payload)
        audio = bundle.get("audio")
        if audio is None:
            handle.write(struct.pack("<i", 0))
        else:
            handle.write(struct.pack("<i", 1))
            handle.write(struct.pack("<3i", audio["channels"],
                                     audio["planes"], audio["t"]))
            handle.write(audio["values"].astype(
                "<f2" if use_f16 else "<f4").tobytes(order="C"))


def describe(label, video):
    import numpy as np

    flat = np.asarray(video, dtype=np.float32)
    print(f"  {label:14s} shape={tuple(flat.shape)} "
          f"min={flat.min():+.4f} max={flat.max():+.4f} "
          f"mean={flat.mean():+.4f} std={flat.std():.4f}")


def to_comfy(arguments):
    import numpy as np
    from safetensors.torch import save_file
    import torch

    bundle = read_bundle(arguments.bundle)
    c, t, h, w = bundle["video"].shape
    describe("bundle video", bundle["video"])
    if bundle["audio"] is not None:
        describe("bundle audio", bundle["audio"]["values"])
        print("  note: ComfyUI's LATENT carries no audio; use --audio-from "
              "when converting back, or it will be dropped")

    tensor = torch.from_numpy(
        np.ascontiguousarray(bundle["video"][None])).float()
    # The marker keeps LoadLatent from rescaling by 1/0.18215.
    save_file({"latent_tensor": tensor,
               "latent_format_version_0": torch.tensor([])},
              arguments.output)
    print(f"  wrote {arguments.output}  [B,C,T,H,W]={tuple(tensor.shape)}")
    print(f"  load it in ComfyUI as input/{os.path.basename(arguments.output)}")


def to_bundle(arguments):
    import numpy as np
    from safetensors.torch import load_file

    if not os.path.isfile(arguments.latent):
        _fail(f"no such latent: {arguments.latent}")
    tensors = load_file(arguments.latent, device="cpu")
    if "latent_tensor" not in tensors:
        _fail(f"{arguments.latent} has no latent_tensor key; keys are "
              f"{sorted(tensors)}")
    if "latent_format_version_0" not in tensors:
        print("  warning: no latent_format_version_0 marker; ComfyUI itself "
              "would have scaled this by 1/0.18215. Continuing unscaled.")

    tensor = tensors["latent_tensor"].float()
    if tensor.dim() == 4:
        tensor = tensor.unsqueeze(0)
    if tensor.dim() != 5:
        _fail(f"expected a 4D or 5D latent_tensor, got {tuple(tensor.shape)}")
    if tensor.shape[0] != 1:
        _fail(f"expected a single batch entry, got {tensor.shape[0]}")
    video = tensor[0].contiguous().numpy()
    describe("latent video", video)

    bundle = {"video": video, "audio": None}
    if arguments.audio_from:
        source = read_bundle(arguments.audio_from)
        bundle["audio"] = source["audio"]
        if bundle["audio"] is None:
            print(f"  note: {arguments.audio_from} carries no audio block")
        else:
            describe("re-attached", bundle["audio"]["values"])

    write_bundle(arguments.output, bundle, not arguments.f32)
    size = os.path.getsize(arguments.output)
    print(f"  wrote {arguments.output}  {size / 1e6:.1f} MB "
          f"({'f16' if not arguments.f32 else 'f32'})")
    print(f"  feed it to: h3 --latent-in {arguments.output} "
          f"--refine-sigma {arguments.refine_sigma}")


def main():
    parser = argparse.ArgumentParser(
        description="Convert between h3c's H3LT latent bundle and ComfyUI's "
                    ".latent safetensors. Values are never rescaled; only the "
                    "axis order changes ([C,T,H,W] <-> [B,C,T,H,W]).")
    subparsers = parser.add_subparsers(dest="command", required=True)

    export = subparsers.add_parser(
        "to-comfy", help="H3LT bundle -> ComfyUI .latent for LoadLatent")
    export.add_argument("bundle", help="h3c --latent-out bundle")
    export.add_argument("-o", "--output",
                        help="destination .latent (default: ComfyUI input/)")
    export.add_argument("--comfyui", default=DEFAULT_COMFYUI,
                        help=f"ComfyUI root (default: {DEFAULT_COMFYUI})")
    export.set_defaults(handler=to_comfy)

    absorb = subparsers.add_parser(
        "to-bundle", help="ComfyUI .latent -> H3LT bundle for --latent-in")
    absorb.add_argument("latent", help=".latent written by SaveLatent")
    absorb.add_argument("-o", "--output", required=True,
                        help="destination H3LT bundle")
    absorb.add_argument("--audio-from", help="original bundle to re-attach "
                                             "the audio block from")
    absorb.add_argument("--f32", action="store_true",
                        help="store float32 instead of float16")
    absorb.add_argument("--refine-sigma", type=float, default=0.6,
                        help="only used in the printed hint (default: 0.6)")
    absorb.set_defaults(handler=to_bundle)

    arguments = parser.parse_args()
    if arguments.command == "to-comfy" and not arguments.output:
        stem = os.path.splitext(os.path.basename(arguments.bundle))[0]
        arguments.output = os.path.join(
            arguments.comfyui, "input", f"{stem}.latent")
        os.makedirs(os.path.dirname(arguments.output), exist_ok=True)
    arguments.handler(arguments)


if __name__ == "__main__":
    main()
