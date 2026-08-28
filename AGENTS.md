# CODEBUDDY.md This file provides guidance to CodeBuddy when working with code in this repository.

This is `h3.c`: a native C/Objective-C (macOS / Apple Silicon only) inference engine for MiniMax-H3, a unified text-to-video-audio (FL2VA) and reference-to-video-audio (Ref2VA) diffusion model. It reimplements the H3 pipeline to run locally on Metal, with no Python/torch dependency. The CLI binary is `h3`; `libh3.a` is the embeddable library.

## Build, test, and run

- `make` (or `make all`): build `h3` CLI and `libh3.a`. Requires Xcode command-line tools (clang) and system `ffmpeg`/`ffprobe` at runtime. The `.m` files are compiled with `-fobjc-arc` and link Metal / MetalPerformanceShaders / Accelerate frameworks.
- **Use Apple clang, not Homebrew LLVM**: build with `make CC="xcrun clang"`. A Homebrew `clang` on `PATH` (e.g. LLVM 16) fails on the macOS 26 SDK's Accelerate/vecLib headers with `unrecognized platform name visionOS`, while Apple clang 21 handles it. Older SDKs (15.4) avoid that error but lack `MTLGPUFamilyMetal4` needed by `h3_metal.m`, so the 26.x SDK + Apple clang is the only working combination.
- All library sources must be listed in `LIB_C` (or `LIB_M`) in the `Makefile`. A source that is only `#include`d but not added to `LIB_C` will link-fail with `Undefined symbols` (this is how `h3_memory_plan.c` was missed).
- `make test`: build and run the full suite. Most binaries auto-skip unless their weights/fixtures exist (see `misc/fixtures/` and the `MiniMax-H3/` model directory). No weights installed => only `h3_tests` and `h3_audio_gpu_tests` actually run.
- `make parity`: Metal vs BF16 numeric parity on the MLX toy-block fixtures (needs `misc/fixtures/h3_dit*.safetensors`). `make real-parity` does the same against real released weights.
- `make clean`: remove all build artifacts, objects, and `.a`.
- Run a single test binary directly, e.g. `./h3_tests`; some take a fixture path argument (e.g. `./h3_metal_tests misc/fixtures/h3_dit.safetensors`, `./h3_tokenizer_tests MiniMax-H3/tokenizer/tokenizer.json`).
- Run the engine: `./h3 --model MiniMax-H3 --prompt "..." --output out.mp4` (plus `--image/--video/--audio/--pr` for Ref2VA, `--mode fl2va|ref2va`).

## Architecture (the big picture)

The engine is pure-C orchestration layered over an Objective-C Metal backend. Cross-cutting data types live in `h3_internal.h` (`h3_ctx`, `h3_host_cond`, `h3_layout`, `h3_params`). Public API surface is in `h3.h` / `h3.c`; everything else is reached through it.

Inference is a six-stage pipeline:

1. Load (`h3_load_dir`): validate model directory layout, parse `h3_model_info` (mode, dimensions, block counts), and open safetensors shards via `h3_safetensors` + `h3_weights` (lazy/streamed BF16, or int8 group/row quantization).
2. Prompt parse (`h3_parse_prompt`): convert the prompt into a `h3_layout` (segment structure, position IDs, condition rows) and a `h3_ref` list of image/video/audio references.
3. Condition build (`h3_build_conditions` -> `h3_multimodal_build`): tokenize text, encode images/video through `h3_vision_encoder`, encode audio through `h3_audio_vae`, assemble `<Picture n>` placeholders, then run the first 50 Qwen3-VL layers (`h3_text_encoder`) to produce BF16 embeddings stored in `h3_host_cond`.
4. Denoise (`h3_denoise` -> `h3_dit_denoise_euler`): 20 Euler steps over a 50-block DiT (`h3_dit`). AdaLN modulation and gate scores are precomputed per step by `h3_dit_schedule` (`h3_dit_schedule_precompute` / `h3_dit_schedule_gate_score`). Acceleration knobs: `core_reuse`, `denoise_reuse`, `token_reduction`, block pruning, int8, and SSD streaming.
5. Decode: video latent through `h3_video_vae` (tiled decode, optional per-step preview), audio latent through `h3_audio_vae` (BigVGAN).
   - `h3_video_vae` has two decode paths chosen by `vae->streaming` (set from `h3_params.video_vae_streaming`): `run_resident_tile` keeps all 36 decoder blocks resident (~9 GiB, fast) while `run_stream_tile` loads/runs/frees one block at a time (~0.25 GiB). Both decode entry points (`decoder_decode_chunk` for the resident decoder and `decode_chunked` for the one-shot path) must branch on this flag — keep them in sync. The block count is exported as `H3_VIDEO_VAE_LAYERS` in `h3_video_vae.h` and used by the memory planner; never hard-code it.
6. Mux (`h3_mux` -> `h3_ffmpeg`): pipe RGB frames + F32 PCM to `ffmpeg` for final MP4+AAC.

Two parallel weight streams exist: text/visual encoders (Qwen3-VL, also used by Ref2VA) and the DiT+VAE stream. `h3_host.c` owns tensor allocation and caches condition tensors between steps. `h3_gpu.h` is the C device abstraction; the actual Metal kernels are inline `.metal` source strings in `h3_metal.m`, with MPS graph paths for stable ops; `h3_gpu.m` wraps device/buffer/command-queue dispatch. Only `h3_metal.m`, `h3_gpu.m`, and `h3_tokenizer.m` are Objective-C; everything else is C.

`h3_memory_plan.c` is the automatic memory-tier planner (ported in spirit from ds4's SSD cache planner), enabled by `h3_params.memory_plan_auto` (default on). It takes the device's `recommended_working_set`, the fully-resident weight total, and a *streaming-aware* resident estimate (`streamed_resident_bytes`: DiT keeps 2 blocks, VAE decoder 1 block, encoders freed per call), then decides `ssd_streaming` / `use_int8_row_fc2` / `video_vae_streaming` / `encoder_streaming` / `dit_layers` / cache budget. Streaming and int8 are orthogonal (unlike an earlier version that forced int8 off under streaming). Inspect the chosen plan at runtime with the `!memory-plan` CLI command.

Key shared constants (`HIDDEN=5376`, `HEADS=56`, `HEAD_DIM=96`, `MLP=21504`, `H3_DIT_BLOCKS=50`, `DIT_IN=24`, `DIT_IN_AUDIO=32`) are duplicated between `h3_dit.h` and `h3_dit_schedule.h` by design — keep them in sync.

## Conventions

- Strict C11 with `-Wall -Wextra -Wpedantic -Wshadow -Wconversion`; respect these in edits. `linenoise.c` is vendored and exempted from `-Wconversion`.
- Tests are standalone `main()` programs in `tests/` named by purpose (`test_*`, `test_real_*` require weights, `bench_*` for benchmarks). Add new coverage there rather than into the library.
- Do not introduce Python or external ML runtimes; the project's value is being a self-contained native engine.
