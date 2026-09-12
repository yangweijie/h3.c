# CODEBUDDY.md This file provides guidance to CodeBuddy when working with code in this repository.

This is `h3.c`: a native C/Objective-C (macOS / Apple Silicon only) inference engine for MiniMax-H3, a unified text-to-video-audio (FL2VA) and reference-to-video-audio (Ref2VA) diffusion model. It reimplements the H3 pipeline to run locally on Metal, with no Python/torch dependency. The CLI binary is `h3`; `libh3.a` is the embeddable library.

## Build, test, and run

- `make` (or `make all`, `make -j8`): build `h3` CLI and `libh3.a`. Requires Xcode command-line tools (clang) and system `ffmpeg`/`ffprobe` at runtime. The `.m` files are compiled with `-fobjc-arc` and link Metal / MetalPerformanceShaders / Accelerate frameworks.
- **Run every binary from the repo root.** Metal kernels are not compiled into the binary: `h3_gpu.m` reads and runtime-compiles the standalone source file `h3_shaders.metal`, and its default path is the literal string `h3_shaders.metal` resolved against the process CWD (tests pass the path explicitly, the CLI does not). Any change to `h3_shaders.metal` needs no relink.
- **Use Apple clang, not Homebrew LLVM**: build with `make CC="xcrun clang"`. A Homebrew `clang` on `PATH` (e.g. LLVM 16) fails on the macOS 26 SDK's Accelerate/vecLib headers with `unrecognized platform name visionOS`, while Apple clang 21 handles it. Older SDKs (15.4) avoid that error but lack `MTLGPUFamilyMetal4` needed by `h3_metal.m`, so the 26.x SDK + Apple clang is the only working combination.
- All library sources must be listed in `LIB_C` (or `LIB_M`) in the `Makefile`. A source that is only `#include`d but not added to `LIB_C` will link-fail with `Undefined symbols` (this is how `h3_memory_plan.c` was missed).
- `make test`: build and run the full suite. Most binaries auto-skip unless their weights/fixtures exist (see `misc/fixtures/` and the `MiniMax-H3/` model directory). No weights installed => only `h3_tests` and `h3_audio_gpu_tests` actually run. `make test AUDIO_VAE_MODEL=<model root>` additionally runs the weights-only AudioVAE end-to-end test (`tests/test_real_audio_vae_e2e.c`), which needs no `misc/fixtures` oracle: it pins the decode shape, finiteness, determinism, dispatch structure, and the shortest legal latent instead.
- `make parity`: Metal vs BF16 numeric parity on the MLX toy-block fixtures (needs `misc/fixtures/h3_dit*.safetensors`). `make real-parity` does the same against real released weights.
- `make clean`: remove all build artifacts, objects, and `.a`.
- Build and run one suite: `make h3_lora_tests && ./h3_lora_tests` (every suite is its own make target, e.g. `h3_vdn_tests`, `h3_real_dit_test`; not all are in `make test`, notably `h3_lora_tests`).
- Run a single test binary directly, e.g. `./h3_tests`; some take a fixture path argument (e.g. `./h3_metal_tests misc/fixtures/h3_dit.safetensors`, `./h3_tokenizer_tests MiniMax-H3/tokenizer/tokenizer.json`).
- `make clipproj-golden`: ClipProj encoder fidelity check (in-engine B vs offline harness); needs the `QWEN4B`/`PROJ`/`CLIPPROJ_MODEL` model paths, overridable as make variables.
- `tests/gen_lora_data.py` writes the `tmp_lora_test/` fixtures that `h3_lora_tests` consumes — run it first or that suite has nothing to compare against.
- Run the engine: `./h3 -d ./MiniMax-H3 -p "..." -o out.mp4` (defaults 864x480, 56 frames, 20 steps). `./h3 -d ./MiniMax-H3 --info` inspects layout/device without mapping weights; without `-p` it starts the interactive session. Ref2VA is selected by the reference flags themselves (`--ref-image`, `--ref-video`, `--ref-silent-video`, `--ref-video-audio V A`, `--ref-audio`), and FL2VA first/last-frame conditioning by `--first-frame`/`--last-frame`. `./h3 --help` is the authoritative flag list.

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

Two parallel weight streams exist: text/visual encoders (Qwen3-VL, also used by Ref2VA) and the DiT+VAE stream. `h3_host.c` owns tensor allocation and caches condition tensors between steps. `h3_gpu.h` is the C device abstraction; `h3_gpu.m` wraps device/buffer/command-queue dispatch **and runtime-compiles the kernel library from `h3_shaders.metal`** (all H3 kernels live in that file; `h3_metal.m` is only `h3_metal_probe` and holds no kernel source). Stable wide ops also have MPSGraph paths selected next to the hand-written kernels. Only `h3_metal.m`, `h3_gpu.m`, and `h3_tokenizer.m` are Objective-C; everything else is C.

Two load-time weight transforms hook into the DiT before denoising, and both are incompatible with `--ssd-streaming` (and with int8 for LoRA):

- **LoRA merge** (`h3_lora.c`, `--lora PATH`): Turbo/distillation adapters (LightX2V, Lightning, VDN) are Diffusers-keyed BF16 `lora_A`/`lora_B` safetensors. `W' = W + (alpha/rank) * (B @ A)` is applied in unified memory right after a block loads, so inference still sees plain BF16; the base checkpoint files are never rewritten. A comma-separated list merges up to 4 adapters in order (VDN uses `default,turbo`); `h3_lora_matches` skips an adapter whose `lora_A` width does not fit the base (e.g. a diffusers-shaped turbo over a pruned ConvRot checkpoint). `h3_lora_merge_blocking` exists because the SSD prefetch thread may merge while the main thread has a command buffer open.
- **VDN linear branch** (`--linear-branch DIR`): adds a learned linear delta-rule branch next to chunk-window softmax attention (`tests/test_vdn_branch.c`, `tests/test_linear_branch.c`).

`h3_memory_plan.c` is the automatic memory-tier planner (ported in spirit from ds4's SSD cache planner), enabled by `h3_params.memory_plan_auto` (default on). It takes the device's `recommended_working_set`, the fully-resident weight total, and a *streaming-aware* resident estimate (`streamed_resident_bytes`: DiT keeps 2 blocks, VAE decoder 1 block, encoders freed per call), then decides `ssd_streaming` / `use_int8_row_fc2` / `video_vae_streaming` / `dit_layers`. Streaming and int8 are orthogonal (unlike an earlier version that forced int8 off under streaming). Inspect the chosen plan at runtime with the `!memory-plan` CLI command.

Key shared constants (`HIDDEN=5376`, `HEADS=56`, `HEAD_DIM=96`, `MLP=21504`, `H3_DIT_BLOCKS=50`, `DIT_IN=24`, `DIT_IN_AUDIO=32`) are duplicated between `h3_dit.h` and `h3_dit_schedule.h` by design — keep them in sync.

## Conventions

- Strict C11 with `-Wall -Wextra -Wpedantic -Wshadow -Wconversion`; respect these in edits. `linenoise.c` is vendored and exempted from `-Wconversion`.
- Tests are standalone `main()` programs in `tests/` named by purpose (`test_*`, `test_real_*` require weights, `bench_*` for benchmarks). Add new coverage there rather than into the library.
- Do not introduce Python or external ML runtimes; the project's value is being a self-contained native engine. The Python that exists (`dbg_*.py`, `clipproj_*.py`, `verify_lora_align.py`, `fastvideo_qad/`, `tests/gen_lora_data.py`) is offline analysis/fixture generation only and is never invoked by `make` or by the engine. Run it with `/Volumes/data/Application/anaconda3/bin/python` — the system `python3` has no `safetensors`/`torch`.
- README's "Implementation and performance notes" is the reference for the `H3_*` environment knobs (`H3_NAX`, `H3_FB_CACHE`, `H3_QWEN_PREFETCH`, `H3_DIT_COMMAND_BLOCKS`, `--use-slower-*` A/B switches, ...). Every fast path is supposed to have a disable/force knob so it can be bisected against the close-reference path; add one when adding a fast path.
