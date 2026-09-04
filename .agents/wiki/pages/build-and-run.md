# 构建与运行

## 构建

```sh
make -j8          # 产出 h3 CLI 与 libh3.a
make clean        # 清除所有产物
```

Makefile 默认 `CC := xcrun clang`。**必须使用 Apple clang，不能是 Homebrew LLVM**：Homebrew clang（如 LLVM 16）在 macOS 26 SDK 的 Accelerate/vecLib 头文件上会报 `unrecognized platform name visionOS`，而 Apple clang 21 可以正常处理。同时，旧 SDK（15.4）虽然能绕过该错误，但缺少 `h3_metal.m` 需要的 `MTLGPUFamilyMetal4`，因此 **26.x SDK + Apple clang 是唯一可用组合**。

编译开关为严格 C11：

```
-std=c11 -O3 -Wall -Wextra -Wpedantic -Wshadow -Wconversion -Wno-sign-conversion
```

`.m` 文件额外加 `-fobjc-arc`，链接 Foundation / Metal / MetalPerformanceShaders / MetalPerformanceShadersGraph / Accelerate 五个 framework 加 `-licucore -lm`。`linenoise.c` 是 vendored 的终端编辑器，单独豁免 `-Wconversion`。

Sources: [Makefile](Makefile#L1-L9), [Makefile](Makefile#L219-L221)

## 库源文件必须登记

`LIB_C` / `LIB_M` 列出所有参与 `libh3.a` 的源文件。**只被 `#include` 而未加进 `LIB_C` 的源文件会在链接期报 `Undefined symbols`**（`h3_memory_plan.c` 就是这样被漏掉的）。新增 `.c` 文件时务必同步登记。

```make
LIB_C := h3.c h3_host.c h3_safetensors.c h3_weights.c h3_text_encoder.c \
	h3_dit_schedule.c h3_dit.c h3_lora.c h3_memory_plan.c
LIB_C += h3_video_vae.c h3_video_encoder.c h3_audio_vae.c h3_ffmpeg.c \
	h3_terminal.c h3_vision_encoder.c h3_multimodal.c
LIB_M := h3_metal.m h3_gpu.m h3_tokenizer.m
```

Sources: [Makefile](Makefile#L11-L16)

## 运行时依赖

- **ffmpeg / ffprobe 必须在 `PATH` 上**，用于参考媒体解码与 MP4 输出。可用 `H3_FFMPEG` / `H3_FFPROBE` 指定可执行文件。
- 无需 Xcode 的完整 Metal 工具链：Metal 源码在**运行时编译**，走的是 Iris 的做法。
- 可选的 `realesrgan-ncnn-vulkan` 用于 `--sr` 超分后处理。

Sources: [h3_ffmpeg.c](h3_ffmpeg.c#L16-L26), [README.md](README.md#L469-L472)

## 三种运行形态

```sh
# 1) 只检查模型与设备，不映射权重
./h3 --info -d ./MiniMax-H3

# 2) 一次性生成
./h3 --profile -d ./MiniMax-H3 \
     -p "A red fox walks through fresh snow in a pine forest." \
     --width 512 --height 512 --frames 22 --steps 20 \
     --layers 45 --reuse 2 --show -o outputs/fox-fast.mp4

# 3) 交互式会话（不给 -p 即进入）
./h3 -d ./MiniMax-H3 --width 512 --height 512 --steps 6
```

会话模式下，精确的 BF16 条件、已准备好的 DiT 与视频解码器都保留在内存里，换种子重跑同一 prompt 无需重新加载与编码。完整命令列表见 [命令行与交互会话](cli-and-session)。

Sources: [README.md](README.md#L12-L64), [main.c](main.c#L17-L73)

## 测试

```sh
make test          # 全量：自动跳过缺少权重/fixture 的用例
make parity        # Metal vs MLX BF16 数值对齐（toy 块 fixture）
make real-parity   # 同上，但针对真实发布权重
```

没有安装任何权重时，实际只跑 `h3_tests` 和 `h3_audio_gpu_tests`。fixture 位于 `misc/fixtures/`，模型目录为 `./MiniMax-H3`。详见 [测试与数值对齐](test-suite)。

也可以直接跑单个测试二进制，部分需要参数：

```sh
./h3_tests
./h3_metal_tests misc/fixtures/h3_dit.safetensors
./h3_tokenizer_tests MiniMax-H3/tokenizer/tokenizer.json
```

Sources: [Makefile](Makefile#L115-L133), [Makefile](Makefile#L198-L205)

## 常见构建陷阱

| 症状 | 原因 | 处理 |
|---|---|---|
| `unrecognized platform name visionOS` | 用了 Homebrew clang | `make CC="xcrun clang"` |
| `MTLGPUFamilyMetal4` 未声明 | SDK 太旧 | 升级到 macOS 26.x SDK |
| `Undefined symbols for architecture arm64` | 新源文件没写进 `LIB_C` | 补登记后 `make clean && make` |
| 运行时 "FFmpeg is not installed" | PATH 上没有 ffmpeg | 用 `H3_FFMPEG` 显式指定 |

Sources: [AGENTS.md](AGENTS.md#L14-L26), [Makefile](Makefile#L1-L4)
