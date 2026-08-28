# 构建与运行 (Build & Run)

## 构建

仓库根通过 `Makefile` 驱动 Apple Clang 编译。关键构建变量：

- `SDK` / `ARCH`：面向 `arm64-apple-macosx`，使用 macOS SDK。
- `METAL_LIB`：由 `.metal` 着色器源（在 `h3_metal.m` 中以字符串常量内联，或独立 `.metal` 文件）编译出的 Metal 库，运行时通过 `shader_source_path` 传给 `h3_gpu_create()`。
- 链接框架：`Metal`、`MetalPerformanceShaders`（MPS）、`MetalFX`（若用）、`AVFoundation` / `CoreMedia`（经 FFmpeg 间接）、`Accelerate`（vImage 高质量缩放）。
- 外部库：`ffmpeg` / `ffprobe` 通过 `H3_FFMPEG` / `H3_FFPROBE` 环境变量或默认 PATH 定位（见 `h3_ffmpeg.c`）。

一个典型的构建会分别编译 C 源（`.c`）与 Objective-C 源（`.m`，含 `h3_tokenizer.m`、`h3_gpu.m`、`h3_metal.m`），最后链接为单个可执行文件（或库 + `main.c`）。`linenoise.c` 提供交互式行编辑。

## 运行

顶层入口为 `main.c`，它解析命令行参数并构造 `h3_params`（见 `h3.h` 中的 `H3_PARAMS_DEFAULT`），随后进入 `h3_cli_run()`（`h3_cli.c`）驱动的交互提示，或在一次性模式下直接调用 `h3_generate()`。

常用参数（均定义在 `h3_params`）：

- `--width` / `--height` / `--frames` / `--steps`：画布与去噪步数。
- `--seed`：随机种子（默认 42）；交互模式下随机。
- `--dit-layers`：保留的 DiT 残差块数（50 准确、45 验证加速、40 激进）。
- `--core-reuse`：每隔 N 步重算 Transformer core（1 准确、4/6 加速）。
- `--denoise-reuse`：每 N 步评估一次去噪器（1 近参考、2 验证快速、3 激进）。
- `--token-reduction`：相邻水平视频 token 配对以减计算。
- `--ssd-streaming`：仅保留 2 个 BF16 DiT 块在内存，重叠 I/O 与执行。
- `--render-width` / `--render-height`：可选更低内部渲染画布。
- `--use-slower-*`：回退到可移植的 BF16/MPS 实现（用于对齐/诊断）。
- `--preview-denoise`：每个 Euler 步解码并投递一帧预览。

输出通过 `h3_params.on_frame` 回调增量交付；`h3_result` 报告最终宽高、帧数、fps（24）、采样率（32000）与种子。

缓存（跨请求复用条件嵌入、DiT、VAE 解码器）默认关闭（`h3_cache_set_enabled()`），一次性调用者保持原始的逐阶段内存生命周期。
