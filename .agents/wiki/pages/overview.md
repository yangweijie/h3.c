# 项目总览 (Overview)

`h3-metal` 是一个面向 Apple Silicon 的原生视频/音频生成推理引擎，用于运行 MiniMax-H3（H3）系列扩散模型。它以纯 C（核心）与 Objective-C（Metal 后端）实现，无 Python 依赖，通过 Metal Performance Shaders（MPS）在 GPU 上执行所有权重计算，并通过 FFmpeg 处理媒体读写。

仓库根目录为 C 源码所在位置（`h3.c` 文件夹），主要源文件包括：

- 公共 API 与编排：`h3.c`、`h3.h`、`h3_internal.h`、`h3_cli.c`、`main.c`、`h3_terminal.c`
- 设备/权重层：`h3_gpu.h`（C 声明）、`h3_gpu.m` / `h3_metal.h` / `h3_metal.m`（Metal 实现）、`h3_safetensors.c`、`h3_weights.c`
- 文本/多模态：`h3_tokenizer.m`、`h3_text_encoder.c`、`h3_vision_encoder.c`、`h3_multimodal.c`
- 扩散主干：`h3_dit.c`、`h3_dit_schedule.c`
- VAE 与媒体：`h3_video_vae.c`、`h3_video_encoder.c`、`h3_audio_vae.c`、`h3_ffmpeg.c`
- 终端交互：`linenoise.c`（第三方行编辑库）

引擎支持两种 Transformer 装配：`FL2VA`（文本到视频/音频）与 `Ref2VA`（带图像/视频/音频参考的参考到视频）。模型权重以 safetensors 分片形式提供，引擎在加载时仅映射元数据、按需流式读取张量到共享 Metal 缓冲区。

版本宏位于 `h3.h`：`H3_VERSION "0.1.0-dev"`；默认画布 864×480、56 帧、20 步去噪、50 个 DiT 层（`H3_DEFAULT_*`、`H3_MIN_DIT_LAYERS=35`）。

详细架构见 [系统架构](architecture.md)，构建与运行见 [构建与运行](build-run.md)。
