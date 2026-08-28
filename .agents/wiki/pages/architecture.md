# 系统架构 (Architecture)

引擎的数据流可分为六个阶段，从命令行/API 入口一直延伸到媒体输出。

## 1. 入口与上下文

`h3_load_dir()`（`h3.c`）读取模型目录元数据并初始化 Metal 设备，但不映射权重。所有状态集中在不透明结构 `h3_ctx`（定义于 `h3_internal.h`），包含模型目录、设备信息、模型信息、条件缓存（conditioning）、以及当前加载的 DiT（`struct h3_dit *`）和视频 VAE 解码器（`struct h3_video_vae_decoder *`）。`h3_cache_*` 一组函数管理跨请求复用的条件嵌入、DiT 与解码器缓存。

## 2. 提示与多模态编码

`h3_generate()` 是高层次生成入口。它先通过 `h3_tokenizer`（Objective-C 实现）做 UTF-8 分词；文本走 `h3_text_encode_bf16()`（前 50 层 Qwen3-VL 语言层，输出 BF16 嵌入）。图像/视频参考先经 `h3_vision_encoder` 的 Qwen 视觉编码器，再与文本在 `h3_multimodal` 中按 `<Picture n>` 呈现格式拼接，调用 `h3_text_encode_multimodal_bf16()`。`h3_host.c` 的 `h3_layout_build()` 计算序列分段、位置、以及图像/音频条件行数，作为后续张量排布的依据。

## 3. 扩散 Transformer (DiT)

`h3_dit` 是 50 个残差块（block）的堆叠（`h3_dit_schedule.h` 中 `H3_DIT_BLOCKS=50`，`HIDDEN=5376`，`HEADS=56`）。权重矩阵（QKV、输出投影、FC1/FC2）可在 BF16 与 int8（分组/逐行量化）之间切换。AdaLN 调制值由 `h3_dit_schedule_precompute()` 在每个去噪步预先投影，并按门控分数（`h3_dit_schedule_gate_score()`）进行块剪枝以加速。去噪主循环 `h3_dit_denoise()` / `h3_dit_denoise_euler()` 按 `h3_sigma_schedule`（由 `h3_schedule_build()` 构造）迭代，逐步从噪声中还原视频与音频潜变量（latent）。

## 4. VAE 解码

视频潜变量（通道数 24、patch 96）经 `h3_video_vae` 的平铺（tiled）解码器还原为 RGB 帧；引擎支持常驻解码器 `h3_video_vae_decoder_load()`，可在每次 Euler 步只解码一帧用于实时预览（`h3_video_vae_decoder_preview()`），最终再完整解码。音频潜变量（`[32,2,T]`）由 `h3_audio_vae` 的 BigVGAN 解码器还原为 32 kHz 立体声 PCM。

## 5. 媒体封装

`h3_ffmpeg` 负责图像/视频/音频的读取（FFprobe 探测尺寸、FFmpeg 解码为 channel-major F32）与写出（RGB24 视频 + F32 PCM 经两条并发管道直接封装为容器，不落中间文件）。

## 6. 设备与权重层

`h3_gpu`（C 声明在 `h3_gpu.h`，实现在 `h3_gpu.m` / `h3_metal.m`）封装 Metal 设备、共享缓冲区（`h3_gpu_tensor`）、以及 F32/BF16/int8 内核与 MPSGraph 调度。`h3_safetensors` 解析 safetensors 头部，`h3_weights` 以"打开头部—按需读 BF16 到共享缓冲区"的模式流式加载权重，避免大规模主机内存分配。

各模块详细文档见左侧目录树。
