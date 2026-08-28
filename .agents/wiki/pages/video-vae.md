# 视频 VAE 解码/编码: h3_video_vae / h3_video_encoder

`h3_video_vae.h` / `h3_video_vae.c` 实现视频潜变量的平铺（tiled）解码与编码；`h3_video_encoder.c` 把输入参考视频编码为潜变量（用于 Ref2VA 的视频参考条件）。二者共用一套 VAE 权重（来自 [权重存储](weights.md)）。

## 视频 VAE 解码器

- `h3_video_vae_decoder_open(ctx, weights, params)` / `h3_video_vae_decoder_close()`：加载并释放解码器权重。支持常驻模式，便于逐步预览。
- `h3_video_vae_decoder_load(ctx, latents, n, params, decoder)`：把 [DiT 主干](dit.md) 输出的视频潜变量（`[24, F/2, H/16, W/16]`，patch=96）解码为 RGB 帧序列（channel-major F32，值范围 0..1）。
- `h3_video_vae_decoder_preview(decoder, latent_i)`：在常驻解码器上只解码第 `i` 个潜变量帧，用于 `--preview-denoise` 的逐 Euler 步预览，避免每步重解全部帧。
- `h3_video_vae_decode_tiled(...)`：平铺解码——把大帧切分为空间块分别解码再拼接，控制峰值内存；块边界用重叠-加窗（overlap-add）消除接缝。

## 视频编码器（参考路径）

- `h3_video_encoder_encode(path, params, &latent)`：把参考视频（经 [FFmpeg](ffmpeg.md) 读入）编码为潜变量，作为 Ref2VA 的视频条件输入，进入 [多模态嵌入编排](multimodal.md) 的序列装配。

## 与 DiT 的耦合

视频 VAE 的通道数（24）与 patch 尺寸（96）直接决定 DiT 输入/输出维度（`DIT_IN=24`），二者必须版本匹配；若权重与代码维度不一致，`h3_load_dir()` 阶段会通过 `h3_model_info` 校验报错。

## 实现要点

解码器大量使用 [Metal 着色器](metal-shaders.md) 中的 `vae_upsample_*`（转置卷积/上采样）与卷积算子，并在 `.m` 后端上以 BF16 运行。平铺策略使 864×480×56 帧的解码峰值内存显著低于整帧一次性解码。

相关：音频侧对称实现见 [音频 VAE](audio-vae.md)；解码结果经 [FFmpeg](ffmpeg.md) 封装输出。
