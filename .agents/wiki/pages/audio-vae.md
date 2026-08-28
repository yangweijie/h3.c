# 音频 VAE (BigVGAN): h3_audio_vae

`h3_audio_vae.c` / `h3_audio_vae.h` 实现音频潜变量的 BigVGAN 解码（以及参考音频的编码），把 [DiT 主干](dit.md) 输出的音频潜变量还原为 32 kHz 立体声 PCM。

## 解码器

- `h3_audio_vae_decoder_open(ctx, weights, params)` / `h3_audio_vae_decoder_close()`：加载 BigVGAN 反卷积权重（来自 [权重存储](weights.md)）。
- `h3_audio_vae_decode(latent, params, &pcm)`：输入音频潜变量 `[32, 2, T]`（32 通道、2 声道、T 帧），经多组反卷积（对应 [Metal 着色器](metal-shaders.md) 的 `bigvgan_conv_transpose_*`）上采样为 F32 PCM，采样率固定 32000。
- `h3_audio_vae_decode_preview(...)`：与 [视频 VAE](video-vae.md) 类似，支持逐块预览音频（在逐 Euler 步时仅合成最新一段）。

## 编码器（参考路径）

- `h3_audio_vae_encode(path, params, &latent)`：把参考音频（经 [FFmpeg](ffmpeg.md) 读入的 PCM）编码为 `[32,2,T]` 潜变量，作为 Ref2VA 的音频条件，进入 [多模态嵌入编排](multimodal.md)。

## 与 DiT 的耦合

音频潜变量的通道数（32）决定 DiT 音频分支的输入维度（`DIT_IN_AUDIO=32`）。与视频 VAE 一样，维度必须在 `h3_load_dir()` 阶段与权重校验一致。

## 实现要点

BigVGAN 的反卷积内核在 `h3_metal.m` 中以 Metal 着色器实现，运行于 BF16 共享缓冲；输出 PCM 直接交给 [FFmpeg](ffmpeg.md) 的音频管道封装，不落中间文件。

相关：视频侧对称实现见 [视频 VAE](video-vae.md)；媒体写出见 [FFmpeg 媒体读写](ffmpeg.md)。
