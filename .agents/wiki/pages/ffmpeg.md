# FFmpeg 媒体读写: h3_ffmpeg

`h3_ffmpeg.c` / `h3_ffmpeg.h` 是引擎与磁盘媒体文件之间的唯一桥梁：读入参考图像/视频/音频，写出最终生成的视频与音频。它通过外部 `ffmpeg` / `ffprobe` 二进制（由 `H3_FFMPEG` / `H3_FFPROBE` 环境变量或 PATH 定位）以管道（pipe）方式通信，避免中间文件。

## 读取

- `h3_ffmpeg_probe(path, &info)`：调用 `ffprobe` 探测宽度、高度、帧数、时长、像素格式，供 [视觉编码器](vision-encoder.md) 与 [视频编码器](video-vae.md) 决定采样策略。
- `h3_ffmpeg_read_image(path, &image)`：把图像解码为 channel-major F32（RGB，值范围 0..1），并交由 Accelerate vImage 做高质量缩放。
- `h3_ffmpeg_read_video(path, frame_indices[], n, &frames)`：按关键帧索引解码视频帧为 F32 序列。
- `h3_ffmpeg_read_audio(path, &pcm)`：解码音频为 F32 PCM（用于 [音频 VAE](audio-vae.md) 编码）。

## 写出

- `h3_ffmpeg_writer_open(params, &writer)`：开启一个写入上下文，内部启动两条并发 `ffmpeg` 子进程管道——一条接收 RGB24 视频帧，一条接收 F32 PCM 音频。
- `h3_ffmpeg_writer_write_frame(writer, rgb, w, h)` / `h3_ffmpeg_writer_write_audio(writer, pcm, n)`：增量推送帧/样本；这使得 [公共 API](api-public.md) 的 `on_frame` 回调可以一边生成一边封装。
- `h3_ffmpeg_writer_close(writer)`：合并两条管道并收尾容器（如 MP4 + AAC），保证音视频时间轴对齐。

## 设计取舍

用管道而非落盘，省去临时文件与额外的编解码往返，使端到端延迟更低、对 SSD 更友好（尤其在 `ssd_streaming` 模式下权重也在流式读取）。代价是依赖外部 `ffmpeg`/`ffprobe` 可用；引擎在 `h3_load_dir()` 时若找不到会报错提示安装。

相关：读取结果喂给 [视觉/视频/音频编码器](vision-encoder.md)；写出消费 [视频 VAE](video-vae.md) 与 [音频 VAE](audio-vae.md) 的输出。
