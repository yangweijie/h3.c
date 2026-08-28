# Qwen 视觉编码器: h3_vision_encoder

`h3_vision_encoder.c` / `h3_vision_encoder.h` 处理图像与视频参考输入，把它们编码为与文本嵌入同源的视觉 token，供 [多模态嵌入编排](multimodal.md) 拼接。

## 核心函数

- `h3_vision_encode_image_bf16(params, path, image_embed)`：把单张参考图（经 [FFmpeg](ffmpeg.md) 读入、vImage 高质量缩放）编码为 BF16 视觉嵌入。用于 Ref2VA 中的 `Image` 段与 `ReferenceImage` 段。
- `h3_vision_encode_video_bf16(params, path, video_embed, frame_indices[], n)`：把参考视频按关键帧索引采样、编码为时序视觉嵌入序列，供视频参考的逐帧条件使用。

## 关键设计

- **与 Qwen3-VL 对齐**：视觉编码器输出维度与文本编码器的隐藏维度一致（5376），使得二者可在同一序列中共存。
- **图像条件行数**：嵌入高度/宽度（patch 化后）决定 [宿主逻辑](host.md) 中 `image_cond_rows` 的数量，进而影响 DiT 序列总长度的布局。
- **高质量缩放**：采样帧在送入编码器前用 Accelerate 的 vImage 做高质量缩放，保持与训练分布一致的预处理。

## 与多模态层的关系

本模块只负责"图/视频 → 视觉嵌入"，不负责把嵌入拼回文本序列；拼接与 `<Picture n>` 占位符的填充由 [多模态嵌入编排](multimodal.md) 完成，最后交由 [Qwen3-VL 文本编码器](text-encoder.md) 的前 50 层统一处理。

相关：参考输入由 [命令行交互](cli.md) 或公共 API 的 `h3_ref` 提供；输出写入 `h3_host_cond.image_embed`。
