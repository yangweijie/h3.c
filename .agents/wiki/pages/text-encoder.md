# Qwen3-VL 文本编码器: h3_text_encoder

`h3_text_encoder.c` / `h3_text_encoder.h` 实现了 H3 文本侧的条件生成：把 token 序列（及可选的视觉嵌入）喂入 Qwen3-VL 语言模型的前 50 层，产出 BF16 条件嵌入。

## 核心函数

- `h3_text_encode_bf16(params, tokenizer, model_name, text, cond)`：纯文本路径。输入提示 → 分词 → 前 50 层 Qwen3-VL → BF16 嵌入，写入 `h3_host_cond.text_embed`。这是 FL2VA（文本到视频/音频）的主路径。
- `h3_text_encode_multimodal_bf16(params, tokenizer, model_name, text, image_embed, audio_embed, n_audio, pr_cond, cond)`：多模态路径。把 [Qwen 视觉编码器](vision-encoder.md) 产生的图像嵌入与音频条件，按 `<Picture n>` 呈现格式插入文本序列对应位置，再走同样的 50 层语言层，输出联合 BF16 嵌入。用于 Ref2VA（参考到视频）。

## 关键设计

- **只取前 50 层**：H3 复用 Qwen3-VL 的语言主干而非完整模型，输出 5376 维（`HIDDEN`）嵌入，与 [DiT 主干](dit.md) 的隐藏维度对齐。
- **BF16 输出**：所有中间与最终嵌入均为 BF16，直接驻留共享 Metal 缓冲，避免 F32 往返开销。
- **条件分离**：`pr_cond`（提示参考条件，如角色设定文本）与 `image_embed`/`audio_embed` 分别传入，由本模块负责在序列中正确放置。

## 与宿主的关系

本模块产出 `h3_host_cond` 的嵌入字段，由 [宿主逻辑](host.md) 持有并在去噪步之间复用（除非 `core_reuse` 强制重算）。嵌入维度与布局必须匹配 `h3_layout_build()` 计算的段结构。

相关：文本序列来自 [Tokenizer](tokenizer.md)；多模态装配见 [多模态嵌入编排](multimodal.md)。
