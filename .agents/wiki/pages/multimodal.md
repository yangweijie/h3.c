# 多模态嵌入编排: h3_multimodal

`h3_multimodal.c` / `h3_multimodal.h` 站在 [Tokenizer](tokenizer.md)、[Qwen3-VL 文本编码器](text-encoder.md) 与 [Qwen 视觉编码器](vision-encoder.md) 之上，负责把异构输入（文本 + 图像/视频/音频参考 + 角色文本）编排成一条可直接喂入 DiT 的联合序列。

## 核心函数

- `h3_multimodal_build(params, tokenizer, prompt, refs, n_refs, &mm, cond, layout)`：主入口。解析提示中的 `<Picture n>` 占位符与参考列表，决定哪些参考是图像/视频/音频/角色文本，并分派到相应编码器。
- `h3_multimodal_free(&mm)`：释放本次请求的中间缓冲。

## 编排逻辑

1. **文本分词**：提示经 `h3_tokenizer_encode` 转为 token，并定位 `<Picture n>` 占位符。
2. **参考分类**：`h3_ref` 数组决定每个参考的类型——图像/视频走 `h3_vision_encode_*`，音频走 [音频 VAE](audio-vae.md) 的条件路径，纯文本（如角色设定）作为对话上下文而非视觉 token。
3. **序列装配**：视觉/音频嵌入按占位符顺序插入文本序列；`pr_cond`（提示参考条件）单独保存，供文本编码器的多模态入口使用。
4. **统一编码**：装配后的序列送入 `h3_text_encode_multimodal_bf16()` 的前 50 层 Qwen3-VL，产出 BF16 联合嵌入。
5. **布局回填**：最终嵌入的尺寸/段信息写回 `h3_layout`（由 [宿主逻辑](host.md) 的 `h3_layout_build` 预设骨架），供 DiT 去噪使用。

## 设计取舍

把"多模态装配"独立成层，使 DiT 与编码器都不必理解 `<Picture n>` 语法与参考类型判断，降低耦合。代价是多一层缓冲管理，但换来清晰的分层与可测试性（各编码器可独立单测）。

相关：输出写入 [宿主逻辑](host.md) 的 `h3_host_cond`，随后进入 [DiT 主干](dit.md)；音频条件来自 [音频 VAE](audio-vae.md)。
