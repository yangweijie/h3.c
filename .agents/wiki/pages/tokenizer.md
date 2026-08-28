# Tokenizer: h3_tokenizer

`h3_tokenizer.h` 是 C 接口声明，`h3_tokenizer.m` 是 Objective-C 实现（基于 Apple 的 `NLTokenizer` / `tokenizer.json` 配置），负责把 UTF-8 提示切分为模型词表 ID。

## 核心类型与函数

- `struct h3_tokenizer`：分词器句柄，持有词表与合并规则。
- `h3_tokenizer_load(path)` / `h3_tokenizer_free()`：从模型目录加载词表与 BPE 合并。
- `h3_tokenizer_encode(tk, text, ids[], *n)`：UTF-8 → token ID 数组。引擎用它在 `h3_generate()` 起点把提示文本转为可喂入 [Qwen3-VL 文本编码器](text-encoder.md) 的序列。
- `h3_tokenizer_decode(tk, ids[], n, text)`：反向解码（用于调试/日志）。

## 与文本编码器的关系

分词结果（token ID 列表）与提示中的 `<Picture n>` 占位符一起，由 [多模态嵌入编排](multimodal.md) 组装成完整序列；文本部分走前 50 层 Qwen3-VL 语言层产生 BF16 嵌入。注意：参考图像/视频/音频不通过本分词器，它们由 [Qwen 视觉编码器](vision-encoder.md) 与音频 VAE 分别处理。

## 实现说明

以 `.m` 结尾表明它依赖 Foundation 框架（而非纯 C）。这是仓库中少数必须作为 Objective-C 编译的源码之一，其余为 `h3_gpu.m` / `h3_metal.m`。

相关：分词输出进入 [多模态嵌入编排](multimodal.md) 与 [宿主逻辑](host.md) 的布局计算。
