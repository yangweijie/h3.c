# 宿主逻辑与张量布局: h3_host

`h3_host.c` / `h3_host.h` 负责把"模型能力 + 提示 + 参考"翻译成确定性的张量排布，并作为条件缓存的拥有者。它不持有权重，只持有布局计划与中间嵌入缓冲。

## 核心职责

- **布局规划**：`h3_layout_build()` 依据提示文本长度、图像/音频参考数量，计算序列分段、绝对位置 ID、以及图像/音频条件行数。它区分 `Text` / `Image` / `ReferenceImage` / `ReferenceAudio` / `Audio` 段，并记录每段的子段（subsegment）起止帧，供位置编码使用。
- **条件缓存**：`h3_host_cond` 结构持有文本嵌入（`text_embed`）、图像嵌入（`image_embed`）、音频条件（`audio_cond`）、以及提示参考条件（`pr_cond`）。对应函数 `h3_host_cond_create()` / `h3_host_cond_free()` / `h3_host_cond_set_*()` 在编码完成后填充，并可由 `h3_cache` 跨请求复用。
- **线性层**：`h3_linear_bf16()` 与 `h3_linear_int8()` 是本模块提供的两个基础投影算子，分别作用于 BF16 与 int8 权重；DiT 与编码器中的多数线性变换都最终落到这两个函数之一，便于统一调度到 GPU。

## 与 DiT 的关系

DiT 去噪主循环需要三类输入：噪声潜变量、调度（由 `h3_dit_schedule` 提供）、以及本模块算好的条件与布局（段/位置）。`h3_host` 在每次请求开始时一次性算好布局，随后在多个去噪步之间复用（除非 `core_reuse`/`denoise_reuse` 要求重算某些部分）。这种"布局一次算、张量多次用"的设计是引擎低延迟的关键。

## 关键类型

- `struct h3_layout`：序列分段、位置、条件行数。
- `struct h3_host_cond`：跨步复用的条件嵌入集合。
- 段枚举 `H3_SEG_*` 与 `h3_subsegment`：描述帧级子段，用以支持参考视频/音频的时序对齐。

相关：条件被送入 [DiT 主干](dit.md)；图像/音频嵌入来自 [多模态嵌入编排](multimodal.md) 与 [Qwen 视觉编码器](vision-encoder.md)。
