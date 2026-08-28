# DiT 主干与去噪: h3_dit

`h3_dit.c` / `h3_dit.h` 是引擎的计算核心：一个 50 块（block）的扩散 Transformer，负责从噪声潜变量逐步去噪出视频与音频潜变量。结构与权重布局由 `h3_dit_schedule.h` 中的宏固定。

## 结构常量（`h3_dit_schedule.h`）

- `H3_DIT_BLOCKS = 50`：残差块数（对应默认 50 层 DiT；`H3_MIN_DIT_LAYERS=35` 为下限）。
- `HIDDEN = 5376`、`HEADS = 56`、`HEAD_DIM = 96`、`MLP = 21504`：隐藏/注意力/MLP 维度，与 Qwen3-VL 嵌入维度对齐。
- `DIT_IN = 24`（视频潜变量通道）、`DIT_IN_AUDIO = 32`（音频潜变量通道）：输入/输出潜变量的通道数。

## 权重布局（每个残差块）

每个块包含 QKV 投影、输出投影、FC1/FC2、AdaLN 调制相关权重，按以下顺序排布（用于 `h3_safetensors` / `h3_weights` 的按需读取）：

`patch_embed` → `x_embedder` → `ada_embedder.1/ada_embedder.2` → 50×（QKV、proj_out、FC1、FC2、AdaLN 调制权重）→ `final_linear` → `final_norm` → `out_layers`。权重可在 BF16 与 int8（group/row）之间切换（`h3_weights` 控制）。

## 去噪主循环

- `h3_dit_denoise(ctx, params, cond, layout, sigmas, schedule, out_latent, on_progress)`：通用去噪入口，按 `h3_sigma_schedule` 迭代。
- `h3_dit_denoise_euler(ctx, params, cond, layout, sigmas, schedule, out)`：Euler 积分实现，每步调用一次块堆叠的前向；若 `preview_denoise` 开启，每个 Euler 步后通过 [视频 VAE](video-vae.md) 解码一帧并投递预览。
- 块前向内部顺序（每个残差块）：RMSNorm → QKV（量化 matmul）→ 注意力（56 头）→ 输出投影 → AdaLN 调制 → FC1（SiLU）→ FC2 → 残差加。AdaLN 调制参数来自 [AdaLN 调度](dit-schedule.md) 的预计算。

## 加速开关

- `core_reuse`：每隔 N 步重算 Transformer core（其余步复用上一结果）。
- `denoise_reuse`：每 N 步评估一次去噪器。
- `token_reduction`：相邻水平视频 token 配对以减少注意力计算。
- 块剪枝：`h3_dit_schedule_gate_score()` 提供每块门控分数，低分块在部分步被跳过。

## 输出

去噪结束得到视频潜变量（`[24, F/2, H/16, W/16]` 量级，patch=96）与音频潜变量（`[32,2,T]`），分别送入 [视频 VAE](video-vae.md) 与 [音频 VAE](audio-vae.md) 还原为媒体。

相关：条件来自 [宿主逻辑](host.md) 与 [多模态嵌入编排](multimodal.md)；调度与剪枝见 [AdaLN 调度与门控剪枝](dit-schedule.md)。
