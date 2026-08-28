# AdaLN 调度与门控剪枝: h3_dit_schedule

`h3_dit_schedule.c` / `h3_dit_schedule.h` 负责两件事：在每个去噪步预计算 DiT 所需的 AdaLN 调制参数，以及为块剪枝提供门控分数。它把"调度"从去噪主循环（`h3_dit_denoise_euler`）中解耦出来，使主循环可以简单地查表使用。

## 结构常量（`h3_dit_schedule.h`）

- `H3_DIT_BLOCKS = 50`、`HIDDEN = 5376`、`HEADS = 56`、`HEAD_DIM = 96`、`MLP = 21504`、`DIT_IN = 24`、`DIT_IN_AUDIO = 32`：与 [DiT 主干](dit.md) 共享同一组维度宏，确保调制参数形状匹配。
- `SIGMA_STEPS = 1000`：连续噪声水平（sigma）的离散化粒度，用于构造去噪调度。

## 调度构造

- `h3_schedule_build(params, steps, &sched)`：依据去噪步数（默认 20）与 `SIGMA_STEPS` 生成 `h3_sigma_schedule`——一组从大到小的 sigma 值，决定每个 Euler 步的噪声水平。
- `h3_dit_schedule_precompute(ctx, params, cond, layout, sigmas, &schedule)`：对每个去噪步，把当前 sigma 与条件投影为 50 块各自的 AdaLN 调制参数（shift/scale/gate），结果存入 `h3_dit_schedule` 供主循环逐块取用。预计算避免在每个块前向内重复投影时间嵌入。

## 门控剪枝

- `h3_dit_schedule_gate_score(schedule, block_index)`：返回某块在"当前去噪阶段"的门控分数。分数低表示该块对当前噪声水平的贡献小，可在部分步跳过以加速（配合 `H3_MIN_DIT_LAYERS` 下限保证质量底线）。
- 该机制与 `core_reuse` / `denoise_reuse` 正交：后者是时间维度的复用，门控剪枝是空间（块）维度的稀疏化。

## 设计取舍

把 AdaLN 调制与时间调度集中在此模块，让 [DiT 主干](dit.md) 的主循环保持"读调制→跑块→采样"的线性结构，易于在 Metal 内核间流水化。代价是需要额外的预计算缓冲，但因其可在 `core_reuse` 周期内复用，实际开销很低。

相关：预计算输出由 [DiT 主干](dit.md) 的块前向内联使用；sigma 调度驱动 `h3_dit_denoise_euler`。
