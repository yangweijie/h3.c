# Metal 着色器与后端: h3_metal / h3_gpu.m

`h3_metal.h` 声明 Metal 内核接口，`h3_metal.m` 以 Objective-C 实现这些内核并把 Metal 着色器源码（`.metal`）作为字符串常量内联，`h3_gpu.m` 则封装设备/缓冲/命令队列并把算子分派到内核或 MPS。二者共同构成引擎的 GPU 后端。

## 着色器源

`h3_metal.m` 持有若干 `.metal` 源字符串（如 RMSNorm、量化 matmul、AdaLN 调制、patch embed、VAE 上采样与 BigVGAN 反卷积）。这些源在 `h3_gpu_create()` 时通过 `shader_source_path` 指定的库加载，或当未指定时回退到内联编译。把源内联的好处是单一可执行文件即可运行，无需随包携带 `.metal` 文件。

## 内核类别

- **归一化**：`rms_norm_*`——BF16/F32 的逐通道 RMS 归一化，DiT 与编码器共用。
- **量化矩阵乘**：`qmatmul_*`（group / row 两种 int8 布局）——对应 [权重存储](weights.md) 的 `int8_group`/`int8_row` 路径；在共享缓冲上做分组/逐行反量化的乘加。
- **激活与调制**：`silu_*`、`adaln_modulate_*`——把 [AdaLN 调度](dit-schedule.md) 投影出的调制参数应用到输入张量。
- **嵌入与采样**：`patch_embed_*`、`vae_upsample_*`、`bigvgan_conv_transpose_*`——VAE 解码与音频合成的核心算子。

## 命令调度

`h3_gpu.m` 维护一个 `MTLCommandQueue`，并暴露：

- `h3_gpu_dispatch_kernel(name, tensors[], params)`：提交单个内核。
- `h3_gpu_commit()` / `h3_gpu_wait()`：刷盘与同步；去噪主循环在每个 Euler 步结束时等待，以保证 `on_frame` 预览的数据已就绪。
- `h3_gpu_profiling`：可选计时，用于 `--preview-denoise` 之外的性能分析。

## 与 MPS 的关系

稳定算子（softmax、卷积转置、某些归一化）走 MPS（`h3_mps_*`），手写内核用于 MPS 不支持或性能更优的路径（如 int8 量化 matmul）。`--use-slower-*` 在前端层把部分算子强制切回 MPS 以获得可复现性。

相关：设备抽象接口见 [Metal / MPS 设备封装](gpu.md)；这些内核被 [DiT 主干](dit.md)、[视频 VAE](video-vae.md) 与 [音频 VAE](audio-vae.md) 调用。
