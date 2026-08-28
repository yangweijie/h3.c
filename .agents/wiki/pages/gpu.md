# Metal / MPS 设备封装: h3_gpu

`h3_gpu.h` 是 C 侧设备抽象（纯声明），实现分散在 `h3_gpu.m`（Objective-C 设备/缓冲/内核封装）与 `h3_metal.m`（Metal 着色器与 MPSGraph 调度）。这是引擎唯一直接触碰 Metal 框架的层。

## 设备与上下文

- `h3_gpu_create(device_index, shader_source_path)` / `h3_gpu_destroy()`：创建/销毁 Metal 设备句柄。`shader_source_path` 指向编译好的 `.metal` 库（或在 `h3_metal.m` 中内联的源）。
- `h3_gpu_info`：返回设备名称、内存、支持的精度（BF16 / int8）等。

## 张量与缓冲区

- `h3_gpu_tensor`：跨算子的共享 Metal 缓冲包装，记录 `dtype`（`bf16`/`f32`/`int8`）、形状、以及"是否是某 DiT 块的独占缓冲"（用于 `ssd_streaming` 的内存复用）。
- `h3_gpu_tensor_create()` / `h3_gpu_tensor_free()` / `h3_gpu_tensor_upload()`（主机→设备）/`h3_gpu_tensor_download()`（设备→主机，用于取回潜变量与最终帧）。
- `h3_gpu_buffer_reuse()`：在去噪步之间复用临时缓冲，避免每步分配。

## 算子调度

- 手写 Metal 内核（`h3_gpu_*_kernel` 系列）：RMSNorm、量化 matmul、SiLU、AdaLN 调制、patch 嵌入等的 GPU 实现，定义在 `h3_metal.m`。
- MPS 图（`h3_mps_*`，如 `h3_mps_softmax`、`h3_mps_conv_transpose`）：对稳定、可融合的算子使用 MPSGraph / MPSCNN，配合 BF16 路径。
- `h3_gpu_dispatch()`：把算子提交到命令队列，并管理 `core_reuse`/`denoise_reuse` 带来的部分重算。

## 精度与回退

`shader_source_path` 区分"快速 Metal 内核"与"可移植 MPS/BF16 实现"；`--use-slower-*` 开关在 CLI 层把某些算子切回 MPS，用于与参考实现对齐/诊断。相应的着色器源在 `h3_metal.m` 中以字符串常量存在，避免了外部 `.metal` 文件的依赖。

相关：权重张量来自 [权重存储](weights.md)；具体内核实现见 [Metal 着色器与后端](metal-shaders.md)。
