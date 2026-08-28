# 权重存储: h3_weights

`h3_weights.c` / `h3_weights.h` 管理"已解析的 safetensors 分片"到"持久化 Metal 张量"的映射。它是引擎内存占用的核心控制点。

## 核心类型

- `struct h3_weights`：一组已打开的 safetensors 分片集合，提供按名查张量。
- `enum h3_weight_type`：`H3_WEIGHT_BF16`（默认）、`H3_WEIGHT_INT8_GROUP`（分组量化）、`H3_WEIGHT_INT8_ROW`（逐行量化）。精度与量化方式由模型配置或运行时开关决定（int8 用于更激进的内存/带宽节省，配合 `h3_host` 的 `h3_linear_int8`）。

## 关键函数

- `h3_weights_open(paths[], n)`：打开一个或多个 safetensors 分片目录。
- `h3_weights_find(w, name)`：跨分片按名定位张量；支持"懒加载"——仅在第一次被 DiT/编码器引用时读取到共享缓冲。
- `h3_weights_load_bf16(w, entry, dst)`：把 BF16 张量提升到目标 GPU 缓冲（通常是 `h3_gpu` 的共享张量）。
- `h3_weights_set_streaming(w, model_size, dit_layers, core_reuse, ssd_streaming)`：配置流式策略——当 `ssd_streaming=1` 时仅保留 2 个 BF16 DiT 块在内存，重叠磁盘 I/O 与 GPU 执行；`core_reuse` 决定多少步重算一次 Transformer core。

## 设计取舍

权重层刻意区分"元数据（始终在内存）"与"张量数据（按需/流式）"。这使得引擎在 24 GB 级别的模型上也能在低内存设备上运行：`h3_dit` 只请求它当前块需要的权重，[宿主逻辑](host.md) 在 `core_reuse` 周期外复用已加载块。量化路径（`int8_group` / `int8_row`）让整机权重带宽进一步下降，代价是精度（由 `--use-slower-*` 可回退到 BF16 对齐）。

相关：张量最终驻留于 [Metal 设备](gpu.md) 的 `h3_gpu_tensor`；解析来自 [Safetensors 解析](safetensors.md)。
