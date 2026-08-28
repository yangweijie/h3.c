# Safetensors 解析: h3_safetensors

`h3_safetensors.c` / `h3_safetensors.h` 提供对 MiniMax 发布的 safetensors 权重分片的只读解析能力。它不分配权重张量，只解析头部（header）并提供按需切片读取。

## 核心类型

- `struct h3_safetensors`：一个分片的句柄，持有已解析的头部与文件映射。
- `struct h3_safetensors_entry`：`{ name, offset, size, dtype, shape[] }`，描述单个张量在分片内的位置与形状。
- `enum h3_safetensors_dtype`：支持 `f32`、`f16`、`bf16`、`f64`、`i64`、`i32`、`i16`、`i8`、`bool`、`f8_e5m2`、`f8_e4m3fn` 等。

## 关键函数

- `h3_safetensors_open(path)` / `h3_safetensors_close()`：打开/关闭分片。
- `h3_safetensors_entries()`：返回头部中所有张量条目。
- `h3_safetensors_entry_get(st, name)`：按名查找张量（用于 `patch_embed`、`x_embedder`、`adaln` 等的按需读取）。
- `h3_safetensors_read(st, entry, dst, len)`：把某个张量切片读到目标缓冲区。权重通常先以 BF16 读入，再提升到 Metal 共享缓冲。
- `h3_safetensors_read_f32()` / `h3_safetensors_read_scaled_f32()`：直接读出为 F32（后者带缩放因子，用于 int8/逐行量化权重的反量化或统计）。

## 设计取舍

解析与加载分离：本模块只解析结构，真正的"映射到一个持久张量"由 [权重存储](weights.md) 完成。这样 safetensors 模块保持纯解析、无状态，便于单元测试（见 [测试套件](tests.md) 中的 `test03_safetensors`）。

相关：权重分片由 [权重存储](weights.md) 按需流式加载到 [Metal 设备](gpu.md) 的共享缓冲区。
