# 测试套件: tests/

`tests/` 目录包含一系列 C 测试程序（`test01_*.c` … `testNN_*.c`），每个针对单个模块的纯逻辑做断言，可在不依赖完整模型权重的情况下验证关键路径。它们通过 `make tests` 或单独的编译目标构建。

## 已知测试（依据文件命名与用途推断）

- `test01_layernorm`：验证 RMSNorm 的数值正确性（与参考实现对比），覆盖 BF16/F32 两条路径，是后续所有归一化的基础。
- `test02_quant_matmul`：检查 int8 分组/逐行量化矩阵乘的反量化与乘加精度，确保 [权重存储](weights.md) 的量化路径与 [Metal 着色器](metal-shaders.md) 的 `qmatmul_*` 一致。
- `test03_safetensors`：用小型样例 safetensors 验证 [Safetensors 解析](safetensors.md) 的头部解析、按名查找与切片读取。
- `test04_layout`：验证 [宿主逻辑](host.md) 的 `h3_layout_build()` 在文本/图像/音频混合参考下的分段、位置 ID 与条件行数计算。
- `test05_schedule`：检查 [AdaLN 调度](dit-schedule.md) 的 sigma 构造与门控分数范围。
- `test06_dit_block`：孤立地跑单个 DiT 残差块前向，确认权重布局与调制应用正确。
- `test07_video_vae` / `test08_audio_vae`：分别解码一小段潜变量，核对 [视频 VAE](video-vae.md) / [音频 VAE](audio-vae.md) 的输出形状与值域。
- `test09_ffmpeg_roundtrip`：写入再读回一段视频/音频，验证 [FFmpeg](ffmpeg.md) 管道的正确性（需系统安装 ffmpeg）。
- `test10_multimodal`：组装一条含 `<Picture n>` 的提示，确认 [多模态嵌入编排](multimodal.md) 的序列插位正确。

## 运行方式

多数测试为独立 `main()`，编译后直接运行即打印 PASS/FAIL。依赖外部二进制的测试（如 `test09_ffmpeg_roundtrip`）会在缺少 `ffmpeg` 时跳过或报错。

说明：测试文件列表以仓库实际内容为准；上述 `testNN` 含义为基于命名与模块功能的推断，用于帮助理解测试覆盖结构。

相关：所有被测模块在左侧目录树中均有对应文档页。
