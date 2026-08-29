# Progress Log: 4B + ClipProj harness

## 2026-08-29

### Done
- 仓库获取：GitCode 镜像需认证弃用；ModelScope `NicoLab28/ClipProj-MiniMax-H3` 下载成功，SHA256 校验通过。
- 投影元数据：`tap=24, d_in=2560, d_out=5120, mlp_hidden=32768, mlp_depth=1, 无 W, cos_test=0.8144`。
- 模型：`Qwen3-VL-4B-Instruct`（8.3GB，`d_in=2560` 确认 4B）；python3.13 + torch2.12 / transformers5.15 / numpy / scipy。
- 编写 `clipproj_harness.py`（PyTorch transformers）：加载 `Qwen3VLModel` + 投影；`--hs-index` 默认 `tap+1=25`。
- Phase 4 实跑产出：`candidate.npz` 已生成（key `hello`, shape `(1, 5120)`, float64）。说明 4B 前向 + ClipProj 投影端到端可产出 `[seq,5120]`。
  - 注：当前仓库仅含 PyTorch 版 harness。**用户反馈 Phase 4 已用 MLX 跑通，但 MLX 脚本未在仓库中**——存疑，待确认/补入（见下）。

### Blocked / Errors
- 早期在本机（16GB，约 8GB 可用）以 PyTorch CPU fp16 加载 8.3GB 模型被 jetsam SIGKILL(137)。`candidate.npz` 应是在更大内存机器或 MLX 下产出。
- 层索引未校准：需读 ComfyUI `sd1_clip.py` 确认 `sm.layer=[24]` → HF `hidden_states` 索引（24 或 25）。

### 量化 / MLX 决策
- 用户选择 int8 预量化路径以在本机跑通；但 Mac CPU 上纯 int8 检查点稀缺（多为 CUDA bitsandbytes 格式），建议改用**原地 torchao int8**（同一 `Qwen3VLModel`，不换模型/运行时）。
- 用户进一步反馈 Phase 4 已用 MLX 跑通 4B+ClipProj。待估算：文本编码独立走 MLX 相比 PyTorch 的提速/省内存（见末尾）。

### MLX 文本编码估算（待实跑验证）
- 速度：Apple Silicon GPU（MLX）文本编码 4B 相较 PyTorch CPU 约 5–15x 前向加速；更关键在加载——PyTorch CPU 加载 4B fp16 约 4 分钟，MLX mmap 加载约 10–30s，单次 run harness 墙钟约 10–20x 提升。
- 内存：fp16 4B 权重均 ~8.3GB（统一内存）。差别在 MLX 可一键加载 4bit/8bit 量化（mlx-lm）→ ~2.3GB/4.2GB，避免本机 16GB 的 OOM；相较 fp16 省 ~2–3.5x。若仅比 PyTorch CPU fp16，绝对占用相近，但 MLX 统一内存 + lazy 分配更不易触发 jetsam。
- 注意：以上为基于 4B 级模型在 Apple Silicon 上典型表现的估算，需实跑计时 + 采样峰值 RSS 确认。

### Pending / Next
- [ ] Phase 5：读 `sd1_clip.py` 确认层索引（24 vs 25）。
- [ ] Phase 6：多 prompt 实跑 + 余弦对照 0.81；与 h3.c 32B 输出对齐。
- [ ] 确认 MLX 脚本位置并补入仓库（若确用 MLX）；或落地 torchao int8 路径（`--enc-quant`）。
- [ ] 实跑 MLX 计时 + 峰值内存以验证上面估算。

### Environment
- 用户建议后续用 `uv` 做可复现安装（当前 pip 已跑通）。
