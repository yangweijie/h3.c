# Progress Log

## 2026-09-01 — fork 流式生成验证（无 lora）

### 20260901 11:13:50.45 — run5 进行中
- 进程存活 PID 78318，`STAT=UN`（I/O 等待，符合流式特征）。
- `denoise 1/4`，错误计数 0。
- 命令：纯 BF16 流式（`--ssd-streaming` + 三处 `--use-slower-bf16-*` 关 int8）。

### 20260901 11:16:40.03 — run5 完成
- 进程正常退出（PROCESS GONE）。
- log tail：`video VAE load 36/36` → `FFmpeg 22/22` → `h3: wrote /tmp/h3out/fork_stream_nolora_256.mp4`。
- 错误计数 0；产物 122408 bytes。

### 20260901 11:16:47.47 — ffprobe 验证
- `codec=h264`，`width=256 height=256`，`avg_frame_rate=24/1`，`duration=0.916667s`，`nb_frames=22`。
- 含 `aac` 音轨（FL2VA 自带）。
- **结论：fork 不接 lora 流式生成 ✅ 成功。**

### 修复回顾
- **P1 (`h3.c`)**：int8/流式互斥误判 → 仅拦 `use_int8_row_fc2` 与 `ssd_streaming`。
- **P2 (`h3_dit.c`)**：流式 read 后无条件 `h3_gpu_submit` → int8 关时跳过（见 findings.md "Bug B"）。

### 下一步
- **P4**：验证默认 int8 开启的流式路径是否同样跑通。
- **P5**：提速优化（砍步数/帧数、MPS 预热、低像素+超分），待用户拍板。
