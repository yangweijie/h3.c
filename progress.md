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

## 2026-09-01 — P4：默认 int8 流式路径（run6）

### 20260901 11:30:45 — P4 完成（run6, PID 80159）
- 进程正常退出（PROCESS GONE）。
- log tail：`video VAE load 36/36` → `FFmpeg 22/22` → `h3: wrote /tmp/h3out/fork_stream_int8_nolora_256.mp4`。
- 错误计数 0；产物 122408 bytes。
- `cmp` 对比 P3/P4 mp4：**逐字节相同（IDENTICAL）** → P4 与 P3 走同一 BF16 计算路径，int8 未启用。
- ffprobe：256×256 h264+aac 24fps 0.92s 22 帧，合法。

### 根因（int8 为何没跑）
- `h3_gpu.m:364-373`：`m5 = [device.name rangeOfString:@"M5"]`；`wantsTensorOps = m5 && (!nax||!*nax||strcmp(nax,"0")!=0)`；
  `tensorOpsEnabled = gpu.library && wantsTensorOps`。
- 本机 GPU 设备名不含 "M5" → `tensorOpsEnabled=false` → `h3_dit.c:1749-1755` 的 `int8_mlp/qkv/attention_out` 全为 0。
- 故即便不传 `--use-slower-bf16-*`，streamed-block int8 也绝不启用，默认 int8 流式路径退化为 BF16 流式。

### P4 结论
- 默认 int8 流式路径在本硬件**退化为 BF16 流式并通过**（无错、产物合法）。
- 真正的 streamed-block int8 + 流式量化/submit 路径（P2 修复的代码）**需 M5 类 GPU 才能端到端验证**，当前硬件无法覆盖。
- 本机可稳定验证的仅是 BF16 流式生成（P3/P4 一致）。
