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

## 2026-09-01 — P5：提速优化诊断

### 20260901 11:42:02 — 模型卷识别
- `df` / `diskutil`：`/Volumes/data` = **USB SSD**（Protocol: USB, Solid State: Yes, HFS+），814 MB/s。
- 内部盘 `/`（`/dev/disk3s1s1`）仅 43Gi 空闲，**装不下 134GB 模型** → 无法搬去更快存储。

### 20260901 11:46:06 — steps=2 实测
- 耗时 **214s**（steps=4 基线 ~240s）→ 砍 2 步仅省 ~26s。证实步数非瓶颈。

### 20260901 11:46:06 — H3_PROFILE 流式读精确数据
- `h3: BF16 SSD stream 72.495 GiB read in 94.376s (0.768 GiB/s), unhidden wait 78.894s`
- 即：每代流式读 72.5 GiB @ 0.768 GiB/s = 94s；其中 **78.9s 是计算端干等盘的 I/O 停滞**。
- 推论：USB SSD 读速是硬上限；每去噪步 ~47s I/O 不可避免（16GB RAM 装不下 36GiB/步 做 page-cache 跨步复用）。

### P5 结论（本硬件）
- 软件杠杆（砍步数/帧数/分辨率、MPS 预热、core_reuse、token_reduction）收益极小，因瓶颈是 I/O 而非算力。
- 真正提速需硬件：更快模型盘（Thunderbolt/USB4 SSD）、或 M5（int8 减半读量）、或 >64GB RAM（常驻去流式）。
- 唯一可做的代码杠杆：更深流式预取回收部分 79s unhidden wait（~30-40% 潜力，改流式核心有风险），待用户拍板。

## 2026-09-01 — 4B vs 50 层 计时对比（P7）

### 编码器隔离计时
- 4B（`test_clipproj_encoder`，"A red fox walking through snow" 6 token）：**7.93s** wall / 0.30 GPU s / 5.4 GiB。
- 50 层（`h3_real_prompt_test`）：**52.41s** wall / 1.84 GPU s / 46.86 GiB。→ 4B 编码器快 **6.6×**，权重少 **8.7×**。

### 完整生成计时（256×256 / 16 帧 / steps 4 / ssd-streaming，同 DiT/VAE 设置）
- 4B（设 `H3_CLIPPROJ_DIR`）：**251.25s**（日志确认 `text encoder (clipproj)` + 25 层，产出 `clipproj_cmp_4b.mp4`）。
- 50 层（`env -u H3_CLIPPROJ_DIR -u H3_CLIPPROJ_PROJ` 强制）：**292.80s**（确认 `text encoder 0/50 ... 50/50`，产出 `clipproj_cmp_50.mp4`）。
- 差值：4B 省 **~41.5s（≈14%）**。

### 陷阱复盘
- 第一次"50 层"完整生成因 `H3_CLIPPROJ_DIR` **持久导出**而实际仍是 4B（251.05s，日志 25 层暴露）。重跑 `env -u` 得真 50 层 292.80s。手动对比必须 `unset`。

### 结论
- 换 4B 省的是编码器阶段（52→8s）与权重体积极大（46.86→5.4 GiB），但端到端仅 ~14%。瓶颈在 DiT/VAE/流式 I/O（`sys` 52–61s），steps=4 时编码器占比已小；steps=20（生产）时占比更小。换 4B 价值在**内存/启动延迟**，非总时长。
