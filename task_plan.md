# Task Plan: fork(h3.c) 16GB 流式生成验证 + lora 合并

## Goal
让 fork 仓库（`/Volumes/data/git/c/h3.c`，antirez 原版 `8974cc0` 之上叠加 16GB 优化 + lora 合并）
在 16GB Mac 上能够：
1. **不接 lora 也能流式生成视频（基线验证）** ✅ 已完成
2. （后续）接入 lora 合并后同样能流式生成 — **当前被 `h3_dit.c:1713` 硬限制阻断，见 P6**

## Phases

### P0 梳理 fork 相对原版的改动 — `complete`
- 目录级 diff：fork 在 antirez 原版基础上叠加 ①16GB 优化 ②lora 合并 ③文档/测试。
- 16GB 优化：自动内存规划器（ssd_streaming + int8 组合）、VAE 双路径（常驻/流式）。
- lora 合并：`h3_lora.c/.h`，`W' = W + scale·(B@A)`，明确"SSD streaming 与 int8 路径不支持合并"。

### P1 修复 int8 / ssd-streaming 互斥误判 — `complete`
- 文件：`h3.c`（约 554 行冲突校验）。
- 旧逻辑把"streamed-block int8"与"use_int8_row_fc2"混为一谈，流式模式被误拦。
- 修复：仅拦截 `use_int8_row_fc2` 与 `ssd_streaming` 的组合。

### P2 修复流式 read 后无条件 `h3_gpu_submit` 导致 prime 失败 — `complete`
- 文件：`h3_dit.c`（`read_stream_layer`，约 768 行）。
- 旧逻辑 int8 关闭时仍调 `h3_gpu_submit`，而 submit 首行 `if (!gpu.command) return 0;` 必然失败；
  且通用文案吞掉真实错误。
- 修复：submit 同样以 `(int8_mlp||int8_qkv||int8_attention_out)` 守卫，int8 关时跳过。

### P3 验证 fork 不接 lora 流式生成 — `complete`
- 命令：`./h3 -d MiniMax-H3 -p "a red ball bouncing on a white floor" \
  -o /tmp/h3out/fork_stream_nolora_256.mp4 --ssd-streaming --steps 4 --frames 16 \
  --width 256 --height 256 \
  --use-slower-bf16-mlp --use-slower-bf16-qkv --use-slower-bf16-attention-output`
- 结果：11:11 启动 → 11:15 写出 mp4；denoise 4/4、video VAE 36/36、ffmpeg 22/22；错误计数 0。
- 产物：256×256，h264+aac，24fps，0.92s，22 帧，122408 bytes。
- **结论：目标 1 ✅ 达成**（走的是 BF16 关 int8 路径）。

### P4 验证默认 int8 流式路径 — `complete`（但 int8 未实际启用）
- 命令：`./h3 -d MiniMax-H3 -p "a red ball..." -o /tmp/h3out/fork_stream_int8_nolora_256.mp4 \
  --ssd-streaming --steps 4 --frames 16 --width 256 --height 256`（不传 `--use-slower-bf16-*`）。
- 结果：denoise 4/4、video VAE 36/36、ffmpeg 22/22，错误计数 0，产出 256×256 h264+aac 0.92s 视频。
- **关键发现**：产出 mp4 与 P3（BF16）逐字节相同（`cmp` IDENTICAL）→ 本机 int8 未启用，P4 实际仍走 BF16 流式。
- 根因：`h3_gpu.m:366` `wantsTensorOps = m5 && (...)`，`m5` = GPU 设备名含 "M5"；本机非 M5 → `tensorOpsEnabled=false`
  → `int8_mlp/qkv/attention_out` 恒为 0 → int8+流式量化路径在本硬件上不可达。
- **结论**：本机仅能验证 BF16 流式（P3/P4 一致通过）；streamed-block int8 + 流式路径需 **M5 类 GPU** 才能端到端验证，当前硬件无法覆盖。

### P5 提速优化 — `in_progress`（诊断完成，受硬件硬上限约束）
- 实测瓶颈（H3_PROFILE）：`BF16 SSD stream 72.495 GiB read in 94.376s (0.768 GiB/s), unhidden wait 78.894s`。
- 模型在 **USB SSD**（`/Volumes/data`，814 MB/s，Protocol: USB）；内部盘 `/` 仅 43Gi 空闲装不下 134GB → 无法搬去更快存储。
- 本机非 M5 → int8 量化不可用；16GB RAM < 134GB → 无法常驻。
- 结论：生成耗时被 USB 读速硬封顶（每个去噪步 ~47s I/O）。软件杠杆在此硬件收益极小：
  - steps 4→2 实测仅省 ~26s（244s→214s）；frames/res 已压到 16/256。
  - MPS 预热、core_reuse、token_reduction 均不解决 I/O 瓶颈。
- 唯一有实质收益的方向（待用户拍板）：
  (a) 更深流式预取（增加 stream slot / prefetch 深度）回收部分 79s unhidden wait（~30-40% 潜力，但改动刚修好的流式核心有风险）；
  (b) 换更快模型盘（Thunderbolt/USB4 SSD）或 M5 / >64GB RAM（硬件层面根治）。

### P6 验证 lora 合并 + 流式 — `blocked`（代码硬限制）
- `h3_dit.c:1712-1716`：若 `lora_path` 非空且 `ssd_streaming`，直接 `fail("LoRA merging is not supported with --ssd-streaming")`。
- 含义：本 fork（feature/lora-merge 分支）核心目标"lora 合并 + 流式生成"在代码层被显式禁止。
- 16GB 上非流式（全量常驻）会 OOM，故 lora 路径唯一可行形态正是流式 —— 当前被堵死。
- 可行出路（待用户拍板）：
  (a) 放宽 `h3_dit.c:1713` 限制，让 lora 权重在流式读取时合并（须确保 merge 在量化/submit 前完成，且不与 int8 流式冲突 —— 见 `h3_lora.h` 注释"SSD streaming 和 int8 路径不支持合并"）；
  (b) 先在大内存机器（>64GB 常驻）验证 lora + 非流式是否可行；
  (c) 评估 lora 合并到常驻权重的时机（加载期一次性 merge，而非流式每块 merge）。

## Errors Encountered
| Error | Phase | Resolution |
|-------|-------|------------|
| "int8 quantization of streamed DiT block failed" | P3 | 反遮罩错误文案 → 露出真错 "DiT SSD stream submit failed" |
| "DiT SSD stream submit failed" | P3 | P2 修复：int8 关时跳过 submit |
| "LoRA merge incompatible with SSD streaming"（早期 lora 失败） | P1 | P1 修复：仅拦截 row-fc2 int8 与流式组合 |

### P7 4B vs 50 层 性能计时对比 — `complete`
- 编码器隔离：4B 7.93s / 5.4 GiB vs 50 层 52.41s / 46.86 GiB（4B 快 **6.6×**、权重少 **8.7×**）。
- 完整生成（256×256 / 16 帧 / steps 4 / ssd-streaming）：4B **251.25s** vs 50 层 **292.80s**（4B 省 **~41.5s ≈14%**）。
- 结论：文本编码器**非总时长瓶颈**（DiT/VAE/流式 I/O 占 `sys` 52–61s）；换 4B 价值在**省内存/权重体积与编码启动延迟**，非显著缩短总生成时间。生产 `--steps 20` 时编码器占比更小、收益进一步缩小。
- 陷阱：`H3_CLIPPROJ_DIR` 持久导出导致首次"50 层"实测仍是 4B（251.05s），须 `env -u`/`unset` 强制默认路径。
