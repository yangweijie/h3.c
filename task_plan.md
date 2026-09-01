# Task Plan: fork(h3.c) 16GB 流式生成验证 + lora 合并

## Goal
让 fork 仓库（`/Volumes/data/git/c/h3.c`，antirez 原版 `8974cc0` 之上叠加 16GB 优化 + lora 合并）
在 16GB Mac 上能够：
1. **不接 lora 也能流式生成视频（基线验证）** ✅ 已完成
2. （后续）接入 lora 合并后同样能流式生成

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

### P4 验证默认 int8 开启的流式路径 — `pending`
- 不传 `--use-slower-bf16-*`，让内存规划器默认开启 int8，验证 streamed-block int8 量化是否也 OK。
- 当前仅证明 BF16 关 int8 路径可用，默认 int8 流式路径未单独验证。

### P5 提速优化 — `pending`（待用户拍板）
- 砍步数（4→更少）、砍帧数、MPS 首步编译预热缓存、量化整盘常驻、低像素+超分。

## Errors Encountered
| Error | Phase | Resolution |
|-------|-------|------------|
| "int8 quantization of streamed DiT block failed" | P3 | 反遮罩错误文案 → 露出真错 "DiT SSD stream submit failed" |
| "DiT SSD stream submit failed" | P3 | P2 修复：int8 关时跳过 submit |
| "LoRA merge incompatible with SSD streaming"（早期 lora 失败） | P1 | P1 修复：仅拦截 row-fc2 int8 与流式组合 |
