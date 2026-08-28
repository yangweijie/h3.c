# Task Plan: 分析+优化 minimax H3 在 32G Mac mini 的运行限制

## Goal
分析为何只能 32GB 跑通；借鉴 ds4 给出并落地优化，让 16/24GB 设备也能跑。

## Phases
1. [done] 收集内存线索、量化根因、ds4 对照思路。
2. [done] 发现：ds4 思路2（自动 cache 预算规划器）**已实现** —— `h3_memory_plan.c/.h` 移植自 ds4，并在 `h3.c:862-882` 接入 `h3_generate`。`h3_params.memory_plan_auto` 默认开。
3. [done] 落地剩余缺口（让规划器真正降低峰值）：
   - (B1) 打破 int8 / ssd_streaming 互斥：规划器在流式时仍建议 int8（ds4 思路1：二者正交）；同步放宽 h3_cli.c 的互斥报错。
   - (B2) 规划器增加"VAE/文本编码器也应流式"的建议字段，供后续 h3_video_vae / h3_text_encoder 实现消费（当前 VAE 整网常驻是 32G 主因）。
   - (B3) CLI 增加 `memory-plan` 检视命令，打印设备 working set 与所选策略（可观测性）。
   - (C1) VAE 解码器权重流式落地：`run_stream_tile` 替代常驻路径，两条解码入口（常驻 `decoder_decode_chunk` + 分块 `decode_chunked`）均按 `vae.streaming` 选择。
   - (C2) DiT int8 + ssd_streaming 引擎级真正同时生效（去掉 `h3_dit.c:1633` 的 `!ssd_streaming` 门控，读入即量化）。
   - (C3) 规划器 streaming-aware 估算：用真实常驻体量（DiT 2 块 + VAE 1 块 + 编码器 0）决策，导出 `H3_VIDEO_VAE_LAYERS=36` 宏替代硬编码 28。
   - (C4) 代码审查(CR)修复：`decode_chunked` 路径忽略 streaming 🔴、VAE 块数硬编码 28 🟡、注释漂移 🟡，均已修复。
4. [done] 验证（make + 运行时）+ 文档收尾。
   - 规避 SDK 不兼容：用 `make CC="xcrun clang"`（Apple clang 21 + 26.5 SDK）替代 Homebrew LLVM 16（其不认识 26.5 SDK vecLib 头的 visionOS 平台名，是此前 make 失败的真正原因之一）。
   - 修复 Makefile：补充漏列的 `h3_memory_plan.c` 到 `LIB_C`（此前 `h3` 链接期缺 `_h3_memory_plan_auto`，整个项目根本链接不出可执行）。
   - 编译验证：干净全量 `make h3 libh3.a h3_tests h3_audio_gpu_tests` **0 errors**，产物齐全（h3 559KB / libh3.a 768KB）；VAE 两测试目标亦编译通过。
   - 运行时验证：`./h3_tests` → ok 1768 checks；`./h3_audio_gpu_tests` → ok（Metal primitives 与 host 一致）；规划器 16/24/32GB 模拟决策符合预期（`_Static_assert(H3_VIDEO_VAE_LAYERS==36)` 通过）。
   - 文档收尾：AGENTS.md 已补充构建须用 `xcrun clang`、源文件须登记 `LIB_C`、VAE 流式双路径约定、内存规划器与 `!memory-plan` 命令；findings.md / progress.md 已更正并同步。
   - 剩余（环境限制，非代码问题）：实机 16/24GB 端到端跑通 + real-* / parity 测试，需 MiniMax-H3 权重与 misc/fixtures（本机均无）。

## 备注
- 不碰 VAE/文本编码器内核（风险高），先让规划器给出正确建议并解除人为互斥，为后续流式落地铺路。
