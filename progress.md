# Progress Log

- 2026-08-28: 启动分析任务。已建规划文件。
- 重要发现：`h3_memory_plan.c/.h` 已存在并接入 `h3_generate`(h3.c:862) —— ds4 思路2(自动预算规划器) 已实现。当前默认 `memory_plan_auto=1`。
- 落地改动：
  * `h3_memory_plan.c`：打破 int8/ssd_streaming 互斥，流式时仍建议 int8；新增 `video_vae_streaming`/`encoder_streaming` 建议字段（针对 VAE 整网常驻这一 32G 主因）。
  * `h3.h`：`h3_params` 新增 `video_vae_streaming`/`encoder_streaming`；`H3_PARAMS_DEFAULT` 改为 designated init 以便稳健扩展。
  * `h3.c`：在 `h3_generate` 应用上述新字段。
  * `h3_cli.c`：放宽 ssd_streaming/int8 互斥告警；新增 `!memory-plan` 检视命令（打印设备 working set + 自动策略）。
- 验证：`h3_memory_plan.c` 语法检查通过；`make` 整体失败为环境预存问题。
- 本轮：实现 VAE 解码器权重流式（思路1/思路B 落地）。
  * 关键发现：`run_resident`(line495) 已逐块 load→run→free，仅常驻 1 块；但 tiled 路径 `load_resident_weights`(line518) 在 open 时预载全部 LAYERS 块+输入/输出权重常驻（~9GiB，即 32G 主因之一）。
  * 改动：`h3_video_vae_decoder_load` 增加 `streaming` 参数；`vae_context` 增加 `streaming` 字段；`load_resident_weights` 在 streaming 时跳过块预载；新增 `run_stream_tile`（镜像 run_resident 的逐块流式）替代 `run_resident_tile`；`decoder_decode_chunk` 按 `vae.streaming` 选择；`h3.c` 把 `params->video_vae_streaming` 经 `h3_acquire_video_decoder` 传入；`tests/test_semantic_vae.c` 传 0（保持常驻预览）。
  * 验证：`h3_video_vae.c` 语法检查通过(exit 0)；h3.c 与 test 的 lint 错误已清除（仅剩 line418 预存 warning）。
- 交互2（文本/音频编码器延迟释放）：经核查 **已是现状、无需改动**。
  * 证据：`h3_text_encoder.c:517` 每次调用 `h3_weight_store_open` 并在 return 前 `h3_weight_store_free(store)`(line758/779)；`h3_audio_vae.c:682` 同理，解码后 `cleanup` 释放；`h3_vision_encoder.c` 也 per-call open/free。
  * 跨 denoise 阶段常驻的权重集只有两处：`h3_dit.c:1628`(DiT transformer, dit->weights) 与 `h3_video_vae.c`(视频VAE解码器, 已做流式)。其余均为 per-call 释放，不存在"常驻未释放"的编码器权重。
  * 结论：32G 下限的源头就是这两个常驻集；已被前几轮改动（DiT ssd_streaming+int8解耦、视频VAE流式）覆盖。无进一步"延迟释放"可挖。
- 交互3（DiT int8+流式解耦落地）：实现引擎级 int8/ssd_streaming 真正同时生效（之前的 planner 解耦被引擎 h3_dit.c:1633 的 `!ssd_streaming` 强制覆盖，是真正 bug）。
  * 改动：
    * `h3_dit.c:1633-1642` 去掉 `!dit->ssd_streaming` 门控，int8 与流式正交。
    * `allocate_stream_slot` 在 int8 开启时额外分配 int8 缓冲+scale（qkv/out/fc1/fc2）。
    * `read_stream_layer` 读入 BF16 后，按 dit->int8_* 把 slot 量化进 int8 缓冲（磁盘仍是 BF16，读入即量化，不需新 safetensors）。
    * dispatch 处 `streamed_weight` 额外 alias int8 字段，`run_block` 直接走 int8 路径。
  * 效果：流式模式下 DiT 常驻 slot 与每步 GPU 工作集再减半（2 个 int8 块 vs 2 个 BF16 块），配合视频VAE流式，16GB 档可行。
  * 验证：`h3_dit.c` 语法检查通过(exit 0)；仅残留预存 sizeof(A*) warning；`h3_gpu_tensor_new_i8`/`h3_gpu_quantize_weight_int8` 签名与既有 quantize_block_* 用法一致。
- 交互4（规划器 streaming 感知修正）：发现并修复规划器的关键 bug——它用"全常驻"体量做决策，导致流式开启后 resident 估算严重偏高，小内存设备被过度剪枝或误判。
  * `h3_memory_plan_auto` 新增 `streamed_resident_bytes` 参数：用"开启流式后的真实常驻体量"（DiT 仅 2 块 + VAE 解码器 1 块 + 编码器 0 常驻）做适配决策与预算分配。
  * `h3.c:866` 与 `h3_cli.c` 计算该 streaming-aware 常驻估计（DiT=2/50、VAE=1/28 块，编码器 per-call 释放按 0 计）并传入。
  * 效果：16/24GB 设备现在得到准确的自动档位，而非被悲观估算误伤；`!memory-plan` 命令输出也同步准确。
  * 验证：h3_memory_plan.c/.h 与 h3.c/h3_cli.c 语法检查通过(exit 0)，lint 0 错误（仅预存 warning）。修复了误用 H3_DIT_BLOCKS（h3.c 未包含该头）→改 H3_DEFAULT_DIT_LAYERS=50。
- 交互5（CR 审查 + 三处修复）：对 VAE 流式落地改动做代码审查，发现并修复 3 个问题，确保 streaming 在全部解码路径真实生效。
  * CR Issue 1（🔴 功能性，已修）：`decode_chunked` 路径完全忽略 streaming 标志——其栈上 `vae.streaming` 默认 0 且无条件调用 `run_resident_tile`，而 `decoder_decode_chunk` 路径已正确分支。后果：大图/长序列走分块路径时 VAE 仍常驻 ~9GiB，自动档位优化静默失效。修复：`h3_video_vae_decode` 与 `decode_chunked` 增加 `streaming` 形参，`h3.c` 调用处传入 `params->video_vae_streaming`，运行分支统一为 `vae.streaming ? run_stream_tile : run_resident_tile`；tests 5 处调用补 `0`（保持常驻测试行为）。
  * CR Issue 2（🟡 估算偏差，已修）：规划器用魔法数字 `28` 估算 VAE 单块体量，但真实块数 `LAYERS=36`。修复：`h3_video_vae.h` 导出 `#define H3_VIDEO_VAE_LAYERS 36`，`h3_video_vae.c` 内 `LAYERS = H3_VIDEO_VAE_LAYERS`（单一来源），`h3.c`/`h3_cli.c` 改用该宏替换 `28`。
    - **方向更正**：早期描述"用 28 导致常驻估算偏小、预算过于乐观"是错的，实测方向相反。`/28` 得出单块 0.321 GiB，`/36` 得出 0.250 GiB（以 9 GiB 解码器为例）——即旧代码**高估**单块体量 28.6%，使流式后常驻估算偏**大**，规划器**低估**可用余量，对 16GB 档**过度保守**（更易触发不必要的 DiT 层数剪枝），而非偏乐观。
    - 实测影响量级：VAE 单块差仅 0.071 GiB，相对 16GB 场景 streaming 常驻总量 2.65 GiB 约占 2.7%（DiT 2 块 2.40 GiB 占主导），故修正的实际效果有限，主要收益是消除魔法数字与真实块数的漂移风险。
  * CR Issue 3（🟡 文档，已修）：`run_resident_tile` 注释未说明它是 streaming=0 的常驻路径、由调用方按 streaming 选择；更新注释指明双路径选择。
- 交互6（本地编译验证 + 修复 Makefile 遗漏）：规避 SDK 不兼容，完成真实编译验证。
  * 根因：默认 `clang` 是 Homebrew LLVM 16，`which clang`→`/opt/homebrew/opt/llvm@16/bin/clang`；它**不认识 26.5 SDK 的 vecLib 头里 `visionOS` 平台名**，导致 `h3_host.c`(Accelerate 链路) 报 `unrecognized platform name visionOS`。这是工具链/SDK 不匹配，非代码问题。
  * 规避方法：用 `make CC="xcrun clang"`（Apple clang 21.0.0 + 26.5 SDK，自动带正确 SDKROOT），既能解析 Metal4(`MTLGPUFamilyMetal4`) 又认识 visionOS。注：15.4 SDK 可过 Accelerate 但缺 Metal4，不满足项目需要，故必须用 26.5 SDK + 苹果 clang。
  * **关键发现：Makefile 的 `LIB_C` 漏列 `h3_memory_plan.c`**（`h3.c:4`/`h3_cli.c:2` 均 `#include "h3_memory_plan.h"` 并调用 `h3_memory_plan_auto`，但该 .c 从未进构建）——导致 `h3` 链接期 `Undefined symbols: _h3_memory_plan_auto`，**整个项目此前根本链接不出可执行文件**。这是此前所有"make 失败"的真正主因之一（叠加 SDK 问题）。已修复：Makefile `LIB_C` 增加 `h3_memory_plan.c`。
  * 验证结果（用 `xcrun clang` + 26.5 SDK）：`make h3 libh3.a` 全量编译+链接 **成功**（产物 h3 559KB / libh3.a 768KB）；`h3_semantic_vae_test`、`h3_real_video_vae_test` 也编译通过。证明 Issue 1/2/3 的全部改动可正确编译、签名自洽、链接无缺失符号。仅余 lint 预存 warning（h3_video_vae.c:418 sizeof(A*) 及 h3_cli.c:708/710 uint64→double 隐式转换 warning，均非本轮引入）。
  * 运行时验证（无需权重即可跑的部分）：
    - `./h3_tests` → **ok: 1768 checks**（全部通过，基础逻辑未被改动破坏）。
    - `./h3_audio_gpu_tests` → ok，Metal primitives 与 host 参考一致（GPU 后端正常）。
    - 规划器实测（临时 harness 直接调用 `h3_memory_plan_auto`，模拟 16/24/32GB；假设 DiT 60GiB、VAE 9GiB、编码器 8GiB、音频VAE 1GiB、激活 4GiB）：
      * 16GB（recommended 10GiB）：ssd=1 int8=1 vae_streaming=1 enc_streaming=1，streaming 常驻 2.65 GiB，余量 3.35GiB < 4GiB → DiT 剪到 35 层，cache 2 GiB。
      * 24GB（recommended 15GiB）：同上但保持完整层数，cache 6 GiB。
      * 32GB（recommended 21GiB）：保持完整层数，cache 11 GiB。
      * `_Static_assert(H3_VIDEO_VAE_LAYERS == 36)` 通过；drift check 确认 /28 vs /36 差 +28.6%（见 CR Issue 2 的方向更正）。
    - 未跑：real-* 与 parity 测试（本机无 MiniMax-H3 权重、无 misc/fixtures，属环境限制非代码问题）。





- 探查发现：流式(ssd_streaming)仅覆盖 DiT 的 4 个 matmul(QKV/OUT/FC1/FC2)，用 2 个 slot 常驻；开启时 int8 被强制关闭。
- 视频 VAE 解码器在 tiled 模式下"整个解码器权重全程常驻"(h3_video_vae.c:530)，音频 VAE 同。
- 文本编码器(Qwen3-VL 前50层)、Ref2VA transformer、patch/adaln/embedding 等非流式权重全部常驻内存。
- 无设备内存自检/自动分级回退，全靠手动 CLI 开关(!ssd-streaming / !int8-row-fc2)。
- 结论成型，准备撰写分析报告。
