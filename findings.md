# Findings: 为何只能 32GB Mac mini 跑 MiniMax-H3

## 根因（代码实证）

1. **权重常驻模型主导显存/统一内存占用**
   - `h3_dit.c:1251-1294`：`ssd_streaming` 仅对 DiT 的 4 个 matmul 做磁盘流式（2 个 `stream_slots` 常驻）。`load_block_norms` 仍把每块 norm/adaln 常驻；`video_patch_proj`/`audio_patch_proj`/`final_*`/`condition_proj`/`token_refiner` 等全部 `bf2/bf1` 常驻。
   - `h3_video_vae.c` 视频 VAE 解码器有两条路径：**常驻路径** `run_resident_tile`（整网 LAYERS=36 块常驻 ~9GiB，tiled 只切激活不切权重）；**流式路径** `run_stream_tile`（镜像 `run_resident` 的逐块 load→run→free，仅常驻 1 块）。由 `vae->streaming` 选择，两条解码入口（`decoder_decode_chunk` 与 `decode_chunked`）均已打通该标志。
   - 文本编码器(Qwen3-VL 前50层)、Ref2VA transformer 全程常驻（`h3.c:426-447` inventory 全量登记）。

2. **流式与 int8 互斥，二选一**
   - `h3_dit.c:1633-1641`：`int8_mlp/int8_qkv/int8_attention_out = !ssd_streaming && ...`，即开流式必关 int8；CLI 也强制 `!int8-row-fc2` 与 `!ssd-streaming` 不能同时开（`h3_cli.c:592-605`）。
   - 结果：要么全量 BF16 常驻(~大)，要么 BF16 流式(仅 DiT matmul 流式，VAE/文本编码器仍常驻)且放弃 int8 压缩。两者都仍吃大量内存。

3. **无自动内存分级 / 设备能力探测后的自适应**
   - `h3_device_info` 有 `physical_memory/recommended_working_set`，但代码未据此自动选模式；全靠手动 `!ssd-streaming` / `!int8-row-fc2` 命令。
   - 没有"按可用内存选择层数/量化/流式组合"的逻辑。

4. **激活与中间缓冲的峰值**
   - DiT 潜变量、视频 VAE 的 tile 拼接、32kHz 立体声 F32 PCM（`h3_audio_vae`）长视频时缓冲显著。
   - `token_refiner` 为 text_rows 分配多组 BF16 张量(`h3_dit.c:827-836`)。

## 来自 ds4 (DwarfStar) 的对照思路（新）

ds4 是同类 native 推理引擎，已解决"小内存 Mac + SSD streaming"问题，其机制可直接迁移到 h3.c：

1. **Tensor/层 级 hotlist 而非整块流式**：`ds4_streaming_hotlist.inc` 用 profiling 数据（`hits/weight`）把每个 routed expert 标记热门/冷门，热门常驻、冷门从 SSD 流式。h3.c 当前只流式 DiT 的 4 个 matmul（2 slot），且 VAE/文本编码器整网常驻——可改为对**每个 DiT 块、每个 VAE 解码层、norm/adaln** 做冷热分级。h3.c 已有 `block_active[]` 门控分数，可作为"热度"代理。

2. **自动 cache 预算规划器**：`ds4_ssd_auto_cache_plan(recommended_bytes, non_routed_bytes, per_expert_bytes, ...)` 取 `recommended_working_set * 80%`，减去必须常驻部分，余量作为可流式缓存预算；`ds4_streaming_manual_cache_safe_bytes` 用 `recommended*7/8` 再扣掉 context/KV/图后端开销。h3.c 已有 `h3_device_info.recommended_working_set`（等价 ds4 的 `recommendedMaxWorkingSetSize`），但**从未用于自动决策**——这是可直接复用的钩子。

3. **CPU/SSD 多级分层**：ds4 的 `ds4_layer_pack` 把层单调连续地分到 GPU/CPU/SSD 三级，超出预算的层 spill 到 CPU。h3.c 可做"GPU 常驻 DiT 头 N 块 + VAE + 文本编码器 → 其余 DiT 块流式/落盘"的等价分层。

4. **profiler 自动产出 hotlist**：`ds4_expert_profile_write_hotlist_file` 在运行期统计命中并写出 hotlist。h3.c 可加一个 `--profile` 模式，按 `block_active` 与 VAE 层复用频度自动生成 `h3_streaming_hotlist.inc`，消除手动调参。

## 量化估算（基于常量）
- 单 DiT 块 matmul: QKV=INNER*3*HIDDEN, OUT=HIDDEN*INNER, FC1=FFN*2*HIDDEN, FC2=HIDDEN*FFN；50 块 BF16 ≈ 数十 GB。int8 可减半但不覆盖 norm/adaln/VAE。
- 视频 VAE 解码器权重：共 LAYERS=36 个 transformer 块（见 `h3_video_vae.h` 导出宏 `H3_VIDEO_VAE_LAYERS`），单块体量 = `video_vae.bytes / 36`。常驻路径全 36 块 ~9GiB；流式路径仅常驻 1 块（~0.25GiB）。**注意早期估算误用硬编码 28，已修正为 36**。
  - 修正的影响方向：旧 `/28` 使单块估算为 0.321 GiB（真实 0.250 GiB），即**高估** 28.6% → 流式后常驻估算偏大 → 规划器低估可用余量、对 16GB 档过度保守（更易触发不必要的 DiT 层数剪枝）。实测该偏差仅占 16GB 场景常驻总量 2.65 GiB 的 ~2.7%（DiT 2 块占主导），故修正的实际收益主要是消除常量漂移风险，而非显著改变档位。
- 音频 VAE 另占。

## 为何下限约 32GB
全量 BF16 常驻(文本编码器+Ref2VA+DiT 全块+双 VAE 解码器+激活) 远超 16/24GB；即便开流式，VAE 解码器+文本编码器+2 个 DiT slot+激活仍逼近 24GB 上限，故实测仅在 32GB 稳定。但经本轮所有修复（DiT int8+流式解耦、视频 VAE 流式路径全入口打通、规划器 streaming-aware 估算用真实块数 36），16/24GB 设备可在自动档位下显著降常驻，不再被默认配置逼到 32GB。

## 流式生成失败的两个根因（本轮新发现，2026-09-01）

在 fork 不接 lora 的流式生成验证中，实测命中两个阻断性 bug，均已精准修复（详见 task_plan P1/P2）：

### Bug A：int8 与 ssd-streaming 互斥校验误判（h3.c 约 554 行）
- 旧逻辑将"streamed-block int8"（流式时仍可量化 DiT matmul 权重）与"use_int8_row_fc2"（行式 fc2 int8）混为一谈，
  在 `params.use_int8_row_fc2 && params.ssd_streaming` 之外，也对 streamed-block int8 报错，
  导致流式模式（即便用户只想纯 BF16 流式）被同款拦截挡下。
- 修复：冲突校验只拦截 `use_int8_row_fc2` 与 `ssd_streaming` 的组合；streamed-block int8 与流式可共存。

### Bug B：流式 read 后无条件调用 h3_gpu_submit（h3_dit.c read_stream_layer 约 768 行）
- 旧逻辑在 `read_stream_layer` 末尾无条件 `if (job->ok && !h3_gpu_submit(dit->gpu))`，
  而 `h3_gpu_submit` 首行 `if (!gpu.command) return 0;`——int8 关闭时从未调用 `h3_gpu_begin` 创建 command buffer，
  故 submit 必然失败，prime 阶段报 "DiT SSD stream submit failed"。
- 更糟：771 行通用文案 "int8 quantization of streamed DiT block failed" 在 submit 失败时**覆盖**了真错误，
  早期把根因误导为 int8 量化问题。
- 修复：submit 与 quantize 一起以 `(int8_mlp || int8_qkv || int8_attention_out)` 守卫，
  int8 关时整段跳过，与纯 CPU `pread` 流式行为（及原版流式）一致。

### 验证结果
- 命令：`--ssd-streaming --steps 4 --frames 16 --width 256 --height 256 --use-slower-bf16-mlp --use-slower-bf16-qkv --use-slower-bf16-attention-output`
- 结果：denoise 4/4、video VAE 36/36、FFmpeg 22/22，错误计数 0，产出 256×256 h264+aac 0.92s 视频。
- 注意：本次验证走的是 **BF16 关 int8** 路径；默认 int8 开启的流式路径（streamed-block int8 量化）尚未单独验证，见 task_plan P4。
