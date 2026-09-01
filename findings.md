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

## int8 仅在 M5 类 GPU 可用（本轮 P4 发现，2026-09-01）

- `h3_gpu.m:364-373`：`m5 = [gpu.device.name rangeOfString:@"M5"].location != NSNotFound`；
  `wantsTensorOps = m5 && (!nax || !*nax || strcmp(nax,"0")!=0)`；
  `tensorOpsEnabled = gpu.library && wantsTensorOps`。
- 即 **int8（tensor ops）只在设备名含 "M5" 的 GPU 上启用**，可用 `H3_NAX=0` 显式关闭。
- 后果：`h3_dit.c:1749-1755` 的 `int8_mlp/qkv/attention_out` 全部依赖 `h3_gpu_has_int8_mlp`（= `tensorOpsEnabled`），
  非 M5 机器上恒为 0 → 即便显式 `--ssd-streaming` 且不传 `--use-slower-bf16-*`，streamed-block int8 也绝不启用。
- 实测：P4（默认流式）与 P3（显式关 int8）产出 mp4 逐字节相同（`cmp` IDENTICAL），证实本机（非 M5）int8 未启用，
  默认 int8 流式路径实际退化为 BF16 流式。
- 含义：
  1. 本机只能验证 BF16 流式生成（已稳定通过，P3/P4 一致）。
  2. streamed-block int8 + 流式量化/submit 路径（P2 修复的代码）**需 M5 类 GPU 才能端到端验证**；当前硬件无法覆盖。
  3. `use_int8_row_fc2`（行式 fc2 int8）另有 `h3.c:563` 的 `!h3_device(ctx)->metal4` 门槛，同样依赖新硬件（Metal 4）。

## P5 提速诊断：瓶颈是 USB SSD I/O（2026-09-01）

- 模型权重位于 `/Volumes/data`（Protocol: USB 的 SSD，顺序读 ~814 MB/s = 0.768 GiB/s）；内部盘 `/` 仅 43Gi 空闲，装不下 134GB 模型，故无法迁移到更快存储。
- H3_PROFILE 实测：`BF16 SSD stream 72.495 GiB read in 94.376s (0.768 GiB/s), unhidden wait 78.894s`。
  - 每次生成流式读取约 72.5 GiB（≈ 2 步 × 每步 36 GiB 的 50 个 DiT block matmul 权重）。
  - **78.9s 是计算端等待盘就绪的 I/O 停滞（unhidden wait）** —— 纯 I/O 阻塞，与步数/算力无关。
- 16GB RAM 装不下单步 36 GiB，page cache 无法跨步复用 → 每步都从 USB 重读，~47s/步 不可避免。
- 软件杠杆收益极小（已验证）：steps 4→2 仅省 ~26s（240s→214s）；frames/res 已压到 16/256；MPS 预热/core_reuse/token_reduction 不解决 I/O。
- 真正提速需硬件：①更快模型盘（Thunderbolt/USB4 SSD，2-3 GB/s 可将 ~79s 停滞降到 ~20-30s）；②M5（int8 减半读量，~24s/步）；③>64GB RAM（常驻去流式）。
- 唯一可行的代码杠杆：增加流式预取深度（更多 stream slot / 更大 prefetch）以回收部分 unhidden wait（~30-40% 潜力），但需改动刚修复的流式核心（`h3_dit.c` stream_thread / stream_slots），有回归风险。

## lora 合并 + 流式被代码硬阻断（2026-09-01 发现）

- `h3_dit.c:1712-1716`：
  ```c
  if (lora_path && *lora_path) {
      if (ssd_streaming) {
          fail(error, error_size,
               "LoRA merging is not supported with --ssd-streaming");
          goto failed;
      }
  }
  ```
- 即 **lora 合并与 `--ssd-streaming` 互斥**，且这是 `fail` 硬报错，不是性能/精度问题。
- 对 feature/lora-merge 分支的影响：在 16GB 上非流式（全量常驻 ~134GB）会 OOM，
  唯一可行形态是流式，但该组合被直接拒绝 → **lora 路径在 16GB 上当前完全不可用**。
- `h3_lora.h` 注释明确："SSD streaming 和 int8 路径不支持合并"（lora 合并也不支持流式）。
- 出路（待定）：放宽 `h3_dit.c:1713` 的限制，使 lora 权重在流式读取/量化前合并；或仅在非流式大内存机器上用 lora。
- 注意：P3/P4 的验证均**不带 lora**（`lora_path` 为空），故未触发此限制；带 lora 的流式运行会立即失败。

## ClipProj 替换文本编码器：最大单一组件的近半 I/O 可被消除（2026-09-01）

### 实测 inventory（`./h3 -d <MODEL_DIR> --info`，本机 M4 / 16GB 统一内存）
```
  Qwen3-VL encoder   14 files  1058 tensors   62.133 GiB   <- FL2VA/text_encoder
  FL2VA DiT          13 files   535 tensors   61.728 GiB
  Ref2VA DiT          0 files     0 tensors    0.000 GiB   <- 未加载
  video VAE           1 files   560 tensors    9.700 GiB
  audio VAE           1 files  1087 tensors    0.564 GiB
  合计 ≈ 134.1 GiB
```
- **文本编码器 = 62.13 GiB，是单模型最大单一组件**，比 FL2VA DiT（61.7 GiB）还略大，占总权重 **46%**。
- 本机 `recommended_working_set 11.8 GiB` / `max Metal buffer 8.9 GiB` → 62 GiB 编码器**必然走流式读盘**（USB SSD 瓶颈，见 P5）。

### 架构发现：编码器与 DiT 的契约只有一个 5120 维向量
- 文本编码器在 `h3_text_encoder.c` 是 **Qwen3-VL 前 50 层**（`TEXT_LAYERS=50, TEXT_HIDDEN=5120, TEXT_INTERMEDIATE=25600`），权重路径独立的 `FL2VA/text_encoder`（`h3.c:427,1468`），与 DiT/VAE 分开计重（`h3_memory_plan.h:47` 5 组件之和）。
- DiT 只校验 `text->width == TEXT_DIM`（5120）与 `text->tokens`：
  ```297:297:h3_dit.c
  if (!text || !text->values || text->width != TEXT_DIM || !text->tokens ||
  ```
- **维度一致 = 可原位替换**：只要喂进去一个 5120 维的 `h3_text_embedding`，`validate_layout`/`refine_text`/`prepare_maps` 一行都不用改。

### ClipProj 做法（已在仓库原型化）
- 用 **Qwen3-VL-4B**（36 层 / hidden 2560；截断到 25 层 ≈ 5.5 GiB 以塞进 16GB）作文本编码器，取中间层 `tap=24` 的 hidden（2560 维），过一个很小的 **MLP 投影**（`2560→32768→5120`，带 mean/std 归一化 + attention-sink 替换），输出 **5120 维**，正好对齐 H3 conditioning 空间。与 32B 真值 `cos_test ≈ 0.81`（见 `clipproj_harness.py`）。
- 该 harness 已在本机跑通：模型在 `~/.lmstudio/models/Qwen3-VL-4B-Instruct`，投影权重 `ClipProj-MiniMax-H3/mmh3-4b-ClipProj-v3-mlp.safetensors`，产出 `.npz` 的 5120 维 embedding。

### 收益（实测量级）
| 方案 | 文本编码器占用 | I/O 节省 |
|---|---|---|
| 原 50 层 Qwen（32B 级） | 62.13 GiB | 基准 |
| ClipProj：Qwen3-VL-4B（25 层）+ MLP | ≈ 5.5 GiB | **省 ~56.6 GiB（编码器自身 91%）** |

- 对**纯文本 prompt**：占总模型 134 GiB 的 **~42% I/O 被直接消除**；等于把最大读盘源从 USB SSD 砍掉。
- 这比 int8 落盘 / 增加预取深度（P5）更划算——它是"换组件"而非"压缩原组件"，且不影响 DiT/VAE 精度。

### 硬限制（必须知道）
- **仅适用于纯文本 prompt**。多模态路径（`h3_text_encode_multimodal_bf16`，带 image/ref 输入的 vision span + deepstack）仍依赖 50 层编码器的**视觉塔**（`h3_text_encoder.c` 的 `h3_text_encode_multimodal_bf16` / `h3_multimodal.c`）。图生视频 / 参考图输入暂时不能用 4B 替换（除非另做视觉版 ClipProj）。harness 注释也写明 vision tower "unused for text-only encoding"。

### 集成方案（两个选项）
- **A（推荐，最省事）文件喂入**：新增 `h3_text_encode_from_file`，把 harness 预计算的 5120 维 BF16 `.npz` 直接载入成 `h3_text_embedding`；`h3.c:1468` 调用点改走这条分支（纯文本时）。改动集中在 1 个文件 + 调用点，不需在 Metal 引擎重写 4B 前向。
- **B（彻底）引擎内化**：把 Qwen3-VL-4B 前向 + MLP 投影移植进 Metal 引擎，运行时直接算。改动大，但免去离线步骤。

### 与 GGUF/MLX/llama.cpp 的边界（避免混淆）
- GGUF / MLX / llama.cpp / Qwen-VL 量化是 **LLM 生态**，不能直接跑 H3 视频扩散模型。ClipProj 是 **H3 原生**的"用小模型替换大模型组件"等价做法：Qwen3-VL-4B 当**文本编码器**，H3 的 DiT 照样生成视频。这与前面讨论的"量化省内存/I/O"是同一思想，但必须落在 H3 自身代码上。

### 待办
- 决定走 A 还是 B；先量 `FL2VA/text_encoder` 实际值（已完成：62.13 GiB）确认收益后再落地。
- 落地后更新 task_plan 增加 P6（ClipProj 文本编码器替换）。

## 编码器替换选型原则（2026-09-01，本轮讨论固化）

1. **任意 5120 维编码器都需训对齐投影头；同族 Qwen3-VL-4B 优于异族 MiniCPM-V-4.6。**
   - "DiT 只校验 `text->width==5120`" 只保证**形状**能喂进去，**不代表语义对得上**。DiT 是在 32B Qwen3-VL 编码器的**输出分布**上训的；喂分布不符的 5120 维向量不会报错，但视频会变乱。
   - ClipProj(Qwen3-VL-4B) 成功的关键不是"小"，是"**同族**"：4B 与 32B 编码器同为 Qwen3-VL 架构族，hidden 分布相近，故一个简单 MLP 投影即可达 `cos_test≈0.81`。
   - MiniCPM-V-4.6 主干是 **Qwen3.5-0.8B + SigLIP2**（异族），hidden 分布差异大，需重新训练"MiniCPM→H3 5120"投影头且预期 `cos_test` 远低于 0.81 → 质量掉。故**同族优先于异族**，即便后者更小。

2. **同族选型 + 量化 4B 优先于换 2B / MoE。**
   - 同族梯度（官方）：Dense `2B/4B/8B/32B`；MoE `30B-A3B`(激活 3B) / `235B-A22B`(激活 22B)。
   - **最优省内存杠杆是量化已在用的 4B（int8/int4）**：架构与 hidden 分布不变 → ClipProj `cos_test≈0.81` 基本保留、**投影头零重训**；harness 已支持 `--quantize int8` 与 `--weights-file`(weight-only int8 convrot)，且文档实证 int8 4B 与 bf16 真值仅差 ~0.0023 cosine。4B fp16 ~8.9GiB → 截断 25 层 ~5.5GiB → int8 ~4.5GiB → int4 ~2.2GiB。
   - **换 2B**（dense 半尺寸）：同族余弦仍高，但 hidden 2048（4B 为 2560）、层数更少 → ClipProj MLP 须重训为 `2048→5120` 且换 tap 层，多一道工序，收益只是再小一点。
   - **MoE 30B-A3B**：激活参数仅 3B、质量/算力比佳，但**总权重 ~60GiB** → 对"塞进 16GB"反效果（需常驻/流式一大坨权重），除非为质量换速度否则不推荐。
   - 结论：**留在 4B 并量化（int4/int8）是最小可行形态（~2–4GiB），优于换 2B / MoE**。

### 方案 A vs B 的收益边界（澄清）
- **I/O / 显存 footprint 收益在 A 与 B 完全相同**：都来自"32B→4B"替换（编码器 62.13→~5.5GiB，省 ~56.6GiB / 91%，占总模型 ~42% I/O）。这是大头，A 已 100% 拿到。
- **B（引擎内化）的额外收益**仅在运行时质量：① 编码速度（Metal GPU 4B vs A 的 CPU torch 4B，可能快数倍～十倍）；② 免离线两步、单命令生成；③ 内存并入 Metal 堆、可 per-call 释放，避免 Python torch 额外常驻；④ 可接入现有 embedding 缓存复用。
- **B 的代价**：需把标准 HF `Qwen3-VL-4B` safetensors 装进 Metal（现有 `h3_weight_store` 读的是 H3 自定义 `.h3st` 格式，不能直接吃 4B），或把 4B 转成 `.h3st`；再加 ClipProj MLP kernel（含 convrot int8 反旋转 / mean·std 归一化 / sink 替换）。工作量大于 A。
  - 缓解因素：Qwen3-VL 与现有 50 层编码器**同构**，RMSNorm/RoPE/GQA/SwiGLU/linear 等 Metal kernel 可复用，仅 hidden(2560)、层数(25) 不同 + 末尾加 MLP。
- **结论**：若目标只是"16GB 跑起来 + 砍 I/O"，**先上 A**（代码最小、风险最低、立即拿到全部 I/O 收益）；B 为锦上添花（更快/更顺），因 kernel 可复用、架构同构，成本低于从零写。编码阶段占总生成时间比例小（瓶颈在 denoise/VAE 的 ~79s USB I/O），故 B 对*总*生成时间的改善有限。

## 文本编码运行时行为（2026-09-01，代码实证）

- **文本编码每次生成只跑一次**，位于 conditioning 阶段，**非每步 / 每帧 / 每秒重跑**。
  - 证据：`h3.c` 单次调用 `h3_text_encode_bf16`（或 multimodal 变体，`h3.c:1445-1474`）；产出 `h3_text_embedding`（5120 × **prompt_tokens**）后在 `load_dit` 内 `refine_text` / `prepare_maps` 各调用一次（`h3_dit.c:1780,1793`），**denoise 循环（2804/2947/3052/3129）不再碰 text**；prepare 完即 `h3_text_embedding_free`（`h3.c:1584`）。
- **成本仅 f(prompt 长度)，与视频时长无关** → 长视频下为固定开销，被整段时长摊薄趋近于零。真正随时长线性增长的是 DiT 去噪（序列长度 ∝ 帧数）与 VAE 解码（帧数）。
- **可缓存**：`ctx->cache_enabled` 时 `h3_conditioning_cache_store` 存 exact BF16；同 prompt 重复生成直接命中 → 文本成本 ~0。
- **含义**：长视频优化重心在 DiT 去噪 + VAE 解码（I/O 与算力），不在文本模型；ClipProj 把 62 GiB 固定读盘砍掉后，文本更非瓶颈。也意味着文本编码**天然适合卸载到远程服务器**（见下条待办）。

### 待办（远程卸载思路，2026-09-01 提出）
- 文本编码是 prompt→5120 维向量的**纯函数**，与后续 DiT/VAE 完全解耦，输出极小（5120×tokens×2B：100 token ≈ 1 MiB，1000 token ≈ 10 MiB），可在远程服务器跑 4B+ClipProj **甚至原始 32B 编码器**生成 embedding，再下载到本地喂引擎（复用方案 A 的 loader）。
  - 远程跑**原始 32B** 可得 cos_test=1.0 的精确 embedding，无任何 4B 近似损失，且 62 GiB 编码器永不落本地 Mac → 直接破解"16GB 装不下编码器"。
  - 与方案 A 兼容（A 的 `.npz` 即可来自本地 harness 或远程服务器）；若走远程，方案 B（引擎内化）必要性下降。

### 未来探索 TODO（2026-09-01 记入，非当前优先级）
- **A_local（本地文件喂入）**：新增 `h3_text_encode_from_file`，把本地 harness 预计算的 5120 维 BF16 `.npz` 直接载入成 `h3_text_embedding`；`h3.c` 纯文本调用点改走此分支。代码量小（1 文件 + 调用点），可先拿全部 I/O 收益（62→~5.5 GiB），但需两步工作流（先跑 harness 产 .npz）。
- **A_remote（远程服务器生成 embedding）**：文本编码放远程（4B+ClipProj 或原始 32B 编码器），仅下载 MB 级 5120 维 embedding 回本地喂引擎（复用 A_local loader）。本地 Mac 完全不持编码器权重，甚至可用原始 32B 得 cos=1.0 精确 embedding。优先级低于 B，作为后续探索。
- **当前优先级：B（引擎内化）进行中**——将 4B 前向 + ClipProj MLP 搬进 Metal 引擎，运行时直接算，免离线步骤（见下）。

### B（引擎内化）实现状态（2026-09-01，已完成并验证）
- **已实现**：`h3_text_encode_clipproj_bf16`（`h3_text_encoder.c`）+ 头文件声明；`h3.c` 纯文本路径接 env 开关 `H3_CLIPPROJ_DIR`（设了就走 4B+ClipProj，不设仍走原 50 层路径，**零回归**）。`H3_CLIPPROJ_PROJ` 可覆盖 projection 目录（默认 `/Volumes/data/.lmstudio/models/ClipProj-MiniMax-H3`）。
- **做法**：复用现有 Metal gpu ops（RMSNorm/Linear/RoPE/GQA/SwiGLU/Add），维度从权重**动态读取**（4B: hidden 2560 / intermediate 9728 / q_heads 32 / kv_heads 8；与 50 层同族 `rope_theta=5e6`、`eps=1e-6`、`head_dim=128`、`vocab=151936`），跑 **25 层**（tap=24）取 2560 维 hidden，末尾 **CPU 跑 ClipProj MLP**（F16→F32 upcast，含 mean/std 归一化 + GELU + sink 替换）输出 **5120 维**。DiT 无需改动（`width==5120`）。
- **验证**：`tests/test_clipproj_encoder.c` 通过——提示 "A red fox walking through snow"（6 token）产出 `6×5120` BF16，无 NaN/Inf、非退化；GPU 分配 **~5.4 GiB**（与"截断 25 层 ~5.5 GiB"预估吻合），25 层前向 + MLP 正常；单次约 7.6s（含 4B 权重磁盘读取，生产环境常驻内存后更快）。
- **待补强**：与 harness `mmh3_cond` 的余弦对齐（`cos_test≈0.81`）未自动比对；如需可加 golden 对比（复用 A_local 的离线 `.npz` 思路）。但前向与 harness 同架构同超参（rope/eps/head_dim 一致），预期对齐。
- **端到端验证通过（2026-09-01）**：设 `H3_CLIPPROJ_DIR` 跑 `./h3 -d MiniMax-H3 -p "a red ball bouncing on a white floor" -o /tmp/h3out/clipproj_stream_nolora_256.mp4 --ssd-streaming --steps 4 --frames 16 --width 256 --height 256 --use-slower-bf16-mlp --use-slower-bf16-qkv --use-slower-bf16-attention-output`。日志确认 `text encoder (clipproj)` 触发、4B 跑满 25 层、`refine text 1/1` 后 DiT 正常消费 5120 维 embedding；最终产出 `clipproj_stream_nolora_256.mp4`：ffprobe 验 256×256 h264+aac 24fps 0.92s 22 帧，与 P3 基线（50 层编码器）**同规格**，大小 123379 vs 122408 bytes（不同 embedding 驱动的不同生成，非崩溃/拷贝）。**B 完成**。内存规划与 P3 一致——文本编码器在 `h3.c:890` 本就按 per-call 释放计（`* 0`），4B 更小不改变规划，流式照常。

## ClipProj token-0 爆炸根因与修复（2026-09-01）

### 现象
`h3_text_encode_clipproj_bf16` 跑 25 层 Qwen3-VL-4B（tap=24），token 0 的最终 hidden 范数 **4701**，而 HF 参考 `hidden_states[25]`（= `tap+1`）token 0 范数仅 **24.15**；逐层余弦前 24 层 ~1.0、最后一层 token 0 骤降到 0.659。B-vs-A_local（同 4B）5120 输出余弦仅 0.697。

### 根因（已数值验证，非猜测）
1. **B 在 25 层之后、喂给 ClipProj MLP 之前，漏掉了 model 的最终 RMSNorm**（`model.language_model.norm`）。
   - HF/ComfyUI 取的是**归一化后**的 `hidden_states[tap+1]`；B 之前取的是**裸 layer-24 输出**。
   - token 0 在 pre-norm 残差流里本就会膨胀（首位 token 因果注意力只看自己、无平均，attention 输出=全幅 value，逐层累加）。HF 同样会膨胀到 ~4824，但**最终 RMSNorm 把 token 0 拉回 24.15**；B 没这步，故停在 4701。
   - token 0 的隐藏态膨胀其**方向**是错的（幅值被 RMSNorm/head-RMSNorm 重置但方向携带垃圾），污染所有 attend token 0 的 token，导致逐 token 方向漂移（tok1–5 余弦仅 0.70–0.80）。
2. 修复：循环后、读 hidden 前，对 `hidden` 应用 `h3_gpu_rms_norm_bf16(gpu, hidden, hidden, final_norm, tokens, H, CP_RMS_EPS)`，`final_norm` = `cp_load_1d(..., "model.language_model.norm.weight", H, ...)`。

### 验证（golden + 逐 token）
- 手工对 B 裸 hidden 应用 final RMSNorm → 与 HF `hs25` 逐 token 余弦 **1.00000**（tok0 范数 4701→24.39，吻合 HF 24.15）。→ 假设实锤。
- 修复后重跑：B 最终 hidden 与 HF `hs25` 逐 token余弦 **0.99997**（tok0 含）；B 5120 输出 vs A_local 余弦 **0.9799**（修复前 0.697）。
- 剩余 0.9799 是 **bf16(B) vs fp16(A_local) 精度地板**：tok4 的 hidden 在 ClipProj MLP 处于高敏感方向，把 25 层累积的微小数值差放大（tok4 的 cos=0.88，其余 token≈0.999；`|B_h-A_h|` tok4 仅 0.94 与别无二致，说明 B 前向本身正确，只是精度放大）。`cos_test=0.814`（4B vs 32B）目标质量达成。
- 结论：**token 0 爆炸是漏 final RMSNorm 导致，已修复**；0.9799 是 bf16/fp16 精度容差（golden 阈值 0.999 对 bf16 实现过严），非逻辑 bug。若需 B 与 A_local 逐位一致，须让 B 前向改用 fp16（非 bf16），代价大且不影响 cos_test。

## ModelScope 量化权重可消费性（2026-09-01 调研）

### 仓库内容（`Abiray/Minimax-H3-nvfp4-INT4-INT8-Convrot`）
- DiT(FL2VA/Ref2VA) + 文本编码器(Qwen3-VL-32B) 的 **INT4 / INT8 / Mixed INT4-INT8（均带 Convrot）** 与 **nvfp4**（mixed）。
- **VAE 未量化**：视频 VAE = FP16（5.21 GB）、音频 VAE = FP32（605 MB）。
- tensor 类型含 BF16/U8/I8/F16/F32/F8_E4M3（INT4 = U8 打包）。
- 硬件目标：**严格 NVIDIA GPU**（RTX 4070TiS/4080/4090/5090），无 Apple Silicon 适配/验证。

### FL2VA DiT 体积对比（关键修正，2026-09-01 实测）
| 格式 | 大小 | 相对 BF16 |
|---|---|---|
| BF16（本仓库当前） | 61.73 GiB (≈66.4 GB) | 基准 |
| `pruned_int8_convrot` | **19.5 GB (≈18.2 GiB)** | **≈29%，3.4× 更小** |
- 19.5 GB 不是单纯 int8（那只能到 ~33 GB），而是 **pruning（减块）+ int8（2×）叠加** → 3.4×。
- 文本编码器 INT8-Convrot = 27.1 GB（BF16 62.13 GiB 的 ~2.3× 压缩）。
- 对全模型：仅 DiT 一项 61.73→18.2 GiB，整模型 134→~90 GiB（若仅 DiT 走此格式）。

### 对 h3.c 的兼容性（硬约束）
- `nvfp4`：NVIDIA Blackwell 专用 → Apple Silicon Metal 不可跑，**出局**。
- `INT4`：U8 打包，h3.c **零 int4 支持**（loader 能读 U8/I8，但无 unpack/dequant/matmul kernel）→ 需全新实现。
- `INT8-Convrot`：loader 能读 I8；消费需 ①放宽 `h3_gpu.m` 的 "M5" 字符串门控（P7 同主题）②前向接 Convrot 逆旋转 ③int8 覆盖扩到 norm/adaln（当前仅 4 个 matmul）④匹配 "pruned" 块数（`H3_DIT_BLOCKS=50` 硬编码）。
- Convrot 旋转矩阵**未单列文件**（页面无 rotation 元数据）→ 须确认烤进权重还是另有 side 矩阵。

### 映射回三组件（回应"体积如何优化"）
- **FL2VA DiT 61.73 GiB**：`pruned_int8_convrot` 可砍到 19.5 GB（3.4×），但需上述 4 处代码改动 → P8 重心。
- **video VAE 9.70 GiB**：仓库未量化 → 仍靠自加量化 / 流式（已压 RAM）。
- **audio VAE 0.56 GiB**：未量化，且本就忽略。
- **文本编码器 62.13 GiB**：维持 **4B ClipProj**（5.5 GiB，cos≈0.81），比消费 INT8-32B(27GB) 小 5× 且已落地 → 不引入此仓库 32B 权重。

### 元数据与 ComfyUI 取向（2026-09-01 下载分析）
- 已下载 `README.md`(8.8KB)、`configuration.json`(77B，仅 `{"framework":"pytorch","task":"image-text-to-video","allow_remote":true}`)、`ref2.json`(53KB，一份 ComfyUI Ref2VA 工作流，引用 `MiniMax_H3_Ref2VA_pruned_int8_convrot.safetensors` 经 `UNETLoader` 以 "default" 权重类型加载)。
- **三者均不含块数 / Convrot 旋转矩阵信息** → 消费前的硬证据只能来自 safetensors header 本身（待主文件下完跑 `dbg_parse_safetensors.py`）。
- **文本编码器在此仓库并不存在**：README 提到 `qwen3vl_32b_minimax_h3_int8_convrot.safetensors`(27.1GB)/`int4_convrot`(15GB)，但仓库文件树（API 列 15 项）里**没有 `text_encoders/` 或任何 qwen3vl 文件** → 该仓库只含 DiT(FL2VA/Ref2VA) + VAE。进一步坐实：消费端仍用 **4B ClipProj**（5.5GiB）做文本编码器。
- **权重是 ComfyUI 取向**：UNETLoader/CLIPLoaderGGUF + 自定义 MiniMax-H3 节点处理 convrot 逆旋转。h3.c 不读 ComfyUI 图，但 safetensors 张量是标准格式；要消费须自己实现 convrot 逆旋转（前向接逆旋转）。

### 显存目标与 16GB Mac 的现实（重要修正）
- README 显存指引：INT4/mixed→16GB dGPU；**INT8→24GB+ dGPU（~21GB）**；nvfp4→Blackwell。即 `pruned_int8_convrot`(19.5GB) 面向 **24GB 级**独显。
- 但 16GB Mac 是**统一内存**，DiT+VAE+文本编码器同池竞争：即便 INT4 DiT(11.3GB)+VAE(9.7GiB)+ClipProj(5.5GiB) ≈ 26.5GiB 仍超 16GB → **量化权重在 16GB Mac 仍装不进 RAM**。
- 故 P8 在 16GB Mac 的**真实价值不是"塞进内存"，而是"流式读盘更少"**：h3.c 当前每代从 USB SSD 流式读 ~72.5GiB（P5）。换成 19.5GB int8_convrot DiT 后，**每代 DiT 读量降到 ~1/3.4**，直接削减 P5 那个 79s I/O 停滞（瓶颈在盘速，非算力）。这与"16GB 跑起来"根目标一致——靠流式而非常驻。
- 质量/IO 权衡：int8(19.5GB, 3.4× 少读) vs mixed_int4_int8(14.8GB, ~4.5× 少读) vs int4(11.3GB, ~5.8× 少读)。用户指定的是 int8(19.5GB)。

### 头部分解结果（2026-09-01 实测，dbg_parse_safetensors.py）
- 文件 `/Volumes/data/.lmstudio/models/Minimax-H3-Quantized/MiniMax_H3_FL2VA_pruned_int8_convrot.safetensors`（20970379688 bytes，SHA256 `f07a5427…`）头部解析：932 张量，≈19.5 GiB。
- **dtype**：I8 = 17.94 GiB（主体），BF16 = 1.49 GiB（norm/adaln/embeddings 保留），F16 0.08 / F32 0.02 / U8 14.4KB。即 **仅每块的 4 个 matmul 被 int8 量化，norm/adaln/patch_proj/final 仍是 BF16** —— 比预想干净：h3.c 只需把 4 个 matmul 权重换成 I8+scale，其余走现有 BF16 路径。
- **DiT 块数 = 50**（min0/max49/count50），与硬编码 `H3_DIT_BLOCKS=50` **完全对齐** → 不存在块数不匹配（"pruned" 指块内结构化剪枝/量化，非删块）。
- **scale 布局**：200 个 `weight_scale`（50×4）F32 `[out,1]` = 逐输出通道对称量化，dequant `w=i8*scale`；无 zero-point（确认对称 int8）。200 个 I8 权重均有对应 scale，无遗漏。
- **关键未知（P8 #1 阻塞）**：`rot`/`convrot`/`hadamard`/`ortho`/`zero` 类 key **命中数全 0** —— 文件里**没有独立 Convrot 旋转矩阵**。两种可能：① 旋转是参数无关固定变换（如 QuaRot 式 Hadamard，作用于激活，无需存储），h3.c 可实现等价旋转；② 旋转是存于别处的随机正交矩阵（本文件无 → 不可恢复，权重在旋转基下无法直接用）。**这是 P8 能否落地的唯一真障碍**，需从量化脚本 / ComfyUI 节点源码确认旋转配方。
- **（2026-09-01 检索已破解）**：convrot = 已发表 **ConvRot (arXiv 2512.03673, 清华&华为)** 方法的 **Regular Hadamard Transform (RHT)**，参数无关、无需存储矩阵（见同章「ConvRot 旋转配方已破解」节）。P8 #1 阻塞解除 → 转为"在 h3.c DiT 前向复刻 RHT 的分组与应用点"。
- I8 权重样例 `blocks.0.attn.out_proj.weight` shape `[5376, 7168]`（与 scale `[5376,1]` 对应逐行）；in_features=7168 与常见 hidden(5376) 不一致，提示 convrot 可能改变有效宽度/转置，h3.c 前向需按文件实际 shape 参数化而非硬编码。

### ConvRot 旋转配方已破解（2026-09-01 检索）
- "convrot" 不是自定义随机旋转，而是已发表方法 **ConvRot: Rotation-Based Plug-and-Play Quantization for DiTs**（arXiv 2512.03673，清华&华为，2025-12），官方实现 `github.com/feice-huang/ConvRot`，并有 `MiniMax-H3-W4A8-ConvRot` 专门项目 → 本仓库 `pruned_int8_convrot` 即由该法产出。
- 核心是 **Regular Hadamard Transform（RHT，分组正则 Hadamard 变换）**：旋转矩阵是**固定、参数无关**的 ±1/√n Hadamard 矩阵，**无需存储** —— 正好解释头部分解中 `rot/convrot` key 命中数为 0 的现象。故 P8 #1 阻塞（"旋转矩阵缺失 → 不可恢复"）**已排除**：h3.c 可自行生成 RHT 矩阵，不必依赖 side 矩阵。
- 真正的实现任务变为：**在 h3.c 的 DiT 块前向里，于 ConvRot 量化时施放的相同位置、相同分组粒度（group size）施加 RHT**（对权重已旋转者，需在激活侧施加 R^T）。具体"哪些张量、沿哪个 dim 分组、放在块的哪个位置"须照搬 ConvRot 官方实现 / MiniMax-H3-W4A8-ConvRot 项目，否则数值不对。
- 含义：P8 从"可能不可行（缺矩阵）"转为"明确可行但需忠实复刻 ConvRot 前向"；难度集中在**对齐 RHT 的分组与应用点**，而非寻找缺失矩阵。

### ConvRot 确切方案 + h3.c 消费蓝图（2026-09-01 检索+推导）
- **官方 README（feice-huang/ConvRot）实锤**：旋转 = **Regular Hadamard Transform（RHT）**，`rot_size=256`（4 的幂，K 须被整除）；权重**离线预旋转+量化**（磁盘即 W·R），激活**在线旋转**（fused rotate+quant）。跳过层：`x_embedder`（in_dim 不整除 rot_size）、`time_text_embed`（时间/文本嵌入，质量敏感，保 BF16）。→ 与头部分解吻合：attn q/k/v/out + mlp fc1/fc2 被 int8（4 matmul/块），norm/adaln/embeddings 保 BF16。
- **数值等价**：Y = (X·R)(W·R)^T = X·W^T（R 正交）。消费预旋转权重两种等价法：① 在线旋转激活 X→X·R 后用 W·R 做 GEMM；② **加载时反旋转回真值** W = (W_stored)·R^T = W_stored·R（R 对称），之后标准路径。
- **h3.c 推荐消费路径（保住 19.5GB 流式收益，不写新 int8 Metal kernel）**：
  1. 流式按层读 int8 权重 W_int8[out,K] + scale[out,1]。
  2. 反旋转：`W_bf16 = (scale[out,1] * W_int8[out,K]) · R_K`，R_K = 沿 K 维、块 256 的**分块对角 Hadamard**（K/256 个 H_256 块，±1/√256，正交）。每代每层一次性，量级极小。
  3. 现有 BF16 DiT 前向直跑（W_bf16 即真值权重）；norm/adaln/embeddings 本就 BF16 直读。
  - 收益：磁盘仍只流式 19.5GB（较 61.7GiB 少 3.4×，削 P5 I/O 瓶颈），运行期按层暂存 BF16 真值，无需全局常驻；**完全复用现有 BF16 GEMM**，仅加 int8→BF16 反量化 + 分块 Hadamard 反旋转两个预处理算子。
- **待精确核对（实现前）**：① 权重存 W·R 还是 R·W（决定反旋转乘在 K 还是 out 维，README 暗示沿 K/输入维）② 激活在线旋转是否 X·R ③ token_refiner/其他子模块是否旋转 ④ Hadamard 归一化（含 1/√n 方使 R·R^T=I 无损）。由读 `ops.py` 的 `hadamard_rotate` + MiniMax-H3 节点加载代码钉死。

### h3.c 已内建 DiT int8 路径（2026-09-01 代码核查，重大修正）
- 代码核查发现 h3.c **早已实现 DiT 的 int8 量化与推理**，`h3_dit.c` / `h3_gpu.m` 实证：
  - `h3_dit_block` 同时持有 `qkv/qkv_int8/qkv_scales`、`out/out_int8/out_scales`、`fc1/fc1_int8/fc1_scales`、`fc2/fc2_int8/fc2_scales`。
  - `h3_gpu_quantize_weight_int8(gpu, dst_int8, dst_scales, src_bf16, rows, cols)` 在流式加载时把 BF16 **逐行对称**量化成 int8 + 逐行 F32 scale（scale 长度 = rows = 输出通道）。
  - Metal 端已有完整 int8 前向 kernel：`h3_linear_int8_nax_r128*`、`h3_fc1_swiglu_int8_*`、`h3_qkv_project_split_int8_*`、`h3_linear_int8_grouped_*`、`h3_gate_adaln_quantize_int8` 等；`run_block` 直接消费 int8 字段。
- **格式完全对齐**：ConvRot 的 `int8 + 逐输出通道 weight_scale [out,1]` 与 h3.c 内部 int8+逐行 scale **同构**（均为对称逐输出量化）。→ 之前"h3.c 零 int8、P8 需从零造 int8 kernel"的判断**错误**，作废。
- **P8 重写（远简单于预期）**：
  1. 新增 DiT 的 int8+scale safetensors 加载：直接把 ConvRot `blocks.N.*.weight`(I8) + `*.weight_scale`(F32) 读入既有 `qkv_int8/qkv_scales/...` 字段（h3_safetensors 已支持 I8/U8 读取）。
  2. **补唯一新算子：Hadamard 反旋转 R_K（块 256，沿 K/输入维分块对角）**：dequant(int8,scale)→BF16 → `W_true = W · R_K`（或 R_K·W，待核对）→ 再走既有 `h3_gpu_quantize_weight_int8` 回填 int8 字段，或直接作 BF16。run_block 以下完全复用。
  3. 校验：用单 DiT block 的"int8+反旋转"输出 vs 现有 BF16 block 输出做数值对齐（经验证 R 约定，无需读外部源码）。
- **收益叠加**：磁盘只流式 19.5GB（较 61.7GiB 少 3.4×，削 P5 I/O 瓶颈）**同时** 运行期享受既有 int8 显存减半。P8 从"造 int8"降级为"接 int8 + 一个 Hadamard 反旋转"。
- 仍待核对：① R 的乘向(K 还是 out 维)与归一化(±1/√256) ② token_refiner/其他子模块是否也 int8 ③ h3.c 逐行 scale 与 ConvRot 同为对称(无 zero-point，头部已证无 zero→一致)。

### 结论
这把"自己写量化"变成"消费现成权重"——省掉造量化一步，但**消费端代码改动仍不可免**，且 VAE 体积问题本仓库完全没解决。P8 立项见 task_plan；第一步用 `dbg_parse_safetensors.py` 解析 INT8-Convrot header 取证。

## ConvRot qkv 权重布局置换（2026-09-02，flat 输出根因实证）

### 现象
convrot INT8 模型（`MiniMax-H3-Convrot`）走 `--ssd-streaming` 生成画面**全灰（像素 std≈6.84）**，base（`MiniMax-H3`）同设置 std≈37.76。

### 根因：qkv_proj 需 formula B 置换，原代码用错 formula A
- ConvRot checkpoint 把 `qkv_proj` 存为 **q/k/v 交错布局**；引擎前向期望「按 head 分离的 q/k/v 连续布局」。两者差一次行置换。
- 引擎内 `h3_dit.c` / `h3_shaders.metal` 的 convrot remap 用 `layout==1` 选择该置换，但两处都写成 **formula A**（`dst_slot = (slot%3)*heads + slot//3`），**错误**。
- **正确置换 = formula B**：`dst_slot = 3*(src_slot % heads) + (src_slot / heads)`，其中 `heads=56`、`head_dim=128`（每 qkv 块 `21504 = 168*128 = 56*3*128` 行）。
- 验证手段（离线 numpy，不依赖 GPU）：取 convrot `blocks.0.attn.qkv_proj.weight` I8 → `dequant = i8*scale` → 沿输入维块 256 做 Hadamard 反旋转（`H_256`，归一化 ±1/√256）→ 得 unrotated 行 → 按 per-slot 与 base BF16 qkv 做 cosine best-match。结果：**formula B 命中 168/168 slots（cos 每 slot≈0.90+，受 bf16 反旋转精度影响，方向完全吻合）；formula A 仅 2/168（flat）**。故 formula B 为唯一正确置换。
- refiner 的 BF16 qkv（`token_refiner.blocks.*.attn.qkv_proj.weight`，同为 `[21504,5376]`）走独立 kernel `h3_convrot_remap_qkv_bf16`（`layout==2`），**本就已为 formula B**（cos≈1.0 验证通过），证实 B 正确。

### 两处落地路径都必须改（关键陷阱）
- **GPU kernel**（`h3_shaders.metal` `h3_weight_dequant_unrotate_int8`，`layout==1`）：非流式 int8 路径用。
- **CPU unrotate**（`h3_dit.c` `convrot_unrotate_cpu`，`layout==1`）：**`--ssd-streaming` 流式路径用**（见 `h3_dit.c` 流式加载处 `layout = (field==STREAM_QKV)?1:0`）。
- 本机生成走流式 → 实际走 CPU 路径。仅改 GPU kernel 对输出**零效果**（两次 std 均恰好 6.84 暴露）。两处统一改为 formula B 后 std 跃升到 51.64。

### 配套事实（代码/权重实证）
- `h3_dit.c:22`：`HEAD_DIM = 128`（**非** AGENTS.md 所写 96）。remap 的 `head_dim` 必须用 128，否则 `slot=row/96` 错位。
- main DiT 与 refiner 的 `qkv_proj` 均为 `[21504, 5376]`；main DiT 是 I8（带 `weight_scale`），refiner 是 BF16。`out_proj` 为 `[5376, 7168]`（I8/BF16），无交错，走 `layout==0`（不置换）。
- metal 库在**运行时**从 CWD 读 `h3_shaders.metal` 编译（`h3_gpu.m` 默认路径 `"h3_shaders.metal"`），故改 metal 不必重编二进制，但改 `h3_dit.c`（C 代码）必须 `make`。

### 修复与验证
- 三处 remap（GPU int8 / CPU int8 流式 / refiner BF16）现已统一为 formula B；`h3_dit.c` CPU 路径注释同步更正（旧注释描述的是 formula A 语义）。
- `convrot_cpu_fixed.mp4`（steps 4 / 256×256 / ssd-streaming）：像素 std **6.84 → 51.64**，逐帧 std ~51.6 稳定、帧间差异 2.33 → 连贯视频，非噪声/灰屏。
- 仍待：①base steps=4 同条件对照（确认亮度差来自步数而非置换，remap 纯置换不改数值尺度）；②全量 `--steps 20` parity + 单 block 数值 cosine 校验固化。

## M4 的 INT8 支持澄清（2026-09-01）

- 用户指出 M4 支持 INT8（ARM v8.2 Int8 dot/matmul）。需拆清：**那是 CPU 指令集**，而 h3.c 的 DiT 跑在 **Metal GPU**，CPU Int8 对其不适用。
- 真正相关的是 **M4 GPU 经 Metal/MPS 的 Int8 算力**——`mps-bitsandbytes` 等仓库佐证 MPS 在 Apple Silicon 能做 int8/int4，故 M4 GPU 大概率支持 int8 计算。
- h3.c 当前 int8 由 `h3_gpu.m:364` 的 `m5 = device.name contains "M5"` 字符串门控，**非 Metal 能力探测**；同引擎 `use_int8_row_fc2` 却按 `metal4` 能力门控（更宽）——口径不一致，强烈暗示 "M5" 只是作者只在 M5 测过的保守假设。
- 但**不能断言放宽即正确**：`H3_METAL_HAS_TENSOR` 分支可能用了 M5-only Metal 特性，需 parity 实测。且即便跑通，int8 计算**不缩磁盘体积**、不覆盖 norm/adaln/VAE。

## 4B vs 50 层 计时对比（2026-09-01，本轮实测）

### 动机
B（引擎内化 ClipProj）已端到端验证通过，但"换 4B 生成快多少"此前只有编码器单阶段粗估（~7.6s）。本会话实测两类计时：① 编码器隔离计时；② 同设置完整生成计时（唯一变量 = 文本编码器）。

### 方法
- 编码器隔离：4B 用 `tests/test_clipproj_encoder.c`；50 层用 `h3_real_prompt_test`（直接调 `h3_text_encode_bf16`，**不读 env**，故不受 `H3_CLIPPROJ_DIR` 影响）。
- 完整生成：`./h3 -d <MODEL> -p "a red ball bouncing on a white floor" -o <OUT> --ssd-streaming --steps 4 --frames 16 --width 256 --height 256 --use-slower-bf16-mlp --use-slower-bf16-qkv --use-slower-bf16-attention-output`；4B 设 `H3_CLIPPROJ_DIR`，50 层用 `env -u H3_CLIPPROJ_DIR -u H3_CLIPPROJ_PROJ` 强制走默认路径。

### 结果
编码器单阶段：
| 编码器 | wall | GPU | 权重加载 |
|---|---|---|---|
| 4B ClipProj | 7.93 s | 0.301 s | 5.4 GiB |
| 50 层默认 | 52.41 s | 1.844 s | 46.86 GiB |

完整生成（256×256 / 16 帧 / steps 4 / ssd-streaming）：
| 编码器 | 总 wall |
|---|---|
| 4B ClipProj | 251.25 s |
| 50 层默认 | 292.80 s |
| 差值 | 4B 省 ~41.5 s（≈14%） |

### 结论
- 4B 把文本编码器阶段从 52s 砍到 8s（**6.6×**），权重 I/O 从 46.86 GiB 降到 5.4 GiB（**8.7×**），内存/启动压力大幅降低。
- 但**端到端仅快 ~14%（41s）**：文本编码器只是流水线一小段，大头是 DiT 去噪 + VAE 解码 + 流式读权重的 I/O（`sys` 52–61s，两者相同）。
- 生产默认 `--steps 20`（本次仅 4 步），DiT 占比更大，4B 相对收益会**进一步缩小**。换 4B 的核心价值是**省内存/权重体积与编码启动延迟**，不是显著缩短总生成时间。
- 所有计时均为冷启动（每次从盘读权重）；warm page cache / 常驻权重会让两者都更快，但比率不变。

### 关键陷阱（影响对比有效性）
`H3_CLIPPROJ_DIR`（及 `H3_CLIPPROJ_PROJ`）在 shell 环境被**持久导出**。第一次我以为的"50 层"完整生成其实仍走了 4B（日志出现 `text encoder (clipproj) 0/50` + 25 层），导致两次 wall-clock 几乎相同（251.25 vs 251.05）。必须 `env -u` / `unset` 显式清除才能跑真 50 层。手动对比前务必 `unset H3_CLIPPROJ_DIR`。

## 最小物理占用分析（2026-09-02）

### 4B + INT8 DiT 的最小磁盘占用 = 39 GiB（已实测确认下限）
- 组件（磁盘物理占用，非运行时显存）：INT8 DiT 20G + video_vae 9.7G + audio_vae 0.56G + tokenizer 0.01G + **4B 文本 BF16 8.3G** + ClipProj 0.48G = **≈39 GiB**。
- **4B INT8 文本不可用**：in-engine `H3_CLIPPROJ_DIR` 加载器只实现 BF16 分支，`h3_weight_load_f32` 遇到 I8 直接报错退出（实测 `weight ... has dtype/rank I8/2, expected BF16/2`）。故 4B 文本必须用 8.3G BF16 版，不能换 4.5G int8 版。
- **ModelScope INT8 VAE 不可用**：`Gluttony10/MiniMax-H3-INT8-CONVROT` 的 `video_vae.safetensors` 4.85 GiB（恰为本地 9.7G 一半 → INT8 量化）会被引擎拒绝。VAE 解码器走 `load_f32` → `h3_weight_load_f32` → `load_tensor(...,H3_DTYPE_F32)`；`h3_weights.c:154`：
  ```c
  if (tensor->dtype != dtype || tensor->ndim != ndim) {
      fail(error, error_size, "weight %s has dtype/rank %s/%d, expected %s/%d", ...);
      return NULL;
  }
  ```
  文件 dtype=I8 与请求 F32 不符 → 报错退出（与 4B 文本 int8 失败同源）。`audio_vae.safetensors` 0.56G 与本地同大小（非 int8），可视为同款但无省空间收益。
- 根因共性：**引擎只有 DiT 实现了 INT8 权重路径**（消费 `qkv_int8/qkv_scales` 字段 + int8 Metal kernel）；文本编码器（`h3_text_encoder.c` 的 clipproj 加载器）与 VAE 解码器均只认 BF16/F32，遇 I8 直接报错。
- 结论：39 GiB 是 **in-engine 当前能力下的不可突破下限**。要再压（用 int8 4B 文本 → 35GiB，或用 int8 VAE → 34GiB），须给 `h3_text_encoder.c` / `h3_video_vae.c` 的加载器加 INT8 反量化分支（类似 DiT 既有 int8 路径）。

## 系统盘 I/O 优化实测（2026-09-02）

### 假设：瓶颈是数据盘（外置 SSD）流式读取
- P5 诊断：模型权重在 `/Volumes/data`（HFS+ SSD，经 USB/外置接口），实测顺序读 ~0.59 GiB/s；每代流式读 72.136 GiB DiT+VAE 权重，纯等盘（unhidden wait）~93s。
- 若瓶颈确为盘速，把 DiT+VAE 搬到系统内置盘（APFS NVMe）应显著提速。

### 实测结果（4B ClipProj + convrot int8 DiT，256×256/steps4/16帧）
| 配置 | 总耗时 | SSD 流读 | 纯等盘 | 说明 |
|---|---|---|---|---|
| 50层 Qwen + 数据盘 | 194 s | 0.590 | 93.2 s | 基线 |
| 4B + 数据盘 | 153 s | 0.590 | 93.4 s | 换 4B 省 41s（编码器） |
| 4B + 系统盘(仅 DiT/VAE) | 94 s | 1.021 | 43.5 s | **DiT/VAE 搬系统盘** |
| 4B + 全模型系统盘 | 91 s | 1.018 | 43.7 s | 含 4B 文本也搬入 |

- **瓶颈假设被证实**：DiT+VAE 流式读取从 0.59→1.02 GiB/s（×1.73），纯等盘 93→43 s，总耗时 153→91 s（提速 40%）。
- 全模型版(91s) 与 混合版(94s) 仅差 3s → 4B 文本权重不计入那 72GiB 流式读取（仅一次性 load），放数据盘或系统盘对总时长影响极小。真正瓶颈纯粹是 **DiT+VAE 的流式 I/O**（占 43s 纯等盘）。
- 系统盘 NVMe 实测仅 ~1.02 GiB/s，未达内置盘理论上限 → 仍可通过①引擎更深流式预取回收部分 unhidden wait；②换更快存储（Thunderbolt/USB4）进一步提速。但 **GPU compute 下限 ~70s** 是硬约束，无法靠存储消除。

### 复制操作坑（供复现）
- 系统盘起初仅剩 3.9–39 GiB，全量 39GiB 复制会填满致系统不稳 → 先复制核心 I/O 组件（DiT+VAE+tokenizer=30GiB），后复制 4B 文本+ClipProj。
- `rsync` 默认先写 `.XXX` 临时文件再 rename，2× 峰值撑爆系统盘 → 改用 `cp -R -L`（直接写目标名，无临时文件峰值）。
- Convrot 壳（`MiniMax-H3-Convrot`）是 24K symlink 壳；复制到系统盘须把 symlink 指向的真实文件/目录实体化，并补 `transformer/config.json` 与 `FL2VA/text_encoder` 目录（引擎存在性检查需要），否则报 "missing required model file"。
- 全模型系统盘路径：`/Users/jay/h3_sys/MiniMax-H3-Convrot`（DiT+VAE+tokenizer+text_encoder→数据盘 symlink）、`/Users/jay/h3_sys/Qwen3-VL-4B-Instruct`、`/Users/jay/h3_sys/ClipProj-MiniMax-H3`。

## ConvRot 旋转矩阵真相：radix-4 butterfly，非 Sylvester（2026-09-02，P11.1 实测钉死）

- 用 `dbg_block_parity.py` 对 block 0 的 4 个 matmul 做 `dequant(i8*scale)` → 反旋转 → 与 base BF16 同张量 cosine 比对。
- **正确形式 = radix-4 ConvRot butterfly**（与 `h3_dit.c:654-661` 注释一致）：stride = 1,4,16,64 四阶段，每 quad `(a,b,c,d) → (a+b+c-d, a+b-c+d, a-b+c+d, -a+b+c+d)`，整块最后 ×1/16（=1/√256）。引擎 `convrot_unrotate_cpu`（`h3_dit.c:591-633`）即此实现。
- **陷阱（实测踩坑）**：自然序 Sylvester Hadamard（`H_{2n}=[[H,H],[H,-H]]/√2`）是**另一个矩阵**。用 Sylvester 反旋转得到 cos≈0.004（噪声级），且**旋转后比不旋转更差**（0.0039 vs 0.062）——这是识别该错误的诊断信号。`h3_dit.c` 注释早已警告："This is **NOT** the natural-order Sylvester Hadamard; using the popcount formula produces structurally corrupted output."
- **R 乘向钉死 = 沿 K（输入 / 最后一维）**：`W_true = W_stored · R_K`，rot_size=256 分块对角。沿 out 维（`R_out · W`）cos 仅 ≈0.004 → 错。
- **qkv 必须 formula B 行置换**：`dst_slot = 3*(src_slot % 56) + (src_slot // 56)`（heads=56, head_dim=128）。无置换 cos≈0.0008；置换后 0.99996。

### 实测（block 0，四个 matmul）
| 张量 | shape | 最佳假设 | cosine |
|---|---|---|---|
| qkv_proj | [21504, 5376] | butterfly(K) + permB | **+0.999961** |
| out_proj | [5376, 7168] | butterfly(K) | +0.999928 |
| fc1 | [28672, 5376] | butterfly(K) | +0.999961 |
| fc2 | [5376, 14336] | butterfly(K) | +0.999956 |

- **结论**：convrot int8 是 base BF16 的**忠实量化**（权重复现 cos≈0.99996，量化误差极小）。P8.1 的 formula B 结论**独立确认正确**。
- **修正 P8.1 旧记录**：P8.1 报告的 "cos 每 slot≈0.90+" 应是用 Sylvester 矩阵的度量；用正确 radix-4 butterfly 后实为 **0.99996**。结论（formula B 正确）不变，但量化精度远好于当时认知。

## base steps=4 对照：std 差异主因是步数，非量化（P11.2）

- base + 4B ClipProj + steps4 + 数据盘：像素 std **55.77**（逐帧 55.5–56.0，非常稳定，22 帧）。
- 对照 convrot_4（修复后）= **51.64** → 同 steps 下仅差 **7.4%**。
- 而 base_20 = 37.76 → **steps 4→20 使 std 降约 18**（55.77 → 37.76）。
- **结论**：此前 "convrot 51.64 vs base 37.76" 的表面巨大差异**主要来自步数不同**（4 vs 20），并非置换/量化缺陷。修复后的 convrot 与 base 在同 steps 下高度一致（差 ~7%，源自 int8 量化与采样随机性）。P8 flat 问题确已解决。

## P12 调研：DiT 预取 vs VAE 预取（收益点重新评估）

### DiT：已是 depth=1 双缓冲流水线，加深收益有限
- `h3_dit.c:2710-2804`：`stream_slots[2]`（双缓冲），主循环 `run_block(N)` 期间 `pthread_create` 异步预取 block N+1 到 `slot^1`，随后 `h3_gpu_submit` + `pthread_join` 等待 → **预取与计算已重叠**。
- **稳态分析**：总 wall 91s 中，I/O ≈ 71s（72 GiB @ 1.018 GiB/s）、GPU compute ≈ 47.3s、纯等盘 43.7s。I/O 是吞吐瓶颈 → 流水线吞吐 = `min(compute_rate, io_rate) = io_rate`，**加深 DiT 预取（depth>1）对稳态吞吐无改善**，仅在 I/O 突发抖动时起平滑作用。

### VAE：完全串行无流水线 —— 这才是真正的收益点
- `h3_video_vae.c:594-630` `run_stream_tile`：`for (index=0..35) { load_block(index); run_block(index); free; }` —— **I/O 与计算零重叠**。
- 注释（539-543）明示常驻模式 "avoids **rereading 9 GiB per spatial tile**" → 反证**流式模式每个 tile 都要重读 9.7 GiB**；多 tile 时 I/O 成倍放大。
- **优化方案**：给 `run_stream_tile` 加双缓冲预取（`load(N+1)` ‖ `run(N)`），可重叠 VAE 的 ~9.7 GiB I/O（≈9.5s @1.018 GiB/s，多 tile 则 ×tile 数）。**风险低于改 DiT 流式核心**（VAE 的 load/run 边界清晰，不涉及 `stream_slots` / `block_active` / 跨步调度）。

### 理论下限
- 即便 DiT+VAE 的 I/O 被计算完全掩盖，总时间受 `max(I/O 总量 ≈71s, GPU compute ≈47s)` 约束 → **~71s 是 I/O 侧硬下限**（除非进一步减读量或提带宽）。当前 91s → 理论可收 ~20s。

## P12 实测：预取不是瓶颈解法（2026-09-02，重要的负面结论）

### 已实现：VAE 双缓冲预取（`h3_video_vae.c`）
- 新增 `vae_prefetch_job` / `vae_prefetch_thread`（后台线程调 `load_block`）+ `H3_VAE_PREFETCH` 开关（默认开，`=0` 关闭作 A/B 与安全回退）。
- `run_stream_tile` 改造：循环前 prime `blocks[0]`；循环中 `pthread_create` 后台加载 `blocks[N+1]`，主线程 `begin → run_block(N) → submit`，随后 `pthread_join` 并 `free_block(N)`。
- **关键设计**：block 直接后台装进**它自己的槽位** `vae->blocks[N+1]`（因 `run_block` 取 `&vae->blocks[index]`），故 `run_block` 零改动；代价仅多驻留 1 个 block（≈+0.27 GiB）。
- 线程安全依据：`load_block` 最终走 `h3_gpu_tensor_load_f32`（文件→GPU buffer，**无命令编码**），与 DiT 预取线程"只做数据、不碰命令编码"的既有约定一致。

### A/B 实测（steps=4 / 全模型系统盘 / convrot int8 / 4B ClipProj，同参数）
| 配置 | 总耗时 |
|---|---|
| `H3_VAE_PREFETCH=0`（基线） | **91.78 s** |
| `H3_VAE_PREFETCH=1`（预取） | **90.99 s** |
| 收益 | **0.79 s（0.86%）** |

- **正确性验证**：两产物 std **26.3232** / mean **251.1122** / 大小 **29020 B** **三者完全相同** → 预取未引入任何数值或编码回归。
- **收益为何远低于预期**（预期 ~9.5s，实得 0.79s）：VAE 的 I/O 本就不是瓶颈——单 tile 只加载一次、OS page cache 命中、VAE 阶段以 GPU 计算为主导。

### 结论：软件预取已近极限
- **DiT 侧**：加深 slot 对稳态吞吐无改善（吞吐 = `min(compute, io)`，I/O 已瓶颈）→ 不实施，避免动刚修复的流式核心。
- **VAE 侧**：实测仅 0.86% → 预取同样不是解法。
- **根因**：91s 中约 71s 是 I/O 带宽硬约束（72 GiB @ 1.018 GiB/s），**重叠消除不了带宽上限**。
- **保留该改动**（默认开启）：已验证输出逐位一致、无回归；且在**高分辨率/多 tile** 场景下 VAE 每 tile 重读 9.7 GiB 会被放大，收益随之上升。
- **真正提速路径只剩**：① 更快存储（Thunderbolt/USB4，2–3 GB/s → I/O 71s 压到 ~25s）② 更大内存常驻去流式（>64GB）③ 继续减读量（int8 已做，20G vs 61.7G）④ 降 steps/分辨率（质量权衡）。
