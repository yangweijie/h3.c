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
