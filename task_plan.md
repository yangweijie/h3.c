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

### P8 消费 ModelScope INT8-Convrot 量化权重 — `in_progress`（qkv 置换 bug 已修 / flat 已解 / 待全量 parity）
- 来源：`Abiray/Minimax-H3-nvfp4-INT4-INT8-Convrot`（ModelScope，2026-08 上传，源自 GitHub 镜像，面向 NVIDIA，无 Apple Silicon 验证）。
- 提供：DiT(FL2VA/Ref2VA) + 文本编码器(Qwen3-VL-32B) 的 INT4/INT8/Mixed-Convrot 与 nvfp4；**VAE 未量化**（视频 FP16 5.21GB / 音频 FP32 605MB）。
- 体积参考：文本编码器 INT8-Convrot=27.1GB（BF16 62.13GiB 的 ~2.3× 压缩）；**FL2VA DiT `pruned_int8_convrot`=19.5GB（BF16 61.73GiB 的 3.4× 压缩，含 pruning+int8 叠加，非单纯 int8 2×）**。
- 兼容性硬约束：
  - `nvfp4` = NVIDIA Blackwell 专用 → Apple Silicon Metal 不可跑，**出局**。
  - `INT4` = U8 打包，h3.c 零 int4 kernel（loader 可读 U8/I8 但无 unpack/dequant/matmul）→ 需全新实现。
  - `INT8-Convrot`（已实测头部分解 + ConvRot 配方已确认）：loader 能读 I8，scale 为逐输出通道 F32 `weight_scale`（无 zero-point）。消费需 ①放宽 `h3_gpu.m` 的 "M5" 门控（P7 同主题）②**Convrot 旋转 = Regular Hadamard Transform (RHT，参数无关，见 ConvRot arXiv2512.03673 / github feice-huang/ConvRot / MiniMax-H3-W4A8-ConvRot)**——无需 side 矩阵，h3.c 自生成；实现任务是在 DiT 前向的对应位置/分组粒度复刻 RHT（对激活施 R^T）③**无需**把 int8 扩到 norm/adaln——实测仅 4 matmul/块是 I8 ④块数=50 已对齐。P8 由"可能不可行"转为"明确可行，难在前向对齐"。
  - Convrot 旋转矩阵未单列文件 → 须确认烤进权重还是另有 side 矩阵（决定前向改造量）。
- 决策：文本编码器维持 **4B ClipProj**（5.5GiB，比消费 INT8-32B=27GB 小 5× 且已落地）→ P8 重心放 **DiT INT8-Convrot**；VAE 体积（9.70GiB）本仓库不覆盖，仍靠自加量化/流式。
- 第一步（**已完成**）：`dbg_parse_safetensors.py` 解析 header 完成。结论：932 张量 / I8 17.94GiB + BF16 1.49GiB；**块数=50 对齐**；scale 逐通道无 zero；**rot/convrot 矩阵缺失**（P8 #1 阻塞）。文件 `/Volumes/data/.lmstudio/models/Minimax-H3-Quantized/MiniMax_H3_FL2VA_pruned_int8_convrot.safetensors`（SHA256 `f07a5427…`）。
- 新认知（2026-09-01）：INT8 DiT 面向 24GB dGPU，16GB Mac 统一内存下仍装不下；P8 真实收益是**流式读盘 3.4× 更少**（削 P5 的 I/O 瓶颈），非塞进 RAM。文本编码器本仓库缺失 → 维持 4B ClipProj。
- 第二步（蓝图已定，待实现；**重大修正**：h3.c 已内建 DiT int8，见 findings「h3.c 已内建 DiT int8 路径」）：消费路径 = 直接从 safetensors 读 ConvRot `int8 + weight_scale` 灌入既有 `qkv_int8/qkv_scales/...` 字段 → 唯一新算子 **Hadamard 反旋转 R_K(块256)**：dequant(int8,scale)→BF16 → `W_true = W·R_K` → 复用既有 `h3_gpu_quantize_weight_int8` 回填 int8 字段（run_block 以下全复用）。磁盘流式 19.5GB + 运行期 int8 显存减半**双重收益**。R 约定(W·R/R·W、归一化)由单 block 数值对齐验证，无需读外部源码。
- 状态：P8 由"可能不可行"→"明确可行，且远简单于预期"——h3.c 已有完整 DiT int8 路径，ConvRot int8 格式与既内 int8+逐行 scale 同构；剩余 = int8+scale 加载器 + 一个 Hadamard 反旋转算子 + 单 block 数值校验。

### P8.1 ConvRot qkv 权重行置换修复 — `complete`（2026-09-02，flat 输出根因）
- **现象**：convrot INT8 模型（MiniMax-H3-Convrot）走 `--ssd-streaming` 生成画面**全灰（像素 std≈6.84）**，而 base（MiniMax-H3）同设置 std≈37.76 → 输出被压成一张近均匀灰屏。
- **根因**：ConvRot checkpoint 的 `qkv_proj` 以 **q/k/v 交错布局**存储，需一次行置换 `dst_slot = 3*(src_slot % heads) + (src_slot / heads)`（formula B，heads=56, head_dim=128）才能恢复引擎期望的「按 head 分离的 q/k/v 连续布局」。原代码在**两处**都用了**错误的 formula A**（`dst_slot = (slot%3)*heads + slot//3`）：
  - GPU dequant kernel（`h3_shaders.metal` 的 `h3_weight_dequant_unrotate_int8`，`layout==1` 分支）；
  - CPU unrotate（`h3_dit.c` 的 `convrot_unrotate_cpu` 流式路径）。
- **为何改 GPU kernel 无效**：本机生成走 `--ssd-streaming` → 实际走 **CPU unrotate** 路径；GPU kernel 改动对 std 零影响（两次运行 std 都恰好 6.84 暴露此点）。修正 CPU 路径后才见效。
- **修复**：GPU kernel 与 CPU unrotate 的 `layout==1` 均改为 formula B；refiner BF16 路径（`h3_convrot_remap_qkv_bf16`，`layout==2`）本已为 B，保持一致。三处 remap 现统一。
- **配套事实**（代码/权重实证）：`h3_dit.c:22` 定义 `HEAD_DIM=128`（非 AGENTS.md 写的 96）；main DiT 与 refiner 的 `qkv_proj` 均为 `[21504,5376]` = 168 slots × 128-dim（head_dim=128, 168=56×3）。
- **验证**：`convrot_cpu_fixed.mp4` 像素 std **6.84 → 51.64**，逐帧 std 稳定 ~51.6（富含结构），帧间差异 2.33（平滑时序，非噪声）。**flat 问题已解决**。产物 256×256/22 帧/h264+aac 合法。
- **剩余（P8 收尾）**：①steps=4 下 convrot(std 51.64) 与 base(std 37.76) 亮度/对比度差主要来自步数而非置换（remap 纯置换不改数值尺度）——可跑 base steps=4 同条件对照确认；②全量 `--steps 20` parity 与数值对齐（单 block cosine）仍未做；③int8 反旋转 `H_256` 的 R 约定（W·R vs R·W）此前已靠反旋转 kernel 隐式实现，需补单 block 数值校验固化。

### P9 最小物理占用分析（4B 文本 + INT8 DiT）— `complete`（2026-09-02）
- 目标：确认 "4B 文本 + INT8 DiT" 配置的最小磁盘物理占用（非运行时显存）。
- 组件明细（均为磁盘占用）：
  | 组件 | 大小 | 说明 |
  |---|---|---|
  | INT8 convrot DiT | 20 GiB | `Minimax-H3-Quantized/..._pruned_int8_convrot.safetensors`（引擎已内建 int8 路径） |
  | video VAE | 9.7 GiB | BF16，引擎只认 BF16/F32 |
  | audio VAE | 0.56 GiB | — |
  | tokenizer | 0.01 GiB | 主流程从 `-d` 目录读 |
  | 4B 文本 BF16 | 8.3 GiB | **必须用 BF16**，int8 版被拒 |
  | ClipProj 投影 | 0.48 GiB | — |
  | **合计（最小可用）** | **≈ 39 GiB** | — |
- 两项 INT8 替代均被引擎拒绝（同根因：引擎仅 DiT 有 int8 路径）：
  - **4B INT8 文本**（4.5G 目录）：in-engine `H3_CLIPPROJ_DIR` 加载器只接受 BF16 → `weight model.language_model.layers.0.self_attn.q_proj.weight has dtype/rank I8/2, expected BF16/2` 报错退出。故 4B 文本必须用 8.3G BF16 版。
  - **ModelScope INT8 VAE**（`video_vae.safetensors` 4.85G，恰为 9.7G 一半 → INT8）：VAE 走 `load_f32` → `h3_weight_load_f32` → `load_tensor(...,H3_DTYPE_F32)`；`h3_weights.c:154` 对 `tensor->dtype != dtype` 直接报错退出（与 4B 文本 int8 失败同源）。audio_vae 0.56G 与本地同大小，非 int8，可用但无省空间收益。
- **结论：最小可用下限 = 39 GiB**，不可经 int8 文本/int8 VAE 进一步压缩（除非改引擎为 VAE/文本编码器加 int8 反量化分支）。

### P10 系统盘 I/O 优化（把 DiT+VAE 搬系统内置盘）— `complete`（2026-09-02）
- 动机：P5 诊断认为瓶颈是数据盘（USB/HFS+ SSD）I/O（0.59–0.77 GiB/s，纯等盘 79–93s）。验证把模型搬到系统内置盘（APFS NVMe）能否提速。
- 空间约束：系统盘起初仅剩 3.9–39 GiB，不能一次复制 39 GiB（会填满致系统不稳）。分两步：
  - **混合版**：仅复制决定 I/O 瓶颈的 DiT+VAE+tokenizer（30 GiB）到 `/Users/jay/h3_sys/MiniMax-H3-Convrot`；4B 文本与 ClipProj 留数据盘 → **94 s**。
  - **全模型版**：系统盘空间回升至 17 GiB 后，复制 4B 文本(8.3G)+ClipProj(0.48G) 入系统盘 → **91 s**。
- 踩坑：① `rsync` 默认建临时文件再 rename，2× 峰值撑爆系统盘 → 改 `cp -R -L` 直接写；② 复制遗漏 `transformer/config.json`（Convrot 壳 symlink 未镜像）与 `FL2VA/text_encoder` 目录 → 逐一补文件/symlink 解决。
- 完整对比（同 16 帧/256/steps4/convrot int8/4B ClipProj）：
  | 配置 | 总耗时 | SSD 流读速度 | 纯等盘 |
  |---|---|---|---|
  | 50层 Qwen + 数据盘 | 194 s | 0.590 GiB/s | 93.2 s |
  | 4B + 数据盘 | 153 s | 0.590 GiB/s | 93.4 s |
  | 4B + 系统盘(仅 DiT/VAE) | 94 s | 1.021 GiB/s | 43.5 s |
  | **4B + 全模型系统盘** | **91 s** | 1.018 GiB/s | 43.7 s |
- 产物校验：256×256 / 22 帧 / h264+aac / 0.92s，像素 std=26.32（与数据盘 4B 版一致，非 flat），质量无差。
- **结论**：瓶颈假设被证实——DiT+VAE 流式读取 0.59→1.02 GiB/s，纯等盘 93→43 s，总耗时 153→91 s（提速 40%）。全模型系统盘 91s 为当前最优；混合版(94s)与之仅差 3s（因 4B 文本权重不计入那 72GiB 流式读取，仅一次性 load）。系统盘 NVMe 实测仅 ~1.02 GiB/s（未达内置盘上限），仍有上行空间（引擎更深预取 / 更快盘）；GPU compute 下限 ~70s 为硬约束。

### P11 全量 parity 与单 block 数值校验 — `in_progress`（2026-09-02）
- 背景：P8.1 修复（qkv 置换 formula A→B）后 convrot 输出从 flat(std 6.84) 恢复到 std 51.64，但**尚未与 base 做同条件 parity**；且 P8.1 的校验只覆盖 `qkv_proj` 一张量。
- 子任务（按"快且能定论"优先）：
  - **P11.1 单 block 数值校验**（离线 numpy，优先）：把 P8.1 的 per-slot cosine best-match 从 `qkv_proj` 扩展到 block 0 的全部 4 个 int8 matmul（`qkv_proj` / `out_proj` / `fc1` / `fc2`）。对每张量做 `dequant(i8*scale)` → 块 256 Hadamard 反旋转 → 与 base BF16 同张量比对，钉死 ① R 乘向（W·R vs R·W）② rot_size=256 分组 ③ qkv 的 formula B 置换。
  - **P11.2 base steps=4 对照**（当前缺失）：已知 base_20 std=37.76、convrot_4(修复后) std=51.64，但 **base steps=4 未跑** → 无法判定 std 差异来自"步数"还是"量化/置换"。须补跑 base steps=4（4B ClipProj 同配置）。
  - **P11.3 convrot steps=20（修复后）**：与 base steps=20 (std 37.76) 直接 parity。已知 convrot_20 **未修复**时为 8.58，修复后未跑。
- 判定标准：同 steps 下 convrot 与 base 的逐帧 std / 像素分布应接近；偏离过大说明 int8/反旋转仍有数值问题。

### P12 引擎更深流式预取（回收剩余 43s 等盘）— `in_progress`（调研中，2026-09-02）
- 背景：P10 证实瓶颈是 DiT+VAE 流式 I/O（全模型系统盘 91s，纯等盘 43.7s，流读 1.018 GiB/s）。
- 目标：增大流式预取深度（prefetch depth / stream slots），让 I/O 与计算更好重叠，回收部分 43.7s unhidden wait。
- 调研清单（**先摸清现状再改**）：
  - `h3_dit.c` 的 `stream_slots` / `stream_thread` / slot 数常量：当前深度、调度方式（`read_stream_layer`、信号量/条件变量同步）。
  - DiT 4 个 streamed matmul（`STREAM_QKV`/`STREAM_OUT`/`STREAM_FC1`/`STREAM_FC2`）的 slot 分配粒度。
  - VAE 流式路径 `run_stream_tile`（`h3_video_vae.c`）的预取策略：是否仅常驻 1 块、有无流水线。
  - **内存账**：加大深度 = 更多 slot 常驻权重。DiT 单块 4 matmul ≈ 61.7GiB/50 ≈ 1.23GiB（int8 减半 ≈0.6GiB）；VAE 单块 ≈9.7GiB/36≈0.27GiB。16GB 统一内存下能开几个 slot（须留激活/文本编码器/ClipProj 空间）。
  - `h3_memory_plan.c` 的 streaming-aware 估算是否随 slot 数变化，需同步更新常量。
- **风险（P5 已警示）**：改动刚修复的流式核心（`h3_dit.c` stream_thread）有回归风险；须保留 `--ssd-streaming` 开关与原有 slot 数作为对照，分步加深。
- **收益上限**：即便预取完美隐藏 I/O，GPU compute 下限 ~70s 仍无法突破 → 理论最多回收 91→~75s（约 16s）。

### P12 实测结论 — `complete`（2026-09-02）：预取不是瓶颈解法，I/O 带宽才是
- **DiT 加深预取：判定为无收益（不实施）**。稳态流水线吞吐 = `min(compute_rate, io_rate)`；当前 I/O 71s > compute 47.3s，I/O 已是瓶颈 → 加深 slot 数不改变吞吐。避免了对刚修复的 `h3_dit.c` 流式核心做高风险改动。
- **VAE 预取：已实现并 A/B 实测**（`h3_video_vae.c`）：新增 `vae_prefetch_job` / `vae_prefetch_thread` 与 `H3_VAE_PREFETCH` 开关；`run_stream_tile` 改为双缓冲——循环前 prime block 0，循环中后台线程加载 `blocks[N+1]`（直接装进它自己的槽位，故 `run_block` 无需改动，仅多驻留 1 个 block ≈ +0.27 GiB）。
- **A/B 实测**（steps=4 / 全模型系统盘 / convrot / 4B ClipProj，同参数）：
  | 配置 | 总耗时 |
  |---|---|
  | `H3_VAE_PREFETCH=0`（基线） | 91.78 s |
  | `H3_VAE_PREFETCH=1`（预取） | 90.99 s |
  | **收益** | **0.79 s（0.86%）** |
- **正确性**：两产物 std 26.3232 / mean 251.1122 / 29020 bytes **完全相同** → 预取未破坏输出，无回归。
- **收益远低于预期（预期 ~9.5s，实得 0.79s）的原因**：VAE 的 I/O 并非瓶颈——单 tile 只加载一次、OS page cache 命中、且 VAE 阶段以 GPU 计算为主导。
- **最终结论**：结合 DiT（加深无稳态收益）与 VAE（实测 0.86%）两端，**软件层预取已接近极限**；91s 中的 ~71s 是 I/O 带宽硬约束，无法靠重叠消除。
- **决定**：保留代码并默认开启（已验证输出逐位一致、无回归；高分辨率/多 tile 场景下 VAE 重读放大、收益会上升），并提供 `H3_VAE_PREFETCH=0` 开关。
- **真正提速只剩硬件/质量权衡**：① 更快存储（Thunderbolt/USB4，2–3 GB/s 可把 71s 压到 ~25s）② 更大内存以常驻去流式（>64GB）③ 继续减读量（int8 已做）④ 降 steps/分辨率。
