# Findings & Decisions: h3.c 超分集成 + 生成调优 + Metal 内核优化

## Requirements (SR 集成，已完成)
- 命令行参数：`--sr-bin` / `--sr-model-dir`（两者有效才启用 SR）
- 内分辨率自动探测；目标 = `--sr-target WxH` 或 内分辨率×`--sr-scale`
- 超分后保留原声音

## Research Findings
- `H3_FPS = 24`（h3_host.h），生成固定 24fps，SR 重编码也用 24fps
- 工程已有 `run_ffmpeg`(posix_spawnp) 与 `h3_ffprobe_visual_size()` 可复用
- realesrgan-ncnn-vulkan 就绪：`/tmp/h3_realesrgan/realesrgan-ncnn-vulkan`；模型含 `realesrgan-x4plus`(×4)、`realesrgan-x4plus-anime`、`realesrnet-x4plus`、`realesr-animevideov3`(x2/x3/x4)
- 模型/LoRA 路径：
  - DiT：`/Users/jay/h3_sys/MiniMax-H3-Convrot`（权重在 FL2VA 子目录）
  - 量化备选：`/Volumes/data/.lmstudio/models/Minimax-H3-Quantized`（MiniMax_H3_FL2VA_pruned_int8_convrot.safetensors 20.9G）
  - CLIPProj：`/Volumes/data/.lmstudio/models/ClipProj-MiniMax-H3`
  - 文本编码器：`/Volumes/data/.lmstudio/models/Qwen3-VL-4B-Instruct-int8-convrot`
  - **本机无任何 Turbo/distill LoRA**（`find` 全盘无 turbo/lora/distill 文件）
- 运行时：macOS Apple M4，Metal；`setsid` 不可用，用 `nohup ... & disown` 后台

## Metal BF16 类型转换（关键发现）
- **问题**：Metal shader 中 `h3_bf16_to_f32()` 接受 `ushort` 参数，但被调用时传入 `bfloat*`，编译器生成错误类型转换代码，导致 kernel 输出全零
- **根因**：`bfloat` 和 `ushort` 是不同 Metal 类型，不能隐式转换
- **修复**：
  - 添加 `h3_bf16_to_f32(device const bfloat *addr)` 重载，使用 `as_type<ushort>(*addr)` 读取
  - 添加 `h3_f32_to_bf16(device bfloat *addr, float value)` 函数，使用 `as_type<bfloat>(ushort(bits >> 16))` 写入
- **经验**：Metal 中 BF16 必须使用 `as_type<>()` 进行显式位转换，不能依赖 C 风格的 `(ushort)` 或 `(bfloat)` 转换

## FlashAttention 内核优化
- **Causal attention** (`h3_flash_attn_causal`)：
  - Online softmax（two-pass: max + sum）
  - 复杂度 O(T²/2)（只处理 k <= q 的 key）
  - 测试：seq=8, heads=1, dim=4, max_err=0.00095
- **Tiled windowed attention** (`h3_flash_attn_tiled_windowed`)：
  - 每个 query 根据类型（text/video/audio）计算 3 个 key 范围
  - 只遍历窗口内的 key，复杂度 O(T·window)
  - 测试：seq=24, F=4, S=4, r=2, max_err=0.00817
- **Linear far-branch 基础设施**：
  - 完整的 SanaDelta delta-rule scan 管线
  - 需要训练权重（alpha、beta、q_norm、gate、to_out_linear），当前 checkpoint 中不存在
  - 7 个测试全部通过

## FP8 量化集成分析
- C 代码已有量化基础设施：
  - `h3_gpu_quantize_bf16_int8_rows` (SIMD vec4)
  - `h3_gpu_quantize_bf16_int8_groups`
  - `h3_gpu_linear_int8_bf16` (cooperative tensor ops)
  - `h3_gpu_mlp_int8_bf16` (SwiGLU 融合)
- FP8 E4M3 在 M4 上可通过 INT8 tensor core 模拟（同一硬件），动态范围更好（±448 vs ±127）

## 色块根因（关键发现）
- 失败运行：target 256×192 / **render 128×96** / layer 40 / token-reduction / SR→1024×768 / 15s → 全屏 3 段水平色带
- 机制：128×96 时 latent token 网格仅 8×6=48 token/帧；token-reduction 在中间 DiT 层把 token 成对池化→约 24/frame，信号量不足以表达结构化画面，VAE 解出平坦色
- 证据链：2s + 原生 256×192 即出内容 → 排除 steps=4 与 token-reduction 为元凶（用户亦证"之前 2s token-reduction 没问题"）
- **结论：元凶是渲染分辨率过低，不是 --steps 4，也不是 --token-reduction**

## 参数对照结论（2s / layer 40 / steps 4 / target 512×384）
| 内部渲染 | token-reduction | SR | 产物 | 体积 | 备注 |
|---|---|---|---|---|---|
| 256×192 原生 | 有 | 无 | /tmp/h3_256_192_native_2s.mp4 | 139K | 对照（出内容）|
| 256×192 原生 | 有 | →1024×768 | /tmp/h3_256_192_native_2s_sr.mp4 | 1.5M | |
| 256×192 render | 有 | →1024×768 | /tmp/h3_512_384_r256_2s_sr.mp4 | 1.5M | |
| 256×192 render | 无 | →1024×768 | /tmp/h3_512_384_r256_notr_2s_sr.mp4 | 1.9M | 比上者细节多 |
| **原生 512×384** | 有 | 无 | /tmp/h3_512_384_native_tr_2s.mp4 | 223K | |
| **原生 512×384** | 无 | 无 | /tmp/h3_512_384_native_notr_2s.mp4 | 283K | 基准最不糊 |
| 原生 512×384（上者 LR）| — | realesrgan-x4plus ×4 手跑 | /tmp/h3_512_384_lr_x4plus.mp4 | 2048×1536 / 4.1M | 最清晰 |
| 原生 512×384 | 无 | 无 | /tmp/h3_512_384_native_notr_15s.mp4 | 1.0M | 15s 基准, exit=0 |

- token-reduction 影响：关掉后体积变大（223K→283K）、峰值内存升高（5.9→7.1 GiB）→ 压缩激活量，关掉更吃显存但细节更多
- 原生 512×384 vs 256×192 render：前者 VAE 解码更慢（~64–75s vs 更快），但内部分辨率高、画面结构更完整

## SR 行为细节
- 引擎默认 `--sr-target` 在"目标 ≤ 输入×4"时：先跑 ×4 模型再 crop/缩到目标
  - 故 1024×768（512×384 的 ×2）= "×4 模型结果缩到 ×2"
  - 纯 ×4 即 2048×1536（512×384 的 ×4）= 模型原始输出，最清晰
- 手跑管线（对 .lr.mp4）：`ffmpeg` 抽帧 → `realesrgan-ncnn-vulkan -n realesrgan-x4plus -s 4 -i 帧目录 -o 帧目录 -m 模型目录` → `ffmpeg -framerate 24 -i %05d.png -c:v libx264 -pix_fmt yuv420p -crf 18 out.mp4`

## 参数校验规则（来自 main.c）
- target 与 render 宽高均需 **32 的倍数、≥32**（报错：`width and height must be multiples of 32 and at least 32`）；render ≤ target；同宽高比
- SR target 与 output 同宽高比且 ≤4×；frames 对齐到 5..362
- 注意：校验**不捕捉**"渲染分辨率过低导致的画质崩坏"（128×96 render 校验通过却出色块）
- **有效 4:3 子 512×384 分辨率**：W=128k, H=96k → 仅 384×288 / 256×192 / 128×96（320×240、192×144、160×120 因高非 32 倍数被拒）

## Issues Encountered
| Issue | Resolution |
|-------|------------|
| ffmpeg 多输入命令把 `-vf` 当输入选项 | 缩放拆独立单输入 pass；抽帧去 `-vf` |
| `setsid` 不可用（macOS）| `nohup ... & disown` 后台 detached |
| h3 非 TTY 下 stdout 全缓冲（日志空到结束）| 监控靠进程 PID / mp4 存在 / FINAL_EXIT 标记 |

## Resources
- 生成入口：`main.c` prompt 分支 → `h3_generate()` → `params.output_path`
- SR 实现：`h3_ffmpeg.c::h3_superres`
- 交互：`h3_cli.c::process_command` 的 `!sr`

# VDN-H3 规格研究 (2026-09-05 session)

## 混合注意力（hybrid_attention.py）
- `softmax_out = to_out[0](softmax_gate(x)⊙window_softmax(q,k,v)) → to_out[1]`；`out[video行] += to_out_linear(output_gate(xv)⊙RMSNorm(linear_readout))`
- 线性分支共享 softmax 的 raw q/k/v（pre-QK-norm、pre-RoPE，NoPE）；只处理 video 行
- full_cover（窗口覆盖全部帧）时走原稠密 attention，线性分支关闭

## softmax 窗口（window.py）
- `window_bounds(F, radius, chunk)`：chunk=0 帧模式 `|t−k|≤r`；chunk=K 块模式 frame t → `[(t//K−r)·K, (t//K+r+1)·K−1]`
- VDN checkpoint: chunk=5, radius=1 → 每 query 帧看 3 个整块（15 帧）
- anchor_frames="both"：帧 0 与 F−1 (a) 作为行 dense（query 看全序列），(b) 作为列（所有 query 可见，不与窗口重复）；此时线性分支 skip_ends（首尾帧输入整段删除、读出=0，bounds 重基 −1）
- text/audio 行稠密双向

## 线性分支（branch.py + scan.py + delta_rule.py + features.py + layers.py）
- 特征：k,v 先 depthwise 5×5 空间 conv（padding2, groups=7168）再 5-tap 时间 conv（零填充非因果）；k: conv→SiLU→L2Norm(1e-6)；q: SiLU→L2Norm；v: conv→SiLU 不归一
- beta=sigmoid(beta_proj(x)) [F,H,S]；A=(βk)ᵀk fp32 对称化；B=(βv)ᵀk fp32
- alpha=exp(−exp(A_log_h)·softplus(up(down(x̄_f))+dt_bias)) fp32，x̄_f=帧均值(dtype=fp32)
- vdn_solve（released checkpoint 用的规则）：chol(I+A)，inv=L⁻ᵀL⁻¹，transition=Diag(alpha)·inv，injection=B@inv（全部 fp32 算完再回 bf16）
- 文本状态：text 行无 conv，beta 同 beta_proj，alpha=ones，c=1/sqrt(L_text) 缩放 A/B 后同规则得 S_text；双向扫描起点 = 0.5·S_text
- 扫描：S_t = S_{t-1}·transition_t + injection_t，forward 前缀 + reverse 后缀（fp32 状态库）
- gather：left=prefix[lo−1]、right=suffix[hi+1]；bridge=α 在 [边,t] 帧区间乘积（log 前缀差）；窗口触界侧读 text_state 按 α 衰减；两侧相加
- 读出：einsum(fshv, fhvk) → RMSNorm(head_dim=128) → ×output_gate（down[128,5376]+up[7168,128]+bias, sigmoid, per token/head/channel）→ reshape [T, 7168]
- softmax_gate：per head [56,5376]+bias，sigmoid 后乘 softmax 输出

## LoRA / 调度
- default: rank64 alpha64 → scale=1.0，targets `attn.orig.to_{q,k,v}`、`attn.orig.to_out.0`、token_refiner 同四件
- turbo: 同 targets、后缀 `.lora_A.turbo`（非 default），另含 `norm_out.linear.lora_*.turbo.weight` [16,2688]/[10752,16]（final layer）
- turbo 8 steps；video_shift=12.0 / audio_shift=3.0 —— 与 h3_host.h 的 H3_VIDEO_SIGMA_SHIFT/H3_AUDIO_SIGMA_SHIFT 完全一致，调度无需改

## h3.c 现状要点（调研结论）
- DiT 常量已匹配：HEADS=56, HEAD_DIM=128, INNER=7168, FFN=14336（h3_dit.c:18-33）
- run_block 在 h3_dit.c:2420-2592；全序列稠密 SDPA（h3_gpu_sdpa_bf16），无窗口/线性分支接线
- 已有未接线基础设施：h3_flash_attn_tiled_windowed（shaders:5468）、h3_sdpa_window_mask（4507）、SanaDelta linear-branch 内核组（4555-4850，一阶截断 ≠ vdn_solve）
- h3_lora.c：仅 6 个固定 target、仅 `.default` 后缀、单 alpha、仅 BF16、不支持 streaming/int8；机制上 target 字符串任意，扩 orig.* 行带映射即可
- 权重目录：h3_weight_store_open glob 目录全部 *.safetensors（多分片 OK）；h3_load_dir 只认 FL2VA 自有布局 → VDN stage 目录需单独加载器
- 缺 depthwise conv Metal 内核；缺批量 Cholesky；kernel 注册流程明确（shaders + h3_gpu.m 名单 + wrapper）

## VDN 集成实现要点 (2026-09-05/06 session)
- 新增 16 个 h3_vdn_* Metal 内核 + h3_vdn_window_args/feat_args/bmm_args（C/Metal 字段对齐）
- bf16 加载陷阱：h3_bf16_to_f32(value) 的隐式转换会产生垃圾 → 必须用指针重载
  h3_bf16_to_f32(ptr + idx)；Metal 无 log1p；线程组内存上限 32KB
- bmm 的 add 缓冲必须始终绑定（缺绑定 → 命令缓冲中止，且无显式报错）——
  用 Metal API Validation（MTL_DEBUG_LAYER=1）抓到
- vdn_solve 精确求逆 = h3_vdn_cholesky（线程=列, 右移更新, 共享列广播）
  + h3_vdn_triinv（行广播前代）+ bmm(XᵀX)；对角线在内核内 +I
- LoRA 流式合并：SSD 预读线程用 h3_lora_merge_blocking /
  h3_gpu_blocking_linear_bf16（私有 command buffer，不与主线程 open buffer 冲突）
- to_out_linear 走 STREAM_LIN_OUT 流式；其余分支张量常驻 ~12MB/块
- 坑：qsort 越界（未初始化的第 5 个 source）——SEGFAULT 在无关位置，靠
  原HEAD对照+栈回溯定位；stream source 需显式 source_count
- 端到端：convrot 基座 + stage-dmd adapters(default+turbo) + linear_branch
  在 16GB 机 streaming 模式下可完成全部 8 步去噪 + VAE 解码

## 视频 VAE streaming 崩溃（既有 bug，与 VDN 无关）
- 现象：audio VAE 7/7 之后、video VAE 解码阶段静默 SIGSEGV（exit 139），无 MP4
- 复现：任何开启 auto memory plan 的运行（planner 会打开 video_vae_streaming）
  —— 基座（无 --linear-branch/--lora）同样 139；显式 --ssd-streaming
  （绕过 planner → video_vae_streaming=0）则 exit 0 正常出片
- 结论：崩溃点在 run_stream_tile 流式 VAE 解码路径，与本次 VDN 改动无关
- 规避：运行 VDN 时显式加 --ssd-streaming（禁用 auto plan）

# INT8 线性分支 + 性能剖析 (2026-09-07 session)

## 线性分支 int8 权重文件事实
- `model_int8_convrot_comfyui.safetensors` 2.30 GB vs bf16 `model.safetensors` 4.28 GB
- 1100 张量 = 650 BF16 + 150 I8 + 150 F32 scale + 150 U8 comfy_quant
- **每 block 仅 3 个张量被量化**（50×3 = 150）：
  - `linear_attention.beta_proj.weight` [56, 5376]
  - `linear_attention.output_gate.down.weight` [128, 5376]
  - `to_out_linear.weight` [5376, 7168]
  - 其余（alpha.* / norm / short_conv / gate up+bias / softmax_gate）保持 BF16
- scale 命名 `{name}.weight_scale`，F32 [rows, 1]；另有 `{base}.comfy_quant` U8[72]（h3.c 不读取）
- 该文件**自包含**（含全部非量化张量），可单独打开，不必与 bf16 文件合并

## 反量化公式与判决性验证
- `w = scale[row] × (1/16) × FWHT_256(i8)`（radix-4 蝴蝶，stride 1/4/16/64，输入维按 256 分块）
- CPU 复现 vs bf16 真值：

| 张量 | 形状 | convrot cos | 不做反旋转 cos |
|---|---|---|---|
| beta_proj.weight | [56, 5376] | 0.999954 | 0.065 |
| output_gate.down.weight | [128, 5376] | 0.999954 | — |
| to_out_linear.weight | [5376, 7168] | 0.999921 | — |

- 三者列维均 `%256==0`，满足旋转块对齐；不做反旋转则完全不可用（cos 0.065）

## h3.c 既有设施（故改动量极小）
- `h3_weights.c::load_int8_dequantized`：I8 → BF16/F32，scale 名 `"%s_scale"`，含 convrot 反旋转（`H3_INT8_UNROTATE=0` 可关）
- `h3_weight_load_bf16` → `load_tensor`：`tensor->dtype==I8 && 请求 BF16` 时**自动**走该路径
- 主模型 `prepare_stream_source`（h3_dit.c:1120）已支持 I8 流式并填 `scale_path/offset/elements/name`

## 代码改动（3 文件，VDN loader 本体零改动）
1. `h3_weights.h/.c`：新增 `h3_weight_store_open_file()`（单文件 store）
2. `h3_dit.c` VDN 打开：候选名单 `model_int8_convrot_comfyui.safetensors` → `model.safetensors` → 整个目录
3. `h3_dit.c::prepare_vdn_stream_source`：接受 I8 并填 scale 字段（原先硬性要求 BF16）
4. `h3_dit.c` 流式消费点：`field == STREAM_LIN_OUT` 时从 `dit->vdn_weights` 取 scale
5. `h3_dit.c::load_convrot_scale_values`：改接收显式 `h3_weight_store *`（主模型传 `dit->weights`，行为不变）

## 关键机制：linear 分支在流式模式下如何加载
- `load_core`（h3_dit.c:2064）：ssd_streaming 分支**只**调 `load_block_norms` + `prepare_stream_layer`，不调 `load_vdn_block`
- `load_block_norms`（1053-1063）：调 `load_vdn_block` 加载全部常驻分支权重后**主动释放** `lin_to_out`
  ```c
  /* to_out_linear is streamed; drop the resident copy the loader made. */
  h3_gpu_tensor_free(block->lin_to_out); block->lin_to_out = NULL;
  ```
- 仅 **2 个 stream slot**（h3_dit.c:218）；每 block 每步从 slot 取 `lin_to_out`（3514-3530）
  → `to_out_linear` **每 block 每步重流**

## 性能实测（`H3_PROFILE`，256×256 / 1s / steps=4 / seed 42 / --ssd-streaming）
| | 有 linear(VDN) | 无 linear(原版) |
|---|---|---|
| 总耗时 | **305 s** | **90 s** |
| SSD 流式读取 | 79.350 GiB / 75.3 s | 72.136 GiB / 66.1 s |
| 吞吐 | 1.053 GiB/s | 1.091 GiB/s |
| **unhidden wait（未被计算掩盖的 I/O 等待）** | **0.001 s** | **23.332 s** |

- 差值 7.214 GiB ÷ (4 步 × 50 block) = 36 MB/次 ≈ `to_out_linear` int8 体积（38.5 MB），精确对上
- VDN 的 I/O **100% 被计算掩盖**（wait ≈ 0）→ 常驻 `to_out_linear` 省 **0 秒**、却要付 50×77MB = **3.85 GB**
- I/O 仅多 9.2 s，总耗时却多 215 s → **VDN 的额外开销是计算（Metal kernel），不是 I/O**
- 反向结论：原版 unhidden wait 23.3 s（占其 90 s 的 26%）→ **原版才是 I/O 受限**，砍流式字节只对原版有效

## 步数结论（"一片黄色"的根因）
- steps=2：VDN 与**无 VDN** 均为橙棕糊（R−B ≈ 52~55，std 18~23，grad ≈ 5）
- steps=4：VDN（int8 与 bf16）与原版均出狐狸
- → 黄色 = **steps=2 欠采样**，与 linear 分支、与 int8 **均无关**

## 环境变量事实（重要）
- `H3_NAX_FORCE`：`grep -rn` 全仓库无引用 → **空操作**（可从命令中删除）
- `H3_NAX`（h3_gpu.m:366）：取值 0 / mlp / qkv-attn，控制 NAX tensor ops
- `H3_VDN_INT8`（存在即生效）：
  - h3_dit.c:2934 主 DiT 共享 QKV 走 int8 matmul
  - h3_dit.c:3072 VDN `to_out_linear` 走 int8 matmul（运行时量化 `weight->lin_to_out`）
  - h3_gpu.m:371 在非 M5 上开启 NAX tensor ops
- **本机 Apple M4（非 M5）**：`wantsTensorOps = (m5 || H3_VDN_INT8) && H3_NAX!="0"` → 默认关闭；
  设 `H3_VDN_INT8=1` 会 `cannot stream DiT block 1: unknown Metal error`（13 s 崩），**不可用**

## 输出一致性证据
- int8 vs bf16（同 seed / 同 steps）：cos = **0.9904**，mean_abs_pixel_diff = **9.3 / 255**
- 用户把两个文件对调（`model.safetensors`=int8，`model_bf32.safetensors`=bf16）后重跑：
  产物 **198461 字节，与之前逐字节一致** → 确定性未破坏，int8 接入正确

## 跨尺度性能验证（"高分辨率能否翻盘"）
- 测试:512×384 / 1s / steps=4 / seed 42 / --ssd-streaming（像素 = 256×256 的 **3.0×**）

| 配置 | 256×256 | 512×384 | 耗时缩放 |
|---|---|---|---|
| VDN | 305 s | **1295 s** | **4.25×**(超线性)|
| 无 VDN | 90 s | **186 s** | **2.07×**(亚线性)|
| VDN 倍率 | 3.4× 慢 | **7.0× 慢** | 差距拉大 |

- 结论:VDN **不能翻盘**,尺度越大越慢。
  - 原版窗口 softmax 亚线性缩放(O(N·window),window 固定 15 帧)
  - VDN 线性分支超线性 —— 大量 per-token kernel(features/stats/cholesky/scan/gather/readout/short_conv/gate)
    随 token 数增长比窗口 softmax 更快
- 根因:VDN 是**叠加混合分支**(h3_dit.c run_block:窗口 softmax 与线性分支**都跑**),
  不是替代;线性分支的常数因子大,且随分辨率恶化更快
- → int8 线性分支接入正确,但其定位是「质量特性」,在当前实现下**任何尺度都不能加速**

## Streaming GPU ConvRot 反量化：失败根因（重要，避免重复踩坑）
- 目标：把 streaming 线程的 CPU 蝶形反量化搬到 GPU，与主线程计算重叠，消除 23s unhidden wait。
- 实现：新增 `h3_gpu_blocking_weight_dequant_unrotate_int8`（h3_gpu.h/.m），
  沿用既有 `h3_gpu_blocking_linear_bf16` 模式——**同一 MTLCommandQueue + 独立命令缓冲 + commit + waitUntilCompleted**。
  这个模式证明"后台线程可独立做 GPU 工作"是可行的（无需第二队列），但本项目未最终采用。
- **两条实现路径都失败**：

| 路径 | 耗时 | 输出 |
|---|---|---|
| dequant 放主线程（join 后，串行） | 151s | ❌ 白色 |
| dequant 放 streaming 线程（blocking 独立命令缓冲） | 144s | ❌ 白色 |
| 基线（CPU 蝶形，streaming 线程） | 90s | ✅ 正常 |

- **更正（重要）：kernel 数值完全正确，上面"根因"是 debug 代码 bug 造成的假象**
  - 最初的诊断代码写错了 bf16→f32 转换：
    ```c
    uint16_t gpu_vals[8];
    memcpy(&g, &gpu_vals[k], 4);   /* ❌ 从 uint16_t* 拷 4 字节 */
    ```
    这会把相邻两个 bf16 拼成一个 float，算出 0.06–0.14 的"偏差"，
    从而误判"layout=0 的 out/fc1/fc2 三个张量错误"。
  - 正确转换是 `bits = (uint32_t)val << 16; memcpy(&f, &bits, 4);`。改正后重测：
    **跨矩阵多行（row 0 / 1344 / 2688 / 4032 / 5376 / 10752 / 16128 / 21504）
    全部 `d=0.000000`**，idx 0–3 全对 → GPU dequant 与 CPU 蝶形**逐位一致，kernel 无需修改**。
  - 注：只比 row 0 是不够的（row 0 在 layout=0/1 下都映射到自身，无法暴露行置换错误），
    必须跨多行采样。
- 真正的失败点在**集成层面**，不是 kernel：
  1. **可见性/同步**：GPU kernel 写入 `storageModeShared` buffer 后，后续 GPU kernel
     （int8 requant）可能读不到该写入。诊断版因为 CPU `memcpy` 覆盖而"固化"了值所以正确，
     纯 GPU 写入版则白色失真。
  2. **性能**：blocking 模式每 source 一次 `commit` + `waitUntilCompleted`，
     外加额外 GPU tensor 分配/释放，开销吃掉全部收益（144s vs 90s）。
- 结论：该方向**即使修好可见性也大概率仍更慢**，已回退。回退后 VDN int8 @4 产物 198461 字节，与回退前一致。
- 教训：**先确认测量工具本身正确，再追根因** —— 错误的一次 memcpy 让我追了两轮假根因。
- **fence/barrier 解决 + 批量 dequant（后续两轮）**：
  - 用 `MTLEvent` 跨 command buffer 同步（dequant signal → requant wait），
    纯 GPU 写入版输出**正确**（R=122.1/93.2/57.8/std=24.4 ≈ 基线），可见性问题解决。
  - 批量 dequant：4 个 source 编码进同一 command buffer、每 block 只提交一次
    （50 次 vs 200 次）。输出正确，但 **182s vs 90s 更慢**。
  - **最终根因（架构级）**：GPU dequant 与主线程计算**共享同一个 GPU**，互相竞争，
    消除了 CPU 蝶形原本享有的"CPU-GPU 异构并行"优势。
    CPU 蝶形在 CPU 上跑（∥ GPU 计算）→ 仅 23s 暴露；
    GPU dequant 在 GPU 上跑（与 GPU 计算串行）→ 全部暴露。
    **GPU dequant 方向从根本上不可行。**

## 端到端参数基准测试(Phase 12)
- 工具链:`benchmark.py` — 参数扫描 + 帧对比评分(SSIM/PSNR/L2,纯 numpy 实现) + 自包含 HTML 报告(内嵌 base64 帧图)
- 测试矩阵按**预期峰值显存从低到高**排序,超过 13 GB 自动停止
- 三轮共 22 个独特配置(去重后),覆盖 steps(4/7/10/20)、layers(40/45/50)、reuse(1/2/3)、core-reuse(2/4/6)、token-reduction、render-resolution(192/256)

**核心结论**:

| 发现 | 说明 |
|---|---|
| 显存不是瓶颈 | 所有测试 9.4–9.8 GB(<13 GB),`--ssd-streaming` 效果显著 |
| step=7 vs step=4 | 同 layers=50/reuse=2:s7-l50-r2(SSIM=0.570) vs s4-l50-r2(SSIM=0.371),画质提升 54% |
| --token-reduction | 省 ~12% 时间(s4-l40_cr4 96.1s → s4-l40_cr4-tr 84.8s) |
| 最快 | s4-l40-cr4-tr-R192: 73s, SSIM=0.214(画质损失大) |
| 最佳权衡 | s7-l50-r2: 177s, SSIM=0.570,效率最高(0.0032) |

**推荐配置**:
- 极速预览:`--steps 4 --layers 40 --core-reuse 4 --token-reduction --render-width 192 --render-height 192`(73s)
- 最佳权衡:`--steps 7 --layers 50 --reuse 2`(177s)
- 高质量:`--steps 10 --layers 45 --reuse 2`(235s)

- 报告:`/tmp/h3_benchmark/report.html`

# 代码审查发现(2026-09-07)

## h3_audio_vae.c

| 类别 | 数量 | 关键项 |
|---|---|---|
| Bug/潜在 Bug | 3 | `hidden_elements` 溢出检查缺失、`add_scaled` 缺少类型转换、`output->values` 失败路径状态 |
| 内存优化 | 2 | `run_stage` 5→3 tensor、qkv slicing |
| 健壮性 | 3 | JSON 键匹配、`getenv` 缓存、`ftell` 错误处理 |
| 设计观察 | 3 | 原地操作别名、无 logvar（预期行为）、双 LayerNorm |

> 整体代码质量很高——错误处理一致、资源清理全面、溢出检查周到。主要的实际风险是 `hidden_elements` 溢出检查缺失和 `run_stage` 的峰值内存。

## h3_cli.c

| 类别 | 数量 | 关键项 |
|---|---|---|
| Bug | 4 | 路径截断静默、`!seconds` 过小值、`open_video` 跨平台、`!again` 失败静默 |
| 安全 | 3 | TOCTOU 竞争、SR 可执行路径注入、错误缓冲区大小 |
| 内存 | 2 | `strdup` 失败未检查（2 处） |
| 设计 | 4 | 巨型 if-else 链、`INT32_MAX` 帧上限、进度比较优化、退出清理 |

> 整体来说这是一个成熟的 CLI 实现——有行编辑历史、进度显示、引用管理、超分辨率后处理等功能。代码结构清晰，错误处理一致。上面列出的大部分是边界情况或可防御性编程的改进，没有严重的功能性缺陷。

## h3_dit_schedule.c

| 类别 | 数量 | 关键项 |
|---|---|---|
| Bug | 3 | 视觉/音频阈值不一致、`time` 张量生命周期管理脆弱、`count * feature_dim` 溢出检查缺失 |
| 安全 | 2 | `gate_score` 大分配无检查、`weight_bf16_any` 32 位截断 |
| 正确性 | 4 | LoRA 跳过静默、`row_map` 无错误信息、调试代码生产路径、浮点相等比较 |
| 性能 | 2 | `gate_score` 每次重新读 GPU、`getenv` 热路径 |

> 最严重的是第 1 点（视觉/音频阈值不一致）和第 2 点（`time` 张量生命周期），前者可能导致数值错误，后者在代码演进时容易引入 double-free 或 leak。其余是防御性编程和代码质量改进。

## h3_dit.c 设计观察(非 bug,但值得注意)

**两级决策顺序(流式开关 → DiT 深度裁剪)**:
- 逻辑清晰,但 `free_after_stream` 的 < 4 GiB 检查发生在 `cache_budget_bytes` 计算**之后**
- 如果 DiT 层被裁剪,激活预算也会缩小,可能释放更多缓存余量
- 当前缓存预算未考虑 DiT 深度减少——可能是有意保守,但在极端设备上缓存可能略微不足

**`dit_layers = 0` 哨兵值**:
- 表示"使用默认/完整深度",是常见 C 习惯用法
- 但与有效层数语义混合,`#define H3_DIT_LAYERS_DEFAULT 0` 能让调用点意图更清晰

**整体评价**:
- 逻辑扎实,渐进降级策略结构良好
- 主要可操作项:溢出防护、误导性 `snprintf` 措辞、两个分支间 `use_int8_row_fc2` 不一致

# 高优先级问题修复记录(Phase 18, 2026-09-08)

审查发现的 5 项高优先级问题已全部修复。**共同特征:正常路径行为不变,是静默风险**
(边界条件触发,或代码演进时引入 bug)。

## 1. 视觉/音频阈值不一致 — `h3_dit_schedule.c:230-234`

```c
/* 修复前 */
if (visual_condition)  … = video >= 0.999f ? … ;   /* 带容差 */
if (audio_condition)   … = audio >= 1.0f   ? … ;   /* 要求精确相等 */

/* 修复后 */
if (audio_condition)   … = audio >= 0.999f ? … ;   /* 与 visual 一致 */
```

- 两者语义相同(判断 sigma 是否已归零 → 用真实行替代 condition 行),但阈值不同
- audio 用 `>=1.0f` 要求浮点精确相等,visual 用 `>=0.999f` 留了容差
- **风险**:浮点舍入下 audio 分支可能永不触发,导致两条 condition 路径行为不一致
- 统一为 `0.999f`(带容差更稳健,可吸收浮点误差)

## 2. `time` 张量生命周期脆弱 — `h3_dit_schedule.c`

- 原代码在 4 个错误路径各写一次 `h3_gpu_tensor_free(time); goto failed;`
- **风险**:新增任意 goto 若忘记 free → leak;若重复 free → double-free
- 修复:
  - 每处 free 后加 `time = NULL`(防 double-free)
  - `failed:` 标签加统一 `h3_gpu_tensor_free(time)`(NULL 安全,防 leak)
- 依据:`h3_gpu_tensor_free` 首行即 `if (!tensor) return;`(h3_gpu.m:852),NULL 安全

## 3 & 4. 分配大小溢出检查缺失

| 位置 | 表达式 | 修复 |
|---|---|---|
| `h3_dit_schedule.c:238` | `malloc(count * feature_dim * sizeof(*features))` | 加 `count > SIZE_MAX / feature_dim` |
| `h3_audio_vae.c:612` | `hidden_elements = STEREO * length * 8` | 加 `length > SIZE_MAX / ((size_t)STEREO * 8)` |

- 原检查用 `TIME_INPUT` 但实际分配用 `feature_dim`(不一致),且未防 `size_t` 回绕
- 注意运算符优先级:`SIZE_MAX / STEREO * 8` 会算成 `(SIZE_MAX/STEREO)*8`,**必须写**
  `SIZE_MAX / ((size_t)STEREO * 8)`

## 5. `use_int8_row_fc2` 语义不一致 — `h3_dit.c:2567`

```c
/* 修复前:存储时被 int8_mlp 静默门控,调用方传 1 也可能不生效 */
dit->use_int8_row_fc2 = dit->int8_mlp && use_int8_row_fc2;

/* 修复后:存原始值,语义 = "调用方请求" */
dit->use_int8_row_fc2 = use_int8_row_fc2;
```

- 门控由使用处外层 `if (dit->int8_mlp && …)`(h3_dit.c:3225)天然保证,无需重复
- 效果:读取代码时不再误解该字段是否已含 int8_mlp 前提

## 顺带清理:让仓库恢复可编译

- **`h3_gpu.m` 重复属性**:`stream_batch` / `stream_batch_pending` 被声明两次
  (HEAD 遗留,Phase 10 编辑残留)→ 编译直接失败。删除重复项。
- **废弃 GPU dequant 函数块**:`h3_gpu_blocking_weight_dequant_unrotate_int8` /
  `stream_dequant_begin|encode|submit` / `encode_wait_stream_event` —— Phase 10 已判定该方向
  不可行(共享 GPU 导致更慢),`h3_dit.c` 早已回到 CPU 蝶形,这些是死代码。删除。
  保留 `h3_gpu_linear_bf16_offset` 与 VDN 存根(VDN 路径需要)。

## 验证

| 项目 | 结果 |
|---|---|
| 编译 | ✅ 通过(binary 665184 bytes) |
| 端到端(256×256/1s/steps=4/seed 42/`--ssd-streaming`) | ✅ 产物 **193579 字节,与基线逐字节一致**,102s,零报错 |

→ 确认 5 项修复**无行为回归**。

## 教训:大文件参数名重构不要用全局字符串替换

中途试图用 `content.replace('h3_gpu *opaque,', 'h3_gpu *gpu,')` 修存根参数名,
**误伤了 58 个正常函数签名**(与函数体内 `H3GPU *gpu = GPU(opaque)` 冲突)。
最终改为 `git checkout` 干净重来 + 精确定位删除,一次通过。

**另一教训**:`h3_gpu.m` 的 HEAD 版本本身不可编译。修复前先用 `git stash` 验证过
「原始代码同样编译失败」,确认问题非本次引入 —— 仓库曾被提交在损坏状态,值得留意。

# 中低优先级清理记录(Phase 18b, 2026-09-08)

## `h3_cli.c`:`strdup` 失败未检查(审查记 2 处 → 实际 5 处)

排查发现审查低估了数量。逐个核实后修复 5 处:

| 行 | 变量 | 风险 | 修复 |
|---|---|---|---|
| 507 | `last_prompt` | **崩溃级**:失败后为 NULL,`repeat` 命令会 `strdup(NULL)` 未定义行为 | `strdup` 后加检查 + 警告 |
| 918 | `repeat` 处 | 同上(消费端) | `state.last_prompt ? strdup(...) : NULL` 双保险 |
| 852 | `sr_model`(初始化) | NULL 被后续 SR 逻辑使用 | 纳入启动失败检查条件 |
| 792 / 796 / 801 | SR `bin`/`model-dir`/`model` | 静默失败,用户以为设置成功 | 失败时 `fprintf` 报错 |

- 已有检查(无需改):276-281(引用拷贝)、349-352、400-403、856-859、918-919
- 教训:**审查记录的数量可能低估**,修复前应自行全量 grep 核实

## `h3_dit_schedule.c`:`weight_bf16_any()` 溢出防护

```c
/* 修复前:elements 累乘无溢出检查 */
for (…) elements *= shape[dimension];

/* 修复后 */
for (…) {
    if (shape[dimension] && elements > UINT64_MAX / shape[dimension]) {
        fail(…, "weight %s shape overflows", name); return NULL;
    }
    elements *= shape[dimension];
}
if (elements > SIZE_MAX / sizeof(float)) { fail(…, "too large to convert"); return NULL; }
```
- 防 `uint64_t` 累乘回绕,以及后续 `(size_t)elements * sizeof(...)` 溢出

## 已核实「无需修改」的项(避免无效改动)

- **`h3_audio_vae.c` `run_stage` 的 `(uint32_t)elements`**:审查怀疑 32 位截断。
  实际查看 533-538 行:
  ```c
  uint64_t elements64 = (uint64_t)STEREO * output_length * channels;
  if (elements64 > UINT32_MAX || elements64 > SIZE_MAX) { fail(…); return 0; }
  size_t elements = (size_t)elements64;
  ```
  **上游已保证 `elements <= UINT32_MAX`**,截断不可能发生 → 无需修改。
  → 教训:审查结论要回到代码验证,不能照单全收。

## 剩余未修(需重构或风险高于收益)
> 以下项在 Phase 18c 中已处理:`run_stage` 5→3(经分析不可行,改务实峰值优化)、`gate_score` 批量读回。

- `h3_cli.c`:巨型 if-else 链拆分、`INT32_MAX` 帧上限、退出清理(纯结构改进)
- `h3_cli.c`:TOCTOU 竞争、SR 可执行文件路径注入(需改成受控执行,改动面较大)
- `h3_audio_vae.c`:`run_stage` qkv slicing(收益极小:仅省 `attention_elements*3*sizeof(f32)`≈几百 KB,且需新增 GPU weight-offset 线性支持)
- `h3_dit.c`:`dit_layers = 0` 哨兵值语义、缓存预算顺序(设计澄清)

## 验证

| 项目 | 结果 |
|---|---|
| 编译 | ✅ 通过(665184 bytes) |
| 端到端(256×256/1s/steps=4/seed 42/`--ssd-streaming`) | ✅ **193579 字节,与基线逐字节一致**,113s,零报错 |

→ 中低优先级清理同样**无行为回归**。

# 内存优化记录(Phase 18c, 2026-09-08)

## `h3_audio_vae.c`:`run_stage()` 峰值内存(`5→3` 不可行的务实改法)

审查建议「5→3 tensor」。逐项分析 5 个 tensor 的生命周期后确认**不可行**:
- `upsampled`:在 3 个 block 各被 copy 一次(作为 block 1/2 的 weight 重置源),必须保留到 block 2
- `sum`:block 0 的 target + 后续累加器
- `work`:block 1/2 的 target(承载 `U + Rk` 后并入 sum)
- `activated` / `branch`:残差计算的乒乓缓冲,算法需要

但发现**真实的峰值浪费**:上一层 `audio->hidden` 在 upsample 之后即不再需要,原代码却保留到块循环结束,
使峰值 = 5 个 stage tensor + 上一层 hidden = **6 份**。

**修复**:上采样后**立即 `h3_gpu_submit` 并 `free_tensor(&audio->hidden)`**,再分配块循环所需的
`sum`/`work`/`activated`/`branch`。峰值从 ~6 份降到 5 份(省一份与 stage tensor 等大的上一层 hidden),
多一次 GPU submit(微秒级),命令序列与数学完全等价。

## `h3_dit_schedule.c` + `h3_dit.c`:`gate_score` 批量读回

原 `h3_dit_schedule_gate_score()` 每调用一次都 `malloc` + GPU readback + `free`。
排序路径(`configure_gate_ranked_blocks`,仅在非 uniform policy 且 active block 数 < 总数时触发)
对 47 个 block 各调一次 → **47 次 malloc/free**。

**修复**:新增 `h3_dit_schedule_gate_scores(schedule, first, count, out)`,内部**复用单个读回缓冲**,
排序路径一次性算出全部分数(47 次 malloc → 1 次)。单次版改为委托批量版,消除重复计算逻辑。

```c
int h3_dit_schedule_gate_scores(const h3_dit_schedule *schedule,
                                unsigned first, unsigned count, double *out);
/* 复用单一读回缓冲;ranking 全栈从「每 block 一次 malloc/free」降为「一次性分配」 */
```

## 验证

| 项目 | 结果 |
|---|---|
| 编译 | ✅ 通过(665232 bytes) |
| 端到端(256×256/1s/steps=4/seed 42/`--ssd-streaming`) | ✅ **193579 字节,与基线逐字节一致**,105s,零报错 |

→ 内存优化**无行为回归**。

# 审查后修复记录(Phase 18d, 2026-09-08)

对前一轮设计/普通/性能审查发现的问题执行修复,并在 B3 清理过程中发现一个新的真实并发缺陷。

## 已修复

### B1(高·休眠 → 拔除):VAE 异步预取崩溃地雷
`h3_video_vae.c` 的 `vae_prefetch_thread` 在共享 `vae->gpu` 上 `load_block` 分配张量,而主线程此时持有 open 命令缓冲解码 block N。代码注释明确承认会
"races and crashes (objc_retain of a dangling pointer)"。仅 `H3_VAE_PREFETCH=1` 可触发,默认关闭。
**修复**:删除 `vae_prefetch_thread` / `vae_prefetch_enabled` / `vae_prefetch_job` 及循环内双缓冲预取逻辑,
改为串行 `load→run→free` 每块的循环。因原串行分支循环内本无 `load_block`(块加载依赖预取线程),
故在循环内改回串行加载,确保每块都被加载。一并移除已无引用的 `#include <pthread.h>`。
**附带消除 B2**:原预取失败路径 `if (job.ok) free_block(...)` 在失败时泄漏 `vae->blocks[next]`,随预取代码整体删除而消除。

### D9(中):删除未实现且误导的 GPU 属性
`h3_gpu.m` 的 `stream_batch` / `stream_batch_pending` / `streamEventValue` 属性全程无函数体引用,
其上方「Batched streaming ConvRot dequant … 一次 submit rather than four」注释描述了一个 Phase 10 已回退、
从未实现的机制。`streamEvent` 保留(在 init 中仍有 `newEvent` 分配)。

### B3(中):修正 `read_stream_layer` 误导性注释
原注释称「no GPU commands from this thread」「GPU work here is only the M5 … requantization below」,
且 `int gpu_work = …; (void)gpu_work;`。**核实发现 `gpu_work` 并非死变量**:它在 `h3_dit.c:1467`
(`h3_gpu_begin`)与 `:1483` (`h3_gpu_submit`) 被使用,中间 `:1472-1482` 多次 `h3_gpu_quantize_weight_int8`。
即 int8 启用时流式线程**确实在共享 `dit->gpu` 上发命令**。修正注释以准确反映该行为,并移除无意义的 `(void)gpu_work`。

## 新并发缺陷(高·条件触发):DiT 流式线程竞态共享 `dit->gpu` 命令缓冲

在 B3 清理中发现的真实问题,推翻了前一轮「子代理误报」的结论:

- **位置**:`h3_dit.c` 主线程序环 `:3519-3596`(创建流式线程 → `run_block(block)` → `h3_gpu_submit` → `pthread_join`)
  与流式线程 `read_stream_layer` 的 int8 requant 段 `:1467-1483`。
- **机制**:当 `gpu_work = int8_mlp || int8_qkv || int8_attention_out` 为真时,流式线程在共享 `dit->gpu` 上
  调用 `h3_gpu_begin`(设 `gpu.command`)、`h3_gpu_quantize_weight_int8`(编码到 `gpu.command`)、`h3_gpu_submit`
  (提交 `gpu.command`)。而主线程**并发**在 `run_block` 中使用同一 `dit->gpu` 的命令缓冲。
- **后果**:`h3_gpu_begin`(h3_gpu.m:943)在 `gpu.command != nil` 时直接 `return 0` → 流式线程或主线程的
  begin 失败;或两者交替写同一 `MTLCommandBuffer` / 递增 `gpu.stats`(非原子),造成静默损坏或崩溃。
- **触发条件**:int8 流式 requant 启用时(本机 M4 上 `int8_mlp` 等可能因内存规划 `use_int8_row_fc2` 等独立
  于 NAX 被开启)。默认 bf16 路径 `gpu_work=0`,不触发。
- **根因与历史**:D9 删除的 `stream_batch` 私有命令缓冲**正是为隔离该竞态预留的**,但从未实现 —— 流式线程
  因此一直复用调用方的 `command` 缓冲。
- **建议修复(后续)**:
  1. 给流式线程一个**独立私有命令缓冲**(重建 `stream_batch` 机制,MPS/compute 各自 begin/commit,用
     `MTLEvent` 排序 requant 结果对主线程的可见性);或
  2. 将 int8 requant 从流式线程**移到主线程 join 之后**执行(牺牲少量重叠,换取绝对正确);
  3. 至少将 `gpu.stats` 改为原子累加 + 给 `h3_gpu_begin/submit` 加可重入锁作为兜底。
- **修复(Phase 18e, 2026-09-08)**:采用方案 2 —— 将 int8 requant 从流式线程**移到主线程 `pthread_join` 之后**,彻底消除共享命令缓冲竞态。
  - 新增 `requant_stream_slot(dit, slot, error, error_size)`:在主线程对刚流式加载的 slot 做
    `h3_gpu_begin` + 3×`h3_gpu_quantize_weight_int8` + `h3_gpu_submit`,使用主线程独占的命令缓冲。
  - `read_stream_layer`(流式线程)删除全部 GPU requant 代码(原 1467-1487)与 `gpu_work` 局部变量;
    现在流式线程**只做 CPU 反量化 + LoRA 合并**,完全不触碰 `dit->gpu`。
  - 主循环在 `if (!stream_job.ok) { ... }` 之后、`dit->stream_ready_slot` 更新之前调用
    `requant_stream_slot(dit, &dit->stream_slots[stream_job.slot], ...)`;此时主线程已 `h3_gpu_submit`
    且流式线程已 join → 无并发,requant 与后续 `run_block` 完全串行。
  - **运行时验证**:`fused_mlp=1`(默认)、`use_slower_bf16_mlp=0`(默认)、M4 `tensorOpsEnabled=true`
    → `int8_mlp=1` → 该路径在 M4 默认测试(`--ssd-streaming`)中**实际运行**;修复前后产物均 **193579 字节一致**,
    证明主线程串行 requant 产出正确且一致输出。
- **当前状态**:**已修复并运行时验证**(非"未实现")。原竞态(两线程并发编码同一 `MTLCommandBuffer`)已彻底消除。

## 验证

| 项目 | 结果 |
|---|---|
| 编译 | ✅ 通过(严格 `-Wall -Wextra -Wpedantic -Wshadow -Wconversion` 无警告,binary 664672 bytes) |
| 端到端(256×256/1s/steps=4/seed 42/`--ssd-streaming`) | ✅ **193579 字节,与基线逐字节一致**,120s(串行 VAE 解码仅轻微变慢),零报错/崩溃 |

# 分支新代码审查 + AudioVAE 端到端验证 (2026-09-11 session)

审查对象:`feature/lora-merge` 相对 `origin/main`(merge-base `92a932c`)的新增 C 代码,
优先覆盖此前未审的 LoRA 合并链路(`h3_lora.c` / `h3_gpu.m` 的 LoRA GEAM / 调用点)与新的 `h3_superres`。

## LoRA 合并链路

### 适配器名三入口不一致 → turbo 适配器静默跳过(🔴,已修)
- `h3_lora_matches()` 用文件解析出的 `lora->adapter`;`h3_lora_merge_blocking()` 传 `NULL` 回落
  `lora->adapter`;而 `h3_lora_apply()` **硬编码 `"default"`**。
- `lora_merge()` 在因子缺失时 `return 1` 是**合法 no-op**(VDN 的 6 个 target 里 `attn.to_q` 与
  `attn.orig.to_q` 必然只有一个存在),所以「找不到」是**静默**的。
- 后果链(真实可触发):turbo LoRA 的键是 `...lora_A.turbo.weight`;
  `merge_adaln_loras()`(`h3_dit_schedule.c:429-437`)先用 `matches`(turbo 命中 → 通过并 **无告警**),
  再用 `apply`(查 `default` → 查不到 → `return 1`)→ **turbo 的 AdaLN/final-layer 适配被丢弃且无提示**。
- 另有路径分叉:常驻加载 `h3_dit.c:885` 传 `blocking=0`(走 `h3_lora_apply`),流式 `h3_dit.c:1501`
  传 `blocking=1`(走 `h3_lora_merge_blocking`)→ 同一份 LoRA 在 `--ssd-streaming` 与非流式下**权重不同**。
- 修法:`h3_lora_apply` 改传 `NULL`(与另两入口统一);显式指定 adapter 仍用 `h3_lora_apply_named`。
- 教训:同一语义若有多个人口,解析规则必须**同源**;「静默 no-op」与「条件跳过」不能共用一个返回值。

### `h3_gpu_begin` 命令缓冲泄漏(🔴,已修)
- `h3_lora.c` 非阻塞分支:`h3_gpu_begin` → `h3_gpu_lora_geam_bf16` → `h3_gpu_submit`,
  但该 GEAM **自建私有 command buffer 并 `waitUntilCompleted`**,根本不使用 `gpu.command`。
- 失败时跳过 `h3_gpu_submit` → `gpu.command` 永不置空;而 `h3_gpu_begin`(h3_gpu.m:939)首行
  `if (!gpu || gpu.command || gpu.inflightCommands.count) return 0;` → 之后**所有** GPU 阶段失败,
  真实根因被 「cannot create Metal command buffer」掩盖。全库确认**没有** `h3_gpu_discard/abort` API
  可用于放弃已开缓冲。
- 且 `h3_gpu_begin` 本身可能失败(调用方已持有 open 编码器,如 denoise prime block)——正是该函数注释
  自己描述的场景 → 把本可成功的 LoRA 合并误报为失败。
- 修法:删除该 begin/submit 包装(GEAM 自带等待语义)。

### 重复实现(🟡,已修)
- `h3_gpu_lora_geam_bf16` 与 `h3_gpu_blocking_lora_geam_bf16` 函数体**逐行等价**(仅错误串不同),
  但头文件把它们描述成两种语义 → `h3_lora.c` 的 `blocking` 开关实际没有区别,且任何修复要改两处。
- 修法:后者保留为实现,前者改为转发;注释合并到保留实现上。

### 防护性补充(🟡,已修)
- `h3_lora.c`:`rank*in_dim` / `rows*rank` / `in_dim*rank` 三个分配大小加 `SIZE_MAX` 防护
  (并先排除 `in_dim/rows == 0` 以防止新增检查自身除零);`rank` 改为在 `dtype/ndim` 校验之后读取并拒绝 `rank==0`。
- `h3_ffmpeg.c::h3_superres`:新增 `remove_tree()`(`/bin/rm` 绝对路径 + 失败 `fprintf` 告警)替换
  5 处静默 `rm -rf`(临时目录可含数十 GB PNG 帧);`waitpid` 非 EINTR 失败时 SIGKILL + 回收(避免孤儿);
  `bin/mbin/mpar` 三处 `snprintf` 加截断检查(`indir/outdir` 由 `mkdtemp` 生成、长度有界 → 未加,避免死代码)。

### 核实为「非缺陷」(避免无效改动)
- `h3_superres` 的 `target_height % inner_h` 看似可能除零 **不可达**:
  `h3_ffprobe_visual_size` 成功返回前强制 `parsed_width >= 1 && parsed_height >= 1`(h3_ffmpeg.c:144-148)。
- `h3_ffmpeg.c` 的 `enc_argv[40]` 最多用 31 项(含 NULL),无数组越界。
- `lora_merge` 的三条失败路径与成功路径都正确释放 host 缓冲与两个 GPU 张量 →
  **无泄漏 / 无 double-free**;`h3_lora_geam_bf16` 内核的 tile 装载、`fma` 索引与两处 barrier 均正确
  (无早退,`row/column` 边界齐备)。

## AudioVAE:submissions 计数推导(解释一条失效断言)
- `tests/test_real_audio_vae.c` 原断言 `submissions != 16`;用官方权重实跑得 **23**。
- 推导:`16 = 1`(`submit AudioVAE input`)+ `7`(`submit AudioVAE stage normalization`,STAGES=7)
  + `7`(`submit AudioVAE stage`)+ `1`(`submit AudioVAE output`);
  Phase 18c 的内存优化在上采样后**立即 submit 并释放上一层 hidden**(h3_audio_vae.c:557)
  → 每 stage +1 → **16 + 7 = 23**。MPS conv 数 **136 不变**。
- 结论:`+7` 是**有意**优化,断言未同步 → 已改为 23 并把推导写进注释。
- 教训:断言里的魔法数字必须可推导(注释写清来源),否则一次内存优化就会让它失效;
  而该测试因 fixture 缺失长期 skip,失效很久未被发现。

## 本地权重与验证入口事实
- `models/minimax-h3/FL2VA/*` 是指向 `/Users/jay/h3_sys/MiniMax-H3-Convrot/…`、`/Volumes/data/.lmstudio/models/…` 的**符号链接**。
- `audio_vae/model.safetensors`(577 MB)**1087 个张量全 F32**,不是 int8/convrot 变体
  → 解码不会引入额外反量化提交(已排除"23 来自反量化"的假设)。
- 仍缺 `misc/fixtures/h3_real_audio_vae_37.safetensors`(MLX oracle)→ **数值 parity 未验证**;
  刻意**未**用 native 输出反造 fixture(那会让测试退化为只验确定性 = 自证)。
- `make test` 的守卫查 `MiniMax-H3/…`,与实际 `models/minimax-h3/…` 不符 → 相关条目默认 skip;
  新测试改用 `AUDIO_VAE_MODEL ?= MiniMax-H3` 变量(`make test AUDIO_VAE_MODEL=models/minimax-h3`)。

## 实测数据(AudioVAE e2e,合成 latent,官方权重)
| 项 | 值 |
|---|---|
| 形状 | 2ch / 29600 samples @ 32000 Hz(latent 37)|
| 统计 | peak 0.333697,rms 0.0532564,全 finite |
| 资源 | 0.543 GiB,0.053 GPU s |
| 结构 | 136 MPS conv,23 submissions |
| 确定性 | 两次解码**逐字节一致** |
| 边界 | latent 1 → 800 samples、latent 2 → 1600 samples(覆盖 `input_length-1==0`)|

## 既有缺陷(编译警告暴露,已修)
- `h3_audio_vae.c::run_stage`:`int ok = …` 位于 `goto done` **之后** → 分配失败时 `return` 未初始化值
  (`-Wsometimes-uninitialized`)。修法:`int ok = 0;` 提到失败分支之前。
- `h3_audio_vae.c::decode_output`:`audio->length` 是 `uint32_t`,与 `SIZE_MAX/(STEREO*8)` 比较在 64 位下
  **恒为假** → 该溢出检查**实际无效**(Phase 18 的修复选错了比较对象)。
  修法:先按 `uint64_t` 计算 `hidden_elements64` 再校验 `> SIZE_MAX` 后转 `size_t`(与 `run_stage` 既有写法一致)。
  → 再次印证:修复溢出检查时要选**与被乘数实际宽度匹配**的上界,否则只是"看起来有检查"。
- `tests/test_lora.c`:未使用的 `gpu` 形参 + `printf("%zu", lora ? 0 : 0)` 格式不匹配且恒打印 0(已删)。

# 流式 video VAE 解码命令缓冲 bug 修复 (2026-09-11)

## 现象
ComfyUI `H3_BinaryT2V` 报 `RuntimeError: H3 引擎失败 (rc=1)`,日志尾部 `audio VAE 7/7` 之后
`h3: begin streamed video VAE transformer block: unknown Metal error`。用官方权重直连 `./h3`
复现,确认是 video VAE **流式解码**路径(由自动内存规划器在权重放不下常驻时开启 `vae->streaming`)
在第一个 block 即失败。

## 根因(GPU 命令缓冲不变量)
- `h3_gpu_begin`(`h3_gpu.m:937`):`if (!gpu || gpu.command || gpu.inflightCommands.count) return 0;`
  —— 命令缓冲已开时**直接返回 0 且不设 `lastError`**,调用方 `h3_gpu_error` 于是回退成
  `"unknown Metal error"`。
- `h3_gpu_submit`(`h3_gpu.m:976`):提交后 `gpu.command = nil`,**不重新打开**(重新打开只在
  `h3_gpu_continue:966`)。所以两阶段之间若要继续编码,必须下一次 `h3_gpu_begin` 重新开缓冲。

`run_stream_tile`(`h3_video_vae.c`)原代码缺配对:
- L614 `h3_gpu_begin("begin streamed video VAE decoder")` 编码 prep ops(L615-628),**此处缺一次 submit**
  → `gpu.command` 一直开着;
- 循环 L638 又 `h3_gpu_begin("begin streamed video VAE transformer block")` → 已开 → 返回 0 →
  "unknown Metal error",整个解码在第一个 block 挂掉;post ops 也因无 begin 而无缓冲。

对照 `run_decoder`(resident,正常):每阶段各有 begin/submit(prep→submit→每 block begin/submit→
output begin/submit)。原 `run_stream_tile` 是 Phase 18c/18d 把异步预取线程改串行时只保留了循环里的
per-block begin,却漏了 prep 之后的 submit 与 output 之前的 begin。

## 修复(镜像 `run_decoder`)
- prep ops 后加 `h3_gpu_submit("submit streamed video VAE prep")`
- 循环里保留每 block 的 `h3_gpu_begin` + `h3_gpu_submit`(per-block 提交让 `free_block` 在该 block
  的 GPU 完成后立即回收权重,保留流式省内存语义)
- post ops 前加 `h3_gpu_begin("begin streamed video VAE output")`

## 验证
- 直连 `./h3`(与 ComfyUI 节点等价参数):`audio VAE 7/7 → FFmpeg 39/39 → wrote mp4`,270 KB,
  ffprobe 确认 448×256 / 39 帧 / 1.625s / 含音频流;日志无 Metal/error 串。
- 全量 `make -j8 all test` 0 警告;`h3_tests`(1768 checks)/`h3_audio_gpu_tests`/
  `h3_real_audio_vae_e2e_test` 全绿。

## 教训
GPU 命令缓冲的 begin/submit 必须**成对且各阶段独立**:`h3_gpu_submit` 不留开缓冲,任何
"在已开缓冲上再 begin"或"submit 后未 begin 就继续编码"都会静默失败成 "unknown Metal error"。
这与 AudioVAE 的 `submissions==23`(Phase 21)同源——都是串行化改造时漏改一处配对导致的隐性 bug。

# ComfyUI 子进程找不到 ffmpeg (2026-09-11)

## 现象
video VAE 流式解码修复后,ComfyUI 跑到 `FFmpeg 0/56` 报 `h3: cannot start FFmpeg: No such file or directory`。
直连 `./h3`(agent shell PATH 含 ffmpeg)能完整产出 → 是 ComfyUI 子进程 PATH 问题。

## 根因
引擎在 mux/读取参考媒体时用 `posix_spawnp(ffmpeg_program(), ...)`(h3_ffmpeg.c,共 8 处),依赖 `environ`
的 PATH 查找可执行。`ffmpeg_program()`/`ffprobe_program()` 优先取 `H3_FFMPEG`/`H3_FFPROBE`,否则裸名
`"ffmpeg"`/`"ffprobe"`。ComfyUI 由 GUI/launchd 拉起时 PATH 仅含 `/usr/bin:/bin` 等,不含本机
`/opt/zerobrew/bin/ffmpeg`(及 `/usr/local/bin` 软链、`/opt/homebrew/bin`)→ `posix_spawnp` 返回 ENOENT。

## 修复(集成层,不碰引擎)
`comfyui_nodes/h3_binary.py`(`/Volumes/data/Documents/ComfyUI/custom_nodes/h3_binary_nodes/`,**不在 h3.c
仓库内**)的 `_run_engine` 在拼装子进程 env 时先调 `_resolve_ffmpeg_env()` 把 ffmpeg/ffprobe 绝对路径注入
`H3_FFMPEG`/`H3_FFPROBE`:先 `shutil.which(name)`,失败再扫 `/opt/zerobrew/bin`、`/usr/local/bin`、
`/opt/homebrew/bin`、`/opt/local/bin`、`/usr/bin`;用户已在环境设置则跳过。无论 ComfyUI 的 PATH 多精简,
引擎都用绝对路径拉起 ffmpeg。

## 验证
`env -i PATH=/usr/bin:/bin H3_FFMPEG=/opt/zerobrew/bin/ffmpeg H3_FFPROBE=/opt/zerobrew/bin/ffprobe ./h3 \
 -d ... --width 448 --height 256 --seconds 1 --steps 4 -o /tmp/h3_verify2.mp4` 完整跑通
(`FFmpeg 39/39 → wrote /tmp/h3_verify2.mp4`,270 KB,与正常 PATH 一致),无 "cannot start FFmpeg"。
节点 `python3 -m py_compile` 通过。

## 教训
引擎把外部工具(ffmpeg)的查找交给 PATH + 可选环境变量覆盖(`H3_FFMPEG`/`H3_FFPROBE`),是刻意解耦设计;
GUI/launchd 启动的程序 PATH 极简,集成层(ComfyUI 节点)有责任把绝对路径注入子进程环境,而非依赖 PATH。
这是"集成环境 ≠ 开发 shell 环境"的典型坑——验证时务必用 `env -i` 最小 PATH 复现,而非只在自己 shell 里跑。

# 空白提示词产生"编辑器截屏"式画面 (2026-09-11)

## 现象
ComfyUI 生成的视频"和提示词无任何关系",画面近黑、像暗色编辑器截屏。

## 诊断
对比帧统计(mean / bright>200占比 / 竖向边缘gy):
- 用户 ComfyUI 输出: mean=28.9 / bright=0.000 / gy=28.93
- 空白提示词 `-p "   "`: mean=49.8 / bright=0.008 / gy=49.76
- 正常提示词"fox": mean=138.7 / bright=0.048 / gy=138.71

空白提示词与用户输出**同形态**(近黑+竖向边缘主导+无亮像素),证实为同一成因。
引擎 prompt 非空检查是 C 字符串 `!*prompt`(h3.c:1218),纯空格通过该检查,但 tokenize 后
嵌入近零 → DiT 跑无条件生成 → 输出模型无条件先验(近黑+竖向结构,恰似暗色代码编辑器)。
ClipProj 路径下 `FL2VA/text_encoder` 权重不加载(h3.c:774-796,容忍缺失),与本次无关。

## 修复
集成层(`comfyui_nodes/h3_binary.py`,不在 h3.c 仓库)_build_cmd 里 `prompt.strip()` 校验,
空白/纯空格抛 ValueError 给出清晰提示。**未改引擎**——引擎的空检查对 CLI 交互式输入足够,
ComfyUI 节点作为集成层应替用户拦住无效输入,避免白等 2 分钟产出垃圾。

## ComfyUI 自定义节点（2026-09-11）

`comfyui_nodes/` 把本项目的 `h3` 二进制封装为 ComfyUI 节点，一节点完成
「文本/图像/参考 → 视频+原生音频(MP4)」全链路，可替换官方工作流的整条采样链。

| 文件 | 说明 |
|------|------|
| `comfyui_nodes/h3_binary.py` | `H3_BinaryT2V` / `H3_BinaryR2V` / `H3_BinaryInfo` |
| `comfyui_nodes/__init__.py` | 节点注册（`NODE_CLASS_MAPPINGS`） |
| `comfyui_nodes/README.md` | 安装、参数、前置条件 |
| `gen_comfyui_workflows.py` | 生成 `h3_binary_t2v.json` / `h3_binary_r2v.json` |

### 关键约束（实现时踩过的坑）
1. **cwd 必须是二进制所在目录**：引擎用相对路径 `"h3_shaders.metal"`
   （`h3.c:1071` 等）运行时编译 Metal kernel，否则报
   `cannot compile h3_shaders.metal`。节点已把子进程 `cwd` 设为该目录。
2. **`--steps` 范围 `[2, 1000]`**（见 `h3_cli.c` 校验）。
3. **宽高必须为 32 的倍数**：`h3_frame_grid` 要求 `latent_h/latent_w` 为偶数，
   而 latent = 像素/16。
4. **`H3_CLIPPROJ_DIR` 设置后 `FL2VA/text_encoder` 可缺失**
   （`h3.c:774-797`：4B+ClipProj 路径替代 50 层编码器）。
5. 输出 `VIDEO` 用 `comfy_api.latest.InputImpl.VideoFromFile(path)` 包装，可直连 `SaveVideo`。

### 实测
```
256×256 / 0.5s / 2 步 / core-reuse 4            → 57.8s
256×256 / 0.5s / 4 步 + turbo LoRA / core-reuse 4 → 69s（auto_steps 自动 20→4）
```

# 生成画面与提示词无关 / 节点无 prompt 输入框 (2026-09-11)

## 现象
ComfyUI 跑 `H3_BinaryT2V` 产出的视频与提示词毫无关系，画面像「编辑器截屏 / 近黑文字帧」；
节点上**看不到 prompt 文本框**（只有一块空白 + 左侧一个多余输入点）。

## 根因（工作流接线错误，不是引擎/节点 bug）
用户工作流 `user/default/workflows/h3_binary_t2v.json` 里，`H3_BinaryT2V`(node 31) 的
`prompt` 控件被 **Convert to Input** 并连了线：
```json
{"name":"prompt","type":"STRING","widget":{"name":"prompt"},"link":2}   // prompt 已变成输入
"links":[[2, 10, 0, 31, 2, "STRING"]]                                    // 来自 node10=H3_BinaryInfo 的输出0(info)
"widgets_values":["", 448, 256, ...]                                     // prompt 本身为空 ""，被连线覆盖
```
即把 **`H3_BinaryInfo`（运行 `h3 --info` 打印设备/权重清单）的整段日志文本**接到了 prompt。
→ 引擎收到的提示词 = `h3 --info` 的文字转储，于是生成了「一堆文字」般的画面。
同时因为控件被转成输入，节点上的文本输入框消失 → 用户以为「没有输入框」。

## 判定方法（无需看界面）
`GET /object_info/H3_BinaryT2V` 显示 `input_order.required` 第一项就是 `prompt`（`multiline:true`），
说明节点定义正常；再读工作流 JSON 看到 `prompt` 带 `link` 即确诊接线导致。

## 修复（界面 3 步）
1. 删除 `H3_BinaryInfo.info → H3_BinaryT2V.prompt` 的连线。
2. 右键 T2V 节点左侧 `prompt` 输入点 → **Convert Input to Widget**（文本框恢复）。
3. 输入真正提示词后重跑。
`H3_BinaryInfo` 的输出只应连到预览/文本显示节点或留空看日志，不得接入 prompt。

## 教训
- ComfyUI 里控件的 `widget`+`link` 同时存在 = 该控件已被 Convert to Input；界面上文本框会消失，
  排查「找不到输入框」应优先怀疑此项。
- ComfyUI 会给 `seed` 自动追加 `control_after_generate` 控件（中文界面显示「生成后控制」，
  值 `randomize/fixed/…`），**不是**自定义节点定义的控件，勿据此误判节点版本。
- 「引擎参数校验非空」拦不住此类错误：prompt 非空（是 info 文本），故生成照常进行。

# Metal 编译缓存调研 (2026-09-12)

## h3.c 现状（源码级确认）
- `h3_gpu_create()`（`h3_gpu.m:338-550`）流程：读 `h3_shaders.metal` → 构造
  `MTLCompileOptions`（`mathMode = MTLMathModeSafe`，可选宏 `H3_METAL_HAS_TENSOR`）→
  `newLibraryWithSource` → 逐个 `newFunctionWithName` + `newComputePipelineStateWithFunction`
  建 ~63 个 pipeline。TensorOps 编译失败会**回退重编一次**（不带宏）。
- **无任何持久化**：全库无 `MTLBinaryArchive` / `newLibraryWithFile` / `metallib` /
  `newDefaultLibrary`。`gpu.pipelines` 是普通 `NSDictionary`，每次 create 全量重建。
- **同进程重复编译 7 处**：`h3_dit.c:2556`、`h3_text_encoder.c:520/1064`、
  `h3_video_vae.c:1029/1174/1334`、`h3_audio_vae.c:708/1322`、`h3_video_encoder.c:768`。
- **唯一查询入口**是 `h3_gpu_pipeline(gpu, name)`（`h3_gpu.m:249`）→ 所有 dispatch 都经它，
  因此 lazy 化只需改这一个函数（外加 create 里的 eager 循环）。
- **线程安全隐含约束**：LoRA 流式合并在 SSD 预读线程上跑
  （`h3_gpu_blocking_lora_geam_bf16`），所以 `h3_gpu_pipeline` 可能被后台线程调用 →
  lazy 缓存必须加锁（不能沿用「建好后只读」的无锁假设）。
- `h3_gpu_free`（`h3_gpu.m:552`）会把 `library` / `pipelines` 逐个置 nil →
  共享模块必须自己持有 library 所有权，不能依赖 H3GPU 实例存活。
- 调用点 7 处但**实际一次典型 T2VA 跑几次**需实测（DiT 1 + 文本编码器 1~2 + video VAE 1 +
  audio VAE 1，video encoder 仅 Ref2VA 用）。

## DeepJIT 可借鉴点
- cache key = 源码 + 追踪的 include 树 + 编译器版本 + 有效编译选项 + 应用提供的依赖签名
- 两级缓存（进程内 + 磁盘）；lazy init 延迟设备/编译器发现
- 磁盘条目用「唯一临时目录 + fsync + 原子 rename」发布，支持多用户/多进程共享同一 cache
→ h3 对应物：源码内容 hash、macOS 版本 + `device.name`（registryID）、`H3_METAL_HAS_TENSOR` 宏状态。
→ Metal 侧实现路径：进程内静态模块缓存（改动 1）+ `MTLBinaryArchive`（改动 2）。

## 预期与风险（待实测修正）
- 改动 1 收益明确（消除 N-1 次源码编译），风险低。
- 改动 2 需实测：`MTLBinaryArchive` 主要省 **pipeline 链接/后端编译**，
  **未必省 `newLibraryWithSource` 的 front-end 源码编译** → 若实测无收益则弃用并记录。
- 改动 3 若 pipeline 构建本身耗时小，收益有限；反之可考虑提升为进程级共享。

## 实测结果（判决性）：假设被推翻

### 微基准 `./h3_metal_bench`（同进程连续建上下文）
| 场景 | library（源码→MTLLibrary） | pipelines（86 个） | 单次合计 |
|---|---|---|---|
| 暖缓存（既有源码） | 0.001–0.003 s | 0.001–0.005 s | **~5 ms** |
| **冷缓存**（`cp` 到 /tmp 后追加一行注释 → 内容变化） | **0.226 s** | 0.002 s | ~0.23 s |
| 对同一份「冷」源码**再跑一次** | 0.001 s | 0.002 s | ~3 ms |
| 进程内第 0 个上下文（含设备初始化） | — | — | 31–51 ms（一次性，与源码无关） |

**机制确认**：Metal 有**系统级、按源码内容命中**的 shader 缓存
（`/private/var/folders/f8/…/C/com.apple.metal`）。同内容第二次编译 0.226 s → 0.001 s。
「86 个 pipeline 只要 2 ms」也说明后端编译被驱动延后/缓存，不是我们在付钱。

### 真实引擎跑（`H3_PROFILE=1 ./h3 -d models/minimax-h3 -p "…" --width 256 --height 256 --seconds 0.5 --steps 2`）
```
count=4   # 一次 T2V 只创建 4 个 Metal 上下文
shader-build  wall=0.013s library=0.005s pipelines=0.007s kernels=86   # 第 1 个
shader-build  wall=0.005s library=0.003s pipelines=0.002s kernels=86
shader-build  wall=0.003s library=0.001s pipelines=0.002s kernels=86
shader-build  wall=0.002s library=0.001s pipelines=0.001s kernels=86
合计 = 23 ms   而同一个 run 的 H3 DiT total wall = 34.680s
```

### 结论
- 三项改动（共享 MTLLibrary / MTLBinaryArchive / lazy pipeline）上限收益 **≈10–15 ms 每次运行（0.04%）**，
  整机冷缓存时也只有一次性 ~0.9 s。**全部不做。**
- **DeepJIT 的磁盘缓存不可移植到 Metal**：CUDA 没有系统级 shader 缓存，Metal 有。
  自己再实现一层 `MTLBinaryArchive` 只是重复 OS 已有机制。
- 该结论**仅在 macOS + Metal 成立**；若将来出现非 Apple 后端（如 CUDA/Vulkan），可重新评估。
- 顺带观察（本次 profile，M4 / 256×256 / 2 steps）：DiT `wait=13.244s`、`root-gpu=2.992s`、
  `encode=0.132s` → **GPU 实际计算只占 3s**，而 wall 34.7s 里大半是 I/O 等待与主机侧编排。
  真正值得优化的是这里，不是 shader 编译。

# 新目标：DiT 去噪的流式权重管线 (2026-09-12)

## 实测 breakdown（M4 / 256×256 / 0.5s / steps=2 / 自动内存规划 → SSD 流式）
```
Euler denoise wall = 33.189 s
  wait (GPU)       = 13.178 s
  root-gpu         =  2.909 s   ← GPU 实际计算只占 8.8%
  encode           =  0.064 s
BF16 SSD stream 36.248 GiB in 33.569 s (1.080 GiB/s)
  unhidden wait    = 19.945 s   ← 主线程阻塞在 pthread_join(prefetch)，占去噪 60%
  pread            = 17.568 s   (52%)
  cpu-unrotate     = 16.000 s   (48%)
```
**17.568 + 16.000 = 33.568 s == `stream_read_seconds`** → 单线程内「读盘 → CPU 反旋转」**严格串行**，
零重叠。这就是全部问题。

## 结构事实（源码级）
- `read_stream_layer`（`h3_dit.c:1379`）每 block 处理 ~4 个 source（qkv/out/fc1/fc2），逐个：
  `load_convrot_scale_values` → `convrot_read_weight`（**每次 open+pread+close**）
  → `convrot_unrotate_cpu`（WHT 蝶形，1024 行/块）→ `h3_gpu_tensor_write_bf16`（写 2× 字节进共享 slot）。
- **每 block 起一个线程**（`h3_dit.c:3607` 附近 `pthread_create(&stream_thread, …, read_stream_layer_thread, …)`），
  主线程 `h3_gpu_submit` 后 `pthread_join`。`sample` 在 10.16s 窗口内抓到 **~37 个
  `read_stream_layer_thread`**，每个存活 ~300 ms → 串行、无跨 block 流水。
- 权重文件 `minimax_h3_fastvideo_4step.safetensors` = **21.33 GiB，其中 I8 占 19.74 GiB**
  （250 个 I8 + 250 个 U8 comfy_quant + F32 scale；BF16 仅 1.49 GiB）→ 已经在走 convrot int8 路径，
  **I/O 字节数已经是最小可用形态**（没有可换的 fp8/int4 导出）。
- 权重在**内置盘** `/dev/disk3s5`（`/Users/jay/h3_sys/…`），非外接盘。该卷用 `disk_speed` 实测顺读
  **2158 MiB/s**（64 MiB 块）；引擎 pread 段实测 36.248/17.568 = **2.06 GiB/s** → **读盘已贴着磁盘上限**。

## 判断与解法
- **可达上限**：把 17.568 s 的读盘与 16.000 s 的 CPU 重叠 → prefetch ≈ max = **17.6 s**，
  去噪 wall 33.2 s → **≈18–19 s（约 1.75×）**。输出应**逐字节一致**（反旋转算法、顺序、结果全不变，
  只是换了执行线程）。
- **地板**：磁盘 2.06 GiB/s × 36.248 GiB = 17.6 s。想再快必须**少读字节**，
  即跨 step 常驻部分权重（8 GiB 缓存预算 ≈ 少读 40% → ~12 s），属内存规划器的范围，另议。
- **不选 GPU 反旋转**：项目已有 `h3_gpu_weight_dequant_unrotate_int8` 且 `h3_convrot_test` 在 M4 通过，
  但 Phase 10 已判定「GPU dequant 与主线程计算争抢同一 GPU，净亏」；
  且此路径要求 GPU 在 61% 空闲时间里接手 16 s 的工作，收益不确定。先用纯 CPU 侧的流水重叠。

## 实现与结果（2026-09-12）

### 做法：按 source 分片并行（不是流水线）
`read_stream_layer` 拆成 `read_stream_sources(job, indices, count)`（纯工作）+ 编排器。
编排器默认起 **2 个 worker**（连续索引区间）+ `H3_DIT_STREAM_WORKERS` 旋钮（1 = 原串行）。

**正确性依据**：块的 4 个 source 各写**不同**的 slot 张量（`stream_slot_target`:
qkv / out / fc1 / fc2 / lin_to_out），并发写域不相交；每个 worker 持有自己的
`h3_dit_stream_job`（含 512B error 缓冲与 bytes/pread/dequant 计数器），无共享可变状态。
LoRA 合并在所有 worker join 之后由编排器执行，`job->seconds` 也只在末尾写。

### A/B 实测（同机，256×256 / 0.5s / steps=2，产物均**逐字节一致**）
| 量 | W=1（原串行 = 基线） | W=2（默认） | 变化 |
|---|---|---|---|
| Euler denoise wall | 33.296 s | **22.686 s** | **−31.9%（1.47×）** |
| unhidden wait | 20.201 s | 9.223 s | −54% |
| pread（线程累计） | 17.826 s | 25.587 s | 并发读提升到 1.58 GiB/s |
| cpu-unrotate（累计） | 15.866 s | 17.229 s | ~持平 |
| 有效吞吐 | 1.076 GiB/s | 1.577 GiB/s | +47% |

### 两个被实测否决的想法（避免重复试）
1. **按字节 LPT 均衡分片**（154.1M vs 231.2M → 192.7M vs 192.6M）：实测 denoise **23.15s，无收益**。
   原因推断：均衡打乱了文件顺序局部性——`prepare_stream_layer` 已 `qsort` 按
   (path, file_offset) 排序，**连续索引区间 = 顺序读**，对 2 GiB/s 的盘有价值。→ 回退连续区间。
2. **4 个 worker**（每 source 一个）：denoise **25.19s，更慢**；pread 累计从 25.05s 涨到 52.05s
   （并发读过多使该 SSD 退化）。**2 个是甜点。**

### 余下瓶颈（Phase 33 时定位）
`unhidden wait` 仍有 9.2s。用测得的量建模：
`pread 25.59/2 ≈ 12.5s wall` + `cpu 17.23/2 ≈ 8.6s wall` = **21.1s** ≈ 实测 22.7s
→ 说明**每个 worker 内部仍是 read→dequant 串行**，两者没有互相填充。

## Phase 35 实现与结果：worker 内 SPSC 两级流水（2026-09-12）

### 做法
每个 worker 内部拆成两个线程，用 **2 槽 SPSC ring** 解耦：
- reader 线程：`stream_fill_slot()` — pread int8 + scale 到 staging slot
- dequantizer（worker 自己的线程）：`stream_consume_slot()` — WHT 反旋转 + 写 slot

槽位下标恒为 `position % H3_STREAM_RING`，握手靠 `filled` 标志，**无需 producer/consumer 游标**。
`H3_DIT_STREAM_PIPELINE=0` 时两阶段在本线程背靠背运行（同一份代码，无重复实现）。

### 2×2 对照矩阵（同一 binary，256×256 / 0.5s / steps=2，产物全部**逐字节一致**）
| 配置 | denoise wall | unhidden wait | 说明 |
|---|---|---|---|
| W=1, P=0 | 35.131 s | 22.101 s | 原始串行参考 |
| W=1, P=1 | 25.189 s | 11.546 s | 只流水、不切片 |
| W=2, P=0 | 23.586 s | 10.239 s | Phase 33（只切片） |
| **W=2, P=1（默认）** | **21.339 s** | **7.957 s** | 切片 + 流水 |

关键派生量（每 block）：串行 337ms → **217ms**；有效吞吐 1.08 → **1.67 GiB/s**。

### 为什么流水只多给 ~10%（而不是预期的 2×）
1. **每个 worker 只有 2 个 source**，2 槽 ring 最多领先 1 个 source。理论模型
   `wall = R₁ + Σmax(Rₖ,Dₖ₋₁) + D_N` 在 R≈D 时给出 `(N+1)/2N` 的效率：
   N=2 → 75%（即最多省 25%），N→∞ → 50%。实测省 ~10%，与 N=2 的量级吻合。
   **瓶颈是 stage 粒度太粗，不是 ring 深度**（推导：ring≥2 时加深 ring 不改善该公式）。
2. **并发读会拖慢 CPU 反旋转**：`cpu-unrotate` 线程累计从 17.2s（P=0）升到 19.0–19.6s（P=1）
   —— reader 以 2.4 GiB/s 读写内存时与 WHT 蝶形抢带宽。
3. 磁盘并发读一多就退化（4 个 reader 时 pread 累计 25→52s），所以**不能靠加 reader 数量解决**。

### 计算出的地板与剩余空间
每 block：read 阶段 ≈ 159ms（2 reader 合计 318ms，约 2.42 GiB/s）、dequant 阶段 ≈ 98ms。
理想 `max(159, 98) = 159ms` → prefetch ≈ 15.9s → 去噪 ≈ 16s。**当前 217ms，差 58ms（27%）**。
按 stage 粒度模型，把 source 再切成 N 段后：N=4→199ms、N=8→179ms、N=16→169ms（收敛到 159ms）。
→ 收益有限（21.3s → ~19.5s），而改动量不小。**更值得做的是加快 CPU 蝶形本身**
（dequant 已是关键路径的一半；蝶形快 2× 可让去噪落到 ~17s）。

### 实现缺陷记录
`H3_DIT_STREAM_PIPELINE=0` 首版错误地把 `&& pipelined` 加到了 **worker 线程的创建条件**上，
导致关闭流水时连 worker 并行一起关掉（量出 35.0s 的"全串行"，误判为 Phase 33 的 22.7s）。
修正：`pipelined` 只决定每个 worker 是否 spawn 自己的 reader 线程，与 worker 之间是否并发无关。

## Phase 36：继续压（直写 slot / NEON / 分块），2026-09-12

### 36a 反旋转直接写 slot 张量 —— 保留
`convrot_unrotate_cpu` 原先把结果写进一个每 source `malloc` 的 `full` 缓冲，再
`h3_gpu_tensor_write_bf16` 整块 memcpy 进 slot。新增
`h3_gpu_tensor_bf16_storage()`（`h3_gpu.h/.m`）拿到共享存储指针后**直接写 slot**，
去掉一个 ~230MB 暂存、一次全量拷贝，以及每 source 一次的 mmap/munmap 首次触碰缺页。
- 实测 `cpu-unrotate` 19.568s → **16.962s（−13%）**，denoise 21.339 → **21.060s（−1.3%）**
- 产物逐字节一致。**保留**（同时是内存卫生改善：不再有 230MB/次 的瞬时分配）

### 36b NEON 重写反旋转 —— **实测否决，未进仓库**
在 `/tmp/convrot_bench.c` 独立基准里写了 NEON 版（`vld4q_f32`/`vst4q_f32` 处理 stride-1 段，
`vld1q_f32` 处理 stride 4/16/64，SIMD 位运算复刻 bf16 的 round-half-up）。
- 先排查出一处真 bug：`vcgeq_u32` 返回的是 **全 1 掩码 0xFFFFFFFF** 而非 1，
  `hi + mask` 变成 `hi - 1`，恰好让 bf16 位模式差 2（与观测的 off-by-2 完全吻合）。
  修法是 `vandq_u32(mask, vdupq_n_u32(1u))`（用 2M 随机样本验证 0 处不一致）。
- 修好后 `ALL BIT-IDENTICAL`，但**性能是 0.84×（更慢）**：
  scalar 8259 MB/s vs neon 6973 MB/s（`out_proj`/`fc2`/`fc1`/`qkv`/`beta_proj` 全部一致）。
- **根因**：`-O3` 已经把标量版自动向量化了。该函数实测 2.75 G 元素/s ≈ 8.3 GB/s 的
  「3 字节/元素」流量，折算约 14 ops/cycle —— 单发射标量做不到，所以编译器显然已经 SIMD 化，
  且比手写 intrinsics 调度得更好（手写的多了 4 路解交错 shuffle 开销）。
- **结论：这个蝴蝶已经是编译器的最优解，别手写 SIMD。** 该负结果避免了后人重复投入。

### 36c 流水 stage 粒度降到 1024 行 —— 保留
模型 `wall = R + D/M`（M = 流水级数）说明 M=2 时末尾的 drain 占总 wall 的 20%。把 ring 的
工作单元从「整个矩阵」改成「1024 行的 chunk」（与反旋转自身的粒度一致，M 从 2 升到 ~30-60）。
- 附带收益：staging 内存从 115MB/worker 降到 **5.5MB/worker**
- 实测 denoise 21.060 → **19.962s**，`unhidden wait` 7.696 → **6.594s**，逐字节一致
- 同时加了 per-source 的 scale 缓存（否则每个 chunk 都重读整个 scale 张量，
  每 block 每 worker ~30 次）。该缓存**实测中性**（20.17 vs 19.96，噪声内），
  保留的理由是避免 `O(chunks × rows)` 的潜在退化，而不是性能

### 最终累计（256×256 / 0.5s / steps=2，全部逐字节一致）
| 阶段 | denoise wall | 累计加速 |
|---|---|---|
| 原始基线 | 33.296 s | 1.00× |
| + 按 source 分片并行（Phase 33） | 23.586 s | 1.41× |
| + worker 内两级流水（Phase 35） | 21.339 s | 1.56× |
| + 反旋转直写 slot（Phase 36a） | 21.060 s | 1.58× |
| + 1024 行分块（Phase 36c） | **19.962 s** | **1.67×** |

### 终点：已贴磁盘地板
每 block 每 worker：read ≈ **173.5ms**（2 reader 合计 347ms ≈ 2.22 GB/s，已等于该卷实测上限
2.16 GiB/s），dequant ≈ 87ms。理想 `max(173.5, 87) = 173.5ms` → 去噪 ≈ **17.4s**。
当前 199.6ms，**只差 15%**，且剩余部分已分散到 condvar 交接延迟与内存带宽争抢，
不再有单一主导项。想再跌破 17.4s 只能**少读字节**（跨 step 常驻部分权重），属内存规划器范畴。

## ⚠️ 决定性 A/B：收益只在低分辨率存在（2026-09-12）

用 `git stash` 回原始源码单独编一个二进制，与优化版在同一机器上逐分辨率对照
（256×256 / 384×384 / 512×512 / 864×480，均 `--seconds 0.5 --steps 2`，产物**逐字节一致**）：

| 分辨率 | 原始 denoise | 优化 denoise | 加速 | 原始 `unhidden wait` | 优化 `unhidden wait` |
|---|---|---|---|---|---|
| 256×256 | 33.296 s | 19.962 s | **1.67×** | 19.945 s | 6.594 s |
| 384×384 | 33.922 s | 26.746 s | **1.27×** | 7.561 s | 0.009 s |
| 512×512 | 46.441 s | 47.454 s | **0.98×（噪声内，略慢）** | **0.001 s** | 0.001 s |
| 864×480 | 78.869 s | 77.579 s | **1.02×（噪声内）** | **0.001 s** | 0.001 s |

**流式耗时本身与分辨率完全无关（这正是优化的设计目标）**：

| | 384×384 | 512×512 | 864×480 |
|---|---|---|---|
| 原始 stream wall | 34.084 s | 35.034 s | 35.337 s |
| 优化 stream wall | 20.032 s | 19.900 s | 20.332 s |
| 降幅 | −41% | −43% | −42% |

### 结论
- 优化**按设计工作**：流式耗时稳定腰斩（35s → 20s），且与分辨率/时长无关
- **但 ≥512×512 时流式在原始代码里就已经被完全隐藏**（`unhidden wait 0.001s`），
  所以整条优化链收益为 **0**（噪声带 ±2%，512 那次甚至略慢——多线程与 GPU 争抢内存带宽）
- **交叉点落在 384×384 与 512×512 之间**（≈448），取决于 GPU 算力 / 磁盘带宽之比：
  GPU 越快或盘越慢，交叉点越低（越值得开）；模型能常驻的机器则该路径根本不执行
- **注意引擎默认就是 864×480 / 56 帧 / 20 步** —— 即默认配置下这套优化**完全没有收益**
  （56 帧比上表的 12 帧更偏 GPU 主导，只会更极端）

### 因此
保留/回退是个产品决策，不是技术决策：
- 常跑 ≤384 小画布（预览、低分辨率快出）→ 值 1.27–1.67×，值得留
- 按默认 864×480 跑 → 白给约 500 行并发代码，建议默认关闭（`H3_DIT_STREAM_WORKERS=1
  H3_DIT_STREAM_PIPELINE=0` 即回到原始行为）或整体回退

---

# memory-plan 未接线字段（2026-09-12 session）

## 现象
给 `--video-vae-streaming 0` 做 A/B 时，逐字段核对 `h3_memory_plan` 的输出发现：
planner 算出的 5 个决策里，**2 个从未被消费**。

| planner 输出 | 消费方 | 状态 |
|---|---|---|
| `ssd_streaming` | h3.c → DiT load | ✅ 接线 |
| `use_int8_row_fc2` | h3.c → DiT load（且被 streaming 强制归零） | ✅ 接线 |
| `video_vae_streaming` | h3.c → decoder load/decode | ✅ 接线 |
| `encoder_streaming` | **无** | ❌ 死字段 |
| `cache_budget_bytes` | **无** | ❌ 死字段 |
| `dit_layers` | h3.c（仅当 >0 时覆盖） | ✅ 接线 |

## git 证据：从未接线，不是撤回
- 引入提交 `d5752a0`（2026-08-28，"分析为何只能 32GB 跑通…让 16/24GB 设备也能跑"）。
- `git log --all -S "params->encoder_streaming"` → **零结果**（全历史从未被读取）。
  只有写入侧：`out->encoder_streaming = …`（planner）与 `eff.encoder_streaming = …`（h3.c）。
- `git log -S "cache_budget" -- h3.c h3_dit.c h3.h` → **零结果**（h3.c/h3_dit.c/h3.h 从未提及）。
  该符号只活在 `h3_memory_plan.c/.h` 内部。
- `d5752a0` 的 h3.c diff 对照：同批次的 `video_vae_streaming` 一路传到
  `h3_video_vae_decoder_load` / `h3_video_vae_decode`；`encoder_streaming` 只有孤零零一行赋值。

## encoder_streaming 为何无意义
- 声明意图（h3.h:112-115）："Release the text/image encoder (Qwen3-VL first 50 layers)
  after condition building instead of keeping it resident through denoise."
- 事实：编码器根本没有可常驻的对象。`h3_text_encode_bf16()` /
  `h3_text_encode_multimodal_bf16()` / `h3_text_encode_clipproj_bf16()` 都是
  "打开权重 → 跑完 → 释放" 的一次性函数（h3_text_encoder.c:517 open、540/758 free），
  只把 `h3_text_embedding` 交给调用方。
- planner 自己也按这个前提估算：`streamed_resident` 里 `text_encoder.bytes * 0
  /* freed per call */`（h3_memory_plan.c:20, h3.c:1253）。
- → 意图已天然满足；接线需要反过来**新增**常驻缓存，无收益场景。

## cache_budget_bytes 为何仍有研究价值
- 意图（h3_memory_plan.c:97-112，移植自 ds4 的 streaming cache planner）：
  `budget = GiB_align(7/8 × recommended_working_set − steady_streamed)`，下限 1 GiB。
  在这台机器上算出 8.0 GiB —— 且只被写进 `plan.reason` 文本
  （"… cache 8.0 GiB"），**日志会误导人以为真占了 8 GiB 内存**。
- 收益指向当前最大瓶颈：denoise 每步重流全部 50 块。
  实测（256×256 / 2 s / 4 步）：`72.136 GiB / 4 步`，denoise 73 s，
  其中 pread 35.0 s + cpu-unrotate 38.5 s + unhidden wait 6.4 s。
- 每块 ≈ 0.36 GiB（72.136 / 4 / 50）→ 8.0 GiB 可常驻 ≈22 块
  → 每步只流 28 块 → 读取量 −44%。
- 障碍：`stream_slots[2]` 是硬编码双槽轮转（h3_dit.c:218、`h3_stream_ring`），
  "部分块常驻"要改 ring 调度；且 8 GiB 常驻与 16 GB 机器的余量冲突
  （订正：`--video-vae-streaming 0` 的 VAE 常驻逻辑预载约 9.0 GiB，但为 F16→F32 加宽的**可驱逐**共享缓冲，并非钉死内存；实测峰值 int6g128(native)@864x480 5s=6.10 GiB、int6g128(native)@512=5.13 GiB、int8g64@512=6.20 GiB，从未 OOM。原「同条件 OOM 过一次」论断在 Apple 统一内存下不成立，自动规划器据此默认选流式是在白白放弃内存/速度。注：实测所用 `native` 变体的 DiT(transformer)实为 int6-g128 量化权重（软链 → `h3_int6g128_native`），并非全精度；真正的未量化全精度模型是 `base` 变体（transformer → `h3_sys/MiniMax-H3-Convrot/FL2VA/transformer`）。各变体的 `video_vae`/`text_encoder`/`audio_vae`/`tokenizer` 均为指向系统盘的全精度软链、跨变体共享、VAE 永不量化，故 VAE 常驻/流式的内存行为在 native/int8/int6/base 下完全一致，变体间峰值差 100% 来自 DiT。15s 长时长验证(@864x480 int6g128):resident 峰值 **7.51 GiB**(8059174912 bytes),VAE 常驻逻辑预载 9.0 GiB 但进程峰值仍远低于此,168 个 chunk 全部走 resident tile 路径,无 OOM——时长从 5s 增至 15s(序列 3×)未改变 resident 更省的结论)。

### 自适应 DiT 部分常驻 + 流水线解耦(2026-09-14)

**问题**:15s@864x480 去噪耗时 54 分钟(DiT SSD 流式每步全量读取 36 GiB int6 权重 + CPU 反量化),且 int6 DiT 在 M4 无张量核上走纯 CPU 反量化极慢。

**改动 1 — 自适应部分常驻**(h3_host.c/h3_dit.c):新增 `h3_host_available_memory()`(mach host_statistics64)查询可用物理内存,`h3_dit_resident_budget()` 按「可用内存 − 激活张量预算(1 GiB + latent 大小)」除以单块成本(保守 0.5 GiB)计算可常驻块数。`load_core()` 中若未设 `H3_DIT_RESIDENT_BLOCKS`,在 SSD 流式启用时自动触发。实测 2s 去噪从 325s(0 块常驻)降至 215s(10 块常驻),**提升 34%**。

**改动 2 — 安全流水线模式**(h3.c):`--pipeline` + `--latent-out` 组合,去噪后用 `madvise(MADV_DONTNEED)` 提示 latent 缓冲区可回收(OS 可将页面回收给 VAE 权重或磁盘缓存),避免危险的 free+re-read(曾在内存压力主机上触发内核 panic)。对长序列(latent > 物理内存)可显著降低峰值 RSS。

**改动 3 — 缓存 key 修复**(h3.c:1383):解码器缓存 key 加入 `video_vae_streaming`,修复交互式会话改 flag 后旧 decoder 被静默忽略的隐患。

**实测对比**(int6g128@864x480,2 步):

| 配置 | 常驻块 | 去噪时间 | 峰值 RSS |
|---|---|---|---|
| 无常驻(手动设 0) | 0 | 325s | 7.95 GiB |
| 自适应常驻 | 10 | 215s | 7.31 GiB |
| 自适应 + 流水线 | 10 | 215s | 7.23 GiB |

**文件变更**:h3_host.h/c(内存查询)、h3_dit.c(自适应常驻)、h3.c(流水线 + key 修复)、findings.md、AGENTS.md、comfyui_nodes/README.md。

### float16 latent 存储(2026-09-14)

**动机**:15s@864x480 的 denoised latent(float32)约 18 GiB,超过 16 GiB 物理内存,导致 VAE 解码阶段 swap。将 latent 检查点存为 float16 可减半磁盘占用(18→9 GiB),且对最终画质无可见影响。

**实现**:`h3.c` 的 `write_latent_bundle`/`read_latent_bundle` 升级为版本 3,新增 `dtype` 字段(0=f32, 1=f16),兼容旧版 2(f32)文件。`H3_LATENT_F16=1` 或 `--pipeline`(默认)写入 f16;`--latent-in` 读回时转回 f32 再送 VAE 解码。新增 `h3_f32_to_f16`/`h3_f16_to_f32` 转换函数。

**质量验证**(同种子 seed=7 对比 f32/f16 latent 解码首帧):**PSNR = 45.93 dB**(远超 40 dB 阈值,像素差异 ~0.2%,肉眼无损)。

**bug 修复**:初版 `h3_f32_to_f16` 进位逻辑误用 `h & 0x400` 检测尾数溢出,实际命中了指数最低位,导致所有 normal 值被放大 2×(PSNR 仅 ~20 dB)。修正为用独立 `mant10 & 0x400` 判定尾数进位并正确+1 到指数后,PSNR 恢复 45.93 dB。

**注意**:f16 存储只缩小**磁盘检查点**;VAE 解码阶段仍需 f32 latent 在内存,故 15s 的 in-memory 峰值仍为 ~18 GiB f32。真正避免 15s swap 需走两进程路径(`--latent-out` 写 f16 → 独立 `--latent-in` 读 9 GiB f16 文件增量转 f32 解码),其峰值显著低于单进程内 18 GiB。流水线模式的 `madvise(DONTNEED)` 曾因清零 VAE 仍需的 latent 而危险,已移除——仅保留安全的 f16 落盘。

## 结论
- `encoder_streaming` → **删除**（冗余，意图已天然实现）。
- `cache_budget_bytes` → 先做代理实验（`--layers 40` 验证线性），再定接线或删除。

# ===== 本任务发现: H3 latent 子系统（音频 latent 与落盘格式）=====

## 音频 latent 的真实格式（关键，此前未 pin 过）
- `h3_audio_latent`（`h3_audio_vae.h:17-24`）：`channels=32, stereo=2, length=T`，
  `values` 布局 **[32,2,T]** channel-major，归一化 posterior means。
- 解码入口 `h3_audio_vae_decode(weight_dir, shader, normalized_latent, latent_length, …)`；
  主路径调用 `h3_audio_vae_decode(audio_vae_path, "h3_shaders.metal", audio, temporal.audio_t, …)`
  —— `audio` 缓冲即该 `[32,2,T]` 张量。
- **与视频 latent 不同秩**：视频是 `[C,T,H,W]`（4D），音频是 `[32,2,T]`（3D、无空间维）。
  → 空间上采样与音频无关；音频只需原样透传。

## 落盘点为何恰好正确
h3.c 主路径顺序：**2119 落盘** → 2135 音频 VAE 解码 → 2158 视频 VAE 解码；
`free(audio)` 在 **2141**。故落盘点处 `audio` 仍有效，且**尚未被 transform**，
正是 `h3_audio_vae_decode` 期望的输入 → 直接 dump 即得**精确往返**（无需逆变换）。
视频 latent 同理：`h3_video_vae_decode` 内部才做 `z*std+mean`，dump 的是变换前的 z。

## latent 文件格式 v1 → v2
```
v2 布局（小端、全 int32/float32）：
  uint32 magic = 0x48334C54 ("H3LT")
  int32  version = 2
  int32  video_t, video_h, video_w, video_c
  c*t*h*w float32     // 视频 latent [C,T,H,W]
  int32  audio_present (0/1)
  if audio_present:
    int32 audio_c, audio_s, audio_t
    c*s*t float32     // 音频 latent [32,2,T]
```
v1 无 version 字段且只有视频 → 无法区分是否含音频，故整体升级（中间产物不向后兼容）。

## 音频合成的两个易错点
1. `h3_ffmpeg_write_av_rgb24_f32` 对 **NULL pcm 直接失败**（`h3_ffmpeg.c:620-624`），
   所以「无音频」分支必须显式 calloc 静音轨，不能传 NULL。
2. `waveform` 的 pcm 由 `h3_audio_vae_decode` 分配 → 必须用
   `h3_audio_waveform_free(&waveform)` 释放（照主路径 h3.c:2266 写法），
   直接 `free(pcm)` 会重复释放。

## 官方工作流音频路径（对照，已用本地官方 JSON 解析确认）
- `video_minimax_h3_r2v.json`：`SamplerCustomAdvanced.output` **同时**连
  `VAEDecode`(video) 与 `VAEDecodeAudio`(audio) → `CreateVideo` 合成。
- 这些官方文件里**没有** `MinimaxH3LatentUpscaler3D` 节点（均为单阶段）；
  `LatentUpscaler3D` 只出现在我们改造的工作流里（且已移除）。
- `MiniMaxH3_Convrot_Workflow.json` 变体：`output`→`VAEDecode`、
  `denoised_output`→`VAEDecodeAudio`，两路并行后 `CreateVideo` 合成。

# ===== int4 / 量化感知蒸馏 调研发现（2026-09-13）=====

## 度量语义（本轮最重要的一条认知）
- `SSIM/PSNR/L2/cosine` 在「**不同权重、同提示**」下量的是**采样轨迹分歧**，不是画质。
  扩散采样会放大任何权重扰动，步数越多分歧越大：int8-g64 的 SSIM 从 2 步 **0.891** 掉到 4 步 **0.739**，
  而其纹理量（detail 0.0122 vs base 0.0129）与饱和度（0.0997 vs 0.0967）与基准持平 —— 即**画质没变**。
- 正确判据（已实现在 `fastvideo_qad/scripts/ab_quant_quality.py`）：
  | 代理 | 定义 | 诊断意义 |
  |---|---|---|
  | `detail` | `mean |d(luma)/dx|` | 高频结构量；**高于参考且饱和度不高于参考 = 噪点，不是纹理** |
  | `saturation` | `mean(max−min over RGB)` | 去饱和 = 退化（只有下限，无上限） |
  | `luma_std` | 对比度；< 0.05 说明参考本身欠采样，全部指标不可信 | 门槛 |
- 参考性反例：int4-g64 的 SSIM 2 步 0.521 → 4 步 **0.598（反而升）**，但它仍是**退化**的
  （2 步 detail ×1.578 判定 `grainy`；4 步 saturation ×0.857 判定 `desaturated`）。

## int4-g64 PTQ 实测（源 = 旋转空间 per-row int8）
| 方案 | 权重 relRMS | 2 步 SSIM | 4 步 SSIM | verdict |
|---|---|---|---|---|
| 源 per-row int8（base） | —（基准） | 1.000 | 1.000 | reference |
| int8 affine group-64（回写 per-row int8） | 0.0052 | 0.891 | 0.739 | **preserved**（2/4 步一致） |
| int4 affine group-64（同上） | 0.0913 | 0.521 | 0.598 | **degrades**（2 步 grainy / 4 步 desaturated） |
- 结论：**int8-g64 可用；int4 PTQ 在无 QAT 前提下不可接受**。这与 §4 的推导一致。
- 评测路径是**代理**：int4 → 反量化 → 重量化 per-row int8 → 现有引擎 int8 路径。
  该代理层引入约 0.5% 底噪（int8-g64 通过它仍判 preserved），对方案差异足够灵敏。
  **引擎目前没有 int4 加载路径**，所以代理是当前唯一可用的评测通道。

## QAD / QAT 事实（FastVideo / FastMetal 侧）
- `FastMetal-1.3B-QAD` 的精度来自 **QAT + DMD2 训练**（GB200 集群），不是格式转换。
- FastVideo 的 QAD 配方（attn-QAT fake-quant + STE，再 DMD2）**只支持 Wan2.1**；
  `fastvideo/train/models/minimax_h3/minimax_h3.py:73-74` 强制 `TORCH_SDPA`，H3 **无 QAT 入口**。
- H3 在 FastVideo 只有推理期量化：MLX weight-only int8/int6/int4（affine g64）、CUDA FFN NVFP4/MXFP8。
- `MLXQuantizationSpec.from_name` 把 group_size **硬编码 64**（`fastwan.py:66-70`）；
  `mx.quantize` 本身接受任意 group_size。
- 本机 anaconda env **无** gptq/awq/llm-compressor（仅 `mlx 0.32.2`）→ 误差补偿 PTQ 需自实现。

## h3.c 引擎能力盘点（决定改动量）
- `h3_safetensors.h:9-24` 已有 `H3_DTYPE_U32`（MLX 打包 int4 用得上），但无任何 U32 量化语义。
- `h3_weights.c::load_int8_dequantized`（179-330）只处理 **I8 + `{name}_scale` F32[rows,1]**，
  且依赖「蝶形先行、scale 后乘」的等价性（`scale * butterfly(i8) / 16`）。
  **group 尺度会破坏该等价性**（尺度按输入维分组）→ int4 必须先解包成 float、按组乘尺度、再反旋转。
- `h3_dit.c::convrot_unrotate_cpu`（668-710）按 `const int8_t *` 读源，输出 BF16，另带 qkv 的 `layout==1` 行重排。
- 流式源描述 `h3_dit_stream_source` 有 `scale_path/scale_offset/scale_elements`，但**没有 zero-point 字段**。
- 无激活 dump 钩子（只有 `H3_DEBUG_CONVROT`、`H3_DEBUG_VISION`）。

## QKV 行序（踩过的坑，务必记住）
- 源 checkpoint 的 `blocks.N.attn.qkv_proj.weight` 是 **module-major**（即 `concat(to_q,to_k,to_v)`），
  **不是** head-interleaved。实测：`relRMS(concat, stored)=0.0056` vs `relRMS(interleave(concat), stored)=1.404`。
- h3.c 自己会在加载时把它重排成 head-interleaved 送进 slot
  （`h3_dit.c::convrot_unrotate_cpu`，`dst = (3*(slot%heads) + slot/heads)*head_dim + dim`）。
- 因此**转换脚本里不需要也不允许再做一次交织**（旧 `mlx_int8_to_h3.py` 正是多做了这一层，导致 int8 与 int4 结果都崩到 SSIM≈0.46）。

## 机器与卷（影响实验设计）
- Apple M4 / 16 GiB；`recommended GPU set 11.8 GiB`；Metal 4 = yes，但 `tensorOpsEnabled` 仍只在 M5 或 `H3_VDN_INT8` 时开
  → **本机没有 int8/int4 张量核路径，量化只省 I/O 不省计算**。
- 顺序读：**内盘 2.31 GB/s** vs **外接 `/Volumes/data` 0.85 GB/s**（差 2.7×）。
  权重放哪个卷直接决定 256² 流式场景 1.5× 的耗时（base 56 s vs 我在外接盘的 86 s）。
- 内盘 `/System/Volumes/Data` 仅剩 ~2 GiB，swap 上限 2 GiB → 任何大产物必须落 `/Volumes/data`。

## Phase 3a：引擎校准激活 dump 钩子 —— **已实现并验证**
- 新增 `H3_DUMP_ACT=<dir>`（+ `H3_DUMP_ACT_ROWS` 默认 256、`H3_DUMP_ACT_LIMIT` 默认 8），
  写在 `h3_dit.c`：结构体加 `dump_dir/dump_rows/dump_limit/dump_calls`，helper `dump_activation()`，
  四个调用点分别对应四个矩阵的**真实输入**：
  | 矩阵 | 输入张量 | 调用点 |
  |---|---|---|
  | qkv | `dit->mod_attention` [rows,5376] | AdaLN 之后 |
  | out | `dit->attention_heads` [rows,7168] | SDPA 之后 |
  | fc1 | `dit->mod_mlp` [rows,5376] | MLP AdaLN 之后 |
  | fc2 | `dit->activated` [rows,14336] | SwiGLU 之后（**须 `H3_DISABLE_FUSED_MLP=1`**，融合 MLP 把中间量留在 kernel 内） |
- **可行性关键**：`h3_gpu_submit` 会 commit **并 `waitUntilCompleted` 所有在飞命令缓冲**（h3_gpu.m:1006-1007），
  且每个 kernel 自带 encoder → 「`submit`（刷+等）→ `read_bf16_range` → `begin`（重开链）」可在 block 内安全插入。
  非 dump 运行在第一行 `if (!dit->dump_dir[0]) return;` 就退出，**零影响**。
- 实测（256²/0.5s/2 步，源 checkpoint）：200 个文件、3.2 GB，每层 **1068 行**样本（2 步 × 534 token），
  **且端到端出片正常**（钩子未破坏流水）。
- 编译：`make CC="xcrun clang"` **0 warning / 0 error**。

## Phase 3b：激活感知量化的评估 —— **决定性负结果**
度量口径（重要修正）：**层输出误差** `err = ‖(Ŵ−W)Xᵀ‖_F / ‖WXᵀ‖_F`，而不是权重 relRMS。
理由：权重 relRMS 与画质**非单调**（int6-g128 权重误差更差但端到端更好），
而输出误差直接对应 `E‖(Ŵ−W)x‖²`，是量化的真实目标。

### 关键诊断：ConvRot 已经把输入各向同性化了
旋转前后每通道二阶矩的 max/min 比值（`x_rot = x_plain @ H`，因为引擎反旋转权重 ⇔ 等价于旋转激活）：

| 层 | 旋转前 | 旋转后 | 改善 |
|---|---|---|---|
| blocks.0.mlp.fc2 | 2.67e9 | 718 | 3.7e6× |
| blocks.1.mlp.fc2 | 9.08e8 | 62.2 | 1.5e7× |
| blocks.0.mlp.fc1 | 7.15e4 | 73.4 | 974× |
| blocks.0.attn.qkv | 3946.7 | 16.7 | 236× |
| blocks.0.attn.out_proj | 5898.6 | 3165.3 | 1.9× |
| **均值（20 个层）** | **1.86e8** | **644.9** | — |

**ConvRot 一个固定的、免费的正交变换就提供了 5–7 个数量级的通道均衡——这正是 GPTQ/AWQ 赖以生效的机制。**

### 后果：AWQ 在这里是**负收益**
| 方案 | 层输出误差（20 层均值） | 权重 relRMS |
|---|---|---|
| int4-g64 plain | **0.05377** | 0.09094 |
| int4-g64 + AWQ 通道缩放 | **0.06038（0.891×，恶化 12%）** | — |

- 20 个层**全部**变差。机制：旋转已把输入摊平，再按 σ_j 缩放只会破坏量化误差分配。
- **结论（仅对 AWQ/对角缩放族成立）：通道缩放式激活感知量化在本模型上是负收益。**
- 附带收获：`plain` 的输出误差在 0.029–0.081 之间有 **2.8× 的层间差异**（不像 relRMS 那样全层齐平）
  → **这才是混合精度需要的敏感度信号**，为 Phase 1c 提供了可用的排序依据。

### 一次性 whitening —— **无效，不是有效上界**
`quantize(W @ Σ^{1/2}) @ Σ^{-1/2}` 实测误差 **550**（plain 0.0535），差 1 万倍。
原因：Σ 极端病态，用 `Σ^{1/2}`/`Σ^{-1/2}` 一对变换会把低能量方向的量化误差放大 `1/√λ` 倍而无界。
→ **不能用一次性变换实现二阶目标**；二阶方法必须用序列式过程（GPTQ）从构造上处理条件数。

### 旋转不变量：特征值谱
`Σ_rot = Hᵀ Σ_plain H` 与 `Σ_plain` **相似 → 特征值完全相同**。所以：
- ConvRot 只能均衡**对角**（这正是它 5–7 个数量级收益的来源），**无法改变特征值谱**。
- 实测能量高度集中：99% 能量落在 **12–169 个方向**（n=1068）→ 序列式二阶方法**理论上仍有空间**。
- 注意：n=1068 样本对 d_in=5376~14336 是严重欠定（H 秩 ≤ 1068），上述"有效秩"部分含小样本假象。

## Phase 3b'：真 GPTQ —— **成功（4.36×），且踩坑记录重要**
`gptq_probe.py`（单层判决性测试）。**第一次实现是错的**，且错法很有教育意义：
- 我用了**完整逆矩阵** `H⁻¹`；GPTQ 必须用 **H⁻¹ 的上三角 Cholesky 因子**（`U = chol(H⁻¹).T`，
  `H⁻¹ = UᵀU`），因为它编码了**序列消元结构**——第 j 列只对"尚未量化"的列做补偿。
- 症状：三层的输出误差 **全部变差**（qkv 0.0387→0.0807、fc1 0.0809→0.1362、fc2 0.0753→0.0914），
  权重 relRMS 爆到 0.19–0.32（补偿量被放大到把值推出组量化范围、触发饱和）。
- **暴露 bug 的实验**：阻尼扫描。damp→∞ 时补偿应趋于消失、结果应收敛回 plain，
  但实测 damp=0.01/2.0/20.0 分别给 0.0807 / 0.0828 / 0.0921，**单调变差且权重 relRMS 恒为 0.32**
  → 说明问题不在条件数而在实现。**「极限行为应当退化到已知基线」是最有效的自检。**

修正后（`blocks.0.attn.qkv_proj`，d_in=5376，n=1068）：

| 方案 | 层输出误差 | 权重 relRMS |
|---|---|---|
| plain int4-g64 | 0.03865 | 0.09099 |
| **GPTQ int4-g64（damp=0.01）** | **0.00886** | **0.35661** |
| 增益 | **4.36×** | 升高 3.9× |

- **权重 relRMS 升高而输出误差降低 4.36×**——GPTQ 的典型签名：牺牲权重域保真度换输出域精度。
  这是本项目里第二次证明「权重 relRMS 不能用来判质量」（第一次是 int6-g128 vs g64）。
- 成本实测：**729 s/层**（qkv，d_in=5376，单线程循环 + BLAS）。
  按 `d_out·d_in²/2` 外推全部 200 层 ≈ **45 h**（fc2 单项就 ~18 h）。可按 block 并行以缩短墙钟。
- **待解决的前提**：校准集 n=1068 ≪ d_in（最大 14336），H 严重欠定。当前 4.36× 是在欠定 Hessian 上
  取得的，放大校准集（更多步数 × 更多 prompt）后数字会更可信、可能更优。

## 环境陷阱
- `h3` 二进制被标 `com.apple.quarantine`（`spctl -a` 判 rejected）→ gatekeeper **SIGKILL**，
  现象是「所有命令 exit=137 且零输出」，连 `--help` 都死。修：`xattr -d com.apple.quarantine h3`。
  排查同类问题时优先看 `xattr -l`，别先怀疑内存。
- `/tmp` 在**内盘**（仅 ~2 GiB 空闲）→ 21 GiB 级别的 checkpoint 必须写 `/Volumes/data`，
  否则 `SafetensorError: No space left on device`。

## Phase 1a：位宽 × group size × 对称性（`quant_scheme_sweep.py`，88 s，纯权重误差）
参考口径 = 源 checkpoint 自身的 per-row 对称 int8 反量化（即引擎今天在跑的东西），旋转空间。

| 方案 | B/param | 全 DiT | vs 源 | relRMS |
|---|---|---|---|---|
| int4-g32-asym | 0.7500 | 13.46 GiB | −25.0% | 0.0807 |
| int4-g64-asym（现 int4） | 0.6250 | 11.22 GiB | −37.5% | 0.0910 |
| int4-g128-asym | 0.5625 | 10.10 GiB | −43.7% | 0.1003 |
| int6-g32-asym | 1.0000 | 17.95 GiB | 0% | 0.0193 |
| int6-g64-asym | 0.8750 | 15.70 GiB | −12.5% | 0.0217 |
| int6-g128-asym | 0.8125 | 14.58 GiB | −18.8% | 0.0239 |
| int8-g64-asym | 1.1250 | 20.19 GiB | **+12.5%（更大）** | 0.0053 |
| int8-g128-asym | 1.0625 | 19.07 GiB | +6.2% | 0.0059 |

- **对称性**：非对称（zero-point）稳定优于对称 **~11%**（int4-g64: 0.1075 sym vs 0.0910 asym）。
  MLX `affine` 就是非对称形式 → 已有代理路径已吃到这部分收益。
- **粒度**：int4 从 g32→g128 仅跨越 0.081↔0.100 → **靠 group size 救不了 int4**（与目标量级差 17×）。
- **位宽**：误差约**每 bit 降 2×**（0.0910 → 0.0217 → 0.0053，各约 4×/2bit）。
- **8 bit 以上无意义**：int8-g64（20.19 GiB）比源 per-row int8（17.95 GiB）**更大**。
  → **int4 与 int6 是仅有的两条有效量化方向**。
- 实现校验：扫描得 int4-g64-asym = **0.09096**，与对 MLX 官方产物的独立实测 **0.09127** 差 0.3% → 互相印证。

## Phase 1a'：int6 端到端 —— **关键正面结果**
`quantize_h3_proxy.py`（新增的代理量化器，直接从源按任意方案量化，其余张量逐字节复制）。

| 变体 | 权重 relRMS | 2 步 SSIM | 2 步 ccos | 4 步 SSIM | 4 步 ccos | 4 步 detail× | 4 步 satur× | verdict |
|---|---|---|---|---|---|---|---|---|
| base（源 per-row int8） | — | 1.0000 | 1.000 | 1.0000 | 1.000 | 1.000 | 1.000 | reference |
| int8-g64 | 0.0052 | 0.8906 | 0.9892 | 0.7386 | 0.8548 | 0.949 | 1.031 | preserved |
| **int6-g64** | **0.0238** | **0.7930** | 0.9481 | **0.6698** | 0.7644 | **0.987** | **1.007** | **preserved（2/4 步一致）** |
| int4-g64 | 0.0913 | 0.5209 | 0.6298 | 0.5977 | 0.6660 | 1.253 | 0.857 | degrades |

- 拼图 `ab_report/montage_steps4_int6.png`（上→下：base / int8 / int6 / int4）：
  **int6 行与 base 行画质等同**（毛发锐度、颜色、松枝、雪面均干净），仅轨迹不同；
  int4 行为明显糊化 + 雪面条状伪影 + 去饱和。
- **由此得到「preserved / degrades」边界**：落在权重 relRMS **0.024（preserved）与 0.091（degrades）之间**。
  这把"还需要多准"从定性变成了定量：**任何新方案只要把 relRMS 压到 ~0.03 以下即大概率通过**。
- 收益：int6-g64 = **−12.5% 字节**；int6-g128（同量级误差 0.0239）= **−18.8% 字节**。
  这才是本任务到目前**唯一可落地的字节收益**（int8 换格式反而更大，纯 int4 质量不过关）。

## Phase 1c：混合位宽探针 —— **边界不可判（重要方法学发现）**
两个候选（`quantize_h3_proxy.py --override`，其余张量逐字节复制）：

| 变体 | 配置 | 字节 | pooled relRMS | 4 步（detail× / satur×）| 2 步（detail× / satur×）|
|---|---|---|---|---|---|
| mixA | MLP=int4, attn=int6 | −27.5%（13.0 GiB） | 0.0577 | 1.041 / **0.887 → degrades** | 0.986 / 1.141 → preserved |
| mixB | attn=int4, MLP=int6 | −22.5%（13.9 GiB） | 0.0576 | 1.208 / 1.091 → preserved | 1.092 / **0.944 → degrades** |

- 两个候选在 2 步/4 步之间**互相翻转**；mixB@2步 仅差 **0.006** 就越过 0.95 阈值。
- 由此标定出代理量的**跨步数波动约 ±0.15~0.25（饱和度比）**，而我原先的判定规则是在
  远离边界的两端（relRMS 0.005→preserved / 0.091→degrades）校准的。
- **结论：pooled relRMS ≈ 0.058 这一档目前不可判**，不能算通过。任何「<14 GiB」的结论
  必须先做评测加固（多 seed × 多 prompt 聚合 + 报告波动），否则只是在噪声里读数。
- 推论（指导 Phase 3）：**要突破 14 GiB，必须把 relRMS 压到 ~0.03 以下**，而不是在 0.06 附近调权重分配。
- 另有观测：mixA/mixB 的 pooled 均值几乎相同（0.0577 / 0.0576）但 max 都是 0.0915（int4 那部分），
  且两者 SSIM 排序（mixA 0.705 > mixB 0.644 @4步）与"哪部分用 int4"有关 →
  MLP 用 int4 比 attn 用 int4 略好，但差距在噪声内，需 1d 加固后才能定论。

## Phase 1a''：int6-g128 端到端 —— **通过（当前最优方案）**
| 变体 | 权重 relRMS | 4 步 SSIM / ccos | 4 步 detail× / satur× | 2 步 SSIM / ccos | 2 步 detail× / satur× | verdict |
|---|---|---|---|---|---|---|
| base | — | 1.0000 / 1.000 | 1.000 / 1.000 | 1.0000 / 1.000 | 1.000 / 1.000 | reference |
| int8-g64 | 0.0052 | 0.7386 / 0.8548 | 0.949 / 1.031 | 0.8906 / 0.9892 | 1.059 / 1.238 | preserved |
| **int6-g128** | **0.0256** | **0.7048 / 0.8267** | **1.076 / 1.093** | 0.7898 / 0.9427 | **1.031 / 1.200** | **preserved（余量充裕）** |
| int6-g64 | 0.0238 | 0.6698 / 0.7644 | 0.987 / 1.007 | 0.7930 / 0.9481 | 0.971 / 1.079 | preserved |

- **结论：int6 affine group-128 可用，字节 17.95 → 14.58 GiB（−18.8%）**，画质与 base 等同
  （`ab_report/montage_steps4_int6g128.png`：base / int8 / int6g128 / int6g64 / int4 五联对比，
  前三者肉眼等价，int4 明显糊化去饱和）。
- 反直觉但一致：g128 的权重误差（0.0256）**略大于** g64（0.0238），但端到端指标反而略好
  （4 步 SSIM 0.705 vs 0.670, ccos 0.827 vs 0.764）→ **权重 relRMS 与画质不是单调关系**，
  组粒度改变的是误差结构分布，不只是幅度。**判定必须靠端到端，不能只看 relRMS。**
- 余量判据的重要性被这轮坐实：
  | 方案 | 距 0.95 饱和度阈值 | 可判性 |
  |---|---|---|
  | int6-g128 | +15% / +26% | ✅ 可信 |
  | int6-g64 | +8% / +14% | ✅ 可信 |
  | mixA / mixB | −7% / −1% | ❌ 在 ±0.15~0.25 的步数波动内 → 不可判 |

# ===== 长时长内存墙 + 运行时守卫 (2026-09-14/15) =====

## 设备与上限（实测 `./h3 --info`）
| 项 | 值 |
|---|---|
| 芯片 / 内存 | Apple M4 / **16.0 GiB** |
| recommended GPU set | **11.8 GiB** |
| max Metal buffer | 8.9 GiB |
| swap | **0.00M（零缓冲）** |
| 权重合计 | ~36 GiB（DiT 17.4 + VAE 9.7 + Qwen 8.3 + audio 0.6）|

**机制**：Apple 统一内存里 Metal/GPU 分配是 **wired（不可压缩、不可 swap）**；当 wired+活跃超物理，
系统进入严重内存压力 → watchdog panic（死机重启），**不是进程级 OOM**。没有硬编码安全数字，
但软线：物理 16 GiB（硬）/ rec 11.8 GiB（GPU 建议）/ 实测 3s 峰值已 12.7 GiB。

## 内存 trace 实测（INT6 native，864×480，steps 4）
| 时长/配置 | DiT 加载 peak | denoise footprint | VAE decode | 结果 |
|---|---|---|---|---|
| 3s | — | — | GPU peak **0.654 GiB** | ✅ 完成（peak footprint 12.7 GiB, RSS 8.1 GiB）|
| 6s（无 token-reduction）| **13.00 GiB** | 10.11 → **13.88 GiB**（第1步 +3.77！）| — | ❌ 触发守卫 |
| 6s（`--token-reduction`）| — | 9.97→13.00（峰）→12.21 | 0.48→2.05 GiB, available 13→6.8↘ | ✅ 完成（2333s）|
| 2s 单段 | — | 峰值 **9.33 GiB** | — | ✅ 完成（698s）|

**两条独立根因**：
1. **denoise 激活 ∝ video token**：6s 比 3s token 翻倍 → 单步 footprint +3.77 GiB；15s 再 ×2.5 必爆。
2. **VAE decode 的 GPU wired 逐 chunk 累积**：6s 时 available 从 13→6.8 GiB（每 chunk ~0.8 GiB），
   且**进程 footprint 仅 ~2 GiB** → 这部分**不计入 `phys_footprint`**，只能从 available 观察；
   15s（~20 chunk）必归零。

## 守卫实现（`h3_host_memory_guard`）
- `h3_host.c`：`h3_host_footprint()`（`task_info(TASK_VM_INFO).phys_footprint`）
  + `h3_host_physical_memory()`（缓存 `hw.memsize`）+ `h3_host_memory_guard(phase,err,sz)`。
- 触发条件（任一）：`available < H3_MEM_GUARD_MB`(默认 1024) 或
  `footprint + H3_MEM_HEADROOM_MB`(默认 2560) `> physical`。
- 插入点：`h3_dit.c` denoise **CPU/GPU 两个步循环**；`h3_video_vae.c` **resident/chunked 两个 chunk 循环**。
- `H3_MEM_TRACE=1` 打印每步 `footprint / available`。
- 效果：6s 在 denoise 1/4 优雅报错退出（**不再死机**）；3s 默认 floor 不误杀。

## 分段生成（硬切）
- 脚本 `gen_segments.sh`：2s × 7 段独立进程（递增 seed），ffmpeg concat。
- 结果 `seg_out/final.mp4` = **16.36s / 864×480 / 392 帧 / 含音频 / 5.8 MB**。
- 段间**硬切**（无跨段条件）。

## 负结果：H3 vision encoder 三处不可得
| 来源 | 结果 |
|---|---|
| 本机 modelscope 缓存 `OpenVDN--vdn-minimax-h3/snapshots/master/h3-base` | 只有 audio_vae/transformer/vae/scheduler |
| HF `OpenVDN/vdn-minimax-h3` | **不可达（curl 000）** |
| ModelScope `OpenVDN/vdn-minimax-h3` | h3-base 无 text_encoder；stage 目录仅 adapters/linear_branch/diffusers(Python) |
| 本地 Qwen3-VL-4B（BF16 / int8-convrot）| `visual.pos_embed` 形状与 H3 期望**不符** |

⇒ `--first-frame`/`--last-frame` 不可用 ⇒ 分段只能硬切。

## 256×256 可行性预估（下一步，用户指示 2026-09-15）
- tokens ∝ 空间×时间：256² 的空间 token 约为 864×480 的 **16%**；
  15s vs 6s 帧数 ×2.5 → 15s@256² 的 token 总量 ≈ 864×480@6s 的 **0.4×**。
- 864×480@6s 单步激活 +3.77 GiB → 外推 15s@256² 单步 ≈ **1.5 GiB** → 单次 15s 大概率可跑通。
- 建议仍带 `h3_host_memory_guard`（已实现）跑，作为安全网。

### 256×256 / 15s 实测（2026-09-15，验证预估 ✅）
| 阶段 | footprint | available |
|---|---|---|
| DiT 加载 | — | 5.9 GiB（9/50 块常驻）|
| denoise 峰值（3/4）| **11.43 GiB** | 2.67 GiB |
| VAE decode | 1.34 → 1.47 GiB | 12.64 → 8.42 GiB |

- 产物 `out256_15s.mp4` = **15.08s / 256×256 / 含音频 / 659 KB**，总 **898s**，RSS 8.37 GiB，零报错。
- **结论：低分辨率单次 15s 可行**（864×480 同条件则触发守卫/死机）。估算 1.5 GiB/步与实际吻合。

### int6 vs int8 @ 256×256 / 15s 对比（2026-09-15）
| 指标 | int6 native | int8（BF16 4step + 运行时 int8）|
|---|---|---|
| denoise 峰值 footprint | 11.43 GiB | **9.91 GiB** |
| DiT 常驻块数 | 9/50 | **8/50** |
| VAE decode available | 12.64→8.42 | 11.56→7.89 |
| 总耗时 | 898s | **853s** |
| RSS | 8.37 GiB | **7.23 GiB** |
| 产物 | 15.08s | 15.08s（627 KB）|

**反直觉结论**：int8 权重严格更大（BF16 4step 22.9 GB / 预量化 int8-g64 更比源 +12.5%），
但 **adaptive residency 只常驻 8 块**（int6 为 9 块）——单块更大 ⇒ 同 available 下常驻块数更少
⇒ **常驻内存更低**，抵消权重劣势并略快（−45s）。**16 GiB 上 int8(BF16+runtime) 比 int6 native 更划算。**

# ===== video VAE 重复加载实测 + DiT GPU 侧剖析 (2026-09-15/16) =====

## 一、VAE 的 146s 在哪：同一份权重被读了 24 次

### 取证（864×480×56 / steps 2）
| 证据 | 值 |
|---|---|
| `[vae-debug] stream tile` 打印次数 | **24**（= 8 tiles × 3 chunks）|
| `per-block_bytes` | 268.6 MB（36 块 = 9.67 GB/次遍历）|
| 预期总读量 | 24 × 9.67 GB = **232.0 GB** |
| profile `alloc` | **216.520 GiB = 232.5 GB**（吻合 99.8%）|

```
video VAE decoder  wall=242.837s  wait=96.700s  root-gpu=22.272s
242.837 − 96.700 = 146.1s ← CPU 侧搬权重，占该阶段 60%
```

### 结构根因
```c
// decoder_decode_chunk（tile 在【外层】）
for (tile_y) for (tile_x) {
    prepare_input(...);
    run_stream_tile(...);   // ← 内部 load 全部 36 块，跑完即 free
}
// run_stream_tile
for (index = 0; index < LAYERS; index++) { load_block(index); run_block(); free_block(); }
```

原注释的判断是错的：
> "Serial decode is correct; the lost overlap only marginally slows the VAE stage."

实测 I/O+CPU 146s vs GPU 96.7s —— **不是 marginal，是 60%**。

## 二、block-major 重构：为什么只需要一份 hidden

### 依赖分析（决定可行性的关键）
- `run_block` 只读写 `hidden/norm/qkv/query/key/value/heads/branch/ff1/activated` + rope。
  **不碰 `latent` / `post` / `patch_hidden`**。
- `unpack_frame_range` 只读 `projected`。

⇒ 每个 tile 的进展状态**只用一份 `hidden`** 就能完整表达，不需要保留整个激活栈。

| 量 | 值 |
|---|---|
| 每状态 = `sequence × HIDDEN × 4` | 2028 × 2048 × 4 = **16.6 MB** |
| 864×480 每 chunk（8 tiles） | 133 MB |
| 864×480 跨 chunk（3×8=24） | **400 MB**（实测 peak 0.654 → 1.025 GiB）|

### 三级路径（自动选择，无开关）
| 路径 | 条件 | load_block 次数 |
|---|---|---|
| **cross-chunk** | `streaming && chunks×tiles×seq×HIDDEN×4 ≤ 1 GiB` | **36** |
| per-chunk | 同上但超限 | 108 |
| per-tile | resident 或更小预算 | 864 |

`states` 上限 1 GiB ≈ 64 个状态 ≈ 22.9 MB 预算/状态。触发回退的规模：
>8s 的 864×480（11×8=88）或 1080p（3×40=120）。

### 逐字节无损的论证与实测
每 tile 的块执行顺序（0→35）完全不变，只是把"tile 间交错"改为"块间交错"；
中间状态通过设备内 `h3_gpu_copy_f32` 存取，不经过主机。实测：

```
864×480×56/2 步，跨 chunk vs 原始 baseline
SSIM Y:1.000000 (inf)  PSNR inf     ← 逐字节一致
```

对照：`H3_VAE_TILE_PIXELS=512`（改 tile 几何）虽然也省 90s，但 **SSIM 0.854 / PSNR 26.7 dB，有损**。

### 实现陷阱
1. **两条 decode 入口都要改**。`h3.c:2273` 是三元表达式：
   `preview_decoder ? h3_video_vae_decoder_decode(...) : h3_video_vae_decode(...)`。
   只改 `decoder_decode_chunk` 时实测 `stream chunk calls: 0`——实际走的是 `decode_chunked`。
2. **copy 必须与 pack 同处一个命令缓冲**。放在 `submit` 之后会静默失败（`exit=1` 且错误信息为空，
   因为 `h3_gpu_error()` 返回空串）。

## 三、DiT 性能测量的两个陷阱

### 陷阱 1：自适应驻留使 A/B 不可复现
```c
// h3_dit.c:2742
if (dit->ssd_streaming && resident_budget == 0) {
    uint64_t avail = h3_host_available_memory();   // ← 运行时系统可用内存
    ... auto_resident = h3_dit_resident_budget(usable, per_block=0.5GiB, ...);
```
| 运行 | `available` | 常驻块 | peak |
|---|---|---|---|
| baseline | 4.3 GiB | 6 | 6.560 GiB |
| tile512 | 4.4 GiB | 6 | 6.560 GiB |
| resident 尝试 | 6.1 GiB | 10 | 9.431 GiB |
| `--layers 45`（未钉住）| 5.7 GiB | 9 | **11.120 GiB → swap → 314s 假数据** |

⇒ **测 DiT 必须 `export H3_DIT_RESIDENT_BLOCKS=6`**（设 `0` 无效——`resident_budget == 0` 正是自适应触发条件）。
钉住后四次运行 peak 稳定在 6.096~6.099 GiB。

### 陷阱 2：`--reuse` 在低步数下无意义
`--reuse 2` 评估首尾 + 每个间隔。steps=2 时无可跳之步 → 75.468s ≈ base 75.015s（+0.6%）。
必须在真实步数（20）下测，才看到 −45.9%。

## 四、DiT A/B 实测（864×480×22，钉住 6 块）

| 配置 | 2 步 | 20 步 | vs base | SSIM(All) | PSNR |
|---|---|---|---|---|---|
| base | 75.015s | 775.689s | — | — | — |
| `--layers 45` | 67.838s | — | −9.6% | 0.807 | 19.23 |
| `--token-reduction` | 49.076s | — | −34.6% | 0.822 | 18.63 |
| `--reuse 2` | 75.468s | 419.618s | −45.9% | 0.745 | 19.25 |
| combo | 44.149s | **273.959s** | **−64.7%** | 0.643 | 14.54 |

**收益与代价都是乘积叠加**：`0.55(reuse) × 0.90(layers) × 0.654(tr) = 0.324`（−67.6%），实测 −64.7%。
dispatch 交叉验证：base `linear=4000`（20 步×50 块×4 个 linear）→ `reuse 2` 的 `2200` 证明**只跑了 11 次前向**。

### 画质：数值差但目视不崩
`base vs combo` 的 SSIM 0.643 / PSNR 14.54 dB 看着很糟，但 `/tmp/s20cmp_{base,combo}.png`
目视：**狐狸结构完整、毛发清晰、背景正常**，差异是**运动相位偏移**（帧 1 位置/朝向不完全一致）。
逐像素 SSIM 对运动相位极敏感，会把这个放大成很差的分数。

⇒ 本质是"**换一条同样合理的生成轨迹**"，不是 artifact。`--reuse` 是时间外推，必然改变轨迹。

## 五、权重 I/O 与分辨率无关（决定优化方向翻转）

DiT 每读一遍权重矩阵的字节数**与激活大小无关**：

| 配置 | 每次评估 I/O | 耗时 @0.997 GiB/s |
|---|---|---|
| `--layers 50` | ~16.0 GiB | ~16.0s |
| `--layers 45` | ~14.4 GiB | ~14.4s |

验证：576×320/10 步/combo = `6 次评估 × 14.4 = 86.4s`，**实测 86.741s**。

| 场景 | denoise wall | GPU wait | SSD stream | unhidden wait |
|---|---|---|---|---|
| 864×480, 20步, base | 775.689s | 774.664s | 335.4s | 0.010s |
| 864×480, 20步, combo | 273.959s | 273.038s | 160.5s | 0.590s |
| **576×320, 10步, combo** | **92.256s** | 58.880s | 86.741s | **33.227s** |

⇒ **降分辨率后 I/O 反超 GPU**（`unhidden wait` 0.010 → 33.227s）。
计算量降 55.6%，时间只降 33%：`864×480` 同配置估 137s → 576×320 实测 92.3s。

| 场景 | 瓶颈 | 有效 | 无效 |
|---|---|---|---|
| ≥864×480 | GPU 计算 | `--token-reduction` | — |
| ≤576×320 | **权重 I/O** | `--layers`、`--reuse` | **`--token-reduction`** |

### 注：`root-gpu` 字段不可信
同一配置下它还报过 0.298s / 0.400s / 121.995s，与 `wait` 完全不成比例。
README 自述它 "can omit child buffers scheduled internally by MPSGraph"。**以 `wait` 为准**。

## 六、三条外部路线对本机的适用性（调研结论）

| 路线 | 判定 | 理由 |
|---|---|---|
| **FREE**（arXiv 2511.20390，CVPR2026 Findings）| 部分/不适用 | draft-verify 的收益来自 batch 并行填满 GPU 空闲；本机长序列单步已 compute bound（`wait` 775s vs I/O 335s），batch-K 验证 ≈ K× 耗时。且本仓库已有同思路的 `H3_FB_CACHE`（零阶重放） |
| **A-SelecT** | 不适用 | 解决的是 DiT 作判别式特征提取器时选时间步，不是生成。且"选时间步子集"这条路 README 已记录走过并否决（linear base grid 胜出）|
| **Chimera** | 等权重 | 混合线性注意力的**算子已就绪**（`--linear-branch` + vdn cholesky/triinv/window kernels），卡在社区权重。且本机 `--linear-branch` 与 `--ssd-streaming` 互斥 → 需更大内存 |

### 附带发现：`H3_FB_CACHE` 在本机不可用
```c
// h3_dit.c:3223
 dit->fb_cache = !ssd_streaming && fb_cache && *fb_cache && strcmp(fb_cache, "0");
```
注释说明原因：SSD 流式的前取环形缓冲假设每个 block 都按顺序消费，中途跳出会失同步。
而 16 GiB 必然 `ssd_streaming=1` ⇒ 该开关恒被忽略、复用率恒为 0。

## 七、GGUF / 量化路线的实测否定

- **仓库无 GGUF reader**（全仓 grep `gguf` 为空）。要走 GGUF 需自写 k-quant 反量化 kernel；
  而引擎**已支持预量化流式读取**（`h3_dit.c:1737` 的 `source->is_grouped`），
  转成 group-quantized safetensors 即可，**零 C 代码改动**。
- **GGUF 本身不压缩**——省 I/O 靠位宽（bit/weight），不靠容器。
- **但现有量化产物全是 int8**：`h3_int4g64`/`h3_int6g128`/`h3_int8g64` 三个目录
  dtype 分布、每张量字节数、目录总大小**完全一致**，`qkv_proj.weight [21504,5376]` 实测
  **8.00 bits/weight**，取值域 −127..127（252 个不同值）。
  ⇒ `int4g64` 未实现；int6 若按 1 byte/weight 存放则**对 I/O 零收益**。
- **量化质量代价实测**（`ab_report/report_steps4.json`）：int6g128(实为 int8) vs BF16
  → SSIM 0.705 / PSNR 17.47 dB。**H3 对量化敏感**。
- 在 compute bound 下，量化省 I/O 换不回时间；只有在 ≤576×320 档才有意义。

## 八、Latent Upscaler 接入：原语、桥接、内存墙（2026-09-16/17）

### 1. B3 最大的未知数半路消失：GroupNorm 仓库里本来就有

放大器需要的归一化是 `GroupNorm(32,C)`，`in_layers` / `TemporalConv.norm` 后面都接 SiLU。
而 `h3_gpu_vae_encoder_group_norm_silu_f32`（`h3_shaders.metal:644`）**正是 NDHWC 布局下的
GroupNorm(groups,C) + SiLU** —— 连 `vae_encoder_norm_args` 都是同一套字段。

⇒ B3 从"新写 GroupNorm"缩成"补一个去掉 SiLU 的变体 + 一个逐通道仿射"。
`out_norm` 之后接的是 AdaLN 式调制（`h*(1+scale)+shift`），所以必须有一个不带激活的版本。

其余全部可复用现成原语：`h3_gpu_silu_f32` / `h3_gpu_linear_f32` / `h3_gpu_add_scaled_f32` /
`h3_gpu_conv3d_pad_f32` ⇒ **网络可用纯 C 原语搭建，不必走 MPSGraph 组合**。

放大器索引映射（此前未理清）：
```
res_index(b)  = b + (b+1)/2        # b = 0..11
temp_index(b) = res_index(b) + 1   # 仅 b 为偶数时存在（temporal_every=2）
```
张量数核对：in_blocks 156 + out_blocks 156 + 边界 10 = **322** ✓

### 2. 两端归一化一致 ⇒ 桥接只换轴序，绝不重缩放

```python
# ComfyUI comfy/ldm/minimax/vae.py:698-702
def decode(self, z, output_buffer=None):
    # z: [B, 24, T_lat, H_lat, W_lat] normalized latents
    z = z * latents_std + latents_mean
```
```c
/* h3c h3_video_vae.c:361 */
rows[row++] = input[source] * deviation[channel] + mean[channel];
```
**逐字等价**，且 per-channel 常量逐位相同。⇒ ComfyUI 的 `latent["samples"]` 与 h3c 的 `z`
是同一空间，桥接只做 `[C,T,H,W] ↔ [B,C,T,H,W]`。

放大节点内部的 `(s-mean)/std` 与出口的 `out*std+mean` 是它自己的进出配对，对桥接透明。

### 3. latent 包读写的三个坑（都是阻塞级）

| # | 坑 | 症状 |
|---|---|---|
| 1 | `_read_h3_latent` 只认 v2，引擎**恒写 v3** | `H3_BinaryLatent` 对任何输入抛"版本不支持" |
| 2 | `_write_h3_latent` 写 v2（24B 头），但 reader **无条件 `fread` 7 字=28B**（`h3.c:2578` 无 rewind） | 4 字节错位：视频整体偏移、`ap` 读到音频头首字段而误判为"有音频"、音频长度取到音频数据的头 4 字节 → `truncated audio latent`。**`H3_BinaryLatentDecode` 从未工作过** |
| 3 | 放大器只返回 `{"samples":...}` | 丢掉 `h3_audio` → 下游退回纯噪声初始化（σ=1）而 schedule 从 refine_sigma 起步 → 分布外输入 |

引擎 writer 恒写 v3 正是因为 reader 恒吃 28 字节 —— 所谓"v2 向后兼容"在 reader 侧实际是坏的。

**实测验证**：修好后 `h3_audio` 穿过放大器的相关系数 **+1.0000**，rms 逐位相同（0.00360）。

### 4. 放大器不是瓶颈（实测推翻估计）

345M 参数、CPU、fp32：

| | 前向 | 外推 |
|---|---|---|
| 20×12→40×24, T=7 | **2.44s** | — |
| 40×22→80×44, T=37 | **60.9s** | 体积 19.4× ⇒ 47s（实测 60.9s）|

此前"CPU 上要十几到几十分钟"的估计是错的。⇒ **B4（把放大器搬进 h3c）的收益只剩
"去掉 ComfyUI 依赖"，性价比存疑。**

### 5. 内存墙：DiT 的 token 上限（决定这条路能走多远）

DiT 激活随 token 数线性增长，**无 token 维度分块**（引擎只做激活别名复用，
864-class 省 99.63 MiB，`H3_DISABLE_DIT_ACTIVATION_ALIAS=1` 可关）。

| 配置 | tokens | 结果 |
|---|---|---|
| render 640×352, 121 帧 | 32,560 | ✅ 跑通 |
| render 1280×704, 22 帧 | 24,640 | ✅ 跑通（96.6s/步）|
| **render 1280×704, 121 帧** | **130,240** | ❌ **系统重启** |

`tokens = (宽/16)×(高/16)×T`，`T = 5k+2`（帧数按 24fps 换算后对齐到 `5+17k`）。
经验上限约 **30k**。

**关键推论：直接全分辨率生成是同一个 token 数 ⇒ 同样跑不动。**
所以"两阶段 vs 直接"在 1280×704/5s 上**不成立**。可行的是
**变体 B（低分生成 → 放大 → 直接 VAE 解码）**：DiT 只在低分辨率下运行。

### 6. 自适应驻留毁测量（再次确认，且量化了）

`H3_DIT_RESIDENT_BLOCKS=0` ⇒ 读系统可用内存自适应决定常驻块数。
**内存越松它越激进，而多常驻反而更慢**（常驻块与流式路径争抢统一内存）。

受控 A/B（256×256 / 1s / 2 步，只改驻留）：

| 常驻 | peak | denoise | SSD 读取 | 带宽 |
|---|---|---|---|---|
| **1** | **2.29 GiB** | **34.4s** | 35.9 GiB | **1.026 GiB/s** |
| 11 | 9.46 GiB | 50.6s | **32.3 GiB**（更少） | 0.888 GiB/s |

常驻=1 时 `denoise(34.4s) ≈ SSD(35.0s)` ⇒ **计算被 I/O 完全掩盖**；
常驻=11 时多出 14.2s 非重叠停顿。

**最严重的一次**：steps=4 报出 422s，钉住 1 块后重测为 **91s —— 差 4.6 倍**。
⇒ 所有跨运行的性能对比必须在 `H3_DIT_RESIDENT_BLOCKS=1` 下做。

### 7. int8 VAE 省磁盘不省内存

| | BF16 | int8_convrot |
|---|---|---|
| 磁盘 | 14.6 GiB | **2.95 GiB** |
| 运行时常驻 | ~9.7 GB | **9.67 GB**（`total_block_bytes=9668689920`，peak 9.365 GiB）|

加载时**反量化成 F32 常驻** ⇒ 对内存天花板无帮助。
DiT 侧同理：文件 21.4 → 20.67 GiB（−4%），`SSD stream` 每步仍读 ~17.7 GiB。

### 8. 8step 蒸馏模型必须走 8 步（横向标尺）

文件名 `fastvideo_fasth3_8step_v2_pruned_int8_convrot` ⇒ 训练步数 8。

标尺：`0.551` = 换种子的两次无关生成；`0.977` = 同参数仅步数不同的常态。

| 对比 | SSIM |
|---|---|
| 2步 vs 8步 | **0.581**（≈ 无关生成）|
| 2步 vs 4步 | 0.603 |
| **4步 vs 8步** | **0.680** |

⇒ 2 步**未收敛**，出的不是提示词对应的内容；4 步在改善但没到位。

耗时（`H3_DIT_RESIDENT_BLOCKS=1`，256×256 / 1s）：
```
总墙钟 ≈ 22.5s（固定开销）+ 17.1s × steps
  2步  57s    4步  91s    8步 159s
```
每步恒定 17.1s，且 `SSD 时间 ≈ denoise 全部时间` ⇒ **该配置纯 I/O bound**（1.03 GiB/s）。

### 9. 引擎的模型发现机制

`-d` 只设根目录，之后按硬编码相对路径解析，两类检查：

- **精确文件**：`FL2VA/transformer/config.json`、`FL2VA/tokenizer/tokenizer.json`
- **目录扫描**（`h3_st_inventory_dir`）：`readdir` 取**所有以 `.safetensors` 结尾且非点开头**
  的文件，读 header 统计；**空目录报错** `no safetensors files`

⇒ **文件名完全无关**，只要放进对的目录。同目录多个 `.safetensors` 会被**合并成一个逻辑组件**
（按张量名查找），所以 DiT 目录被报成 "2 files 1102 tensors"（主权重 + time_embedder）。

`FL2VA/text_encoder` 仅在 `H3_CLIPPROJ_DIR` 未设/为 `0`/`off` 时才必须（`h3.c:775-798`）。
`Ref2VA/transformer/model.safetensors.index.json` 存在与否开关 R2V。

---

# ===== Findings（当前任务）: ANE int8 投影移植（h3.c-ane → h3c）=====

## F1. 决策 A 关闭：磁盘 int8 payload 已经是 ANE 要的格式

`dbg_ane_int8_format.py` 对 `fastvideo_fasth3_8step_v2_pruned_int8_convrot.safetensors`
block 0 的 qkv/out/fc1/fc2 四项实测：

- dtype=I8，shape 与图常量假设的 `[out,in]` 完全一致
- `weight_scale` 为 **每输出行一个 F32**（count == rows），ANE 的
  `constexpr_affine_dequantize(axis=0, scale=fp16[N], zero_point=int8(0))` 直接可用
- `zero` 张量不存在（对称量化）→ 无需重新居中
- bias 不存在 → MIL conv 无 bias 操作数这件事不构成障碍
- scale 转 fp16 最大相对误差 ~4.8e-4，0 个 subnormal / 0 个 flush-to-zero

⇒ **不需要任何 requant / 转 per-row / 重排预处理**。第 1 步的前提成立。

## F2. LoRA 不能折进 int8 权重，只能走高精度低秩旁路

`dbg_lora_fuse_convrot.py` 的判据：`SNR = RMS(delta)·sqrt(12) / per-row step`。
delta 只有 ~0.2%·||W||，与 per-row int8 噪声同量级；折叠后
「有效 delta vs 真 delta」relRMS ≈ 1.0、cos 0.02–0.2 ⇒ LoRA 被量化噪声淹没。
（陷阱：只比折叠前后整个权重的话，relRMS 一直 ~0.002，看不出问题。）

fork 的答案是**图内多 band 旁路**：打包时预旋转 `A_rot = A·R`，
于是旁路可以复用主 conv 已经算出的旋转激活 `rᵢ`——
不加输入绑定、不多一次 eval；每 band 自己的行 → `concat(axis=1)` → 一次 `add`。
qkv 是 module-major，需要 3 个 band（to_q/to_k/to_v）；其余 1 个。

## F3. 上午「08:37 全 PASS → 08:57 真实 gate FAIL」不是代码回归，是磁盘

证据链（全部今天实测）：

1. `/tmp/ane_base_only.log` 08:37：含 real-qkv（126.5 MiB 权重）全 PASS。
2. `/tmp/ane_final.log` 08:57：real-qkv + real-lora 双双 `ANECCompile() FAILED`。
3. `/tmp/ane_repro_0919.log`：只有 real-lora FAIL，错误串是 `"?"`
   ——`h3_ane_bridge.m:210` 在 NSError 为 nil 时打印 `"?"`，即编译框架没给原因。
4. 直接前台复跑：**SIGBUS，exit 138 / 经 script 包装后 10**，日志 0 字节
   （SIGBUS 杀进程时管道全缓冲的 stdout 直接丢失，所以「空日志」本身就是被杀的信号）。
5. `df`：`/System/Volumes/Data` 228 Gi 用了 199 Gi，**只剩 4.8 Gi（98%）**。
6. 缓存目录 mtime 与崩溃时刻吻合：real-qkv 命中缓存所以「通过」，
   real-lora 需要全新的大编译 → 失败。**只有新鲜编译会挂**，这正是磁盘症状。

⇒ ANE_PORT_SUMMARY §7 预写的「磁盘不足会 SIGBUS/InvalidMILProgram」被命中。

## F4. TMPDIR 搬迁对 ANE 编译无效（重要，别再试）

试过 `TMPDIR=/Volumes/data/tmp/ane-tmp ./h3_ane_int8_test ...`：
崩溃点、exit code 完全一样，且新目录**始终是空的**，而
`/var/folders/f8/.../T/h3-ane-cache` 的 mtime 跟着每次运行更新。

原因：`h3_ane_bridge.m:77` 和 `:183` 用的是 `NSTemporaryDirectory()`，
它走 `confstr(_CS_DARWIN_USER_TEMP_DIR)`，**不读 TMPDIR 环境变量**。
所以 bridge 的 staging 目录和 h3-ane-cache 永远落在系统卷。
（顺带：`h3_ane_linear.m:515` 的 `H3_ANE_DUMP_MIL` 更是硬编码 `/tmp/ane_dump_%s`。）

⇒ 唯一可行的解法是**给系统卷腾空间**，不是改环境变量。

## F5. 空间账本（Data 卷，用于决定清什么）

| 路径 | 大小 | 归属 |
|------|------|------|
| `~/h3_sys` | 31G | 用户模型权重 |
| `~/Library/Application Support` | 18G | 用户 |
| `/private/var/folders`（整个 Darwin 临时域） | 4.0G | 混合 |
| `~/Library/Caches` | 3.9G | 用户，可再生但需确认 |
| `~/Library/Containers` | 3.4G | 用户 |
| `~/.cache` | 2.5G | 用户 |
| `T/h3-ane-cache` | 400M | **我们的**，删了只是重新编译 |
| `T/94FBBC93…`（两个孤立 staging） | 131M + 263M | **我们的**，崩溃残留 |

## F6. 「真实 gate 跑不过」的真因是 fork 测试里的两处少乘 sizeof，不是 ANE、也不只是磁盘

`lldb` 抓到的崩溃现场：`main+13116` 处 `str d1, [x24, x21, lsl #3]`，
`EXC_BAD_ACCESS code=2`，故障地址每次都在**页起始**（ASLR 变、页对齐不变）
= 写到一段映射之后的第一个未映射页 = 堆溢出。
内层 `cmp x5, #0x20`（32 = rows）定位到参考实现里 `want[n*rows+s]` 的写入循环。

`tests/test_ane_int8.c` 里两处漏了元素大小（fork 测试代码，非库代码）：

| 行 | 原代码 | 需要 | 实际 |
|----|--------|------|------|
| 886 | `float *x = malloc((size_t)in_dim * rows)` | 688 KB | 172 KB（写越界 4 倍） |
| 926 | `double *want = malloc((size_t)out_dim * rows)` | 5.5 MB | 688 KB（写越界 8 倍） |

同类分配在 `:38 / :415 / :619` 都正确带 `sizeof(double)`，只有 `real_lora_gate` 这两处漏了。
（`x` 那处更阴：不崩，只是参考值读到越界垃圾，即使侥幸不死也过不了 gate。）

改完 `make h3_ane_int8_test` 后全绿：

```
real-qkv   K=5376 N=21504 kc=1024 rows=32 gs=256: cos=0.9999998 rel_l2=6.415e-04 nonfinite=0 weights=126.5 MiB PASS
real-lora    K=5376 N=21504 kc=1024 rows=32 gs=256: cos=0.9999998 rel_l2=6.729e-04 max_abs=7.039e-02 nonfinite=0 PASS
0 failure(s)
```

⇒ **Phase A 关闭**：真实 checkpoint 的 int8 主卷积 + 真实 Turbo LoRA 的
3-band 旁路，端到端对 double 参考余弦 0.9999998。图本身没有任何问题。

## F7. 两个成因要分清，别把磁盘结论过度外推

- 08:37→08:57 那两次 `ANECCompile() FAILED`（含 real-qkv）发生在**编译期**，
  而两处少 sizeof 的越界发生在**其之后**的参考计算，所以那次确实像磁盘紧；
  腾出 ~800 MB（删我们自己的 `h3-ane-cache` 400M + 崩溃残留 staging 394M）后新鲜编译即可通过。
- 今天 repro 的 SIGBUS 则是**堆溢出**，与磁盘无关。
- 也就是说：`ANE ... compile failed: ?`（`h3_ane_bridge.m:210`，NSError 为 nil 时打 `"?"`）
  这条错误串信息量为零，将来排障要先把 NSError 的 domain/code 打出来。
- 系统卷目前仍只有 ~5.0 Gi 空闲（98%）。**Phase B 之前需要用户决定是否清理**
  `~/Library/Caches` 3.9G / `~/.cache` 2.5G 这类可再生大目录（不确认不动）。

## F8. 移植注意：带这两处 bug 的是 fork 的测试文件

Phase B 把 `h3_ane_bridge` / `h3_ane_linear` 搬进 h3c 时，
测试也要一起搬，但必须带上这两个 `sizeof` 修正——
否则会在新仓库里复现同一个「神秘 SIGBUS / compile failed ?」。

## F9. 「ANE 为什么不在外置 SSD 上产生临时缓存」——实测：不能，卡在编译器沙箱

先量化一次新鲜大编译到底花掉多少内部盘空间（`H3_ANE_CACHE=0`，real-qkv 126.5 MiB 权重）：

| 指标 | 前 | 后 | 差 |
|------|-----|-----|-----|
| `/System/Volumes/Data` free | 5,503,808 KiB | 5,372,356 KiB | **−128 MiB** |
| `getconf DARWIN_USER_TEMP_DIR` 用量 | 67,080 KiB | 196,668 KiB | **+129,588 KiB** |

⇒ 掉的空间**全部等于我们交给 aned 的 staging 目录**（`MIL 文本 + weight.bin +
编译产物 data/net.plist/compiled.ok`）。缓存条目实测 127M/图（`data` 里已经嵌了常量）。
也就是说不存在「apple 偷偷在内部盘存了个巨大的东西」——东西是我们自己写的，
只是写在了内部盘。

于是试着把 staging 挪走（给 `h3_ane_bridge.m:77/183` 加 `H3_ANE_TMP` 基准目录，
`bridge_cache_entry` 和 staging 一起换根，保证 `bridge_mirror()` 的硬链接仍同卷）：

| `H3_ANE_TMP` | 结果 |
|--------------|------|
| 未设（`NSTemporaryDirectory()`，系统卷） | `0 failure(s)`，新鲜编译通过 |
| `/Volumes/data/tmp/ane-staging`（USB SSD，1.9 T 剩 546 G） | **16 个 gate 全挂**：`ANECCompile() FAILED` |
| `/tmp/ane-base`（普通内部卷路径，但不在 Darwin temp） | **同样 16 个全挂** |

⇒ 决定性对照：**不是外置 SSD 的问题，是路径必须在用户的 Darwin 临时域里**。
`ANECompilerService`（root 的 XPC 沙箱服务）只被允许访问
`confstr(_CS_DARWIN_USER_TEMP_DIR)` 那一块；换任何别的路径它都读不到源，
直接 ANECCompile 失败。所以「把 ANE 编译缓存搬到 SSD」这条路是**死的**。

副作用记录：搬到 SSD 那次 `DARWIN_USER_TEMP_DIR` 仍涨了 ~141 MB（我们的 staging 不在那儿），
说明编译器服务还会在自己的容器里另写一份工作副本——进一步印证腾空间只能在系统卷上做。
`TMPDIR` 环境变量同样无效（F4），因为 `NSTemporaryDirectory()` 不读它。

**该 patch 已回退**（证明不可用的开关留在代码里只会变成下一个坑）；
fork 现在只剩两处 `sizeof` 修正。

### 由此得到的硬结论

内部卷剩余空间是 ANE 的**前置条件**，不是可以绕开的细节。预算：
每张不同的图 ≈ 128 MiB staging（编译期峰值）+ 127 MiB 缓存产物；
整 block 一张图的产物更大。多形状扫档时必须留 **GiB 级**余量，
否则又会看到 F7 那种 `ANECCompile FAILED`。

## F10. ANE 到底需要多少盘 —— 编译几乎免费，19 GiB 是"全量缓存"的成本

用真实 checkpoint 直接量（gate 里加了 `compile=%.2fs cache=%d` 打印）：

| 图（rows=32, kc=1024） | payload | 新鲜编译 | 缓存命中 | 缓存条目 |
|------|---------|----------|----------|----------|
| real-qkv `K=5376 N=21504` | 126.5 MiB | **0.11 s** | 0.04 s | 129.6 MiB |
| real-lora（同图 + 3-band 旁路，最大图） | ~130 MiB | **0.56 s** | 0.00 s | 134.6 MiB |

其他实测：
- 编译期瞬时峰值 ≈ 1× payload（`df` 掉 128 MiB，等于我们写进 staging 的 MIL+weight.bin）；
  **`h3_ane_model_free()` 会 `removeItemAtPath:staging_directory`**，所以进程正常退出零残留。
- A/B 隔离验证 `H3_ANE_CACHE`：`=0` → cache 目录 **0 个条目**；`=1` → **8 个条目**。
  之前一次观测到的"关缓存也留了 264 MiB"是同一命令串里**前一个开缓存的 run** 写的，看错时间戳了。
- 硬链接镜像（`bridge_mirror` 的 `linkItemAtPath`）⇒ staging 和 cache 条目共享 extent，
  不翻倍；unload 删 staging 后由 cache 条目持有那些块。

按 payload 汇总（直接读 safetensors header，不加载）：

```
transformer  attn  9,415 MiB + mlp 11,031 MiB = 19.0 GiB   (50 block, I8)
             每 block ≈ 414 MiB（qkv 126.5 + out ~37 + fc1 ~147 + fc2 ~73）
audio_vae    577 MiB (F32)      video_vae 3.0 GiB（子目录，需另算）
```

⇒ **ANE_PORT_SUMMARY 里那个"~19 GB/形状"不是编译门槛，而是"把 50 个 block 的编译产物全量缓存在盘上"的成本。**
真实需求分两种：

| 策略 | 常驻盘 | 一次性时间 | 结论 |
|------|--------|-----------|------|
| 关缓存（或 LRU 上限） | ~0，瞬时峰值 = 最大单图（整 block 一张图 ≈ 414 MiB/shape） | 50 block 重编译 ≈ 20–60 s/shape | **内部卷留 2 GiB 就够跑全流程** |
| 全量缓存 | **19 GiB / shape**，多一档 shape 再 ×19 GiB | 每次启动省 20–60 s | 当前 14 Gi 空闲装不下，必须加 LRU/白名单 |

顺带：内部卷 avail 从上午的 4.9 Gi 自己涨到 **14 Gi**（APFS purgeable 被回收），
说明当时那几次 `ANECCompile FAILED` 有"可回收空间没及时让出来"的成分。
即便如此，稳态仍建议 ≥ 8–10 Gi 空闲，且**Phase B 不再需要用户清理 `~/Library/Caches`**
——关缓存路线 2 GiB 就够。

移植时要注意的实现点：`h3_ane_bridge.m` 的 cache 目前**无上限、无淘汰**（只有
`bridge_cache_evict` 在 free 时按 identifier 删自己）。要上 50 block × 多 shape，
必须补一个按总字节的 LRU，或默认 `H3_ANE_CACHE=0` + 进程内复用。

---

## F11（2026-09-19 10:50，Phase B2a 完成）：h3c host 依赖已就位，且两条 un-rotate 路径可证等价

### 新增文件（全部在 h3c 侧，未动 fork）

| 文件 | 行数 | 来源 |
|------|------|------|
| `h3_convrot.h` / `h3_convrot.c` | 22 / 84 | fork 逐字移植（Hadamard 表缓存 + `cblas_sgemm` 分块 derotate） |
| `tests/test_int8_raw.c` | 155 | **新写**（fork 无此测试），Makefile 目标 `h3_int8_raw_test` |
| `h3_weights.c` | +143 | 新增 `h3_weight_load_int8_raw` + statics `weight_sidecar_name` / `comfy_quant_group_size` / `read_int8_scales` |
| `Makefile` | +9 | `LIB_C += h3_convrot.c`；`h3_int8_raw_test` 目标 + `test` 里带 `DIT_MODEL` 守卫的调用 |

### 移植时按 h3c 约定改写（不是复制）的三处

1. **读 payload 用 `pread_bytes()`**（h3c `h3_weights.c:232` 的本地约定），不用 fork 的
   `h3_st_read_data()` ⇒ 少一个 `detail[384]` 中间缓冲，错误信息按 h3c 风格自己 `fail()`。
2. **scale 名按 h3c 的 `"%s_scale"`**（`qkv_proj.weight` → `qkv_proj.weight_scale`），
   和 fork 的 `sidecar_name(..., "weight_scale")` 结果等价，但和 `load_int8_dequantized`
   共用同一条命名路径；`.comfy_quant` 侧车仍需剥掉 `.weight` 再接后缀。
3. **只接受 F32 scale**（`int8_tensorwise` 的磁盘契约），不像 `h3_dit.c:625` 那样
   兼容 BF16 —— 那个宽容是给流式路径的，ANE 侧要的是原始 F32，宽容只会掩盖格式漂移。

### 把「两套 Hadamard 实现」变成被测不变量（而不是隐患）

h3c 原本有两处 ConvRot：`h3_weights.c` 的 radix-4 butterfly（`convrot_unrotate_row`）
和 `h3_dit.c:780 build_convrot_hadamard`（喂 Metal kernel）。fork 的 `h3_convrot.c` 是
第三种写法（H4 的 Kronecker 幂）。三者数学上是同一个矩阵：butterfly 的 stride=1,4,16,64
就是在 base-4 各位上依次右乘 H4，Kronecker 递归正是同一件事。

实测坐实（两个测试，都不依赖 ANE）：

- `tests/test_convrot_unrotate.c` 扩了两项检查：
  `convrot table vs radix-4 butterfly: max_abs=0.000e+00`（表元素是 ±1/16，bf16 精确可表示，
  所以能要求**逐元素 0**），`table derotate vs cpu exact: max_abs=0.000e+00`。
- `tests/test_int8_raw.c` 在真实 checkpoint 上比对「表 derotate」vs「butterfly un-rotate」：
  4 个投影 `rel_rms=1.3e-08 ~ 3.9e-08`，`max_abs≤8.4e-07`（只剩 sgemm 与 butterfly 的
  浮点求和次序差），全部 `0 failure(s)`。

⇒ 以后任何人改坏其中一条路径，`make test` 当场红，不用等到 ANE 投影对不上 Metal。

### 真实 checkpoint 侧车确认（读 header，不加载权重）

```
/Users/jay/h3_sys/MiniMax-H3-Convrot/FL2VA/transformer/
  fastvideo_fasth3_8step_v2_pruned_int8_convrot.safetensors   21 GiB
  blocks.0.attn.qkv_proj.weight        I8  [21504, 5376]
  blocks.0.attn.qkv_proj.weight_scale  F32 [21504, 1]
  blocks.0.attn.qkv_proj.comfy_quant   U8  [72] =
      {"format": "int8_tensorwise", "convrot": true, "convrot_groupsize": 256}
```
（`test_int8_raw.c` 首次运行还顺带量到每投影的行 scale 区间：
qkv 6.5e-4~7.1e-3、out 1.5e-3~9.9e-3、fc1 2.4e-4~7.9e-3、fc2 2.2e-3~9.8e-3。）

### 待办衔接

- `h3_dit.c:780 build_convrot_hadamard` 与 `h3_weights.c` butterfly 仍各自持有实现，
  现在有了测试兜底，**收敛成一处**是后续独立小改动（不阻塞 Phase B2b）。
- B2b 需要：`-framework IOSurface` 尚未加进 `Makefile` 的 `FRAMEWORKS`。

---

## F12（2026-09-19 11:05，Phase B2b 完成）：ANE bridge/linear 已在 h3c 内独立跑通全部 12 个关

### 落地的文件

| h3c 新增/改动 | 行数 | 说明 |
|---------------|------|------|
| `h3_ane_bridge.h/.m` | 53/345 | fork 逐字拷贝，一次编译通过（只依赖 Foundation/IOSurface/objc_msgSend） |
| `h3_ane_linear.h/.m` | 207/1053 | 同上；它需要的 host 依赖恰好就是 B2a 补的那三个（`h3_convrot`、`h3_weight_load_int8_raw`、`h3_gpu` 老 API） |
| `h3_gpu.{h,m}` | +26/+90 | `h3_gpu_tensor_wrap_f32`（无拷贝收养 IOSurface 基址）、`h3_gpu_tensor_host_pointer`、`h3_gpu_pack_ane_input_bf16`、`h3_gpu_unpack_ane_output_bf16` |
| `h3_shaders.metal` | +41 | kernel `h3_ane_pack_bf16` / `h3_ane_unpack_bf16` |
| `Makefile` | +12 | `LIB_M += h3_ane_{bridge,linear}.m`、`-framework IOSurface`、3 个测试目标 |
| `tests/test_ane_int8.c` | 1030 | 从 fork 拷入，**带上了两处 `sizeof` 修复**和 compile/cache 打点 |
| `tests/test_ane_staging.c` | 165 | **新写**（fork 没有）：Metal pack/unpack 位级往返 |

### 移植中撞到的两处 h3c 差异（都不是复制粘贴能解决的）

1. **`h3_gpu_require_bf16` 已存在**（h3c `h3_gpu.m:2567`，我最初 grep 用 `head -5` 截断了输出，
   以为没有，就在文件尾部补了一份 → `redefinition`）。删掉重复实现，复用原有的。
2. **kernel 必须显式注册**：h3c 的 `gpu.pipelines` 是一张写死的名单
   （`h3_gpu.m:534` 起，`[names addObject:...]`），名单外的 kernel 运行期取不到
   （表现为 `missing Metal pipeline h3_ane_pack_bf16`，而不是加载失败）。
   且这段名单嵌在 `if (gpu.tensorOpsEnabled)` 里 ⇒ ANE staging 两个 kernel 注册在
   **该 if 之外**，否则关掉 TensorOps 的档位就取不到 pipeline。

### 数值结果（h3c 内，`H3_ANE_CACHE=0`）

```
rot-padded/plain-2048/bypass-r32/r128/zeroB/zeroA/1chunk/bands-qkv/cache-shape  全 PASS
real-qkv  K=5376 N=21504 gs=256: cos=0.9999998 rel_l2=6.415e-04 compile=0.43s cache=0 PASS
real-lora K=5376 N=21504 gs=256: cos=0.9999998 rel_l2=6.729e-04 compile=0.50s cache=0 PASS
0 failure(s)
```
真实 Turbo LoRA 文件：
`/Volumes/data/.lmstudio/models/LightX2V/MiniMax-H3-Turbo/minimax_h3_fl2v_turbo_4step_v1.1_768p_bf16.safetensors`
（50 block × `attn.to_{q,k,v}.lora_{A,B}.default.weight`，`__metadata__` alpha=128/rank=128 ⇒ scale=1.0）。

### 对 F10「关缓存进程退出零残留」的修正（重要）

`tests/test_ane_int8.c` 的 `cache_gates()` 会 `setenv("H3_ANE_CACHE","1",1)` 且**从不还原**，
所以同一个进程里其后的 real-qkv/real-lora 实际是**开着跨进程缓存**跑的：
外部 `H3_ANE_CACHE=0` 被静默覆盖，`cache=1`、每次退出留 ~130 MiB/形状
（实测两次运行后 `T/h3-ane-cache` = 2 条目 258 MiB）。⇒ 已补上「保存并还原调用方的值」，
还原后重跑：12 关全 PASS、真机关显示 `cache=0`、`T/h3-ane-cache` 剩 **0 B**。

（顺带：那个目录里 11 个 1.3–1.6 MiB 的十六进制命名目录**不是我们的** staging——它们是 10:03 之前
就有的、ANE 编译器服务自己的产物，本次只删了本会话写的 `h3-ane-cache`。）

### 尚未接线的部分

`h3_gpu_pack/unpack_ane_*` 现在只有独立测试在用；**h3_dit.c 还没有任何一条投影走 ANE**。
下一步（原计划 B3/B4）是把 `blocks.N.attn.qkv/out, mlp.fc1/fc2` 挂到 `H3_ANE_LINEARS` 开关后面，
并保留 Metal 对照路径 + cos 检查。注意 h3c 侧两个既有事实会决定接线方式：
- qkv/out/fc1/fc2 在 h3c 是**融合** kernel（`h3_gpu_grouped_qkv_linear_rope_int8`
  `h3_dit.c:4007`、`h3_gpu_mlp_int8_bf16` `h3_dit.c:4109`），走 ANE 等于把 RoPE/SwiGLU 拆回两步；
- h3c 的 LoRA 在加载时**已并入权重**（`h3_lora.c:170 h3_lora_apply`），
  而 F2 的结论是 int8 下 LoRA 必须以高精度旁路存在 ⇒ 接线前必须先确认这两条不冲突。

---

## F13：B3 接线前提核对 —— F12 提的「两个冲突」里只有一个成立（2026-09-19 11:55）

读源码把 F12 结尾两条待确认项各自查实，结论是**LoRA 那条不是冲突，融合那条才是真代价**。

### 1. LoRA：h3c 做的是「合并后重新量化」，不是 F2 否决的「折进既有 int8 网格」

- 流式 int8 路径顺序（`h3_dit.c:2084` 起，注释原文 "before the int8 requantization bakes
  them into the resident slot"）：
  **pread int8 → dequant+un-rotate 成 BF16 → `merge_block_loras(..., blocking=1)` 把
  ΔW 加进 BF16 → 再 requant 成驻留 int8（per-row scale 重新算）**。
- F2 的「LoRA 折进 int8 会掉到噪声底」针对的是**沿用原 scale/zero-point** 的就地折叠；
  h3c 在合并之后重算了行 scale ⇒ 是不同的算子，结论不能直接套过来。
- 所以对 ANE 而言：**喂 h3c 已有的驻留 int8 权重即可**，不需要旁路，也不需要改 h3c 的 LoRA 策略。
  合并+重量化若有精度损失，那是 h3c 现路径**已经付掉**的成本（Metal/ANE 两条线吃同一个张量），
  不是「选 ANE 引入的新损失」⇒ `cos(ANE, Metal)` 仍是有意义的对照。
- fork 里那条 `real-lora` 旁路 gate 的价值不变（它证明的是「旁路写法正确」），
  但**不属于 h3c 接线的必要条件**。

### 2. 权重已 un-rotate ⇒ ANE 图用「无旋转」那档（gs=0），不需要图内 Hadamard

`h3_dit.c:819` 注释：ConvRot 权重的产出是 "BF16 tensor holding the true (unrotated) weight,
ready for the existing quantize/forward"，由 `h3_gpu_weight_dequant_unrotate_int8`
+ `dit->convrot_hadamard`（`build_convrot_hadamard`，`h3_dit.c:780`）完成。
⇒ 激活侧不需要再做 grouped Hadamard，fork 图里那一段 conv 在 h3c 接线时**应当整个省掉**，
只留纯 int8 GEMM（对应 fork 已 PASS 的 `plain-2048` 档）。
顺带：qkv 的 q/k/v 交错→分离重排（`h3_gpu_convrot_remap_qkv_bf16`、`convrot_unrotate_cpu` 的
`layout == 1`）h3c 也已经在权重侧做完，ANE 侧不必重复。

### 3. 真正的代价：融合 kernel 会被拆成 2~3 步

- `h3_gpu_grouped_qkv_linear_rope_int8`（`h3_dit.c:4007`）= GEMM+per-head 分组+RoPE 一步；
  `h3_gpu_mlp_int8_bf16`（`h3_dit.c:4109`）= fc1+SwiGLU 一步。
- ANE 只能吃 GEMM，接线后 RoPE / SwiGLU 要退回独立 kernel ⇒ 多两趟 HBM 往返 + 两次
  pack/unpack 到 ANE 平面。**净收益必须是「ANE GEMM 省下的时间 > 拆融合的代价」**，
  这个数现在没有，只能实测。
- 建议第一步只做 `attn.out` 和 `mlp.fc2`（两者下游本来就无融合，SwiGLU 在 fc1、RoPE 在 qkv，
  这两条是「GEMM 完就写回」的纯矩阵乘），拿到 ANE/Metal 单算子计时后再决定要不要动 qkv/fc1。

### 4. 接线范围约束：只有 int8 驻留路径能走 ANE

私有 `AppleNeuralEngine` 的 `constexpr_affine_dequantize` 输入是 int8 载荷；
h3c 的 BF16 直载路径（`LOAD_BF16_CONVROT`，`h3_dit.c:908`）拿到的是 BF16 权重，
若要走 ANE 必须先量化 —— 那等于给 BF16 档位凭空加一层量化误差。
⇒ `H3_ANE_LINEARS` 应与 int8/int6 驻留权重的开关**互锁**（非量化模式下自动回退 Metal），
而不是独立生效。

---

## F14：整 block 单图（B3'）移植结果 + 一整套决定性数字（2026-09-19 11:35）

fork 的 `h3_ane_block.{h,m}`（857 行，把 adaln→int8 ConvRot 投影→per-head norm→RoPE→
full softmax→gate 残差→SwiGLU MLP 编成**一张** ANE 图，激活每 block 只跨界一次）
+ `tests/test_ane_full_block.c`（真实 checkpoint + f64 回放参考）移植进 h3c。

### 移植成本：几乎为零（依赖早在 B2a/B2b 就位）

- 拷贝后**只改 4 行**：h3c 的 `-Wenum-float-conversion` 比 fork 严，
  `1.0 / BLK_HIDDEN`、`1.0 / BLK_HEAD_DIM`（`h3_ane_block.m:413/490`）和测试里
  `qs / HEAD_DIM`、`ks / HEAD_DIM` 要显式 `(double)`；顺带修了 usage banner 把 binary 名
  写成 `h3_ane_block_test` 的笔误，并补了一句「rotation 那条腿需要 `H3_ANE_CACHE=1`」。
- 依赖核对（fork `h3_ane_block.m` 调用的全部 h3_ 符号）：`h3_weight_load_int8_raw`、
  `h3_convrot_hadamard`（B2a）、`h3_ane_model_*`/`h3_ane_bridge_surface`/`h3_ane_bridge_available`、
  `h3_weight_find`、`h3_st_read_data`、`h3_st_tensor_elements`（h3c 全有）⇒ **一次编译通过**。
- Makefile：`LIB_M` 加 `h3_ane_block.m`，新 target `h3_ane_full_block_test`。

### h3c 内实测（真实 `MiniMax-H3-Convrot/FL2VA/transformer` 的 blocks.0，`H3_ANE_CACHE=1`）

```
compiled in 4.19s, blob 380.8 MiB
  segment 0 rows [0,6):  cos=0.99999 max_abs=5.744e+01
  segment 1 rows [6,20): cos=0.99999 max_abs=1.740e+01
  segment 2 rows [20,64): cos=0.99999 max_abs=2.566e+01
block S=64: cos=0.999988 rel_l2=4.898e-03 max_abs=5.744e+01 nonfinite=0 PASS
timing S=1904: best 334.3 ms/block (compile 3.21s)
rotation S=1904: unload 5 ms + reload 29 ms, 3 cycles bit-identical PASS
```

- 精度关（cos≥0.999、rel_l2≤3e-2、nonfinite=0）**在 h3c 的构建/依赖下成立**，
  整 block 走 fp16 激活 + int8 权重后 rel_l2 只有 4.9e-3。
- **常驻轮换是真的可用**：`unload 5ms + reload 29ms` = eval(334ms) 的 8.7%，
  而且 reload 之后输出**逐字节相同** ⇒ 「800MB wired/block，50 个装不下」可以靠轮换绕开
  （fork README 的判断成立，但代价可接受）。

### 决定性的一条：编译产物 381 MiB/block-形状（磁盘关）

本次跑完 `T/h3-ane-cache` = **762 MiB / 2 条目**（S=64 与 S=1904 各一），即
**381 MiB ≈ 1.0× int8 blob**。乘回去：

| 量 | 数值 |
|---|---|
| 一个形状 × 50 block 的跨进程缓存 | 50 × 381 MiB ≈ **18.6 GiB** |
| 再算上 aned 自己那份副本（fork README 实测翻倍） | ≈ **37 GiB** |
| 本机两个卷的 avail | **12 GiB**（Data 与系统卷同一容器） |

⇒ fork 估的「~19GB/形状」在 h3c 依赖下**逐 MiB 得到验证**；
**当前磁盘条件不支持「50 block 全量 + 跨形状缓存」**，这不是可调参绕开的，是硬墙。
关缓存（`H3_ANE_CACHE=0`）退出零残留，但每次换形状要重付 50 × 3.2~4.2 s ≈ **170~210 s** 冷编译。

### 与 h3c 现有瓶颈对账：低分辨率下 ANE 打不过「少读字节」

- h3c 的流式去噪在 256×256/0.5s/steps=2 已经是 **SSD 地板**：每 block 每 worker
  read ≈ 173.5 ms（2.22 GB/s，等于该卷实测上限），dequant ≈ 87 ms，
  整 block ≈ 200 ms（F 前文「终点：已贴磁盘地板」）。
- ANE 那条路**一样要从盘上读 414 MiB/block**，还多付每形状每 block 3~4 s 编译
  ⇒ 在「权重装不进内存、必须每步重流」的低分辨率档，ANE 没有位置。
- ANE 有位置的唯一情形：**行数大到计算开始盖过读盘**（S=1904 时单 block 图 eval 334 ms）
  **且同一形状被重复求值**（多 step / 多 chunk 复用一次编译）。
- ⚠️ 但 334 ms 只是 `h3_ane_block_eval`，**不含** pack/unpack 与 h3c 那两趟 GEMM 的 Metal 对照，
  所以现在还不能下「ANE 更快」的结论。

### 因此 B4 的前置量（先测再决定接线）

需要 **h3c 自身在 rows≈1904、权重常驻（不流式）时单 block 的 Metal 时间**，
和 334 ms 对齐比较；fork 的 `tests/test_ane_block.c` 正是这个对照
（4 投影 ANE vs 纯 Metal vs 融合 Metal，合成权重，argv=ROWS，gate cos≥0.999）。
⇒ 建议把它作为 **B4-pre** 移植进来出曲线，而不是先去改 `h3_dit.c`。

---

## F15：B4-pre 量完了 —— 「4 投影分别上 ANE」被判死刑，且顺带挖出一个计时口径 bug（2026-09-19 12:00）

fork 的 `tests/test_ane_block.c` 移植进 h3c **零改动**（443 行，编译+链接一次过；
`h3_gpu_{grouped_qkv_linear_rope,grouped_qkv_rope,mlp,swiglu,sdpa,rms_norm,copy,add,linear}_bf16`
这批 Metal API 在 h3c 全都还在，签名没漂）。它跑的是：
**同一个 block，四条投影分别走 ANE，其余算子留在 Metal**，对照「纯 Metal（融合）」和「Metal（拆融合）」。
合成权重、真实 DiT 形状（qkv 21504×5376 / out 5376×21504 / fc1 28672×5376 / fc2 5376×14336）。

### 干净环境下的 rows 扫描（机器上无其它负载，`H3_ANE_CACHE=0`，best/mean 毫秒）

| rows | Metal 融合 best | 拆投影 ANE best | ratio | ANE mean | 分阶段（staged 模式）pack / ANE / rest |
|---|---|---|---|---|---|
| 384 | 101.5 | 68.3 | **1.49×** | 90.0 | 20.8 / 30.1 / 56.9 |
| 1536 | 436.7 | 418.8 | **1.04×** | **549.2（比 Metal 慢）** | 71.7 / 114.8 / 196.3 |
| 3072 | 903.7 | 842.7 | **1.07×** | **943.0（比 Metal 慢）** | 157.9 / 248.7 / 475.7 |

数值关三档全 PASS（cos≈0.99999、rel_l2≈4.8e-3、nonfinite=0）。

### 计时口径 bug（继承自 fork，会骗人）

`h3_ane_projection_apply()` 里 `pack_seconds += packed - started`，而 `started` 取在
**pack 之前**，pack 又是往**已经排着前面所有 Metal 工作的命令缓冲**里追加
⇒ plain 模式的「pack+submit」**把整条 Metal 队列的排空时间记成了 pack**：
1536 行下 plain 报 384.9 ms、staged 报 71.7 ms（其中真 pack 只有 1.0~2.1 ms/投影）。
⇒ 任何「pack 太慢」的结论都必须用 `H3_ANE_PROFILE_STAGES=1` 复核；
plain 的那个数字只能当「ANE 之前的 Metal 尾巴」。

### 第二个陷阱：后台有并发时数字整体虚高 ~2×

第一次 1536 行的 run 与后台 `h3_ane_int8_test` 撞车，同一 binary 同一参数报出
`metal_default=855 ms`（干净时 436.7 ms）。⇒ ANE/Metal 计时 gate 必须独占机器。

### 判读

1. **拆投影走 ANE 不值**：只有小 rows（384）拿到 1.49×，rows 越大越贴平，mean 还反超为慢。
   原因是硬账：**每投影一次 pack+unpack 跨界**，而 ANE 省下的只是那趟 GEMM；
   加上每形状每投影 0.16~0.87 s 编译 × 4 投影 × 50 block ≈ **30~170 s 冷编译**。
2. **唯一有肉的是「整 block 一张图」**（F14：S=1904 时 eval 334 ms，而本表插值出
   Metal 在 1904 行 ≈ 530 ms ⇒ 约 1.6×，且激活每 block 只跨界一次），
   这与 fork README 端到端 26.3 s vs 31.9 s 的量级一致。
3. 但整 block 图的跨进程缓存是 **381 MiB/block-形状 ⇒ 18.6 GiB/形状**（F14），本机 12 GiB 装不下；
   关缓存又要为每个形状重付 50 × 3~4 s ≈ 170~210 s。
4. 所以对 h3c 的判断是：**DiT 侧 ANE（无论拆还是整块）在这台机器上都不划算**；
   ANE 真正合适的目标是**形状固定、payload 小、每形状只求值一次的模块**
   ——即 Phase D 的三个 VAE（video/vision/audio tile 尺寸固定，编译产物按 tile 计）。
   这条建议待用户拍板后再改 task_plan 的 C/D 阶段划分。
---

## F16：D0 放行 —— video VAE 投影在 ANE 上实测 2.9~3.7×，rel_l2 ≤ 7.9e-4，编译/缓存账比 DiT 轻两个量级（2026-09-19 12:45）

新建 `tests/test_ane_vae.c`（Makefile 目标 `h3_ane_vae_test`）：真实
`decoder.transformer_blocks.0` 的四条投影，对照 h3c 自己的 `h3_gpu_linear_f32`，
另加一路 `cblas_sgemm` 宿主参考。**参考实现零误差**（Metal fp32 vs BLAS
rel_l2=0.000e+00 max_abs=0.000e+00，四条形），所以下面的误差全是 ANE 侧引入的。

### 先纠一条前提（Phase D 计划里写错了）

盘上不是 F16：`FL2VA/video_vae/source/model.safetensors` 是**单片 10,415,548,320
字节、560 张量全 F32**（脚本读头部清点）。⇒ **权重 fp32→fp16 的舍入本身就在误差
预算里**，不能当"零损失"。好在实测它和激活舍入加起来仍然过闸（见下）。
顺带确认形状与假设一致：qkv 6144×2048 / out 2048×2048 / w1 16384×2048 /
w2 2048×8192，decoder 恰好 36 个 `transformer_blocks`，bias 全在（F32，本 gate 两边都不加）。

### 速度 / 误差（机器独占，`H3_ANE_CACHE=0`，best 毫秒）

| rows | qkv | out | w1 | w2 | block 合计 | ratio | rel_l2 范围 |
|---|---|---|---|---|---|---|---|
| 1797 | 16.02→5.03 | 5.36→1.82 | 42.88→13.37 | 26.93→7.38 | **90.7→28.0** | **3.24×** | 5.1e-4 ~ 7.9e-4 |
| 512 | 4.49→1.54 | 1.65→0.60 | 11.25→3.87 | 6.26→2.18 | **23.7→8.2** | **2.89×** | 5.1e-4 ~ 7.9e-4 |

- cos=1.000000，`nonfinite=0`，**fp16 不可表示的权重 = 0**（最大权重 0.36，参考峰值
  13.4，离 fp16 上限 65504 极远）⇒ 三条放行判据（≥2×、≤1e-3、溢出 0）**全过**。
- **小 rows 不掉速**：512 行仍然是 2.89×，说明不需要"rows 太小退回 Metal"的门槛。
  这与 DiT 的拆投影曲线（F15：1536 行只有 1.04×）完全不同，因为这里权重是 fp16、
  N/K 比小，ANE 的 MAC 阵列吃得满。
- **rel_l2 与 rows 无关**（1797 和 512 一模一样到 1e-7）⇒ 误差纯来自 fp16 的
  权重/激活舍入，不是累加长度。w2 的 7.9e-4 最大（K=8192 且 8 个分片求和），
  是四条里最该盯的那条。
- 外推到整轮解码：**一条 pass 的四投影 3.26 s → 1.01 s**，24 pass ≈ **78 s → 24 s**
  （只算 GEMM，未扣 pack/unpack；未算 ANE 免掉的每 pass 读盘）。

### 编译与磁盘：比 DiT 轻两个量级，且**与 rows 无关**

- 每张图编译 **0.04~0.18 s**，一个 block 四种投影合计 0.5 s ⇒ 36 block **16 s**
  冷编译（DiT 整 block 单图是 3.2 s/**block**，50 block ≈ 170 s，见 F14/F15）。
- 缓存条目实测只含三样：`data` + `net.plist` + `compiled.ok`，其中 `data` 是
  **编译器的产物常量段**（`bridge_cache_store` 明确跳过 `weights`/`model.mil` 源文件），
  大小恰好 = N·K·2 字节：24 / 8 / 64 / 32 MiB ⇒ **每 block 128 MiB，36 block = 4.5 GiB**。
- **128 MiB 在 rows=1797 和 rows=512 两次 run 里完全相同** ⇒ 产物字节只由权重决定，
  与形状无关（形状只改变 `net.plist`，那是 KB 级）。⇒ 多一个 rows 形状就是**再复制
  一整份 4.5 GiB**。这条取代 F14 对 fp16 图的估算：DiT 整 block 的 381 MiB/block 是
  **int8 图特有**（`constexpr_affine_dequantize` 把反量化结果也落进产物），fp16 图是 1× 权重字节。
- 缓存恢复是**逐位一致**的：`H3_ANE_CACHE=1` 重跑同参数，四条 rel_l2 与 CACHE=0 完全相同。

### rows 形状数：一轮解码只有 1 个（这是 D2 能成立的关键）

`vae->sequence = CHUNK_LATENT_TIME · latent_h · latent_w + SUFFIX`，而 tile 是**等宽平铺**：
`tile_axis_build` 让相邻 tile 共享 `TILE_OVERLAP_MIN` 重叠而不是把边界 tile 切短，
`configured_tile_pixels` 只在 `TILE_PIXELS..320`（步长 SPATIAL_RATIO）里选一个全局尺寸。
⇒ 同一次解码里所有 tile 的 rows 相同；只有当视频比一个 tile 还小时才会退化成
`axis->length = extent` 的短形状。
⇒ **D2 应该把 rows 向上取整到固定桶再编译**：多付 ≤1 个桶的算力（1797→2048 是 +14%），
换来「跨分辨率复用同一份 4.5 GiB 产物」，否则每个新分辨率都要重付一遍 4.5 GiB + 16 s。

### 还没做的（别当成已验证）

- 本 gate 两边都**不带 bias**，真实解码带；激活是 `[-2,2)` 均匀伪随机，不是真实
  VAE 激活分布（真实分布若有长尾，fp16 相对误差不变但**溢出**风险要重估）。
- pack/unpack 未在测：CPU 侧填平面在 w2 上要 41 ms（比省下的 19.5 ms 还贵），
  这是**harness 的 strided gather**，不是引擎路径的账；D1 必须用 Metal pack kernel
  重测这条，否则结论会被这一项翻掉。
---

## F17：D1 量完 —— 把 staging 和 bias 都算进去，block 端到端仍然 2.26~2.43×；rows=512 只有 1.88×（2026-09-19 13:20）

D0 的账是"裸图 vs 裸 kernel"，不能拿来做决策。D1 补上引擎真正要走的那条路：
**Metal pack → ANE 图 → Metal unpack（顺手加 bias）**，对照今天带 bias 的
`h3_gpu_linear_f32`。

### 新增的东西（都在 gate 之外，是给 D2 用的地基）

- `h3_shaders.metal`：`h3_ane_pack_f32`（row-major F32 → `[kc][plane_rows]` 纯转置）、
  `h3_ane_unpack_f32`（转回 row-major F32，`with_bias` 时加 `[output_dim]` 通道 bias）。
  **bias 不进图**：图只表达 reduction，加法放在 unpack，一次读一次写。
- `h3_gpu.{h,m}`：`h3_gpu_pack_ane_input_f32` / `h3_gpu_unpack_ane_output_f32(bias 可空)`，
  两个 pipeline 名字加进**显式登记名单**（在 `if (tensorOpsEnabled)` 之外，和 bf16 那两个同一处）。
- `h3_ane_linear.{h,m}`：`h3_ane_projection_create_f16`（host 端 row-major fp16 权重）
  和 `h3_ane_projection_apply_f32`；bf16 的 `h3_ane_projection_apply` 现在只是
  共用 static `ane_projection_apply(..., f32_staging=0)` 的薄壳。
  调用约定不变：**前置要有开着的命令缓冲，返回时 unpack 还在缓冲里没提交**，由调用方 submit。

### 端到端实测（机器独占，`H3_ANE_CACHE=0`，best 毫秒；比例对照 fp32+bias）

| rows | qkv | out | w1 | w2 | block 合计 | block ratio | staging 合计 |
|---|---|---|---|---|---|---|---|
| 1797 | 14.98→6.89 (2.17×) | 5.16→2.99 (**1.73×**) | 39.60→16.84 (2.35×) | 22.65→9.72 (2.33×) | **82.4→36.4** | **2.26×** | 8.9 ms |
| 1797 复跑（缓存热） | 2.27× | 1.68× | 2.49× | 2.69× | 90.2→37.1 | **2.43×** | 9.1 ms |
| 512 | 4.26→2.43 (1.76×) | 1.61→1.24 (**1.30×**) | 10.98→5.41 (2.03×) | 6.14→3.18 (1.93×) | 23.0→12.3 | **1.88×** | 3.97 ms |

- **一张图一次的 staging ≈ 1.0~3.7 ms**，四条合计 ≈ 9 ms，相当于在裸图 28 ms 上
  再加 32% —— 但相对 fp32 仍然是 2.26×。⇒ D0 那个 3.24× 是上限，**决策请用 2.26~2.43×**。
- 换算：**36 block 一条 pass 2.97 s → 1.31 s**（1797 行）。
- **误差没被 staging 污染**：端到端 rel_l2 与裸图逐位同级（5.322e-04 vs 5.323e-04），
  说明 fp32 加 bias 不引入新误差。
- **最小的那条 `out`（2048×2048）单独看只有 1.68~1.73×**：它裸图 2.0 ms，而两次跨界就要
  1.0 ms。⇒ 单投影 ≥2× 是错误的闸；闸要放在 **block 合计**，单投影只判"别退化"
  （测试里改成 ≥1.2× 才报 REGRESSION）。绝对值上 `out` 走 ANE 仍比 fp32 快（2.99 vs 5.16 ms），
  所以 D2 **不需要**"小投影退回 Metal"的策略。
- **rows=512 的 block 1.88× 不过 2× 闸**：跨界开销基本固定，GEMM 越小越吃比例。
  ⇒ 小 tile 要么不路由，要么靠"整 block 一张图"摊掉三次额外跨界（D2 的可选项）。

### 一个必须记住的桥接坑

`h3_ane_bridge.m:181` 的目录名来自 **模型内容 identifier**（`hexStringIdentifier`），
不含我们传的 `name` ⇒ **同一份 MIL + 权重的两张图会共用同一个 staging 目录**。
gate 里因此先 `h3_ane_linear_free` 再建 projection（改图名是无效的，试过了）。

### 回归面（改了共享代码，全部重跑）

`h3_tests` 1829 checks ok；`h3_ane_staging_test`、`h3_ane_int8_test`、`h3_convrot_test`
0 failure；bf16 那条 `h3_ane_block_test 384` 仍 PASS（cos 0.999989，ane 49.0 ms vs
metal 86.4 ms = 1.76×）⇒ `ane_projection_apply` 重构没动坏。
`h3_metal_tests` 跑不了：`misc/fixtures/h3_dit.safetensors` 不在（既有状态，非本次改动）。

## F18：D2a 常驻探针量完 —— 硬上限是 126 个存活句柄（不是内存），平面才是内存大头；缓存恢复会偶发失败并自毁条目（2026-09-19 14:20）

探针：`tests/test_ane_vae_residency.c`（`make h3_ane_vae_residency_test`），
`usage: h3_ane_vae_residency_test VIDEO_VAE_DIR [capacity|reload|pass|window] [ROWS] [BLOCKS] [WINDOW]`。
为此给共享代码加了：`h3_ane_linear_unload/reload`、`h3_ane_projection_unload/reload`、
`h3_ane_projection_plane_bytes`、`h3_ane_projection_cache_hit`（都是薄壳，转调 bridge 已有的
`h3_ane_model_unload/reload`），以及 `h3_ane_bridge.m` 里 `H3_ANE_CACHE_DEBUG=1`
时打印「cache restored / miss / cached load failed(+原因)」。`h3_video_vae.c` 一行没动。

### 1. 句柄上限 ≈126，park 不能腾名额（决定接线形状）

`capacity 64 36` 与 `window 64 36 8`（后者每张建好立刻 unload，任意时刻只有 1 张连着）
**都在第 127 张（b31_w1）挂**：
`createProgramInstanceForModel:...: Program load failure (0x50004)`，
此时 footprint 只有 412 MiB、可用内存 6.2 GiB、机器 16 GiB —— **不是内存也不是磁盘，是按存活
model 句柄计数**。⇒ 36 block × 4 投影 = 144 张图**连"建出来放着"都做不到**，
"全 decoder 常驻"这条路从源头就没有。

### 2. 轮换单价（DiT 的 5/29 ms 确实不能套用）

12 张 rows=2048 图、`reload` 模式 5 轮：**unload 0.47~1.14 ms、reload 3.40~3.77 ms/张**。
32 张（8 block）全驻留时的 `pass` 模式里 reload=5.15 ms、unload=1.40 ms（张数多了略涨）。

### 3. 端到端 pass 曲线（rows=2048、8 block/32 张、独占机器）

| 调度 | pass | 相对全常驻 |
|---|---|---|
| 全常驻 | **333.5 ms**（10.4 ms/张 ⇒ block 41.7 ms） | 1.00 |
| window=8（每 pass 淘汰 32 次） | 557.5 ms（其中重连 128.8 ms） | **1.67** |
| 每张每 pass 都 unload+reload | 616.4 ms | **1.85** |

⇒ **ANE 的收益只在"基本不轮换"时成立**：D1 的 2.26× 一旦被每 pass 全量重连吃掉，
只剩 ≈1.2×。所以接线必须以「常驻不轮换」为默认，轮换只是兜底。

### 4. 平面（planes）才是内存大头，必须跨 block 共享

一张图的常驻 = 烘进去的 fp16 权重 **+ 自己的输入/输出平面**。rows=2048 时每 block
平面 **320 MiB**（w1 的 `[16384][2048]` 输出平面独占 128 MiB），36 block = **11.5 GiB**
⇒ "每图自持平面"在 16 GiB 机器上直接判死。而 36 block 形状完全相同，
**池化后只要 320 MiB**。实测 footprint 边际 ≈200 MiB/block（平面与权重的记账在
IOSurface/驱动侧不完全落进 phys_footprint，别把 412 MiB/126 张当"权重不占内存"的结论用）。
⇒ D2 的前置工程量是「让 `h3_ane_linear` 接外部平面」，不是「怎么轮换」。

### 5. 缓存这一条要在接线前修，否则每次冷启动都在赌

- `bridge_cache_store` 是逐文件硬链接（省空间），但 `bridge_cache_restore` 走的是
  **目录级 `copyItemAtPath`**（目录硬链接不被允许，`bridge_mirror` 的 fallback）。
- 本机缓存堆到 150 条目 / 系统卷只剩 4.4 GiB 时，实测出现
  **「cache restored」但 `loadWithQoS` 失败** → 回退重编译；而重编译后的
  `bridge_cache_store` 会先 `removeItemAtPath(entry)` ⇒ **一次瞬时失败就把这张图的条目抹掉**，
  下次只能再冷编译。这就是 F16 之前那次"同一个图一会儿命中一会儿不命中"的真相。
- 本轮结束前已删除自己的可再生缓存 `T/h3-ane-cache`（**5.7 GiB**），系统卷回到 12 GiB 空闲；
  删完复测冷编译路径：`./h3_ane_block_test 384` PASS（cos 0.999989，compile 0.5~2.3 s/图）。
  用户目录一个字节没动。
- ⇒ D2 的 LRU 上限必须先有「restore 用硬链接」+「cached load 失败不许删条目」，
  否则任何上限都只是把重编译排队。

### 6. 给 D2 的结论（架构定形）

144 > 126 ⇒ 只有两条路：
1. **block 数封顶**：≤31 block 走 ANE（124 张）且**全常驻不轮换**，其余 block 留 fp32 Metal。
   按 §3+D1 外推一条 pass：31×41.7 ms + 5×~95 ms ≈ **1.8 s vs 全 Metal ~3.4 s（1.9×）**。
2. **减少每 block 图数**：把 w1+w2 融成一张 FFN 图 ⇒ 3 张/block = 108 张，全进 ANE。
   代价：一张图要 4(w1 chunk)+4(w2 chunk)=8 个输入，正好顶满 F1 的 8 输入绑定上限，
   没有余量；且要新写 MIL + 重做精度关。

推荐先做 ①（平面池化 + 图数上限 + 不轮换），②留作 D2 之后的可选优化。

**未验证**：126 这个数是否随驱动状态/是否创建 `ANERequest` 而变（park 时若连 request
和 IOSurface 引用一起释放，名额可能回来 —— 下一步最便宜的实验就是这个）。

## F19：D2b 完成 —— 缓存改成硬链接 restore + 原子 store + 5 GiB LRU，rows 分桶实测跨分辨率共用同一批 identifier（2026-09-19 15:10）

用户 2026-09-19 拍板：走**路线①（block 封顶 31）**，并且**先修缓存再把上限定在 5 GiB**。
改动全部在 `h3_ane_bridge.m`（+ `h3_ane_linear.{h,m}` 一个纯函数、探针几行），
`h3_video_vae.c` 仍未动。

### 1. restore 不再是目录 copy（磁盘峰值减半，且这是 5 GiB 上限能成立的前提）

`bridge_mirror()` 原来是「目录 `linkItemAtPath`（必然失败）→ 目录 `copyItemAtPath`」，
所以**每次命中都要把整份产物再抄一遍**：12 张 rows=2048 图 = 384 MiB 变 768 MiB。
现在改成逐文件硬链接 + 目录递归重建，单文件复制只作 fallback。
实测 `capacity 2048 3` → `reload` 模式 12 张图 3 轮轮换，缓存全程稳定在 **1.2 GiB**
（旧实现会涨到 ~1.6 GiB），且 reload 单价不变（unload 0.40~0.45 ms、reload 3.18~4.65 ms
/张，与 F18 一致 ⇒ 硬链接没有拖慢 load）。

### 2. store 变成「写 `<id>.tmp` → `move` 换入」，好条目不会再被瞬时失败抹掉

原 `bridge_cache_store` 第一件事就是 `removeItemAtPath(entry)`：中途任何失败
（磁盘紧、链接失败、进程被杀）都会留下**空/半截条目**，下次只能冷编译 —— 这就是 F18 §5
"一会儿命中一会儿不命中"的机制。现在旧条目只在新条目**完整写好之后**才被替换；
`compiled.ok` 写成失败也会清掉 `.tmp`。另外每次 trim 顺手清扫 `<id>.tmp` 残留。

### 3. LRU 上限：`H3_ANE_CACHE_MAX_MIB`，默认 **5120**，按 `compiled.ok` mtime 淘汰

restore 会 touch `compiled.ok`，所以排序是「最近使用」而不是「最近编译」。
`keep` 参数保证刚写的那条不会被自己挤掉。实测把上限设成 1000 MiB 跑一次新形状：
`evicted 3 entries（252/73/336 MiB）→ 811 MiB`，`du` 复核 917 MiB，符合预期。
**注意**：淘汰粒度是**单张图**。5 GiB 上限装得下一个 rows 形状（4.5 GiB）但装不下两个，
所以换分辨率时会把旧形状逐张挤掉 —— 这是有意的（宁可重编译 16 s 也不让系统卷掉到 5 GiB 以下）。

### 4. rows 分桶实测：1797 与 2048 拿到**逐字相同**的图 identifier

`h3_ane_rows_bucket(rows)`（`H3_ANE_ROW_BUCKET`，默认 256，0 关闭）加在
`h3_ane_linear.{h,m}`，探针启动就套用并打印 `rows=1797 (bucket 2048)`。
实测四条投影的 identifier（`D6B80868…`/`442F2D29…`/`622B058F…`/`0DECFC1F…`）
与之前 rows=2048 那轮**完全一致**并全部 `cache restored`，create 10~36 ms、
compile 0.00~0.10 s ⇒ 「所有分辨率共用一份 4.5 GiB 产物」这条前提坐实。

### 5. 闸复核（rows=2048，缓存热）

block 端到端 **96.2 → 41.2 ms = 2.33×**，36 block 一条 pass **3.46 → 1.48 s**；
单投影 1.89×（out）~2.44×（w1）；rel_l2 5.1e-4~7.9e-4，fp16 不可表示 0，nonfinite 0。
与 F17（rows=1797 的 2.26~2.43×）同级。

### 6. 回归与磁盘

`make -j8 all` 无新告警；`h3_ane_int8_test` 的缓存用例（hit/rehit/miss + `bitexact=1`）
在硬链接/原子 store 之下仍通过；`h3_ane_staging_test` 0 failure；`h3_ane_block_test 384`
PASS（cos 0.999989）。系统卷 12 GiB 空闲、缓存 926 MiB。本轮只在 `T/` 里增删自己的产物。

## F20：平面池化实测 —— 16 张图 footprint 2340→1379 MiB，池子按形状收一份（2026-09-19 15:35）

`h3_ane_linear.m` 里加了进程内的平面池（`ane_plane_set` 按
`{chunks, input_bytes, output_bytes}` 索引），`h3_ane_linear.h` 暴露三个口：

```c
void     h3_ane_planes_share(int enable);  /* 之后建的图共用同形状平面 */
uint64_t h3_ane_planes_bytes(void);        /* 池子实际持有的字节 */
void     h3_ane_planes_clear(void);        /* 图都释放之后再清 */
```

形状不同的图自然拿到不同集合（VAE 四投影 = 4 个集合）；**共享是 opt-in**，
默认仍走每图自持，所以既有的单图测试语义没变。引用计数：每次 acquire 都给调用方
自己的一份 `CFRetain`，池子另留一份 ⇒ 释放某张图不会把平面从别的图底下抽走。

实测（`H3_ANE_RESIDENCY_SHARE=1`，rows=2048、4 block/16 张，缓存热）：

| | planes 名义 | 池子实际 | footprint |
|---|---|---|---|
| 每图自持 | 1280 MiB | 0（无池） | **2340 MiB** |
| 池化 | 1280 MiB | **320 MiB** | **1379 MiB** |

释放 16 张图之后两条都回到 1059 MiB，池化那条池子仍留 320 MiB（等 `clear`）
⇒ 计数正确，没有提前释放也没有泄漏。

**为什么这是 D2 的前置而不是优化**：D2a 的账是 36 block × 320 MiB = 11.5 GiB
（每图自持平面），本机 16 GiB 直接判死；池化后是固定 320 MiB，与 block 数无关。
剩下的唯一约束就回到 F18 的 126 句柄上限 ⇒ 路线①（≤31 block）成立。

**并发口径**：`h3_gpu_submit()` 是 commit + `waitUntilCompleted`，
且 `ane_projection_apply` 在 ANE eval 之前一定 submit 一次 ⇒ 上一张图的 unpack
已经跑完，下一张同形状图才会写同一块平面 ⇒ 串行链路下共享安全。
（若将来做 ANE/GPU 流水，这个前提要重新审。）

## F21：路线①实测成立 —— 124 张图 + 池化平面常驻 footprint 1584 MiB、一条 pass 1.78 s；但缓存装不下 31 block（新增磁盘下限）（2026-09-19 16:05）

### 1. 31 block × 4 = 124 张图可以**同时常驻**，且池化后内存有余量

`H3_ANE_RESIDENCY_SHARE=1 ./h3_ane_vae_residency_test $VAE pass 2048 31`：
124 张图全部建好（离 F18 的 126 上限还剩 2 个名额），
**footprint 1584 MiB**（权重名义 3968 MiB、平面名义 9920 MiB、池子实际 320 MiB
⇒ 烘进去的 fp16 常量不落在本任务的 phys_footprint 里，是驱动侧的账）。
释放 124 张之后 footprint 回到 1406 MiB、池子仍记 320 MiB（等 `clear`）⇒ 计数干净。

### 2. 全常驻 pass 1.78 s，但每张图的成本随常驻图数上升

| 常驻图数 | 全常驻 pass | 每张 | 每 block |
|---|---|---|---|
| 32（8 block，池化） | **345.1 ms** | 10.78 ms | 43.1 ms |
| 124（31 block，池化） | **1780.6 ms** | 14.36 ms | **57.4 ms** |

对照 D1 单 block 端到端 41.2 ms ⇒ **每张图的 ANE 成本随常驻图数线性上涨**
（拟合 ≈38.2 + 0.62×block 数 ms/block）。轮换（每张每 pass unload+reload）在 124 张时
2374 ms（reload 5.71 ms/张、unload 0.76 ms/张）⇒ 仍是每 pass 全量重连更贵，但差距
从 F18 的 1.85× 缩到 1.33×。

**为什么还是选满 31 block**：边际一个 block 在 N=31 处成本 57.4 ms，而它替掉的
fp32 Metal block 是 96.2 ms ⇒ 每多一个 block 净赚 ≈39 ms，单调更好，上限只由 126 决定。
一条 36 block 解码 pass：31×57.4 + 5×96.2 ≈ **2.26 s vs 全 Metal 3.46 s = 1.53×**
（比 F18 §6 外推的 1.9× 低，原因就是上面那条成本上涨）。

### 3. 池化不但省内存，还更快更稳（SHARE=0/1 对照，32 张图）

| | 首轮 pass | 最好 pass | footprint |
|---|---|---|---|
| 每图自持平面 | 719.1 ms | 377.2 ms | 4582 MiB |
| 池化 | **398.8 ms** | **345.1 ms** | **2341 MiB** |

（首轮含预热，池化少写 2.2 GiB 新页。）

### 4. 缓存装不下 31 block：新增 `H3_ANE_CACHE_MIN_FREE_MIB`（默认 6144）

31 block 的一套产物 ≈ **3.9 GiB**，而本机在缓存 4.4 GiB 时系统卷只剩 **5.1 GiB**
—— 正好是 F18 观测到 ANE load 开始失败的位置。所以「5 GiB 上限」单独不够用：
它管的是缓存自己多大，不管卷上还有谁在写。改成
**`target = min(上限, 当前空闲 - 下限 + 缓存字节)`**，并且冷编译前先 `trim` 腾地方。
实测生效：跑一张新图时连淘 14 条（147/64/32/24/8 MiB…），
**空闲 5.1 → 6.1 GiB**、缓存 4.4 → 3.3 GiB，日志带上 target 与实时空闲。

⇒ 这台机器的现实约束是：**「31 block 全热缓存」和「留 6 GiB 安全垫」二选一**。
代价可量化：124 张图冷启动合计 **create 71.4 s**（均值 576 ms/张，缓存命中时 11~46 ms/张），
所以缓存只在"进程会重启"的场景值钱；常驻服务一次编译就够。
D2d 接线时不把性能押在缓存命中上（正确性也不依赖它）。

### 5. 回归

`make -j8` 全量无新告警；`h3_ane_int8_test`（含缓存 hit/rehit/miss + bitexact）、
`h3_ane_staging_test` 0 failure 均通过；`h3_ane_vae_test 2048` 端到端闸 2.33× 复现。
本轮删改仅限自己的 `T/h3-ane-cache` 与 `T/` 下 ANE staging 目录。

---

## F22（2026-09-19）：D2c —— ANE 图权重的宿主侧读取口转正为 `h3_weight_load_f16_raw`

video VAE 的 36×4 个投影在盘上是 **F32**（`FL2VA/video_vae/source/model.safetensors`，
10.4 GB），ANE 图要的是 **行主序 fp16 constexpr**；探针里那份临时的
`read_weight_f16()` 就是缺的那个口，现在落到 `h3_weights.{h,c}`：

```c
int h3_weight_load_f16_raw(const h3_weight_store *store, const char *name,
                           uint64_t output_dim, uint64_t input_dim,
                           uint16_t **weights, char *error, size_t error_size);
```

- 接受 **F32 与 F16** 两种存储：F32 逐元素 `(__fp16)` 舍入，F16 直接 `pread` 到目标缓冲
  （已是图要的表示，逐字拷贝）。
- 为什么保留 F16 分支不是多余设计：`h3_weight_load_f32` 早就支持 F16 存储的 VAE
  （`load_f16_as_float`），ANE 侧若不接就出现"Metal 能读、ANE 不能读"的存储。
- 形状/dtype 不符一律拒绝并给出 `name is not F32/F16 [rows][cols]`。

**覆盖**（本机没有 F16 存储的 video VAE，所以分两处测两条分支）：
1. `h3_ane_vae_test <FL2VA video_vae> 2048`：四个投影逐字节比对
   `h3_weight_load_f16_raw` 与闸门自算的 fp16 payload ⇒ **F32→fp16 分支一致**
   （12582912/4194304/33554432/16777216 halves 全 ok），闸门仍 3.24× block、端到端 2.34× PASS。
2. `h3_int8_raw_test <convrot transformer>`：该分片里有未量化的 F16 矩阵
   `blocks.0.adaln_proj.linear.weight` [96768][8] ⇒ 用它验 **F16 逐字分支**
   （`first and last 1048576 bytes match the shard`）＋ **拒绝 I8**。两处 0 failure。

`tests/test_ane_vae_residency.c` 的本地实现删掉、改为调用新口（实测权重字节数
24/8/64/32 MiB 与 fp16 预期一致，8 张图正常编译执行）。

---

# F23（2026-09-19 17:40）：D2d 接线实测 —— 正确、能优雅回退，但**在这台机器上净亏**

`h3_video_vae.c` 接好后（`H3_ANE_VAE` 默认 off、`H3_ANE_VAE_MIN_ROWS=1024`、
`H3_ANE_VAE_MAX_BLOCKS=31`、rows 分桶 2048、平面池化、常驻不轮换、逐 block 回退），
用 `tests/test_ane_vae_decode.c`（新 Makefile 目标 `h3_ane_vae_decode_test`）对同一份
确定性 latent 做 A/B：先跑 fp32 Metal 当参考，再跑 ANE，逐像素比对并计时。
`REPEAT` 用来把"首轮建图"和"稳态"分开，否则会把编译成本读成解码成本。

## 1. 正确性：PASS

| ANE block 数 | cosine | rel_l2 | max_abs | 非有限 |
|---|---|---|---|---|
| 6 | 1.000000 | 2.178e-04 | 1.505e-03 | 0 |
| 16 | 1.000000 | 4.926e-04 | 2.858e-03 | 0 |
| 31（实际停在 30） | 1.000000 | 5.185e-04 | 2.972e-03 | 0 |

门是 `cos>=0.999 && rel_l2<=0.02`，量级与 D1 闸门单投影的 fp16 误差一致（5~8e-4）。
形状、帧数（12 latent → 39 帧 256×256）、跨 chunk 路径都对。

## 2. 回退：按设计工作

`vae.b26.w1 compile failed` / `vae.b30.w1 compile failed` 时只打印
"stopped at block N of 31"，该 block 之后的 block 照常 fp32 Metal，输出仍然正确
（上表 31 那行其实是 30 张常驻 + 6 个 block 回退的结果）。`H3_ANE_VAE_STRICT=1` 才让
prepare 失败变成硬错误。

## 3. 性能：稳态 0.95~1.00×，即**没有收益**；产物留不住时 0.13~0.38×

latent 12×16×16（2 个 chunk、1797 行 → 2048 桶）、streaming=1：

| ANE block | 稳态墙钟 | 同轮 metal | 比值 | ANE 侧 pack+eval |
|---|---|---|---|---|
| 6 | 20.24 / 20.45 / 20.43 s | 20.18 s | **1.00×** | — |
| 16 | 20.80 / 20.61 s | 19.71 s | **0.95×** | 0.94 + 1.16 s |
| 8（产物被淘光，每 pass 重编译 32 张） | 56.56 s | 20.58 s | 0.36× | 0.49 + 1.21 s |
| 31（同上，104~124 张） | 122~158 s | 20.30 s | 0.13~0.17× | — |

三个原因相乘，任何一个都足以否掉收益：

1. **占比太小**：D1 闸门量的"36 block 一条 pass 3.22 s → 1.05 s"只占这次解码墙钟
   （20 s）的 16%，理论上限 ~2.2 s（11%）。这 20 s 里大头是 10.4 GiB F32 权重的流式读
   与注意力，不是那四个 GEMM。
2. **跨界要 drain GPU**：`h3_ane_projection_apply` 前后各一次 `h3_gpu_submit()`
   （commit + `waitUntilCompleted`）。16 block × 2 chunk = 128 次调用，ANE 侧只花
   2.10 s，被替换的 Metal GEMM 是 16×2×89.5 ms ≈ 2.86 s，账面该省 0.76 s，
   实测反而 **+0.9 s** ⇒ 每次调用约 **13 ms 的流水线空泡**（GPU 等 CPU、CPU 等 GPU 排空），
   正好把理论收益全吃掉。闸门里没有这个空泡，因为它连跑同一张图。
3. **产物根本留不住**：一个 rows 形状 4.5 GiB，本机系统卷只有 9 GiB 空闲，而
   `H3_ANE_CACHE_MIN_FREE_MIB` 默认 6144。`bridge_cache_target()` 取
   "min(上限, 空闲−下限+已有)"，空闲≈下限时 target 塌到 ~0 ⇒ LRU 疯狂淘汰，
   而被存活图硬链接钉住的条目删掉也不返还空间（代码注释已承认这点），
   于是**缓存被清空、空闲却没涨**，每轮解码重付 0.6~1.1 s/张的编译。
   要看稳态只能临时把下限压到 1 GiB（表里 16 block 那行就是这么测的）。

## 4. 唯一能让 ANE 摊薄固定成本的形状，这台机器测不了

收益要成立得靠大 tile（GEMM 时间随时长线性涨，而跨界/建图成本近似不变）。
但 latent 32×32 ⇒ rows 7173 ⇒ 桶 7424 ⇒ 池化平面按 D2a 的账（rows 2048 时 124 张
1584 MiB）外推到 ~5.7 GiB，加上 10.4 GiB F32 权重流式读，超过本机 **16 GiB** 统一内存。
换 rows 形状还要再付一份 4.5 GiB 产物 —— 与第 3 点冲突。

## 5. 由此定下的收尾口径

- 接线保留（正确性、回退、`H3_ANE_VAE_STATS` 分阶段计数都在），**默认 off 不变**。
- Phase D 不再往"整 block 一张图/融合 FFN"投入：第 1、2 点与图的数量无关，
  少建几张图只会让第 3 点的编译成本按比例下降，收益上限仍是 11% 减去空泡。
- 缓存的空闲下限与"钉住条目不返还空间"是**独立缺陷**（DiT/int8 闸门同样受影响），
  记在这里待单独处理：淘汰前应看 `st_nlink`，或空闲低于下限时干脆不写。

## F23 更正（同一轮，紧接着的复测）：第 3 条原因写重了，产物留不住的**机制没定论**

`H3_ANE_CACHE_DEBUG=1` 复跑两遍（8 block 一次、16 block 一次）：
**evicted 行 0 条**，16 block 那轮是 64 条 `cache restored` + 0 条 `cache miss`，
缓存目录稳定在 2186 MiB、系统卷空闲 6334 MiB ⇒ 我上写的"LRU 疯狂淘汰"不成立。

站得住的部分：
- 默认下限（6144 MiB）下，**31 block 那一轮结束时缓存是 0 字节**、16 block 的三轮
  （144/150/61 s）结束时是 33 MiB；把下限临时降到 1024 MiB 后，同一形状稳定留住
  ~2 GiB 产物，稳态才量得到 0.95×。⇒ "默认配置下这套产物不可靠地留不住"是真的。
- 机制没定：可能是 store 阶段直接失败（`.tmp` 写不进、或被清扫），也可能是
  会话开头的 trim 一次性清空（最早那轮 6 block 冷编译前空闲确实只有 5.4 GiB < 下限，
  当时 DEBUG 没开）。要定性得单独开一轮：带 DEBUG 从空缓存起跑，逐条看 miss→store→下轮命中。
- 因此 **Phase D 的否掉结论只依赖第 1、2 条**（占比 16% 与每次跨界 ~13 ms 空泡），
  它们与缓存无关；第 3 条只解释"为什么冷跑是 0.13~0.38× 而不是 0.95×"。

# F24 DiT 逐算子剖析（2026-09-21，`H3_DIT_OP_PROFILE=1`）

## 0. 为什么要自己加括号
整块 DiT 编码进**一条** command chain（`encode_forward` 只在块边界 submit/continue），
所以没有括号时驱动只报得出"一块多少毫秒"，单个 kernel 看不见。
新增开关 `h3_dit.c`：`run_block` 内的 `OP()` 在 `H3_DIT_OP_PROFILE=1` 时给每个算子
补上 `submit`+`begin`，于是每个算子有独立 command buffer 和一个可测窗口。
只碰 `run_block`——`encode_forward` 里同名 `OP` 的调用本身就是链的 begin/continue/submit。
读的是 `stats.command_encode_seconds`/`command_wait_seconds` 的增量（主机时钟围着
`waitUntilCompleted`，可信），**不读** root `GPUEndTime-GPUStartTime`（既有记录注过：MPSGraph 会
把注意力排进子 buffer，同一配置它曾报过 0.298 s / 0.400 s / 121.995 s，不可信）。

## 1. 测量条件
`/Users/jay/h3_sys/MiniMax-H3-Convrot`（int8 convrot 权重，本机 M4 16 GB 无 TensorOps
⇒ 计算走 BF16 路径）、`--ssd-streaming --steps 2 --reuse 1 --seconds 2`、
`H3_DIT_RESIDENT_BLOCKS=6`（必须钉住：自适应驻留会随当时可用内存漂移，A/B 就不可比）、
`caffeinate -is`（第一次 576 那两轮
是机器睡眠把 `MTLCompilerService` 弄死后失败的；`864_unfused` 那一轮墙钟 124 min，
而同一次运行里用 `CLOCK_MONOTONIC` 计出的括号窗口只有 221 s（macOS 睡眠期间该时钟
不走），也是同一件事）。
每标签 100 calls = 2 步 × 50 块。

| 档 | sequence | 每块括号内合计 |
|---|---|---|
| 864×480 | 7074 rows | 2451.6 ms |
| 576×320 | 3249 rows | 921.3 ms |

## 2. 默认路径（融合 MLP + 融合 gate/AdaLN）逐算子

864×480 / 7074 rows：

| 算子 | ms/call | 占比 |
|---|---|---|
| DiT fused MLP（FC1+SwiGLU+FC2）| 1109.2 | **45.2%** |
| DiT QKV projection/norm/RoPE | 607.9 | 24.8% |
| DiT full attention（SDPA）| 503.4 | 20.5% |
| DiT attention output | 206.4 | 8.4% |
| DiT fused MLP gate + next attention AdaLN | 15.8 (×98) | 0.6% |
| DiT fused attention gate + MLP AdaLN | 7.4 | 0.3% |
| DiT attention AdaLN（每步第 0 块）| 82.3 (×2) | 0.1% |

576×320 / 3249 rows：MLP 476.8（**51.8%**）、QKV 242.6（26.3%）、SDPA 112.4（12.2%）、
attention output 85.0（9.2%）、两种 fused gate+AdaLN 各 ~2.0 ms（合计 0.4%）、
attention AdaLN 14.6 ms（×2）。

⇒ **份额随分辨率移动的方向只有一个：注意力。** SDPA 是 O(M²)，12.2%→20.5%；
MLP 51.8%→45.2%。两档相加 MLP+QKV 都占 70~78%，AdaLN/gate 全线 ≤1%
（此前那轮 AdaLN 融合优化已经把这类逐元素算子压到零头，没有余量了）。

## 3. 拆开 FC1 / SwiGLU / FC2（`H3_DISABLE_FUSED_MLP=1 H3_DISABLE_FUSED_GATE_ADALN=1`）

| 算子 | 864 ms/call | 576 ms/call |
|---|---|---|
| DiT MLP input（FC1，K=5376→N=28672）| 681.0 | 313.0 |
| DiT SwiGLU | 13.8 | 3.7 |
| DiT MLP output（FC2，K=14336→N=5376）| 336.9 | 160.6 |
| DiT attention gate / MLP AdaLN | 4.4 / 2.1 | 1.5 / 1.1 |

SwiGLU 0.4~0.6%，FC1:FC2 ≈ 2:1（正好是 2×FFN 输出宽度的比）。

## 4. 折算成有效算力（扣掉括号空泡后）
`TFLOPS` 用 `2·M·K·N`，注意力用 `4·M²·d·heads`：

| 算子 | 864 TFLOPS | 576 TFLOPS |
|---|---|---|
| FC1 | 3.29 | 3.39 |
| FC2 | 3.42 | 3.51 |
| QKV | 2.77 | 3.45 |
| attention output | 2.89 | 3.74 |
| SDPA | 2.96 | 3.19 |
| fused MLP（一个 kernel 装三件事）| 3.00 | 3.27 |

⇒ **非 TensorOps 的 BF16 路径已经跑到 2.8~3.7 TFLOPS**，各 GEMM 之间只差三成，
kernel 层没有大肉。省时间的路子只剩两条：**少算**（`--reuse`/`--layers`/
`--token-reduction`/块稀疏注意力）或**低比特上 TensorOps**（本机 M4 拿不到，
`h3_gpu.m:378` 的闸门是 M5 或 `H3_VDN_INT8`）。
唯一可疑的一格：fused MLP 比同一档拆开的两次 GEMM 慢 8~10%，看着像省 405 MB 激活
（7074×28672 BF16）付出的代价。但**同一次运行外不可比**——见 §6。
> **修订（同日 F25）**：这一格已由同进程交替 A/B 判为噪声，fused 无可测代价，撤回"慢 8~10%"。

## 5. 剖析自身的失真（必须先量，否则占比是假的）
576 同配置、只差开关的配对：

| | denoise wall | submissions |
|---|---|---|
| 无括号 | 81.410 s | 90 |
| 有括号 | 92.182 s | 692 |

⇒ 每个括号 `(92.182−81.410)/602 = ` **17.89 ms** 纯排空空泡。
用它校正：每块 921.3 − 6.02×17.89 = **813.6 ms**，与无括号的 814.1 ms **闭合到 0.06%**；
且括号内算子合计 92.126 s ≈ 有括号的 denoise 全程 92.182 s（99.9%）
⇒ 块外算子（embed/final/add/velocity）合计不到 0.1%，不用再单列。
864 那档扣同一常数后仍比无括号高 5.7%（2343.9 vs 2217.7 ms/块）⇒ M 越大排空越贵。

**读数规矩**：绝对 ms 带 ~18 ms/call 的正偏，≤2 ms 的标签只说明"在括号地板以下"，
不能判零；占比是同一次运行内的相对量，可信。

## 6. 跨进程同算子能差两成 ⇒ 只在同一次运行内比较
`DiT QKV projection/norm/RoPE` 在 864 融合那次是 607.9 ms，在不融合那次是 499.3 ms
（−18%），而它两边都不是被改动路径；`DiT attention AdaLN` 更是 82.3 vs 39.8 ms（×2 次）。
差异只能来自进程间分配器/MPSGraph 状态。所以 §4 里 "fused MLP 慢 8~10%" 这句话
的置信度低于这个噪声，要定罪得做同进程 A/B（一次运行内两种 MLP 路径交替）。

## 7. 与 F23 的互相印证，以及下一步的天花板
- F23 从"6 block 与 16 block 两轮各自反推"得到的 **每次跨界 ~12-13 ms 空泡**，
  这里被独立量到：一次纯 GPU `submit`+`wait` 括号就是 **17.9 ms**。DiT↔ANE 的
  每次包边界至少要付这个量级，不是 ANE 特有的开销。
- 给"再上加速器"划的线：DiT 里值得 offload 的只有 MLP（45~52%）和 QKV（25~26%）；
  只搬注意力在 864 的天花板是 20.5%、在 576 只有 12.2%，还要先解决上面那条
  每次跨界 ~18 ms 的空泡（每块 4 次跨界 = 72 ms ≈ 576 一档 MLP 份额的 15%）。
- 块稀疏/窗口注意力的价值随分辨率单调上升：864 把 SDPA 砍一半 ≈ 全局 −10%，
  576 只有 −6%。这与既有那条"≥864 时瓶颈在 GPU 计算、≤576 时瓶颈在权重 I/O，
  所以 `--token-reduction` 只在高分辨率有效"是同一个方向。

## 8. 复现
```sh
export H3_DIT_RESIDENT_BLOCKS=6 H3_DIT_OP_PROFILE=1
export H3_CLIPPROJ_DIR=/Volumes/data/.lmstudio/models/Qwen3-VL-4B-Instruct-int8-convrot
export H3_CLIPPROJ_PROJ=/Volumes/data/.lmstudio/models/ClipProj-MiniMax-H3
caffeinate -is ./h3 -d /Users/jay/h3_sys/MiniMax-H3-Convrot -p "A red fox" \
    --width 864 --height 480 --seconds 2 --steps 2 --reuse 1 --ssd-streaming --profile \
    -o /tmp/opprof_864_fused.mp4
# 拆 FC1/SwiGLU/FC2 与 gate/AdaLN：再加 H3_DISABLE_FUSED_MLP=1 H3_DISABLE_FUSED_GATE_ADALN=1
```
原始日志 `/tmp/opprof_{864,576}_{fused,unfused}.log`、`/tmp/opprof_{864,576}_plain.log`。

# F25 同进程 A/B：fused MLP 不比拆开的两次 GEMM 慢（2026-09-21，`h3_mlp_fusion_bench`）

## 0. 要判的案
F24 §4 留了一格可疑：864 一档 `DiT fused MLP` 1109.2 ms/call，而同一档拆开的
FC1+SwiGLU+FC2 = 681.0+13.8+336.9 = 1031.7 ms ⇒ fused 看着慢 7.5%。
但 F24 §6 量到同一个 QKV 算子跨进程能差 18%，这条 7.5% 落在那条噪声带里，
定罪和释放都不够证据。这一轮把两条路径搬进**同一进程、同一套 buffer、逐轮交替**。

## 1. 工具与口径
`tests/bench_mlp_fusion.c` → `h3_mlp_fusion_bench`（独立 target，已进 `clean`，不在 `all` 里）。
- 形状取 DiT 真值：`HIDDEN 5376`、`FFN 14336`、FC1 输出宽度 `2×FFN`；rows 取两档
  sequence 3249 / 7074。
- 融合侧一次 `h3_gpu_mlp_bf16`；拆开侧 `h3_gpu_linear_bf16` + `h3_gpu_swiglu_bf16` +
  `h3_gpu_linear_bf16`，多物化一块 405 MB（7074×28672 BF16）的 FC1 输出。
- 窗口外 `h3_gpu_begin`，窗口内"编码 + 一次 `submit`（含 `waitUntilCompleted`）"
  ⇒ 每一测只排空一次，两侧对称，测完链是关着的。
- 逐轮交替、奇偶轮交换先后 ⇒ 热漂移与时钟偏置均摊到两列。
- buffer 全部 `memset` 清零，只测时间不测数值。
- 条件：本机 16 GB M4（`int8 off` ⇒ 走 BF16 MPSGraph）、`caffeinate -is`、
  跑法 `./h3_mlp_fusion_bench 13 3249 7074`。

## 2. 读数
| rows | 轮数 | fused 中位 ms | unfused 中位 ms | fused/unfused | 逐轮比值范围 | 单列极差/中位 |
|---|---|---|---|---|---|---|
| 3249 | 7 | 432.715 | 437.683 | 0.989 | 0.942..1.002 | 3.9% / 5.3% |
| 7074 | 7 | 998.180 | 990.884 | 1.007 | 0.970..1.020 | 6.4% / 6.2% |
| 3249 | 13 | 458.637 | 458.249 | 1.001 | 0.921..1.036 | 8.4% / 10.9% |
| 7074 | 13 | 1120.313 | 1126.224 | 0.995 | 0.937..1.101 | 11.5% / 11.3% |

四次独立跑（两档 × 两种轮数）的比值：0.989、1.007、1.001、0.995 ⇒ 全在 1.0 的
1.1% 以内，且两次 7074 的符号相反（0.995 / 1.007）。按 13 轮那组做配对统计：
7074 一档逐轮比值 mean 1.011、单轮 sd 5.3% ⇒ sem 1.5%、95% 置信区间
**1.011 ± 3.2%（上限 1.042）**；3249 一档 mean 0.993 ± 1.7%。⇒ F24 那句"慢 8~10%"落在区间外，**排除**；
能保留的最坏说法是"864 档可能慢 ≤4.2%"，而那已经不是值得动手的量。

## 3. 顺量到的两条噪声性质
- **同进程逐轮**极差 3.9~11.5%（轮数越多采样越全，暴露的尾部越大），中位数把它压到 1% 级。
  ⇒ 这台机器上任何单算子 A/B，**少于 3% 的差都得用"同进程 + 交替 + 多轮中位数"**才谈得上。
- **同一 binary、不同次运行**绝对值仍能差 6~12%：3249 fused 中位 432.7 → 458.6（+6.0%）、
  7074 fused 中位 998.2 → 1120.3（+12.2%）。这是 F24 §6 那条跨进程噪声在同一 binary 上的重现
  （电源/热状态），也再次说明**只有同一次运行内的比值可信**。

## 4. 结论与影响
- **无罪**：fused MLP 图本身没有可测的时间代价 ⇒ `H3_DISABLE_FUSED_MLP=1` 只作为
  数值对参考的口子保留，默认融合路径不动。省下 405 MB 常驻激活（864 档）这一点白拿。
- F24 §7 的 offload 池不变（MLP 45~52% + QKV 25~26%），但 MLP 那 45~52% 里
  **没有 kernel 组织形式的漏**：之前以为"改回两次 GEMM 能白捡 8%"不存在，
  要动它只剩"少算"或低比特上 TensorOps。
- 本结论**不覆盖**两种情形：① 块内真实数据/缓存邻居下的相互作用（bench 是零 buffer、
  孤立窗口，只能判"融合图本身不亏"，不能证明 864 档块内一定零差）；② M5 上的 int8/TensorOps
  融合 MLP 路径（本机拿不到，`h3_gpu.m:378` 的闸门是 M5 或 `H3_VDN_INT8`）。

# F26 现成窗口注意力内核每对便宜不了：比稠密 SDPA 慢 70~78 倍（2026-09-21，`h3_attention_bench`）

## 0. 要判的案
F24 §7 之后排在第一位的方向是"块稀疏/窗口注意力在 864 一档占 20.5%，砍一半 ≈ 全局 −10%，
而且内核现成（`h3_gpu_sdpa_window_mask_bf16` / `h3_flash_attn_windowed` /
`h3_flash_attn_tiled_windowed`），只差接线 + 画质 A/B"。接线之前先量"每对便宜不便宜"，
因为整条估算默认**稀疏内核算一对和稠密 MPS 算一对一样贵**——这个前提没人量过。

## 1. 工具与口径
`tests/bench_attention.c` → `h3_attention_bench`（独立 target，已进 `clean`，不在 `all`）。
- 同一进程、同一套 q/k/v/out（各 `sequence×56×128` BF16），两侧交替、奇偶轮交换先后；
  窗口外 `h3_gpu_begin`，窗口内"编码 + 一次 `submit`（含 wait）"（F25 定的那套口径）。
- A 侧 = DiT 真正在用的 `h3_gpu_sdpa_bf16`（MPSGraph 稠密）；
  B 侧 = `h3_gpu_flash_attn_bf16`，`causal=0` ⇒ 走 `h3_flash_attn_tiled_windowed`。
- 几何按 F24 那两档真实分解：`7074 = 189(文本+音频) + 17×405`、`3249 = 189 + 17×180`
  （两档都是 17 帧 + 189 非视频行，互相印证）。
- `radius=16` 是**对照**：17 帧全可见，B 侧算的对数和 A 侧基本相同 ⇒ 直接读出每对成本比。

## 2. 读数（3 轮，中位数）
| sequence | B 侧窗口 | 稠密 A ms | 窗口 B ms | B/A | 该窗口实际算的对数占稠密比 |
|---|---|---|---|---|---|
| 7074 | radius 1 | 540.5 | 7169.8 | **13.27** | 18.73% |
| 7074 | radius 16（对照，几乎全可见） | 569.3 | 38730.0 | **68.03** | 97.40% |
| 3249 | radius 1 | 103.9 | 1690.8 | **16.28** | 20.86% |
| 3249 | radius 16（对照） | 125.4 | 8983.1 | **71.63** | 94.52% |

对数占比按内核的可见域逐帧算出（文本行只见文本、视频行见"文本 + 半径内整帧"，
边缘帧按截断算），不是估的。

**自洽校验**：`(B/A) ÷ 对数占比` = 每对成本比 ——
7074：13.27/0.1873 = 70.8 对 68.03/0.9740 = 69.8（差 1.4%）；
3249：16.28/0.2086 = 78.0 对 71.63/0.9452 = 75.8（差 2.9%）。
两种窗口宽度、两种几何都落在 **每对比稠密 MPS 贵 70~78 倍** ⇒ 这是内核性质，不是巧合。

## 3. 为什么这么慢（读 `h3_shaders.metal:5600-5712`）
`h3_flash_attn_tiled_windowed` 名字里的 tiled 只是"一个 threadgroup 64 行 query"，
没有 KV 分块、没有 threadgroup 复用：**一线程一 query 行**，
- 每个 key 都从 global 读一整行 K（128 通道）再点积，64 个线程各读各的 ⇒ 同一批 K 行被读 64 遍；
- online softmax 的 `acc[128]` **每个 key 重标一遍**（又多 128 次乘法和一次 V 行读取）；
- 全程标量 fp32 FMA，无 TensorOps/无 matmul 分块。
⇒ 带宽而非算力封顶，比 MPSGraph 的融合注意力差两个数量级是必然结果。
（`h3_flash_attn_windowed`、`test_flash_attn.c` 那一族是同一写法；
`h3_gpu_sdpa_window_mask_bf16` 只是**造 mask** 再喂稠密 SDPA，一对都不省。）
另外：`h3_gpu_flash_attn_bf16` 在 `h3_dit.c`/`h3.c` 里**没有任何调用点**，只有测试在用。

## 4. 结论与影响
- **F24 之后那条"只差接线"的第一优先方向作废**：接线不会带来全局 −10%，
  会把注意力那一个算子从 540 ms 变成 7170 ms（13.3 倍）——按它 20.5% 的份额，
  每块时间变成 2.5 倍。
  块稀疏注意力的 20.5% 天花板仍然成立，但要吃到它得**新写一个分块 flash 内核**
  （threadgroup 里 tile K/V、matmul 累加、只保留窗口内的 KV tile），
  这是一周内核工程，不是接线活；且要赢过 2.96 TFLOPS 的稠密 MPS SDPA 才有意义。
  > **修订（同日 F27）**：这句只对"用库里那个窗口内核"成立。把一次稠密 SDPA 拆成 G 次
  > 小稠密 SDPA 时，每对成本在 540 行以上不涨（0.8~0.9×），"新写 flash 内核"不是前置；
  > 真正缺的是让 SDPA 支持 query 行数 ≠ key 行数（现在是方阵）。见 F27 §2、§4。
- 顺带校准了剖析口径的**绝对**可信度：本 bench 的稠密中位（7074 档 540.5 ms）
  对上 F24 块内括号读到的 503.4 ms（同一算子、不同进程），差 7% —— 与 F25 §3
  那条"同 binary 两次运行差 6~12%"一致，绝对值仍只当量级用。
- 对"还有没有肉"的回答没有变：MLP 45~52% + QKV 25~26% 的 offload 池、
  以及"少算 / 低比特上 TensorOps"这两条省时路子（F24 §4）不受影响。

# F27 把一次稠密 SDPA 拆成 G 次小稠密 SDPA：每对成本几乎不涨（2026-09-21）

## 0. 问题从哪来
F26 §4 的结论是"要吃到注意力那 20.5% 就得新写一个分块 flash 内核，一周工程"。
那句话只覆盖了"用库里那个窗口内核"这一条路。块稀疏真正的成本问句是：
**同样跑稠密 MPSGraph SDPA，把一次 M×M 拆成 G 次 (M/G)×(M/G)，每对 token 还值多少钱？**
- 每对成本随 M 变小而明显上涨 ⇒ 稀疏必须新内核，F26 成立；
- 每对成本基本不变 ⇒ G 次小稠密调用本身就是现成的稀疏内核，"一周内核工程"不存在。

## 1. 工具与口径
`tests/bench_attention.c` 重写为三档：`dense`（一次全序列，产品路径）、
`split`（G 次小稠密）、`window`（F26 那个库内核，只作对照，默认不跑）。
- `split` 用 **G 套各自独立**的小 buffer（不是复用同一组，免得做出人造的热 cache）；
  G 次编码进**同一条** chain，一次 `submit`（commit+wait）计一段墙钟。
- 每个 shape 首次使用要编自己的 MPSGraph ⇒ 先各 warm 一次再进计时轮。
- 5 轮、轮间轮换顺序、取中位；每档另报 CPU 编码 / GPU 排空两段。
- 同进程配对，符合 F25 §3 那条规矩（绝对 ms 只当量级，比值才是结论）。

## 2. 结果
| M | keep=1/G → 每次行数 | time × | pairs × | 每对 × | 相对稠密效率 |
|---|---|---|---|---|---|
| 7074 | 3 → 2358 行 | 0.293 | 0.333 | 0.9 | 1.14 |
| 7074 | 6 → 1179 | 0.140 | 0.167 | 0.8 | 1.19 |
| 7074 | 9 → 786 | 0.099 | 0.111 | 0.9 | 1.12 |
| 7074 | 12 → 589 | 0.079 | 0.083 | 0.9 | 1.06 |
| 7074 | 18 → 393 | 0.059 | 0.056 | 1.1 | 0.94 |
| 7074 | 24 → 294 | 0.044 | 0.041 | 1.1 | 0.94 |
| 3249 | 3 → 1083 行 | 0.309 | 0.333 | 0.9 | 1.08 |
| 3249 | 6 → 541 | 0.164 | 0.166 | 1.0 | 1.02 |
| 3249 | 9 → 361 | 0.129 | 0.111 | 1.2 | 0.86 |
| 3249 | 12 → 270 | 0.098 | 0.083 | 1.2 | 0.84 |
| 3249 | 18 → 180 | 0.075 | 0.055 | 1.4 | 0.74 |
| 3249 | 24 → 135 | 0.073 | 0.041 | 1.8 | 0.57 |

绝对值（中位，ms）：7074 稠密 623.9~652.2，split G=6 89.6 / G=12 50.1 / G=24 28.8；
3249 稠密 120.5~125.7，split G=6 20.6 / G=12 12.3 / G=24 8.8。

⇒ **每对成本在 ~540 行以上完全不掉（0.8~0.9×，甚至略赚）**，540 行以下才开始涨，
到 135 行涨到 1.8×。对照 F26 那个窗口内核的 **70~78×**：同一个"少看几对"的需求，
一条路贵两个数量级，一条路免费。轮间散布 ±3~6%，G=3/6/9/12 那几档的"每对 0.8~1.0"
是稳的，1.1/1.2 这种一档之别落在噪声里，不必细抠。

## 3. 折算天花板（配上 F24 的份额）
F24：864×480 / 7074 档每块括号内 2451.6 ms，其中 full attention 503.4 ms = **20.5%**；
576×320 / 3249 档每块 921.3 ms，SDPA 112.4 ms = **12.2%**。

| keep | 864 SDPA → ms/块 | 864 全局省 | 576 SDPA → ms/块 | 576 全局省 |
|---|---|---|---|---|
| 1/6 | 70.5 | **−17.7%** | 18.4 | **−10.2%** |
| 1/12 | 39.8 | −18.9% | 11.0 | −11.0% |
| 1/24 | 22.2 | −19.6% | 8.8 | −11.2% |

（份额取自 F24 另一个进程，比值取自本进程，属于跨来源相乘，按 6~12% 的绝对噪声读；
量级结论不受影响。）此前"窗口注意力最多全局 −10%"那句是保守了：
按 1/6 keep 且不吃质量亏，864 一档的上限是 −17~18%，keep 再狠也只多拿不到 2 个点
⇒ **这项的取舍几乎全在质量侧，不在速度侧**。

## 4. 那么缺的是形状，不是内核
`h3_gpu_sdpa_bf16` 只能算**方阵**：`h3_gpu.m:1542-1568` 建图时 Q 与 K/V 共用同一个
`sequence`，`:1693` 又把 batch 钉成 1。
"每个 query 块看它 top-k 选中的 key 块" = (B 行 query) × (k·B 行 key)，是**矩形**；
方阵只表达得出**块对角**（每个 query 块只看自己那一片）。
块对角对视频几乎必然判死：帧与帧之间看不到，视频行还看不到那 189 行文本条件。
⇒ 前置改动是给 SDPA 支持 `query_rows ≠ key_rows`（顺手 batch>1）。
gather 侧**不需要**新内核：选中的 key 块都是连续段，现成的
`h3_gpu_copy_bf16(gpu, dst, dst_offset, src, src_offset, elements)` 就能打包，
流量 O(rows·d) 对算力 O(rows·key·d) 可忽略。
另外确认了一件省心的：G 次调用的 CPU 编码只有 0.1~1.3 ms（G=24 时 1.0~1.3），
全编码进同一条 command buffer，**不付** F24 §5 那 17.9 ms/括号。

## 5. 已知未测
- 矩形 (B × kB) 的每对成本**没测** —— 现在的 API 表达不了它。加完 API 第一件事就是量它，
  尤其 kB 远大于 B 时（query 少 key 多）是不是还平。

  > **修订（同日 F28）**：API 已加（`h3_gpu_sdpa_rect_bf16`）并测完 —— 7074 档每个形状
  > 每对都是 **0.9~1.0×** 稠密，带打包流量也不涨；唯一变贵的是"单次调用活量 <~100k 对"。
  > 结论从"最多 −17.7%"改到 **−19.4%**（keep 5%），见 F28 §2/§4。
- 50 层各用不同 keep ⇒ 几十个不同 shape 的 MPSGraph 编译：首步 warm-up 时间与
  `sdpaCache` 的显存占用没量（16 GB 机器上这条必须先量再上）。
- buffer 全 0：这些内核没有数据相关早退，理论上耗时与值无关，但没排除 MPSGraph
  按值选 kernel 的可能。
- 质量：块对角 / 窗口 / top-k 选块都要真实渲染 A/B，这一步省不掉。
  按 §3，速度侧 1/6 keep 就基本吃满，所以**该从最保守的 keep 起测**。

## 6. 复现
```sh
make -j8 h3_attention_bench
caffeinate -is ./h3_attention_bench 5 ds \
  7074:189:17:405:0:3 7074:189:17:405:0:6 7074:189:17:405:0:9 \
  7074:189:17:405:0:12 7074:189:17:405:0:18 7074:189:17:405:0:24 \
  3249:189:17:180:0:3 3249:189:17:180:0:6 3249:189:17:180:0:9 \
  3249:189:17:180:0:12 3249:189:17:180:0:18 3249:189:17:180:0:24
# MODE = SEQUENCE:TEXT:FRAMES:TOKENS_PER_FRAME:RADIUS:GROUPS，radius 0 关掉窗口对照
```
原始日志 `/tmp/split_dense.log`。

# F28 矩形 SDPA 每对成本 = 稠密：块稀疏的定价彻底改口，代价只在质量侧（2026-09-21，`h3_gpu_sdpa_rect_bf16` + `h3_attention_bench`）

## 0. 这一轮补的是 F27 §5 的第一条
F27 只测了方阵（`kB = B`），因为老 API 表达不了别的形状。它留下的问句是：
**query 少、key 多的矩形调用是不是还平？** 现在的稀疏方案全是这个形状
（一个 query 块只看它选中的几片 key），如果不平，F27 那张表就用不上。
本轮先补 API，再立刻量它。

## 1. 改了什么
- `h3_gpu.m`：建图函数拆出 `key_sequence`，Q 与 K/V 各自的行数和 shape
  （`H3SDPA` 多一个 `keyShape`，cache key 多一个字段），新增
  `h3_gpu_sdpa_rect_bf16(gpu, out, q, k, v, query_rows, key_rows, heads, dim, scale)`。
  原 `h3_gpu_sdpa_bf16` 走 `sequence, sequence`，产品路径不变。
- `tests/test_flash_attn.c`：`test_sdpa_rectangular` —— 不依赖 safetensors fixture 的数值闸
  （已接进 `make test`）。三件事：rect(5,5) 与老方阵调用**逐位相同**；
  rect(5,11) 是 dense(11) 前 5 行的子集；把两段不连续的 key 行用
  `h3_gpu_copy_bf16` 打包成 11 行再喂 rect，结果对 BF16 CPU 参考仍只有 2.2e-04 误差。
  ⇒ 打包 gather 不需要新内核这点是**验过**的，不是推测。
- `tests/bench_attention.c`：加 `rect`（纯调用，key 视为已打包）和 `rectg`
  （每次调用前真的从全尺寸 K/V 里 `copy_bf16` 出这段窗口），MODE 多一个 `KEY_ROWS` 字段。
  一轮里同时出 dense / split / rect / rectg 四档，同进程配对。

## 2. 结果：每对成本就是 1.0×
`time × / pairs ×`，189 文本行、17 帧、5 轮中位；`rect` 是纯注意力，`rectg` 带上打包流量。

| M | 调用 × B 行 × S key | keep(=pairs) | split 每对 | rect 每对 | rectg 每对 |
|---|---|---|---|---|---|
| 7074 | 6 × 1179 × 354 | 0.050 | 0.9 | **1.0** | **1.0** |
| 7074 | 6 × 1179 × 707 | 0.100 | 0.9 | **0.9** | **0.9** |
| 7074 | 12 × 589 × 354 | 0.050 | 1.0 | **1.0** | 1.1 |
| 7074 | 18 × 393 × 354 | 0.050 | 1.0 | **1.0** | 1.2 |
| 7074 | 18 × 393 × 707 | 0.100 | 1.0 | **1.0** | 1.2 |
| 7074 | 18 × 393 × 1179 | 0.167 | 1.1 | **0.9** | 1.1 |
| 3249 | 6 × 541 × 361 | 0.111 | 1.0 | **1.0** | 1.2 |
| 3249 | 6 × 541 × 180 | 0.055 | 1.0 | 1.2 | 1.3 |
| 3249 | 9 × 361 × 361 | 0.111 | 1.1 | 1.1 | 1.3 |
| 3249 | 9 × 361 × 180 | 0.055 | 1.2 | 1.4 | 1.4 |
| 3249 | 18 × 180 × 180 | 0.055 | 1.4 | 1.3 | 1.6 |

7074 档绝对值：稠密中位 564~613 ms；keep=5% 的 rect **30.0 ms**（rectg 30.3~35.9）。
⇒ "少看 key"这件事本身**不再收钱**：F26 那个库窗口内核每对 70~78×，
稠密小调用每对 1.0×。同一需求的价格差从两个数量级变成零。

## 3. 贵的只剩一件事：每次调用的算力太小
把三档放一起看，掉不分块的形状、也不掉 B 或 S 单独的大小，掉的是**单次调用的活量**：
- 每次 ≥ ~140k 对（≈4 GFLOP，如 393×354、541×361、1179×354）⇒ 每对 0.9~1.0；
- 每次 ~97k 对（541×180）⇒ 1.2；~65k（361×180）⇒ 1.4；~32k（180×180）⇒ 1.3~1.6。
所以 3249 一档想吃到 5% keep，**不要**用 18 个小块（那是 split 只能给的对角形状），
要用 6 个大 query 块 × 180 key。这条直接就是"块怎么划"的选型依据。

顺带一条 F27 没点破的：方阵 split 只能表达**块对角**且 keep 被钉死成 1/G；
矩形把 keep 和调用次数解耦了 —— 上表 6×1179×354 用 6 次调用做到 5% keep，
方阵要 20 次调用才勉强凑出同样的 keep，而且形状还是错的（帧间不可见）。

## 4. 折算天花板（份额仍取自 F24）
864×480：SDPA 占块内 20.5%（503.4 / 2451.6 ms）；576×320：12.2%（112.4 / 921.3）。

| keep | 864 最优形状的 rectg time× | 864 全局省 | 576 最优 | 576 全局省 |
|---|---|---|---|---|
| 5% | 0.051（6×1179×354） | **−19.4%** | 0.073（6×541×180） | **−11.3%** |
| 10% | 0.093（6×1179×707） | −18.6% | 0.133（6×541×361） | −10.6% |
| 16.7% | 0.181（18×393×1179） | −16.8% | — | — |

和 F27 §3 的估计同量级，但这张表**已经把打包流量算进去了**（keep 越小、块越多，
gather 占比越明显：18 块 5% keep 时 rect→rectg 是 30.1→35.9 ms，+19%；
6 块时 30.0→30.3 ms，几乎为零）。⇒ 结论没变、更硬：**速度侧已经吃满，
剩下 −19% / −11% 能不能拿，全看质量**。再往 20% 以下压 keep 只多不到 1 个点，
不值得为它冒画质风险。

## 5. 显存：真实管线要的比 bench 少两个数量级
bench 给每个块配一套独立 buffer（最狠的一档 18×393×1179 ≈ 813 MB），
是为了不造人造热 cache。真实路径里 copy→SDPA 是**串行消费**同一块 scratch 的：
一整套打包 K/V 只要 `2 × S × 56 × 128 × 2` 字节，S=707 时 **20.3 MB**。
未验：同一条 chain 里复用同一块 scratch 是否被正确排序（要不要像 rect 那样补一个
逐位比对闸）。上真机前必须先把这条闸做出来。

> **修订（同日 F29）**：闸已补（`test_sdpa_scratch_reuse`，进 `make test`）—— 3 块
> 共用一块 scratch、含两段 gather，误差与"每块独立 buffer"完全相同（2.18e-04），
> 相邻块输出不逐位相同。串行复用成立，20.3 MB 那笔账可以用。

## 6. 已知未测
- 真实稀疏模式（文本列全看 + 帧窗口 + 局部）下 50 层的**渲染质量** A/B —— 现在才有
  了可测的形状，之前连表达都表达不出来。
- 每块多段不连续 key（3~5 段）时的编码开销：本轮 rectg 每块只搬 1 段，
  段数上去后 copy 的 op 数翻倍，字节数不变，估计吃的是 CPU 编码那 0.1~1.3 ms。
- 50 层各异 keep ⇒ 几十个不同 shape 的 MPSGraph：首步编译时间 + `sdpaCache` 常驻，
  16 GB 机器上这条**先于**任何逐层表接线。
- buffer 全 0：这些内核没有数据相关早退，理论上与值无关，但没排除 MPSGraph 按值选 kernel。
- 单块内 top-k 选块（谁是那 354 行）需要一个打分通路，本轮完全不涉及。

## 7. 复现
```sh
make -j8 h3_attention_bench h3_flash_attn_tests && ./h3_flash_attn_tests
caffeinate -is ./h3_attention_bench 5 dsgr \
  7074:189:17:405:0:18:354 7074:189:17:405:0:18:707 7074:189:17:405:0:18:1179 \
  7074:189:17:405:0:12:354 7074:189:17:405:0:6:354 7074:189:17:405:0:6:707
caffeinate -is ./h3_attention_bench 5 dsgr \
  3249:189:17:180:0:18:180 3249:189:17:180:0:9:180 3249:189:17:180:0:6:180 \
  3249:189:17:180:0:6:361 3249:189:17:180:0:9:361
# MODE = SEQUENCE:TEXT:FRAMES:TOKENS_PER_FRAME:RADIUS:GROUPS:KEY_ROWS
# 变体 d 稠密 / s 方阵拆分 / r 矩形纯调用 / g 矩形带打包 / w 库窗口内核（默认不跑）
```
原始日志 `/tmp/attn_rect_7074.log`、`/tmp/attn_rect_3249.log`。

# F29 打包 K/V 可以全块共用一块 scratch：显存账从 813 MB 掉到 20 MB（2026-09-21，`test_sdpa_scratch_reuse`）

## 1. 为什么先量这个
F28 §5 留了一条"未验"：bench 为了不做人造热 cache，给每个 query 块配一整套
打包 buffer，最狠的一档（18 × 393 × 1179）光 slice 就 813 MB。
真实逐块循环只需要**一套** scratch —— 前提是同一条 command buffer 里
`copy → SDPA` 的读写序被 Metal 守住。这条不成立的话，块稀疏在 16 GB 机器上
要么改每块一套（显存吃掉一大截），要么每块之间插排空（把 F24 §5 那 17.9 ms/括号
重新请回来）。所以它排在质量 A/B 前面。

## 2. 闸怎么做
`tests/test_flash_attn.c::test_sdpa_scratch_reuse`（跟着 `make test` 跑）：
13 行源 K/V、11 行打包 scratch、3 个 query 块**共用这一块**，每块先写 scratch
再紧跟自己那次 `h3_gpu_sdpa_rect_bf16`，全在一条 chain、一次 submit。
三块选的行互不相同，其中一块是**两段**（5..12 + 0..2），顺带把多段 gather 的
正确性也占了。判据两层：
- 每块输出对它自己的选行参考：`max_err=2.18e-04`，与 F28 里"每块独立 buffer"
  那次 packed 路径的误差**一模一样** ⇒ 没有串味；
- 相邻两块输出不得逐位相同 ⇒ 挡住"三次都算了最后一次内容"这种退化。

## 3. 结论与影响
- 复用安全，`copy` 与后续 rect 的序守住了 ⇒ 打包 K/V 常驻只要
  `2 × S × 56 × 128 × 2` 字节：S=707 时 **20.3 MB**（F28 §5 那笔账成立）。
- 每块**多段**（本轮 2 段）不需要额外 buffer，只是多几条 copy 编码；
  编码字节数没变，代价落在 CPU 编码那 0.1~1.3 ms 上，量它仍是未测项。
- 这不外推的东西：跨 command buffer / 跨层的复用、以及几十个不同 shape 图
  的编译与 `sdpaCache` 常驻 —— 那条仍是逐层表接线前的硬闸（F28 §6）。

## 4. 复现
```sh
make h3_flash_attn_tests && ./h3_flash_attn_tests
# 期望：sdpa_scratch_reuse: OK (... two-segment gather included) max_err=0.000218
```

---

# F30 逐层各异 keep 的三道前置闸：图缓存显存、首步编译、多段 gather 编码（2026-09-21）

## 0. 这一轮要答什么
F27 §5、F28 §6、F29 §3 留的是同两句：**50 层各用不同 keep ⇒ 几十个不同 shape 的
MPSGraph 首步编译时间和 `sdpaCache` 常驻没量过**；**每块 3~6 段不连续 key 的 gather
编码开销没量过**。16 GB 机器上这两条不过关，逐层静态表就只能退回"全层共用一个 keep"，
§task#20 之后没有空间。

## 1. 先把"显存归因"做对，否则结论是反的
第一次量：50 个 shape 跑完 resident 801 → 4042 MB ⇒ 看着像 65 MB/shape，
结论会是"逐层表直接毙"。
**对照组**：把 50 个不同 shape 换成**同一个 shape 重复 50 轮**，resident 一样爬到
4835 MB，而 `engine tensors` 全程钉在 774/774 MB。
⇒ 那个斜坡是 bench 自己每个 MODE 重新 alloc/free 一套 buffer 池的**抖动**，跟 shape
数无关。修法：给 bench 加共享池 —— 所有 MODE 的 `sequence` 和 `groups` 相同才启用，
按最大的 `key_rows` 开一套给全部 shape 复用，这样 resident 的增量才只剩"图缓存"一个来源。
（共享池只在 sequence+groups 一致时开；不同 groups 混进来 slice 尺寸会失真。）

## 2. 闸一：图缓存 ≈ 1 MB/shape —— 绿灯
共享池、50 个不同 `key_rows` 的 shape：resident **794 → 842 MB**，即
**50 张图合计 +48 MB ≈ 1 MB/shape**；`engine tensors` 全程 774/774 MB（就是那套池本身）
⇒ 新 shape 的图**没有**带进额外常驻 buffer。16 GB 机器上"逐层 50 张图"是几十 MB 量级，
不是 GB 量级 ⇒ 不构成否决。

## 3. 闸二：首步编译 3.9 ms/shape，50 层一次性 201 ms —— 绿灯
每个新 shape 第一次用的 CPU 侧 `enc` = 2.9~7.2 ms（中位 3.9），50 个合计 **201 ms**；
shape 缓存命中后每次 re-encode 0.3~0.9 ms。
对照 F24 §5：一个纯 GPU 括号 17.89 ms ⇒ 50 层各编一张图约等于一个括头，且**只在首步付一次**。

## 4. 闸三：段数不花钱 —— 绿灯
| 形状 | 段 1 | 2 | 3 | 4 | 6 | 8 |
|---|---|---|---|---|---|---|
| 6 块 ×1179 行 ×707 key | 54.6 | 55.5 | 55.3 | 56.4 | 56.8 | 55.5 |
| 18 块 ×393 行 ×354 key | 34.6 | — | — | 34.6 | — | 34.4 |

毫秒，5 轮中位；**每对成本全程 0.9 / 1.1，段数从 1 到 8 没有可测代价**。
CPU 编码：段 1 那档 12 条 copy → 0.3~0.5 ms；段 8 且 18 块那档 **288 条 copy + 18 次
rect → 0.8 ms**。一次 submit 不变 ⇒ 不会把 F24 §5 那 17.89 ms/括号请回来。
⇒ "每块看 帧窗口 + 那 189 行文本 + 若干段" 这种真实选块模式，gather 侧真的只是记账。

## 5. keep 带曲线补全：固定 B 动 S，从 6.8% 到 16.7% 一条直线
`B=1179` 固定，只动 `key_rows`（这就是"层数不变、只调 keep"的形状）：

| key_rows | 480 | 560 | 640 | 720 | 800 | 880 | 960 | 1040 | 1120 | 1180 |
|---|---|---|---|---|---|---|---|---|---|---|
| ms | 35.3 | 42.0 | 45.8 | 53.8 | 58.4 | 64.9 | 71.0 | 76.5 | 78.7 | 85.6 |
| 每对 × | 0.8 | 0.9 | 0.9 | 0.9 | 0.8 | 0.8 | 0.8 | 0.8 | 0.8 | 0.8 |

**单调、无悬崖，每对 0.8~0.9**（5 轮中位，dense 对照全程 591~629 ms）。
拟合：每 1% keep ≈ 5.1 ms/块，与 F28 的 keep=5% → 30.0 ms 相接。
⇒ 逐层表里每层 keep 可以**任意取值**，不需要为"避开坏 shape"做量化对齐。

## 6. 踩坑：机器状态会伪造一个 shape 悬崖，而且"同进程配对"救不了它（方法学）
第一轮 50 shape 扫描（`rounds=1`）跑到第 15 个 shape，**全序列 dense 对照**从
547~589 ms 变成 900~1200 ms，此后 20 多分钟不恢复（90 s 空闲、换更小 geometry 都没用；
3249 dense 也从 F28 的 120.5~125.7 ms 变成 168~209 ms）。当时看到
"S=440→500 每对成本跳 4×"，差点记成"MPSGraph 在某个 key_rows 上换了内核"。
两点把它证伪：
- 同一 shape（6×1179×707）在好窗口是 time ×0.092 / 每对 0.9，坏窗口 ×0.145 / 每对 1.5
  ⇒ **小调用比大调用在争用下亏得更多**，所以坏窗口的"每对成本"系统性偏高，
  F25/F27 那条"同进程配对"只保证同状态内可比，**跨状态不守恒**；
- §4/§5 在好窗口重测（5 轮）完全单调，且那组里 dense 满载连跑 27 s 都没退化
  ⇒ 退化不是我自己压出来的，是外部占用（`pmset -g therm` 无告警、`h3-mlx` 已退出，
  在场的是 WindowServer / Qoder Helper / UURemoteServer / VTEncoderXPCService）。
⇒ 闸已经写进 bench：每个 sweep 自带的那次全序列 dense 就是探针，中位超过**本次运行同
sequence 最快 dense 的 1.1×** 就打 `NOTE: ... discard the ratios above`。
以后凡是跨 sweep 比 shape 的扫描（`rounds=1` 那种），先扫有没有 NOTE 再看数字。
坏窗口日志只保留 `resident` 与 `enc` 这两类**非 GPU 计时**字段（§2/§3 用的正是它们，
所以不受影响）。

## 7. 结论：逐层静态表现在只剩质量一道门
三道前置闸全绿：图缓存 +48 MB/50 shape、一次性编译 201 ms、多段 gather 编码 ≤0.8 ms，
外加 keep 任意取值不掉性能。工程可行性这条不用再验；剩下的全在 F28 §4 的天花板
（864 −19.4% / 576 −11.3%）和 task#20 的渲染质量 A/B 上。

## 8. 复现
```sh
make -j8 h3_attention_bench
# 闸一/二：50 个不同 key_rows，sequence+groups 一致才吃得到共享池
typeset -a modes; for i in {1..50}; do modes+=("7074:189:17:405:0:6:$((180 + i * 20)):1"); done
caffeinate -is ./h3_attention_bench 1 dr "${modes[@]}"   # 只看 resident 与 enc 两列
# 闸三：段数
typeset -a seg; for s in 1 2 3 4 6 8; do seg+=("7074:189:17:405:0:6:707:${s}"); done
for s in 1 4 8; do seg+=("7074:189:17:405:0:18:354:${s}"); done
caffeinate -is ./h3_attention_bench 5 dg "${seg[@]}"
# keep 带
typeset -a band; for k in 480 560 640 720 800 880 960 1040 1120 1180; do
  band+=("7074:189:17:405:0:6:${k}:1"); done
caffeinate -is ./h3_attention_bench 5 dr "${band[@]}"
```
MODE 现在是 8 段：`SEQUENCE:TEXT:FRAMES:TOKENS_PER_FRAME:RADIUS:GROUPS:KEY_ROWS:SEGMENTS`。
原始日志：`/tmp/attn_shapes50_shared.log`（闸一/二，S≥480 段的计时作废）、
`/tmp/attn_shapes50.log` 与 `/tmp/attn_shapecontrol.log`（§1 归因对照）、
`/tmp/attn_segments4.log`（闸三）、`/tmp/attn_keepband.log`（§5）、
`/tmp/attn_segments3.log`（整份作废，§6）。

---

# F31 固定 keep 的块稀疏接通产品路径：等价闸位精确，但质量在有利可图的区间就崩了（2026-09-21）

## 1. 接线（`h3_dit.c`，`H3_SPARSE_ATTN=BLOCK_FRAMES[:RADIUS]`）

- video 行按帧分块：每块查询看"自己这 block 的帧 ± RADIUS 帧"的窗口，**外加所有非 video 行**
  （文本/音频条件对每块都全量可见，所以条件不会被稀疏掉）。
- 条件行只打包一次进共享 scratch，每块只重写窗口那一段；video 段之外（前 133/189 行与尾行）
  直接用原始 K/V 走"矩形稠密 key"调用，不打包。
- 默认 off；`--token-reduction`、VDN、几何不整除、`BLOCK_FRAMES > 总帧数` 时自动回退稠密。
  `H3_SPARSE_ATTN=0:0` 这类非法值在 DiT 创建时就报错退出。
- 走稀疏时强制放弃 head-major 的 int8 注意力输出消费（rect 输出是 row-major，布局不兼容），
  所以 A/B 两边要么都 `--use-slower-bf16-attention-output`，要么承认消费路径也不同数值。

## 2. 等价闸：keep=100% 时逐位相同

576×320/1 s（12 帧 × 180 行，序列 2293），`H3_SPARSE_ATTN=12:0` ⇒ 每块看到全部 2293 行。
layer 0 的注意力输出（`H3_DUMP_ACT`，512 行）对比稠密：

| 行段 | max\|d\| | cos |
|---|---|---|
| video 行 | 0.0000 | 1.000000（逐位相同） |
| 条件行 | 0.0156 | 1.000000（稠密自身读回抖动 0.0076%） |

⇒ 打包、偏移、写回、块序全对。后面所有差异都只能归因于 keep，不是接线 bug。

## 3. 隔离算子增量（同 seed、输入逐位相同）

`qkv.00.bin`（注意力输入）在稠密/稀疏两边 max|d|=0.0000 ⇒ 这一层的差异纯粹是"少看了 key"。
layer 0 video 行的 cos：

| keep | 13.7% | 29.4% | 45.1% | 100% |
|---|---|---|---|---|
| layer0 video cos | 0.729 | 0.842 | 0.892 | 1.000 |

13.7% 那档逐层往下：layer0 0.729 → layer34 0.453，连条件行都从 1.000 漂到 0.76~0.80
（隐状态已经被改掉，不再是"只有注意力变"）。

## 4. 端到端 A/B（864×480/2 s，50 层，2 步，reuse 1，ssd-streaming，resident 6）

| 配置 | keep | Euler denoise | Δ | 最终 latent cos | detail | sat | sharp |
|---|---|---|---|---|---|---|---|
| dense | 100% | 209.7 s | — | 1.000 | 6.30 | 41.74 | 8.77 |
| `1:0` | 8.4% | 164.6 s | **−21.5%** | 0.639 | 9.23 | 78.71 | 13.96 |
| `1:1` | 19.8% | 172.2 s | −17.9% | 0.748 | 4.09 | 62.76 | 5.87 |
| `1:2` | 31.3% | 176.6 s | −15.8% | 0.801 | 3.94 | 53.18 | 5.43 |
| `1:3` | 42.7% | 178.9 s | −14.7% | 0.851 | 3.99 | 50.29 | 5.46 |

图（帧 0/8/16 肉眼）：`1:0` 主体散架成条纹噪渣；`1:1` 鬼影、脸没了；`1:2`/`1:3` 构图与主体在，
但毛皮糊掉、饱和度整体偏高（dense 帧 8 是清晰的脸，`1:2` 同帧是糊的）。
detail/sat/sharp 是 17 帧 PPM 上的梯度、饱和度、拉普拉斯均值；2 步下"换了个样本"也会动这些数，
所以它们只能和图一起读，不能单独当结论。

## 5. 结论

- 省的时长基本线性 ≈ 23%×(1−keep)，和 F28 的天花板一致；CPU 编码 3.0~3.2 s（1800 次调用），
  F30 的"多段 gather 编码不要钱"在产品路径上成立。
- 但质量的可用点在 keep ≳ 43%，那里只剩 −14.7%；而 −45.9% 用 `--reuse 2`（20 步实测）不碰注意力就能拿到。
  **固定 keep（时间窗）这条路在能赚钱的区间被质量卡死 ⇒ 关掉，不再投。**
- 唯一还站得住的变体不是"看近的几个块"，而是"看对的块"（top-k / 离线标注的逐层 keep 表）。
  判据这轮量化了：要在 keep ≤ 30% 把 detail/sat 拉回 dense 水平、layer0 video cos 从 0.80 提到 ~0.95+。
  时间窗做不到这件事，说明远块里带着近块补不回来的信息（全局构图/颜色），这正是 top-k 的假设前提。
- 接线本身留着（默认 off）：等价闸证明它对，将来做 top-k 直接换"选哪些 key"即可，不必再动通路。

## 6. 复现

```sh
export H3_DIT_RESIDENT_BLOCKS=6
export H3_CLIPPROJ_DIR=/Volumes/data/.lmstudio/models/Qwen3-VL-4B-Instruct-int8-convrot
export H3_CLIPPROJ_PROJ=/Volumes/data/.lmstudio/models/ClipProj-MiniMax-H3
H3_SPARSE_ATTN=1:3 caffeinate -is ./h3 -d ~/h3_sys/MiniMax-H3-Convrot -p "A red fox" \
  --width 864 --height 480 --seconds 2 --steps 2 --reuse 1 --ssd-streaming --profile \
  --latent-out /tmp/x.bin --frames-dir /tmp/x -o /tmp/x.mp4
# 隔离算子增量（输入位同 ⇒ 差异纯来自算子）：
H3_DUMP_ACT=/tmp/dump H3_DUMP_ACT_ROWS=512 H3_DUMP_ACT_LIMIT=1 ... --use-slower-bf16-attention-output
# 比 qkv.00.bin（输入）与 out.00.bin（输出），bf16 用 (u16<<16).view(f32) 解析
```

本轮 wall 数在负载 7~16 的窗口里测，只用于同窗口内相对比较；latent 与图像指标不受负载影响。

---

# F32 reuse 的收益只取决于"评估几次"，形状上均匀优于堆尾（2026-09-21）

## 0. 先纠 F31 的一处归因

F31/续 18/README 里写的"`H3_REUSE_STEPS=2` 已 −45.9%"是错的：−45.9% 来自 CLI **`--reuse 2`**
（20 步实测，见本文早先 §四）。`H3_REUSE_STEPS` 是"自定义哪几步真的评估"的步号列表
（h3_dit.c:5042 `parse_reuse_steps`），且**只有 `--reuse ≥2` 时才生效**（`reuse_interval > 1` 才会去 parse），
值必须递增且包含 0 与 steps-1，否则报错。它此前从未被测过，README 也没写。四处文字已就地改正。

## 1. 设计：把"评估次数"和"调度形状"分开

864×480/2 s、`--steps 6`、50 层、`--ssd-streaming`、`H3_DIT_RESIDENT_BLOCKS=6`、同种子。

| 配置 | 调度 | 新鲜评估 | 形状 |
|---|---|---|---|
| `base6` | `--reuse 1` | 6/6 | 全新鲜（参照） |
| `auto4` | `--reuse 2` ⇒ 0,2,4,5 | 4/6 | 均匀 |
| `lated4` | `--reuse 2 H3_REUSE_STEPS=0,3,4,5` | 4/6 | 堆尾 |
| `r3` | `--reuse 3` ⇒ 0,3,5 | 3/6 | 均匀 |

`auto4` 与 `lated4` 是**等成本对照**：评估次数相同，只换哪几步新鲜。

## 2. 结果

| 配置 | Euler denoise | vs base | s/评估 | detail | sat | 拉普拉斯 | 帧间差 | latent cos |
|---|---|---|---|---|---|---|---|---|
| base6 | 685.2 s | — | 114.2 | 7.51 | 44.03 | 9.83 | 4.29 | 1.0000 |
| auto4 | 480.7 s | −29.8% | 120.2 | 6.77 (−9.9%) | 43.66 (−0.8%) | 8.55 | 4.61 | 0.9158 |
| lated4 | 542.0 s | −20.9% | 135.5 | 5.73 (−23.7%) | 41.21 (−6.4%) | 7.01 | 4.55 | 0.8884 |
| r3 | 355.7 s | −48.1% | 118.6 | 5.99 (−20.3%) | 38.10 (−13.5%) | 7.39 | 4.87 | 0.8603 |

`s/评估` 这一列兼作窗口自检：114.2 / 120.2 / **135.5** / 118.6 —— `lated4` 那一档整段慢约 12%，
所以它的 −20.9% 不能和 `auto4` 的 −29.8% 比时长（F30 的教训在渲染上同样成立）。
**形状结论不受影响**：两边评估次数相同、做的功相同，只有质量指标可比。

## 3. 结论

1. **收益 = 少做几次整网评估**，与实现无关：省下的比例 ≈ 1 − 评估数/步数。
   6 步下 4 评估 −29.8%、3 评估 −48.1%；20 步下 `--reuse 2`（11 评估）−45.9% —— 同一条规律。
2. **等成本下形状有代价**：均匀 0,2,4,5 比堆尾 0,3,4,5 的 detail 高 13.8 pt、sat 高 5.6 pt、
   latent cos 0.916 vs 0.888。中间步的速度外推一旦过期，尾部补得再密也换不回结构。
   ⇒ 默认就用自动间隔调度；`H3_REUSE_STEPS` 只在"明确知道某几步必须新鲜"时才手写，且**不要拿它把预算堆到尾部**。
3. 目视（帧 8）：`auto4` 与 base 同锐度同曝光，只是头位相略偏；`r3` 主体仍完整，眼睛偏糊、胸前多几缕拉丝。
   ⇒ 6 步这一档 `--reuse 2` 基本是免费午餐，`--reuse 3` 开始要付质量。
4. 与 F31 并读：固定 keep 稀疏省 14.7% 时 detail 已经 −37%；`--reuse 2` 省 29.8% 时 detail 只 −9.9%。
   **同样"少算"，用时间外推做比砍注意力做性价比高一个数量级** ⇒ 注意力线让位给 reuse。
5. 未控变量，别外推：`detail`（梯度能量）对质量非单调 —— F31 的 2 步 dense detail 6.30 就高于本轮 r3 的 5.99，
   而两者步数不同、σ 轨迹不同。跨步数比较需要另设对照，本轮不下结论。

## 4. 默认建议

- `--steps ≥ 4` 时保持 `--reuse 2` 为默认（README 已如此标注，本轮支持）。
- `--steps 2..3` 下 `--reuse` 无步可跳，实测 +0.6% 噪声 ⇒ 别在低步数下指望它。
- 要更快：`--steps 6 --reuse 3`（3 评估，−48.1%，眼睛略糊）优于继续砍注意力。
- `H3_REUSE_STEPS` 保持"高级覆写"定位，不进预设表。

## 5. 复现

```sh
export H3_DIT_RESIDENT_BLOCKS=6
export H3_CLIPPROJ_DIR=/Volumes/data/.lmstudio/models/Qwen3-VL-4B-Instruct-int8-convrot
export H3_CLIPPROJ_PROJ=/Volumes/data/.lmstudio/models/ClipProj-MiniMax-H3
./h3 -d ~/h3_sys/MiniMax-H3-Convrot -p "A red fox" --width 864 --height 480 --seconds 2 \
  --steps 6 --reuse 2 --ssd-streaming --profile --frames-dir /tmp/reuse/auto4 --latent-out /tmp/reuse/auto4.bin
H3_REUSE_STEPS=0,3,4,5 ... --reuse 2          # 等成本换形状
```
日志与帧：`/tmp/reuse/{base6,auto4,lated4,r3}.{log,bin,mp4}` 及同名目录；分析脚本 `/tmp/reuse/analyze.py`。

## 6. 补（同轮）：把 detail 拆成主体/背景，并目视 lated4 —— 形状结论成立，但量级要改口

上一节只报了全图梯度能量。因为 `lated4` 那一档我其实没看过帧，且"均匀调度基本不伤主体"这句
是从全图数字推的，需要验证：损伤落在主体还是背景？

**做法**（零 GPU 开销，只用已有 `/tmp/reuse/*/frame-*.ppm`）：在 `base6` 上取固定主体掩膜
—— 多帧平均 chroma > 60 **且** R−B > 20，占画面 20.0%；同一掩膜套到四档，水平/垂直梯度分别只在
"两端都落在该区域"的相邻像素对上求均值（掩膜对齐到差分网格，避免边界串区）。
全图那一列与 F32 §2 的 detail 完全吻合（7.514/6.768/5.734/5.986），所以差异只来自区域划分。

| 配置 | 全图 detail | 主体梯度 | 背景梯度 |
|---|---|---|---|
| base6 | 7.514 | 14.847 | 5.723 |
| auto4 | −9.9% | **−19.0%** | −4.2% |
| lated4 | −23.7% | **−37.4%** | −15.0% |
| r3 | −20.3% | **−43.6%** | −5.6% |

三条结论：

1. **损伤集中在主体**。`r3` 的背景几乎没变（−5.6%），主体掉了 43.6% —— 目视看到的"眼睛糊、
   胸前拉丝"就是这一列。速度外推过期时，先塌的是运动中的主体细节，不是背景纹理。
2. **全图 detail 会低估主体损失**（−20.3% 对应主体 −43.6%），因为它被大面积低梯度背景稀释。
   ⇒ 后续 reuse/稀疏 A/B 必须至少报主体区域梯度，全图单指标不够判。
3. **形状结论方向不变、量级改口**：均匀 0,2,4,5 相对堆尾 0,3,4,5，主体 19.0% vs 37.4%、
   背景 4.2% vs 15.0%，两个区域都是均匀更优 ⇒ §3.2 的判据仍然成立。但"均匀 ≈ 免费"要收回：
   均匀那一档主体也掉了 19.0%。

目视（帧 8，`/tmp/reuse/{base6,auto4,lated4,r3}-0008.png`）：`lated4` 的红狐构图完整、曝光正确，
但眼圈/口鼻明显比 `auto4`、`base6` 软，胸前毛偏泥 —— 与主体那一列一致。

复现：`python3 /tmp/reuse/regions.py`（同时重算全图列，作为与 `analyze.py` 的一致性检查）。

# F33（2026-09-21 23:54）：等预算下"少步全新鲜"优于"多步 + reuse 外推" —— 低步数档应砍 steps，不是砍 reuse

## 0. 问题

F32 比的是**同步数下换形状**（哪几步新鲜）。没比过的是另一根轴：**同样 4 次整网评估**，
预算该买"更密的 σ 网格 + 外推"（`--steps 6 --reuse 2`）还是"更粗的网格但每步都真算"
（`--steps 4 --reuse 1`）。这直接决定低步数预设推荐哪一个。

## 1. 设计与附带对照

864×480/2 s/50 层/`--ssd-streaming`/`H3_DIT_RESIDENT_BLOCKS=6`/默认种子，两档**同一时间窗串行**跑。
`--steps 6 --reuse 2` 这一档是 **F32 `auto4` 的完全重复**（同配置同种子），顺带给出跨窗口对照。

两侧做的功相同，由 profile 计数证明：`submissions=180 direct=628 linear=800 attention=200` 完全一致
（外推那两步不跑网络，只跑 Euler 更新，可忽略）。

## 2. 结果

| 配置 | steps | 新鲜评估 | Euler denoise | s/评估 | 全图 detail | 主体梯度 | 背景梯度 | sat | latent cos | 帧间差 | 二阶差 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| base6（参照） | 6 | 6 | 685.2 s | 114.2 | 7.514 | 14.847 | 5.723 | 44.029 | 1.0000 | 4.287 | 6.292 |
| `--steps 4 --reuse 1` | 4 | 4 | 479.2 s | 119.8 | 6.832 (−9.1%) | **13.032 (−12.2%)** | 5.318 (−7.1%) | 43.281 (−1.7%) | **0.9656** | 3.814 | 5.727 |
| `--steps 6 --reuse 2`（本轮） | 6 | 4 | 419.9 s | 105.0 | 6.768 (−9.9%) | **12.027 (−19.0%)** | 5.483 (−4.2%) | 43.657 (−0.8%) | **0.9158** | 4.605 | 6.659 |
| 同上（F32 auto4，另一窗口） | 6 | 4 | 480.7 s | 120.2 | 6.768 (−9.9%) | 12.027 (−19.0%) | 5.483 (−4.2%) | 43.657 (−0.8%) | 0.9158 | — | — |

主体/背景口径与 F32 §6 完全一致（`base6` 上 chroma>60 且 R−B>20 的固定掩膜，占画面 20.0%）。

## 3. 结论

1. **等预算下少步全新鲜更好**：主体梯度 −12.2% vs −19.0%，latent cos 0.9656 vs 0.9158，
   运动也更平滑（帧间差 3.814 vs 4.605，二阶差 5.727 vs 6.659 —— 外推档帧间跳得更多）。
   唯一反向的是背景梯度（−7.1% vs −4.2%）：4 点网格让背景纹理略少，但幅度小。
   ⇒ **整网速度外推的误差，比"少走一步 σ"的离散化误差更贵**。
2. 与 F32 并读，低预算这条轴上排序清楚了：同样 4 次评估，`steps4 密集 −12.2%` <
   `steps6 均匀外推 −19.0%` < `steps6 堆尾外推 −37.4%`；3 次评估的 `--reuse 3` 是 −43.6%。
   ⇒ **20 步档 `--reuse 2` 仍然划算**（那里网格本来就密，外推只是省重复），
   但**低步数档要砍 `--steps` 并保持 `--reuse 1`**，而不是留着 reuse 去换步数。
3. 目视（帧 8，`/tmp/eqcost/{s4dense,s6reuse2}-0008.png`）：两档主体都完整、曝光正确；
   `s4dense` 的口鼻须毛和胸前毛更利落，`s6reuse2` 偏"绘画感"、头位相与背景石头位置不同。与数字一致。

## 4. 计时学（F30 规则的又一次确认，这次是免费对照）

同一配置、同一种子在两个窗口：**质量指标逐位相同**（6.768/12.027/5.483/43.657/0.9158 完全一致），
**时长差 12.6%**（480.7 s vs 419.9 s）。所以渲染质量指标可跨窗口比，wall 不行。
同一窗口内还有**顺序效应**：先跑的 `s4dense` 119.8 s/评估，后跑的 `s6reuse2` 105.0 s/评估。
可见的一部分来自流式路径预热（`cpu-unrotate` 44.8 s → 37.1 s，video VAE 110.1 s → 103.4 s）。
⇒ 本轮**不解读**"479 vs 420 = 密集档更慢"，两侧算子计数相同，成本按构造相等；
以后同窗串行要把待比较的两档**各跑两次换序**，或者干脆只报算子计数。

## 5. 复现

```sh
# 见 /tmp/eqcost/matrix.sh（串行两档，caffeinate -is，同种子）
python3 /tmp/eqcost/analyze.py    # 三列梯度 + sat + latent cos，参照 /tmp/reuse/base6
python3 /tmp/eqcost/flicker.py    # 帧间差 / 二阶差
```
产物：`/tmp/eqcost/{s4dense,s6reuse2}.{log,bin,mp4}` 与同名帧目录、`*-0008.png`。

# F34（2026-09-22 00:25）：F33 在第二个 shape 上复现且更强 —— 但 576×320/1 s 上 reuse 外推直接把主体压塌，且指标抓不到

## 0. 问题

F33 的"等预算下少步密集优于多步外推"只在 864×480/2 s + 一个提示词上成立，而它已经写进预设表。
本轮补外部有效性：换 576×320/1 s（sequence 2293 = 133 上下文 + 12×180）重跑同一组对照。

## 1. 设计与一处口径警告

同窗串行三档：`d6`（`--steps 6 --reuse 1`，参照）、`s4dense`（`--steps 4 --reuse 1`）、
`s6reuse2`（`--steps 6 --reuse 2`），同种子、`--ssd-streaming`、`H3_DIT_RESIDENT_BLOCKS=6`、50 层。

**主体/背景拆分不能跨 shape 照搬**：F32 §6 的掩膜口径（chroma>60 且 R−B>20）在 864 上选出 20.0% 的狐身，
在这次的 576 画面（草地 + 全身小狐）上选出 71.1%，而且"主体"梯度 9.587 **低于**"背景"18.210 ——
掩膜抓到的是草叶。所以本轮主体/背景两列**只作参考**，判定用全图 detail、latent cos 和目视。

## 2. 结果

| 配置 | steps | 新鲜评估 | Euler denoise | s/评估 | 全图 detail | sat | latent cos | 帧间差 | 二阶差 |
|---|---|---|---|---|---|---|---|---|---|
| d6（参照） | 6 | 6 | 193.3 s | 32.2 | 12.214 | 64.246 | 1.0000 | 3.077 | 5.018 |
| `--steps 4 --reuse 1` | 4 | 4 | 130.2 s | 32.5 | **12.133 (−0.7%)** | 60.849 (−5.3%) | **0.9682** | 2.992 | 5.016 |
| `--steps 6 --reuse 2` | 6 | 4 | 130.9 s | 32.7 | **10.907 (−10.7%)** | 63.127 (−1.7%) | **0.9171** | 2.531 | 4.278 |

目视（帧 0/6/10/16 拼贴 `/tmp/eq576/{s4dense,s6reuse2}-tile.png`）：
`s4dense` 四帧都是完整锐利的红狐（与 `d6` 同级）；**`s6reuse2` 每一帧都是压塌的双头狐身**，
不是单帧抖动也不是构图偏移 —— 结构性崩坏。

## 3. 结论

1. **F33 复现，且幅度更大**：等 4 次评估下，4 步密集档的全图 detail 离 6 评估参照只差 −0.7%
   （864 上是 −9.1%），外推档 −10.7%。⇒ "少走一步 σ 的离散化误差 < 整网外推误差"在两个 shape 上都成立。
2. **`--reuse ≥2` 在低步数下是风险开关，不是免费午餐**。F32 说的"6 步下 `--reuse 2` 目视免费"
   只在 864×480/2 s 成立；换 576×320/1 s 同配置直接崩主体。
   低步数档的推荐因此收紧为 `--steps 4 --reuse 1`（两个 shape 都验过）。
3. **指标抓不到崩塌（本轮最重要的方法论收获）**：崩掉的 `s6reuse2` latent cos 0.9171，
   和 864 那次"质量可接受"的 0.9158 几乎一样；sat 只 −1.7%。
   因为 cos/detail 量的是**与参照轨迹的偏离度**，分不出"不同但合理"与"结构错误"。
   ⇒ 渲染质量判定**必须目视**，指标只能当筛选器；跨 shape 时尤其不能只报数字。
4. **计时学补强**：这一窗三档 s/评估 32.2/32.5/32.7（1.5% 内），两个等成本档 wall 130.2 vs 130.9（差 0.5%）。
   ⇒ 反过来印证 F33 里 864 那 14% 的差是窗口/顺序噪声，不是成本差；也说明"窗口是否稳定"可以用
   s/评估 的极差当场判定（极差 >5% 就别报绝对时长）。
5. 未定：`--reuse 2` 在 20 步档是否也在小 shape 上不安全，本轮没测 —— 而 20 步 + `--reuse 2` 正是 README 的默认档。

## 4. 复现

```sh
# /tmp/eq576/matrix.sh（三档串行，caffeinate -is，同种子）
python3 /tmp/eq576/analyze.py
ffmpeg -y -i /tmp/eq576/s6reuse2.mp4 -vf "select='eq(n,0)+eq(n,6)+eq(n,10)+eq(n,16)',tile=2x2" -frames:v 1 /tmp/eq576/tile.png
```
产物：`/tmp/eq576/{d6,s4dense,s6reuse2}.{log,bin,mp4}` 与帧目录、`*-tile.png`。

# F35（2026-09-22 07:10）：20 步 + `--reuse 2` 在 576×320/1 s 上安全 —— 省 46.6%、目视无差，默认档站住

## 0. 问题

F34 让 `--reuse 2` 在 6 步档（4 次评估）于小 shape 上崩了主体，而 README 的默认档是 20 步 + `--reuse 2`。
那一档只在 864×480/2 s 与 512 方形上验过 ⇒ 必须在刚崩过的那个 shape 上复验，否则"默认"两个字没有依据。

## 1. 结果

576×320/1 s、50 层、同种子、同窗串行（`d20` 先跑）。

| 配置 | steps | 新鲜评估 | Euler denoise | s/评估 | 全图 detail | sat | latent cos | 帧间差 | 二阶差 |
|---|---|---|---|---|---|---|---|---|---|
| `--steps 20 --reuse 1` | 20 | 20 | 644.0 s | 32.2 | 12.743 | 70.605 | 1.0000 | 1.921 | 3.277 |
| `--steps 20 --reuse 2` | 20 | 11 | 343.6 s | 31.2 | 12.920 (+1.4%) | 68.502 (−3.0%) | **0.9767** | 2.667 | 4.295 |

时长省 −46.6%，与 F32 §3.1 的规律吻合（1 − 11/20 = 45%）。
窗口自检：s/评估 32.2 vs 31.2（极差 3%），可用。
目视（帧 0/6/10/16 拼贴 `/tmp/eq20/{d20,r20x2}-tile.png`）：两档都是同一只站立的红狐、同一姿态，
锐度与曝光同级，`r20x2` 只是毛色偏冷一点。**没有崩塌、没有鬼影。**

## 2. 结论

1. **`--reuse 2` 的危险不在"步数"，在"新鲜评估次数"**：4 次评估（6 步 + reuse 2）在小 shape 上塌主体；
   11 次评估（20 步 + reuse 2）在同一个 shape 上目视无差。⇒ README 默认档（20 步 + `--reuse 2`）站住。
2. **两个 shape 的边界说法统一了**：低预算要"少步 + `--reuse 1`"（F33/F34），
   20 步这一档"多评估 + reuse"是划算的（F32/F35）。中间地带（8~16 步）没测，暂不外推。
3. 顺带：`d20` 的帧间差 1.921 明显低于 `d6` 的 3.077（同 shape 同步数不同的另一条证据：
   20 步轨迹更平滑），但 reuse 档的帧间差方向在 6 步/20 步两例里相反 ⇒ 再次确认帧间差不能当质量判据。
4. latent cos 这次与目视同向（0.9767 = 安全；对照 F34 的 0.9171 = 崩塌）。
   但**单点同向不构成可用阈值** —— 864 那次 0.9158 目视是可接受的。cos 只能当筛选器，判定仍要目视。

## 3. 复现

```sh
# /tmp/eq20/matrix.sh（两档串行，caffeinate -is，同种子）
python3 /tmp/eq20/analyze.py
ffmpeg -y -i /tmp/eq20/r20x2.mp4 -vf "select='eq(n,0)+eq(n,6)+eq(n,10)+eq(n,16)',tile=2x2" -frames:v 1 /tmp/eq20/tile.png
```
产物：`/tmp/eq20/{d20,r20x2}.{log,bin,mp4}` 与帧目录、`*-tile.png`。

# F36（2026-09-22 08:00）：中间档补测 —— 7 次新鲜评估下两种调度都合格，"密集优于外推"只在 4 次那一档成立

## 0. 问题

"新鲜评估次数"这条轴上只有两个端点：4 次（6 步 + reuse 2，小 shape 塌主体，F34）和 11 次
（20 步 + reuse 2，目视无差，F35）。产品上最常用的是中间那一档，且 F33/F34 得出的
"等成本下少步密集优于多步外推"只在 4 次评估处验过 —— 换一个新鲜次数复验，才知道那条结论是规律还是端点效应。

576×320/1 s、50 层、同种子、串行同窗：`--steps 7 --reuse 1`（7 次评估）vs `--steps 12 --reuse 2`（7 次评估）。

## 1. 等成本成立

两档算子计数完全相同：`submissions=315 direct=1099 linear=1400 attention=350`。
`s12x2` 日志有 `selected reuse schedule has 7 evaluations`。

## 2. 结果

| 配置 | steps | 新鲜评估 | Euler denoise | s/评估 | 全图 detail | sat | 帧间差 | latent cos（对 d20） |
|---|---|---|---|---|---|---|---|---|
| `--steps 20 --reuse 1`（参照） | 20 | 20 | 644.0 s | 32.2 | 12.743 | 70.605 | 1.921 | 1.0000 |
| `--steps 7 --reuse 1` | 7 | 7 | 225.9 s | 32.3 | 12.506 (−1.9%) | 65.224 (−7.6%) | 3.141 | 0.9039 |
| `--steps 12 --reuse 2` | 12 | 7 | 219.6 s | 31.4 | 12.852 (+0.9%) | 66.493 (−5.8%) | 2.692 | 0.8719 |

窗口自检：三档 s/评估 32.2/32.3/31.4（极差 2.8%），可用。两个等成本档 wall 差 −2.8%，在窗口噪声内。
相对 20 步全新鲜：7 次评估省 −65%（644 → 222 s）。

**目视（帧 0/5/11/16 拼贴 `/tmp/mid/{d20,s7dense,s12x2}-tile.png`）：两档都是结构完整的狐狸**，
单头、四腿、黑掌、毛皮锐利，无崩塌无鬼影。两档彼此也同场景同姿态（低头叼食），
但都与 20 步参照（抬头站立）不同 —— 这是新鲜次数少带来的**内容漂移**，不是模糊化。

## 3. 结论

1. **崩塌阈值夹在 4 与 7 之间**：4 次评估 + 外推会塌主体，7 次评估 + 外推目视合格。
   ⇒ README 原先"新鲜次数低于约 8 就把 `--reuse >= 2` 当风险开关"偏保守，据本轮改为"低于 7 未验、4 已证崩"。
2. **"少步密集优于多步外推"不是普适规律，是 4 次评估那一档的端点效应**。7 次评估处两档互有胜负：
   外推档 detail +0.9% vs 密集档 −1.9%、sat −5.8% vs −7.6%（外推略好），
   只有 latent cos 密集档更高（0.9039 vs 0.8719）。而 cos 已被 F34 证明分不出"不同"与"错"
   ⇒ 这一档两者都可用，按 F34 §3 的规矩以目视为准。
3. **帧间差第三次翻方向**：本轮外推档 2.692 < 密集档 3.141，F33 里却是 4.605 > 3.814。
   ⇒ 彻底废弃"帧间差当质量判据"；它只反映运动幅度，与调度好坏无单调关系。
4. **内容漂移是低评估数的真实代价**：7 次评估的两档都换了一个姿态。
   对"要复现某条片子"的用法，这一条比 detail/sat 的百分位更重要。

## 4. 复现

```sh
# /tmp/mid/matrix.sh（两档串行，caffeinate -is，同种子）
python3 /tmp/mid/analyze.py
python3 /tmp/mid/tile.py
```
产物：`/tmp/mid/{s7dense,s12x2}.{log,bin,mp4}` 与帧目录、`*-tile.png`；
参照 latent/帧复用 `/tmp/eq20/d20*`。

# F37（2026-09-22 08:12）：5 与 6 次新鲜评估（外推档）都不崩 —— 但"危险变量是新鲜次数"这条被本轮削弱

## 0. 问题

F36 把崩塌阈值夹在"4 次崩、7 次安全"之间，中间 5~6 次没测；运行时闸的阈值要落在这个区间里，
所以必须补。576×320/1 s、50 层、同种子、`caffeinate -is` 串行同窗：
`--steps 8 --reuse 2`（日志 5 evaluations）与 `--steps 10 --reuse 2`（6 evaluations）。
密集档不重复测：4 次（F34 `s4dense`）与 7 次（F36 `s7dense`）都已证安全，5/6 属内插。
算子计数按评估数线性缩放（5 次 225/785/1000/250，6 次 270/942/1200/300），调度无异常。

## 1. 完整阶梯（全部锚在 `d20`：20 步全新鲜，同 shape 同种子）

| 配置 | steps | 新鲜评估 | Euler denoise | s/评估 | detail | sat | 帧间差 | lat cos | 目视 |
|---|---|---|---|---|---|---|---|---|---|
| `--steps 20 --reuse 1` | 20 | 20 | 644.0 | 32.2 | 12.743 | 70.605 | 1.921 | 1.0000 | 参照 |
| `--steps 6 --reuse 2` | 6 | 4 | 130.9 | 32.7 | −14.4% | −10.6% | 2.531 | 0.8655 | **塌主体（双头）** |
| `--steps 8 --reuse 2` | 8 | 5 | 161.4 | 32.3 | **+10.2%** | −6.7% | 3.482 | 0.8557 | 完整狐狸 |
| `--steps 10 --reuse 2` | 10 | 6 | 193.9 | 32.3 | −0.4% | −3.8% | 3.358 | 0.8709 | 完整狐狸 |
| `--steps 12 --reuse 2` | 12 | 7 | 219.6 | 31.4 | +0.9% | −5.8% | 2.692 | 0.8719 | 完整狐狸 |
| `--steps 20 --reuse 2` | 20 | 11 | 343.6 | 31.2 | +1.4% | −3.0% | 2.667 | 0.9767 | 完整狐狸 |

窗口自检：五档 s/评估 32.7/32.3/32.3/31.4/31.2，极差 4.8%（阈值 5% 内）。
`--steps 6 --reuse 2` 一行的 detail/cos 与 F34 报的 −10.7%/0.9171 不同 —— 因为锚点从 6 步 dense 换成了 20 步，
本轮表内所有比值同锚，可横向比。

目视（帧 0/5/11/16 拼贴 `/tmp/boundary/{s8x2,s10x2}-tile.png`）：两档都是单头、四腿、黑掌、毛皮锐利的完整狐狸，
姿态是低头嗅地/叼物，与 F36 的 7 次档同一类内容，与 20 步参照（抬头站立）不同。

## 2. 结论

1. **崩塌只发生在 4 次评估那一档，5 次就没了** ⇒ 危险区比 F36 猜的窄得多，不是"<7 都危险"。
2. **但"危险变量是新鲜评估次数"这条不能照抄**：本轮三个变量完全同向共线 ——
   步数 6→8→10→12→20 与新鲜次数 4→5→6→7→11 一起变，没有任何一档把它们拆开。
   所以"是次数还是步数在决定崩塌"目前**未证**，F35 §2.1 的表述要降级为"至少与步数相关"。
   可判别的实验（~3 min）：`--steps 8 --reuse 3`（调度 0,3,6,7 = 4 次评估、但网格比 6 步密）。
   若它安全 ⇒ 决定量是**网格粗细/步数**；若它崩 ⇒ 决定量确实是新鲜次数。运行时闸的写法取决于这个答案。
3. **cos 不只是"看不出崩塌"，在 4 次 vs 5 次这一对上是反着的**：崩掉的 0.8655 高于安全的 0.8557。
   4~7 次这一整段 cos 几乎平的（0.856~0.872），11 次才跳到 0.977 ⇒ cos 只能区分"低预算 vs 高预算"，
   在低预算段内部没有任何排序能力。判定必须目视（第四次确认）。
4. **detail 再次证明不可用作单调指标**：5 次评估那档 detail +10.2%（比参照还"高"），
   而它并不比 6/7 次档更好 —— 高出来的部分是草地/杂讯梯度，不是主体细节。
5. 帧间差第四次翻方向（4 次档 2.531 反而低于 5/6 次档的 3.482/3.358）⇒ 该判据彻底废弃（F36 §3.3 再确认）。
6. 收益侧：5 次评估比 20 次省 −74.9%，6 次省 −69.9%（同窗 s/评估一致，可信）。

## 3. 复现

```sh
# /tmp/boundary/matrix.sh（两档串行，caffeinate -is，同种子）
python3 /tmp/boundary/analyze.py   # 打印 4/5/6/7/11/20 全阶梯，锚 d20
python3 /tmp/boundary/tile.py
```
产物：`/tmp/boundary/{s8x2,s10x2}.{log,bin,mp4}` 与帧目录、`*-tile.png`；
其余档位复用 `/tmp/eq576/s6reuse2`、`/tmp/mid/s12x2`、`/tmp/eq20/{d20,r20x2}`。

# F38（2026-09-22 08:42）：判别档 `--steps 8 --reuse 3` —— 4 次新鲜评估换网格仍崩 ⇒ 决定量是新鲜次数

## 0. 设计

F37 留下的问题：阶梯上步数与新鲜次数全程共线（6/4、8/5、10/6、12/7、20/11），
"崩不崩由谁决定"未证。判别实验 = 固定网格 8 步、把新鲜次数压回 4：`--steps 8 --reuse 3`
（调度 0,3,6,7，日志核对 `selected reuse schedule has 4 evaluations`）。
576×320/1 s、50 层、同种子；与 `--steps 6 --reuse 2`（4 次、崩）和 `--steps 8 --reuse 2`（5 次、安全）直接对照。

## 1. 结果

| 配置 | steps | 新鲜评估 | Euler denoise | s/评估 | detail | sat | 帧间差 | lat cos | 目视 |
|---|---|---|---|---|---|---|---|---|---|
| `--steps 20 --reuse 1` | 20 | 20 | 644.0 | 32.2 | 12.743 | 70.605 | 1.921 | 1.0000 | 参照 |
| `--steps 6 --reuse 2` | 6 | 4 | 130.9 | 32.7 | −14.4% | −10.6% | 2.531 | 0.8655 | 塌主体（双头） |
| `--steps 8 --reuse 3` | 8 | 4 | 112.1 | 28.0 | **−18.7%** | **−17.2%** | 1.530 | **0.8181** | **面部畸形 + 全身条带状伪影** |
| `--steps 8 --reuse 2` | 8 | 5 | 161.4 | 32.3 | +10.2% | −6.7% | 3.482 | 0.8557 | 完整狐狸 |
| `--steps 10 --reuse 2` | 10 | 6 | 193.9 | 32.3 | −0.4% | −3.8% | 3.358 | 0.8709 | 完整狐狸 |
| `--steps 12 --reuse 2` | 12 | 7 | 219.6 | 31.4 | +0.9% | −5.8% | 2.692 | 0.8719 | 完整狐狸 |
| `--steps 20 --reuse 2` | 20 | 11 | 343.6 | 31.2 | +1.4% | −3.0% | 2.667 | 0.9767 | 完整狐狸 |

目视（`/tmp/discrim/s8x3-tile.png`，帧 0/5/11/16）：趴着的躯体上覆盖深色的横向条带（编织感），
脸部塌陷、两眼变成黑色块，四帧几乎同一姿势。这是**结构性失败**，不是"偏软"。
计时注记：本档 s/评估 28.0，比同窗其它档（31~33）低 13% —— 单档墙钟不可跨窗比，
本轮结论只依赖目视与比值型指标（质量指标是确定性的）。

## 2. 结论

1. **决定量是"新鲜评估次数"，不是步数/网格粗细**：同样 4 次评估，6 步网格崩（双头）、
   8 步网格也崩（面部畸形 + 条带），且 8 步那档三项指标全部更差（cos 0.8181 为全阶梯最低、
   sat −17.2%）。⇒ F37 §2.2 提出的共线疑问**当场关掉**，F35 §2.1 的原始表述恢复成立（这次是有判别实验的版本）。
2. **边界收紧为"4 次崩、5 次起安全"**：5/6/7/11 四个新鲜次数点（含两种网格）目视全部合格。
   ⇒ 运行时闸的判据可以定了：`reuse_interval >= 2` 且算出的新鲜次数 < 5 时降级为 `--reuse 1` 或直接报错。
   该条件覆盖的实际组合：`--reuse 2` 的 steps 4..7（3~4 次）、`--reuse 3` 的 steps 6..10（3~4 次）。
3. **指标这次与目视同向，但不能当阈值**：cos 0.8181（崩）< 0.8557（安全）是本轮唯一的正向案例，
   而 F37 已经出现过 0.8655（崩）> 0.8557（安全）的反例 ⇒ 同向只是巧合级别，判定仍必须看帧。
4. **帧间差第五次失效**：崩溃档给出全阶梯最低值 1.530（低于 20 步参照 1.921），
   原因只是四帧几乎静止。⇒ 该指标已彻底废弃，本轮再次确认。

## 3. 复现

```sh
# /tmp/discrim/matrix.sh（caffeinate -is，同种子）
python3 /tmp/discrim/analyze.py   # 打印 4(6步)/4(8步)/5/6/7/11 全阶梯
python3 /tmp/discrim/tile.py
```
产物：`/tmp/discrim/s8x3.{log,bin,mp4}` 与帧目录、`s8x3-tile.png`；
其余档位复用 `/tmp/eq576/s6reuse2`、`/tmp/boundary/{s8x2,s10x2}`、`/tmp/mid/s12x2`、`/tmp/eq20/{d20,r20x2}`。

---

# F39 打包粒度装不下有用的稀疏性：可打包的 top-k 与固定时间窗在等 keep 下打平，而赢的形态按每调用开销算净负（2026-09-22）

## 0. 把"要不要做 top-k"变成一次离线测量

- 新增 `H3_DUMP_ATTN_LAYERS`（`h3_dit.c`，默认 off）：抓 norm/RoPE **之后**、SDPA 真正消费的
  Q/K/V，整条序列全量行（576×320/1 s 就是 2293×7168 bf16，每层每字段一次）。
  有了 Q/K/V，所有选择器都在离线比，一帧都不用重渲染。
- 抓取：`--steps 2 --reuse 1`，只落第 1 次评估；层 0/12/25/38/49。
- **harness 闸**：离线稠密重建 vs 引擎自身的注意力输出（`out.NN.bin`，8 行 × 7168）
  五层全部 cos = 1.0000 ⇒ bf16 解码、`(row, head, dim)` 布局、softmax·V 通路全对。
- **与 F31 的口径差**：同一 harness 重算固定时间窗 window0/1/2 = 0.791/0.879/0.919，
  F31 在引擎内测得 0.729/0.842/0.892。差 0.03~0.06 的来源是行子集：F31 只 dump 前 512 行
  = 帧 0..2，而边界帧的时间窗更窄（r=1 时帧 0 只有 2 个块），它的样本偏向"看得更少"的块。
  ⇒ 绝对值不可跨 harness 比，**本文所有对比只在离线 harness 内做**。

## 1. 运行时的打包单位是 token 行，因此"可选的形式"只有三种

rect SDPA 一次打包是一个 token 的全部 7168 个 key 维，且同一查询块的 180 行共用那份 scratch
⇒ 选择必须是 **跨头共享 × 查询块内共享**。逐行逐头的 top-k 数学上最好，但表达不出来。

L0 video 行 cos（括号内 p10），keep 为打包比例：

| 形式 | 可打包 | 13.7% | 21.5% | 29.4% | 37.2% | 52.9% |
|---|---|---|---|---|---|---|
| 逐行逐头 oracle | ✗ | 0.854(.669) | 0.921(.815) | 0.952(.888) | 0.969(.930) | 0.987(.972) |
| 逐行逐头 pooled | ✗ | 0.831(.596) | 0.903(.764) | 0.937(.848) | 0.958(.900) | 0.980(.956) |
| oraclehs（块共享、跨头） | ✓ | 0.791(.468) | 0.849(.627) | 0.885(.721) | 0.914(.788) | 0.957(.910) |
| poolhs（同上，可负担打分） | ✓ | 0.791(.468) | 0.850(.625) | 0.885(.718) | 0.913(.788) | 0.945(.881) |
| glob（整层一份） | ✓ 最便宜 | 0.719(.321) | 0.791(.463) | 0.837(.560) | 0.869(.642) | 0.917(.786) |
| 固定时间窗（同 harness） | ✓ 已接线 | 0.791(.468) | — | 0.879@28.0(.699) | — | 0.919@41.1(.804) |

三点读法：
1. **poolhs ≈ oraclehs**（最大差 0.036，出现在 L38 的 13.7% 档；L0 的 13.7%/29.4% 两档三位小数相同）
   ⇒ 可打包形式里**打分器不是瓶颈**：oracle 已经知道每块真正收到多少注意力，也照样到不了判据线。
   共享本身才是瓶颈，评分侧（学习打分、离线逐层 keep 表）不用再投。
2. F31 §5 的判据（keep ≤30% 且 L0 cos ~0.95+）只有**不可打包**的逐行逐头形式勉强够
   （oracle 0.952，且 p10 只 0.888）。可打包最好形式在 29.4% 是 0.885。
3. 与固定窗等 keep 对照，五层均值：**13.7% 档 top-k 反而差 0.014**（poolhs 0.804 vs window 0.818），
   29.4% 档只**好 0.006**（0.917 vs 0.911）——而 window1 那一行实际 keep 是 28.0%，
   即 top-k 多花 1.4 pt keep 才换来这 0.006。L38 上 top-k 全程落后（0.881 vs 0.892 @~28%、
   0.926 vs 0.943 @~41%）。⇒ 在唯一能打包的粒度上，"看对的块"对"看近的块"**没有净收益**。

## 2. 唯一的逃生口是"多份 scratch"，而它的价格已被实测过

G = 把 56 个头分成 G 组，每组打包自己的 top-k 并集（G=1 即 §1 的共享形式，G=56 即逐头）。
L0（keep 为实际打包比例）：

| k | G=1 | G=2 | G=4 | G=8 | G=16 | G=56 |
|---|---|---|---|---|---|---|
| 2 | .999/97.4% | .997/92.8% | .989/80.4% | .976/64.0% | **.952/42.0%** | .892/21.5% |
| 3 | 1.000/100% | .999/98.4% | .997/92.8% | .988/79.1% | .973/55.9% | .927/29.4% |

L38 更贵：G=16 要到 0.932@38.4%、0.961@51.6%（同 keep 下比 L0 差，且固定窗在 L38 本就更强）。

- **56 个头的 top-2 并集覆盖 97.4% 的 token**（G=1，k=2）⇒ 头部偏好近乎不相交。
  这不是"选得不好"，是稀疏性不在 head 共享的粒度上。
- 判据换算：本 harness 里固定窗到 0.95 需 keep ~64%，poolhs 需 ~55%（差 9 pt，两者都是 L0 内插）；
  最激进的 G=16 需 ~42%（L0）但 L38 需 ~48%。按 `省时 ≈ SDPA 占比 × Δkeep`
  （864×480 SDPA 占块时长 20.5%）⇒ 收益 **1.9%~4.4% 时长**。
- 代价：调用数 ×G。F31 实测 1800 次 rect 调用 ≈ 3.0~3.2 s CPU 编码（≈1.8 ms/次）
  ⇒ G=8 约 +24 s、G=16 约 +48 s，而整段 Euler denoise 才 165~210 s。
  **净负 4~7 倍**，且这还没算 G 份 scratch 的显存/带宽。
- 要把编码搬到 GPU 侧 gather 属于新建通路，不是"换个选择器"，超出这条线的范围。

## 3. 结论

- 固定 keep 时间窗（F31）死于"不够省"；本轮把它剩下的唯一变体——"看对的块"——也关掉，死因更根本：
  **有用的稀疏性长在 head 维（和行维）上，而 token 打包天生 head 共享**；在共享粒度上 top-k ≈ 时间窗（13.7% 档还更差）。
- 判据现在可量化、可复用：同 harness 里 L0 到 cos 0.95 所需的 keep —— 固定窗 ~64%、poolhs ~55%、
  G=16 ~42%（但最难的 L38 要 ~48%）。任何新方案先报这个数，再谈渲染。
- 不再投的方向：更聪明的离线打分器、逐层 keep 表、per-row 稀疏（打包表达不出来）。
- 留下的可复用件：`H3_DUMP_ATTN_LAYERS`（默认 off）+ 离线 harness（`/tmp/topk/study*.py`）。
  若将来真有 GPU 侧 per-head gather，可用同一份 dump 直接重价，不必再渲染。
  接线本身（`H3_SPARSE_ATTN`）保持默认 off，F31 §2 的位等价闸仍然有效。

## 4. 复现

```sh
# 1) 抓 Q/K/V（5 层，每层 3×32.9 MB）
/tmp/topk/dump.sh                       # H3_DUMP_ATTN_LAYERS=0,12,25,38,49 + H3_DUMP_ACT
# 2) 逐行逐头三选择器 + 与引擎 out.NN 的位对拍（cos 1.0000）
python3 /tmp/topk/study.py     > /tmp/topk/result.txt
# 3) 可打包形式（oraclehs/poolhs/glob）
python3 /tmp/topk/study2.py    > /tmp/topk/result2.txt
# 4) 同 harness 的固定窗 + 全体头顶并集
python3 /tmp/topk/study3.py    > /tmp/topk/result3.txt
# 5) G 组曲线（keep 为实际打包比例）
python3 /tmp/topk/study4.py    > /tmp/topk/result4.txt
# 三个脚本的几何常量都硬编码为 576×320/1 s：12 帧 × 180 行 + 133 条件行 = 2293
```

---

# F40 AdaLN 调制缓存：checkpoint 只带调制不带权重，省 24.29 GiB；且一份缓存覆盖三种条件模式（2026-09-26）

## 0. 问题：24.29 GiB 的差额能不能从 checkpoint 里拿掉

上一轮把一个 mere-run MLX affine q4 的 checkpoint 转成 h3.c 可加载的 4-bit grouped 格式，
结果是 35.93 GiB，而 mere-run 自己那份只有 10.55 GiB。逐类目分桶后差额定位得很干净：
**51 个 `adaln_proj.linear` 矩阵占 24.288 GiB**，mere-run 用
`cache_covered_weights_omitted: true` 把它们整个省掉。

能不能省，取决于这些权重**被读几次**。答案是**一次**：
`h3_dit_schedule_precompute` 在 denoise 之前就把所有 step 的 AdaLN 调制一次算完，
逐块投影、算完立刻 `free_tensor(&weight)`。所以 checkpoint 完全可以只存那份**产出**
（每行 `50 × 96768 × 2 B + 10752 × 2 B + 4 B = 9,698,308 B = 9.25 MiB`），不存权重。

这也解释了为什么换 dtype 省不掉：F16 与 BF16 同为 16 bit/param（而 `weight_bf16_any` 也只收
BF16/F16/F32），换成输出缓存才是数量级的差别 —— 24.29 GiB 的权重对应 83.2 / 157.2 / 379.2 MiB
的三档缓存（steps 4/8/20），即 158× 上下。

## 1. 关键发现：三种条件模式的行数是嵌套的，所以一份缓存能全覆盖

行数 = 每步 1 行（video sigma == audio sigma）或 2 行，**第 0 步恒为相等**，所以无条件下
`time_rows = 2*steps - 1`。实测 `--steps 2` → 3 行（dump 29,094,948 B）、`--steps 8` → 15 行
（dump 145,474,644 B），两次都与 `24 + rows*9,698,308` 精确吻合。

条件行**追加在 step 行之后**，顺序固定 visual 再 audio（`h3_dit_schedule.c` 的
`visual_condition_row = count++` 在前）。于是三种模式的行数是嵌套的：

| 模式 | time_rows | 条件行 |
|---|---|---|
| text-to-video | `2*steps - 1` | 无 |
| 首/末帧 I2V（`!first`/`!last`） | `2*steps` | visual |
| 参考图 Ref2VA（`!ref-image`） | `2*steps + 1` | visual + audio |

`visual_condition` 来自 `layout->img_cond_rows != 0`（`H3_SEG_COND` + `H3_SEG_REF_IMAGE`），
`audio_condition` 来自 `H3_SEG_REF_AUDIO`（只有 `!ref-image` 会带）。

因为短模式的 `times[]` 恰好是长模式的**前缀**，导出时把两个条件行都算出来，一份缓存就能
服务全部三种模式（短模式按前缀命中）。所以 `H3_DIT_ADALN_CACHE_DUMP` 现在**同时强制
`visual_condition = audio_condition = 1`**。

**这里踩过一个坑**：最初只强制 visual。这样 Ref2VA 需要 `2s+1` 行而缓存只有 `2s` 行，会被
拒绝并提示"重建缓存"—— 而重建出来的还是 `2s` 行，**这个提示永远修不好问题**。同时强制两个
条件行后缓存是 `2s+1` 行，三种模式分别按前缀/精确命中，代价只有两行用不上的 ~20 MiB。

**唯一不覆盖的模式**：纯音频参考（`H3_LAYOUT_REF_AUDIO` 无图）。它只有 1 个条件行，会落在
visual 行的位置上，于是 memcmp 在那一行失败 → 明确拒绝而不是套错。该模式 CLI/REPL 都构造
不出来（没有 `!ref-audio`，CLI 也没有引用参数），只有 C API 可达。

## 2. 等价性是逐字节的，连 mp4 容器都相同

契约：`adaln_cache_times_s{steps}` F32 `[rows]` 是**校验键**，加载时与本次 sigma schedule 算出的
`times[]` 逐位 memcmp、要求前缀匹配；`blocks.N.adaln_cache_s{steps}` BF16 `[rows, 96768]`、
`final_layer.adaln_cache_s{steps}` BF16 `[rows, 10752]` 是调制本体。
`blocks.0.adaln_cache_s{steps}` 存在即选中缓存路径，无需开关。

dump 字节数（`32 + rows*9,698,308`）三档逐字节命中预测：4 步 87,284,804 / 8 步 164,871,268 /
20 步 397,630,660。裁剪后目录 12.25 GiB（11.65 权重 + 619.7 MiB 三个缓存），`--verify` PASS。

A/B（A = 全权重 `h3c-q4-native`，B = 缓存模型，同 prompt/256×256/22 帧/seed 42）：

| steps | A | B | latent md5 | mp4 md5 |
|---|---|---|---|---|
| 4 | 151 s | 119 s | `6e9c2d8786cea097ddf9bdd45e5593bd` 同 | 同 |
| 8 | 250 s | 219 s | `a7f85c8bfd6b8fe0282c4084adbad9c5` 同 | 同 |
| 20 | 556 s | 501 s | `dc8c5d41f9779fc0b77e8172cf4603bd` 同 | 同 |

`relRMS=0.000000 cos=1.000000 max|d|=0.000000e+00`。**mp4 容器也逐字节相同** ⇒ 整条链路
（含 VAE 解码）无任何差异，省下的就是那 24.29 GiB 不再被读。

steps=8 的 md5 与"改成按步数分键**之前**"那版完全一致，反证这次重构与"多算两个条件行"
没有改变数值 —— 因为条件行只追加在尾部，step 行的索引与取值不受影响。

## 3. 结论

- **24.29 GiB 是可省的**，且省法不是换 dtype，而是让 checkpoint 只带 AdaLN 的**输出**。
  缓存成本是每行 9.25 MiB，三档合计 619.7 MiB。
- **一份缓存能覆盖三种条件模式**，因为条件行追加在尾部、三种模式的行数嵌套
  （`2s-1` ⊂ `2s` ⊂ `2s+1`）。导出时把两个条件行都算出来即可。
- **缓存按 `--steps` 分键**（`adaln_cache_s{steps}.safetensors`），同目录多档并存，
  换步数只改 `--steps`。加载侧不需要任何开关，`h3_st_inventory_dir` 天然认得新文件。
- **校验必须是逐位前缀匹配**：缓存套错步数会"看起来完全健康"地在每一步套错调制，
  所以宁可硬报错。
- **`!layers N` 块剪枝在缓存模式下依然精确**：`h3_dit_schedule_gate_scores` 读的是
  `schedule->blocks[block]` 的 slot 2/5，而非缓存路径下 `h3_gpu_linear_bf16` 也把投影结果
  写进同一个数组 —— 两条路装的都是**调制**，缓存逐位复现它。（早先记的"缓存模式下 gate 排名
  会坏"是误读，已纠正。）
- **与 LoRA 互斥**：`merge_adaln_loras` 要合并进缓存已丢弃的权重，加载时直接拒绝。

## 4. 复现

```sh
# 1) 从还带权重的 checkpoint 导出（每个步数一次；导出运行本身是完整运行，可当 A 侧基线）
for S in 4 8 20; do
  H3_DIT_ADALN_CACHE_DUMP=/tmp/h3ab/v2/adaln_s$S.raw \
    ./h3 -d /Volumes/data/MODELS/h3c-q4-native --ssd-streaming \
    -p "A red fox walking through snow" --width 256 --height 256 \
    --frames 22 --steps $S --seed 42 --latent-out /tmp/h3ab/v2/s${S}A_latent.bin \
    -o /tmp/h3ab/v2/s${S}A.mp4
done
# 2) 裁剪 + 装三份缓存（写新目录，源模型不动），再 --verify
python fastvideo_qad/scripts/export_h3_adaln_cache.py \
  --dump /tmp/h3ab/v2/adaln_s4.raw --dump /tmp/h3ab/v2/adaln_s8.raw \
  --dump /tmp/h3ab/v2/adaln_s20.raw \
  --model /Volumes/data/MODELS/h3c-q4-native/FL2VA/transformer \
  --trim-to /Volumes/data/MODELS/h3c-q4-adalncache/FL2VA/transformer
python fastvideo_qad/scripts/export_h3_adaln_cache.py --dump ... (同上三个) \
  --model /Volumes/data/MODELS/h3c-q4-adalncache/FL2VA/transformer --verify
# 3) B 侧（缓存模型）
# 前置条件（F45 补记，F47 定稿）：这棵树没有 FL2VA/text_encoder，必须先设 H3_CLIPPROJ_DIR 与
# H3_CLIPPROJ_PROJ 两个变量，否则在 h3_load_dir 就 fatal —— 本节最初抄录时漏了这句，
# 2026-10-04 照抄因此跑不通（F43 §6）。
# F47 实测确定：上表那几个 latent md5 对应的 4B 是 int8-convrot 那一版，不是库旧的 BF16 默认
#   H3_CLIPPROJ_DIR=/Volumes/data/.lmstudio/models/Qwen3-VL-4B-Instruct-int8-convrot
#   H3_CLIPPROJ_PROJ=/Volumes/data/.lmstudio/models/ClipProj-MiniMax-H3
# 换成 BF16 的 Qwen3-VL-4B-Instruct 会得到不同的 latent（256²/22f/4步/seed42：
# 6e9c2d8786cea097ddf9bdd45e5593bd 对 a63e27a4c42621c6a7388d56c9942961）。
for S in 4 8 20; do
  ./h3 -d /Volumes/data/MODELS/h3c-q4-adalncache --ssd-streaming \
    -p "A red fox walking through snow" --width 256 --height 256 \
    --frames 22 --steps $S --seed 42 --latent-out /tmp/h3ab/v2/s${S}B_latent.bin \
    -o /tmp/h3ab/v2/s${S}B.mp4
done
# 4) A/B
python /tmp/h3ab/compare_latent.py /tmp/h3ab/v2/s8A_latent.bin /tmp/h3ab/v2/s8B_latent.bin
md5 -q /tmp/h3ab/v2/s8A_latent.bin /tmp/h3ab/v2/s8B_latent.bin   # 应相同
# 5) 负向对照（没有缓存的步数）
./h3 -d /Volumes/data/MODELS/h3c-q4-adalncache --ssd-streaming ... --steps 6 -o /tmp/x.mp4
#   → exit 1, "this checkpoint ships AdaLN caches but none for 6 steps, and it has no
#      adaln_proj weights to fall back on; export a cache at these steps with
#      H3_DIT_ADALN_CACHE_DUMP"
```

环境：必须从仓库根运行（`h3_shaders.metal` 按 CWD 解析）、`H3_CLIPPROJ_DIR` /
`H3_CLIPPROJ_PROJ`、16 GiB 机器上加 `H3_DIT_RESIDENT_BLOCKS=4`。
dump 字节数可直接验算：`32 + rows*9,698,308`，`rows = 2*steps + 1`。

---

# F41（2026-10-01）：Strata 对照清单的前提核对 —— "流式存 BF16"半错、"uncached 全放弃"方向相反、"每调用一个 fd"成立但不是 4 KiB 病理

用户给了一份 Strata（LLM offload）与 h3c 的对照优化清单，末尾要求"参考上面的优化"。清单里
三条是对 h3c 现状的**事实断言**，动手前先逐条核到代码。本轮不改行为、无 GPU 轮次。

## 1. 断言①"流式档位盘上存 BF16；Strata 的收益来自权重不反量化" —— 对 ConvRot int8 checkpoint 不成立

`read_stream_layer()` 按张量是否 ConvRot 编码分两条读法：

- **ConvRot**：`convrot_read_weight()` 读 **int8** + 每组 scale（h3_dit.c:1649，函数体
  h3_dit.c:773-786）→ CPU Walsh-Hadamard 反旋转 → BF16 写进槽 → 主线程 `requant_stream_slot()`
  再把槽重量化成 int8 供 GPU 直算（h3_dit.c:1510-1546）。
- **非 ConvRot（BF16 checkpoint）**：`h3_gpu_tensor_stream_file_bf16()` 整块读 BF16
  （h3_dit.c:1663）。

也就是说读法由 **checkpoint 决定**，而本机近期所有轮次用的
`/Users/jay/h3_sys/MiniMax-H3-Convrot` 走的就是 int8 读 —— "盘上存 BF16"只描述 BF16 checkpoint。
int8 计算也没因流式而关：`dit->int8_mlp / int8_qkv / int8_attention_out` 的设置与流式无关
（h3_dit.c:3370-3377），注释 3365-3369 明确写着 "Previously the two were
force-mutually-exclusive"。

清单把 `--use-int8-row-fc2` 和"int8"混为一谈：`h3.c:918` 拒绝的只是那个 **M5 专属 FC2 kernel
变体**（h3.c:927 要求 Metal 4），不是 int8 整体。

真正还没吃到的那部分，措辞应换成：ConvRot 流式绕了 **int8 → BF16 → int8** 一圈（反旋转破坏了
原 scale 网格，所以必须重量化），代价是 CPU 反旋转 + 一次主线程串行 requant。README:827-831 实测
33.6 s 的流里读 17.6 s、反旋转 16.0 s ⇒ 反旋转占生产者忙时约 48%，这才是要定价的对象。

顺带两处**文档/打印与代码不一致**（免费可修）：

- README:142 "This uses the original BF16 checkpoint without conversion or quantization" 与
  README:793-797 "Two complete BF16 matrix slots alternate … Darwin uncached reads avoid
  retaining a second copy" —— 和 README:826-831 自己写的 int8 读法并存，读者无法判断默认路径。
- `h3_dit.c:5673` 的 `H3_PROFILE` 行不分路径都印 "BF16 SSD stream"，而 `job->bytes` 在 int8
  分支累加的是 int8 字节（h3_dit.c:1658）⇒ 那行 GiB/s 是 int8 流量贴了 BF16 标签。

## 2. 断言②"uncached 读 = 放弃保留，retention is the whole game" —— 对 BF16 路径成立，对本机在用的 int8 路径方向相反

`F_NOCACHE` 全仓只有一处（h3_gpu.m:823），只有 `h3_gpu_tensor_stream_file_bf16()` 传
`uncached=1`（h3_gpu.m:857-862）。ConvRot 的 int8 读（h3_dit.c:775 `open(path, O_RDONLY)`）和
`h3_weights.c:235` 的 `pread_bytes()` 都**没有** F_NOCACHE ⇒ 这些读照样填页缓存，保留与否交给
OS，我们既没控制也没测。所以"h3c 放弃了保留"这句话在 int8 路径上是错的，正确的是**放弃得不完
全、而且没记账**。

已有的分层保留也比清单假设的多一层：`load_core()` 的自适应部分常驻（h3_dit.c:2756-2853）按内存
余量把**前 N 个 active block** 反旋转一次、跨所有 step 复用，`H3_DIT_RESIDENT_BLOCKS` 可钉数 ——
但选谁常驻按**块序号**而非实测成本，且与页缓存无关。

"缺'这次读是否本可以命中'这一维"这条**成立**：`H3_PROFILE` 只有 unhidden wait / pread /
cpu-unrotate 三个数（h3_dit.c:5670-5680）。macOS 没有现成的 per-read 命中计数，但有免费代理：
同一批权重每个 step 重读一次，**同一 block 第 1 次与第 20 次的 pread 带宽比**就是命中率的直接读数。

## 3. 断言③"每张量 open/close 一个 fd" —— 成立，但请求尺寸不是 Strata #230 那种 4 KiB

两处确认：`h3_weights.c:234-247`（14 个调用点）、`h3_dit.c:773-786`。粒度实际比"每调用一个 fd"
更细一档：开快速路径后按 `H3_STREAM_CHUNK_ROWS = 1024` 切块（h3_dit.c:1485、2033），
1024 行 × 5376 列 int8 ≈ 5.25 MB/次；默认串行路径整矩阵一次。Strata 0.03 → 1.49 → 3.25 GiB/s
那段收益来自把 2048 个 4 KiB 读合成大块，我们本来就是 MB 级 ⇒ 同样手法预期上限小得多，值得测但
别按那个倍数预期。

## 4. 清单里两条按现有红线不该排期

- **"ANE 编译产物当专家层管理"**：ANE 线已按用户决定关闭，代码保留默认 off，删除需单独批准 ——
  不在"参考上面优化"里顺手重开。
- **"无损优先：验证者就是大模型本身"**：这条被我们自己的证据挡回来。reuse / token-reduction 的
  失败模式是主体崩塌，而 F35-F38 的教训是**同族偏离度指标对崩塌会反向**（崩了的渲染 cosine
  0.866、结构完好的 0.856；另一组崩的 0.9171 ≈ 合格的 0.9158）。用 rel-L2/cosine 自验证会把
  "要防的那类错"判成通过 ⇒ 真要做，先用已确认崩塌的 4-次新鲜评估档当负样本，检验候选指标能否
  把它和 5-次档分开，再谈接线。

## 5. 工程纪律三条的对齐情况

env 臂 + 保留旧路是 h3c 惯例（`H3_DIT_STREAM_WORKERS`、`H3_REUSE_STEPS`、`H3_DISABLE_*`），无需
新学；**逐块 FNV-1a 指纹没做过**，且正对我们痛点 —— F39 §0 的 parity 靠整层 cos、F40 靠 latent
md5 全同，都定位不到"从第几块开始动"；**看门狗没做过**，而 `h3_memory_plan.c:43-46` 的注释自己
承认静态 80%/85% 上限（47、51 行）判错时长 clip 能把物理内存打穿到系统 panic。

## 6. 复现命令

```bash
# 断言①③：读法分支、fd 生命周期、唯一的 uncached 点
rg -n 'convrot_read_weight|h3_gpu_tensor_stream_file_bf16|requant_stream_slot' h3_dit.c
rg -n 'open\(path, O_RDONLY\)|F_NOCACHE' h3_dit.c h3_weights.c h3_gpu.m
# 断言②：madvise 只存在于注释里（h3.c:2318 明确写了不用 DONTNEED）
rg -n 'F_NOCACHE|madvise|MADV_' --glob '*.{c,h,m}'
# 断言①的文档自相矛盾三处 + 静态启发器两处
sed -n '138,146p;786,800p;822,836p' README.md
sed -n '20,55p' h3_memory_plan.c
```
---

# F42（2026-10-04）：F40 AdaLN 缓存那轮的代码评审 —— 8 处可改，其中 1 处一刀切、1 处建议不可执行、2 处实测踩到

评审对象是工作区**未提交**的 F40 那轮：`h3_dit_schedule.{c,h}`（+270/-25）、
`fastvideo_qad/scripts/export_h3_adaln_cache.py`（451 行，新）、
`export_h3_q4_native.py`（新）、README 新小节（+73）。本轮不改行为、不跑 GPU；凡能本机证伪的
（getenv 语义、导出器参数组合、trim 的丢文件、safetensors 对齐）全部跑了，现场在 `/tmp/f42`。

## 0. 先记做对的部分（下轮别把这些当问题）

- `prepare_rows` 拆成 `prepare_times` + `prepare_features` 是**真简化而非搬家**：缓存路径因此能在
  读任何权重、碰任何 GPU 之前先校验 sigma 前缀；`count > SIZE_MAX / feature_dim` 也跟着搬到真正
  做那次分配的地方（h3_dit_schedule.c:277）。
- 该 TU 的编译告警从 **11 条降到 1 条**。HEAD 版有 4 处 `-Wsometimes-uninitialized`（`time` 被判
  可能在 if 条件里未初始化）与 1 处 `-Wshadow`；本轮把 `time` 提到函数入口并初始化（:599-601 的
  注释正是为此）后全消。剩下唯一一条在 :419（`(size_t)rows * dim` 传给 uint32_t 形参），
  **早于本轮且本轮未触碰该函数** —— 用 `git show HEAD:h3_dit_schedule.c` 单独编译复核过。
- 前缀校验顺带把"纯音频参考"这个唯一不覆盖的模式**自然拒掉**了：音频条件行落在视觉行的下标上
  （0.999 vs 1.000），memcmp 不等即拒，不需要特例分支。
- 失败路径不重复释放：成功路径在 dump 之后才 `free(times)`（:831），`goto failed` 只可能发生在
  它之前；缓存分支提前 return 前也自己 free（:707）。

## 1. P1：LoRA 闸一刀切，误杀"只有注意力目标"的适配器（h3_dit_schedule.c:688）

`if (lora_count > 0) fail(...)` 拒掉**任何**适配器。但计算路径里 `merge_adaln_loras` 对不匹配的
适配器是**警告跳过**：`h3_lora_matches`（h3_lora.c:154-168）按
`{prefix}.{target}.lora_A.{adapter}.weight` 查键，查不到返回 0，于是 :446-451 印一句 warning 继续。
所以"适配器不含 AdaLN 目标"从来不是错误。

按 Phase 1 实测的 VDN adapter 事实：`.default` 只有 `attn.orig.{to_q,to_k,to_v,to_out.0}` ⇒ 无 AdaLN
目标 ⇒ 与缓存本可共存；`.turbo` 另带 `norm_out.linear.lora_*.turbo`（final 层 AdaLN）⇒ 真冲突。
现在两种一起被拒。

**这不是隐藏 bug**：README:1120 明写 "LoRA adapters are refused"，头注释也写了，属于已文档化的保守
设计。代价是缓存档用不了 VDN default adapter。收窄的障碍是**信息不足且诚实**：缓存档已经没有
`adaln_proj` 探针，`time_dim` 只能停在默认 2688，无法判断 8-wide 剪枝档的 AdaLN 适配器是否冲突。
要真做得先给缓存加宽度（导出侧 Python 读源 checkpoint 的 `blocks.0.adaln_proj.linear.weight`
shape[1]，写成一个标量键或塞进 dump 头字段），C 侧再按它跑 `h3_lora_matches`。

## 2. P2：空串 env 被当"已设置"，改了行布局却什么都不导出（实测）

三处判据不一致：:679 与 :699 用 `getenv(...)` 非 NULL，:823 用 `dump_path && *dump_path`。
`/tmp/f42/getenv_probe.c` 实测三档：

```
unset          -> 不强制行、不导出
H3_..=（空串） -> 强制 visual+audio（多 2 行 ≈18.5 MiB GPU + 两次多余投影），但不写 dump
H3_=…/x.raw    -> 强制 + 导出
```

即在缓存档上还会印一句假警告 "H3_DIT_ADALN_CACHE_DUMP ignored; this checkpoint already ships an
AdaLN cache"（:699-702）。修法：函数入口读一次 `const char *dump_path`，三处统一
`dump_path && *dump_path`，顺带去掉 3 次重复 getenv。

## 3. P3：纯音频参考的报错建议不可执行（:494-499）

memcmp 分支的话术是 `... rebuild the cache at these steps`。但导出路径**无条件**把 visual 行排在
前面（:679-682），所以无论哪个步数重建，那一行都是 0.999，永远等不上这条 run 需要的 1.000。
这正是本轮自己的注释（:676-678）批评过的那类错误 —— "with 'rebuild the cache' advice that
rebuilding could never satisfy"。修法：缓存分支加前置判定 `audio_condition && !visual_condition`
⇒ 直说"缓存不覆盖无图音频参考（H3_LAYOUT_REF_AUDIO）"。

## 4. P4：dump 非原子、丢 errno

`dump_adaln_cache` 直接 `fopen(path, "wb")` 写目标路径，中途失败（ENOSPC/EINTR）留下截断文件；
所有写失败统一报 "cannot write AdaLN cache dump %s"（:563-566），不带 `errno`。
下游 Python 用长度核对兜住了 —— :196-199 实测会对截断 dump 报 "the dump is truncated or the
header is wrong"，所以功能上不致命。但 157/379 MiB 的导出在 /tmp 留半截文件不体面，且"cannot
write"掩盖真实原因。修法：写 `path.tmp` 再 `rename`；消息里带 `strerror(errno)`。

## 5. P5：导出器 `--trim-to` 缺 `--model` 抛裸 traceback（实测）

```
$ python export_h3_adaln_cache.py --dump adaln_s1.raw --trim-to out_a
  File ".../export_h3_adaln_cache.py", line 441, in main
    trim(args.model, args.trim_to, dumps)
  File ".../export_h3_adaln_cache.py", line 257, in trim
    shards = sorted(model.glob("*.safetensors"))
AttributeError: 'NoneType' object has no attribute 'glob'
```

同文件 `--verify` 有干净守卫（:409-410 `raise SystemExit("--verify needs --model")`）。补一行即可。

## 6. P6：trim 静默丢掉源目录里已有的缓存，且 `--verify` 看不见（实测）

脚本 :265-266 `if is_cache_file(shard): continue` ⇒ 源目录的 `adaln_cache_s*.safetensors` **整文件
不进输出**，而模块文档 :80-81 与 README 都写"只丢 adaln_proj 矩阵"。实测：

```
源 model/ = adaln_cache_s4.safetensors + shard0.safetensors
--dump adaln_s1.raw --model model --trim-to out_b
out_b/   = adaln_cache_s1.safetensors + shard0.safetensors     # s4 无声消失
再对 out_b 跑 --verify  ->  PASS                                # 它只核对传入的 dump
```

影响面不小：给已裁剪模型**追加一档步数**是常规操作（README 的循环就是这么教的），若走 `--trim-to`
就把旧档位丢了，而"多档并存"正是这轮的设计卖点。修法二选一：未覆盖的 `adaln_cache_*.safetensors`
原样 `shutil.copy2` 过去；或至少在 trim 结束时列出被跳过的缓存分片，提示把它们的 dump 一起传进来。

## 7. P7：缓存路径零测试，而唯一的 AdaLN 回归测试根本没接进 make

- `tests/test_real_dit_schedule.c` 断言了行布局、与 MLX 的 modulation parity，以及
  `stats.submissions == 52`（逐块投影"一次一块"的常驻纪律）。但它只出现在自己的规则（Makefile:131）
  和 `clean`（:323）里；`test:` 的依赖与运行清单、`real-parity:` 都不含它（count-mode 复核：
  `grep -c test_real_dit_schedule Makefile` = 1 行）。⇒ 本轮改动前后都**没有可跑的闸**。
- 缓存路径的新逻辑（前缀校验、过短拒绝、LoRA 拒绝、dump↔safetensors 往返）目前只有 F40 手测的
  latent/mp4 md5 全同。
- 这是 **C↔Python 跨语言 codec**：`H3ADALN2` + 6×uint32 + times + 50×block + final，魔数和字段序
  两侧各写一遍（C :536-540 与 Python :96-97/173-174），改任一侧只有整轮渲染才发现。按本仓库
  既有规矩，codec fixture 要**双向锚定**（decode(真字节)==参考、pack(参考)==真字节）。本轮已把
  原料做出来：`/tmp/f42/mkdump.py` 能生成合法 dump，Python 读回后 C 侧 verify 全绿；差的只是把
  它变成 `make test` 里的一个用例。
- 附带一条：接进 make 时该测试要按路径分支 —— `submissions` 只在两处自增（h3_gpu.m:993/1028，
  均为提交路径），`h3_gpu_tensor_load_file` 不经过它们 ⇒ 缓存档 precompute 后应为 0，`== 52` 必然
  失败（**代码路径判定，未跑实测**）。

## 8. P8：小项

- `cache_header`(:610) 与 `adaln_header`(:613) 只被写、从不被读；`h3_weight_find` 明确接受 NULL 头参
  （h3_weights.c:169/174）⇒ 两处可传 NULL，各省一行一个假"输出"。
- `prepare_features` 无表分支按 `feature_dim`(2688) 分配、每行只写 `TIME_INPUT`(256)：17 行档 182 KB
  里只有 17 KB 有用。安全性依赖一条没写出来的不变量 —— `table == NULL` 蕴含
  `feature_dim == H3_DIT_TIME_DIM ≥ TIME_INPUT`（:627-628 只有 shape[1]≠2688 才走表分支）。注释说了
  "故意超配"但没给下界条件。建议无表分支按 `min(feature_dim, TIME_INPUT)` 分配，并把不变量写进注释。
- :277 的溢出检查没算 `* sizeof(float)`。原 `prepare_rows` 同形，非本轮引入；实际 count ≤ 2*steps+1、
  feature_dim ≤ 2688，够不着。
- `H3_ADALN_CACHE_NAME_MAX 64` 够用（最长约 27 字符），snprintf 返回值未核 —— 非问题，留说明。

## 9. 已核过、确认不是问题的（免得下轮重复怀疑）

- 前缀规则对三种条件模式都成立：steps=20 时 39/40/41 行的嵌套关系与 `test_real_dit_schedule.c:50-57`
  实测行号一致（39 步行 + visual=39）。
- `cache_rows` 取自 block 0，其余 51 张由 `load_tensor` 的逐维 shape 核对兜住
  （h3_weights.c:951-957 "shape mismatch at dimension N"）。
- dump 布局与 Python 端一致：C 写 `{steps, time_rows, 50, 96768, 10752, 0}` 六个 uint32，
  Python `HEADER_BYTES = 8 + 6*4`、`struct.unpack("<6I")`；旧魔数 `H3ADALN1` 有专门提示。
- 缓存档 GPU 常驻与计算档同形（都是 50×time_rows×96768 BF16 + final），不是新增开销。
- 对齐：Python `st_write` 不补 8 字节对齐，实测真档 `adaln_cache_s20.safetensors` 里 51/52 张量偏移
  非 8 倍数（block 0 起于 164）；但官方 `safetensors` 0.8.0 reader 实测**接受**错位偏移，h3.c 走
  pread 更无所谓 ⇒ 不是缺陷，只是把裁剪目录交给 mmap 零拷贝消费者时的潜在拷贝。

## 10. 下一轮入口

1. P2/P3/P4/P5/P6/P8 全是十行内的局部改，不碰算法、不需要 GPU 轮次，可一批做完；P5/P6 各配一条
   Python 侧负例（缺 `--model` 要干净报错；已有缓存分片要被带上或被列出）。
2. P7 是唯一要新写闸的：双向锚定的 codec fixture + 把 `h3_real_dit_schedule_test` 接进 `make test`
   并给缓存档分支。
3. P1 只在"缓存档要用 VDN default adapter"成为需求时才做，前提是先给缓存加 AdaLN 宽度字段。

## 11. 复现命令

```bash
# P2：空串 env 的三处判据不一致
cc -std=c11 -O0 /tmp/f42/getenv_probe.c -o /tmp/f42/getenv_probe && /tmp/f42/getenv_probe
H3_DIT_ADALN_CACHE_DUMP= /tmp/f42/getenv_probe
# P5/P6：导出器的两个参数组合缺陷（合成 dump 约 9.7 MB，不碰 GPU）
python3 /tmp/f42/mkdump.py /tmp/f42/adaln_s1.raw && python3 /tmp/f42/mkmodel.py
python3 fastvideo_qad/scripts/export_h3_adaln_cache.py --dump /tmp/f42/adaln_s1.raw --trim-to /tmp/f42/out_a
python3 fastvideo_qad/scripts/export_h3_adaln_cache.py --dump /tmp/f42/adaln_s1.raw --model /tmp/f42/model --trim-to /tmp/f42/out_b
ls /tmp/f42/model /tmp/f42/out_b            # 源里的 adaln_cache_s4 未出现在 out_b
# P7：唯一 AdaLN 回归测试没接进 make（count-mode，别用截断的 content 列表下"不存在"结论）
grep -c "test_real_dit_schedule" Makefile
awk '/^test:/,/^[a-z$_].*:$/' Makefile | grep -n "real_dit_schedule"   # 无输出
# 0. 告警 11 -> 1
git show HEAD:h3_dit_schedule.c > /tmp/f42/head.c && cc -std=c11 -O3 -Wall -Wextra -Wpedantic \
  -Wshadow -Wconversion -Wno-sign-conversion -D_DARWIN_C_SOURCE -I. -fsyntax-only /tmp/f42/head.c
```
---

# F43（2026-10-04）：F42 评审的 8 条全部落地 —— 逐条改动、闸与实测结果

按 F42 的清单把 8 条做完，无一轮 GPU 渲染。逐条对照：

## 1. P1 LoRA 闸收窄：可选 `adaln_cache_meta_s{steps}`（U32[1] = AdaLN 输入宽度）

- `h3_dit_schedule.h`：新增 `H3_ADALN_CACHE_META_FORMAT`，并把磁盘契约（magic、头字段序与
  宽度、`H3_ADALN_CACHE_DUMP_BYTES(rows)` 尺寸式、6 个 `H3_ADALN_CACHE_FIELD_*` 槽位）从
  `.c` 的文件内 enum 提到头文件，成为两侧唯一定义处；`.c` 里 `BLOCK_OUTPUT`/`FINAL_OUTPUT`
  改为引用这些宏。`dump_adaln_cache` 的字段数组改成**指定初始化器**按槽位赋值。
- `h3_dit_schedule.c`：新增 `adaln_cache_time_dim()`（键缺失 ⇒ 宽度 0；键畸形 ⇒ 硬失败）与
  `adaln_cache_accepts_adapters()`。判定顺序是 audio-only → 读宽度 → 逐适配器 `h3_lora_matches`
  （`transformer_blocks.0` + `adaln_proj.linear`，或 `""` + `norm_out.linear`）。
  **宽度未知时保持原来的整批拒绝**，所以已入库的三档缓存行为一字不变；raw dump 格式与魔数
  也不动，因此不需要重新导出。宽度已知时打印一句 note，说明"这些适配器不碰 AdaLN，缓存调制原样使用"。
- `export_h3_adaln_cache.py`：`weight_time_dim()`（从幸存的 `adaln_proj` shape 读）+
  `source_time_dim()`（前者为空时回落到目录里已有缓存的 meta）；`install()` 多写一个 meta 张量。
  **这一步纯磁盘操作，不需要 GPU。**

## 2. P2~P4、P8

- P2：`h3_dit_schedule_precompute` 入口读一次 env 并折叠成 `int dumping`，三处判据统一成
  `dump_path && *dump_path` 的语义。实测过的空串缺陷（强制 2 个条件行却不导出、缓存档印假警告）
  随之消失；其运行时影响只剩"少 2 行显存与两次多余投影 + 少一句假警告"，现有夹具观测不到，
  所以这一条按 getenv 探针（F42 §2 已实测）+ 判据统一来认定修好，**不声称端到端验证过**。
- P3：缓存分支前置判定 `audio_condition && !visual_condition`，报"缓存不覆盖无图音频参考"，
  不再建议一次永远满足不了的重建。
- P4：dump 先写 `<path>.tmp` 再 `rename`，任何失败路径 `remove()` 临时文件；错误消息带
  `strerror(errno)`，读不回 GPU 时给单独措辞。路径长度超过 4091 直接干净报错。
- P8：删掉只写不读的 `cache_header`/`adaln_header`（`h3_weight_find` 接受 NULL）；
  `prepare_features` 按分支真正用到的行跨度分配（表路径 `feature_dim`，正弦路径 `TIME_INPUT`），
  溢出检查算进 `sizeof(float)`，注释改成写清"每个分支的跨度就是消费者读的跨度"。

## 3. P5/P6 导出器

实测（`/tmp/f42`，合成 dump + 双分片假模型目录）：
`--trim-to` 缺 `--model` 现在是 `--trim-to needs --model` 且不建目录；
trim 会把源目录里未被本次 `--dump` 覆盖的 `adaln_cache_s*.safetensors` 原样 `copy2` 过去并打印
`carried 1 existing cache shard(s) across: adaln_cache_s4.safetensors`；随后对该输出跑 `--verify`
报出 53 键（52 + meta）与 `adaln_cache_meta_s1 is a non-zero U32 [1]`，PASS。

## 4. P7 两个闸

- `tests/adaln_cache_probe.c` + `tests/test_adaln_cache_codec.py`：探测程序**只用头文件宏**，
  打印契约常量、字段槽位、1/3/9/17/41 行的 dump 尺寸，并能按槽位解析头部与列出键名。
  测试把它编译出来（`$CC`，缺编译器则整个 `make test` 本来也跑不动），与导出器模块常量逐项比对，
  再用 distinct sentinel 头部验字段序、用 C 算出的长度验 Python 的尺寸闸（短 8 字节必须被拒），
  最后比对 52 个键名拼写与 meta 键拼写。
  **外部锚是真的**：探测出的 `dump_bytes_41/17/9` 与三档**真实已安装**缓存的载荷字节完全吻合
  （397,630,628 + 头部 5,200 = 397,635,828 等三行），所以公式不是自证。
  > F53 补注（2026-10-06）：这三份后来被就地补上了 `adaln_cache_meta_s*` 键，每份 +92 B
  > （头部 5,192→5,280、载荷尾 +4 B），s20 现在是 397,635,920 B。**这条锚没被推翻**：
  > `dump_bytes_41` 比的仍是那 52 个张量的字节，meta 的 4 字节在载荷末尾另算。
  Mutation 控制 5 条全红且可归因：A 改 Python 常量、B 改 Python 字段序、C 改 C 侧槽位宏、
  D 改 C 侧键名拼写、E 改 C 侧尺寸式 —— 每条都必须是 FAIL 行而不是崩栈；B/E 一开始是靠
  `Dump` 抛 SystemExit 崩出来的，已把这类拒绝收敛成判定行（`parse_dump()`）。
- `tests/gen_adaln_cache_fixture.py` + `tests/test_adaln_cache_lora_gate.c`：缓存档的 schedule key
  故意全填 `-12345.0`（任何 `1-sigma` 行都取不到），于是**凡是越过 LoRA/条件闸的用例都会停在
  前缀检查**——"different sigma schedule" 就成了"闸确实放行"的证据，而拒绝类用例根本走不到
  那 51 张大张量，所以整个夹具只有 2.1 MiB、不需要模型也不提交任何 GPU 工作。8 个用例全绿。
  该闸的 mutation 控制 4 条（一刀切复原 / 撤 audio-only / 忽略 meta 宽度 / 只查 norm_out）
  全部 red 且咬到的正是被监视的那条判定；每条控制都单独 `rm` 掉 object 与 binary 再重建，
  批次结束用 sha256 确认源码回到原样。
- Makefile：`test:` 现在跑 codec、生成夹具并跑闸；`h3_real_dit_schedule_test` 既进了 `test:`
  的依赖与守卫运行，也进了 `real-parity`。它原本那句 `submissions == 52` 断言按路径分支成
  `from_cache ? 0 : 52`，并打印走的是哪条路（**本机无模型，这一分支未实测**）。

## 5. `make test` 复验

`make test` **exit 0**：codec 全绿（含真档三行尺寸吻合）、闸 8/8 绿、其余套件 15 条 skip
（模型/夹具未装）、无 FAIL 无编译错误。这 15 条 skip 不是绿灯：本机上唯一新得的真绿灯是
codec 与 LoRA 闸两个，`h3_real_dit_schedule_test` 在这里是 skip。

## 6. 一处被推翻的前提：F40 的端到端复现小节今天跑不起来

想复跑一次 4 步缓存渲染去对 `6e9c2d87` 时撞到：
`./h3 -d /Volumes/data/MODELS/h3c-q4-adalncache` 报
`FL2VA/text_encoder: No such file or directory`。查目录：该树只有
`audio_vae`/`tokenizer`/`video_vae` 三个软链 + `transformer`，**没有 text_encoder**；
`h3c-q4-native` 与 `h3c-official` 同样没有（`find -maxdepth 3 -type d -name text_encoder`
在 `/Volumes/data/MODELS` 下返回空）。而 `h3.c:776` 说明把蒸馏编码器关掉时
`FL2VA/text_encoder` 才不参与运行 ⇒ F40 那轮用的显然不是小节里抄录的那条命令
（缺一个组件路径或一个蒸馏/离线编码器入口）。**在拿到真实调用方式之前，本轮不声称端到端复跑过**，
F40 的复现小节也应就地标注为"缺组件，命令不完整"。

## 7. 复现命令

```bash
python3 tests/test_adaln_cache_codec.py \
  --model /Volumes/data/MODELS/h3c-q4-adalncache/FL2VA/transformer   # 含真档尺寸吻合
python3 tests/gen_adaln_cache_fixture.py tmp_adaln_cache_fixture
./h3_adaln_cache_lora_gate_test tmp_adaln_cache_fixture              # 8 个用例
# 导出器两条（合成夹具，9.7 MB，无 GPU）
python3 fastvideo_qad/scripts/export_h3_adaln_cache.py --dump /tmp/f42/adaln_s1.raw --trim-to /tmp/f42/out_a
python3 fastvideo_qad/scripts/export_h3_adaln_cache.py --dump /tmp/f42/adaln_s1.raw \
  --model /tmp/f42/model --trim-to /tmp/f42/out_b && \
python3 fastvideo_qad/scripts/export_h3_adaln_cache.py --dump /tmp/f42/adaln_s1.raw \
  --model /tmp/f42/out_b --verify
# 端到端复跑今天跑不起来（F43 §6）
./h3 -d /Volumes/data/MODELS/h3c-q4-adalncache --ssd-streaming ... --steps 6 -o /tmp/x.mp4
```
---

# F44（2026-10-04）：整仓评审 —— 优化路径清单（4 条我逐行复核，其余为代理报告的机制）

用户要求"审核全部代码，看是否有优化路径"。本轮**只评审、不改代码、不跑 GPU**。
按子系统并行派 4 个评审代理（DiT+流式权重 / Metal 层+着色器 / VAE+编码器+ffmpeg /
引擎胶水+内存规划），每个都带上了已判死的红线（ANE、F31+F39 稀疏注意力、F32~F38 reuse、F26），
回来后我逐条复核行号与机制。**分级如下，级别就是证据级别**：

## 1. 我已读源码复核（机制成立，可直接排期）

### 1.1 video VAE 流式路径每块每 tile 两次全量 blit，而 hidden 本来就是原地累加器
- `h3_video_vae.c:934` `state_elements = vae->sequence * HIDDEN`；`:1327` 每个 state 张量按
  **同一个** `state_elements` 分配 ⇒ 与 `vae->hidden` 同尺寸。
- `:952-954` `copy states[state] → vae->hidden`，`:956` `run_block`，`:957-959` `copy
  vae->hidden → states[state]`。
- `run_block` 对 hidden 的全部写法是 `h3_gpu_scale_add_f32(gpu, vae->hidden, vae->hidden,
  branch, ...)`（`:653`、`:663`，原地累加），读法是 `:641`/`:655` 的 rms_norm ⇒ 把 state 张量
  直接当成 hidden 传进去，两条 blit 全可删。
- **两个代理各自独立命中同一处**（一处报 `:952/:957`，一处报 `:951-961`）。
- 收益类别：GPU 提交数与 blit 带宽；纯搬运不改算术 ⇒ 应当逐位不变，只动 `direct_dispatches`
  计数。同样的拷贝模式在 `:1234/:1262`、`:1279/:1287`、`:1319` 重复出现三次，
  而 `:1569-1603` 与 `:1739-1768` 两段 cross-chunk 代码几乎逐行相同（代理数字，未实测）。

### 1.2 ClipProj 三处判据互相矛盾，且库内硬编码了本机绝对路径 —— 这正是 F43 端到端跑不通的原因
- `h3.c:2051-2057`：`H3_CLIPPROJ_DIR` **未设** ⇒ 默认启用 ClipProj，并把
  `"/Volumes/data/.lmstudio/models/Qwen3-VL-4B-Instruct"` 写死在库里。
- `h3.c:376-388`（持久缓存 key）：未设 ⇒ `clipmodel=none`，即"50 层编码器"。
- `h3.c:781-792`（`h3_load_dir`）：未设 ⇒ `clipproj_active=0` ⇒ 缺 `FL2VA/text_encoder`
  **直接 fatal**。
- 同一个"env 未设"在三条路径上得到三个不同答案。后果：① 缓存 key 与真实编码器选择不一致
  （`:373-375` 注释自己要求跨进程锁定）；② 只有 transformer + 三个软链、没有 text_encoder 的
  模型树（`h3c-q4-adalncache`/`h3c-q4-native`/`h3c-official` 都是这样，F43 §6 实测）在
  `h3_load_dir` 直接 fatal ⇒ **F43 那条"照 F40 小节抄的端到端命令"就是死在这里**；
  ③ 库代码里带机器路径 = 换机器/换目录即坏。
- 一个 helper 折叠默认值、三处共用，零数值改动。

### 1.3 两处内存估算用错常量，方向正好是"把守卫放松"
- `h3.c:1375-1377`：activation 估式 `w*h*frames/16/16 * 56 * 4`，其中 **56 是 DiT 的
  `HEADS`（`h3_dit.c:24`）**，而 latent 通道是 `VIDEO_CHANNELS = 24`（`h3_dit.c:28`，
  真实字节数就按它算：`h3_dit.c:4960`）。
- `h3_dit.c:2771-2774`：注释写 `[C=16, T, H, W]`、按 **16** 预留，同样与 24 不符 ⇒ **少留 1/3**，
  于是自适应常驻块数偏多，误差方向是 `h3_memory_plan.c:43-46` 自己承认的那个 panic 侧。
- 代理称 `h3.c` 那条"高估约 8 倍"（还叠加了时间压缩缺失），**这个倍数是它的推算，未实测**；
  常量用错是确定的。修法：导出一个 `h3_dit_latent_bytes()`，把 `h3.c:1368-1394` 与
  `h3_cli.c:745-766` 两份已经漂移的输入式（`h3.c:1392-1393` 有 `*0` 死乘）合并成一个。

### 1.4 文本编码器把整个词表 embedding 传上 GPU，只为取几百行
- `h3_text_encoder.c:623-625`：`load_2d("model.language_model.embed_tokens.weight",
  TEXT_VOCAB, TEXT_HIDDEN)`，常量在 `:14-15` 是 `151936 × 5120`，BF16 ⇒ **约 1.45 GiB**
  一次性成为 GPU 张量；`:628-631` 的 `h3_gpu_embedding_bf16` 只 gather `tokens` 行。
- `:639-646` 随后还用视觉 span 覆写 pad 行 ⇒ 取到的行里还有一部分立刻被盖掉。
- 收益类别：峰值物理内存 + 磁盘读 + 上载时间（16 GB 机器与外置 SSD 上都在意）。
  纯 gather 无算术 ⇒ `h3_text_tests` / `h3_real_multimodal_text_test` 应逐位不变。
  改法：CPU 侧按去重后的 id 做 strided pread（几百行 ≈ 几 MB）再 `write_bf16_range`。

## 2. 代理报告的机制（我复核了行号存在，占比/收益是它们的估算，未实测）

1. **`h3_video_vae.c:1566-1567`**：`cross_chunk = streaming && states_bytes <= 1<<30` 是全有或全无，
   超 1 GiB 就退回 `:1379` 每 chunk 完整 `load_block` 36 块（注释 `:743` 给的是 fp32 约 268 MB/块）
   ⇒ 磁盘读乘以 chunk 数。改法是按预算分组的 chunk 批。
2. **`h3_dit.c:1542-1543`**：`requant_stream_slot` 末尾自开 `h3_gpu_submit`，紧接着 `:1551`
   才 `h3_gpu_begin`；而 `h3_gpu_begin`（`h3_gpu.m:967-968`）要求 `inflightCommands.count == 0`
   ⇒ 每块多一次排空。**注意**：这条不是"删 5 行"，它动的是全引擎依赖的命令缓冲纪律
   （`requant` 的注释 `:1511-1516` 明确说 worker 绝不碰共享 command buffer），要在 M5 上先取数。
3. **`h3_gpu.m:3073`**：用 label **字符串相等** `@"int8 MLP FC2 input"` 选 vec4 量化器，于是权重
   反量化（`h3_gpu.m:3215`）与 `@"int8 QKV input"`（`:4288`）落到标量对照 kernel
   （`h3_shaders.metal:1626`，注释自陈"仅为对照 vec4 路径保留"）；且 rows/groups 量化器
   （`:1554`/`:1715`）都是先求 max 再重读原行，本文件已有单遍解法（`:1662`、`:1871`）。
   max 满足结合律、逐元素 `rint` 不变 ⇒ 位相同；本机 M4 无 TensorOps，GPU 时间**未测**。
4. **`h3_shaders.metal:4126`** `h3_head_rms_norm_bf16`：一个线程独占整个 head，128 次 2 字节标量读；
   host `h3_gpu.m:4465` 走 `dispatch_2d` ⇒ 固定 256 线程/threadgroup（`h3_gpu.m:288`），组内相邻
   线程行距 3584 元素、无行内共享。照 `:3916` 的 coop 做法改成 128 线程协同装载、仍由 lane0 做
   **完全同序**的求和 ⇒ 逐位不变。
5. **NAX GEMM 收尾**（`h3_shaders.metal:1416/1480/1539/2899`）：`index += 128` 排 8192 元素、
   每 lane 2 字节 store ⇒ 32-lane 组只覆盖 64 B（半条缓存线）；改 bfloat4 视图即整线写。M5 未测。
6. **`h3_vision_encoder.c:557-564`**：视觉塔权重按"每张参考图"重读（27 层 ≈ 813 MB），
   `run_merger` 每次重载 merger（`:365-376`），一次编码内被调 4 次（`:574`、`:585`），
   `:451-453` 每次还重新 `h3_weight_store_open` ⇒ 单张图约 1 GB 读 + 上传。
   仓库已有 `h3_acquire_video_decoder` 的常驻范式可抄。
7. **`h3_ffmpeg.c:289` + `:369`**：同一帧数据同时持有 packed 与 planar 两份全尺寸缓冲
   ⇒ 峰值物理内存可减一份。`h3_ffmpeg.c:114-121` 逐字节 `read()` 取 ffprobe 输出，
   ffprobe 按 reference 逐个 spawn（`h3.c:1669`）。
8. **`h3_terminal.c:180-219`/`:264-269`**：预览每帧 mkstemp 写裸 RGB + fork 一个 ffmpeg 编 PNG
   再从磁盘读回，而 `h3.c:1330-1341` 是**每帧**回调 ⇒ `--show` 长片段要起几百个进程。
9. **`h3_dit.c:5015-5020`/`:5064`**：每评估一次做 6 次 latent 尺寸 malloc/free；
   `gpu_sampler_requested`（`:5140-5145`）在非 M5 为假 ⇒ M4 走这条，而 GPU 版已把同类缓冲提到步环外
   （`:5211`）。收益类别是分配次数，实测占比未知。
10. **`h3_dit.c:704`** 与 **`h3_weights.c:292/861/933`**：butterfly 反旋转纯标量；三条解码路径都是
    "全量展 F32 → 全量 pack → 第二遍 unrotate"，4bit 峰值 6.5 B/元素对盘上 0.5 B；
    同一张量重复算 grouped spec（`:171` 线性查找各 3 次）。NEON 化要保持
    `((a+b)+c)-d` 次序才逐位不变，`tests/test_convrot_unrotate.c` 是现成闸。
11. **`h3_memory_plan.c:74-75` vs `h3.c:1402`**：规划器的 int8 建议在流式分支置 1 后立刻被清零
    （`h3.c:911-914` 判互斥），而 `h3_memory_plan.c:22-25` 与 `h3_memory_plan.h:21-28` 两处注释都
    教"正交可叠加" ⇒ 文档与实际相反（与 F41 待办 2 的三处描述修正同类，现又多两处：
    `main.c:82-85` 与 `h3.h:157-162` 声称 `--pipeline` 会 `madvise(DONTNEED)`/重读盘，
    实际 `h3.c:2313-2320` 明写不 madvise，只有 `use_f16=1` 加一行日志）。
12. **`h3.c:1366-1367` + `:1421`**：用户显式给 `--ssd-streaming/--int8` 就整块跳过规划，随后
    `video_vae_streaming` 落到 0（常驻约 9 GiB）⇒ 16 GB 上"手动流式"恰是最需要省内存的组合却不省。
    兜底守卫在 chunk 循环内（`h3_video_vae.c:1584`），而 `:1571-1580` 在守卫之前一次性 calloc 全部
    tile 状态，上限硬编码 1 GiB（`:1567`）与设备无关。
13. **`h3_memory_plan.c:80-82`**：用未夹的 `rec` 算 `free_after_stream`，而 `:47-53` 正因不信任
    `rec` 才夹成 `target` ⇒ 极端预算判据更乐观。改一行，但会动 16 GB 默认档，需复跑校验。
14. **ANE 判死线仍在构建**：`LIB_M` 常编 3 个 `.m`（约 2.6k 行），产品唯一入口默认关
    （`h3_video_vae.c:293`），全仓 28 个 `H3_ANE_*` 旋钮、7 个测试目标，`make test` 每轮仍跑
    `h3_ane_staging_test`（`Makefile:189,213`）。**删除需单独批准**（红线），这里只记账。

## 3. 我建议的顺序（全部不需要新学一套方法）

1. **1.2 ClipProj 三处判据 + 硬编码路径**：它同时是 bug 和"端到端验证被卡"的根因，
   修完才谈得上把 F43 待办 1 的 md5 复跑补上。
2. **1.3 内存估算常量**：方向是放松守卫，16 GB 机器上有 panic 代价；纯常量与一个 helper。
3. **1.1 VAE 双 blit**：删的是纯搬运，逐位不变，收益在最长耗时的 decode 段。
4. **1.4 全词表上载**：省 1.45 GiB 峰值 + 读盘，风险低。
5. 第 2 节的 3/4/5（Metal 侧）都需要先在 M5 上取数再动；6/7/8 属于"省 IO/省进程"，
   收益明确但没有占比实测。
6. **11/12/13** 是"免费描述修正 + 一行守卫"，和 F41 待办 2 那批合并做最省。

## 4. 复现命令

```bash
# 1.1 / 1.2 / 1.3 / 1.4 的四处证据
sed -n '934p;952,959p;1327p;641,663p' h3_video_vae.c
sed -n '2051,2057p;376,388p;781,792p' h3.c
sed -n '1375,1377p' h3.c; sed -n '2771,2774p' h3_dit.c; grep -n "VIDEO_CHANNELS = 24\|HEADS = 56" h3_dit.c
sed -n '12,16p;623,646p' h3_text_encoder.c
# 1.2 的后果：缺 text_encoder 的树在 h3_load_dir 直接 fatal（F43 §6）
./h3 -d /Volumes/data/MODELS/h3c-q4-adalncache --ssd-streaming ... --steps 6 -o /tmp/x.mp4
```
---

# F45（2026-10-04）：F44 §1.2 落地 —— ClipProj 三处判据折成一个规则，顺带查出一个静默缓存投毒

按计划先做 F44 待办 1（`h3.c` 的 ClipProj 选择）。本轮结论：**F44 那条描述还不够重**，
真实危害是三处不一致里有一处会让条件缓存把 A 编码器的产物当 B 的读回去。

## 1. 取证修正：机器路径是两条不是一条，而且库的默认值和全仓其它地方不一致

- 库内硬编码其实有**两处**：`h3.c:2056`（4B 目录）与 `h3.c:2061` / `h3.c:389`（投影目录）。
- **`README.md` 里 `H3_CLIPPROJ*` 出现 0 次**（count-mode 复核）⇒ 这两个开关此前完全没文档，
  这正是 F43 §6"照 F40 小节抄的命令跑不通"的土壤。
- 库默认指向 **BF16** 的 `Qwen3-VL-4B-Instruct`，而 `comfyui_nodes/h3_binary.py:40`、
  `benchmark/benchmark.py:31`、`fastvideo_qad/scripts/ab_quant_quality.py:58`、
  `gen_comfyui_workflows.py:56` 全部指向 **`...-int8-convrot`** ⇒ 裸跑 `./h3` 与脚本跑的是
  不同的编码器。
- 契约的权威说法在两处文档里，且互相一致：`comfyui_nodes/__init__.py:19`"设置
  `H3_CLIPPROJ_DIR` + `H3_CLIPPROJ_PROJ`（**设了则 `FL2VA/text_encoder` 可缺失**）"、
  `h3_text_encoder.h:75-76`"Enable by pointing H3_CLIPPROJ_DIR at ..."。
  三者里唯一的异类是 `h3.c:2051-2056` 的"未设也默认启用 ClipProj"。

## 2. 真正的危害：未设时"用 ClipProj 编码、却按 50 层记 key"

`h3.c:2051-2056` 把 `cp_dir` 先填成内置默认，**再**用 `strcmp(cp_dir,"0")/("off")` 判
`use_clipproj` ⇒ 未设必然走 ClipProj 分支。而磁盘 key 那边（`h3.c:376-391`）未设时记
`clipmodel=none`、`clipproj=none`，注释里 `none` 的含义是"50 层编码器"。
于是：一棵**有** `FL2VA/text_encoder` 的树、`H3_CLIPPROJ_DIR` 未设 ⇒ 实际编码用 4B+ClipProj，
缓存 key 却声明"50 层"。下一个同样未设的进程会把这份 ClipProj 条件当作 50 层编码器的结果读回去
——**key 是按编码器身份算的，这层不一致没有任何地方能发现**。
（此前它没被观察到，只是因为无 text_encoder 的树在 `h3_load_dir` 就先 fatal 了。）

## 3. 改成什么

单一规则 + 单一读取点（`h3.c`）：
- `h3_clipproj_classify(dir, proj, ...)` —— 纯函数，只吃两个字符串，返回
  `50_LAYER` / `CLIPPROJ` / `INCOMPLETE`；
- `h3_clipproj_resolve(...)` —— 全仓**唯一**读 `getenv("H3_CLIPPROJ_*")` 的地方（复核：
  `grep -n H3_CLIPPROJ h3.c` 只剩注释、解析器本身与报错文案）；
- `h3_clipproj_disk_identity(...)` —— key 用同一个解析器，未配置/半配置都记 `none`；
- 语义：**ClipProj 是 opt-in**，DIR 与 PROJ 必须都给；只给一个 ⇒ 生成路径明确报错
  （不再偷偷用内置投影目录）；未设 ⇒ 必须加载 `FL2VA/text_encoder`。
- 库里**不再保留任何默认路径**（`grep -c "\.lmstudio" h3.c` = 0）。
- `h3_load_dir` 的容忍条件改为"只要请求了任一 ClipProj 模式就容忍 text_encoder 缺失"，
  这样半配置的报错来自能说出两个变量名的生成路径，而不是一句误导的"组件缺失"。
- 半配置在 key 侧记成 `none`（=回退档），这样它不会写出一个"配置正确后会被当成自己的"条目。

**为什么这个语义而不是保留默认**：三条独立证据都指向"设了才算启用"
（`__init__.py:19`、`h3_text_encoder.h:75-76`、以及 `:781-784` 本来就是这条规则），
而保留默认的那条分支在现实中不可达（无 text_encoder 的树先 fatal，有 text_encoder 的树
则静默错配 key）。同时所有真实调用方（ComfyUI 节点、benchmark、ab_quant、`clipproj_golden.sh`）
都给全了两个变量 ⇒ **没有任何现在能工作的用法被改变**。

## 4. 验证

- 新闸 `tests/test_clipproj_selection.c`：把 `h3.c` 作为单一 TU 包含进来直接测私有规则，
  10 个用例（未设/空串/`=0`/`=off`/两者都给/只给 DIR/给空 PROJ + 三种 key 身份），全绿；
  已接进 `make test`（`Makefile` 专用规则：`-Wno-unused-parameter` + `-undefined dynamic_lookup`，
  并显式声明 `h3.c` 为依赖，改引擎即重编）。
- Mutation 控制 4 条全红且各归因到被监视的那条判定：撤掉 `0/off` 回退、空串当已设置、
  不再拒绝半成品、key 不再回退。每条控制前删二进制重编，批后用 sha256 确认 `h3.c` 回原样
  （`hash_ok=True`）。其中一条最初被预检挡下——我的替换文本正好是基线行的子串，
  `count(new)=1` 判成 harness fault，换成不重叠的写法后重跑才成立。
- `make test` **exit 0**（三闸齐绿：codec `PASS`、AdaLN 闸 8/8、ClipProj 表 10/10）。
  注意：我只用 `tail -50` 留了日志尾部，所以"12 条 skip"是截断后的下界；
  **全套无失败是靠 exit 0 证的**（每个测试都是独立 recipe 行，任何非零都会让 make 非零）。
- **真机两证**（`h3c-q4-adalncache`，无 `FL2VA/text_encoder` 的裁剪树）：
  ① 两个变量都给 ⇒ 装载通过、`text encoder (clipproj) 0/50` 真的跑起来，最后停在 F40 那条
  负向对照 "ships AdaLN caches but none for 6 steps"；
  ② 只给 DIR ⇒ 新报错 `H3_CLIPPROJ_DIR is set but H3_CLIPPROJ_PROJ is not; ...`。
  ⇒ **F43 §6 的堵点解除**，F43 待办 1 的 4 步 md5 复跑现在具备条件。
- 文档同步：新增 README「Text encoder choice」小节（此前 0 处提及），
  并把 `h3_text_encoder.h:75-76` 那句"`H3_CLIPPROJ_PROJ` overrides the projection dir"
  改成"两个都必须给，缺一个就报错；引擎内不含默认路径"——原句教的正是被本轮删掉的回退。

## 5. 顺带记下的一处小缺陷（未改）

`h3_conditioning_cache_dump` 的形参 `conditioning_key`（现 `h3.c:494`）从未被使用，
`-Wextra` 早就在报。已用 `git show HEAD:h3.c` 单编复核：**该告警早于本轮**，
与 `h3_dit_schedule.c:422` 那条同等对待，只记账不顺手改。

## 6. 复现命令

```bash
grep -c "H3_CLIPPROJ" README.md                  # 本轮前为 0
grep -c "\.lmstudio" h3.c                        # 本轮后为 0
grep -n "H3_CLIPPROJ_DIR\|H3_CLIPPROJ_PROJ" h3.c # 只剩注释/解析器/报错文案
make h3_clipproj_selection_test && ./h3_clipproj_selection_test
# 真机两证
H3_CLIPPROJ_DIR=/Volumes/data/.lmstudio/models/Qwen3-VL-4B-Instruct \
H3_CLIPPROJ_PROJ=/Volumes/data/.lmstudio/models/ClipProj-MiniMax-H3 \
  ./h3 -d /Volumes/data/MODELS/h3c-q4-adalncache --ssd-streaming -p "A red fox walking through snow" \
  --width 256 --height 256 --frames 22 --steps 6 --seed 42 -o /tmp/x.mp4
H3_CLIPPROJ_DIR=/Volumes/data/.lmstudio/models/Qwen3-VL-4B-Instruct \
  ./h3 -d /Volumes/data/MODELS/h3c-q4-adalncache --ssd-streaming -p x ... --steps 4   # 应报缺 PROJ
```
---

# F46（2026-10-05）：从 splash（Apple silicon LLM 引擎）看 MiniMax-H3 的加速路径 —— 能搬的三处与不能搬的三处

用户让看 `/Volumes/data/git/c/splash` 能给 MiniMax-H3 加速什么。**本轮纯研究，不改代码、不跑 GPU。**
splash 是 incoai 的**自回归 LLM** 推理引擎（Qwen3.8-27B 稠密 / Qwen3.6-35B-A3B MoE，GGUF 1–8bit 与 MLX 4bit），
MiniMax-H3 是**扩散 Transformer**（每步过整条序列、迭代去噪若干步）。所以每条机制都要问一遍
"splash 里为什么省"在扩散情形下还成不成立。

注：用户指定的 `docs-researcher` agent 类型在本会话未注册，改用 `general-purpose` 承接同一份 brief。

**证据分级**：标【核】的是我读了原文/源码的；标【报】的是代理报告、我未逐行复核的。

## 1. 最值得搬：内存规划从"静态比例"变成"实测 + 水位回收"【核，部分报】

- splash：`hardBudget = min(--max-memory, recommendedMaxWorkingSet − margin)`，
  `margin = max(1 GiB, 2%)` —— `runtime/engine/MemoryPlan.hpp:63-86`。**注意它自己也用百分比**，
  区别是百分比叠在 OS 给的 recommendedMaxWorkingSet 上，且可用内存是实测的：
  `host_statistics64` 的 `free − speculative + fileBacked + purgeable + 压缩节省`，
  Critical 时不计压缩【报：MemoryGovernor.cpp:19-63】。
- 回收是**带水位的目标式**：Warning 每 1 s 最多一次、目标字节 = 恢复水位 − 当前余量，
  只有 OS Critical 才清空全部可逐出缓存；`hostConstrained_` 用 1/2 GiB 双阈值做**迟滞**防 thrash；
  reserve-then-allocate，驱动仍拒绝才算失败【报：MemoryGovernor.hpp:63-67、.cpp:278-330、210-226】。
- **可直接抄且我们缺的那一个**：`contextTokensWithin(bytes)` —— "给定这些字节能撑多长的反查"
  【报：MemoryPlan.cpp:259-266】。h3c 只有正算（画幅×帧数→字节），长片段撞墙就是这么来的：
  `h3_memory_plan.c:47-53` 的 80%/85% 是静态启发式，注释自己承认长片段能判错到系统 panic
  （F41/F44 都记过）。搬法不是抄它的数，是**加这道反查**：从"这份预算能开多少帧"倒推分段长度。
- `--idle-release`：所有 buffer 挂在 command queue 的 residency set 上跨请求常驻，
  10 分钟无命令才释放权重镜像【报：DEVELOPMENT.md:546-565】。对我们这种"一次渲染一进程"用处小。

## 2. 注意力 K/V 降到 INT8：容量理由不成立，**带宽**理由成立【核】

- splash 设计我读到原文：每 `(head, token)` 一个 fp32 对称 scale（`splash_kv_scale_element`），
  K 存 token-major、V 存 dim-major（`runtime/metal/abi/KvExtent.h:43-59`）；量化融在写页 kernel 里
  与落页同一次完成；读侧**不单独跑反量化**：int8 指针直接当 matmul 的 device operand，
  key scale 乘在 QK 之后的分数上、value scale 乘在概率上【报：paged_attention_tile.h:208-276】。
- 扩散情形：**没有跨步复用的 KV**，所以"省容量"这条不成立；但窗口内每个 K/V tile 会被窗内
  各 query tile 重读，**bf16→int8 就是把这部分带宽减半**，这条成立。
- 两条必须先自测的：① splash 自己明说 BF16/INT8 之差"不是通用提速"，512-token 的 cycle 差在
  ±0.8% 内【报：dev/benchmarks/kv-formats.md:5-6、46-48】；② h3c 的 head_dim = 5376/56 = **96**，
  splash 是 256，且它的 int8 operand 路径依赖 M5 的 matmul2d —— **M4 无 TensorOps 吃不到**。
- 质量代价它有数：与 llama.cpp 首位一致率 INT8 99.23–99.25%(27B)/97.92–97.94%(35B)，
  BF16 99.30–99.45%【报：docs/performance.md:40-49，M5 Pro】。**这是 LLM 逐 token 一致率，
  不能当我们的画质预算**（按红线：文档数字带条件，且我们的画质闸是目视 + 同形状实测）。

## 3. VDN 线性分支的扫描 kernel 有对口参考，但**不能照抄**【核】

- h3c 现状我读了本体：`h3_shaders.metal:6170 h3_vdn_scan_step` —— 一个线程一个输出元素
  （`index = gid.x` 覆盖 `dim*dim`），内层 `for k < dim(=128)` 串行 FMA，每线程 128 次独立 load。
- splash 的 GDN kernel：每 lane 持连续 4 个 fp32、点积靠 `simd_sum` 跨 lane 归约；prefill 把状态行
  留在寄存器里整块推进、16 token 的 q/k/v/decay/beta 进 threadgroup memory 并预取下一块；
  phase 间用 threadgroup memory 交接（"先 load 后 store"）换来 1.2~1.4× 且逐位不变
  【报：kernels/decode/gdn.metal:140-200、kernels/prefill/gdn.metal:1-33 及其注释 8-18】。
  我核对到它的类型混合是真的（`float decay[]` + `bfloat q/k/v/rows/beta`），
  但"state 是每 value head 的 128×128 fp32"未逐行确认【报】。
- **为什么不能照抄**：`simd_sum` 跨 lane 归约 = 改变求和顺序 = 改浮点结果。splash 自己就
  **否掉过**一个"数值测试通过但改变求和顺序"的并行分母归约【报：
  dev/benchmarks/remaining-decode-optimizations.md:14-18】——这条纪律和我们完全一致。
  可行的是：先测现 kernel 到底是不是带宽/延迟受限，再考虑共享内存 tile + **保持同序**的分块归约。
- 定位要说清：VDN 分支在我们 Phase 9 的实测里是**质量叠加**（同画幅比无 VDN 慢 3.4~7×），
  所以搬这个 kernel 是"降低 VDN 的价格"，**不是让基座变快**。

## 4. 明确不能搬（省下轮次）

- **投机解码 / DFlash2 的"无损省步"**。判据我读到原文：逐位置 `uniform * q < p` 接受
  （`runtime/metal/kernels/decode/sampling.metal:1407-1416`），拒绝后从残差分布
  `(p − q·保留质量)⁺` 重抽使边际分布逐位等于目标（同文件 682-692 附近注释"never below zero,
  so the draw follows the residual distribution"），且 draft 必须**与目标联合训练**、按 family
  签名校验才能装（`DEVELOPMENT.md:383-401`）。扩散拿不到这道判据的具体理由有两条：
  ① 验证器一次前向给的是整块 x 的 score/velocity，**不是候选的归一化密度 p**，算不出 `p/q`；
  ② 更致命——LLM 拒绝一个 token 只作废那一个位置、前缀仍成立，而扩散**拒绝第 t 步就要重跑第 t 步**，
  所以不存在"一次前向验证 k 步并保留已接受前缀"。这就是 reuse 只能靠质量指标（5 次新鲜评估那道闸）
  的结构性原因。重开缺的门槛：每步一个可计算、无偏、归一化的 `p_target/q_draft` 密度比
  （或等价的耦合构造）。**不据此排期。**
- **把 `F_NOCACHE` 照搬到流式权重读**。我读了它两处用途：① 权重镜像的**写**侧
  —— "writers read their sources uncached (`F_NOCACHE`) so the page cache keeps no second copy
  of the model"（`DEVELOPMENT.md:522-524`）；② KV/GDN state 的 disk tier 传输（`1197` 附近，
  1 MiB 对齐块直传）。两者都不是 h3c 这种"**每个去噪步重读同一批字节**"的情形，照搬等于主动放弃
  页缓存。splash 也没有这维度的数据 ⇒ "页缓存保留值多少钱"仍只能我们自己 A/B（F41 待办 3）。
- **token 稀疏注意力**：记一条事实供存档——splash 的 attention 打包单位是
  "一个 threadgroup = 一个 KV head × 其 GQA 组若干行"【报：paged_attention_tile.h:17-20、181-194】，
  即 F39 说我们缺的"更细单位"在别人那里确实做到这个粒度。这不改变判死（我们的 rect-SDPA
  没有对应的 per-head gather，且 ~1.8 ms/次编码没解耦），只是说明"若哪天真重开，这条路工程上成立"。

## 5. 方法学上最该学的一条（比任何单点收益都快）

splash 的规矩：规则**只在原生机器实测到增益才采纳**（按模拟核数选 split tile 曾误判 −26%~+20%）、
GPU 核数取自 IORegistry、不做 per-SKU 表；ABBA 交替、31 次取中位、要求逐位相等；
并且明写"单 kernel 1.2–1.7× 到整图只剩 1.005–1.13×"【报：device-policy.md:33-40、
remaining-decode-optimizations.md:60-72】。这与我们的"时间窗口内才可比、只报比率"是同一纪律，
但它**把设备事实集中成一处**，而 h3c 的 M4/M5 差异现在散在运行时 if 里 —— 这条是可抄的结构。

## 6. 未找到 / 建议我自己再读

- INT8 与 BF16 KV 在**长上下文**下 attention 时间或带宽的直接对比数值：未找到
  （只有"BF16 在长上下文可能更慢"一句，`DEVELOPMENT.md:201-202`）。
- GDN state 的低精度化实验：未找到，仓库始终 fp32。
- 代理点名的锚点（若开工值得精读）：`runtime/metal/kernels/prefill/gdn.metal` 全篇
  （分块扫描的取/算重叠，对着 `h3_vdn_scan_step` 比着读）、
  `runtime/engine/StateCache.cpp` + `KvPool.cpp` 的三级受害者排序、`dev/tuning/LinearTuning.cpp`
  的离线标定流程（标定不改变服务默认值）、`dev/benchmarks/attention_sweep.mm` 的
  `--compare-metallib` 基线比对法。
---

# F47（2026-10-05）：F44 待办 2 三条落地 —— 三次真机 A/B 证逐位不变，并查明 F40 基线用的是哪个 4B

**改了什么**（全部零新增 GPU 轮次以外的假设）

1. **内存估算的常量错误**（F44 §1.3）
   - 新增 `h3_dit_latent_elements(latent_t, latent_h, latent_w, audio_t)`（`h3_dit.h:96`，实现
     `h3_dit.c`）：通道常量继续只在 `h3_dit.c` 一处定义。
   - `h3_dit.c` 的自适应常驻估算：删掉手写的 `[C=16,...]`（真实 `VIDEO_CHANNELS=24`）与
     `32*2`，改调 `h3_dit_video_elements()` + `h3_dit_audio_elements()` —— 就是 `h3_generate`
     实际 malloc 的那两个尺寸，从此不可能漂移。**方向是增加预留 ⇒ 常驻块数变少 ⇒ 更安全。**
   - `h3.c` 自动规划的 activation 项：原来 `w*h*frames/16/16*56*4` 里 **56 是 DiT 的 HEADS**，
     且漏了时间压缩；改成 `h3_temporal` + `h3_latent_canvas` + 上面的 helper。顺带把
     `render_width/render_height` 的解析提到规划之前共用（旧式用 `params->width`，忽略显式
     render 覆盖）。`h3_cli.c` 的 `!memory-plan` 同一处口径。
2. **video VAE 流式路径每块每 tile 两次全量 blit**（F44 §1.1）
   `run_block` 加 `hidden` 形参；`run_stream_chunk` 直接把 `states[state]` 当 hidden 传进去，
   两条 `h3_gpu_copy_f32` 删除。三处调用点里两处仍传 `vae->hidden`（非流式路径行为不变）。
   静态审计：`run_stream_chunk` 的三个调用点后面接的都是 `finish_chunk_states`
   （每次先 `states[index] → vae->hidden` 再 `finish_hidden`），`decoder_decode_chunk:1375` 同路
   ⇒ 没有任何读者依赖被我删掉的那次拷贝留下的陈旧 `vae->hidden`。
3. **整块词表 embedding 上载**（F44 §1.4）
   新增 `h3_weight_gather_bf16_rows()`（`h3_weights.c`）：按 id 逐行 `pread` 再整块上传；
   两条编码器路径都改用它，并删掉因此失去用途的 `ids` GPU 张量（含 `activations[]` 数组的
   下标边界 `13 → 12` 同步修正）。原 kernel `h3_embedding_bf16` 是纯行拷贝（越界补零），
   新实现越界改为**明确报错**——两条路径都在更早处校验过 id 范围。

**真机验证（这台 16 GB M4，缓存档 `h3c-q4-adalncache`，256×256 / 22 帧 / steps 4 / seed 42）**

| 二进制 | 编码器 4B | latent md5 | mp4 md5 |
|---|---|---|---|
| 我的三处改动 | int8-convrot | `6e9c2d8786cea097ddf9bdd45e5593bd` | `ac0d0940434ad216d8131f7469154b70` |
| 只回退 `h3_text_encoder.c`（整载全表） | int8-convrot | 同一 md5（`cmp` 差异 0） | 同一 md5 |
| 只回退 `h3_video_vae.c`（保留两条 blit） | int8-convrot | `6e9c2d87…` | `ac0d0940…`，**mp4 `cmp -l` 差异 0 字节** |
| 我的三处改动 | BF16 4B（库旧默认） | `a63e27a4c42621c6a7388d56c9942961` | `fa5f8c2cfb74bbdfe67e13f64945e22e` |

⇒ **三处改动逐位不变**（第 3 条由 latent+mp4 双证，第 2 条由 mp4 直证，不只是"上游没变"）。
⇒ **F43 待办 1 结清**：F40 记的 `6e9c2d8786cea097ddf9bdd45e5593bd` 逐字符复现。

**顺带查清的一件事**：F40 的基线用的是 **int8-convrot** 的 4B（与本仓所有脚本一致），
不是库那个 BF16 内置默认。所以 F45 删掉那个默认值不只是清理——那个"默认"本来就
和项目实际跑的东西不一致。

**`make test`**：exit 0，46 行绿、无 FAIL、无 make Error，15 条 skip（模型/夹具未装）。
三个新闸（codec、AdaLN LoRA 闸、ClipProj 选择表）全绿。

## 一处必须留档的保留意见（第 1 条的 h3.c 半边）

- `h3.c` 那项修正让 planner 看到的 activation **变小**：256²/22f 从 1.2 MiB→0.15 MiB；
  864×480/360f 从 ≈124.6 MiB→≈10.8 MiB（约 11×）。**方向是把档位推向更激进**，
  而 `h3_memory_plan.c:43-46` 的注释自己承认长片段判错会把物理内存打穿到 panic。
- 更深一层：**新旧两个式子都没建模 DiT 自身的逐块激活**（按 rows×hidden 的那几份 bf16/f32 缓冲，
  长片段是 GB 级），真正的占用主导项不在式子里。旧式因为把 56 当通道而**偶然偏保守**。
- 本次渲染**完全没走到这段代码**：自动规划的门槛是 `eff.ssd_streaming == 0`，而我们显式给了
  `--ssd-streaming` ⇒ 这段的正确性只有静态证据，没有真机档位验证。
- 结论：第 1 条里 **`h3_dit.c` 那半边是净收益**（增大预留、有真机逐位证）；
  **`h3.c`/`h3_cli.c` 那半边是"算对了但少了个主导项"**，应并入 F46 待办 2 的
  "实测可用内存 + 反查可撑帧数"一起做，别当成安全修复交付。
- 另外这台机器上 activation 项对档位几乎不起作用：权重 35 GiB 对 `target`（16 GiB 机器上约
  1.2 GiB）永远超 ⇒ 必然流式。差 100 MiB 不改变任何决定，只有贴近边界的形状才可能被推动。

## 复现命令

```bash
# 逐位闸（三条任一）
H3_CLIPPROJ_DIR=/Volumes/data/.lmstudio/models/Qwen3-VL-4B-Instruct-int8-convrot \
H3_CLIPPROJ_PROJ=/Volumes/data/.lmstudio/models/ClipProj-MiniMax-H3 \
  ./h3 -d /Volumes/data/MODELS/h3c-q4-adalncache --ssd-streaming \
  -p "A red fox walking through snow" --width 256 --height 256 --frames 22 \
  --steps 4 --seed 42 --latent-out /tmp/s4.bin -o /tmp/s4.mp4
md5 -q /tmp/s4.bin   # 6e9c2d8786cea097ddf9bdd45e5593bd
# 单文件 A/B 的做法：cp 走我的版本 -> git checkout HEAD -- 该文件 -> 重建 -> 同命令 -> 比 md5 -> 还原并核对哈希
```
---

# F48（2026-10-05）：校准实验推翻 F47 待办 1 的前提 —— 真正的内存墙主因是"每块常驻成本"写死 0.5 GiB，实测 0.72 GiB

原计划给 `h3.c` 的 activation 项补上"DiT 逐块激活"这个主导项，并用 `H3_PROFILE` 校准。
校准跑出来的结果说明**主导项判断错了**，而且顺手复现了一次真实的 OOM 击杀。

## 1. 两次同画幅、不同帧数的实测

同一台 16 GB M4、同一缓存档、同 prompt/seed/steps=4，只改 `--frames`，带 `H3_PROFILE=1`：

| frames | 自适应常驻 | 它报的 available | DiT load 的 peak | 结局 |
|---|---|---|---|---|
| 22 | **9**/50 块 | 5.6 GiB | 8.040 GiB（denoise 后仍 8.040） | 跑完，latent/mp4 落盘 |
| 44 | **13**/50 块 | 7.6 GiB | 10.995 GiB | **denoise 中 rc=137（SIGKILL）** |

（44f 那次的日志尾部就停在 `denoise 0/4`，进程被系统杀掉，没有我方错误信息。）

## 2. 两点求斜率 ⇒ 每块常驻的真实成本是 0.72 GiB，代码按 0.50 GiB 算

- 峰值差 `10.995 − 8.040 = 2.955 GiB`，常驻块差 `13 − 9 = 4` 块 ⇒ **0.739 GiB/块**。
- **更正（2026-10-05，F51 复算）**：下面这条原来用 `latent_t = (f+3)/4` 手算，那是
  `h3_video_encoder_latent_t()`（`h3_host.c:35-37`，编码器下采样）的式子，不是 DiT 序列的
  `h3_video_latent_t()`（`:30-33`，`((f-5)/17)*5+2`）。改成直接调真函数测：
  256²、文本 6 行、无附加条件下
  `--frames 22` → 对齐 **22**、估计 **528 行**（6 文本 + 448 视频 + 74 音频）、激活 0.0582 GiB；
  `--frames 44` → 对齐 **56**（梯子同余于 5 mod 17：5/22/39/56…，我先前记成的"58"也不对）、
  **1280 行**、激活 0.1410 GiB。
  于是 rows 差 = `1280 − 528 = 752` 行，按 `allocate_activations()` 默认活集合
  （MLP 融合开、别名开）118,272 B/行 ≈ **0.083 GiB** ⇒ 每块
  `(2.955 − 0.083)/4 = 0.718 GiB`。
- 独立核对（不靠拟合）：`allocate_stream_slot()` 的 BF16 权重是
  `INNER*3*HIDDEN + HIDDEN*INNER + FFN*2*HIDDEN + HIDDEN*FFN` 个元素
  = 385,874,432 元素 × 2 B = **736 MiB = 0.719 GiB/块** —— 与拟合值 **0.718** 相符。
  （更正前那版算出的 0.729 也落在同一量级，结论没变，但两条独立路径现在能对上到 0.001 GiB。）
- 而 `h3_dit.c:2779` 用的是字面量 `per_block = 0.5 * 1024^3`，其注释的理由是
  "int8 权重占主导，约 0.39 GiB，取保守的 0.5 GiB"。
  **这个前提只在 `--int8` 开着时成立**；`--ssd-streaming` 不带 `--int8`（M4 上的默认路径）
  槽位是 BF16 ⇒ 0.719 GiB ⇒ **低估 1.44×**，于是多钉约 44% 的块。
  44f 那次 13 块 ≈ 9.3 GiB 槽位，加上 11 GiB 峰值，16 GiB 机器直接在去噪中被杀。

⇒ **F47 待办 1 里"activation 是长片段主导项"这个假设不成立**：activation 在 44f（对齐 56 帧）
实测 0.141 GiB（118,272 × 1280 ≈ 151 MB），而每块成本误差一项就是 13 × 0.22 ≈ 2.9 GiB。
长片段确实把它顶大了（0.058 → 0.141 GiB），但那是 2.4×，而致命的是 13 块 × 0.22 GiB 的累积。

## 3. 第二个实测事实：`h3_host_available_memory()` 同形状两次给出 2.6 / 5.6 GiB

本轮 256²/22f 跑出 9 块常驻，而 F47 那两次同命令跑出 **3** 块（available 2.6/2.7 GiB）。
同画幅同帧数同二进制，常驻块数差 3 倍，说明这个"可用内存"读数被页缓存/压力状态左右，
而常驻决策**只**靠它 ⇒ 常驻策略成了抽奖。这不是新问题（F41 待办 3 就缺这一维的记账），
但现在有了具体数字：**同形状 3↔9 块、对应 available 2.6↔5.6 GiB**。

> F51 复跑同一形状（256²/22f/steps4/seed42，两串逐位锚不变，F49 修后的二进制）钉了 **6** 块。
> 注意这**不能**与上面 3 / 9 排成一条同代码的序列 —— 那两次是 F49 之前的二进制
> （`per_block` 还是 0.5 GiB、也没有物理内存那道夹）。所以能说的只是：修后仍然只有一个
> 观测值 6，而抽奖的源头（`available` 自身不稳定）没有被 F49 动过 —— 加的是上夹，不是去抖。

## 4. 修正后的做法（下一步该做的）

1. `per_block` 从字面量改成按实际槽位算：BF16 槽 = `allocate_stream_slot()` 的那四项之和 × 2 B；
   开 int8 时是 int8 字节 + scale 缓冲。代码里已有这些尺寸，没有新数学。
   （这条能直接把 44f 的过钉从 13 块降到 ≈9 块。）
2. 常驻上限除了 `available` 还要被**物理内存 − 已有峰值**夹住：`h3_memory_plan.c:43-46`
   的注释早就承认 GPU 固定分配不可换出，但 `h3_dit.c` 的自适应那段只看 `available`。
3. activation 主导项仍然该补（它是真实的一项，只是不是主导），按 118,272 B/行 +
   别名/融合关闭时的差额来算；`h3.c` 的 activation 因此**变大**，
   与 F47 那条"修正让预留变小"的告警方向抵消掉，反而更安全。
4. 反查（"这份预算能撑多少帧"）要有意义，得等 1~3 之后 —— 现在反查的输入本身不可信。

## 5. 复现

```bash
# 44f 的击杀与 13 块过钉
H3_PROFILE=1 H3_CLIPPROJ_DIR=...-int8-convrot H3_CLIPPROJ_PROJ=... \
  ./h3 -d /Volumes/data/MODELS/h3c-q4-adalncache --ssd-streaming \
  -p "A red fox walking through snow" --width 256 --height 256 --frames 44 \
  --steps 4 --seed 42 --latent-out /tmp/f44.bin -o /tmp/f44.mp4
echo $?            # 137
grep "partial residency" /tmp/... # 13 of 50 ... per-block 0.50 GiB
grep "h3 profile: H3 DiT  *load"  # peak=10.995GiB
# 每块 0.719 GiB 的独立算式
grep -n "INNER \* 3 \* HIDDEN\|HIDDEN \* INNER\|FFN \* 2 \* HIDDEN\|HIDDEN \* FFN" h3_dit.c | sed -n '1,4p'
```
---

# F49（2026-10-05）：F48 的修正落地 —— 常驻成本按实测算 + 物理内存夹一道，44f 从被杀变成跑完

改的是 `h3_dit.c` 的自适应常驻决策（三处），零数值路径改动。

## 1. 改了什么

1. **`per_block` 不再写死 0.5 GiB**：新增 `stream_slot_bytes(dit)`，按四个投影各自的实际形态算
   （int8 开 = int8 字节 + 每输出行一个 f32 scale，因为 `release_stream_slot` 在 requant 后
   会释放 BF16 原件；int8 关 = BF16；VDN 分支再加一份 `HIDDEN*INNER`）。
   与 `allocate_stream_slot()` 共用同一批常量，不会再各写一遍漂移。
2. **常驻上限再被物理内存夹一道**：`usable = min(avail − activation_reserve,
   physical − footprint − H3_MEM_HEADROOM_DEFAULT_MB)`。为此把 `h3_host_physical_memory()`
   从 h3_host.c 的 static 提为公开（`h3_host_footprint()`/`h3_host_memory_guard()` 本来就公开，
   规则与 guard 里的 `footprint + headroom > physical` 同一套）。
   动机是 F48 量到的：同形状同二进制 `available` 给出 2.6 与 5.6 GiB ⇒ 只看它等于抽奖。
3. **activation 项补上 DiT 自己的激活arena**：新增公开的 `h3_dit_activation_bytes(sequence)`
   = `sequence × 2B × (3*HIDDEN + 6*INNER)` = **118,272 B/行**（默认配置：MLP 融合开、
   attention heads 与 MLP modulation 别名进 qkv）；`h3_dit.c` 用真实 `dit->sequence` 计入。
   `h3.c`/`h3_cli.c` 仍只算了 latent（它们手上没有可信的 sequence，硬凑就是再造一处漂移源）——
   这条留在待办里和"反查"一起做。

## 2. 验收：F48 那条判据是可证伪的，两条都过了

| 用例 | 修正前 | 修正后 |
|---|---|---|
| 256²/44f/steps4/seed42 | 钉 **13** 块、DiT load peak **10.995 GiB**、去噪中 **SIGKILL(137)** | 钉 **4** 块、DiT load peak **4.535 GiB**、**EXIT=0 跑完并写出 mp4** |
| 256²/22f/steps4/seed42 | latent `6e9c2d87…`、mp4 `ac0d0940…` | **两串逐字符不变**（常驻块数 9→4 的情况下） |

新日志把决策的输入都印出来了，不再只有一个 `available`：
`adaptive DiT partial residency: 4 of 50 blocks resident (available 4.5 GiB, footprint
3.4/16.0 GiB, budget 3.3 GiB, activation reserve 1.1 GiB, per-block 0.72 GiB)`

⇒ **常驻块数不影响输出数值**这次是被实测证的（9 块与 4 块得到同一串 latent+mp4），之前只是假设。
⇒ F47 那条"修正让预留变小、方向更激进"的告警被 ③ 抵掉：activation 现在计入，预留变大。

**回归**：`make test` exit 0（48 行绿、15 条 skip、无 FAIL），编译期没有新告警。

## 3. 还没做的（按顺序）

- `h3.c`/`h3_cli.c` 的 activation 仍缺 DiT 激活项：需要一个可信的 sequence 估计
  （`video_rows = latent_t × (latent_h/2) × (latent_w/2)` + audio + 文本行，文本行数要 tokenize），
  或者把规划挪到 layout 建好之后。
- "给定字节能撑多少帧"的反查排在它后面 —— 现在正算的可信度只到 `h3_dit.c` 那一层。
- `h3_memory_plan.c:80-82` 用未夹的 `rec` 算 `free_after_stream`（F44 §2 #13）仍未动。
- 入库状态（本条写就时更正）：`d51d414` 已收走 F40/F42/F43/F45 与新测试/脚本/README/Makefile；
  F47+F49 的代码在 `0301614`，F46~F49 的记录在 `dbe42d2`。工作区干净，未 push。

## 4. 复现

```bash
# 曾经被杀的用例，现在必须 EXIT=0
H3_PROFILE=1 H3_CLIPPROJ_DIR=/Volumes/data/.lmstudio/models/Qwen3-VL-4B-Instruct-int8-convrot \
H3_CLIPPROJ_PROJ=/Volumes/data/.lmstudio/models/ClipProj-MiniMax-H3 \
  ./h3 -d /Volumes/data/MODELS/h3c-q4-adalncache --ssd-streaming \
  -p "A red fox walking through snow" --width 256 --height 256 --frames 44 \
  --steps 4 --seed 42 -o /tmp/f44.mp4
grep "partial residency" /tmp/...   # 4 of 50 ... per-block 0.72 GiB
# 逐位锚（22f）
md5 -q <latent> <mp4>   # 6e9c2d8786cea097ddf9bdd45e5593bd / ac0d0940434ad216d8131f7469154b70
```
---

# F50（2026-10-05）：给内存规划补上可信的 sequence 估计，并让它自证

F49 §3 留的那条：`h3.c`/`h3_cli.c` 的 activation 项缺 DiT 自己的激活 arena，原因是规划点跑在
layout 与 tokenize 之前，拿不到真实行数。这轮补上，并且**不给没有自证的估计**。

## 1. 新增的三个口子（`h3_dit.c`，声明在 `h3_dit.h`）

- `h3_dit_sequence_estimate(width, height, frames, text_rows_upper_bound, condition_count,
  reference_count)` —— 按 `h3_layout_build()` 的真实构成写：`(latent_h/2)*(latent_w/2)` 行/帧 ×
  `latent_t` 帧，一帧首尾关键条件再加一份 frame_rows，音频每 latent 帧 2 行（`h3_audio_grid`
  就是 `audio_t*2`），每个 reference 按**满画幅一份序列**计入。
- `h3_dit_activation_bytes(rows)` —— F49 加的 118,272 B/行。
- `h3_dit_plan_bytes(...)` = latent + activation，**两个调用点共用它**，这样" generate 侧算了
  latent、REPL 预览漏掉 latent"这种事没法再发生（我第一版就在 CLI 侧漏了 latent 项，被这条结构
  逼回来）。

文本行数：`h3.c` 有 prompt，用 `strlen(prompt)` 当上界（BPE 不会产生比字节数更多的 token）；
REPL 的 `!memory-plan` 预览没有 prompt（prompt 是每命令参数，不在 session params 里），
那里传 0 并在注释里写明它少这一项、权威检查在 `h3_generate`。

## 2. 自证：估完必须和真实 layout 对账

`h3_generate` 在 `h3_layout_build()` 成功后比对：
- **低估**（`layout.seq_len > planned_rows`）⇒ 打印 warning，因为这是唯一会让档位过于乐观的方向；
- 否则只在 `H3_PROFILE=1` 下打印 `h3: [mem] planned N token rows, layout M`，让估计平时可校准。

实测（256²/22f/t2v/steps4，int8-convrot 4B + 缓存档）：`planned 552 token rows, layout 528`
⇒ **高估 4.5%，安全方向**；同一次运行 latent `6e9c2d8786cea097ddf9bdd45e5593bd`、
mp4 `ac0d0940434ad216d8131f7469154b70` 两串逐字符不变（估的是内存，不动数值）。
`make test` exit 0（48 行绿、15 skip、无 FAIL），无新告警（剩下的 3 条 `h3_cli.c:752/754`、
`h3.c:494` 都是 HEAD 就有的）。

## 3. 边界与已知不精确（不要当成已验证）

- reference 项是**上界**：一张参考图实际只加 `frame_rows`，我按满画幅一整个序列算 ⇒ 多估数倍。
  方向安全，代价是档位可能偏保守一丁点；量化上界是每参考 ~458 行 × 118 KB ≈ 54 MB 的激活。
- **首尾关键帧 / reference 的估计路径未实测**（本机没有可跑的 i2v 锚，跑一次要图 + 另一组基线）。
  若它低估，运行时会打印上面那条 warning —— 这是有意留的兜底而不是"应该没问题"。
- 溢出检查：行数量级到 768²/360f 约 5.2 万行 × 118 KB ≈ 6.1 GB，`size_t` 无溢出；
  参考数与条件数都是小整数。

## 4. 复现

```bash
H3_PROFILE=1 ./h3 -d <cached> --ssd-streaming -p "..." --width 256 --height 256 \
  --frames 22 --steps 4 --seed 42 -o /tmp/x.mp4 2>&1 | grep "\[mem\]"
# h3: [mem] planned 552 token rows, layout 528
grep -n "h3_dit_sequence_estimate\|h3_dit_plan_bytes" h3.c h3_cli.c h3_dit.c h3_dit.h
```

---

# F51（2026-10-05）：把"这个画幅最长能出多少帧"接成反查，并让正/反规划共用同一条预算式

splash 的 `MemoryPlan.hpp` 旁边有个反向查询 `contextTokensWithin`（给定预算问最长上下文），
它的价值不在算法而在**共用规则**：正向判定和反向查询如果各抄一遍预算折扣，迟早漂。
本轮照这个形状补 h3 这边缺的另一半。

## 1. 改了什么

- `h3_memory_plan_budget_bytes(device)`（`h3_memory_plan.c:13`，新公开）：把
  `h3_memory_plan_auto()` 内部那条 `min(建议工作集×80%, (物理内存−4 GiB)×85%)` 抽出来，
  正向档位判定与反向最长查询都走它。`h3_memory_plan_auto()` 里因此不再用的 `physical`
  局部一并删掉（留着就是 `-Wunused-variable`）。
- `h3_memory_plan_frames_within()`（`h3_memory_plan.c:29`）：在 **`22 + 17k`** 这个梯子上二分
  `h3_dit_plan_bytes()`，搜索区间 `(H3_PLAN_FRAMES_CEILING − 22)/17`。连"22 帧一个训练块都
  放不下"这种情形都返回 0，而不是返回一个请求了也会被对齐改写的数。
- `H3_PLAN_FRAMES_CEILING 4051`（`h3_memory_plan.h:78`，= `22 + 17×237` = `5 + 17×238`）：
  命中它表示"至少这么多"，注释写明它只覆盖 latent + activation 的增长，**是估计不是承诺**。
- CLI `!memory-plan`（`h3_cli.c:780-805`）：在同一条 ceiling 上先扣掉流式常驻权重，再打印
  `Longest clip at WxH: N frames (+) (room of a C GiB ceiling)`；放不下 22 帧时改打印缺多少。

## 2. 新测试当场抓到一个"够不着的上限"——这是真缺陷，不是 harness 毛病

我第一版把上限写成 **4065**。`h3_align_frame_count()`（`h3_host.c:22-28`）要求
`f ≡ 5 (mod 17)`，而 `align(4065) = 4068` —— 4065 根本不在梯子上。后果是二分的最大可达答案是
4051，于是 `longest == H3_PLAN_FRAMES_CEILING` 这个"撞没撞到上限"的等式**永远为假**，
`+` 号一次都不会出现，而用户会把 4051 读成硬上限。

`tests/test_memory_plan_inverse.c:85` 的成员断言（`align(CEILING) == CEILING`）把它抓住，
失败信息就是返回的 4051。改成 4051 后全绿。**这条断言之所以有意义，是因为它检查的是
"外部世界能不能表示这个值"，而不是函数自洽。**

## 3. 实测数字（全部走真函数，不再手算）

256²/文本上界 64 行/无条件无参考：

| 反查预算 | 答案 | 正向 `h3_dit_plan_bytes()` |
|---|---|---|
| 8 GiB @ 256×256 | **3252** 帧 | 7.9707 GiB（放得下） |
| — 梯子下一级 | 3269 帧 | **8.0122 GiB（放不下）** |
| 8 GiB @ 768×432 | **753** 帧 | 7.9389 GiB（放得下） |
| — 梯子下一级 | 770 帧 | **8.1175 GiB（放不下）** |
| 64 GiB @ 256×256 | **4051 帧**（撞上限，打 `+`） | — |

测试的每个用例都是"问一次 → 用正向函数复核答案真的放得下 → 复核梯子下一级放不下"，
所以这张表不是另算一套，而是断言本身。另外三条边界也钉住了：差一字节放不下 22 帧、
恰好一份 22 帧返回 22、`plan(39)+1` 字节仍返回 39。单调性四向（预算/画幅/关键帧条件/参考图）
各一条。

## 4. 已知不精确（写进了头文件注释，别当已验证）

- 只计入 **latent + activation** 两项增长。延续 F50 的口径，参考图按满画幅整个序列算（上界），
  页缓存状态它无从知道；`h3_generate()` 仍然会拿自己估的行数和真正 build 出来的 layout 对账。
- CLI 那行刻意把文本行留成 0（prompt 是逐命令给的，不在会话参数里），所以预览是**乐观端**。
- 4051 只是让二分收敛的搜索边界，不代表任何机器跑得动这么长的片段。
- **本机 REPL 里那一行长什么样，还没眼验**：试着用 `printf '…\nquit\n' | ./h3 -d <cached>`
  喂 linenoise，进程不响应管道输入（挂住、无输出），我把它停了。这一行的正确性目前只在
  纯 C 层（同一组函数）由上面的测试覆盖。想真看一眼需要在交互终端里手打 `!memory-plan`。

## 5. 顺带的就地更正：F48 §2 的手算用错了 latent 式子

见 F48 §2 的更正块。原因是 `h3_host.c` 有两个同名近似的 helper：
`h3_video_encoder_latent_t()`（`:35-37`，`(f+3)/4`，VAE 编码器下采样）和 DiT 序列真正用的
`h3_video_latent_t()`（`:30-33`，`((f-5)/17)*5+2`）。手算抓错了那一个， rows 差算成 356
行（实际 **752** 行），顺带把 44 帧对齐后的长度记成 58（实际 **56**）。
改成实测后：22f = 528 行 / 0.0582 GiB，56f = 1280 行 / 0.1410 GiB，每块
`(2.955 − 0.083)/4 = 0.718 GiB`，与独立算出的 0.719 GiB 对上到 0.001。
`h3_dit.c:2839-2842` 那句"~0.1 GiB"的注释一并改成实测 0.141 GiB。

**教训**：这类"两个名字像的 helper"正是手算最容易踩的坑；凡是能被真函数回答的问题就别手算，
本轮的做法（把数字搬进 probe / 测试）以后固定下来。

## 6. 复现

```bash
make h3_memory_plan_inverse_test && ./h3_memory_plan_inverse_test   # EXIT=0，14 条 ok、0 FAIL
grep -n "h3_memory_plan_budget_bytes\|h3_memory_plan_frames_within\|H3_PLAN_FRAMES_CEILING" \
  h3_memory_plan.c h3_memory_plan.h h3_cli.c
grep -n "更正（2026-10-05" findings.md   # F48 §2 的更正块
```

---

# F52（2026-10-05）：档位判据从"Metal 建议值"改回"规划真正执行的那条上限"

F44 §2 #13 挂着的一条一行级缺陷，本轮结掉，并给它补上第一个覆盖档位决策的测试。

## 1. 缺陷本体

`h3_memory_plan_auto()` 里两处判据用的**不是同一个数**：
- 第一档（能不能全常驻）比的是 `h3_memory_plan_budget_bytes()` —— 建议工作集 ×80% 与
  `(物理内存 − 4 GiB) ×85%` 取小；
- 第二档（流式之后还剩多少余量 ⇒ 要不要把 DiT 深度降到 `H3_MIN_DIT_LAYERS`）比的是
  **未夹的 `rec`**（Metal 原始建议值）。

后果有方向性：小内存机器上 `rec` 通常远大于夹后上限（这台 16 GB M4 实测 `rec=20 GiB`、
上限 `10.2 GiB`），于是"余量 < 4 GiB 就削深度"这条**在最需要削的机器上永远不触发** ——
`rec − 6.5 GiB = 13.5 GiB` 看着很宽裕，而上限那边只剩 `3.7 GiB`。

改成 `target`（同 F51 抽出来的那条规则）后，两档共用一个数，注释里写清了为什么不能用 `rec`。
顺带删掉 `h3_memory_plan.h:83-86` 那段**没有对应函数**的尾注释（"recommended_working_set * 7/8"）：
`grep` 全仓，这个名字只在头部注释里作为 ds4 出处出现，没有实现 —— 它教的是和本仓实际规则
相反的算法（既不是 7/8，也没夹物理内存），按"文档教过期配方等同于缺陷"处理。

## 2. 新测试 `tests/test_memory_plan_tiers.c`（此前档位决策**零覆盖**）

`grep -rn h3_memory_plan_auto tests/` 之前无命中 —— 也就是说这两档的取数规则一直没被测过，
所以缺陷能长期活着。测试 13 条断言，其中最关键的一条是**用例自证前提**：

> `room_by_rec >= 4 GiB && room_by_target < 4 GiB`

即"这个用例真的横跨阈值"。没有这条，"削深度"的断言可能只是两条规则给出同一答案时的假绿。
其余覆盖：流式档的开关组合、余量宽（4.7 GiB）时保持满深度、能全常驻时不碰任何旋钮、
`rec == 0` 时留默认，以及**rationale 里打印的 GiB 数必须等于 `h3_memory_plan_budget_bytes()`**
（wiring 类断言：防止以后改成"打印一个数、按另一个数决策"）。

## 3. 变异对照：证明这条断言确实在测这条规则

按单文件变异流程（先 `grep` 前置：待替换文本基线出现 1 次、变异文本基线出现 0 次；
每次重建前 `rm` 掉 `.o` 与测试二进制；变异/测量/还原在同一次调用里；还原后比 sha256）：

| 手臂 | 结果 |
|---|---|
| 变异：把 `target` 换回 `device->recommended_working_set` | exit=1，**恰好 1 条变红**："tight against the enforced ceiling trims DiT depth" |
| 基线（还原后 sha256 一致） | exit=0，0 条红 |

红只落在那一条上，正是它该有的形状 —— 其余 12 条在两个手臂下都同色，说明它们不是这条规则的
覆盖来源。另一侧（`rec=2 GiB` 小于物理夹后上限）在变异下仍绿，因为两条规则在那个方向给出的
余量都低于阈值，测试里没有把它当成"证明"。

## 4. 顺带记下、本轮没动的一处

`h3.c:1465-1467` 用 `bytes * 0 + ...` 表达"这个分量每次调用后释放"。能编译、语义正确（结果恒 0），
但写法比直接不加大项更绕，且 `-Wextra` 不会提醒它。留作观察项，不单开一步。

## 5. 验收

- `./h3_memory_plan_tiers_test`：EXIT=0，**13 条断言全绿**（含那条自证前提）。
- 变异对照：exit=1 且**恰好 1 条**变红，基线还原后 sha256 一致（见 §3 表）。
- `make test`：EXIT=0，63 行 `  ok  `（比上一轮 +13，即新测试全量进闸门）+ 11 行 `ok…` 收尾、
  15 skip、0 FAIL。该次日志里没有 `warning:` 行 —— 这**不等于**全仓无告警：本轮只重编了改动
  涉及的文件，`h3.c:494` 那条 `unused parameter 'conditioning_key'` 是 HEAD 就有的、没被这次
  构建触及，仍按"原有告警"记账。另外单独 `make h3` 重编过 `h3_cli.c`，给出 2 条整型→浮点转换，
  本步没让它变多；"原有"这次是查出来的而不是沿用记忆：`git show 0301614:h3_cli.c` 单独
  `-fsyntax-only` 编出来的还是 `:752`/`:754` 同两处、同行号。
- 逐位锚：`--ssd-streaming` 走的路径其实**不经过**规划器（`h3.c:1436-1437` 的门是
  `memory_plan_auto && ssd_streaming == 0 && use_int8_row_fc2 == 0`），但按"改内存决策类代码就
  回锚"的规矩仍复跑一次 256²/22f/steps4/seed42 ⇒ EXIT=0，latent
  `6e9c2d8786cea097ddf9bdd45e5593bd`、mp4 `ac0d0940434ad216d8131f7469154b70` **两串逐字符不变**
  （常驻仍 6/50 块）。这条是保险，不是证据来源 —— 证据是上面那条变异对照。

## 6. 复现

```bash
make h3_memory_plan_tiers_test && ./h3_memory_plan_tiers_test   # EXIT=0，13 条 ok
grep -n "free_after_stream" h3_memory_plan.c
grep -Fn "7/8" h3_memory_plan.h    # 计数应为 0（那段注释已删）
```

---

# F53（2026-10-06）：给装早了的 AdaLN 缓存就地补 meta 键，P1 的收窄第一次在真文件上生效

F43 把 C 侧判据从"有缓存就拒一切适配器"收窄成"只有真打 AdaLN 的适配器才拒"，但收窄靠
`adaln_cache_meta_s{steps}` 报宽度；`h3c-q4-adalncache` 这三份是分片键存在**之前**装的
（`weight_time_dim=None`、`source_time_dim=None`，实测），所以实际行为仍是一刀切。
F52 之后的待办就是补这三个键。

## 1. 改之前的真实行为（不靠推理，直接跑）

用 `tests/gen_adaln_cache_fixture.py` 生成的三个夹具适配器打在**真已装分片**上跑 `./h3 -d ... --lora`：

| 适配器 | 目标 | 补键前 | 补键后 |
|---|---|---|---|
| `adapter_attn_only` | `attn.orig.to_q` | 「has no adaln_cache_meta_s4 width key …」一律拒 | `note: none of the 1 LoRA adapter(s) target AdaLN, so the cached AdaLN modulation for 4 steps is used unmodified`，并走到 `precompute AdaLN 50/50` |
| `adapter_adaln` | `transformer_blocks.0.adaln_proj.linear` | 同上（说不清） | `LoRA adapter 1 of 1 carries an AdaLN factor for a **2688**-wide input …` |
| `adapter_norm_out` | `norm_out.linear` | 同上 | 同 `2688` 那条 |

补键前三个都是同一句"无宽度键 ⇒ 全拒"，这正是 P1 的失效面；补键后两条路径按预期分叉，
且拒绝句里的 2688 只能来自我写进去的那个键 —— 也就是说 C 侧真的读到了。

## 2. 宽度不是写死的

2688 由 `weight_time_dim()` 从**这批缓存导出时所用的那棵树**读来
（`h3c-q4-native` 的 `blocks.0.adaln_proj.linear.weight` shape[1]；`h3c-official` 同值），
脚本拿不到就 `SystemExit`，不猜。第三处独立吻合：`gen_adaln_cache_fixture.py:32` 早写着
`TIME_DIM = 2688`，而夹具的 `block_output` 宽 96,768 = 36 × 2688、`final_output` 10,752 = 4 × 2688。

## 3. 磁盘改动只做一件事：头部变长

新张量的 4 字节追加在数据段**末尾**，所以既有 52 个张量的相对偏移一个都不变，载荷字节整段照抄。
每份净 +92 B（s20 头部 5,192→5,280、载荷尾 +4），三份都自动复核：
按备份算 SHA-256 区间摘要 ⇒ **载荷逐字节不变**（`51ffd1cfe389..` / `874fb8848a34..` /
`d24018d8e87e..`），再回读 meta 得 2688。原文件先 `copy2` 到
`/Volumes/data/tmp/h3scr/f53/backup/`（619 MB）才 `rename`，写的是同目录 `.tmp` 再原子换。
工具落在 `fastvideo_qad/scripts/add_h3_adaln_cache_meta.py`：留着它的理由是
"没有它就只能把三个步数各重跑一遍去重新 dump"。

## 4. 端到端与保留意见

- 上表的 `adapter_attn_only` 越闸之后**停在形状不符**：夹具那张 `lora_A` 是 `[16, 2688]`，
  真 to_q 要 `[16, 5376]` —— 夹具本来就只为 AdaLN 尺寸造的，不是本步引入的问题。
- 所以另造了一个**形状正确、值为零**的 rank-16 to_q 适配器（`[16,5376]`+`[7168,16]` BF16 全零）
  打在缓存档上跑完整 256²/22f/steps4/seed42，与同一轮的不带适配器那臂配对：

  | 臂 | 结果 |
  |---|---|
  | 无适配器 | EXIT=0，latent `6e9c2d8786cea097ddf9bdd45e5593bd`、mp4 `ac0d0940434ad216d8131f7469154b70` —— **补键之后两串逐字符不变** |
  | 零值真形状适配器 | 先出 `note: none of the 1 LoRA adapter(s) target AdaLN …`，EXIT=0，**同样两串** |

  也就是说"非 AdaLN 适配器 + 缓存档端到端跑通"这次是真跑通的，而且合并零不移动任何权重
  这条性质顺带被同一对数字证明（合并路径本身没有把 BF16 权重绕一圈改掉）。
- 三份之外的缓存没扫（本机 `find` 只命中这三份）；工具**可重入**：再跑一次报
  `already carries adaln_cache_meta_s4 = 2688; left alone`。
- 备份留在外置卷 `/Volumes/data/tmp/h3scr/f53/backup/`（619 MB），在仓库外所以不进 `.gitignore`，
  要清理由你一句话。
- 新工具**还没有测试**：它改的是用户模型文件，值得一条"改头不动载荷 + 无宽度源就拒"的夹具测试
  （`gen_adaln_cache_fixture.py` 已经有 `meta: bool` 这一维），本轮没做，记进待办。

## 5. 复现

```bash
FIX=/Volumes/data/tmp/h3scr/f53/fixtures/adapters     # 由 gen_adaln_cache_fixture.py 生成
env H3_CLIPPROJ_DIR=... H3_CLIPPROJ_PROJ=... ./h3 -d /Volumes/data/MODELS/h3c-q4-adalncache \
  --ssd-streaming -p "A red fox walking through snow" --width 256 --height 256 \
  --frames 22 --steps 4 --seed 42 --lora $FIX/adapter_attn_only.safetensors   # 应见 note 行
python3 fastvideo_qad/scripts/add_h3_adaln_cache_meta.py \
  --cache-dir /Volumes/data/MODELS/h3c-q4-adalncache/FL2VA/transformer \
  --source-model /Volumes/data/MODELS/h3c-q4-native/FL2VA/transformer \
  --backup-dir /Volumes/data/tmp/h3scr/f53/backup --dry-run                   # 先看字节账
```
