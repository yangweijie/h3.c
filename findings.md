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
