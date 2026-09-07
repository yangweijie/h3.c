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
