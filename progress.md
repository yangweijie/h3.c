# Progress Log

## 2026-09-01 — fork 流式生成验证（无 lora）

### 20260901 11:13:50.45 — run5 进行中
- 进程存活 PID 78318，`STAT=UN`（I/O 等待，符合流式特征）。
- `denoise 1/4`，错误计数 0。
- 命令：纯 BF16 流式（`--ssd-streaming` + 三处 `--use-slower-bf16-*` 关 int8）。

### 20260901 11:16:40.03 — run5 完成
- 进程正常退出（PROCESS GONE）。
- log tail：`video VAE load 36/36` → `FFmpeg 22/22` → `h3: wrote /tmp/h3out/fork_stream_nolora_256.mp4`。
- 错误计数 0；产物 122408 bytes。

### 20260901 11:16:47.47 — ffprobe 验证
- `codec=h264`，`width=256 height=256`，`avg_frame_rate=24/1`，`duration=0.916667s`，`nb_frames=22`。
- 含 `aac` 音轨（FL2VA 自带）。
- **结论：fork 不接 lora 流式生成 ✅ 成功。**

### 修复回顾
- **P1 (`h3.c`)**：int8/流式互斥误判 → 仅拦 `use_int8_row_fc2` 与 `ssd_streaming`。
- **P2 (`h3_dit.c`)**：流式 read 后无条件 `h3_gpu_submit` → int8 关时跳过（见 findings.md "Bug B"）。

### 下一步
- **P4**：验证默认 int8 开启的流式路径是否同样跑通。
- **P5**：提速优化（砍步数/帧数、MPS 预热、低像素+超分），待用户拍板。

## 2026-09-01 — P4：默认 int8 流式路径（run6）

### 20260901 11:30:45 — P4 完成（run6, PID 80159）
- 进程正常退出（PROCESS GONE）。
- log tail：`video VAE load 36/36` → `FFmpeg 22/22` → `h3: wrote /tmp/h3out/fork_stream_int8_nolora_256.mp4`。
- 错误计数 0；产物 122408 bytes。
- `cmp` 对比 P3/P4 mp4：**逐字节相同（IDENTICAL）** → P4 与 P3 走同一 BF16 计算路径，int8 未启用。
- ffprobe：256×256 h264+aac 24fps 0.92s 22 帧，合法。

### 根因（int8 为何没跑）
- `h3_gpu.m:364-373`：`m5 = [device.name rangeOfString:@"M5"]`；`wantsTensorOps = m5 && (!nax||!*nax||strcmp(nax,"0")!=0)`；
  `tensorOpsEnabled = gpu.library && wantsTensorOps`。
- 本机 GPU 设备名不含 "M5" → `tensorOpsEnabled=false` → `h3_dit.c:1749-1755` 的 `int8_mlp/qkv/attention_out` 全为 0。
- 故即便不传 `--use-slower-bf16-*`，streamed-block int8 也绝不启用，默认 int8 流式路径退化为 BF16 流式。

### P4 结论
- 默认 int8 流式路径在本硬件**退化为 BF16 流式并通过**（无错、产物合法）。
- 真正的 streamed-block int8 + 流式量化/submit 路径（P2 修复的代码）**需 M5 类 GPU 才能端到端验证**，当前硬件无法覆盖。
- 本机可稳定验证的仅是 BF16 流式生成（P3/P4 一致）。

## 2026-09-01 — P5：提速优化诊断

### 20260901 11:42:02 — 模型卷识别
- `df` / `diskutil`：`/Volumes/data` = **USB SSD**（Protocol: USB, Solid State: Yes, HFS+），814 MB/s。
- 内部盘 `/`（`/dev/disk3s1s1`）仅 43Gi 空闲，**装不下 134GB 模型** → 无法搬去更快存储。

### 20260901 11:46:06 — steps=2 实测
- 耗时 **214s**（steps=4 基线 ~240s）→ 砍 2 步仅省 ~26s。证实步数非瓶颈。

### 20260901 11:46:06 — H3_PROFILE 流式读精确数据
- `h3: BF16 SSD stream 72.495 GiB read in 94.376s (0.768 GiB/s), unhidden wait 78.894s`
- 即：每代流式读 72.5 GiB @ 0.768 GiB/s = 94s；其中 **78.9s 是计算端干等盘的 I/O 停滞**。
- 推论：USB SSD 读速是硬上限；每去噪步 ~47s I/O 不可避免（16GB RAM 装不下 36GiB/步 做 page-cache 跨步复用）。

### P5 结论（本硬件）
- 软件杠杆（砍步数/帧数/分辨率、MPS 预热、core_reuse、token_reduction）收益极小，因瓶颈是 I/O 而非算力。
- 真正提速需硬件：更快模型盘（Thunderbolt/USB4 SSD）、或 M5（int8 减半读量）、或 >64GB RAM（常驻去流式）。
- 唯一可做的代码杠杆：更深流式预取回收部分 79s unhidden wait（~30-40% 潜力，改流式核心有风险），待用户拍板。

## 2026-09-01 — ModelScope 量化权重调研 + P8 立项

### 调研（`Abiray/Minimax-H3-nvfp4-INT4-INT8-Convrot`）
- 覆盖 DiT + 文本编码器(Qwen3-VL-32B) 的 INT4/INT8/Mixed-Convrot 与 nvfp4；**VAE 未量化**（视频 FP16 5.21GB / 音频 FP32 605MB）。
- 体积：文本编码器 INT8-Convrot=27.1GB（BF16 62.13GiB 的 ~2.3× 压缩）；面向 NVIDIA，无 Apple Silicon 验证。
- 硬约束：nvfp4=NVIDIA Blackwell 出局；INT4=U8 打包需全新 kernel；INT8-Convrot 需放宽 M5 门控+Convrot 逆旋转+扩 int8 到 norm/adaln+匹配 pruned 块数；Convrot 旋转矩阵未单列文件。

### 决策
- 文本编码器维持 **4B ClipProj**（5.5GiB，比消费 INT8-32B=27GB 小 5×，已落地）→ P8 重心放 **DiT INT8-Convrot**。
- VAE 体积（9.70GiB）本仓库不覆盖，仍靠自加量化/流式。
- 顺带澄清：用户提的 "M4 支持 int8" 是 **CPU** ARM v8.2；h3.c 跑 **GPU(Metal)**，相关的是 MPS GPU int8（mps-bitsandbytes 佐证 M4 支持）；而 h3.c 的 "M5" 门控是字符串启发式非能力探测，可能过严。

### P8 立项（task_plan 新增）
- 第一步：写 `dbg_parse_safetensors.py` 解析 INT8-Convrot header 取证（key/dtype/scale 布局/旋转矩阵/块数）。脚本已创建。

## 2026-09-01 — FL2VA DiT `pruned_int8_convrot` = 19.5 GB（P8 体积修正）

- 用户实测：pruned_int8_convrot 格式的 FL2VA DiT 模型大小 **19.5 GB**（非早先估的"int8 约减半"）。
- 相对 BF16 61.73 GiB (≈66.4 GB)：19.5 GB ≈ **29%，即 3.4× 更小**。
- 修正认知：19.5 GB 不是单纯 int8（应≈33GB），而是 **pruning（减块）+ int8（2×）叠加** → 3.4×。
- 对全模型：仅 DiT 一项 61.73→18.2 GiB（19.5GB÷1.0737），整模型 134→~90 GiB（若仅 DiT 走此格式）。
- 已更新 task_plan P8 与 findings.md「ModelScope 量化权重可消费性」节。
- 仍待办：用 `dbg_parse_safetensors.py` 实际解析该文件 header，确认块数/旋转矩阵/scale 布局，作为 P8 开工硬证据。

### 下载执行（unfetch MCP，2026-09-01）
- 4 个下载任务已提交（save_dir `/Volumes/data/.lmstudio/models/Minimax-H3-Quantized`）：
  - 主文件 `MiniMax_H3_FL2VA_pruned_int8_convrot.safetensors`（19.5GB，SHA256 校验）→ 任务 `fb3b8782-1c64-4625-aeca-5579fde53553`，downloading ~42.6MB/s，ETA ~8min。
  - `README.md`/`configuration.json`/`ref2.json` → 已 done。
- 元数据分析：`configuration.json` 仅框架标记；`ref2.json` 是 ComfyUI 工作流（无块数/旋转矩阵）；**文本编码器不在本仓库**（README 提及的 qwen3vl_32b_* 未出现在文件树）→ 继续用 4B ClipProj。
- 关键修正：INT8 DiT 面向 24GB 级 dGPU；16GB Mac 统一内存下即便 INT4 也装不下（DiT+VAE+文本 ≈ 26GiB）。故 P8 在 16GB Mac 的真实价值是**流式读盘更少**（19.5GB vs 61.7GiB，3.4× 削减 P5 的 I/O 瓶颈），而非塞进 RAM。
- 下一步：主文件下完后跑 `dbg_parse_safetensors.py` 解析 header（块数/Convrot 旋转矩阵/scale 布局），作为 P8 开工硬证据。

### 头部分解结果（2026-09-01 实测，`dbg_parse_safetensors.py`）
- 文件 20970379688 bytes（完整下载），932 张量，≈19.5 GiB。
- dtype：I8 17.94 GiB（主体）+ BF16 1.49 GiB（norm/adaln/embeddings）+ F16/F32/U8 微量。**结论：仅每块 4 个 matmul 被 int8 量化，其余 BF16** → h3.c 只需换 4 matmul 为 I8+scale，无需扩大 int8 覆盖。
- **DiT 块数=50**（min0/max49）↔ `H3_DIT_BLOCKS=50` **对齐**，无块数改动。
- scale：200 个 `weight_scale` F32 `[out,1]`（50×4），逐输出通道对称量化，无 zero-point；全部 I8 权重均有 scale。
- **P8 #1 阻塞**：`rot`/`convrot`/`hadamard`/`ortho`/`zero` key 命中数全 0 → **无独立旋转矩阵**。需查量化脚本/ComfyUI 节点源码确认 Convrot 旋转配方（疑为参数无关 Hadamard）；否则权重在旋转基下无法直接用。
- 样例 I8 `blocks.0.attn.out_proj.weight` shape `[5376,7168]`（in=7168≠常见 hidden 5376），提示 convrot 改变有效宽度/转置，h3.c 前向需按文件实际 shape 参数化。

### ConvRot 旋转配方已破解（2026-09-01 检索）
- "convrot" = 已发表 **ConvRot: Rotation-Based Plug-and-Play Quantization for DiTs**（arXiv 2512.03673，清华&华为，2025-12），官方 `github.com/feice-huang/ConvRot`，并有 `MiniMax-H3-W4A8-ConvRot` 项目 → 本仓库 `pruned_int8_convrot` 由该法产出。
- 核心是 **Regular Hadamard Transform (RHT)**：旋转矩阵固定、参数无关（±1/√n），**无需存储** —— 完美解释头部分解中 `rot/convrot` key 为 0。
- **P8 #1 阻塞解除**：旋转可自生成，不必依赖 side 矩阵。剩余工程任务 = 在 h3.c DiT 前向复刻 ConvRot 的 RHT 分组与应用点（对激活施 R^T）。
- 下一步：读 ConvRot 官方实现 / MiniMax-H3-W4A8-ConvRot，提取"哪些张量、沿哪个 dim 分组、放在块哪个位置"的具体方案，作为 h3.c 集成蓝图。

### ConvRot 确切方案与 h3.c 蓝图（2026-09-01 检索+推导）
- 官方 README（feice-huang/ConvRot）：旋转 = Regular Hadamard Transform，`rot_size=256`（K 须被整除）；权重离线预旋转+量化（磁盘即 W·R），激活在线旋转；跳过 `x_embedder` / `time_text_embed`（保 BF16）。与头部分解吻合（attn q/k/v/out + mlp fc1/fc2 为 int8，norm/adaln/embed 保 BF16）。
- 数值等价 Y=(X·R)(W·R)^T=X·W^T；消费法 = 加载时反旋转 W=W_stored·R（R 对称）走标准路径。
- h3.c 蓝图：按层流式读 int8 → `W_bf16=(scale*W_int8)·R_K`（R_K=沿 K 维块 256 分块对角 Hadamard）→ 复用现有 BF16 前向。保 19.5GB 流式收益，仅加 int8→BF16 反量化 + 分块 Hadamard 两预处理算子，无需新 int8 Metal kernel。
- 待核对（实现前）：权重存 W·R 还是 R·W；激活旋转方向；token_refiner 是否旋转；Hadamard 归一化。由读 `ops.py`/`hadamard_rotate` + MiniMax-H3 节点钉死。

### h3.c 已内建 DiT int8（2026-09-01 代码核查，重大修正）
- `h3_dit.c`/`h3_gpu.m` 实证 h3.c **早已实现 DiT int8**：`h3_dit_block` 含 `qkv_int8/qkv_scales/out_int8/out_scales/fc1_int8/fc1_scales/fc2_int8/fc2_scales`；`h3_gpu_quantize_weight_int8` 逐行对称量化(BF16→int8+逐行 F32 scale)；Metal kernel `h3_linear_int8_*`/`h3_fc1_swiglu_int8_*`/`h3_qkv_project_split_int8_*` 等，`run_block` 直接消费 int8 字段。
- ConvRot `int8 + weight_scale[out,1]` 与 h3.c 内部 int8+逐行 scale **同构** → 之前"h3.c 零 int8、P8 需从零造 kernel"判断作废。
- P8 重写：直接读 ConvRot int8+scale 灌入既有 int8 字段 + 补唯一新算子 **Hadamard 反旋转 R_K(块256)**（dequant→W·R_K→复用 `h3_gpu_quantize_weight_int8`）。磁盘 19.5GB 流式 + 运行期 int8 显存减半双重收益。R 约定由单 block 数值对齐验证。
- 结论：P8 由"造 int8"降级为"接 int8 + 一个 Hadamard 反旋转 + 数值校验"。

## 2026-09-01 — 4B vs 50 层 计时对比（P7）

### 编码器隔离计时
- 4B（`test_clipproj_encoder`，"A red fox walking through snow" 6 token）：**7.93s** wall / 0.30 GPU s / 5.4 GiB。
- 50 层（`h3_real_prompt_test`）：**52.41s** wall / 1.84 GPU s / 46.86 GiB。→ 4B 编码器快 **6.6×**，权重少 **8.7×**。

### 完整生成计时（256×256 / 16 帧 / steps 4 / ssd-streaming，同 DiT/VAE 设置）
- 4B（设 `H3_CLIPPROJ_DIR`）：**251.25s**（日志确认 `text encoder (clipproj)` + 25 层，产出 `clipproj_cmp_4b.mp4`）。
- 50 层（`env -u H3_CLIPPROJ_DIR -u H3_CLIPPROJ_PROJ` 强制）：**292.80s**（确认 `text encoder 0/50 ... 50/50`，产出 `clipproj_cmp_50.mp4`）。
- 差值：4B 省 **~41.5s（≈14%）**。

### 陷阱复盘
- 第一次"50 层"完整生成因 `H3_CLIPPROJ_DIR` **持久导出**而实际仍是 4B（251.05s，日志 25 层暴露）。重跑 `env -u` 得真 50 层 292.80s。手动对比必须 `unset`。

### 结论
- 换 4B 省的是编码器阶段（52→8s）与权重体积极大（46.86→5.4 GiB），但端到端仅 ~14%。瓶颈在 DiT/VAE/流式 I/O（`sys` 52–61s），steps=4 时编码器占比已小；steps=20（生产）时占比更小。换 4B 价值在**内存/启动延迟**，非总时长。

## 2026-09-02 — P8.1：ConvRot qkv 行置换修复（flat 输出根因）

### 诊断
- convrot INT8 模型（`MiniMax-H3-Convrot`）走 `--ssd-streaming` 生成画面**全灰**，像素 std≈6.84（base 同设置≈37.76）。
- 离线 numpy 验证：convrot `blocks.0.attn.qkv_proj.weight` I8 → dequant → 块 256 Hadamard 反旋转 → 与 base BF16 qkv 做 per-slot cosine best-match。**formula B**（`dst=3*(s%56)+s//56`）命中 168/168 slots；**formula A**（`dst=(s%3)*56+s//3`）仅 2/168。→ 正确置换是 B。
- 关键陷阱：仅改 GPU kernel（`h3_shaders.metal`）后重跑 std 仍恰好 6.84 → 暴露本机流式实际走 **CPU unrotate**（`h3_dit.c` `convrot_unrotate_cpu`），GPU kernel 改动对输出零效果。

### 修复（三处 remap 统一为 formula B）
- `h3_shaders.metal` `h3_weight_dequant_unrotate_int8` `layout==1` → B（非流式 GPU 路径）。
- `h3_dit.c` `convrot_unrotate_cpu` `layout==1` → B（**流式路径，真正生效处**）。
- `h3_shaders.metal` `h3_convrot_remap_qkv_bf16` `layout==1/2` → B（refiner BF16，本已为 B）。
- 配套：`h3_dit.c:22` `HEAD_DIM=128`；qkv [21504,5376]=168×128；CPU 路径注释同步更正（原注释描述 formula A 语义）。

### 验证（grep 产物 / ffprobe）
- `convrot_cpu_fixed.mp4`（steps 4 / 256×256 / ssd-streaming / 三处 `--use-slower-bf16-*`）：像素 std **6.84 → 51.64**；逐帧 std ~51.6 稳定、帧间差异 2.33 → 连贯视频，非灰屏/噪声。
- 比对序列：base_20=37.76 / convrot_20=8.58 / convrot_s4=7.12 / convrot_fixed(仅改GPU)=6.84 / convrot_cpu_fixed(GPU+CPU都改)=51.64。
- 构建：`make CC="xcrun clang"`（因改了 `h3_dit.c` C 代码；改 metal 文件本身不需重编，运行时从 CWD 读）。

### 剩余（P8 收尾）
- base steps=4 同条件对照（确认亮度差源于步数，非置换）。
- 全量 `--steps 20` parity + 单 block 数值 cosine 校验固化 R 约定（W·R vs R·W）。

## 2026-09-02 — 最小物理占用分析 + 系统盘 I/O 优化

### 20260902 — 最小物理占用（P9）
- 确认 4B + INT8 DiT 的最小磁盘占用：**39 GiB**（DiT 20 + video_vae 9.7 + audio_vae 0.56 + tokenizer 0.01 + 4B文本BF16 8.3 + ClipProj 0.48）。
- 实测拒绝 INT8 4B 文本：`H3_CLIPPROJ_DIR` 指向 `Qwen3-VL-4B-Instruct-int8-convrot`（4.5G）→ `weight model.language_model.layers.0.self_attn.q_proj.weight has dtype/rank I8/2, expected BF16/2`，2s 退出。结论：4B 文本必须 BF16 8.3G。
- 评估 ModelScope `Gluttony10/MiniMax-H3-INT8-CONVROT` 的 INT8 VAE：`video_vae 4.85G`（=9.7G 半价→INT8）被 `h3_weights.c:154` dtype 检查拒绝（VAE 走 `load_f32`→`H3_DTYPE_F32`，I8≠F32）；`audio_vae 0.56G` 与本地同大小（非 int8，无省空间）。结论：两个 INT8 VAE 均不可用。
- **下限 39 GiB 不可突破**（除非改引擎为 VAE/文本加 int8 反量化分支）。

### 20260902 — 系统盘 I/O 优化（P10）
- 动机：验证 "瓶颈是数据盘 I/O" 假设。系统盘 `/`（Macintosh HD APFS SSD）起初仅剩 3.9–39Gi，不能全量复制。
- 混合版：复制 DiT+VAE+tokenizer（30GiB）到 `/Users/jay/h3_sys/MiniMax-H3-Convrot`；4B 文本/ClipProj 留数据盘。遇 rsync 2× 临时文件峰值撑爆盘 → 改 `cp -R -L`；补 `transformer/config.json` 与 `FL2VA/text_encoder` symlink。→ 跑通，**94s**（数据盘 153s）。
- 全模型版：系统盘空间回升至 17Gi 后，复制 4B 文本(8.3G)+ClipProj(0.48G) 入系统盘。→ **91s**。
- 完整对比：50层+数据盘 194s / 4B+数据盘 153s / 4B+系统盘(仅DiT/VAE) 94s / 4B+全模型系统盘 91s。SSD 流读 0.590→1.021 GiB/s，纯等盘 93→43s。
- 产物校验：256×256/22帧/h264+aac/0.92s，像素 std=26.32（与数据盘 4B 版一致，非 flat）。
- **结论**：瓶颈假设证实；全模型系统盘 91s 为当前最优。4B 文本位置影响仅 3s；GPU compute 下限 ~70s 是硬约束。

## 2026-09-02 — P11 单 block 校验 + base steps=4 对照 + P12 预取调研

### P11.1 单 block 数值校验（脚本 `dbg_block_parity.py`，新建）
- **踩坑**：第一版用 Sylvester 自然序 Hadamard 矩阵，四张量 cosine 全 ≈0.06，且旋转后比不旋转更差（0.0039 vs 0.062）→ 诊断信号指向"矩阵形式错"。
- **根因**：`h3_dit.c:654-661` 注释明示引擎用 **radix-4 ConvRot butterfly**（stride 1/4/16/64，quad→`(a+b+c-d, a+b-c+d, a-b+c+d, -a+b+c+d)`，×1/16），"**NOT** the natural-order Sylvester Hadamard"。改用 butterfly 后：
  | 张量 | 最佳假设 | cosine |
  |---|---|---|
  | qkv | butterfly(K) + permB | **+0.999961** |
  | out | butterfly(K) | +0.999928 |
  | fc1 | butterfly(K) | +0.999961 |
  | fc2 | butterfly(K) | +0.999956 |
- **钉死**：R 沿 K（输入）维、rot_size=256、radix-4；沿 out 维仅 0.004（错）；qkv 必须 formula B 置换（无置换 0.0008）。convrot int8 = base 的忠实量化（cos 0.99996）。

### P11.2 base steps=4 对照（PID 57371，已完成）
- 产物 `/tmp/h3out/base_4b_s4.mp4`（135288 B，256×256/22帧/h264+aac）。
- 像素 std **55.77**（逐帧 55.5–56.0）。convrot_4 = 51.64 → 同 steps 差仅 7.4%。
- base_20 = 37.76 → steps 4→20 使 std 降 ~18。**结论：先前 std 差异主因是步数，非量化/置换**。

### P11.3 convrot steps=20（PID 63336，**已完成**）
- 产物 `/tmp/h3out/convrot_4b_s20.mp4`（29576 B，256×256/22帧）。
- 像素 std **37.40**（逐帧 35.83–44.84，avg 37.30），mean 248.2。
- **parity 达成**：base_20 std = 37.76 → 差异仅 **0.96%**。
- mean 248 符合 prompt 语义（"a red ball bouncing on a **white floor**"，白地板占画面主体）；steps 4→20 使 mean 从 153→248、std 从 ~55→37，即去噪更收敛、浮现出白地板与红球结构。
- **P11 结论**：convrot INT8（P8.1 修复后）与 base BF16 在 steps=4（差 7.4%）与 steps=20（差 0.96%）下均高度一致 → **量化路径数值正确，P8 flat 问题彻底解决**。

### P12 预取调研（结论：收益点在 VAE，不在 DiT）
- DiT（`h3_dit.c:2710-2804`）**已有** depth=1 双缓冲流水线（`stream_slots[2]`，算 N ‖ 读 N+1）；I/O 是吞吐瓶颈（71s vs compute 47.3s）→ 加深预取对稳态吞吐无改善。
- VAE（`h3_video_vae.c:594-630` `run_stream_tile`）**完全串行无流水线**：`load→run→free` 逐块，I/O 与计算零重叠；且流式模式每 tile 重读 9.7 GiB。
- **下一步**：给 `run_stream_tile` 加双缓冲预取（load(N+1) ‖ run(N)），预期回收 ~9.5s（多 tile 则×N）；风险低于改 DiT 核心。理论下限 ~71s（当前 91s）。

### P12 实施与 A/B 实测（已完成）
- **实施**：`h3_video_vae.c` 加 `vae_prefetch_job`/`vae_prefetch_thread` + `H3_VAE_PREFETCH` 开关；`run_stream_tile` 改双缓冲（prime block0，循环中后台加载 `blocks[N+1]` 到自身槽位，`run_block` 零改动，+0.27 GiB）。编译通过（lint 0 错误）。
- **DiT 加深预取：判定无收益，不实施**（稳态吞吐 = min(compute, io)，I/O 已瓶颈 → 加深不改吞吐；避免动刚修复的 `h3_dit.c` 核心）。
- **A/B 实测**（steps=4，全模型系统盘，convrot，4B ClipProj，同参数）：
  - `H3_VAE_PREFETCH=0` → **91.78s**
  - `H3_VAE_PREFETCH=1` → **90.99s**
  - 收益 **0.79s（0.86%）**
- **正确性**：两产物 std 26.3232 / mean 251.1122 / 29020 B **完全相同** → 无回归。
- **结论**：收益远低于预期（VAE I/O 非瓶颈：单 tile + page cache + 计算主导）。**预取不是瓶颈解法，I/O 带宽才是**。保留代码（默认开，高分辨率/多 tile 场景收益会放大）。
- 真提速只剩：更快存储（TB/USB4）→ I/O 71s→~25s；更大内存常驻；继续减读量；降 steps/分辨率。

## 2026-09-02 — P13 加载器 dtype 转换分支（突破 39 GiB 下限）

### 调研（含更正）
- 4B INT8 文本（4.53 GiB）：I8+`_scale`[out,1]，对拍 BF16 → cos(dequant) 0.062 / cos(unrotate) **0.99996** → 必须 radix-4 反旋转；q/k/v 分离无需 formula B。
- **VAE 更正**：ModelScope "INT8-CONVROT" 包的 video_vae 实为 **FP16**（560 张量全 F16，4.85 GiB = 本地 F32 10.4G 的一半），需 F16→F32 加宽而非 int8 反量化。

### 实现（h3_weights.c，编译通过）
- `load_int8_dequantized`（dequant + ConvRot radix-4 反旋转，`H3_INT8_UNROTATE=0` 可跳过）+ `load_f16_as_float`（f16→f32 加宽）；`load_tensor` 的 dtype 检查为两类放行并分流。

### 验证（steps4 / 256×256，22 帧产物）
| 配置 | 耗时 | std | mean |
|---|---|---|---|
| BF16-text + F32-VAE（基线） | 90.99 s | 26.3232 | 251.1122 |
| INT8-text + F32-VAE | 94.53 s | 25.0403 | 251.4139 |
| BF16-text + FP16-VAE | 91.19 s | 26.3390 | 251.1147 |
| **INT8-text + FP16-VAE（最小）** | 93.85 s | **25.0432** | 251.4139 |

- FP16 VAE 近乎无损（+0.06%）；INT8 文本 −4.9%；误差可加无交互劣化。
- **新下限 ≈30.0 GiB**（原 39 GiB，省 9 GiB / 23%）。
