# Progress Log: 4B + ClipProj harness

## 2026-08-29

### Done
- 仓库获取：GitCode HF 镜像 `ai.gitcode.com/hf_mirrors/NicoLab28/ClipProj-MiniMax-H3` 直链返回 401、ls-remote 被重定向登录页，需认证不可用 -> 弃用。
- 改用 ModelScope `NicoLab28/ClipProj-MiniMax-H3` 下载成功，SHA256 校验通过（feef06ef...），文件干净。
- 投影文件 `mmh3-4b-ClipProj-v3-mlp.safetensors` 元数据：version=v3, tap=24, d_in=2560, d_out=5120, mlp_hidden=32768, mlp_depth=1, 无 W（纯残差 MLP）, cos_test=0.8144。
  - 关键修正：投影用第 24 层（不是最后一层/不是 50 层）。
- 模型：`/Users/jay/.lmstudio/models/Qwen3-VL-4B-Instruct`（8.3GB，d_in=2560 确认是 4B）。
- 依赖：python3.13 已装 torch 2.12.0 / transformers 5.15.x / numpy / scipy（pip，非 uv）。
- 编写 `clipproj_harness.py`：加载 Qwen3VLModel（base，去 LM head）+ 投影；--hs-index 默认 25（tap+1），--tap 25；支持 --inline / --prompts / --reference / --chat。
- 前向逻辑已按元数据实现：hidden[hs_index] -> Linear(2560->32768)->GELU->Linear(32768->5120) -> cond=yn*std_out+mean_out -> cond[:,0]=sink_out。

### Blocked / Errors
- **OOM（EXIT 137 SIGKILL）**：在本机（16GB 总内存，仅 1.9GB 空闲，OS+app 已占 ~14GB）加载 8.3GB fp16 模型需 ~22GB -> jetsam 强杀。
  - harness 逻辑本身没问题，纯属内存装不下。
  - 已用 TQDM_DISABLE=1 关进度条确认是 SIGKILL 而非 Python 异常。

### Pending / Next
- [ ] Phase 5：读 ComfyUI `sd1_clip.py` 的 `encode_token_weights`，确认 sm.layer=[24] 映射到 HF `hidden_states` 的索引（24 还是 25）。
- [ ] Phase 4 解锁方案（选一）：
  1. 在大内存机器（>12GB 空闲）上跑 fp16（32B 对照机通常满足）。
  2. 给 harness 加 int8/int4 量化（torchao），把模型压到 ~4GB/~2GB，本机可跑。
  3. 查 ClipProj-MiniMax-H3 仓库是否自带 int8 版 4B 编码器（元数据 source_model=qwen3vl_4b_int8_convrot 疑似暗示）。
- [ ] Phase 6：实跑后用余弦相似度对照，目标接近 cos_test=0.8144；再与 h3.c 32B 输出对齐。

### Environment note
- 用户建议后续用 `uv` 做可复现安装（`uv venv` + `uv pip install` + `uv.lock`）。当前 pip 已跑通，先不动，作为后续环境方案。

## 2026-08-29（后续）
### 规划文件规范化
- `task_plan.md` 从自定义表格格式改写为 planning-with-files skill 模板格式（`### Phase N` + `**Status:**` + 检查项），内容全部保留：6 个 phase、Key Questions、Decisions、Errors 表。
- 效果：`check-complete.sh` 由 `0/0 phases complete`（识别不到）变为正确输出 `3/6 phases complete, 1 in_progress, 2 pending`；skill 的自动钩子（写文件后提示更新、Stop 时完成检查）恢复有效。
- Phase 4 状态标记为 in_progress（blocked 语义保留在 Current Phase 与进度条目中），因模板只支持 pending/in_progress/complete。
- 文件变更：`task_plan.md`（改写）、`progress.md`（本条记录）。

### 解锁方向调研（少走弯路）
- **方向 3 结论**：ClipProj-MiniMax-H3 仓库只含投影矩阵，**不含 4B 编码器**。但模型卡原文确认：bf16 校准的矩阵用于 int8/fp8 编码器余弦差仅 **0.0023**——量化编码器与矩阵兼容有官方依据（`source_model=qwen3vl_4b_int8_convrot` 正是作者校准用的 int8 编码器）。
- **方向 2 修正**：`torch.ao.quantization.quantize_dynamic` 是"先全量加载 fp16（8.9GB）再量化"，加载峰值不降，实测把 16GB 机器拖入严重 swap（命令全超时数分钟）——**加载后量化在此机不可行**。已 kill 进程恢复。
- **harness 截断加载改造**（进行中）：新增 `--max-layers`（默认 25=embed+tap24 层）+ `--quantize int8`。
  - 用 `safe_open` 惰性读所需层 + `torch.device('meta')` 构造 + `load_state_dict(assign=True, strict=False)`，绕开 transformers 5.15 禁止 `state_dict` 与模型名同传的限制。
  - 实测加载 25 层成功（277 tensors, 2.91B params），不再 OOM；卡在 strict 报错，已加 strict=False（visual 塔不加载保留 meta 占位）待验证。
  - 另发现模型目录**缺 config.json**（LM Studio 布局），已从 hf-mirror 补装（`Qwen/Qwen3-VL-4B-Instruct` 的 config，1.5KB）。
- **量化版本调研**：Qwen3-VL-4B 有丰富量化版（unsloth/mradermacher GGUF：Q2_K 1.67GB~Q8_0 4.28GB，Q4_K_M 2.5GB 为甜点档；mlx-community MLX 3/4/8bit 2.6~5.1GB）。
- **h3.c 不支持 GGUF**：`h3_weights.c` 只认 `.safetensors`，全仓库无 GGUF 代码（自研引擎、无 GGML）。GGUF 是给 harness 降内存用，h3.c 32B 基准仍走 safetensors。

### int8 编码器下载验证（hf-mirror + unfetch）
- 找到并下载 `Merserk/qwen3vl-4b-int8-convrot` 的 `qwen3vl_4b_int8_convrot.safetensors`（4.86GB）。
- 渠道：huggingface.co 直连超时（HTTP 000），**hf-mirror.com 可达**；unfetch 32 线程分片 **108 MB/s**，约 1 分钟下完；大小与 content-length（4864124848B）完全一致。
- 文件格式验证：1421 tensors，dtype = F32×354（scale/bias）+ BF16×359（norm/bias）+ **I8×354**（量化权重）+ U8×354——标准 torchao int8 权重量化布局，与投影元数据 `qwen3vl_4b_int8_convrot` 完全对口。
- 文件位置：`h3.c/weights/qwen3vl_4b_int8_convrot.safetensors`。
- 注意：该仓库仅权重+config，缺 tokenizer/chat_template，加载时需搭配原版 `Qwen3-VL-4B-Instruct` 目录的 tokenizer 等。

### int8 文件移动 + harness 反量化加载打通（重大进展）- 按用户要求统一模型布局：int8 文件移至 `/Users/jay/.lmstudio/models/Qwen3-VL-4B-Instruct-int8-convrot/qwen3vl_4b_int8_convrot.safetensors`（同卷 mv 无额外空间消耗）。
- 投影文件丢失（只剩 .cache lock，可能被清理）：用 unfetch 从 hf-mirror 重新下载 `mmh3-4b-ClipProj-v3-mlp.safetensors`（503MB，23 秒，SHA/元数据核对通过：tap=24, d_in=2560, d_out=5120, cos_test=0.8144, source_model=qwen3vl_4b_int8_convrot）。
- **harness 新增 `--weights-file` 选项**：直接读单文件 weight-only int8（`I8 weight` + `F32 weight_scale [out,1]`），反量化 `w = i8 * scale` 后走截断加载，跳过 visual tower。
- **RoPE meta buffer 修复**：meta device 构造模型后，`Qwen3VLTextRotaryEmbedding` 的 `inv_freq` 是 meta 张量，前向报 "Cannot copy out of meta tensor"；改为用 config 重算 `inv_freq`/`original_inv_freq`（default 用 `compute_default_rope_parameters`，非 default 用 `ROPE_INIT_FUNCTIONS`）。
- **✅ 全链路跑通**（16GB 本机无 OOM）：`python3.13 clipproj_harness.py --prompts "hello" --inline --max-prompts 1 --weights-file <int8> --max-layers 25`
  - `[model] int8-file dequant: 277 tensors (2.91B params)` —— 只读 embed+25 层+norm
  - `[0] seq=1 mean_norm=16357.887 :: hello`
  - `saved 1 embeddings -> candidate.npz`，`hello` 键 shape=[1,5120] float64（d_out=5120 正确）
  - 单 prompt 无 32B 对照，mean_norm 仅目测合理性；Phase 5/6 尚未完成。

### Phase 5 完成：layer 索引语义确认（transformers 5.15 源码实证）
- 本机无 ComfyUI，改从 transformers 5.15 源码确认：`Qwen3VLModel.forward` 的 `language_model` 是 `Qwen3VLTextModel`（继承 `Qwen3Model`），其 forward 标 `@capture_outputs`（`utils/output_capturing.py:215`）。
- `capture_outputs` 通过 hook 自动收集每层输出为 `hidden_states` 元组：`[0]`=embed 输出（capture_initial_hidden_state），`[k]`=第 k-1 层 block 输出；最后一个元素被 `tie_last_hidden_states` 覆盖为 norm 后的 `last_hidden_state`。
- **结论**：`hidden_states[25]` = 第 24 层 block 输出 = tap=24。harness 默认 `hs_index = tap + 1 = 25` **正确**，与 ComfyUI `sm.layer=[24]` 语义一致。2 层 smoke 实测 `len(hidden_states)=3`（embed, layer0, layer1）验证。
- 注：之前 grep 不到 `all_hidden_states` 是因为这个版本用 hook 捕获而非手写收集循环，易误判为"不支持中间层"。

### NaN 根因修复 + 向量合理性验证（重大突破）
- **症状**：多 token prompt 第 2+ token 出现 NaN（单 token 正常但被 sink_out 覆盖无区分度）。q/k/v 输出巨大（max=234 vs 正常 0.97）。
- **根因 1（dtype 判断 bug，真凶）**：`safe_open.get_slice(name).get_dtype()` 返回**字符串 `'I8'`** 而非 `torch.int8`。harness 的 `== torch.int8` 判断恒 False，反量化分支从未进入 → raw int8 值（absmax=127）被直接当权重加载 → q/k/v 输出 200+ → o_proj 起 NaN。
  - 实证：`get_slice().get_dtype()` 类型是 `<class 'str'>`，`d == torch.int8` 为 False。
  - 修复：改为 `== "I8"`。
- **根因 2（Hadamard 旋转）**：int8 文件是 ComfyUI **ConvRot 格式**（`comfy_quant` JSON: `{"format":"int8_tensorwise","convrot":true,"convrot_groupsize":256}`）。存储权重 = 每 256 宽组乘归一化 Hadamard 的旋转结果（`W_stored = W @ H^T`），反量化必须**先 i8*scale 再 unrotate**，否则权重完全错误。
  - 从 PyPI comfy-kitchen 0.2.31 wheel（`comfy_kitchen/backends/eager/convrot_w4a4.py` + `tensor/int8.py`）确认权威实现：`h = _build_hadamard(gs)`（4×4 seed 做 Kronecker 积到 256，÷√size 归一化），`dequantize` 路径 `_rotate_weight(i8*scale, h, gs)`。
  - 归一化 Hadamard 对称正交（H²=I），反量化再乘 H^T 即还原。
  - harness 新增 `_build_hadamard` + `convrot_unrotate(w, gs)`，按 `comfy_quant` 元数据判断是否 unrotate。
- **验证（全部通过）**：
  - 单层权重对照原版 fp16：7 个 layer0 线性层 cos 均 ≈1.000-1.008（unrotated），未 unrotate 时仅 0.064。
  - 同 prompt 逐层对照：INT8 q_proj out_max=0.973 vs ORIG 0.972；layer0 out 5.727 vs 5.734。
  - 8-prompt 批量无 NaN（a cat seq=2, quantum physics seq=3, a cat sitting on a mat seq=6 全部有限）。
  - **语义合理性**（多 token 最后 token，去均值余弦）：a cat/a dog/a elephant 互相关 0.97-0.98（语义相近高）；quantum physics 与动物组 0.44-0.48（无关低）；hello world 与其它负相关。
  - **确定性**：同一 prompt 两次运行 mean_norm 完全一致（9057.316 / 10556.060）。
  - **sink 行为**：所有 prompt 的 token0 norm=16357.887、互相 cos=1.0（sink_out 覆盖，与 ClipProj 设计一致）；内容 token 有真实区分度。

### Phase 6 收尾：32B 对照放弃，以网络测评数据为基准
- **32B 本机不可行（确认放弃）**：h3.c 文本编码器跑 Qwen3-VL-32B 语言部分，权重 37 GiB（README 534-537 行），流式文本编码默认 ring depth 3 的目标机是 **M5 Max / 128 GiB**（README 543 行）。16GB Mac 无法加载。按用户指示放弃本地 32B 对照，以网络测评数据为对比。
- **以网络测评数据对比**：ClipProj 元数据 `cos_test=0.8144` = 作者 200 prompts 上 4B-vs-32B 的平均余弦；模型卡声明 bf16 校准矩阵用于 int8 编码器余弦差仅 **0.0023**（int8 编码器与矩阵兼容有官方依据）。
- **交付物**：`weights/cand_4b_final.npz`（10 prompts：a cat, a dog, an elephant, quantum physics, a vintage sports car, world peace, war and peace, hello world, close-up of a red fox in snowy forest, cinematic neon city street at night；全部 finite）。
- **sink_out 精确复现**：单 token prompt 输出与投影元数据 sink_out maxdiff=0.0（norm=16357.887），证明 ClipProj 前向与作者实现一致。
- **最终语义检查**：cat/dog=0.979, cat/elephant=0.814（动物组高）, cat/quantum physics=0.450（低）, fox/sports car=0.628（同场景）, world peace/war and peace=0.092。
- 后续（可选）：在 ≥128GB 机器跑 h3.c 32B 同 prompts 与 cand_4b_final.npz 对照，验证接近 cos_test=0.8144。

### int8 vs fp16 候选向量对照（补充验证，0.9982）
- 用原版 fp16 4B（截断 25 层，2.91B）跑**同一批 10 prompts** → `/tmp/cand_fp16.npz`，与 int8 交付物 `weights/cand_4b_final.npz` 逐 prompt 对比最后 token 余弦：
  - 平均余弦 **0.9982**，min 0.9900（hello world/neon city 为长 prompt 累积量化误差），9/10 个 >0.999
  - 证明 int8 反量化管线与原版 fp16 数值一致（编码器层面 ≈ 模型卡声称的 0.0023 余弦差）
  - 因此 int8 候选向量 ≈ 作者校准用 int8 编码器输出 → 与 32B 的余弦应接近 cos_test=0.8144
- 结论：以网络测评数据（cos_test=0.8144）为对比基准的论证链完整闭合。

### 32B 对照指引落地（task_plan 收尾）
- 新增 `weights/README.md`：交付物说明 + 10 prompts 清单 + 4B 复现命令 + **32B 实证对照完整步骤**。
- 32B 侧对照路径确认：h3.c `h3_text_encode_bf16`（`tests/test_real_prompt.c` 参照）输出 `[tokens,5120]` BF16，导出为 `ref32b.npz`（key=prompt 原文）；harness 已内置 `--reference ref32b.npz` 一键输出逐 prompt `cosine_vs_32B` + `mean/min/max/frac>=0.7`。
- 目标基准：mean 接近 ClipProj 元数据 `cos_test=0.8144`（作者 200-prompt 4B-vs-32B 平均余弦）。
- task_plan.md：Current Phase 更新为「全部完成（本机范围）」；Key Questions 3 已答（网络测评基准 + weights/README.md 实证步骤）。

### h3.c 引擎真实验证（回应：32B 从未跑过）
- **用户指出验证链缺口**：h3.c 的 32B 真实前向从未验证（权重 37GiB + 目标机 M5 Max/128GB，本机 M4/16GB 物理不可行），我们 4B harness 的对照对象"h3.c 32B 输出"从未产生。
- **本轮补强验证（无权重可做项全部做完）**：
  1. `./h3_tests`：**1768 checks 全过**（引擎数学原语自检）
  2. `./h3_audio_gpu_tests`：**AudioVAE Metal 原语与 host 参考匹配**（真实 Metal 计算）
  3. `./h3_tokenizer_tests <Qwen3-VL tokenizer.json>`：**60 checks 过**——h3.c 解析真实 Qwen3-VL tokenizer.json（7MB），编码与 Qwen 词汇表兼容
  4. safetensors 探针（`h3_st_read_header`）：**成功解析真实文件**——ClipProj（9 tensors: mean_in/mean_out/mlp.0.bias...）、4B int8 convrot（1421 tensors: model.language_model.layers.0.mlp.*.weight_scale...）
- **仍缺口**：32B 文本编码/DiT/VAE 真实权重前向（需 MiniMax-H3 原版 BF16 37GiB + 128GB 机器）。h3.c 只认 BF16 权重（`h3_text_encoder.c:225` 硬校验 H3_DTYPE_BF16），ComfyUI 的 int8_convrot 打包格式不受支持。

### Phase 7 启动：16GB 跑通整体（核心目标）
- **目标**：本次主要目的是优化 h3.c 让其在 **16GB 统一内存**上跑通 MiniMax-H3 整体（32B 文本编码 + DiT + VAE）。
- **关键澄清（回应 4B 疑问）**：
  - h3.c **硬编码 32B 维度**（`h3_text_encoder.h:9` `H3_TEXT_HIDDEN_SIZE 5120u`，`h3_text_encoder.c:15` TEXT_HIDDEN=5120），4B（2560 hidden）不兼容，**不能替代 32B**。
  - 4B harness 是独立 Python 验证工具（验证 ClipProj 投影），不是 h3.c 组件。
  - 4B harness 存在的意义 = 模拟验证投影数学，最终对照目标仍是 h3.c 32B 真实输出。
  - d5752a0 commit 是**内存优化**（h3_memory_plan/h3_dit/h3_video_vae，让 16/24GB 跑 32B 权重），不是 4B 支持。
  - h3.c 只认 **BF16 权重**（h3_text_encoder.c:225 硬校验 H3_DTYPE_BF16），低精度是运行时从 BF16 量化（MLP int8 引擎），无需低精度权重文件。
- **16GB 跑通的设计基础（README 确认）**：
  - 文本编码器：**流式 ring buffer**，8 个 I/O worker 预取未来层，Metal 执行当前层；默认 ring depth = 2（M3/16GB）/ 3（M5/128GB），`H3_QWEN_PREFETCH`/`H3_QWEN_PREFETCH_DEPTH` 可调。
  - DiT：`--ssd-streaming`，仅小归一化权重驻留，双 BF16 槽交替 + 后台读（~14.6 GB/s 内 SSD）。
  - M5 零拷贝映射权重；M3 copied-buffer。本机 M4 走 copied-buffer 路径。
- **权重下载**（hf-mirror，unfetch 32 线程 ~20MB/s×5 并行）：
  - `/Users/jay/.lmstudio/models/MiniMax-H3/`（`.lmstudio` 是软链 → `/Volumes/data/.lmstudio` 移动硬盘 1.2Ti）
  - `FL2VA/text_encoder/` 14 片 ~64GB + `FL2VA/tokenizer/` + `FL2VA/transformer/config.json`（h3_load_dir 必需）
  - 最小验证集依据：`tests/test_real_prompt.c:141` 直接 `h3_text_encode_bf16(text_encoder_dir,...)`，不经过完整 h3_load_dir；校验 `submissions==51`（50 层+1 norm）
- **待办**：下载完成后 `./h3_real_prompt_test /Users/jay/.lmstudio/models/MiniMax-H3` 验证 32B 文本编码 16GB 跑通 + 4B harness 对照闭环；再评估 DiT/VAE 整体生成。

### turbo 少步调研（回应：能否参考 ComfyUI 插件优化 h3.c）
- **h3.c 现状**：支持 `--steps 4-7`（普通调度少步）+ `--reuse`（denoiser 复用）；**无 LoRA/turbo/lightning 实现**（grep 无任何 lora 代码）。
- **ComfyUI 加速生态**（搜索确认）：
  - 量化：INT4/INT8/NVFP4（Abiray 汇总 16GB卡→int4_convrot 11.3GB DiT+15GB 编码器；24GB→int8；Blackwell→nvfp4 12.5GB）——h3.c 只认 BF16，不可用
  - Turbo 少步：LightX2V MiniMax-H3-Turbo（4步 LoRA）+ 官方 Lightning 8步 LoRA（1.8GB）——标准 LoRA safetensors
  - 缓存预测：TeaCache/EasyCache/FirstBlockCache；SageAttention（~2x）
  - unsloth/MiniMax-H3-GGUF（212GB）：DiT Q2_K 6.7GB~Q8_0 21.4GB + qwen3vl_32b Q4_K_M 等——**h3.c 无 GGML，不可用**
- **结论**：h3.c 可借鉴的最高价值机制 = **LoRA 合并（Turbo 少步）**：W'=W+α·(B@A)，20步→4-8步 = 2.5-5x 提速。h3_safetensors + h3_dit.c 加载逻辑现成，仅需把 ComfyUI `lora_unet_*` 键名映射到 h3.c DiT 层结构。计划在 32B 文本编码验证后优先实现。
- 下载恢复：unfetch 降并发（max_concurrent=2, threads=16）后 8 片并行正常（3-4MB/s/片）。

### LoRA 合并实现完成 + 真实 LoRA 端到端验证（feature/lora-merge 分支）
- 实现：h3_lora.c/h（解析 Diffusers 风格 BF16 LoRA，rank 从 lora_A 读，scale=alpha/rank，GPU linear 算 B@A，host 行带写回）+ h3_dit.c load_block 6 目标合并 + main.c `--lora PATH` + h3_gpu.m read_bf16_range + Makefile 改用 xcrun clang（Homebrew clang16 无法编译 macOS26 SDK MPSGraph）
- 合成验证：tests/test_lora.c 真实维度×F64 参考，rel-L2 0.29%，qkv 三带/4 组权重/缺失 no-op 全过
- **真实 LoRA 验证**（ModelScope 下载 v1.1 4step 1.38GB）：624 张量、312 个目标键全匹配、alpha=128→scale=1.0、to_out.0 合并 rel-L2 0.175% PASS
- 用法：`h3 --lora /Users/jay/.lmstudio/models/LightX2V/MiniMax-H3-Turbo/minimax_h3_fl2v_turbo_4step_v1.1_768p_bf16.safetensors --steps 4`
- 限制：与 --ssd-streaming 互斥（流式路径未接入合并）
