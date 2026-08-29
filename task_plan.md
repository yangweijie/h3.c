# Task Plan: 4B + ClipProj 候选向量 harness（用于对照 h3.c 32B 流水线）

## Goal
用 Python 复现 MiniMax-H3 的「4B 文本编码器 (Qwen3-VL-4B) + ClipProj 投影」前向，产出 [seq,5120] 候选向量，
与 h3.c 在 32B 上跑出的结果做余弦相似度对照（ClipProj 自带 cos_test=0.8144 参考值）。

## Current Phase
Phase 8（in_progress）

**一句话状态：** 4B+ClipProj harness 全部 6 个 phase 完成；**Turbo LoRA 合并已实现并通过真实 LoRA 端到端验证**（feature/lora-merge 分支）。MiniMax-H3 FL2VA 下载完成大部分：text_encoder **14/14 片（~64GB）** + tokenizer + transformer/config.json 已就位（62GB 汇总，移动硬盘）；**Turbo LoRA v1.1 4step 1.38GB 已下载并验证**。下一步：跑 `h3_real_prompt_test` 验证 32B 文本编码，然后下载 DiT/VAE 做整体生成。

## Phases

### Phase 7: 16GB 跑通整体（h3.c + MiniMax-H3 32B）— 大部分完成
- [x] 下载 MiniMax-H3 FL2VA：text_encoder 14 片（~64GB）+ tokenizer + transformer/config.json → `/Users/jay/.lmstudio/models/MiniMax-H3/`（移动硬盘）— **14/14 片齐全、大小匹配，共 62GB**
- [ ] `./h3_real_prompt_test /Users/jay/.lmstudio/models/MiniMax-H3` 验证 32B 文本编码在 16GB 跑通（`submissions==51`，流式 ring depth 2）
- [ ] 对照验证链闭环：h3.c 32B 文本输出 ↔ 4B harness 候选向量（cos 目标 ≈0.8144）
- [ ] 评估内存峰值；若 OOM 则按 h3.c 流式参数调优（`H3_QWEN_PREFETCH`/`H3_QWEN_PREFETCH_DEPTH`）
- [ ] （通过后）下载 DiT（~58GB）+ video/audio VAE，`h3 --ssd-streaming` 整体生成验证
- **Status:** in_progress

### Phase 8: Turbo LoRA 合并（4-8 步少步生成）— 完成
- [x] 调研：ComfyUI/unsloth H3 加速路径梳理 → h3.c 唯一可行路径是 Turbo/蒸馏 LoRA 合并（2.5-5x）；GGUF/int8 权重与 BF16-only 加载器不兼容
- [x] 下载真实 LoRA：ModelScope `LightX2V/MiniMax-H3-Turbo` `minimax_h3_fl2v_turbo_4step_v1.1_768p_bf16.safetensors`（1.38GB，2 分钟）
- [x] 实现 `h3_lora.c/h`：解析 Diffusers 风格 BF16 LoRA（rank 从 lora_A 读，scale=alpha/rank），GPU `linear_bf16` 算 B@A，host 行带写回 W += scale·(B@A)
- [x] 接入 `h3_dit.c`：load_block 每块 6 目标合并（to_q/to_k/to_v→qkv 带、to_out.0、ff.net.0.proj、ff.net.2）；键前缀映射 `blocks.N`→`transformer_blocks.N`、`token_refiner.blocks.N`→`refiner_blocks.N`；与 `--ssd-streaming` 互斥
- [x] CLI `--lora PATH`（main.c + params.lora_path）；h3_gpu.m 新增 `h3_gpu_tensor_read_bf16_range`
- [x] 合成验证：`tests/test_lora.c`（真实维度 5376/7168/14336 × F64 参考），rel-L2 0.29%，qkv 三带 + 4 组权重 + 缺失目标 no-op 全过
- [x] **真实 LoRA 验证**：624 张量、312 个目标键（50 transformer + 2 refiner 块 × 6 目标 × A/B）全匹配、alpha=128→scale=1.0、to_out.0 合并 vs Python 参考 rel-L2 0.175% PASS
- [x] 提交：`af61e47`（实现）+ `76fbbba`（真实验证记录），分支 `feature/lora-merge`
- [ ] （待 32B 全链路跑通后）端到端：`./h3 --lora <turbo> --steps 4` vs `--steps 20` 质量/时间对比
- **Status:** complete（实现 + 数值验证）；端到端待整体流水线就绪

### Phase 1: 获取 ClipProj 仓库并校验
- [x] ModelScope `NicoLab28/ClipProj-MiniMax-H3` 下载成功，SHA256 校验通过（feef06ef...），文件干净
- [x] GitCode HF 镜像（ai.gitcode.com）直链 401、需认证，已弃用
- **Status:** complete

### Phase 2: 解析投影元数据 + 确认 4B 模型
- [x] 投影文件 `mmh3-4b-ClipProj-v3-mlp.safetensors` 元数据：version=v3, tap=24, d_in=2560, d_out=5120, mlp_hidden=32768, mlp_depth=1, 无 W（纯残差 MLP）, cos_test=0.8144
- [x] 关键修正：投影用第 24 层（不是最后一层/不是 50 层）
- [x] 模型 `Qwen3-VL-4B-Instruct`（8.3GB，d_in=2560 确认是 4B）
- **Status:** complete

### Phase 3: 编写 harness (clipproj_harness.py)
- [x] 加载 Qwen3VLModel（base，去 LM head）+ 投影；--hs-index 默认 25（tap+1），--tap 25；支持 --inline / --prompts / --reference / --chat
- [x] 前向逻辑按元数据实现：hidden[hs_index] -> Linear(2560->32768)->GELU->Linear(32768->5120) -> cond=yn*std_out+mean_out -> cond[:,0]=sink_out
- **Status:** complete

### Phase 4: 实跑前向验证流水线
- [x] OOM 解锁：改用 `Merserk/qwen3vl-4b-int8-convrot`（torchao int8 单文件 4.86GB）反量化加载（w=i8*scale），绕开 fp16 全量加载峰值
- [x] `--max-layers 25` 截断（embed+tap24 层），跳过 visual tower 与尾部层；meta 构造 + assign=True + RoPE buffer 重建
- [x] 全链路跑通（16GB 无 OOM）：277 tensors (2.91B params) -> `[0] seq=1 mean_norm=16357.887 :: hello` -> `candidate.npz` [1,5120]
- [x] **NaN 根因修复**：get_dtype() 返回字符串 'I8' 而非 torch.int8（反量化分支从未进入）+ ConvRot Hadamard 解旋转（comfy_quant 元数据驱动）——修复后多 token 无 NaN
- **Status:** complete

### Phase 5: 确认 ComfyUI 层索引映射
- [x] 本机无 ComfyUI，改从 transformers 5.15 源码实证：`Qwen3VLModel` 的 `language_model`（Qwen3VLTextModel）forward 标 `@capture_outputs`，hidden_states 元组 [0]=embed、[k]=第 k-1 层输出、末元素=norm 后 last_hidden_state
- [x] `hidden_states[25]` = 第 24 层 block 输出 = tap=24，与 ComfyUI `sm.layer=[24]` 一致；harness 默认 hs_index=tap+1=25 正确
- **Status:** complete

### Phase 6: 余弦对齐到 0.81 / 与 h3.c 32B 对照
- [x] 向量合理性验证（多 token 语义余弦：cat/dog 高、无关词低、确定性、sink 行为正确）
- [x] 交付物：`weights/cand_4b_final.npz`（10 prompts，全部 finite，无 NaN）
- [x] **32B 本机对照不可行**：32B 文本编码器权重 37 GiB（README 534-537 行：M5 零拷贝映射 37 GiB 模型文件；543 行：默认流式深度 3 层目标机 128 GiB），16GB Mac 装不下
- [x] 按用户指示以网络测评数据为对比：ClipProj 元数据 `cos_test=0.8144` 即作者 4B-vs-32B 的 200-prompt 平均余弦；模型卡声明 bf16 校准矩阵用于 int8 编码器余弦差仅 0.0023
- [x] int8 vs fp16 对照：同 10 prompts 最终向量 mean cos=0.9982（min 0.99），论证链闭合
- [x] 32B 后续对照指引落地：`weights/README.md`（prompts 清单、4B 复现命令、32B 侧导出 `ref32b.npz` 步骤、`--reference` 一键对照、目标 mean≈0.8144）
- **Status:** complete（本机范围；32B 实证对照待 ≥128GB 机器）

## Key Questions
1. ~~ComfyUI 的 sm.layer=[24] 对应 HF `hidden_states` 的哪个索引？~~ 已答：`hidden_states[25]`（tap+1，embed 算第 0 层），源码实证
2. ~~本机 16GB 跑不下 8.3GB fp16：怎么解锁？~~ 已解：int8 convrot 反量化 + 截断 25 层加载，无 OOM
3. ~~Phase 6：单 prompt 结果如何与 cos_test=0.8144 对齐？~~ 已答：cos_test 是作者在 200 prompts 上 4B-vs-32B 的余弦；本机无 32B 权重，按用户指示以网络测评数据为基准，实证对照步骤见 `weights/README.md`（需 ≥128GB 机器）
4. ~~Turbo LoRA 是 Diffusers 还是 ComfyUI 格式？真实文件键名与 h3.c 生成器是否匹配？~~ 已答：Diffusers `minimax-h3-diffusers`；312 目标键（50 transformer + 2 refiner 块 × 6 目标）真实文件全匹配，rank=128/alpha=128
5. ~~alpha 如何解析？~~ 已答：`__metadata__.alpha`（JSON 字符串 "128"），scale=128/128=1.0 真实验证

## Decisions Made
| Decision | Rationale |
|----------|-----------|
| 投影只取第 tap 层隐藏态，前向为纯残差 MLP（无 W）：Linear(2560->32768)->GELU->Linear(32768->5120) -> cond=yn*std_out+mean_out -> cond[:,0]=sink_out | 由投影元数据 version=v3 推导 |
| 层索引默认 tap+1=25（嵌入算第 0 层） | Phase 5 源码实证确认：hidden_states[25]=第 24 层输出=tap=24，与 ComfyUI sm.layer=[24] 一致 |
| 环境用 pip（python3.13）已装 torch2.12/transformers5.15/numpy/scipy | 已跑通；用户建议后续用 uv 做可复现安装，作为后续环境方案 |
| int8 编码器用 `Merserk/qwen3vl-4b-int8-convrot`（hf-mirror 下载，unfetch 108MB/s），harness `--weights-file` 反量化加载 | 本机 16GB 跑不下 fp16 8.9GB；该模型正是投影元数据 source_model 同名，数值兼容（模型卡：cos 差 0.0023） |
| int8 反量化必须 **i8*scale 再解 ConvRot Hadamard 旋转**（每 256 宽组乘归一化 Hadamard，H²=I），按 `comfy_quant` 元数据 `convrot:true` 驱动；dtype 判断用 `get_slice().get_dtype()=="I8"`（字符串！） | comfy-kitchen 0.2.31 源码实证：`dequantize` = `_rotate_weight(i8*scale, H, gs)`；`get_dtype()` 返回 'I8' 字符串而非 torch.int8 |
| `--max-layers 25` 截断 + meta 构造 + strict=False + RoPE buffer 重建 | 只加载所需 25 层，跳过 visual tower；meta 构造省内存，strict 保留 visual 占位，RoPE inv_freq 需用 config 重算 |
| LoRA 目标 = Diffusers `key_format: minimax-h3-diffusers`（非 ComfyUI）；实测文件 `LightX2V/MiniMax-H3-Turbo` 4step v1.1 768p bf16 | 312 目标键实测全匹配（50 transformer + 2 refiner 块 × 6 目标） |
| 合并 W += scale·(B@A)，scale=alpha/rank（alpha 从 __metadata__ 读），qkv_proj 分 to_q/to_k/to_v 三带合并 | Diffusers LoRA 语义；rank=128、alpha=128 → scale=1.0 真实验证 |
| 合并在统一内存中执行、源文件不修改；`--ssd-streaming` 显式拒绝；int8 Quanti 路径合并 BF16 但发生在量化后 | 保持 h3.c BF16-only 模型，不引入 GGUF/int8 权重路径 |
| Makefile `CC := xcrun clang` | Homebrew clang 16 无法编译 macOS 26 SDK 的 MPSGraph.h（`dispatchQueue retain` 非对象类型报错）；Xcode clang 21 正常 |

## Errors Encountered
| Error | Attempt | Resolution |
|-------|---------|------------|
| GitCode 镜像需认证（直链 401 / ls-remote 重定向登录页） | 1 | 弃用，改 ModelScope 下载 |
| OOM（EXIT 137 SIGKILL）：16GB 机器加载 8.3GB fp16 需 ~22GB | 1 | 用 int8 convrot 反量化 + 截断加载，本机跑通 |
| transformers 5.15 `state_dict 与模型名同传` ValueError | 1 | meta 构造 Qwen3VLModel(cfg) + load_state_dict(assign=True, strict=False) |
| RoPE `Cannot copy out of meta tensor` | 1 | meta 构造后重算 Qwen3VLTextRotaryEmbedding 的 inv_freq/original_inv_freq/attention_scaling |
| 投影文件丢失（目录只剩 .cache lock） | 1 | unfetch 从 hf-mirror 重新下载（503MB, 23s），元数据核对通过 |
| **多 token NaN（q/k/v 输出 200+）** | 3 | **双根因**：(a) `safe_open.get_slice().get_dtype()` 返回字符串 `'I8'` 而非 torch.int8，`==torch.int8` 恒 False → 反量化分支从未进入，raw int8(absmax=127) 当权重；(b) ConvRot 权重须解 Hadamard 旋转（`convrot_unrotate`）。修复后单层权重 vs 原版 cos≈1.00，q_proj 输出 0.973 vs 0.972 |
| LoRA apply 报 `unknown Metal error` | 1 | 根因：用 `h3_gpu_tensor_read_f32_range` 读 BF16 权重（F32-only API 返回空/错值）；修复：新增 `h3_gpu_tensor_read_bf16_range` 走 BF16 路径 |
| h3_gpu.m 无法编译：`MPSGraph.h:137 property with 'retain' must be of object type` | 2 | PATH 里 Homebrew clang 16 不认识 macOS 26 SDK 的 `dispatch_queue_t` retain 属性；改 Makefile `CC := xcrun clang`（Xcode clang 21）编译通过 |
| 合成 LoRA 测试数值不匹配（rel-L2 0.11，max-rel 1e7） | 1 | 陈旧二进制：h3_gpu.o 是仓库预编译，改动 h3_gpu.h 后 make 用 Homebrew clang 重编失败、测试仍链老 .o；修复编译后 rel-L2 0.29%（BF16 vs F64 预期噪声），max-rel 是近零元素无意义值，判定只看 rel-L2 |
| LoRA key 构造越界（`blocks.0.attn.to_out.0` 前缀拼接截断） | 1 | 首次只拼 `transformer_blocks.0.attn.to_out.0` 导致 length-6 溢出；用 `snprintf` 检查 + 完整前缀 `%s.%s.lora_A.default.weight` |
| hf-mirror 下载 text_encoder 片 7-14 中途暂停/掉速 | 1 | 换 ModelScope `MiniMax/MiniMax-H3` 重下 7-14 片；最终 14/14 齐全（01-06 用 hf-mirror 完成版，07-14 用 ModelScope 版，大小与 total_bytes 完全一致） |

## Notes
- Update phase status as you progress: pending → in_progress → complete
- Re-read this plan before major decisions (attention manipulation)
- Log ALL errors - they help avoid repetition
