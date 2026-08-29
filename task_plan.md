# Task Plan: 4B + ClipProj 候选向量 harness（用于对照 h3.c 32B 流水线）

## Goal
用 Python 复现 MiniMax-H3 的「4B 文本编码器 (Qwen3-VL-4B) + ClipProj 投影」前向，产出 [seq,5120] 候选向量，
与 h3.c 在 32B 上跑出的结果做余弦相似度对照（ClipProj 自带 cos_test=0.8144 参考值）。

## Current Phase
全部完成（本机范围）

**一句话状态：** 4B+ClipProj 候选向量 harness 全部 6 个 phase 完成。产出 `weights/cand_4b_final.npz`（10 prompts）并附 `weights/README.md`（prompts 清单、4B 复现命令、32B 对照完整指引）。NaN 双根因修复、int8 vs fp16 对照（mean cos 0.9982）、网络测评基准（cos_test=0.8144）论证链全部闭合。32B 实证对照需 ≥128GB 机器，按 README 步骤可一键复现。

## Phases

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

## Decisions Made
| Decision | Rationale |
|----------|-----------|
| 投影只取第 tap 层隐藏态，前向为纯残差 MLP（无 W）：Linear(2560->32768)->GELU->Linear(32768->5120) -> cond=yn*std_out+mean_out -> cond[:,0]=sink_out | 由投影元数据 version=v3 推导 |
| 层索引默认 tap+1=25（嵌入算第 0 层） | Phase 5 源码实证确认：hidden_states[25]=第 24 层输出=tap=24，与 ComfyUI sm.layer=[24] 一致 |
| 环境用 pip（python3.13）已装 torch2.12/transformers5.15/numpy/scipy | 已跑通；用户建议后续用 uv 做可复现安装，作为后续环境方案 |
| int8 编码器用 `Merserk/qwen3vl-4b-int8-convrot`（hf-mirror 下载，unfetch 108MB/s），harness `--weights-file` 反量化加载 | 本机 16GB 跑不下 fp16 8.9GB；该模型正是投影元数据 source_model 同名，数值兼容（模型卡：cos 差 0.0023） |
| int8 反量化必须 **i8*scale 再解 ConvRot Hadamard 旋转**（每 256 宽组乘归一化 Hadamard，H²=I），按 `comfy_quant` 元数据 `convrot:true` 驱动；dtype 判断用 `get_slice().get_dtype()=="I8"`（字符串！） | comfy-kitchen 0.2.31 源码实证：`dequantize` = `_rotate_weight(i8*scale, H, gs)`；`get_dtype()` 返回 'I8' 字符串而非 torch.int8 |
| `--max-layers 25` 截断 + meta 构造 + strict=False + RoPE buffer 重建 | 只加载所需 25 层，跳过 visual tower；meta 构造省内存，strict 保留 visual 占位，RoPE inv_freq 需用 config 重算 |

## Errors Encountered
| Error | Attempt | Resolution |
|-------|---------|------------|
| GitCode 镜像需认证（直链 401 / ls-remote 重定向登录页） | 1 | 弃用，改 ModelScope 下载 |
| OOM（EXIT 137 SIGKILL）：16GB 机器加载 8.3GB fp16 需 ~22GB | 1 | 用 int8 convrot 反量化 + 截断加载，本机跑通 |
| transformers 5.15 `state_dict 与模型名同传` ValueError | 1 | meta 构造 Qwen3VLModel(cfg) + load_state_dict(assign=True, strict=False) |
| RoPE `Cannot copy out of meta tensor` | 1 | meta 构造后重算 Qwen3VLTextRotaryEmbedding 的 inv_freq/original_inv_freq/attention_scaling |
| 投影文件丢失（目录只剩 .cache lock） | 1 | unfetch 从 hf-mirror 重新下载（503MB, 23s），元数据核对通过 |
| **多 token NaN（q/k/v 输出 200+）** | 3 | **双根因**：(a) `safe_open.get_slice().get_dtype()` 返回字符串 `'I8'` 而非 torch.int8，`==torch.int8` 恒 False → 反量化分支从未进入，raw int8(absmax=127) 当权重；(b) ConvRot 权重须解 Hadamard 旋转（`convrot_unrotate`）。修复后单层权重 vs 原版 cos≈1.00，q_proj 输出 0.973 vs 0.972 |

## Notes
- Update phase status as you progress: pending → in_progress → complete
- Re-read this plan before major decisions (attention manipulation)
- Log ALL errors - they help avoid repetition
