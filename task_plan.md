# Task Plan: 4B + ClipProj 候选向量 harness（对照 h3.c 32B 流水线）

## Goal
用 Python 复现 MiniMax-H3 的「4B 文本编码器 (Qwen3-VL-4B) + ClipProj 投影」前向，产出 [seq,5120] 候选向量，
与 h3.c 在 32B 上跑出的结果做余弦相似度对照（ClipProj 自带 cos_test=0.8144 参考值）。

## Phases
| # | Phase | Status | 验证 |
|---|-------|--------|------|
| 1 | 获取 ClipProj 仓库并校验 | complete | ModelScope 下载完成，SHA256 通过；GitCode 镜像需认证已弃 |
| 2 | 解析投影元数据 + 确认 4B 模型 | complete | tap=24, d_in=2560, d_out=5120, mlp_hidden=32768, cos_test=0.8144 |
| 3 | 编写 harness (clipproj_harness.py) | complete | 文件已跟踪，--hs-index 可调（默认 tap+1=25） |
| 4 | 实跑前向验证流水线 | complete* | candidate.npz 已产出 shape (1,5120)；但 MLX 脚本未在仓库、层索引未校准 |
| 5 | 确认 ComfyUI 层索引映射 | pending | 需读 sd1_clip.py 的 encode_token_weights |
| 6 | 余弦对齐 0.81 / 与 h3.c 32B 对照 | pending | 依赖 4 与 5 |

*Phase 4 用户反馈已用 MLX 跑通；仓库当前仅 PyTorch harness，MLX 脚本待确认/补入。

## Decisions
- 投影只取第 tap 层隐藏态，前向为纯残差 MLP（无 W）：Linear(2560->32768)->GELU->Linear(32768->5120) -> cond=yn*std_out+mean_out -> cond[:,0]=sink_out。
- 层索引默认 tap+1=25（嵌入算第 0 层）；需经 Phase 5 校准，可能应为 24。
- 量化：选 int8 预量化路径以本机跑通；Mac CPU 上纯 int8 检查点稀缺，建议原地 torchao int8（同一 Qwen3VLModel）。用户反馈 Phase 4 已用 MLX，待核实。

## Open Questions
- ComfyUI 的 sm.layer=[24] 对应 HF hidden_states 的哪个索引（含/不含 embedding 偏移）？
- Phase 4 是否真用 MLX？若是，MLX 脚本在何处、是否需补入仓库？
- MLX 文本编码相比 PyTorch 的提速/省内存实测值多少（见 progress.md 估算，待验证）？
