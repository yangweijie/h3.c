# Task Plan: h3.c 完整支持 VDN-H3（线性注意力混合分支 + LoRA + turbo）

## Goal
让 h3.c 完整支持 VDN-MiniMax-H3（OpenVDN/vdn-minimax-h3 checkpoint，位于
/Users/jay/.cache/modelscope/models/OpenVDN--vdn-minimax-h3/snapshots/master）：
- 每 DiT block 增加线性注意力分支（vdn_solve delta rule，与 softmax 窗口分支精确互补）
- LoRA adapter（default rank64/alpha64；turbo 额外一层，`.turbo` 后缀）加载合并
- softmax 分支改为 chunk 窗口（c5）+ anchor_frames=both（替代全稠密 SDPA）
- 8-step turbo 推理（video_shift=12 / audio_shift=3 — h3.c 常量已匹配，无需改）

## 权重事实（来自 safetensors 头部实测）
- 每 block 16 个新张量（800 = 16×50），key `transformer_blocks.N.attn.…`：
  - `linear_attention.alpha.{A_log[56], dt_bias[7168], down.weight[128,5376], up.weight[7168,128]}`
  - `linear_attention.beta_proj.weight[56,5376]`、`linear_attention.norm.weight[128]`
  - `linear_attention.short_conv.{k,v}_{sp[7168,1,5,5],tm[7168,1,5]}.weight`
  - `linear_attention.output_gate.{down.weight[128,5376], up.weight[7168,128], up.bias[7168]}`
  - `softmax_gate.up.{weight[56,5376], bias[56]}`
  - `to_out_linear.weight[5376,7168]`
- adapter：`transformer_blocks.N.attn.orig.{to_q,to_k,to_v,to_out.0}.lora_A/B.default.weight`
  rank64；turbo 同 targets 但后缀 `.turbo`，另含 `norm_out.linear.lora_*.turbo`。
- 基座权重沿用现有 FL2VA/transformer（convrot），只需映射 `transformer_blocks.N`→`blocks.N`、
  `attn.orig.to_{q,k,v}`→融合 qkv 的行带 [0,7168)/[7168,14336)/[14336,21504)。

## 算法规格（源码级确认）
1. 线性分支只吃 video 行，输入=共享 QKV 的 raw q/k/v（pre-QK-norm、pre-RoPE，NoPE）。
2. 特征：k,v = depthwise 5×5 空间 conv + 5-tap 时间 conv（零填充、非因果、跨帧）→ SiLU → L2Norm(eps=1e-6, 仅 k)；q = SiLU→L2Norm；v = conv→SiLU（不归一）。
3. beta = sigmoid(beta_proj(x))；A[f,h]=(βk)ᵀk（fp32+对称化）、B[f,h]=(βv)ᵀk（fp32）。
4. alpha（fp32）：delta=up(down(x̄_f))+dt_bias，alpha=exp(−exp(A_log_h)·softplus(delta))，x̄_f=帧均值。
5. vdn_solve：chol(I+A) → inv=L⁻ᵀL⁻¹；transition=Diag(alpha)·inv；injection=B@inv。
6. 文本状态：S_text=B_text@(I+A_text)⁻¹（text 行无 conv，beta 同投影，ones alpha）；双向扫描起点=0.5·S_text。
7. 扫描：S_t=S_{t-1}·transition_t+injection_t，forward+reverse 两套状态库（fp32）。
8. gather（skip_ends：首尾帧读出=0，bounds 减 1 重基）：left=prefix[lo-1]、right=suffix[hi+1]，bridge=α 乘积（log 前缀差）；窗口触界侧读 0.5·S_text 衰减入。
9. 读出：out_fs = Σ_k linear_state[f,h,v,k]·q[f,s,h,k] → RMSNorm[128] → ×output_gate(xv) → to_out_linear，加到 attention 输出的 video 行。
10. softmax 侧：gate=sigmoid(softmax_gate.up(x)) per head 乘到 softmax 输出（to_out 之前）。
11. softmax 窗口：chunk=5 模式，frame t 窗口=[(t//5−1)·5, (t//5+2)·5−1]（clamp）；anchor both：帧 0/F−1 的 query 行 dense，且所有 query 额外看到帧 0/F−1（列）。text/audio 行保持稠密。

## Phases
### Phase 1: 规格研究 + 计划落盘 — complete
- [x] Python 参考实现精读（branch/delta_rule/scan/features/layers/hybrid_attention/window）
- [x] checkpoint 权重清单实测
- [x] h3.c 现状调研（DiT 前向/LoRA/加载/调度/kernel 注册）

### Phase 2: LoRA 扩展 — in_progress
- [ ] 支持 `attn.orig.to_q/to_k/to_v` → qkv 行带映射（C 端 target 表）
- [ ] 支持 `.default` 与 `.turbo` 两套后缀（多 adapter，按顺序合并）
- [ ] 验证：加载 stage-dmd adapters，检查合并后 delta 数值（h3_lora_tests 扩展）

### Phase 3: 线性分支权重加载
- [ ] h3_dit_block 新增 16 字段（norm 常驻 + 大矩阵可流式 STREAM_LIN_*）
- [ ] 加载/映射 linear_branch/model.safetensors（transformer_blocks.N → blocks.N）
- [ ] 验证：张量计数、形状断言

### Phase 4: Metal 内核
- [ ] 特征：depthwise 5×5 空间 conv + 5-tap 时间 conv + SiLU + L2Norm（融合）
- [ ] frame stats（A/B fp32）、alpha、softmax_gate、output_gate（可并入现有 adaln/linear）
- [ ] vdn_solve：批量 128×128 Cholesky + 三角求逆（threadgroup/矩阵）
- [ ] scan（逐帧 bmm，主机驱动；forward+reverse）
- [ ] gather + readout epilogue（readout@q、RMSNorm、gate）
- [ ] 单元测试：CPU 黄金对照（tests/ 新增）

### Phase 5: run_block 接线
- [ ] raw qkv 输出（fused qkv kernel 需输出 pre-norm/pre-rope 值，或拆分）
- [ ] softmax 窗口注意力接入主路径（chunk c5 + anchors，改造 h3_flash_attn_tiled_windowed/h3_sdpa_window_mask）
- [ ] 线性分支插入（softmax out + to_out_linear 后、残差前）
- [ ] 验证：单 block 与 Python golden 对比

### Phase 6: 目录/CLI/调度
- [ ] --vdn-dir 或 lora/linear 分开参数；8 步 turbo（steps=8，shift 已匹配）
- [ ] token_refiner（文本编码器侧）LoRA 合并检查（h3_lora 已支持 refiner 目标名）

### Phase 7: 端到端验证
- [ ] make test 全绿
- [ ] 用 stage-dmd-step-250 生成短片，对比视觉质量/与 Python 输出合理性

## Decisions Made
| Decision | Rationale |
|---|---|
| vdn_solve 用精确 Cholesky 求逆（而非 SanaDelta 一阶截断）| released checkpoint 就是用 vdn_solve 训练的；h3.c 已有 SanaDelta 骨架但不能等价替换 |
| 基座权重继续用 convrot FL2VA 目录 | VDN stage 目录只含新增权重；基座即 MiniMax-H3 |
| chunk 窗口用 per-frame bounds 表示 | 与现有 window mask/flash kernel 数据结构兼容，bounds 改为 c5 生成即可 |
| 扫描 v1 用主机逐帧驱动 bmm | 102×2 次/层/步的 launch 开销可接受（~0.5s/步），先求正确 |

## Errors Encountered
| Error | Attempt | Resolution |
|-------|---------|------------|
| 子代理无写文件权限 | 1 | 报告由主代理落盘 |

## Notes
- VDN 配置：chunk=5, radius=1（chunk 模式下 radius 为 chunk 跨度）、anchor_frames=both、
  bridge=alpha、a_fp32、enable_text_state、short_conv=[k,v]、linear_head_dim=128。
- turbo: 8 steps, video_shift=12.0, audio_shift=3.0（= h3_host.h 现有常量）。
- 详细规格见 findings.md「VDN-H3 规格研究」节。
