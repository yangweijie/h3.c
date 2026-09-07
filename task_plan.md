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

### Phase 2: LoRA 扩展 — complete
- [x] 支持 `attn.orig.to_q/to_k/to_v` → qkv 行带映射（C 端 target 表）
- [x] 支持 `.default` 与 `.turbo` 两套后缀（多 adapter，按顺序合并）
- [x] 形状不兼容时告警跳过（pruned convrot 基座的 8-wide AdaLN 层）

### Phase 3: 线性分支权重加载 — complete
- [x] h3_dit_block 新增 16 字段（常驻 + 大矩阵流式 STREAM_LIN_OUT）
- [x] 加载/映射 linear_branch/model.safetensors（transformer_blocks.N → blocks.N）
- [x] 张量计数、形状断言

### Phase 4: Metal 内核 — complete
- [x] 特征：depthwise 5×5 空间 conv + 5-tap 时间 conv + SiLU + L2Norm（融合）
- [x] frame stats（A/B fp32）、alpha、softmax_gate、output_gate
- [x] vdn_solve：批量 128×128 Cholesky + 三角求逆
- [x] scan（逐帧 bmm，主机驱动；forward+reverse）
- [x] gather + readout epilogue（readout@q、RMSNorm、gate）
- [x] tests/test_vdn_branch.c 加入 make test

### Phase 5: run_block 接线 — complete
- [x] raw qkv（pre-QK-norm / pre-RoPE）输出
- [x] softmax 窗口注意力接入主路径（chunk c5 + anchors）
- [x] 线性分支插入（softmax out + to_out_linear 后、残差前）

### Phase 6: 目录/CLI/调度 — complete
- [x] `--linear-branch DIR` 参数（--lora 逗号分隔顺序合并）
- [x] 端到端跑通进入去噪循环

### Phase 7: 端到端验证 — complete
- [x] 短片生成成功（256×256 / 1s），有狐狸
- [x] 与 bf16 线性分支逐像素对比（cos=0.9904）

### Phase 8: INT8（ConvRot）线性分支接入 — complete
- [x] 实测确认 int8 文件为 convrot 格式：反量化 vs bf16 真值 cos≈0.99995（非旋转仅 0.065）
- [x] 新增 `h3_weight_store_open_file()`（单文件 store，h3_weights.h/.c）
- [x] VDN 打开改为候选名单确定性单文件：`model_int8_convrot_comfyui.safetensors` → `model.safetensors` → 整个目录
- [x] `prepare_vdn_stream_source()` 支持 I8 + 填 scale 字段（原先硬性要求 BF16）
- [x] 流式反量化按 `field == STREAM_LIN_OUT` 从 `dit->vdn_weights` 取 scale
- [x] `load_convrot_scale_values()` 改接收显式 store（主模型行为不变）
- [x] 端到端跑通；对调文件后复现出逐字节一致产物

### Phase 9: 性能剖析与加速验证 — complete
- [x] 步数对照：steps=2 为欠采样（**无 linear 同样糊**）→ 黄色与 linear/int8 无关
- [x] 端到端计时：VDN **305s** vs 原版 **90s**（M4 / 256×256 / 1s / steps=4）→ **慢 3.4×**
- [x] `H3_PROFILE` 流式剖析：79.35 vs 72.14 GiB；**unhidden wait 0.001s vs 23.3s**
- [x] 结论：VDN 瓶颈是**计算**不是 I/O；`to_out_linear` 常驻省 0 秒却要 3.85 GB
- [x] 高分辨率对照：512×384 下 VDN **1295s** vs 原版 **186s** → **慢 7.0×**；VDN 超线性缩放(4.25×)、原版亚线性(2.07×)，差距随尺度拉大
- [x] 最终结论：VDN 是**叠加混合分支**(窗口 softmax + 线性分支都跑),任何尺度都不能加速
- [x] 诊断：`h3_convrot_test` 在 M4 **通过** → 基础 GPU convrot kernel(`h3_gpu_weight_dequant_unrotate_int8`)可用；`H3_VDN_INT8` 崩溃原因 = NAX/tensor-ops(M5 专属),非 convrot kernel

### Phase 10: Streaming GPU ConvRot 反量化 — 失败，已回退（保留记录以免重复踩坑）
- [x] 新增 `h3_gpu_blocking_weight_dequant_unrotate_int8`(h3_gpu.h/.m)，沿用
      `h3_gpu_blocking_linear_bf16` 模式：同一队列 + 独立命令缓冲 + commit/wait，不开 NAX
- [x] 诊断（第一轮，结论**错误，已推翻**）：误判"layout=0 三个张量错误"。
      实际是诊断代码自身 bug：`memcpy(&g,&gpu_vals[k],4)` 从 `uint16_t*` 拷 4 字节，
      把两个 bf16 拼成一个 float。
- [x] 诊断（第二轮，**正确**）：改用 `bits=(uint32_t)val<<16` 转换，跨矩阵多行比对，
      idx 0–3 / 多行全部 `d=0.000000` → **kernel 与 CPU 蝶形逐位一致，kernel 无 bug**。
- [x] 结论：失败在**集成层面**（GPU 写入对后续 kernel 的可见性 + blocking commit/wait 开销），
      而非 kernel。性能 ❌（144s vs 90s）→ **回退**。
- [x] 回退：`git checkout h3_dit.c` 恢复 CPU 反量化；验证 VDN int8 @4 产物 198461 字节一致，零报错。
- [x] **fence/barrier 解决（第二轮）**：用 `MTLEvent` 跨 command buffer 同步
      （dequant signal → requant wait），纯 GPU 写入版输出**正确**
      （R=122.1/93.2/57.8/std=24.4 ≈ 基线），可见性问题已解决。
- [x] **批量 dequant（第三轮）**：新增 `h3_gpu_stream_dequant_begin/encode/submit`，
      4 个 source 编码进同一 command buffer、每 block 只提交一次（50 vs 200 次）。
      输出正确，但 **182s vs 90s 更慢**。
- [x] **最终根因（架构级）**：GPU dequant 与主线程计算**共享同一个 GPU**，
      互相竞争，消除了 CPU 蝶形原本享有的"CPU-GPU 异构并行"优势。
      **GPU dequant 方向从根本上不可行，即使修好 fence 也不会更快。**
- [ ] 若日后重启：kernel 是正确的，不要修它。要解决的是 GPU 资源竞争
      （需多命令队列真正并行，或接受此方向不可行）。

### Phase 16: 端到端参数基准测试 — complete
- [x] 编写 `benchmark.py` 工具链(参数扫描 + SSIM/PSNR/L2 评分 + HTML 报告)
- [x] 发现模型 text_encoder 不完整(`.unfetch_state`),切换至用户工作模型
- [x] 三轮测试(Round 1/2/3),共 22 个独特配置
- [x] 按显存从低到高排序,所有测试 <13 GB(9.4–9.8 GB)
- [x] 生成 HTML 报告 `/tmp/h3_benchmark/report.html`
- [x] 结论:显存不是瓶颈;step=7 比 step=4 画质提升 54%;最佳权衡 s7-l50-r2(177s/SSIM=0.57)

## Decisions Made
| Decision | Rationale |
|---|---|
| vdn_solve 用精确 Cholesky 求逆（而非 SanaDelta 一阶截断）| released checkpoint 就是用 vdn_solve 训练的；h3.c 已有 SanaDelta 骨架但不能等价替换 |
| 基座权重继续用 convrot FL2VA 目录 | VDN stage 目录只含新增权重；基座即 MiniMax-H3 |
| chunk 窗口用 per-frame bounds 表示 | 与现有 window mask/flash kernel 数据结构兼容，bounds 改为 c5 生成即可 |
| 扫描 v1 用主机逐帧驱动 bmm | 102×2 次/层/步的 launch 开销可接受（~0.5s/步），先求正确 |
| 复用 `h3_weight_load_bf16` 已有的 I8 反量化，不为 VDN 另写加载器 | `load_tensor` 在 `dtype==I8` 且请求 BF16 时自动走 `load_int8_dequantized`（反量化 + convrot 反旋转）；其 scale 名规则 `"%s_scale"` 正好匹配 int8 文件的 `beta_proj.weight_scale`，故 VDN loader 本体零改动 |
| VDN 线性分支只打开**单个**权重文件（候选名单） | int8 与 bf16 导出并存且张量同名，合并进同一 store 会让 `h3_weight_find` 命中排序靠前的那份，且白白映射 4.3 GB bf16 |
| `to_out_linear` **不**改为常驻 | `H3_PROFILE` 实测 unhidden wait = 0.001s，I/O 已被计算完全掩盖；常驻省 0 秒却要 50×77MB = 3.85 GB |

## Errors Encountered
| Error | Attempt | Resolution |
|-------|---------|------------|
| 子代理无写文件权限 | 1 | 报告由主代理落盘 |
| `VDN streaming weight is absent or has the wrong schema: transformer_blocks.0.attn.to_out_linear.weight` | 1 | `prepare_vdn_stream_source` 硬性要求 `dtype==BF16`；改为接受 I8 并填 `scale_path/scale_offset/scale_name` |
| 流式反量化报 `required weight_scale absent: …to_out_linear.weight_scale` | 1 | 消费点固定从 `dit->weights` 按名找 scale；改为按 `field == STREAM_LIN_OUT` 选 `dit->vdn_weights` |
| `H3_VDN_INT8=1` → `cannot stream DiT block 1: DiT stream begin failed: unknown Metal error`（13s 退出）| 1 | NAX tensor ops 是 M5 路径，M4 上强制开启会崩；本机不可用，回退 bf16 计算 |
| 用户把 int8/bf16 两个文件对调（`model.safetensors`=int8，`model_bf32.safetensors`=bf16）导致同名张量重复进 store | 1 | 候选名单确定性单文件打开（见 Decisions）；已验证复现逐字节一致产物 |
| 误判"steps=4 仍是黄糊"（只能看统计量、看不到图）| 1 | 用户肉眼确认 4 步有狐狸；统计指标（std/grad）不足以判断语义正确性，结论须以实际观看为准 |

## Notes
- VDN 配置：chunk=5, radius=1（chunk 模式下 radius 为 chunk 跨度）、anchor_frames=both、
  bridge=alpha、a_fp32、enable_text_state、short_conv=[k,v]、linear_head_dim=128。
- turbo: 8 steps, video_shift=12.0, audio_shift=3.0（= h3_host.h 现有常量）。
- 详细规格见 findings.md「VDN-H3 规格研究」节。
- **本机 Apple M4（非 M5）**：`wantsTensorOps = (m5 || getenv("H3_VDN_INT8")) && !(H3_NAX=="0")`
  （h3_gpu.m:371）→ M4 上 NAX tensor ops 默认关闭，且 `H3_VDN_INT8` 强开会崩。
- `H3_NAX_FORCE` 全仓库无任何引用 = **空操作**；真实变量是 `H3_NAX`（0 / mlp / qkv-attn）。
- `--linear-branch` 是 VDN 的**唯一入口**（main.c:308/409），不受任何环境变量影响；
  不传它则 `vdn=0`，线性分支的解析/加载/前向全部不执行（有效的"无 linear"对照）。
- 性能基线（256×256 / 1s / steps=4 / seed 42 / `--ssd-streaming`）：VDN **305s** vs 原版 **90s**。
- 跨尺度（256×256 → 512×384,像素 3×）：VDN 慢 **3.4× → 7.0×**；VDN 超线性缩放、原版亚线性。
- **最终结论：VDN 是叠加混合分支(窗口 softmax + 线性分支都跑),在当前 h3.c 实现下任何尺度都不能加速。**
  int8 线性分支接入是正确的,但它的定位是「质量特性」而非「加速特性」。
  详见 findings.md「INT8 线性分支 + 性能剖析 (2026-09-07)」。
