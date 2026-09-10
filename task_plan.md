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

### Phase 17: 代码审查 — complete
- [x] 审查 `h3_audio_vae.c`(Bug 3/内存 2/健壮性 3/设计 3)
- [x] 审查 `h3_cli.c`(Bug 4/安全 3/内存 2/设计 4)
- [x] 审查 `h3_dit_schedule.c`(Bug 3/安全 2/正确性 4/性能 2)
- [x] 审查 `h3_dit.c` 设计观察(缓存预算顺序/dit_layers 哨兵值/use_int8_row_fc2 不一致)
- [x] 记录 30+ 项发现到 findings.md,评估严重性

### Phase 18: 修复代码审查高优先级问题 — complete
已修复 5 项高优先级问题(均为静默正确性/安全风险,不影响正常路径行为):

| # | 文件 | 问题 | 修复 |
|---|---|---|---|
| 1 | `h3_dit_schedule.c:233` | 视觉/音频阈值不一致(visual `>=0.999f` vs audio `>=1.0f`) | audio 改为 `>= 0.999f`,与 visual 一致 |
| 2 | `h3_dit_schedule.c` | `time` 张量生命周期脆弱(4 处 free,新增 goto 易漏) | 每处 free 后加 `time = NULL`;`failed:` 标签统一 `h3_gpu_tensor_free(time)` |
| 3 | `h3_dit_schedule.c:238` | `count * feature_dim` 分配无溢出检查 | 加 `count > SIZE_MAX / feature_dim` |
| 4 | `h3_audio_vae.c:612` | `hidden_elements` 分配无溢出检查 | 加 `audio->length > SIZE_MAX / ((size_t)STEREO * 8)` |
| 5 | `h3_dit.c:2567` | `use_int8_row_fc2` 存储时被 `int8_mlp` 静默门控,语义不一致 | 存储原始值;门控由使用处外层 `if (dit->int8_mlp && …)` 保证 |

顺带清理(使代码可编译):
- `h3_gpu.m`:删除重复的 `stream_batch` / `stream_batch_pending` 属性声明(HEAD 遗留,导致编译失败)
- `h3_gpu.m`:删除已废弃的 GPU dequant 函数块(`blocking_weight_dequant_unrotate_int8` /
  `stream_dequant_begin|encode|submit` / `encode_wait_stream_event`);保留 `h3_gpu_linear_bf16_offset`
  与 VDN 存根

- [x] 验证:编译通过;端到端(256×256/1s/steps=4/seed 42/`--ssd-streaming`)产物
      **193579 字节,与基线逐字节一致**,耗时 102s,零报错 → 无回归

### Phase 18b: 清理中低优先级问题 — complete
- [x] `h3_cli.c` **`strdup` 失败未检查**(审查记 2 处,实际排查出 **5 处**):
  - `last_prompt`(507):失败后为 NULL,后续 `repeat` 会 `strdup(NULL)` **崩溃** → 加检查;
    另在 918 行加 `state.last_prompt` NULL 保护(双保险)
  - `sr_model` 初始化(852):纳入启动失败检查条件
  - SR `bin` / `model-dir` / `model`(792 / 796 / 801):原先静默失败 → 改为 `fprintf` 报错
- [x] `h3_dit_schedule.c` `weight_bf16_any()`:`elements` 累乘与转换缓冲区大小加溢出检查
      (防 `uint64_t` 回绕 / `size_t` 转换溢出)
- [x] 已核实**无需修改**的项:`h3_audio_vae.c` `run_stage` 的 `(uint32_t)elements`
      —— 上游 533-538 行已有 `elements64 > UINT32_MAX` 检查,截断不可能发生
- [x] 验证:编译通过;端到端产物 **193579 字节,与基线逐字节一致**,113s,零报错 → 无回归

### Phase 18c: 清理内存优化项（run_stage / gate_score）— complete
- [x] `h3_audio_vae.c` `run_stage()` 内存峰值优化（审查建议「5→3 tensor」的务实实现）
  - 审查建议的「5→3」经**逐项生命周期分析不可行**：5 个 tensor 都必需——
    `upsampled` 在 3 个 block 各被 copy 一次、`sum` 是累加器、`work` 是 block 1/2 的 target、
    `activated`/`branch` 乒乓。但发现**真实的峰值浪费**：上一层 `audio->hidden` 在 upsample
    之后即不再需要，却保留到块循环结束，使峰值 = 5 个 stage tensor + 上一层 hidden = **6 份**。
  - 修复：上采样后**立即 submit 并释放 `audio->hidden`**，再分配块循环的 4 个张量。
    峰值从 ~6 份降到 5 份（省一份与 stage tensor 等大的上一层 hidden），命令序列与数学完全等价。
- [x] `h3_dit_schedule.c` + `h3_dit_schedule.h` + `h3_dit.c`：`gate_score` 批量读回
  - 原 `h3_dit_schedule_gate_score()` 每调用一次都 `malloc` + GPU 读回 + `free`；
    排序路径（`configure_gate_ranked_blocks`）对 47 个 block 各调一次 → **47 次 malloc/free**。
  - 修复：新增 `h3_dit_schedule_gate_scores(schedule, first, count, out)`，**复用单个读回缓冲**，
    排序路径一次性算出全部分数（47 次 malloc → 1 次）。单次版改为委托批量版，消除重复逻辑。
- [x] 验证:编译通过;端到端(256×256/1s/steps=4/seed 42/`--ssd-streaming`)产物
      **193579 字节,与基线逐字节一致**,耗时 105s,零报错 → 无回归

### Phase 18d: 审查后修复(B1 崩溃地雷 / D9 死属性 / B3 注释 + 新并发缺陷) — complete
- [x] **B1(高·休眠 → 拔除)**:`h3_video_vae.c` 删除异步 VAE 预取线程(`vae_prefetch_thread` /
      `vae_prefetch_enabled` / `vae_prefetch_job` 及循环中的双缓冲预取逻辑),改为串行
      `load→run→free`。该线程在共享 `vae->gpu` 上分配张量、主线程同时持有 open 命令缓冲,
      并发 ObjC/Metal 访问会崩溃(objc_retain of a dangling pointer),仅 `H3_VAE_PREFETCH=1`
      可触发。一并移除已无引用的 `#include <pthread.h>`。**B2(禁用路径泄漏)随之消除**。
- [x] **D9(中)**:`h3_gpu.m` 删除未实现且误导的 `stream_batch` / `stream_batch_pending` /
      `streamEventValue` 属性及「批量 dequant 一次 submit」注释(`streamEvent` 保留,仍有分配)。
- [x] **B3(中)**:`h3_dit.c` 修正 `read_stream_layer` 误导性注释(原称「无 GPU 命令」实为错误)——
      经核实 `gpu_work` **并非死变量**,它在 `h3_dit.c:1467` / `:1483` 被使用:int8 启用时流式线程
      在共享 `dit->gpu` 上执行 `h3_gpu_begin` / `h3_gpu_quantize_weight_int8` / `h3_gpu_submit`。
- [x] **新发现(高·条件触发)**:DiT 流式线程在 int8 启用时与主线程**竞态共享 `dit->gpu` 命令缓冲**
      (流式线程 `h3_dit.c:1467-1483` 与主线程序行 `run_block` 同时写 `gpu.command` / `gpu.stats`)。
      `h3_gpu_begin` 在 `gpu.command` 非空时直接返回 0 → 任一线程失败或缓冲损坏。D9 移除的
      `stream_batch` 私有命令缓冲本是为解决此问题预留(从未实现)。详见 findings.md「新并发缺陷」。
- [x] 验证:编译通过(严格 `-Wall -Wextra -Wpedantic -Wshadow -Wconversion` 无警告);端到端
      (256×256/1s/steps=4/seed 42/`--ssd-streaming`)产物 **193579 字节,与基线逐字节一致**,
      120s(串行 VAE 解码仅轻微变慢),零报错。

### Phase 18e: 修复 DiT int8 流式线程竞态(高·条件触发) — complete
- [x] **根因**:`read_stream_layer`(流式线程)当 `gpu_work = int8_mlp || int8_qkv || int8_attention_out` 为真时,
      在共享 `dit->gpu` 上执行 `h3_gpu_begin`(:1467) + 3×`h3_gpu_quantize_weight_int8`(:1472-1482) +
      `h3_gpu_submit`(:1483),而主线程序发 `run_block`(:3571)与 `h3_gpu_submit`(:3588)使用同一命令缓冲。
      两线程并发编码同一 `MTLCommandBuffer`(Metal 明确禁止)→ 数据竞争/崩溃。
      `h3_gpu_begin`(h3_gpu.m:943)在 `gpu.command` 非空时直接 `return 0`,使流式线程复用主线已开缓冲并发编码。
- [x] **修复**:采用「移到主线程」方案 —— 新增 `requant_stream_slot(dit, slot, error, error_size)` 在主线程
      对刚流式加载的 slot 做 `h3_gpu_begin`+3×`h3_gpu_quantize_weight_int8`+`h3_gpu_submit`;`read_stream_layer`
      删除全部 GPU requant 代码与 `gpu_work` 变量(现在流式线程只做 CPU 反量化 + LoRA 合并);主循环在
      `pthread_join` 后、`stream_ready_slot` 更新前调用它(此时主线程独占 `dit->gpu`,无并发)。
- [x] **运行时验证**:`fused_mlp=1`(默认)/`use_slower_bf16_mlp=0`(默认)/M4 `tensorOpsEnabled=true` → `int8_mlp=1`,
      该路径在 M4 默认测试(`--ssd-streaming`)中实际运行;修复前后产物均 **193579 字节一致** → 串行 requant 正确。
- [x] 编译:通过(严格 `-Wall -Wextra -Wpedantic -Wshadow -Wconversion` 无警告);端到端 193579 字节,122s,零报错。

### Phase 19: 分支新代码审查 + 修复 6 项 — complete
审查对象:`feature/lora-merge` 相对 `origin/main`(merge-base `92a932c`)的新增 C 代码,
优先覆盖此前未审的 LoRA 合并链路与新的 `h3_superres`。
- [x] 🔴 `h3_lora.c` 适配器名三入口不一致 → `h3_lora_apply` 硬编码 `"default"`,
      而 `h3_lora_matches` / `h3_lora_merge_blocking` 用文件解析出的 adapter。
      turbo-only 适配器被**静默跳过**;且常驻(`blocking=0`)与流式(`blocking=1`)路径结果不同。
      修复:`h3_lora_apply` 改传 NULL 统一语义 + 头注释同步。
- [x] 🔴 `h3_lora.c` 非阻塞分支 `h3_gpu_begin` 开的命令缓冲从未提交(GEAM 自建私有缓冲并等待),
      失败时 `gpu.command` 永久非空 → 之后所有 GPU 阶段失败(全库无 `h3_gpu_discard`)。
      修复:删除该 begin/submit 包装。
- [x] 🟡 `h3_gpu.m`:`h3_gpu_lora_geam_bf16` 与 `..._blocking_...` 函数体逐行等价 → 改为转发,头注释合并。
- [x] 🟡 `h3_lora.c`:分配前加 `SIZE_MAX` 溢出防护(含 `in_dim/rows == 0` 除零防护)。
- [x] 🟡 `h3_lora.c`:`rank` 改为在 `dtype/ndim` 校验之后读取,并拒绝 `rank == 0`。
- [x] 🟡 `h3_ffmpeg.c::h3_superres`:新增 `remove_tree()`(`/bin/rm` 绝对路径 + 失败告警)替换 5 处静默清理;
      `waitpid` 非 EINTR 失败时 SIGKILL + 回收;`bin/mbin/mpar` 加 `snprintf` 截断检查。
- [x] 核实为**非缺陷**(避免无效改动):`target_height % inner_h` 除零不可达
      (`h3_ffprobe_visual_size` 已保证 ≥1);`enc_argv[40]` 实际最多用 31 项。
- [x] 验证:`make -j8 h3 h3_lora_tests` 零新增警告;`h3_tests` 1768 checks、
      `h3_lora_tests`(rel-L2 0.002844 与修复前一致)、`h3_audio_gpu_tests` 全绿。

### Phase 20: 修复既有缺陷(编译警告暴露) — complete
- [x] `h3_audio_vae.c::run_stage`:`int ok = …` 位于 `goto done` 之后 → 分配失败路径
      `return` 未初始化值(`-Wsometimes-uninitialized`)。修复:`int ok = 0;` 提前到失败分支之前。
- [x] `h3_audio_vae.c::decode_output`:`audio->length` 是 `uint32_t`,与 `SIZE_MAX/(STEREO*8)`
      比较在 64 位下**恒为假**(该溢出检查实际无效)。修复:先按 `uint64_t` 计算再校验转 `size_t`。
- [x] `tests/test_lora.c`:删除未使用的 `gpu` 形参(+4 处调用点)、删除格式不匹配且恒打印 0 的调试 `printf`。
- [x] 验证:全量 `make -j8 all test` **0 warnings / 0 errors**;三个套件全绿,数值不变。

### Phase 21: AudioVAE 端到端验证(官方权重)+ 失效断言修复 — complete
- [x] 确认 `models/minimax-h3/FL2VA/*` 为符号链接;`audio_vae/model.safetensors` **1087 张量全 F32**
      (非 int8/convrot 变体)
- [x] 临时 harness 直连 `libh3.a`,用合成 latent 跑真实权重:2ch/29600/@32kHz、全 finite、
      两次解码逐字节一致、136 convs、边界 latent 1→800 / 2→1600
- [x] **发现失效断言**:`tests/test_real_audio_vae.c` 期望 `submissions == 16`,实跑 **23**。
      推导:16 = 1 input + 7 stage-norm + 7 stage + 1 output;Phase 18c 的「上采样后立即 submit
      并释放上一层 hidden」每 stage +1 → 7+16 = 23。→ 已修正断言并写明推导
- [x] 未验证:与参考 oracle 的波形数值 parity(`misc/fixtures/h3_real_audio_vae_37.safetensors` 缺失);
      刻意**未**用 native 输出反造 fixture(会使测试退化为自证)

### Phase 22: 新增不依赖 fixture 的 AudioVAE 端到端测试 — complete
- [x] 新增 `tests/test_real_audio_vae_e2e.c`:断言输出几何、全 finite、确定性(两次逐字节)、
      分发结构(136 convs / 23 submissions,含推导注释)、最短合法 latent(1/2 帧)
- [x] `Makefile`:新目标 `h3_real_audio_vae_e2e_test`、新变量 `AUDIO_VAE_MODEL ?= MiniMax-H3`,
      接入 `make test`(仅需权重)、加入 `clean`
- [x] `AGENTS.md` 登记该入口
- [x] 验证:负例(错误模型路径)exit=1;默认 `make test` skip;
      `make test AUDIO_VAE_MODEL=models/minimax-h3` 实际执行并通过
- [ ] 待办(可选):把 `make test` 里 `h3_real_audio_vae_test` 的守卫路径也参数化到 `AUDIO_VAE_MODEL`

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
| LoRA 适配器名统一为「文件自带 adapter」 | 三个入口(`matches` / `merge_blocking` / `apply`)必须同源，否则 turbo-only 适配器被静默跳过；显式指定仍走 `h3_lora_apply_named` |
| 删除 `lora_merge` 非阻塞分支的 `h3_gpu_begin/submit` | GEAM 已自建私有缓冲并等待；保留只会多一个失败点，并在失败时泄漏未提交的 `gpu.command`（全库无 discard API） |
| `h3_gpu_lora_geam_bf16` 改为转发到 `..._blocking_...` | 两者函数体逐行等价，保留双份实现只会让修复必须改两处 |
| AudioVAE 提交数断言取 **23** 而非 16 | 16 是 Phase 18c 之前的基线；每 stage 多一次「上采样后立即 submit」→ +7。结构变更时两处同时更新 |
| 端到端测试做成**不依赖 fixture** 的版本 | fixture(MLX oracle)在本机缺失，导致整条 AudioVAE 断言长期 skip；新测试只依赖权重，改由几何/确定性/结构/边界兜底 |

## Errors Encountered
| Error | Attempt | Resolution |
|-------|---------|------------|
| 子代理无写文件权限 | 1 | 报告由主代理落盘 |
| `VDN streaming weight is absent or has the wrong schema: transformer_blocks.0.attn.to_out_linear.weight` | 1 | `prepare_vdn_stream_source` 硬性要求 `dtype==BF16`；改为接受 I8 并填 `scale_path/scale_offset/scale_name` |
| 流式反量化报 `required weight_scale absent: …to_out_linear.weight_scale` | 1 | 消费点固定从 `dit->weights` 按名找 scale；改为按 `field == STREAM_LIN_OUT` 选 `dit->vdn_weights` |
| `H3_VDN_INT8=1` → `cannot stream DiT block 1: DiT stream begin failed: unknown Metal error`（13s 退出）| 1 | NAX tensor ops 是 M5 路径，M4 上强制开启会崩；本机不可用，回退 bf16 计算 |
| 用户把 int8/bf16 两个文件对调（`model.safetensors`=int8，`model_bf32.safetensors`=bf16）导致同名张量重复进 store | 1 | 候选名单确定性单文件打开（见 Decisions）；已验证复现逐字节一致产物 |
| 误判"steps=4 仍是黄糊"（只能看统计量、看不到图）| 1 | 用户肉眼确认 4 步有狐狸；统计指标（std/grad）不足以判断语义正确性，结论须以实际观看为准 |
| 临时 harness 自加 `(void)next_value;`（误判为未使用参数，实际在用）| 1 | 直接重写该文件而非局部打补丁；临时程序也要编译告警零容忍 |
| harness 断言 `submissions == 16` 失败 | 1 | 不是我的改动导致的：逐项推导出 16 + 7(STAGES) = 23，根因是 `tests/test_real_audio_vae.c` 的断言在 Phase 18c 后未同步 → 修正测试 |
| AudioVAE 真实权重测试在 `make test` 中一直 skip | 1 | 守卫查 `MiniMax-H3/…`，实际权重在 `models/minimax-h3/…`；新增 `AUDIO_VAE_MODEL` 变量并让新测试可用它 |

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
- **本机权重布局**：`models/minimax-h3/FL2VA/*` 是指向 `/Users/jay/h3_sys/MiniMax-H3-Convrot/…`、
  `/Volumes/data/.lmstudio/models/…` 的符号链接；`audio_vae/model.safetensors` 为 577 MB / 1087 张量全 F32。
- **AudioVAE 验证入口**：`./h3_real_audio_vae_e2e_test models/minimax-h3`（单跑）或
  `make test AUDIO_VAE_MODEL=models/minimax-h3`（随套件）。不依赖 `misc/fixtures`；
  与参考 oracle 的数值 parity 仍需 `misc/fixtures/h3_real_audio_vae_37.safetensors`。
- **提交数/分发结构不变量**：AudioVAE 解码 = 136 MPS conv + 23 submissions
  （23 = 1 input + 7 stage-norm + 7 stage + 1 output + 7 早提交，STAGES=7）。
- 详细审查发现与推导见 findings.md「代码审查与修复 + AudioVAE 端到端验证 (2026-09-11 session)」。
