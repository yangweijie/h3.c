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

### Phase 23: 修复流式 video VAE 解码命令缓冲 bug — complete
- [x] 复现:用官方权重直连 `./h3`(ComfyUI `H3_BinaryT2V` 节点等价参数)跑 T2V,日志在
      `audio VAE 7/7` 后报 `h3: begin streamed video VAE transformer block: unknown Metal error`,
      rc=1,无输出(`vae->streaming` 由自动内存规划器在权重放不下常驻时开启)
- [x] 根因:`run_stream_tile`(`h3_video_vae.c`)L614 用 `h3_gpu_begin` 开了命令缓冲编码 prep ops,
      但**直到 L652 才 submit**,循环里 L638 又 `h3_gpu_begin` → 此时 `gpu.command` 已非空 →
      `h3_gpu_begin` 直接 `return 0` 且不设 `lastError` → 错误串回退成 "unknown Metal error"。
      `h3_gpu_submit` 提交后 `gpu.command = nil` **不重新打开**(重新打开只在 `h3_gpu_continue`),
      故每阶段必须自带 begin。`run_resident_tile`/`run_decoder` 分别用单 begin 包全部 / 每 block 自带 begin,均正常
- [x] 修复:镜像 `run_decoder` 的每阶段 begin/submit——prep 单独 `submit`;循环恢复每 block 的
      `h3_gpu_begin`+`submit`(per-block 提交让 `free_block` 在 GPU 完成后回收权重,保持流式省内存);
      post ops 前补一个 `h3_gpu_begin`
- [x] 验证:直连 `./h3` 完整跑通(448×256,1s,4 steps)`audio VAE 7/7 → FFmpeg 39/39 →
      wrote /tmp/h3_verify.mp4`(270 KB;ffprobe: 448×256,39 帧,1.625s,含音频流);日志无
      Metal/error;全量 `make -j8 all test` 0 警告,      三套件全绿。本 session 此前改动未碰 video VAE

### Phase 24: 修复 ComfyUI 子进程找不到 ffmpeg — complete
- [x] 现象:video VAE 修复后,ComfyUI 跑到 `FFmpeg 0/56` 报 `h3: cannot start FFmpeg: No such
      file or directory`(rc=1)。直连 `./h3`(agent shell PATH 含 ffmpeg)能过 → 是 ComfyUI 子进程
      PATH 不含 ffmpeg
- [x] 根因:引擎 `posix_spawnp("ffmpeg",...,environ)`(h3_ffmpeg.c,8 处)依赖 PATH 查找;已支持
      `H3_FFMPEG`/`H3_FFPROBE` 覆盖为绝对路径。ComfyUI 由 GUI/launchd 拉起时 PATH 仅含
      /usr/bin:/bin 等,不含本机 /opt/zerobrew/bin/ffmpeg(及 /usr/local/bin 软链)→ ENOENT
- [x] 修复(集成层,不碰引擎):`comfyui_nodes/h3_binary.py` 的 `_run_engine` 在传 env 前经
      `_resolve_ffmpeg_env()` 把 ffmpeg/ffprobe 绝对路径注入 `H3_FFMPEG`/`H3_FFPROBE`
      (先 `shutil.which`,失败再扫 zerobrew/homebrew/macports/system 常见目录;用户已设则跳过)。
      **该节点文件在 ComfyUI 目录,不在 h3.c 仓库内**
- [x] 验证:`env -i PATH=/usr/bin:/bin H3_FFMPEG=/opt/zerobrew/bin/ffmpeg ...` 直连 `./h3` 完整跑通
      `FFmpeg 39/39 → wrote /tmp/h3_verify2.mp4`(270 KB);节点 `python3 -m py_compile` 通过

### Phase 25: 修复空白提示词产生无意义画面 — complete
- [x] 现象:ComfyUI 生成的视频"和提示词无任何关系,像编辑器截屏"(近黑+竖向边缘结构)
- [x] 诊断:用户帧 mean=28.9 / bright=0.000 / gy=28.93(竖向边缘主导);空白提示词 `-p "   "` 复现
      完全同形态(mean=49.8 / bright=0.008 / gy=49.76);正常提示词 mean=138.7 / bright=0.048。
      三者对比证实:空白提示词通过引擎 `!*prompt` 检查后 tokenize 近零嵌入 → DiT 生成无条件先验
      (近黑+竖向结构,恰似暗色编辑器截屏)
- [x] 修复(集成层):`_build_cmd` 在发命令前 `prompt.strip()` 校验,空白/纯空格抛 ValueError 给
      出清晰提示(而非等 2 分钟产出垃圾)。**节点文件在 ComfyUI 目录,不在 h3.c 仓库内**
- [x] 验证:`python3 -m py_compile` 通过;引擎侧仅缺 ClipProj 时才需要 text_encoder 权重
      (h3.c:774-796,ClipProj 激活时容忍 text_encoder 缺失)

### Phase 26: 定位"画面与提示词无关 + 节点无 prompt 框"真因(接线错误) — complete
- [x] 现象:Phase 25 修复后仍产出近黑"文字/编辑器截屏"画面,且节点上**找不到 prompt 输入框**
- [x] 诊断:`GET /object_info/H3_BinaryT2V` 显示 required 第一项确为 `prompt`(multiline),节点定义正常;
      再读 `user/default/workflows/h3_binary_t2v.json` 发现 node31 的 `prompt` 带 `"link":2`,
      由 node10 `H3_BinaryInfo.info` 连入 → prompt 被 Convert to Input(故界面无文本框),
      引擎实际收到的是 `h3 --info` 的整段环境日志文本 → 生成"文字"般画面
- [x] 结论:**非引擎/节点 bug**,是工作流把"环境检查节点输出"误接进了 prompt
- [x] 修复指引(界面):删除该连线 → 右键 prompt 输入点 Convert Input to Widget → 填真实提示词重跑
- [x] 记录:findings.md「生成画面与提示词无关 / 节点无 prompt 输入框 (2026-09-11)」

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


# ===== Task Plan（当前任务）: Metal 上下文编译缓存 — DeepJIT 启发的三项加速 =====

> 上方为历史任务（VDN-H3 支持 / 代码审查 / AudioVAE 验证）的完整记录，保留备查。

## Goal
参考 DeepJIT（「源码 hash + 编译选项 + 编译器版本 → 缓存编译产物 + lazy init」），
在**不改变任何算术**（端到端产物必须逐字节一致）的前提下削减 Metal 上下文创建耗时。

h3.c 现状：`h3_gpu_create()`（`h3_gpu.m:338`）用 `newLibraryWithSource` 编译
`h3_shaders.metal`（280 KB / ~180 个 kernel），**无任何二进制缓存**；同进程内 7 处调用点
（DiT / Qwen 文本编码器 ×2 / video VAE ×3 / audio VAE ×2 / video encoder）
会重复编译同一份源码。编译耗时当前被并入 "Metal context" 的 wall，未单独计量。

## 三项改动（独立实施、独立测量收益）
| # | 改动 | 机制 |
|---|---|---|
| 1 | 进程内共享 `MTLLibrary` | 静态模块缓存，key = 设备 + 源码 + 宏状态；同进程只编译一次 |
| 2 | 磁盘二进制缓存 | `MTLBinaryArchive` 持久化编译产物，跨进程复用 |
| 3 | lazy pipeline 创建 | 只建实际用到的 pipeline（是否提升为进程级共享由实测决定） |

## 验收标准（每项改动各自过一遍）
1. `make -j8 all test` 0 warning / 0 error
2. `h3_tests` / `h3_audio_gpu_tests` / `h3_real_audio_vae_e2e_test` 全绿
3. 端到端 `256×256 / 1s / steps=4 / seed 42 / --ssd-streaming` 产物**逐字节一致**
4. 给出量化收益：`tests/bench_metal_context.c` 微基准 + `H3_PROFILE` 实测

## ⛔ 结论（2026-09-12 实测后更新）：三项改动**全部不做**

实测（Apple M4 / macOS 26）：

| 测量 | 数字 |
|---|---|
| 单次上下文创建（暖，系统缓存命中） | **~5 ms**（library 1–3 ms + 86 个 pipeline 1–7 ms） |
| 冷缓存（源码内容改动 → 系统 shader 缓存 miss） | library **0.226 s**，pipeline 仍 **2 ms** |
| 真实 T2V 一次生成的上下文数 | **4 个** |
| 4 次 shader-build 合计 | **23 ms** |
| 同一次运行的 DiT 主体 wall | **34.7 s** |

→ 三项改动的**上限收益 ≈ 10–15 ms/次（0.04%）**；即便整机冷缓存，一次性也只有 ~0.9 s。
→ 投产判断：**不值得增加任何缓存代码**。

**根因（关键认知）**：Metal 自带**系统级 shader 缓存**
（`/private/var/folders/*/C/com.apple.metal`，按源码内容命中；同内容第二次编译 0.226 s → 0.001 s）。
DeepJIT 的磁盘缓存对 CUDA 有价值，是因为 CUDA 侧**没有**这层系统缓存；
这份经验**不可移植到 Metal** —— 自己再做一个 `MTLBinaryArchive` 只是重复 OS 已有的机制。

## Phases
### Phase 27: 基线测量 + 计时插桩 — complete
- [x] `h3_gpu_create` 加 `H3_PROFILE` 计时：`library=`（源码→MTLLibrary）、`pipelines=`（构建数）
- [x] 新增 `tests/bench_metal_context.c` + `make h3_metal_bench` 目标（同进程连续 N 次 create）
- [x] 记录基线数字；并用「改一行源码制造缓存 miss」做判决性对照，推翻「编译很贵」的假设

### Phase 28: 改动 1 — 进程内共享 MTLLibrary — **取消（收益 ~10ms）**
### Phase 29: 改动 2 — 磁盘二进制缓存（MTLBinaryArchive）— **取消（OS 已有同层缓存）**
### Phase 30: 改动 3 — lazy pipeline 创建 — **取消（86 个 pipeline 合计 ~2-7ms）**
### Phase 31: 汇总收益 + 收尾 — complete（结论：不做，保留诊断工具）

## 保留物（唯一产出）
| 产物 | 位置 | 用途 |
|---|---|---|
| `H3_PROFILE` 的 shader-build 计时 | `h3_gpu.m`（`h3_gpu_create`，~15 行） | 本次判决性诊断的工具；后续任何 Metal 启动相关排查都可复用 |
| `h3_metal_bench` 微基准 | `tests/bench_metal_context.c` + Makefile | `make h3_metal_bench && ./h3_metal_bench 5` 复测上下文创建成本 |

## Decisions Made（本任务）
| Decision | Rationale |
|---|---|
| **先插桩再优化**（本次奏效） | 假设「280KB 源码编译很贵」是错的：暖缓存仅 3 ms。不做测量就会写出 300 行无收益的缓存代码 |
| 用「改源码制造 cache miss」做对照 | 仅测暖缓存会把结论建立在 OS 缓存上；冷/暖对照才暴露出真实机制 |
| 三项改动全部放弃 | 上限收益 0.04%，与代码复杂度/风险不成比例（原则：不为一次性问题增加抽象） |
| 保留计时与基准、不保留缓存代码 | 诊断工具可复用且零风险；缓存代码是纯负债 |

## Errors Encountered（本任务）
| Error | Attempt | Resolution |
|---|---|---|
| 真实引擎首次探针跑成交互模式（日志只有 `Goodbye.`） | 1 | `nohup ... ./h3 -d ...` 漏了 `-p`；补 `-p "..."` 后正常出 `shader-build` 行 |
| 初始假设「编译 280KB / 180 kernel 需数秒」 | — | 被实测推翻（暖 3 ms / 冷 226 ms）。**教训与 Phase 10 同源：先确认测量对象，再谈优化** |


# ===== Task Plan（当前任务）: DiT 流式权重管线的读/算重叠 =====

## Goal
去噪 wall 的 60% 花在主线程等 prefetch 线程 `pthread_join`（`unhidden wait 19.945s / 33.189s`），
而该线程内部 **pread 17.568s 与 CPU 反旋转 16.000s 严格串行**。把这两段重叠起来。

**目标**：去噪 33.2s → **≈18–19s（约 1.75×）**，端到端产物**逐字节一致**。
**地板**：磁盘 2.06 GiB/s × 36.248 GiB = 17.6s（读盘已贴该卷实测上限 2158 MiB/s）。

## 方案（选定）
在 `read_stream_layer`（`h3_dit.c:1379`）内做**生产者/消费者拆分**：
- 生产者线程：按 source 顺序 pread（含 scale）到 staging ring（2 槽）
- 消费者（现有 prefetch 线程）：取 staging → `convrot_unrotate_cpu` → `h3_gpu_tensor_write_bf16`

**为什么不用 GPU 反旋转**：已有 kernel 且 M4 通过测试，但 Phase 10 已判定 GPU dequant 与主线程
计算争抢 GPU 净亏；先把纯 CPU 侧的重叠做掉。

## 验收标准
1. `make -j8 all test` 0 warning / 0 error；`h3_tests` / `h3_audio_gpu_tests` 全绿
2. 端到端 `256×256 / 0.5s / steps=2` 产物与 `/tmp/h3_probe3.mp4` **逐字节一致**
3. `unhidden wait` 从 19.9s 降到 <3s；去噪 wall ≤20s

## Phases
### Phase 32: 相位计时插桩 — complete
- [x] `read_stream_layer` 拆分 `stream_pread_seconds` / `stream_dequant_seconds`
- [x] teardown 打印 `pread %.3fs, cpu-unrotate %.3fs`
- [x] 基线：`pread 17.568s / cpu-unrotate 16.000s / unhidden wait 19.945s / denoise 33.189s`

### Phase 33: 按 source 分片并行 — complete
（**修正**：最终实现的是「按 source 分片并行」而非原计划的「生产者/消费者流水线」——
后者每 block 需 condvar + staging ring，先用更小的改动拿到大部分收益。）

- [x] `h3_dit_stream_source` 循环拆成 `read_stream_sources(job, indices, count)`
- [x] 新增 `h3_dit_stream_worker`（自带 job 计数器 + source 索引表）
- [x] `read_stream_layer` 变为编排器：默认 2 个 worker（连续区间），pthread 并发，
      合并 counters/错误，再在两个线程都结束后做 LoRA 合并
- [x] 旋钮 `H3_DIT_STREAM_WORKERS`（1 = 原串行参考路径）
- [x] 安全前提：每个 source 写**不同** slot 张量（qkv/out/fc1/fc2/lin_to_out），
      并发写互不相交；每个 worker 拥有自己的 job（含 error 缓冲与计数器），无共享可变状态

### Phase 34: 验证 byte-identical + 量化收益 — complete
见下方「结果」。

### Phase 35: worker 内 SPSC 两级流水 — complete
- [x] `h3_stream_ring`（2 槽 SPSC）+ `stream_fill_slot`（reader）/ `stream_consume_slot`（dequantizer）
      + `run_stream_pipeline(worker, pipelined)`
- [x] 旋钮 `H3_DIT_STREAM_PIPELINE=0`（同一份代码背靠背跑，无重复实现）
- [x] 2×2 对照矩阵见下「最终结果」

### Phase 36: 继续压（直写 slot / NEON / 分块）— complete
- [x] **36a 反旋转直写 slot**：新增 `h3_gpu_tensor_bf16_storage()`，去掉 230MB 暂存 + 一次全量拷贝
      → `cpu-unrotate` 19.57 → 16.96s，denoise 21.339 → **21.060s**，逐字节一致。保留
- [x] **36b NEON 反旋转 — 实测否决（0.84×，未进仓库）**：`-O3` 已自动向量化标量版，
      手写 intrinsics 反而多了 4 路解交错 shuffle 开销。过程中修掉一个真 bug：
      `vcgeq_u32` 返回全 1 掩码（不是 1），`hi+mask` 变成 `hi-1`，正好造成 bf16 off-by-2
- [x] **36c 流水 stage 降到 1024 行**：ring 单元 矩阵 → chunk，M 由 2 升到 ~30–60
      → denoise 21.060 → **19.962s**，staging 内存 115MB → 5.5MB/worker，逐字节一致。保留
- [x] per-source scale 缓存（否则每 chunk 重读整个 scale 张量）；实测中性，按「避免
      O(chunks×rows) 潜在退化」保留
- [x] 验收：`make test` **exit 0**；`PIPELINE=0` 不挂死且逐字节一致（22.55s）；构建仅
      `h3_dit.c`/`h3_gpu.m` 零警告

### Phase 37: 终点结论 — **已贴磁盘地板**
每 block 每 worker：read 173.5ms（2 reader 合计 2.22 GB/s ≈ 该卷上限 2.16 GiB/s）、
dequant 87ms → 理想 `max = 173.5ms` → 去噪 ≈17.4s，**当前 199.6ms，只差 15%**，
且剩余已分散（condvar 交接 + 内存带宽争抢），无单一主导项。

### Phase 38: 逐分辨率 A/B（git stash 原始二进制对照）— complete
**关键结论：优化只在低分辨率有效，默认配置（864×480）无收益。**

| 分辨率 | 原始 | 优化 | 加速 | 原始 unhidden wait |
|---|---|---|---|---|
| 256×256 | 33.296 s | 19.962 s | **1.67×** | 19.945 s |
| 384×384 | 33.922 s | 26.746 s | **1.27×** | 7.561 s |
| 512×512 | 46.441 s | 47.454 s | 0.98× | **0.001 s** |
| 864×480 | 78.869 s | 77.579 s | 1.02× | **0.001 s** |

- 流式耗时本身与分辨率无关（原始 34.1–35.3s → 优化 19.9–20.3s，稳定 −42%），
  证明优化按设计工作；但 ≥512×512 流式在**原始代码里就已被隐藏**，故整体收益为 0
- 产物在 384/512/864 三档均**逐字节一致**
- 交叉点 ≈448×448（取决于 GPU 算力/磁盘带宽之比）
- **引擎默认是 864×480 / 56 帧 / 20 步 → 默认配置下收益为 0**

### Phase 39: 决策 = **A（保留 + 默认关闭 + 一键开关）** — complete
- [x] 翻转默认：`workers=1`、`pipelined = workers>1`
- [x] **`H3_DIT_STREAM_WORKERS=2` 一个变量 = 整条快路径**（分块 + 直写 + 分片 + 流水）
- [x] 关键修正：**分块与反旋转直写也必须挂到同一开关**。它们原先没挂，导致「默认关闭」
      实际不是原始行为 —— 实测分块在 512×512 是 **+3.7%**、直写在 512×512 是 **+1.2~2.8%**
      （只在低分辨率/关键路径上才划算）
- [x] 默认态 = 原始：512×512 背靠背 **47.588s vs 47.631/47.653s**，逐字节一致
- [x] 快路径：`H3_DIT_STREAM_WORKERS=2` @256×256 = **19.760s（1.70×）**，逐字节一致
- [x] ComfyUI `fast_stream` BOOLEAN → `H3_DIT_STREAM_WORKERS=2`，附自动提示
- [x] README「Streamed DiT weight prefetch (opt-in)」小节
- [ ] 待办：同步仓库外的 ComfyUI 部署副本（`/Volumes/data/Documents/ComfyUI/custom_nodes/h3_binary_nodes/`）

### 交付总表
| 项 | 结果 |
|---|---|
| 默认行为 | 与改动前逐字节一致、耗时相同（512×512 −0.1%） |
| 一键加速 | `H3_DIT_STREAM_WORKERS=2`：256×256 **1.70×**，384×384 **1.27×**，≥512 无效 |
| 改动文件 | `h3_dit.c` / `h3_gpu.h` / `h3_gpu.m`（0 警告）、`comfyui_nodes/h3_binary.py`、`README.md` |
| 验收 | `make test` exit 0；`h3_tests` 1768 checks；所有 A/B 产物逐字节一致 |

## Decisions Made（本任务）
| Decision | Rationale |
|---|---|
| 只做 CPU 侧重叠，先不碰 GPU 反旋转 | Phase 10 已证 GPU dequant 争抢；且读盘是地板，CPU 降到 17.6s 以下无额外收益 |
| 不改反旋转算法/顺序 | 保证 byte-identical，验收可 `cmp` |
| 保留 ring 深度 2 | staging 内存翻倍但可控（~+350MB），避免撑爆 16GB 机的 9.5 GiB working set |

## 结果（A/B，同一机器，256×256 / 0.5s / steps=2）
| 量 | `H3_DIT_STREAM_WORKERS=1`（= 基线） | 默认（2 workers） | 变化 |
|---|---|---|---|
| Euler denoise wall | **33.296 s** | **22.686 s** | **−31.9%（1.47×）** |
| unhidden wait | 20.201 s | 9.223 s | −54.3% |
| stream pread（线程累计） | 17.826 s | 25.587 s | 并发读让 SSD 吞吐上升 |
| stream cpu-unrotate（累计） | 15.866 s | 17.229 s | ~持平 |
| 有效吞吐 | 1.076 GiB/s | 1.577 GiB/s | +46.6% |
| 产物 | — | — | **逐字节一致** |

- `make -j8 test AUDIO_VAE_MODEL=models/minimax-h3` → **exit 0**，全部可跑套件绿
- 构建 **0 warning / 0 error**（严格 `-Wall -Wextra -Wpedantic -Wshadow -Wconversion`）

## 最终结果（2×2 对照，同一 binary，产物全部逐字节一致）
| 配置 | denoise wall | unhidden wait |
|---|---|---|
| W=1, P=0（原始串行参考） | 35.131 s | 22.101 s |
| W=1, P=1（只流水） | 25.189 s | 11.546 s |
| W=2, P=0（只切片 = Phase 33） | 23.586 s | 10.239 s |
| **W=2, P=1（默认）** | **21.339 s** | **7.957 s** |

（W = `H3_DIT_STREAM_WORKERS`，P = `H3_DIT_STREAM_PIPELINE`；对照最初未改动的基线 33.296 s。）

- 相对原始基线累计 **33.296 → 21.339 s（1.56×）**；每 block 337ms → 217ms；吞吐 1.08 → 1.67 GiB/s
- 验收：`make -j8 test AUDIO_VAE_MODEL=models/minimax-h3` **exit 0**；构建 0 warning

## 余下空间（已量化，未做）
每 block：read 阶段 ≈159ms（2 reader 合计 318ms ≈2.42 GiB/s）、dequant 阶段 ≈98ms。
理想 `max(159,98)=159ms` → prefetch ≈15.9s → 去噪 ≈16s。**当前 217ms，差 27%。**
按 stage 粒度模型（`wall = R₁ + Σmax(Rₖ,Dₖ₋₁) + D_N`），把 source 再切 N 段：
N=4→199ms、N=8→179ms、N=16→169ms（收敛 159ms）→ 只到 ~19.5s，且改动不小。
**更值得做的是加快 `convrot_unrotate_cpu` 蝶形本身**——dequant 已占关键路径一半。

## Errors Encountered（本任务）
| Error | Attempt | Resolution |
|---|---|---|
| `-Wshadow`：`read_stream_sources` 新参数名 `count` 与循环内既有的 `uint32_t count` 冲突 | 1 | 参数改名 `source_count`（不动既有局部变量，遵守「精准修改」） |
| 按数量均分导致负载不均（154.1M vs 231.2M 元素/块） | 1 | 改用 LPT 按字节均衡 → 实测 **23.15s，无收益**（噪声内）→ **回退为连续区间**：source 已按文件偏移排序，连续区间保住 SSD 顺序局部性 |
| 4 个 worker 反而更慢（denoise 25.19s，pread 累计 52.05s） | 1 | 并发读过多会让该 SSD 退化；2 个是甜点，旋钮保留以便复测 |

---

# 任务：memory-plan 未接线字段处置（encoder_streaming / cache_budget_bytes）

## Goal
给 `--video-vae-streaming 0` 做 A/B 时发现 `h3_memory_plan` 的输出里有两个字段从未被消费。
查清引入时机与是否被撤回，按「合理撤回→删除 / 有研究对比意义→接线实测后再定」处置。

## 调查结论（git 证据，已完成）
| 字段 | 引入提交 | 是否曾被消费 | 性质判定 |
|---|---|---|---|
| `encoder_streaming` | `d5752a0`（2026-08-28，"让 16/24GB 设备也能跑"） | **否** —— `git log -S "params->encoder_streaming"` 全历史零结果 | 意图已由架构天然实现，冗余 |
| `cache_budget_bytes` | `d5752a0` 同批 | **否** —— `git log -S "cache_budget" -- h3.c h3_dit.c h3.h` 全历史零结果 | 意图未实现，属预留 |

- `d5752a0` 的 h3.c diff 里，同批次的 `video_vae_streaming` 一路传到 decoder load/decode；
  `eff.encoder_streaming = plan.encoder_streaming;` 是孤立赋值，`cache_budget` 在 h3.c 根本没出现。
- 所以**不是"逻辑被撤回"，而是"从未接线"**。

### encoder_streaming 无意义的原因
- 其声明意图（h3.h）："Release the text/image encoder after condition building"。
- 实际：`h3_text_encode_bf16` / `h3_text_encode_clipproj_bf16` 是**一次性函数** ——
  内部 `h3_weight_store_open()` → 跑完 → `h3_weight_store_free()`
  （h3_text_encoder.c:517/522/540/758），无持久 encoder 对象。
- planner 自己的注释就写着 `text_encoder.bytes * 0 /* freed per call */`（h3_memory_plan.c:20）。
- → 意图已天然满足，接线反而要新增"encoder 常驻缓存"，无收益场景。

### cache_budget_bytes 有研究对比意义的原因
- 意图（源自 ds4 的 streaming cache planner）：`7/8 × working_set − steady_streamed`，
  GiB 对齐、下限 1 GiB（h3_memory_plan.c:97-112）。
- 直击当前最大瓶颈：denoise 每步重流全部 50 块（实测 72.136 GiB / 4 步，73 s，
  unhidden wait 6.4 s + pread 35.0 s + cpu-unrotate 38.5 s）。
- 量化：每块 ≈ 0.36 GiB（72.136/4/50）；8.0 GiB 预算可常驻 ≈22 块 → 每步只流 28 块
  → 读取量约 −44%。
- 代价：`stream_slots[2]` 是**硬编码双槽轮转**（h3_dit.c:218 + `h3_stream_ring`），
  "部分块常驻"需改 ring 调度，属中等改动。

## Phases
### Phase 1: 调查与判定 — complete
- [x] git 历史定位（`d5752a0`）
- [x] 验证「从未接线」而非「撤回」
- [x] 判定 `encoder_streaming` 冗余（freed-per-call 天然实现）
- [x] 判定 `cache_budget_bytes` 有研究价值（直击 SSD 重流瓶颈）

### Phase 2: 代理实验（先行验证收益假设，不写代码）— complete
- [x] `--layers 40`（块数 −20%）→ SSD 读 57.781 GiB（**−19.9%**）、denoise 58.84 s（**−18.4%**）→ 线性成立
- [x] 推导出关键路径模型（见下），据此估算常驻缓存的收益

**关键路径模型**（两个数据点一致）：
| 块数 | denoise wall | pread + cpu-unrotate | root-gpu |
|---|---|---|---|
| 50 | 72.11 s | 35.00 + 38.53 = 73.53 s | 14.31 s |
| 40 | 58.84 s | 28.35 + 30.90 = 59.25 s | 11.36 s |

`denoise ≈ (pread + cpu-unrotate) × 0.98`，GPU 只占 14 s —— **读取/反旋转链完全在关键路径上**。
所以「缓存 K 块」的收益 = `denoise × (1 − K/50)`，而非 layers 那种「少算又少读」。
- 8.0 GiB / 0.36 GiB每块 ≈ 22 块 → denoise 72 → ~41 s → 总时间 112.5 → ~81 s（**−28%**）

### Phase 3: 删除 encoder_streaming — complete
- [x] h3.h / h3_memory_plan.h/.c / h3.c / h3_cli.c / AGENTS.md 全链路移除（无 tests 引用）
- [x] reason 文本 "SSD+VAE+encoder streaming on" → "SSD+VAE streaming on"
- [x] 编译通过；端到端 md5 `0d353faf2173acaab0badd4f9af62bf4` 与删除前**逐字节一致**（117.1 s，噪声内）

### Phase 4a: 实现常驻缓存（选项 A）— complete（结论：本机无收益）
- [x] 前置确认：`quantize_block_*` 会释放 bf16（仅诊断开关下保留）；
      `fuse_next_attention` 只读 `norm1`（流式块也有）；`load_core` 在
      `configure_gate_ranked_blocks` 之后调用
- [x] 实现：`H3_DIT_RESIDENT_BLOCKS`（env-only opt-in，默认 0）+ `resident_blocks`
      / `block_resident[]` + `first_streamed_block` / `next_streamed_block`
      + `resident_needs_bf16` + `adopt_slot_weights` + `load_core`/主循环分流
- [x] 编译 0 warning（顺带删除因改动而孤立的 `next_active_block`）
- [x] **正确性（踩坑后修好）**：第一版常驻块走 `load_block`（`bf2_convrot`）→
      md5 变（`82a84345…`）。改成让常驻块也走流式读取路径（同源）后，
      K=4 / K=8 产物 md5 均 = 基线 `0d353faf2173acaab0badd4f9af62bf4` ✓
- [x] 回归：K=0 与改动前完全一致（md5 + 读取量 72.136 GiB）

**实测（256×256 / 2 s / 4 步 / seed 42）**
| K | 读取量 | pread+unrotate | denoise | peak | md5 |
|---|---|---|---|---|---|
| 0（基线） | 72.136 GiB | 73.53 s | 72.11 s | 1.65 GiB | `0d353faf…` |
| 4 | 67.830 GiB | 70.52 s | 73.27 s | 4.52 GiB | `0d353faf…` |
| 8 | 63.523 GiB | 67.94 s | **74.02 s** | 7.39 GiB | `0d353faf…` |

**判定：磁盘 I/O 不是 denoise 的瓶颈。** 读取量降 12%、读取链短 5.6 s，
denoise 却 +1.9 s。按 200 次块执行反推：流式块 ≈0.36 s/次（预取主导），
常驻块反而 ≈0.42 s/次 —— 8 块常驻 5.8 GiB 后统一内存压力抵消了 I/O 节省。
这与 VAE 常驻是两回事：VAE 常驻逻辑预载 9.0 GiB 但为**可驱逐**共享缓冲，实测峰值仅 5.1–6.2 GiB，从未 OOM（见 findings.md 订正），不能据此类比。

**外推**：常驻块本身更快（0.42 vs 0.36 是在本机被内存压力拖累后的结果；
无压力时按计算量应 ≈0.21 s/次）。若内存宽裕，K=20 可让 denoise ≈60 s（−17%）。
即该功能**在 ≥32 GB 机器上可能有效，在 16 GB 上无效**。

### Phase 4b: 处置 — complete（用户选 A + C）
- [x] **A**：保留 `H3_DIT_RESIDENT_BLOCKS` 为 env-only opt-in（默认 0，零影响），
      并写入 README 新小节「Partially resident DiT blocks (opt-in)」——
      含 K=0/4/8 实测表 + 「必须走流式读取路径，否则数值会变」的坑
- [x] **C**：删除 `cache_budget_bytes` 字段、`h3_memory_cache_budget_bytes()`
      函数与声明，清理 planner reason 里误导的 "cache X GiB"，同步 AGENTS.md
- [x] 编译 0 warning；代码零残留；端到端 md5 仍为 `0d353faf2173acaab0badd4f9af62bf4`

**为什么不自动接线**：按 `cache_budget` 换算 K（8.0 GiB / 0.72 ≈ 11 块）会比
K=8 更慢 —— 该公式（`7/8 × working_set − steady`）过于乐观，未计入 macOS
统一内存在压力下的退化，实测 8 GiB 预算就已伤性能。
**重要**：这里「接线」不是接一根线，而是要**新建一个「流式权重常驻缓存」机制**。
- 现状约束（h3_dit.c:3974-3998）：`stream_slots[2]` 双槽轮转，且主循环有严格顺序断言
  `stream_ready_layer != block || stream_ready_slot > 1` → fail。
- 实现路径（最小侵入）：流式模式下把**前 K 块完整 load 进 `dit->blocks[0..K-1]`**
  （复用既有 `load_block`），主循环 `block < K` 时跳过 stream 分支，预取序列从 K 起。
- 成本：中等，触及 DiT 核心调度；风险：8 GiB 部分 DiT 块常驻在 16 GB 上可行性未知
  （注：不能拿 VAE 常驻作反证——VAE 常驻 9.365 GiB 的「OOM 过一次」论断已证伪：常驻权重可驱逐，峰值仅 ~6 GiB）。
- 选项 A：按上述实现 → 实测 → 定「新增开关」或「回退删除」
- 选项 B：直接删除（含 `h3_memory_cache_budget_bytes()` 与 reason 里的 "cache X GiB" 误导文本）

## 待验证的假设
1. denoise wall 与「每步流式块数」近似线性（`--layers 40` 应降到 ~80%）。
2. 8 GiB 部分 DiT 块常驻在 16 GB 上的可行性需另测（见 H3_DIT_RESIDENT_BLOCKS 实测：wall time
   几乎不降，因与流式路径争抢统一内存）；**不能**用 VAE 常驻作反证——VAE 常驻逻辑预载 9.0 GiB
   但为可驱逐共享缓冲，实测峰值仅 5.1–6.2 GiB，从未 OOM（原「VAE 9.365 GiB OOM 过一次」论断在
   统一内存下不成立）。
3. `cache_budget` 与 `video_vae_streaming=0` **不再互斥**：VAE 常驻权重可驱逐、不钉死 9 GiB；
   且 `cache_budget` 本身为未接线的死字段（见 findings.md）。

## Errors Encountered（本任务）
（暂无）

# ===== Task Plan（当前任务）: H3 latent 子系统（ComfyUI）往返保留音频 =====

## Goal
补完「生成 latent → latent 空间上采样 → latent 回解视频」这条 ComfyUI 节点链
（`H3_BinaryLatent` / `H3_BinaryLatentUpscale` / `H3_BinaryLatentDecode`），
并让它在往返过程中**保留 H3 原生音频** —— 与官方
「`VAEDecode` + `VAEDecodeAudio` 并行 → `CreateVideo`」等效。
用户选定方案 **(i)：空间上采样 + 音频 latent 透传**。

## 硬约束（源码级确认，非假设）
- 去噪产出两个**独立**缓冲：`video`（z 空间 `[C=24,T,H,W]`）与 `audio`
  （归一化音频 latent，`[32,2,T]` channel-major，见 `h3_audio_vae.h:17-24`）。
- 音频 latent 是 **3D、无 H,W 空间维** → 空间上采样只动视频 latent 的 H,W，
  音频原样透传；时间维不变则音视天然对齐。
- `h3_ffmpeg_write_av_rgb24_f32` **不接受 NULL pcm**（`h3_ffmpeg.c:620-624` 直接失败），
  故「无音频」时也须显式写静音轨。

## Phases
### Phase 1: 确认音频 latent 真实 shape — complete
- [x] `h3_audio_vae.h:17-24` 坐实 `h3_audio_latent{channels=32,stereo=2,length}` → `[32,2,T]`
- [x] 确认落盘点（h3.c:2119）在音频 VAE 解码（2135）与 `free(audio)`（2141）**之前**，
      `audio` 缓冲有效且未被 transform → 直接 dump 即精确往返

### Phase 2: 引擎 latent 格式升级 v1→v2（携带音频）— complete
- [x] `write_video_latent` → `write_latent_bundle`（同时写视频 latent + 音频 latent）
- [x] `read_video_latent` → `read_latent_bundle`（读回两者；含 version 校验）
- [x] `h3_decode_latent` 末尾：有音频 latent → `h3_audio_vae_decode` 解波形再 mux；
      无 → 回退静音轨（兼容旧 latent）

### Phase 3: Python 节点接线 — complete
- [x] `_read/_write_h3_latent` 升级 v2，LATENT 字典携带 `h3_audio` 键
- [x] `H3_BinaryLatent`：输出 LATENT 含 `h3_audio`
- [x] `H3_BinaryLatentUpscale`：**只动 `samples` 空间维，透传 `h3_audio`**（音频不失真的核心）
- [x] `H3_BinaryLatentDecode`：写回时带 `h3_audio` → 引擎据此合成带音轨 MP4

### Phase 4: 端到端验证 — complete
- [x] 编译 0 error（唯一 warning 为既存 `conditioning_key` 未使用，与本任务无关）
- [x] factor=1 往返：解码音频与原片音频 **MD5 完全一致**（`56ab826…`）→ 真透传、非静音
- [x] factor=2 上采样：视频 `16×16→32×32` → 512×512；音频 `[32,2,93]` 未动，仍解出原音频

## Decisions Made（本任务）
| 决策 | 理由 |
|---|---|
| 音频 latent 与视频 latent 存**同一 bundle 文件**（而非两个文件） | 保持 `--latent-out/--latent-in` 单路径设计，与既有实现一致 |
| 上采样限定为**空间（H,W）**，不改时间维 | 音频 latent 无空间维；T 不变则音视帧数天然对齐，无需扩音频时序 |
| 格式升到 **v2**（与 v1 不兼容） | v1 无 version 字段、无法区分是否含音频；中间产物无需向后兼容 |
| 音频 VAE 路径取 `FL2VA/audio_vae` | latent 节点属 FL2VA（T2V）定位；R2V latent 未涉及 |

## Errors Encountered（本任务）
（无 —— 首轮即通过；验证阶段未出现失败重试）

---

# ===== Task Plan（当前任务）: H3 DiT int4 路线（量化感知蒸馏的前置 + 本地可执行部分）=====

## Goal
让 H3 DiT 的 **int4 权重**在质量可接受的前提下真正可用：把「离线方案搜索 → 引擎可加载 → 端到端验收」
在本机（M4 / 16 GiB）范围内走完；把必须依赖 GPU 集群的量化感知蒸馏（QAT/DMD2）整理成可交接规格。

## 成功标准（可验证）
| # | 标准 | 验证方式 |
|---|---|---|
| S1 | 256×256 / 0.5s / **4 步** 下 int4 方案相对 base 判定 `preserved`（saturation ≥0.95× 且 detail ∈[0.80,1.25]×），拼图肉眼无伪影/糊化 | `ab_quant_quality.py` + `montage_steps4.png` |
| S2 | 全 DiT 权重 ≤ 14 GiB（源 21.33 GiB，≥ −34%）；单块流式 ≤ 0.25 GiB（源 0.359） | 导出目录字节数 |
| S3 | 引擎原生加载 int4 并端到端出片（rc=0、无 Metal 报错） | `./h3 -d <int4>`、`--info` |
| S4 | 可复现：脚本 + 报告 + 拼图落盘，命令可复制粘贴 | 目录清单 |
| S5 | QAT 规格书：数据配方、显存/时间估算、FastVideo 侧改动点 | 文档 |

## 价值主张与边界（先钉死，避免做完才发现没用）
- 本机 M4 **无 int8/int4 张量核路径**（`h3_gpu.m:378-380` 仅在 M5 或 `H3_VDN_INT8` 时开 `tensorOpsEnabled`）
  → int4 在本机**只省 I/O，不省计算**。
- 引擎默认 864×480 下 I/O 已被完全掩盖（`unhidden wait` 0.001 s）→ **默认配置 int4 收益 = 0**。
- 收益区间：① ≤384² 小画布预览（I/O 在关键路径），上限约 1.3–1.5×；② **M5+/未来硬件**（张量核下 int4 吞吐 ≈ int8 的 2×）。
- 结论：本任务定位是「**为低分辨率预览与未来硬件准备 int4 格式与质量**」，不是「加速默认生成」。

## 已知事实（本轮实测，详见 findings.md）
- int4-g64 PTQ 权重 relRMS **0.0913**（int8-g64 为 0.0052）
- 端到端：2 步 SSIM 0.521 / 4 步 0.598（base=1.0）；画质代理判定 int4 **degrades**（2 步 grainy、4 步 desaturated），int8-g64 **preserved**
- 引擎**目前无法加载 int4**：前述 A/B 走的是代理路径（int4 → 反量化 → 重量化 per-row int8 → 现有 int8 路径）
- `h3_safetensors.h:19` 已有 `H3_DTYPE_U32`；但 `h3_weights.c::load_int8_dequantized` 只处理 I8 + `_scale`
- 引擎无激活 dump 钩子（只有 `H3_DEBUG_CONVROT` / `H3_DEBUG_VISION`）
- FastVideo `MLXQuantizationSpec.from_name` 把 group_size 硬编码为 64（`fastwan.py:66-70`）；`mx.quantize` 本身支持任意 group_size
- 本机 anaconda env 无 gptq/awq/llm-compressor（只有 mlx 0.32.2）→ 误差补偿 PTQ 需自实现

## Phases

### Phase 0: 基线固化与代理灵敏度校验 — pending
- [ ] 归档 int8-g64 / int4-g64 checkpoint、AB 报告、拼图（`/Volumes/data/work/h3_qad/ab_report/`）
- [ ] 代理灵敏度：证明「重量化回 per-row int8」的代理层不会掩盖方案差异（int8-g64 已 preserved → 代理可用）
- 验证：`--skip-run` 复现同样数字

### Phase 1: 离线量化方案扫描（不改引擎，先做）— in_progress
- [x] 1a 位宽 × group size 曲线：{4,6,8} × {32,64,128}（`quant_scheme_sweep.py`，88 s）
- [x] 1a' **int6-g64 端到端验证 → 通过（preserved，2/4 步一致）**，−12.5% 字节
- [x] 1a'' **int6-g128 端到端验证 → 通过（preserved，2/4 步一致，余量充裕）**，−18.8% 字节
- [ ] 1d **评测加固**（必须做，先于任何「<14 GiB」的结论）：
      - 现规则只在远离边界处校准过；混合方案（pooled 0.058）在 2/4 步之间互相翻转 → **边界附近不可判**
      - 手段：多 seed（≥3）× 多 prompt（≥2）聚合，并报告代理量的**跨样本波动**，把阈值改成「均值 ± 波动」
- [ ] 1b scale/zero 用 F16 省 0.06 B/param（权重误差不变，纯省字节）
- [ ] 1c 混合位宽：**已探边界但不可判**，暂停至 1d 或 Phase 3 完成
- [ ] 交付：候选方案表（字节 / 权重 relRMS / 端到端 verdict + 波动）
- 验证：权重部分沿用 `mlx_int8_to_h3.py --verify` 同款误差口径；端到端用 `ab_quant_quality.py`
- 预计：权重扫描秒级；每个候选导出 ~2.5 min + AB ~5 min；1d 加固每候选 ×6 次运行

#### Phase 1 目前可落地的结论（2026-09-13）
| 方案 | 字节 | 权重 relRMS | 端到端 verdict | 状态 |
|---|---|---|---|---|
| int6-g64 | −12.5%（15.70 GiB） | 0.0238 | **preserved（2/4 步一致，视觉等同）** | ✅ 可用 |
| **int6-g128** | **−18.8%（14.58 GiB）** | 0.0256 | **preserved（2/4 步一致，余量充裕：satur× 1.093/1.200）** | ✅ **当前最优** |
| int6-g128 + F16 累加器 | −21.9%（14.02 GiB） | 不变 | 待验（Phase 1b） | ⏳ 下一步 |
| int4-g64 | −37.5%（11.22 GiB） | 0.0913 | degrades（2 步 grainy / 4 步 desaturated） | ❌ 需 Phase 3 |
| mixA（MLP=int4） | −27.5%（13.0 GiB） | 0.0577 | **翻转**：4 步 degrades / 2 步 preserved | ⚠️ 不可判 |
| mixB（attn=int4） | −22.5%（13.9 GiB） | 0.0576 | **翻转**：4 步 preserved / 2 步 degrades | ⚠️ 不可判 |
| int8-g64 | **+12.5%（20.19 GiB）** | 0.0053 | preserved | 无收益 |

**判据边界（实测标定）**：preserved ⇔ 权重 relRMS ≲ 0.026；degrades ⇔ ≳ 0.091。
0.058 一档不可判 → 说明**要突破 14 GiB 必须先把 relRMS 压到 ~0.03 以下**，即走 Phase 3。
「余量」是判定可靠性的关键：int6-g128 距阈值 +15%/+26%（远大于 ±0.15~0.25 的步数波动），
而 mixA/mixB 距阈值仅 −7%/−1%（在噪声内）→ 前者可信，后者不可判。

### Phase 1b（立即可做，纯收益）— pending
- [ ] int6-g128 的 scale/zero 从 F32 换 F16：字节 14.58 → **14.02 GiB（−21.9%）**，权重误差不变（缩放因子精度足够）
- [ ] 验证：同 `quantize_h3_proxy.py` 加 `--accumulator f16` + A/B；
      注：原生交付也需此改动（Phase 2 的格式里 scale/zero 用 F16）

#### Phase 1a 结果（已完成，2026-09-13）
| 方案 | B/param | 全 DiT | vs 源 | relRMS |
|---|---|---|---|---|
| int4-g32-asym | 0.7500 | 13.46 GiB | −25.0% | 0.0807 |
| int4-g64-asym（现 int4） | 0.6250 | 11.22 GiB | −37.5% | 0.0910 |
| int4-g128-asym | 0.5625 | 10.10 GiB | −43.7% | 0.1003 |
| int6-g32-asym | 1.0000 | 17.95 GiB | 0% | 0.0193 |
| int6-g64-asym | 0.8750 | 15.70 GiB | −12.5% | 0.0217 |
| int6-g128-asym | 0.8125 | 14.58 GiB | −18.8% | 0.0239 |
| int8-g32-asym | 1.2500 | 22.43 GiB | +25.0% | 0.0047 |
| int8-g64-asym | 1.1250 | 20.19 GiB | **+12.5%（更大）** | 0.0053 |
| int8-g128-asym | 1.0625 | 19.07 GiB | +6.2% | 0.0059 |

（对称变体一致地差 ~11%，如 int4-g64-sym=0.1075；MLX `affine` 即非对称形式）

三条硬结论：
1. **非对称优于对称 ~11%**，且 MLX affine 已是此形式（代理路径已吃到）。
2. **group size 空间很小**：int4 从 g32→g128 仅跨越 0.081↔0.100 → **靠粒度救不了 int4**（目标量级差 17×）。
3. **误差约每 bit 降 2×**；**int8-g64 比源 per-row int8 更大** → 8 bit 以上无量化意义。
   **int4 与 int6 是仅有的两条有效方向。**

→ 重排：Phase 3（误差补偿）成为 int4 的关键路径；Phase 1a'（int6 是否 already 可用）是当下最高性价比的补测。
   校准验证：`int4-g64-asym` 扫描值 0.09096 vs 对 MLX 产物实测 0.09127（差 0.3%）→ 实现与实测互相印证。

### Phase 2: 引擎原生 grouped 量化加载路径（int6-g128 已验证；int4 复用同路径待 Phase 3）— complete
- 格式（自描述，无元数据张量；相对原提案改为 **U8 打包 + bias**，与导出器 `export_h3_int6_native.py` 对齐）：
  - `{name}.weight` **U8** `[rows, cols*bits/8]`（6-bit=4 值/3 字节；4/8-bit 同理）
  - `{name}.weight_scale` / `{name}.weight_bias` **F16**（Phase 1b 的省字节改动已一并落地）`[rows, cols/group]`
  - 识别：`h3_weight_grouped_spec` 由 dtype==U8 + 形状推导 `bits = packed_row_bytes*8/cols`、`group = cols/scale_cols`（group 须为 4 的倍数）
  - dequant：`w = code*scale + bias`，**在 ConvRot 反旋转之前按组乘**（group=128 非旋转块 256 的倍数 → 先解包成 float、按组乘、再反旋转）
- 改动（654 行净增）：
  - `h3_weights.h/.c`：spec 推导 + `h3_weight_dequantize_grouped` + `h3_weight_unrotate_rows`（复用 int8 路径的 radix-4 蝴蝶）+ `h3_weight_load_grouped_accumulators`（常驻/流式共享）
  - `h3_dit.c`：常驻 `load_block` 与 SSD 流式 `prepare_stream_source`/`stream_consume_slot` 都接入 grouped 分支（解包 float → 按组乘 → 反旋转 → 写 BF16 slot）；int8 路径零改动
- 验证（已闭环）：
  - 单测 `tests/test_grouped_weights.c`（`make h3_grouped_tests`）：block 0 四矩阵（qkv/out/fc1/fc2）经 `h3_weight_load_bf16` 真实入口加载，相对 Python golden **relRMS 0.000e+00**（容差 1e-3）→ 逐值等价 ✓
  - 编译：严格标志 0 warning / 0 error ✓
  - 端到端：`/Volumes/data/work/h3_qad/ab/native`（transformer→`h3_int6g128_native`）；`montage_native_int6g128.png` 五联对比、2 步/4 步均 `preserved`，`/tmp/h3_nat_stream.mp4` rc=0 无 Metal 报错 ✓
  - 字节：int6-g128 原生 **14.58 GiB（−18.8%）真实落盘**（代理路径恒 21.4 GiB，不体现收益）✓
- **结论**：int6-g128 原生加载 = Phase 2 交付完成，质量等价于 Phase 1 代理判据（preserved）。
  int4（bits=4）**复用同一加载器**，无新引擎代码；可用性门槛由 Phase 3 GPTQ 把 relRMS 压到 ~0.03 以下决定，3e 通过即直接可用（`export_h3_int6_native.py --bits 4`）。

### Phase 3: 校准激活导出 + 误差补偿 PTQ（GPTQ）— in_progress
- [x] 3a 引擎加 `H3_DUMP_ACT=<dir>` 钩子（`h3_dit.c`，4 个调用点，0 warning）
- [x] 3b 激活感知方案诊断：**AWQ 负收益（0.891×）**；一次性 whitening 无效（病态）；
      ConvRot 已提供 5–7 个数量级对角均衡
- [x] 3b' **单层真 GPTQ 判决性测试：4.36× 输出误差下降**（0.03865 → 0.00886）
- [x] **修正记录**：初版 Python 用完整 `H⁻¹` 而非 `H⁻¹` 的上三角 Cholesky 因子 → 三层全部变差；
      靠「damp→∞ 应收敛回 plain」的自检发现
- [ ] 3c **放大校准集**：现 n=1068 ≪ d_in(5376~14336)，H 严重欠定；
      需 ~10k+ 行/层（更多步数 × 更多 prompt，dump 体积从 3.2 GB → ~30–60 GB）
- [ ] 3d **全 200 层 GPTQ 跑批**：实测 729 s/层（qkv），外推 ≈45 h 单线程；
      按 block 并行（M4 10 核）目标墙钟 ~6–8 h；产出 h3 格式代理 checkpoint
- [ ] 3e **端到端 A/B**：int4-GPTQ vs base（2/4 步）。注意**不能用权重 relRMS 判据**
      （GPTQ 故意抬高 relRMS），只能用 `ab_quant_quality.py` 的 verdict
- 风险：45 h 长跑 + 校准集放大是一次真正的工程投入；若 3e 不通过，int4 路线即终止

#### Phase 3 已验证结论
| 项 | 结果 |
|---|---|
| 引擎 dump 钩子 | 200 文件 / 3.2 GB / 每层 1068 行；端到端出片正常；非 dump 运行零影响 |
| ConvRot 对角均衡（每通道二阶矩 max/min） | 1.86e8 → 645（均值），单层最高 **3.7e6×** |
| AWQ 通道缩放 | 层输出误差 0.05377 → 0.06038（**0.891×，负收益**），20/20 层变差 |
| 一次性 whitening | 失效（病态，误差 550）→ 不是有效上界 |
| **GPTQ（修正后，单层 qkv）** | **0.03865 → 0.00886（4.36×）**；权重 relRMS 0.091 → 0.357 |
| GPTQ 成本 | 729 s/层（d_in=5376）→ 全 200 层 ≈45 h 单线程 |

### Phase 4: 旋转自由度（仅分析，低优先）— pending
- 现状：变换矩阵硬编码（`h3_dit.c::build_convrot_hadamard`），改变它需引擎支持可配置 T
- 仅做离线分析：不同 Hadamard 变体 / 符号翻转下 int4 误差分布，判断是否值得动引擎

### Phase 5: 端到端验收与产物固化 — pending
- 2 步 + 4 步 AB、拼图、报告、复跑命令写入 README 小节

### Phase 6: QAT 规格交接（本地不可执行）— pending
- 显存/时间估算（19.27B 参数 QAT 的规模）；FastVideo 侧改动（H3 的 `ATTN_QAT_TRAIN`/DMD2 端口）；数据配方

## Decisions Made（本任务）
| Decision | Rationale |
|---|---|
| 先做 Phase 1 离线扫描，再动引擎 | 引擎支持 int4 是大改动（新 dtype + group 尺度 + 反旋转顺序）；先用秒级权重误差曲线筛掉不可能达标的方案 |
| 评估沿用「重量化回 per-row int8」代理路径 | 引擎暂无 int4 加载；已实测该代理层只引入 ~0.5% 底噪（int8-g64 仍判 preserved）→ 灵敏度足够 |
| 位宽/粒度扫描自实现（不用 FastVideo 导出器） | ① 其 group_size 被硬编码 64；② 其输入是已删除的 66 GB diffusers 中间产物；③ 源即旋转空间 int8，自实现可精确控制旋转语义 |
| 交付格式用形状/dtype 自描述，不加元数据张量 | 避免再出现 `comfy_quant` 那种「文件里有、引擎不读」的字段 |

## Errors Encountered（本任务）
| Error | Attempt | Resolution |
|---|---|---|
| `./h3` 所有命令 `exit=137`、零输出（`--help` 也死） | 1 | 根因是二进制被标 `com.apple.quarantine` → Gatekeeper SIGKILL（`spctl` 判 rejected）；`xattr -d com.apple.quarantine h3` 解决。与权重/内存/目录无关 |
| 首版 int8-g64 与 int4-g64 结果不可分辨（SSIM 均 ≈0.46） | 1 | 根因是 QKV 行序错误：旧脚本多套一层 head-interleave。实测源为 module-major（stored 0.0047 vs interleaved 1.404）；删除置换后 int8 0.891 / int4 0.521 |
| 首版 `--verify` 通过但输出是错的（自洽假阳性） | 1 | 两侧使用同一置换，误差被约掉；改为同时报 stored 与 interleaved 两个数 |
| 用 SSIM 误读「4 步比 2 步更差」 | 1 | SSIM 量的是轨迹分歧不是画质；新增 `quality_proxies`（detail/saturation/luma_std）+ `verdict()` 并写入模块文档 |

## Next Action
Phase 1a：{4,6,8} bit × {32,64,128} group 的权重 relRMS 曲线（无副作用、秒级）→ 决定哪些组合进端到端 AB。

---

# ===== Task Plan（当前任务）: video VAE 解码流式化（消除分辨率×时长内存墙）=====

## Goal
video VAE 解码不再把整段 RGB 累积在内存：改为逐时间分片（temporal chunk）解码，
每个 chunk 解出后立刻 `f32→u8 →(resize)→(on_frame 预览)→ pipe` 进 FFmpeg。
峰值从「整段 RGB = 帧数×宽×高×3」降到「一个 chunk ≈ 0.5 GiB@1080p + decoder 权重」，
与时长/帧数解耦。两条生成路径都改：主生成（`h3_generate`）与 `--latent-in` 调试路径。

## 改动（已闭环）
| 文件 | 改动 |
|---|---|
| `h3_ffmpeg.h/.c` | 新增 `h3_ffmpeg_mux`（spawn ffmpeg，video+audio pipe，异步音频线程；`h3_ffmpeg_mux_open`/`pipe_frame`/`close`） |
| `h3_video_vae.c` | `decode_chunked`（多 chunk）与 unpack_frames 单 chunk 分支加 `sink` 回调；`h3_video_vae_decode` API 加 `sink`/`sink_opaque` 形参 |
| `h3_video_vae.h` | 新增 `h3_vae_frame_sink` 签名 + `h3_vae_sink_ctx` |
| `h3.c` | 新增 `h3_vae_frame_sink`（f32→u8→resize→on_frame→pipe）；主路径改用 sink 流式；`h3_decode_latent`（`--latent-in`）改用 sink 流式（先解 audio/silent 开 mux，再流式 video） |

## 关键设计点
- 时序重叠混合同原逻辑：`decode_chunked` 用 `temporal_overlap`（5 帧前段）与每 chunk 后 12 帧拼出完整 17 帧 chunk；单 chunk 路径（t==2）解 5 帧。sink 只替换「累积 final_rgb」这一步。
- 像素级等价：解码/重叠/混合代码未改，sink 仅逐 chunk 消费 final_rgb 的等价切片 → 输出与旧路径逐字节一致（同种子对照）。
- resize 在 sink 内每 chunk 做（native→params），等价原一次性 resize 整段。
- `--latent-in`：因 mux open 需 audio，调整顺序为 audio/silent 先解 → open mux → 流式 video decode；silent 时长用 `((t-2)/5)*17+5`（= VAE `output_frames`）。

## 验收（已闭环）
- 编译：严格 `-Wall -Wextra -Wpedantic -Wshadow -Wconversion` **0 warning / 0 error**（仅 1 个预存 `conditioning_key` 未使用 warning，与本次无关）；`libh3.a` 链接成功
- 主路径端到端跑通（448×256 / 1s / 4 steps）→ `FFmpeg 39/39 → wrote …mp4`（270 KB，ffprobe 含音视频流），日志无 Metal/error
- 像素级等价：**待用户实测对照**（代码审查保证；旧路径可用 `git stash` 对照，或对比流式前构建）

## 结论
- 峰值从「整段 RGB（1080p×长视频可达数 GiB）」降到「单 chunk ≈ 0.5 GiB@1080p + VAE 权重」
- 与分辨率/时长**解耦**：时长只影响耗时（chunk 数 × 每 chunk 解码），不影响峰值
- **step 数增长对峰值无影响**（见 progress.md 本 session 节论证）

---

# ===== Task Plan（当前任务）: 长时长 15s 内存墙诊断 + 运行时守卫 + 分段生成 =====

## Goal
在 Apple M4 / **16 GiB / swap=0** 上产出 15s @ 864×480 / steps 4 的视频。
先回答「VAE 流式化后，decode 单 chunk 的 int6/int8、同时长下峰值与时长有无撞墙」。

## 最终结论
- **单次 15s 在本机不可行（硬件墙）**：16 GiB 物理 + swap=0；denoise 激活 ∝ video token；
  VAE decode 的 GPU wired 内存逐 chunk 累积（且**不计入进程 `phys_footprint`**）。
- **交付**：分段硬切拼接 `seg_out/final.mp4` = **16.36s / 864×480 / 含音频 / 5.8 MB**。

## Phases
### Phase 40: 权重与量化路径定位 — complete
- int6=`h3_int6g128_native`；int8 预量化(h3_int8 / h3_int8g64) 均缺 `condition_proj.bias`
  → 依用户指示改用 BF16 `minimax_h3_fastvideo_4step.safetensors` + 运行时 int8。

### Phase 41: 15s 崩溃诊断（3s/6s 实测取证） — complete
- 3s：int6/int8 均成功，VAE 单 chunk GPU peak **0.654 GiB**（流式化生效）。
- 6s：denoise 第 1 步 footprint 10.11 → **13.88 GiB**（+3.77/步）。
- 6s + `--token-reduction`：峰值降到 13.00 → 完成。
- 15s 单次：**整机死机重启 ×2** → 定性为硬件墙。

### Phase 42: 运行时内存守卫 `h3_host_memory_guard` — complete
- `h3_host.h/.c` 新增守卫（available + footprint 双条件）+ `h3_host_footprint` + physical 查询。
- 插入 `h3_dit.c` denoise **CPU/GPU 两个步循环**、`h3_video_vae.c` **resident/chunked 两个 chunk 循环**。
- `h3_memory_plan.c` 预算收紧 `min(rec×0.8,(physical−4GiB)×0.85)`。
- 验证：6s 在 denoise 1/4 优雅退出（不再死机）；3s 默认 floor 不误杀。

### Phase 43: `--token-reduction` 降激活验证 — complete
- 6s denoise 峰值 13.88 → 13.00 GiB，使 6s 可跑完。

### Phase 44: 分段生成（硬切拼接） — complete
- `gen_segments.sh`：2s × 7 段独立进程（递增 seed），ffmpeg concat。
- 产物 `seg_out/final.mp4` = 16.36s。

### Phase 45: 跨段视觉条件调研 — complete（负结果）
- H3 vision encoder 三处不可得 → `--first-frame` 不可用 → 段间只能硬切。

## Decisions Made（本任务）
| Decision | Rationale |
|---|---|
| 守卫以 **available 为主、footprint 为辅** | VAE decode 时进程 footprint 仅 ~2 GiB 而系统 available 降 6.2 GiB → GPU wired 不计 footprint，只看 footprint 会漏 |
| plan 预算 `min(rec×0.8,(physical−4GiB)×0.85)` | Metal 建议值过于乐观；以物理内存为硬上限更安全 |
| 段间**硬切**（无 `--first-frame`） | H3 vision encoder 三处均不可得，跨段视觉条件无法使用 |
| 段长取 2s（56 帧） | 单段峰值 ~9.3 GiB，远离 16 GiB 墙；3s 已达 12.7 GiB 太贴边 |

## Errors Encountered（本任务）
| Error | Attempt | Resolution |
|---|---|---|
| 15s 单次 → **整机死机重启** ×2 | 2 | 判定硬件墙；改用 2s 分段 + 运行时守卫 |
| int8 加载报 `condition_proj.bias` absent | 1 | int8 导出缺该张量 → 用 BF16 transformer + 运行时 int8 |
| `--first-frame` 报 `visual.pos_embed` shape mismatch | 2 | 缺 H3 vision encoder（HF 不可达/ModelScope 无）→ 放弃 |
| 第 7 段 `Killed: 9`(SIGKILL) | 1 | 系统内存压力保护（**未死机**）；单独重跑成功 |
| 后台 nohup 进程被工具会话终止 | 1 | 改前台 + `tr '\r' '\n'` 实时输出 |

## Next（用户指示 2026-09-15）
1. **降分辨率测试**：下次用 **256×256** 测 15s 单次生成。
   预估：256² 空间 token 约为 864×480 的 **16%**；15s vs 6s 帧数 ×2.5
   → 15s@256² 的 token 总量 ≈ 864×480@6s 的 **0.4×** → 单步激活 ≈ **1.5 GiB**
   （864×480@6s 为 3.77）→ 单次 15s 大概率可跑通。仍带 `h3_host_memory_guard` 作安全网。
2. **段间连贯**：需补 H3 vision encoder（`h3-base/text_encoder`，含 `model.visual.*`），
   之后把 `gen_segments.sh` 每段加 `--first-frame <上段末帧>` 即得连贯（替代硬切）。

### Phase 46: 256×256 / 15s 单次生成 — complete（✅ 成功）
- 实测：denoise 峰值 **11.43 GiB**（available 2.67 GiB）→ **未触发守卫**；
  VAE decode footprint 1.47 GiB、available 12.64→8.42 GiB（~20 chunks，单调下降但充足）。
- 产物 `out256_15s.mp4` = **15.08s / 256×256 / 含音频 / 659 KB**，总 **898s**，RSS 8.37 GiB，零报错。
- **结论：低分辨率（256×256）下单次 15s 完全可行**，与 864×480 的硬件墙形成对照。

# ===== Task Plan（当前任务）: video VAE 跨 chunk 权重复用 + DiT GPU 侧剖析 =====

## Goal
承接 256×256/15s 之后的性能线，两个独立目标：
1. **VAE**：消除 `run_stream_tile` 每 tile 重读整份 VAE 权重的浪费（要求无损、逐字节可验证）。
2. **DiT**：在 16 GiB M4（**无 M5 TensorOps/int8 路径**）上量化各加速开关的真实收益，
   并解决"DiT 性能测量不可复现"的问题。

## 最终结论
- **VAE 跨 chunk 复用完成：242.8s → 103.5s（−57.4%），SSIM 1.000000/inf = 逐字节无损**。
  两条 decode 入口（`decode_chunked` / `h3_video_vae_decoder_decode`）均已接入，**默认开启、无开关**。
- **DiT 是 compute bound，I/O 被完全隐藏**（20 步 base：wall 775.7s vs SSD stream 335.4s，
  `unhidden wait 0.010s`）。
- **权重 I/O 与分辨率无关**：每次真实评估固定读 ~14.4 GiB（layers 45）→ 低分辨率下 I/O 反超 GPU
  （576×320：GPU 58.9s vs I/O 86.7s，`unhidden wait` 0.010→33.227s）。
- 方法论发现：**DiT 自适应驻留必须钉住**（`H3_DIT_RESIDENT_BLOCKS`），否则测量全部失效。

## Phases
### Phase 47: VAE 权重重复加载定位 — complete
- 实测 `run_stream_tile` 被调用 **24 次**（8 tiles × 3 chunks），每次完整读 36 块（268.6 MB/块）。
- 累计 `alloc=216.520 GiB` ↔ 24 × 9.67 GB = 232.0 GB（吻合 99.8%）。
- CPU 侧 146.1s = `242.837 − 96.700(GPU wait)` → **60% 的 VAE 阶段在搬权重**。
- 代码注释自述"lost overlap only marginally slows"的判断是**错的**（实测 60%）。

### Phase 48: block-major 重构（per-chunk 粒度） — complete
- 抽出 `pack_hidden` / `finish_hidden`；新增 `run_stream_chunk`：块循环提到外层，
  每块加载一次、对全部 tile 各跑一遍。
- 关键实现约束：`h3_gpu_copy_f32` 必须与 `pack_hidden` 在**同一命令缓冲**内（首次漏了导致空错误退出）。
- 结果：864 → **108 次 load**，242.8 → **116.5s**，SSIM 1.0。

### Phase 49: 跨 chunk 复用（36 次 load，理论下限） — complete
- `decode_chunked` 先打包全部 chunk 的 states（3×8=24 个，400 MB）→ 一次块遍历 →
  再按原顺序 finish/stitch（时间融合有 chunk 间依赖，未动）。
- 结果：**103.5s（−57.4%）**，alloc 216.5 → 9.784 GiB，peak 0.654 → 1.025 GiB，submissions 912 → 84。
- 1 GiB states 上限 + 两级回退（per-chunk → per-tile）。

### Phase 50: resident decoder 路径同步 + 测试补齐 — complete
- `decoder_decode_chunk` 加 `states` 参数（NULL = 原行为）；`h3_video_vae_decoder_decode` 接入跨 chunk。
- 该路径 CLI 不可达（`preview_denoise` 需真终端、`h3_terminal_detect()` 非交互返回 NONE；
  `model_cache_enabled` 无 CLI 调用者）→ 用测试覆盖。
- 新增 `tests/test_semantic_vae.c --streaming-parity` + 合成 fixture；
  `Makefile` 加 `VIDEO_VAE_MODEL` 变量 + 测试钩子。
- 验证：**两条流式路径逐字节一致**（`memcmp` 整帧）。

### Phase 51: DiT GPU 侧 A/B — complete
- 踩坑：自适应驻留漂移（peak 6.1~11.1 GiB）→ 首轮 A/B 得到 **314s 假数据 + 一次 OOM**。
- 钉住 `H3_DIT_RESIDENT_BLOCKS=6` 后 peak 稳定 6.096~6.099 GiB，数据才可比。

### Phase 52: 分辨率 vs 瓶颈翻转 — complete
- 576×320/10 步/combo：`unhidden wait` 0.010s → **33.227s**，I/O 反超 GPU。

## DiT A/B 实测（864×480×22，`H3_DIT_RESIDENT_BLOCKS=6`）
| 配置 | 2 步 | 20 步 | vs base |
|---|---|---|---|
| base | 75.015s | 775.689s | — |
| `--layers 45` | 67.838s (−9.6%) | — | — |
| `--token-reduction` | 49.076s (−34.6%) | — | — |
| `--reuse 2` | 75.468s（2 步下无效，无步可跳） | **419.618s** | **−45.9%** |
| combo（三者叠加） | 44.149s (−41.1%) | **273.959s** | **−64.7%** |

dispatch 交叉验证：base `linear=4000`（20×50×4）；`reuse 2` → `2200`（**只跑 11 次前向**）；combo → `1980`（11×45×4）。

## 每步评估的 I/O 下限（分辨率无关）
| 配置 | 每次评估 | 耗时 @0.997 GiB/s |
|---|---|---|
| `--layers 50` | ~16.0 GiB | ~16.0s |
| `--layers 45` | ~14.4 GiB | ~14.4s |
| `--layers 40` | ~12.8 GiB | ~12.8s |

10 步 + `--reuse 2` = 6 次评估 → 86.4s，**实测 86.741s**（吻合）。

## Decisions Made（本任务）
| Decision | Rationale |
|---|---|
| 跨 chunk **默认开启、无开关** | 触发条件纯运行时判定（streaming + states ≤ 1 GiB），与 `--video-vae-streaming` 正交 |
| states 上限 **1 GiB** | 16 GiB 机器余量；超限回退 per-chunk（仍比 per-tile 快 8 倍） |
| 保留 per-tile 路径 | 作为 ≥1080p / 长视频（>8s）的回退 |
| 放弃 `H3_VAE_TILE_PIXELS=512` | block-major 后 load 次数与 tile 数无关，而 512 让 GPU wait +20s |
| DiT 侧不引入新开关 | `--token-reduction/--layers/--reuse` 已覆盖，且三者均为有损 |

## Errors Encountered（本任务）
| Error | Attempt | Resolution |
|---|---|---|
| `stream chunk calls: 0`（新路径未生效） | 1 | 实际走 `decode_chunked` 而非 `decoder_decode_chunk`，**两条入口都要改** |
| 首个 tile 后 `exit=1`、`h3:` 后错误信息为空 | 1 | `h3_gpu_copy_f32` 被放在 `submit` 之后 → 移入同一命令缓冲 |
| `--layers 45` 跑出 314s（应 −10%） | 1 | 自适应驻留漂移 + swap → 钉住 `H3_DIT_RESIDENT_BLOCKS=6` |
| `--token-reduction` `exit=137`（OOM） | 1 | 同上 |
| `--show` 无法触发 resident 路径 | 3 | `h3_terminal_detect()` 非交互返回 NONE（`script` 造 pty 亦无效）→ 改用测试覆盖 |
| `--render-width 432×240` 被拒 | 1 | 非 32 倍数；合法档位仅 288×160 / 576×320 / 864×480 |
| `h3_semantic_vae_test` 缺 fixture | 1 | 生成合成 latent（parity 测试只需路径一致，不需 MLX 真值） |

## 本轮改动文件
| 文件 | 改动 |
|---|---|
| `h3_video_vae.c` | 净 +~400 行：`pack_hidden`/`finish_hidden`/`pack_chunk_states`/`finish_chunk_states`/`run_stream_chunk`；`decode_chunk_streaming` 重写；`decode_chunked` + `h3_video_vae_decoder_decode` 接入跨 chunk |
| `tests/test_semantic_vae.c` | +56 行，`--streaming-parity` |
| `Makefile` | +9 行，`VIDEO_VAE_MODEL` 变量 + 测试钩子 |
| `misc/fixtures/h3_vae_streaming_parity_256x256x39_f32.safetensors` | 新增（合成 latent，seed 20260916） |

## Next
1. **DiT 侧决策**：三开关均有损。combo 画质 PSNR 14.54 dB（vs base）、SSIM 0.643，
   但**目视结构完整、仅运动相位偏移**（`/tmp/s20cmp_*.png`）。保守选项：
   `--reuse 1 --token-reduction --layers 45`（−41%，无跨步外推）。
2. **1 GiB states 上限**：长视频（>8s）/1080p 会回退到 per-chunk，可考虑放宽或按可用内存自适应。
3. **低分辨率 I/O bound**：576×320 档已贴住 I/O 下限（86.7s），进一步只能动 `--layers`/`--reuse`，
   `--token-reduction` 在此档基本无效。

# ===== Task Plan（当前任务）: Latent Upscaler 接入（两阶段 hires-fix）=====

## Goal
把 ComfyUI 的 `Comfyui_Minimax_h3_latent_Upscaler`（神经潜空间放大器）工作流接入 h3c：
低分辨率生成 → latent 放大 → 高分辨率 refine，用便宜的小分辨率步替代部分昂贵的大分辨率步。

## 背景数据（决定这条路值得做）
- 权重 **659 MB**（`ComfyUI/models/latent_upscale_models/minimax_h3_latent_upscaler_3d_fp16.safetensors`，
  322 张量全 fp16）
- 结构：24× ResBlockEmb3D + 12× TemporalConv + conv_in/conv_out；
  **48 个 Conv3d(512→512, 3×3×3)** 占绝大部分计算
- **总计算量 15.58 TFLOPs**（864×480 的 latent 14×54×30）→ M4 十核约 **5–12 秒**（fp16 峰值 ~9 TFLOPS，小核效率按 30%）
- 对照：1280×704 的一步 denoise ≈ **238s** ⇒ 放大成本 ≈ **0.05 步**，可忽略
- 官方管线（`workflow3.json`）：`ResolutionSelector 0.2MP` → 5 步 → 放大到 `1 MP` → 4 步 refine
- **注意**：官方 sigma（`ManualSigmas [0.9035, 0.8, 0.6316, 0.3158, 0]`）来自
  `BasicScheduler(simple, 8)`，与 h3c 的 `h3_serving_schedule_build`（linear base grid）**不同源**

## Phases
### Phase 53: 能力缺口定位 — complete
- h3c 缺"从 latent 续跑"：`--latent-in` 的语义是 `skipping denoise`，只解码
- 放大器网络未移植（需 Conv3d kernel）

### Phase 54: `--refine-sigma`（img2img / hires-fix） — complete
- `h3.h`：`h3_params.refine_sigma`（0 = 保持原解码语义）
- `main.c`：`--refine-sigma S`（新增 `parse_float`，范围 [0,1] 且拒绝 NaN）+ usage；
  无 prompt 时警告并回落到解码路径
- `h3_host.c/.h`：新增 `h3_refine_schedule_build(start_sigma, steps, *schedule)`
  —— 反解 `sigma = shift·b/(1+(shift−1)·b)` 求起始 base，保持发布曲线形状；
  **两模态各自反解**，故 video/audio 都从同一 sigma 起步
- `h3.c`：新增 `h3_seed_refine_latents()` —— 读 bundle、校验视频元素数、
  按 `x = (1−σ)·clean + σ·noise` 重新加噪（复用同一 per-modality RNG 流 ⇒ 固定 seed 可复现）；
  音频尺寸不匹配时警告并退回纯噪声
- 测试：`tests/test_h3.c` +19 行（起始 sigma 精度、单调性、边界拒绝）→ 1768 → **1788 checks**

### Phase 55: 端到端验证 — complete
| 对照 | SSIM(All) | 含义 |
|---|---|---|
| C（refine σ=0.05，2 步）vs A（5 步生成） | **0.9575** | 从 A 的 latent 起跑，几乎复现 |
| D（纯噪声，2 步）vs A | 0.7880 | 独立生成 |

- 尺寸校验通过；`linear` dispatch 计数符合预期（5 步 = 1000，4 步 = 800）

### Phase 56: 放大器所需 GPU 原语（B1–B3） — complete

**关键转折：B3 最大的未知数（GroupNorm）发现仓库已有现成实现。**
`h3_gpu_vae_encoder_group_norm_silu_f32` 就是 NDHWC 布局下的 `GroupNorm(groups,C)+SiLU`，
正好是放大器 `in_layers` / `TemporalConv.norm` 的形式。于是 B3 从"要新写 GroupNorm"
缩成"只补一个去掉 SiLU 的变体"。

| 子项 | 内容 | 状态 |
|---|---|---|
| B1 | `h3_gpu_conv3d_pad_f32`（padding + groups 参数化） | ✅ |
| B2 | `h3_resize_bilinear_f32`（CPU，对 PyTorch 参考值逐点吻合） | ✅ 41 项测试 |
| B3 | `h3_gpu_group_norm_f32`（无 SiLU）+ `h3_gpu_channel_scale_shift_f32` | ✅ 编译零 error |

- 放大器索引映射：`res_index(b) = b + (b+1)/2`，`temp_index(b) = res_index(b)+1`（b 为偶数时）
- 张量数核对：in_blocks 156 + out_blocks 156 + 边界 10 = **322** ✓ 与 safetensors 一致
- 已有原语可直接复用：`h3_gpu_silu_f32` / `h3_gpu_linear_f32` / `h3_gpu_add_scaled_f32` / `h3_gpu_conv3d_pad_f32`
  ⇒ **网络可完全用现有 C 原语搭建，不必走 MPSGraph 组合**

### Phase 57: 放大器网络组装（B4） — pending（已降级，见 Decisions）
| 子项 | 内容 | 风险 |
|---|---|---|
| B4a | 权重加载（322 张量 + 索引映射） | 机械，低 |
| B4b | 前向组装（24 ResBlock + 12 TemporalConv） | 中 |
| B4c | CLI + 管线接线 | 低 |
| B4d | 对 PyTorch 逐点验证 | **真正的验收门槛** |

### Phase 58: ComfyUI 桥接 + 工作流 — complete（并修掉 3 个阻塞级 bug）

`h3_binary_nodes` 是**软链到本仓库**的，故改动直接生效。

| # | Bug | 后果 |
|---|---|---|
| 1 | `_read_h3_latent` 只认 v2，而引擎 **恒写 v3** | `H3_BinaryLatent` 对任何输入抛异常 |
| 2 | `_write_h3_latent` 写 v2，但 reader **无条件先吃 28 字节**（`h3.c:2578`） | 4 字节错位 → `truncated audio latent`；`H3_BinaryLatentDecode` 从未工作过 |
| 3 | 放大器只返回 `{"samples":...}`，**丢掉 `h3_audio`** | 会掉进"纯噪声音轨"坑 |

- 新增 `H3_BinaryLatentRefine`（带 `prompt`；引擎要 `-p` 才走去噪，否则退化成解码）
- 新增 `gen_h3_two_stage_workflow.py` → 产出两份 UI 工作流：
  - `h3_two_stage_pipeline.json`（①→②→Refine）
  - `h3_two_stage_upscale_decode.json`（①→②→Decode）
- **端到端实跑成功**（ComfyUI HTTP API，`POST /prompt` + 轮询 `/history`）：
  两条链路各产出 mp4；`h3_audio` 穿过放大器的相关系数 **+1.0000**

### Phase 59: 两阶段收益实测 — **变体 B 全尺寸跑通并被接受**

| 配置 | tokens | 结果 |
|---|---|---|
| render 640×352, 121 帧 | 32,560 | ✅ 跑通（DiT 721s + VAE 175s ≈ 15 min） |
| render 1280×704, 22 帧 | 24,640 | ✅ 跑通（96.6s/步） |
| **render 1280×704, 121 帧** | **130,240** | ❌ **触发系统重启** |

**推论：直接全分辨率生成是同一个 token 数，同样跑不动。**
⇒ "两阶段 vs 直接"在 1280×704/5s 上**不成立**，两条路都越界。

**替代路线（变体 B）已跑通并被用户接受**（2026-09-16 21:15 实跑）：

```
低分生成(① steps=4, render 640×352) → 放大器(→1280×704) → H3_BinaryLatentDecode
→ output/video/h3_upscale_decode_00001_.mp4
   1280×704 / 124 帧 / 含音轨 / 4.1 MB
   帧内对比度 std=74.9，相邻帧差=4.62   → 真实画面 + 有运动
   音轨 rms=0.00089（噪声失败案例是 0.486，低 546×）→ 真实音频
   音轨时长 5.18s ≈ 124 帧@24fps = 5.17s         → 音视频长度对齐
```

**用户结论：效果可以接受** ⇒ **B4（把放大器搬进 h3c）无限期搁置**——
它唯一剩下的收益是"去掉 ComfyUI 依赖"，而现有链路已经能出目标尺寸且画质可接受。

附带说明：音轨 rms 0.00089 比 640×384 那次（0.00360）安静约 4 倍，
判断为内容相关（雪林环境音），非缺陷；如需确认可目视/试听。

### Phase 60: 新权重（pruned-int8 DiT / int8 VAE / 8step 蒸馏）实测 — complete

- 引擎**按目录扫描**发现新权重（与文件名无关），`--info` 确认
- `steps=2 / 1s / 256×256 / --ssd-streaming` **能成功出片**（39 帧、含音轨、120s、无警告）
- 但 **2 步对 8step 蒸馏模型未收敛**：2步 vs 8步 SSIM 0.581 ≈ 换种子的无关生成 0.551
- **横向标尺**：4步 vs 8步 = 0.680，2步 vs 4步 = 0.603 ⇒ 4 步在收敛但仍未到位
- 耗时公式（`H3_DIT_RESIDENT_BLOCKS=1`）：`总墙钟 ≈ 22.5s + 17.1s × steps`

## Decisions Made（本任务）
| Decision | Rationale |
|---|---|
| 复用 `--latent-in` 而非新增路径参数 | `--latent-in` + `--refine-sigma` = img2img；只给 `--latent-in` = 原解码语义，零破坏 |
| refine 用**独立构造**的 schedule，而非截取 | 两边 sigma 调度不同源；独立构造才能精确指定起始 sigma |
| 两模态各自反解 base | 同一 start_sigma 下 video/audio 都能从该 sigma 起步（各自 shift 不同） |
| 先做 A 再做 B | A 能立刻用 ComfyUI 放大跑通管线验证收益，避免在未实测前投入 B 的 ~600 行 |
| **B4 无限期搁置** | 放大器 CPU 前向只要 2.4s（1280×704 时 60.9s）——**它不是瓶颈**；而变体 B 已跑通 1280×704/124 帧并经用户确认"效果可以接受"。B4 唯一剩下的收益是"去掉 ComfyUI 依赖"，不值得 350–400 行 C |
| **保留 steps=4** | 用户实跑后确认可接受（变体 B 无精修，① 的步数直接决定最终画质；8step 蒸馏权重下 4 步属欠收敛，但画质够用即止）|
| **所有性能测量必须加 `H3_DIT_RESIDENT_BLOCKS=1`** | 自适应驻留随系统内存漂移；不钉住则同一参数能差 4.6×（见 Errors） |

## Errors Encountered（本任务）
| Error | Attempt | Resolution |
|---|---|---|
| `truncated audio latent` | 1 | 根因是 `_write_h3_latent` 写 v2 而 reader 恒吃 28 字节 → 4 字节错位。改为写 v3（dtype=0） |
| `--refine-sigma needs a prompt (-p)` | 1 | refine 节点缺 `prompt` 输入；补上并在空值时硬报错 |
| 放大器丢 `h3_audio` | 1 | 改为保留 LATENT 字典附加键 |
| **`h3: missing required model file`** | 1 | 用户工作流里 `binary` 指向 `/Volumes/data/git/c/h3.c/h3`，该目录**不存在**（仓库已改名 `h3c`）。新工作流用节点默认值 |
| **系统重启（OOM）** | 1 | 1280×704 + 121 帧 = 130,240 tokens。改用变体 B 或降时长 |
| **steps=4 报出 422s** | 2 | **测量错误**，非真实结果：未钉住驻留块数，引擎自适应选了 11 块。钉住 1 块后重测为 **91s**（差 4.6×） |
| `exec_command` 工具不存在 | 1 | 工具名应为 `execute_command` |
| zsh 不分词 `$VAR` | 1 | 多 flag 不要用变量承载（zsh 无引号变量不分词） |
| `pkill -f "ComfyUI/main.py"` 无效 | 1 | 实际命令行是 `./.venv/bin/python3 main.py`；改用 `lsof -ti:8188` 精确定位 |

## 本轮改动文件
| 文件 | 改动 |
|---|---|
| `h3.h` / `main.c` / `h3_host.h` / `h3_host.c` / `h3.c` | `refine_sigma`（阶段 A）+ `H3_LATENT_VERSION_F16` 读写 |
| `h3_gpu.h/.m` / `h3_shaders.metal` | `conv3d_pad_f32`、`group_norm_f32`、`channel_scale_shift_f32`（+ kernel 注册与参数镜像） |
| `h3_host.c/.h` | `h3_resize_bilinear_f32` |
| `tests/test_h3.c` | schedule 测试 + bilinear 测试 → **1829 checks** |
| `comfyui_nodes/h3_binary.py` | `_read_h3_latent`(v2+v3/f32+f16)、`_write_h3_latent`(v3)、**新增 `H3_BinaryLatentRefine`** |
| `comfyui_nodes/__init__.py` | 注册 refine 节点 |
| `h3_latent_bridge.py`（仓库根） | 终端桥接工具（H3LT bundle ↔ ComfyUI `.latent`） |
| `gen_h3_two_stage_workflow.py` | 生成两份 UI 工作流 |
| `README.md` | §10「Two-stage latent pipeline」 |
| `ComfyUI/custom_nodes/Comfyui_Minimax_h3_latent_Upscaler/.../minimax_h3_latent_upscaler_3d.py` | 保留 `h3_audio`（第三方节点，1 行补丁） |

## Next
1. **变体 B 已验证可用** —— 直接用于生产；`steps=4` 经用户确认可接受
2. **B4 无限期搁置**（见上）
3. 若日后仍要做 B4，先补 `B4d`（对 PyTorch 逐点对拍），否则接线无法证明正确
4. **所有后续性能测量一律加 `H3_DIT_RESIDENT_BLOCKS=1`**
5. 新权重：本任务用 4 步已够；追求上限可试 8 步
6. 变体 A（含精修）仍受 token 上限约束，大尺寸请用变体 B

## 用法（当前）
```bash
# 变体 A（有精修，受 token 上限约束）—— 直接在 ComfyUI 打开工作流
#   h3_two_stage_pipeline.json

# 变体 B（无精修，能出 1280x704 / 5s）—— h3_two_stage_upscale_decode.json
#   ① 输出 1280x704 + extra_args "--render-width 640 --render-height 352" + seconds 5
#   ② target dimensions 1280x704
#   ③ H3_BinaryLatentDecode(1280, 704)

# 终端路线（等价，无需 ComfyUI）
./h3 -d MODEL -p "..." -o s1.mp4 --width 1280 --height 704 \
     --render-width 640 --render-height 352 --frames 121 --steps 5 \
     --latent-out s1.h3lt
python3 h3_latent_bridge.py to-comfy s1.h3lt          # → ComfyUI input/
#   （ComfyUI 内放大后 SaveLatent）
python3 h3_latent_bridge.py to-bundle <saved.latent> -o up.h3lt --audio-from s1.h3lt
./h3 -d MODEL -p "..." -o final.mp4 --width 1280 --height 704 --frames 121 \
     --steps 4 --latent-in up.h3lt --refine-sigma 0.4

# 性能测量必须钉住驻留：
export H3_DIT_RESIDENT_BLOCKS=1
```

# ===== Task Plan（当前任务）: ANE int8 投影移植（h3.c-ane fork → h3c）=====

## Goal
把 fork（/Volumes/data/git/c/h3.c-ane）里已验证的 ANE int8 线性层能力移植进 h3c：
DiT 4 投影（qkv/out/fc1/fc2）走 ANE int8 图 + LoRA 低秩旁路，Metal 保留对照路径。
（背景与硬约束见 ANE_PORT_SUMMARY.md；其第 6 节的两个决策已闭环，见下。）

## 决策闭环状态（2026-09-19 凌晨/上午）
- [x] **决策 A（int8 格式是否满足 ANE）→ 满足**：`dbg_ane_int8_format.py` 实测
      `fastvideo_fasth3_8step_v2_pruned_int8_convrot.safetensors` block 0 四投影：
      per-row F32 scale（行数==输出行数）、zero  absent（对称）、bias absent、
      scale fp16 往返最大相对误差 4.9e-4、无 flush-to-zero/subnormal →
      `constexpr_affine_dequantize(axis=0)` 契约直接成立，无需重排/重中心化。
- [x] **决策 前置 2（LoRA 可否折入 int8）→ 不可**：`dbg_lora_fuse_convrot.py` 判定
      delta≈0.2%||W||，恰在 per-row int8 噪声量级 → 折叠后 effective delta cos≈0.02-0.2（埋没）。
      **必须走低秩旁路**。fork 已在图内实现：`h3_ane_linear_create_int8_bands`
      （A_rot 复用主 conv 已算出的旋转激活 rᵢ，不多绑输入、不多一次 eval；qkv 3 band）。
- [x] fork 侧扩展：`h3_ane_linear.{h,m}` 今日 08:16 加 int8 payload + bands 接口；
      `tests/test_ane_int8.c` 08:55 建成 `h3_ane_int8_test`。

## Phase A（当前卡点）：真实尺寸 ANE 编译失败诊断
- 08:37 `ane_base_only.log`：**real-qkv K=5376 N=21504 kc=1024 rows=32 PASS**
  （cos=0.9999998, weights=126.5 MiB, 0 failures）
- 08:57 `ane_final.log`：**real-qkv 与 real-lora-qkv 均 FAIL**
  （`_ANECompiler : ANECCompile() FAILED`），10 个合成 gate 全 PASS
- 期间仅测试文件重编（库 .o 停在 08:18 未变）→ 疑因：aned 服务/缓存状态、内存压力、
  或 08:48-08:55 测试改动（如先建 lora 图改变顺序）
- 08:59 两次重跑日志为空（会话中断）
- [ ] 复现中：`/tmp/ane_repro_0919.log`（后台跑当前 binary，同参数）

## Phase B（待 A 绿后）：移植入 h3c
- 搬 `h3_ane_bridge.{h,m}` + `h3_ane_linear.{h,m}` + `h3_ane_int8_test`，knob 门控（如 H3_ANE_LINEARS=1）
- 单 block 输出 vs Metal 路径余弦 >0.99 验证；保留 Metal 回退
- 再议 `h3_ane_block`（整 block 一张图，收益最大但每形状 ~19GB 编译产物）

## 边界与红线（沿用 ANE_PORT_SUMMARY.md 第 7 节）
- 未确认范围前不改 h3c 的任何 .c/.m/.h（fork 里的实验性改动不受此限）
- 所有 ANE 路径保留 Metal 对照 + 余弦校验 knob

## Phase 状态更新（2026-09-19 09:50）

- **Phase A（fork 侧真实 gate 打通）：COMPLETE**
  `real-qkv` 与 `real-lora`（真实 Turbo LoRA，3 band rank 64，N=21504/K=5376）
  对 double 参考 cos=0.9999998、rel_l2=6.7e-4、nonfinite=0，`0 failure(s)`，exit 0。
  卡点不是 ANE/MIL，也不是纯磁盘：`tests/test_ane_int8.c:886/926` 两处 malloc
  少乘元素大小 → 堆溢出 → SIGBUS。已在 fork 内修正（只改测试文件，未动库）。
- 顺手清掉我们自己产生的 ANE 编译残留（`T/h3-ane-cache` 400M + 孤立 staging 394M）。
  系统卷仍只剩 ~5.0 Gi（98%）。
- **Phase B（搬进 h3c）：BLOCKED，等用户两点确认**
  1. 移植范围（ANE_PORT_SUMMARY §6 决策 B）：只搬「4 投影 + 整 block」，
     还是同时规划 video/vision/audio VAE 那批（§2 的 4–8 号）？
  2. 是否授权为编译腾空间清理 `~/Library/Caches`(3.9G)/`~/.cache`(2.5G) 这类
     用户可再生目录；不授权则我在 Phase B 期间只能靠 ~5 Gi 硬跑，
     新鲜大编译随时可能再挂。

## 决策更新（2026-09-19 10:05）

- **决策 B 已定**：移植范围**不只** 4 投影 + 整 block，要把 **video / vision / audio VAE**
  一并纳入规划（ANE_PORT_SUMMARY §2 的 4/5/6 号）。⇒ 阶段划分改成 A→B→C→D：
  B=投影+block，C=形状档位收敛（VAE tile / seq 分档），D=三个 VAE 逐个 fp16 化并过精度关。
- **磁盘结论（F9）**：ANE 编译 staging **无法**搬到外置 SSD（编译器沙箱只认
  `DARWIN_USER_TEMP_DIR`），所以 Phase B 开始前系统卷必须先腾出 GiB 级空间。
  等用户批准清理 `~/Library/Caches`(3.9G) / `~/.cache`(2.5G) 或指定别的清理清单。

- **磁盘前置（F10 修正）**：ANE 编译只需 0.1–0.6 s/图，关缓存后进程退出零残留
  ⇒ 不再需要清理用户目录，内部卷 avail 已回到 14 Gi。策略改为
  **默认关跨进程缓存 + 进程内复用**，跨进程缓存要做得带 LRU 上限（当前 bridge 无上限无淘汰）。

## Phase B2a 完成（2026-09-19 10:50）

- h3c 侧新增 `h3_convrot.{h,c}`（fork 逐字移植）、`h3_weight_load_int8_raw()`
  （按 h3c 的 `pread_bytes` / `%s_scale` / 仅 F32 scale 约定重写）+ `tests/test_int8_raw.c`。
- 「表 derotate」与 h3c butterfly un-rotate 的等价性已变成测试断言：
  `h3_convrot_test` 要求逐元素 0，`h3_int8_raw_test` 在真实 checkpoint 上 rel_rms ≤ 4e-8。
- 验证：`make -j8 all` 无新 warning；`./h3_convrot_test` PASS；
  `./h3_int8_raw_test ~/h3_sys/MiniMax-H3-Convrot/FL2VA/transformer` → `0 failure(s)`。
- 下一步 **B2b**：移植 `h3_ane_bridge.{h,m}` + `h3_ane_linear.{h,m}`，
  `FRAMEWORKS` 补 `-framework IOSurface`，测试搬 `tests/test_ane_int8.c`
  （**必须带上 fork 里那两处漏 `sizeof` 的修复和 compile/cache 打点**），
  验收 = h3c 内 `real-qkv` + `real-lora` cos ≥ 0.9999。

## Phase B2b 完成（2026-09-19 11:40）

- h3c 内 ANE 通路**自洽可用**：`h3_ane_bridge.{h,m}` + `h3_ane_linear.{h,m}` 从 fork 逐字移植，
  构建接线完成（`LIB_M` 收两个 .m、`FRAMEWORKS` 补 `-framework IOSurface`、
  新增 target `h3_ane_int8_test` / `h3_ane_staging_test`）。
- 补齐了 fork 没有的一块：`h3_gpu.{h,m}` + `h3_shaders.metal` 的 **Metal pack/unpack 暂存**
  （`h3_ane_pack_bf16` / `h3_ane_unpack_bf16`），并新增 `tests/test_ane_staging.c`
  做位级往返校验（fork 里这条路径一直无测试覆盖）。
- **验收达成**（h3c 内，真实 ConvRot checkpoint + 真实 Turbo LoRA，`H3_ANE_CACHE=0`）：
  12 关全 PASS，`real-qkv cos=0.9999998 / real-lora cos=0.9999998` ≥ 0.9999，
  `0 failure(s)`；compile 分别 0.43s / 0.50s，`T/h3-ane-cache` 退出后 0 B。
- 回归：`make -j8 all` 无新 warning，`./h3_tests` → `ok: 1829 checks`，
  `./h3_convrot_test` PASS，`./h3_ane_staging_test` → `0 failure(s)`。
- 移植中撞到的两处 h3c 特有差异（`h3_gpu_require_bf16` 重复定义、
  pipeline 名字硬编码名单）与 `cache_gates()` 静默强制开缓存的坑，全部记在 **F12**。
- **Phase B2 至此关闭**（B2a host 依赖 + B2b bridge/linear/测试）。

## 下一步（B3/B4）待确认，不擅自开工

接线 `h3_dit.c` 的 `blocks.N.attn.qkv/out, mlp.fc1/fc2` 到 ANE（`H3_ANE_LINEARS` 开关 +
Metal 对照 + cos 校验）之前，先解决 h3c 侧两个既有事实与移植结论的冲突：

1. **融合 kernel**：qkv 与 MLP 在 h3c 分别是
   `h3_gpu_grouped_qkv_linear_rope_int8`（`h3_dit.c:4007`）和
   `h3_gpu_mlp_int8_bf16`（`h3_dit.c:4109`）；走 ANE 意味着把 RoPE / SwiGLU 拆回两步，
   需评估融合收益与 ANE 收益谁大。
2. **LoRA 合并**：h3c 在加载时已把 LoRA 并入权重（`h3_lora.c:170 h3_lora_apply`），
   而 F2 的结论是 int8 下 LoRA **必须**以高精度低秩旁路存在（delta ≈0.2%‖W‖ ≈ int8 噪声底）。
   ⇒ 要么 ANE 路径改走「未合并权重 + 旁路」，要么明确接受合并带来的精度损失并量化它。

## Phase B3 前置核对完成（2026-09-19 11:25）→ 原「两个冲突」只剩一个（F13）

- **LoRA 不是冲突**：h3c 是「合并进 BF16 后**重新量化**驻留 int8」（`h3_dit.c:2084`），
  F2 否决的是「沿用原 scale 就地折进 int8」⇒ ANE 直接吃 h3c 驻留 int8 即可，不需要旁路。
- **权重已在权重侧 un-rotate + qkv 重排** ⇒ ANE 图用 **gs=0 纯 GEMM 档**，
  fork 图里那段 grouped Hadamard 在 h3c 接线时应当省掉。
- 剩下的真代价：qkv/MLP 是融合 kernel，走 ANE 要把 RoPE/SwiGLU 拆回独立 kernel。
- 接线约束：`H3_ANE_LINEARS` 与 int8/int6 驻留开关**互锁**（BF16 直载档自动回退 Metal）。

## Phase B3' 完成（2026-09-19 11:40）：整 block 单图移植进 h3c 并过真实 gate

- 移植 `h3_ane_block.{h,m}` + `tests/test_ane_full_block.c`；**一次编译通过**，
  只改了 4 处 `(double)` 显式转换（h3c 的 `-Wenum-float-conversion`）+ 2 处注释（binary 名、
  rotation 腿需要 `H3_ANE_CACHE=1`）。`LIB_M` 加 `h3_ane_block.m`，新 target `h3_ane_full_block_test`。
- 真实 blocks.0 gate：`cos=0.999988 rel_l2=4.898e-03 nonfinite=0 PASS`、
  `timing S=1904: 334.3 ms/block (compile 3.21s)`、`rotation: unload 5ms + reload 29ms 3 圈逐字节相同`。
- **磁盘硬墙量化**（F14）：编译产物 **381 MiB / block-形状**（≈1× int8 blob）
  ⇒ 50 block × 1 形状 ≈ 18.6 GiB，算上 aned 副本 ≈ 37 GiB，本机 avail 只有 12 GiB。
  关缓存则每换形状重付 ≈ 170~210 s 冷编译。
- **B4 方向据此调整**：h3c 低分辨率档去噪已贴 SSD 地板（每 block read 173.5 ms / 2.22 GB/s），
  ANE 不减字节只加编译 ⇒ 先移植 fork 的 `tests/test_ane_block.c` 做
  **ANE vs 纯 Metal vs 融合 Metal 的 rows 扫描曲线**（B4-pre），
  拿到「行数多大 ANE 才开始赢」之后才谈 `h3_dit.c` 接线。
- 下一步（B4-pre）：移植 `tests/test_ane_block.c`（合成权重、argv=ROWS、gate cos≥0.999），
  在 h3c 里补上「权重常驻、rows≈1904」的 Metal 单 block 计时，与 334 ms 对齐比较。

## Phase B4-pre 完成（2026-09-19 12:05）：拆投影上 ANE 判死，DiT 侧接线不做（F15）

`tests/test_ane_block.c` 零改动移植（新 target `h3_ane_block_test`），干净环境 rows 扫描：

| rows | Metal 融合 best | ANE 拆投影 best | ratio | ANE mean |
|---|---|---|---|---|
| 384 | 101.5 ms | 68.3 ms | 1.49× | 90.0 ms |
| 1536 | 436.7 ms | 418.8 ms | 1.04× | **549.2 ms（慢于 Metal）** |
| 3072 | 903.7 ms | 842.7 ms | 1.07× | **943.0 ms（慢于 Metal）** |

- 结论 1：**B3（`H3_ANE_LINEARS` 接 4 条投影）取消**，不再改 `h3_dit.c`；
  收益 ≤1.5× 且只在 rows=384 出现，mean 反超为慢，还要付 4×50 个图 × 0.16~0.87 s 冷编译。
- 结论 2：DiT 侧唯一有肉的是「整 block 一张图」（≈1.6×），但缓存 18.6 GiB/形状 vs 本机 12 GiB
  ⇒ **在 h3c 这台机器上 DiT-ANE 整体不成立**，模块与 gate 留在树里备用即可。
- 结论 3（建议，待用户拍板）：ANE 的力气改投 **Phase D 的三个 VAE**：
  tile 形状固定、payload 小、每形状编译一次 ⇒ 才是 ANE 的适用面。
- 顺带修掉两个会骗人的地方：`h3_ane_projection_timings` 的 `pack_seconds`
  含「排空前面 Metal 队列」的口径（已在头文件注释写明，真实 pack 只有 1~2 ms/投影）；
  计时 gate 必须独占机器（并发时同一参数虚高 2×：855 vs 436.7 ms）。

### 本轮验证与清理
`make -j8 all` + 5 个 ANE/回归 gate 全绿；`T/h3-ane-cache` 只删本会话写的 762 MiB，卷 avail 回到 13 GiB。

---

# Phase D（2026-09-19 12:20 开）：范围按「摊销判据」重排 —— 只做 video VAE

## D 的准入判据（先立规矩，避免第三次返工）

ANE 的权重是 **constexpr 烘进编译产物**的 ⇒ 一个模块值不值得上 ANE，取决于
**「同一份权重 + 同一形状，会被复用多少次」**，而不是单算子快不快：

| 模块 | 同一权重复用次数 | 结论 |
|---|---|---|
| **video VAE** | 36 block × **24 pass**（864×480×56 实测，findings:1576） | ✔ 唯一能摊销的，见下 |
| vision encoder | 27 层 × **1 pass**（每次生成只跑一遍），rows 还随分辨率变 | ✘ 编译无法摊销 |
| audio VAE | 0.56 GiB conv，**1 pass** 且形状随 clip 时长变；`conv_transpose1d` 在 ANE 上**未验证**；fp32→fp16 音频质量风险（ANE_PORT_SUMMARY:51） | ✘ 三重不利 |

⇒ **vision / audio 从 Phase D 出局**（原 ANE_PORT_SUMMARY §2 的 5/6 号作废，这条要写回总结文档，
免得以后又重新评估一遍）。

## 为什么 video VAE 反而worth做：瓶颈是「权重搬运」，而 ANE 恰好把它消掉

- video VAE 根本不是 conv，是 **36 层纯 ViT**（HIDDEN 2048 / HEADS 32 / FFN 8192，`h3_video_vae.c:19-27`），
  且**全程 fp32**（`h3_gpu_linear_f32` + `h3_gpu_sdpa_f32`，`h3_video_vae.c:445-478`）。
- 盘上是 **F16**（4.85 GiB / 560 张量），运行时展成 **9.67 GB fp32**；流式解码每 pass 全量重读，
  24 pass = **232 GB** 读盘，实测 242.8 s 里 **~146 s 是 CPU 侧权重搬运**（findings:1576-1585）。
- ANE 图把权重烘进编译产物 ⇒ 每个 block 的权重**只在编译时读一次**，24 个 pass 复用同一张已加载的图。
  ⇒ 省的是**读盘 + fp32 展宽 + 每 pass 的 GPU 上传**这一整块，而不只是 GEMM 加速。
- 磁盘账（按 F14 实测「产物 ≈ 1× 权重字节」）：fp16 权重每 block ≈ **134 MiB**
  ⇒ 36 block × 1 形状 ≈ **4.8 GiB**（+ aned 副本）。比 DiT 的 18.6 GiB/形状小 4 倍，
  但仍然吃满本机 13 GiB ⇒ **必须带字节上限的 LRU + 默认关跨进程缓存**。
- 顺带：权重在盘上已是 F16 ⇒ 转 fp16 对**权重零损失**，ANE 引入的唯一精度损失是**激活 fp32→fp16**。

## D0（本阶段第一个动作，不碰 `h3_video_vae.c`）：真实 VAE 投影的 ANE-vs-Metal 关

新建 `tests/test_ane_vae.c`：取真实 `decoder.transformer_blocks.0` 的四条投影
（qkv 6144×2048 / out 2048×2048 / w1 16384×2048 / w2 2048×8192），rows=**1797**
（256px tile 的真实序列长，`TILE_PIXELS=256` ⇒ `7·lh·lw+5`），权重走 `H3_ANE_W_F16`，
对照 h3c 自己的 `h3_gpu_linear_f32`。三个输出量：
1. **速度比**（ANE vs Metal fp32，同一 shape 多次求值取 best/mean）；
2. **激活 fp16 化的误差**（cos / rel_l2 / max_abs，另加**峰值幅度与 fp16 溢出计数**——
   fork 在 DiT 上撞到过 fc2 峰值 4.3e4 逼近 fp16 上限）；
3. **每图编译秒数 + blob 字节**（外推到 36 block × 24 pass 的净账）。
- 放行判据：**速度比 ≥ 2× 且 rel_l2 ≤ 1e-3 且 fp16 溢出 = 0** 才进 D1
  （VAE 是最终像素的直接生产者，容差要比 DiT 的 3e-2 严一个量级）。

## D1..D3（D0 放行后才展开）
- D1：bias 语义（`h3_gpu_linear_f32` 带 bias，现图不支持）+ f32↔ANE plane 的 pack/unpack。
- D2：36 block 的常驻轮换（F14 实测 unload 5 ms / reload 29 ms 可用）+ **带字节上限的缓存 LRU**。
- D3：端到端画质回归（既有逐字节/SSIM 基线）+ `H3_ANE_VAE` knob（默认关，保留 Metal 对照）。

---

# Phase D0 完成（2026-09-19 12:45）：放行，进 D1

`tests/test_ane_vae.c` + Makefile 目标 `h3_ane_vae_test`（该 .o 单独特例
`-DACCELERATE_NEW_LAPACK`，为了用不弃用的 `cblas_sgemm` 原型）。编译零告警，
未碰 `h3_video_vae.c`。完整数据在 **F16**。

结论：**三条判据全过** —— ratio 3.24×（rows=1797）/ 2.89×（rows=512）、
rel_l2 ≤ 7.9e-4、fp16 溢出 0。一处前提被推翻：**盘上权重是 F32 不是 F16**，
所以 fp16 舍入的账算在误差里（仍然过）。

同时修正了磁盘账：fp16 图的编译产物 = **1× 权重字节**（每 block 128 MiB、
36 block 4.5 GiB），且**与 rows 无关** ⇒ 每个新 rows 形状再复制一份 4.5 GiB。
⇒ D2 的范围据此调整为**「rows 分桶 + 一份跨分辨率复用的缓存」**，见下。

## D1（现在展开）
1. `h3_shaders.metal`：`h3_pack_ane_planes_f32` / `h3_unpack_ane_output_f32`
   （row-major f32 ↔ channel-major `[kc][plane_rows]` f32），bias 在 unpack 侧加
   （bias 是 `[output_dim]` f32，ANE 图不管它）。
2. `h3_gpu.m` 的 pipeline 名单里**显式登记**这两个 kernel（tensorOps 关闭时也要登记）。
3. 在同一个 gate 里加"GPU pack + ANE + bias unpack"的**端到端单投影计时**，
   因为 F16 已经证明 CPU 填平面在 w2 上（41 ms）能吃掉全部收益 ——
   **D1 的放行判据用端到端比，不再用裸 GEMM 比：端到端 ≥ 2× 才继续**。
4. 保持 `H3_ANE_VAE` 语义预设：Metal 对照路径不删。

---

# Phase D1 完成（2026-09-19 13:20）：端到端闸放行（1797 行 2.26~2.43×），512 行不过闸

代码见 F17。要点：bias 放 unpack 不进图；`h3_ane_pack_f32`/`h3_ane_unpack_f32` +
`h3_ane_projection_create_f16`/`apply_f32`；block 端到端 **2.26×**（冷）/**2.43×**（热），
36 block 一条 pass **2.97 s → 1.31 s**；误差与裸图同级。回归全绿。

**D1 修正了 D0 的判据口径**：裸图 3.24× 不能用来决策，staging 一张图要吃 1.0~3.7 ms，
四条合计 ≈9 ms（+32%）。今后所有放行/回退判断都按端到端算。

## D2 待拍板的三个决定（这一步要动 `h3_video_vae.c`，先对齐再改）

1. **rows 闸**：512 行端到端只有 1.88×（不过 2×）。建议
   `H3_ANE_VAE_MIN_ROWS` 默认 **1024**：小 tile 继续走 Metal fp32，大 tile 走 ANE。
   备选：整 block 一张图（把三次跨界摊成一次），但那是 B3' 级别的工程量，且 fp16 图
   没有 int8 图那种反量化膨胀，产物仍是 4.5 GiB/形状。
2. **rows 分桶**：把 `sequence` 向上取整到固定桶（1797→2048，多付 ≤14% 算力），
   让**所有分辨率共用同一份 4.5 GiB 编译产物**；不分桶则每遇到一个新分辨率就再付
   4.5 GiB + 16 s 冷编译。建议做，桶粒度先取 256 的倍数。
3. **缓存字节上限**：`T/h3-ane-cache` 现在**没有 LRU 也没有上限**（Phase B 就欠着）。
   本机数据卷只剩 13 GiB，一个形状 4.5 GiB ⇒ 建议 **上限 6 GiB + LRU 淘汰**，
   超了宁可重编译（16 s/36 block，比 SIGBUS 便宜）。

D2 的实施顺序（拍板后）：① `h3_weights` 加 F32→fp16 的权重装载口（现在只有
`load_bf16`/`load_f32`/`load_i8`）；② `h3_video_vae.c` 里按 block 建 36 张图 +
unload/reload 轮换（B3' 实测 5 ms/29 ms）；③ 缓存 LRU；④ 与 Metal 路径逐 block 对拍。

---

# D2 拍板（2026-09-19 13:35）：阈值以下留 Metal + rows 分桶 + 缓存上限 6 GiB

用户确认：
1. **小 tile 留 Metal**：新增 `H3_ANE_VAE_MIN_ROWS`，默认 **1024**，rows 不到就走现有
   fp32 路径（Metal 对照路径天然保留，`H3_ANE_VAE=0` 整体关闭）。
2. **rows 分桶 + LRU 上限 6 GiB**：`sequence` 向上取整到固定桶（先按 256 的倍数），
   让所有分辨率复用同一份编译产物；`T/h3-ane-cache` 加字节上限与 LRU 淘汰。

## D2a（先做，因为它决定架构）：常驻容量探针

架构岔路口：36 block × 4 投影 = **144 张已加载的图**，常量合计 4.5 GiB。
- 若一次全常驻 → 轮换成本为 0，每 pass 纯赚 1.66 s；
- 若常驻有上限（DiT 侧的经验是"编译后 block ~800 MB 必须轮换"）→ 每 pass 要付
  unload/reload，F14 在 381 MiB 图上实测 5 ms/29 ms，144 张 × 部分 reload 的账
  要先算清楚再决定接线方式。

所以 D2a 只测三件事，不动 `h3_video_vae.c`：
1. **同时 load 得下多少张图**（逐张 load，记录第一次失败的位置与实际驻留字节）；
2. **128 MiB 级 fp16 图的 unload/reload 各多少毫秒**（DiT 的数不能直接套用）；
3. **每 pass 走一遍 36 block 的净时间**：全常驻 vs 轮换 K 张 两条曲线。

---

# D2a 完成（2026-09-19 14:20）：常驻上限是 126 个句柄，架构必须改形（数据在 F18）

探针 `tests/test_ane_vae_residency.c` 量完三件事，全部写进 F18。结论按影响排序：

1. **句柄数（≈126）是硬墙，与内存/磁盘无关**：`capacity 64 36` 和
   `window 64 36 8`（每张建好立刻 unload）都在**第 127 张**挂
   （`Program load failure (0x50004)`，当时 footprint 412 MiB、可用 6.2 GiB）。
   ⇒ 36 block × 4 投影 = 144 张连"建出来放着"都不可能，**park 不腾名额**。
2. **轮换单价便宜但吃不起**：unload 0.47~1.14 ms、reload 3.40~5.15 ms/张（128 MiB 级
   fp16 图）。rows=2048 / 32 张实测：全常驻 333.5 ms、window=8 557.5 ms（×1.67）、
   每张每 pass 重连 616.4 ms（×1.85）⇒ D1 的 2.26× 会掉到 ≈1.2×。
   **接线的默认必须是"常驻、不轮换"，轮换只当兜底。**
3. **内存大头是平面不是权重**：rows=2048 时每 block 平面 320 MiB（w1 输出平面单独
   128 MiB），36 block 自持平面 = 11.5 GiB ⇒ 判死。36 block 形状一致，
   **池化后 320 MiB**。⇒ D2 的真正前置工程是"让 `h3_ane_linear` 接受外部平面"。
4. **缓存会自毁条目**：restore 是目录级 `copyItemAtPath`；系统卷紧到 4.4 GiB 时出现
   "cache restored 但 `loadWithQoS` 失败" → 回退重编译 → `bridge_cache_store` 先
   `removeItemAtPath(entry)` ⇒ 一次瞬时失败就把条目抹掉。**LRU 之前必须先修这条**
   （restore 改逐文件硬链接 + cached load 失败不许删条目）。

因此 D2 的架构只有两条路（F18 §6）：
- **①（推荐先做）**：ANE block 数封顶 ≤31（124 张图，全常驻不轮换）+ 平面池化，
  其余 block 留 fp32 Metal。外推一条 pass ≈ **1.8 s vs 全 Metal ≈3.4 s（1.9×）**。
- **②**：把 w1+w2 融成一张 FFN 图 ⇒ 3 张/block = 108 张，全部进 ANE；但正好顶满
  8 输入绑定上限、无余量，且要新 MIL + 重做精度关。

**待用户重拍**：原"LRU 上限 6 GiB"是在我误报"系统卷只剩 13 GiB"的前提下定的。
实测：一个 rows 形状的产物就要 **4.5~4.6 GiB**（另加每张图 live 的 staging 副本），
而本机数据卷空闲降到 ~5 GiB 时 ANE load 就开始失败（本轮清掉自己的 5.7 GiB 缓存后
回到 12 GiB）。⇒ 6 GiB 上限约等于"只装得下一个形状，且和 staging 抢空间"。

**D2b 计划（拍板后）**：① 缓存两处修正（硬链接 restore / 失败不删条目）；
② rows 分桶（256 的倍数）；③ LRU 上限（按新磁盘账重定，可能 10 GiB 或按形状数封顶）；
④ 平面池化（`h3_ane_linear` 接外部 IOSurface）。之后 D2c（`h3_weights` 加 F32→fp16
装载口）、D2d（`h3_video_vae.c` 接线，默认关，逐 block 与 Metal 对拍）。

---

# D2b + 平面池化完成（2026-09-19 16:05）：路线①坐实，收益修正为 1.53×

代码与数据见 F19/F20/F21。用户重拍：走路线①（block 封顶 31）+ 先修缓存再定 5 GiB 上限。

已完成：
1. **缓存三修**：restore 逐文件硬链接（不再整份复制）、store 走 `.tmp` + `move`
   （瞬时失败不再抹条目）、LRU = `H3_ANE_CACHE_MAX_MIB`（5120）
   **+ 磁盘空闲下限 `H3_ANE_CACHE_MIN_FREE_MIB`（6144）**，冷编译前先腾地方。
2. **rows 分桶**：`h3_ane_rows_bucket()`（`H3_ANE_ROW_BUCKET`，默认 256），
   实测 1797 与 2048 共用同一批 identifier ⇒ 跨分辨率复用一份产物。
3. **平面池化**：`h3_ane_planes_share/bytes/clear`，同形状图共用一套平面；
   124 张图常驻 footprint 1584 MiB、一条 pass 1.78 s。

**收益口径修正**：全 Metal 36 block ≈ 3.46 s；路线① ≈ **2.26 s（1.53×）**
（比 F18 的 1.9× 低，因为 ANE 每张图的成本随常驻图数上涨：43→57 ms/block）。

**新发现的约束（影响 D2d 的默认值）**：31 block 一套产物 3.9 GiB，
与"系统卷留 6 GiB"互斥 ⇒ **不能把性能押在缓存命中上**。
D2d 因此按"冷启动也要能看"设计：常驻不轮换是默认，但首轮编译 71 s 是明码标价的成本，
`H3_ANE_VAE_MAX_BLOCKS` 要能压低（少几张图＝少编译、少句柄、也少产物）。

D2c/D2d 待做（不变）：`h3_weights` 的 F32→fp16 读取口；`h3_video_vae.c` 接线
（`H3_ANE_VAE` 默认 off、`H3_ANE_VAE_MIN_ROWS=1024`、`H3_ANE_VAE_MAX_BLOCKS=31`、
池化平面、逐 block 与 Metal 对拍）。

# D2c 完成（2026-09-19 16:35）：`h3_weight_load_f16_raw` 转正

`h3_weights.{h,c}` 新增行主序 fp16 读取口（F32 舍入 / F16 逐字拷贝，形状与 dtype 不符即拒绝），
探针里的临时 `read_weight_f16()` 删除并改为调用它。两条分支各有真实数据覆盖：
`h3_ane_vae_test`（F32 存储，逐字节对齐闸门自算结果）＋ `h3_int8_raw_test`
（分片内 F16 矩阵 `blocks.0.adaln_proj.linear.weight`，验逐字读取与拒绝 I8）。
数据与理由见 F22。

D2d 待做：`h3_video_vae.c` 接线 —— `H3_ANE_VAE` 默认 off、`H3_ANE_VAE_MIN_ROWS=1024`、
`H3_ANE_VAE_MAX_BLOCKS=31`（可压低：少几张图＝少编译/少句柄/少产物）、rows 分桶、
池化平面、常驻不轮换、逐 block 与 fp32 Metal 对拍 + 余弦门。

# D2d 完成（2026-09-19 17:55）：接线正确、回退干净，但**性能判负**，建议 Phase D 收尾

代码：`h3_video_vae.c`（`H3_ANE_VAE` 默认 off / `MIN_ROWS=1024` / `MAX_BLOCKS=31` /
rows 分桶 / 池化平面 / 常驻不轮换 / 逐 block 回退 / `H3_ANE_VAE_STATS` 分阶段计数）、
`h3_ane_linear.{h,m}`（`h3_ane_projection_set_activation_rows`，把"编译用的行数桶"
与"激活真正的行数"分开，1×1 卷积逐位置，所以桶比真实行多是安全的）、
`h3_weights.{h,c}`（D2c 的 `h3_weight_load_f16_raw`）、
`tests/test_ane_vae_decode.c` + Makefile 目标 `h3_ane_vae_decode_test`（同一 latent
先 metal 后 ANE 的 A/B，`REPEAT` 分离建图与稳态，门 `cos>=0.999 && rel_l2<=0.02`）。

实测（12×16×16 latent、1797 行→2048 桶、streaming=1，全部数据见 F23）：
- 正确性 PASS：cosine 1.000000，rel_l2 2.2e-04（6 block）~5.2e-04（30 block），max_abs ≤3e-3。
- 回退 PASS：`w1` 编译失败只停在该 block，其余走 fp32 Metal，输出仍正确。
- 性能判负：稳态 **6 block 1.00×、16 block 0.95×**；产物没留住的那几轮 0.13~0.38×。
  原因（与缓存无关的两条）：四个 GEMM 只占解码墙钟的 16%（理论上限 11%），
  而每次跨界要 `submit+waitUntilCompleted`，**约 12~13 ms/次调用**的流水线空泡
  （6 block 与 16 block 两轮独立反推一致）把上限吃光。

**建议**：Phase D 到此为止，接线保留、默认 off，不再投"整 block 一张图/融合 FFN"
（那两条改不了上面两个原因），D3 只保留"默认 off 时的画面回归"这一项，已完成的部分：
`h3_tests` 1829 checks、`h3_semantic_vae_test --streaming-parity` 逐字节一致、
`h3_ane_staging_test`/`h3_ane_int8_test` 0 failure、`h3_ane_vae_test 1797` 仍 2.43× PASS。
待用户拍板：是否就此收 Phase D；两个另开的缺陷（跨界空泡能否用双缓冲消掉、
ANE 产物在本机留不住的机制）是否值得单独一轮。

# 新增（2026-09-21 06:25）：DiT 逐算子剖析完成（F24 / progress 续 10）

`H3_DIT_OP_PROFILE=1` 已落地（只改 `h3_dit.c` + README 一段），两档分辨率的逐算子
ms/占比/有效算力都量了，括号自身的失真也做了对照闭合（0.06%）。剖析开关与产品输出
逐字节一致，默认关。

**这张表改变了什么判断**
- 任何再上加速器（ANE 或别的）的**唯一有效目标是 MLP（45~52%）+ QKV（25~26%）**；
  只搬注意力在 864 的天花板是 20.5%、576 只有 12.2%，而每次跨界 ~18 ms 的空泡
  （本轮独立量出，印证 F23）在 576 一档就把 4 次跨界吃成 MLP 份额的 15%。
  ⇒ Phase D 的"否掉"结论现在有了逐算子依据，不再只是单 block 闸门外推。
- GEMM 已跑在 2.8~3.7 TFLOPS（非 TensorOps BF16 近本机上限）⇒ kernel 层没有大肉，
  省时间只剩"少算"（reuse / layers / token-reduction / 稀疏注意力）与低比特上 TensorOps。
- AdaLN/gate 全线 ≤1% ⇒ 这条线上已经没有可做的活，之前的 AdaLN 融合到此为止。

**新出现、待用户点头的三条（按性价比排序）**
1. 块稀疏/窗口注意力在 864 的 20.5% 份额：砍一半 ≈ 全局 −10%，且现成内核
   （`h3_gpu_sdpa_window_mask_bf16` / `h3_flash_attn_windowed`）已在库里，只差接线 +
   画质 A/B。分辨率越高越值。
2. fused MLP 比拆开的两次 GEMM 慢 8~10%（**但在跨进程噪声内**，见 F24 §6：同一算子
   跨进程能差 18~20%）。要定罪需同进程 A/B；若为真值 ~4% 全块时间，代价是 +405 MB 激活。
3. `--token-reduction` 之外没有别的减 token 手段这条假设：SDPA 是 O(M²)，864 与 576
   的份额差 8 个点，值得回头看有没有中等分辨率的专用档。

**仍然挂着的旧决策项**（未变）：Phase D 是否就此收尾；~13-18 ms 跨界空泡值不值得
再做一轮双缓冲；ANE 产物默认配置下留不住的机制（trim vs store 失败）要不要单独定性。

# 新增（2026-09-21 06:55）：fused MLP 判无罪（F25 / progress 续 11）

上一节三条待点头里的第 2 条已经自己结掉：新 target `h3_mlp_fusion_bench` 做同进程交替 A/B，
fused/拆开的比值 0.989~1.007（两次 7074 符号相反），13 轮配对 95% CI 上限 1.042
⇒ F24 §4 那句"融合 MLP 慢 8~10%"撤回，默认融合路径不动，405 MB 激活节省白拿，
`H3_DISABLE_FUSED_MLP=1` 退回纯数值对参考口。

顺带定死的**方法学闸门**（后续任何性能结论都要遵守）：这台机器上单算子 A/B
小于 3% 的差只在"同进程 + 逐轮交替 + 多轮中位数"下才算数；同 binary 两次运行的绝对值
能差 6~12%，所以 F24/F25 里的绝对 ms 只能当量级，跨运行的比较一律用比值。

**剩下两条待点头（重排后）**
1. 块稀疏/窗口注意力（864 一档 SDPA 占 20.5%，砍一半 ≈ 全局 −10%；现成内核
   `h3_gpu_sdpa_window_mask_bf16` / `h3_flash_attn_windowed`，只差接线 + 画质 A/B）。
   现在这是唯一还有肉的方向：MLP 那条"白捡 8%"已确认不存在。
2. `--token-reduction` 之外是否还有中等分辨率的减 token 手段。

**仍然挂着的旧决策项**（未变）：Phase D 是否就此收尾；~13-18 ms 跨界空泡值不值得再做一轮
双缓冲；ANE 产物默认配置下留不住的机制（trim vs store 失败）要不要单独定性。

# 新增（2026-09-21 07:15）：窗口注意力"只差接线"这条已量死（F26 / progress 续 12）

新 target `h3_attention_bench` 在同一进程、同一套 q/k/v 上比 `h3_gpu_sdpa_bf16`（DiT 真用的
MPSGraph 稠密）与 `h3_gpu_flash_attn_bf16`（`h3_flash_attn_tiled_windowed`），并用
`radius=16`（17 帧全可见 ≈ 和对数相同的稠密对照）单独读出"每对成本"：
**每对贵 70~78 倍**，两种分辨率、两种窗口宽度互相闭合到 1~3%。
原因在内核写法（`h3_shaders.metal:5600-5712`）：一线程一 query 行、K/V 整行从 global 逐 key
重读、online softmax 每个 key 重标 128 通道累加器、纯标量 FMA；`h3_gpu_sdpa_window_mask_bf16`
只是造 mask 喂稠密 SDPA，一对都不省。`h3_gpu_flash_attn_bf16` 在 `h3_dit.c`/`h3.c` 里没有任何调用点。

⇒ 上一节"剩下两条"里的第 1 条（块稀疏/窗口注意力，估全局 −10%）**作为"接线活"结束**：
接线会把那一个算子放大 13.3 倍、每块时间变成 2.5 倍。
20.5% 的份额还在，但要吃到必须**新写一个 threadgroup 分块 + matmul 累加的 flash 内核**，
而且要先赢过 2.96 TFLOPS 的稠密 MPS SDPA —— 这是一周内核工程，性价比要重新评估。

**当前待点头清单（重排）**
1. 要不要为块稀疏注意力投入"新写分块 flash 内核"这一周级的活（不是接线）。
2. `--token-reduction` 之外还有没有中等分辨率的减 token 手段。
3. 旧三项未变：Phase D 是否收尾；~13-18 ms 跨界空泡要不要做双缓冲一轮；
   ANE 产物默认配置下留不住的机制（trim vs store 失败）要不要单独定性。

# 决策（2026-09-21 07:30）：ANE 这条线放弃（用户拍板）

`ane放弃` ⇒ Phase D 就此收尾，不再开新轮；`h3_ane_bridge/linear/block` 与 video VAE 的
`H3_ANE_VAE`（默认 off）接线**保留在树里但不继续投入**，理由链已经齐：
F23（占比 16% + 每次跨界 ~13 ms 空泡 ⇒ 判负）、F24 §7（可 offload 池 MLP+QKV 才是大头，
注意力天花板只有 20.5%/12.2%）、F21/F23 更正（31 block 产物 3.9 GiB 撞本机磁盘下限，
默认配置下稳态产物留不住，机制未定性）。

**因此从待办里划掉**：ANE 产物持久化机制定性（trim vs store 失败）、
~13-18 ms 跨界空泡的双缓冲轮 —— 两条都只在 ANE 继续活的情况下才有意义。
如果哪天真要复活，前置是：先确认机器有 ≥10 GiB 系统卷余量，否则冷编译会一直吃 0.13~0.38×。

**注意**：这只是"停止投入"，**不等于删代码**。删除 `h3_ane_*.{c,m,h}`、`--ane-*` 开关
或 video VAE 里的 ANE 分支需要单独点头。
# 新增（2026-09-21 07:45）：块稀疏的定价改口（F27 / progress 续 14）

F26 那句"要吃到注意力天花板得新写分块 flash 内核"这轮被自己收窄了：
把一次稠密 SDPA 拆成 G 次小稠密 SDPA，**每对成本在 ~540 行以上不涨（0.8~0.9×）**，
7074 档 G=6 是 time ×0.140 / pairs ×0.167，G=24 仍 1.1×；
只有拆到 135~180 行才涨到 1.4~1.8×。同一台机器、同一次进程内配对。
⇒ 稀疏注意力的算力侧是免费的，"一周内核工程"这条前置取消。

配 F24 份额的上限（keep 越狠边际越小）：
| keep | 864 全局 | 576 全局 |
|---|---|---|
| 1/6 | −17.7% | −10.2% |
| 1/12 | −18.9% | −11.0% |
| 1/24 | −19.6% | −11.2% |
⇒ 速度侧 1/6 就吃满，**这项的取舍全部落在质量侧**，原型该从最保守的 keep 起测。

## 真正的前置：矩形 SDPA，不是新内核
`h3_gpu_sdpa_bf16` 建图时 Q 与 K/V 共用同一个 `sequence`（`h3_gpu.m:1542-1568`），
batch 在 `:1693` 钉成 1 ⇒ 只能方阵 ⇒ 只能表达**块对角**；块对角对视频必判死
（帧间不通，视频行还看不到那 189 行文本条件）。
"每个 query 块看它选中的 k 个 key 块" 需要 (B 行 query) × (k·B 行 key)。
gather 侧不用新东西：选中的 key 块是连续段，现成 `h3_gpu_copy_bf16` 打包即可。

## 重排后的待办
1. **（等拍板）** 给 SDPA 加 `query_rows ≠ key_rows`（顺手 batch>1）：动的是全项目共用
   的 GPU API，且现有 cache key/shape/feeds 三处都要分叉，先对齐再改。
2. 改完第一件事量矩形 (B × kB) 的每对成本 —— 现在的表只覆盖方阵 kB=B；kB≫B 未证。
3. 同批要量的：50 层各用不同 keep ⇒ 几十张不同 shape 的 MPSGraph 首步编译时间 +
   `sdpaCache` 显存（16 GB 机器上这条不过关就别铺开）。
4. 质量闸：固定 keep 1/6 起做渲染 A/B；过了再谈逐层 keep 静态表怎么接
   （注意 `H3_DIT_LAYER_POLICY` 今天跳的是**整块**，注意力 keep 需要新旋钮）。
5. `H3_REUSE_STEPS=2`（已 −45.9%）与稀疏是乘法关系，先叠哪个都行，不冲突。

## 未测 / 不可信项（照实记）
- buffer 全 0（这些内核无数据相关早退，但没排除 MPSGraph 按值选 kernel）。
- F27 §3 的换算是跨进程相乘（份额取 F24、比值取本进程），按 6~12% 绝对噪声读量级。

# 新增（2026-09-21 09:15）：矩形 SDPA 已落地并定价（F28 / progress 续 15）

上一节的两条前置都做完了：
- **待办 1**（给 SDPA 加 `query_rows ≠ key_rows`）：`h3_gpu_sdpa_rect_bf16` 已实现，
  数值闸在 `tests/test_flash_attn.c::test_sdpa_rectangular`（已接进 `make test`）：
  rect(5,5) 与方阵调用逐位相同、rect ⊂ dense、`copy_bf16` 打包两段不连续 key 行后
  对 BF16 参考 2.2e-04。老路径传 `sequence, sequence`，产品行为不变。
- **待办 2**（量矩形每对成本）：**kB ≫ B 和 kB ≪ B 都平**。7074 档 6 个形状全在
  0.9~1.0×，keep=5% 的注意力 30.0 ms（稠密 564~613 ms）；带打包流量的 `rectg`
  在 6 个大 query 块时 30.3 ms（几乎零代价），18 个小块时 35.9 ms（+19%）。

新量出来的唯一失效边界不是形状、是**单次调用的活量**：≥~140k 对/次 ⇒ 1.0×；
~97k ⇒ 1.2×；≤65k ⇒ 1.3~1.6×。⇒ 选形状的规则写成一句：**query 块要少而高，
keep 用 S 调，不要用 G 调**（方阵 split 做不到这点，它的 keep 被钉成 1/G）。

天花板（F24 份额 × 本轮带 gather 的比值）：864 **−19.4%**@5%、−18.6%@10%；
576 −11.3%@5.5%、−10.6%@11%。比上一节的估计又高一点，因为它没算 gather 而
现在算了。**速度侧到此吃满，剩余全在质量侧。**

## 重排后的待办
1. **（等拍板）单 scratch 复用闸**：真实路径每块的打包 K/V 只有 20.3 MB（S=707），
   前提是一条 chain 里 `copy → SDPA` 串行复用同一块 buffer 被正确排序。
   这条不过关就得退回"每块一套"，最狠一档 813 MB ⇒ 16 GB 机器上先验它。
2. **（等拍板）多段 gather 的编码开销**：真实模式每块要拼 3~5 段不连续 key 行
   （自己的帧窗口 + 那 189 行文本），本轮每块只搬 1 段。字节数不变、op 数翻倍，
   吃的是 CPU 编码那 0.1~1.3 ms；量一遍确认不构成新瓶颈。
3. 50 层各异 keep ⇒ 几十个不同 shape 的 MPSGraph 首步编译时间 + `sdpaCache` 常驻。
   **这条是逐层静态表接线前的硬闸**，不过关就全层共用一个 keep。
4. 质量闸（真正没测的那个大头）：固定 keep 从最保守的 1/6 起做渲染 A/B。
   现在才有可表达的形状（块对角对视频必判死）。过了再谈逐层 keep 静态表怎么接 ——
   注意 `H3_DIT_LAYER_POLICY` 今天跳的是**整块**，注意力 keep 要新旋钮。
5. top-k **选块**通路（谁进那 354 行）本轮完全不涉及，是 4 之后的独立工程。
6. `H3_REUSE_STEPS=2`（已 −45.9%）与稀疏是乘法关系，先叠哪个都行。

## 未测 / 不可信项（照实记）
- 质量：一次都没测过。上面所有百分比都是"若质量等价"的上限。
- buffer 全 0（无数据相关早退，但没排除 MPSGraph 按值选 kernel）。
- 天花板仍是跨来源相乘（份额 F24 / 比值本轮），按 6~12% 绝对噪声读量级。

# 新增（2026-09-21 09:55）：待办 1 关掉 —— scratch 复用闸过了（F29 / progress 续 16）

上一节待办 1（单 scratch 复用）验完，**结论是可以复用**：
`test_sdpa_scratch_reuse` 让 3 个 query 块在一条 chain 里共用同一块 11 行打包
K/V（其中一块故意分两段 gather），每块对它自己的选行参考 `max_err=2.18e-04`，
与"每块独立 buffer"那次误差完全相同，且相邻块输出不逐位相同。

影响两笔账：
- **显存**：打包 K/V 常驻 = `2·S·56·128·2` ⇒ S=707 时 **20.3 MB**，不是 bench 那种
  每块一套的 813 MB。块稀疏的显存侧从此不构成障碍。
- **排空**：不需要"每块之间插一次 submit"这条退路，所以 F24 §5 那 17.9 ms/括号
  不会被请回来（那是 keep=5% 时注意力 30 ms 的 60%，一旦被迫每块排空方案就废了）。
- 附带收获：多段 gather 的**正确性**已占（两段），只剩它的**编码开销**没量。

产品代码这轮未动，只加测试；`make test` exit 0。

## 重排后的待办（覆盖上一节编号）
1. 多段 gather 的编码开销（每块 3~5 段：帧窗口 + 那 189 行文本）：bench 加一档
   "每块 N 段 copy"，看 CPU 编码那 0.1~1.3 ms 会不会被 op 数翻倍顶起来。
   顺手同批测：50 层各异 keep ⇒ 几十个不同 shape 的 MPSGraph **首步编译时间** +
   `sdpaCache` 常驻。**这条仍是逐层静态表接线前的硬闸**，不过关就全层共用一个 keep。
2. **（等拍板）** 固定 keep 的渲染质量 A/B：从最保守的 1/6 起，形状按 F28 §3
   "query 块少而高、keep 用 S 调"。需要新旋钮（`H3_DIT_LAYER_POLICY` 跳的是整块），
   保留稠密 Metal 对照 + 余弦校验。
3. top-k **选块**通路（谁进那几行）是 2 之后的独立工程，本轮仍不涉及。
4. `H3_REUSE_STEPS=2`（已 −45.9%）与稀疏是乘法关系，叠法上仍排在质量闸之后。

## 未测 / 不可信项（照实记）
- 质量：仍然一次都没测过；所有百分比都是"若质量等价"的上限。
- scratch 复用只在"一条 chain、串行、小 shape"下验过，没验跨 command buffer 的复用。
- buffer 全 0；天花板跨来源相乘（按 6~12% 绝对噪声读量级）。

# 新增（2026-09-21 14:45）：待办 1 关掉 —— 三道前置闸全绿（F30 / progress 续 17）

上一节待办 1（多段 gather 编码 + 50 shape 编译/常驻）**全部量完，没有一条否决逐层表**：
- **图缓存显存**：共享池下 50 个不同 shape 合计 **+48 MB（≈1 MB/shape）**，
  `engine tensors` 全程 774/774 MB ⇒ 之前那次 801→4042 MB 是 bench 每 MODE 重开池子的
  抖动（"同一 shape 跑 50 轮"对照一样爬到 4835 MB 已证伪）。
- **首步编译**：2.9~7.2 ms/shape，中位 3.9，50 层一次性 **201 ms**；命中后 re-encode
  0.3~0.9 ms。一个纯 GPU 括号 17.89 ms ⇒ 相当于"多一个括头、只付一次"。
- **多段 gather**：段数 1→8 在 6×1179×707 上是 54.6→55.5 ms、18×393×354 上是
  34.6→34.4 ms（每对 0.9/1.1 不变）；288 条 copy + 18 次 rect 的 CPU 编码 **0.8 ms**。
- **keep 带**：固定 B=1179 扫 key_rows 480→1180（keep 6.8%→16.7%）单调、每对 0.8~0.9，
  每 1% keep ≈ 5.1 ms/块 ⇒ 逐层表的 keep 取值不需要往特定形状上凑。

方法学新增一条硬规矩（F30 §6，已落进 bench）：**跨 shape 比较的任何一轮，先看
`NOTE: dense is X× the fastest dense at sequence N`；出现即该 sweep 丢弃**。
本轮就是靠这条抓出一次外部占用伪造的"每对 4× 悬崖"（同进程配对救不了跨机器状态，
小调用比大调用亏得更多）。

## 重排后的待办（覆盖上一节编号）
1. **（等拍板）** 固定 keep 的渲染质量 A/B —— 现在这是唯一门槛。从最保守的 keep 起
   （建议先 5% 一档，其次 10%），形状按 F28 §3"query 块少而高、keep 用 `key_rows` 调"，
   每块选行 = 自己那片帧窗口 + 那 189 行文本（两段 gather 的通路 F29/F30 已验）。
   需要新旋钮：`H3_DIT_LAYER_POLICY` 跳的是整块，表达不了"注意力看几行"。
   保留稠密 Metal 对照 + 余弦校验，两档 geometry（864×480 / 576×320）各一次。
2. 上真机前的最后一道非质量闸：**稀疏通路的端到端接线**（DiT 里替换那一次
   `h3_gpu_sdpa_bf16` ⇒ 打包 scratch + N 次 `h3_gpu_sdpa_rect_bf16`，逐层 keep 先常量）。
   这一步的产物应该正好是 1 的渲染器，所以和 1 合并做，不必单列工期。
3. 逐层静态 keep 表（离线标注）：等 1 有"keep→质量"曲线之后再谈，别提前造旋钮。
4. top-k **选块**通路（谁进那几行）仍是 3 之后的独立工程。
5. `H3_REUSE_STEPS=2`（已 −45.9%）与稀疏是乘法关系，叠法仍排在质量闸之后。

## 未测 / 不可信项（照实记）
- **质量：一次都没测过。** 所有百分比（864 −19.4% / 576 −11.3%）都是"若质量等价"的上限。
- 那两次降速窗口的日志（`/tmp/attn_segments3.log`、`attn_shapes50_shared.log` 的
  S≥480 段）计时作废，只留 resident / enc。
- 降速成因没定位：`pmset -g therm` 无告警、无 `h3-mlx`，在场的是 WindowServer /
  Qoder Helper / UURemoteServer / VTEncoderXPCService，16 GB + swap 常驻 3.7 GB。
- scratch 复用只在一条 chain、小 shape 下验过；跨 command buffer 没验。
- bench 的 buffer 全 0（内核无数据相关早退，但没排除 MPSGraph 按值选内核）。

---

## 新增（2026-09-21 19:45）：待办 1+2 关掉 —— 固定 keep 被质量判死（F31 / progress 续 18）

块稀疏已接进 DiT（`H3_SPARSE_ATTN`，默认 off），等价闸位精确，质量 A/B 跑完四档 keep：
可用质量点在 keep ≳43%，那里只省 −14.7%，低于 `--reuse 2` 的 −45.9%（20 步实测，不碰注意力）。
⇒ **固定 keep（时间窗）这条线关闭**，不再排队做逐层静态 keep 表。

重排后的待办：
1. （原 5）把 `H3_REUSE_STEPS` 与分段/时长策略的组合收益在 864×480 上量一遍，给出默认建议 —— 这是目前唯一确定能拿的时长收益。
2. 注意力若还要压：只剩"选对的块"（top-k / 离线标注逐层表），验收判据已在 F31 §5 量化（keep ≤30% 且 detail/sat 回到 dense、layer0 video cos ≥0.95）。没有满足判据的候选选择器之前不启动。
3. `H3_SPARSE_ATTN` 通路保留在树里（默认 off），作为 2 的现成载体；不做兼容层、不加配置项。
4. 已放弃：ANE 线（用户决定）；库自带 windowed attention kernel（F26，70× 每对成本）。

---

## 新增（2026-09-21 22:58）：待办"reuse 组合收益量化"关掉（F32 / progress 续 19）

- 归因修正：−45.9% 属 `--reuse 2`（20 步），`H3_REUSE_STEPS` 是自定义步号列表且需 `--reuse ≥2`，此前未测。
- 6 步实测：省下的时长 ≈ 1 − 新鲜评估数/步数（4 评估 −29.8%、3 评估 −48.1%），与 20 步那次同一条规律。
- 等成本换形状（0,2,4,5 vs 0,3,4,5）：均匀优于堆尾 13.8 pt detail ⇒ `H3_REUSE_STEPS` 只做高级覆写，不进预设。
- 默认建议：`--steps ≥4` 用 `--reuse 2`（6 步下目视免费）；要更快用 `--steps 6 --reuse 3`。

重排后的待办：
1. 把 F32 的默认建议落进 README/预设表（`--reuse` 与 `--steps` 的配对说明 + `H3_REUSE_STEPS` 一行文档），
   并在文档里明确"低步数下 reuse 无效"。
2. 注意力线（top-k / 逐层 keep 表）继续挂起，验收判据见 F31 §5；除非 reuse 的收益被吃满，不启动。
3. 已关闭：固定 keep 时间窗稀疏（F31）；库自带 windowed kernel（F26）；ANE 线（用户决定）。

补（同轮）：上面待办 1 已随手做完 —— README「Sampler and DiT controls」一节补了
reuse 的收益规律（1 − 新鲜评估数/步数）、等成本换形状的实测（均匀优于堆尾）、
`H3_REUSE_STEPS` 的语义与"高级覆写、不进预设"定位，以及低步数下无效的提醒。
下一轮的可选项只剩：把 `--steps 6 --reuse 3` 这条"更快档"补进 §4 预设表的示例命令，
或在 20 步上复核一次形状结论（成本更高，收益只是重复验证）。

---

## 新增（2026-09-21 23:18）：F32 补主体/背景拆分（§6 / progress 续 20），文档已落地

同一轮的收尾验证（零 GPU 开销）：把 detail 拆成主体/背景两列，并第一次目视 `lated4`。
主体损失 auto4 −19.0% / lated4 −37.4% / r3 −43.6%，背景 −4.2% / −15.0% / −5.6%。
⇒ 形状结论方向不变（两区域都是均匀优于堆尾），但两条要记住：
① **reuse 的代价落在运动主体上**，背景几乎不动；② **全图 detail 会低估主体损失一倍以上**，
以后 reuse/稀疏的 A/B 至少要报主体区域梯度（`/tmp/reuse/regions.py` 的掩膜口径）。

文档：README §4 预设表补"2..3 步下 `--reuse` 无步可跳"，新增"更快档"示例
`--steps 6 --reuse 3`（−48.1%，代价写成"先掉运动主体"）；
「Sampler and DiT controls」补主体/背景数字。

重排后的待办：
1. **等成本换"少步稠密 vs 多步外推"**：`--steps 4 --reuse 1` 与 `--steps 6 --reuse 2` 都是 4 次新鲜评估，
   前者把预算全给 σ 网格、后者给调度形状。F32 只比过同一步数下的形状，这一组没测过，
   而它直接决定"低步数档"该推荐哪一个。约 800 s GPU（同窗口跑两档，用 `s/评估` 做窗口自检，
   指标按 F32 §6 口径报主体/背景两列）。
2. 注意力线（top-k / 逐层 keep 表）继续挂起；验收判据除 F31 §5 外，按 §6 补一条：
   **主体区域梯度回到 dense**。
3. 已关闭：固定 keep 时间窗稀疏（F31）；库自带 windowed kernel（F26）；ANE 线（用户决定）；
   reuse 收益与形状量化（F32 + §6）。

---

## 新增（2026-09-21 23:58）：待办"等成本换少步密集"关掉 —— 密集档胜出（F33 / progress 续 21）

同 4 次整网评估下（算子计数逐位相同）：`--steps 4 --reuse 1` 主体梯度 −12.2%、
latent cos 0.9656、帧间差 3.814；`--steps 6 --reuse 2` 分别 −19.0%、0.9158、4.605。
⇒ **整网外推的误差比"少走一步 σ"的离散化更贵**，低步数档应砍 `--steps` 保留 `--reuse 1`；
`--reuse` 回到 20 步档的位置。README §4 示例与「Sampler and DiT controls」已按此改口。
附带免费对照：`--steps 6 --reuse 2` 与 F32 `auto4` 同配置同种子 —— 质量指标逐位相同、wall 差 12.6%，
F30 规则在渲染上再次成立；同窗内还有顺序效应（先跑慢 14%），故本轮不用 wall 差下结论。

重排后的待办：
1. **给 F33 补外部有效性**（便宜，先做）：在 576×320/1 s 上重跑同一组等成本对照
   （`--steps 4 --reuse 1` vs `--steps 6 --reuse 2`，约 2×150 s）。
   现在"密集优于外推"只在 864×480/2 s + fox 这一个 shape/种子/提示词上成立，
   而它要写进预设表 ⇒ 至少要有第二个 shape 支持。指标按 F32 §6 三列口径。
2. **3 评估档**：`--steps 3 --reuse 1` vs `--steps 6 --reuse 3`（同为 3 次评估），
   决定"极速预览"预设。约 16 min。等 1 的结论出来再排。
3. 同窗换序复测（把 wall 差归零）—— 低价值，算子计数已经证明成本相等，除非要报绝对时长。
4. 注意力线（top-k / 逐层 keep 表）继续挂起；判据 = F31 §5 + F32 §6（主体区域梯度回到 dense）。
5. 已关闭：固定 keep 时间窗稀疏（F31）；库自带 windowed kernel（F26）；ANE 线（用户决定）；
   reuse 收益与形状（F32+§6）；等成本换"密集 vs 外推"（F33）。

---

## 新增（2026-09-22 00:27）：待办"给 F33 补外部有效性"关掉（F34 / progress 续 22）

576×320/1 s 同窗三档：`--steps 4 --reuse 1` 全图 detail 只离 6 评估参照 −0.7%，
`--steps 6 --reuse 2` −10.7% **且主体压塌成双头狐身（每一帧都塌）** ⇒ F33 方向复现、幅度更大。
两条方法论收获写进 README：**① 崩塌类失败指标抓不到**（cos 0.9171 vs 可接受的 0.9158）
⇒ 质量必须目视；**② 主体掩膜口径不可跨 shape 照搬**（576 上抓到草叶，71.1%）。
另外该窗 s/评估 极差仅 1.5%、等成本两档 wall 差 0.5% ⇒ 反证 F33 里 864 的 14% 差是窗口噪声。

重排后的待办：
1. **20 步 + `--reuse 2` 在小 shape 上是否安全**（README 的默认档，现在只有 864/6 步与 512/20 步的证据，
   而 576/6 步刚崩过一次）。同窗跑 `--steps 20 --reuse 1` 与 `--steps 20 --reuse 2` 于 576×320/1 s，
   目视为准、指标为辅。约 32.5 s/评估 × (20+11) ≈ 17 min。
   若这里也崩 ⇒ `--reuse 2` 不该再挂"默认"，要按 shape/步数设闸（闸的行为改动需先请示）。
2. 3 评估档（`--steps 3 --reuse 1` vs `--steps 6 --reuse 3`）—— 优先级降到 1 之后，
   因为 F34 已经给了低步数的答案（砍 steps、留 reuse 1）。
3. 同窗换序复测：不再单列，改为**每轮顺手记录 s/评估 极差**（>5% 就不报绝对时长）。
4. 注意力线（top-k / 逐层 keep 表）继续挂起；判据 = F31 §5 + F32 §6（主体区域梯度回到 dense，
   且**同一 shape 内**取掩膜）。
5. 已关闭：固定 keep 时间窗稀疏（F31）；库自带 windowed kernel（F26）；ANE 线（用户决定）；
   reuse 收益与形状（F32+§6）；等成本"密集 vs 外推"（F33）及其外部有效性（F34）。

---

## 新增（2026-09-22 07:14）：待办"20 步 + reuse 2 在小 shape 是否安全"关掉（F35 / progress 续 23）

576×320/1 s 同窗：`--steps 20 --reuse 2`（11 次评估）比 20 次全新鲜省 **46.6%**，
目视同狐同姿态同锐度（只毛色偏冷），detail +1.4%、sat −3.0%、latent cos 0.9767。
⇒ **危险变量是新鲜评估次数，不是步数**：4 次评估在小 shape 塌主体（F34），11 次无差（F35）。
README 默认档（20 步 + `--reuse 2`）站住；低预算档用"少步 + `--reuse 1`"（F33/F34）。

重排后的待办：
1. **补中间档**（便宜，~10 min）：576×320/1 s 上 `--steps 12 --reuse 2`（7 次评估）
   vs `--steps 7 --reuse 1`（7 次评估）。现在"新鲜次数"这条轴只有 4（崩）和 11（安全）两个端点，
   中间 7~8 次正是产品上最常用的一档；同时这也是"少步密集 vs 多步外推"在 20 步以外的第二次复验。
2. **是否加运行时闸**（等 1 的边界再定，行为改动需先请示）：低新鲜次数下把 `--reuse ≥2` 自动降级为 1
   或直接报错。目前 main.c:601 只有一条 warning，而 F34 的失败是"每一帧都塌主体"级别的。
3. 注意力线（top-k / 逐层 keep 表）继续挂起；判据 = F31 §5 + F32 §6（**同一 shape 内**取主体掩膜、主体梯度回到 dense）。
4. 收尾杂务：`/tmp` 里本轮系列的产物约 400 MB（`reuse`/`ab864`/`eqcost`/`eq576`/`eq20`），
   内置盘只剩 4.8 GiB；F31 的 `/tmp/ab864` 结论已归档，可在下轮前清掉。
5. 已关闭：固定 keep 时间窗稀疏（F31）；库自带 windowed kernel（F26）；ANE 线（用户决定）；
   reuse 收益与形状（F32+§6）；等成本"密集 vs 外推"（F33）+ 外部有效性（F34）+ 20 步默认档复验（F35）。

## 新增（2026-09-22 08:05）：待办"补中间档"关掉 —— 7 次评估两档都合格（F36 / progress 续 24）

576×320/1 s 同窗、算子计数完全相同（315/1099/1400/350）：
`--steps 7 --reuse 1` 225.9 s vs `--steps 12 --reuse 2` 219.6 s（各 7 次新鲜评估，相对 20 次省 −65%）。
**目视两档都是结构完整的狐狸**（单头四腿、毛皮锐利），崩塌阈值夹在 **4 与 7 之间**，不是"低于 8"。
两档互有胜负且都在噪声内：外推 detail +0.9% / 密集 −1.9%，sat −5.8% / −7.6%，
只有 latent cos 偏向密集（0.8719 vs 0.9039）。⇒ F33/F34 的"密集优于外推"是 **4 次评估那一档的端点效应**，
不外推到 7 次。帧间差第三次翻方向（本轮外推 2.692 < 密集 3.141）⇒ 正式废弃该判据。
新增代价认识：低评估数真正丢的是**内容**（7 次评估两档都换了姿态，与 20 步参照不同），不是清晰度。
README 已改：§4 删掉"低于 8 档当风险开关"、补中间档段落；「Sampler and DiT controls」同步；
预设表一行改为"约 4 次新鲜评估用少步 + `--reuse 1`，约 7 次两种调度都可用"。

重排后的待办：
1. **是否加运行时闸**（行为改动，需先请示）：现在数据只支持"新鲜评估 <4 会崩、≥7 安全"，
   中间 5~6 次没测 ⇒ 要么先补这一格（~8 min）再定阈值，要么把阈值定在保守的"<6 降级为 `--reuse 1`"。
   目前 main.c:601 只有一条 warning。
2. 注意力线（top-k / 逐层 keep 表）继续挂起；判据 = F31 §5 + F32 §6（**同一 shape 内**取主体掩膜、主体梯度回到 dense）。
3. **环境**：内置盘 2.4 GiB、swap 6.77/7.17 GB（本轮 VAE 解码被 I/O 拖慢的主因）。
   下一轮计时前要先确认这两项，否则 s/评估 极差会超阈值。
4. 已关闭：固定 keep 时间窗稀疏（F31）；库自带 windowed kernel（F26）；ANE 线（用户决定）；
   reuse 收益与形状（F32+§6）；等成本密集 vs 外推（F33）+ 外部有效性（F34）+ 20 步默认档（F35）+
   中间档 7 次评估（F36）；`/tmp` 已清（progress 清理记录 2026-09-22 07:25）。

## 新增（2026-09-22 08:20）：待办"补 5/6 次那一格"关掉 —— 都不崩，但次数/步数未拆开（F37 / progress 续 25）

576×320/1 s 同窗：`--steps 8 --reuse 2`（5 次评估，161.4 s，省 −74.9%）与
`--steps 10 --reuse 2`（6 次，193.9 s，省 −69.9%）**目视都是完整狐狸**。
⇒ 在案的崩塌只有 `--steps 6 --reuse 2` 一个配置；F36 的"阈值在 4~7 之间"过宽。
⇒ 新暴露的问题：**步数与新鲜次数全程共线**，"是次数还是步数在决定崩塌"未证。

重排后的待办：
1. **判别实验（~3 min，跑完这条线就闭环）**：`--steps 8 --reuse 3`（调度 0,3,6,7 = 4 次评估，网格仍 8 步）。
   安全 ⇒ 决定量是**步数/网格粗细**，运行时闸应按 `--steps` 判；崩 ⇒ 决定量是**新鲜次数**，按次数判。
   同 shape 同种子同窗，与 `/tmp/eq576/s6reuse2`（4 次、6 步、崩）直接对照。
2. **是否加运行时闸**（行为改动，需先请示）：等 1 的答案再定阈值与判据。目前 main.c:601 只有 warning。
3. 注意力线（top-k / 逐层 keep 表）继续挂起；判据 = F31 §5 + F32 §6（**同一 shape 内**取主体掩膜、主体梯度回到 dense）。
4. 已关闭：固定 keep 时间窗稀疏（F31）；库自带 windowed kernel（F26）；ANE 线（用户决定）；
   reuse 收益与形状（F32+§6）；等成本密集 vs 外推（F33）+ 外部有效性（F34）+ 20 步默认档（F35）+
   中间档 7 次（F36）+ 5/6 次边界（F37）；`/tmp` 已清一轮。

## 新增（2026-09-22 08:48）：待办"判别实验"关掉 —— 决定量是新鲜评估次数（F38 / progress 续 26）

`--steps 8 --reuse 3`（网格仍 8 步、4 次新鲜评估）目视**同样崩**：面部塌陷成黑色块、躯体覆盖深色横向条带，
cos 0.8181（全阶梯最低）、sat −17.2%。对照 `--steps 6 --reuse 2`（4 次、崩）与 `--steps 8 --reuse 2`（5 次、安全）。
⇒ 步数与次数的共线当场拆开：**主因是新鲜评估次数**；边界 = **4 崩 / 5 起安全**（两种网格各验一点）。
README 两处 + 预设表已按此改写。

重排后的待办：
1. **运行时闸（行为改动，等用户拍板，不自行实施）**：`reuse_interval >= 2` 且新鲜次数 < 5 ⇒
   报错 / 静默降级为 `--reuse 1` / 降级并打印提示（现状 main.c:601 只有 warning）。
   覆盖组合：`--reuse 2` 的 steps 4..7、`--reuse 3` 的 steps 6..10。实现点：h3_dit.c 的
   `h3_dit_reuse_schedule` 已经算出 evaluations 数，把它接到 main 的校验路径即可。
2. 注意力线（top-k / 逐层 keep 表）继续挂起；判据 = F31 §5 + F32 §6（**同一 shape 内**取主体掩膜、主体梯度回到 dense）。
3. reuse 这条线的测量已闭环：收益（F32）、等成本（F33/F34）、默认档安全（F35）、中间档（F36）、
   边界（F37）、主因判别（F38）。除非产品要新的默认预设，不再排测量轮。
4. 已关闭：固定 keep 时间窗稀疏（F31）；库自带 windowed kernel（F26）；ANE 线（用户决定）；
   reuse 阶梯全部六点（F32~F38）；`/tmp` 已清一轮。

## 新增（2026-09-22 11:50）：待办"运行时闸"完成（progress 续 27）

用户拍板"降级 + 打印提示"。实现放在 `h3.c: h3_generate`（`h3_valid_params` 之后），
判据 = `denoise_reuse > 1` 且 `H3_REUSE_STEPS` 未设且 `h3_dit_reuse_schedule` 算出的评估数 `< 5`
⇒ 打印说明并置 `denoise_reuse = 1`；`main.c` 旧的按步数触发的 warning 删除。
验证：`--steps 6 --reuse 2` 的 latent 与 `--steps 6 --reuse 1` 逐字节相同；
`--steps 10 --reuse 2` 与 F37 的 `s10x2.bin` 逐字节相同（不触发路径零扰动）。

重排后的待办：
1. 注意力线（top-k / 逐层 keep 表）继续挂起；判据 = F31 §5 + F32 §6（**同一 shape 内**取主体掩膜、主体梯度回到 dense）。
   打开它之前要先有非 reuse 的收益来源，否则与已闭环的 reuse 线抢同一份时间。
2. 杂务：本轮系列产物 `/tmp/reuse`(100M) `/tmp/eqcost`(49M) `/tmp/eq576`(34M) `/tmp/eq20`(21M)
   `/tmp/mid` `/tmp/boundary` `/tmp/discrim` `/tmp/guard`（后四个各约 9~20 MB）。
   F33~F38 的复现小节都引用它们，要腾空间优先删 `/tmp/eqcost`（结论已被 F34 复验）。
3. 未提交：工作区还带着 ANE 移植、矩形 SDPA、稀疏接线、文档与本轮闸的全部改动（`git status` 十余项）。
   用户未要求提交，保持现状；真要提交时应先按主题拆分。
4. 已关闭：固定 keep 时间窗稀疏（F31）；库自带 windowed kernel（F26）；ANE 线（用户决定）；
   reuse 阶梯与主因（F32~F38）；低预算运行时闸（本轮）。

## 新增（2026-09-22 13:05）：待办"未提交改动"关掉 —— 按主题拆成 5 个提交（progress 续 28）

`d52faa1` 之后新增 `0b9c038`（矩形 SDPA + 块稀疏 + 逐算子剖析/基准）、`b6bb4a0`（ANE 移植，默认 off）、
`133cdc5`（构建接线）、`2aafd31`（reuse 低预算闸）、`27081aa`（F36~F38 与文档）。
`h3_gpu.{h,m}`、`Makefile`、`README.md` 四个混合文件按 hunk 分到各自主题；中间提交做过 `-fsyntax-only`
与符号对照，工作区现已干净，**未 push**。

重排后的待办：
1. 注意力线（top-k / 逐层 keep 表）继续挂起；判据 = F31 §5 + F32 §6（**同一 shape 内**取主体掩膜、主体梯度回到 dense）。
   打开它之前要先有非 reuse 的收益来源，否则与已闭环的 reuse 线抢同一份时间。
2. 杂务：本轮系列产物 `/tmp/reuse` `/tmp/eqcost` `/tmp/eq576` `/tmp/eq20` `/tmp/mid` `/tmp/boundary`
   `/tmp/discrim` `/tmp/guard`（F33~F38 的复现小节都引用它们）。要腾空间优先删 `/tmp/eqcost`（结论已被 F34 复验）。
3. 可选收尾：ANE 的 4 个新文件带 755 位入库（拷贝遗留），要改成 644 需单独确认；`~/h3_sys`、缓存目录按红线不动。
4. 已关闭：固定 keep 时间窗稀疏（F31）；库自带 windowed kernel（F26）；ANE 线（用户决定）；
   reuse 阶梯与主因（F32~F38）；低预算运行时闸；本轮未提交改用的主题拆分。


## 新增（2026-09-22 13:20）：注意力 top-k 线做完离线可行性闸，判死（F39 / progress 续 29）

挂起的"top-k / 逐层 keep 表"按用户指示打开一次，全程不渲染（新增默认 off 的
`H3_DUMP_ATTN_LAYERS` 抓 SDPA 的 Q/K/V，离线重放选择器）。结论见 F39：可打包的
head 共享形式与固定时间窗打平（13.7% 档还更差），能达标的形式要么不可打包，
要么按 1.8 ms/次的实测编码算净负。

重排后的待办：
1. 抓取件已按用户拍板"一起提交"入库（`H3_DUMP_ATTN_LAYERS` + `dump_bf16_capture` 抽取，默认 off）；
   提交前复验：重抓的 `attn_{q,k,v}.00.bin` 与 F39 用的那份逐字节相同。
   保留理由：F39 的判据换算全靠它，将来若真有 GPU 侧 per-head gather 可直接重价而不用重渲染。
2. 注意力线**关闭**（两条独立死因：F31 固定 keep 不够省；F39 打包粒度装不下有用的稀疏性）。
   重新打开的前提不是"更好的打分器"，而是"更细的打包单位"——即 GPU 侧 per-head/per-row gather
   与每调用 ~1.8 ms 编码的解耦；在那之前不再排测量轮。
3. 杂务：`/Volumes/data/tmp/topk`（本轮 dump，约 500 MB：15 个 attn_*.bin × 32.9 MB + 50 层
   qkv/out/fc*）、`/tmp/topk/{study*.py,result*.txt,dump.sh}`（F39 复现小节引用）。
   旧账仍在：`/tmp/reuse` `/tmp/eqcost` `/tmp/eq576` `/tmp/eq20` `/tmp/mid` `/tmp/boundary`
   `/tmp/discrim` `/tmp/guard`，优先删 `/tmp/eqcost`。
4. 可选收尾：ANE 的 4 个新文件 755 位入库（要改 644 需单独确认）；`~/h3_sys`、缓存目录按红线不动。
5. 已关闭：固定 keep 时间窗稀疏（F31）与 top-k / 逐层 keep 表（F39）；库自带 windowed kernel（F26）；
   ANE 线（用户决定）；reuse 阶梯与主因（F32~F38）；低预算运行时闸；未提交改动的主题拆分。

## 新增（2026-09-26 16:00）：AdaLN 调制缓存落地并预生成 4/8/20 三档（F40 / progress 续 30）

上一轮遗留的第 1 条（"缓存按 `--steps` 绑定，换步数要重新导出"）本轮做掉，并顺手把
24.29 GiB 的 AdaLN 权重从 checkpoint 里整个拿掉。

**做了什么**
- `h3_dit_schedule.h`：`H3_ADALN_CACHE_{TIMES,BLOCK,FINAL}_FORMAT` 全部带 `s%d` 步数后缀，
  同一 transformer 目录可并存多份缓存；`blocks.0.adaln_cache_s{steps}` 存在即选中缓存路径。
- `h3_dit_schedule.c`：`prepare_rows` 拆成 `prepare_times`（纯 CPU，先于任何权重读）+
  `prepare_features`；新增 `load_cached_adaln`（逐位前缀校验）与 `dump_adaln_cache`
  （`H3ADALN2` 头，含 steps）；`H3_DIT_ADALN_CACHE_DUMP` 导出时**同时强制 visual+audio 两个
  条件行**；新增 `!cache && !adaln_probe` 的明确报错。
- `fastvideo_qad/scripts/export_h3_adaln_cache.py`：`--dump` 可重复、按步数命名/校验、
  流式写不落内存；`--trim-to` 写新目录、`--verify` 只读复查。
- 产物 `/Volumes/data/MODELS/h3c-q4-adalncache`：**12.25 GiB**（原 35.93），带三档缓存。

**验证**（细节见 F40）：dump 字节数三档逐字节命中预测；`--verify` PASS；
4/8/20 三档 A/B 的 latent **与 mp4 容器** md5 全同、`max|d|=0`；
负向对照 `--steps 6` → exit 1 且报错可读。

**重排后的待办**
1. 本轮改动**未提交**：`h3_dit_schedule.{c,h}` + 两个新脚本 + README 新小节 +
   `findings.md`/`progress.md`/`task_plan.md`。提交前建议再跑一次 `--verify`（只读）。
2. 缓存与 **LoRA 互斥**（要合并进已丢弃的 AdaLN 权重，直接拒绝）——已写进 README 与头注释，
   不打算支持；若要支持得让 LoRA 携带自己的 AdaLN 增量。
3. **纯音频参考**（`H3_LAYOUT_REF_AUDIO` 无图）是唯一不覆盖的条件模式，被 memcmp 明确拒绝。
   CLI/REPL 都构造不出来，只有 C API 可达；若将来加 `!ref-audio` 之类的入口，要先扩缓存行数
   （把 audio 条件行放到 visual 之前，或按模式分键）。
4. 想要别的步数就按 README 新小节的循环重跑一遍导出+裁剪；脚本的 `--dump` 已支持一次装多档。
5. 旧账未动：`/tmp/reuse` `/tmp/eqcost` `/tmp/eq576` `/tmp/eq20` `/tmp/mid` `/tmp/boundary`
   `/tmp/discrim` `/tmp/guard`、`/Volumes/data/tmp/topk`；本轮新增 `/tmp/h3ab/v2`（约 1 GB，
   三个 dump + A/B latent + 日志，F40 复现小节引用）。
6. 已关闭：注意力 top-k / 逐层 keep 表（F39）、固定 keep 时间窗（F31）、库自带 windowed kernel（F26）、
   ANE 线、reuse 阶梯与主因（F32~F38）、低预算运行时闸。

## 新增（2026-10-01）：Strata 对照清单落地为 h3c 待办前先核前提（F41 / progress 续 31）

用户给了一份 Strata（LLM offload）与 h3c 的对照优化清单，要求"参考上面的优化"。本轮只把清单里
三条**事实断言**核到代码（细节与行号见 F41），未改任何行为、未跑 GPU。

**核对结果**：①"流式档位盘上存 BF16"只对 BF16 checkpoint 成立 —— ConvRot int8 checkpoint 的流式
读本来就是 int8（h3_dit.c:1649）并且保留 int8 计算（`requant_stream_slot` h3_dit.c:1517）；没吃到
的是 int8→BF16→int8 这一圈往返（反旋转 16.0 s / 33.6 s 流）。②"uncached 读=放弃保留"方向相反：
`F_NOCACHE` 全仓只有一处且在 BF16 helper（h3_gpu.m:823），int8 流式读根本没设它 —— 保留交给 OS 且
没记账；已有 `load_core()` 的自适应前 N 块常驻（h3_dit.c:2756-2853）。缺的那一维确认成立：
`H3_PROFILE` 无命中/带宽按块拆分（h3_dit.c:5670-5680），并且那行不分路径都印 "BF16 SSD stream"。
③"每调用一个 fd"成立（h3_weights.c:234-247、h3_dit.c:773-786），但请求是 ~5 MB 一块
（`H3_STREAM_CHUNK_ROWS=1024`），不是 Strata #230 的 4 KiB ⇒ 预期倍数小得多。

**重排后的待办**
1. **先清前置杂务**：F40 那轮改动（`h3_dit_schedule.{c,h}` + 两个导出脚本 + README + 三个计划文件）
   从 2026-09-26 挂到现在未提交，按上轮待办 1 的建议先只读跑一次 `--verify` 再入库，别和新轮混提交。
2. **免费（零行为改动）**：修三处与代码不一致的描述 —— README 流式小节按 checkpoint 分述 BF16/int8
   两条读法、`h3.c:918` 的报错点明被拒的是 M5 专属 row-FC2 kernel 变体（不是 int8 整体）、
   `H3_PROFILE` 那行按实际路径印 int8/BF16。
3. **免费 profiling**：给流式读取补"这次读本可以命中吗"的代理指标 —— 按 block 记第 1 次与后续各次的
   pread 秒数与 GiB/s，一次渲染即可回答"页缓存保留值多少钱"，也是②的后续决策前提。
4. **便宜原型 A**：每 worker 常驻 fd（去掉每块 open/close），`H3_STREAM_KEEP_FD` 臂 + 旧路保留，
   量 `pread` 秒与 wall。预期小（MB 级请求），定价半小时一轮。
5. **便宜原型 B（工程纪律）**：逐块 FNV-1a 指纹放在校验/dump 路径（不进热路径），把"cos 动了"变成
   "第 x 块开始动" —— 补 F39 §0 整层 cos 与 F40 latent md5 都定位不到块号的短板。
6. **等第 3 步数据再定**：un-rotate 能否不进 BF16 域（直接 int8 反旋 + 重量化）；常驻前 N 块从"按序号"
   改"按实测成本选块"。两者都要新 kernel 或新调度，需单独批准。
7. **不排期**：ANE 产物当专家层管理（线已按用户决定关闭）；"验证者=大模型本身"的自验证无损优先闸
   （F35-F38 已证同族偏离度指标对崩塌反向，要用就得先拿 4-次新鲜评估崩塌档当负样本验收）；
   `--calibrate` 5-10 min 实测搜索（最贵，且静态 80%/85% 上限的错是 panic 风险不是速度）。
8. 已关闭不变：注意力稀疏全线（F31+F39）、库自带 windowed kernel（F26）、ANE 线、reuse 阶梯与主因、
   低预算运行时闸。


## 新增（2026-10-04）：评审 F40 那轮未提交改动，挑出 8 处可优化点（F42 / progress 续 32）

用户要求"审核代码，看看还有什么可以优化的"。评审范围 = 工作区未提交的 F40 那轮
（`h3_dit_schedule.{c,h}` +270/-25、两个新导出脚本、README 新小节）。本轮**不改行为、不跑 GPU**；
凡能本机证伪的都跑了（现场 `/tmp/f42`，含一个 9.7 MB 合成 dump 与一个双分片假模型目录）。

**做对的（别当问题）**：`prepare_rows` 拆成 `prepare_times`+`prepare_features` 让缓存路径能在读权重、
碰 GPU 之前先验 sigma 前缀；该 TU 的编译告警 **11 → 1**（消掉 4 处 `-Wsometimes-uninitialized` 和 1 处
`-Wshadow`，剩那条在 :419 早于本轮）；前缀校验顺带自然拒掉唯一不覆盖的"无图音频参考"模式。

**8 处可改，按严重度**：
1. P1 LoRA 一刀切（`h3_dit_schedule.c:688`）拒掉**所有**适配器，但计算路径对不含 AdaLN 目标的适配器
   历来是警告跳过（`h3_lora.c:154`）。`.default`（仅 `attn.orig.*`）本可与缓存共存，`.turbo`
   （带 `norm_out.linear`）才真冲突。README:1120 已写"refused"，属**已文档化的保守设计**；收窄的
   真实障碍是缓存档没有 `adaln_proj` 探针 ⇒ `time_dim` 只能停在 2688，判不了 8-wide 剪枝档。要做得先
   给缓存加宽度字段。
2. P2 `H3_DIT_ADALN_CACHE_DUMP=`（空串）实测**强制 2 个条件行但不导出任何东西**，在缓存档还印假警告
   —— 三处判据不一致（:679/:699 用非 NULL，:823 用 `*dump_path`）。入口读一次即修。
3. P3 无图音频参考的报错建议 "rebuild the cache at these steps" **不可执行**（导出永远把 visual 排在
   前，:679-682），正是本轮注释 :676-678 自己批评的那类话术。加一句前置判定即可。
4. P4 `dump_adaln_cache` 非原子（截断文件留在 /tmp）且丢 errno；下游 Python 用长度核对兜住了，
   所以不致命但报告不可读。`.tmp`+`rename`、消息带 `strerror`。
5. P5 实测：`--trim-to` 缺 `--model` 抛裸 `AttributeError`（脚本 :441→:257），同文件 `--verify` 有守卫。
6. P6 实测：**trim 静默丢掉源目录已有的 `adaln_cache_s*.safetensors`**（:265-266 整文件 skip），
   而文档说"只丢 adaln_proj 矩阵"；对输出跑 `--verify` 仍 PASS。"给已裁剪模型追加一档"是 README
   教的操作，这条会吃掉旧档位。
7. P7 缓存路径**零测试**，且唯一的 AdaLN 回归测试 `tests/test_real_dit_schedule.c` 没接进 `make test`
   /`real-parity`（count-mode 复核：Makefile 里只 1 行，即它自己的规则；`clean` 里那行是目标名）。
   dump↔safetensors 是 **C↔Python 跨语言 codec**，按本仓库规矩要双向锚定 fixture；原料本轮已做出
   （合成 dump 能被 Python 读回并 verify 全绿）。附带：接入时要按路径分支，缓存档 `submissions` 应为
   0 而非 52（代码路径判定，**未实测**）。
8. P8 小项：`cache_header`/`adaln_header` 写了没读（`h3_weight_find` 接受 NULL）；无表分支按 2688 分配
   只写 256（依赖一条没写出来的不变量）；:277 溢出检查没算 `sizeof(float)`（原同形，够不着）。

**已核过、不是问题**：三种条件模式的前缀嵌套成立；51 张的 shape 由 `load_tensor` 逐维核对兜住；
C/Python 的 `H3ADALN2` 布局一致；缓存档 GPU 常驻与计算档同形；`st_write` 不补 8 字节对齐但官方
`safetensors` 0.8.0 实测接受错位偏移（真档 s20 里 51/52 张量偏移非 8 倍数），h3.c 走 pread 无影响。

**重排后的待办**
1. **F41 待办 1 仍未做**：F40 那轮改动从 2026-09-26 挂到现在未入库。现在多了 F42 的 8 条评审结论，
   建议顺序改成：先做 F42 待办 2 的低风险局部改 → 再按主题提交（提交前跑一次只读 `--verify`）。
2. **一批做完 P2/P3/P4/P5/P6/P8**：十行内的局部改，不碰算法、无需 GPU 轮次；P5/P6 各配一条 Python
   侧负例（缺 `--model` 干净报错；已有缓存分片被带上或被列出）。
3. **P7 单独一轮**：双向锚定的 codec fixture + 把 `h3_real_dit_schedule_test` 接进 `make test`
   并给缓存档分支。
4. **P1 挂起**：只在"缓存档要用 VDN default adapter"成为需求时才做，前提是先给缓存加 AdaLN 宽度字段。
5. 杂务新增：`/tmp/f42`（约 20 MB：合成 dump + 假模型目录 + getenv_probe，F42 复现小节引用）。
   旧账未动：`/tmp/reuse` `/tmp/eqcost`（优先删）`/tmp/eq576` `/tmp/eq20` `/tmp/mid` `/tmp/boundary`
   `/tmp/discrim` `/tmp/guard` `/tmp/h3ab/v2`、`/Volumes/data/tmp/topk`。
6. 已关闭不变：注意力稀疏全线（F31+F39）、库自带 windowed kernel（F26）、ANE 线、reuse 阶梯与主因、
   低预算运行时闸、未提交改动的主题拆分。


## 新增（2026-10-04）：F42 的 8 条全部落地，两个新闸接进 make（F43 / progress 续 33）

零 GPU 轮次。逐条改动与实测细节在 F43。

**改了什么**
- **P1** 新增可选键 `adaln_cache_meta_s{steps}`（U32[1]=AdaLN 输入宽度）。宽度已知时逐适配器判
  `h3_lora_matches`（块侧 `transformer_blocks.0.adaln_proj.linear`、final 侧 `norm_out.linear`），
  只拒真带 AdaLN 因子的；**宽度未知时保持原来的整批拒绝**，所以已入库的 s4/s8/s20 三档行为不变。
  raw dump 格式与魔数没动 ⇒ 不需要重新导出；写 meta 是导出器的纯磁盘步骤，也不需要 GPU。
  磁盘契约（magic/字段序/宽度/尺寸式/键名）从 `.c` 的文件内 enum 提到 `h3_dit_schedule.h`，成为唯一定义处。
- **P2** env 入口读一次并折叠成 `dumping`，三处判据统一（空串不再"强制 2 行却不导出"）。
- **P3** 缓存分支前置拒绝"无图音频参考"，理由说到位，不再建议无法满足的重建。
- **P4** dump 改 `.tmp` + `rename`，失败清理，错误带 `strerror(errno)`。
- **P5/P6** `--trim-to` 缺 `--model` 干净报错；trim 不再静默丢源目录已有的缓存分片（带过去并打印）。
- **P8** 删只写不读的头参局部；`prepare_features` 按各分支真正读到的行跨度分配；溢出检查算 sizeof。
- **P7** 两个闸：`tests/adaln_cache_probe.c` + `test_adaln_cache_codec.py`（C 常量→编译探针→比对
  Python 侧抄的常量；sentinel 头部验字段序；C 尺寸式喂给 Python 尺寸闸；52 键名拼写）；
  `gen_adaln_cache_fixture.py` + `test_adaln_cache_lora_gate.c`（2.1 MiB 夹具、不提交 GPU 工作，
  靠"越过后停在 times 前缀"来证明闸放行，8 个用例）。`h3_real_dit_schedule_test` 接进 `test:` 与
  `real-parity`，其 `submissions==52` 按路径分支为 `from_cache ? 0 : 52`。
- README AdaLN 小节：键清单加 meta 行，LoRA 那句改成"按目标判定 + 宽度未知则全拒"。

**验证**
- `make test` **exit 0**；codec 绿、闸 8/8 绿；15 条 skip（模型/夹具未装）。
- codec 的外部锚：C 的 `dump_bytes_41/17/9` 与三档**真实已安装**缓存的载荷字节逐行吻合。
- Mutation：codec 5 条、LoRA 闸 4 条，全部 red 且归因到被监视的那条判定；每控制单独删
  object+binary 重建，批后 sha256 确认源码回原样。
- 导出器 P5/P6/P1 用 `/tmp/f42` 合成夹具实测过（含 `--verify` 从 52 键变 53 键仍 PASS）。

**重排后的待办**
1. **端到端复跑被一个事实挡住**：三棵模型树下都没有 `FL2VA/text_encoder`
   （`find /Volumes/data/MODELS -maxdepth 3 -type d -name text_encoder` 为空），F40 复现小节里那条
   `./h3 -d /Volumes/data/MODELS/h3c-q4-adalncache ...` 今天照抄直接报缺组件 ⇒ 拿不到用户真实调用方式
   之前，**本轮不声称端到端复跑过**，F40 小节要就地标注"命令不完整"。要验的是"缓存档 4 步 latent
   md5 仍等于 `6e9c2d87`"。
2. 本机 15 条 skip 里包含刚接进去的 `h3_real_dit_schedule_test` ⇒ 它的 `from_cache ? 0 : 52` 分支
   **未实测**，只有在装了模型与 block0 夹具的机器上才会第一次跑到。
3. 想真正用上 P1 的收窄，需要对 s4/s8/s20 各跑一次 `install`（纯磁盘，无需 GPU）把 meta 写进去；
   已入库缓存现在仍是"全拒"。
4. 提交：F40 + F42/F43 两轮改动现在都在工作区未入库（`h3_dit_schedule.{c,h}`、README、Makefile、
   4 个 tests/新文件、2 个导出脚本、三个计划文件）。按主题拆：ANE/矩形 SDPA 那批早已单独提交，
   本轮至少分「AdaLN 缓存功能」「缓存的闸与文档」两笔。
5. 杂务：新增 `tmp_adaln_cache_fixture`（2.1 MiB，测试自产）与 `/tmp/f42`（F43 复现小节引用）。
   旧账：`/tmp/reuse` `/tmp/eqcost`（优先删）`/tmp/eq576` `/tmp/eq20` `/tmp/mid` `/tmp/boundary`
   `/tmp/discrim` `/tmp/guard`、`/Volumes/data/tmp/topk`。**F40 的 `/tmp/h3ab/v2` 已被清掉**，
   其复现小节引用的三个 dump 与 A/B latent 都不在了。
6. 已关闭不变：注意力稀疏全线（F31+F39）、库自带 windowed kernel（F26）、ANE 线、reuse 阶梯与主因、
   低预算运行时闸。


## 新增（2026-10-04 下午）：整仓评审出一张优化路径表，按证据分两级（F44 / progress 续 34）

用户要求"审核全部代码，看是否有优化路径"。本轮**只评审、不改代码、不跑 GPU**。
四个子系统并行派代理评审（DiT+流式权重 / Metal 层+着色器 / VAE+编码器+ffmpeg / 引擎胶水+内存规划），
brief 里带上了判死红线（ANE、F31+F39 稀疏注意力、F32~F38 reuse、F26），回来后我逐行复核。
证据分级与全部条目在 F44。

**我读源码复核成立的 4 条（可直接排期）**
1. `h3_video_vae.c:934/952-959/1327` —— 流式路径每块每 tile 两次全量 blit，而 `vae->hidden`
   本来就是 `scale_add` 原地累加器（`:653`/`:663`），state 张量与 hidden 同尺寸 ⇒ 两条 blit 全可删，
   纯搬运不改算术。**两个代理各自独立命中同一处。** 同一模式在 `:1234/:1262`、`:1279/:1287`、
   `:1319` 又出现三次。
2. `h3.c:2051-2057` / `:376-388` / `:781-792` —— 同一个"`H3_CLIPPROJ_DIR` 未设"在三条路径上得到
   三个相反答案（运行路径默认启用 ClipProj 并**把本机绝对路径写死进库**；缓存 key 记成 none；
   `h3_load_dir` 判"必须加载 text_encoder 否则 fatal"）。**这正是 F43 §6 端到端复跑被卡的根因**，
   不是"用户少给了个参数"。
3. `h3.c:1375-1377` 用 `56`（那是 DiT 的 `HEADS`）当 latent 通道数，真实是 `VIDEO_CHANNELS=24`
   （`h3_dit.c:28`，`h3_dit.c:4960` 就按它算）；`h3_dit.c:2771-2774` 按 `16` 预留 ⇒ **少留 1/3**，
   误差方向正是把内存守卫放松（`h3_memory_plan.c:43-46` 自认的 panic 侧）。
   代理说的"高估 8 倍"是它的推算，标未实测。
4. `h3_text_encoder.c:623-625` + 常量 `:14-15` —— 把 `151936×5120` BF16 ≈ **1.45 GiB** 整块词表
   embedding 传上 GPU，只 gather `tokens` 行，且 `:639-646` 随后用视觉 span 覆写 pad 行。

**代理报告、我未复核收益的 14 条**记在 F44 §2：cross-chunk 的 1 GiB 全有或全无阈值、requant 末尾
多余 submit（但改法动命令缓冲纪律，不是"删 5 行"）、label 字符串选量化器导致落到标量对照 kernel、
`h3_head_rms_norm_bf16` 一线程一 head、NAX 收尾半线写、视觉塔每参考图重读 ~1 GB、ffmpeg 双缓冲、
终端每帧 fork、M4 每步 6 次 malloc、butterfly/4bit 三遍展-pack-unrotate、memory_plan 的 int8 建议
被立刻清零 + 4 处文档与实际相反、显式覆盖后不重规划、`free_after_stream` 用未夹的 `rec`、
ANE 判死线仍在构建（删除需单独批准）。

**重排后的待办**
1. **先修 F44 §1.2**（ClipProj 三处判据 + 库内机器路径）：它既是 bug，也是"F43 待办 1 的端到端
   md5 复跑补不上"的根因。修完这条，`./h3 -d /Volumes/data/MODELS/h3c-q4-adalncache` 那类
   只有 transformer + 软链的树才能按 README 的说法跑起来。
2. 再修 F44 §1.3（两个常量）与 §1.1（删纯搬运 blit）、§1.4（别上载整块词表）。这四条都不需要
   GPU 轮次就能推进，且都有现成闸口（`h3_semantic_vae_test --streaming-parity`、
   `h3_text_tests`/`h3_real_multimodal_text_test`）。
3. Metal 侧三条（F44 §2 的 3/4/5）**先在 M5 上取 GPU 时间**再动，本机 M4 没有 TensorOps，
   测不出来的东西不要写成结论。
4. "免费描述修正"这批又攒了 4 处（`main.c:82-85`、`h3.h:157-162`、
   `h3_memory_plan.c:22-25` 与 `.h:21-28`），和 F41 待办 2 的三处合并成一轮做。
5. 上一轮遗留未动：F40+F42/F43 两轮改动仍**未入库**；`s4/s8/s20` 的 meta 键还没写（纯磁盘）。
6. 杂务：`/tmp/f42` 与 `tmp_adaln_cache_fixture`（2.1 MiB，测试自产，已进 .gitignore）。
   `/tmp/h3ab/v2` 已不存在。旧账 `/tmp/reuse` `/tmp/eqcost`（优先删）等仍在。
7. 已关闭不变：注意力稀疏全线（F31+F39）、F26、ANE 线、reuse 阶梯与主因、低预算运行时闸。


## 新增（2026-10-04）：F44 待办 1 做完 —— ClipProj 选择折成一个规则，查出静默缓存投毒（F45 / progress 续 35）

**改了什么**（`h3.c` + `h3_text_encoder.h` + README + Makefile + 新测试）
- 单一规则、单一读取点：`h3_clipproj_classify`（纯函数，可表测）→ `h3_clipproj_resolve`
  （全仓唯一 `getenv` 处）→ `h3_clipproj_disk_identity`（缓存 key 用同一条规则）。
  装载路径、生成路径、key 路径此前各自手推，得到三个不同答案。
- **语义定为"ClipProj 是 opt-in"**：DIR 与 PROJ 必须都给；只给一个明确报错；未设 ⇒ 必须加载
  `FL2VA/text_encoder`。依据三条独立文档证据（`comfyui_nodes/__init__.py:19`、
  `h3_text_encoder.h:75-76`、`h3.c:781-784` 本来就是这条），且所有真实调用方都给全了两个变量
  ⇒ 没有现用法被改变。库里不再保留任何默认路径（`grep -c .lmstudio h3.c` = 0）。
- 查出的真危害：未设时生成路径**用 ClipProj 编码、却按 `clipmodel=none`(=50 层) 记缓存 key**，
  于是下一个未设的进程会把 ClipProj 条件当 50 层的结果读回去。此前不可见，只因为无 text_encoder
  的树在 `h3_load_dir` 先 fatal。

**验证**：新闸 `tests/test_clipproj_selection.c` 10/10（已进 `make test`）；mutation 控制 4 条
全红且归因正确（其中一条被预检挡下后换写法重跑）；`make test` exit 0；真机两证（裁剪树 +
两个变量 ⇒ 装载通过并停在 F40 的 6 步负向对照；只给 DIR ⇒ 新报错）。README 新增
「Text encoder choice」小节（此前 0 处提及），并修掉 `h3_text_encoder.h` 那句教回退的注释。

**F43 §6 的堵点解除**：F43 待办 1（缓存档 4 步 latent md5 仍等于 `6e9c2d87`）现在具备条件。

**重排后的待办**
1. **补 F43 待办 1**：按 F45 §6 的两个变量把 4 步渲染跑完，对 `6e9c2d87`；顺带把 F40 复现小节
   补上"必须先设 H3_CLIPPROJ_DIR/PROJ"这一句（那小节此前抄下来是跑不通的）。
2. F44 待办 2 剩余三条：内存估算常量（`h3.c:1375` 用 56 当通道、`h3_dit.c:2772` 用 16 而真实 24）、
   video VAE 每块两次全量 blit（`h3_video_vae.c:952-959`）、整块词表 1.45 GiB 上载
   （`h3_text_encoder.c:623-625`）。
3. F44 §2 的 Metal 三条（label 选量化器 / head rmsnorm / NAX 收尾）**先在 M5 上取 GPU 时间**再动。
4. 攒一批"免费描述修正"：`main.c:82-85` 与 `h3.h:157-162`（`--pipeline` 声称 madvise/重读盘，
   实际 `h3.c:2313-2320` 明写不 madvise）、`h3_memory_plan.c:22-25` 与 `.h:21-28`（说 int8 可叠加，
   实际 `h3.c:1402` 立刻清零）、F41 待办 2 的三处。
5. 小账：`h3_conditioning_cache_dump` 的死参数 `conditioning_key`（现 `h3.c:494`，HEAD 就报，
   未顺手改）；F44 §1.2 的描述"库内机器路径"实际是**两条**且与脚本用的 int8-convrot 目录不一致。
6. 未入库账不变：F40 + F42/F43 + 本轮 F45 都在工作区；`s4/s8/s20` 的 meta 键仍未写（纯磁盘）。
7. 已关闭不变：注意力稀疏全线、F26、ANE 线、reuse 阶梯与主因、低预算运行时闸。


## 新增（2026-10-05）：splash 对照研究出 3 条可搬 / 3 条不可搬（F46 / progress 续 36）

用户让从 `/Volumes/data/git/c/splash`（Apple silicon 自回归 LLM 引擎）看 MiniMax-H3 的加速路径。
纯研究，未改码、未跑 GPU。**结构差异是判据**：扩散"每步过整条序列"，LLM"逐 token、KV 随长度增长"。

**可搬（按价值序）**
1. **内存规划补一道"反查"**：splash 的 `hardBudget = min(--max-memory, recommendedMaxWorkingSet − max(1 GiB,2%))`
   我读了原文（`MemoryPlan.hpp:63-86`）——它自己也用百分比，差别在**叠在 OS 实测量上** + 可用内存走
   `host_statistics64` + 回收是带水位与双阈值迟滞的目标式。我们真正缺的是 `contextTokensWithin(bytes)`
   这种"给定字节能撑多长"的反查：`h3_memory_plan.c:47-53` 只会正算，长片段撞墙与 80%/85% 判错到 panic
   都源于此。搬法是加反查，不是抄它的数。
2. **注意力 K/V 走 INT8**：省容量不成立（扩散无跨步 KV），**省带宽成立**（窗内 tile 被各 query 重读）。
   前置自测两条：splash 自陈 BF16/INT8 之差非通用提速（512-token ±0.8% 内）；我们 head_dim=96 而它 256，
   且 int8 operand 依赖 M5 TensorOps ⇒ **M4 吃不到**。
3. **VDN 扫描 kernel**：`h3_shaders.metal:6170 h3_vdn_scan_step` 是"一线程一元素 + 串行 128 次 FMA"，
   splash GDN 用 lane 连续 4 个 fp32 + `simd_sum` + threadgroup 相位交接。**但跨 lane 归约改求和顺序就改结果**，
   且 VDN 是质量叠加（Phase 9 慢 3.4~7×）⇒ 这条是"降低 VDN 价格"，不是加速基座。

**不可搬（省下轮次）**：投机解码的无损省步（判据 `u·q < p` + 残差重抽要候选有可算密度，而扩散"拒绝第 t 步
就得重跑第 t 步"，结构上没有"一次前向验证 k 步保留前缀"）；`F_NOCACHE` 照搬到流式权重（它在 splash 用于
权重镜像写侧与 state disk tier，都不是"每步重读同一批字节"，抄来即主动放弃页缓存）；token 稀疏注意力
（只记事实：splash 打包单位是"一个 threadgroup = 一个 KV head × GQA 组若干行"，即 F39 缺的更细单位，
但不改变判死）。

**方法学抄一条**：设备事实集中在 IORegistry、"只在原生机器实测到增益才采纳规则"（模拟核数曾误判
−26%~+20%），而 h3c 的 M4/M5 分支现在散在运行时 if 里。

**重排后的待办**（在原有基础上插入，原 1~7 条不废）
1. 先做 F44 待办 2 的三条（内存估算常量 56/24 与 16/24、`h3_video_vae.c:952-959` 双 blit、
   `h3_text_encoder.c:623-625` 整块词表上载）——它们与 splash 无关、成本低、已有闸口。
2. **内存反查**升级为独立一项：把 `h3_memory_plan.c` 从"正算 + 静态比例"改成"实测可用 + 反查可撑帧数"，
   参考 `MemoryPlan.cpp:259-266` 的接口形状（不是抄它的 margin）。这项直接消掉已知的 panic 风险。
3. INT8 注意力 K/V 只在 M5 上评估；开工前先测 M4 的 head_dim=96 下 bf16 SDPA 是不是带宽受限。
4. VDN kernel 改造前先测 `h3_vdn_scan_step` 的受限类型；求和顺序必须保持（splash 自己也守这条）。
5. 上轮遗留不变：F43 待办 1 的 4 步 md5 复跑仍未跑（F45 已解除堵点）；四轮改动未入库；
   `s4/s8/s20` 的 meta 键未写。


## 新增（2026-10-05）：F44 待办 2 三条落地，三次真机 A/B 证逐位不变（F47 / progress 续 37）

**做了什么**
1. 内存估算常量：新增 `h3_dit_latent_elements()`（通道常量仍只在 `h3_dit.c` 一处）；
   `h3_dit.c` 的常驻估算改调 `h3_dit_video_elements/h3_dit_audio_elements`（原来手写
   `C=16`，真实 `VIDEO_CHANNELS=24`）；`h3.c` 规划项原来把 **56=DiT HEADS** 当通道、还漏了
   时间压缩，改为 `h3_temporal`+`h3_latent_canvas`+helper，并把 `render_width/height` 的解析
   提到规划前共用（旧式忽略显式 render 覆盖）；`h3_cli.c` 同口径。
2. `h3_video_vae.c`：`run_block` 加 `hidden` 形参，流式路径把 `states[state]` 直接当 hidden，
   删掉每块每 tile 两次全量 `copy_f32`。静态审计确认无读者依赖被删的拷贝
   （三个调用点之后都走 `finish_chunk_states`，它自己先还原）。
3. 新增 `h3_weight_gather_bf16_rows()`：按 id 逐行 pread 再上传，两条编码器路径都不再
   整块上载词表（50 层 1.45 GiB / 4B 778 MiB），并删掉随之失去用途的 `ids` 张量。

**真机验证**（16 GB M4，缓存档，256²/22f/steps4/seed42）
- 只回退 `h3_text_encoder.c` ⇒ latent 与 mp4 **同一 md5**。
- 只回退 `h3_video_vae.c` ⇒ **mp4 `cmp -l` 差异 0 字节**（不只靠"上游没变"）。
- 三处改动全在的二进制 + int8-convrot 4B ⇒ latent `6e9c2d8786cea097ddf9bdd45e5593bd`，
  **逐字符复现 F40 基线** ⇒ **F43 待办 1 结清**。
- `make test` exit 0，46 行绿，15 条 skip，三个新闸全绿。

**两条必须记住的结论**
- F40 的基线用的是 **int8-convrot** 的 4B，不是库那个 BF16 内置默认 —— 反过来印证 F45 删掉
  那个默认是对的（它和本仓所有脚本实际用的东西都不一致）。已把这两条 env 写进 F40 的复现小节。
- `h3.c` 那半边是**算对了但少了主导项**：activation 从 124.6 MiB 掉到 10.8 MiB（864×480/360f），
  方向是把档位推向更激进；而新旧两式都没建模 DiT 逐块激活（长片段 GB 级）。本次渲染还
  **完全没走到这段**（自动规划门槛是 `ssd_streaming == 0`，我们显式给了 `--ssd-streaming`）。
  ⇒ 不把它当安全修复交付，并入 F46 待办 2 的"实测可用内存 + 反查可撑帧数"一起做。

**重排后的待办**
1. `h3.c`/`h3_cli.c` 的 activation 项：补上 DiT 逐块激活这一主导项，并做成"实测 + 反查"
   （参考 splash `contextTokensWithin` 的接口形状），用 `H3_PROFILE` 的峰值存活字节校准。
2. F44 §2 剩余可做的：视觉塔每参考图重读 ~1 GB（`h3_vision_encoder.c:557-564`）、
   ffmpeg 双缓冲、终端每帧 fork、`--pipeline` 等 4+3 处文档与实际相反。
3. Metal 侧三条仍需先在 M5 取 GPU 时间（本机 M4 无 TensorOps）。
4. 逐位闸已固化：`6e9c2d87…` + `ac0d0940…`（256²/22f/steps4/seed42 + int8-convrot 4B）；
   改任何 DiT/VAE/编码器的数值路径都应回到这两串对上。产物在 /tmp 会被回收，别把文件当锚，
   要锚就抄这串 md5。
5. 上轮遗留不变：四轮改动（F40/F42/F43/F45）+ 本轮 F47 全部**未入库**；`s4/s8/s20` 的 meta 键未写。
6. 已关闭不变：注意力稀疏全线、F26、ANE 线、reuse 阶梯与主因、低预算运行时闸。


## 新增（2026-10-05）：校准实验否证了 F47 待办 1 的主导项假设，抓到内存墙真主因（F48 / progress 续 38）

**实测**（16 GB M4，同 prompt/seed/steps=4，只变 `--frames`，带 `H3_PROFILE=1`）：
22f ⇒ 钉 9/50 块、DiT load peak 8.040 GiB、跑完；44f ⇒ 钉 **13/50** 块、peak **10.995 GiB**、
**去噪中被 SIGKILL（rc=137）**。

**结论：主导项不是 activation，是"每块常驻成本"。**
- 两点斜率 `(10.995−8.040)/4 = 0.739 GiB/块`，扣掉激活差（356 行 × 118,272 B ≈ 0.04 GiB）
  ⇒ **≈0.729 GiB/块**。
- 独立核对（不靠拟合）：`allocate_stream_slot()` 四项 BF16 权重 = 385,874,432 元素 × 2 B
  = **736 MiB = 0.719 GiB/块**。
- 而 `h3_dit.c:2779` 写死 `per_block = 0.5 GiB`，注释理由是"int8 权重占主导 ≈0.39 GiB"
  —— **只对 `--int8` 成立**；M4 上 `--ssd-streaming` 不带 `--int8` 的默认路径是 BF16 槽
  ⇒ 低估 1.44× ⇒ 多钉 44% 的块 ⇒ 13 块 ≈9.3 GiB 槽位 + 11 GiB 峰值，16 GiB 机器被杀。
- activation 在 44f 只有 ~0.09 GiB；每块成本这一项的误差是 13×0.22 ≈ **2.9 GiB**。
- 附带量到：同画幅同二进制两次得到 available **2.6 vs 5.6 GiB**、常驻 **3 vs 9 块**
  ⇒ 只看这一个读数做常驻决策，等于抽奖。

**重排后的待办（替代 F47 待办 1 的做法）**
1. `per_block` 改成按实际槽位算（BF16 = 那四项 ×2 B；int8 = int8 字节 + scale 缓冲），
   尺寸代码里已有，无新数学。预期把 44f 的过钉从 13 降到 ≈9。
2. 常驻上限除 `available` 外再夹一道**物理内存 − 当前峰值**（GPU 固定分配不可换出，
   `h3_memory_plan.c:43-46` 自己承认过）。
3. activation 项照补（118,272 B/行的默认活集合，别名/融合关闭时的差额另加）；它让预留**变大**，
   正好抵掉 F47 那条"修正后预留变小"的告警。
4. "给定字节能撑多少帧"的反查排在 1~3 之后——输入本身不可信时反查没有意义。
5. 逐位锚已固化（`6e9c2d87…` / `ac0d0940…`，256²/22f/steps4/seed42 + **int8-convrot** 4B）；
   1~3 改的是常驻决策与预留，不动数值路径，改完仍须对上这两串。
6. 未入库账继续累积：F40/F42/F43/F45/F47 五轮 + 本轮 F48。


## 新增（2026-10-05）：F48 的三条修正落地，44f 从被杀变跑完（F49 / progress 续 39）

`h3_dit.c` 的自适应常驻决策：① `per_block` 由 `stream_slot_bytes()` 按实际形态算（不再是 0.5 GiB
字面量；BF16 槽 = 0.719 GiB，与我独立算的 736 MiB 和两点拟合的 0.729 GiB 一致）；
② `usable` 除 `available` 外再夹一道 `physical − footprint − headroom`（为此公开
`h3_host_physical_memory()`，规则与 runtime guard 同一套）；③ 新增 `h3_dit_activation_bytes()`
（118,272 B/行，默认配置）按真实 `dit->sequence` 计入预留。

**验收（F48 定的可证伪判据，两条都过）**：256²/44f 从"钉 13 块、DiT load peak 10.995 GiB、
去噪中 SIGKILL(137)"变成"钉 4 块、peak 4.535 GiB、**EXIT=0 写出 mp4**"；
256²/22f 的 latent `6e9c2d87…` 与 mp4 `ac0d0940…` **两串逐字符不变**（顺带实测证明
"常驻块数不影响数值"，此前只是假设）。`make test` exit 0（48 行绿、15 skip），无新告警。

**重排后的待办**（F49 §3 为准，覆盖 F48 的编号）
1. `h3.c`/`h3_cli.c` 的 activation 仍缺 DiT 激活项：要么给可信的 sequence 估计
   （`latent_t × (latent_h/2) × (latent_w/2)` + audio + 文本行数，文本要 tokenize），
   要么把规划挪到 layout 建好之后。做完才能谈 2。
2. "给定字节能撑多少帧"的反查（splash `contextTokensWithin` 的接口形状）。
3. `h3_memory_plan.c:80-82` 用未夹的 `rec` 算 `free_after_stream`（F44 §2 #13）。
4. F44 §2 未做的：视觉塔每参考图重读 ~1 GB、ffmpeg 双缓冲、终端每帧 fork、
   4+3 处文档与实际相反（`--pipeline` 等）。Metal 三条要 M5 数据。
5. 逐位锚：latent `6e9c2d8786cea097ddf9bdd45e5593bd` / mp4 `ac0d0940434ad216d8131f7469154b70`
   （256²/22f/steps4/seed42 + **int8-convrot** 4B + 缓存档）。改常驻/内存决策类代码都应回对这两串。
6. **入库状态（2026-10-05 收尾更正）**：上一版这条写"六轮未入库"是**过期前提**——
   `d51d414`（10-05 14:30）已经收走 F40/F42/F43/F45 与新测试/脚本/README/Makefile；
   本轮另起两笔：`0301614`（F47+F49 代码）、`dbe42d2`（F46~F49 记录）。工作区已干净、**未 push**。
   上面 2362/2414/2453/2491/2533/2565 各行里的"未入库"是当时的状态，保留不改。


## 新增（2026-10-05）：内存规划的 sequence 估计落地并带自证（F50 / progress 续 40）

`h3_dit_sequence_estimate()` + `h3_dit_plan_bytes()`（latent+activation 合式，两个调用点共用）；
`h3.c` 用 `strlen(prompt)` 当 token 行上界，REPL 预览无 prompt 传 0 并注明。
自证：layout 建好后与估计对账，**低估打印 warning**，`H3_PROFILE=1` 打印双方数字。
实测 256²/22f ⇒ `planned 552 / layout 528`（高估 4.5%，安全方向），两串逐位锚不变，
`make test` exit 0。关键帧与 reference 的估计路径**未实测**（reference 是上界），低估由 warning 兜。

**重排后的待办**
1. 提交本步（4 个代码文件 + 记录），仍不 push。
2. "给定字节能撑多少帧"的反查现在有可信输入了：`h3_memory_plan` 加一个
   `h3_memory_plan_rows_within(...)` 形状的反查（照 splash `contextTokensWithin` 的接口，不抄它的数）。
3. `h3_memory_plan.c:80-82` 用未夹 `rec` 算 `free_after_stream`（F44 §2 #13，一行）。
4. i2v/reference 的估计校准：需要一个可跑的参考图锚；跑通后把 F50 §3 那条"未实测"消掉。
5. F44 §2 未做项：视觉塔每参考图重读 ~1 GB、ffmpeg 双缓冲、终端每帧 fork、4+3 处文档纠偏；
   Metal 三条等 M5 数据。
6. `s4/s8/s20` 的 meta 键仍未写（纯磁盘，让 F43 的 P1 收窄生效）。


## 新增（2026-10-05）：反查"这个画幅最长多少帧"落地，测试抓到一个够不着的上限（F51 / progress 续 41）

`h3_memory_plan_budget_bytes()` 抽出来给正/反两条查询共用（否则预算折扣规则会漂）；
`h3_memory_plan_frames_within()` 在 **`22 + 17k`** 梯子上二分 `h3_dit_plan_bytes()`，
答案一定是 `h3_align_frame_count()` 认得的长度；`H3_PLAN_FRAMES_CEILING = 4051`；
CLI `!memory-plan` 多打一行 `Longest clip at WxH: N frames (+)`。

**新测试 `tests/test_memory_plan_inverse.c` 当场抓到真缺陷**：上限第一版写成 4065，
而 `align(4065) = 4068` ⇒ 4065 不在梯子上 ⇒ "撞到上限"的等式永远为假、`+` 号永不出现。
断言 `align(H3_PLAN_FRAMES_CEILING) == H3_PLAN_FRAMES_CEILING` 抓住它，改成 4051 后全绿。
实测对账（256²/文本上界 64）：8 GiB ⇒ **3252** 帧（7.9707 GiB），梯子下一级 3269 放不下
（8.0122 GiB）；768×432 ⇒ **753**（7.9389）/ 下一级 770 放不下（8.1175）。

**顺带就地更正 F48 §2 的手算**：那处用了 `h3_video_encoder_latent_t()`（`(f+3)/4`）而不是
DiT 的 `h3_video_latent_t()`（`((f-5)/17)*5+2`），rows 差算成 356（实为 **752**），
并把 44 帧对齐长度记成 58（实为 **56**）。改实测后每块 `(2.955−0.083)/4 = 0.718 GiB`，
与独立算出的 0.719 对上到 0.001。`h3_dit.c` 那句 "~0.1 GiB" 注释同步改成实测 0.141 GiB。
**今后凡真函数能答的问题不要手算。**

**验收**：`make test` exit 0（新测试 14 条断言全绿、0 FAIL；全仓 50 行 `  ok  ` + 11 行 `ok…`
收尾（含 `ok: 1829 checks`）、15 skip、0 FAIL、该次构建 0 条 `warning:`）；
两串逐位锚 `6e9c2d87…`/`ac0d0940…` 不变；`h3_cli.c` 只剩 2 条原有告警。

**未眼验**：CLI 那一行只在纯 C 层由测试覆盖。管道喂 linenoise REPL 不响应
（`printf '…\nquit\n' | ./h3 -d <cached>` 挂住无输出），要看真打印得在交互终端手打 `!memory-plan`。

**重排后的待办**
1. 提交本步（`h3.c h3_cli.c h3_dit.c h3_dit.h h3_memory_plan.c h3_memory_plan.h h3_host.c
   h3_host.h Makefile` + 新测试 + 三份记录），**不 push**。
2. `h3_memory_plan.c:80-82` 用未夹 `rec` 算 `free_after_stream`（F44 §2 #13，一行）。
3. meta 键装到 `s4/s8/s20`（纯磁盘活，让 F43 的 P1 收窄真正生效）。
4. i2v/reference 估计路径的校准：需要一个可跑的参考图锚（现在是上界 + warning 兜底）。
5. F44 §2 未做的：视觉塔每参考图重读 ~1 GB、ffmpeg 双缓冲、终端每帧 fork、4+3 处文档相反。
   Metal 三条要 M5 GPU 数据（本机 M4 无 TensorOps）。
6. 逐位锚不变：latent `6e9c2d8786cea097ddf9bdd45e5593bd` / mp4 `ac0d0940434ad216d8131f7469154b70`。
