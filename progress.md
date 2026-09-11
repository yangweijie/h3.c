# Progress Log

## Session: 2026-09-05 (FlashAttention + Tiled Windowed Softmax)
### Phase 12: FlashAttention 内核调试 — complete
- **根本原因**：Metal shader 中 `h3_bf16_to_f32()` 接受 `ushort` 参数，但被调用时传入 `bfloat*`，编译器生成错误类型转换代码
- **修复方案**：
  - 添加 `h3_bf16_to_f32(device const bfloat *addr)` 重载
  - 添加 `h3_f32_to_bf16(device bfloat *addr, float value)` 函数
  - 使用 `as_type<ushort>()` 和 `as_type<bfloat>()` 进行显式位转换
- **新增 kernel**：
  - `h3_flash_attn_causal` — causal attention (online softmax)
  - `h3_flash_attn_tiled_windowed` — windowed attention (O(T·window) 复杂度)
- **测试结果**：
  - `flash_attn_causal`: OK (seq=8, heads=1, dim=4) max_err=0.00095
  - `tiled_windowed`: OK (seq=24, F=4, S=4, r=2) max_err=0.00817
  - 1768 个现有测试全部通过

### Phase 13: Tiled Windowed Softmax — complete
- **核心优化**：将 attention 复杂度从 O(T²) 降到 O(T·window)
- **实现**：
  - 每个 query 根据类型（text/video/audio）计算 3 个 key 范围
  - Range 1: text keys (0 到 text_rows)
  - Range 2: video keys (窗口 [t-r, t+r] 帧)
  - Range 3: audio keys (所有 audio tokens)
  - 只遍历范围内的 key，跳过窗口外计算
- **测试结果**：tiled_windowed OK (seq=24, F=4, S=4, r=2) max_err=0.00817

### Phase 14: FP8 量化集成分析 — complete
- 分析 Python 参考项目 `ops/fp8_linear.py`
- C 代码已有很好的量化基础设施：
  - `h3_gpu_quantize_bf16_int8_rows` (SIMD vec4)
  - `h3_gpu_quantize_bf16_int8_groups`
  - `h3_gpu_linear_int8_bf16` (cooperative tensor ops)
  - `h3_gpu_mlp_int8_bf16` (SwiGLU 融合)
- FP8 E4M3 在 M4 上可通过 INT8 tensor core 模拟（同一硬件），动态范围更好（±448 vs ±127）

### Phase 15: Linear Far-Brain 分支基础设施 — complete
- 实现完整的 SanaDelta delta-rule scan 管线：
  - `frame_stats` → `symmetrize_A` → `factor_apply` → `scan_frame` → `log_alpha_prefix` → `gather` → `output`
- **注意**：需要训练权重（alpha、beta、q_norm、gate、to_out_linear），当前 checkpoint 中不存在
- **测试结果**：7 个 linear-branch 测试全部通过

## Session: 2026-09-02 (SR 集成)
### Phases 1–5: 完成
- 在 h3.c 集成 Real-ESRGAN 超分后处理（h3_ffmpeg.c/h/main.c/h3_cli.c 四文件改完，编译通过，端到端 256→864 含音轨验证通过）
- 详见 findings.md / task_plan.md（Phase 1–5）
- git 状态：README.md、h3.c modified，未提交

## Session: 2026-09-03 (色块诊断 + 参数调优)
### Phase 6: 色块根因 — complete
- 失败运行（前序）：target 256×192 / render 128×96 / layer 40 / token-reduction / SR→1024×768 / 15s → 全屏色块
- 对照证明：2s + 原生 256×192 即出内容 → 元凶=128×96 渲染过低，非 steps=4 / 非 token-reduction
- **结论：渲染分辨率过低导致 DiT token 网格不足以表达画面**

### Phase 7: 2×2 对照 — complete（均 2s / layer 40 / steps 4 / target 512×384）
| 内部渲染 | token-reduction | SR | 产物 | 体积 |
|---|---|---|---|---|
| 256×192 原生 | 有 | 无 | /tmp/h3_256_192_native_2s.mp4 | 139K |
| 256×192 原生 | 有 | →1024×768 | /tmp/h3_256_192_native_2s_sr.mp4 | 1.5M |
| 256×192 render | 有 | →1024×768 | /tmp/h3_512_384_r256_2s_sr.mp4 | 1.5M |
| 256×192 render | 无 | →1024×768 | /tmp/h3_512_384_r256_notr_2s_sr.mp4 | 1.9M |
| 原生 512×384 | 有 | 无 | /tmp/h3_512_384_native_tr_2s.mp4 | 223K |
| 原生 512×384 | 无 | 无 | /tmp/h3_512_384_native_notr_2s.mp4 | 283K |
- 手跑 realesrgan-x4plus ×4（512×384 LR）→ /tmp/h3_512_384_lr_x4plus.mp4（2048×1536, 4.1M，最清晰）

### Phase 8: 15s 正式基准 — complete
- 原生 512×384 / 无 token-reduction / 无 SR / 15s
- 命令：`nohup bash -c '... ./h3 -d … -o /tmp/h3_512_384_native_notr_15s.mp4 --profile --ssd-streaming --reuse 1 --seconds 15 --width 512 --height 384 --layers 40 --steps 4' > … & disown`
- 产物：/tmp/h3_512_384_native_notr_15s.mp4（1.0M, 512×384/24fps/362帧/15.08s, exit=0）
- 耗时 ~33min；日志 /tmp/h3_512_384_native_notr_15s.txt、/tmp/time_512_384_native_notr_15s.txt
- 监控：h3 子进程 PID 88102，轮询 FINAL_EXIT 标记确认完成

### Phase 9: 最小清晰分辨率扫描 — complete
- 启动 /tmp/h3_res_scan.sh：6 档 4:3 分辨率 × layer 40/45/50（原生 / 无 token-reduction / 不超分 / 0.5s / steps 4）
- **关键约束**：h3 校验 `width and height must be multiples of 32` → 320×240/192×144/160×120 因高非 32 倍数被拒（EXIT=1，瞬时）
- 有效 4:3 子 512×384 分辨率只有 384×288 / 256×192 / 128×96（W=128k, H=96k）
- 产物 9 个有效文件（均 EXIT=0），日志 /tmp/h3_res_scan.log，06:47:27 DONE

| 分辨率 | L40 | L45 | L50 | 备注 |
|---|---|---|---|---|
| 384×288 | 94K ✓ | 81K ✓ | 81K ✓ | |
| 256×192 | 54K ✓ | 71K ✓ | 59K ✓ | |
| 128×96 | 38K ✓ | 43K ✓ | 43K ✓ | 最小档，清晰度待肉眼判 |
| 320×240 | ✗ | ✗ | ✗ | 240 非 32 倍数 |
| 192×144 | ✗ | ✗ | ✗ | 144 非 32 倍数 |
| 160×120 | ✗ | ✗ | ✗ | 120 非 32 倍数 |

- 待用户肉眼挑最小清晰档（结论：4:3 下最小只能是 128×96；更细粒度需放宽 32 倍数或改非 4:3）

### Phase 10: 人脸清晰度 1:1 矩阵 — complete
- 修正记录：第一版去音+单layer 被叫停；第二版 `--frames 5` 全部 EXIT=1（VAE 需 ≥1 个 22-frame 解码 chunk）→ 改用 **--seconds 0.5（=12 帧）**
- 配置：原生 / 无 token-reduction / 不超分 / steps 4 / 0.5s(12 帧) / 保留音轨 / layer 40/45/50
- 39 档全部 EXIT=0；产物 /tmp/h3_face_<W>_L<L>_05s.mp4 ×39；日志 /tmp/h3_face_matrix.log（08:20:30 DONE）

| 分辨率 | L40(size/time) | L45 | L50 |
|---|---|---|---|
| 256×256 | 58K/82s | 50K/90s | 64K/95s |
| 288×288 | 63K/81s | 61K/88s | 58K/93s |
| 320×320 | 63K/82s | 68K/89s | 73K/96s |
| 352×352 | 96K/93s | 115K/99s | 120K/105s |
| 384×384 | 119K/92s | 140K/98s | 126K/106s |
| 416×416 | 142K/93s | 132K/99s | 155K/105s |
| 448×448 | 147K/96s | 148K/105s | 197K/112s |
| 480×480 | 182K/106s | 193K/115s | 218K/129s |
| 512×512 | 243K/119s | 239K/128s | 237K/137s |
| 544×544 | 271K/132s | 291K/143s | 273K/155s |
| 576×576 | 282K/148s | 287K/157s | 334K/169s |
| 608×608 | 269K/166s | 392K/177s | 389K/191s |
| 640×640 | 262K/176s | 355K/193s | 413K/217s |

- 观察：≤320×320 体积均 ~50–73K（细节偏少）；352×352 起体积跳升（96→120K），为清晰度拐点；向上随分辨率稳步增大；layer 对体积影响小、对耗时影响明显（L50 最慢）
- 用户挑最清晰档（建议从 352×352 起看；256/288 可能仍偏糊）

### Phase 11: 加超分出品 — pending（待用户决定）
- 候选：在选定最清晰档上加 `--sr --sr-target 1024x768`（×2）或 `2048x1536`（纯×4）
- 可选：`--steps` 10/20

## Test Results (本 session)
| 项 | 配置 | 结果 |
|---|---|---|
| 色块复现 | render 128×96 | 色块（根因确认）|
| 根因对照 | 原生 256×192 2s | 出内容（排除 steps/tr）|
| 15s 基准 | 原生 512×384 无 tr 无 SR | 1.0M, exit=0, ~33min |

## Error Log
| Time | Error | Attempt | Resolution |
|------|-------|---------|------------|
| 09-03 | `setsid: command not found` | 1 | `nohup … & disown` |
| 09-03 | 长命令超时风险（15s 约 27–33min）| — | 后台 detached + 30s 轮询 FINAL_EXIT |

## 5-Question Reboot Check
| Question | Answer |
|----------|--------|
| Where am I? | Phase 10 人脸矩阵完成（39 档全部 EXIT=0）；Phase 11 待用户挑最清晰后加超分 |
| Where am I going? | 用户挑最清晰档 → 加超分出品（1024×768 或 2048×1536）或提 steps |
| What's the goal? | 找最小清晰分辨率并产出可用视频（已定位色块根因=渲染过低）|
| What have I learned? | 见 findings.md（色块根因 + 帧数下限 12 + SR 行为 + 分辨率矩阵）|
| What have I done? | 分辨率扫描 + 人脸 1:1 矩阵（13 分辨率 × 3 layer）已出 |

## Session: 2026-09-05 (VDN-H3 完整支持)
### Phase 1: 规格研究 — complete
- Python 参考实现精读完成；checkpoint 权重清单实测（每 block 16 张量）；h3.c 调研完成
- 规格见 findings.md「VDN-H3 规格研究」

### Phase 2: LoRA 扩展 — complete（代码）
- h3_lora: 新增 h3_lora_apply_named（显式 adapter 名）、h3_lora_matches（形状兼容探测）、
  自动推断 adapter 名（default/turbo 后缀）
- h3_dit: lora[4] 数组，--lora 逗号分隔多 adapter 顺序合并；load_block 增加 attn.orig.* targets
- h3_dit_schedule: precompute 增加 loras 参数；adaln_proj.linear / norm_out.linear 合并
  （形状不匹配时告警跳过，兼容 pruned convrot 基座）
- main.c: --linear-branch DIR 参数；h3.h: params.linear_branch_path
- 全库编译通过（仅预存警告）

### Phase 3: 线性分支权重加载 — complete（代码）
- h3_dit_block 新增 16 个 lin_* 字段 + free
- load_vdn_block: 从 vdn_weights store 加载（sp/tm 卷积权重按真实 4/3 维校验）
- load_dit 新参数 linear_branch_path；ssd_streaming 互斥；两公开 API + 全部调用点更新

### Phase 4: Metal 内核 — complete（代码 + 单测）
- 新增 16 个 h3_vdn_* Metal 内核（窗口注意力/门控/特征/短卷积/alpha/Cholesky/三角求逆/bmm/
  行缩放/扫描/gather/读出）+ h3_gpu.m 包装 + h3_gpu.h 声明 + pipeline 注册
- 复用既有 h3_linear_branch_frame_stats/symmetrize/log_alpha_prefix
- 修复：bmm 缺 add 绑定导致命令缓冲中止（Metal 校验层定位）；bf16 值加载必须用指针重载；
  cholesky 对角写 d 而非 1；triinv 清零上三角；log1p 不存在
- tests/test_vdn_branch.c：cholesky 求逆 rel=0、窗口注意力 vs CPU rel_l2=0.00167（含 chunk
  bounds + anchor both 语义），已加入 make test

### Phase 5: run_block 接线 — in_progress

### Phase 5: run_block 接线 — complete（代码）
- run_block: VDN 模式 = 窗口注意力 + softmax gate + bf16 QKV（禁 int8 qkv/头主输出/
  token reduction/激活别名）+ 线性分支（features→beta/alpha→stats→cholesky 求逆→双向扫描
  →gather→readout→to_out_linear→加到 video 行）
- prepare_vdn：c5 窗口 bounds（softmax 未夹紧 + skip_ends 重基）、全部 scratch 缓冲
- load_vdn_block：alpha down/up/dt_bias 转 fp32（Python 的 alpha fp32 岛）

### Phase 6: streaming 支持 — complete（代码）
- 16GB 机器无法全驻留 → STREAM_LIN_OUT 流式 to_out_linear；其余分支张量常驻（~12MB/块）
- LoRA 在流式读取线程内合并：merge_block_loras(blocking=1) +
  h3_lora_merge_blocking / h3_gpu_blocking_linear_bf16（私有命令缓冲，不碰主线程 open buffer）
- 修复内存规划器同时建议 ssd_streaming+int8_row_fc2 的既有互斥冲突
- 端到端跑通进入去噪循环（8 步 turbo，约 7-8 分钟/步，步骤 1/8 已完成）

## Session: 2026-09-07 (INT8 线性分支接入 + 性能剖析)

### Phase 8: INT8（ConvRot）线性分支接入 — complete
- **判决性验证先行**：CPU 复现 convrot 反量化，3 个量化张量 vs bf16 真值
  cos = 0.999954 / 0.999954 / 0.999921；不做反旋转仅 **0.065** → 确认必须走 convrot 路径
- 改动 3 文件 5 处（`h3_weights.h/.c`、`h3_dit.c`），VDN loader 本体**零改动**
- 首次运行即报错 `VDN streaming weight is absent or has the wrong schema:
  transformer_blocks.0.attn.to_out_linear.weight` → 定位 `prepare_vdn_stream_source` 硬性要求 BF16
- 二次报错（预判）：消费点按名从 `dit->weights` 找 scale 会失败 → 改为按 `field==STREAM_LIN_OUT` 选 store
- 修复后跑通；int8 vs bf16 输出 **cos 0.9904 / mean_abs_diff 9.3**

### Phase 9: 步数与性能剖析 — in_progress
- 步数对照：steps=2 时 **VDN 与无 VDN 同样糊** → 黄色 = 欠采样，与 linear/int8 无关
- 端到端计时：**VDN 305s vs 原版 90s（慢 3.4×）**
- `H3_PROFILE` 剖析：79.35 vs 72.14 GiB；**unhidden wait 0.001s vs 23.332s**
- 结论：VDN 瓶颈是**计算**；`to_out_linear` 常驻省 0 秒、需 3.85 GB → **不做**

### 运行清单（本 session，均 256×256 / 1s / seed 42 / --ssd-streaming）
| # | 配置 | 产物 | 结果 |
|---|---|---|---|
| 1 | VDN int8 @2 | /tmp/int8_test.mp4 | 橙棕糊（用户反馈"一片黄"）|
| 2 | VDN int8 @4 | /tmp/int8_test_s4.mp4 | 有狐狸 |
| 3 | VDN bf16 @4（`/tmp/lb_bf16` 符号链接）| /tmp/bf16_test_s4.mp4 | 有狐狸；与 #2 cos 0.9904 |
| 4 | **无 VDN @2** | /tmp/novdn_s2.mp4 | **同样糊**（关键对照，证伪 linear/int8 嫌疑）|
| 5 | VDN int8 @4（计时）| /tmp/linear_s4_timed.mp4 | **308 s** |
| 6 | VDN + `H3_VDN_INT8=1` @4 | — | **崩溃** 13s，`unknown Metal error` |
| 7 | 无 VDN @4（计时）| /tmp/novdn_s4_timed.mp4 | **90 s** |
| 8 | `H3_PROFILE` ×2 | /tmp/prof_vdn.mp4、/tmp/prof_novdn.mp4 | 305s / 90s + 流式统计 |
| 9 | 文件对调后重跑 VDN @4 | /tmp/linear_s4_timed.mp4 | **198461 B，与 #2 逐字节一致** |

### 产物证据（帧统计，t=0.8s）
| 样本 | R | G | B | R−B | std | grad |
|---|---|---|---|---|---|---|
| VDN+int8 @2 | 89.9 | 64.1 | 34.7 | 55.2 | 17.9 | 4.88 |
| 无 VDN @2 | 85.8 | 61.7 | 33.5 | 52.4 | 22.7 | 5.10 |
| VDN+int8 @4（有狐狸）| 127.3 | 98.9 | 64.9 | 62.4 | 23.5 | 5.51 |

### Error Log（本 session）
| Error | Attempt | Resolution |
|---|---|---|
| `VDN streaming weight is absent or has the wrong schema: to_out_linear.weight` | 1 | `prepare_vdn_stream_source` 改为接受 I8 并填 scale 字段 |
| 预判：scale 按名从 `dit->weights` 找不到 | 1 | 消费点按 `field==STREAM_LIN_OUT` 选 `dit->vdn_weights` |
| `H3_VDN_INT8=1` → `DiT stream begin failed: unknown Metal error` | 1 | M4 不支持（NAX 为 M5 路径），放弃该开关 |
| 误判 steps=4 仍糊（只看统计量）| 1 | 用户肉眼确认有狐狸；统计指标不能判断语义正确性 |

### 高分辨率对照（方向 ②）— complete
- 512×384 / 1s / steps=4 / seed 42 / --ssd-streaming
- VDN **1295 s** vs 原版 **186 s** → **7.0× 慢**
- 对比 256×256(3.4× 慢):像素 3×,VDN 缩放 4.25×、原版缩放 2.07×
- **VDN 超线性缩放,原版亚线性** → 尺度越大越不能翻盘

### Phase 10: Streaming GPU ConvRot 反量化 — 失败，已回退
- 方案：给 h3_gpu 新增 `h3_gpu_blocking_weight_dequant_unrotate_int8`
  （沿用既有 `h3_gpu_blocking_linear_bf16` 模式：同一队列 + 独立命令缓冲 + commit/wait），
  让 streaming 线程自己做 GPU dequant，与主线程计算重叠。
- **更正：kernel 数值完全正确，第一轮"根因"是 debug 代码 bug 的假象**
  - 首版诊断写错 bf16→f32：`uint16_t gpu_vals[8]; memcpy(&g,&gpu_vals[k],4);`
    （从 uint16_t* 拷 4 字节，把相邻两个 bf16 拼成一个 float）→ 算出虚假的 0.06–0.14 偏差。
  - 改用正确转换 `bits=(uint32_t)val<<16` 后，跨矩阵多行重测：
    **idx 0–3、row 0/1344/2688/4032/5376/10752/16128/21504 全部 `d=0.000000`**
    → GPU dequant 与 CPU 蝶形**逐位一致**。kernel 没有 bug。
  - 只比 row 0 会漏掉行置换错误（row 0 在两种 layout 下都映射到自身）。
- **真实的失败点在集成层面**：
  1. 可见性/同步：GPU 写入 shared buffer 后，后续 GPU kernel（int8 requant）读不到；
     诊断版因 CPU memcpy 覆盖而正确，纯 GPU 写入版白色失真。
  2. 性能：blocking 每 source 一次 commit+waitUntilCompleted + 额外 tensor 分配/释放。
- **结果**：白色失真（R=235/G=217/B=197, std=8.5），且 **144s vs 基线 90s（更慢）**。
- **决策：回退**。`git checkout h3_dit.c` 恢复 CPU 反量化版本。
  确认 HEAD 版本已含 VDN int8 修复（`prepare_vdn_stream_source` 的 I8 分支），功能完好。
- 回退后验证：VDN int8 @4 产物 **198461 字节**，与回退前一致，零报错 → ✅ 功能完好。
- **fence/barrier 解决（第二轮）**：用 `MTLEvent` 跨 command buffer 同步
  （dequant signal → requant wait），纯 GPU 写入版输出**正确**
  （R=122.1/93.2/57.8/std=24.4 ≈ 基线），证明可见性问题已解决。
- **批量 dequant（第三轮）**：4 个 source 编码进同一 command buffer、
  每 block 只提交一次（50 次 vs 之前 200 次）。输出正确，但 **182s vs 90s 更慢**。
- **最终根因（架构级）**：GPU dequant 与主线程计算**共享同一个 GPU**，
  互相竞争，消除了 CPU 蝶形原本享有的"CPU-GPU 异构并行"优势。
  CPU 蝶形在 CPU 上跑（∥ GPU 计算）→ 仅 23s 暴露；
  GPU dequant 在 GPU 上跑（与 GPU 计算串行）→ 全部暴露。
  **结论：GPU dequant 方向从根本上不可行，即使修好 fence 也不会更快。**

### Phase 16: 端到端参数基准测试 — complete
- 工具链:编写 `benchmark.py`(参数扫描 + SSIM/PSNR/L2 评分 + HTML 报告)和 `fix_text_encoder.sh`
- 发现模型 text_encoder 不完整(`.unfetch_state` 占位符),切换至用户工作模型 `/Users/jay/h3_sys/MiniMax-H3-Convrot`
- 三轮测试(按显存从低到高排序,>13GB 停止):

| 轮次 | 测试数 | 说明 |
|---|---|---|
| Round 1 | 7 | steps×layers×reuse 基础扫描 |
| Round 2 | 6 | step=4, layer=40, core-reuse/token-reduction 组合 |
| Round 3 | 11 | step=7 变体(追加到表格) |

- 去重后共 **22 个独特配置**,所有测试均 <13 GB(9.4–9.8 GB)

**关键结果**(按效率 SSIM/time 排序):

| # | 配置 | 时间 | 显存 | SSIM | 效率 |
|---|---|---|---|---|---|
| 1 | s4-l40-cr4-tr-R192 | 73s | 9.52 | 0.214 | 0.0029 |
| 20 | **s7-l50-r2** | **177s** | 9.75 | **0.570** | **0.0032** |
| 17 | s4-l50-r2 | 144s | 9.77 | 0.371 | 0.0026 |

- **关键发现**:
  1. 显存不是瓶颈(所有测试 9.4–9.8 GB)
  2. step=7 vs step=4 差异显著:s7-l50-r2(SSIM=0.570) vs s4-l50-r2(SSIM=0.371),同 layers/reuse 画质提升 54%
  3. --token-reduction 省 ~12% 时间
  4. 最快 73s(SSIM=0.214),最佳权衡 177s(SSIM=0.570)
- 报告:`/tmp/h3_benchmark/report.html`(含嵌入帧对比图)

### Phase 17: 代码审查 — complete
- 审查范围:`h3_audio_vae.c` / `h3_cli.c` / `h3_dit_schedule.c` / `h3_dit.c`(设计观察)
- 发现统计:Bug 10 项、安全 5 项、内存 4 项、正确性 4 项、性能 2 项、设计 7 项 + h3_dit.c 设计观察 3 项
- 最严重问题:
  - `h3_dit_schedule.c`:视觉/音频阈值不一致(可能导致数值错误)、`time` 张量生命周期脆弱(易引入 double-free/leak)
  - `h3_audio_vae.c`:`hidden_elements` 溢出检查缺失、`run_stage` 峰值内存
  - `h3_cli.c`:TOCTOU 竞争、SR 可执行路径注入
- h3_dit.c 设计观察(非 bug):
  - 两级决策顺序:缓存预算在 DiT 深度裁剪**之前**计算,极端设备可能缓存略不足
  - `dit_layers = 0` 哨兵值语义混合,建议 `#define H3_DIT_LAYERS_DEFAULT 0`
  - 两个分支间 `use_int8_row_fc2` 不一致
- 整体评价:代码质量高,错误处理一致,资源清理全面。主要是边界情况和防御性编程改进,无严重功能性缺陷。
- 详细发现已记录在 findings.md「代码审查发现」节。

### Phase 18: 修复代码审查高优先级问题 — complete

**已修复 5 项**(均为静默风险:正常路径行为不变,但边界/演进下可能出错):

| # | 文件:行 | 问题 | 修复 |
|---|---|---|---|
| 1 | `h3_dit_schedule.c:233` | 视觉/音频阈值不一致(visual `>=0.999f` vs audio `>=1.0f`)→ 可能导致数值行为不一致 | audio 改 `>= 0.999f` |
| 2 | `h3_dit_schedule.c`(4 处 + `failed:`) | `time` 张量在多个错误路径 free,新增 goto 易漏(leak)或重复 free | 每处 free 后 `time = NULL`;`failed:` 标签加统一 `h3_gpu_tensor_free(time)`(NULL 安全) |
| 3 | `h3_dit_schedule.c:238` | `count * feature_dim` 分配无溢出检查 → 极端 rows 可堆溢出 | 加 `count > SIZE_MAX / feature_dim` |
| 4 | `h3_audio_vae.c:612` | `hidden_elements = STEREO * length * 8` 无溢出检查 | 加 `length > SIZE_MAX / ((size_t)STEREO * 8)` |
| 5 | `h3_dit.c:2567` | `dit->use_int8_row_fc2 = dit->int8_mlp && use_int8_row_fc2` 静默吞掉调用方请求,语义不一致 | 存原始值;门控交给使用处外层 `if (dit->int8_mlp && …)`(已天然保证) |

**顺带清理(使代码可编译)**:
- `h3_gpu.m` 有重复 `stream_batch` / `stream_batch_pending` 属性声明(HEAD 遗留)→ 编译失败。删除重复项。
- 删除已废弃 GPU dequant 函数块(`blocking_weight_dequant_unrotate_int8` /
  `stream_dequant_begin|encode|submit` / `encode_wait_stream_event`)—— Phase 10 已判定该方向
  不可行,`h3_dit.c` 早已回到 CPU 蝶形。保留 `h3_gpu_linear_bf16_offset` 与 VDN 存根。

**过程中的重要教训**:
- 试图用全局字符串替换修 `h3_gpu.m` 的 `opaque`/`gpu` 参数名,误伤了 58 个正常函数签名
  (把 `h3_gpu *opaque` 改成 `h3_gpu *gpu`,与函数体内 `H3GPU *gpu = GPU(opaque)` 冲突)。
  **最终改为 `git checkout` 干净重来 + 精确删除**,一次编译通过。
  → 原则:大文件参数名重构不要用全局字符串替换;宁可 checkout 重来。
- `h3_gpu.m` 的 HEAD 版本本身不可编译(重复属性),说明仓库曾被提交在损坏状态;
  修复前先 `git stash` 验证过「原始代码同样编译失败」,确认非本次引入。

**验证**:
- 编译通过(binary 665184 bytes)
- 端到端:256×256 / 1s / steps=4 / seed 42 / `--ssd-streaming`
  → 产物 **193579 字节,与基线逐字节一致**,耗时 102s,零报错 ✅ 无回归

### Phase 18b: 清理中低优先级问题 — complete

**修复**:
- `h3_cli.c` 5 处 `strdup` 失败未检查(审查记 2 处,实际 5 处):
  - `last_prompt`(507)→ 失败后 NULL 会让 `repeat` 命令 `strdup(NULL)` **崩溃**,加检查;
    另在 918 行加 NULL 保护双保险
  - `sr_model` 初始化(852)→ 纳入启动失败检查
  - SR `bin`/`model-dir`/`model`(792/796/801)→ 静默失败改为 `fprintf` 报错
- `h3_dit_schedule.c` `weight_bf16_any()`:`elements` 累乘 + 转换缓冲区加溢出检查

**核实后判定「无需修改」**(避免无效改动):
- `h3_audio_vae.c` `run_stage` 的 `(uint32_t)elements`:上游 533-538 行已有
  `elements64 > UINT32_MAX` 检查,截断不可能发生 → 审查结论需回代码验证

**验证**:编译通过;端到端产物 **193579 字节,与基线逐字节一致**,113s,零报错 ✅

**剩余未修**(本轮已清理 run_stage / gate_score 两项,余下需重构或收益低于风险):
- `h3_cli.c` 巨型 if-else 拆分 / TOCTOU 竞争 / SR 可执行文件路径注入(需重构)
- `h3_dit.c` `dit_layers = 0` 哨兵值语义、缓存预算顺序(设计澄清)

### Phase 18c: 清理内存优化项（run_stage / gate_score）— complete
- `h3_audio_vae.c` `run_stage()`:审查建议「5→3 tensor」经逐项生命周期分析**不可行**(每个 tensor 必需);
  但发现真实峰值浪费——上一层 `audio->hidden` 在 upsample 后即不用却保留到块循环结束(峰值 6 份)。
  改为上采样后立即 submit 并释放,再分配块循环张量,**峰值 6→5 份**,命令序列与数学等价。
- `h3_dit_schedule.c/.h` + `h3_dit.c`:新增 `h3_dit_schedule_gate_scores()` 复用单个读回缓冲,
  排序路径 47 次 malloc → 1 次;单次版委托批量版,消除重复逻辑。
- 验证:端到端产物 **193579 字节,与基线逐字节一致**,105s,零报错 ✅

### Phase 18d: 审查后修复(B1 / D9 / B3 + 新并发缺陷) — complete
- `h3_video_vae.c`:删除异步 VAE 预取线程(崩溃地雷,env 可触发)→ 串行 `load→run→free`;`B2` 禁用路径泄漏随之消除
- `h3_gpu.m`:删除未实现且误导的 `stream_batch`/`stream_batch_pending`/`streamEventValue` 属性
- `h3_dit.c`:修正 `read_stream_layer` 注释(`gpu_work` 非死变量,在 `:1467/1483` 用于 int8 流式 requant)
- **新发现**:DiT 流式线程在 int8 启用时竞态共享 `dit->gpu` 命令缓冲(高·条件触发),`stream_batch` 私有缓冲本是其预留解但未实现
- 验证:端到端产物 **193579 字节一致**,120s,零报错

### Phase 18e: 修复 DiT int8 流式线程竞态(高·条件触发) — complete
- `h3_dit.c`:新增 `requant_stream_slot()` 在主线程对刚流式加载的 slot 做 int8 GPU requant;`read_stream_layer`
  (流式线程)删除全部 GPU requant 代码与 `gpu_work` 变量 → 流式线程完全不触碰共享 `dit->gpu`
- 主循环在 `pthread_join` 后调用 `requant_stream_slot`,消除两线程并发编码同一 `MTLCommandBuffer` 的竞态
- **运行时验证**:M4 上 `int8_mlp=1`(fused_mlp 默认开 + tensorOpsEnabled),`--ssd-streaming` 默认测试实际走该路径;
  修复前后产物均 **193579 字节一致** → 修复正确且已验证
- 编译通过(严格标志无警告);端到端 122s,零报错

### 5-Question Reboot Check（更新至 Phase 18e 末）
| Question | Answer |
|---|---|
| Where am I? | Phase 18 + 18b + 18c 完成:5 项高优先级 + 中低优先级(strdup/溢出) + 内存优化(run_stage 峰值 / gate_score 批量)全部修复验证通过 |
| Where am I going? | 剩余仅 h3_cli.c 需重构项(TOCTOU/路径注入/if-else 拆分)与 h3_dit.c 设计澄清(哨兵值/预算顺序),建议暂停 |
| What's the goal? | 识别并修复潜在 bug,提升代码健壮性 |
| What have I learned? | 修复后产物逐字节一致 → 确认全部无回归;审查的「5→3 tensor」需回代码验证生命周期,不可照单全收;大文件全局字符串替换风险高(曾误伤 58 个签名) |
| What have I done? | 审查 3 文件 + 记录 30+ 发现 + 修复 5 项高优先级 + 清理 5 处 strdup + 溢出防护 + run_stage 峰值 + gate_score 批量 + 清理编译阻塞 + 三轮端到端验证 |

## Session: 2026-09-11 (分支新代码审查 + 修复 + AudioVAE 端到端验证)

### Phase 19: 审查 `feature/lora-merge` 新增 C 代码并修复 6 项 — complete
审查范围:该分支相对 `origin/main` 的新增 C 代码(34 文件 / +9640 行),优先此前未审的
LoRA 合并链路与新的 `h3_superres`。`findings.md` 已记录的项(如 DiT int8 流式竞态)不重复报告。

| # | 严重度 | 位置 | 问题 | 修复 |
|---|---|---|---|---|
| 1 | 🔴 | `h3_lora.c` | 适配器名三入口不一致(`apply` 硬编码 `"default"` vs `matches`/`merge_blocking` 用文件自带 adapter)→ turbo-only 适配器**静默跳过**;且常驻/流式两路径结果不同 | `h3_lora_apply` 改传 NULL 统一语义 |
| 2 | 🔴 | `h3_lora.c` + `h3_gpu.m` | 非阻塞分支 `h3_gpu_begin` 开的缓冲从未提交(GEAM 自建私有缓冲并等待)→ 失败时 `gpu.command` 永久非空,之后**所有** GPU 阶段失败 | 删除 begin/submit 包装 |
| 3 | 🟡 | `h3_gpu.m` | `h3_gpu_lora_geam_bf16` 与 `..._blocking_...` 函数体逐行等价 | 改为转发,单一实现 |
| 4 | 🟡 | `h3_lora.c` | `rank*in_dim` 等分配大小无溢出检查 | 加 `SIZE_MAX` 防护(含除零) |
| 5 | 🟡 | `h3_lora.c` | `rank` 在 `dtype/ndim` 校验前读取 | 调整顺序并拒绝 `rank==0` |
| 6 | 🟡 | `h3_ffmpeg.c` | SR 临时目录清理 best-effort(`rm` 走 PATH、失败静默)、`waitpid` 失败不回收子进程、路径 `snprintf` 截断未检查 | `remove_tree()` + SIGKILL 回收 + 截断检查 |

- 核实为**非缺陷**(避免无效改动):`h3_superres` 的 `target_height % inner_h` **不可达除零**
  (`h3_ffprobe_visual_size` 成功前强制 `parsed_width/height >= 1`,`h3_ffmpeg.c:144-148`)。
- 验证:`make -j8 h3 h3_lora_tests` 零新增警告;`h3_tests` 1768 checks、`h3_lora_tests`
  qkv/out/fc1/fc2 rel-L2 **0.002844**(与修复前一致)、`h3_audio_gpu_tests` 全绿。

### Phase 20: 修复编译警告暴露的既有缺陷 — complete
| 位置 | 问题 | 修复 |
|---|---|---|
| `h3_audio_vae.c::run_stage` | `int ok = …` 在 `goto done` 之后 → 分配失败时 `return` 未初始化值(`-Wsometimes-uninitialized`) | `int ok = 0;` 提到失败分支前;顺带对齐因此位移的 5 行续行缩进 |
| `h3_audio_vae.c::decode_output` | `audio->length` 是 `uint32_t`,与 `SIZE_MAX/(STEREO*8)` 比较 64 位下恒为假 → 溢出检查**实际无效** | 先按 `uint64_t` 计算再校验转 `size_t`(与 `run_stage` 既有写法一致) |
| `tests/test_lora.c` | `compare()` 未使用的 `gpu` 形参;`printf("%zu", lora ? 0 : 0)` 格式不匹配且恒打印 0 | 删形参 + 4 处调用点;删除该调试输出 |

- 验证:全量 `make -j8 all test` **0 warnings / 0 errors**;三套件全绿,数值不变。

### Phase 21: 用官方权重端到端验证 AudioVAE + 修复失效断言 — complete
- 用户指明权重在 `@models/minimax-h3/FL2VA`(符号链接到 `/Users/jay/h3_sys/MiniMax-H3-Convrot/…`)。
- `audio_vae/model.safetensors`:577 MB / **1087 张量全 F32** → 非 int8/convrot 变体,不会引入额外反量化提交。
- 临时 harness(直连 `libh3.a`,合成 latent)实跑结果:

| 项 | 值 |
|---|---|
| 形状 | 2ch / 29600 samples @ 32000 Hz(latent 37)|
| 统计 | peak 0.333697,rms 0.0532564,全 finite |
| 资源 | 0.543 GiB,0.053 GPU s |
| 结构 | 136 MPS conv,**23** submissions |
| 确定性 | 两次解码**逐字节一致** |
| 边界 | latent 1 → 800 samples、latent 2 → 1600 samples |

- **发现失效断言**:`tests/test_real_audio_vae.c` 期望 `submissions == 16`,实跑 **23**。
  推导:16 = 1 `submit AudioVAE input` + 7 `stage normalization` + 7 `stage` + 1 `output`(STAGES=7);
  Phase 18c 的「上采样后立即 submit 并释放上一层 hidden」每 stage +1 → **7 + 16 = 23**(conv 数 136 不变)。
  → 修正断言为 23 并把推导写进注释。
- **仍未验证**:与参考 oracle 的波形数值 parity —— 缺 `misc/fixtures/h3_real_audio_vae_37.safetensors`;
  刻意**未**用 native 输出反造 fixture(会使测试退化为自证)。

### Phase 22: 新增不依赖 fixture 的 AudioVAE 端到端测试 — complete
- 新增 `tests/test_real_audio_vae_e2e.c`:输出几何、全 finite、确定性(两次逐字节)、
  分发结构(136 convs / 23 submissions,含推导注释)、最短合法 latent(1/2 帧,覆盖 `input_length-1==0`)。
- `Makefile`:目标 `h3_real_audio_vae_e2e_test`、变量 `AUDIO_VAE_MODEL ?= MiniMax-H3`、
  接入 `make test`(仅需权重)、加入 `clean`;`AGENTS.md` 登记该入口。
- 三条路径实测:错误模型路径 → **exit 1**(测试确实会失败);默认 `make test` → skip;
  `make test AUDIO_VAE_MODEL=models/minimax-h3` → 套件内实际执行并通过。

### Phase 23: 修复流式 video VAE 解码命令缓冲 bug — complete
- 复现:ComfyUI `H3_BinaryT2V` 报 rc=1,日志 `audio VAE 7/7` 后
  `begin streamed video VAE transformer block: unknown Metal error`;直连官方权重 `./h3` 同样复现。
- 根因:`h3_gpu_submit` 提交后 `gpu.command` 置 nil 且不重新打开(重新打开只在 `h3_gpu_continue`);
  `h3_gpu_begin` 在已开缓冲上直接 `return 0` 且**不设 lastError** → 错误串回退成 "unknown Metal error"。
  `run_stream_tile` 在 prep 后缺一次 submit,导致循环首个 `h3_gpu_begin` 撞上已开缓冲而失败(此前改动未碰 video VAE,为既有 bug)。
- 修复:镜像 `run_decoder` 的每阶段 begin/submit——prep 单独 submit、循环恢复每 block 的 begin+submit、post 前补 begin。
- 验证:直连 `./h3` 跑通(448×256,1s,4 steps),`audio VAE 7/7 → FFmpeg 39/39 → wrote /tmp/h3_verify.mp4`
  (270 KB;ffprobe:448×256,39 帧,1.625s,含音频流);日志无 Metal/error;全量 `make -j8 all test` 0 警告,三套件全绿。

### 运行清单(本 session)
| # | 命令 | 结果 |
|---|---|---|
| 1 | `make -j8 h3 h3_lora_tests` | 0 warning / 0 error |
| 2 | `./h3_tests` / `./h3_lora_tests` / `./h3_audio_gpu_tests` | 1768 checks / rel-L2 0.002844 / 全绿 |
| 3 | `./h3_real_audio_vae_e2e_test models/minimax-h3` | PASS(见 Phase 21 表)|
| 4 | `./h3_real_audio_vae_e2e_test /tmp/definitely-not-a-model` | exit 1(负例) |
| 5 | `make test`(默认) | 新测试 skip: `released AudioVAE weights are not installed` |
| 6 | `make -j8 all test` | 0 warnings / 0 errors,可运行套件全绿 |

### Error Log(本 session)
| Error | Attempt | Resolution |
|---|---|---|
| 临时 harness 里误加 `(void)next_value;`(其实在用) | 1 | 直接重写该文件;临时程序也按编译告警零容忍处理 |
| harness 断言 `submissions == 16` 失败 | 1 | 逐项推导出 16 + 7 = 23,根因是测试断言在 Phase 18c 后未同步(非本次改动)→ 修正测试 |
| `make test` 中 AudioVAE 真实权重条目一直 skip | 1 | 守卫查 `MiniMax-H3/…`,实际在 `models/minimax-h3/…` → 新增 `AUDIO_VAE_MODEL` 并让新测试可用 |

### 5-Question Reboot Check(更新至 Phase 22 末)
| Question | Answer |
|---|---|
| Where am I? | Phase 19–22 完成:分支新代码审查并修复 6 项 + 既有 2 缺陷 + AudioVAE 官方权重端到端验证 + 新增不依赖 fixture 的 e2e 测试 |
| Where am I going? | 可选:把 `h3_real_audio_vae_test` 的守卫路径也参数化到 `AUDIO_VAE_MODEL`;补 `misc/fixtures` oracle 后可验证数值 parity;`h3_cli.c` 重构项与 `h3_dit.c` 设计澄清仍挂起 |
| What's the goal? | 收敛分支新增代码的正确性/资源管理风险,并让 AudioVAE 在没有 oracle fixture 的机器上也能被验证 |
| What have I learned? | 同一 API 的多入口必须同源解析(adapter 三处不一致导致静默跳过);断言里的魔法数字必须给出推导,否则优化后必然失效(16→23);审查怀疑项要回代码验证(`target_height % inner_h` 不可达、`enc_argv` 无越界) |
| What have I done? | 审查并修复 6+2 项 + 官方权重端到端实跑 + 修正失效断言 + 新增 `tests/test_real_audio_vae_e2e.c` 与 Makefile/AGENTS 接线(全量 0 警告) |

## Session: 2026-09-12 (Metal 上下文编译缓存 — DeepJIT 启发)

### Phase 27: 基线测量 + 计时插桩 — complete
- [x] 调研落盘：见 findings.md「Metal 编译缓存调研」
- [x] `h3_gpu_create` 加 `H3_PROFILE` 计时：`library=`（源码→MTLLibrary）+ `pipelines=`（86 个构建耗时）
- [x] 新增 `tests/bench_metal_context.c` + `make h3_metal_bench` 目标
- [x] 记录基线数字：

| 场景 | library | pipelines | 单次合计 |
|---|---|---|---|
| 暖缓存 | 0.001–0.003 s | 0.001–0.005 s | **~5 ms** |
| 冷缓存（源码内容改动） | **0.226 s** | 0.002 s | ~0.23 s |
| 同「冷」源码再跑 | 0.001 s | 0.002 s | ~3 ms |

- [x] 判决性对照：`cp h3_shaders.metal /tmp/… && 追加一行注释` 制造缓存 miss →
  证明暖缓存的 3 ms 来自**系统 shader 缓存**（`/private/var/folders/*/C/com.apple.metal`）

### Phase 28–30: 三项改动 — **全部取消（有实测依据）**
- [x] 真实引擎跑（`H3_PROFILE=1`，256×256 / 0.5s / steps 2）：一次 T2V 只建 **4 个上下文**，
      shader-build 合计 **23 ms**；同 run 的 DiT 主体 wall = **34.680 s**
- [x] 判断：上限收益 ≈10–15 ms/次（0.04%）→ **不写任何缓存代码**
- [x] 根因：Metal 有系统级按内容命中的 shader 缓存；DeepJIT 的磁盘缓存经验来自 CUDA
      （无此机制），**不可移植**

### Phase 31: 汇总收益 + 收尾 — complete（结论为「不做」）
- [x] 保留 `H3_PROFILE` 的 shader-build 计时（诊断工具，零风险）
- [x] 保留 `tests/bench_metal_context.c` + `h3_metal_bench`（可复测）
- [ ] 待确认：是否把该结论写进 README 的 performance notes（避免后人重复踩坑）

### 运行清单（本 session）
| # | 命令 | 结果 |
|---|---|---|
| 1 | `make -j8 h3_metal_bench` | 0 warning / 0 error |
| 2 | `H3_PROFILE=1 ./h3_metal_bench 5` | 5 个上下文，合计 0.077 s |
| 3 | 同上，换成内容改动的 `/tmp/h3_shaders_cold.metal` | 首个 library **0.226 s**（缓存 miss） |
| 4 | 再跑同一冷源码 | 0.001 s → 确认系统缓存按内容命中 |
| 5 | `H3_PROFILE=1 ./h3 -d models/minimax-h3 -p "…" 256×256/0.5s/2steps` | 4 个上下文，shader-build 合计 23 ms |

### Error Log（本 session）
| Error | Attempt | Resolution |
|---|---|---|
| 首次真实引擎探针跑成交互模式（日志仅 `Goodbye.`） | 1 | `nohup ./h3 -d …` 漏 `-p`；补提示词后正常 |
| 初始假设「编译 280KB/180 kernel 需数秒」 | — | 被实测推翻（暖 3 ms / 冷 226 ms）。教训：先确认测量对象 |

## Session: 2026-09-12 (换目标：DiT 流式权重管线读/算重叠)

### Phase 32: 相位计时插桩 — complete
- [x] `h3_dit.c`：`h3_dit` 加 `stream_pread_seconds` / `stream_dequant_seconds`；
      `read_stream_layer` 分别累加「文件读（含 scale）」与「CPU 反旋转 + 写 slot」
- [x] teardown 打印扩展为 `..., unhidden wait %.3fs, pread %.3fs, cpu-unrotate %.3fs`
- [x] 基线实测（256×256 / 0.5s / steps=2，自动规划走 SSD 流式）：

| 量 | 值 |
|---|---|
| Euler denoise wall | **33.189 s** |
| wait (GPU) | 13.178 s |
| root-gpu | 2.909 s |
| encode | 0.064 s |
| **unhidden wait** | **19.945 s**（60% 的 wall） |
| stream pread | 17.568 s |
| stream cpu-unrotate | 16.000 s |

- [x] **关键等式**：`17.568 + 16.000 = 33.568 s == 该线程总忙碌时间` → 读盘与 CPU **零重叠**

### 支撑证据（本次调研）
- `sample` 抓取：主线程 61% 在 `pthread_join`、39% 在 `waitUntilCompleted`；
  流式线程 ~46% 在 `pread`、~46% 在 `read_stream_layer`（内联的 WHT 蝶形）、~6% `memmove`
- 10.16s 采样窗口内抓到 **~37 个 `read_stream_layer_thread`** → 每 block 起一个线程、串行 join
- 权重文件 `minimax_h3_fastvideo_4step.safetensors` = 21.33 GiB，其中 **I8 19.74 GiB**
  （250×I8 + 250×U8 comfy_quant + F32 scale）→ 已是 convrot int8，"减字节"这条路没有现成空间
- 权重在内置盘 `/dev/disk3s5`；`disk_speed` 实测该卷顺读 **2158 MiB/s**，
  引擎 pread 段 2.06 GiB/s → **读盘已贴上限**

### Phase 33–34: 按 source 分片并行 + 验证 — complete
- [x] `h3_dit.c`：`read_stream_layer` 拆为 `read_stream_sources(job, indices, count)` + 编排器；
      新增 `h3_dit_stream_worker`（自带 job 计数器 + source 索引表）
- [x] 默认 2 个 worker 并发（连续索引区间）；`H3_DIT_STREAM_WORKERS=1` 回到原串行参考路径
- [x] 并发安全性：每个 source 写不同 slot 张量，写域不相交；每 worker 独立 job，无共享可变状态；
      LoRA 合并与 `job->seconds` 留在 join 之后的编排器里
- [x] **A/B 结果（产物均逐字节一致）**：

| 量 | W=1（串行 = 基线） | W=2（默认） | 变化 |
|---|---|---|---|
| Euler denoise wall | 33.296 s | **22.686 s** | **−31.9%（1.47×）** |
| unhidden wait | 20.201 s | 9.223 s | −54% |
| 有效吞吐 | 1.076 GiB/s | 1.577 GiB/s | +47% |

- [x] `make -j8 test AUDIO_VAE_MODEL=models/minimax-h3` → exit 0，可跑套件全绿；构建 0 警告
- [x] 被实测否决并回退：按字节 LPT 均衡（23.15s，无收益，破坏顺序局部性）、4 workers（25.19s，更慢）

### Phase 35: worker 内 SPSC 两级流水 — complete
- [x] `h3_dit.c`：新增 `h3_stream_ring`（2 槽 SPSC，槽位 = `position % 2`，握手靠 `filled` 标志）；
      `stream_fill_slot()`（reader 线程：pread int8 + scale）/ `stream_consume_slot()`
      （dequantizer：WHT 反旋转 + 写 slot）/ `run_stream_pipeline(worker, pipelined)`
- [x] 旋钮 `H3_DIT_STREAM_PIPELINE=0`：两阶段在同一线程背靠背跑（复用同一份代码，无重复实现）
- [x] **2×2 对照（同一 binary，产物全部逐字节一致）**：

| 配置 | denoise wall | unhidden wait |
|---|---|---|
| W=1, P=0（原始串行参考） | 35.131 s | 22.101 s |
| W=1, P=1 | 25.189 s | 11.546 s |
| W=2, P=0（Phase 33） | 23.586 s | 10.239 s |
| **W=2, P=1（默认）** | **21.339 s** | **7.957 s** |

- [x] 相对最初基线（33.296 s）累计 **1.56×**；每 block 337ms → 217ms；吞吐 1.08 → 1.67 GiB/s
- [x] `make -j8 test AUDIO_VAE_MODEL=models/minimax-h3` → exit 0；构建 0 警告

### Phase 36: 继续压（直写 slot / NEON / 分块）— complete
- [x] **36a 反旋转直写 slot 张量**（新增 `h3_gpu_tensor_bf16_storage()`）：
      `cpu-unrotate` 19.57 → 16.96s；denoise 21.339 → **21.060s**；逐字节一致；顺带去掉 230MB/次瞬时分配
- [x] **36b NEON 重写反旋转 → 实测 0.84×（更慢），否决，未进仓库**
      - 过程中修掉真 bug：`vcgeq_u32` 返回全 1 掩码而非 1 → `hi+mask` = `hi-1` → bf16 差 2
      - 修正后 `ALL BIT-IDENTICAL`，但 scalar 8259 MB/s vs neon 6973 MB/s
      - 根因：`-O3` 已把标量版自动向量化（实测 ~14 ops/cycle），手写 intrinsics 多付解交错 shuffle
- [x] **36c 流水 stage 粒度 矩阵 → 1024 行 chunk**：denoise 21.060 → **19.962s**；
      staging 115MB → 5.5MB/worker；逐字节一致
- [x] per-source scale 缓存（避免每 chunk 重读整个 scale 张量）；**实测中性**，按潜在退化防护保留
- [x] 验收：`make test` exit 0；`PIPELINE=0` 不挂死、逐字节一致（22.55s）

### 本任务累计结果（256×256 / 0.5s / steps=2，全部逐字节一致）
| 阶段 | denoise wall | 累计 |
|---|---|---|
| 原始基线 | 33.296 s | 1.00× |
| + 按 source 分片并行 | 23.586 s | 1.41× |
| + worker 内两级流水 | 21.339 s | 1.56× |
| + 反旋转直写 slot | 21.060 s | 1.58× |
| **+ 1024 行分块（最终默认）** | **19.962 s** | **1.67×** |

### Phase 37: 终点 — 已贴磁盘地板
每 block 每 worker：read 173.5ms（2 reader 合计 2.22 GB/s ≈ 该卷实测上限 2.16 GiB/s）、
dequant 87ms → 理想 173.5ms → 去噪 ≈17.4s；当前 199.6ms，差 15% 且已分散（condvar 交接 +
内存带宽争抢），无单一主导项。**再快只能少读字节**（跨 step 常驻权重），属内存规划器范畴。

### Phase 38: 逐分辨率 A/B（`git stash` 原始二进制对照）— complete
**结论：优化只在低分辨率有效；引擎默认配置（864×480）收益为 0。**

| 分辨率 | 原始 denoise | 优化 denoise | 加速 | 原始 unhidden wait |
|---|---|---|---|---|
| 256×256 | 33.296 s | 19.962 s | **1.67×** | 19.945 s |
| 384×384 | 33.922 s | 26.746 s | **1.27×** | 7.561 s |
| 512×512 | 46.441 s | 47.454 s | **0.98×** | **0.001 s** |
| 864×480 | 78.869 s | 77.579 s | **1.02×** | **0.001 s** |

- 流式耗时与分辨率无关：原始 34.084/35.034/35.337 s → 优化 20.032/19.900/20.332 s（稳定 −42%）
- **≥512×512 时流式在原始代码里就已被完全隐藏**（unhidden wait 0.001s）→ 整链收益 0
- 三档产物均**逐字节一致**；交叉点 ≈448×448
- 方法论：`cp h3 /tmp/h3_opt` → `git stash push -- <4 文件>` → 编原始 → `cp /tmp/h3_orig`
  → `git stash pop` → 干净重建。**注意**：stash 期间编出的 `h3_gpu.o` 在 pop 后会变成
  陈旧对象（缺新符号）导致链接失败，需 `make clean` 全量重建

### Phase 39: 决策 = A（保留 + 默认关闭）— complete
用户选择 A：保留代码、默认关闭、并给出环境变量与 ComfyUI 一键开关。

- [x] `h3_dit.c` 翻转默认：`workers=1`、`pipelined = workers>1`。
      **`H3_DIT_STREAM_WORKERS=2` 一个变量开启整条快路径**；`H3_DIT_STREAM_PIPELINE`
      单独覆盖读/算重叠（=0 保留分片去掉 reader 线程，=1 只开流水）
- [x] **把「分块」和「反旋转直写 slot」也挂到同一开关**（原先漏挂，导致默认态仍非原始）：
      - 分块：256 时 −16.5%，512 时 **+3.7%**（只多 syscall）→ 默认改为「整矩阵一块」
      - 直写：256 时 −13% CPU，512 时 **+1.2~2.8%**（QKV 乱序行写共享内存 vs
        私有页顺序写 + memcpy）→ 默认改回暂存缓冲 + 全量拷贝
- [x] 默认态验证 = 原始：512×512 背靠背 **47.588s vs 原始 47.653s（−0.1%）**，逐字节一致
- [x] 快路径验证：`H3_DIT_STREAM_WORKERS=2` @256×256 = **19.760s（1.70×）**，逐字节一致
- [x] ComfyUI 节点 `comfyui_nodes/h3_binary.py`：新增 `fast_stream` BOOLEAN（默认关），
      经 `_engine_env()` 注入 `H3_DIT_STREAM_WORKERS=2`；并在 ≤384×384 且 ≤1.0s 时
      自动提示可开启。`py_compile` 通过，两个节点均已暴露该参数
- [x] README performance notes 新增「Streamed DiT weight prefetch (opt-in)」小节
      （含开关、四条分辨率实测表、交叉点随 GPU/磁盘比移动的说明）
- [ ] 待办：ComfyUI 侧的**部署副本** `/Volumes/data/Documents/ComfyUI/custom_nodes/h3_binary_nodes/h3_binary.py`
      仍是旧版（仓库外，需用户确认再同步）

### 最终交付（本任务）
| 项 | 状态 |
|---|---|
| 代码 | `h3_dit.c` / `h3_gpu.h` / `h3_gpu.m`，0 警告 |
| 默认行为 | 与改动前**逐字节一致**且耗时相同（512×512 −0.1%） |
| 一键加速 | `H3_DIT_STREAM_WORKERS=2` → 256×256 **1.70×** / 384×384 **1.27×** |
| ComfyUI | 新增 `fast_stream` 开关 + 自动提示 |
| 文档 | README 新增小节；planning 三件套已同步 |
