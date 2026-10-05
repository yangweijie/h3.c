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

---

# Session 2026-09-12：memory-plan 未接线字段

## 起因
用户问「除了 denoise 和文本编解码还有哪些步骤可优化」→ 逐阶段计时 →
发现 `--video-vae-streaming` 无独立开关 → 加开关做 A/B →
逐字段核对 planner 输出时发现 2 个死字段。

### 本 session 已完成的旁支工作
- [x] `gen_comfyui_workflows.py`：补 `fast_stream` + `control_after_generate`，
      修 widgets_values 错位（这是 ComfyUI 那次 FileNotFoundError 的根因）
- [x] 校验加强：按名字锚定 `binary` / `model_dir` 槽位，错位直接 FATAL
- [x] `_common_required()`：`lora` 从 optional 移入（optional 会被前端折叠成 `shape:7` 看不见）；
      顺序不变，旧工作流仍对位
- [x] `_fast_stream_hint`：去掉 `seconds <= 1.0`，改为纯分辨率判据
- [x] 新增 `--video-vae-streaming 0|1`（三态 `-1`=auto），planner 尊重显式设置
- [x] ComfyUI 部署副本确认与仓库**同一 inode**（硬链接，无需同步）

### 阶段计时基线（256×256 / 2 s / 4 步 / seed 42 / M4 16 GB）
| 配置 | 总耗时 | DiT denoise | VAE 解码 | 备注 |
|---|---|---|---|---|
| 默认（planner，VAE 流式） | 126.2 / 112.5 s | 82.1 / 72.1 s | 31.5 / 29.3 s | 两次（有/无干扰） |
| `--ssd-streaming` | 108.7 s | 74.1 s | 22.1 s | |
| `--video-vae-streaming 0` | 104.7 s | 73.2 s | 19.8 s | 干净环境；有干扰时 123.7 s（历史偶发 OOM 实为统一内存被其他进程挤占，VAE 常驻权重可驱逐、峰值仅 ~6 GiB，非 9 GiB 钉死） |
| 带 turbo LoRA（4 步） | 166.3 s | 122.1 s | 31.3 s | LoRA 在流式路径阻塞 merge，+40.1 s |

关键：三种成功的产物 md5 全为 `0d353faf2173acaab0badd4f9af62bf4`（逐字节一致）。

## 当前任务：死字段处置
### Phase 1 调查 — complete
- [x] `encoder_streaming` / `cache_budget_bytes` 均由 `d5752a0`（2026-08-28）引入
- [x] 证明**从未接线**（`git log -S` 对读取侧全历史零结果），不是撤回
- [x] `encoder_streaming`：意图（释放 encoder）已由「一次性编码函数」天然实现 → 冗余
- [x] `cache_budget_bytes`：意图未实现，且指向 denoise 最大瓶颈（每步重流 50 块）→ 有研究价值

### Phase 2 代理实验 — complete
- [x] `--layers 40`：总 100.5 s、denoise 58.84 s（−18.4%）、SSD 读 57.781 GiB（−19.9%）→ 线性
- [x] 推出关键路径模型 `denoise ≈ (pread + cpu-unrotate) × 0.98`，GPU 仅 14 s/72 s
      → 读取链完全在关键路径上，缓存的收益按 `× (1 − K/50)` 计算
- [x] 估算：8.0 GiB 常驻 22 块 → denoise 72 → ~41 s → 总时间 −28%

### Phase 3 删除 encoder_streaming — complete
- [x] 6 个文件移除（h3.h / h3_memory_plan.h / h3_memory_plan.c ×2 / h3.c / h3_cli.c / AGENTS.md）
- [x] 无 tests 引用；代码零残留；编译通过
- [x] 端到端 117.1 s、md5 `0d353faf2173acaab0badd4f9af62bf4` —— 与删除前逐字节一致

### Phase 4 cache_budget_bytes — 待用户决策
范围已从「删死代码」升级为「实现流式权重常驻缓存」（详见 task_plan.md 的 Phase 4）。

---

# Session 2026-09-12 续：部分常驻缓存实现（选项 A）

## 实现（`H3_DIT_RESIDENT_BLOCKS`，env-only，默认 0 = 关闭）
| 位置 | 改动 |
|---|---|
| `h3_dit` struct | `resident_blocks` + `block_resident[H3_DIT_BLOCKS]` |
| `load_dit` | 解析 `H3_DIT_RESIDENT_BLOCKS`（0..50，越界 fail） |
| `load_core` | clamp 到 `active-2`；按前缀标记常驻；常驻块也走 `load_block_norms` |
| `load_core` 尾部 | 常驻块用 `read_stream_layer` + `requant_stream_slot` 灌进 `blocks[]`，再 `adopt_slot_weights` |
| 新增 | `first_streamed_block` / `next_streamed_block` / `resident_needs_bf16` / `adopt_slot_weights` |
| 主循环 | `if (ssd_streaming && !block_resident[block])`；预取目标改用 `next_streamed_block` |
| 删除 | `next_active_block`（因改动而孤立） |

## 关键教训：走 `load_block` 会改变数值
第一版让常驻块走 `load_block`（`bf2_convrot`），结果 **md5 变了**
（`82a84345…` ≠ 基线 `0d353faf…`）—— 与流式块的 `pread + convrot_unrotate_cpu`
不是同一条权重来源。改成让常驻块也走流式读取路径后，md5 恢复一致。

## 实测（256×256 / 2 s / 4 步 / seed 42，同机同种子）
| 配置 | 总耗时 | denoise | 读取量 | pread+unrotate | unhidden wait | md5 |
|---|---|---|---|---|---|---|
| K=0（基线） | 112.5~119.1 s | 72.11~73.67 s | 72.136 GiB | 73.53 s | 7.25 s | `0d353faf…` |
| K=4 | 117.3~119.1 s | 73.27~73.67 s | **67.830 GiB** | 70.52 s | 6.47 s | `0d353faf…` ✓ |

- 读取量精确按比例下降（188 次块读取 = 4 + 4×46）✓
- **但时间没有下降**：读取链短了 3.0 s，denoise 反而 +1.2 s（噪声内）
- 常驻块实际占 **0.72 GiB/块**（peak 1.646→4.517 GiB），即只保留了 bf16 ——
  该基座的 `int8_qkv/mlp/attention_out` 未启用，`resident_needs_bf16` 判定为真

## 最终结论：A + C（用户选定）
| 项 | 处置 |
|---|---|
| `H3_DIT_RESIDENT_BLOCKS`（部分常驻） | **保留**为 env-only opt-in（默认 0，零影响）+ README 新小节 |
| `cache_budget_bytes` + `h3_memory_cache_budget_bytes()` | **删除**（含 planner reason 里的 "cache X GiB"、AGENTS.md 引用） |

**判定依据**：K=8 读取量降 12%（72.136 → 63.523 GiB）、读取链短 5.6 s，denoise 却
+1.9 s（74.02 s vs 72.11 s）。磁盘 I/O 不是 denoise 瓶颈；16 GB 机器上常驻块
与流式路径争同一片统一内存，节省被抵消。产物全程逐字节一致（`0d353faf…`）。

**为什么不自动接线**：`cache_budget` = `7/8 × working_set − steady` = 8.0 GiB，
换算 K≈11 会比 K=8 更慢 —— 公式过于乐观，未计入内存压力下的退化。

**验证**：编译 0 warning；`cache_budget` 代码零残留；K=0 与删除后端到端 md5
均为 `0d353faf2173acaab0badd4f9af62bf4`；planner 日志不再出现误导性的 cache 数值。

## Errors Encountered（本 session）
| Error | Attempt | Resolution |
|---|---|---|
| ComfyUI 节点 `binary` 收到模型目录路径 | 1 | 定位为 widgets_values 整体错位（缺 `fast_stream`/`control_after_generate`），修生成器并重生成 |
| `--video-vae-streaming 0` 进程 rc=-9 | 1 | 非代码问题：用户同时开浏览器导致内存不足；干净环境重跑 104.7 s 成功 |

# ===== 本 session: H3 latent 子系统（ComfyUI）往返保留音频 =====

## 起因
用户指出：官方工作流能「视频 VAE + 音频 VAE 并行 → CreateVideo」保留音频，
而我们的 `H3_BinaryLatent` 只 dump 视频 latent → 走 latent 往返会**丢音频**。
该判断经核对为**正确**：`--latent-out` 原实现只写 `video`，`audio` 缓冲从未落盘；
`H3_BinaryLatentDecode` 也只做视频 VAE 解码并写静音轨。

## 用户选定方案
**(i) 空间上采样 + 音频透传保留音频**（另两选项：接受无声分支 / 连时间维也上采样）。

## 改动清单
| 文件 | 改动 |
|---|---|
| `h3.c` | `write_video_latent`→`write_latent_bundle`（写视频+音频 latent）；`read_video_latent`→`read_latent_bundle`（含 version 校验）；`h3_decode_latent` 末尾改「有音频 latent 则解音频并 mux，否则静音」 |
| `comfyui_nodes/h3_binary.py` | `_read/_write_h3_latent` 升级 v2 携带 `h3_audio`；`H3_BinaryLatentUpscale` 透传 `h3_audio`；`H3_BinaryLatentDecode` 写回带音频；更新两处 docstring + 1 处 DESCRIPTION |

## 验证（真机，256×256 / 2s / 20 步，模型 MiniMax-H3-Convrot）
1. 生成 + `--latent-out`：日志 `latent 17/17`、`audio VAE 7/7`，写出 `/tmp/h3_out.mp4`。
2. `ffprobe` 原片：`0,video` + `1,audio` ✓
3. `--latent-in` 回解：写出 `/tmp/h3_dec.mp4`，`ffprobe` 同样是 `0,video` + `1,audio`。
4. **决定性验证**（排除静音兜底）：抽取两份音轨为 f32le wav →
   **字节数 598108 相同、MD5 完全一致 `56ab82615d7b9054464f6bb13cd1187d`**；
   解码音频最大振幅 **1.005 > 0**（非静音）。→ 音频 latent 被逐字节带回。
5. 真实节点链（anaconda python 调 `H3_BinaryLatentUpscale`）：
   - 读回 `samples (1,24,17,16,16)` + `h3_audio (32,2,93)`
   - factor=2 `trilinear` 上采样 → `samples (1,24,17,32,32)`、`h3_audio (32,2,93)` **未动** ✓
   - 回解 → `0,video,512,512` + `1,audio`，解码音频振幅 1.005（音频透传）✓

## 结论
方案 (i) 落地完成：**空间上采样不影响音轨，latent 往返保留 H3 原生音频**。

## 已知限制
- latent 格式 v2 与改动前生成的旧 `.bin` 不兼容（中间产物，无影响）。
- 音频 VAE 路径硬编码 `FL2VA/audio_vae` —— 与 latent 节点的 FL2VA(T2V) 定位一致；
  若将来做 R2V latent 需按 `ref2va` 切换为 `Ref2VA/audio_vae`。

## Errors Encountered（本 session）
（无）

---

# Session: int4 量化路线 kickoff（2026-09-13）

## 本 session 做了什么（按时间）

1. **三项目对比分析**（FastMetal-1.3B-QAD / FastVideo / h3c）→ 产出瓶颈定位与 QAD 可行性分析。
2. **执行 §4.4-1**：修 `fastvideo_qad/scripts/mlx_int8_to_h3.py` 的 **5 个**缺陷并重导出两份 checkpoint。
3. **执行 §4.4-2**：端到端质量 A/B（2 步 / 4 步），并修正度量语义（新增画质代理与判定）。
4. **启动 int4 路线**：建立本任务的规划文件（task_plan / findings / progress）。

## 关键数字

| 项 | 值 |
|---|---|
| 引擎二进制 | `/Volumes/data/git/c/h3c/h3`，665616 B，`xattr` 隔离属性已清除 |
| 源 DiT | `minimax_h3_fastvideo_4step.safetensors` 21.33 GiB（I8 19.74 + BF16 1.49 + F16 0.08 + F32 0.018 + U8 250×72B）；1086 张量 |
| 单块流式 | 4 个矩阵 = 0.359 GiB（50 块/步 = 17.95 GiB/步） |
| / 每步 20 步 | ~359 GiB/次生成；内盘 2.06 GiB/s → 纯 I/O 地板 ~174 s |
| 256²/2 步 denoise | base 56.4 s；int8-g64 85.7 s；int4-g64 94.7 s（后两者在外接盘，受 0.85 GB/s 拖累） |
| 256²/4 步 denoise | base 91.4 s；int8-g64 148.9 s；int4-g64 149.1 s |
| base 画面统计 | luma mean 0.665 / std 0.311（非糊图，可判读） |

## 产物（本轮落盘）

| 路径 | 内容 |
|---|---|
| `/Volumes/data/work/h3_qad/h3_int8g64` | 21.4 GiB / 11 分片；relRMS mean 0.00518 max 0.00635 |
| `/Volumes/data/work/h3_qad/h3_int4g64` | 21.4 GiB / 11 分片；relRMS mean 0.09127 max 0.09170 |
| `/Volumes/data/work/h3_qad/ab/{base,int8g64,int4g64}` | 三个单变量对照模型目录（除 `FL2VA/transformer` 外全部同源符号链接） |
| `/Volumes/data/work/h3_qad/ab_report/` | 2 步/4 步的 mp4（6 个）、拼图 2 张、report json 2 份 |
| `fastvideo_qad/scripts/mlx_int8_to_h3.py` | 重写（修 5 缺陷 + 非自洽校验） |
| `fastvideo_qad/scripts/ab_quant_quality.py` | 新增（divergence + quality proxies + verdict） |

残留 `_steps4` 验证用的中间帧：`/tmp/h3_ab/f_*.png`、`/tmp/h3_ab4/f_*.png`（可删）。

## 验收记录（本 session）

| 验收项 | 结果 |
|---|---|
| int8-g64 权重 relRMS | 0.0047–0.0064（预检与整份导出一致）✓ |
| int4-g64 权重 relRMS | 0.0911–0.0917 ✓ |
| QKV 行序校验 | stored 0.0047 vs interleaved 1.404（有判别力）✓ |
| 2 步 A/B | int8 `preserved`（SSIM 0.891 / ccos 0.989）、int4 `degrades (grainy)`（0.521） |
| 4 步 A/B | int8 `preserved`（SSIM 0.739，画质代理持平）、int4 `degrades (desaturated)`（0.598） |
| 脚本编译 / lint | `py_compile` 通过、0 diagnostics ✓ |

## 结论

- **int8 affine group-64 可用**；**int4 group-64 纯 PTQ 不可接受**（需要 QAT）。
- int4 在这些评测里走的是**代理路径**（重量化回 per-row int8）——引擎尚无 int4 加载。
- 本机 int4 只省 I/O、不省计算；默认 864×480 下 I/O 已被掩盖 → **int4 的收益只在 ≤384² 预览与 M5+/未来硬件**。

## Next

Phase 1a：{4,6,8} bit × {32,64,128} group 的权重 relRMS 曲线（秒级、无副作用），
据此筛出值得进端到端 AB 的组合。见 `task_plan.md` 的 Phases。

---

# Session: int4 路线 Phase 1a / 1a' / 1a'' / 1c（2026-09-13 续）

## 新增脚本
| 脚本 | 作用 |
|---|---|
| `fastvideo_qad/scripts/quant_scheme_sweep.py` | Phase 1a：位宽 × group × 对称性的**纯权重误差**扫描（88 s，无副作用） |
| `fastvideo_qad/scripts/quantize_h3_proxy.py` | Phase 1：把任意方案落到 h3 格式代理 checkpoint（直接从源量化，其余张量逐字节复制，支持 `--override` 逐矩阵配置） |

## 本轮实验（4 个 checkpoint + 12 次引擎运行）

| 变体 | 权重 relRMS | 原生字节 | 4 步 verdict | 2 步 verdict | 结论 |
|---|---|---|---|---|---|
| int6-g64 | 0.0238 | 15.70 GiB (−12.5%) | preserved | preserved | ✅ 可用 |
| **int6-g128** | 0.0256 | **14.58 GiB (−18.8%)** | preserved（satur× 1.093） | preserved（satur× 1.200） | ✅ **当前最优** |
| mixA（MLP=int4） | 0.0577 | 13.0 GiB (−27.5%) | degrades（0.887） | preserved | ⚠️ 不可判 |
| mixB（attn=int4） | 0.0576 | 13.9 GiB (−22.5%) | preserved | degrades（0.944） | ⚠️ 不可判 |

## 判定边界（本轮最重要的定量产出）
- **preserved ⇔ 权重 relRMS ≲ 0.026；degrades ⇔ ≳ 0.091**（用源 per-row int8 反量化为基准口径）。
- 0.058 一档**不可判**：代理量跨步数波动 ±0.15~0.25，而候选距阈值仅 −7%/−1%。
- 「余量」比数值本身更重要：int6-g128 距阈值 +15%/+26% → 可信；mixA/B 贴线 → 不可判。
- 权重 relRMS 与画质**非单调**：int6-g128 误差略大于 g64，端到端反而略好。

## 验收记录
| 项 | 结果 |
|---|---|
| 扫描实现校验 | 扫得 int4-g64-asym 0.09096 vs 对 MLX 产物实测 0.09127（差 0.3%）✓ |
| int6-g128 视觉 | `ab_report/montage_steps4_int6g128.png` 五联对比：base / int8 / int6g128 肉眼等价，int4 明显退化 ✓ |
| 确定性 | base 两次独立运行 md5 一致（`906150b0…`）✓ |
| 脚本 | `py_compile` 通过（3 个新/改脚本）✓ |

## 已知问题
- 代理 checkpoint 恒为 21.4 GiB（per-row int8 交付），**不体现方案的字节收益**；
  字节收益来自 `quant_scheme_sweep.py` 的模型，要落地必须做 Phase 2（引擎原生 int6/int4 加载）。
- 混合方案（<14 GiB）在评测加固（Phase 1d）之前不得下结论。

## Next
1. **Phase 1b**：int6-g128 的 scale/zero 改 F16 → 14.02 GiB（−21.9%），权重误差不变；一次导出 + A/B。
2. **Phase 1d**：评测加固（多 seed ≥3 × 多 prompt ≥2 聚合 + 报告波动），使边界可判。
3. **Phase 2**：引擎原生 int6 加载路径（格式与改动点已写入 `task_plan.md`）。
4. **Phase 3**：校准激活导出 + GPTQ-lite，把 int4 的 relRMS 从 0.091 压到 ~0.03 以下。

---

# Session: Phase 3（校准激活 + 误差补偿 PTQ）2026-09-13 续

## 引擎改动（唯一一处，102 行，`h3_dit.c`）
- 新增 `H3_DUMP_ACT=<dir>` / `H3_DUMP_ACT_ROWS`(默认 256) / `H3_DUMP_ACT_LIMIT`(默认 8)
- 结构体字段 `dump_dir / dump_rows / dump_limit / dump_calls[50][4]`；helper `dump_activation()`
- 4 个调用点 = 4 个被量化矩阵的真实输入：`mod_attention`(qkv) / `attention_heads`(out) /
  `mod_mlp`(fc1) / `activated`(fc2，须 `H3_DISABLE_FUSED_MLP=1`)
- 机制：`h3_gpu_submit`（commit + 等所有在飞缓冲）→ `read_bf16_range` → `h3_gpu_begin`（重开链）
- 编译 `0 warning / 0 error`；`git diff --stat h3_dit.c` = **+102 行，无其他文件改动**
- 非 dump 运行在 helper 第一行返回 → 零影响

## 新增脚本
| 脚本 | 作用 |
|---|---|
| `quant_calib_analysis.py` | 层输出误差度量 + 对角各向异性 + AWQ + whitening oracle |
| `gptq_probe.py` | 单层真 GPTQ 判决性测试 |

## 本轮实验矩阵

| 实验 | 结果 | 判定 |
|---|---|---|
| 校准 dump | 200 文件 / 3.2 GB / 每层 1068 行；出片正常 | ✅ 钩子可用 |
| ConvRot 对角均衡 | 1.86e8 → 645（均值），单层最高 3.7e6× | ✅ 解释 AWQ 失败的机制 |
| AWQ 通道缩放 | 0.05377 → 0.06038（0.891×），20/20 层变差 | ❌ 负收益 |
| 一次性 whitening | 误差 550（病态） | ⚠️ 无效，非有效上界 |
| GPTQ 初版（完整 H⁻¹） | 三层全变差（0.0387→0.0807 等），relRMS 爆到 0.32 | ❌ 实现 bug |
| **GPTQ 修正版（H⁻¹ 上三角因子）** | **0.03865 → 0.00886（4.36×）** | ✅ **成功** |

## 两个方法论产出（可复用）
1. **「极限行为应退化到已知基线」是最强的自检**：damp→∞ 时 GPTQ 补偿应消失、结果应收敛回 plain；
   实测反而单调变差 → 立刻定位到实现 bug（而非条件数问题）。
2. **权重 relRMS 不能判质量**（第二次证明）：GPTQ 把 relRMS 从 0.091 抬到 0.357，
   同时把输出误差降 4.36×。判据必须是 `ab_quant_quality.py` 的端到端 verdict 或层输出误差。

## 成本实测
- GPTQ：**729 s/层**（qkv, d_in=5376）；按 `d_out·d_in²/2` 外推 200 层 ≈ **45 h** 单线程
- 校准集 n=1068 ≪ d_in(5376~14336) → H 严重欠定；放大到 ~10k+ 行/层需 dump 增至 ~30–60 GB

## Next（Phase 3 后续）
1. 3c 放大校准集（更多步数 × 更多 prompt）→ 重测单层增益
2. 3d 全 200 层 GPTQ 跑批（按 block 并行），产出 h3 格式代理 checkpoint
3. 3e 端到端 A/B（int4-GPTQ vs base，2/4 步）——**这是 int4 路线的最终判据**
4. 若 3e 不通过 → int4 终止，交付已确认的 int6-g128（−18.8%）+ Phase 2 原生加载

---

# Session: int4 路线 Phase 2（引擎原生 grouped 加载路径）完成 — 2026-09-13

## 做了什么
- 实现引擎**原生 grouped 量化加载器**，直接吃 `export_h3_int6_native.py` 产出的打包权重（U8 打包 + F16 scale/bias），不再走「重量化回 per-row int8」代理。
- 654 行净增（`h3_weights.c` +344 / `h3_dit.c` +272 / `h3_weights.h` +67）；resident 与 SSD 流式两条路径都接入；int8 路径零改动。

## 关键设计落地
- 格式自描述（dtype + 形状推导 bits/group，不新增元数据张量）。相对原提案把 `U32+zero` 改为 **`U8+bias`**，与导出器对齐。
- dequant 顺序：`code*scale+bias` → **在 ConvRot 反旋转之前按组乘**（group=128 非旋转块 256 的倍数，故不能先反旋转再乘）。
- 复用 int8 加载器的 radix-4 蝴蝶做反旋转（单实现，常驻/流式共享）。

## 验证（已闭环）
| 项 | 结果 |
|---|---|
| `make h3_grouped_tests` 单测 | block 0 四矩阵 vs Python golden **relRMS 0.000e+00**（容差 1e-3），4/4 PASS |
| 编译 | 严格标志 0 warning / 0 error |
| 端到端 `/tmp/h3_nat_stream.mp4` | rc=0、无 Metal 报错；2 步/4 步均 `preserved` |
| 拼图 `montage_native_int6g128.png` | base / int8 / int6g128 肉眼等价 |
| 字节真实收益 | int6-g128 **14.58 GiB（−18.8%）落盘**（代理路径恒 21.4 GiB，不体现） |

## 结论
- int6-g128 原生加载 = **Phase 2 交付完成**，质量等价于 Phase 1 代理判据（preserved），且字节收益真实落盘。
- int4（bits=4）**复用同一加载器**，无新引擎代码；可用性门槛 = Phase 3 GPTQ 把 relRMS 压到 ~0.03 以下 + 3e 端到端通过。
- 回退方案已无必要：3e 不通过也直接交付 int6-g128 原生（−18.8%），无需退回代理路径。

## Next
- Phase 3 的 3d（全 200 层 GPTQ 跑批，~6–8h 墙钟）+ 3e（端到端 A/B）—— int4 路线最终判据。
- 3e 通过 → int4 checkpoint 用同一 `export_h3_int6_native.py --bits 4` 导出，引擎零改动。

---

# Session 2026-09-14：video VAE 解码流式化（消除分辨率×时长内存墙）

## 起因
VAE 解码原把整段 RGB 累积在 `final_rgb`（帧数×宽×高×3），峰值与「分辨率×时长」成正比
→ 长视频/高分辨率会 OOM。改为逐时间分片流式解码 + pipe FFmpeg。

## 改动
- 新增 `h3_ffmpeg_mux`（video+audio pipe，异步音频线程）替代一次性 `h3_ffmpeg_write_av_*`
- `h3_video_vae_decode` 加 `sink`/`sink_opaque`；`decode_chunked` 与单 chunk 分支在每 chunk 解出后调 sink
- `h3.c` 新增 `h3_vae_frame_sink`（f32→u8→resize→on_frame 预览→pipe）
- 主路径 `h3_generate` 与 `--latent-in` 路径均改用 sink 流式

## 验证
- 编译 0 warning（仅预存 `conditioning_key` 警告，无关）；链接 `libh3.a` 成功
- 主路径端到端跑通：448×256 / 1s / 4 steps → `FFmpeg 39/39 → wrote mp4`（ffprobe 含音视频流）
- 像素级等价：代码审查保证（sink 仅替换累积步骤，解码/重叠/混合未改）；待用户 `git stash` 对照

## 关键结论

### step 数增长对内存峰值**无影响**
- denoise 在**固定** latent 张量上重复 DiT forward；step 数只乘 forward 次数 → **耗时线性增长**，不分配新大张量。
- 代码证据：`h3_dit_denoise_euler_preview`（`h3_dit.c:5068`）的 `video_velocity` / `audio_velocity` /
  `last_video` / `previous_video` 等全在 `for (step …)` **循环外**一次性分配（大小 = latent 元素数，固定），
  循环体内无 per-step `malloc`。
- 与 step 相关的唯一内存是 `h3_dit_schedule_precompute` 的 per-step AdaLN modulation / gate 张量
  （每 step ~KB 级，可忽略）和可选的 `row_maps`（token reduction 的小映射，已在 `load_dit` 一次性分配）。
- 内存峰值由以下**独立于 step** 的因素决定：
  1. DiT 权重 resident 模式（全常驻 / 流式 2 槽 / 部分常驻 K）
  2. video latent 大小（分辨率×时长，固定）
  3. VAE 单 chunk（已流式化，与时长解耦）
  4. 条件张量（文本/视觉编码，一次性）
- VAE 流式化后，峰值**进一步与时长解耦**：时长只增耗时（chunk 数 × 每 chunk 解码），不增峰值。

### 反向提示（避免误判）
- 「step 增加画质变好」是**质量/算力**维度（如 Phase 16 测 step7 vs step4 SSIM +54%），不是内存维度。
- 若观察到 step 增多时 RSS 上升，几乎一定来自别处（如缓存/调度张量或系统内存压力），非 denoise 循环本身。

## Session 2026-09-14/15：长时长 15s 内存墙 + 守卫 + 分段生成

### 起因
用户要求「VAE 流式化后，decode 单 chunk、int6 与 int8、864×480、15s、steps 4 的时长与峰值有无撞墙损耗」。

### 过程与结论
1. **int6/int8 路径定位**：int6=`h3_int6g128_native`；int8 预量化(h3_int8/h3_int8g64) 均缺 `condition_proj.bias`
   → 改用 BF16 `minimax_h3_fastvideo_4step.safetensors` + 运行时 int8（用户指明）。
2. **3s 实测**：int6/int8 均成功，VAE 单 chunk GPU 峰值 **0.654 GiB**（流式化生效）。
3. **6s 实测（trace）**：denoise 第 1 步 footprint 10.11→**13.88 GiB**（+3.77/步）触发守卫；
   加 `--token-reduction` 后峰值降到 13.00 完成 → 证明激活 ∝ token。
4. **15s 单次 → 整机死机重启 ×2** → 判定硬件墙（16 GiB + swap=0）。
5. **实现运行时守卫**（见 findings）→ 之后 6s 触发时**优雅退出，不再死机**。
6. **分段生成**：`gen_segments.sh` 2s×7，产出 `seg_out/final.mp4`（16.36s）。
   第 7 段曾 `Killed: 9`（系统内存保护，未死机），单独重跑成功后拼接。
7. **跨段条件调研（负结果）**：H3 vision encoder 三处不可得 → `--first-frame` 不可用 → 段间硬切。

### 本 session 改动文件
| 文件 | 改动 |
|---|---|
| `h3_host.h/.c` | `h3_host_memory_guard` + `h3_host_footprint` + physical 查询；`H3_MEM_GUARD_MB`/`H3_MEM_HEADROOM_MB`/`H3_MEM_TRACE` |
| `h3_dit.c` | include `h3_host.h`；denoise CPU/GPU 两个步循环插入守卫 |
| `h3_video_vae.c` | include `h3_host.h`；resident/chunked 两个 chunk 循环插入守卫 |
| `h3_memory_plan.c` | 预算 `target = min(rec×0.8, (physical−4GiB)×0.85)` |
| `gen_segments.sh` | 新增：分段生成脚本（硬切拼接） |

### 运行清单
| # | 命令 | 结果 |
|---|---|---|
| 1 | INT6 3s 864×480 steps4 | ✅ wrote mp4；VAE peak 0.654 GiB |
| 2 | INT8（BF16 4step + runtime int8）3s | ✅ wrote mp4；同 peak |
| 3 | INT6 6s（无 tr）| ❌ 守卫触发（denoise 13.88 GiB）|
| 4 | INT6 6s + `--token-reduction` | ✅ 完成（峰值 13.00）|
| 5 | INT6 15s / INT8 15s | ❌ **整机死机**（两次）|
| 6 | `gen_segments.sh`（2s×7）| ✅ `final.mp4` 16.36s |

### Error Log
| Error | Attempt | Resolution |
|---|---|---|
| 15s 单次死机 | 2 | 判为硬件墙 → 分段 + 守卫 |
| int8 缺 condition_proj.bias | 1 | 改用 BF16 transformer + 运行时 int8 |
| `--first-frame` pos_embed 不符 | 2 | 缺 H3 vision encoder → 硬切 |
| 第 7 段 Killed:9 | 1 | 系统保护（未死机）；单独重跑成功 |

### 5-Question Reboot Check
| Question | Answer |
|---|---|
| Where am I? | 守卫 + 分段生成完成，`seg_out/final.mp4` 16.36s 已产出 |
| Where am I going? | 若要段间连贯需补 H3 vision encoder；否则可用交叉溶解软化硬切 |
| What's the goal? | 16 GiB 机产出 15s @ 864×480 |
| What have I learned? | 16 GiB+swap=0 单次 15s 不可行；denoise 激活 ∝ token；VAE wired 不计 footprint；vision encoder 不可得 |
| What have I done? | 守卫 + token-reduction 验证 + 分段生成 + 3/6s 实测 + 15s 死机定性 |

### Next（用户指示 2026-09-15）
- 下次测试**降分辨率 256×256**（空间 token ≈ 864×480 的 16% → 预估单次 15s 可行，单步激活 ~1.5 GiB）。
- 段间连贯需补 **H3 vision encoder**（`h3-base/text_encoder`），之后启用 `--first-frame` 替代硬切。

### Phase 46: 256×256 / 15s 单次生成 — ✅ 成功
- 命令：`./h3 -d h3_int6g128_native -p "…" --width 256 --height 256 --seconds 15 --steps 4 -o out256_15s.mp4`
- 结果：`wrote out256_15s.mp4`；**15.08s / 256×256 / 含音频 / 659 KB**；总 **898s**，RSS 8.37 GiB，零报错。
- 内存：denoise 峰值 **11.43 GiB**（未触发守卫）；VAE decode 1.47 GiB、available 12.64→8.42 GiB。
- **关键结论：低分辨率下 15s 单次可行**，无需分段；分段仅在高分辨率（≥864×480）时需要。

# Session 2026-09-15/16：video VAE 跨 chunk 权重复用（无损）+ DiT GPU 侧 A/B

## 起因
用户先问"FREE / A-SelecT / Chimera 能否加速现有算法"，调研后转为实测：
- FREE（CVPR2026 Findings，arXiv 2511.20390）的 draft-verify 依赖 batch 并行填满 GPU，
  而本机长序列单步已 compute bound → 该红利不存在；A-SelecT 是判别式任务选特征层，不适用。
- 顺势查 `H3_FB_CACHE`，发现它与 `ssd_streaming` **硬互斥**，而本机 16 GiB 必然 streaming
  → 该路径不可用。转向 VAE 阶段，发现真正的浪费。

## 改动
- `h3_video_vae.c`：把"每 tile 重读整份 VAE 权重"改为 block-major（详见 findings）。
  抽出 `pack_hidden`/`finish_hidden`/`pack_chunk_states`/`finish_chunk_states`，
  新增 `run_stream_chunk`；`decode_chunked` 与 `h3_video_vae_decoder_decode` 均接入跨 chunk。
- `tests/test_semantic_vae.c`：新增 `--streaming-parity`（覆盖 resident decoder 的跨 chunk，
  该路径 CLI 不可达）。
- `Makefile`：`VIDEO_VAE_MODEL` 变量 + 测试钩子。
- `misc/fixtures/h3_vae_streaming_parity_256x256x39_f32.safetensors`：新增合成 latent。

## 验证
- 编译 **0 warning**（`-Wall -Wextra -Wpedantic -Wshadow -Wconversion` 全开）；lint 0。
- `make test`：`h3_tests` 1768 checks、audio GPU primitives、convrot PASS、vdn passed，
  新增 parity 测试 `ok: resident and one-shot streamed VAE decodes are byte-identical`。
- **端到端逐字节等价**：864×480×56/2 步，跨 chunk vs 原始 baseline
  `SSIM Y:1.000000 (inf)` / `PSNR inf`。

## 运行清单
| # | 命令 | 结果 |
|---|---|---|
| 1 | base 864×480×56 steps2 | VAE 242.837s（alloc 216.5 GiB，24 次块遍历）|
| 2 | `H3_VAE_TILE_PIXELS=512` | VAE 152.357s（6 次遍历）但**输出有损** SSIM 0.854 |
| 3 | `--video-vae-streaming 0` | ❌ `exit=137` SIGKILL（OOM）|
| 4 | per-chunk block-major | VAE 116.549s，SSIM 1.0 |
| 5 | 跨 chunk（最终） | **VAE 103.487s**，SSIM 1.0，submissions 912→84 |
| 6 | 复跑确认 | VAE 103.641s（波动 0.15%）|
| 7 | DiT A/B（864×480×22 steps2，钉住 6 块）| base 75.015 / l45 67.838 / tr 49.076 / r2 75.468 / combo 44.149 |
| 8 | DiT A/B（steps20）| base **775.689s** / r2 **419.618s** / combo **273.959s** |
| 9 | 576×320 steps10 combo | 92.256s，I/O 反超 GPU（unhidden wait 33.227s）|

## 关键结论

### 1. VAE 的 146s 是"同一份权重读了 24 次"
`run_stream_tile` 在 tile×chunk 循环内，每 tile 完整 load 36 块再逐块 free。
`alloc=216.520 GiB` ↔ 24 × 9.67 GB（吻合 99.8%）。代码注释认为去掉 overlap 只是
"marginally slows"，实测 **CPU 侧占 VAE 阶段的 60%**。

### 2. block-major 重构（无损）
把块循环提到 tile 外层，为每个 tile 保留一份 hidden 状态（16.6 MB/tile）：
- per-chunk：864 → 108 次 load
- 跨 chunk：108 → **36 次**（3 chunks 的 states 一起保留，400 MB）

### 3. DiT 性能测量必须钉住驻留块数
`h3_host_available_memory()` 使驻留块数在 5~10 间漂移，peak 6.1~11.1 GiB。
未钉住时 A/B 得到 **314s 假数据**（swap）和一次 OOM。`H3_DIT_RESIDENT_BLOCKS=6` 后 peak 稳定 6.096~6.099 GiB。

### 4. 权重 I/O 与分辨率无关
每次真实评估固定读 ~14.4 GiB（layers 45）≈ 14.4s @0.997 GiB/s。因此：
- 864×480：计算主导，`--token-reduction` 有效（−34.6%）
- 576×320：**I/O 主导**，`unhidden wait` 0.010→33.227s，token-reduction 白费
- 10 步 + `--reuse 2` = 6 次评估 → 86.4s，实测 86.741s

## Error Log
| Error | Attempt | Resolution |
|---|---|---|
| `stream chunk calls: 0` | 1 | 实际走 `decode_chunked`（`h3.c:2277`），我改的是 `decoder_decode_chunk`（2274）；两条入口都要改 |
| 首 tile 后 `exit=1`、错误信息为空 | 1 | copy 放在 `submit` 之后，无 open command buffer |
| `--layers 45` 跑 314s / tr 被 OOM | 1 | 自适应驻留漂移 → 钉住 `H3_DIT_RESIDENT_BLOCKS=6` |
| `--show` 触发不了 resident 路径 | 3 | `h3_terminal_detect()` 非交互=NONE；`script` 造 pty 无效 → 用测试覆盖 |
| `--render-width 432×240` 拒绝 | 1 | 非 32 倍数 → 576×320 |

## 5-Question Reboot Check
| Question | Answer |
|---|---|
| Where am I? | VAE 跨 chunk 已完成并双向验证；DiT A/B 数据齐全；576×320 档已确认 I/O bound |
| Where am I going? | 由用户决定：DiT 开关取舍、1 GiB 上限放宽与否 |
| What's the goal? | 消除 VAE 权重重复加载（无损）+ 量化 DiT 加速空间 |
| What have I learned? | 权重 I/O 与分辨率无关；DiT 自适应驻留毁测量；FREE/A-SelecT/GGUF 三条路对本机无效或非必要 |
| What have I done? | VAE −139.4s 无损；测试补齐；DiT A/B（2 步 + 20 步）；发现两个方法论陷阱 |

## Next
1. DiT 侧：三开关均有损；保守选项 `--reuse 1 --token-reduction --layers 45`（−41%，无跨步外推）。
2. 1 GiB states 上限在长视频（>8s）/1080p 会回退，可放宽或按可用内存自适应。
3. 低分辨率档已贴 I/O 下限，进一步只能动 `--layers`/`--reuse`。

---

# Session 2026-09-16 → 17：两阶段管线打通 + 新权重实测

## 起因
延续「Latent Upscaler 接入」任务：阶段 A（`--refine-sigma`）已完成，需验证两阶段收益，
并决定是否投入 B4（把放大器搬进 h3c，~350–400 行 C）。

## 完成

### B 阶段原语（B1–B3 全部完成）
| 子项 | 内容 | 状态 |
|---|---|---|
| B1 | `h3_gpu_conv3d_pad_f32`（padding + groups） | ✅ |
| B2 | `h3_resize_bilinear_f32`（CPU，对 PyTorch 参考值逐点吻合） | ✅ 41 项新测试 |
| B3 | `h3_gpu_group_norm_f32`（无 SiLU）+ `h3_gpu_channel_scale_shift_f32` | ✅ |

`make` 零 error，`./h3_tests` → **1829 checks**（1768 → 41[phase A] → +19[bilinear]）。

**关键转折**：B3 最大的未知数（GroupNorm）发现仓库已有 `h3_gpu_vae_encoder_group_norm_silu_f32`，
正是 NDHWC 的 `GroupNorm(groups,C)+SiLU`。B3 只剩"去掉 SiLU 的变体 + 逐通道仿射"。

### 格式桥（`h3_latent_bridge.py`）
- f32 往返**逐字节一致**；f16 精确半尺寸（65260 = 28+60480+16+4736）
- 实现细节：h3c writer **恒写 version 3**（即使 f32），按 v2 写会差 4 字节

### ComfyUI 桥接（发现既有节点 + 修 3 个 bug + 加 1 个节点）
- 仓库 `comfyui_nodes` 是**软链**到 `<ComfyUI>/custom_nodes/h3_binary_nodes`
- 修 `_read_h3_latent`（v2+v3 / f32+f16）
- 修 `_write_h3_latent`（v2 → v3，4 字节错位）
- 修放大器的 `h3_audio` 丢失（第三方节点 1 行补丁）
- **新增 `H3_BinaryLatentRefine`**（带 `prompt`；引擎要 `-p` 才走去噪）
- 新增 `gen_h3_two_stage_workflow.py` → 两份 UI 工作流

### 端到端实跑（ComfyUI HTTP API，无人工干预）
启动 ComfyUI → `/object_info` 取 schema → `POST /prompt` → 轮询 `/history`：
- 变体 A（①→②→Refine）640×384 ✅ 产出 mp4
- 变体 B（①→②→Decode）640×384 ✅ 产出 mp4 + 音轨
- **`h3_audio` 穿过放大器：相关系数 +1.0000，rms 逐位相同**

### 阶段 1（1280×704 / 640×352 render / 121 帧 / 5 步）
DiT 721.4s + video VAE 174.9s ≈ **15.2 min**；latent `[C=24,T=37,H=22,W=40]`。
放大器（真实模型）`40×22 → 80×44, T=37` 实测 **60.9s**。

## 关键结论

### 1. 放大器不是瓶颈
CPU/fp32 前向：小尺寸 2.44s，1280×704 时 60.9s。
⇒ **B4 的收益只剩"去掉 ComfyUI 依赖"，性价比存疑**（此前估计"十几到几十分钟"是错的）。

### 2. 内存墙 —— 这条路的天花板
| 配置 | tokens | 结果 |
|---|---|---|
| render 640×352, 121帧 | 32,560 | ✅ |
| render 1280×704, 22帧 | 24,640 | ✅ |
| **render 1280×704, 121帧** | **130,240** | ❌ 系统重启 |

**直接全分辨率生成是同一个 token 数 ⇒ 同样跑不动。**
⇒ "两阶段 vs 直接"在 1280×704/5s 上不成立；可行的是变体 B（放大后直接解码）。

### 3. 自适应驻留毁测量（最严重的一次测量错误）
steps=4 报 422s，钉住 `H3_DIT_RESIDENT_BLOCKS=1` 后重测为 **91s（差 4.6×）**。
受控 A/B：常驻 1 块 34.4s / 2.29GiB vs 常驻 11 块 50.6s / 9.46GiB。
**多常驻反而更慢**（争抢统一内存）。⇒ 所有性能测量必须钉住。

### 4. int8 VAE 省磁盘不省内存
磁盘 14.6 → 2.95 GiB，但运行时反量化成 F32 常驻，**peak 仍 9.365 GiB**。

### 5. 8step 蒸馏模型必须走 8 步
2步/4步/8步 SSIM 标尺：2步 vs 8步 = 0.581 ≈ 无关生成 0.551。
耗时公式 `22.5s + 17.1s × steps`，且**纯 I/O bound**（denoise 时间 ≈ SSD 时间）。

## Error Log
| Error | Attempt | Resolution |
|---|---|---|
| `truncated audio latent` | 1 | `_write_h3_latent` 改写 v3（reader 恒吃 28B） |
| `--refine-sigma needs a prompt (-p)` | 1 | refine 节点补 `prompt` 输入 + 空值硬报错 |
| 放大器丢 `h3_audio` | 1 | 保留 LATENT 字典附加键 |
| `missing required model file` | 1 | 工作流 `binary` 指向已不存在的 `.../h3.c/h3` |
| 系统重启（OOM） | 1 | 130,240 tokens；改用变体 B 或降时长 |
| steps=4 报 422s（错误数据） | 2 | 未钉驻留块数；钉住后 91s |
| `exec_command` 不存在 | 1 | 正确工具名 `execute_command` |
| zsh `$VAR` 不分词 | 1 | 多 flag 不用变量承载 |
| `pkill -f "ComfyUI/main.py"` 无效 | 1 | 用 `lsof -ti:8188` |
| 命令超 8000 字节 | 1 | 先写脚本文件再执行 |
| `/tmp` 被重启清空 | — | 中间产物丢失，改用 `/Volumes/data` 存关键产出 |

## 5-Question Reboot Check
| Question | Answer |
|---|---|
| Where am I? | 两阶段链路**已跑通并验证**；B4 因内存墙发现而降级 |
| Where am I going? | 等用户目视确认变体 B 画质 → 决定走 A/B 还是做 B4 |
| What's the goal? | 用便宜的低分辨率步替代昂贵的高分辨率步 |
| What have I learned? | 放大器不是瓶颈；token 上限 ~30k；自适应驻留毁测量；int8 不省内存 |
| What have I done? | B1–B3 完成；桥接 + 3 bug 修复 + 1 新节点；两份工作流并实跑；新权重标定 |

## Next
1. 打开 `h3_two_stage_upscale_decode.json` 目视确认画质（唯一能出 1280×704/5s 的路线）
2. 若 OK → 用它生产，B4 无限期搁置
3. 若要 B4 → 先补 `B4d`（对 PyTorch 逐点对拍）
4. 所有后续测量加 `H3_DIT_RESIDENT_BLOCKS=1`
5. 新权重生产建议 `--steps 8`

---

## 2026-09-17 收尾：变体 B 全尺寸验收 + 工作流副本同步

### 用户验收
实跑 `h3_two_stage_upscale_decode.json`（变体 B，全尺寸）→ 产出成功，
**用户结论：效果可以接受**。

```
output/video/h3_upscale_decode_00001_.mp4
  1280×704 / 124 帧 / 含音轨 / 4.1 MB
  帧内对比度 std=74.9   相邻帧差=4.62      → 真实画面 + 有运动
  音轨 rms=0.00089（噪声失败案例 0.486，低 546×）→ 真实音频
  音轨 5.18s ≈ 124帧@24fps = 5.17s        → 音视频长度对齐
```

⇒ **B4（把放大器搬进 h3c）无限期搁置**：链路已能出目标尺寸且画质可接受，
B4 唯一剩余收益是"去掉 ComfyUI 依赖"，不值 350–400 行 C。
⇒ `steps=4` 保留（用户确认可接受）。

### 工作流副本同步（差点提交错版本）
生成器同一份 dict 写两处，理论上应一致，但 `cmp` 失败 → **ComfyUI 前端重写过文件**。

| | 仓库副本 | ComfyUI 版 |
|---|---|---|
| 变体 A ① steps | 2 | **4**（用户改） |
| 变体 B ① steps | 20 | **4**（用户改） |

其余差异（`codec`/`filename_prefix`/note `text`）只是前端把位置数组归一化成命名值。

**若直接提交会静默丢掉这次编辑。** 已处理：
1. 用 ComfyUI 版覆盖仓库副本（保住编辑）
2. 生成器默认值对齐 steps=4，补 `control_after_generate`、SaveVideo/Note 的 `widgets_values_named`
3. **修掉根因**：生成器原会静默覆盖两处（过程中一度冲掉用户 ComfyUI 版，已从备份恢复）；
   改为默认跳过已存在文件，覆盖需 `--force`
4. 格式统一为缩进（变体 B 原是单行 0 换行的紧凑 JSON）

最终：两份 754 / 686 行，仓库版 == ComfyUI 版 == 生成器语义输出。
前端的 `extra.ds`（画布缩放平移状态）完整保留。

`make` ✓，`./h3_tests` → 1829 checks。

### 教训
- **JSON 生成类工具必须防覆盖**：生成物会被下游（前端）重写并携带用户编辑，
  "重新生成"等于丢数据。默认 skip + 显式 `--force` 才安全。
- **提交前检查生成物是否被下游改过**，不能假设"生成器写的就是最新的"。

---

## Session 2026-09-19（续）: ANE 真实 gate 排障 → Phase A 关闭

接手时状态：fork 的 `h3_ane_int8_test` 真实 checkpoint gate 神秘失败，
上一轮把矛头指向「系统卷只剩 4.9 Gi」，并试了 TMPDIR 搬迁。

时间线：
1. 读 `/tmp/ane_tmpdir_run.log` → 0 字节；`/Volumes/data/tmp/ane-tmp` 始终为空。
2. 定位原因：`h3_ane_bridge.m:77/183` 用 `NSTemporaryDirectory()`，走
   `confstr(_CS_DARWIN_USER_TEMP_DIR)`，**不读 TMPDIR**；缓存在
   `/var/folders/.../T/h3-ane-cache`（mtime 跟每次运行同步）。⇒ 搬迁路线作废。
3. 删自己的残留（h3-ane-cache 400M + 94FBBC93 staging 394M），复跑：仍 exit 138。
4. 分步定位：只跑合成 gate → 0 failure；加 checkpoint（real-qkv）→ PASS；
   加 LoRA（real-lora）→ 死。峰值 RSS 1308 MiB、swap 还剩 1.2 G ⇒ 不是内存/磁盘。
5. `lldb`：`EXC_BAD_ACCESS code=2`，故障地址每次页起始、ASLR 变页对齐不变，
   `str d1,[x24,x21,lsl #3]` 且内层 `cmp x5,#0x20`(=rows) ⇒ 参考实现的 `want` 写入越界。
6. 读到源码：`:926 double *want = malloc((size_t)out_dim * rows)`（少 `sizeof(double)`，
   688 KB 缓冲写 5.5 MB）；顺带发现 `:886 float *x = malloc((size_t)in_dim * rows)`
   同样少 `sizeof(float)`。同文件 `:38/:415/:619` 都写对了，只有 `real_lora_gate` 漏。
7. 改这两行 → `make h3_ane_int8_test` → 全绿：
   `real-lora cos=0.9999998 rel_l2=6.729e-04 nonfinite=0 PASS`，`0 failure(s)`。

教训：
- **`compile failed: ?` 这种空错误串是自欺欺人的入口**。NSError 为 nil 时打 `"?"`
  （`h3_ane_bridge.m:210`）把「框架没给原因」和「没走到报错」混在一起；
  排障第一反应应该是 lldb 抓现场，而不是猜磁盘。
- **SIGBUS + 空日志 ≠ 磁盘满**：管道全缓冲下堆溢出被杀同样是 0 字节。
  判据是「页起始地址 + ASLR 变动但页对齐不变」。
- 磁盘紧是**真实但次要**的因素（它解释了 08:57 编译期的 ANECCompile FAILED），
  别把它外推成今天崩溃的原因。findings F7 专门记了这条边界。

---

## Session 2026-09-19（续 2）: Phase B2a + B2b，ANE 通路落地 h3c

范围确认后可开工（用户答复：连 video/vision/audio VAE 一并规划；缓存/磁盘走「默认关跨进程
缓存 + 进程内复用」）。本轮只做「移植 + 在 h3c 内自证」，**不碰 h3_dit 的推理接线**。

### B2a（host 依赖）
1. `h3_convrot.{h,c}`：fork 的 H4 Kronecker 幂表 + `h3_convrot_derotate_f32`（cblas 分 slab）。
2. `h3_weight_load_int8_raw()`：按 h3c 约定重写（本地 `pread_bytes`、`%s_scale`、仅收 F32 scale），
   sidecar `<base>.comfy_quant` 解析 `int8_tensorwise` / `convrot` / `convrot_groupsize`。
3. `tests/test_int8_raw.c`：loader 形状正反面、字节对分片 head/tail 1 MiB 逐字节一致、
   **表 derotate vs h3c butterfly 逐元素相等**。

### B2b（bridge / linear / Metal 暂存）
1. `h3_ane_bridge.{h,m}`、`h3_ane_linear.{h,m}` 逐字拷入；`LIB_M` + `FRAMEWORKS(-framework IOSurface)` 接线。
2. 移植测试时带上了 fork 里那两处 `sizeof` 修复和 compile/cache 打点。
3. fork 缺的一环在 h3c 补上：`h3_shaders.metal` 的 `h3_ane_pack_bf16`/`h3_ane_unpack_bf16`
   + `h3_gpu.{h,m}` 四个 host 入口 + 新写 `tests/test_ane_staging.c`（位级往返、含尾部零填充、
   以及在活动命令窗口内拒绝过小输出张量）。

### 本轮的三个坑（详见 F12）
- grep 用 `head -5` 截断输出 ⇒ 误判 `h3_gpu_require_bf16` 不存在而重复定义；
  而且第一次「修正」又粘贴了一份，最终靠单次 Edit 删掉两份重复。
  ⇒ **grep 截断的结论不能当「不存在」用**。
- h3c 的 Metal pipeline 是**写死的名单**（`h3_gpu.m` 里 `[names addObject:]`），
  名单外的 kernel 只在 dispatch 时才报 `missing Metal pipeline`；且名单嵌在
  `if (gpu.tensorOpsEnabled)` 内部 ⇒ ANE 两个 kernel 注册到该 if 之外。
- `tests/test_ane_int8.c` 的 `cache_gates()` 会 `setenv("H3_ANE_CACHE","1",1)` 且从不还原，
  静默覆盖外部 knob ⇒ F10 的「关缓存零残留」当时是被这个测试自身破坏的。补保存/还原后，
  机关显示 `cache=0`、`T/h3-ane-cache` 退出 0 B。

### 最终验证（h3c，真实 checkpoint + 真实 Turbo LoRA）
```
make -j8 all                    无新 warning
./h3_tests                      ok: 1829 checks
./h3_convrot_test               PASS（含表==蝶形、derotate 精确）
./h3_int8_raw_test <transformer> 0 failure(s)
./h3_ane_staging_test           0 failure(s)
H3_ANE_CACHE=0 ./h3_ane_int8_test <transformer> <turbo-lora>
  real-qkv  cos=0.9999998 rel_l2=6.415e-04 compile=0.43s cache=0 PASS
  real-lora cos=0.9999998 rel_l2=6.729e-04 compile=0.50s cache=0 PASS
  12 关全 PASS / 0 failure(s)
```

### 停在哪儿
Phase B2 关闭。**B3/B4（把 h3_dit 的 4 条投影挂到 `H3_ANE_LINEARS` 后面）未开工**，
因为 h3c 有两个既有事实与移植结论正面冲突：qkv/MLP 已是融合 kernel（RoPE/SwiGLU 拆开会亏），
以及 `h3_lora_apply` 在加载时已把 LoRA 并入权重（与 F2「int8 下 LoRA 必须走高精度旁路」相悖）。
等用户定方向再动 `h3_dit.c`。

---

## Session 2026-09-19（续 3）: B3 前提核对 + 整 block 单图（B3'）落地 h3c

### 先把 F12 留的两个「冲突」查实（结论：只有一个成立）
读 `h3_dit.c:2084` 的流式路径顺序后确认：h3c 做的是 **dequant→BF16→并 LoRA→重新量化**，
per-row scale 是合并之后重算的；F2 否决的是「沿用原 scale 就地折进 int8」。
⇒ 两个不是同一件事，**ANE 不需要 LoRA 旁路**，也不必改 h3c 的 LoRA 策略。
同理 `h3_dit.c:819` 说明驻留权重是 **un-rotate 后的真值**，q/k/v 交错也已在权重侧重排完
⇒ ANE 用 **gs=0 纯 GEMM 档**，fork 图里那段图内 Hadamard 应当省掉。
真代价只剩一个：qkv/MLP 在 h3c 是融合 kernel（RoPE、SwiGLU 同趟），走 ANE 要拆成两步。
全部记在 **F13**。

### B3'：fork 的整-block-单图直接落地
`h3_ane_block.{h,m}`（857 行：adaln + int8 ConvRot 投影 + per-head norm + RoPE +
full softmax + gate 残差 + SwiGLU，一张图；带 `unload/reload` 常驻轮换）
+ `tests/test_ane_full_block.c`（真实 checkpoint + f64 回放）拷进来 **一次编译通过**
——B2a/B2b 铺的依赖（`h3_weight_load_int8_raw`、`h3_convrot_hadamard`、`h3_ane_bridge`）刚好够。
只动了 4 处 `(double)` 显式转换（h3c 的 `-Wenum-float-conversion` 比 fork 严）
和 2 处注释（usage 里 binary 名写错、rotation 腿需要 `H3_ANE_CACHE=1`）。

### 实测（blocks.0 真实权重）
```
compiled in 4.19s, blob 380.8 MiB
block S=64:  cos=0.999988 rel_l2=4.898e-03 nonfinite=0 PASS
timing S=1904: best 334.3 ms/block (compile 3.21s)
rotation S=1904: unload 5 ms + reload 29 ms, 3 cycles bit-identical PASS
```

### 本轮最值钱的一条：把「磁盘硬墙」从估算变成量出来的数
跑完 `T/h3-ane-cache` = 762 MiB / 2 条目 ⇒ **381 MiB ≈ 1.0× int8 blob / block-形状**。
50 block × 1 形状 = **18.6 GiB**，再加 aned 自己那份副本 ≈ 37 GiB；本机 avail 只有 12 GiB。
⇒ fork README 的「~19GB/形状」被逐 MiB 验证，**50 block 全量 + 跨形状缓存在这台机器上不可行**。

### 方向修正（写进 task_plan）
h3c 低分辨率档的去噪早就贴在 SSD 读盘地板上（每 block read 173.5 ms @2.22 GB/s）。
ANE **既不减少要读的字节、又每形状每 block 多付 3~4 s 编译**
⇒ 先别碰 `h3_dit.c`，把 fork 的 `tests/test_ane_block.c`
（ANE vs 纯 Metal vs 融合 Metal，argv=ROWS）作为 **B4-pre** 移植进来出 rows 扫描曲线，
量到「行数多大 ANE 才开始赢」再谈接线。

### 回归
`make -j8 all` 无新 warning；`./h3_tests` 1829 checks；`h3_convrot_test` PASS；
`h3_ane_staging_test` / `h3_int8_raw_test` 各 `0 failure(s)`；
`H3_ANE_CACHE=0 ./h3_ane_int8_test`（12 关，含 real-qkv/real-lora）见本轮日志。
本轮只删自己写的 `T/h3-ane-cache`（762 MiB），未碰任何用户目录。

### 紧接着把 B4-pre 也量完了（结论：DiT 侧接线取消）
`tests/test_ane_block.c` 零改动移植（`h3_gpu_*_bf16` 那批 Metal API 在 h3c 全没漂），
干净环境扫 rows：384 → ANE 1.49×；1536 → 1.04×（mean 反而慢）；3072 → 1.07×（mean 反而慢）。
⇒ **「4 条投影分别上 ANE」这条路不接了，`h3_dit.c` 一行没改**。
两个排障收获（细节在 F15）：
1. plain 模式的 `pack+submit` 会把「排空前面 Metal 队列」算进 pack（1536 行报 385 ms，
   staged 真值 72 ms，其中真 pack 只有 1~2 ms/投影）⇒ 已在头文件注释写清口径。
2. 第一次 1536 的 run 和后台 `h3_ane_int8_test` 撞车，同一参数报出 855 ms（干净 436.7 ms）
   ⇒ 计时 gate 必须独占机器。

## 续 4（2026-09-19 12:45）：Phase D0 —— video VAE 投影 ANE gate 跑通并放行

- 新建 `tests/test_ane_vae.c`（+ Makefile `h3_ane_vae_test`）：真实 decoder block0
  四投影，ANE fp16 vs h3c 的 `h3_gpu_linear_f32`，宿主侧再拿 `cblas_sgemm` 对一遍
  （Metal 与 BLAS 逐位相同 ⇒ 误差全部归 ANE）。零告警构建；为此给这个 .o 单独加了
  `-DACCELERATE_NEW_LAPACK`（老 `cblas_sgemm` 原型在 macOS 13.3 起被标弃用）。
- 实测（独占机器、`H3_ANE_CACHE=0`）：rows=1797 block 合计 90.7→28.0 ms（**3.24×**），
  rows=512 23.7→8.2 ms（**2.89×**，小 rows 不掉速）；cos=1.000000、
  rel_l2 5.1e-4~7.9e-4、fp16 溢出 0。外推一轮解码 GEMM 78 s→24 s。
- 纠偏两处前提：① 盘上 VAE 权重是 **F32**（560 张量全 F32，单片 10.4 GB）不是 F16，
  所以权重舍入进误差预算（仍过闸）；② fp16 图的缓存产物 = **1× 权重字节**且**与 rows
  无关**（128 MiB/block、4.5 GiB/36 block，两次不同 rows 的 run 完全同字节），
  F14 的 381 MiB/block 是 int8 图特有。缓存恢复逐位一致。
- 读到的一条结构性事实（决定 D2 形状）：VAE tile 是**等宽平铺**（边界 tile 靠重叠而非
  切短），所以一轮解码只有一个 rows 形状 ⇒ D2 用 rows 分桶换跨分辨率复用同一份产物。
- 未做（F16 已标）：bias 没测；激活是伪随机；**pack/unpack 没测**，而 CPU 填平面在 w2
  上 41 ms 比省下的 19.5 ms 更贵 ⇒ D1 的放行判据改成"端到端 ≥2×"。
- 磁盘卫生：本轮只新增/删除过自己的 `T/h3-ane-cache`（当前留 128 MiB 实验产物，可再生），
  用户模型/缓存目录未动。

## 续 5（2026-09-19 13:20）：Phase D1 —— f32 staging + bias 落地，端到端闸改口径

- 新增 `h3_ane_pack_f32` / `h3_ane_unpack_f32`（bias 在 unpack 侧加，图不管 bias），
  两个 pipeline 进 `h3_gpu.m` 显式名单（与 bf16 那两个同在 tensorOps 分支之外）；
  `h3_gpu_{pack_ane_input,unpack_ane_output}_f32`；
  `h3_ane_projection_create_f16` + `h3_ane_projection_apply_f32`，
  bf16 的 `h3_ane_projection_apply` 退化成共用 static `ane_projection_apply` 的薄壳。
- gate 加了"pack → ANE → unpack(+bias)"的端到端计时与校验：**rows=1797 block 82.4→36.4 ms
  = 2.26×**（缓存热时 2.43×），36 block 一条 pass **2.97 s → 1.31 s**；
  rel_l2 与裸图同级（5.322e-04）。**rows=512 只有 1.88×，不过 2× 闸**。
- 据此把判据口径从"裸图比"改成"端到端比"，并且**单投影只判退化（≥1.2×）、block 合计才判 2×**：
  最小的 `out` 单独只有 1.68~1.73×，但绝对值仍比 fp32 快，所以不需要"小投影退回 Metal"。
- 踩到并记下桥接事实：staging 目录名取自**模型内容 identifier**（`hexStringIdentifier`），
  不含传入的 `name` ⇒ 同 MIL+权重的两张图共用一个目录；改图名无效，只能先释放裸图。
- 回归：`h3_tests` 1829 checks、`h3_ane_staging_test`/`h3_ane_int8_test`/`h3_convrot_test`
  全 0 failure，`h3_ane_block_test 384` 仍 PASS（1.76×）⇒ apply 重构没动坏。
  `h3_metal_tests` 因 `misc/fixtures/h3_dit.safetensors` 缺失跑不了（既有状态）。
- 本轮改动文件：`h3_shaders.metal`、`h3_gpu.{h,m}`、`h3_ane_linear.{h,m}`、
  `tests/test_ane_vae.c`、`Makefile`（新目标 + clean 列表补齐 ANE 家族 + 该 .o 单加
  `-DACCELERATE_NEW_LAPACK`）。**`h3_video_vae.c` 仍未动**；D2 的三个决定待拍板。

## 续 6（2026-09-19 14:20）：Phase D2a —— 常驻容量探针量完，架构定形为「≤31 block + 全常驻 + 平面池化」

- 新建 `tests/test_ane_vae_residency.c`（4 种模式 capacity/reload/pass/window，
  Makefile 目标 `h3_ane_vae_residency_test`）。**`h3_video_vae.c` 一行没动**。
  为探针给共享代码加了薄壳 API：`h3_ane_linear_unload/reload`、
  `h3_ane_projection_unload/reload`、`h3_ane_projection_plane_bytes`、
  `h3_ane_projection_cache_hit`；`h3_ane_bridge.m` 加 `H3_ANE_CACHE_DEBUG=1`
  打印缓存命中/失败原因（纯诊断，不改行为）。
- 三个决定性数字（详见 F18）：**126 个存活句柄是硬墙**（第 127 张 `0x50004`，
  且 park 不腾名额 ⇒ 144 张不可能存在）；**轮换 reload 3.4~5.15 ms/张**，
  每 pass 全量重连把 2.26× 压成 ≈1.2×；**平面 320 MiB/block**（rows=2048）
  才是内存大头，池化后 36 block 共用一套。
- 抓到缓存的真缺陷：**"cache restored 但 load 失败" 会顺手把条目删掉**
  （restore 走目录 copy；store 前无条件 `removeItemAtPath`）。这解释了 F16 之前
  "同一张图一会儿命中一会儿不命中"。修法（硬链接 restore + 失败不删）排在 LRU 之前。
- 磁盘卫生：本轮唯一删除的是自己的可再生 `T/h3-ane-cache`（5.7 GiB，缓存涨到 150 条目时
  系统卷只剩 4.4 GiB，正是 ANE 开始失败的位置）→ 回到 12 GiB。用户目录/模型/系统缓存未动。
  删后复测冷编译：`./h3_ane_block_test 384` PASS（cos 0.999989、compile 0.5~2.3 s/图）。
- 修正了自己给过的一个错数：**"系统卷只剩 13 GiB" 是错的**，所以用户据此拍的
  "LRU 上限 6 GiB" 需要重拍（一个 rows 形状就 4.5~4.6 GiB 产物 + live staging 副本）。
- 回归：`make -j8 all` 无新告警；`./h3_tests` 1829 checks；`h3_convrot_test` PASS；
  `h3_ane_staging_test` 0 failures。`h3_metal_tests` 仍因缺 fixture 跑不了（既有状态）。
- 下一步：等用户重拍磁盘预算 + 确认走「① block 封顶」，然后进 D2b（缓存两修 + 分桶 +
  LRU + 平面池化）。

## 续 7（2026-09-19 16:05）：D2b + 平面池化完成，路线①的可行性用 124 张常驻图坐实

用户重拍后开工（**`h3_video_vae.c` 仍未动**）：

- **缓存三修（`h3_ane_bridge.m`）**：① `bridge_mirror` 改成逐文件硬链接递归
  （原来是目录 `copyItemAtPath`，命中一次就把整份产物再抄一遍）；
  ② `bridge_cache_store` 改成写 `<id>.tmp` → `moveItemAtPath` 换入，
  瞬时失败不再把好条目抹掉；③ 新增 LRU：`H3_ANE_CACHE_MAX_MIB`（默认 5120）
  **加上 `H3_ANE_CACHE_MIN_FREE_MIB`（默认 6144）的磁盘空闲下限**，
  restore 会 touch `compiled.ok` 作为"最近使用"，冷编译前先 `trim` 腾地方，
  顺手清扫 `.tmp` 残留。`H3_ANE_CACHE_DEBUG=1` 能看到 restored/miss/evicted 全过程。
- **rows 分桶**：`h3_ane_rows_bucket()`（`H3_ANE_ROW_BUCKET`，默认 256），探针里已套用，
  实测 rows=1797 与 2048 得到逐字相同的 identifier 且全部命中缓存 ⇒ 跨分辨率复用产物成立。
- **平面池化**：`h3_ane_planes_share/bytes/clear`（`h3_ane_linear.{h,m}`），
  按 `{chunks, input_bytes, output_bytes}` 收一套 IOSurface，调用方各持一份引用。
  实测 32 张图 footprint 4582 → **2341 MiB**、pass 377 → **345 ms**；
  124 张图（31 block）常驻 footprint 只有 **1584 MiB**、一条 pass **1.78 s**
  ⇒ 路线①（31 block 走 ANE + 5 block 留 fp32 Metal）≈ **2.26 s vs 3.46 s（1.53×）**。
- **两条要记住的成本规律**（F21）：ANE 每张图的成本随**常驻图数**上涨
  （43 ms/block@8 → 57 ms/block@31，拟合 38.2+0.62·N）；
  31 block 一套产物 3.9 GiB，与"留 6 GiB 空闲"在同一块系统卷上互斥
  ⇒ 接线不把性能押在缓存命中上（124 张冷启动 create 合计 71.4 s，命中时 11~46 ms/张）。
- 磁盘：缓存从 4.4 GiB 被自己的下限淘到 3.3 GiB，系统卷空闲 5.1 → 6.1 GiB。
  只动过 `T/h3-ane-cache` 与 `T/` 下 ANE staging 目录，用户数据未动。
- 回归：`make -j8 all` 无新告警；`h3_ane_int8_test`（含缓存 hit/rehit/miss + bitexact=1）、
  `h3_ane_staging_test` 0 failure；`h3_ane_vae_test 2048` 端到端 2.33× PASS；
  `h3_ane_block_test 384` PASS。
- 下一步：D2c（`h3_weights` 加 F32→fp16 读取口，探针里的 `read_weight_f16()` 转正）
  → D2d（`h3_video_vae.c` 接线：`H3_ANE_VAE` 默认 off、`H3_ANE_VAE_MIN_ROWS=1024`、
  `H3_ANE_VAE_MAX_BLOCKS` 默认 31、池化平面、逐 block 与 Metal 对拍）。

## 续 8（2026-09-19 16:35）：D2c —— fp16 权重读取口转正

- `h3_weight_load_f16_raw(store, name, output_dim, input_dim, &halves, err, n)`：
  F32 存储逐元素 `(__fp16)` 舍入，F16 存储 `pread` 逐字拷贝；形状/dtype 不符拒绝。
  动机与逐条验证见 F22。
- 探针 `tests/test_ane_vae_residency.c` 改调用新口（本地实现删除）。
- 覆盖两条分支的实测：
  - `h3_ane_vae_test <FL2VA video_vae source> 2048`：qkv/out/w1/w2 四个投影
    与闸门自算的 fp16 payload **逐字节一致**；闸门本身仍 block 3.24×、端到端 2.34× PASS。
  - `h3_int8_raw_test <convrot transformer>`：新增 F16 逐字读取用例
    （`blocks.0.adaln_proj.linear.weight` [96768][8]）与"I8 必须被 fp16 口拒绝"，0 failure。
  - `h3_ane_vae_residency_test ... capacity 2048 2`：8 张图正常（24/8/64/32 MiB 权重）。
- 回归：`make -j8 all` 无新告警；`h3_ane_staging_test`、`h3_ane_int8_test` 各 0 failure。
- 未动用户数据；本轮只读 `~/h3_sys`（convrot 分片）与 FL2VA 目录。
- 下一步：D2d 接线（`h3_video_vae.c`）。

## 续 9（2026-09-19 17:55）：D2d 接线完成并实测 —— 正确性 PASS、性能判负，建议收 Phase D

- `h3_video_vae.c` 接线落地：`H3_ANE_VAE` 默认 off、`H3_ANE_VAE_MIN_ROWS=1024`、
  `H3_ANE_VAE_MAX_BLOCKS=31`、rows 分桶（1797→2048）、平面池化、常驻不轮换、
  ANE 覆盖的 block 跳过 fp32 矩阵上传、逐 block 优雅回退、`H3_ANE_VAE_STATS` 分阶段计数。
  配套：`h3_ane_projection_set_activation_rows`（编译桶 vs 真实行数解耦）、
  Makefile 新目标 `h3_ane_vae_decode_test`。
- A/B 闸门 `tests/test_ane_vae_decode.c`（同 latent 先 metal 后 ANE，`REPEAT` 分离建图/稳态）：
  **正确性 PASS**（cos 1.000000、rel_l2 2.2e-04~5.2e-04、max_abs ≤3e-3、非有限 0），
  **回退 PASS**（`vae.b26/b30.w1 compile failed` 时后面 block 自动走 fp32 Metal，输出仍对）。
  **性能判负**：稳态 6 block 1.00×、16 block 0.95×；产物没留住的轮次 0.13~0.38×。
  两条与缓存无关的根因：四 GEMM 只占解码墙钟 16%（上限 11%）＋跨界 `submit+wait`
  每次 ~12~13 ms 空泡（6/16 block 两轮独立反推一致）。全部数据与推导见 F23。
- 自己写重了一条并已更正：F23 初稿说"LRU 疯狂淘汰"，`H3_ANE_CACHE_DEBUG=1` 复跑是
  **0 条 evicted、64 restored、0 miss** ⇒ 改为"默认下限下产物留不住是事实、机制未定"，
  并把结论收敛到只依赖上面两条根因（见 F23 更正）。
- 内存账：本机 **16 GiB** 统一内存，大 tile（rows 7173 ⇒ 池化平面 ~5.7 GiB + 10.4 GiB
  权重流式读）根本放不下，所以唯一可能翻盘的形状在这台机器上不可测。
- ANE 侧本身没有退化：同一形状 `h3_ane_vae_test 1797` 仍 block 3.07×、端到端 2.43× PASS。
- 回归：`make -j8 all` 无新告警；`h3_tests` 1829 checks；
  `h3_semantic_vae_test --streaming-parity`（默认 off 的 video VAE 路径）逐字节一致；
  `h3_ane_staging_test`、`h3_ane_int8_test`（convrot 分片）各 0 failure。
- 文档：`ANE_PORT_SUMMARY.md` 补 §9，把 §8 里「video VAE 合适/唯一值得做」按实测作废。
- 磁盘：本轮只动自己的 `T/h3-ane-cache`（最早一轮冷编译前空闲 5.4 GiB < 下限，
  被自己的 trim 清过；结束时 2.1 GiB、卷空闲 6.3 GiB），用户数据未动。
- 待用户拍板：是否就此收 Phase D（接线与默认 off 保留）；另两轮候选 ——
  跨界空泡能否用双缓冲/一次跨界多投影消掉、ANE 产物在本机留不住的机制定性。

## 续 10（2026-09-21 06:25）：DiT 逐算子剖析落地（`H3_DIT_OP_PROFILE`），记 F24

**为什么做**：ANE 三轮（F14/F15 整 block、F23 video VAE）都死在"每次跨界 ~13 ms 空泡"
这类**没有实测就只是猜测**的量上；而 DiT 侧此前只有整块耗时，没有任何逐算子分解。
用户点名要 SDPA / QKV / FC1 / FC2 / adaLN 在 864×480 与 576×320 两档各多少毫秒。

**代码改动（只有 `h3_dit.c`，117 行）**
- `struct h3_dit`：`op_profile` / `op_profile_wall` / `op_profile_stats`。
- `run_block` 的 `OP(call, label)` 在 `H3_DIT_OP_PROFILE=1`（非空且非 `0`）时变成括号版：
  调用前记 wall+stats，调用后 `h3_gpu_submit` → 取 `command_encode_seconds` /
  `command_wait_seconds` 增量 → `h3_gpu_begin` 重开链，按 label 累加到文件级表。
  `encode_forward` 里**同名** `OP` 保持原样（它的调用就是链的 begin/continue/submit）。
- `h3_dit_free` 里 `op_profile_report()` 打印按 wall 降序的表（ms/call、encode、wait、占比）。
- README「DiT 快路径」段后新增该开关的说明。

**为什么必须自己加括号**：整块编码进一条 command chain，驱动只报得出块总量；
root `GPUEndTime-GPUStartTime` 又不可信（MPSGraph 子 buffer，既有记录里同一配置报过
0.298/0.400/121.995 s）。读 `wait`（主机时钟围着 `waitUntilCompleted`）才可信。

**结果（详表在 findings F24）**：864×480（7074 rows）每块括号内 2451.6 ms ——
MLP 1109.2 ms（45.2%）、QKV 607.9（24.8%）、SDPA 503.4（20.5%）、attention out 206.4（8.4%）、
gate+AdaLN 合计 ~1%。576×320（3249 rows）每块 921.3 ms —— MLP 476.8（51.8%）、
QKV 242.6（26.3%）、SDPA 112.4（12.2%）、out 85.0（9.2%）。
拆 FC1/SwiGLU/FC2：864 = 681.0/13.8/336.9 ms，576 = 313.0/3.7/160.6 ms。
折算有效算力 2.8~3.7 TFLOPS（非 TensorOps BF16 已近本机上限 ⇒ kernel 层无大肉）。
**括号自身的失真也量了**：576 同配置 81.410 s（90 submissions）vs 92.182 s（692）
⇒ 每括号 17.89 ms；校正后每块 813.6 vs 814.1 ms，闭合 0.06%，且括号内算子合计
92.126 s = 有括号 denoise 全程的 99.9%（块外算子 <0.1%，不用单列）。

**顺带独立印证 F23**：一次纯 GPU `submit`+`wait` 括号就是 17.9 ms，
F23 反推出的"每次跨界 ~12-13 ms 空泡"量级正确、且不是 ANE 特有。

**踩到的坑（记进 F24）**
1. 括号版最初在 `op_profile_start` 里也 `submit`，把链关掉后第一个算子直接
   `h3_gpu_begin() was not called` 死掉 ⇒ start 只记时钟，提交/重开全在 stop。
2. 第一次的 576 两轮开局即 `Unable to reach MTLCompilerService`，`864_unfused` 墙钟
   124 min 而 monotonic 只有 221 s ⇒ **机器睡了**，唤醒后 Metal 编译服务已死。
   长跑一律 `caffeinate -is`。
3. `H3_DIT_OP_PROFILE=`（空值）在 `getenv()!=NULL` 式闸门下等于**开**，参照跑差点
   没关掉 ⇒ 改成非空且非 `0` 才算开。

**回归**：`make -j8 all` 无新警告；`h3_tests` 1829 checks ok；
同一 256×256/0.5s/2 步配置**带括号与不带括号的 mp4 逐字节一致**
（sha256 前缀 `c934a43c78efb0a8`）⇒ 剖析开关不改语义。
`h3_semantic_vae_test --streaming-parity` 这次起不来：它按仓库根相对路径找
`MiniMax-H3/FL2VA/video_vae/source/../config.json`，本机没有那个符号链接（与本轮改动无关）。

**临时物**：`/tmp/run_op_profile.sh`、`/tmp/run_op_profile2.sh`、`/tmp/opprof_*.log|mp4`，
未进仓库。

## 续 11（2026-09-21 06:55）：同进程 A/B 判掉"fused MLP 慢 8~10%"，记 F25

**为什么做**：F24 §4 唯一可疑的一格是 864 档 fused MLP 1109.2 ms vs 拆开
FC1+SwiGLU+FC2 1031.7 ms（+7.5%），但 F24 §6 自己量到同一算子跨进程能差 18%
⇒ 这个数既不能定罪也不能释放。用户没点名，是剖析结果自己浮出来的最便宜的一条判据。

**新增**：`tests/bench_mlp_fusion.c` → `h3_mlp_fusion_bench`（独立 target，已进 `clean`，
不在 `all`）。两档 rows 3249/7074，融合侧一次 `h3_gpu_mlp_bf16`，拆开侧
`h3_gpu_linear_bf16`+`h3_gpu_swiglu_bf16`+`h3_gpu_linear_bf16`，同一套 buffer；
窗口外 `h3_gpu_begin`、窗口内"编码+一次 `submit`（含 wait）"，测完链是关着的
⇒ 两侧各只有一次排空、对称，且两档 rows 之间不残留状态；奇偶轮交换先后。
用法 `caffeinate -is ./h3_mlp_fusion_bench 13 3249 7074`。

**结果**：四次（两档 × 7/13 轮）比值 0.989 / 1.007 / 1.001 / 0.995，两次 7074 符号相反。
13 轮 7074 配对：mean 1.011、sd 5.3%、95% CI ±3.2% ⇒ "慢 8~10%" 排除，最坏 ≤4.2%。
⇒ **默认融合路径不动**，405 MB 常驻激活的节省白拿，`H3_DISABLE_FUSED_MLP=1` 退回纯数值口子。

**噪声地板（本轮的真实产出）**：同进程逐轮极差 3.9~11.5%，同 binary 两次运行绝对值仍差
6~12%（3249 fused 中位 432.7→458.6，7074 998.2→1120.3）
⇒ 这台机器上 <3% 的单算子差**只能**用"同进程 + 交替 + 多轮中位数"来谈。

**踩到的两个自己的坑**：① 首版 `sweep()` 用 `return 1` 同时表示成功和失败，
7074 那档第一次直接没跑（`h3_gpu_begin` 因链未关而失败）却打印 PASS —— 失败路径改 `return 0`，
并把 `h3_gpu_begin` 移进 `measure()` 开头；② 打印比值前先算中位数会排序原地改数组，
min/max 只能在排序后取，已按此写法核对过。

**边界**：零 buffer、孤立窗口 ⇒ 只判"融合图本身不亏"，不覆盖块内缓存邻居交互，
也不覆盖 M5 TensorOps/int8 那条融合 MLP（本机 `int8 off`）。

## 续 12（2026-09-21 07:15）：窗口注意力内核量死（每对 70~78 倍），记 F26

**为什么做**：F25 结掉 fused MLP 之后，计划里排第一的是"块稀疏/窗口注意力占 864 的 20.5%、
内核现成只差接线"。这条估算隐含"稀疏内核算一对和稠密 MPS 一样贵"，从没量过；
接线前先用最便宜的方式量它。

**新增**：`tests/bench_attention.c` → `h3_attention_bench`（独立 target，已进 `clean`）。
同进程、同一套 q/k/v/out（`sequence×56×128` BF16）、两侧交替、F25 那套计时窗口。
几何用 F24 两档的真实分解 `7074 = 189 + 17×405`、`3249 = 189 + 17×180`；
另设 `radius=16` 的**稠密对照**（对数几乎不变），把"内核快慢"和"少算多少"解耦。

**结果**：B/A = 13.27（7074/r1）、68.03（7074/r16 对照）、16.28、71.63。
除以精确算出的对数占比（18.73%/97.40%/20.86%/94.52%）⇒ **每对 70~78 倍**，
两几何两窗口互合到 1~3%。内核是"一线程一 query 行 + K/V 逐 key 从 global 重读 +
每 key 重标 128 通道累加器 + 纯标量 FMA"（`h3_shaders.metal:5600-5712`），
且 `h3_gpu_flash_attn_bf16` 在产品代码里零调用点。⇒ "只差接线"判死，
真要拿那 20.5% 得新写分块 flash 内核（周级工程）。

**自我纠正一处**：表格里"对数占比"我先手算成 19.8%/97.3%/22.4%/94.2%，
用脚本逐帧重算后是 18.73%/97.40%/20.86%/94.52%；标题的"68 倍"也改成 70~78 倍，
"+1200%" 改成"那一个算子 13.3 倍、每块 2.5 倍"。改完才让文档被别人读到。

**回归**：`make -j8 all` 无警告、`./h3_tests` 1829 checks ok、`make clean && make -j8 all` 通过；
两个 bench target 都能单独建。稠密侧绝对值（7074 档 540.5 ms）对上 F24 块内的 503.4 ms（差 7%），
在 F25 §3 那条"跨运行 6~12%"带内。

## 续 13（2026-09-21 07:30）：ANE 线按用户决定收尾，不继续投入

`ane放弃` ⇒ Phase D3 不再开轮，task_plan 记决策与理由链（F23 占比+跨界空泡判负、
F24 §7 可 offload 池、F21/F23 更正的 3.9 GiB 产物撞磁盘下限）。
两条挂在清单上的 ANE 收尾项（产物留不住的机制定性、跨界空泡双缓冲）随之一并划掉。
代码不动：`h3_ane_*` 与默认 off 的 `H3_ANE_VAE` 接线留在树里，删不删等单独点头。

顺带把外部 key 的复查漏斗走完：`grep -rIl "typesafe\|TYPESAFE\|sk-[A-Za-z0-9]{8}"` 在这轮
新增/改动文件里 0 命中，唯一命中是仓库既有的第三方
`.agents/wiki/script/highlight.min.js`，逐条看是 `sk-composite` / `sk-position` 两个
语法高亮 class 名，与 key 无关（该文件在 d5752a0 就进了历史）。key 本身的轮换是用户侧动作。

## 续 14（2026-09-21 07:45）：拆小稠密 SDPA 每对不涨 ⇒ 块稀疏不用新内核，记 F27

接着 F26 往回纠一件事：F26 说"要吃到注意力 20.5% 就得新写分块 flash 内核"，
那句话隐含"稠密 SDPA 拆小了每对会变贵"。这轮直接把这句测掉。

`tests/bench_attention.c` 重写为 dense/split（window 留作 F26 对照，默认不跑）：
`split` = **G 套各自独立**的小 buffer，每次 `h3_gpu_sdpa_bf16(rows=⌊M/G⌋)`，
G 次编码进同一条 chain、一次 submit；每 shape 首用编图，所以先 warm 再计时；
5 轮轮换顺序取中位，每档另报 CPU 编码 / GPU 排空。
（上一版草稿的 split 是"同一个全尺寸调用重复 G 次"，那量出来是 G×稠密，白测；
另放宽了整除要求，行数下取整、`pair_fraction` 用同一个取整，报表自洽。）

结果（同进程中位比）：M≥~540 行时 split 每对只花稠密的 **0.8~0.9×**，
7074 档 G=6 是 `time ×0.140 / pairs ×0.167`；到 294 行仍 1.1×；
只有 3249 档拆到 135~180 行才涨到 1.4~1.8×。对照 F26 库内窗口内核的 **70~78×**。
CPU 侧 G 次编码 0.1~1.3 ms，不付 F24 §5 那 17.9 ms/括号。

配 F24 份额折算：864 一档 keep 1/6 全局上限 **−17.7%**、1/24 −19.6%；
576 一档 1/6 −10.2%。⇒ keep 从 1/6 再狠也多拿不到 2 个点，**这项的取舍全在质量侧**。

真正的缺口是**形状不是内核**：`h3_gpu_sdpa_bf16` 建图时 Q 与 K/V 共用一个 `sequence`
（`h3_gpu.m:1542-1568`）、batch 又钉成 1，只能算方阵 ⇒ 只表达得出块对角，
而块对角对视频必然判死（帧间不通、视频行看不到那 189 行文本）。
选中的 key 块是连续段，打包用现成 `h3_gpu_copy_bf16` 就够 ⇒ 下一步是
给 SDPA 加 `query_rows ≠ key_rows`，然后第一件事量矩形 (B × kB) 的每对成本。
未测项按老实记：矩形没量、50 层各异 keep 带来的几十张图编译与 `sdpaCache` 显存没量、
buffer 全 0、质量侧完全没碰。

改动：`tests/bench_attention.c` 重写、README 增一段价格表口径、
findings 追加 F27 并在 F26 §4 那条"要新写内核"下面加了修订行。
`make -j8 h3_attention_bench` 无警告。日志 `/tmp/split_dense.log`。

## 续 15（2026-09-21 09:10）：矩形 SDPA 落地并定价，每对成本 = 稠密，记 F28

补 F27 §5 第一条。三件事一起做完：

1. **API**：`h3_gpu.m` 建图函数把 Q 与 K/V 的行数分开（`H3SDPA` 加 `keyShape`、
   cache key 多一个字段、causal mask 用 `query×key`），新增
   `h3_gpu_sdpa_rect_bf16(..., query_rows, key_rows, ...)`；老的
   `h3_gpu_sdpa_bf16` 传 `sequence, sequence`，产品路径逐位不变。
2. **数值闸**：加在 `tests/test_flash_attn.c`（先写在 `test_bf16.c`，量完发现
   `h3_bf16_tests` 要 `misc/fixtures/h3_dit*.safetensors`，这台机器上根本没装，
   闸就跑不起来 ⇒ 用 `git checkout --` 撤回，改成不依赖 fixture 的版本，
   并把 `h3_flash_attn_tests` 接进 `make test`）。验三件事：
   rect(5,5) 与方阵调用**逐位相同**、rect(5,11) ⊂ dense(11)、
   两段不连续 key 行用 `h3_gpu_copy_bf16` 打包后对 BF16 参考误差 2.2e-04。
3. **定价**：bench 加 `rect`（纯调用）与 `rectg`（每次调用前真的打包窗口），
   MODE 多一个 `KEY_ROWS`；一轮同时出 d/s/r/g 四档。

结论：**7074 档每个形状每对都是 0.9~1.0×稠密**，keep=5% 的注意力只要 30.0 ms
（稠密 564~613 ms）。带打包流量也几乎不涨（6 块 1179×354：30.0→30.3 ms；
18 块 393×354：30.1→35.9 ms，+19%）。唯一变贵的因素是**单次调用的活量**：
每次 ≥~140k 对就平，掉到 65k 对（3249 档 9×361×180）变 1.4×，32k 对变 1.3~1.6×。
⇒ 3249 想吃 5% keep 该用 6 个大 query 块 × 180 key，不是 18 个小块；
而且方阵 split 根本给不出 5% keep（keep 被钉成 1/G）且只能表达块对角。

折算天花板（F24 份额 × 本轮带 gather 的比值）：864 一档 **−19.4%**（5% keep）、
−18.6%（10%）；576 一档 −11.3% / −10.6%。⇒ 速度侧已经吃满，能不能拿全看质量。

踩到并修掉的坑：`key_rows < rows` 时那一套 slice 的 K/V buffer 不够方阵 split 用
（`SDPA key tensor is absent or too small`）⇒ slice 按 `max(rows, key_rows)` 分配，
两种稀疏探针共用一套，SDPA 只读自己图里写明的行数。

显存账纠了个方向：bench 每块一套 buffer 最狠 813 MB，真实路径 copy→SDPA 串行消费
同一块 scratch ⇒ 打包 K/V 只要 20.3 MB（S=707）。但"同 chain 复用同一 scratch 是否
被正确排序"还没闸，上真机前得先补。

未测（按新到旧）：真实稀疏模式的 50 层渲染质量 A/B（现在才有形状可表达）、
每块 3~5 段不连续 key 的编码开销、50 层各异 keep 的几十张图首步编译 + `sdpaCache`
常驻（16 GB 机器上先于任何逐层表接线）、buffer 全 0。

回归：`make -j8 all` 干净（只剩 `h3_cli.c` 两条早就在的 float 转换警告）、
`./h3_tests` 通过、`./h3_flash_attn_tests` 含新闸全过。
日志 `/tmp/attn_rect_7074.log`、`/tmp/attn_rect_3249.log`。

## 续 16（2026-09-21 09:55）：打包 scratch 可以全块共用，显存账 813 MB → 20.3 MB，记 F29

F28 §5 那条"未验"是块稀疏上真机的第一个显存/排空岔路：bench 每块一套 buffer 最狠
813 MB；若逐块循环只能每块一套，就得要么吃显存、要么每块之间插排空
（把 F24 §5 那 17.9 ms/括号又请回来）。所以它排在质量 A/B 前面。

`tests/test_flash_attn.c::test_sdpa_scratch_reuse`（已进 `make test`）：13 行源 K/V +
11 行**共用**打包 scratch + 3 个 query 块，每块"写 scratch → 紧接自己的 rect 调用"，
全在一条 chain 一次 submit。三块选行互不相同，其中一块故意做成**两段**
（5..12 + 0..2），顺手把多段 gather 的正确性也占了。两层判据：
- 每块对它自己的选行参考 `max_err=2.18e-04`，与 F28"每块独立 buffer"那次 packed
  路径误差一模一样 ⇒ 没有串味；
- 相邻两块输出不得逐位相同 ⇒ 挡住"三次都算了最后一次内容"的退化。

⇒ 复用安全，打包 K/V 常驻 = `2·S·56·128·2`，S=707 时 **20.3 MB**；
多段 gather 只是多几条 copy 编码，不需要额外 buffer。
仍未测：多段 gather 的**编码开销**（字节数不变、op 数翻倍），以及几十个不同 shape
的 MPSGraph 首步编译 + `sdpaCache` 常驻（这条仍是逐层表前的硬闸）。

改动只在 `tests/test_flash_attn.c`（新增一个函数 + 注册），产品代码未动。
`make test` exit 0、`./h3_tests` 1829 checks、新闸在报表里可见。日志 `/tmp/make_test_f29.log`。

## 续 17（2026-09-21 14:40）：逐层静态表的三道前置闸全绿，记 F30

task#19 收口。逐层各异 keep 这件事，此前一直挂着两句"没量过"：50 张不同 shape 的
图要编译多久、常驻多少；每块多段 gather 会不会把 CPU 编码顶起来。本轮三条都量了，
**全是绿灯**，所以逐层表剩下的只有质量 A/B 一道门。

- **显存归因差点反过来**：第一版量到 50 shape 让 resident 从 801 爬到 4042 MB，
  看着像 65 MB/shape 足以否决逐层表。用"同一个 shape 重复 50 轮"做对照，一样爬到
  4835 MB ⇒ 那是 bench 自己每 MODE 重开池子的抖动。给 bench 加共享池后真实数字是
  **50 个 shape 合计 +48 MB（≈1 MB/shape）**，`engine tensors` 全程 774/774 MB。
- **首步编译 3.9 ms/shape（2.9~7.2），50 层一次性 201 ms**；缓存命中后 re-encode
  0.3~0.9 ms。一个纯 GPU 括号是 17.89 ms（F24 §5），所以这是"约一个括头、只付一次"。
- **段数不花钱**：6 块 ×707 key 从 1 段到 8 段 54.6→55.5 ms，18 块 ×354 key
  1→8 段 34.6→34.4 ms，每对全程 0.9/1.1；288 条 copy + 18 次 rect 的 CPU 编码只有
  0.8 ms，不需要每块之间排空。
- **keep 带补全**：固定 B=1179 只动 key_rows，480→1180（keep 6.8%→16.7%）单调无悬崖、
  每对 0.8~0.9，拟合每 1% keep ≈ 5.1 ms/块 ⇒ 逐层表的 keep 可以任意取，不用量化对齐。

方法学踩坑（写进 F30 §6，也写进 bench）：50 shape 那轮跑到中途，全序列 dense 对照
自己从 560 掉到 1000+ ms 且 20 分钟不恢复，制造出一个假的"S=440→500 每对 4× 悬崖"。
同进程配对**救不了**这种跨状态比较 —— 小调用比大调用亏得更多，坏窗口的每对成本系统性
偏高。现在 bench 每个 sweep 自带 `NOTE: dense is X× the fastest dense at sequence N`
（阈值 1.1×，按 sequence 分别跟踪），跨 shape 扫描先看有没有 NOTE 再看数字。
本轮日志：`/tmp/attn_segments4.log`、`/tmp/attn_keepband.log`、
`/tmp/attn_shapes50_shared.log`（其中 S≥480 的计时作废）、
`/tmp/attn_segments3.log`（整份作废）。

改动仍在测试侧：`tests/bench_attention.c`（共享池 + dense 自检闸）、
`tests/test_flash_attn.c`（F28/F29 两道闸），产品代码未动。
回归：`make -j8 all` 干净、`./h3_tests` 1829 checks、`./h3_flash_attn_tests` 全过、
`make test` exit 0（`/tmp/make_test_f30.log`）。

## 续 18（2026-09-21 19:40）：块稀疏接进 DiT，等价闸位精确，质量 A/B 判死固定 keep，记 F31

待办 1（固定 keep 质量 A/B）做完，和待办 2（端到端接线）合并成同一轮。

1. **接线**：`h3_dit.c` 加 `H3_SPARSE_ATTN=BLOCK_FRAMES[:RADIUS]`。video 行按帧分块，每块看
   自己窗口 ± RADIUS 帧 + 全部非 video 行；条件行打包一次进共享 scratch，块外用矩形稠密调用。
   默认 off，`--token-reduction`/VDN/几何不整除/`block > 帧数` 自动回退稠密，非法值创建时报错。
2. **等价闸**：`H3_SPARSE_ATTN=12:0`（keep=100%）在 576×320/1 s 上 layer0 注意力输出 video 行
   与稠密**逐位相同**（max|d|=0、cos=1.000000）⇒ 通路正确，后面的差异只可能来自 keep。
   顺带确认 `qkv.00.bin` 两边逐位相同，所以隔离的是算子而不是轨迹。
3. **质量 A/B（864×480/2 s/2 步）**：keep 8.4/19.8/31.3/42.7% ⇒ denoise −21.5/−17.9/−15.8/−14.7%，
   最终 latent cos 0.639/0.748/0.801/0.851。图上 `1:0` 主体散架、`1:1` 鬼影无脸、
   `1:2`/`1:3` 构图在但毛皮糊 + 饱和度偏高（detail −37%、sat +20~27%，相对 dense）。
   **可用 keep ≳43% 只剩 −14.7%，而 `--reuse 2` 不碰注意力就有 −45.9%（20 步实测） ⇒ 固定 keep 关掉。**
4. 时长收益 ≈ 23%×(1−keep)，与 F28 天花板一致；CPU 编码 3.0~3.2 s / 1800 次调用，
   印证 F30"多段 gather 编码不要钱"。
5. 遗留判据（若将来做 top-k）：keep ≤30% 时把 detail/sat 拉回 dense、layer0 video cos 从 0.80 → ~0.95+。
   时间窗做不到 ⇒ 远块带近块补不回的信息，这正是"选对的块"的假设前提。接线保留，将来只换取哪些 key。

回归：`make -j8 all` 干净、`./h3_tests` 1829 checks、`./h3_flash_attn_tests` 全过；
稠密路径未受影响（knob 不设时行为与改前一致，`99:0` 回退验证通过）。
产物：`/tmp/ab864/`（dense 与四档 keep 的 mp4/ppm/latent/日志）、`/tmp/dd|ds|de`（隔离增量 dump，已可删）。
本轮 wall 数在负载 7~16 的窗口测，只作同窗口相对比较。

## 续 19（2026-09-21 22:55）：reuse 收益 = 评估次数，等成本下均匀调度优于堆尾，记 F32

1. **先纠错**：F31/续 18/README/task_plan 里"`H3_REUSE_STEPS=2` 已 −45.9%"是归因错误 ——
   那是 CLI `--reuse 2` 在 20 步下的实测。`H3_REUSE_STEPS` 是自定义"哪几步真评估"的步号列表，
   且只有 `--reuse ≥2` 才生效（h3_dit.c:5042），此前从未被测、README 未写。四处已就地改正。
2. **等成本对照**（864×480/2 s、6 步、50 层、钉住 6 块、同种子）：
   base6(6 评估) 685.2 s / auto4(0,2,4,5) 480.7 s −29.8% / lated4(0,3,4,5) 542.0 s / r3(0,3,5) 355.7 s −48.1%。
   detail 7.51 → 6.77(−9.9%) / 5.73(−23.7%) / 5.99(−20.3%)；latent cos 0.916 / 0.888 / 0.860。
3. **结论**：收益只取决于"少做几次整网评估"（≈1−评估数/步数，与 20 步下 −45.9% 同规律）；
   等成本下均匀调度比堆尾好 13.8 pt detail ⇒ `H3_REUSE_STEPS` 保持高级覆写定位，不进预设。
   目视：`--reuse 2` 在 6 步下基本免费（同锐度同曝光，仅头位相偏），`--reuse 3` 开始付质量（眼睛糊、胸前拉丝）。
4. **窗口自检**：`s/评估` 列 114.2/120.2/**135.5**/118.6 暴露 lated4 整段慢 12% ⇒ 那一档的时长不与 auto4 可比，
   只有质量可比（两边做的功相同）。F30 的教训在渲染路径上同样成立。
5. 默认建议：`--steps ≥4` 保持 `--reuse 2`；要更快用 `--steps 6 --reuse 3`，而不是砍注意力（对照 F31：
   reuse −29.8% 时 detail −9.9%，稀疏 −14.7% 时 detail 已 −37%）。

本轮无代码改动（只改文档/计划文件）。产物：`/tmp/reuse/`（四档 log/bin/mp4/帧 + analyze.py），约 200 MB。
磁盘仍紧张（本轮起 6.0 GiB 可用），下轮前先确认是否清理。

## 续 20（2026-09-21 23:17）：detail 拆主体/背景 + lated4 目视，README 补"更快档"，记 F32 §6

零 GPU 开销的一轮（只用 `/tmp/reuse/*/frame-*.ppm`），因为 F32 只报了全图梯度能量，
而 `lated4` 那一档我根本没看过帧。

1. **主体/背景拆分**：在 `base6` 上取固定掩膜（多帧平均 chroma>60 且 R−B>20，占画面 20.0%），
   同一掩膜套四档，梯度只在"两端同区"的相邻像素对上求。全图列与 F32 §2 完全吻合（一致性检查通过）。
   主体损失：auto4 −19.0% / lated4 −37.4% / r3 −43.6%；背景：−4.2% / −15.0% / −5.6%。
2. **三条结论**：① 外推过期先塌的是运动主体，不是背景（r3 背景几乎不动）；
   ② 全图 detail 被低梯度背景稀释，会低估主体损失一倍以上 ⇒ 以后 reuse/稀疏 A/B 至少报主体区域梯度；
   ③ 形状结论方向不变（两区域都是均匀优于堆尾），但"均匀 ≈ 免费"收回 —— 均匀档主体也掉 19.0%。
3. **目视**：`lated4` 帧 8 构图与曝光正确，眼圈/口鼻明显偏软、胸前毛偏泥，与主体列一致。
4. **文档落地**：README §4 预设表补 `--reuse` 在 2..3 步无步可跳；新增"更快档"示例
   `--steps 6 --reuse 3`（−48.1%，附"掉的是运动主体"的代价说明）；
   「Sampler and DiT controls」补主体/背景两列数字，并把"低步数 keep --reuse 1"改成准确说法。

本轮无代码改动。脚本 `/tmp/reuse/regions.py`。

## 续 21（2026-09-21 23:58）：等预算对照 —— 4 步全新鲜 优于 6 步+reuse 外推，记 F33

按计划做待办 1：同样 4 次整网评估，预算买"更密 σ 网格 + 外推"还是"更粗网格但每步真算"。
864×480/2 s/50 层/同种子，两档同窗串行（`/tmp/eqcost/matrix.sh`）。

1. **成本按构造相等**：两档 profile 计数完全一致（`submissions=180 direct=628 linear=800 attention=200`），
   外推那两步不跑网络。
2. **质量：少步密集更好**。主体梯度 −12.2%（`--steps 4 --reuse 1`）vs −19.0%（`--steps 6 --reuse 2`）；
   latent cos 0.9656 vs 0.9158；运动更平滑（帧间差 3.814 vs 4.605，二阶差 5.727 vs 6.659）。
   唯一反向是背景梯度（−7.1% vs −4.2%），幅度小。目视一致：密集档口鼻须毛/胸前毛更利落。
3. **免费拿到跨窗口对照**：`--steps 6 --reuse 2` 是 F32 `auto4` 的完全重复 ——
   质量指标**逐位相同**（6.768/12.027/5.483/43.657/0.9158），wall 差 12.6%（480.7 → 419.9 s）。
   ⇒ F30 规则再确认：质量可跨窗口比，wall 不行。另有**同窗顺序效应**（先跑 119.8 s/评估、后跑 105.0），
   一部分能归到流式预热（cpu-unrotate 44.8→37.1 s、video VAE 110.1→103.4 s）。
   所以本轮不解读"479 vs 420 = 密集档更慢"。
4. **默认建议改口**：低步数档砍 `--steps` 并保留 `--reuse 1`；`--reuse` 留给 20 步档（那里网格密、跳过近乎免费）。

文档：README §4 预设表 + "更快档"示例改为 `--steps 4 --layers 50 --reuse 1`（附等成本数据），
「Sampler and DiT controls」补等成本一段并修正"低步数怎么砍"的说法。本轮无代码改动。
产物：`/tmp/eqcost/`（两档 log/bin/mp4/帧 + analyze.py/flicker.py），约 60 MB。

## 续 22（2026-09-22 00:26）：576×320/1 s 复现等成本对照 —— 密集档优势更大，外推档把主体压塌且指标抓不到，记 F34

按计划做待办 1（给 F33 补外部有效性）。同窗三档：`d6`（6 步全新鲜，参照）、`s4dense`（4 步全新鲜）、
`s6reuse2`（6 步 4 次评估），576×320/1 s、同种子。

1. **F33 复现且更强**：`s4dense` 全图 detail 只比 6 评估参照差 −0.7%（864 上是 −9.1%），
   `s6reuse2` −10.7%；latent cos 0.9682 vs 0.9171。
2. **外推档在该 shape 上结构崩坏**：帧 0/6/10/16 全是压塌的双头狐身，不是抖动也不是换构图。
   ⇒ F32"6 步下 `--reuse 2` 目视免费"不跨 shape 成立；低步数档推荐收紧为 `--steps 4 --reuse 1`。
3. **指标盲区（本轮最值钱的一条）**：崩坏的 cos 0.9171 ≈ 864 那次可接受的 0.9158，sat 只 −1.7%。
   cos/detail 量的是"与参照轨迹的偏离度"，分不出"不同但合理"和"结构错误"
   ⇒ 质量判定必须目视，指标只当筛选器。顺带：帧间差在两 shape 里方向相反，也不能当规律。
4. **口径警告**：F32 §6 的主体掩膜（chroma>60 且 R−B>20）在 864 上选 20.0% 狐身，
   在 576 这张草地全身图上选到 71.1%，且"主体"梯度反而低于"背景"（9.587 vs 18.210）—— 抓的是草叶。
   ⇒ 主体/背景拆分只在同一 shape 内有意义，跨 shape 不可照搬。
5. **计时学补强**：该窗三档 s/评估 32.2/32.5/32.7（极差 1.5%），等成本两档 wall 130.2 vs 130.9（差 0.5%）。
   反证 F33 里 864 的 14% 差是窗口/顺序噪声 ⇒ 用 s/评估 极差当场判窗口是否可用（>5% 就不报绝对时长）。

文档：README §4 与「Sampler and DiT controls」都补了 576 崩塌、指标盲区和"低步数 `--reuse≥2` 是风险开关"。
本轮无代码改动。产物 `/tmp/eq576/`（三档 log/bin/mp4/帧 + analyze.py + `*-tile.png`），约 40 MB。
未定：20 步 + `--reuse 2`（README 默认档）在小 shape 上是否安全，没测。

## 续 23（2026-09-22 07:12）：20 步 + `--reuse 2` 在 576×320/1 s 复验通过，默认档站住，记 F35

F34 之后必须补的一块证据：README 默认档（20 步 + `--reuse 2`）此前只在 864×480/2 s 与 512 方形上验过，
而 6 步档刚在 576 上崩过。同窗跑 `--steps 20 --reuse 1`（参照）与 `--steps 20 --reuse 2`（11 次评估）。

1. **省 46.6%**（644.0 → 343.6 s），与 1 − 11/20 = 45% 的规律吻合；窗口 s/评估 32.2/31.2（极差 3%，可用）。
2. **目视无差**：帧 0/6/10/16 拼贴里两档是同一只站立红狐、同姿态同锐度，`reuse` 档只是毛色偏冷。
   全图 detail +1.4%、sat −3.0%、latent cos 0.9767。
3. **结论收口**：`--reuse` 的危险变量是**新鲜评估次数**，不是步数 ——
   4 次评估在小 shape 塌主体（F34），11 次评估在同一 shape 无差（本轮）。
   ⇒ 低预算用"少步 + `--reuse 1`"（F33/F34），20 步档继续默认 `--reuse 2`；中间 8~16 步未测，不外推。
4. 顺带两条方法论：cos 与目视这次同向（0.9767 安全 vs 0.9171 崩塌），但 864 的 0.9158 目视可接受
   ⇒ 单点同向不构成阈值，cos 只当筛选器；帧间差在 6 步/20 步两例里方向相反 ⇒ 不能当质量判据。

文档：README §4 与「Sampler and DiT controls」都补了"危险变量是新鲜评估次数"和本轮实测。
本轮无代码改动。产物 `/tmp/eq20/`（两档 log/bin/mp4/帧 + `*-tile.png`），约 60 MB。

## 清理记录（2026-09-22 07:25）：删掉已关闭线的 `/tmp` 中间产物

按用户要求清 `/tmp`。只删**本轮系列自己做过的、结论已归档在 findings 里**的实验产物：

- `/tmp/ab864/`（130 MB）：F31 固定 keep 四档 + dense 的帧/latent/mp4 —— 该线已由 F31 判死，数字全在 F31 §4。
- 散落的早期 smoke 产物：`/tmp/ab_{dd,de,dense,ds,eq,sp,sp1}.mp4`、`/tmp/ab_sp.bin`、
  `/tmp/k.mp4`、`/tmp/g99.mp4`、`/tmp/k_1:{1,2}.log`、空目录 `/tmp/dumps`。

**保留**（仍被 findings 的"复现"小节引用，且分析脚本要读）：`/tmp/reuse`（F32/§6，100 MB）、
`/tmp/eqcost`（F33，49 MB）、`/tmp/eq576`（F34，34 MB）、`/tmp/eq20`（F35，21 MB）、`/tmp/mid`（进行中）。
⇒ F31 §6 里指向 `/tmp/ab864/` 的复现路径已失效：要重跑按 F31 §6 的命令重新生成即可（结论不会变，
该线已关闭）。若还要继续腾空间，下一个最可牺牲的是 `/tmp/eqcost`（F33 的结论已由 F34 在第二个 shape 复验）。

## 续 24（2026-09-22 08:05）：中间档补测完成 —— 7 次新鲜评估下两种调度都合格（F36）

576×320/1 s、50 层、同种子、`caffeinate -is` 串行同窗：
`--steps 7 --reuse 1` 与 `--steps 12 --reuse 2`（各 7 次新鲜评估）。
等成本由算子计数证明：两档都是 `submissions=315 direct=1099 linear=1400 attention=350`。

| 配置 | steps | 评估 | Euler denoise | s/评估 | detail | sat | 帧间差 | lat cos |
|---|---|---|---|---|---|---|---|---|
| `--steps 20 --reuse 1`（参照，复用 /tmp/eq20/d20） | 20 | 20 | 644.0 | 32.2 | 12.743 | 70.605 | 1.921 | 1.0000 |
| `--steps 7 --reuse 1` | 7 | 7 | 225.9 | 32.3 | −1.9% | −7.6% | 3.141 | 0.9039 |
| `--steps 12 --reuse 2` | 12 | 7 | 219.6 | 31.4 | +0.9% | −5.8% | 2.692 | 0.8719 |

窗口自检：s/评估极差 2.8% ⇒ 可用。目视（帧 0/5/11/16 拼贴）两档都是单头四腿、毛皮锐利的完整狐狸，
无崩塌；但两档的姿态都与 20 步参照不同（低头叼食 vs 抬头站立）。

结论四条：
1. 崩塌阈值夹在 **4 与 7 之间** ⇒ README 原先"低于约 8 档把 `--reuse ≥2` 当风险开关"过保守，已改。
2. F33/F34 的"少步密集优于多步外推"是 **4 次评估处的端点效应**，7 次处两档在噪声内互有胜负。
3. 帧间差第三次翻方向 ⇒ 正式废弃为质量判据。
4. 低评估数的真实代价是**内容漂移**（换了姿态），不是变糊 —— 对"要复现某条片子"的用法比百分位重要。

文档：README §4 预设表一行改写 + 新增中间档段落；「Sampler and DiT controls」同步补一段。
本轮无代码改动。产物 `/tmp/mid/`（两档 log/bin/mp4/帧 + `*-tile.png` + `analyze.py`/`tile.py`），约 9 MB。

环境记一笔：本轮 VAE 解码明显变慢，`ps` 显示渲染进程长期 `UN`（不可中断 I/O 等待）、
36 s 窗口内 CPU 时间只推进 1.1 s；swap 6.77/7.17 GB、内置盘 2.4 GiB。
⇒ 计时轮的墙钟会被内存压力污染，下轮开跑前先看这两项（s/评估 极差是当场判据）。

## 续 25（2026-09-22 08:20）：5/6 次新鲜评估补测完 —— 都不崩，但"次数 vs 步数"没拆开（F37）

按用户指示补 F36 留的空档：576×320/1 s、同种子同窗，`--steps 8 --reuse 2`（5 evaluations）
与 `--steps 10 --reuse 2`（6 evaluations）。密集档不重复测（4 次、7 次都已证安全）。
算子计数按评估数线性缩放（225/785/1000/250 与 270/942/1200/300）。

全阶梯（同锚 `d20`，20 步全新鲜）：

| steps | 评估 | Euler denoise | s/评估 | detail | sat | lat cos | 目视 |
|---|---|---|---|---|---|---|---|
| 6 | 4 | 130.9 | 32.7 | −14.4% | −10.6% | 0.8655 | **塌主体** |
| 8 | 5 | 161.4 | 32.3 | +10.2% | −6.7% | 0.8557 | 完整 |
| 10 | 6 | 193.9 | 32.3 | −0.4% | −3.8% | 0.8709 | 完整 |
| 12 | 7 | 219.6 | 31.4 | +0.9% | −5.8% | 0.8719 | 完整 |
| 20 | 11 | 343.6 | 31.2 | +1.4% | −3.0% | 0.9767 | 完整 |

窗口自检：五档 s/评估极差 4.8%（阈值 5% 内）。收益：5 次评估省 −74.9%，6 次省 −69.9%。

关键结论（三条）：
1. **崩塌只覆盖 `--steps 6 --reuse 2` 这一个配置**，5 次评估就正常 ⇒ F36 的"阈值夹在 4~7"应收窄。
2. **但决定量仍未证**：步数与新鲜次数在所有档位上完全共线（6/4、8/5、10/6、12/7、20/11），
   没有任何一档把两者拆开。判别实验 = `--steps 8 --reuse 3`（4 次评估、网格仍是 8 步）：
   安全 ⇒ 决定量是步数/网格粗细；崩 ⇒ 决定量是新鲜次数。**运行时闸的写法取决于这个答案，先不写代码。**
3. **两个指标当场失效的新证据**：cos 在"崩(0.8655) vs 安全(0.8557)"这一对上是**反的**，
   且 4~7 次整段几乎平（0.856~0.872）；detail 在安全的 5 次档给出 +10.2%（比参照还高）。
   帧间差第四次翻方向。⇒ 目视是唯一判据（F34 规则的第四次确认）。

文档：README §4 与「Sampler and DiT controls」把"危险变量是新鲜评估次数"降级为
"passes that run the network 决定成本 + 次数/步数未拆开 + 判别实验"，预设表一行同步。
本轮无代码改动。产物 `/tmp/boundary/`（两档 log/bin/mp4/帧 + `*-tile.png` + `analyze.py`/`tile.py`）。

## 续 26（2026-09-22 08:48）：判别档跑完 —— 4 次新鲜评估换网格仍崩，决定量确认为新鲜次数（F38）

`--steps 8 --reuse 3`（576×320/1 s、同种子；日志核对 `reuse schedule has 4 evaluations`，
调度 0,3,6,7；算子计数 180/628/800/200，正好是 4 次的量）。

| 对照 | steps | 评估 | detail | sat | lat cos | 目视 |
|---|---|---|---|---|---|---|
| `--steps 6 --reuse 2` | 6 | 4 | −14.4% | −10.6% | 0.8655 | 塌主体（双头） |
| `--steps 8 --reuse 3` | 8 | 4 | −18.7% | −17.2% | **0.8181** | **面部畸形 + 全身条带** |
| `--steps 8 --reuse 2` | 8 | 5 | +10.2% | −6.7% | 0.8557 | 完整狐狸 |

⇒ **共线拆开了**：固定 8 步网格、只把新鲜次数压回 4，渲染同样坏（而且三项指标更差）。
所以"崩不崩"由**新鲜评估次数**决定，不由步数/σ 网格决定 ⇒ F37 §2.2 的疑问关闭，
F35 的原表述以有判别实验的版本恢复；边界收紧为 **4 崩 / 5 起安全**。

计时注记：本档 s/评估 28.0，比同窗其它档（31~33）低 13% ⇒ 单档墙钟跨窗不可比，本轮只用目视与比值指标。
指标注记：cos 本轮与目视同向（0.818 崩 < 0.856 安全），但 F37 已有 0.8655（崩）> 0.8557（安全）的反例
⇒ 同向是巧合，不是阈值；帧间差第五次失效（崩溃档 1.530 全阶梯最低，只因四帧近乎静止）。

文档：README §4 与「Sampler and DiT controls」把"仍未证/待判别"改为已定结论（4 崩两种网格、5 起安全），
预设表一行改为"低于 5 次新鲜评估会坏主体"。本轮无代码改动。产物 `/tmp/discrim/`。

**下一步待拍板（行为改动，不自行实施）**：运行时闸。判据已可定死 ——
`reuse_interval >= 2` 且算出的新鲜次数 < 5 时，降级为 `--reuse 1` 或直接报错。
覆盖组合：`--reuse 2` 的 steps 4..7（3~4 次）、`--reuse 3` 的 steps 6..10（3~4 次）。
现状只有 main.c:601 一条 warning。需要用户选：报错 / 静默降级 / 降级+提示 / 不加。

## 续 27（2026-09-22 11:50）：运行时闸落地 —— 新鲜评估 <5 时降级为全新鲜并打印说明

按用户拍板（"降级为 --reuse 1 并打印提示"）实现。

**位置**：`h3.c` 的 `h3_generate`，紧跟 `h3_valid_params` 之后（约 1424~1438 行），
不是原先计划的 `main.c`。原因：交互式 REPL 走 `h3_cli.c:621` 自己写 `params.denoise_reuse`，
只挡 `-p` 一次性路径会留一半漏洞；放在库里一处覆盖 CLI、REPL 和直接调库的调用方。
`main.c` 里那条按"步数 2..7"触发的旧 warning 已删除（它既误报也会漏报：真正的主因是新鲜次数）。

**判据**：`denoise_reuse > 1` 且 `H3_REUSE_STEPS` 未设 ⇒ 调 `h3_dit_reuse_schedule(steps, reuse, ...)`
拿真实评估数，`< 5` 就打印 `h3: reuse R at N denoising steps leaves only K fresh model
evaluations, which breaks the subject; running all N passes instead`，并把 `eff.denoise_reuse` 置 1。
`H3_REUSE_STEPS` 是自选调度的专家开关，明确不拦。

**端到端验证**（576×320/1 s、同种子、`caffeinate -is`，`/tmp/guard/matrix.sh`）：

| 档 | 闸 | latent sha256[:16] | 结论 |
|---|---|---|---|
| `--steps 6 --reuse 2` | 触发（4 次） | `55379884d32575da` | 与下一档逐字节相同 ⇒ 降级等价于"跑满 6 步" |
| `--steps 6 --reuse 1` | — | `55379884d32575da` | 基线 |
| `--steps 10 --reuse 2` | 不触发（6 次） | `6d754b612d098a6c` | 与 F37 的 `/tmp/boundary/s10x2.bin` 逐字节相同 |

顺带又确认一次：质量侧输出跨窗口完全确定（第三、四次印证 F30/F33 的"只有墙钟不可跨窗比"）。
`--steps 8 --reuse 3`（4 次）与 `--steps 4 --reuse 3`（2 次）、`--steps 7 --reuse 2`（4 次）、
`--steps 2 --reuse 2`（2 次）、`--steps 9 --reuse 3`（4 次）都触发；
`--steps 11 --reuse 3`（5 次）与 `--steps 12 --reuse 2`（7 次）不触发 —— 用假模型目录即可看到提示行，
不需要真渲染。

编译：`make h3` 干净，`h3.c`/`main.c` 无新增告警（`main.c:20` 那条 usage 字符串超长的
`-Woverlength-strings` 是既有问题，与本轮无关）。
文档：README §4 补了闸的行为与 `H3_REUSE_STEPS` 例外。产物 `/tmp/guard/`。

## 续 28（2026-09-22 13:05）：未提交改动按主题拆成 5 个提交

用户拍板"按你的方案拆开提交"。`git status` 实际比上次的记录更多：除十余项 modified，还有 21 项
untracked（`h3_ane_*`、`h3_convrot.*`、7 个 ANE gate、2 个 bench、`test_int8_raw.c`、2 个 dbg 脚本、
`ANE_PORT_SUMMARY.md`）。先扫过一遍密钥（`api[_-]?key|typesafe|sk-|bearer`）—— 无命中。

拆分结果（`d52faa1` 之后，未 push）：

| 提交 | 主题 | 文件 |
|---|---|---|
| `0b9c038` | 矩形 SDPA + 时序块稀疏原型 + 逐算子剖析与基准 | `h3_dit.c`、`tests/{test_flash_attn,bench_attention,bench_mlp_fusion}.c`、`h3_gpu.{h,m}` 的矩形部分、README 稀疏/剖析两节 |
| `b6bb4a0` | ANE 移植（默认关闭） | `h3_ane_{bridge,linear,block}.*`、`h3_convrot.*`、`h3_weights.{c,h}`、`h3_video_vae.c`、`h3_shaders.metal`、`h3_gpu.{h,m}` 的 ANE 暂存部分、ANE/convrot/int8 测试、`dbg_*.py`、`ANE_PORT_SUMMARY.md`、Makefile 的库源文件+`-framework IOSurface` |
| `133cdc5` | 构建接线 | Makefile 余下 7 个 hunk（测试目标、`DIT_MODEL`、`make test` 列表、`clean`、`ACCELERATE_NEW_LAPACK`） |
| `2aafd31` | reuse 低预算运行时闸 | `h3.c` +16 / `main.c` −6 |
| `27081aa` | F36~F38 测量归档与低预算文档改写 | `README.md`（reuse 三处）、`findings.md`、`progress.md`、`task_plan.md` |

**混合文件按 hunk 拆**：`h3_gpu.h`（矩形声明 vs ANE 声明）、`h3_gpu.m`（H3SDPA keyShape + SDPA 矩形化 vs
dispatch 名单 + ANE 暂存实现）、`Makefile`（库接线 vs 测试目标）、`README.md`（稀疏/剖析文档 vs reuse 文档）。
工具是临时写的 `/tmp/split/hunk.py`（列 hunk / 按序号拼出可选子集，喂 `git apply --cached`）；注意部分暂存后
`git diff` 变成"相对 index"，hunk 序号会重排，第二个提交那批一开始按旧序号选导致空补丁。
`h3_dit.c`、`h3_weights.*`、`h3_video_vae.c` 无交叉，整文件走。

**校验**：① 每个中间提交用 `git archive` 解到临时目录，对 `h3_gpu.m`/`h3_dit.c`/`h3_video_vae.c`
`-fsyntax-only`（A、B 两档均通过），并对暂存后的 `h3_dit.c`/`h3_video_vae.c` 做符号对照（调用的
`h3_*` 全部能在同提交可见的声明里找到）—— 保证拆分不是"看起来分好了"；② C 档 `make -n test`
能展开新目标；③ `git diff d52faa1..HEAD` = 37 files / 13470+ / 73−，与原工作区改动（tracked 16 files
5218+/73− 加 21 个新文件）一致，工作区现已干净。
ANE 有 4 个新文件在磁盘上是 755 位（`h3_ane_block.{h,m}`、`tests/test_ane_block.c`、
`tests/test_ane_full_block.c`，拷贝带来的可执行位），已按原样入库，未顺手改权限。

## 续 29（2026-09-22 13:20）：注意力 top-k 线开了一次离线闸，判死在打包粒度上（F39）

用户"继续：挂起的注意力 top-k / 逐层 keep 表" ⇒ 把这条线从挂起里取出来，先做一次不渲染的可行性闸。

**做了什么**
1. `h3_dit.c` 加 `H3_DUMP_ATTN_LAYERS`（默认 off）：抓 norm/RoPE 之后、SDPA 真正消费的
   Q/K/V 全量行（2293×7168 bf16，每层每字段一次），并把原 `dump_activation` 里的
   "读回→写文件"抽成 `dump_bf16_capture` 供两处共用。`make h3` 干净（新增变量曾撞
   `-Wshadow`，改名后无新 warning）。
2. 渲染抓数：`/tmp/topk/dump.sh`（576×320/1 s、`--steps 2 --reuse 1`、层 0/12/25/38/49、
   `H3_DISABLE_HEAD_MAJOR_ATTENTION_OUTPUT=1`）。第一次排队失败：`--steps 1` 被参数校验拒
   （`[2, 1000]`）；改 `--steps 2` 后 exit 0，落 15 个 attn_*.bin × 32.9 MB。
   另一会话在同一仓库同一二进制上渲染时**未抢占**，用 `/tmp/topk/waitrun.sh` 等空闲。
3. 离线 harness 四个脚本（纯 numpy，不渲染）：`study.py` 逐行逐头三选择器 + 与引擎
   `out.NN.bin` 位对拍（**5 层全 cos=1.0000**，先证明解码/布局/通路对）；
   `study2.py` 可打包形式 oraclehs/poolhs/glob；`study3.py` 同 harness 重算固定窗 +
   全体头顶并集；`study4.py` G 组（1/2/4/8/16/56）的 keep↔cos 曲线。
   三个脚本先用合成数据（故意长程耦合）做过冒烟测，能区分形式好坏才敢上真实 dump。

**结论（判死，且死因与 F31 不同）**
- 可打包形式里 `poolhs ≈ oraclehs`（L0 的 13.7%/29.4% 两档三位小数相同）⇒ 打分器不是瓶颈，**跨头共享本身是瓶颈**，
  逐层 keep 表/学习打分不用再投。
- 与固定时间窗在等 keep 下：13.7% 档 top-k **差 0.014**、29.4% 档**好 0.006**（五层均值），
  L38 全程落后 ⇒ 在能打包的粒度上"看对的块"对"看近的块"没有净收益。
- 56 个头各取 top-2 的并集覆盖 **97.4%** 的 token ⇒ 头部偏好近乎不相交，有用的稀疏性长在 head/row 维。
- 逃生口 G=16（16 份 scratch、16× 调用）到 cos 0.95 需 keep ~42%(L0)/48%(L38)，
  按 F31 实测 ≈1.8 ms/次的编码，代价 +48 s 换收益 1.9~4.4% 时长 ⇒ **净负 4~7 倍**。
- 判据换成可复用的形式：同 harness 到 0.95 所需 keep = 固定窗 ~64% / poolhs ~55% / G=16 ~42~48%。

**顺手改的在案说法**：README 稀疏小节原文"留作将来 top-k 的载体"已被本轮否掉，就地改成 "F39 离线验过：可打包的选择形式与固定窗打平，稀疏性长在 head 维、打包装不下"。

**提交前的返工与复验**（用户拍板"一起提交"之后）：解析块改成无空分支的嵌套写法，并补一行
`h3: attention Q/K/V capture on for %u files...`（`all` 会写 150 个整序列文件，先报数量再落盘）。
`make h3` 干净。两条行为探针（`/tmp/topk/probe.sh`，跑完删产物）：
未设 `H3_DUMP_ACT` 时按预期打印 "needs H3_DUMP_ACT=<dir>; ignoring" 并忽略；
`H3_DUMP_ATTN_LAYERS=0` 重抓的 `attn_{q,k,v}.00.bin` 与 F39 用的那份 dump **逐字节相同**
⇒ 返工没有改动抓取语义，F39 的全部数字对提交后的代码依然成立。
（探针第一版失败：`export` 不跨 Bash 调用保留，丢了 `H3_CLIPPROJ_DIR` 报
`text_encoder: no safetensors files`；把环境变量写进脚本才通。）

**提交**：用户拍板"一起提交"⇒ `h3_dit.c` 的 `H3_DUMP_ATTN_LAYERS`（默认 off）、README 稀疏小节、
`findings.md`/`progress.md`/`task_plan.md` 同一个提交，未 push。

## 续 30（2026-09-26 16:00）：AdaLN 调制缓存落地（F40）—— 省掉 24.29 GiB 权重，并预生成 4/8/20 三档可自由切换

起点是上一轮的产物：一个由 mere-run q4 转来的 4-bit grouped checkpoint 有 35.93 GiB，
而 mere-run 自己那份只有 10.55 GiB。差额算清楚了 —— **24.288 GiB 全是 51 个
`adaln_proj.linear` 矩阵**，mere-run 靠 `cache_covered_weights_omitted: true` 把它们整个省掉，
改成随 checkpoint 携带 AdaLN 的**输出**（调制）。本轮在 h3.c 里把同一招做出来，并顺手把
"换步数"这件事做成开箱可用。

**为什么能省**：`h3_dit_schedule_precompute` 本来就在 denoise 之前**一次性算完所有 step 的
AdaLN 调制**，逐块投影、算完立刻 `free_tensor(&weight)` —— 24.288 GiB 权重**只被读一次**。
所以 checkpoint 完全可以只存那份产出（每行 9.25 MiB），不存权重。

**契约**（`h3_dit_schedule.h`）：缓存**按 `--steps` 分键**，同一目录可并存多份 ——

```
adaln_cache_times_s{steps}        F32  [rows]      ← 校验键
blocks.N.adaln_cache_s{steps}     BF16 [rows, 96768]
final_layer.adaln_cache_s{steps}  BF16 [rows, 10752]
```

`blocks.0.adaln_cache_s{steps}` 存在即选中缓存路径，**不需要任何开关**；文件名
`adaln_cache_s{steps}.safetensors`。`h3_st_inventory_dir` 只按 `*.safetensors` 后缀扫目录，
天然认得新文件，加载侧无需改动。校验是把本次 sigma schedule 算出的 `times[]` 与缓存键
**逐位 memcmp**，要求是**前缀**；太短或不匹配就硬报错，绝不静默套错调制。

**关键发现：三种条件模式的行数是嵌套的**。条件行**追加在 step 行之后**，顺序固定
visual 再 audio，所以：

| 模式 | time_rows | 条件行 |
|---|---|---|
| text-to-video | `2*steps - 1` | 无 |
| 首/末帧 I2V | `2*steps` | visual |
| 参考图 Ref2VA | `2*steps + 1` | visual + audio |

（是 `2*steps - 1` 不是 `2*steps`：第 0 步 video==audio，只占 1 行。实测 `--steps 2` → 3 行、
`--steps 8` → 15 行，两次都对得上。）

因为短模式的 `times[]` 恰好是长模式的**前缀**，一份缓存能服务全部三种模式 —— 前提是导出时
把两个条件行都算出来，所以 `H3_DIT_ADALN_CACHE_DUMP` 现在**同时强制
`visual_condition = audio_condition = 1`**。

**修掉一个死循环陷阱**：先前只强制了 visual。这样 Ref2VA 需要 `2s+1` 行而缓存只有 `2s` 行，
会被拒绝并提示"重建缓存"—— 但重建出来的还是 `2s` 行，**永远修不好**。同时强制两个条件行后
缓存是 `2s+1` 行，三种模式分别按前缀/精确命中，代价是两行用不上的 ~20 MiB。

**顺手补的报错**：缓存专用 checkpoint 连 `blocks.0.adaln_proj.linear.weight` 都没有，跑一个
没缓存的步数时原来只报 `required weight is absent: blocks.0.adaln_proj.linear.weight`，看不出
"这个模型只是没存这一档"。新增判断 `!cache_tensor && !adaln_probe`（只在注定失败的路径上成立），
改成 `this checkpoint ships AdaLN caches but none for N steps, and it has no adaln_proj weights
to fall back on; export a cache at these steps with H3_DIT_ADALN_CACHE_DUMP`。

**纠正一处早先的误读**：上轮曾记"缓存模式下 `schedule->blocks[]` 装的是调制而不是权重，
所以 gate 排名会坏"。**这是错的**：非缓存路径下 `h3_gpu_linear_bf16` 也把投影结果写进
`schedule->blocks[]`，两条路装的都是调制；`h3_dit_schedule_gate_scores` 读的正是调制
slot 2/5。缓存逐位复现同一内容，且 `time_rows` 取本次运行的值，所以 `!layers N` 块剪枝在
缓存模式下**结果完全一致**。

**导出与验证**。dump 格式 `magic[8]="H3ADALN2"` + `uint32[6]={steps, time_rows, 50, 96768,
10752, 0}` + `f32 times[rows]` + `u16 blocks[50][rows*96768]` + `u16 final[rows*10752]`，头 32 B。
预测 `32 + rows*9,698,308` 字节，三档**逐字节命中**：

| steps | time_rows | dump 字节 | 预测 | md5 |
|---|---|---|---|---|
| 4 | 9 | 87,284,804 | ✓ | 13d2bdb3… |
| 8 | 17 | 164,871,268 | ✓ | 35f7a335… |
| 20 | 41 | 397,630,660 | ✓ | 73867699… |

裁剪：丢 102 个 AdaLN 张量（24.29 GiB），留 11.65 GiB，装 3 个缓存 619.7 MiB，
目录 **12.25 GiB**（原 35.93）。`--verify` PASS：3 份缓存各 52 键、schedule 键逐位、
block 0/49 与 final 逐字节、残留 `adaln_proj` 0 个。

A/B（同一 prompt/256×256/22 帧/seed 42，A = 全权重 `h3c-q4-native`，B = 缓存模型）：

| steps | A | B | latent md5 | mp4 md5 |
|---|---|---|---|---|
| 4 | 151 s | 119 s | 6e9c2d8786cea097ddf9bdd45e5593bd（同） | 0330454e…（同） |
| 8 | 250 s | 219 s | a7f85c8bfd6b8fe0282c4084adbad9c5（同） | 35e5fdb3…（同） |
| 20 | 556 s | 501 s | dc8c5d41f9779fc0b77e8172cf4603bd（同） | 569f5be2…（同） |

`relRMS=0.000000 cos=1.000000 max|d|=0`，**连 mp4 容器都逐字节相同** ⇒ 整条链路（含 VAE 解码）
无差异；省下的就是那 24.29 GiB 不再被读。steps=8 的 md5 与改成按步数分键**之前**那版完全一致，
反证这次重构与"多算两个条件行"没有改变数值 —— 因为条件行只追加在尾部，step 行的索引与取值不受影响。

负向对照：缓存模型跑 `--steps 6` → `exit=1`，报上面那条新错误。

**交付物**：`/Volumes/data/MODELS/h3c-q4-adalncache/FL2VA/transformer/` 含
`adaln_cache_s{4,8,20}.safetensors` + 18 个权重 shard，共 12.25 GiB；
用法 `./h3 -d /Volumes/data/MODELS/h3c-q4-adalncache --ssd-streaming ... --steps 8`。
代码：`h3_dit_schedule.{c,h}`（按步数分键 + 前缀校验 + 双条件导出 + 缺失提示）、
`fastvideo_qad/scripts/export_h3_adaln_cache.py`（`--dump` 可重复、按步数命名/校验）。
脚本与日志在 `/tmp/h3ab/v2/`。

**遗留**：缓存与 LoRA 互斥（`merge_adaln_loras` 要合并进已丢弃的权重，直接拒绝）；
纯音频参考（`H3_LAYOUT_REF_AUDIO` 无图）是唯一不覆盖的模式，会被 memcmp 明确拒绝，
该模式 CLI/REPL 都构造不出来。本轮改动**未提交**。

## 续 31（2026-10-01）：Strata 对照清单只做了前提核对（F41），三条事实断言两条与代码不符

用户给了一份 Strata（LLM offload）与 h3c 的对照优化清单，末尾要求"参考上面的优化"。清单里的
建议要成立，得先知道 h3c 现状是哪样，所以本轮只核代码、不改行为、不跑 GPU。

**核对结论**（行号与复现命令在 F41）：

| 断言 | 结论 |
|---|---|
| ① `--ssd-streaming` 盘上存/读 BF16，Strata 的收益来自"权重不反量化" | 只对 BF16 checkpoint 成立。ConvRot int8 checkpoint 的流式读就是 int8（h3_dit.c:1649），且 int8 计算保留（主线程 `requant_stream_slot`，h3_dit.c:1517）。真正没吃到的是 int8→BF16→int8 这圈往返（反旋转 16.0 s / 33.6 s 流）。清单还把 `--use-int8-row-fc2`（M5 专属 kernel 变体，h3.c:927 要求 Metal 4）当成"int8 整体" |
| ② uncached 读 = 放弃保留，h3c 缺"这次读本可以命中"这一维 | 前半句方向相反：`F_NOCACHE` 全仓仅一处（h3_gpu.m:823）且只在 BF16 helper，int8 流式读根本没设 ⇒ 保留交给 OS 且没记账。后半句成立，但 `H3_PROFILE` 还有个附带错误：不分路径都印 "BF16 SSD stream"（h3_dit.c:5673）而字节数是 int8 流量（1658）。另外已有 `load_core()` 自适应前 N 块常驻（2756-2853），只是按序号不按成本 |
| ③ 每张量 open/close 一个 fd | 成立（h3_weights.c:234-247 共 14 处、h3_dit.c:773-786），但粒度实际更细：开快速路径后按 `H3_STREAM_CHUNK_ROWS=1024` 切，约 5.25 MB/次 —— 不是 Strata #230 那种 2048×4 KiB，所以别按 0.03→3.25 GiB/s 的倍数预期 |

清单另两条按现有红线不排期：ANE 产物当专家层管理（线已按用户决定关闭）；"验证者=大模型本身"的
自验证无损优先闸 —— 我们自己的 F35~F38 恰好证明同族偏离度指标对崩塌会反向（崩的 0.866 vs 好的
0.856；崩的 0.9171 ≈ 合格的 0.9158），这类指标会把要防的那类错判成通过。

**下一轮入口**：task_plan.md 新"待办" 2~5 条，顺序仍是"免费描述修正 → 免费 profiling → 便宜原型"：
先修 README/h3.c/H3_PROFILE 三处与代码不一致的说法，再给流式读取补按 block 的首次 vs 后续 pread
带宽对比（命中率代理），然后才是常驻 fd 的 A/B 与逐块 FNV-1a 指纹。

**未提交状态**：本轮只写 F41 + task_plan 新小节 + 本条续记；F40 那轮的代码与脚本改动仍从
2026-09-26 挂起未入库。

## 续 32（2026-10-04）：评审 F40 AdaLN 缓存那轮，挑出 8 处可优化点（F42）

用户要求"审核代码，看看还有什么可以优化的"。评审对象 = 工作区未提交的 F40 那轮
（`h3_dit_schedule.{c,h}` +270/-25、`export_h3_adaln_cache.py` 451 行、`export_h3_q4_native.py`、
README 新小节 +73）。本轮不改行为、不跑 GPU；能本机证伪的都跑了，现场在 `/tmp/f42`。

**做对的**：`prepare_rows` 拆成 `prepare_times`+`prepare_features` 是真简化（缓存路径得以在读权重、
碰 GPU 之前先验 sigma 前缀）；该 TU 告警从 **11 条降到 1 条**（`git show HEAD:` 单独编译复核，
剩那条 :419 早于本轮）；前缀校验顺带自然拒掉唯一不覆盖的"无图音频参考"。

**8 条结论**（细节与行号在 F42）：

| 编号 | 结论 | 证据形态 |
|---|---|---|
| P1 | LoRA 一刀切（:688）拒所有适配器；但 `h3_lora_matches` 对不含 AdaLN 目标的一向是警告跳过 ⇒ `.default`（仅 attn.orig.*）本可共存，只有 `.turbo`（带 norm_out.linear）真冲突。README:1120 已文档化，属保守设计而非隐藏 bug；障碍是缓存档无 `adaln_proj` 探针、`time_dim` 判不了剪枝档 | 代码路径 + Phase 1 adapter 事实 |
| P2 | `H3_DIT_ADALN_CACHE_DUMP=`（空串）强制 2 个条件行却不导出，缓存档还印假警告 | **实测**（getenv_probe 三档） |
| P3 | 无图音频参考的报错说"rebuild the cache at these steps"，而导出永远把 visual 排前面 ⇒ 建议不可执行，正是 :676-678 注释自己批评的那类话术 | 代码路径 |
| P4 | dump 非原子 + 丢 errno；Python :196-199 用长度核对兜住，故不致命 | 代码路径 |
| P5 | `--trim-to` 缺 `--model` 抛裸 `AttributeError`（:441→:257），同文件 `--verify` 有守卫 | **实测** |
| P6 | trim 静默丢掉源目录已有的 `adaln_cache_s*.safetensors`（:265-266 整文件 skip），文档却说"只丢 adaln_proj 矩阵"；对输出跑 `--verify` 仍 PASS | **实测**（双分片假模型目录） |
| P7 | 缓存路径零测试；`tests/test_real_dit_schedule.c` 没接进 `make test`/`real-parity`（count-mode：Makefile 只 1 行即其自身规则）。dump↔safetensors 是 C↔Python codec，按规矩需双向锚定 fixture，原料本轮已做出 | 实测 + 代码路径 |
| P8 | `cache_header`/`adaln_header` 写了没读；无表分支按 2688 分配只写 256（不变量没写出来）；:277 没算 `sizeof(float)`（原同形） | 代码路径 |

**核过不是问题的一条，值得记下过程**：`st_write` 不补 8 字节对齐，真档 `adaln_cache_s20.safetensors`
里 51/52 张量偏移非 8 倍数（block 0 起于 164）。看着像缺陷，但用官方 `safetensors` 0.8.0 造了个
刻意错位的 F32 文件实测**能读**，h3.c 又走 pread ⇒ 判为"交给 mmap 零拷贝消费者时的潜在拷贝"，
不列入待办。

**中途自我纠正两处**：① 一度准备写"README 没提 LoRA 不兼容"，count-mode grep 到 README:1120 明写
"LoRA adapters are refused"，撤；② 一度把 :419 那条告警当本轮引入，用 `git show HEAD:` 单编复核后
确认早于本轮，改成"告警 11→1"记为改进。

**下一轮入口**：F42 §10 与 task_plan 新待办 2~4 —— 一批做完 P2/P3/P4/P5/P6/P8（十行内、无需 GPU
轮次，P5/P6 各配负例）→ P7 单独一轮（codec 双向锚定 + 接入 make）→ P1 只在需要 default adapter 时做。

**未提交状态**：本轮只写 F42 + task_plan 新小节 + 本条续记；F40 的代码与脚本改动仍从 2026-09-26
挂起未入库（F41 待办 1 至今未做）。

## 续 33（2026-10-04）：F42 的 8 条优化全部落地，补了两个闸并接进 make（F43）

零渲染轮次。改完 `h3_dit_schedule.{c,h}`、`export_h3_adaln_cache.py`、README、Makefile，
新增 `tests/adaln_cache_probe.c`、`tests/test_adaln_cache_codec.py`、
`tests/gen_adaln_cache_fixture.py`、`tests/test_adaln_cache_lora_gate.c`。

**最要紧的两条判断**
- **P1 用"可选键 + 缺省保守"实现**，而不是直接放宽：宽度键缺失时仍整批拒绝 ⇒ 已入库的三档缓存
  行为一字不变，raw dump 格式与魔数不动（所以不必重新导出，写 meta 只是导出器的磁盘步骤）。
  磁盘契约同时从 `.c` 的 enum 提到头文件，成了唯一定义处。
- **夹具靠"越过后必然停在 times 前缀检查"来证明闸放行**（schedule key 全填 -12345.0，任何
  `1-sigma` 行都取不到）：于是 2.1 MiB 的夹具既不需要模型也不提交 GPU 工作，而"gate accepted"
  仍是可观测事实而不是推断。

**验证**：`make test` exit 0（codec 绿、闸 8/8 绿、15 条 skip）；codec 的锚是外部的——C 的
`dump_bytes_41/17/9` 与三档真实已安装缓存的载荷字节逐行吻合。Mutation 共 9 条（codec 5 + 闸 4）
全部 red 且归因到被监视的那条判定。

**两处踩坑并当场纠正（都记进 F43）**
1. 第一批闸控制把 old/new 写反、又只在 `finally` 还原，导致控制 3/5 与"还原后基线"跑的其实是
   上一批的二进制（`RESTORED baseline RED!`）。改成每条控制前删 object+binary、跑完立刻还原、
   批后用 sha256 核对源码，重跑后 4 条全部正确归因、还原基线绿。
2. 想端到端复跑缓存档对 `6e9c2d87` 时撞墙：三棵模型树下都没有 `FL2VA/text_encoder`，F40 复现
   小节那条命令今天照抄直接报缺组件。**没拿到真实调用方式之前不声称端到端验证过**，并要把 F40
   小节标注为"命令不完整"。另外 `/tmp/h3ab/v2` 已被清掉，F40 引用的三个 raw dump 与 A/B latent
   都不在了。

**未实测的项（显式标出）**：`h3_real_dit_schedule_test` 的 `from_cache ? 0 : 52` 分支（本机 15 条
skip 包含它）；P2 的运行时效果（现有夹具观测不到，只按 getenv 探针 + 判据统一认定）。

**下一轮入口**：task_plan 新待办 1~4 —— 先要真实调用方式把端到端 md5 复跑补上，再决定是否为
s4/s8/s20 补写 meta，然后按主题提交这两轮改动。

## 续 34（2026-10-04 下午）：整仓评审出优化路径表（F44），只评审未改代码

用户把范围从"F40 那轮未提交改动"扩到**整仓**。做法：四个子系统并行派评审代理，brief 里写死
判死红线（ANE / F31+F39 稀疏注意力 / F32~F38 reuse / F26）与"行号必须真读到、未测就标未测"，
回来我逐行复核。**本轮不改代码、不跑 GPU。**

**复核成立 4 条**（细节与证据在 F44 §1）
- `h3_video_vae.c:934/952-959` 流式每块每 tile 两次全量 blit，而 `hidden` 是原地累加器
  （`:653`/`:663`）、state 与 hidden 同尺寸（`:1327`）⇒ 两条都能删。**两个代理独立命中同一处**，
  这是本轮最强的一条。
- `h3.c:2051-2057`/`:376-388`/`:781-792` 对"env 未设"给出三个相反答案，并把
  `/Volumes/data/.lmstudio/...` 本机绝对路径写死在库里 —— **F43 端到端复跑被卡的根因找到了**，
  不是用户少传参数。
- `h3.c:1375-1377` 拿 `HEADS=56` 当 latent 通道数（真实 `VIDEO_CHANNELS=24`）、
  `h3_dit.c:2771-2774` 按 16 预留 ⇒ 少留 1/3，误差方向是放松内存守卫。
- `h3_text_encoder.c:623-625` 把约 1.45 GiB 的整块词表 embedding 传上 GPU 只为 gather 几百行，
  且 pad 行随后被视觉 span 覆写。

**其余 14 条**按"代理报告、收益未实测"记在 F44 §2，没有混进结论。其中我特意下调了一条：
`requant_stream_slot` 末尾那次多余 submit 不是"删 5 行"的便宜事，它动的是全引擎依赖的命令缓冲
纪律（`:1511-1516` 的注释明确禁止 worker 碰共享 command buffer），必须先在 M5 上取数。

**方法论上守住的两点**：① 代理给的倍数（"8× 高估""19 GB/chunk""12% 块时间"）全部标成它们的
估算而非实测；② 三条 Metal kernel 优化在 M4 上根本测不出来（无 TensorOps），所以写成"先取 M5
数据再动"，不排在前面。

**下一轮入口**：task_plan 新待办 1 —— 先修 ClipProj 那三处判据（修完才补得上 F43 欠的端到端
md5 复跑），再依次是内存估算常量、VAE 双 blit、整块词表上载。上一轮遗留仍未动：F40+F42/F43
两轮改动未入库、s4/s8/s20 的 meta 键未写。

## 续 35（2026-10-04）：F44 待办 1 落地 —— ClipProj 一条规则，顺带查出静默缓存投毒（F45）

只做一个步骤：修 F44 §1.2。结论是**我上轮对这条的描述偏轻了**——三处判据不一致里有一处
不是"混乱"而是"静默错"：未设 `H3_CLIPPROJ_DIR` 时，生成路径先把目录填成内置默认再判
`use_clipproj`，于是必然用 ClipProj 编码；而缓存 key 那边未设记的是 `clipmodel=none`，
注释含义是"50 层编码器"。**下一个同样未设的进程会把 ClipProj 的条件当 50 层的结果读回去**，
key 是按编码器身份算的，没有任何一层能发现。此前看不见，只因为无 `FL2VA/text_encoder`
的树在 `h3_load_dir` 先 fatal。

**取证修正**（都是 count-mode 复核的）：机器路径是两条不是一条；README 对这两个开关
**0 处提及**；库默认指 BF16 的 4B 目录，而 ComfyUI 节点/benchmark/ab_quant 全指
`-int8-convrot` ⇒ 裸跑与脚本跑的是不同编码器。

**改法**：`classify`（纯函数）→ `resolve`（全仓唯一 getenv）→ `disk_identity`，装载/生成/key
三条路径共用；语义取"ClipProj 是 opt-in，DIR+PROJ 必须都给"，依据是三条独立文档证据
（`comfyui_nodes/__init__.py:19`、`h3_text_encoder.h:75-76`、`h3.c:781-784` 本来就这样），
且所有真实调用方都给全了变量 ⇒ 无现用法被改变。库里删掉所有默认路径。

**验证**：`tests/test_clipproj_selection.c` 10/10 并进 `make test`；mutation 4 条全红且归因正确
（一条被我的预检挡下——替换文本正好是基线行的子串，换写法才成立）；`make test` exit 0；
真机两证：裁剪树 + 两个变量 ⇒ 装载通过、ClipProj 真跑起来、停在 F40 的 6 步负向对照；
只给 DIR ⇒ 新报错。**F43 §6 的堵点因此解除**，欠的 4 步 md5 复跑具备条件了。

**两处诚实标注**：① `make test` 我只 `tail -50` 留了尾部，"12 条 skip"是截断后的下界，
全套无失败是靠 exit 0 证的；② `h3.c:494` 的死参数告警用 `git show HEAD:h3.c` 单编确认
**早于本轮**，与 `h3_dit_schedule.c:422` 同等对待，只记账不顺手改。

**下一轮入口**：task_plan 新待办 1（补 4 步 md5 + 修 F40 小节缺的那句前置），
再往下是 F44 的内存估算常量与 video VAE 双 blit。

## 续 36（2026-10-05）：从 splash 看 MiniMax-H3 的加速路径（F46），纯研究未改码

用户让看 `/Volumes/data/git/c/splash` 能给我们什么加速思路。splash 是 **Apple silicon 上的自回归 LLM 引擎**，
MiniMax-H3 是**扩散 Transformer**，所以每条机制都得问一遍"为什么省"还成不成立。
（说明：用户指定的 `docs-researcher` agent 类型本会话未注册，改用 `general-purpose` 承接同一份 brief。）

**能搬的三处**（分级标注，【核】=我读了原文/源码）：
1. 【核】内存规划：`hardBudget = min(--max-memory, recommendedMaxWorkingSet − max(1 GiB, 2%))`
   （`MemoryPlan.hpp:63-86`）——它自己也用百分比，差别在叠在 OS 实测量上、可用内存走
   `host_statistics64`、回收是带水位+双阈值迟滞的目标式。真正缺的一个是
   `contextTokensWithin(bytes)` 这种"**给定字节能撑多长**"的反查：h3c 只有正算，
   长片段撞墙和 80%/85% 静态判错的 panic 风险都来自这里。
2. 【核】注意力 K/V 从 BF16 到 INT8：**省容量不成立**（扩散无跨步 KV），**省带宽成立**
   （窗口内 K/V tile 被窗内各 query 重读）。splash 的 scale 是每 (head,token) 一个、
   读侧不跑反量化 pass（`KvExtent.h:43-59` 我读了）。但它自陈 BF16/INT8 之差不是通用提速
   （512-token ±0.8% 内），且 head_dim 它是 256 我们是 96、int8 operand 靠 M5 TensorOps
   ⇒ M4 吃不到，必须自测。
3. 【核】VDN 线性分支的扫描 kernel 有对口参考：我们的 `h3_vdn_scan_step`（`h3_shaders.metal:6170`）
   是"一线程一元素 + 串行 128 次 FMA"，splash GDN 是"lane 持连续 4 个 fp32 + `simd_sum` +
   threadgroup memory 相位交接（逐位不变）"。**但跨 lane 归约改求和顺序就改浮点结果**，
   splash 自己否掉过这类改动；且 VDN 是质量叠加（Phase 9 慢 3.4~7×），搬它是降低 VDN 价格不是加速基座。

**不能搬的三处**（省下轮次）：投机解码的"无损省步"——判据 `u·q < p` + 残差重抽要求候选有可算的
归一化密度 p，而扩散的"拒绝第 t 步要重跑第 t 步"，结构上就没有"一次前向验证 k 步保留前缀"；
`F_NOCACHE` 照搬——它在 splash 用于权重镜像写侧与 state disk tier，都不是"每步重读同一批字节"，
抄过来是主动放弃页缓存；token 稀疏注意力——只记一条事实：splash 的打包单位确实是
"一个 threadgroup = 一个 KV head × GQA 组若干行"，即 F39 说我们缺的更细单位，但不改变判死。

**方法学最值得抄的一条**：它的设备事实集中在 IORegistry + "只在原生机器实测到增益才采纳规则"
（模拟核数曾误判 −26%~+20%），而 h3c 的 M4/M5 分支现在散在运行时 if 里。

**未动的事**：F45 末尾问的"4 步 md5 复跑"还没跑；F44 待办 2 的三条（内存常量、VAE 双 blit、
整块词表上载）仍未动；F40+F42/F43/F45 四轮改动未入库。

## 续 37（2026-10-05）：F44 待办 2 三条落地，三次真机 A/B 证逐位不变（F47）

做了内存常量、video VAE 双 blit、整块词表上载三条。验证方式是**单文件 A/B**：
把某个文件回退到 HEAD、其余全不动、重建、跑同一条命令比 md5，再还原并核对哈希。

**结果**
- 回退 `h3_text_encoder.c`（整载全表）⇒ 与我改的按行 gather **latent 与 mp4 同 md5**。
- 回退 `h3_video_vae.c`（保留两条全量 blit）⇒ **mp4 `cmp -l` 差异 0 字节**。这条尤其要实测：
  删掉拷贝后 `vae->hidden` 会留着陈旧内容，静态审计（三个调用点之后都走会自行还原的
  `finish_chunk_states`）说得出理由，但只有 mp4 逐字节相同才算数。
- 三处改动全在的二进制 + int8-convrot 4B ⇒ latent 复现 F40 的 `6e9c2d8786cea097ddf9bdd45e5593bd`，
  **F43 待办 1 结清**。
- `make test` exit 0（46 行绿、15 skip、三个新闸全绿）。

**过程里两次被自己的记录坑到，都已纠正**
1. 头两次渲染得到 `a63e27a4…`，与基线不符。当时最容易下的结论是"我改坏了"；实际是
   **基线用的 4B 变体不同**（库内置默认是 BF16，本仓所有脚本和 F40 用的是 int8-convrot）。
   归因路径：先证明这次渲染不可能受 h3.c 影响（自动规划门槛 `ssd_streaming == 0`，而我们显式
   给了 `--ssd-streaming`）、又不可能受 VAE 影响（latent 在 VAE 上游），只剩文本编码器一条，
   才去做单文件 A/B；A/B 相等 ⇒ 差异必在基线侧，再换 int8-convrot 一击命中。
2. `/tmp/f42/render/` 被系统回收，一次后台渲染因此**静默失败**（日志文件都没生成）。
   产物挪到 `/Volumes/data/tmp/h3scr`，并把要留的锚写成 md5 而不是文件路径。

**一条不当成"修好了"交付的保留意见**：`h3.c` 的 activation 项虽然算对了通道与时间压缩，
但**新旧两式都没建模 DiT 逐块激活**（长片段是 GB 级主导项），而且修正让预留变小、
档位方向更激进。已写进 F47 并进待办 1，等和"实测可用内存 + 反查可撑帧数"一起做。

**可复用的逐位锚**：256²/22f/steps4/seed42 + int8-convrot 4B ⇒
latent `6e9c2d8786cea097ddf9bdd45e5593bd`、mp4 `ac0d0940434ad216d8131f7469154b70`。
F40/F45/F47 的 env 前置也已写回 F40 的复现小节。

## 续 38（2026-10-05）：校准实验推翻了自己的前提，抓到内存墙的真实主因（F48）

要做的本是"给 activation 补主导项并用 H3_PROFILE 校准"。校准跑了两个形状（256²、22f 与 44f，
同 seed 同 steps），结论是**主导项假设错了**：

- 44f 那次**被 SIGKILL（rc=137）**，日志停在 `denoise 0/4`；它自适应钉了 **13/50 块**，
  DiT load 阶段 peak 已达 10.995 GiB（16 GiB 机器）。
- 两点求斜率：`Δpeak 2.955 GiB / Δ4 块 = 0.739`；扣掉激活差 356 行 × 118,272 B ≈ 0.04 GiB ⇒
  **每块 ≈ 0.729 GiB**。不靠拟合的独立核对：`allocate_stream_slot()` 的四项 BF16 权重
  = 385,874,432 元素 × 2 B = **736 MiB = 0.719 GiB/块**。而 `h3_dit.c:2779` 写死
  `per_block = 0.5 GiB`，注释理由是"int8 权重占主导 ≈0.39 GiB"——**那只在对 `--int8` 成立**；
  M4 上 `--ssd-streaming` 不带 `--int8` 的默认路径槽位是 BF16 ⇒ 低估 1.44× ⇒ 多钉 44%。
- activation 在 44f 上只有 ~0.09 GiB，而每块成本这一项的误差就是 13×0.22 ≈ **2.9 GiB**。
  所以 F47 待办 1 里"activation 是长片段主导项"不成立。
- 另一条顺手量到的：同画幅同二进制，两次跑出 available **2.6 vs 5.6 GiB**、常驻 **3 vs 9 块**。
  常驻决策只看这个读数 ⇒ 实际是抽奖。

**方法上的收获**：如果我按原计划直接"补上 118 KB/行的激活项"就交付，代码会看起来更严谨、
数字仍然错 1.4×，而 44f 照样被杀。是"先校准再改"这一步把方向纠回来的——虽然它否证的正是
我自己上一轮的判断。F47 里我写的保留意见（"缺一个主导项"）方向对、定位错：缺的主导项是
**每块槽位成本**，不是激活。

**下一步（已改进 task_plan）**：`per_block` 按实际 dtype 算；常驻上限除 available 外再用
物理内存夹一道；activation 主导项照旧补（它会让预留变大，正好抵掉 F47 那条"修正后预留变小"的告警）；
反查要等这三条之后才有意义。

**本轮另外结清的事**：F43 待办 1 的逐位复现拿到了（`6e9c2d87…` + int8-convrot 4B），
三处改动各自真机 A/B 证逐位不变（F47），并查明 F40 基线用的是 int8-convrot 版 4B。

## 续 39（2026-10-05）：F48 的三条修正落地，44f 从 SIGKILL 变跑完（F49）

按 F48 的结论改 `h3_dit.c` 的常驻决策：`per_block` 按实际槽位算（0.72 GiB，与独立推导 736 MiB
和两点拟合 0.729 GiB 三方一致）、上限再被 `physical − footprint − headroom` 夹一道
（公开 `h3_host_physical_memory()`，与 runtime guard 同规则）、activation 计入
`h3_dit_activation_bytes()`（118,272 B/行）。

**F48 定的判据两条都过**：44f 从"13 块 / peak 10.995 GiB / 去噪中被杀"变成"4 块 / peak 4.535 GiB /
EXIT=0 写出 mp4"；22f 的 latent `6e9c2d87…` 与 mp4 `ac0d0940…` 两串逐字符不变。
后者还顺带把"常驻块数不影响输出数值"从假设变成实测（9 块 vs 4 块同结果）。`make test` exit 0。

**方法上记一笔**：这一项差点按我上一轮的错误假设去做（"activation 是长片段主导项"）。是"先跑两个
形状校准、再动手"把它拽回来的 —— 校准否证的正是我自己刚写的结论，而 44f 那次真实被杀给了
一个无法争辩的反例。教训形态：**内存/性能类的估算改动的验收判据，应该是"曾经失败的那个用例现在过"，
不是"数字看起来更严谨"。**

**仍未做**：`h3.c`/`h3_cli.c` 的 activation 缺 DiT 激活项（需要先有可信 sequence 估计）；
反查"字节能撑多少帧"；`h3_memory_plan.c:80-82` 的未夹 `rec`；视觉塔每参考图重读、ffmpeg 双缓冲、
终端每帧 fork、4+3 处文档纠偏；Metal 三条等 M5 数据。

**入库更正**：本条原先写"六轮改动未入库"是过期前提 —— `d51d414`（10-05 14:30）已收走
F40/F42/F43/F45 与新测试/脚本/README/Makefile；本轮补两笔 `0301614`（F47+F49 代码）与
`dbe42d2`（F46~F49 记录）。工作区干净，未 push。

## 续 40（2026-10-05）：内存规划补上可信的 sequence 估计，并让它自证（F50）

做 F49 待办 1。`h3_dit.c` 新增 `h3_dit_sequence_estimate()`（按 `h3_layout_build()` 的真实构成：
每帧 `(latent_h/2)(latent_w/2)` 行 × `latent_t`、首尾关键条件各一份 frame_rows、音频 `2*audio_t`、
每个 reference 按满画幅一整个序列计入）与 `h3_dit_plan_bytes()`（latent + activation 合在一处，
**两个调用点共用**），`h3.c` 用 `strlen(prompt)` 当 token 行上界，REPL 预览没有 prompt 就传 0
并注明差这一项。

**给了它一个自证**：`h3_generate` 在 layout 建好后与 `planned_rows` 对账 —— 低估就打印 warning
（只有这个方向会让档位过于乐观），`H3_PROFILE=1` 时两种情况都打印数字。实测 256²/22f
得 `planned 552, layout 528` ⇒ 高估 4.5%，安全方向；两串逐位锚不变；`make test` exit 0，无新告警。

**过程中一次自我纠正**：我第一版把 CLI 侧改成只算 activation，**静默丢了 latent 项**——正是我这轮
要避免的那类漂移。改成 `h3_dit_plan_bytes()` 由结构上保证两边同式，不是靠人记住。

**明确未验证**：首尾关键帧与 reference 的估计路径（本机没有 i2v 锚可跑）。reference 是**上界**
（一张图实际只加一份 frame_rows，我按整个序列算，方向安全、代价是可能偏保守一丁点）；
真低估了会由那条 warning 抓出来，这是留的兜底，不是"应该没问题"。

**状态**：4 个代码文件已改未提交（`h3.c h3_cli.c h3_dit.c h3_dit.h`）；上一轮已提交
`0301614`/`dbe42d2`/`a88c17d`，仍未 push。

## 续 41（2026-10-05）：反查"最长能出多少帧"落地，测试抓到一个够不着的上限（F51）

做 F50 待办 2，照 splash `contextTokensWithin` 的**接口形状**（不抄它的数）补上反向那一半。

**关键设计是"共用而不是重抄"**：把 `h3_memory_plan_auto()` 内部的预算折扣规则抽成公开的
`h3_memory_plan_budget_bytes()`，正向判档位和反向查最长片段都走它 —— 否则两条式子迟早漂。
`h3_memory_plan_frames_within()` 在 **`22 + 17k`** 梯子上二分 `h3_dit_plan_bytes()`（F50 刚建的
那个共用式），所以返回的长度一定是 `h3_align_frame_count()` 认得的、调用方真能请求的长度；
22 帧一个训练块都塞不下就返回 0。CLI `!memory-plan` 多打一行
`Longest clip at WxH: N frames (+)`。

**新测试抓到的是我写的真缺陷，不是用例毛病**：上限第一版 `4065` 不在梯子上
（`align(4065) = 4068`，梯子要 `f ≡ 5 mod 17`），于是二分最大只能答 4051，
`longest == H3_PLAN_FRAMES_CEILING` 恒假 ⇒ "撞到上限"的 `+` 号一次都不会出现，用户只能把
4051 误读成硬顶。断言 `align(CEILING) == CEILING` 直接失败把它抓到，改成 4051 后全绿。
这条断言的价值在于它检查"外部世界能否表示这个返回值"，而不是函数自洽 —— 后者永远发现不了。

**数字都改成实测**（256²/文本上界 64）：8 GiB ⇒ 3252 帧（正向复核 7.9707 GiB 放得下，
梯子下一级 3269 是 8.0122 GiB 放不下）；768×432 ⇒ 753 / 下一级 770 放不下。测试的每个用例
就是"问一次 → 正向复核放得下 → 复核下一级放不下"，所以这张表是断言本身，不是另算。

**顺带把 F48 §2 的手算就地更正了**：那处用了 `h3_video_encoder_latent_t()`（`(f+3)/4`，
VAE 编码器下采样），而 DiT 序列用的是 `h3_video_latent_t()`（`((f-5)/17)*5+2`）。名字太像，
抓错一个就全盘错：rows 差算成 356（实测 **752**），44 帧对齐长度记成 58（实测 **56**）。
更正后每块 `(2.955 − 0.083)/4 = 0.718 GiB`，与独立算出的 0.719 对上到 0.001 —— 结论没变，
但两条独立路径现在真能闭合。`h3_dit.c` 里"~0.1 GiB"那句注释同步改成实测 0.141 GiB。
**规矩定下来：真函数能答的问题不要手算。**

**明确未眼验**：CLI 那一行只在纯 C 层由同一组函数覆盖。想用管道喂 REPL 亲眼看一次打印没成功
（`printf '…\nquit\n' | ./h3 -d <cached>` 不响应 stdin 输入，挂住无输出，已停）。
要看真打印得在交互终端手打 `!memory-plan`。

**逐位锚本轮复跑过**（不是沿用上一轮）：同一命令 256²/22f/steps4/seed42 + int8-convrot 4B
⇒ EXIT=0，latent `6e9c2d8786cea097ddf9bdd45e5593bd`、mp4 `ac0d0940434ad216d8131f7469154b70`
两串逐字符不变，自证行仍是 `h3: [mem] planned 552 token rows, layout 528`。
这次常驻给了 **6/50** 块（F48 §3 那条"抽奖"在修后的二进制上只有一个观测值，已记清不能与
修前的 3 / 9 排成同代码序列）。`make test` exit 0：新测试 14 条断言全绿、全仓 50 行 `  ok  `
+ 11 行 `ok…`、15 skip、0 FAIL、0 `warning:`。

**状态**：本步代码 6 个文件 + 1 个新测试未提交（`h3.c h3_cli.c h3_dit.c h3_dit.h
h3_memory_plan.c h3_memory_plan.h Makefile tests/test_memory_plan_inverse.c`）；
上一轮已提交 `0301614`/`dbe42d2`/`a88c17d`，**仍未 push**。
