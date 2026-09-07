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

### 5-Question Reboot Check（更新至 2026-09-07 末）
| Question | Answer |
|---|---|
| Where am I? | Phase 16 完成:22 个配置的端到端基准测试,含 HTML 报告 |
| Where am I going? | 用户决定是否继续(如测试更高分辨率/更长视频) |
| What's the goal? | 找到最佳速度/画质权衡参数 |
| What have I learned? | 显存不是瓶颈;step=7 比 step=4 画质提升 54%;--token-reduction 有效;最佳权衡 s7-l50-r2(177s/SSIM=0.57) |
| What have I done? | 编写 benchmark.py 工具链 + 三轮 22 次测试 + HTML 报告 |
