# Progress Log

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
