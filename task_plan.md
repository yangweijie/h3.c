# Task Plan: h3.c — 超分集成 + 视频生成参数调优

## Goal
1. (已完成) 把 Real-ESRGAN 超分作为 CLI 后处理集成进 h3.c，保留音轨。
2. (进行中) 诊断"生成只有色块、无画面"问题，并调出可用的 15s 视频参数组合（分辨率 / token-reduction / 超分）。

## Current Phase
Phase 7：2×2 参数对照已完成；15s 基准（原生 512×384 / 无 token-reduction / 无超分）已产出。
下一步：用户决定是否①给 15s 基准加超分出品（×2→1024×768 或 纯×4→2048×1536），②或把 `--steps` 提到 10/20。

## Phases (SR 集成 — 已完成)
### Phase 1: Requirements & Discovery — complete
### Phase 2: Planning & Structure — complete
### Phase 3: Implementation — complete
### Phase 4: Testing & Verification — complete
### Phase 5: Delivery — complete（口头说明，git 未提交）

## Phases (色块诊断 + 参数调优 — 进行中)
### Phase 6: 色块根因定位
- [x] 最初失败运行：target 256×192 / render 128×96 / layer 40 / token-reduction / SR→1024×768 / 15s → 全屏 3 段水平色带，无画面
- [x] 假设① --steps 4 过低（模型原生 20 步，且本机无 Turbo/distill LoRA）→ 被用户推翻："之前 2s token-reduction 没问题"
- [x] 对照：2s + 原生 256×192（无降 render）即出现内容 → 证伪 render 之外的因素
- [x] **结论：根因是 128×96 内部渲染分辨率过低**（latent token 网格仅 8×6=48 token/帧，token-reduction 再对半后约 24，不足以表达结构化场景）
- **Status:** complete

### Phase 7: 2×2 参数对照（2s / layer 40 / steps 4 / target 512×384）
- [x] 原生 512×384 + token-reduction + 无 SR → OK（223K）
- [x] 原生 512×384 + 无 token-reduction + 无 SR → OK（283K，最不糊基准）
- [x] 256×192 render + token-reduction + SR→1024×768 → OK（1.5M）
- [x] 256×192 render + 无 token-reduction + SR→1024×768 → OK（1.9M，比上者细节多）
- [x] 手跑 realesrgan-x4plus ×4（512×384 LR → 2048×1536）→ 最清晰（4.1M）
- **Status:** complete

### Phase 8: 15s 正式基准
- [x] 原生 512×384 / 无 token-reduction / 无 SR / 15s → `/tmp/h3_512_384_native_notr_15s.mp4`（1.0M, exit=0, ~33min）
- **Status:** complete

### Phase 9: 最小清晰分辨率扫描（进行中）
- [x] 脚本 /tmp/h3_res_scan.sh：6 档 4:3 分辨率（384×288/320×240/256×192/192×144/160×120/128×96）× layer 40/45/50
- [x] 统一：原生（无 --render 降采样）/ 无 token-reduction / 不超分 / --seconds 0.5 / --steps 4
- [ ] 产出 18 个文件 /tmp/h3_<W>x<H>_L<L>_05s.mp4，等用户肉眼挑最小清晰档
- **Status:** in_progress

### Phase 10: 人脸清晰度 1:1 矩阵（完成）
- [x] 脚本 /tmp/h3_face_matrix.sh：1:1 分辨率 256→640（32 倍数，13 档）× layer 40/45/50（39 档）
- [x] 统一：原生（无 render）/ 无 token-reduction / 不超分 / `--steps 4` / **`--seconds 0.5`（=12 帧，因 `--frames 5` 被 VAE 22-frame chunk 限制拒掉）/ 保留音轨
- [x] 提示词改为"整画面人脸特写"，便于肉眼比清晰度
- [x] 39 个文件全部 EXIT=0，记录耗时+体积（见 progress.md 表）；用户挑最清晰
- **Status:** complete

### Phase 11: 加超分出品（待用户决定）
- [ ] 在选定最清晰档上加 `--sr --sr-target 1024x768`（×2）或 `2048x1536`（纯×4）
- [ ] 可选：把 `--steps` 抬到 10/20 看细节变化
- **Status:** pending

## Key Questions
1. 色块根因 → 128×96 渲染分辨率过低（非 steps / 非 token-reduction）
2. 内分辨率如何取得 → `h3_ffprobe_visual_size()` 自动探测
3. 目标非整数倍 → ×4 超分后再 ffmpeg 缩放
4. 本机无 Turbo/distill LoRA → steps=4 在"渲染分辨率足够"时仍可用（不是色块元凶）

## Decisions Made
| Decision | Rationale |
|----------|-----------|
| 诊断用 2s 短片 + 原生分辨率做 A/B，而非直接 15s | 快速、低成本定位根因 |
| 最终 15s 基准用原生 512×384 + 无 token-reduction（最不糊）| 内部分辨率越高画面结构越完整；token-reduction 关掉后细节更多（代价更吃内存）|
| 长任务用 `nohup ... & disown` 后台跑（macOS 无 setsid）| 避免命令超时被杀；h3 stdout 非 TTY 全缓冲，靠进程 PID/mp4/FINAL_EXIT 监控 |

## Errors Encountered
| Error | Attempt | Resolution |
|-------|---------|------------|
| `setsid: command not found`（macOS）| 1 | 改用 `nohup bash -c '...' & disown` |
| 长命令疑似超时风险（15s 原生 512×384 约 27–33min）| — | 后台 detached 启动 + 轮询（每 30s 查 FINAL_EXIT / mp4 / 进程存活）|

## Notes
- 修改文件（SR 集成）：`h3_ffmpeg.c` `h3_ffmpeg.h` `main.c` `h3_cli.c`（git 显示 modified，未提交）
- 模型路径见 findings.md
- 参数校验不捕捉"分辨率过低导致的画质崩坏"：128×96 render 与 256×192 target 校验通过却出色块
