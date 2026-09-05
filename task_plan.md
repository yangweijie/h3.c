# Task Plan: h3.c — 超分集成 + 视频生成参数调优 + Metal 内核优化

## Goal
1. (已完成) 把 Real-ESRGAN 超分作为 CLI 后处理集成进 h3.c，保留音轨。
2. (已完成) 诊断"生成只有色块、无画面"问题，并调出可用的 15s 视频参数组合。
3. (已完成) 参考 vdn-minimax-h3 Python 项目，优化 Metal 内核：FlashAttention + Tiled Windowed Softmax。

## Current Phase
Phase 15：Metal 内核优化完成（FlashAttention + Tiled Windowed Softmax + Linear Far-Brain 基础设施）。
下一步：用户决定是否继续优化其他方向（FP8 量化、Ulysses 序列并行、训练 linear far-branch 权重）。

## Phases (SR 集成 — 已完成)
### Phase 1: Requirements & Discovery — complete
### Phase 2: Planning & Structure — complete
### Phase 3: Implementation — complete
### Phase 4: Testing & Verification — complete
### Phase 5: Delivery — complete（口头说明，git 未提交）

## Phases (色块诊断 + 参数调优 — 已完成)
### Phase 6: 色块根因定位 — complete
### Phase 7: 2×2 参数对照 — complete
### Phase 8: 15s 正式基准 — complete
### Phase 9: 最小清晰分辨率扫描 — complete
### Phase 10: 人脸清晰度 1:1 矩阵 — complete
### Phase 11: 加超分出品 — pending（待用户决定）

## Phases (Metal 内核优化 — 已完成)
### Phase 12: FlashAttention 内核调试 — complete
- [x] 修复 BF16 类型转换 bug（`h3_bf16_to_f32` 参数类型不匹配）
- [x] 实现 `h3_flash_attn_causal` kernel
- [x] 实现 `h3_flash_attn_tiled_windowed` kernel
- [x] 测试通过：causal max_err=0.00095, tiled_windowed max_err=0.00817

### Phase 13: Tiled Windowed Softmax — complete
- [x] 实现 O(T·window) 复杂度的 windowed attention
- [x] 支持 text（dense）、video（windowed）、audio（dense）三种 query 类型
- [x] 测试通过：tiled_windowed OK (seq=24, F=4, S=4, r=2)

### Phase 14: FP8 量化集成分析 — complete
- [x] 分析 Python 参考项目 `ops/fp8_linear.py`
- [x] 确认 C 代码已有量化基础设施（int8 tensor core）
- [x] FP8 E4M3 可通过 INT8 模拟，动态范围更好（±448 vs ±127）

### Phase 15: Linear Far-Brain 分支基础设施 — complete
- [x] 实现完整的 SanaDelta delta-rule scan 管线
- [x] 7 个 linear-branch 测试全部通过
- [ ] 需要训练权重（alpha、beta、q_norm、gate、to_out_linear）

## Key Questions
1. 色块根因 → 128×96 渲染分辨率过低（非 steps / 非 token-reduction）
2. Metal BF16 类型转换 → 必须使用 `as_type<ushort>()` / `as_type<bfloat>()` 显式位转换
3. FlashAttention 复杂度 → causal O(T²/2), tiled_windowed O(T·window)
4. Linear far-branch → 需要训练权重，当前 checkpoint 中不存在

## Decisions Made
| Decision | Rationale |
|----------|-----------|
| 诊断用 2s 短片 + 原生分辨率做 A/B，而非直接 15s | 快速、低成本定位根因 |
| 最终 15s 基准用原生 512×384 + 无 token-reduction（最不糊）| 内部分辨率越高画面结构越完整 |
| Metal BF16 使用 `as_type<>()` 显式位转换 | 避免编译器生成错误的类型转换代码 |
| Tiled windowed 按 query 类型分 3 个 key 范围处理 | 减少不必要的 key 遍历 |

## Errors Encountered
| Error | Attempt | Resolution |
|-------|---------|------------|
| `setsid: command not found`（macOS）| 1 | 改用 `nohup bash -c '...' & disown` |
| Metal kernel 输出全零 | 多个 | 修复 `h3_bf16_to_f32` 参数类型不匹配 |
| `(ushort)` cast 产生错误 BF16 值 | 多个 | 使用 `as_type<bfloat>(ushort(bits >> 16))` |
| `h3_f32_to_bf16()` 被编译器错误内联 | 多个 | 使用 `as_type<uint>()` 直接位操作 |

## Notes
- 修改文件（SR 集成）：`h3_ffmpeg.c` `h3_ffmpeg.h` `main.c` `h3_cli.c`
- 修改文件（Metal 内核）：`h3_shaders.metal` `h3_gpu.m` `h3_gpu.h`
- 新增测试：`tests/test_flash_attn.c` `tests/test_tiled_windowed.c` `tests/test_linear_branch.c`
- 模型路径见 findings.md
