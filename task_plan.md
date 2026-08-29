# Task Plan: MiniMax-H3 on 16GB Mac（4B+ClipProj 验证 + 跑通/优化）

## Goal
在 16GB Mac 上跑通并优化 MiniMax-H3 视频生成；用 4B+ClipProj/MLX 文本编码替代 32B 以省内存/提速，并验证与 h3.c 32B 输出的余弦对齐。

## Workstream A: 4B+ClipProj harness（验证用）
| # | Phase | Status | 验证 |
|---|-------|--------|------|
| 1 | 获取 ClipProj 仓库并校验 | complete | ModelScope 下载完成，SHA256 通过；GitCode 镜像需认证已弃 |
| 2 | 解析投影元数据 + 确认 4B 模型 | complete | tap=24, d_in=2560, d_out=5120, mlp_hidden=32768, cos_test=0.8144 |
| 3 | 编写 harness (clipproj_harness.py) | complete | 文件已跟踪，--hs-index 可调（默认 tap+1=25） |
| 4 | 实跑前向验证流水线 | complete* | candidate.npz 已产出 shape (1,5120)；**用户确认 Phase 4 已用 MLX 跑通**；MLX 脚本待补入仓库 |
| 5 | 确认 ComfyUI 层索引映射 | pending | 需读 sd1_clip.py 的 encode_token_weights |
| 6 | 余弦对齐 0.81 / 与 h3.c 32B 对照 | pending | 依赖 4 与 5 |

## Workstream B: 16GB 跑通与优化
| # | Phase | Status | 验证 |
|---|-------|--------|------|
| B1 | 32B 文本编码耗时拆解 | complete | 22.5min=46.9GB@35MB/s 流式；GPU 仅 1.9s；非加载慢 |
| B2 | 磁盘/USB 瓶颈定位 | in_progress | disk_speed.c 实测写 33MB/s；ioreg 见 USB2 Hub；待换口验证 |
| B3 | 下载 DiT 权重(58GB) | pending | 前置：需快盘/快口 |
| B4 | 合并 feature/lora-merge + --lora DiT 验证 | pending | DiT 权重下载后置 |
| B5 | h3.c 外部条件注入接口 | pending | 读 npz 替换 text encoder 输出 |
| B6 | 4B+ClipProj/MLX 文本编码 offload | pending | Phase4 已产 [seq,5120]；脚本待补入 |
| B7 | h3.c int8 权重加载器 | pending | 方向1：int8 62→15.5GB 驻留 |

## Decisions
- 文本编码现状：硬编码 32B / TEXT_DIM=5120；4B 需 ClipProj 升维 + 注入接口。
- MLX 文本编码可行（Phase 4 已验证产 [seq,5120]）；llama.cpp/LM Studio 不可行（无 hidden states）。
- 估算：4B+MLX 文本编码相较 32B-in-h3.c 冷启 5–20x、热启 ~1000x；内存从"必须流式 62GB"变为"8.3GB 驻留"。
- 最高杠杆：先确认/修复 USB2 瓶颈（方向5），再下 58GB DiT 权重。

## Open Questions
- Phase 4 MLX 脚本在哪、是否补入 h3.c 仓库？
- 外部条件注入接口的具体 API 形态？
- int8 加载器（B7）vs 4B offload（B6）哪个优先落地？
