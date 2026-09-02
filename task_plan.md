# Task Plan: 把 Real-ESRGAN 超分集成进 h3.c 命令行

## Goal
为 `h3.c` 增加命令行超分参数(`--sr` / `--sr-bin` / `--sr-model-dir` / `--sr-model` / `--sr-target` / `--sr-scale`),在视频生成成功后调用 `realesrgan-ncnn-vulkan` 把低清内分辨率超分到目标分辨率,并保留原音轨。交互模式加 `!sr` 命令。

## Current Phase
Phase 5（已完成，待交付）

## Phases

### Phase 1: Requirements & Discovery
- [x] 理解意图:生成后做 SR 后处理(非 VAE 自带上采样),复用真实 realesrgan 二进制
- [x] 确认工程用 getopt_long + Apple clang,H3_FPS=24
- **Status:** complete

### Phase 2: Planning & Structure
- [x] 集成点选定:生成写入 output_path 后,改名 `.lr.mp4` → 抽帧 → realesrgan → ffmpeg 重编码(带原音轨)写回 output_path,失败则还原
- [x] 目标分辨率语义:正好是内分辨率整数倍(2/3/4)直接该倍超分;否则 ×4 后再缩到目标
- **Status:** complete

### Phase 3: Implementation
- [x] `h3_ffmpeg.h`:声明 `h3_superres()`
- [x] `h3_ffmpeg.c`:加 `#include <limits.h>`、`run_command()`、`h3_superres()`(抽帧 + realesrgan + 重编码;缩放走独立单输入 pass)
- [x] `main.c`:加 include、枚举、options[]、解析、校验、`prompt` 分支里 SR 后处理、usage 文本
- [x] `h3_cli.c`:状态字段、初始化、清理释放、`generate()` 后调 SR、`!sr` 命令、help/status
- **Status:** complete

### Phase 4: Testing & Verification
- [x] `make CC="xcrun clang"` 编译通过(`-Wall -Wextra -Wpedantic -Wshadow -Wconversion` 无警告)
- [x] 端到端测试:独立 C 程序调用 `h3_superres` 对真实低清视频做 256→864,产物 864×864 且含 audio,`ok=1`
- [x] CLI 接线校验:`--help` 含 SR 选项;`--sr` 缺 bin/model-dir 时报错退出 2
- **Status:** complete

### Phase 5: Delivery
- [ ] 向用户汇报用法(已完成口头说明,待确认是否提交 git)
- **Status:** complete

## Key Questions
1. 内分辨率如何取得 → 用 `h3_ffprobe_visual_size()` 从生成产物自动探测
2. 目标非整数倍时怎么办 → ×4 超分后再 ffmpeg 缩放(独立 pass)
3. realesrgan-x4plus 固定 ×4 → 精确 ×4 用 `--sr-target 1024x1024` 或 216→864

## Decisions Made
| Decision | Rationale |
|----------|-----------|
| 生成后做 SR 后处理(改名 .lr.mp4,写回原路径,失败还原) | 与已验证的 bash 脚本一致,最简单且不影响主生成流程 |
| 缩放拆成独立单输入 ffmpeg pass,不在双输入命令里用 `-vf` | ffmpeg 会把双输入命令里的 `-vf` 误判为输入选项而报错 |
| 抽帧不用 `-vf format=rgb24` | realesrgan 直接读 PNG 即可,避免 `-vf` 放置导致的 ffmpeg 报错 |
| 用 posix_spawnp 跑外部程序(复用已有 run_ffmpeg 模式) | 与工程既有风格一致,二进制经 PATH 查找 |

## Errors Encountered
| Error | Attempt | Resolution |
|-------|---------|------------|
| ffmpeg "Option vf cannot be applied to input url" (抽帧命令带 -vf) | 1 | 去掉 `-vf format=rgb24`,直接抽 PNG |
| ffmpeg 同样报错(重编码双输入命令里带 -vf scale) | 2 | 把缩放拆成独立单输入 pass,再与音轨混流 |
| replace_in_file 失败(old_str 用 `sizeof out_pattern` 无括号) | 3 | 实际文件是 `sizeof(out_pattern)` 带括号,按精确文本重写 |

## Notes
- 修改文件:`h3_ffmpeg.c` `h3_ffmpeg.h` `main.c` `h3_cli.c`(git 已显示 modified)
- 之前的 `/tmp/sr_pipeline.sh` bash 版可丢弃(功能现已在 C 里)
- 未提交 git(用户未要求)
