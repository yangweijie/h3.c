# Findings & Decisions: h3.c 超分集成

## Requirements
- 命令行参数:指定超分 bin 目录(`--sr-bin`)与模型目录(`--sr-model-dir`);两参数都有效时启用 SR
- 内分辨率 = 实际要超分前的分辨率(生成视频的实际尺寸,自动探测)
- 目标分辨率 = 超分后分辨率(由 `--sr-target WxH` 或 内分辨率×`--sr-scale` 决定)
- 超分后保留原声音

## Research Findings
- `H3_FPS = 24`(在 `h3_host.h`),生成结果固定 24fps,SR 重编码也用 24fps 保帧率一致
- `h3.c` 已有 `--render-width/--render-height`(低清内部分辨率)与 `--width/--height`(输出)概念,但内部上采样是 VAE 自带,不是 Real-ESRGAN
- 工程已有 `run_ffmpeg`(posix_spawnp 跑 ffmpeg)与 `h3_ffprobe_visual_size()` 可直接复用
- realesrgan-ncnn-vulkan 二进制已就绪:`/tmp/h3_realesrgan/realesrgan-ncnn-vulkan`,模型 `realesrgan-x4plus` 在 `/tmp/h3_realesrgan/models`

## Technical Decisions
| Decision | Rationale |
|----------|-----------|
| SR 作为生成后处理,不改主生成管线 | 最小侵入,失败可还原,不影响无 SR 场景 |
| `h3_superres()` 签名:input/output/bin_dir/model_dir/model_name/scale/target_w/target_h | 覆盖精确倍率与非整数倍两种路径 |
| 非整数倍时 ×4 超分 → 独立单输入 ffmpeg scale pass | 规避双输入 `-vf` 误判 |
| 抽帧直接 `ffmpeg -i in %05d.png`(无 `-vf`) | realesrgan 读 PNG 无格式要求 |

## Issues Encountered
| Issue | Resolution |
|-------|------------|
| ffmpeg 在多输入命令里把 `-vf` 当作输入选项 | 缩放改用独立单输入 pass;抽帧去掉 `-vf` |
| replace_in_file 多次失败 | 旧串 `sizeof out_pattern` 漏写括号,按 `cat -e` 核对精确文本后成功 |

## Resources
- 生成入口:`main.c` 的 `prompt` 分支 → `h3_generate()` → `params.output_path`
- SR 实现:`h3_ffmpeg.c::h3_superres`
- 交互命令:`h3_cli.c::process_command` 的 `!sr` 分支
