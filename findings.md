# Findings & Decisions: h3.c 超分集成 + 生成调优

## Requirements (SR 集成，已完成)
- 命令行参数：`--sr-bin` / `--sr-model-dir`（两者有效才启用 SR）
- 内分辨率自动探测；目标 = `--sr-target WxH` 或 内分辨率×`--sr-scale`
- 超分后保留原声音

## Research Findings
- `H3_FPS = 24`（h3_host.h），生成固定 24fps，SR 重编码也用 24fps
- 工程已有 `run_ffmpeg`(posix_spawnp) 与 `h3_ffprobe_visual_size()` 可复用
- realesrgan-ncnn-vulkan 就绪：`/tmp/h3_realesrgan/realesrgan-ncnn-vulkan`；模型含 `realesrgan-x4plus`(×4)、`realesrgan-x4plus-anime`、`realesrnet-x4plus`、`realesr-animevideov3`(x2/x3/x4)
- 模型/LoRA 路径：
  - DiT：`/Users/jay/h3_sys/MiniMax-H3-Convrot`（权重在 FL2VA 子目录）
  - 量化备选：`/Volumes/data/.lmstudio/models/Minimax-H3-Quantized`（MiniMax_H3_FL2VA_pruned_int8_convrot.safetensors 20.9G）
  - CLIPProj：`/Volumes/data/.lmstudio/models/ClipProj-MiniMax-H3`
  - 文本编码器：`/Volumes/data/.lmstudio/models/Qwen3-VL-4B-Instruct-int8-convrot`
  - **本机无任何 Turbo/distill LoRA**（`find` 全盘无 turbo/lora/distill 文件）
- 运行时：macOS Apple M4，Metal；`setsid` 不可用，用 `nohup ... & disown` 后台

## 色块根因（关键发现）
- 失败运行：target 256×192 / **render 128×96** / layer 40 / token-reduction / SR→1024×768 / 15s → 全屏 3 段水平色带
- 机制：128×96 时 latent token 网格仅 8×6=48 token/帧；token-reduction 在中间 DiT 层把 token 成对池化→约 24/frame，信号量不足以表达结构化画面，VAE 解出平坦色
- 证据链：2s + 原生 256×192 即出内容 → 排除 steps=4 与 token-reduction 为元凶（用户亦证"之前 2s token-reduction 没问题"）
- **结论：元凶是渲染分辨率过低，不是 --steps 4，也不是 --token-reduction**

## 参数对照结论（2s / layer 40 / steps 4 / target 512×384）
| 内部渲染 | token-reduction | SR | 产物 | 体积 | 备注 |
|---|---|---|---|---|---|
| 256×192 原生 | 有 | 无 | /tmp/h3_256_192_native_2s.mp4 | 139K | 对照（出内容）|
| 256×192 原生 | 有 | →1024×768 | /tmp/h3_256_192_native_2s_sr.mp4 | 1.5M | |
| 256×192 render | 有 | →1024×768 | /tmp/h3_512_384_r256_2s_sr.mp4 | 1.5M | |
| 256×192 render | 无 | →1024×768 | /tmp/h3_512_384_r256_notr_2s_sr.mp4 | 1.9M | 比上者细节多 |
| **原生 512×384** | 有 | 无 | /tmp/h3_512_384_native_tr_2s.mp4 | 223K | |
| **原生 512×384** | 无 | 无 | /tmp/h3_512_384_native_notr_2s.mp4 | 283K | 基准最不糊 |
| 原生 512×384（上者 LR）| — | realesrgan-x4plus ×4 手跑 | /tmp/h3_512_384_lr_x4plus.mp4 | 2048×1536 / 4.1M | 最清晰 |
| 原生 512×384 | 无 | 无 | /tmp/h3_512_384_native_notr_15s.mp4 | 1.0M | 15s 基准, exit=0 |

- token-reduction 影响：关掉后体积变大（223K→283K）、峰值内存升高（5.9→7.1 GiB）→ 压缩激活量，关掉更吃显存但细节更多
- 原生 512×384 vs 256×192 render：前者 VAE 解码更慢（~64–75s vs 更快），但内部分辨率高、画面结构更完整

## SR 行为细节
- 引擎默认 `--sr-target` 在"目标 ≤ 输入×4"时：先跑 ×4 模型再 crop/缩到目标
  - 故 1024×768（512×384 的 ×2）= "×4 模型结果缩到 ×2"
  - 纯 ×4 即 2048×1536（512×384 的 ×4）= 模型原始输出，最清晰
- 手跑管线（对 .lr.mp4）：`ffmpeg` 抽帧 → `realesrgan-ncnn-vulkan -n realesrgan-x4plus -s 4 -i 帧目录 -o 帧目录 -m 模型目录` → `ffmpeg -framerate 24 -i %05d.png -c:v libx264 -pix_fmt yuv420p -crf 18 out.mp4`

## 参数校验规则（来自 main.c）
- target 与 render 宽高均需 **32 的倍数、≥32**（报错：`width and height must be multiples of 32 and at least 32`）；render ≤ target；同宽高比
- SR target 与 output 同宽高比且 ≤4×；frames 对齐到 5..362
- 注意：校验**不捕捉**"渲染分辨率过低导致的画质崩坏"（128×96 render 校验通过却出色块）
- **有效 4:3 子 512×384 分辨率**：W=128k, H=96k → 仅 384×288 / 256×192 / 128×96（320×240、192×144、160×120 因高非 32 倍数被拒）

## Issues Encountered
| Issue | Resolution |
|-------|------------|
| ffmpeg 多输入命令把 `-vf` 当输入选项 | 缩放拆独立单输入 pass；抽帧去 `-vf` |
| `setsid` 不可用（macOS）| `nohup ... & disown` 后台 detached |
| h3 非 TTY 下 stdout 全缓冲（日志空到结束）| 监控靠进程 PID / mp4 存在 / FINAL_EXIT 标记 |

## Resources
- 生成入口：`main.c` prompt 分支 → `h3_generate()` → `params.output_path`
- SR 实现：`h3_ffmpeg.c::h3_superres`
- 交互：`h3_cli.c::process_command` 的 `!sr`
