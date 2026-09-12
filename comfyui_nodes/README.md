# H3 引擎 ComfyUI 节点（h3.c 项目）

把 `h3.c` 编译出的 **C 引擎二进制**（同级的 `h3`）当作推理后端，封装为 ComfyUI 节点。

一个节点完成「文本/图像/参考 → 视频 + 原生音频(MP4)」，可直接替换官方工作流的整条采样链：

```
UNETLoader + CLIPLoader + MiniMaxH3ImageToVideo / MiniMaxH3ReferenceToVideo
 + SamplerCustomAdvanced + VAEDecode + VAEDecodeAudio + CreateVideo
→ 只留 [H3 Engine T2V/R2V] + [SaveVideo]
```

## 安装

```bash
# 1) 编译二进制
cd /Volumes/data/git/c/h3.c && make

# 2) 软链接到 ComfyUI（目录名用 h3_binary_nodes，与 PHP 项目的 h3_nodes 区分）
ln -s /Volumes/data/git/c/h3.c/comfyui_nodes \
      /Volumes/data/Documents/ComfyUI/custom_nodes/h3_binary_nodes

# 3) 重启 ComfyUI
```

## 节点

| 节点 | 用途 | 输出 |
|------|------|------|
| `H3_BinaryT2V` | 文本 / 首尾帧 → 视频（FL2VA） | `VIDEO`, `video_path` |
| `H3_BinaryR2V` | 参考图/视频/音频 → 视频（Ref2VA） | `VIDEO`, `video_path` |
| `H3_BinaryInfo` | 跑 `h3 --info` 看设备/权重清单 | `info` |
| `H3_BinaryLatent` | 文本/首尾帧 → 视频 + 视频 latent（FL2VA） | `VIDEO`, `LATENT`, `video_path` |
| `H3_BinaryLatentUpscale` | 视频 latent 空间上采样（纯 Python） | `LATENT` |
| `H3_BinaryLatentDecode` | 视频 latent → 视频（VAE 解码，无音频） | `VIDEO`, `video_path` |

### 参数

主要输入：`prompt, width, height, resolution_preset, seconds, steps, seed`

可选：`lora`（逗号分隔多个）、`auto_steps`、`first_frame/last_frame`（T2V）、
`ref_image_1/2`、`ref_video_path`、`ref_audio_path`（R2V）、
`core_reuse/reuse/layers`（加速）、`model_dir`、`binary`、`clipproj_dir`、`clipproj_proj`。

#### `resolution_preset` — 分辨率预设（对齐官方 ResolutionSelector）

选非 `custom` 时**覆盖** `width/height`。内置 18 个预设：

| 预设 | 尺寸 |
|------|------|
| `custom (用 width/height)` | 手填，自动校正为 32 的倍数 |
| `官方 16:9 0.2MP` … `0.98MP` | **608×352, 736×416, 864×480, 960×544, 1056×608, 1152×640, 1216×672, 1280×736, 1344×768(768p)** |
| `9:16` / `1:1` / `4:3` / `3:4` / `21:9` | 0.4MP / 0.98MP（如 `9:16 0.4MP (480x864)`） |

> 16:9 使用**官方表精确值**；其他比例由公式「短边向上取整、长边取最近 32 倍数」求得。
> 官方默认 `16:9 0.4MP = 864×480`，与引擎默认一致。

#### `auto_steps` — 步数自动校正（默认开）

| LoRA 名 | 行为 |
|---------|------|
| 含 `Nstep`（如 `..._8step_...`） | → `N` 步 |
| 含 turbo / lightning / dmd / lcm | → 4 步 |
| 无 LoRA 且 steps < 8 | 仅提示「建议 20 步」 |
| 无法推断 | 仅提示，不改动 |

关闭后只提示不修改。日志会打印实际参数：`[H3] 最终参数: 864x480, 4 步`。

## 前置条件

1. **二进制**：`cd /Volumes/data/git/c/h3.c && make`（本包默认用同级的 `h3`）
2. **ClipProj 环境**（设了之后 `FL2VA/text_encoder` 可缺失）：
   ```
   H3_CLIPPROJ_DIR=/Volumes/data/.lmstudio/models/Qwen3-VL-4B-Instruct-int8-convrot
   H3_CLIPPROJ_PROJ=/Volumes/data/.lmstudio/models/ClipProj-MiniMax-H3
   ```
3. **R2V**：需安装 `Ref2VA/` 权重（`H3_BinaryInfo` 显示 `Ref2VA DiT 0 files` 即未安装）

## 工作流

用生成器产出（写入项目根目录 + ComfyUI workflows 目录）：

```bash
python3 gen_comfyui_workflows.py
# → h3_binary_t2v.json（H3_BinaryInfo + H3_BinaryT2V + SaveVideo）
# → h3_binary_r2v.json（LoadImage×2 + H3_BinaryInfo + H3_BinaryR2V + SaveVideo）
```

生成时会自动校验 `widgets_values` 与节点声明顺序一致，防止参数错位。

## 注意

- 引擎运行时从**二进制所在目录**读取 `h3_shaders.metal` 并编译 Metal kernel
  → 节点已自动把子进程 `cwd` 设为该目录。
- `steps` 合法范围 `[2, 1000]`；宽高必须为 32 的倍数（节点会自动校正）。
- 实测：256×256 / 0.5s / 4 步（turbo LoRA）/ core-reuse 4 ≈ **69 秒**。

## 特殊开关与环境变量说明

> 与 `H3_BinaryNote` 节点（显示名 "H3 Engine Notes (Markdown)"）的默认文本一致，
> 可直接加到工作流里随时查看；复制该节点的 `notes` 输出到 ComfyUI 内置 **Note** 节点即可在画布上渲染 Markdown。
> 这些开关都是「近似 / 提速」手段：开大会更快，但输出**不再与精确路径逐 bit 一致**；
> 追求可复现 / 最高质量时全部回到默认值（core_reuse=1, reuse=1, layers=0, fast_stream=关）。

### 节点加速开关

- **core_reuse**（`--core-reuse`，默认 1，1~6）：核心张量复用间隔。每 N 步才完整重算一次「核心」中间激活，其间复用上一次结果做近似。1=精确(close)，4=快速(fast)，6=激进(aggressive)。质量优先保持 1；快速草稿用 6。
- **reuse**（`--reuse`，默认 1，常用 1~3）：去噪步间细粒度复用，复用 DiT block 内部中间结果（attention 上下文、投影缓存等），颗粒比 core_reuse 更细。1=close，2=fast，3=aggressive。**与 core_reuse 互斥**：二者不能同时 >1（引擎报错 `core reuse and denoiser reuse cannot be combined`），只能二选一。
- **layers**（`--layers`，默认 0=50 块，0~50）：实际参与的 DiT block 数。0=完整(精确)，45=快，40=激进。只缩短 DiT 主干，不影响 tokenizer / VAE；低质量快速预览用 45/40。
- **fast_stream**（节点开关 → `H3_DIT_STREAM_WORKERS=2`，默认关）：把 DiT 权重流式预取拆成多 worker 并与 CPU 反旋转重叠。仅在低分辨率（≤384²）有效：实测 256²≈1.67×、384²≈1.27×；512² 及以上无效且略慢。低分辨率批量出图时打开。
- **extra_args**（附加 CLI 参数）：传节点未暴露的开关，例如 `--video-vae-streaming 0|1`（强制 VAE 解码器常驻/流式，见下）。

### 环境变量（进程级；节点 extra_args / 系统环境均可设）

- **H3_DIT_RESIDENT_BLOCKS=N**（默认 0）：前 N 个 DiT 块常驻、其余流式，用内存换 I/O。实测 16 GB 上 wall time 几乎不降（甚至略升），因常驻块与流式路径争抢统一内存；每块约 0.7 GiB。仅在内存充裕（≥20 块能轻松放下）且磁盘是瓶颈时才有收益；16 GB 不要开。
- **H3_VIDEO_VAE_STREAMING / --video-vae-streaming**（0|1|-1 auto，默认 auto）：0=VAE 解码器常驻（快但占 ~9 GiB，16 GB 干扰下可能 OOM），1=流式（只占 ~0.25 GiB，慢一些），-1=由内存规划器决定。
- **H3_DIT_STREAM_WORKERS / H3_DIT_STREAM_PIPELINE**：fast_stream 的底层旋钮（=2 启用分块多 worker；PIPELINE 只调重叠），一般直接用节点 fast_stream 开关。
- **H3_PROFILE=1**：打印 I/O 字节、吞吐、未被 GPU 隐藏的读等待，性能诊断用。
- **H3_CLIPPROJ_DIR / H3_CLIPPROJ_PROJ**：节点已通过 clipproj_dir / clipproj_proj 参数自动注入。
- **H3_FFMPEG / H3_FFPROBE**：节点已自动探测并注入；**H3_BINARY / H3_MODEL_DIR**：节点默认读取，也可参数覆盖。

### 调参优先级

1. 默认（1/1/0/关）：精确基准。
2. 想快：低分辨率开 fast_stream；低质量预览把 layers 降到 45/40，或把 core_reuse / reuse 调大。
3. 质量没达标：全部回到精确（1/1/0）。
4. 内存紧张（16 GB）：保持 video_vae_streaming 默认（流式），不要开 H3_DIT_RESIDENT_BLOCKS。
5. 内存充裕且要榨速度：video_vae_streaming=0（常驻 VAE）+ fast_stream（低分辨率）+ 视情况 H3_DIT_RESIDENT_BLOCKS（>16 GB 且磁盘慢）。

## Latent 子系统（生成 / 上采样 / 解码）

把 H3 引擎当可「中断在 VAE 解码前」的推理后端：先产出**视频 latent**（去噪结果，z 空间 `[C=24, T, H, W]`），
后续的 latent 空间操作（上采样、合成、插值）都更便宜，最后再用 VAE 解码成视频。适合做「高分辨率放大」「latent 编辑」等玩法。

| 节点 | 用途 | 输出 |
|------|------|------|
| `H3_BinaryLatent` | 等同 T2V/R2V（文本/首尾帧 → 视频+音频），**额外**同时写出视频 latent | `VIDEO`, `LATENT`, `video_path` |
| `H3_BinaryLatentUpscale` | 对 latent 的 H,W 做空间放大（factor 倍，T/C 不变） | `LATENT` |
| `H3_BinaryLatentDecode` | latent → 视频（直接 VAE 解码，跳过 DiT 去噪；**无音频**） | `VIDEO`, `video_path` |

### 典型工作流（latent 上采样放大分辨率）

```
H3_BinaryLatent ──VIDEO──▶ SaveVideo
        │
      LATENT
        │
        ▼
H3_BinaryLatentUpscale (factor=2) ──LATENT──▶ H3_BinaryLatentDecode ──VIDEO──▶ SaveVideo
```

- `H3_BinaryLatent` 内部等价于 T2V，只是多一步把去噪后的 latent 存到 `--latent-out`（raw 文件，节点自动读取成 ComfyUI `LATENT`）。
- `H3_BinaryLatentUpscale` 是纯 Python（torch 插值），只放大空间 H,W，保持时间 T 与通道 C，以维持 VAE 时序对齐。
- `H3_BinaryLatentDecode` 把 latent 写回 raw 文件，调引擎 `--latent-in` 直接走 VAE 解码出视频。因为 latent 不含音频，解码出的视频**无音轨**（要音频请用 `H3_BinaryT2V`/`H3_BinaryR2V`）。

### latent 文件格式（引擎 ↔ 节点共用）

raw 二进制，自描述：
- 头部 20 字节：`uint32 magic=0x48334C54` + `int32 t, h, w, c`
- 随后 `c*t*h*w` 个 `float32`，按 `[C, T, H, W]` 连续存放（与引擎 `h3_video_vae_decode` 内部索引顺序一致）

> 注意：这是 **H3 自己的 latent 空间**（24 通道、16× 空间压缩、H3 专属归一化），
> **不能**直接喂给官方 `MinimaxH3LatentUpscaler3D` / 官方 `VAEDecode`；本子系统内部自洽即可。

### 命令行等价验证（已实测通过）

```bash
# 生成视频 + 落 latent（需设 H3_CLIPPROJ_DIR/PROJ，见「前置条件」）
./h3 -d <model> -p "..." --width 256 --height 160 --seconds 2 --steps 4 \
     -o /tmp/direct.mp4 --latent-out /tmp/latent.bin
# 用 latent 反解视频（与直出视频逐帧 md5 一致）
./h3 -d <model> --latent-in /tmp/latent.bin --width 256 --height 160 -o /tmp/decoded.mp4
# latent 空间 2x 上采样后解码 → 512x320 视频（上采样在节点内用 torch 完成）
```
