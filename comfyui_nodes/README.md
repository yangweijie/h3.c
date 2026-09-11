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
