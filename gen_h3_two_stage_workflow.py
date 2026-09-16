#!/usr/bin/env python3
"""生成 H3 两阶段管线的 ComfyUI 工作流（两个变体）。

共同前两段：H3_BinaryLatent（低分生成）→ MinimaxH3LatentUpscaler3D（潜空间放大）

  变体 A  h3_two_stage_pipeline.json       ③ = H3_BinaryLatentRefine（全分辨率精修）
  变体 B  h3_two_stage_upscale_decode.json ③ = H3_BinaryLatentDecode（直接解码）

变体 B 的 DiT **只在低分辨率下运行**，全分辨率只做 VAE 解码，因此能在 16 GB
机器上出 1280x704 / 5 秒；代价是没有精修。

两份产物各写两处：
  <ComfyUI>/user/default/workflows/<name>.json    ← 前端可加载
  <repo>/comfyui_nodes/<name>.workflow.json       ← 随仓库分发

只写「连线输入」与按名绑定的 widget 值；其它 widget 由前端按节点 schema 重建，
这样节点以后再增删参数也不会错位（按位置的 widgets_values 会）。
"""

import json
import os
import sys

PROMPT = "A red fox walks through fresh snow in a pine forest."
MODELS = "/Users/jay/h3_sys/MiniMax-H3-Convrot"
BINARY = "/Volumes/data/git/c/h3c/h3"
CLIPPROJ_DIR = "/Volumes/data/.lmstudio/models/Qwen3-VL-4B-Instruct-int8-convrot"
CLIPPROJ_PROJ = "/Volumes/data/.lmstudio/models/ClipProj-MiniMax-H3"
UPSCALER = "minimax_h3_latent_upscaler_3d_fp16.safetensors"

# 尺寸语义与约束（两个变体共用）
SIZE_DOC = """## 三个尺寸怎么填（三个节点会显示成同一个数，这是正常的）

| 节点 | 字段 | 含义 |
|---|---|---|
| ① | `width` / `height` | 低分首轮的**输出视频**尺寸 |
| ① | `extra_args` 里的 `--render-width/--render-height` | **内部画布 ← 真正的「低分辨率」旋钮** |
| ② | `mode.width` / `mode.height` | 放大的**目标像素**尺寸 |
| ③ | `width` / `height` | 最终输出尺寸 |

硬约束：

- **② 必须等于 ③**：放大器输出的 latent 直接决定 ③ 的 token 网格，对不上引擎报元素数不匹配。
- **① 的 `render-*` 与 ① 的输出同宽高比，且是 32 的倍数**（取输出的一半最省）。
"""

MEMORY_DOC = """## 内存信封（16 GB 机器实测）

DiT 激活 ≈ 空间位置数 × T；**超过约 30k tokens 就会撑爆内存**。
（实测 32,560 tokens 跑通；130,240 tokens 导致系统重启。）

`tokens = (输出宽/16) × (输出高/16) × T`，`T` 由 `seconds` 决定（引擎按 24fps 换算后
对齐到 `5+17k` 帧，latent 时间长度为 `T = 5k+2`）：

| `seconds` | 0.5 | 1.0 | 2.0 | 3.0 | 5.0 |
|---|---|---|---|---|---|
| 帧数 | 22 | 39 | 56 | 73 | 124 |
| latent T | 7 | 12 | 17 | 22 | 37 |

| 最终输出 | 空间 | 0.5s | 1s | 2s | 5s |
|---|---|---|---|---|---|
| 640x384 | 960 | 6,720 | 11,520 | 16,320 | 35,520 ⚠ |
| 1024x576 | 2,304 | 16,128 | 27,648 | 39,168 ⚠ | 85,248 ❌ |
| 1280x704 | 3,520 | 24,640 | 42,240 ⚠ | 59,840 ❌ | 130,240 ❌ |

⚠ 接近上限，❌ 会崩。
"""

AUDIO_DOC = """## h3_audio 必须传下来

`H3_BinaryLatent` 把音频 latent 放在 LATENT 字典的 `h3_audio` 键。
下游节点靠它保留音轨：

- 变体 A 的 `H3_BinaryLatentRefine` 用它按**同一个 `refine_sigma`** 重新加噪音频并与
  视频联合去噪，音视频一致性才成立。若它丢失，引擎会改用单位正态噪声（σ=1）初始化，
  而 schedule 从 `refine_sigma` 起步 —— 分布外输入，结果是**明显更响的噪声音轨，
  且加步数救不回来**。
- 变体 B 的 `H3_BinaryLatentDecode` 直接用它做音频 VAE 解码；丢了则退化为静音。

（放大器节点原先只返回 `{"samples": ...}` 会丢掉这个键，本仓库已修。）
"""

SIGMA_DOC = """## refine_sigma 的取舍（实测，对比第一阶段画面）

| `refine_sigma` | 0.05 | 0.1 | 0.2 | 0.4 | 0.6 |
|---|---|---|---|---|---|
| SSIM vs 第一阶段 | 0.95 | 0.92 | 0.89 | 0.87 | 0.84 |

参照标尺：画面未变 = 0.99，换种子重新生成 = 0.84。
**0.3~0.5 是可用区间**；0.6 以上基本等于重新生成。
"""

NOTE_A = f"""# H3 两阶段管线（变体 A）：低分生成 → 潜空间放大 → **全分辨率精修**

`H3_BinaryLatent` → `MinimaxH3LatentUpscaler3D` → `H3_BinaryLatentRefine` → `SaveVideo`

{SIZE_DOC}
{AUDIO_DOC}
{SIGMA_DOC}
{MEMORY_DOC}
## 本变体的问题

③ 会在**全分辨率**下跑 DiT，因此受上面的 token 上限约束 —— 想清楚目标尺寸再配。
**要出 1280x704 / 5 秒请改用变体 B**（`h3_two_stage_upscale_decode.json`）。

## 耗时参考（640x384 / 0.5s / 2 步，实测）

| 阶段 | 耗时 |
|---|---|
| ① 低分生成 | 57.9s |
| ② 潜空间放大（CPU，345M 参数） | 2.4s（1280x704 时 60.9s） |
| ③ 全分辨率精修 | 87.4s |
"""

NOTE_B = f"""# H3 两阶段管线（变体 B）：低分生成 → 潜空间放大 → **直接解码**

`H3_BinaryLatent` → `MinimaxH3LatentUpscaler3D` → `H3_BinaryLatentDecode` → `SaveVideo`

## 与变体 A 的区别

变体 A 的第三步 `H3_BinaryLatentRefine` 会在全分辨率下跑 DiT，因此被 token 上限卡住
（1280x704 / 5 秒 = 130,240 tokens，会撑爆 16 GB）。

**本变体把第三步换成 `H3_BinaryLatentDecode`：放大后直接做 VAE 解码，全分辨率 DiT
一次都不跑。** 于是 DiT 只在 ① 的低分辨率画布上运行，而输出仍是全分辨率视频。

代价是**没有精修** —— 放大器的输出直接上屏，画面细节不如变体 A 干净。

{SIZE_DOC}
## 本变体的尺寸选择

既然 ③ 不跑 DiT，token 上限只约束 ① 的低分辨率画布：

| 最终输出 | ① `extra_args` | ① tokens (5s) | 结果 |
|---|---|---|---|
| 1280x704 | `--render-width 640 --render-height 352` | 40x22x37 = 32,560 | ✅ 实测跑通（DiT 12 分钟） |
| 1024x576 | `--render-width 512 --render-height 288` | 32x18x37 = 21,312 | ✅ 更省 |
| 640x384 | `--render-width 320 --render-height 192` | 20x12x37 = 8,880 | ✅ 最快 |

{AUDIO_DOC}
## 耗时参考

① 的 render 640x352 / 5 秒 / 5 步实测：DiT 721s + VAE 175s ≈ **15 分钟**。
② 的放大器在 1280x704 上实测 **60.9s**。
③ 只做 VAE 解码，量级与 ① 的 VAE 相当（约几分钟）。

{MEMORY_DOC}
"""


def link_input(name, kind, link):
    return {"localized_name": name, "name": name, "type": kind, "link": link}


def node(nid, ntype, pos, size, title, order, inputs, outputs,
         widgets_values, widgets_values_named=None):
    entry = {
        "id": nid, "type": ntype, "pos": pos, "size": size, "flags": {},
        "order": order, "mode": 0, "inputs": inputs, "outputs": outputs,
        "properties": {"Node name for S&R": ntype},
        "widgets_values": widgets_values,
    }
    if title:
        entry["title"] = title
    if widgets_values_named is not None:
        entry["widgets_values_named"] = widgets_values_named
    return entry


def generate_node(output_w, output_h, render_w, render_h, seconds, steps):
    """① 低分生成。真正决定「低分辨率」的 render-* 藏在 extra_args，写进标题里可见。"""
    extra = f"--render-width {render_w} --render-height {render_h}"
    return node(
        nid=1, ntype="H3_BinaryLatent", pos=[70, 120], size=[430, 600],
        title=f"① 低分生成 — 内部画布 {render_w}x{render_h}（在 extra_args 里）",
        order=0,
        inputs=[link_input("first_frame", "IMAGE", None),
                link_input("last_frame", "IMAGE", None)],
        outputs=[{"localized_name": name, "name": name, "type": kind,
                  "links": [1] if name == "latent" else None}
                 for name, kind in (("video", "VIDEO"), ("latent", "LATENT"),
                                    ("video_path", "STRING"))],
        widgets_values=[PROMPT, output_w, output_h, "custom (用 width/height)",
                        seconds, steps, 42, "fixed", "", False, MODELS, "",
                        1, 1, 0, False, BINARY, CLIPPROJ_DIR, CLIPPROJ_PROJ,
                        extra],
        widgets_values_named={
            "prompt": PROMPT, "width": output_w, "height": output_h,
            "resolution_preset": "custom (用 width/height)",
            "seconds": seconds, "steps": steps, "seed": 42,
            # ComfyUI 前端会给 seed 自动加这个命名伴生值；生成器不写就会
            # 每次重新生成都与前端保存的版本产生无意义差异。
            "control_after_generate": "fixed",
            "lora": "",
            "auto_steps": False, "model_dir": MODELS, "output_path": "",
            "core_reuse": 1, "reuse": 1, "layers": 0, "fast_stream": False,
            "binary": BINARY, "clipproj_dir": CLIPPROJ_DIR,
            "clipproj_proj": CLIPPROJ_PROJ, "extra_args": extra},
    )


def upscale_node(target_w, target_h):
    """② 潜空间放大（CPU）。target 必须等于 ③ 的宽高。"""
    return node(
        nid=2, ntype="MinimaxH3LatentUpscaler3D", pos=[560, 220], size=[300, 210],
        title=f"② 潜空间放大（CPU）→ {target_w}x{target_h}", order=1,
        inputs=[link_input("latent", "*", 1)],
        outputs=[{"name": "latent", "type": "*", "links": [2]}],
        widgets_values=[UPSCALER, "target dimensions", target_w, target_h, 32,
                        False, True, "cpu", "fp32"],
        widgets_values_named={
            "model_name": UPSCALER, "mode": "target dimensions",
            "mode.width": target_w, "mode.height": target_h, "align": 32,
            "enable_temporal_chunking": False, "force_unload": True,
            "device": "cpu", "precision": "fp32"},
    )


def refine_node(width, height, seconds, steps, sigma):
    """③(A) 全分辨率精修。width/height 必须等于 ② 的 target。"""
    return node(
        nid=3, ntype="H3_BinaryLatentRefine", pos=[560, 480], size=[430, 400],
        title="③ 全分辨率精修（受 token 上限约束）", order=2,
        inputs=[link_input("latent", "LATENT", 2)],
        outputs=[{"localized_name": name, "name": name, "type": kind,
                  "links": [3] if name == "video" else None}
                 for name, kind in (("video", "VIDEO"),
                                    ("video_path", "STRING"))],
        widgets_values=[PROMPT, width, height, seconds, steps, sigma, 42, "fixed",
                        MODELS, BINARY, CLIPPROJ_DIR, CLIPPROJ_PROJ, ""],
        widgets_values_named={
            "prompt": PROMPT, "width": width, "height": height, "seconds": seconds,
            "steps": steps, "refine_sigma": sigma, "seed": 42,
            "control_after_generate": "fixed",
            "model_dir": MODELS, "binary": BINARY,
            "clipproj_dir": CLIPPROJ_DIR, "clipproj_proj": CLIPPROJ_PROJ,
            "extra_args": ""},
    )


def decode_node(width, height):
    """③(B) 直接解码。不跑 DiT，只做 VAE 解码。"""
    return node(
        nid=3, ntype="H3_BinaryLatentDecode", pos=[560, 480], size=[430, 330],
        title=f"③ 直接 VAE 解码（不跑全分辨率 DiT）→ {width}x{height}",
        order=2,
        inputs=[link_input("latent", "LATENT", 2)],
        outputs=[{"localized_name": name, "name": name, "type": kind,
                  "links": [3] if name == "video" else None}
                 for name, kind in (("video", "VIDEO"),
                                    ("video_path", "STRING"))],
        widgets_values=[width, height, MODELS, BINARY, CLIPPROJ_DIR, CLIPPROJ_PROJ],
        widgets_values_named={
            "width": width, "height": height, "model_dir": MODELS,
            "binary": BINARY, "clipproj_dir": CLIPPROJ_DIR,
            "clipproj_proj": CLIPPROJ_PROJ},
    )


def save_node(prefix):
    return node(
        nid=4, ntype="SaveVideo", pos=[1050, 520], size=[320, 180],
        title=None, order=3,
        inputs=[link_input("video", "VIDEO", 3)],
        outputs=[{"name": "video", "type": "VIDEO", "links": None}],
        widgets_values=[prefix, "auto", "auto"],
        # format 是 DynamicCombo，前端把它归一化成 format / codec / format.codec
        widgets_values_named={"filename_prefix": prefix, "format": "auto",
                              "codec": "auto", "format.codec": "auto"},
    )


def note_node(text):
    return node(nid=5, ntype="MarkdownNote", pos=[70, 760], size=[690, 620],
                title="说明 / 注意事项", order=4, inputs=[], outputs=[],
                widgets_values=[text], widgets_values_named={"text": text})


def assemble(workflow_id, nodes):
    return {
        "id": workflow_id, "revision": 0, "last_node_id": 5, "last_link_id": 3,
        "nodes": nodes,
        # [link_id, source_node, source_slot, target_node, target_slot, type]
        "links": [[1, 1, 1, 2, 0, "LATENT"],
                  [2, 2, 0, 3, 0, "LATENT"],
                  [3, 3, 0, 4, 0, "VIDEO"]],
        "groups": [], "config": {}, "extra": {}, "version": 0.4,
    }


def build_variant_a():
    """小尺寸默认值：640x384 输出 / 320x192 内部画布 / 0.5s，已实测跑通。

    steps 默认 4：ComfyUI 里手动改过并已同步回仓库副本，这里保持一致，
    重新生成才不会覆盖掉那次编辑。
    """
    return assemble("h3-two-stage-pipeline", [
        generate_node(640, 384, 320, 192, 0.5, 4),
        upscale_node(640, 384),
        refine_node(640, 384, 0.5, 2, 0.4),
        save_node("video/h3_two_stage"),
        note_node(NOTE_A),
    ])


def build_variant_b():
    """1280x704 / 5 秒：① 的 render 640x352 是安全上限内，③ 不跑 DiT。

    steps 默认 4：ComfyUI 里手动改过并已同步回仓库副本，这里保持一致。
    注意本变体没有精修，① 的步数直接决定最终画质 —— 若用的是
    8step 蒸馏权重，4 步仍属欠收敛（实测 4步 vs 8步 SSIM 0.680），
    追求画质应改回 8。
    """
    return assemble("h3-two-stage-upscale-decode", [
        generate_node(1280, 704, 640, 352, 5.0, 4),
        upscale_node(1280, 704),
        decode_node(1280, 704),
        save_node("video/h3_upscale_decode"),
        note_node(NOTE_B),
    ])


def main():
    # 默认**不覆盖已存在的文件**。这套工作流会在 ComfyUI 前端被编辑（UI 会补
    # bgcolor/color、把位置数组归一化成 widgets_values_named、并改写的参数），
    # 盲目重跑生成器会把那些编辑冲掉 —— 已经发生过一次。要更新用 --force，
    # 并先把前端版同步回来（见文件头说明）。
    force = "--force" in sys.argv[1:]
    here = os.path.dirname(os.path.abspath(__file__))
    targets = {
        "h3_two_stage_pipeline": build_variant_a(),
        "h3_two_stage_upscale_decode": build_variant_b(),
    }
    for name, workflow in targets.items():
        for path in (f"/Volumes/data/Documents/ComfyUI/user/default/workflows/{name}.json",
                     os.path.join(here, "comfyui_nodes", f"{name}.workflow.json")):
            if os.path.exists(path) and not force:
                print(f"  skip   {path}  (已存在，加 --force 覆盖)")
                continue
            os.makedirs(os.path.dirname(path), exist_ok=True)
            with open(path, "w", encoding="utf-8") as handle:
                json.dump(workflow, handle, ensure_ascii=False, indent=1)
            print(f"  wrote  {path}  ({os.path.getsize(path)} bytes)")


if __name__ == "__main__":
    main()
