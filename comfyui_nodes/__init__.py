"""H3 引擎 ComfyUI 节点（h3.c 项目）。

把 h3.c 编译出的 `h3` 二进制封装为 ComfyUI 节点：
一节点完成「文本/图像/参考 → 视频 + 原生音频(MP4)」全链路，
可直接替换官方工作流的整条采样链。

安装（二选一）：
    # 软链接（推荐，改代码即生效）
    ln -s /Volumes/data/git/c/h3.c/comfyui_nodes \\
          <ComfyUI>/custom_nodes/h3_binary_nodes
    # 或复制
    cp -r /Volumes/data/git/c/h3.c/comfyui_nodes \\
          <ComfyUI>/custom_nodes/h3_binary_nodes

> 目录名建议用 `h3_binary_nodes`，与 PHP 项目的 `h3_nodes` 区分，避免包名冲突。

前置：
1. 在 h3.c 目录执行 `make` 生成 `h3`（本包默认使用同级的 `h3`）。
2. 设置 `H3_CLIPPROJ_DIR` + `H3_CLIPPROJ_PROJ`（设了则 `FL2VA/text_encoder` 可缺失）。
3. R2V 需安装 `Ref2VA/` 权重。
"""
from .h3_binary import (H3_BinaryT2V, H3_BinaryR2V, H3_BinaryInfo, H3_BinaryNote,
                         H3_BinaryLatent, H3_BinaryLatentUpscale, H3_BinaryLatentDecode)

NODE_CLASS_MAPPINGS = {
    "H3_BinaryT2V": H3_BinaryT2V,
    "H3_BinaryR2V": H3_BinaryR2V,
    "H3_BinaryInfo": H3_BinaryInfo,
    "H3_BinaryNote": H3_BinaryNote,
    "H3_BinaryLatent": H3_BinaryLatent,
    "H3_BinaryLatentUpscale": H3_BinaryLatentUpscale,
    "H3_BinaryLatentDecode": H3_BinaryLatentDecode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "H3_BinaryT2V": "H3 Engine T2V (C binary)",
    "H3_BinaryR2V": "H3 Engine R2V (C binary)",
    "H3_BinaryInfo": "H3 Engine Info (C binary)",
    "H3_BinaryNote": "H3 Engine Notes (Markdown)",
    "H3_BinaryLatent": "H3 Engine Latent (C binary)",
    "H3_BinaryLatentUpscale": "H3 Latent Upscale",
    "H3_BinaryLatentDecode": "H3 Latent Decode (C binary)",
}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
