#!/usr/bin/env python3
"""生成 H3 引擎(C 二进制) 的 ComfyUI 工作流。

产物：
- h3_binary_t2v.json — 文本/首尾帧 → 视频（H3_BinaryT2V + SaveVideo）
- h3_binary_r2v.json — 参考图/视频/音频 → 视频（H3_BinaryR2V + SaveVideo + LoadImage×2）

同时写入项目根目录与 ComfyUI 的 workflows 目录。

用法：
    python3 gen_comfyui_workflows.py [--comfy-workflows DIR]
"""
import argparse
import json
import os
import sys
import uuid

PROJ = os.path.dirname(os.path.abspath(__file__))
DEFAULT_COMFY_WORKFLOWS = "/Volumes/data/Documents/ComfyUI/user/default/workflows"

# 从节点模块读取常量（同时用于校验 widget 顺序一致性）
CUSTOM_RES = "custom (用 width/height)"
_NODE_OK = False
try:
    sys.path.insert(0, os.path.join(PROJ, "comfyui_nodes"))
    import h3_binary as _hb  # noqa: E402
    CUSTOM_RES = _hb.CUSTOM_RESOLUTION
    _NODE_OK = True
except Exception as exc:  # 生成器可独立运行
    print(f"[warn] 无法导入 h3_binary（跳过 widget 校验）: {exc}")


def _widget_names(cls):
    """列出节点的 widget 类参数（排除 IMAGE 等连线型输入），顺序与前端一致。"""
    widget_types = {"INT", "FLOAT", "STRING", "BOOLEAN"}
    names = []
    for section in ("required", "optional"):
        for name, spec in cls.INPUT_TYPES().get(section, {}).items():
            ty = spec[0]
            if isinstance(ty, list):          # COMBO
                names.append(name)
            elif ty in widget_types:
                names.append(name)
    return names

H3_BINARY = os.environ.get("H3_BINARY", "/Volumes/data/git/c/h3.c/h3")
MODEL_DIR = os.environ.get("H3_MODEL_DIR", "/Users/jay/h3_sys/MiniMax-H3-Convrot")
CLIPPROJ_DIR = os.environ.get(
    "H3_CLIPPROJ_DIR", "/Volumes/data/.lmstudio/models/Qwen3-VL-4B-Instruct-int8-convrot")
CLIPPROJ_PROJ = os.environ.get(
    "H3_CLIPPROJ_PROJ", "/Volumes/data/.lmstudio/models/ClipProj-MiniMax-H3")

T2V_PROMPT = (
    "Realistic live-action cinematic look, a red fox walks through fresh snow in a pine "
    "forest, medium tracking shot, natural winter light, realistic fur.\n\n"
    "Audio: light wind through the pines, soft snow crunch under paws, distant birds."
)
R2V_PROMPT = (
    "Use <Picture 1> as the subject character and <Picture 2> as the environment style.\n\n"
    "Bold comic-book ink style, heavy linework, night city. The character moves through "
    "the scene with confident motion, dynamic camera, consistent identity across shots.\n\n"
    "Audio: low synth score, city ambience, a soft accent hit on each cut."
)


def _node(nid, ntype, pos, size, widgets=None, inputs=None, outputs=None,
          title=None, order=0, props=None):
    n = {
        "id": nid, "type": ntype, "pos": list(pos), "size": list(size),
        "flags": {}, "order": order, "mode": 0,
        "inputs": inputs or [], "outputs": outputs or [],
        "properties": props or {"Node name for S&R": ntype},
    }
    if title:
        n["title"] = title
    if widgets is not None:
        n["widgets_values"] = widgets
    return n


def _link_input(name, itype, link=None, optional=False):
    d = {"localized_name": name, "name": name, "type": itype, "link": link}
    if optional:
        d["shape"] = 7
    return d


def _widget_input(name, itype, widget_name=None):
    return {"localized_name": name, "name": name, "type": itype,
            "widget": {"name": widget_name or name}, "link": None}


def _output(name, otype, links=None, display=None):
    return {"localized_name": display or name, "name": name,
            "type": otype, "links": links}


def save_video_node(nid, pos, video_link, order, prefix):
    return _node(
        nid, "SaveVideo", pos, [508, 106], order=order,
        inputs=[
            _link_input("video", "VIDEO", video_link),
            _widget_input("filename_prefix", "STRING"),
            _widget_input("format", "COMFY_DYNAMICCOMBO_V3"),
            _widget_input("format.codec", "COMFY_DYNAMICCOMBO_V3",
                          widget_name="format.codec"),
            _widget_input("codec", "COMFY_DYNAMICCOMBO_V3"),
        ],
        outputs=[_output("video", "VIDEO")],
        widgets=[prefix, "auto", "auto", "auto"],
        props={"cnr_id": "comfy-core", "ver": "0.33.0",
               "Node name for S&R": "SaveVideo"},
    )


def load_image_node(nid, pos, filename, order, out_link):
    return _node(
        nid, "LoadImage", pos, [320, 320], order=order,
        inputs=[_widget_input("image", "COMBO"),
                _widget_input("upload", "IMAGEUPLOAD")],
        outputs=[_output("IMAGE", "IMAGE", [out_link] if out_link else None,
                         display="图像"),
                 _output("MASK", "MASK", display="遮罩")],
        widgets=[filename, "image"],
        props={"Node name for S&R": "LoadImage", "cnr_id": "comfy-core",
               "ver": "0.30.0"},
    )


def info_node(nid, pos, order):
    return _node(
        nid, "H3_BinaryInfo", pos, [420, 130], order=order,
        inputs=[],
        outputs=[_output("info", "STRING")],
        widgets=[MODEL_DIR, H3_BINARY, CLIPPROJ_DIR, CLIPPROJ_PROJ],
        title="H3 环境检查（可选，运行后可看设备/权重清单）",
    )


def t2v_workflow():
    # 组件顺序必须与节点 INPUT_TYPES 一致（IMAGE 类输入走 slot，不占 widget）
    widgets = [
        T2V_PROMPT,        # prompt
        864,               # width
        480,               # height
        CUSTOM_RES,        # resolution_preset
        2.0,               # seconds
        20,                # steps
        42,                # seed
        "",                # lora
        True,              # auto_steps
        MODEL_DIR,         # model_dir
        "",                # output_path
        1,                 # core_reuse
        1,                 # reuse
        0,                 # layers
        H3_BINARY,         # binary
        CLIPPROJ_DIR,      # clipproj_dir
        CLIPPROJ_PROJ,     # clipproj_proj
        "",                # extra_args
    ]
    main = _node(
        20, "H3_BinaryT2V", [300, 300], [430, 400], order=1,
        inputs=[_link_input("first_frame", "IMAGE", optional=True),
                _link_input("last_frame", "IMAGE", optional=True)],
        outputs=[_output("video", "VIDEO", [1]),
                 _output("video_path", "STRING")],
        widgets=widgets,
    )
    return {
        "id": str(uuid.uuid4()), "revision": 0,
        "last_node_id": 30, "last_link_id": 1,
        "nodes": [
            info_node(10, [300, 60], 0),
            main,
            save_video_node(30, [770, 320], 1, 2, "video/H3_Binary_T2V"),
        ],
        "links": [[1, 20, 0, 30, 0, "VIDEO"]],
        "groups": [{"id": 1, "title": "H3 引擎（C 二进制）",
                    "bounding": [280, 40, 1030, 700], "color": "#3f789e",
                    "flags": {}}],
        "config": {}, "extra": {}, "version": 0.4,
    }


def r2v_workflow():
    widgets = [
        R2V_PROMPT,        # prompt
        864,               # width
        480,               # height
        CUSTOM_RES,        # resolution_preset
        2.0,               # seconds
        20,                # steps
        42,                # seed
        "",                # lora
        True,              # auto_steps
        MODEL_DIR,         # model_dir
        "",                # output_path
        1,                 # core_reuse
        1,                 # reuse
        0,                 # layers
        H3_BINARY,         # binary
        CLIPPROJ_DIR,      # clipproj_dir
        CLIPPROJ_PROJ,     # clipproj_proj
        "",                # extra_args
        "match",           # ref_image_size
        "",                # ref_video_path
        "",                # ref_audio_path
    ]
    main = _node(
        20, "H3_BinaryR2V", [560, 300], [430, 420], order=2,
        inputs=[_link_input("ref_image_1", "IMAGE", 1, optional=True),
                _link_input("ref_image_2", "IMAGE", 2, optional=True)],
        outputs=[_output("video", "VIDEO", [3]),
                 _output("video_path", "STRING")],
        widgets=widgets,
    )
    return {
        "id": str(uuid.uuid4()), "revision": 0,
        "last_node_id": 30, "last_link_id": 3,
        "nodes": [
            load_image_node(10, [60, 100], "example.png", 0, 1),
            load_image_node(11, [60, 460], "AA1LhIKt.jpeg", 1, 2),
            info_node(12, [60, 820], 3),
            main,
            save_video_node(30, [1030, 320], 3, 4, "video/H3_Binary_R2V"),
        ],
        "links": [[1, 10, 0, 20, 0, "IMAGE"],
                  [2, 11, 0, 20, 1, "IMAGE"],
                  [3, 20, 0, 30, 0, "VIDEO"]],
        "groups": [{"id": 1, "title": "H3 引擎（C 二进制）— Ref2VA",
                    "bounding": [40, 80, 1490, 900], "color": "#3f789e",
                    "flags": {}}],
        "config": {}, "extra": {}, "version": 0.4,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--comfy-workflows", default=DEFAULT_COMFY_WORKFLOWS)
    args = ap.parse_args()

    targets = [PROJ]
    if os.path.isdir(args.comfy_workflows):
        targets.append(args.comfy_workflows)
    else:
        print(f"[warn] ComfyUI workflows 目录不存在，仅写入项目: {args.comfy_workflows}")

    made = (("h3_binary_t2v.json", t2v_workflow(), "H3_BinaryT2V"),
            ("h3_binary_r2v.json", r2v_workflow(), "H3_BinaryR2V"))
    for name, data, cls_name in made:
        if _NODE_OK:
            cls = getattr(_hb, cls_name)
            expect = _widget_names(cls)
            node = next(n for n in data["nodes"] if n["type"] == cls_name)
            got = node["widgets_values"]
            if len(expect) != len(got):
                raise SystemExit(
                    f"[FATAL] {cls_name}: 声明 {len(expect)} 个 widget "
                    f"但工作流给了 {len(got)} 个\n  声明: {expect}")
            print(f"[check] {cls_name}: {len(got)} 个 widget 顺序一致 ✓")
        text = json.dumps(data, ensure_ascii=False, indent=2)
        for d in targets:
            path = os.path.join(d, name)
            with open(path, "w", encoding="utf-8") as f:
                f.write(text)
            print(f"写出 {path}")


if __name__ == "__main__":
    main()
