"""H3 引擎（C 二进制）ComfyUI 节点 —— 属于 h3.c 项目。

本包位于 `<h3.c>/comfyui_nodes/`，默认使用**同级的 `h3` 二进制**
（在 h3.c 目录执行 `make` 生成）。

把已编译的 C 引擎当作推理后端，一个节点完成
「文本/图像/参考 → 视频+原生音频(MP4)」全链路。

可直接替换官方工作流中的整条采样链：
    UNETLoader + CLIPLoader + MiniMaxH3ImageToVideo / MiniMaxH3ReferenceToVideo
    + SamplerCustomAdvanced + VAEDecode + VAEDecodeAudio + CreateVideo
→ 只留本节点 + SaveVideo。

节点：
- H3_BinaryT2V: 文本 / 首尾帧 → 视频（FL2VA）
- H3_BinaryR2V: 参考图 / 参考视频 / 参考音频 → 视频（Ref2VA）

运行前提（二进制要求）：
- `H3_CLIPPROJ_DIR` + `H3_CLIPPROJ_PROJ` 指向 4B Qwen3-VL(int8-convrot) 与 ClipProj；
  设置后 `FL2VA/text_encoder` 不再需要（C: h3.c:774-797）。
- Ref2VA 需要 `Ref2VA/` 权重，未安装时 R2V 不可用（`--info` 会显示 0 files）。
"""
import math
import os
import re
import shutil
import subprocess
import time

# 本包位于 <h3.c>/comfyui_nodes/ → 上溯一级即项目根，二进制默认取同级的 h3
_H3_ROOT = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))

_DEFAULT_BINARY = os.environ.get("H3_BINARY", os.path.join(_H3_ROOT, "h3"))
_DEFAULT_MODEL_DIR = os.environ.get("H3_MODEL_DIR", "/Users/jay/h3_sys/MiniMax-H3-Convrot")
_DEFAULT_CLIPPROJ_DIR = os.environ.get(
    "H3_CLIPPROJ_DIR", "/Volumes/data/.lmstudio/models/Qwen3-VL-4B-Instruct-int8-convrot")
_DEFAULT_CLIPPROJ_PROJ = os.environ.get(
    "H3_CLIPPROJ_PROJ", "/Volumes/data/.lmstudio/models/ClipProj-MiniMax-H3")

CATEGORY = "H3"

CUSTOM_RESOLUTION = "custom (用 width/height)"

# 官方 video_minimax_h3_t2v.json 的 ResolutionSelector 参考表（16:9, multiple=32）
_OFFICIAL_16_9 = [
    (0.2, 608, 352), (0.3, 736, 416), (0.4, 864, 480), (0.5, 960, 544),
    (0.6, 1056, 608), (0.7, 1152, 640), (0.8, 1216, 672), (0.9, 1280, 736),
    (0.98, 1344, 768),
]
_ASPECTS = {
    "16:9": (16, 9), "9:16": (9, 16), "1:1": (1, 1),
    "4:3": (4, 3), "3:4": (3, 4), "21:9": (21, 9),
}
MULTIPLE = 32  # H3 画布对齐（短边 768 原生、上限 768×1344，均为 32 的倍数）


def _ceil_mult(value: float, m: int = MULTIPLE) -> int:
    return max(m, int(math.ceil(value / m)) * m)


def _round_mult(value: float, m: int = MULTIPLE) -> int:
    return max(m, int(round(value / m)) * m)


def calc_resolution(aspect: str, megapixels: float, m: int = MULTIPLE):
    """按 宽高比 + 百万像素 求 (宽, 高)：短边向上取整，长边按比例取最近倍数。

    与官方 ResolutionSelector 同思路（官方 16:9 表已内置为精确值，见下）。
    """
    wr, hr = _ASPECTS[aspect]
    ratio = max(wr, hr) / min(wr, hr)
    short = _ceil_mult(math.sqrt(megapixels * 1e6 / ratio), m)
    long_ = _round_mult(short * ratio, m)
    return (long_, short) if wr >= hr else (short, long_)


def _build_resolution_presets():
    presets = {CUSTOM_RESOLUTION: None}
    for mp, w, h in _OFFICIAL_16_9:          # 官方表，精确对齐
        presets[f"官方 16:9 {mp}MP ({w}x{h})"] = (w, h)
    for aspect, mps in (("9:16", (0.4, 0.98)), ("1:1", (0.4, 0.98)),
                        ("4:3", (0.4, 0.98)), ("3:4", (0.4,)),
                        ("21:9", (0.4,))):
        for mp in mps:
            w, h = calc_resolution(aspect, mp)
            presets[f"{aspect} {mp}MP ({w}x{h})"] = (w, h)
    return presets


RESOLUTION_PRESETS = _build_resolution_presets()
RESOLUTION_CHOICES = list(RESOLUTION_PRESETS)

# LoRA 名 → 推荐步数（turbo/lightning/dmd 蒸馏模型需要少步数）
_FEW_STEP_HINTS = ("turbo", "lightning", "dmd", "lcm", "schnell")


def suggest_steps(lora: str):
    """从 LoRA 名推断推荐去噪步数。返回 (推荐步数 or None, 说明)。"""
    name = (lora or "").strip()
    if not name:
        return None, "未使用 LoRA → 建议 20 步"
    match = re.findall(r"(\d+)\s*[-_]?\s*step", name, flags=re.I)
    if match:
        n = int(match[-1])
        return n, f"LoRA 名含 '{n}step' → 建议 {n} 步"
    low = name.lower()
    if any(k in low for k in _FEW_STEP_HINTS):
        return 4, "检测到 turbo/lightning/dmd 类 LoRA → 建议 4 步（或 8 步）"
    return None, "无法从 LoRA 名推断步数（按 checkpoint 自带建议）"


# ---------------------------------------------------------------------------
# 工具函数
# ---------------------------------------------------------------------------

def _comfy_temp_dir() -> str:
    """ComfyUI 临时目录（用于中转输入图 / 引擎输出 MP4）。"""
    try:
        import folder_paths
        d = folder_paths.get_temp_directory()
    except Exception:
        import tempfile
        d = tempfile.gettempdir()
    os.makedirs(d, exist_ok=True)
    return d


def _save_image(img, name: str) -> str:
    """把 ComfyUI IMAGE ([H,W,C] float 0-1) 存成 PNG，返回路径。"""
    import numpy as np
    from PIL import Image

    arr = img
    for attr in ("detach", "cpu"):
        if hasattr(arr, attr):
            arr = getattr(arr, attr)()
    arr = np.clip(np.asarray(arr, dtype="float32"), 0.0, 1.0)
    if arr.ndim == 3 and arr.shape[0] == 1:      # 单帧 [1,H,W,C]
        arr = arr[0]
    d = os.path.join(_comfy_temp_dir(), "h3_binary_in")
    os.makedirs(d, exist_ok=True)
    path = os.path.join(d, name)
    Image.fromarray((arr * 255.0).astype("uint8")).save(path)
    return path


def _make_video(path: str):
    """把 MP4 文件包装成 ComfyUI 的 VIDEO 对象（可直连 SaveVideo）。"""
    try:
        from comfy_api.latest import InputImpl
    except Exception as exc:  # ComfyUI 过旧
        raise RuntimeError(
            "当前 ComfyUI 不支持 comfy_api.latest（需要较新版本才能返回 VIDEO）。"
            f"原始错误: {exc}\n可改用输出中的 video_path 接 LoadVideo。") from exc
    return InputImpl.VideoFromFile(path)


def _interrupted() -> bool:
    try:
        import comfy.model_management as mm
        return bool(mm.processing_interrupted())
    except Exception:
        return False


def _resolve_ffmpeg_env():
    """引擎用 posix_spawnp 查找 ffmpeg/ffprobe（依赖 PATH）。ComfyUI 由 GUI/launchd
    拉起时 PATH 往往不含 /opt/zerobrew/bin、/usr/local/bin 等，导致
    'cannot start FFmpeg: No such file or directory'。这里把绝对路径通过
    H3_FFMPEG / H3_FFPROBE 注入子进程环境（引擎已支持这两个覆盖变量）。"""
    result = {}
    for var, name in (("H3_FFMPEG", "ffmpeg"), ("H3_FFPROBE", "ffprobe")):
        if os.environ.get(var):
            continue
        path = shutil.which(name)
        if not path:
            for cand in ("/opt/zerobrew/bin", "/usr/local/bin",
                         "/opt/homebrew/bin", "/opt/local/bin", "/usr/bin"):
                p = os.path.join(cand, name)
                if os.path.exists(p):
                    path = p
                    break
        if path:
            result[var] = path
        else:
            print(f"[H3] 警告: 未找到 {name}，请安装 ffmpeg 或设置 {var} 环境变量")
    return result


def _run_engine(cmd, env_extra, tag="H3"):
    """运行引擎二进制，实时转发日志；返回 (returncode, 输出全文)。

    注意：引擎在运行时从 **当前工作目录** 读取 `h3_shaders.metal` 并编译 Metal
    kernel（C 源码中为相对路径 "h3_shaders.metal"），因此必须把 cwd 设为
    二进制所在目录，否则报 "cannot compile h3_shaders.metal"。
    """
    env = os.environ.copy()
    env.update(_resolve_ffmpeg_env())
    env.update({k: v for k, v in env_extra.items() if v})
    work_dir = os.path.dirname(os.path.abspath(cmd[0]))
    print(f"[{tag}] exec: {' '.join(cmd)}")
    print(f"[{tag}] cwd: {work_dir}")
    t0 = time.time()
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT, text=True,
                            bufsize=1, env=env, cwd=work_dir)
    lines = []
    try:
        for raw in proc.stdout:
            line = raw.rstrip()
            if line:
                lines.append(line)
                print(f"[{tag}] {line}")
            if _interrupted():
                proc.kill()
                raise RuntimeError("用户取消（已终止引擎进程）")
    finally:
        if proc.poll() is None:
            proc.wait()
    dt = time.time() - t0
    print(f"[{tag}] 引擎结束 rc={proc.returncode} 用时 {dt:.1f}s")
    return proc.returncode, "\n".join(lines)


class _H3BinaryBase:
    """两个节点共用的参数与执行逻辑。"""

    CATEGORY = CATEGORY
    RETURN_TYPES = ("VIDEO", "STRING")
    RETURN_NAMES = ("video", "video_path")
    FUNCTION = "generate"

    @classmethod
    def _common_required(cls):
        return {
            "prompt": ("STRING", {"multiline": True, "default": ""}),
            "width": ("INT", {"default": 864, "min": 64, "max": 2048, "step": 32,
                              "tooltip": "resolution_preset 为 custom 时生效"}),
            "height": ("INT", {"default": 480, "min": 64, "max": 2048, "step": 32,
                               "tooltip": "resolution_preset 为 custom 时生效"}),
            "resolution_preset": (RESOLUTION_CHOICES, {
                "default": CUSTOM_RESOLUTION,
                "tooltip": "分辨率预设（对齐官方 ResolutionSelector）；"
                           "选非 custom 时覆盖 width/height"}),

            "seconds": ("FLOAT", {
                "default": 2.0, "min": 0.2, "max": 15.0, "step": 0.5,
                "tooltip": "时长(秒@24fps)。引擎会向上对齐到 5+17k 帧"}),
            "steps": ("INT", {"default": 20, "min": 2, "max": 1000,
                              "tooltip": "引擎要求 2~1000；turbo LoRA 用 4/8，无 LoRA 用 20"}),
            "seed": ("INT", {"default": 42, "min": 0, "max": 2**31 - 1}),
        }

    @classmethod
    def _common_optional(cls):
        return {
            "lora": ("STRING", {
                "default": "",
                "tooltip": "LoRA 路径；多个用逗号分隔，按顺序合并（如 default,turbo）"}),
            "auto_steps": ("BOOLEAN", {
                "default": True,
                "tooltip": "按 LoRA 名自动校正步数：turbo/4step/8step → 4/8 步，无 LoRA → 20 步"}),
            "model_dir": ("STRING", {"default": _DEFAULT_MODEL_DIR,
                                     "tooltip": "含 FL2VA/（R2V 还需 Ref2VA/）的模型根目录"}),
            "output_path": ("STRING", {
                "default": "",
                "tooltip": "留空则写入 ComfyUI 临时目录；指定则额外另存一份"}),
            "core_reuse": ("INT", {"default": 1, "min": 1, "max": 6,
                                   "tooltip": "核心刷新间隔：1 精确，4 快速，6 激进"}),
            "reuse": ("INT", {"default": 1, "min": 1, "max": 3,
                              "tooltip": "去噪复用：1 close，2 fast，3 aggressive"}),
            "layers": ("INT", {"default": 0, "min": 0, "max": 50,
                               "tooltip": "DiT 块数：0=默认，50 精确，45 快，40 激进"}),
            "binary": ("STRING", {"default": _DEFAULT_BINARY}),
            "clipproj_dir": ("STRING", {"default": _DEFAULT_CLIPPROJ_DIR}),
            "clipproj_proj": ("STRING", {"default": _DEFAULT_CLIPPROJ_PROJ}),
            "extra_args": ("STRING", {"default": "",
                                      "tooltip": "附加 CLI 参数（直接拼到命令行）"}),
        }

    def _resolve_inputs(self, width, height, resolution_preset,
                        steps, lora, auto_steps):
        """应用分辨率预设、自动步数、32 对齐校验；返回 (width, height, steps)。"""
        # 1) 分辨率预设（对齐官方 ResolutionSelector）
        if resolution_preset and resolution_preset != CUSTOM_RESOLUTION:
            preset = RESOLUTION_PRESETS.get(resolution_preset)
            if preset:
                if (width, height) != preset:
                    print(f"[H3] 分辨率预设 '{resolution_preset}' → "
                          f"{preset[0]}x{preset[1]}（覆盖 {width}x{height}）")
                width, height = preset

        # 2) 宽高必须是 32 的倍数（latent = 像素/16 且需为偶数）
        if width % MULTIPLE or height % MULTIPLE:
            nw = _round_mult(width)
            nh = _round_mult(height)
            print(f"[H3] 警告: 宽高需为 {MULTIPLE} 的倍数，{width}x{height} → {nw}x{nh}")
            width, height = nw, nh

        # 3) 步数建议 / 自动校正（turbo 蒸馏模型必须少步数）
        recommended, reason = suggest_steps(lora)
        if recommended is None:
            if steps < 8:
                print(f"[H3] 提示: steps={steps}；{reason}")
        elif steps == recommended:
            print(f"[H3] 步数 OK: steps={steps}（{reason}）")
        elif auto_steps:
            print(f"[H3] 自动步数: steps {steps} → {recommended}（{reason}）")
            steps = recommended
        else:
            print(f"[H3] 提示: {reason}（当前 steps={steps}，"
                  f"开启 auto_steps 可自动校正）")

        print(f"[H3] 最终参数: {width}x{height}, {steps} 步")
        return width, height, steps

    def _build_cmd(self, binary, model_dir, prompt, width, height, seconds,
                   steps, seed, lora, output_path, core_reuse, reuse, layers,
                   extra_args, extra=None):
        if not os.path.isfile(binary):
            raise FileNotFoundError(
                f"H3 引擎二进制不存在: {binary}\n"
                f"请先在 h3.c 目录执行 `make`，或用 `binary` 参数/环境变量 H3_BINARY 指定路径")
        if not os.path.isdir(model_dir):
            raise FileNotFoundError(f"模型目录不存在: {model_dir}")
        if not (prompt and prompt.strip()):
            raise ValueError(
                "提示词为空：ComfyUI 节点的 prompt 框未填写或仅含空白。"
                "请输入有效提示词（否则引擎会生成与提示词无关的噪声画面）。")

        cmd = [binary, "-d", model_dir, "-p", prompt.strip(),
               "--width", str(width), "--height", str(height),
               "--seconds", f"{float(seconds):g}",
               "--steps", str(steps), "--seed", str(seed),
               "-o", output_path]
        if lora:
            cmd += ["--lora", lora]
        if core_reuse and core_reuse > 1:
            cmd += ["--core-reuse", str(core_reuse)]
        if reuse and reuse > 1:
            cmd += ["--reuse", str(reuse)]
        if layers and layers > 0:
            cmd += ["--layers", str(layers)]
        if extra:
            cmd += extra
        if extra_args:
            cmd += extra_args.split()
        return cmd

    def _output_path(self, output_path, prefix):
        if output_path:
            # 绝对化：引擎的 cwd 是二进制目录，相对路径会落在那里
            abs_path = os.path.abspath(output_path)
            os.makedirs(os.path.dirname(abs_path), exist_ok=True)
            return abs_path
        d = os.path.join(_comfy_temp_dir(), "h3_binary_out")
        os.makedirs(d, exist_ok=True)
        return os.path.join(d, f"{prefix}_{int(time.time() * 1000)}.mp4")

    def _finish(self, rc, out_path, log):
        if rc != 0 or not os.path.isfile(out_path):
            tail = "\n".join(log.splitlines()[-25:])
            raise RuntimeError(
                f"H3 引擎失败 (rc={rc})，输出文件{'存在' if os.path.isfile(out_path) else '不存在'}\n"
                f"--- 日志尾部 ---\n{tail}")
        size_mb = os.path.getsize(out_path) / 1024 / 1024
        print(f"[H3] 完成: {out_path} ({size_mb:.2f} MB)")
        return (_make_video(out_path), out_path)


class H3_BinaryT2V(_H3BinaryBase):
    """文本 / 首尾帧 → 视频（FL2VA 路径）。"""

    @classmethod
    def INPUT_TYPES(cls):
        req = cls._common_required()
        opt = cls._common_optional()
        opt.update({
            "first_frame": ("IMAGE", {"tooltip": "首帧条件图（可选）"}),
            "last_frame": ("IMAGE", {"tooltip": "尾帧条件图（可选）"}),
        })
        return {"required": req, "optional": opt}

    DESCRIPTION = ("H3 引擎(C 二进制) 文本/首尾帧 → 视频+音频；"
                   "可替换官方 t2v 工作流的整条采样链，直连 SaveVideo")

    def generate(self, prompt, width, height, resolution_preset, seconds, steps, seed,
                 lora="", auto_steps=True, model_dir=_DEFAULT_MODEL_DIR, output_path="",
                 core_reuse=1, reuse=1, layers=0, binary=_DEFAULT_BINARY,
                 clipproj_dir=_DEFAULT_CLIPPROJ_DIR,
                 clipproj_proj=_DEFAULT_CLIPPROJ_PROJ, extra_args="",
                 first_frame=None, last_frame=None):
        width, height, steps = self._resolve_inputs(
            width, height, resolution_preset, steps, lora, auto_steps)
        extra = []
        if first_frame is not None:
            extra += ["--first-frame", _save_image(first_frame, "first_frame.png")]
        if last_frame is not None:
            extra += ["--last-frame", _save_image(last_frame, "last_frame.png")]

        out_path = self._output_path(output_path, "h3_t2v")
        cmd = self._build_cmd(binary, model_dir, prompt, width, height, seconds,
                              steps, seed, lora, out_path, core_reuse, reuse,
                              layers, extra_args, extra)
        rc, log = _run_engine(cmd, {"H3_CLIPPROJ_DIR": clipproj_dir,
                                    "H3_CLIPPROJ_PROJ": clipproj_proj}, tag="H3_T2V")
        return self._finish(rc, out_path, log)


class H3_BinaryR2V(_H3BinaryBase):
    """参考图 / 参考视频 / 参考音频 → 视频（Ref2VA 路径）。"""

    @classmethod
    def INPUT_TYPES(cls):
        req = cls._common_required()
        opt = cls._common_optional()
        opt.update({
            "ref_image_1": ("IMAGE", {"tooltip": "第 1 张参考图（提示词中用 <Picture 1> 引用）"}),
            "ref_image_2": ("IMAGE", {"tooltip": "第 2 张参考图（<Picture 2>）"}),
            "ref_image_size": (["match", "max"], {"default": "match"}),
            "ref_video_path": ("STRING", {"default": "", "tooltip": "参考视频文件路径（含音频）"}),
            "ref_audio_path": ("STRING", {"default": "", "tooltip": "独立参考音频文件路径"}),
        })
        return {"required": req, "optional": opt}

    DESCRIPTION = ("H3 引擎(C 二进制) 参考图/视频/音频 → 视频+音频；"
                   "需已安装 Ref2VA 权重；可替换官方 r2v 工作流的整条采样链")

    def generate(self, prompt, width, height, resolution_preset, seconds, steps, seed,
                 lora="", auto_steps=True, model_dir=_DEFAULT_MODEL_DIR, output_path="",
                 core_reuse=1, reuse=1, layers=0, binary=_DEFAULT_BINARY,
                 clipproj_dir=_DEFAULT_CLIPPROJ_DIR,
                 clipproj_proj=_DEFAULT_CLIPPROJ_PROJ, extra_args="",
                 ref_image_1=None, ref_image_2=None, ref_image_size="match",
                 ref_video_path="", ref_audio_path=""):
        width, height, steps = self._resolve_inputs(
            width, height, resolution_preset, steps, lora, auto_steps)
        extra = []
        for idx, img in enumerate((ref_image_1, ref_image_2), start=1):
            if img is not None:
                extra += ["--ref-image", _save_image(img, f"ref_image_{idx}.png")]
        if ref_image_size:
            extra += ["--ref-image-size", ref_image_size]
        if ref_video_path:
            if not os.path.isfile(ref_video_path):
                raise FileNotFoundError(f"参考视频不存在: {ref_video_path}")
            extra += ["--ref-video", ref_video_path]
        if ref_audio_path:
            if not os.path.isfile(ref_audio_path):
                raise FileNotFoundError(f"参考音频不存在: {ref_audio_path}")
            extra += ["--ref-audio", ref_audio_path]

        out_path = self._output_path(output_path, "h3_r2v")
        cmd = self._build_cmd(binary, model_dir, prompt, width, height, seconds,
                              steps, seed, lora, out_path, core_reuse, reuse,
                              layers, extra_args, extra)
        rc, log = _run_engine(cmd, {"H3_CLIPPROJ_DIR": clipproj_dir,
                                    "H3_CLIPPROJ_PROJ": clipproj_proj}, tag="H3_R2V")
        return self._finish(rc, out_path, log)


class H3_BinaryInfo:
    """运行 `h3 --info`，查看设备与模型权重清单（排查环境问题用）。"""

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "model_dir": ("STRING", {"default": _DEFAULT_MODEL_DIR}),
            "binary": ("STRING", {"default": _DEFAULT_BINARY}),
            "clipproj_dir": ("STRING", {"default": _DEFAULT_CLIPPROJ_DIR}),
            "clipproj_proj": ("STRING", {"default": _DEFAULT_CLIPPROJ_PROJ}),
        }}

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("info",)
    FUNCTION = "run"
    CATEGORY = CATEGORY
    DESCRIPTION = "运行 h3 --info（设备 + 各组件权重文件数/GiB）"

    def run(self, model_dir, binary, clipproj_dir, clipproj_proj):
        if not os.path.isfile(binary):
            raise FileNotFoundError(f"H3 引擎二进制不存在: {binary}")
        rc, log = _run_engine([binary, "-d", model_dir, "--info"],
                              {"H3_CLIPPROJ_DIR": clipproj_dir,
                               "H3_CLIPPROJ_PROJ": clipproj_proj}, tag="H3_INFO")
        return (log,)
