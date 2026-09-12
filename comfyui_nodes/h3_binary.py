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
import shlex
import shutil
import subprocess
import time
from collections import deque

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
    arr = np.asarray(arr, dtype="float32")
    # ComfyUI 的 IMAGE 是 [N,H,W,C]（N 通常为 1）。原先只判断 ndim==3，
    # 于是 [1,H,W,C] 会被原样交给 PIL 而报错；这里按维度统一归一到 [H,W,C]。
    if arr.ndim == 4:
        if arr.shape[0] != 1:
            print(f"[H3] 提示: 输入是 {arr.shape[0]} 帧的批次，只使用第 1 帧")
        arr = arr[0]
    if arr.ndim != 3 or arr.shape[-1] not in (3, 4):
        raise ValueError(
            f"图像张量形状无法识别: {tuple(arr.shape)}（期望 [N,H,W,C]）")
    arr = np.clip(arr, 0.0, 1.0)
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
    # 进度行会逐条刷新，长任务下日志可以很长；只保留尾部即可
    # （错误信息只用到最后 25 行）。
    lines = deque(maxlen=2000)
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
        if proc.stdout is not None:
            proc.stdout.close()
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
            # lora 放 required：optional 里的 widget 会被前端折叠（序列化成
            # "shape": 7）而看不见。它本就是 optional 首项，移到这里不改变
            # 前端 widget 顺序（required 整体排在 optional 之前），旧工作流
            # 的 widgets_values 依然对位。
            "lora": ("STRING", {
                "default": "",
                "tooltip": "LoRA 路径；多个用逗号分隔，按顺序合并（如 default,turbo）"}),
            # 以下几项原先在 optional：optional 里的 widget 会被前端折叠
            # （序列化成 "shape": 7）而在节点上看不见。把它们整体前移成 required
            # 的「连续前缀」，前端 widget 顺序就与改动前逐项一致，旧工作流的
            # widgets_values 依然对位 —— 不能只挑其中几项移动，那样会打乱顺序。
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
            "fast_stream": ("BOOLEAN", {
                "default": False,
                "tooltip": "低分辨率流式提速（H3_DIT_STREAM_WORKERS=2）：把 DiT 权重流式预取"
                           "拆成多 worker 并与 CPU 反旋转重叠。实测 256×256 1.70×、384×384 1.27×；"
                           "512×512 及以上无效（GPU 已完全掩盖预取）且略慢，故默认关闭"}),
        }

    @classmethod
    def _common_optional(cls):
        return {
            # 纯环境配置，默认值即正确值，留在 optional（折叠）里避免节点过长。
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

        # 引擎硬限制：core reuse 与 denoiser reuse 不能同时 >1，
        # 否则引擎直接报错 `core reuse and denoiser reuse cannot be combined`。
        if core_reuse and core_reuse > 1 and reuse and reuse > 1:
            raise ValueError(
                "core_reuse 与 reuse 不能同时 > 1（引擎硬限制：二者互斥）。\n"
                "二者都是「复用近似」提速手段，只能二选一：\n"
                "  - core_reuse：每 N 步重算一次核心激活（1 精确 / 4 快速 / 6 激进）\n"
                "  - reuse：去噪步间细粒度复用（1 close / 2 fast / 3 aggressive）\n"
                "把其中一个设回 1 即可。")

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
            try:
                cmd += shlex.split(extra_args)
            except ValueError as exc:
                raise ValueError(
                    f"extra_args 解析失败（引号不配对？）: {extra_args}") from exc
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

    def _engine_env(self, clipproj_dir, clipproj_proj, fast_stream):
        """引擎子进程环境变量。fast_stream → H3_DIT_STREAM_WORKERS=2，
        引擎侧即开启「分块 + 多 worker + 读/算重叠」整条流式快路径。"""
        env = {"H3_CLIPPROJ_DIR": clipproj_dir, "H3_CLIPPROJ_PROJ": clipproj_proj}
        if fast_stream:
            env["H3_DIT_STREAM_WORKERS"] = "2"
        return env

    @staticmethod
    def _fast_stream_hint(width, height, fast_stream):
        """低分辨率落在这个提速真正有效的区间里（实测数据见 h3.c README
        「Streamed DiT weight prefetch」），提醒用户可以打开。

        判据只看分辨率：该提速是否生效取决于「权重预取是否在关键路径上」，
        即 GPU 算力量，与视频时长无关。README 实测 256x256 1.67x、
        384x384 1.27x、512x512 及以上无效（预取已被 GPU 完全掩盖）。"""
        if fast_stream:
            return
        if width * height <= 384 * 384:
            print(f"[H3] 提示: {width}x{height} 属于流式提速有效区间，"
                  f"开启 fast_stream 可获约 1.2~1.7x（实测数据，非估计）")

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
                 core_reuse=1, reuse=1, layers=0, fast_stream=False,
                 binary=_DEFAULT_BINARY,
                 clipproj_dir=_DEFAULT_CLIPPROJ_DIR,
                 clipproj_proj=_DEFAULT_CLIPPROJ_PROJ, extra_args="",
                 first_frame=None, last_frame=None):
        width, height, steps = self._resolve_inputs(
            width, height, resolution_preset, steps, lora, auto_steps)
        self._fast_stream_hint(width, height, fast_stream)
        extra = []
        if first_frame is not None:
            extra += ["--first-frame", _save_image(first_frame, "first_frame.png")]
        if last_frame is not None:
            extra += ["--last-frame", _save_image(last_frame, "last_frame.png")]

        out_path = self._output_path(output_path, "h3_t2v")
        cmd = self._build_cmd(binary, model_dir, prompt, width, height, seconds,
                              steps, seed, lora, out_path, core_reuse, reuse,
                              layers, extra_args, extra)
        rc, log = _run_engine(cmd, self._engine_env(clipproj_dir, clipproj_proj,
                                                    fast_stream), tag="H3_T2V")
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
                 core_reuse=1, reuse=1, layers=0, fast_stream=False,
                 binary=_DEFAULT_BINARY,
                 clipproj_dir=_DEFAULT_CLIPPROJ_DIR,
                 clipproj_proj=_DEFAULT_CLIPPROJ_PROJ, extra_args="",
                 ref_image_1=None, ref_image_2=None, ref_image_size="match",
                 ref_video_path="", ref_audio_path=""):
        width, height, steps = self._resolve_inputs(
            width, height, resolution_preset, steps, lora, auto_steps)
        self._fast_stream_hint(width, height, fast_stream)
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
        rc, log = _run_engine(cmd, self._engine_env(clipproj_dir, clipproj_proj,
                                                    fast_stream), tag="H3_R2V")
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
    # 输出没有被其它节点消费时，ComfyUI 不会执行这个节点；而它的用途恰恰是
    # 「跑一下看日志」，所以必须自行声明为输出节点，否则不连线就完全不跑。
    OUTPUT_NODE = True

    def run(self, model_dir, binary, clipproj_dir, clipproj_proj):
        if not os.path.isfile(binary):
            raise FileNotFoundError(f"H3 引擎二进制不存在: {binary}")
        rc, log = _run_engine([binary, "-d", model_dir, "--info"],
                              {"H3_CLIPPROJ_DIR": clipproj_dir,
                               "H3_CLIPPROJ_PROJ": clipproj_proj}, tag="H3_INFO")
        return (log,)


# ---------------------------------------------------------------------------
# 特殊开关 / 环境变量 说明（Markdown）。作为可编辑文本节点呈现；复制本节点的
# notes 输出到 ComfyUI 内置 "Note" 节点即可在画布上渲染 Markdown。
# ---------------------------------------------------------------------------
H3_SWITCHES_DOC = """\
# H3 引擎 特殊开关与环境变量

> 这些开关都是「近似 / 提速」手段：开大会更快，但输出**不再与精确路径逐 bit 一致**。
> 追求可复现 / 最高质量时全部回到默认值（core_reuse=1, reuse=1, layers=0, fast_stream=关）。

## 节点加速开关

### core_reuse（--core-reuse，默认 1，1~6）
核心张量复用间隔。去噪时每 N 步才完整重算一次「核心」中间激活（如部分 hidden state），其间的步直接复用上一次结果做近似。
- 1 = 每步重算（close，最精确、最慢）
- 4 = 快速（fast）
- 6 = 激进（aggressive，最快、画质损失最大）
适用：质量优先保持 1；想提速又不舍太多画质用 4；快速草稿 / 预览用 6。
⚠ 与 `reuse` 互斥：二者不能同时 >1（引擎报错 `core reuse and denoiser reuse cannot be combined`），只能二选一。

### reuse（--reuse，默认 1，常用 1~3，引擎上限更高）
去噪步间细粒度复用。在连续去噪步之间复用 DiT block 内部的中间结果（attention 上下文、投影缓存等），比 core_reuse 颗粒更细。
- 1 = close（逐步精确）
- 2 = fast
- 3 = aggressive（最快）
适用：与 core_reuse **互斥**——二者不能同时 >1（引擎报错 `core reuse and denoiser reuse cannot be combined`），只能二选一用于提速，不要同时开。

### layers（--layers，默认 0 = 50 块，0~50）
实际参与的 DiT block 数。
- 0 = 用 checkpoint 完整块数（默认 50，精确）
- 45 = 快
- 40 = 激进
直接砍掉后半段 transformer 层，提速显著但画质下降。只缩短 DiT 主干，不影响 tokenizer / VAE。
适用：低质量快速预览。

### fast_stream（节点开关 → H3_DIT_STREAM_WORKERS=2，默认关）
把 DiT 权重流式预取拆成多 worker 并与 CPU 反旋转重叠。仅在低分辨率（≤384²）权重预取处于关键路径时有效：实测 256²≈1.67×、384²≈1.27×；512² 及以上 GPU 已完全掩盖预取，无效且略慢，故默认关。节点会在低分辨率未开启时主动提示。
适用：低分辨率批量出图时打开。

### extra_args（附加 CLI 参数）
直接拼到引擎命令行，用于传节点未暴露的开关，例如：
- `--video-vae-streaming 0`：强制 VAE 解码器常驻（快，但常驻占 ~9 GiB；16 GB 机器在干扰下可能 OOM）
- `--video-vae-streaming 1`：强制 VAE 流式（只占 ~0.25 GiB，慢一些）
- 留空则由自动内存规划器决定（auto）。

## 环境变量（进程级，节点 extra_args / 系统环境均可设）

### H3_DIT_RESIDENT_BLOCKS=N（部分常驻 DiT 块，默认 0）
保留前 N 个 DiT 块常驻、只流式其余块，用内存换 I/O。实测在 16 GB 上 wall time 几乎不降（甚至略升），因为常驻块与流式路径争抢统一内存；每块约 0.7 GiB，8 块已占 5+ GiB。仅在内存充裕（约 20 块以上能轻松放下）且磁盘是瓶颈时才有收益；16 GB 不要开。

### H3_VIDEO_VAE_STREAMING / --video-vae-streaming（0|1|-1 auto，默认 auto）
见上「extra_args」。auto 由内存规划器决定：内存紧用流式，内存松用常驻。

### H3_DIT_STREAM_WORKERS / H3_DIT_STREAM_PIPELINE
即 fast_stream 的底层旋钮（=2 启用分块多 worker；PIPELINE 只调重叠）。一般直接用节点的 fast_stream 开关即可。

### H3_PROFILE=1
打印 I/O 字节、吞吐、未被 GPU 隐藏的读等待，用于性能诊断。

### H3_CLIPPROJ_DIR / H3_CLIPPROJ_PROJ
Qwen3-VL 与 ClipProj 路径。节点已通过 clipproj_dir / clipproj_proj 参数自动注入，无需手动设。

### H3_FFMPEG / H3_FFPROBE
ffmpeg / ffprobe 绝对路径覆盖。节点已自动探测 /opt/zerobrew/bin、/usr/local/bin 等并注入，无需手动设。

### H3_BINARY / H3_MODEL_DIR
二进制与模型路径。节点已默认读取，也可在 binary / model_dir 参数覆盖。

## 调参优先级（在节点里）
1. 默认（1/1/0/关）：精确基准。
2. 想快：低分辨率开 fast_stream；低质量预览把 layers 降到 45/40，或把 core_reuse / reuse 调大。
3. 质量没达标：全部回到精确（1/1/0）。
4. 内存紧张（16 GB）：保持 video_vae_streaming 默认（流式），不要开 H3_DIT_RESIDENT_BLOCKS。
5. 内存充裕且要榨速度：video_vae_streaming=0（常驻 VAE）+ fast_stream（低分辨率）+ 视情况 H3_DIT_RESIDENT_BLOCKS（>16 GB 且磁盘慢）。
"""


class H3_BinaryNote:
    """H3 引擎特殊开关 / 环境变量说明（Markdown 文本节点）。

    ComfyUI 仅在自带的 "Note" 节点上做画布级 Markdown 渲染，自定义节点无法获得
    该渲染；本节点把说明作为可编辑文本呈现，复制其 notes 输出到内置 Note 节点即可
    在画布上预览。
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "notes": ("STRING", {"multiline": True, "default": H3_SWITCHES_DOC,
                                 "dynamicPrompts": False}),
        }}

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("notes",)
    FUNCTION = "show"
    CATEGORY = CATEGORY
    OUTPUT_NODE = True
    DESCRIPTION = ("H3 引擎特殊开关与环境变量说明（Markdown）；"
                   "复制 notes 到 ComfyUI 内置 Note 节点即可画布预览")

    def show(self, notes):
        return (notes,)
