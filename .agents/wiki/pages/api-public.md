# 公共 API: h3.h / h3.c

`h3.h` 是引擎对外的纯 C 头文件，定义了版本宏、参数结构体、结果结构体、以及全部公共函数声明。`h3.c` 实现这些函数的编排逻辑，是生成流程的"中央调度器"。

## 版本与宏

- `H3_VERSION "0.1.0-dev"`
- 默认值：`H3_DEFAULT_WIDTH=864`、`H3_DEFAULT_HEIGHT=480`、`H3_DEFAULT_FRAMES=56`、`H3_DEFAULT_STEPS=20`、`H3_DEFAULT_DIT_LAYERS=50`
- 约束：`H3_MIN_DIT_LAYERS=35`、`H3_MIN_CORE_REUSE=1`、`H3_MAX_CORE_REUSE` 等

## 参数与结果

`struct h3_params`（默认值 `H3_PARAMS_DEFAULT`）携带所有生成配置：画布尺寸、帧数、步数、种子、各加速/质量开关，以及输出回调 `on_frame`、日志/进度回调、`on_start` / `on_end` 钩子。`struct h3_result` 报告最终 `width/height/frames/fps/sample_rate/seed`。这两个结构体与 `h3_get_param()` / `h3_set_param()` 一起，让宿主在不重新链接的情况下调参。

## 生命周期 API

- `h3_load_dir(const char *dir, const char *shader_source_path, const char *ffmpeg, const char *ffprobe)` → `h3_ctx *`：打开模型目录与 Metal 着色器，初始化设备。不映射权重。
- `h3_unload_dir(h3_ctx *)`：释放上下文与已加载权重。
- `h3_model_info(h3_ctx *)` / `h3_device_info(h3_ctx *)`：返回只读元数据。

## 生成 API

- `h3_generate(h3_ctx *, h3_params *, const char *prompt, const h3_ref *refs, int n_refs, h3_result *out)`：一次性文本/参考到视频-音频生成，结果通过回调增量交付。
- `h3_generate_incremental(...)`：逐块（chunk）生成，便于交互式预览。

## 缓存 API

`h3_cache_enabled()` / `h3_cache_set_enabled()`、`h3_cache_clear()`：控制跨请求复用的条件/DiT/解码器缓存；默认关闭以匹配一次性调用的内存生命周期。

## 多模态引用类型

`h3_ref`（见 `h3.h`）描述参考输入：图片、视频、音频、或文本"角色"（如 `system`/`assistant`）；`h3_new_image_ref` / `h3_new_video_ref` / `h3_new_audio_ref` / `h3_new_text_ref` 构造它们。文本引用（如角色设定）不进入视觉编码器，而是作为对话上下文拼接到提示。

编排实现要点：在 `h3.c` 中，`h3_generate()` 负责调用 tokenizer→text/vision encoder→multimodal→dit→vae→ffmpeg 的全链路；它通过 `h3_host` 计算张量布局，通过 `h3_cache_*` 决定复用，通过 `h3_dit_denoise_euler()` 驱动去噪，并通过 `h3_ffmpeg` 与 `on_frame` 回调产出最终媒体。下游模块见 [宿主逻辑与张量布局](host.md) 与 [命令行交互](cli.md)。
