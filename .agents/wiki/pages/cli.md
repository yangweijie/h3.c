# 命令行交互: h3_cli / main / terminal

命令行体验由三部分组成：`main.c`（进程入口与参数解析）、`h3_cli.c`（生成循环与交互提示）、`h3_terminal.c` + `linenoise.c`（行编辑/历史/信号）。

## 入口

`main.c` 负责：
- 解析命令行选项到 `h3_params`（默认值 `H3_PARAMS_DEFAULT`）。
- 解析 `--model`（含 `shader_source_path`、`H3_FFMPEG`/`H3_FFPROBE` 定位）。
- 调用 `h3_load_dir()` 建立 `h3_ctx`。
- 进入 `h3_cli_run()`；或在非交互模式下直接 `h3_generate()`。

## 交互循环

`h3_cli_run()` 维护一个交互式提示，允许用户反复提交提示与参考（图片/视频/音频/角色文本），每次提交触发一次 `h3_generate()`。它把 `on_frame` 回调接驳到终端预览（配合 `linenoise` 实现无闪烁刷新），并通过 `on_progress` / `on_elapsed` 显示进度与耗时。交互模式下种子默认随机，与一次性模式的默认 42 不同。

## 终端原语

`h3_terminal.c` 封装 ANSI 转义、逐帧绘制（用于 `--preview-denoise` 的逐 Euler 步预览）、以及 SIGINT 处理（干净中断生成）。`linenoise.c` 是第三方行编辑库（MIT），提供历史回溯与基本补全。

## 与公共 API 的关系

CLI 只是 `h3.h` 公共 API 的一个宿主示例。任何宿主（GUI、服务端、脚本）都可以跳过 `h3_cli`，直接调用 `h3_load_dir()` → `h3_generate()` → `h3_unload_dir()`，并通过 `on_frame` 回调接收结果。见 [公共 API](api-public.md)。
