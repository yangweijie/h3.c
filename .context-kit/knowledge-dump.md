# h3.c 知识库（context-kit 导出）

- 工作区：`/Volumes/data/git/c/h3c`
- 条目数：45
- 生成时间：2026-09-13T03:01:17.606Z
- 数据位置：`~/.context-kit/data/knowledge/h3c-8cd7ead6/`

> 本文件由 context-kit 知识库导出，用于人工审阅。条目正文存放在
> `entries/*.md`，`index.json` 只保存元数据与摘要。

## 目录

- [MiniMax-H3 原生推理引擎（仓库总览）](#minimax-h3-原生推理引擎-仓库总览-)  `overview / 仓库级`
- [构建、测试与运行命令](#构建-测试与运行命令)  `unique_setup_and_commands / 仓库级`
- [环境变量与运行时开关](#环境变量与运行时开关)  `configuration_system / 仓库级`
- [错误处理与日志约定](#错误处理与日志约定)  `error_handling / 仓库级`
- [构建系统（Makefile）](#构建系统-makefile-)  `build_system / 仓库级`
- [公共 API 与生成流水线](#公共-api-与生成流水线)  `overview / 模块`
- [公共 API 与生成流水线 - Architecture Design](#公共-api-与生成流水线-architecture-design)  `architecture_design / 模块`
- [公共 API 与生成流水线 - Coding Conventions](#公共-api-与生成流水线-coding-conventions)  `coding_conventions / 模块`
- [Host 基础层（几何、调度、布局、RNG）](#host-基础层-几何-调度-布局-rng-)  `overview / 模块`
- [Host 基础层（几何、调度、布局、RNG） - Architecture Design](#host-基础层-几何-调度-布局-rng-architecture-design)  `architecture_design / 模块`
- [Host 基础层（几何、调度、布局、RNG） - Tech Stack](#host-基础层-几何-调度-布局-rng-tech-stack)  `tech_stack / 模块`
- [GPU / Metal 后端](#gpu-metal-后端)  `overview / 模块`
- [GPU / Metal 后端 - Architecture Design](#gpu-metal-后端-architecture-design)  `architecture_design / 模块`
- [GPU / Metal 后端 - Tech Stack](#gpu-metal-后端-tech-stack)  `tech_stack / 模块`
- [GPU / Metal 后端 - Coding Conventions](#gpu-metal-后端-coding-conventions)  `coding_conventions / 模块`
- [权重加载与 safetensors](#权重加载与-safetensors)  `overview / 模块`
- [权重加载与 safetensors - Architecture Design](#权重加载与-safetensors-architecture-design)  `architecture_design / 模块`
- [权重加载与 safetensors - Tech Stack](#权重加载与-safetensors-tech-stack)  `tech_stack / 模块`
- [文本与视觉编码器（Qwen3-VL）](#文本与视觉编码器-qwen3-vl-)  `overview / 模块`
- [文本与视觉编码器（Qwen3-VL） - Architecture Design](#文本与视觉编码器-qwen3-vl-architecture-design)  `architecture_design / 模块`
- [文本与视觉编码器（Qwen3-VL） - Tech Stack](#文本与视觉编码器-qwen3-vl-tech-stack)  `tech_stack / 模块`
- [Tokenizer（BPE）](#tokenizer-bpe-)  `overview / 模块`
- [Tokenizer（BPE） - Architecture Design](#tokenizer-bpe-architecture-design)  `architecture_design / 模块`
- [多模态条件构建](#多模态条件构建)  `overview / 模块`
- [多模态条件构建 - Architecture Design](#多模态条件构建-architecture-design)  `architecture_design / 模块`
- [DiT 去噪主干](#dit-去噪主干)  `overview / 模块`
- [DiT 去噪主干 - Architecture Design](#dit-去噪主干-architecture-design)  `architecture_design / 模块`
- [DiT 去噪主干 - Coding Conventions](#dit-去噪主干-coding-conventions)  `coding_conventions / 模块`
- [DiT 调度与 AdaLN 预计算](#dit-调度与-adaln-预计算)  `overview / 模块`
- [DiT 调度与 AdaLN 预计算 - Architecture Design](#dit-调度与-adaln-预计算-architecture-design)  `architecture_design / 模块`
- [LoRA 合并](#lora-合并)  `overview / 模块`
- [LoRA 合并 - Architecture Design](#lora-合并-architecture-design)  `architecture_design / 模块`
- [显存分层规划](#显存分层规划)  `overview / 模块`
- [显存分层规划 - Architecture Design](#显存分层规划-architecture-design)  `architecture_design / 模块`
- [视频 VAE 解码器](#视频-vae-解码器)  `overview / 模块`
- [视频 VAE 解码器 - Architecture Design](#视频-vae-解码器-architecture-design)  `architecture_design / 模块`
- [音频 VAE（BigVGAN）](#音频-vae-bigvgan-)  `overview / 模块`
- [音频 VAE（BigVGAN） - Architecture Design](#音频-vae-bigvgan-architecture-design)  `architecture_design / 模块`
- [媒体封装与 ffmpeg 管线](#媒体封装与-ffmpeg-管线)  `overview / 模块`
- [媒体封装与 ffmpeg 管线 - Architecture Design](#媒体封装与-ffmpeg-管线-architecture-design)  `architecture_design / 模块`
- [CLI 与交互会话](#cli-与交互会话)  `overview / 模块`
- [CLI 与交互会话 - Architecture Design](#cli-与交互会话-architecture-design)  `architecture_design / 模块`
- [CLI 与交互会话 - Setup & Commands](#cli-与交互会话-setup-commands)  `unique_setup_and_commands / 模块`
- [测试体系](#测试体系)  `overview / 模块`
- [测试体系 - Build System](#测试体系-build-system)  `build_system / 模块`

---

## MiniMax-H3 原生推理引擎（仓库总览）

category: `overview` · type: `repository`

`h3.c` 是一个**纯 C / Objective-C**（仅 macOS + Apple Silicon）实现的 MiniMax-H3 推理引擎：把官方 H3 的 FL2VA（text-to-video-audio）与 Ref2VA（reference-to-video-audio）扩散流水线重写到 Metal 上本地运行，**不依赖 Python 与 torch**。CLI 二进制是 `h3`，可嵌入库是 `libh3.a`。

**六阶段流水线**（描述任何模块时先定位它属于哪一阶段）：
1. 加载 —— `h3_load_dir` 校验模型目录布局、解析 `h3_model_info`、惰性打开 safetensors 分片。
2. 提示解析 —— `h3_parse_prompt` 把提示词转成 `h3_layout`（分段结构、位置 ID、条件行）与 `h3_ref` 引用列表。
3. 条件构建 —— 文本 tokenize、图像/视频过视觉编码器、音频过 AudioVAE、拼装 `<Picture n>` 占位符，再跑 Qwen3-VL 前 50 层得到 BF16 条件 embedding。
4. 去噪 —— `h3_denoise` → `h3_dit_denoise_euler`，在 50 块 DiT 上做 20 步 Euler 采样。
5. 解码 —— 视频 latent 过 `h3_video_vae`（分块解码，可选逐步预览），音频 latent 过 `h3_audio_vae`（BigVGAN）。
6. 封装 —— `h3_mux` / `h3_ffmpeg` 把 RGB 帧 + F32 PCM 交给外部 `ffmpeg` 产出 MP4+AAC。

**规模**：约 2.7 万行。最大的三个文件是 `h3_gpu.m`(5821)、`h3_dit.c`(5136)、`h3.c`(2505)。全部内核源码集中在单个 `h3_shaders.metal`（121 个 `kernel void`，见 `h3_shaders.metal:1-9`）。

**默认参数**（`h3.h:155-165`）：864×480、56 帧、20 步、seed 42；版本 `H3_VERSION "0.1.0-dev"`（`h3.h:12`）。

**权重依赖**：需要 Hugging Face 的 MiniMax-H3 快照（`FL2VA/`、`Ref2VA/`、`text_encoder/`、`video_vae/`、`audio_vae/`）与运行期可用的 `ffmpeg` / `ffprobe`。

---

## 构建、测试与运行命令

category: `unique_setup_and_commands` · type: `repository`

**构建**：`make`（或 `make -j8`）产出 `h3` 与 `libh3.a`（`Makefile:31-37`）。必须用 Apple clang：`make CC="xcrun clang"`。Homebrew LLVM 会在 macOS 26 SDK 的 Accelerate/vecLib 头文件上失败（`unrecognized platform name visionOS`），而旧 SDK（15.4）缺少 `MTLGPUFamilyMetal4`，所以只有「26.x SDK + Apple clang」这一组合可用。

**必须从仓库根目录运行**：`h3_gpu.m` 运行时编译 `h3_shaders.metal`，默认路径是相对进程 CWD 的字面量字符串（`h3_gpu.m:364-366`）。测试会显式传路径，CLI 不会。

**运行**：`./h3 -d ./MiniMax-H3 -p "..." -o out.mp4`；`./h3 -d ./MiniMax-H3 --info` 只检查布局与设备、不映射权重；不给 `-p` 则进入交互会话。`./h3 --help` 是权威参数列表。

**条件选择靠参数本身**：Ref2VA 由 `--ref-image` / `--ref-video` / `--ref-silent-video` / `--ref-video-audio V A` / `--ref-audio` 触发；FL2VA 的首/尾帧条件用 `--first-frame` / `--last-frame`。

**测试**：`make test` 跑全套（`Makefile:139-227`）。绝大多数二进制在缺权重/fixture 时**自动跳过**：没有权重时真正运行的只有 `h3_tests` 与 `h3_audio_gpu_tests`。
- 单个套件：`make h3_lora_tests && ./h3_lora_tests`。注意 `h3_lora_tests` **不在** `make test` 内，且需要先跑 `tests/gen_lora_data.py` 生成 `tmp_lora_test/` fixture。
- 直接跑：`./h3_tests`；部分需要参数，如 `./h3_metal_tests misc/fixtures/h3_dit.safetensors`、`./h3_tokenizer_tests MiniMax-H3/tokenizer/tokenizer.json`。
- `make test AUDIO_VAE_MODEL=<model root>` 会额外跑 AudioVAE 端到端测试（`tests/test_real_audio_vae_e2e.c`），它不需要 `misc/fixtures` oracle。
- 数值一致性：`make parity`（Metal vs BF16，需 `misc/fixtures/h3_dit*.safetensors`）、`make real-parity`（对真实权重）。
- ClipProj 保真度：`make clipproj-golden`，模型路径用 `QWEN4B` / `PROJ` / `CLIPPROJ_MODEL` 覆盖（`Makefile:19-21`）。
- `make clean` 清理产物。

---

## 环境变量与运行时开关

category: `configuration_system` · type: `repository`

项目**没有配置文件**：全部运行期行为通过命令行参数 + 环境变量控制。代码中共出现 100+ 个 `H3_*` 环境变量，绝大多数是「诊断 / A-B 对比」用途，用于关闭某条优化路径以确认它的贡献。

**按用途分组（有代表性的）：**
- 性能诊断：`H3_PROFILE`（分阶段计时）、`H3_DEBUG_GPU_MEMORY`、`H3_NAX_DIAGNOSTIC`、`H3_DEBUG_CONVROT`、`H3_DEBUG_VISION`。
- 优化开关的「反向」变量（命名约定是 `H3_DISABLE_*`，置位即退回可移植实现）：`H3_DISABLE_INT8_QKV` / `H3_DISABLE_INT8_MLP` / `H3_DISABLE_INT8_ATTENTION_OUT`、`H3_DISABLE_NAX_LINEAR` / `H3_DISABLE_NAX_MLP` / `H3_DISABLE_NAX_MORTON*`、`H3_DISABLE_FUSED_MLP`、`H3_DISABLE_TOKEN_REDUCTION`、`H3_DISABLE_GRAPH_DATA_CACHE`、`H3_DISABLE_COOP_QKV`、`H3_DISABLE_CACHED_QKV` 等。
- 流式与驻留：`H3_DIT_RESIDENT_BLOCKS`、`H3_DIT_STREAM_WORKERS`、`H3_DIT_STREAM_PIPELINE`、`H3_DIT_COMMAND_BLOCKS`、`H3_ZERO_COPY_WEIGHTS`。
- 采样与复用：`H3_REUSE_STEPS`（覆盖 `denoise_reuse` 的命中步）、`H3_CPU_SAMPLER`、`H3_GPU_SAMPLER`、`H3_GPU_SAMPLER_WINDOW`。
- Token reduction 细调：`H3_TOKEN_REDUCTION`、`H3_TOKEN_REDUCTION_BLOCKS`、`H3_TOKEN_REDUCTION_EARLY`、`H3_TOKEN_REDUCTION_SCALE`。
- 外部程序路径：`H3_FFMPEG`、`H3_FFPROBE`。
- 首块缓存：`H3_FB_CACHE`、`H3_FB_CACHE_THRESHOLD`。
- 层裁剪策略：`H3_DIT_LAYER_POLICY`。
- ClipProj 诊断：`H3_CLIPPROJ_DIR`、`H3_CLIPPROJ_PROJ`、`H3_CLIPPROJ_LAYERS`、`H3_CLIPPROJ_DUMP_*`。

注意 `H3_METAL_HAS_TENSOR` 不是用户开关，而是构建期注入 Metal 源码的宏（`h3_gpu.m:373-384`）。

**配置驱动的优先级**：`h3_params` 里的显式设置会覆盖自动内存规划器的建议（见 `h3_memory_plan.h:16-17`）。

---

## 错误处理与日志约定

category: `error_handling` · type: `repository`

**没有断言**：全项目 `.c` / `.m` 中 `assert(` 出现次数为 0。错误一律用返回值 + 错误字符串表达。

**两层错误通道：**
- 公共层：`h3_ctx` 内嵌 `char error[512]`（`h3_internal.h:14`），通过 `h3_set_error()` 写入，格式串带 `__attribute__((format(printf,2,3)))` 编译期校验（`h3_internal.h:38-39`）；调用者用 `h3_last_error(ctx)` 读出（`h3.h:206`）。`h3.c` 内共 118 处 `h3_set_error` 调用。
- GPU 层：`h3_gpu_set_error(gpu, fmt, ...)` / `h3_gpu_error(gpu)` 存取一个 NSString（`h3_gpu.m:224-231`、`h3_gpu.m:1038-1042`）。

**返回码约定**：C API 返回 `int`，0 表示失败 / 1 表示成功；指针 API 失败返回 `NULL`。ObjC 层同样以 `NULL` 表示失败（例如 kernel library 编译失败时 `h3_gpu.m:409-416`）。

**日志**：没有专用日志宏或日志库，直接 `fprintf(stderr, ...)`，由环境变量开关控制。分布很不均匀：`h3_cli.c`(54)、`main.c`(36)、`h3_dit.c`(15)、`h3.c`(14)、`h3_dit_schedule.c`(12)，而 `h3_gpu.m` 只有 5 处（`h3_gpu.m:200,360,398` 分别受 `H3_PROFILE` / `H3_DEBUG_GPU_MEMORY` / `H3_NAX_DIAGNOSTIC` 控制）。

**计量而非日志**：性能数据走 `h3_gpu_stats` 结构体与 `profile_mark()` 记账（`h3_gpu.h:17-33`、`h3_gpu.m:1050-1065`），再由上层渲染，而不是散落的 printf。

**失败即降级、不崩溃**：多处优化路径在探测失败时静默回退（例如 TensorOps 编译失败会清空宏重编译一次并置 `tensorOpsEnabled=NO`，`h3_gpu.m:391-407`）；`h3_lora_matches` 不匹配时跳过适配器并告警（`h3_lora.c:154-168`）。

---

## 构建系统（Makefile）

category: `build_system` · type: `repository`

手写 Makefile，无 CMake / Bazel / xcodebuild。

**编译器与标志**：`CC := xcrun clang`（`Makefile:1`）；`CFLAGS` = `-std=c11 -O3 -MMD -MP -Wall -Wextra -Wpedantic -Wshadow -Wconversion -Wno-sign-conversion -D_DARWIN_C_SOURCE`（`Makefile:3-4`）；ObjC 文件追加 `-fobjc-arc`（`Makefile:5`）。警告等级很高，`-Wconversion` 默认开启。

**链接**：`-framework Foundation -framework Metal -framework MetalPerformanceShaders -framework MetalPerformanceShadersGraph -framework Accelerate`，外加 `-licucore -lm`（`Makefile:6-9`）。

**源文件清单是手维护的，这是最常见的踩坑点**：`LIB_C`（`Makefile:11-15`）与 `LIB_M`（`Makefile:16`）必须列出所有库源文件。一个只被 `#include` 但没加进清单的 `.c` 会以 `Undefined symbols` 链接失败——`h3_memory_plan.c` 历史上就是这样被漏掉的。
```make
LIB_C := h3.c h3_host.c h3_safetensors.c h3_weights.c h3_text_encoder.c \
    h3_dit_schedule.c h3_dit.c h3_lora.c h3_memory_plan.c
LIB_C += h3_video_vae.c h3_video_encoder.c h3_audio_vae.c h3_ffmpeg.c \
    h3_terminal.c h3_vision_encoder.c h3_multimodal.c
LIB_M := h3_metal.m h3_gpu.m h3_tokenizer.m
```

**每个测试套件都是独立 target**（`Makefile:39-137`），模式规则 `%.o: %.c` 与 `%.o: %.m`（`Makefile:241-248`）。`tests/bench_dit_864.o` 是唯一带额外 `-D` 的目标（`Makefile:114-116`）。

**依赖跟踪**：`-MMD -MP` 生成 `.d`，末尾 `-include $(wildcard *.d tests/*.d)`（`Makefile:254`）实现头文件依赖自动重建。

**为 vendored 代码局部放宽警告**：`linenoise.o: CFLAGS += -Wno-conversion -Wno-variadic-macro-arguments-omitted`（`Makefile:250-252`）——让主项目保持严格，而不去改写这个终端编辑器。

**测试的条件执行**：`test` target 用 shell `if test -f ...` 判断权重/fixture 是否存在，不存在则打印 `skip: ...` 而非失败（`Makefile:147-227`）。这是「无权重也能 `make test` 通过」的原因。

---

## 公共 API 与生成流水线

category: `overview` · type: `module`

`h3.c`(2505 行) + `h3.h`(238 行) 是整个引擎的门面与编排层：对外只暴露不透明句柄 `h3_ctx` / `h3_result` 与一组 `h3_*` 函数（`h3.h:20-21`），内部把工作派发给各子系统。

**API 表面**（`h3.h:202-233`）：`h3_load_dir` / `h3_free` / `h3_last_error` / `h3_device` / `h3_model` / `h3_generate` / `h3_decode_latent` / `h3_result_free`，加上缓存控制 `h3_cache_set_enabled` / `h3_cache_set_models_enabled` / `h3_cache_clear` / `h3_cache_get_info` / `h3_cache_set_disk_dir`。

**两个核心配置结构**：`h3_params`（`h3.h:65-153`）集中了所有生成期旋钮——尺寸/帧数/步数/seed、参考输入、以及一大批加速开关（`denoise_reuse` / `dit_layers` / `core_reuse` / `token_reduction` / `use_int8_row_fc2` / `ssd_streaming` / `lora_path` / `linear_branch_path` / `video_vae_streaming` / `memory_plan_auto`）；`h3_device_info` / `h3_model_info` / `h3_component_info` 描述设备与权重构成（`h3.h:167-191`）。默认值宏是 `H3_PARAMS_DEFAULT`（`h3.h:155-165`）。

**上下文**：`struct h3_ctx`（`h3_internal.h:12-36`）持有 `model_dir`、错误缓冲、设备/模型信息、条件缓存（含磁盘缓存键与值）、以及 `dit` / `video_decoder` 两个阶段缓存指针。

**回调式输出**：生成结果通过 `h3_frame_callback` 增量交付帧，`h3_progress_callback` 汇报阶段进度（`h3.h:61-63`）；`h3_frame` 带 `denoise_step` 字段以区分「中间去噪预览」与最终帧（`h3.h:56-58`）。

---

## 公共 API 与生成流水线 - Architecture Design

category: `architecture_design` · type: `module` · parent: `公共 API 与生成流水线`

**分层方向是单向的**：`h3.c` → 各子系统（`h3_host` / `h3_dit` / `h3_video_vae` / `h3_text_encoder` / `h3_ffmpeg` …）→ `h3_gpu` 抽象 → Metal。子系统之间不互相直接调用，只在 `h3.c` 里被编排；跨模块共享的类型集中在 `h3_internal.h`（`h3_ctx`）与 `h3_host.h`（`h3_layout`、`h3_sigma_schedule`、`h3_position`），公共头 `h3.h` 只放对外 API。

**两条并行权重流**：文本/视觉编码器（Qwen3-VL，Ref2VA 也用）与 DiT+VAE 流。`h3_host.c` 负责张量分配并在步与步之间缓存条件张量。

**生成主路径的关键交接点**（都在 `h3.c` 内）：
- 图像读入 `h3_ffmpeg_read_image_f32`（`h3.c:1710`）→ 视频 VAE 编码 `h3_video_vae_encode`（`h3.c:1785`）→ Qwen 视觉编码 `h3_vision_encode_bf16`（`h3.c:1825`、`h3.c:1857`）→ 音频 `h3_audio_vae_encode`（`h3.c:1652`）。
- 多模态分派按条件类型二选一：`h3_multimodal_encode_fl2va_bf16` 或 `..._ref2va_bf16`（`h3.c:1880-1889`）。
- 条件张量交给 `h3_dit_load_conditioned`（`h3.c:2001`）。
- 自动显存规划在 `h3_generate` 内、加载之前完成（`h3.c:1263-1283`）。

**缓存是显式开关、默认关闭**：`h3_cache_set_enabled` 注释说明默认关闭是为了让一次性调用者保留原有的「按阶段释放内存」的生命周期（`h3.h:210-215`）；交互会话才打开，从而复用文本条件、已准备的 DiT 与视频解码器。

**磁盘条件缓存的设计**（`h3.h:217-226`）：缓存键覆盖提示词、每个引用媒体的 stat、以及编码器/tokenizer/VAE 目录的模型指纹，因此任何输入或权重变化都会自然落空重算；文件原子写入并在使用前校验，损坏条目被丢弃重编码。

---

## 公共 API 与生成流水线 - Coding Conventions

category: `coding_conventions` · type: `module` · parent: `公共 API 与生成流水线`

- **命名**：所有 C 符号前缀 `h3_`、snake_case；类型名 `h3_xxx`；宏 `H3_` 全大写。ObjC 类用 `H3` 前缀的驼峰（`H3GPU` / `H3Tensor`）。
- **不透明句柄 + 访问器**：公共头只前置声明 `typedef struct h3_ctx h3_ctx;`（`h3.h:20`），字段定义留在 `h3_internal.h`，避免把内部布局暴露给嵌入方。
- **配置结构用指定初始化器 + 默认宏**：新增旋钮应加入 `h3_params` 并在 `H3_PARAMS_DEFAULT` 给默认值，而不是加全局变量。
- **注释解释「为什么」而不是「做什么」**：`h3.h` 里几乎每个旋钮的注释都写明了它属于哪个档位（例如 `denoise_reuse`：1 = close-reference、2 = validated fast、3 = aggressive，`h3.h:77-79`；`dit_layers`：50 精确 / 45 验证过的快速档 / 40 更激进，`h3.h:80-82`）。新增旋钮请沿用这个习惯。
- **向后兼容的命名**：一批 `use_slower_*` 参数（`h3.h:121-141`）刻意命名为「用更慢的」，表示它们是回退到可移植实现的开关，用于 A/B 定位。
- **单位与布局写进注释**：如 latent 文件是 `[C=24,T,H,W]` z-space（`h3.h:147-148`）。

---

## Host 基础层（几何、调度、布局、RNG）

category: `overview` · type: `module`

`h3_host.c`(647 行) + `h3_host.h`(142 行) 是不碰 GPU 的纯主机侧基础库，被几乎所有模块依赖。它承载四类职责：
1. **几何与时间形状**：`h3_align_frame_count`、`h3_video_latent_t`、`h3_temporal`、`h3_latent_canvas`、`h3_adapt_canvas`、`h3_reference_image_canvas`、`h3_reference_video_canvas`（`h3_host.h:95-110`）。
2. **sigma 调度**：`h3_schedule_build` / `h3_serving_schedule_build` 产出 `h3_sigma_schedule`（`h3_host.h:82-86`、`h3_host.h:112-116`）。
3. **序列布局**：`h3_layout_build` 把 `h3_layout_spec` 展开成带分段与位置 ID 的 `h3_layout`（`h3_host.h:118-120`）。
4. **随机数与图像缩放**：`h3_rng_seed` / `h3_rng_u32` / `h3_rng_normal` / `h3_rng_fill_normal`（`h3_host.h:123-126`），以及基于 Accelerate/vImage 的高质量 RGB24 缩放 `h3_resize_rgb24_high_quality`（`h3_host.h:128-134`）。

**关键常量**（`h3_host.h:7-14`）：画布必须 32 的倍数（`H3_CANVAS_MULTIPLE 32`）、最大像素 `768*1344`、输出 24 fps（`H3_FPS 24`）、音频 latent 40 fps（`H3_AUDIO_LATENT_FPS 40`）、VAE 空间压缩比 16（`H3_VAE_SPATIAL_RATIO 16`）、视频/音频 sigma shift 分别为 12.0 / 3.0、最大步数 1000。

**布局模型**：`h3_segment_kind` 有 6 种段（文本、条件、参考图、参考音频、音频、视频，`h3_host.h:28-35`）；`h3_layout` 除分段与位置外还带 `img_cond_rows` / `img_target_rows` / `audio_cond_rows` / `audio_target_rows` 与一个 5 元 `signature[5]` 用于识别布局是否变化（`h3_host.h:70-80`）。

---

## Host 基础层（几何、调度、布局、RNG） - Architecture Design

category: `architecture_design` · type: `module` · parent: `Host 基础层（几何、调度、布局、RNG）`

**为什么单独成层**：这些计算（画布对齐、sigma 网格、mRoPE 位置、RNG）都是纯数值的、可确定性测试的，把它们与 GPU 隔离让 `h3_tests` 能在没有权重和 Metal 的情况下验证核心几何与调度逻辑。

**画布自适应规则是双向的**：`h3_adapt_canvas` 处理目标输出画布（对齐到 32 的倍数、像素上限 `768*1344`）；`h3_reference_image_canvas` 则**只缩不放**且保持宽高比，`max_short_edge=0` 表示匹配输出像素面积，正值表示限制短边（`h3_host.h:101-106`）；`h3_reference_video_canvas` 同样是「小于目标时不放大」（`h3_host.h:107-110`）。

**两套 sigma 网格**：视频与音频有各自独立的 shifted 网格（shift 12.0 / 3.0），由 `h3_time_shift_sigma` / `h3_time_shift_slope` 做时间步重映射（`h3_host.h:112-113`）。

**求解器步进与调度解耦**：`h3_euler_velocity_step` 与 `h3_res_step` 只做「给定 sigma 与速度、更新样本」这一件事（`h3_host.h:136-140`），调度由 `h3_sigma_schedule` 提供——这样换求解器不需要改调度，换调度不需要改求解器。

**RNG 状态是可序列化的结构体**而非全局：`h3_rng` 含 `state` / `increment` / `spare` / `has_spare`（`h3_host.h:88-93`），支持缓存 `spare` 以节省一次采样。

---

## Host 基础层（几何、调度、布局、RNG） - Tech Stack

category: `tech_stack` · type: `module` · parent: `Host 基础层（几何、调度、布局、RNG）`

- **语言**：C11，无外部依赖；只用 `<stddef.h>` / `<stdint.h>` 与 Accelerate。
- **图像缩放**用 Apple Accelerate 的 vImage 高质量重采样，而不是手写双线性（`h3_host.h:128-130`）。注释明确说明「几何完全相同也返回独立副本」，即调用方始终拥有 `*output` 的所有权。
- **时间/帧数对齐**：`h3_temporal_shape` 同时给出 `frame_count` / `video_t` / `audio_t` 三个量（`h3_host.h:16-20`），把「帧数 → 视频 latent T / 音频 latent T」的换算集中在一处。
- **位置用双精度**：`h3_position` 的 `t` / `h` / `w` 都是 `double`（`h3_host.h:22-26`），避免长序列上累积误差。
- **sigma 表定长**：`h3_sigma_schedule` 内嵌 `float video[H3_MAX_STEPS+1]` / `audio[...]`（`h3_host.h:82-86`），避免动态分配。

---

## GPU / Metal 后端

category: `overview` · type: `module`

`h3_gpu.h`(900 行) / `h3_gpu.m`(5821 行) / `h3_metal.m`(47 行) / `h3_shaders.metal` 构成设备层。`h3_gpu.h` 是纯 C 抽象，`h3_gpu.m` 是唯一的实现。

**职责边界**：`h3_metal.m` 只做设备探测（`h3_metal_probe` 填充 `h3_device_info`，`h3_metal.m:15-47`），**不含任何 kernel 源码**；所有 H3 内核都集中在 `h3_shaders.metal`（121 个 `kernel void`，`h3_shaders.metal:1-9`），由 `h3_gpu.m` 在运行时读取并编译。

**对外 ABI 极小**：只有两个不透明句柄 `h3_gpu` / `h3_gpu_tensor`，加 dtype 枚举与 `h3_gpu_stats`（`h3_gpu.h:7-33`）。

**双实现策略**：稳定且宽的算子同时有「手写 Metal 内核」与「MPSGraph」两条路径，运行时按形状与设备能力选路；Conv1d / ConvTranspose1d / Conv3d 无条件走 MPSGraph。

**存储**：所有张量用 `MTLResourceStorageModeShared`（`h3_gpu.m:616-617`），在统一内存架构上主机可直接读写；权重可选择零拷贝 wrapping。

---

## GPU / Metal 后端 - Architecture Design

category: `architecture_design` · type: `module` · parent: `GPU / Metal 后端`

**内核库在运行时从源码编译，没有 metallib 也没有磁盘缓存**：默认路径 `h3_shaders.metal`，用 `NSString contentsOfFile` 读源码后 `newLibraryWithSource`（`h3_gpu.m:364-384`）。随后按一张名字列表逐个建 pipeline，任一失败即返回 `NULL`（`h3_gpu.m:533-556`）。这就是「所有二进制必须从仓库根目录运行」的根因。

**编译选项与条件宏**：`mathMode = MTLMathModeSafe`；当设备是 M5，或设了 `H3_VDN_INT8` 且 `H3_NAX≠0` 时注入宏 `H3_METAL_HAS_TENSOR=1`（`h3_gpu.m:373-384`）。

**优雅降级**：TensorOps 编译失败会清空宏重编译一次，置 `tensorOpsEnabled=NO`，退回普通 MPSGraph / 直接 Metal（`h3_gpu.m:391-407`）。只有源码读不到或最终 library 为 nil 才写错误并返回 `NULL`（`h3_gpu.m:409-416`）。

**选路判据（关键函数）**：
- BF16 Linear 三路（NAX 直核 / MPSGraph / 朴素核）：NAX 需要 `tensorOpsEnabled && naxShape && bias==NULL && rows>=128`（`h3_gpu.m:2516-2547`）；否则 MPS 路径要求 `rows>=32 && in/out>=256`（`h3_gpu.m:2611-2614`）。
- F32 Linear：先匹配特例 tiled 核（`output_dim==5376 && input_dim∈{32,96}`），否则用同一套 MPS 阈值，最后退回 `h3_linear_f32`（`h3_gpu.m:1133-1170`）。
- **SDPA 只有 MPSGraph 一条生产实现**（`h3_gpu.m:1608-1660`）；手写的 `h3_gpu_flash_attn_bf16` 只被测试调用，生产仍走 `sdpa_bf16`（`h3_gpu.m:5104`、`h3_dit.c:3775-3780`）。
- GQA 默认直核，仅 `H3_MPS_GQA` 时改走 MPS（`h3_gpu.m:4443-4445`）。
- Conv1d / ConvTranspose1d / Conv3d 无条件走 MPSGraph（`h3_gpu.m:2076-2087`、`h3_gpu.m:2116-2129`、`h3_gpu.m:1969`）；BF16 MLP 走 MPS（`h3_gpu.m:2682`），NAX / int8 MLP 走直核（`h3_gpu.m:2706-2711`）。
- grouped QKV 投影：条件满足时用 NAX 融合 QKV+RoPE，否则退回 linear + rope 两步（`h3_gpu.m:3956-3971`）。

---

## GPU / Metal 后端 - Tech Stack

category: `tech_stack` · type: `module` · parent: `GPU / Metal 后端`

- **Metal** 为核心计算后端；**MetalPerformanceShaders** 提供部分原语；**MetalPerformanceShadersGraph** 承载所有宽算子（Linear / SDPA / Conv / MLP）的图路径。
- **MPSGraph 图缓存**：`H3GPU` 持有五类 MPSGraph 缓存（`h3_gpu.m:93-135`），README 也说明宽 BF16 矩阵乘与 SDPA 使用缓存图（`README.md:465-466`）。
- **统一内存**：设备能力由 `h3_metal.m:34` 的 `hasUnifiedMemory` 上报；张量一律 `StorageModeShared`。
- **零拷贝权重**：transformer 权重或设了 `H3_ZERO_COPY_WEIGHTS=1` 时，用 `mmap(MAP_PRIVATE)` + `newBufferWithBytesNoCopy`，deallocator 内 `munmap`（`h3_gpu.m:672-710`）。非 no-copy 路径用 `pread` 直接写入 shared buffer（`h3_gpu.m:744-760`）；流式版本额外加 `F_NOCACHE` 绕过页缓存（`h3_gpu.m:812-813`、`h3_gpu.m:847-850`）。
- **ObjC 桥接**：C↔ObjC 用 `GPU()` / `TENSOR()` 两个 `__bridge` 转换宏（`h3_gpu.m:137-145`），不引入额外句柄包装结构。
- **无 buffer 池**：每次 `newBufferWithLength`、`free` 释放，仅做 stats 记账（`h3_gpu.m:606-631`）。
- **主机直读**：`h3_gpu.h:94-99` 暴露 BF16 后备存储指针，允许主机侧直接读写张量。
- **没有第三方 GPU 抽象层**（无 MLX、无 Metal-cpp），直接使用 MTL* / MPS* API。

---

## GPU / Metal 后端 - Coding Conventions

category: `coding_conventions` · type: `module` · parent: `GPU / Metal 后端`

- **C API 全 `h3_gpu_` 前缀 + snake_case，返回 `int`(0/1) 或指针 `NULL`**；ObjC 类名 `H3GPU` / `H3Tensor`（`h3_gpu.m:20-135`）。
- **pipeline 字符串与 kernel 名一一对应**：`h3_shaders.metal` 里的 `kernel void` 名字必须与 C 侧 pipeline 列表中的字符串完全一致，否则建 pipeline 失败（`h3_gpu.m:533-556`）。改内核名要同步改两处。
- **不要新增全局日志宏**：沿用 `fprintf(stderr, ...)` + `H3_*` 开关的模式。
- **错误统一走 `h3_gpu_set_error` / `h3_gpu_error`**，不要自己拼错误字符串（`h3_gpu.m:224-231`）。
- **性能记账走 `h3_gpu_stats` 与 `profile_mark()`**，而不是打印计时（`h3_gpu.h:17-33`）。
- **回退路径要保留且可控**：每个激进优化都应能用 `H3_DISABLE_*` 关掉；新增优化时同时加反向开关，并在选路函数里显式判据。
- **同步约定**：只有一条串行 queue（`h3_gpu.m:346`）；`submit` 会等所有 inflight command 完成并检查 status（`h3_gpu.m:1003-1035`），而 `continue` 只 commit 不等待、依赖队列有序性（`h3_gpu.m:972-990`）。后台/阻塞路径自建私有 command buffer 并立即 wait（`h3_gpu.m:5313-5335`）。`MTLSharedEvent streamEvent` 已分配但**当前未使用**（`h3_gpu.m:128-134`、`h3_gpu.m:347`）；也没有 semaphore / lock / fence，仅 `lastError` 声明为 atomic（`h3_gpu.m:114-118`）。

---

## 权重加载与 safetensors

category: `overview` · type: `module`

`h3_weights.c`(489 行) + `h3_safetensors.c`(584 行) 负责第 1 阶段：把 safetensors 分片变成可用的张量。

**分工**：`h3_safetensors` 是格式层——解析 safetensors 头部与张量索引；`h3_weights` 是策略层——惰性/流式加载 BF16，或做 int8 分组/按行量化。

**加载模式**：`h3_load_dir` 先校验模型目录布局并解析 `h3_model_info`（模式、维度、block 数量），然后惰性打开分片；权重默认**不映射**，`--info` 可以只查布局与设备。

**组件计量**：`h3_component_info` 分别统计 `text_encoder` / `fl2va_transformer` / `ref2va_transformer` / `video_vae` / `audio_vae` 的 `bytes` / `tensor_bytes` / `files` / `tensors`（`h3.h:178-191`），这组数字正是内存规划器的输入。

**量化**：支持 int8 分组（group）与按行（row）量化；`use_int8_row_fc2` 用「FC2 每行一个 activation scale + M5 full-K kernel」，比分组 int8 更快但数值上更激进（`h3.h:90-92`）。

---

## 权重加载与 safetensors - Architecture Design

category: `architecture_design` · type: `module` · parent: `权重加载与 safetensors`

**惰性 + 流式是关键设计**：权重不在 `h3_load_dir` 时全部驻留，而是按需加载。这是让 16/24 GB Mac 也能跑起来的前提，也是 `--ssd-streaming` 能工作的基础。

**SSD 流式的实现位置在 DiT 侧**：DiT 用两槽环形缓冲，`load_core` 负责槽位轮转（`h3_dit.c:2620-2703`），前向时预取下一块（`h3_dit.c:4128-4228`）。默认只保 2 个 BF16 block 驻留（`h3.h:96-98`）。

**两种读盘路径**：零拷贝用 `mmap(MAP_PRIVATE)` + `newBufferWithBytesNoCopy`（`h3_gpu.m:672-710`）；否则 `pread` 直接写进 shared buffer（`h3_gpu.m:744-760`），流式版本加 `F_NOCACHE` 避免污染页缓存（`h3_gpu.m:812-813`）。

**量化与流式是正交的**：`h3_memory_plan.h:20-24` 明确说明两者不再互斥——streaming 决定「权重放在哪」，int8 决定「权重怎么压缩」，对应 ds4 里解耦的 routing / expert-cache 设计。

**与加载期权重改写的冲突**：LoRA 合并与 VDN linear-branch 都要求在统一内存里改写权重，因此**都与 `--ssd-streaming` 不兼容**，LoRA 还额外与 int8 不兼容（见 `h3_lora.h:23-26`）。

---

## 权重加载与 safetensors - Tech Stack

category: `tech_stack` · type: `module` · parent: `权重加载与 safetensors`

- **safetensors** 是唯一的权重格式（BF16 为主），带分片（如 `text_encoder/model-00014-of-00014.safetensors`）。
- **文件 IO** 直接用 POSIX：`pread` 读入、`mmap` 零拷贝、`F_NOCACHE` 绕过页缓存（`h3_gpu.m:672-710`、`h3_gpu.m:744-760`、`h3_gpu.m:812-813`）。
- **量化**：int8 分组与按行两条路径；`H3_INT8_*` 系列环境变量可细调分组大小、缓存、activation clip、是否保留 BF16 的 QKV/MLP/attention-out（见「环境变量与运行时开关」）。
- **磁盘测速工具**：仓库里有一个独立的 `disk_speed.c`（103 行），用于评估流式路径的实际带宽。
- **无 Python 依赖**：解析 safetensors 是纯 C 实现，不经过 Python/`safetensors` 包。仓库里的 `dbg_parse_safetensors.py` 等脚本只是调试对照，不参与运行。

---

## 文本与视觉编码器（Qwen3-VL）

category: `overview` · type: `module`

`h3_text_encoder.c`(1348 行) 实现 Qwen3-VL 文本塔；`h3_vision_encoder.c`(633 行) 与 `h3_video_encoder.c`(814 行) 实现视觉塔；`h3_tokenizer.m`(521 行) 是 BPE 分词器。三者共同完成第 3 阶段的条件构建。

**层数**：`TEXT_LAYERS = 50`（`h3_text_encoder.c:13`）。代码注释只说明这是「the released first 50」（`h3_text_encoder.h:31`），**为什么只需要前 50 层在代码中没有进一步依据**——这是一个已知的开放问题，不要臆测原因。

**入口**：`text_encode_bf16_impl`（`h3_text_encoder.c:474`），逐层 `layer_weights_load`（`h3_text_encoder.c:145`）+ `encode_layer`（`h3_text_encoder.c:409`）。权重前缀 `model.language_model.layers.%d.`，每层 11 个张量（`h3_text_encoder.c:149`、`h3_text_encoder.c:166-177`）；`embed_tokens.weight` 在 `h3_text_encoder.c:623`。

**输出**：BF16 embedding 写入 `h3_text_embedding.values`（`h3_text_encoder.h:11-19`、`h3_text_encoder.c:724-736`），由 `h3.c:1333` 接收并交给 `h3_dit_load_conditioned`（`h3.c:2001`）。注意：AGENTS.md 提到的 `h3_host_cond` **在代码中找不到对应定义**，实际承载条件的是 `h3_text_embedding` 与 `h3_ctx` 上的 `conditioning_*` 字段。

**另一条 ClipProj 路径**：`h3_text_encode_clipproj_bf16`（`h3_text_encoder.c:985`）只跑 Qwen3-VL-4B 的前 25 层（`CP_TAP_LAYERS`，`h3_text_encoder.c:789`）再经 MLP 2560→5120（`h3_text_encoder.c:793`），用于保真度对照。

---

## 文本与视觉编码器（Qwen3-VL） - Architecture Design

category: `architecture_design` · type: `module` · parent: `文本与视觉编码器（Qwen3-VL）`

**多模态融合方式**：视觉 embedding **覆盖**对应 span 的 base embedding（`h3_text_encoder.c:638-646`），而 deepstack 特征在第 0/1/2 层之后相加（`h3_text_encoder.c:675-679`）。也就是说视觉信息既在输入端注入、又在浅层以残差方式补充。

**权重逐层加载、逐层释放**：`layer_weights_load` 与 `encode_layer` 成对出现，编码器整塔跑完即释放——这正是内存规划器把「编码器按 0 字节常驻」计入 `streamed_resident_bytes` 的依据（`h3.c:1250-1261`）。

**视觉塔有独立的预取**：`H3_QWEN_PREFETCH` / `H3_QWEN_PREFETCH_DEPTH` 控制 Qwen 权重预取，说明视觉/文本塔的加载是流水线化的。

**呈现层负责位置编码**：`encode_presentation` 计算 mRoPE positions 与 tags（`h3_multimodal.c:151-195`），然后才调用 `h3_text_encode_multimodal_bf16`（`h3_multimodal.c:189`）。位置编码由多模态呈现决定，不是文本塔内部的事。

**为什么分离视觉/视频两个编码器**：`h3_vision_encoder` 面向图像参考，`h3_video_encoder` 面向视频参考（含 24 fps 有界解码、VAE 的因果 `ceil(T/4)` 压缩、Qwen 两帧采样，见 `README.md:824-826`）。

---

## 文本与视觉编码器（Qwen3-VL） - Tech Stack

category: `tech_stack` · type: `module` · parent: `文本与视觉编码器（Qwen3-VL）`

- **Qwen3-VL** 提供文本塔（前 50 层）与视觉塔；视觉塔权重在 `text_encoder/` 分片里，Ref2VA 也复用它。
- **BF16 端到端**：条件 embedding 以 BF16 存储与传递（`h3_text_embedding` 是 `uint16_t*` 形态），避免中间转换开销。
- **特殊 token ID 硬编码**：`VISION_START=151652`、`VISION_END=151653`、`IMAGE_PAD=151655`（`h3_multimodal.c:10-13`），`H3_PAD_TOKEN_ID=151643`（`h3_tokenizer.h:7`）。
- **Accelerate / vImage** 用于条件图像的高质量缩放（`h3_host.h:128-134`）。
- **媒体解码**不依赖 Python：图像/视频读取走 `h3_ffmpeg_read_image_f32`（`h3.c:1710`）与 `ffprobe`/`ffmpeg`（`H3_FFMPEG` / `H3_FFPROBE` 可指定可执行文件）。

---

## Tokenizer（BPE）

category: `overview` · type: `module`

`h3_tokenizer.m`(521 行) 是纯 Objective-C/Foundation 实现的字节级 BPE 分词器，读取 Hugging Face 的 `tokenizer.json`。它是本项目仅有的 3 个 ObjC 文件之一。

**格式校验**：要求 `model.type == "BPE"` 且 `normalizer == "NFC"`，否则拒绝（`h3_tokenizer.m:316-320`）。

**读取字段**：`model.vocab`（`h3_tokenizer.m:325`）、`model.merges`（`h3_tokenizer.m:350`）、`added_tokens`（`h3_tokenizer.m:330`）、`normalizer`（`h3_tokenizer.m:315`）。

**预分词是 GPT-2 风格**：处理 contraction（`h3_tokenizer.m:101-117`）以及 letter / number / space 分类切分（`h3_tokenizer.m:75-91`）。

**字节级 BPE**：324 项 `byteEncoder`（`h3_tokenizer.m:392-405`）、`mergeRanks` 合并优先级表（`h3_tokenizer.m:348-367`）、`bpeCache` 缓存已算过的词（`h3_tokenizer.m:209`、`h3_tokenizer.m:253`）。

**特殊 token**：`added_tokens` 按长度降序做最长匹配（`h3_tokenizer.m:383-389`、`h3_tokenizer.m:273-287`），命中直接输出其 id（`h3_tokenizer.m:431-444`）；显式拒绝 `single_word` / `lstrip` / `rstrip` / `normalized` 这类带副作用的标志（`h3_tokenizer.m:371-374`）。空串在 `pad_empty` 时补 `H3_PAD_TOKEN_ID=151643`（`h3_tokenizer.h:7`、`h3_tokenizer.m:448`）。

---

## Tokenizer（BPE） - Architecture Design

category: `architecture_design` · type: `module` · parent: `Tokenizer（BPE）`

**为什么用 ObjC 而不是 C**：`tokenizer.json` 是 JSON，Foundation 的 `NSJSONSerialization` 省掉了自己写 JSON 解析器；代价是这个文件必须用 `-fobjc-arc` 编译（`Makefile:5`、`Makefile:16`）。

**接口面很窄**：`h3_tokenizer.h` 只有 26 行，暴露加载与编码两个动作，不泄漏内部结构。

**最长匹配 + 拒绝副作用标志**的设计取向是「宁可少支持也不要静默错误」：与其猜测 `lstrip` 的语义，不如直接拒绝并让调用方知道。

**缓存层是必需的**：BPE 合并是逐对迭代的，`bpeCache` 按词缓存结果（`h3_tokenizer.m:209`），否则长提示词会明显变慢。

**与多模态的耦合点**：tokenizer 只负责产出 id 序列；`<Picture n>` 等占位符文本与 `VISION_START/IMAGE_PAD/VISION_END` 特殊 id 的拼装由 `h3_multimodal.c` 负责（`h3_multimodal.c:221-228`、`h3_multimodal.c:138-147`）。

---

## 多模态条件构建

category: `overview` · type: `module`

`h3_multimodal.c`(360 行) 负责把「文本 + 图像 + 视频 + 音频参考」拼装成模型能吃的单条序列，是第 3 阶段的收口处。

**真实入口（注意与 AGENTS.md 的差异）**：代码里**不存在** `h3_multimodal_build` / `h3_build_conditions`（AGENTS.md 里提到的这两个名字在代码中找不到）；实际入口是 `h3_multimodal_encode_fl2va_bf16`（`h3_multimodal.c:197`）与 `h3_multimodal_encode_ref2va_bf16`（`h3_multimodal.c:243`），由 `h3.c:1880-1889` 按条件类型分派。

**FL2VA 拼装顺序**：逐图 `snprintf("<Picture %zu>: ")`（`h3_multimodal.c:221`）→ `tokenize_append`（`h3_multimodal.c:224`）→ `append_vision` 写入 `VISION_START` / `IMAGE_PAD` / `VISION_END`（`h3_multimodal.c:138-147`）→ **最后才 tokenize 提示词本身**（`h3_multimodal.c:228`）。

**Ref2VA 额外语法**：插入 `<Audio n>:`（`h3_multimodal.c:295`）、`<Video n>:`（`h3_multimodal.c:319`）与 `<x.x seconds>` 时间戳（`h3_multimodal.c:325`）；视频按每 2 帧块一个 span 展开（`h3_multimodal.c:315-335`）。

**收尾**：`encode_presentation` 计算 mRoPE 位置与 tags（`h3_multimodal.c:151-195`），再调 `h3_text_encode_multimodal_bf16`（`h3_multimodal.c:189`）。

---

## 多模态条件构建 - Architecture Design

category: `architecture_design` · type: `module` · parent: `多模态条件构建`

**呈现（presentation）与编码（encoding）分离**：`h3_multimodal.c` 只决定「序列长什么样」——占位符、特殊 token、时间戳、mRoPE 位置；真正的张量计算全部委托给编码器。这让 FL2VA / Ref2VA 的差异局限在这一层。

**上游职责边界清晰**（都在 `h3.c` 内按序调用）：读图 `h3_ffmpeg_read_image_f32`（`h3.c:1710`）→ 视频 VAE 编码 `h3_video_vae_encode`（`h3.c:1785`）→ Qwen 视觉编码 `h3_vision_encode_bf16`（`h3.c:1825`、`h3.c:1857`）→ 音频 `h3_audio_vae_encode`（`h3.c:1652`）。多模态层拿到的是已经编码好的特征。

**参考音频的合成规则**（README 明示，`README.md:828-834`）：解码为 32 kHz 立体声 F32 → 走原生 AudioVAE 的 posterior-mean 路径 → 按 0.999 干净 latent + 0.001 带种子噪声混合 → 钉在音频条件时间步 1.0 → 以宽度 32 的行打包到与视觉参考相同的 rotary 时间轴上。约束：音频输入 2–15 秒、最多 3 段、总时长上限 15 秒，且**独立音频参考必须与图像或视频参考组合使用**。

**FL2VA 与 Ref2VA 是两套 transformer**：`h3_model_info` 分别统计 `fl2va_transformer` 与 `ref2va_transformer`（`h3.h:187-188`），`--ref-image` 会切换到 Ref2VA 那套权重与「只缩不放、保持宽高比」的参考画布（`README.md:821-823`）。

---

## DiT 去噪主干

category: `overview` · type: `module`

`h3_dit.c`(5136 行) 是全仓最大的模块，实现第 4 阶段的 50 块 DiT 主干与前向。

**结构**：`H3_DIT_BLOCKS 50`（`h3_dit_schedule.h:11`），实例数组 `blocks[H3_DIT_BLOCKS]`（`h3_dit.c:217`）。每个 block 含 `norm1` / `norm2`、`attn.qkv_proj`、`attn.q_norm`、`attn.k_norm`、`attn.out_proj`、`mlp.fc1`、`mlp.fc2`（`h3_dit.c:870-877`）。**AdaLN 调制不在 block 内部**，而是每 block 独立的 `blocks.N.adaln_proj.linear` 投影、6 个 slot（`h3_dit_schedule.c:512-518`、`h3_dit_schedule.h:15`）。另有 2 个 token_refiner block（`h3_dit.c:2084`、`h3_dit.c:2124-2127`）与头部 `final_layer.norm / video_out / audio_out`（`h3_dit.c:2712-2721`）。

**调用链**：加载 `h3_dit_load_t2va`（`h3_dit.c:3219`）/ `h3_dit_load_conditioned`（`h3_dit.c:3264`）→ `load_dit`（`h3_dit.c:2983`）；单步前向 `h3_dit_forward`（`h3_dit.c:4375`）→ `encode_forward`（`h3_dit.c:3942`）→ 每块 `run_block`（`h3_dit.c:3711`）。

**采样**：`h3_dit_denoise_euler`（`h3_dit.c:4919`）→ `_preview`（`h3_dit.c:4794`）；在 M5 / GPU sampler 条件下走 `denoise_euler_gpu`（`h3_dit.c:4546`、`h3_dit.c:4807-4811`）。另有一个 RES 求解器 `h3_dit_denoise`（`h3_dit.c:4720`），**不是当前 serving 路径**。

---

## DiT 去噪主干 - Architecture Design

category: `architecture_design` · type: `module` · parent: `DiT 去噪主干`

**AdaLN 与 gate 预计算是显存优化手段**：`h3_dit_schedule_precompute` 为每个 step 物化全部 AdaLN（`h3_dit_schedule.c:443`），逐 block 计算 `time @ W` 写入 `schedule->blocks[block]`（`h3_dit_schedule.c:516-552`），最终层同理（`h3_dit_schedule.c:577-605`）。动机写得很明确：一次只提交一个投影，让 498 MiB 的 block 投影在加载下一个之前就被释放（`h3_dit_schedule.h:24-30`）。消费点是 `run_block` 取 `h3_dit_schedule_block` 作为 modulation（`h3_dit.c:3719`），送入 AdaLN / gate kernel（`h3_dit.c:3733`、`h3_dit.c:3818`、`h3_dit.c:3894`），最终层在 `h3_dit.c:4251` 取用。调用时机在 `load_dit` 中、`load_core` 之前（`h3_dit.c:3187-3202`）。

**gate score 用于层裁剪**：`h3_dit_schedule_gate_scores` 对每 block 的 slot 2、5 逐元素取绝对值求均值（`h3_dit_schedule.c:669-709`）；`configure_gate_ranked_blocks` 按 score 升序淘汰（`h3_dit.c:2500-2518`），随后 `h3_dit_schedule_prune` 释放对应张量（`h3_dit.c:3193-3194`）。README 补充：裁剪时保护结构上重要的首尾块（`README.md:495-498`）。

**时间步与 Euler 转移**：步数来自 `sigmas->steps`（`h3_dit_schedule.c:198`）；video 与 audio 各有独立 shifted 网格，时间行都是 `1 - sigma`（`h3_dit_schedule.c:216-218`、`h3_dit_schedule.c:259-260`）。转移用 `h3_gpu_euler_bf16`，步长 `sigma[step]-sigma[step+1]`（`h3_dit.c:4652-4661`）；CPU 路径是 `h3_euler_velocity_step`（`h3_dit.c:4890-4895`）。

**复用调度（denoise_reuse）**：`h3_dit_reuse_schedule` 选中 step 0、末步、以及 `step % interval == 0` 的步（`h3_dit.c:4473-4488`），可用 `H3_REUSE_STEPS` 覆盖（`h3_dit.c:4490-4510`）；被跳过的步用速度外推 `extrapolate_velocity` 补上（`h3_dit.c:4458-4471`、`h3_dit.c:4638-4647`）。

**加速开关各自的生效位置**：`core_reuse` 由 `evaluate_core` 判据（`h3_dit.c:4053-4056`）并缓存/复用残差（`h3_dit.c:4238-4248`）；`token_reduction` 由 `configure_token_reduction` 配置，默认作用在 block 4:30（`h3_dit.c:466`、`h3_dit.c:471`），前向中由 `use_token_reduction` 判据进入/退出（`h3_dit.c:4057`、`h3_dit.c:4091-4111`）；层裁剪有均匀策略 `configure_active_blocks`（`h3_dit.c:2453-2462`）与 gate 排名策略（`h3_dit.c:2487-2526`）；int8 在 `sequence>=128` 且设备支持时置位（`h3_dit.c:3152-3159`），在 `run_block` 生效（`h3_dit.c:3737`、`h3_dit.c:3784`、`h3_dit.c:3835`）。

**SSD 流式**：`dit->ssd_streaming` 走两槽环形缓冲（`load_core`，`h3_dit.c:2620-2703`），前向中预取（`h3_dit.c:4128-4228`）。**首块缓存**（fb_cache）另有独立探针 `fb_cache_probe`（`h3_dit.c:3905-3940`），判据在 `h3_dit.c:4067-4071` 与 `h3_dit.c:4229-4233`。

---

## DiT 去噪主干 - Coding Conventions

category: `coding_conventions` · type: `module` · parent: `DiT 去噪主干`

- **块数与层数永远用宏/导出常量**：`H3_DIT_BLOCKS`（`h3_dit_schedule.h:11`）、`H3_DEFAULT_DIT_LAYERS` / `H3_MIN_DIT_LAYERS`（`h3.h:17-18`）、`H3_VIDEO_VAE_LAYERS`（`h3_video_vae.h:24`）。硬编码 50 / 36 这类数字是明确的禁忌。
- **优化必须有对应的 `H3_DISABLE_*` 反向开关**：例如 `H3_DISABLE_TOKEN_REDUCTION`、`H3_DISABLE_FUSED_MLP`、`H3_DISABLE_INT8_*`，并且在选路处用 `getenv(...) == NULL` 的形式内联判断（`h3_dit.c:3027`、`h3_dit.c:3032`）。
- **融合（fused）路径成对出现**：每个 `fused_*` 布尔都在 `load_dit` 里统一初始化（`h3_dit.c:2772-2776`、`h3_dit.c:2846-2859`、`h3_dit.c:3027-3032`），而不是散落在使用点，便于审计当前启用了哪些融合。
- **环境变量解析集中在配置函数**：`configure_token_reduction`（`h3_dit.c:466`）、`configure_active_blocks` / `configure_gate_ranked_blocks`（`h3_dit.c:2453-2526`）、流式 worker 数解析（`h3_dit.c:1805-1815`）——不要在热路径里读环境变量。
- **实测结论写进注释/README 而不是只留在 commit 里**：例如 `H3_TOKEN_REDUCTION_EARLY` / `_SCALE` 的默认值选择、以及 README 里「哪些候选方案输给了 released linear base grid」的对比（`README.md:487-493`）。

---

## DiT 调度与 AdaLN 预计算

category: `overview` · type: `module`

`h3_dit_schedule.c`(784 行) + `h3_dit_schedule.h`(69 行) 是 DiT 的「时间维」侧翼：把时间步嵌入投影成每 block 的 AdaLN 调制量与 gate 分数，并负责层裁剪时的张量释放。

**对外接口**：`h3_dit_schedule_precompute`（预计算）、`h3_dit_schedule_gate_scores` / `h3_dit_schedule_gate_score`（gate 分数）、`h3_dit_schedule_block`（取某 block 的调制量）、`h3_dit_schedule_prune`（裁剪后释放）。

**常量**：`H3_DIT_BLOCKS 50`（`h3_dit_schedule.h:11`）、每 block 6 个 AdaLN slot（`h3_dit_schedule.h:15`）。

**预计算的动机**（`h3_dit_schedule.h:24-30`）：AdaLN 投影张量很大（约 498 MiB），如果全部常驻会吃掉显存；逐块计算可以做到「用完即释放」。

---

## DiT 调度与 AdaLN 预计算 - Architecture Design

category: `architecture_design` · type: `module` · parent: `DiT 调度与 AdaLN 预计算`

**它是 DiT 与显存规划之间的解耦层**：DiT 前向不再自己算时间嵌入，而是从预计算好的 `schedule->blocks[block]` 取；这让「按 step 重算」与「复用上一步结果」成为纯调度层的事。

**gate 分数同时服务两个消费者**：一是前向时的 gate 调制，二是层裁剪的排序依据（`h3_dit_schedule.c:669-709` 计算，`h3_dit.c:2500-2518` 消费）。

**LoRA 合并挂钩在这里**：`adaln_proj.linear` 与 `norm_out.linear` 的 LoRA 在调度预计算前合并（`h3_dit_schedule.c:544-546`、`h3_dit_schedule.c:586-588`），也就是说调度层必须先于前向拿到「已合并」的权重。

**裁剪必须配套释放**：`h3_dit_schedule_prune`（`h3_dit.c:3193-3194` 调用）释放被裁掉 block 的调度张量——只改 `dit_layers` 而不调 prune 会留下悬空显存占用。

**sigma 网格也在这里组装**：video / audio 两条独立 shifted 网格、时间行 `1 - sigma`（`h3_dit_schedule.c:198-260`），与 `h3_host.h` 的 `h3_sigma_schedule` 结构配合。

---

## LoRA 合并

category: `overview` · type: `module`

`h3_lora.c`(308 行) + `h3_lora.h`(79 行) 在加载期把 Turbo / 蒸馏适配器合并进 DiT 权重，用于 4 步 / 8 步调度器。

**公式**：`W' = W + (alpha/rank) * (B @ A)`，即 scale 先乘到 B 上（`h3_lora.h:20-22`、`h3_lora.c:130`）。

**适配器格式**：Diffusers 键名的 BF16 `lora_A` / `lora_B` safetensors；支持逗号分隔最多 4 个适配器按顺序合并（VDN 用 `default,turbo`）。

**应用时机**：在统一内存里、**紧接某个 block 加载之后**立即合并，因此推理阶段看到的仍是普通 BF16 权重；基础 checkpoint 文件永不被改写（`h3_lora.h:23-26`）。

**约束**：与 `--ssd-streaming` 互斥（流式路径下权重不常驻统一内存），且与 int8 路径不兼容。

---

## LoRA 合并 - Architecture Design

category: `architecture_design` · type: `module` · parent: `LoRA 合并`

**合并点分散但规则统一**：`adaln_proj.linear` 与 `norm_out.linear` 在调度预计算前合并（`h3_dit_schedule.c:544-546`、`h3_dit_schedule.c:586-588`）；block 的 qkv / out / fc1 / fc2 在 `load_block` 内合并（`h3_dit.c:880-899`，核心是 `merge_block_loras`，`h3_dit.c:998-1047`）；流式 block 在 `read_stream_layer` 里合并（`h3_dit.c:1948`）。

**为什么需要 `h3_lora_merge_blocking`**：SSD 预取线程可能需要在主线程已经开着一个 command buffer 的时候做合并。该函数在**私有且立即提交**的 command buffer 上执行（`h3_lora.h:35-37`、`h3_lora.c:189-196`、`h3_lora.c:285-288`），从而不与主线程的编码冲突。

**兼容性判据 `h3_lora_matches`**：要求存在 BF16、2 维、且 `lora_A.shape[1] == in_dim` 的因子（`h3_lora.c:154-168`）。不匹配就跳过并告警（`h3_dit_schedule.c:429-435`）。这解决了「diffusers 形状的 turbo 套在被裁剪过的 ConvRot checkpoint 上」这类问题。

**计算实现**：scale 先乘到 B，A 转置后通过**单个融合的 GPU GEAM** 完成（`h3_lora.c:268-298`），而不是逐张量做两次乘加。

---

## 显存分层规划

category: `overview` · type: `module`

`h3_memory_plan.c`(87 行) + `h3_memory_plan.h`(62 行) 是自动内存档位规划器，让 16/24 GB 的 Mac 不需要手工调 `--ssd-streaming` / `--int8` 也能跑起来。设计上「在精神上移植自 ds4 / DwarfStar 的 SSD 流式缓存规划器」（`h3_memory_plan.h:10-13`）。

**输入**（`h3_memory_plan.h:52-56`）：`device`（含 `recommended_working_set`）、`total_weight_bytes`（全驻留时的 BF16 权重总量）、`streamed_resident_bytes`（**应用流式之后**的实际驻留量）、`activation_bytes`（峰值激活缓冲的粗估）。

**输出**（`h3_memory_plan.h:20-38`）：`ssd_streaming`、`use_int8_row_fc2`、`dit_layers`、`video_vae_streaming`，以及一段纯文本 `reason[256]` 供 `--verbose` 展示。

**性质**：规划结果是**建议性的**——调用方可以显式设置对应的 `h3_params` 字段来覆盖它（`h3_memory_plan.h:16-17`）。

---

## 显存分层规划 - Architecture Design

category: `architecture_design` · type: `module` · parent: `显存分层规划`

**决策算法**（`h3_memory_plan.c`）：
1. `target = recommended_working_set * 80 / 100`（`h3_memory_plan.c:40`）——留 20% 余量。
2. `steady = total_weight_bytes + activation_bytes`（`h3_memory_plan.c:41`）。
3. 若 `steady <= target` → 全驻留（不开流式、不量化，`h3_memory_plan.c:45-57`）。
4. 否则同时开启 SSD 流式 + int8 + VAE 流式（`h3_memory_plan.c:61-63`）——三者是正交的档位，不是互斥选项。
5. 极端预算下：`free_after_stream = recommended_working_set - steady_streamed`，若小于 4 GiB 则把 `dit_layers` 压到 `H3_MIN_DIT_LAYERS`（`h3_memory_plan.c:67-70`）。

**`streamed_resident_bytes` 的估算是这个模块的关键**（`h3.c:1250-1261`）：DiT 按「保 2 个 block」折算（`2 * (dit_blocks / H3_DEFAULT_DIT_LAYERS)`）；视频 VAE 解码器按「保约 1 个 block」折算（`video_vae.bytes / H3_VIDEO_VAE_LAYERS`）；编码器按 0 计（每次调用后释放）。注释说明这样做是为了让规划器「不会长期高估内存」。

**调用点**：`h3_generate` 内（`h3.c:1263-1283`）。注意一个容易踩的细节：**流式开启时会强制关闭 `use_int8_row_fc2`**（`h3.c:1269`）。

**为什么 32 GB 是实际下限**：`h3_memory_plan.h:32-35` 指出视频 VAE 解码器是最大的单块固定占用，也是内存吃紧时建议开启 `video_vae_streaming` 的原因。

---

## 视频 VAE 解码器

category: `overview` · type: `module`

`h3_video_vae.c`(1348 行) + `h3_video_vae.h`(61 行) 实现第 5 阶段的视频侧解码：把视频 latent 解码成 RGB 帧，支持分块（tiled）解码与逐步预览。

**两条解码路径，由 `vae->streaming` 选择**（字段 `h3_video_vae.c:93`，赋值 `h3_video_vae.c:1017`）：
- `run_resident_tile`（`h3_video_vae.c:550`）：36 个 decoder block 全部常驻，约 9 GiB，**快**（`h3_video_vae.c:535`、`h3_video_vae.c:544-549`）。
- `run_stream_tile`（`h3_video_vae.c:600`）：逐块 load → run → free，仅约 1 个 block 常驻，约 0.25 GiB（`h3_video_vae.c:596-599`、`h3_video_vae.c:641-653`）。

**块数由 `H3_VIDEO_VAE_LAYERS = 36` 导出**（`h3_video_vae.h:24`），被 `h3_video_vae.c:19` 的 `LAYERS`、内存规划 `h3.c:1257`、CLI `h3_cli.c:764` 共同引用。

**参数来源**：`h3_params.video_vae_streaming`（`h3.h:112`，`-1` = 交给自动规划器，`0` = 强制常驻，`1` = 强制流式），传入点见 `h3.c:2070`、`h3.c:2151`、`h3.c:2166`、`h3.c:2415`。

---

## 视频 VAE 解码器 - Architecture Design

category: `architecture_design` · type: `module` · parent: `视频 VAE 解码器`

**两个解码入口都必须分支 `streaming`——这是本模块最重要的维护约束**：`decoder_decode_chunk`（常驻 decoder 的入口）在 `h3_video_vae.c:957-959` 分支，`decode_chunked`（一次性路径的入口）在 `h3_video_vae.c:1219-1221` 分支。两者共享同一个 `vae_context`，且各有服务场景：常驻 decoder 服务预览与最终解码（`h3_video_vae.c:1045`、`h3_video_vae.c:1069`），而 `h3_video_vae_decode` 在大尺寸 / 多 chunk 时会转向 `decode_chunked`（`h3_video_vae.c:1321-1328`）。**改了其中一处就必须同步另一处**，否则流式开关在一条路径上失效。

**常驻 vs 流式是典型的「时间换空间」**：常驻快但吃 9 GiB，流式省到 0.25 GiB 但每块都要读盘。这正是内存规划器把 VAE 流式作为独立档位的原因（`h3_memory_plan.h:32-35`）。

**分块解码与预览共用一条路径**：`decoder_decode_chunk` 同时被预览（每个 Euler 步一张代表帧）与最终解码使用，因此预览帧与最终帧的解码逻辑天然一致。

**空间压缩比 16**：`H3_VAE_SPATIAL_RATIO 16`（`h3_host.h:11`）决定了 latent 画布与像素画布的换算，`h3_latent_canvas` 是唯一换算入口（`h3_host.h:99`）。

---

## 音频 VAE（BigVGAN）

category: `overview` · type: `module`

`h3_audio_vae.c`(1372 行) + `h3_audio_vae.h`(48 行) 实现音频侧的编解码：解码是流式原生 BigVGAN / AudioVAE，编码用于参考音频。

**关键常量**（`h3_audio_vae.c:13-24`）：`LATENT_CHANNELS=32`、`LATENT_DIM=2048`、`DECODER_DIM=1024`、`STEREO=2`、`STAGES=7`、`SAMPLE_RATE=32000`、`HOP_LENGTH=800`。

**上采样率与卷积核**（`h3_audio_vae.c:26-27`）：`{5,5,2,2,2,2,2}` 与 `{9,9,4,4,4,4,4}`——乘积恰好 800，等于 `HOP_LENGTH`。这是解码器结构自洽的校验点。

**解码流程**：`prepare_input` 反归一化 `z*std+mean`（`h3_audio_vae.c:351`、`h3_audio_vae.c:366`）→ `dec_in_proj` / `decoder.conv_pre`（`h3_audio_vae.c:379-384`）→ 7 个 stage（`h3_audio_vae.c:715-722`）→ `decode_output`（`h3_audio_vae.c:726`）。激活函数是 alias-free SnakeBeta（`h3_audio_vae.c:500`），输出 clip 到 [-1,1]（`h3_audio_vae.c:661-664`），waveform 形状 `[2, samples]`（`h3_audio_vae.c:677-679`）。

**编码器**：5 个 stage、stride `{2,4,4,5,5}`（`h3_audio_vae.c:749`），输出 32 通道 latent（`h3_audio_vae.c:1294`），输入右侧补零到 800 边界（`h3_audio_vae.c:796`）。

---

## 音频 VAE（BigVGAN） - Architecture Design

category: `architecture_design` · type: `module` · parent: `音频 VAE（BigVGAN）`

**与视频侧一致：解码走流式**。README 明确「公开生成路径用流式原生 BigVGAN/AudioVAE 解码联合音频 latent」（`README.md:814-815`），这也和内存规划器把编码器按 0 常驻计算一致。

**数值保真度有量化记录**：原生 waveform 与修正后的 MLX oracle 相对 L2 为 `6.94e-5`（`README.md:815-816`）；原生音频编码器在真实 2 秒立体声 fixture 上相对 L2 为 `3.59e-6`（`README.md:836-837`）。

**一个必须知道的坑**（`README.md:837-839`）：原始 MLX reshape 把左右声道样本交错，而官方 PyTorch / SGLang 路径是把完整立体声通道折进 batch 维度。这个修正对音频正确性是关键，改音频相关代码时要保持立体声通道不被交错。

**时间轴与视频对齐**：音频 latent 是 40 fps（`H3_AUDIO_LATENT_FPS 40`，`h3_host.h:10`），输出 24 fps 视频（`H3_FPS 24`），两者各自有独立的 sigma shift（音频 3.0、视频 12.0）。

**有独立的 trace 开关**：`H3_AUDIO_ENCODER_TRACE`（`h3_audio_vae.c:755`）用于导出编码器中间张量做对照。

---

## 媒体封装与 ffmpeg 管线

category: `overview` · type: `module`

`h3_ffmpeg.c`(941 行) + `h3_ffmpeg.h`(60 行) 完成第 6 阶段：把 RGB 帧与 F32 PCM 交给外部 `ffmpeg` 进程，产出最终 MP4 + AAC。

**入口**：`h3_ffmpeg_write_av_rgb24_f32`（`h3_ffmpeg.c:614`），调用点在 `h3.c:2213` 与 `h3.c:2484`。

**用管道，不用临时文件**：开两条 `pipe()`（`h3_ffmpeg.c:647`），头文件也明示这一点（`h3_ffmpeg.h:41-42`）。README 补充：不产生中间未压缩媒体文件（`README.md:471-472`）。

**编码参数**（`h3_ffmpeg.c:668-679`）：输入 `rawvideo` / `rgb24` 与 `f32le`，输出 `-c:v libx264 -crf 18 -pix_fmt yuv420p -c:a aac -b:a 192k -movflags +faststart`。

**音频参数来自实际输出**：采样率/声道取自 BigVGAN 的 `waveform.channels` / `sample_rate`（`h3.c:2215-2216`）；无音频时用 24000 Hz 静音兜底（`h3.c:2473-2482`）。

---

## 媒体封装与 ffmpeg 管线 - Architecture Design

category: `architecture_design` · type: `module` · parent: `媒体封装与 ffmpeg 管线`

**进程模型**：`posix_spawnp` 启动 ffmpeg（`h3_ffmpeg.c:693`），然后开**两个 pthread** 分别写视频与音频管道（`h3_ffmpeg.c:718-730`），`waitpid` 收尾（`h3_ffmpeg.c:735`）。两个写线程是必需的：ffmpeg 需要同时拿到音视频流，单线程顺序写会死锁。

**PCM 需要先交错**：写之前把音频排成 `[samples, channels]` 交错布局（`h3_ffmpeg.c:641-644`），匹配 `f32le` 输入的期望。

**SIGPIPE 被忽略**（`h3_ffmpeg.c:706-710`）：ffmpeg 提前退出时写管道会触发 SIGPIPE，忽略它才能让主流程走到 `waitpid` 并报告真实错误，而不是整个进程被信号杀掉。

**外部依赖是运行期要求**：`ffmpeg` 与 `ffprobe` 必须在 `PATH` 上；可用 `H3_FFMPEG` / `H3_FFPROBE` 指定具体可执行文件（`README.md:469-470`）。测试 `h3_av_mux_test` 也会在缺 ffmpeg 时跳过（`Makefile:185-189`）。

**图像/视频输入也走 ffmpeg 家族**：`h3_ffmpeg_read_image_f32`（`h3.c:1710`）负责读图，参考视频用有界 24 fps 解码（`README.md:824-826`）。

---

## CLI 与交互会话

category: `overview` · type: `module`

`main.c`(665 行) + `h3_cli.c`(950 行) + `h3_terminal.c`(304 行) + vendored 的 `linenoise.c`(1763 行) 构成命令行前端。

**两种运行模式**：给 `-p` 时一次性生成；不给 `-p` 时进入 Iris 风格的交互会话（`README.md:29-33`）。

**会话复用**：交互会话把精确的 BF16 提示条件、已准备的 DiT、视频解码器保留在内存里，因此换个 seed 重复同一提示可以跳过加载与编码（`README.md:35-39`）。这是 `h3_cache_set_enabled` 的主要用途（`h3.h:210-215`）。

**会话内命令**：`!status`、`!seed random`、`!seconds 2`、`!show`、`!save output.mp4`、`!cache`，以及 `!first` / `!last` 锚点、`!ref-image` / `!ref-remove` / `!refs` 参考管理（`README.md:38-63`）。`!help` 给出完整短清单。

**重要约束**：Ref2VA 参考与 `!first` / `!last` 锚点**不能混用**（`README.md:62-63`）。

---

## CLI 与交互会话 - Architecture Design

category: `architecture_design` · type: `module` · parent: `CLI 与交互会话`

**CLI 是薄壳**：`main.c` / `h3_cli.c` 只负责参数解析、会话状态与把 `h3_params` 填好，真正的推理全部在 `h3.c` 的公共 API 后面。CLI 不直接触碰 DiT / VAE 内部结构。

**终端能力是探测的**：`h3_terminal.c` 负责终端尺寸、是否支持图形化逐帧预览等判断；去噪预览在支持的图形终端里显示演化中的中间帧（`README.md:67-70`）。

**linenoise 是 vendored 的**：来自 Iris 项目，`Makefile:250-252` 特意为它局部放宽 `-Wconversion` 等警告，让主项目保持严格——这是「不要为了 vendored 代码改写它」的明确取向。

**会话状态与公共 API 的边界**：会话把「已缓存的条件 + 已准备的 DiT + 视频解码器」放进 `h3_ctx`（`h3_internal.h:12-36` 的 `conditioning_*`、`dit`、`video_decoder`），并用 `dit_key` / `video_decoder_key` 做失效判断。

**CLI 也参与内存估算**：`h3_cli.c:764` 引用 `H3_VIDEO_VAE_LAYERS` 来展示每块占用，说明 CLI 层只读取导出常量、不重复推导。

---

## CLI 与交互会话 - Setup & Commands

category: `unique_setup_and_commands` · type: `module` · parent: `CLI 与交互会话`

**典型调用**：
```sh
make -j8
mkdir -p outputs
./h3 --info -d ./MiniMax-H3                       # 只查布局与设备
./h3 -d ./MiniMax-H3 -p "..." -o out.mp4          # 一次性生成
./h3 -d ./MiniMax-H3 --width 512 --height 512 --steps 6   # 进入交互会话
```
（`README.md:19-33`）

**会话内命令**（`README.md:38-63`）：
```text
!status / !help / !show / !cache
!seed random / !seconds 2 / !save output.mp4
!first opening.png     # 首帧锚点
!last ending.png       # 尾帧锚点
!first clear / !last clear
!ref-image person.png  # Ref2VA 参考图，按顺序追加为 <Picture 1> …
!refs / !ref-remove N / !refs clear
```

**参考视频/音频的参数形式**（`README.md:824-834`）：`--ref-video` 保留内嵌音轨，`--ref-video-audio VIDEO AUDIO` 显式替换音轨，`--ref-audio` 追加独立音频片段（必须与图像或视频参考组合）。

**诊断**：`--profile` 分阶段报告墙钟时间、CPU 侧命令编码耗时、commit-to-fence 等待、root command GPU 时间戳、峰值 tensor 存储、累计分配与 dispatch 次数（`README.md:846-851`）。`--latent-out` / `--latent-in` 可把去噪后的视频 latent 落盘再单独解码（`h3.h:147-152`）。

---

## 测试体系

category: `overview` · type: `module`

`tests/` 下 30+ 个独立 C 测试程序，每个都是自己的 make target（`Makefile:39-137`），没有测试框架——每个文件自带 `main()` 与断言式比较。

**三类测试**：
1. **无依赖测试**：`h3_tests`（主机侧几何/调度/布局）、`h3_audio_gpu_tests`、`h3_convrot_test`、`h3_vdn_tests`。这些在没有权重时也会真正运行。
2. **fixture 驱动测试**：需要 `misc/fixtures/*.safetensors`，例如 `h3_metal_tests` / `h3_bf16_tests`（MLX 玩具 block）、`h3_text_tests`、各 `h3_real_*_test`。缺文件时由 Makefile 打印 `skip:` 而不失败。
3. **权重驱动测试**：直接读 `MiniMax-H3/` 真实 checkpoint，例如 `h3_real_prompt_test`、`h3_real_dit_block_test`、`h3_real_video_vae_test`。

**一致性（parity）测试是核心方法论**：`make parity` 用 MLX 玩具 block 的具名输出校验 Metal 与 BF16 两条路径（`Makefile:229-232`）；`make real-parity` 对真实权重做同样的事（`Makefile:234-236`）。README 说明这覆盖了 F32 诊断路径与生产 BF16 存储路径两者（`README.md:460-467`）。

**数据生成脚本**：`tests/gen_lora_data.py` 生成 `tmp_lora_test/` fixture，供 `h3_lora_tests` 使用——**必须先跑它**，否则该套件没有可比对的数据。

---

## 测试体系 - Build System

category: `build_system` · type: `module` · parent: `测试体系`

**测试在 Makefile 里的组织方式**：每个测试是 `h3_<name>_test: tests/test_<name>.o $(LIB_OBJ)` 形式的 target，统一链接全部 `LIB_OBJ` 与 `LDLIBS`（`Makefile:39-137`）。新增测试要照抄这个模式。

**条件执行的写法**（`Makefile:147-227`）——这是本项目测试的惯用法，新增依赖权重的测试请沿用：
```make
@if test -f MiniMax-H3/tokenizer/tokenizer.json; then \
    ./h3_tokenizer_tests MiniMax-H3/tokenizer/tokenizer.json; \
else \
    echo "skip: released tokenizer is not installed"; \
fi
```

**`make test` 的实际覆盖范围**（`Makefile:139-145` 的依赖列表 + 147-227 的执行体）：依赖 `h3_tests h3_metal_tests h3_bf16_tests h3_tokenizer_tests h3_text_tests h3_audio_gpu_tests h3_real_audio_vae_test h3_real_audio_vae_e2e_test h3_real_audio_encoder_test h3_av_mux_test h3_real_video_encoder_test h3_real_qwen_vision_test h3_real_multimodal_text_test h3_real_ref_video_text_test h3_convrot_test h3_vdn_tests`。

**不在 `make test` 里的套件**（容易漏跑）：`h3_lora_tests`、`h3_real_dit_test`、`h3_real_dit_schedule_test`、`h3_semantic_dit_test`、`h3_real_video_vae_test`、`h3_semantic_vae_test`、`h3_linear_branch_tests`、`h3_flash_attn_tests`、`h3_tiled_windowed_tests`、`h3_clipproj_test`、各 bench。

**bench 的特殊处理**：`tests/bench_dit_864.o` 用额外的 `-DH3_BENCH_LATENT_H=30 -DH3_BENCH_LATENT_W=54` 编译（`Makefile:114-116`），是唯一带专用编译参数的目标。

**运行时编译内核让测试有特殊要求**：Metal 内核在测试运行时才编译，因此测试二进制也必须能从其工作目录找到 `h3_shaders.metal`——测试的做法是显式传路径（`Makefile:150-151`），CLI 则依赖 CWD。

---

