# ClipProj 轻量文本路径

ClipProj 用一个 **4B 模型 + 一个小型投影 MLP** 替代 **62 GiB 的 50 层 Qwen3-VL 编码器**，把文本编码的读取量降到 ~5.5 GiB。

## 动机

H3 的文本编码器（Qwen3-VL 前 50 层）是整个 checkpoint 里第二大的组件。但对**纯文本**路径而言，它的 50 层深度是冗余的——真正需要的只是最后那 5120 维条件嵌入。

ClipProj 观察到：截断到某个抽取层的 Qwen3-VL-**4B** 已经能给出足够好的中间表示，只需一个投影把它升到 H3 的条件空间。

| 路径 | 读取量 | 适用 |
|---|---:|---|
| 50 层编码器 | ~62 GiB | 多模态（有视觉跨度）、对齐检查 |
| 4B + ClipProj | ~5.5 GiB | 纯文本（默认） |

Sources: [h3_text_encoder.h](h3_text_encoder.h#L67-L76)

## 结构

```mermaid
flowchart LR
    A["token ids"] --> B["Qwen3-VL-4B<br/>截断到抽取层"]
    B --> C["2560 维抽取隐藏"]
    C --> D["mean/std 归一化"]
    D --> E["2 层投影 MLP"]
    E --> F["attention sink 替换"]
    F --> G["h3_text_embedding<br/>width = 5120"]
    G --> H["DiT（完全无改动）"]
```

三步后处理（归一化 → 投影 → attention sink 替换）与 `clipproj_harness.py` 中的离线参考实现严格对应。

**关键性质**：输出宽度正是 `H3_TEXT_HIDDEN_SIZE == 5120`，与 50 层路径逐位兼容，因此 DiT 侧不需要任何改动。

Sources: [h3_text_encoder.h](h3_text_encoder.h#L67-L84)

## 实现位置

`h3_text_encode_clipproj_bf16`（`h3_text_encoder.c:985`）是本路径的入口，配套一组 `cp_*` 辅助函数：

| 辅助 | 行 | 职责 |
|---|---:|---|
| `cp_bf16_to_f32` / `cp_f32_to_bf16` | 798 / 805 | dtype 转换 |
| `cp_f16_to_f32` | 813 | 投影权重可能是 F16 |
| `cp_load_2d` / `cp_load_1d` / `cp_load_f32` | 834 / 841 / 850 | 权重加载 |
| `cp_gelu` | 888 | 投影的激活函数 |
| `cp_encode_layer` | 892 | 4B 模型的单层 |

Sources: [h3_text_encoder.c](h3_text_encoder.c#L798-L985)

## 开关

```c
const char *cp_dir = getenv("H3_CLIPPROJ_DIR");
if (!cp_dir || !*cp_dir)
    /* 默认开启 */
    cp_dir = "/Volumes/data/.lmstudio/models/Qwen3-VL-4B-Instruct";
int use_clipproj = strcmp(cp_dir, "0") != 0 && strcmp(cp_dir, "off") != 0;
```

| 环境变量 | 默认 | 语义 |
|---|---|---|
| `H3_CLIPPROJ_DIR` | `/Volumes/data/.lmstudio/models/Qwen3-VL-4B-Instruct` | BF16 Qwen3-VL-4B-Instruct 目录；设为 `0` 或 `off` 回退到 50 层编码器 |
| `H3_CLIPPROJ_PROJ` | `/Volumes/data/.lmstudio/models/ClipProj-MiniMax-H3` | 投影权重目录 |

Sources: [h3.c](h3.c#L1490-L1500)

## 对模型加载的影响

**这是唯一一处会放宽目录校验的地方。** 当 ClipProj 激活时，`FL2VA/text_encoder` 缺失不再致命：

```c
if (!h3_inventory(ctx, "FL2VA/text_encoder", &ctx->model.text_encoder)) {
    if (!clipproj_active) { /* 报错返回 */ }
    memset(&ctx->model.text_encoder, 0, sizeof(ctx->model.text_encoder));
    h3_global_error[0] = '\0';
    ctx->error[0] = '\0';
}
```

换句话说，只做纯文本生成的用户**不需要下载 text_encoder 那 62 GiB**。

Sources: [h3.c](h3.c#L434-L457)

## 适用范围

ClipProj **只在纯文本路径上生效**（`h3.c:1484` 的 `else` 分支，即 `visual_count == 0`）。一旦存在视觉条件——FL2VA 锚点或 Ref2VA 参考——就必须走 `h3_multimodal_encode_*`，因为视觉跨度注入与 deepstack 需要完整的 50 层。

Sources: [h3.c](h3.c#L1484-L1519)

## 保真度验证

仓库提供了一套 golden 对照脚本：

| 文件 | 作用 |
|---|---|
| `clipproj_harness.py` | 离线参考实现（A_local） |
| `clipproj_golden.sh` | 编排：跑引擎侧 B，与 A_local 比对 |
| `clipproj_golden_compare.py` | 数值比对 |

Makefile 目标：

```sh
make clipproj-golden
```

相关路径由 Makefile 顶部的变量给出（`QWEN4B`、`PROJ`、`CLIPPROJ_MODEL`、`CLIPPROJ_PROMPT`）。

Sources: [Makefile](Makefile#L18-L22), [Makefile](Makefile#L207-L208)
