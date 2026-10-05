# h3c 项目长期记忆

## 代码库上下文工具（context-kit）

- 索引与知识库数据在 `~/.context-kit/data/`，按工作区键 `h3c-8cd7ead6` 隔离（键 = 目录名 + 路径哈希，由 `paths.workspaceKey` 生成）。
- 常用命令（必须带 `--cwd`，且从仓库根运行）：
  - 刷新索引：`node ~/.workbuddy-ai/skills/context-kit/src/cli.mjs index --cwd /Volumes/data/git/c/h3c`
  - 代码检索：`... search --kind workspace --cwd <root> --query "..."`
  - 知识检索：`... search --kind knowledge --mode search|fetch --cwd <root> --query "..." / --titles "精确标题"`
  - 生成注入文本：`echo '{"cwd":"<root>"}' | ... context-prompt`
- 项目级前置配置在 `.context-kit/repowiki/wiki_plan.yaml`（未被 .gitignore 覆盖）。
- 知识库当前 45 条条目，内容由 Agent 阅读源码撰写、带 `path:line` 依据。**该仓库没有配置 LLM provider**，所以新增/更新条目要走 `KnowledgeBuilder.write()` 结构化写入，而不是期望自动生成。

## 本地对 context-kit 的补丁（非上游）

`~/.workbuddy-ai/skills/context-kit` 是用户级安装、无 `agent_created` 标记，因此补丁是直接改源码的，**升级或重装该 skill 会丢失**：

1. `src/knowledge/schema.mjs` 新增 `entryContentKey(id)`：条目正文文件名 = 短 slug（≤48）+ id 短哈希。修复存储层 `paths.sanitizeKey` 的 ASCII-only 清洗 + 64 字符截断导致的中文标题文件名碰撞与正文错配。`builder.mjs` / `store.mjs` 共用此函数。
2. `schema.normalizeEntry` 增加 `summary`；`builder.write` 落一份单行摘要；`store.renderTreeLines` 优先读 `entry.summary`。修复概览树只有标题、没有摘要的问题。

重新生成知识库前建议先 `builder.clear()`，否则历史碰撞产生的同名旧文件会残留。

## 项目约定（来自本次代码审计）

- 只允许 Apple clang 构建；所有二进制必须从仓库根运行（`h3_shaders.metal` 按 CWD 相对路径解析）。
- 新增库源文件必须加入 `Makefile` 的 `LIB_C` / `LIB_M`，否则 `Undefined symbols`。
- 块数/层数一律用导出常量（`H3_DIT_BLOCKS`、`H3_VIDEO_VAE_LAYERS` 等），禁止硬编码。
- 全项目零 `assert`：错误用返回码 + 错误字符串（`h3_set_error` / `h3_gpu_set_error`）。

## 权重格式契约（packed group-quantized）

- `{name}` **U8** `[rows, cols*bits/8]`；`{name}_scale` / `{name}_bias`
  F16 或 F32 `[rows, cols/group]`；`w = code*scale + bias`。
  引擎从 dtype+shape 反推：`bits = packed_row_bytes*8/columns`，`group = columns/scale_cols`，
  `group` 必须是 4 的倍数。见 `h3_weights.h` / `h3_weight_grouped_spec`。
  **命名陷阱**：`{name}` 是**含 `.weight` 的完整矩阵键**，辅助张量是把 `_scale` / `_bias`
  直接接在其后，所以实际叫 `blocks.0.attn.qkv_proj.weight_scale` —— **不是**
  `...weight_weight_scale`，也不是 `...qkv_proj.weight_scale` 之外的任何形式。
  `h3_weight_grouped_spec`（`h3_weights.c:623-628`）用 `name + "_scale"` 精确查表；
  `h3_dit.c:1243` 的 `"%.*s.weight_scale"`（`nl-7`）是同一约定的 printf 写法。
  名字写错时 `h3_weight_grouped_spec` 返回 **0**（不是 -1），于是报
  `grouped streaming weight schema mismatch` —— 注意它**不会**报"缺少 scale"，
  因为返回 0 的三条路径（weight 非 U8 / 找不到 `_scale` / 找不到 `_bias`）共用同一条外层错误。
  `--info` 只统计张量个数，**不会**发现这个错误，必须真正跑一次生成。
- **该格式契约上位于 ConvRot 旋转后的空间**。流式 grouped 分支无条件去旋转
  （`h3_dit.c:1785`），且没有 `H3_INT8_UNROTATE` 开关；常驻路径
  （`h3_weights.c:853`）有开关且不做 qkv 重排 —— 两条路对 qkv 行序的假设不一致。
- **grouped 权重只能走 `--ssd-streaming`**：常驻 `load_block` → `bf2_convrot`
  按 dtype 分派，只认 I8 / BF16。
- 引擎的 qkv_proj 行序是**按 head 交错**（`3*head + which`），内核读
  `qkv[base+d]` / `qkv[base+head_dim+d]` / `qkv[base+2*head_dim+d]`；
  而存储侧（发布版 BF16、ConvRot INT8、mere-run 产物）是**全局 slab** `[all-q;all-k;all-v]`。
  `h3_shaders.metal:80-85` 的注释把两个布局命名写反了，**以代码为准**。

## mere-run 量化产物（MLX affine）

- `{name}.weight` U32 `[rows, cols/8]`（8 个 4-bit 码/U32，**首个码占低位**）；
  `.scales`/`.biases` BF16 `[rows, cols/64]`；`w = code*scale + bias`。
- `/Volumes/data/MODELS/h3-16gb-q4/transformer.safetensors` 由发布版 BF16 直接量化而来，
  因此是**未旋转**空间、qkv 为**全局 slab**（`qkv_layout: global-qkv-slabs`），
  且**省略了全部 50 个 `blocks.N.adaln_proj.linear.weight`**（每个 496 MiB，共 24.2 GB）。
  它的 11.33 GB 主要来自省略 AdaLN，不是 4-bit 的功劳。
  **实测账**：q4 10.56 GiB vs h3c 可加载产物 35.93 GiB，差额 +25.38 GiB 中
  **+24.28 GiB 是 51 个 AdaLN 投影**（q4 里 `adaln` 键数量为 **0**，
  `cache_covered_weights_omitted: true` —— mere-run 缓存 AdaLN 的**输出**）；
  **4-bit 那一半两边一样大**（0.5625 B/param）。
  这个差额**换 dtype 减不掉**：F16 与 BF16 同为 16 bit/param；引擎的 AdaLN 加载器
  `weight_bf16_any`（`h3_dit_schedule.c:94`）只收 BF16/F16/F32，存不了 packed 4-bit。
  注意 AdaLN 的 dtype **不**参与格式探测（`h3_dit_schedule.c:469` 只看 ndim/shape）。
  **但可以靠 AdaLN 输出缓存整体省掉 —— 见下节。**

## ConvRot INT8 逐行量化格式（pruned 发布版 / ComfyUI）

`/Users/jay/h3_sys/MiniMax-H3-Convrot/FL2VA/transformer/` 是**另一套参考模型**，h3c 已原生支持：

- 每个量化矩阵三件套：`{name}.weight` **I8** `[rows, cols]`；
  `{name}.weight_scale` **F32** `[rows, 1]`（**逐行一个 scale，无 bias**）；
  `{base}.comfy_quant` **U8[72]**（`{base}` = 去掉 `.weight`，**不是** `.weight.comfy_quant`）。
  侧车内容 `{"format":"int8_tensorwise","convrot":true,"convrot_groupsize":256}`。
  解析在 `h3_weights.c:348-460`（`weight_sidecar_name` / `comfy_quant_group_size`）。
- **1.00074 B/参数**，比 h3c 的 4-bit group-64（0.5625）**大 1.78×**。实测 MLP 比值 1.779。
- **`adaln_t_table` table mode**（`h3_dit_schedule.c:390-500`）：pruned 版把 AdaLN 输入
  从 2688 维时间嵌入换成 `adaln_t_table` F32 `[1025, 8]` 查表 + 线性插值
  （`interpolate_curve_table`，`pos = t * (rows-1)`）。
  `blocks.N.adaln_proj.linear.weight` 变成 **F16 `[96768, 8]`**（发布版是 BF16 `[96768, 2688]`）。
  探针逻辑：`shape[1] != H3_DIT_TIME_DIM(2688)` 就走 table mode，且此时 `adaln_t_table` 缺失即报错。
- **`attn.to_gate_compress` 在全部 `.c/.h/.m` 里零引用** —— VSA 专用，h3c 不实现。
  拿这类 checkpoint 直接跑会**缺掉 VSA 门控**，语义不等价。
- 命名：`blocks.N.attn.qkv_proj.weight` I8 `[21504, 5376]`，out_proj `[5376, 7168]`，
  mlp fc1 `[28672, 5376]`，fc2 `[5376, 14336]`。qkv 是**全局 slab**（`concat(to_q,to_k,to_v)`），
  引擎自己转交错，**不要**再置换（`mlx_int8_to_h3.py` 的 5 条 correctness notes 逐条讲了这个）。

## AdaLN 输出缓存（已实现，按步数多缓存）—— 省掉 24.29 GiB 的正路

`h3_dit_schedule_precompute`（`h3_dit_schedule.c`）**一次性算出所有 step 的 AdaLN 调制**：
每块算完立刻 `free_tensor(&weight)` —— 24.288 GiB 权重**只被读一次**，产出只有
`50 × time_rows × 96768 × 2 B`，即**每行 9,698,308 B = 9.25 MiB**（含 4 B times）。
行数由步数决定，**无视觉条件时 `time_rows = 2*steps - 1`**（第 0 步 video==audio 只占 1 行，
之后每步 2 行）：实测 `--steps 2` → 3 行（29,094,948 B）、`--steps 8` → 15 行（145,474,644 B）。

**缓存按 `--steps` 分键，可共存**（`H3_ADALN_CACHE_{TIMES,BLOCK,FINAL}_FORMAT`）：

    adaln_cache_times_s{steps}        F32  [rows]      ← 校验键
    blocks.N.adaln_cache_s{steps}     BF16 [rows, 96768]
    final_layer.adaln_cache_s{steps}  BF16 [rows, 10752]

`blocks.0.adaln_cache_s{steps}` 存在即选中缓存路径，无需开关。加载时把本次 sigma schedule
算出的 `times[]` 与缓存键**逐位 memcmp**，要求是**前缀**；太短/不匹配就硬报错。

**一个缓存覆盖三种条件模式**（"前缀"规则的全部意义）。条件行**追加在 step 行之后**，
顺序固定 visual 再 audio，故三种模式的 `times[]` 嵌套：

    text-to-video      2*steps - 1 行   无条件行
    首/末帧 (I2V)      2*steps     行   visual 行
    参考图 (Ref2VA)    2*steps + 1 行   visual 行 + audio 行

`H3_DIT_ADALN_CACHE_DUMP` 导出时**同时强制 `visual_condition = audio_condition = 1`**，
于是缓存是 `2*steps + 1` 行，短模式按前缀命中。（只强制 visual 会让参考图差一行被拒，
而"重建缓存"根本修不好 —— 死循环陷阱，已修。）纯音频参考（`H3_LAYOUT_REF_AUDIO` 无图）
是唯一不覆盖的模式，其条件行落在 visual 行的位置而被 memcmp 拒绝；该模式 CLI/REPL 都到不了，
只有 C API 能构造。

**用法**：① 从**还带权重**的 checkpoint 导出，每个步数一次：
`H3_DIT_ADALN_CACHE_DUMP=/tmp/adaln_s$N.raw ./h3 -d <完整模型> ... --steps $N`
（导出运行本身是完整运行，可顺带当 A/B 的 A 侧基线）。
② `python fastvideo_qad/scripts/export_h3_adaln_cache.py --dump ...s4.raw --dump ...s8.raw
--dump ...s20.raw --model <完整dir>/FL2VA/transformer --trim-to <新dir>/FL2VA/transformer`，
再加 `--verify` 复查。`--trim-to` 写**新目录**，源模型不动；只丢 `".adaln_proj.linear."` 键
（`final_layer.norm_out.linear` 保留）。

产物 `/Volumes/data/MODELS/h3c-q4-adalncache`：**12.25 GiB**（11.65 权重 + 619.7 MiB 缓存），
原 35.93 GiB；与 mere-run 的 10.55 GiB 同量级，但**能直接被 h3c 加载**。

**实测等价（4/8/20 三档全部 bit-identical）**：

| steps | time_rows | 缓存 | A 全权重 | B 缓存 | latent md5 | mp4 md5 |
|---|---|---|---|---|---|---|
| 4 | 9 | 83.2 MiB | 151 s | 119 s | 6e9c2d87… | 0330454e… |
| 8 | 17 | 157.2 MiB | 250 s | 219 s | a7f85c8b… | 35e5fdb3… |
| 20 | 41 | 379.2 MiB | 556 s | 501 s | dc8c5d41… | 569f5be2… |

latent 与 mp4 容器 **md5 全同**，`relRMS=0.000000 cos=1.000000 max|d|=0`。
steps=8 的 md5 与改成按步数分键**之前**那版完全一致 —— 证明重构与"多算两个条件行"
没有改变任何数值。负向对照：缓存模型跑 `--steps 6` → `exit=1`，
`this checkpoint ships AdaLN caches but none for 6 steps, and it has no adaln_proj weights
to fall back on; export a cache at these steps with H3_DIT_ADALN_CACHE_DUMP`
（缓存专用 checkpoint 连 `blocks.0.adaln_proj.linear.weight` 也没有，所以判断
`!cache && !adaln_probe` 只在注定失败的路径触发，只是把报错变清楚）。

**其它事实**：
- 缓存是 `(sigmas, visual_condition, audio_condition)` 的纯函数，**与分辨率/帧数/seed 无关**。
- **LoRA 除外**：`merge_adaln_loras` 要合并进 AdaLN 权重，缓存模式下直接拒绝。
- **`!layers N` 块剪枝在缓存模式下依然正确**：`h3_dit_schedule_gate_scores` 读的是
  `schedule->blocks[block]` 的 slot 2/5，而**非缓存路径下这个数组装的也是调制**
  （`h3_gpu_linear_bf16` 把投影结果写进去），不是权重 —— 缓存逐位复现同一内容，
  且 `time_rows` 用本次运行的值，故 gate 分数与剪枝结果完全一致。
  （早先记的"缓存模式下 gate 排名会坏"是误读，已纠正。）
- 发布版 50 个 AdaLN 权重哈希**互不相同**，没有共享去重空间。
- 导出格式：`magic[8]="H3ADALN2"` + `uint32[6]={steps, time_rows, 50, 96768, 10752, 0}` +
  `f32 times[rows]` + `u16 blocks[50][rows*96768]` + `u16 final[rows*10752]`，头 32 B。
  脚本按 `steps` 反推文件名 `adaln_cache_s{steps}.safetensors`；旧版 `H3ADALN1`
  （24 B 头、无 steps）的 dump 会被明确拒绝并提示重新导出。

## 判断数值确定性的正确方法

**不要比 mp4 的 md5** —— 同一 latent 解码出的 mp4 容器字节可能不同（实测同尺寸
87237 B、解码帧 PSNR = inf，但 md5 不同）。要比 `--latent-out` 的 latent
（逐字节）或解码帧（PSNR）。实测：同一模型同一参数连跑 5 次 latent 逐字节相同；
常驻 9 块与 4 块也得到同一结果。

## 运行 h3c 的前置

- `FL2VA/text_encoder` 缺失时必须设 ClipProj，否则 `h3_create` 直接失败：
  `H3_CLIPPROJ_DIR=/Volumes/data/.lmstudio/models/Qwen3-VL-4B-Instruct-int8-convrot`
  `H3_CLIPPROJ_PROJ=/Volumes/data/.lmstudio/models/ClipProj-MiniMax-H3`
- `--steps ∈ [2,1000]`；`--frames` 必须是 17n+5 且 **≥ 22**（至少一个训练过的解码 chunk）。
- `h3_st_inventory_dir` 扫目录内任意 `*.safetensors`，不读 index.json。
- **16 GiB 机器上必须限制常驻块数**。自适应常驻预算
  （`h3_dit.c:2756-2790`，`(available - activation_reserve) / 0.5 GiB`）会挑 10~13 块
  并几乎吃满可用内存，然后在 denoise 第 0 块被 macOS **SIGKILL（exit 137）**。
  **BF16 发布版同样会被杀**，与量化无关。稳定跑法：`H3_DIT_RESIDENT_BLOCKS=4`。
  常驻块数**不影响数值**（13 块与 4 块的输出 md5 完全相同）。
- 流式失败的真实原因可能被吞成 `unknown Metal error`：`h3_dit.c:1906` 的
  非流水线分支 `||` 短路后不会调用 `stream_consume_slot`，而只有它会把
  `ring->error` 拷进 `job->error`。**排查时设 `H3_DIT_STREAM_PIPELINE=1`**
  强制走读线程，真实错误就会显出来。

## 校验量化权重是否被引擎正确解读

最省事也最严格的办法：`H3_DEBUG_CONVROT=1` 会让引擎打印它**去旋转后写进 slot 的
BF16 权重**（`h3_dit.c:1806-1838`），即引擎对文件的最终解读。用 Python 按同一套
规则复算自己的文件去比对，误差应 ≤ 1 个 bf16 ULP。实测（4-bit 转换产物）：
`out_proj[0][0..7]` Δ=3.5e-4、qkv 元素 7168 Δ=7.4e-4、元素 14336 Δ=4.1e-4，全部达标。

两个坑：
1. **探针的第二个参数是元素偏移，不是行偏移**（`h3_gpu.m:913`
   `h3_gpu_tensor_read_bf16_range`）。qkv 探针传 `{0, 7168, 14336}` 却按行号打
   `q[0]/k[0]/v[0]` 标签，实际读到的是交错布局 slot 的行 1 列 1792 与行 2 列 3584，
   标签是错的（q[0] 恰好对是因为元素 0 就是行 0）。按行探应传
   `{0, 56*128*5376, 112*128*5376}`。
2. Python 侧复算要**再旋转一次**：文件里存的是 `R(W)`，引擎会再做一次 R
   得回 `W`（R 是对合）。qkv 还要先按 `dst=(3*(s%56)+s//56)*128+dim` 复现置换。

## 四个模型的体积构成（同一分桶口径实测，GiB）

| | AdaLN | MLP | Attention | token_refiner | 其它 | 合计 |
|---|---|---|---|---|---|---|
| h3c-official BF16 | 24.288 | 21.533 | 14.355 | 1.436 | 0.116 | **61.73** |
| 同架构 int8（推算） | 12.144 | 10.767 | 8.972 | 0.718 | 0.116 | **32.72** |
| ConvRot int8 pruned（实测） | **0.081** | 10.773 | 8.979 | 0.718 | 0.057 | **20.61** |
| h3c-q4-native 4bit | 24.288 | 6.056 | 4.037 | 1.436 | 0.116 | **35.93** |
| mere-run q4 | **0** | 6.056 | 4.037 | 0.404 | 0.057 | **10.55** |

**int8 的 22 GB 不是 int8 更省，是模型被剪了 12 GiB**：AdaLN 从 `[96768,2688]` BF16
（24.288 GiB）变成 `[96768,8]` F16（0.072 GiB，336× 压缩）。`config.json` 的
`num_layers:50 / hidden_size:5376 / ffn_hidden_size:14336` 全与发布版一致 ——
剪的是**条件注入路径**，不是层数或宽度。
h3c-q4-native 与 mere-run q4 的 MLP/Attention 字节**完全相同**（6.056 / 4.037），
35.93−10.55 = 25.38 GiB 的差额就是 AdaLN 24.288 + token_refiner 1.03。

## 已产出的可加载模型目录

- `/Volumes/data/MODELS/h3c-official` — 发布版 BF16 参考，FL2VA DiT 13 shards / 61.73 GiB。
- `/Volumes/data/MODELS/h3c-q4-native` — 由 mere-run q4 转换来的 4-bit，
  18 shards / 935 tensors / **35.93 GiB**，由 `fastvideo_qad/scripts/export_h3_q4_native.py`
  生成（`--source q4` 二次量化，对 BF16 的 relRMS mean 0.1318；`--source bf16` 约 0.091）。
  `token_refiner.*` 保持 BF16（文本编码器硬要求）。
- **4-bit 的精度上限是格式本身**：q4 文件相对发布版 BF16 就已经是
  mean 0.096 / max 0.104，二次量化只额外加 3.6 个点。想更准得上 6-bit / 更小 group。

