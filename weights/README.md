# weights/ — 4B 编码器候选向量与 32B 对照指引

## 交付物

`cand_4b_final.npz`：MiniMax-H3「4B 文本编码器 (Qwen3-VL-4B int8-convrot, 前 25 层) + ClipProj 投影」候选向量。

- 每个 key = prompt 原文，value = `[seq, 5120]` float64
- `seq` = token 数；`[0]` 是注意力 sink token（ClipProj 用 `sink_out` 覆盖，恒等于元数据 sink_out）
- 语义参考用最后一个 token `[-1]`（真实编码）
- 10 prompts 全部 finite、无 NaN；与 fp16 原版对照平均余弦 0.9982

## Prompts 清单（与 npz 完全一致）

```
a cat
a dog
an elephant
quantum physics
a vintage sports car
world peace
war and peace
hello world
close-up of a red fox in snowy forest
cinematic neon city street at night
```

## 4B 侧复现命令（本机 16GB Mac 可直接跑）

```bash
cd /Volumes/data/git/c/h3.c
python3.13 clipproj_harness.py \
  --prompts "a cat,a dog,an elephant,quantum physics,a vintage sports car,world peace,war and peace,hello world,close-up of a red fox in snowy forest,cinematic neon city street at night" \
  --inline --max-prompts 10 \
  --model-dir /Users/jay/.lmstudio/models/Qwen3-VL-4B-Instruct \
  --projection /Users/jay/.lmstudio/models/ClipProj-MiniMax-H3/mmh3-4b-ClipProj-v3-mlp.safetensors \
  --weights-file /Users/jay/.lmstudio/models/Qwen3-VL-4B-Instruct-int8-convrot/qwen3vl_4b_int8_convrot.safetensors \
  --max-layers 25 --out weights/cand_4b_final.npz
```

## 32B 侧对照（需 ≥128GB 内存机器）

> 本机 16GB 无法加载 32B 文本编码器（37 GiB 权重；h3.c README 目标机为 M5 Max/128 GiB），
> 因此本仓库以网络测评数据为基准：ClipProj 元数据 `cos_test=0.8144`（作者 200-prompt
> 4B-vs-32B 平均余弦）。下面的步骤供后续在 32B 机器上做实证对照。

### 1. 32B 侧产出 `ref32b.npz`

h3.c 的 32B 文本编码入口（输出即 `[tokens, 5120]` BF16，无需 ClipProj）：

```c
#include "h3_text_encoder.h"
/* h3_text_encode_bf16(weight_dir, "h3_shaders.metal", token_ids, token_count,
                        progress, NULL, &embedding, err, sizeof err);
 * embedding.values = BF16 [embedding.tokens, 5120] */
```

参照 `tests/test_real_prompt.c`。导出为 npz（key=prompt 原文，value=[seq,5120] float32）：

```c
/* 用 numpy 的 C API 或经 fwrite 落盘后由 python 读入:
 *   data_f32 = [bf16_to_f32(v) for v in embedding.values]  -> (tokens, 5120)
 * python 侧 np.savez('ref32b.npz', **{prompt: arr})        // 与 4B 侧同 key */
```

权重目录：Hugging Face `MiniMaxAI/MiniMax-H3` snapshot（README 顶层 `./MiniMax-H3`）。

### 2. 一次跑出对照

```bash
cd /Volumes/data/git/c/h3.c
python3.13 clipproj_harness.py \
  --prompts "a cat,a dog,an elephant,quantum physics,a vintage sports car,world peace,war and peace,hello world,close-up of a red fox in snowy forest,cinematic neon city street at night" \
  --inline --max-prompts 10 \
  --model-dir /Users/jay/.lmstudio/models/Qwen3-VL-4B-Instruct \
  --projection /Users/jay/.lmstudio/models/ClipProj-MiniMax-H3/mmh3-4b-ClipProj-v3-mlp.safetensors \
  --weights-file /Users/jay/.lmstudio/models/Qwen3-VL-4B-Instruct-int8-convrot/qwen3vl_4b_int8_convrot.safetensors \
  --max-layers 25 \
  --reference /path/to/ref32b.npz --out /tmp/cand_vs_32b.npz
```

harness 会打印每个 prompt 的 `cosine_vs_32B` 及汇总 `mean/min/max/frac>=0.7`。
**目标：mean 接近 `cos_test=0.8144`**（作者 200-prompt 均值；单 prompt 可高于或低于 0.7，
`frac>=0.7` 反映多数 prompt 相关性）。

### 3. 已知精度链（为何 0.8144 是合理目标）

| 环节 | 余弦 | 来源 |
|---|---|---|
| int8 反量化 4B vs 原版 fp16 4B（编码器层） | ≈0.998 | 本机实测（layer0 权重 cos≈1.00；10 prompts 最终向量 mean 0.9982） |
| bf16 校准矩阵用于 int8 编码器 | 差 0.0023 | ClipProj 模型卡 |
| 4B+ClipProj vs 32B | **0.8144** | ClipProj 元数据 `cos_test`（200 prompts） |
