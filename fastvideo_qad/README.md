# FastVideo QAD 对接 h3.c

将 MiniMax H3 ConvRot 模型转换为 FastVideo MLX INT8 格式。

## 流水线

```
ConvRot BF16 (66GB)  →  [convrot_to_diffusers.py]  →  Diffusers BF16 (66GB)  →  [FastVideo]  →  MLX INT8 (~16GB)
                              ↑                                              ↑
                         格式转换(SSD)                                   量化+清理
```

中间产物写 SSD，量化后自动删除，只保留最终 INT8。

## 使用方法

```bash
python scripts/run_qad_pipeline.py \
    --input /Users/jay/h3_sys/MiniMax-H3-Convrot/FL2VA/transformer \
    --ssd-work /Volumes/data/work/h3_qad \
    --fastvideo-root /Volumes/data/git/python/FastVideo \
    --formats int8
```

输出：
- `/Volumes/data/work/h3_qad/h3_mlx/int8/` — 最终 MLX INT8 checkpoint

## 空间需求

| 阶段 | 大小 | 位置 | 生命周期 |
|---|---|---|---|
| ConvRot 输入 | ~66 GB | 原始位置 | 永久 |
| Diffusers 中间 | ~66 GB | SSD | 量化后删除 |
| MLX INT8 输出 | ~16 GB | SSD | 永久 |

SSD 需要至少 **132 GB** 临时空间（中间 + 输出）。

## 后续：h3.c 加载

MLX INT8 checkpoint 格式与 h3.c 不兼容，需要额外转换：

```bash
# TODO: 编写 mlx_int8_to_h3.c_loader.py
```

或者直接用 FastVideo MLX 运行时推理（`mlx_fasth3.py`）。
