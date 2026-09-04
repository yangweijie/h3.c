# 性能分析与诊断开关

引擎保留了大量**用于精确 A/B 的环境变量**。它们不是给用户调的，而是在同一进程内把某条融合路径关掉、回到"近参考"的两内核实现，从而定位数值或性能回归。

## --profile 报告什么

`--profile` 逐 Metal 阶段报告六个维度：

| 维度 | 说明 |
|---|---|
| wall time | 阶段墙钟时间 |
| command encode | CPU 侧命令编码时间 |
| wait | **完整的 commit-to-fence 周转** |
| root GPU timestamp | 根命令缓冲的 GPU 时间戳 |
| peak live / cumulative alloc | 峰值活跃张量存储、累计分配量 |
| dispatch counts | 直接 / MPS linear / conv / sdpa / blit 各类 dispatch 数 |

关于时间口径有一条重要说明：

> wait 测量的是完整的命令周转；**根 GPU 时间戳单独看会漏掉 MPSGraph 内部调度的子缓冲**，因此被相应标注。

Sources: [README.md](README.md#L758-L765), [h3_gpu.h](h3_gpu.h#L28-L32)

## 阶段名

进度回调里的 phase 字符串即剖析单元：

```
tokenizer, text encoder, Qwen vision, video VAE encoder, audio VAE encoder,
text encoder (clipproj), preview VAE load, video VAE load, audio VAE, FFmpeg
+ DiT 透传的 phase
```

Sources: [h3.c](h3.c#L679-L710)

## 诊断开关总表

### DiT 融合

| 环境变量 | 关闭的对象 |
|---|---|
| `H3_DISABLE_FUSED_MLP=1` | 融合的 `fc1→SwiGLU→fc2` 图，回到分离操作边界 |
| `H3_DISABLE_FUSED_GATE_ADALN=1` | 注意力残差门控 + 后续 MLP AdaLN 的融合 |
| `H3_DISABLE_FUSED_CROSS_BLOCK_ADALN=1` | 跨块的 MLP 门控 → 下一块注意力 AdaLN |
| `H3_DISABLE_FUSED_FINAL_SLICE=1` | 最终 AdaLN 直接绑偏移，回到拷贝 + AdaLN |
| `H3_DISABLE_FUSED_FINAL_HEAD=1` | 最终头加载瓦片时顺带施加 AdaLN |
| `H3_DIT_F32_FINAL=1` | 恢复近参考的 F32 输出头 |
| `H3_DISABLE_FUSED_PATCH_CAST=1` | patch 投影的融合舍入 |
| `H3_SCALAR_PATCH=1` | 标量诊断路径 |
| `H3_DISABLE_FUSED_PATCH_PACK=1` | patch 输出直接绑进打包隐藏流 |
| `H3_DISABLE_DIT_ACTIVATION_ALIAS=1` | 激活缓冲的生命周期别名复用 |
| `H3_DIT_COMMAND_BLOCKS=0` / `1..50` | 命令缓冲拆分；0 恢复单个 |

Sources: [README.md](README.md#L502-L519), [README.md](README.md#L683-L697), [README.md](README.md#L767-L770)

### int8 / Metal 4

| 环境变量 | 作用 |
|---|---|
| `H3_NAX=0` | 完全禁用 TensorOps |
| `H3_NAX=1` | 强制更宽的原生 BF16 线性（微基准更快，完整 forward 目前更慢） |
| `H3_NAX=mlp` | 专门的 Metal 4 MLP 路径 |
| `H3_DISABLE_NAX_MLP=1` | 在此方式创建的上下文里保留 MPSGraph MLP，用于同进程 A/B |
| `H3_DISABLE_GRAPH_DATA_CACHE=1` | 恢复所有张量的瞬态 MPSGraph 包装器 |
| `H3_REUSE_MPS_COMMAND=0` / `1` | 覆盖 MPSCommandBuffer 复用的自动选择 |

对应的 CLI 回退开关见 [int8 与 TensorOps 路径](int8-tensorops)。

Sources: [README.md](README.md#L634-L652), [README.md](README.md#L698-L709)

### Token 缩减与首块缓存

| 环境变量 | 作用 |
|---|---|
| `H3_TOKEN_REDUCTION_BLOCKS` | 覆盖后续的 `4:30` 区间 |
| `H3_TOKEN_REDUCTION_EARLY=STEPS:END` | 覆盖早期调度；`0` 禁用 |
| `H3_DISABLE_TOKEN_REDUCTION=1` | 上下文内的精确对照 |
| `H3_DISABLE_FUSED_TOKEN_POOL_ADALN=1` | 恢复双内核入口边界 |
| `H3_DISABLE_FUSED_TOKEN_ADALN=1` | 恢复双内核出口边界 |
| `H3_FB_CACHE=1` | 启用首块缓存（默认关） |
| `H3_FB_CACHE_THRESHOLD` | 相对 L1 阈值，默认 0.12 |

Sources: [README.md](README.md#L548-L553), [README.md](README.md#L563-L575)

### 权重与编码

| 环境变量 | 作用 |
|---|---|
| `H3_ZERO_COPY_WEIGHTS=0` | 关闭 M5 的 zero-copy 权重映射 |
| `H3_QWEN_PREFETCH=0` / `1..8` | 文本编码器预取；0 恢复单层同步，1–8 指定线程数 |
| `H3_QWEN_PREFETCH_DEPTH=1..6` | 覆盖预取环深度（M3/老硬件默认 2，M5 默认 3） |
| `H3_CLIPPROJ_DIR=0` / `off` | 回退到 50 层编码器 |
| `H3_VAE_TILE_PIXELS=256` | 恢复保守的视频 VAE 瓦片计划 |
| `H3_DEBUG_CONVROT` | 首块 `STREAM_OUT` 的 ConvRot 调试输出 |

Sources: [README.md](README.md#L600-L612), [h3_dit.c](h3_dit.c#L1105-L1106)

### 采样器

| 环境变量 | 作用 |
|---|---|
| `H3_CPU_SAMPLER=1` | 在 M5 上恢复 CPU 采样器 |
| `H3_GPU_SAMPLER=1` | 显式选择 GPU 状态路径 |
| `H3_GPU_SAMPLER_WINDOW=0` | 较慢的无界提前编码诊断模式 |

Sources: [README.md](README.md#L716-L719)

## 一个诊断流程示例

怀疑某个融合改变了数值：

```sh
# 1) 基线（近参考）
H3_DISABLE_FUSED_GATE_ADALN=1 H3_DISABLE_FUSED_MLP=1 \
  ./h3 --profile -d ./MiniMax-H3 -p "..." --seed 42 \
       --width 512 --height 512 --frames 22 --steps 20 --layers 50 --reuse 1 \
       -o /tmp/base.mp4

# 2) 默认（全部融合）
./h3 --profile -d ./MiniMax-H3 -p "..." --seed 42 \
       --width 512 --height 512 --frames 22 --steps 20 --layers 50 --reuse 1 \
       -o /tmp/fused.mp4
```

同种子、同分辨率、同帧数、同步数是比较的前提。README 也提醒：

> 这类负载对热节流敏感，比较时应使用重复运行，并在机器预热期间交替变体。

Sources: [README.md](README.md#L99-L102)

## 需要注意的默认行为

`ssd-streaming` 不是默认档位，`token_reduction` 也不是；`memory_plan_auto` 默认**开启**，会在用户未显式指定流式/int8 时替用户做选择。做 A/B 时若想完全确定路径，应显式指定 `ssd_streaming` 与 `use_int8_row_fc2`（任一非零即禁用自动规划）。

Sources: [h3.c](h3.c#L886-L888), [README.md](README.md#L156-L158)
