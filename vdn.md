# 如何 支持 https://github.com/OpenVDN/vdn-minimax-h3

## 支持vdn
当前 h3.c（h3-metal）**尚不支持** VDN-Minimax-H3。两者虽然共享 MiniMax-H3 的底座权重，但 VDN-H3 在架构上做了重要扩展。下面是差距分析和可行的集成路径。

---

### VDN-H3 相比基础 H3 增加了什么

| 组件 | 说明 | h3.c 现状 |
|---|---|---|
| **线性注意力分支** | 每个 DiT block 附加一个 frame-wise linear attention 分支，与 softmax 分支并行 | ❌ 不存在。当前只有标准 softmax attention |
| **LoRA 适配器 (default)** | 对 QKV 和 O 投影加 LoRA（50 步蒸馏） | ❌ 无 LoRA 加载/合并机制 |
| **LoRA 适配器 (turbo)** | 在 default 之上再加 turbo LoRA（8 步蒸馏） | ❌ 同上 |
| **权重目录结构** | `stage-b-step-2000/` 和 `stage-dmd-step-250/`，内含 `linear_branch/` + `adapters/` | ❌ 只识别 `h3-base/` 布局 |

### 集成所需的关键修改

**1. DiT Block 结构扩展** — [h3_dit.c](h3_dit.c#L47-L64) 中的 `h3_dit_block` 只有 `qkv`/`out`/`fc1`/`fc2`，需新增：
- `linear_qkv`、`linear_out`（线性注意力分支的投影矩阵）
- 对应的 int8 量化版本（可选）

**2. 线性注意力 Metal 内核** — [h3_shaders.metal](h3_shaders.metal) 需新增：
- Frame-wise linear attention 的 GPU 内核（不需要 softmax，复杂度 O(n)）
- 与现有 softmax 分支的结果合并逻辑

**3. LoRA 权重加载与合并** — [h3_weights.h](h3_weights.h) 需扩展：
- 支持从 `adapters/default/` 和 `adapters/turbo/` 读取 LoRA 权重
- 在加载时将 LoRA 合并到 base 权重中（`W_merged = W_base + α * W_lora`），VDN-H3 的设计正是"plug-and-play"，推理时合并不修改 backbone

**4. 权重目录识别** — [h3_safetensors.c](h3_safetensors.c) 需支持 VDN 的多阶段目录结构：
```
ckpts/
  h3-base/              ← h3.c 已支持
  stage-b-step-2000/    ← 新增：linear_branch + adapters/default
  stage-dmd-step-250/   ← 新增：adapters/turbo
```

**5. 调度适配** — VDN-H3 使用不同的 shift 参数（8-step turbo 用 video shift=6, audio shift=3），[h3_dit_schedule.c](h3_dit_schedule.c) 需支持这些配置。

### 好消息

VDN-H3 的设计哲学是 **plug-and-play**：线性注意力分支和 LoRA 可以在推理时合并进 backbone 权重，不改变前向传播的调用接口。这意味着：
- 合并后的模型仍然是标准 DiT 前向，**不需要在推理循环中切换分支**
- 但线性注意力分支本身的前向计算是新增的，不能简单跳过

### 实际操作建议

如果目标只是用 VDN-H3 的 **turbo LoRA 做 8 步快速推理**，最务实的路径是：

1. **在 Python 端合并权重**：用 VDN-H3 的 Python 代码将 LoRA 合并到 base，导出为 safetensors
2. **在 h3.c 端直接加载合并后的权重**：利用现有的 [h3_safetensors.c](h3_safetensors.c) 加载流程
3. **线性注意力分支暂不实现**：先只走 softmax 路径，质量会有一定损失但能跑通

如果要**完整支持**（含线性注意力分支以获得 VDN-H3 的质量优势），则需要上述全部 5 项修改，工作量不小。


## 流式

**能。** 现有的 SSD 流式机制可以自然扩展到完整 VDN-H3，内存增幅很小。

---

### 现有流式机制回顾

当前 h3.c 的 SSD 流式逻辑（[h3_dit.c](h3_dit.c#L100-L104)）：

| 机制 | 实现 |
|---|---|
| 仅保留 2 个 block 的 GPU 槽位 | `stream_slots[2]`，双缓冲交替 |
| 常驻内存只留 norm 权重 | `load_block_norms()` 加载 `norm1`/`norm2`/`q_norm`/`k_norm`，每块约 20 KiB |
| 大矩阵按需从 SSD 读入 | `STREAM_QKV`/`STREAM_OUT`/`STREAM_FC1`/`STREAM_FC2`，4 个矩阵 per block |
| 后台 pthread 预读下一层 | GPU 跑当前 block 时，`read_stream_layer_thread` 读下一个 block |

效果：DiT 显存从 **~36.5 GiB 降到 ~2.0 GiB**（512²），代价是 26–84% 的速度损失。

### VDN-H3 扩展对流式的影响

VDN-H3 每个 block 新增两类权重：

**① LoRA 适配器（default + turbo）**

VDN-H3 的设计是 plug-and-play：LoRA 在推理时合并进 backbone（`W_merged = W_base + α·A·B`），合并后张量尺寸与 base 完全一致。因此：
- **LoRA 不增加每块流式体积**，只需在首次加载时多读几个小矩阵做合并
- 合并操作可以发生在 host 端，再写入临时 safetensors，或不合并、在流式读取后即时融合

**② 线性注意力分支**

每个 block 额外有 `linear_qkv` + `linear_out`（frame-wise linear attention），尺寸大致是：
- `linear_qkv`: `[INNER × 3, HIDDEN]` → 与 softmax QKV 同形
- `linear_out`: `[HIDDEN, INNER]` → 与 softmax OUT 同形

流式扩展非常直接：

```c
// 当前
enum { STREAM_QKV, STREAM_OUT, STREAM_FC1, STREAM_FC2, STREAM_MATRICES };

// 扩展后
enum { STREAM_QKV, STREAM_OUT, STREAM_FC1, STREAM_FC2,
       STREAM_LIN_QKV, STREAM_LIN_OUT,               // 新增
       STREAM_MATRICES };                              // 4 → 6
```

对应修改：

| 修改点 | 内容 |
|---|---|
| `h3_dit_stream_layer.sources[]` | 扩为 6 个 source，加入 `linear_qkv`/`linear_out` 的路径和偏移 |
| `h3_dit_block stream_slots[2]` | 每个槽位多分配 2 个 BF16 tensor（`linear_qkv`/`linear_out`） |
| `prepare_stream_layer()` | 追加 `SOURCE(4, "linear_attn.qkv_proj.weight", ...)` 等 |
| `stream_slot_target()` | 映射 `STREAM_LIN_QKV` → `slot->linear_qkv` |
| `run_refiner_block()` | 在 softmax attention 后、MLP 前插入线性注意力前向 |

### 内存估算

| 组件 | 全驻留（50 blocks） | SSD 流式（2 slots） |
|---|---|---|
| Softmax block (QKV+OUT+FC1+FC2) | ~36.5 GiB | ~1.46 GiB |
| 线性分支 block (LIN_QKV+LIN_OUT) | ~14.4 GiB 估算 | ~0.58 GiB |
| Norm 常驻（所有 blocks） | ~1.0 MiB | ~1.0 MiB |
| **合计 DiT** | **~51 GiB** | **~2.1 GiB** |

线性注意力分支没有 FC1/FC2（它共享主 MLP），所以实际增量更小。**流式模式下总 DiT 驻留约 2.0–2.5 GiB**，与当前 H3 base 几乎持平。

### 速度代价

线性注意力是 frame-wise（O(n) 而非 O(n²)），计算开销远小于 softmax 分支。瓶颈仍在 SSD 读取带宽——每步多读约 40% 的数据，预计流式开销从当前的 26–84% 上升至约 35–100%，具体取决于分辨率和 SSD 速度。

---

### 总结

| 问题 | 答案 |
|---|---|
| 流式加载是否可行？ | ✅ 完全可行，现有双缓冲架构天然兼容 |
| 内存省多少？ | 50 blocks → 2 slots，DiT 从 ~51 GiB 降到 ~2.1 GiB |
| 需要改什么？ | 枚举扩展、stream source 补充、槽位多分 2 个 tensor、前向插入线性分支 |
| 有没有坑？ | LoRA 合并时机需设计：建议 host 端预合并写临时文件，避免流式路径中做矩阵运算 |
