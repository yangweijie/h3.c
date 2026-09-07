# 时间维 Fast-Mode 设计备忘录

> 状态:设计草稿(无代码改动)  
> 范围:MiniMax-H3 DiT 推理的时间轴加速  
> 日期:2026-09-07

## 1. 动机

现有 fast-mode 沿三个轴裁剪:

| 轴 | 参数 | 机制 | 当前档位 |
|---|---|---|---|
| 步间 | `denoise_reuse` | 跳 Euler 步 | 1/2/3 |
| 核心 | `core_reuse` | 隔 N 步重算 transformer | 1/4/6 |
| 空间 | `token_reduction` | 相邻视频 token 配对 | 0/1 |

视频是**时间序列**,相邻帧的潜在表示高度相关。当前实现把每帧独立送入 DiT,浪费了帧间冗余。本设计补齐**第四轴:时间维 fast-mode**。

## 2. 核心观察

### 2.1 帧间相关性结构

在 Euler 去噪过程中:
- **早期步**(高噪声):帧间相关性低(噪声主导),每帧需独立计算
- **后期步**(低噪声):帧间相关性高(信号主导),相邻帧的 Q/K/V 近似

### 2.2 关键帧 vs 中间帧

MiniMax-H3 的 temporal attention 覆盖全部 `video_t` 帧。若只对**关键帧子集**做完整 DiT,中间帧通过**时间传播**复用关键帧的 K/V 或输出,可省大量计算。

### 2.3 与现有机制正交

时间维 fast-mode 独立于 denoise_reuse / core_reuse / token_reduction,可叠加。

## 3. 设计方案

### 3.1 时间稀疏计算(Temporal Sparse Computation)

**思路**:每 N 个时间 token 只算 1 个,其余通过插值或注意力复用。

```
时间轴:  [f0] [f1] [f2] [f3] [f4] [f5] [f6] [f7]
计算:     ✔    ✗    ✗    ✔    ✗    ✗    ✗    ✔    (temporal_sparse = 3)
```

**实现**:
- 在 DiT block 的 temporal attention 中,对 Q 做 mask:只保留关键帧位置的 Q
- K/V 仍用全部帧(保证信息完整),但 Q 只计算关键帧位置
- 输出:关键帧位置写完整值,中间帧位置通过**时间线性插值**填充

**Metal 内核修改**:
```metal
// 当前:所有时间位置都参与 attention
// 新增:根据 temporal_sparse_mask[time_idx] 决定是否计算 Q
if (temporal_sparse_mask[q_time_idx]) {
    q = compute_q(k, v, k_norm, ...);
} else {
    q = 0; // 后续插值
}
```

### 3.2 时间 K/V 复用(Temporal K/V Reuse)

**思路**:相邻帧的 K/V 在低噪声步高度相似,可隔帧复用。

```
步数:    t0    t1    t2    t3    t4    t5
帧:      f0    f0    f0    f0    f0    f0
         f1    f0    f1    f0    f1    f0    (奇数帧复用偶数帧 K/V)
         f2    f2    f2    f2    f2    f2
         f3    f2    f3    f2    f3    f2
```

**实现**:
- 新增 `temporal_kv_reuse` 参数(1 = 关闭, 2 = 每 2 帧复用, 3 = 每 3 帧复用)
- 在 `h3_dit_block` 的 attention 计算前,根据 `temporal_kv_reuse` 把 K/V 复制到复用位置
- 仅对 `core_reuse > 1` 的步生效(这些步本身就在省计算)

### 3.3 渐进式时间分辨率(Temporal Resolution Schedule)

**思路**:早期步用低时间分辨率(帧合并),后期步恢复全分辨率。

```
步数:    0-4    5-9    10-14   15-19
分辨率:  t/4    t/2    t/2     t
```

**实现**:
- 在 `h3_dit_schedule` 中维护 `active_temporal_resolution`
- 早期步:把相邻 4 帧的 token 平均合并,计算后再上采样回原始分辨率
- 后期步:恢复正常 temporal attention

## 4. 参数设计

```c
typedef struct {
    // ... 现有字段 ...

    /* Temporal fast-mode: compute 1 of every N temporal tokens, interpolate
     * the rest. 1 is exact, 2 is fast, 3 is aggressive. Orthogonal to
     * denoise_reuse / core_reuse / token_reduction. */
    int temporal_sparse;

    /* Reuse K/V from adjacent frames every N denoiser steps. 1 is exact,
     * 2 reuses every other frame, 3 reuses every third. Only active when
     * core_reuse > 1. */
    int temporal_kv_reuse;

    /* Temporal resolution schedule: start at 1/N resolution and ramp to
     * full by the final denoising step. 1 is exact, 2 is half, 4 is
     * quarter. */
    int temporal_resolution_schedule;
} h3_params;
```

### 4.1 推荐档位

| 模式 | temporal_sparse | temporal_kv_reuse | temporal_resolution | 预估加速 | 质量影响 |
|---|---|---|---|---|---|
| exact | 1 | 1 | 1 | 1x | 基线 |
| fast | 2 | 2 | 2 | ~1.5x | 极低 |
| aggressive | 3 | 3 | 4 | ~2.5x | 中等 |
| turbo | 3 | 3 | 4 | ~3x | 较高(需验证) |

## 5. 与现有 fast-mode 的叠加

时间维 fast-mode 与现有机制正交,可叠加:

```
总加速 ≈ denoise_reuse × core_reuse × token_reduction × temporal_sparse
```

**示例**(fast 模式):
- denoise_reuse = 2 (2x)
- core_reuse = 4 (1.5x)
- token_reduction = 1 (1.3x)
- temporal_sparse = 2 (1.5x)
- **总加速 ≈ 5.8x**

## 6. 实现优先级

### Phase 1:时间 K/V 复用(低风险,易实现)
- 修改 `h3_dit_block` 的 attention 路径
- 新增 `temporal_kv_reuse` 参数
- 仅在 `core_reuse > 1` 时生效

### Phase 2:时间稀疏计算(中等风险)
- 新增 temporal attention mask
- 实现时间插值内核
- 新增 `temporal_sparse` 参数

### Phase 3:渐进式时间分辨率(高风险,需验证)
- 修改 schedule 逻辑
- 实现帧合并/上采样内核
- 新增 `temporal_resolution_schedule` 参数

## 7. 风险与缓解

| 风险 | 影响 | 缓解 |
|---|---|---|
| 时间插值引入闪烁 | 视觉质量下降 | 仅在低噪声步使用;后期步恢复全分辨率 |
| K/V 复用导致运动模糊 | 快速运动场景质量下降 | 检测运动强度,动态调整复用率 |
| Metal 内核复杂度增加 | 维护成本 | 每个阶段独立验证,保持 exact 路径不变 |
| 与 VDN 线性分支冲突 | 未知 | 先验证 base H3,再扩展到 VDN |

## 8. 验证计划

1. **单元测试**:构造已知时间序列,验证插值/复用精度
2. **质量测试**:对比 exact vs fast 模式的 FID/VGGSIM
3. **速度测试**:测量各档位在 M4 Pro 上的实际加速比
4. **消融实验**:单独关闭每个时间维机制,量化贡献

## 9. 参考

- `denoise_reuse` 实现:`h3.c:2085` `h3_dit_denoise_euler_preview`
- `core_reuse` 实现:`h3_dit.c:2318` `core_reuse_interval`
- `token_reduction` 实现:`h3_dit.c:455` `configure_token_reduction`
- temporal attention:`h3_dit.c` temporal attention 路径
- VDN 线性分支:`vdn.md` 中的线性注意力设计

---

*本文档为设计草稿,实现前需进一步验证。*
