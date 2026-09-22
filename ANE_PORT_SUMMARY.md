# ANE 移植可行性总结（h3c ← 参考 h3.c-ane）

> 状态：仅做总结与规划，**未改动任何代码**。所有结论来自对 `/Volumes/data/git/c/h3.c-ane`（ANE fork）与
> `/Volumes/data/git/c/h3c`（本项目）两份代码的对比调研。
> 代码改动需经用户确认第 5 节落地顺序与前置决策后再启动。

## 0. 目标与边界

- **目标**：把 ANE（Apple Neural Engine）int8 线性层能力从 `h3.c-ane` 移植进本项目 `h3c`，用 ANE 跑
  静态形状 + conv/matmul 为主的算子来做性能优化。
- **边界（硬约束，来自 fork 实测）**：
  - ANE 只接受 **fp16 激活 + int8 常量（per-row scale）**，没有 fp32 GEMM。
  - **静态形状**：一个 shape 一套编译产物（~19GB/形状，aned 再存一份）→ 必须先收敛成少数固定档位。
  - 私有 `AppleNeuralEngine.framework` + `objc_msgSend`，任何 macOS 更新都可能断，不能进 App Store。
  - 跨界成本高（pack/unpack 走 Metal），唯一正确粒度是**整块一张图**。

## 1. fork 里已验证能跑的算子（可直接照搬）

MIL 图原语位置：`h3_ane_block.m:364-640`。

| 算子 | MIL 实现 | 坑 |
|------|----------|----|
| GEMM | conv 当 1×1 卷积（权重 `[N,C,1,1]`） | K 轴必须切块，块数 ≤8 |
| Hadamard/ConvRot 旋转 | conv（groups=k/256），权重 `[k,256,1,1]` | 分组卷积，不是真 WHT |
| int8 反量化 | `constexpr_affine_dequantize(axis=0)` | per-row scale；分组量化用不了 |
| RMSNorm | `reduce_sum` + `pow(x,-0.5)`，eps 折进去 | ANE 无 `reduce_mean`/`rsqrt` |
| RoPE | `slice`+`concat`+`mul/add` 手工拼 rotate-half | 无 rotate-half 原语 |
| Softmax attention | matmul→softmax→matmul，Q 按 512 行分块 | 不分块 score 张量 1.3GB 会 OOM |
| SwiGLU / AdaLN / gated residual | sigmoid+mul、`slice_by_size`+mul+add | — |
| fp16 溢出保护 | 2 的幂缩放残差、fc2 前后 ×1/16 与 ×16 | 指数缩放无损，必须做 |

**关键组合技巧**：整块一张图（norm→mod→qkv→head norm→rope→attn→out→gate→mlp→gate），每 block 只在
Metal↔ANE 之间跨界一次。单投影（`h3_ane_linear.m`）因每次 pack/eval/unpack 只是调试/回退路径。

**常量一致性**：fork `h3_ane_block.m:11-24` 的
`BLK_HIDDEN=5376, BLK_HEADS=56, BLK_HEAD_DIM=128, BLK_INNER=56*128, BLK_FFN=14336`
与本项目 `h3_dit.c:23-27` 完全一致 → DiT block 的 MIL 图几乎可原样搬。

> ⚠️ **文档纠错**：`AGENTS.md` 写的 `HEAD_DIM=96 / MLP=21504` 与代码不符（实际 128 / 14336）。
> 照 AGENTS.md 写 ANE 图会算错，应以 `h3_dit.c` 源码常量为准。

## 2. h3c 里还能搬的（按性价比排序）

| # | 模块 | 形状/精度 | 可行性 | 主要约束 |
|---|------|-----------|--------|----------|
| 1 | DiT 4 投影 qkv/out/fc1/fc2 | `[21504,5376]`/`[5376,7168]`/`[28672,5376]`/`[5376,14336]`，int8/bf16 | 已验证（`h3_ane_linear.h`） | 需 per-row scale int8；group=128 packed 要先转 per-row |
| 2 | 整 DiT block | rows×5376，full softmax | 已验证，收益最大 | rows 随 T/H/W 变 → 每形状一套编译（19GB/形状） |
| 3 | final head + patchify proj | `[5376→96]`/`[5376→32]`，bf16 | 高，形状固定 | 小 GEMM，收益小但顺手 |
| 4 | video VAE transformer | 纯 ViT：2048/32 heads/64 dim，ridge tile 固定像素 | 合适：TILE_PIXELS=256/512 固定 → seq 固定 | 全 fp32 内核，需 fp16 化并验证精度敏感；tile 档位收敛成少数几档 |
| 5 | vision encoder（27 层 ViT） | 1152/4304/16 heads/72 dim，full SDPA + 2D RoPE | 高，无 window attention | seq 随分辨率变 → 按档位编译；LayerNorm（MIL 有 `layer_norm`） |
| 6 | audio VAE（BigVGAN） | 纯 conv1d 转置卷积 k=9/4，空洞 1/3/5，7 级上采样 | ANE 强项（就是 conv） | fp32→fp16 音频质量风险；长度随 duration 变；conv_transpose 支持度待验 |
| 7 | text encoder（Qwen3-VL 前 50 层） | 5120/25600，GQA 64:8，MRoPE | 理论可行 | 权重 51GB 编译缓存爆炸；seq 动态 → 建议 pad 固定档位或继续走预计算 conditioning |
| 8 | ClipProj MLP | 2560→32768→5120 + GELU，现 CPU fp32 | 可行 | 一次性成本，优先级最低 |

## 3. 明确别搬的

- **VDN linear branch / chunk-window attention**：`h3_dit.c` 的 chunk=5 帧窗口 + 线性扫描，带
  `gather`/`log_alpha`/fp32 state plane，是显式扫描结构，MIL 表达不了，且 fp32 状态 ANE 没有。
- **ConvRot 的 CPU 反旋转**（`convrot_unrotate_cpu`）：只能在图内当分组 conv 做，出图就失去意义。
- **token reduction 的 merge/gather**（`row_map`/`token_pool_sources`）：逐 token 索引；另起形状→编译缓存翻倍。
- **SSD 流式权重**：fork 里 `H3_ANE_LINEARS` 与 `--ssd-streaming` 直接互斥（权重打包进 blob 后不再走流式 slot）。
- **time embedder**（`256→5376→2688` fp32）：太小，跨界开销吃掉收益。

## 4. 决定可行性的硬约束（筛选用）

1. **静态形状**：一个 shape 一套编译产物，~19GB/形状，aned 再存一份。先固定几档分辨率/时长，别让形状爆炸。
2. **只有 fp16 激活 + int8 常量**，没有 fp32 GEMM。任何 fp32 模块（video VAE、audio VAE、time embedder）
   都得先过精度关。
3. **请求输入 ≤8 个，绑定按参数名字母序**（不是声明序）——`h3_ane_bridge.h:13-16` 明确写的坑。
   K 分块因此最多 8 块。
4. **行数对齐 16，mod 平面宽 32。**
5. **wired memory**：编译后的 block ~800MB，50 个必须 rotate + 内容寻址缓存。
6. **跨界成本**：pack/unpack 走 Metal，所以「尽量整块一张图」是唯一正确粒度。
7. 私有 `AppleNeuralEngine.framework` + `objc_msgSend`，任何 macOS 更新可能断，不能进 App Store。

## 5. 建议落地顺序（需用户确认后启动）

1. **搬 `h3_ane_bridge.{h,m}` + `h3_ane_linear.{h,m}`**，用 `H3_ANE_LINEARS=1` 这类 knob 只跑 4 个投影。
   - 验证点：单 block 输出与 Metal 路径余弦 >0.99。
2. **搬 `h3_ane_block.{h,m}`**，用 `H3_ANE_FULL_BLOCKS=N` 逐块推进。
   - 验证点：2 步渲染 frame-cosine 0.99、6 步视觉一致（fork 实际数字）。
3. **形状收敛**：给 video VAE tile 和 token-reduction 后的 seq 各设少数固定档位，再做全 block。
4. 之后才考虑 4/5/6 号模块（VAE/ vision/ audio）。

## 6. 落地前需确认的两个决策（阻塞项）

- **A. 前置动作优先级**：先确认现有 int8 权重能否转成 ANE 需要的 **per-row scale** 格式
  （fork 的 `constexpr_affine_dequantize(axis=0)` 不支持 group=128 packed 格式）。
  这是第 1 步能否跑通的前提，可能比直接搬 bridge/linear 更划算先做。
- **B. 移植范围**：只搬「4 投影 + 整 block」（步骤 1–2，收益最大、风险最低），还是一上来就规划 4–8 号模块？

## 7. 风险与红线

- 不覆盖现有 `task_plan.md` / `findings.md`（属其他已完成任务）。
- 不在未确认第 5/6 节前改任何 `.c/.m/.h`。
- ANE 编译产物巨大（~19GB/形状）且依赖私有框架，需保证磁盘余量（参考历史：编译 21504×5376 图需 ~700MB+ 空闲，
  磁盘不足会 SIGBUS/InvalidMILProgram）。
- 所有 ANE 路径必须保留 Metal 对照路径 + 余弦校验 knob，以便 bisect。

---

## 8. 实测之后对本文件 above 的更正（2026-09-19，Phase B4-pre/D0 之后）

第 2、4、6 节的表格是**移植前的估算**，以下数字是 h3c 本机实测，以实测为准（细节见
`findings.md` 的 F14/F15/F16）。

| 本文件原判断 | 实测结论 |
|---|---|
| §2-1「DiT 4 投影」性价比第一 | **判死**：拆投影走 ANE 只有 1.04~1.07×（rows 1536/3072），mean 反超为慢（F15）。不接线。 |
| §2-2「整 DiT block 收益最大」 | 数值成立（S=1904 eval 334 ms，cos 0.999988），但**磁盘判死**：int8 图 **381 MiB/block-形状 ⇒ 18.6 GiB/形状**（+aned 副本），本机 13 GiB 装不下；关缓存则每形状重付 ~170 s 冷编译（F14/F15）。 |
| §2-4「video VAE 合适」 | **成立且是唯一值得做的**：真实 block0 四投影 ANE/Metal = **3.24×**（rows=1797）/2.89×（rows=512），rel_l2 ≤ 7.9e-4，fp16 溢出 0；一轮 pass 的 GEMM 3.26 s→1.01 s（F16）。 |
| §2-5「vision encoder 高」 | **出局**：27 层 × 每次生成 1 pass 且 seq 随分辨率变 ⇒ 编译无法摊销（ANE 权重是 constexpr，摊销判据 = 同权重同形状复用次数）。 |
| §2-6「audio VAE 是 ANE 强项」 | **出局**：1 pass、形状随 clip 时长变、`conv_transpose1d` 未验证、fp16 音频质量风险 —— 四重不利。 |
| §4-1「~19GB/形状」 | 量级对，且**只在 int8 图成立**。fp16 图的产物 = **1× 权重字节、与 rows 无关**：video VAE 每 block 128 MiB、36 block **4.5 GiB/形状**。 |
| §4-2「VAE 得先过精度关」 | 已过（F16），但前提要更正：**video VAE 盘上是 F32 不是 F16**（560 张量全 F32，单片 10.4 GB），所以权重舍入本身在误差预算内。 |

**形状账的关键结构事实**（决定 video VAE 能不能落地）：VAE tile 是**等宽平铺**（边界
tile 靠 `TILE_OVERLAP_MIN` 重叠而非切短），所以**一轮解码只有一个 rows 形状**；
再叠加"产物字节与 rows 无关"，⇒ 把 rows 向上取整到固定桶就能让**所有分辨率共用同一份
4.5 GiB 产物**，只多付 ≤1 个桶的算力（1797→2048 是 +14%）。

**仍未验证**（D1 的前提，别当已过）：bias 语义、真实激活分布下的 fp16 溢出、
以及 **Metal pack/unpack 的端到端账** —— F16 里 CPU 填平面在 w2 上花 41 ms，
比省下的 19.5 ms 更贵，所以 D1 的放行判据已改为「端到端 ≥2×」而不是裸 GEMM 比。

## 9. Phase D 实测结论：video VAE 接线完成但**性能判负**，默认关闭（2026-09-19，D2d）

§8 里「video VAE 合适 / 唯一值得做」这条按 D2d 的整机实测作废（数据见 `findings.md` F23）：

| 闸门里量到的 | 接进 `h3_video_vae.c` 之后 |
|---|---|
| 四投影 block 端到端 **2.43~3.24×**（rows 1797/2048，一条 pass 3.22 s→1.05 s） | 整次解码稳态 **6 block 1.00×、16 block 0.95×**；产物没留住的轮次 0.13~0.38× |
| 裸图/单 block 连跑同一形状，ANE 一直热 | 每次跨界要 `submit + waitUntilCompleted`，实测 **约 12~13 ms/次调用**的流水线空泡（6 与 16 block 两轮独立反推一致） |
| 「GEMM 占大头」 | 在 1797 行的流式解码里四个 GEMM 只占墙钟 **16%**（20 s 里约 3.2 s），理论收益上限 11%，被空泡吃光 |

正确性与鲁棒性是达标的：cosine 1.000000、rel_l2 2.2e-04~5.2e-04、max_abs ≤3e-3；
`w1` 编译失败时只停在该 block、其余回退 fp32 Metal 且输出仍正确。
所以这不是实现缺陷，而是**这台机器上量不够摊薄跨界成本**：想赢要靠大 tile
（rows 7173 ⇒ 池化平面 ~5.7 GiB + 10.4 GiB F32 权重流式读 > 本机 16 GiB 内存），
且每换一个 rows 桶要再付一份 4.5 GiB 产物（系统卷只有 ~9 GiB 空闲）。

**移植结论修正**：ANE 在本仓库三条路（DiT 拆投影 F15、整 DiT block F14/F15、video VAE F23）
全部因为同一个结构原因止步 —— 权重是 constexpr ⇒ 必须摊销，而摊销要常驻，
常驻的产物/平面与本机的磁盘与统一内存冲突，再叠加每次跨引擎边界的同步空泡。
`H3_ANE_VAE` 保留为默认 off 的诊断开关（`h3_ane_vae_decode_test` 可随时复测对拍），
不再投入「整 block 一张图 / 融合 FFN」这类改动。

## 10. DiT 逐算子剖析：为什么 ANE 只能盯着 MLP + QKV（2026-09-21，F24）

`H3_DIT_OP_PROFILE=1` 给 `run_block` 的每个算子补一次 commit+wait，于是单 kernel 成本
第一次可见（50 块 × 2 步，每标签 100 calls）：

| 档 | 每块 | MLP | QKV | SDPA | attn out | gate+AdaLN |
|---|---|---|---|---|---|---|
| 864×480（7074 rows）| 2451.6 ms | **45.2%** | 24.8% | 20.5% | 8.4% | ~1.0% |
| 576×320（3249 rows）| 921.3 ms | **51.8%** | 26.3% | 12.2% | 9.2% | ~0.5% |

拆开看：FC1 681.0 / SwiGLU 13.8 / FC2 336.9 ms（864），313.0 / 3.7 / 160.6 ms（576）。
有效算力 2.8~3.7 TFLOPS，非 TensorOps 的 BF16 路径已近本机上限。

三条对本移植的硬约束：
1. 可 offload 的池子只有 MLP + QKV（合计 70~78%）；只搬注意力的天花板是 12~21%，
   而 SDPA 份额随分辨率上升（O(M²)），越低分辨率越不值。
2. 本轮独立量到**一次纯 GPU 排空括号 = 17.89 ms**（576 同配置 81.410 s 无括号 vs
   92.182 s 有括号，602 次括号），印证 §9 里"每次跨界 ~13 ms 空泡"不是 ANE 特有的，
   而是任何 GPU↔ANE 交叉都躲不掉的同步代价。每块 4 次跨界 = 72 ms，相当于 576 一档
   MLP 份额的 15%。
3. 括号失真的闭合检查（校正后 813.6 vs 实测 814.1 ms/块，0.06%）同时说明：块外算子
   （embedding / final / velocity）合计 <0.1%，没有漏掉的第三方大头。
