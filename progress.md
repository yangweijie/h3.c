# Progress Log: MiniMax-H3 on 16GB Mac（含 4B+ClipProj harness）

## 2026-08-29 — 4B+ClipProj harness（验证用）
### Done
- 仓库获取：GitCode 镜像需认证弃用；ModelScope `NicoLab28/ClipProj-MiniMax-H3` 下载成功，SHA256 校验通过。
- 投影元数据：`tap=24, d_in=2560, d_out=5120, mlp_hidden=32768, mlp_depth=1, 无 W, cos_test=0.8144`。
- 模型：`Qwen3-VL-4B-Instruct`（8.3GB，`d_in=2560` 确认 4B）；python3.13 + torch2.12 / transformers5.15 / numpy / scipy。
- 编写 `clipproj_harness.py`（PyTorch transformers）：加载 `Qwen3VLModel` + 投影；`--hs-index` 默认 `tap+1=25`。
- Phase 4 实跑产出：`candidate.npz` 已生成（key `hello`, shape `(1, 5120)`, float64）。用户确认 **Phase 4 已用 MLX 跑通 4B+ClipProj 产出 [seq,5120]**；但 MLX 脚本尚未补入 h3.c 仓库（待补）。

### Blocked / Errors
- 早期在本机（16GB，~8GB 可用）以 PyTorch CPU fp16 加载 8.3GB 模型被 jetsam SIGKILL(137)。MLX 路径已规避。
- 层索引未校准：需读 ComfyUI `sd1_clip.py` 确认 `sm.layer=[24]` → HF `hidden_states` 索引（24 或 25）。

---

## 2026-08-29 (2) — 32B 文本编码耗时拆解 + 5 方向优化

### 关键发现：22.5 分钟是什么
- 这 22.5 分钟只是**单次文本编码**（一个 prompt 的 32B 前向，50 层）——视频生成流水线的**第一步**，不是采样步。DiT 采样 + VAE 还没走，DiT 权重（58GB）还没下载。
- 时间构成：GPU 计算 1.9s（Metal 快）；等待磁盘 ~1345s（从 USB 盘流式读 46.9GB 权重 @ 35MB/s）。不是"模型加载慢"，而是每层都从磁盘流式读（16GB 放不下 62GB，必须逐层读+算+丢）。磁盘读总量 46.9GB → 物理上 ~1340s。

### 对整条视频生成流水线的预估（用户）
- 标准 20 步：22min 文本 + 20×~28min/步 ≈ 9.5 小时
- LoRA 4 步：22min + 4×28min ≈ 2.1 小时
- Turbo LoRA 核心价值：采样步 20→4，时间砍 5 倍。文本编码每轮只做 1 次（22min 可接受），瓶颈在 DiT 步数 × 磁盘读取。

### 5 方向分析（用户）
1. **h3.c 支持量化/低精度权重？** 当前不支持（`h3_weights.c:150` load_tensor 校验 dtype，只走 bf16/f32；GGUF/int8 不兼容）。可行：加载时反量化到 BF16（~1-2 天）。收益：FP16 62→31GB；int8 62→15.5GB（16GB 可整体驻留，彻底摆脱磁盘流式）。
2. **文本编码必须 FL2VA 32B？** 目前必须。`DiT` 条件 `TEXT_DIM=5120` 硬编码（h3_dit.c:16）。32B 直出 [seq,5120]；4B 出 [seq,2560] 需 ClipProj 升到 5120（harness 已验证）。h3.c 无外部条件注入接口——需加"读 npz 替换 text encoder 输出"接口。
3. **文本编码独立走 MLX / LM Studio？** MLX ✅（Phase 4 已跑通 4B+ClipProj 产出 [seq,5120]，但需补 h3.c 注入接口）；LM Studio/llama.cpp ❌（不输出中间层 hidden states，且 Qwen3-VL 非 embedding 模型）；MLX 32B ❌（16GB 跑不下 64GB）。
4. **模型存系统盘 + 软链**：内置盘仅 1.4GB 空闲，需先腾 ~65GB；Library 占 57G 可挖，但先做方向 5 更划算。
5. **测速程序已交付**（`/Volumes/data/git/c/disk_speed.c`，不在 h3.c 仓库）。实测外置盘 WRITE 33.1MiB/s（写也这么慢）→ ioreg 发现链路有 "USB2 Hub"（480Mbps），磁盘极可能挂 USB2.0 通道。建议换 Type-C 口/直连主机测：若挂 USB3，35→800MB/s+，提速 ~23 倍，磁盘问题直接解决。

### MLX 文本编码估算（本任务所求：对比 32B-in-h3.c 路径）
基线（用户数据）：32B 文本编码 = 22.5min = 1350s，其中磁盘读 46.9GB@35MB/s=1345s，GPU 1.9s；内存必须流式（62GB>16GB）。
4B+ClipProj 走 MLX：
- 权重：4B fp16≈8.3GB，int4≈2.3GB，int8≈4.2GB（约 32B 的 1/7.5）。
- 冷启（USB2 35MB/s）：fp16 读 8.3GB≈237s(~4min)；int4 读 2.3GB≈66s(~1.1min)；GPU 计算 <1s。
- 热启（4B 可驻留 16GB，加载后不再读盘）：仅计算 ~0.2–1s。
- **速度**：冷启 22.5min→~4min(fp16)/~1min(int4) ≈ 5–20x；热启 → 秒级 ≈ 700–2700x（32B 每次仍须重读 46.9GB）。
- **内存**：32B 路径必须流式 62GB；4B fp16 8.3GB 直接驻留 16GB → **彻底摆脱文本编码的磁盘依赖**；int4 仅 2.3GB 余量充足。
- 范围提醒：文本编码仅占整条流水线一步（每次生成 1 次）。20 步全流水线中省 22min≈4%；LoRA 4 步中≈17%。真正大头是 DiT 步数×磁盘读。但 4B+ClipProj/MLX 的价值在：(a) 产出与 DiT 期望一致的 [seq,5120]（harness 已验证），可经"外部条件注入接口"替换 32B；(b) 消除文本编码磁盘依赖，把 USB 带宽让给 DiT 流式；(c) 避免加载 62GB 32B 这一最大内存/磁盘负担。
- 等价路径：方向 1 的 int8 32B（15.5GB 驻留）也能消除 32B 文本编码磁盘流式；但 4B+ClipProj 更轻（8.3GB vs 15.5GB）且复用已验证 harness。
- 以上为估算，需实跑 MLX 计时 + 峰值 RSS 确认。

### Pending / Next
- [ ] 用 `disk_speed.c` 换 Type-C 口实测，确认 USB2 vs USB3 瓶颈（最高杠杆）
- [ ] 下载 DiT 权重（58GB）前置：先在快盘/快口上做
- [ ] 主干合并到 `feature/lora-merge`，跑 `--lora` DiT 数值/流程验证
- [ ] h3.c 加"外部条件向量注入"接口（读 npz 替换 text encoder 输出）
- [ ] 落地 4B+ClipProj/MLX 文本编码 offload（Phase 4 已产 [seq,5120]，脚本待补入仓库）
- [ ] 评估 h3.c int8 权重加载器（方向 1）

### Environment
- 用户建议后续用 `uv` 做可复现安装（当前 pip 已跑通）。
- 注意：`disk_speed.c` 在父目录 `/Volumes/data/git/c/`，不在 h3.c 仓库；提交需在该仓库进行。
