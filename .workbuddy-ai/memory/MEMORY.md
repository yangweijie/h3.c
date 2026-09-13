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
