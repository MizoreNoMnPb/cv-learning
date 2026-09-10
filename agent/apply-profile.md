# Apply fork-profile

本文件是步骤。Forker 只改根目录 `fork-profile.md`。执行本文件时，先读 profile 的 YAML，再读 Intent。缺字段则停止，不要猜。

对照表：[`path-glossary.md`](path-glossary.md)。单篇深挖：[`deepdive.md`](deepdive.md)。不可覆盖的库规：根目录 `AGENTS.md`。

---

## 0. 完成标准（全程有效）

一次 apply 结束时必须同时成立：

- 工作方式等于 `fork-profile.md` 里的 `paths` / `markdown_language` / `knowledge_unit` / `notes`。
- PDF 一个都没删。
- 每个被改过的 Markdown 相对链接都能解析。
- `reorganize: false` 时，`01`–`04` 的 PDF 与已有串讲 **没有被移动或重写**。
- 生成笔记的页眉含 profile 的 `notes.header`。

`AGENTS.md` 的收录边界（无代码、无部署脚本、证据分层）不被 profile 放宽。

## 0.1 硬停（先于选分支）

出现下列任一情况则停止，不改树、不写笔记：

1. 当前仓库仍是本库的 `master` 或 `zh`（未 fork，或 fork 后仍跟上游目录合同），**并且** `reorganize: true`。改树是 fork 的事。
2. 当前分支是 `zh`。`zh` 只由 `agent/generate-zh.py` 从 `master` 生成。
3. 走 Notes 分支且 `interest` 为空列表。要求 forker 列出 glossary id，不要给全库写笔记。

---

## 1. 读 profile

1. 解析根目录 `fork-profile.md` 的 YAML。`profile_version` 必须为 `1`。
2. 记下：`paths`、`markdown_language`、`knowledge_unit`、`notes.output`、`notes.commit`、`reorganize`、`interest`、Intent 正文。
3. 枚举值不在注释所列集合里 → 停止并列出合法值。

---

## 2. 选分支（只走一条）

按下面的表选 **恰好一个** 分支，然后只执行该分支。先过 §0.1。

| 条件 | 分支 |
|---|---|
| `reorganize: false` 且 `knowledge_unit` 为 `family-synthesis` | **Filter**：按 `interest` 给阅读清单，不写新树 |
| `reorganize: false` 且 `knowledge_unit` 为 `per-paper` 或 `both` | **Notes**：按 `deepdive.md` 把笔记写到 `notes.output` |
| `reorganize: false` 且 `paths` 与磁盘不一致 | **Index**：生成 `view/index.md`，中文标题链到现有英文路径；不改名 |
| `reorganize: true` | **Reorganize**：先写计划，再等明确执行 |

`interest` 非空时，所有分支都只处理这些 id（必须能在 `path-glossary.md` 里查到）。查不到 → 停止并说本库没有该族，不要编。

---

## 3. 分支 Filter

1. 打开 [learning-path.md](../learning-path.md) 与两个 `problem-setting-and-overview.md`。
2. 列出 forker 该读的串讲路径（当前分支上的目录名）和中文标题。
3. 空族写「尚无串讲 / 尚无 PDF」，不生成假串讲。

完成：清单覆盖 `interest`（或全库第一波）中每一个 id。

---

## 4. 分支 Notes

1. 若 `interest` 为空 → 停（见 §0.1）。
2. 目标目录：`scratch/` 或 PDF 同目录，以 `notes.output` 为准。
3. 每个目标 PDF 一篇 md。已有合格串讲则笔记里 **链到串讲**，不复制成第二套理论。
4. `notes.commit: false` 时路径必须被 `.gitignore` 覆盖。
5. 页眉写入 `notes.header`。正文语言 = `markdown_language`。

完成：`interest` 内每个目标 PDF 都有对应笔记文件，或被显式跳过并写了原因（例如无 PDF）。

---

## 5. 分支 Index

1. 新建 `view/index.md`（可 gitignore）。
2. 用 glossary 的 Chinese 列做标题，href 指向磁盘上的现有路径。
3. 不创建中文文件夹，不复制 PDF。

完成：`interest` 范围内每个公开 md 在 `view/index.md` 有且仅有一条链接，且链接可打开。

---

## 6. 分支 Reorganize

### 6.1 计划（未得到「执行计划」之前停在这里）

写 `agent/plan.md`，表格列为：`from`、`to`、动作（`rename` / `generate-note` / `skip`）、原因。

- `paths: english-academic`：每个路径对到 glossary 的 English 列。
- `paths: chinese-academic`：每个路径对到 glossary 的 zh-dir 列。
- 论文 PDF 已是 `paper.pdf` 则 `skip`。
- 不删 PDF。不把 `scratch/` 笔记升格为公开串讲。

完成（计划阶段）：`agent/plan.md` 存在，且每一行 `to` 都能在 glossary 里解释。

### 6.2 执行（仅当用户明确说执行该计划）

1. 按 `agent/plan.md` 逐行 rename/move。
2. 重写所有 Markdown 相对链接。
3. 公开 md 的 **正文语言** 转为 `markdown_language` 仅当用户在 Intent 里要求翻译；默认 **不翻译** 已有中文串讲。
4. 删除 `agent/plan.md` 或把状态标为 `applied`。

完成：计划中的每一行都已执行；随机抽 10 条相对链接可解析。

---

## 7. 禁止用 profile 做的事

把这些写成失败，而不是「尽量避免」：

- 用 Intent 覆盖 YAML 枚举（Intent 只补充兴趣与强调，不改字段）。
- 为了中文路径创建 git symlink。
- 把 DeepSpeed 等 `99-Inbox` 条目写进第一波清单，除非 Intent 明确只要系统论文。
- 在 `reorganize: false` 时移动 `01`–`04`。
- 在本库 `zh` 上改树或把人手提交写进 `zh`。
