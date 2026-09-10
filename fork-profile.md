---
# fork-profile.md is the only file a forker edits to change layout or generation.
# Invariants that protect the library live in AGENTS.md and cannot be relaxed here.
profile_version: 1

# Path and file names. Markdown *body* language is independent.
# english-academic = glossary English column (Title-Case)
# chinese-academic = glossary zh-dir column
paths: english-academic          # english-academic | chinese-academic
markdown_language: zh            # zh | en

# What counts as a knowledge unit in *your* working copy.
# family-synthesis = one md threads a method family or a foundation topic
# per-paper        = one md beside each PDF
# both             = keep family md as library; generate per-paper as notes
knowledge_unit: family-synthesis

notes:
  output: scratch                # scratch | alongside-pdf
  commit: false                  # false = gitignore; true = you own the notes in your fork
  header: "[agent-generated/unverified]"

# false: agent only filters, indexes, or writes notes. It does not move the tree.
# true:  agent may rename/move after writing agent/plan.md and getting an explicit go.
reorganize: false

# Optional interest filter. Empty = whole library for Filter / Index.
# Notes with empty interest must stop (do not deepdive the whole library).
# Examples: [vlm-ad, memory-prototype, dann]
interest: []
---

# Intent（给 agent 的自由文本）

本仓库作者的格式，也是 fork 的默认值：

- 分类目录用英文学术术语 Title-Case（对照 `agent/path-glossary.md` 的 English 列）。
- Markdown **正文**用中文。
- 公开知识单元是 **族 / 主题串讲**：一篇 md 穿一条 AD/DA 方法族，或一块基础领域（CNN、Faster R-CNN→DETR）。
- 单篇论文精读不是成品。需要时由 `agent/deepdive.md` 写到 `scratch/`。
- 论文目录是查找键 + `paper.pdf` + `sources.md`，见 `agent/paper-unit.md`。

本文件只给 **已经 fork、且不再跟本仓库目录合同走** 的人改。clone `master` / `zh` 的读者不要改本文件，也不要跑 `agent/apply-profile.md` 改树。

Forker 若要改格式：只改本文件顶部的字段和本节 Intent，然后让 agent 执行 `agent/apply-profile.md`。不要直接口述「帮我改成中文目录」而不改本文件——下次会话会丢。

示例（每篇论文一篇中文 md，并允许改树）：

```yaml
paths: chinese-academic
markdown_language: zh
knowledge_unit: per-paper
notes:
  output: alongside-pdf
  commit: true
reorganize: true
```
