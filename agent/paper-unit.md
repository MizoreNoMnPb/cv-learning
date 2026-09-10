# Paper unit

一篇已归档论文在磁盘上只占一个目录。

## 布局

```text
<num>-<field>/<subfield>/[<VENUE YEAR>] <CoreIdea> - <Title>/
    paper.pdf
    sources.md
```

- `<num>-<field>`、`<subfield>`：对照 `path-glossary.md` 的 English 列（Title-Case + 缩写全大写）。
- 论文目录名：查找键，给人类扫目录用。
- `paper.pdf`：该目录内唯一 PDF，正式版本（会议/期刊优先于 arXiv）。
- `sources.md`：书目与链接。文件夹名做不到的信息放这里。

不要再套一层 `paper/`。综述也是一篇一个目录，不要把 PDF 直接堆在 `survey/` 根上。

## 目录名语法

```text
[<VENUE YEAR>] <CoreIdea> - <Title>
```

| 段 | 规则 |
|---|---|
| `VENUE` | 正式出版物缩写：`CVPR`、`ICML`、`JMLR`、`NeurIPS`、`TPAMI`。未发表才写 `arXiv`。 |
| `YEAR` | 与该版本一致。arXiv 年 ≠ 会议年时，用会议/期刊年。 |
| `CoreIdea` | 社区已用的短名（PatchCore、DANN、SHOT）。没有短名就省略，不要自造。 |
| `Title` | 官方题名，可截断到 Windows 路径安全长度；不要把题名再复制进 PDF 文件名。 |

不要写成 `( [CVPR 2022] PatchCore)`：括号套空格，查找键和题名重复。

## sources.md

每篇必有。模板：

```markdown
# <Official title>

- authors:
- venue:
- year:
- doi:
- arxiv:
- open_access:
- notes:
```

`notes` 只写对读者有用的版本信息，例如正式版与 arXiv 不是同一份。不要写方法串讲，也不要写整理过程。

链接用 Markdown URL，不要用 Windows `.url` / `.lnk`。

## 文件夹名不是书目

查找键可以编码 venue、年、短名、题名。完整作者、DOI、arXiv 与正式版的区别写在 `sources.md`。

引用论文结论时仍按 `AGENTS.md`：标题、全部作者、会议或期刊、年份、页码。页码来自 `paper.pdf`，不来自文件夹名。
