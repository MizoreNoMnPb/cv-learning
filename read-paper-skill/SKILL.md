---
name: read-paper-for-group-meeting
description: Read a paper deeply and produce an evidence-grounded Chinese group-meeting note with a fixed eight-section structure, presentation guidance, a timed talk script, and Q&A preparation. Use when the user gives a paper title, DOI, arXiv ID, URL, local PDF, or existing paper note and asks for 论文精读、论文总结、组会汇报、组会讲稿、逐字稿、方法与实验分析、问答准备, or asks to rewrite notes into the established group-meeting format. For multiple papers, create one note per paper unless the user explicitly requests a combined survey.
---

# Read Paper for Group Meeting

Produce a research-grade reading note that can be used directly for a Chinese lab meeting. Include a complete Chinese translation of the paper's abstract, then reconstruct the evidence chain beyond the abstract.

## Workflow

Follow this order for each paper:

1. Resolve the exact paper identity and publication type.
2. Acquire the strongest available source, preferring a local PDF.
3. Read the complete extractable paper, including appendix material needed for methods, settings, and limitations.
4. Build an evidence plan before drafting.
5. Draft the eight-section note.
6. Validate structure and evidence hygiene.
7. Perform a final quality review, then a readability review.
8. Save one Markdown file per paper in the requested directory.

If the user provides multiple papers, process them independently. Do not merge them into one file unless the user explicitly asks for a combined comparison or survey.

## Source and Evidence Rules

Use sources in this order:

1. User-provided local PDF.
2. Existing canonical raw text, source manifest, note plan, or verified deep note for the same paper.
3. Zotero attachment when available.
4. DOI, publisher, arXiv, or another open full-text source.

Fail closed when the full paper or equivalent evidence is unavailable. Do not present an abstract-only rewrite as a completed note.

Translate the abstract only from the complete, verified abstract in the paper or canonical full-text source. Do not translate a truncated search-result snippet or metadata excerpt as though it were the full abstract.

At the top of the note, state the fact source and evidence convention:

~~~markdown
> **事实源**：
> - path/to/paper.pdf
> - path/to/evidence/
>
> 论文页数、抽取材料和核验范围。以下用 **[论文事实]**、**[作者主张]**、**[汇报者评价]** 区分证据层级。
~~~

Use the labels as follows:

- **[论文事实]**: Directly supported by the paper's text, formula, table, figure, appendix, or verified metadata.
- **[作者主张]**: The authors' interpretation, positioning, causal explanation, or broad claim.
- **[汇报者评价]**: Your evidence-based synthesis, criticism, comparison, or boundary judgment.

Do not label every sentence mechanically. Use labels where readers could otherwise confuse evidence with interpretation.

## Evidence Plan

Before writing, identify:

- paper type and evidence ceiling;
- problem and task definition;
- a complete mechanism inventory covering model or Agent components, intermediate states and representations, search or scheduling, memory, rule-based guardrails, tools, and environment execution feedback;
- datasets or source materials;
- baselines and budget fairness;
- metrics and their exact operational meaning;
- main quantitative results;
- ablations and mechanism-result mapping;
- negative, limiting, or non-leading results;
- latency, token, API, GPU, human, or deployment costs;
- what the paper proves and does not prove;
- three questions likely to arise in a group meeting.

For Demo papers, treat architecture and cases as existence evidence, not comparative evidence. For Benchmark papers, emphasize task construction, ground truth, metrics, judge reliability, and leakage. For method or system papers, emphasize the executable mechanism, intermediate state, feedback loop, and ablations. For training papers, explain the objective, data, reward, trajectory, and optimization assumptions.

## Output Contract

Use this exact top-level structure. Do not add YAML frontmatter.

~~~markdown
# 论文简称：一句中文定位

> **事实源**：...

## 1. 论文定位
### 摘要完整翻译
## 2. 核心思想
## 3. 详细方法论
## 4. 实验分析
## 5. 优势与局限
## 6. 组会讲法
## 7. 独立逐字稿（约 2 分钟）
## 8. 问答准备
~~~

Default filename: ShortTitle_论文总结.md

### 1. 论文定位

Use a compact table with:

- title;
- venue and DOI or arXiv ID;
- paper type;
- task;
- data or system scope;
- research category;
- one-sentence contribution.

Immediately state the evidence ceiling. Distinguish Research, Benchmark, Demo, Industry, and survey evidence.

After the metadata table and evidence ceiling, add `### 摘要完整翻译`. Translate the complete original abstract into Chinese, preserving its sentence order, scope, uncertainty, comparisons, numbers, and technical terms. Do not summarize, omit sentences, merge away qualifications, strengthen claims, or insert commentary into the translation. Keep author claims as author claims rather than rewriting them as independently verified conclusions.

If the canonical paper genuinely has no abstract, write `原文未提供可核验的摘要` under this heading and do not fabricate one. A complete abstract translation is source-aligned content; it does not replace sections 2--5 or raise the paper's evidence ceiling.

### 2. 核心思想

Normally use:

~~~markdown
### 2.1 关键问题
### 2.2 主要洞察
### 2.3 与已有工作的区别
~~~

Explain why the problem exists, what conceptual move the paper makes, and what is genuinely different from nearby routes. Do not use generic praise such as "improves accuracy and efficiency."

### 3. 详细方法论

Reconstruct the actual flow:

~~~text
input -> state or intermediate representation -> decision -> tool or execution -> feedback -> output
~~~

Before drafting prose, build a component ledger from the full paper and appendix. Separate:

- model or Agent components;
- intermediate states and representations;
- search, routing, planning, or scheduling mechanisms;
- memory and history windows;
- deterministic rules, validators, and safety guardrails;
- tools, environment execution, and external feedback.

Do not treat only named Agents as methodology. A tree search, memory window, rule checker, replay executor, ranker, or stopping policy is also a method component when it changes system behavior or appears in an experiment.

For every iterative system, explain the complete loop:

1. input and output;
2. prompt fields or state fields;
3. candidate generation;
4. candidate validation, filtering, and ranking;
5. tool or environment execution;
6. feedback and memory or state update;
7. stopping condition, iteration budget, or convergence rule.

Use one term for each component across methods, experiments, script, and Q&A. Every component named by an ablation, replacement, or merge experiment must already be defined here, including what it consumes, changes, and produces.

Cover roles only when they correspond to distinct information, state, or control responsibilities. Explain central formulas with variable meanings, assumptions, and engineering consequences. Use standard Markdown math delimiters.

For Benchmark papers, adapt this section to task construction, ground truth, evaluation protocol, judge or simulator, and leakage controls.

### 4. 实验分析

Start with datasets, models, baselines, budgets, and metrics. Then present the central comparison.

Use a compact Markdown table when comparing three or more models, datasets, settings, metrics, or ablations. Explain the table immediately afterward.

Apply these rules:

- Bind every number to a task, metric, table, figure, or setting.
- Distinguish percent from percentage points.
- Do not average incompatible metrics unless the paper defines the aggregation.
- Do not treat missing baseline results as wins.
- Do not combine maximum gains from different datasets into one simultaneous claim.
- Report negative results and cases where another method wins.
- Extract every row from the paper's ablation table before summarizing it. Do not silently select only favorable or familiar rows; if a row is omitted, state why.
- Map every `w/o`, removal, replacement, merge, or budget variant to a mechanism already explained in section 3.
- For each ablation, state what was actually removed or changed, how the result changed, and which claim that change supports or weakens.
- Group large ablation tables by mechanism and include an `实际删除内容` or `实际改动` column. Use `不适用` for genuinely undefined metrics instead of collapsing cells or leaving ambiguous blanks.
- Recheck every numeric row against the original table or figure after Markdown transcription, especially when copied source text has lost column boundaries.
- Compute relative improvement as `(method - baseline) / baseline` unless the paper defines another formula. Name the denominator and distinguish relative percent from percentage-point change.
- Separate final quality from efficiency, cost, and safety.

### 5. 优势与局限

Separate evidence-backed strengths from limitations, costs, and risks.

Always check:

- semantic correctness versus execution success;
- model-judge or simulator dependence;
- baseline strength and budget fairness;
- external validity across data, models, systems, and users;
- token, GPU, API, latency, replay, or human cost;
- memory contamination, leakage, sandbox, permission, and rollback risk;
- reproducibility gaps or inconsistent settings.

State what the paper does not prove.

### 6. 组会讲法

Include all four subsections:

~~~markdown
### 推荐叙事顺序（约 2 分钟）
### 推荐原论文配图和表格
### 容易讲混的概念
### 承上启下句
~~~

Choose a 2-minute script by default. Use about 4 minutes only for a central paper whose mechanism or training objective must be explained in depth.

Recommend only figures and tables that carry the argument. Say what to point at and what not to infer.

The transition sentence must place the paper in the larger narrative rather than merely announce the next title.

### 7. 独立逐字稿

Rewrite the note as spoken Chinese:

1. problem and paper type;
2. central mechanism;
3. two or three decisive results;
4. one important boundary or counterexample;
5. one takeaway and transition.

Do not read section headings, enumerate every module, or stack unexplained numbers. Keep the script consistent with the evidence analysis.

### 8. 问答准备

Provide three default questions:

- one about mechanism or terminology;
- one about metrics, comparison, or attribution;
- one about evidence boundary, cost, safety, or external validity.

Answer directly. Do not add facts absent from the paper.

## Writing Style

Write for researchers and engineers, not a general audience.

- Prefer natural Chinese; retain stable paper, model, dataset, database, and metric names when needed.
- Explain jargon at first use.
- Use formulas only when they change how the method or result should be understood.
- Prefer mechanism-level criticism over generic limitation lists.
- Keep the note self-contained.
- Translate the abstract completely and faithfully; keep analysis, criticism, and supplementary explanation outside the translation subsection.

## Figures

Do not embed every extracted image. In the default group-meeting note, recommend the strongest original figures or tables in section 6.

Embed images only when the user asks for an illustrated note or when an image is necessary to understand the mechanism. Verify every local path and explain the evidence carried by the figure.

## Validation

Run the bundled validator after drafting:

~~~bash
python scripts/validate_group_note.py path/to/Paper_论文总结.md
~~~

Run it on every output file for a multi-paper request. Fix all errors before final delivery.

After structural validation, perform these reviews:

1. **Final quality review**: compare the translated abstract sentence by sentence with the complete source abstract, then verify the central evidence chain, key settings and numbers, strong alternatives, mechanistic limitations, proven/unproven separation, and reusable takeaway. Confirm that section 3 covers every mechanism named by an ablation; every ablation explains what changed, the result change, and the supported claim; method and experiment terminology match one-to-one; large ablation tables contain every source row or an explicit omission reason; and all derived percentages use a stated denominator.
2. **Final readability review**: reread the complete note for natural Chinese, table interpretation, formula placement, spoken rhythm, and consistency between analysis, script, and Q&A.

If either review changes the note, rerun the validator.

## Completion Criteria

Treat the task as complete only when:

- the full paper or equivalent canonical evidence was read;
- one Markdown file exists per paper;
- all eight sections are present in order;
- fact source and three evidence levels are explicit;
- the complete source abstract is faithfully translated under `摘要完整翻译`, or the canonical paper's lack of an abstract is stated explicitly;
- central settings, numbers, ablations, negative results, and limitations are included;
- methodology components completely cover the ablation, replacement, merge, and budget variants reported in the experiments;
- the group-meeting guidance, timed script, and three Q&A items are complete;
- the validator passes;
- final quality and readability reviews pass.
