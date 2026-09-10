#!/usr/bin/env python3
"""Validate the fixed group-meeting paper-note format."""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from urllib.parse import urlparse

REQUIRED_SECTIONS = [
    (1, "论文定位"),
    (2, "核心思想"),
    (3, "详细方法论"),
    (4, "实验分析"),
    (5, "优势与局限"),
    (6, "组会讲法"),
    (7, "独立逐字稿"),
    (8, "问答准备"),
]

MIN_SECTION_CHARS = {
    1: 100,
    2: 150,
    3: 200,
    4: 200,
    5: 150,
    6: 180,
    7: 300,
    8: 150,
}

PLACEHOLDER_RE = re.compile(r"\b(?:TODO|TBD)\b|待补|placeholder", re.IGNORECASE)
LINK_RE = re.compile(r"!?\[[^\]]*\]\(([^)]+)\)")
H2_RE = re.compile(r"^## ([1-8])\. (.+)$", re.MULTILINE)
H1_RE = re.compile(r"^# (?!#)(.+)$", re.MULTILINE)
QUESTION_RE = re.compile(r"^### Q([1-3])：(.+)$", re.MULTILINE)
ITERATIVE_METHOD_RE = re.compile(r"树搜索|迭代|循环|搜索|beam search|tree search", re.IGNORECASE)
FEEDBACK_RE = re.compile(r"反馈|回报|奖励|评价|得分|feedback|reward", re.IGNORECASE)
STOP_RE = re.compile(r"停止|终止|预算|上限|次数|收敛|stop|terminat|budget|converg", re.IGNORECASE)
ABLATION_VARIANT_RE = re.compile(
    r"^(?:w/o\b|without\b|no\b|无|去掉|移除|关闭|删除|只保留|仅保留|替换|.*(?:换成|替代|合并))",
    re.IGNORECASE,
)
TABLE_SEPARATOR_CELL_RE = re.compile(r"^:?-{3,}:?$")


def section_bodies(text: str, matches: list[re.Match[str]]) -> dict[int, str]:
    result: dict[int, str] = {}
    for index, match in enumerate(matches):
        end = matches[index + 1].start() if index + 1 < len(matches) else len(text)
        result[int(match.group(1))] = text[match.end():end].strip()
    return result


def check_local_links(path: Path, text: str) -> list[str]:
    errors: list[str] = []
    for raw_target in LINK_RE.findall(text):
        target = raw_target.strip().strip("<>")
        parsed = urlparse(target)
        if parsed.scheme or target.startswith("#"):
            continue
        target = target.split("#", 1)[0]
        if not target:
            continue
        resolved = (path.parent / target).resolve()
        if not resolved.exists():
            errors.append(f"broken local link: {raw_target}")
    return errors


def split_table_row(line: str) -> list[str]:
    """Split a simple Markdown table row while preserving escaped pipes."""
    stripped = line.strip()
    if stripped.startswith("|"):
        stripped = stripped[1:]
    if stripped.endswith("|") and not stripped.endswith(r"\|"):
        stripped = stripped[:-1]
    return [cell.strip() for cell in re.split(r"(?<!\\)\|", stripped)]


def markdown_table_issues(text: str) -> list[str]:
    """Catch missing separator rows and collapsed columns after table transcription."""
    issues: list[str] = []
    lines = text.splitlines()
    index = 0
    while index < len(lines):
        if not lines[index].lstrip().startswith("|"):
            index += 1
            continue

        start = index
        block: list[str] = []
        while index < len(lines) and lines[index].lstrip().startswith("|"):
            block.append(lines[index])
            index += 1

        if len(block) < 2:
            issues.append(f"Markdown table at line {start + 1} has no separator row")
            continue

        header = split_table_row(block[0])
        separator = split_table_row(block[1])
        if len(header) != len(separator) or not all(
            TABLE_SEPARATOR_CELL_RE.fullmatch(cell) for cell in separator
        ):
            issues.append(f"Markdown table at line {start + 1} has an invalid separator row")
            continue

        for offset, row in enumerate(block[2:], start=2):
            if len(split_table_row(row)) != len(header):
                issues.append(
                    f"Markdown table row at line {start + offset + 1} has a collapsed or extra column"
                )
    return issues


def count_ablation_variants(section: str) -> int:
    count = 0
    for line in section.splitlines():
        if not line.lstrip().startswith("|"):
            continue
        cells = split_table_row(line)
        if cells and ABLATION_VARIANT_RE.search(cells[0].strip("* `")):
            count += 1
    return count


def validate(path: Path) -> dict[str, object]:
    errors: list[str] = []
    warnings: list[str] = []

    if not path.is_file():
        return {"path": str(path), "errors": ["file does not exist"], "warnings": []}

    text = path.read_text(encoding="utf-8")
    first_nonempty = next((line.strip() for line in text.splitlines() if line.strip()), "")

    if first_nonempty == "---":
        errors.append("YAML frontmatter is not allowed in the group-meeting format")

    h1_matches = H1_RE.findall(text)
    if len(h1_matches) != 1:
        errors.append(f"expected exactly one H1 title, found {len(h1_matches)}")

    h2_matches = list(H2_RE.finditer(text))
    observed = [(int(m.group(1)), m.group(2).strip()) for m in h2_matches]
    if len(observed) != 8:
        errors.append(f"expected 8 numbered H2 sections, found {len(observed)}")
    else:
        for (number, expected), (actual_number, actual_title) in zip(
            REQUIRED_SECTIONS, observed
        ):
            if actual_number != number or not actual_title.startswith(expected):
                errors.append(
                    f"section {number} must start with '{expected}', got "
                    f"'{actual_number}. {actual_title}'"
                )

    if "> **事实源**" not in text:
        errors.append("missing top-level fact-source block")

    for label in ("论文事实", "作者主张", "汇报者评价"):
        if f"**[{label}]**" not in text:
            errors.append(f"missing evidence label: {label}")

    if PLACEHOLDER_RE.search(text):
        errors.append("placeholder text remains")

    if text.count("$$") % 2:
        errors.append("unbalanced display-math fences")

    errors.extend(markdown_table_issues(text))

    bodies = section_bodies(text, h2_matches)
    for number, minimum in MIN_SECTION_CHARS.items():
        body = bodies.get(number, "")
        if len(body) < minimum:
            errors.append(
                f"section {number} is too short: {len(body)} chars, minimum {minimum}"
            )

    section_one = bodies.get(1, "")
    if "|---" not in section_one.replace(" ", ""):
        errors.append("section 1 must contain a compact metadata table")
    if "一句话贡献" not in section_one:
        errors.append("section 1 must contain a one-sentence contribution")

    section_four = bodies.get(4, "")
    if not re.search(r"\d", section_four):
        warnings.append("section 4 contains no numeric evidence")
    if "|---" not in section_four.replace(" ", ""):
        warnings.append("section 4 contains no Markdown comparison table")

    section_three = bodies.get(3, "")
    paper_is_benchmark = bool(re.search(r"\bBenchmark\b|基准论文|基准数据集", section_one, re.IGNORECASE))
    if not paper_is_benchmark and ITERATIVE_METHOD_RE.search(section_three):
        if not FEEDBACK_RE.search(section_three):
            warnings.append("iterative or search method does not explain its feedback signal")
        if not STOP_RE.search(section_three):
            warnings.append("iterative or search method does not explain its stopping condition or budget")

    ablation_variants = count_ablation_variants(section_four)
    if ablation_variants >= 5:
        if not re.search(r"完整数据流|整体闭环|完整流程|机制流程|组件职责", section_three):
            warnings.append(
                "large ablation study is not anchored to a complete method flow or component ledger"
            )
        if not re.search(r"实际(?:删除内容|改动)", section_four):
            warnings.append(
                "large ablation table should include an '实际删除内容' or '实际改动' column"
            )

    section_five = bodies.get(5, "")
    if not re.search(r"不能|未|缺少|局限|风险|下降|低于|不等于", section_five):
        errors.append("section 5 does not state a concrete limitation or boundary")

    section_six = bodies.get(6, "")
    for marker in ("推荐叙事", "推荐原论文配图", "容易讲混", "承上启下"):
        if marker not in section_six:
            errors.append(f"section 6 missing subsection or marker: {marker}")

    questions = list(QUESTION_RE.finditer(text))
    question_numbers = [int(m.group(1)) for m in questions]
    if question_numbers != [1, 2, 3]:
        errors.append(f"expected Q1, Q2, Q3 exactly once, found {question_numbers}")
    else:
        for index, match in enumerate(questions):
            end = questions[index + 1].start() if index + 1 < len(questions) else len(text)
            answer = text[match.end():end].strip()
            if len(answer) < 30:
                errors.append(f"Q{match.group(1)} answer is too short")

    errors.extend(check_local_links(path, text))

    if not path.name.endswith("_论文总结.md"):
        warnings.append("filename does not follow <ShortTitle>_论文总结.md")

    return {"path": str(path), "errors": errors, "warnings": warnings}


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Validate eight-section Chinese group-meeting paper notes."
    )
    parser.add_argument("paths", nargs="+", type=Path)
    parser.add_argument("--json", action="store_true", dest="as_json")
    args = parser.parse_args()

    results = [validate(path) for path in args.paths]
    passed = all(not result["errors"] for result in results)
    payload = {
        "status": "passed" if passed else "failed",
        "files": results,
    }

    if args.as_json:
        print(json.dumps(payload, ensure_ascii=False, indent=2))
    else:
        for result in results:
            print(result["path"])
            for error in result["errors"]:
                print(f"  ERROR: {error}")
            for warning in result["warnings"]:
                print(f"  WARNING: {warning}")
            if not result["errors"] and not result["warnings"]:
                print("  OK")
        print(f"status: {payload['status']}")

    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
