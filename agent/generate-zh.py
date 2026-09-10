#!/usr/bin/env python3
"""Generate Chinese classification paths from agent/path-glossary.md.

Default: preview directory moves. --check also validates classification coverage
and local Markdown links. --apply requires clean master, builds zh in an isolated
worktree, validates it, then updates zh and leaves master untouched.
"""
from __future__ import annotations

import argparse
import posixpath
import re
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import quote, unquote, urlsplit

ROOT = Path(__file__).resolve().parents[1]
SECTION_PARENT = {
    "Top-level": "",
    "Foundations": "01-Foundations",
    "Vision tasks": "02-Vision-Tasks",
    "Anomaly detection": "03-Anomaly-Detection",
    "Domain adaptation": "04-Domain-Adaptation",
}
# Angle destinations preserve spaces, brackets and parentheses in paper keys.
# Bare destinations support balanced parentheses, percent escapes and a title.
LINK_RE = re.compile(
    r'(?P<pre>!?\[(?:\\.|[^\[\]\n]|\[[^\]\n]*\])*\]\(\s*)'
    r'(?P<url><[^>\n]+>|(?:\\.|[^\s()\\]|\((?:\\.|[^()\\]|\([^()]*\))*\))+)'
    r'(?P<post>(?:\s+"[^"\n]*"|\s+\'[^\'\n]*\')?\s*\))'
)
BACKTICK_RE = re.compile(r"`([^`\n]+)`")
BANNER_RE = re.compile(
    r"^> 本分支由 `master` `[0-9a-f]+` 生成。[^\n]*\n(?:\n)?", re.M
)
# These files describe canonical master paths; only their working links change.
CANONICAL_PREFIXES = ("agent/", "read-paper-skill/")
CANONICAL_FILES = {"fork-profile.md"}


@dataclass(frozen=True)
class DirMap:
    ident: str
    english: str
    zh_dir: str
    parent: str


def die(message: str) -> None:
    raise ValueError(message)


def git(args: list[str], *, cwd: Path | None = None, check: bool = True):
    result = subprocess.run(
        ["git", *args], cwd=cwd or ROOT, capture_output=True,
        text=True, encoding="utf-8", check=False,
    )
    if check and result.returncode:
        die(f"git {' '.join(args)}: {(result.stderr or result.stdout).strip()}")
    return result


def parse_glossary(text: str) -> list[DirMap]:
    rows = []
    parent = None
    for raw in text.splitlines():
        line = raw.strip()
        if line.startswith("## "):
            section = line[3:].split("(", 1)[0].strip()
            parent = SECTION_PARENT.get(section)
            continue
        if parent is None or not line.startswith("|"):
            continue
        cells = [c.strip().strip("`") for c in line.strip("|").split("|")]
        if len(cells) < 4 or not cells[1].endswith("/"):
            continue
        ident, english, chinese = cells[:3]
        if chinese in {"", "-", "—"}:
            die(f"missing zh-dir: {english}")
        english, chinese = english.rstrip("/"), chinese.rstrip("/")
        for name in (english, chinese):
            if name in {"", ".", ".."} or re.search(r'[\\/:]', name):
                die(f"classification name must be one relative component: {name}")
        rows.append(DirMap(ident, english, chinese, parent))
    if not rows:
        die("glossary has no classification directory mappings")
    by_path = {}
    identifiers = set()
    for row in rows:
        path = english_full(row)
        if path.casefold() in by_path or row.ident in identifiers:
            die(f"duplicate classification mapping or id: {path}")
        by_path[path.casefold()] = row
        identifiers.add(row.ident)
    by_full = {english_full(row): row for row in rows}
    for row in rows:
        if row.parent and row.parent not in by_full:
            die(f"missing parent mapping: {row.parent}")
    targets = [zh_full(row, by_full).casefold() for row in rows]
    if len(targets) != len(set(targets)):
        die("multiple classifications map to the same Chinese path")
    return rows


def english_full(row: DirMap) -> str:
    return f"{row.parent}/{row.english}" if row.parent else row.english


def zh_full(row: DirMap, by_full: dict[str, DirMap]) -> str:
    if not row.parent:
        return row.zh_dir
    return f"{zh_full(by_full[row.parent], by_full)}/{row.zh_dir}"


def build_dir_renames(rows: list[DirMap]) -> list[tuple[str, str]]:
    by_full = {english_full(row): row for row in rows}
    pairs = [(english_full(row), zh_full(row, by_full)) for row in rows]
    return sorted(pairs, key=lambda p: len(p[0]), reverse=True)


def validate_classifications(rows: list[DirMap], root: Path) -> None:
    expected = {english_full(row) for row in rows}
    actual = {
        path.name for path in root.iterdir()
        if path.is_dir() and re.match(r"^(?:0[1-4]|99)-", path.name)
    }
    for parent in SECTION_PARENT.values():
        if parent and (root / parent).is_dir():
            actual.update(
                f"{parent}/{child.name}" for child in (root / parent).iterdir()
                if child.is_dir() and not child.name.startswith(".")
            )
    unknown = sorted(actual - expected)
    if unknown:
        die("classification directories missing from glossary: " + ", ".join(unknown))


def plan_renames(pairs, root: Path):
    return (
        [(src, dst) for src, dst in pairs if (root / src).is_dir()],
        [src for src, _ in pairs if not (root / src).is_dir()],
    )


def leaf_git_moves(rows, existing_english: set[str]):
    by_full = {english_full(row): row for row in rows}
    result = []
    for row in sorted(rows, key=lambda r: english_full(r).count("/")):
        if english_full(row) not in existing_english:
            continue
        parent = zh_full(by_full[row.parent], by_full) if row.parent else ""
        src = posixpath.join(parent, row.english)
        dst = posixpath.join(parent, row.zh_dir)
        if src != dst:
            result.append((src, dst))
    return result


def repository_files(root: Path, *, include_untracked: bool = False) -> list[str]:
    args = ["ls-files", "--cached", "-z"]
    if include_untracked:
        args += ["--others", "--exclude-standard"]
    return sorted({
        name for name in git(args, cwd=root).stdout.split("\0")
        if name and (root / name).is_file()
    })


def file_move_map(dir_pairs, root: Path | None = None, *, include_untracked=False):
    root = root or ROOT
    paths = set(repository_files(root, include_untracked=include_untracked))
    for filename in list(paths):
        parent = posixpath.dirname(filename)
        while parent:
            paths.add(parent)
            parent = posixpath.dirname(parent)
    pairs = sorted(dir_pairs, key=lambda p: len(p[0]), reverse=True)
    moves = {}
    for old in sorted(paths):
        moves[old] = old
        for src, dst in pairs:
            if old == src or old.startswith(src + "/"):
                moves[old] = dst + old[len(src):]
                break
    return moves


def outside_code(text: str, transform) -> str:
    """Apply a prose transform without changing fenced or inline code."""
    result = []
    fence = None
    for line in text.splitlines(keepends=True):
        marker = re.match(r"^\s{0,3}(`{3,}|~{3,})", line)
        if marker:
            run = marker.group(1)
            if fence is None:
                fence = run
            elif run[0] == fence[0] and len(run) >= len(fence):
                fence = None
            result.append(line)
        elif fence:
            result.append(line)
        else:
            # Recognize whole links before code spans inside their labels.
            tokens = re.finditer(LINK_RE.pattern + r"|(?P<code>`+[^`\n]*`+)", line)
            parts = []
            start = 0
            for token in tokens:
                parts.append(transform(line[start:token.start()]))
                value = token.group(0)
                parts.append(value if token.group("code") else transform(value))
                start = token.end()
            parts.append(transform(line[start:]))
            result.append("".join(parts))
    return "".join(result)


def local_target(raw: str, old_file: str):
    angled = raw.startswith("<") and raw.endswith(">")
    url = raw[1:-1] if angled else raw
    if url.startswith(("#", "//", "/")) or urlsplit(url).scheme:
        return None
    parts = urlsplit(url)
    if not parts.path:
        return None
    target = posixpath.normpath(
        posixpath.join(posixpath.dirname(old_file), unquote(parts.path))
    )
    if target == ".." or target.startswith("../"):
        die(f"local link escapes repository: {old_file}: {raw}")
    suffix = url[len(parts.path):]
    return target, suffix, angled, parts.path


def rewrite_links(text: str, old_file: str, moves: dict[str, str]) -> str:
    def replace(match):
        raw = match.group("url")
        info = local_target(raw, old_file)
        if info is None:
            return match.group(0)
        target, suffix, angled, original_path = info
        new_file = moves.get(old_file, old_file)
        new_target = moves.get(target, target)
        if new_file == old_file and new_target == target:
            return match.group(0)
        url = posixpath.relpath(new_target, posixpath.dirname(new_file) or ".")
        if original_path.endswith("/") and not url.endswith("/"):
            url += "/"
        if "%" in original_path or not angled:
            url = quote(url, safe="/-_.~")
        url += suffix
        if angled:
            url = "<" + url + ">"
        return match.group("pre") + url + match.group("post")
    return outside_code(text, lambda part: LINK_RE.sub(replace, part))


def rewrite_backticks(text: str, pairs, old_file=None, moves=None) -> str:
    ordered = sorted(pairs, key=lambda p: len(p[0]), reverse=True)
    def replace(match):
        inner = match.group(1)
        for src, dst in ordered:
            if inner.rstrip("/") == src or inner.startswith(src + "/"):
                return "`" + dst + inner[len(src):] + "`"
        if old_file and moves and "/" in inner:
            # Tables often name a family relative to the overview document.
            target = posixpath.normpath(posixpath.join(posixpath.dirname(old_file), inner))
            if target in moves:
                new_file = moves.get(old_file, old_file)
                relative = posixpath.relpath(moves[target], posixpath.dirname(new_file) or ".")
                if inner.endswith("/"):
                    relative += "/"
                return "`" + relative + "`"
        return match.group(0)
    lines = []
    fence = None
    for line in text.splitlines(keepends=True):
        marker = re.match(r"^\s*(`{3,}|~{3,})", line)
        if marker:
            run = marker.group(1)
            if fence is None:
                fence = run
            elif run[0] == fence[0] and len(run) >= len(fence):
                fence = None
            lines.append(line)
        else:
            lines.append(line if fence else BACKTICK_RE.sub(replace, line))
    return "".join(lines)


def check_links(root: Path, files: list[str]) -> list[str]:
    errors = []
    for filename in files:
        if not filename.endswith(".md"):
            continue
        def inspect(part):
            for match in LINK_RE.finditer(part):
                info = local_target(match.group("url"), filename)
                if info and not (root / info[0]).exists():
                    errors.append(f"{filename}: {match.group('url')}")
            return part
        outside_code((root / filename).read_text(encoding="utf-8"), inspect)
    return errors


def rewrite_markdown_tree(moves, pairs, root: Path) -> None:
    for old, new in moves.items():
        if not old.endswith(".md") or not (root / new).is_file():
            continue
        path = root / new
        original = path.read_text(encoding="utf-8")
        updated = rewrite_links(original, old, moves)
        if not old.startswith(CANONICAL_PREFIXES) and old not in CANONICAL_FILES:
            updated = rewrite_backticks(updated, pairs, old, moves)
        if updated != original:
            path.write_text(updated, encoding="utf-8", newline="\n")


def insert_banner(root: Path, sha: str):
    path = root / "README.md"
    text = BANNER_RE.sub("", path.read_text(encoding="utf-8"))
    first, separator, rest = text.partition("\n")
    banner = f"> 本分支由 `master` `{sha}` 生成。内容修改请在主干完成。\n\n"
    if first.startswith("# "):
        text = first + "\n\n" + banner + rest.lstrip("\n")
    else:
        text = banner + text
    path.write_text(text, encoding="utf-8", newline="\n")


def git_mv(src, dst, root: Path):
    if src.casefold() == dst.casefold() and src != dst:
        temporary = src + ".__rename__"
        if (root / temporary).exists():
            die(f"temporary rename path already exists: {temporary}")
        git(["mv", src, temporary], cwd=root)
        git(["mv", temporary, dst], cwd=root)
    else:
        git(["mv", src, dst], cwd=root)


def require_master_clean(root: Path) -> str:
    branch = git(["branch", "--show-current"], cwd=root).stdout.strip()
    if branch != "master":
        die(f"run on master, current branch is {branch or '(detached)'}")
    if git(["status", "--porcelain"], cwd=root).stdout.strip():
        die("master must be clean; commit the intended changes first")
    if "branch refs/heads/zh" in git(["worktree", "list", "--porcelain"], cwd=root).stdout.splitlines():
        die("zh is checked out in another worktree")
    return git(["rev-parse", "HEAD"], cwd=root).stdout.strip()


def apply(pairs, git_moves, sha, root: Path) -> str:
    previous = git(["rev-parse", "--verify", "refs/heads/zh"], cwd=root, check=False)
    old_sha = previous.stdout.strip() if previous.returncode == 0 else "0" * len(sha)
    # This directory contains only a generated worktree, never the user's notes.
    with tempfile.TemporaryDirectory(prefix="cv-learning-zh-") as temporary:
        generated = Path(temporary) / "tree"
        added = False
        try:
            git(["worktree", "add", "--detach", str(generated), sha], cwd=root)
            added = True
            moves = file_move_map(pairs, generated)
            for src, dst in git_moves:
                git_mv(src, dst, generated)
            rewrite_markdown_tree(moves, pairs, generated)
            insert_banner(generated, sha)
            errors = check_links(generated, repository_files(generated))
            if errors:
                die("generated links are broken:\n" + "\n".join(errors))
            git(["add", "-A"], cwd=generated)
            git(["commit", "-m", f"regen(zh): from master {sha}"], cwd=generated)
            result = git(["rev-parse", "HEAD"], cwd=generated).stdout.strip()
            git(["update-ref", "refs/heads/zh", result, old_sha], cwd=root)
            return result
        finally:
            if added:
                git(["worktree", "remove", "--force", str(generated)], cwd=root)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="validate local links and directory coverage")
    parser.add_argument("--apply", action="store_true", help="generate and validate zh in an isolated worktree")
    args = parser.parse_args(argv)
    rows = parse_glossary((ROOT / "agent/path-glossary.md").read_text(encoding="utf-8"))
    validate_classifications(rows, ROOT)
    pairs, missing = plan_renames(build_dir_renames(rows), ROOT)
    moves = leaf_git_moves(rows, {src for src, _ in pairs})
    print("classification directory map:")
    for src, dst in sorted(pairs):
        print(f"  {src}/ -> {dst}/")
    for src in missing:
        print(f"  absent, not created: {src}/")
    if args.check or args.apply:
        errors = check_links(ROOT, repository_files(ROOT, include_untracked=True))
        if errors:
            die("broken local links:\n" + "\n".join(errors))
        print("classification coverage and local links: OK")
    if args.apply:
        sha = require_master_clean(ROOT)
        if not moves:
            die("nothing to translate")
        result = apply(pairs, moves, sha, ROOT)
        print(f"zh generated: {result}; master unchanged")
    else:
        print("preview only; use --apply after committing master")


if __name__ == "__main__":
    try:
        main()
    except (ValueError, OSError) as error:
        print(f"error: {error}", file=sys.stderr)
        raise SystemExit(1)
