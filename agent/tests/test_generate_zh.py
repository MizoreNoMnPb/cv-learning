"""Regression tests for classification moves and Markdown link preservation."""
from __future__ import annotations

import importlib.util
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch
from urllib.parse import quote

SCRIPT = Path(__file__).resolve().parents[1] / "generate-zh.py"
spec = importlib.util.spec_from_file_location("generate_zh", SCRIPT)
gen = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = gen
spec.loader.exec_module(gen)

GLOSSARY = """## Top-level
| id | English | zh-dir | Chinese |
| vision | `02-Vision-Tasks/` | `02-视觉/` | 视觉 |
| ad | `03-Anomaly-Detection/` | `03-异常/` | 异常 |
| da | `04-Domain-Adaptation/` | `04-适配/` | 适配 |
## Vision tasks
| image-generation | `Generative/` | `生成式/` | 生成式 |
## Anomaly detection
| ad-survey | `Survey/` | `综述/` | 综述 |
| absent | `VAE/` | `VAE/` | VAE |
## Domain adaptation
| translation | `Generative/` | `生成式翻译/` | 翻译 |
"""


class MappingTests(unittest.TestCase):
    def setUp(self):
        self.rows = gen.parse_glossary(GLOSSARY)
        self.pairs = gen.build_dir_renames(self.rows)

    def test_parent_specific_names_and_absent_families(self):
        pairs = dict(self.pairs)
        self.assertEqual(pairs["02-Vision-Tasks/Generative"], "02-视觉/生成式")
        self.assertEqual(pairs["04-Domain-Adaptation/Generative"], "04-适配/生成式翻译")
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "03-Anomaly-Detection/Survey").mkdir(parents=True)
            present, absent = gen.plan_renames(self.pairs, root)
            self.assertIn("03-Anomaly-Detection/VAE", absent)
            moves = gen.leaf_git_moves(self.rows, {src for src, _ in present})
            self.assertEqual(moves, [
                ("03-Anomaly-Detection", "03-异常"),
                ("03-异常/Survey", "03-异常/综述"),
            ])
            self.assertFalse((root / "03-Anomaly-Detection/VAE").exists())

    def test_invalid_and_duplicate_mapping(self):
        for glossary in [
            GLOSSARY.replace("`综述/`", "—"),
            GLOSSARY.replace("| translation |", "| ad-survey |"),
            GLOSSARY.replace("`04-适配/`", "`03-异常/`"),
            GLOSSARY.replace("`Survey/`", "`../Survey/`"),
        ]:
            with self.subTest(glossary=glossary):
                with self.assertRaises(ValueError):
                    gen.parse_glossary(glossary)

    def test_unknown_classification_fails(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "03-Anomaly-Detection/New-Family").mkdir(parents=True)
            with self.assertRaisesRegex(ValueError, "New-Family"):
                gen.validate_classifications(self.rows, root)


class LinkTests(unittest.TestCase):
    def setUp(self):
        self.old = "03-Anomaly-Detection/Survey/[TMLR 2022] A survey　(v5)/paper.pdf"
        self.new = self.old.replace("03-Anomaly-Detection/Survey", "03-异常/综述")
        self.moves = {"README.md": "README.md", self.old: self.new}

    def test_spaces_brackets_title_query_and_anchor(self):
        text = f'[论文](<{self.old}?download=1#page=2> "原文")'
        expected = f'[论文](<{self.new}?download=1#page=2> "原文")'
        self.assertEqual(gen.rewrite_links(text, "README.md", self.moves), expected)

    def test_encoded_destination_and_code_label(self):
        text = f'[`paper.pdf`]({quote(self.old)}#page=2)'
        expected = f'[`paper.pdf`]({quote(self.new)}#page=2)'
        self.assertEqual(gen.rewrite_links(text, "README.md", self.moves), expected)

    def test_images_and_relative_source_change(self):
        old_file = "03-Anomaly-Detection/Survey/note.md"
        new_file = "03-异常/综述/note.md"
        moves = {**self.moves, old_file: new_file, "02-Vision-Tasks/a.png": "02-视觉/a.png"}
        text = "![图](../../02-Vision-Tasks/a.png)"
        expected = "![图](../../02-%E8%A7%86%E8%A7%89/a.png)"
        self.assertEqual(gen.rewrite_links(text, old_file, moves), expected)

    def test_code_examples_and_external_links_untouched(self):
        link = f"[论文](<{self.old}>)"
        text = f"`{link}`\n```md\n{link}\n```\n~~~md\n{link}\n~~~\n[外部](https://example.invalid/a)\n[节](#one)"
        self.assertEqual(gen.rewrite_links(text, "README.md", self.moves), text)

    def test_relative_family_paths_in_prose(self):
        old_file = "03-Anomaly-Detection/overview.md"
        moves = {old_file: "03-异常/overview.md", "03-Anomaly-Detection/Survey": "03-异常/综述"}
        text = "分类 `Survey/`；示例 `unknown/path`。"
        self.assertEqual(
            gen.rewrite_backticks(text, [], old_file, moves),
            "分类 `综述/`；示例 `unknown/path`。",
        )

    def test_local_escape_rejected(self):
        with self.assertRaisesRegex(ValueError, "escapes repository"):
            gen.rewrite_links("[越界](../secret.md)", "README.md", self.moves)


class GenerationTests(unittest.TestCase):
    def test_generation_isolated_repeatable_and_failure_preserves_branch(self):
        with tempfile.TemporaryDirectory(prefix="cv-zh-test-") as temporary:
            root = Path(temporary) / "repo"
            root.mkdir()
            env = dict(os.environ)
            for key in list(env):
                if key.startswith(("GIT_", "HUSKY")):
                    env.pop(key)
            env.update(GIT_CONFIG_GLOBAL=os.devnull, GIT_CONFIG_NOSYSTEM="1")
            with patch.dict(os.environ, env, clear=True):
                def git(*args):
                    return gen.git(list(args), cwd=root).stdout.strip()
                git("init", "-b", "master")
                git("config", "user.name", "Test Author")
                git("config", "user.email", "test@example.invalid")
                git("config", "core.autocrlf", "false")
                git("config", "core.longpaths", "true")
                paper = "03-Anomaly-Detection/Survey/[2022] Survey　(v5)/paper.pdf"
                for name, content in {
                    ".gitignore": "scratch/\n",
                    "README.md": f"# Example\n\n[`PDF`](<{paper}>)\n",
                    paper: "PDF fixture",
                    "02-Vision-Tasks/Generative/note.md": "# 生成\n",
                    "04-Domain-Adaptation/Generative/note.md": "# 翻译\n",
                    "agent/path-glossary.md": GLOSSARY,
                    "agent/guide.md": f"`03-Anomaly-Detection/Survey/`\n[原文](<../{paper}>)\n",
                }.items():
                    path = root / name
                    path.parent.mkdir(parents=True, exist_ok=True)
                    path.write_text(content, encoding="utf-8")
                git("add", "-A")
                git("commit", "-m", "Initial fixture")
                sha = gen.require_master_clean(root)
                (root / "scratch").mkdir()
                (root / "scratch/private.md").write_text("[missing](not-found)", encoding="utf-8")
                rows = gen.parse_glossary(GLOSSARY)
                pairs, missing = gen.plan_renames(gen.build_dir_renames(rows), root)
                moves = gen.leaf_git_moves(rows, {src for src, _ in pairs})
                before = {p.relative_to(root): p.read_bytes() for p in root.rglob("*") if p.is_file() and ".git" not in p.parts}
                zh = gen.apply(pairs, moves, sha, root)
                self.assertEqual(git("rev-parse", "HEAD"), sha)
                self.assertEqual(git("rev-parse", "zh^"), sha)
                self.assertEqual(git("status", "--porcelain"), "")
                self.assertEqual(git("branch", "--show-current"), "master")
                self.assertEqual(git("show", "zh:03-异常/综述/[2022] Survey　(v5)/paper.pdf"), "PDF fixture")
                guide = git("show", "zh:agent/guide.md")
                self.assertIn("`03-Anomaly-Detection/Survey/`", guide)
                self.assertIn("../03-异常/综述/", guide)
                tracked = git("-c", "core.quotepath=false", "ls-tree", "-r", "--name-only", "zh")
                self.assertNotIn("scratch", tracked)
                self.assertNotIn("VAE", tracked)
                self.assertNotIn("Distribution-Shift-AD", tracked)
                after = {p.relative_to(root): p.read_bytes() for p in root.rglob("*") if p.is_file() and ".git" not in p.parts}
                self.assertEqual(before, after)
                (root / "README.md").write_text("# Example revised\n", encoding="utf-8")
                with self.assertRaisesRegex(ValueError, "must be clean"):
                    gen.require_master_clean(root)
                git("add", "README.md")
                git("commit", "-m", "Revise fixture")
                next_sha = gen.require_master_clean(root)
                next_zh = gen.apply(pairs, moves, next_sha, root)
                self.assertEqual(git("rev-parse", "zh^"), next_sha)
                old_ancestor = gen.git(["merge-base", "--is-ancestor", zh, next_zh], cwd=root, check=False)
                self.assertEqual(old_ancestor.returncode, 1)
                with patch.object(gen, "check_links", return_value=["injected broken link"]):
                    with self.assertRaisesRegex(ValueError, "generated links are broken"):
                        gen.apply(pairs, moves, next_sha, root)
                self.assertEqual(git("rev-parse", "zh"), next_zh)
                self.assertEqual(git("rev-parse", "HEAD"), next_sha)
                self.assertEqual(git("status", "--porcelain"), "")
                self.assertEqual(git("worktree", "list", "--porcelain").count("worktree "), 1)


if __name__ == "__main__":
    unittest.main()
