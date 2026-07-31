#!/usr/bin/env python3
"""Lint documentation markdown and optionally refresh size table / verification stamps."""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
from dataclasses import dataclass
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DOC_DIR = ROOT / "documentation"
README = DOC_DIR / "README.md"
FOOTER_RE = re.compile(r"\n<!-- docs:verified .+ -->\s*$")
LINK_RE = re.compile(r"\[[^\]]*\]\(([^)]+)\)")
BACKTICK_PATH_RE = re.compile(
    r"`((?:configs|scripts|models|training|inference|utils|tests|data)/[A-Za-z0-9_./-]+)`"
)

ALLOW_MISSING_PATHS = {
    "data/pretrain_chinchilla",
    "data/pretrain_chinchilla/",
    "data/pretrain_smoke",
    "data/pretrain_smoke/",
    "data/scripts/download_raw.py",
    "data/scripts/pack_shards.py",
    "data/shards/shard_NNNNN.bin",
    "data/manifest.json",
    "data/config/mixture.yaml",
    "data/shared_data",
    "data/state",
    "models/__init__.py",
    "scripts/launch_a100.sh",
}

STALE_PATTERNS: list[tuple[str, str]] = [
    (r"\{,\}", "LaTeX thousand separator `{,}`"),
    (r"\b190 tests?\b", "stale test count (use 187)"),
    (r"\b185 tests?\b", "stale test count (use 187)"),
    (r"\b130 tests?\b", "stale test count (use 187)"),
    (r"\b600-line\b", "stale ATTENTION_SINKS line count"),
    (r"moe_triton\.md", "use moe.md (Triton section)"),
    (r"triton_kernels\.md", "merged into moe.md"),
    (r"transformer\.md", "merged into architecture.md"),
    (r"\battention\.md\b", "merged into ATTENTION_SINKS.md"),
    (r"\brotary\.md\b", "merged into rope_yarn.md"),
    (r"\byarn\.md\b", "merged into rope_yarn.md"),
    (r"configs\.md", "merged into training.md"),
    (r"scripts\.md", "merged into operations.md"),
    (r"utils\.md", "merged into operations.md"),
    (r"OPTIMIZATIONS\.md", "merged into operations.md"),
    (r"ENABLE_TRITON_KERNELS", "removed env-var gate; use moe_dispatch config"),
    (r"when published", "stale placeholder link language"),
]

SIZE_TABLE_START = "## Doc size reference"
SIZE_TABLE_HEADER = "| Doc | ~Lines | Status |"
SIZE_TABLE_DIVIDER = "|---|---|---|"


@dataclass
class Issue:
    path: Path
    line: int
    message: str

    def format(self) -> str:
        rel = self.path.relative_to(ROOT)
        return f"{rel}:{self.line}: {self.message}"


def git_short_head() -> str:
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=ROOT,
            stderr=subprocess.DEVNULL,
            text=True,
        )
        return out.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


def iter_doc_files() -> list[Path]:
    return sorted(DOC_DIR.glob("*.md"))


def check_control_chars(path: Path, text: str) -> list[Issue]:
    issues: list[Issue] = []
    for i, ch in enumerate(text):
        if ord(ch) < 32 and ch not in "\n\r\t":
            line = text.count("\n", 0, i) + 1
            issues.append(Issue(path, line, f"control character U+{ord(ch):04X}"))
    return issues


def check_stale_patterns(path: Path, text: str) -> list[Issue]:
    issues: list[Issue] = []
    for pattern, desc in STALE_PATTERNS:
        for match in re.finditer(pattern, text):
            line = text.count("\n", 0, match.start()) + 1
            issues.append(Issue(path, line, desc))
    return issues


def resolve_link(source: Path, target: str) -> Path | None:
    target = target.strip()
    if not target or target.startswith(("http://", "https://", "mailto:")):
        return None
    if target.startswith("#"):
        return None
    path_part, _, _anchor = target.partition("#")
    if path_part.startswith("/"):
        candidate = ROOT / path_part.lstrip("/")
    else:
        candidate = (source.parent / path_part).resolve()
    return candidate


def is_doc_link(target: str) -> bool:
    path_part = target.strip().partition("#")[0]
    return path_part.endswith(".md") or path_part.startswith("documentation/")


def check_markdown_links(path: Path, text: str) -> list[Issue]:
    issues: list[Issue] = []
    for match in LINK_RE.finditer(text):
        raw = match.group(1)
        if not is_doc_link(raw):
            continue
        resolved = resolve_link(path, raw)
        if resolved is None:
            continue
        if not resolved.exists():
            line = text.count("\n", 0, match.start()) + 1
            issues.append(Issue(path, line, f"broken link: {raw}"))
    return issues


def check_backtick_paths(path: Path, text: str) -> list[Issue]:
    issues: list[Issue] = []
    for match in BACKTICK_PATH_RE.finditer(text):
        rel = match.group(1).rstrip("/")
        if "*" in rel or "..." in rel:
            continue
        if rel in ALLOW_MISSING_PATHS or f"{rel}/" in ALLOW_MISSING_PATHS:
            continue
        candidate = ROOT / rel
        if not candidate.exists():
            line = text.count("\n", 0, match.start()) + 1
            issues.append(Issue(path, line, f"missing path: `{rel}`"))
    return issues


def collect_issues() -> list[Issue]:
    issues: list[Issue] = []
    for path in iter_doc_files():
        text = path.read_text(encoding="utf-8")
        issues.extend(check_control_chars(path, text))
        issues.extend(check_stale_patterns(path, text))
        issues.extend(check_markdown_links(path, text))
        issues.extend(check_backtick_paths(path, text))

    for root_doc in (ROOT / "README.md", ROOT / "AGENTS.md", ROOT / "SKILLS.md"):
        if root_doc.is_file():
            text = root_doc.read_text(encoding="utf-8")
            issues.extend(check_stale_patterns(root_doc, text))
            issues.extend(check_markdown_links(root_doc, text))
    return issues


def line_counts() -> list[tuple[str, int]]:
    rows: list[tuple[str, int]] = []
    for path in iter_doc_files():
        if path.name == "README.md":
            continue
        rows.append((path.name, sum(1 for _ in path.open(encoding="utf-8"))))
    rows.sort(key=lambda item: item[1], reverse=True)
    return rows


def render_size_table(rows: list[tuple[str, int]]) -> str:
    total = sum(count for _, count in rows)
    lines = [
        SIZE_TABLE_START,
        "",
        SIZE_TABLE_HEADER,
        SIZE_TABLE_DIVIDER,
    ]
    for name, count in rows:
        lines.append(f"| {name} | {count:,} | Comprehensive |")
    lines.append(f"| **Total** | **{total:,}** | |")
    lines.append("")
    return "\n".join(lines)


def update_size_table() -> bool:
    text = README.read_text(encoding="utf-8")
    rows = line_counts()
    new_block = render_size_table(rows)
    pattern = re.compile(
        r"## Doc size reference\n\n\| Doc \| ~Lines \| Status \|\n\|---\|---\|---\|\n(?:\|[^\n]+\n)+",
    )
    if not pattern.search(text):
        print("check_docs: could not find doc size table in documentation/README.md", file=sys.stderr)
        return False
    updated = pattern.sub(new_block + "\n", text, count=1)
    if updated == text:
        return False
    README.write_text(updated, encoding="utf-8")
    return True


def stamp_footers(commit: str, verified: str) -> int:
    footer = f"\n<!-- docs:verified {verified} · {commit} -->\n"
    changed = 0
    for path in iter_doc_files():
        text = path.read_text(encoding="utf-8")
        stripped = FOOTER_RE.sub("", text)
        if not stripped.endswith("\n"):
            stripped += "\n"
        new_text = stripped + footer
        if new_text != text:
            path.write_text(new_text, encoding="utf-8")
            changed += 1
    return changed


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate GPT-OSS-Lite documentation.")
    parser.add_argument("--update-sizes", action="store_true")
    parser.add_argument("--stamp-footers", action="store_true")
    args = parser.parse_args()

    if args.update_sizes:
        if update_size_table():
            print("Updated documentation/README.md doc size table")
        else:
            print("Doc size table already up to date")

    if args.stamp_footers:
        n = stamp_footers(git_short_head(), date.today().isoformat())
        print(f"Stamped {n} documentation file(s)")

    issues = collect_issues()
    if issues:
        print(f"check_docs: {len(issues)} issue(s)", file=sys.stderr)
        for issue in issues:
            print(issue.format(), file=sys.stderr)
        return 1

    if not args.update_sizes and not args.stamp_footers:
        print(f"check_docs: OK ({len(iter_doc_files())} files)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
