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
DOC_DIR = ROOT / "docs"
README = DOC_DIR / "README.md"
FOOTER_RE = re.compile(r"\n<!-- docs:verified .+ -->\s*$")
LINK_RE = re.compile(r"\[[^\]]*\]\(([^)]+)\)")
FENCE_RE = re.compile(r"```.*?```", re.DOTALL)
# Heading-anchor machinery: a link target may carry a `#fragment`, which must
# match a heading id in the target file. GitHub derives ids as: drop inline
# markup, lowercase, keep word chars and hyphens, spaces -> hyphens, and give
# a repeated heading the `-1`, `-2` suffix.
HEADING_RE = re.compile(r"^#{1,6}\s+(.*?)\s*#*\s*$")
SETEXT_RE = re.compile(r"^(?:=+|-{2,})\s*$")
MD_LINK_RE = re.compile(r"!?\[([^\]]*)\]\([^)]*\)")
HTML_TAG_RE = re.compile(r"<[^>]+>")
SLUG_STRIP_RE = re.compile(r"[^\w\s-]")
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

# Anchor format shared with tests/test_doc_refs.py: `file.py:Symbol`.
ANCHOR_RE = re.compile(r"([A-Za-z_][A-Za-z0-9_./-]*\.py):([A-Za-z_][A-Za-z0-9_.]*)")
# Literal metavariables from templates/contracts are not anchors.
PLACEHOLDER_ANCHORS = {"file.py:Symbol", "file.py:Class.method", "file.py:function"}
# Every public module-level symbol in these modules must be cited in the docs.
# Kept in sync with tests/test_doc_refs.py:CORE_MODULES — same list, same intent:
# that file is the pytest entry point, this is the CLI entry point.
COVERAGE_MODULES = [
    "models/transformer.py",
    "models/attention.py",
    "models/moe.py",
    "models/moe_triton.py",
    "models/yarn.py",
    "models/rotary.py",
    "training/pretrain.py",
    "inference/generate.py",
    "inference/long_context.py",
    "utils/checkpoint.py",
    "utils/memory.py",
    "utils/logging.py",
    "data/prepare_data.py",
]

STALE_PATTERNS: list[tuple[str, str]] = [
    (r"\{,\}", "LaTeX thousand separator `{,}`"),
    (r"\b185 tests?\b", "stale test count (run pytest tests/ -q)"),
    (r"\b130 tests?\b", "stale test count (run pytest tests/ -q)"),
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
    """All markdown under docs/ (top-level + concepts/ + references/ + guides/), plus the index."""
    files = sorted(DOC_DIR.glob("*.md"))
    for sub in ("concepts", "references", "guides"):
        files += sorted(DOC_DIR.glob(f"{sub}/*.md"))
    return files


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
    return path_part.endswith(".md") or path_part.startswith("docs/")


_SLUG_CACHE: dict[Path, set[str]] = {}


def _blank_fences(text: str) -> str:
    """Blank out fenced code but preserve every offset, so the line numbers
    reported below still point at the right source line."""
    return FENCE_RE.sub(lambda m: re.sub(r"[^\n]", " ", m.group(0)), text)


def _slug(text: str) -> str:
    """GitHub heading id for a heading's text."""
    text = MD_LINK_RE.sub(r"\1", text)
    text = re.sub(r"`([^`]*)`", r"\1", text)
    text = HTML_TAG_RE.sub("", text)
    text = re.sub(r"[*~]", "", text).lower()
    # GitHub maps EACH space to its own hyphen — never collapse runs, or
    # "Part B — Config" would yield `part-b-config` instead of `part-b--config`.
    return SLUG_STRIP_RE.sub("", text).strip().replace(" ", "-")


def heading_slugs(path: Path) -> set[str]:
    """Every anchor id `path` exposes. Setext headings count, thematic
    breaks do not, and a repeated heading gets GitHub's `-1`, `-2` suffix.
    """
    if path in _SLUG_CACHE:
        return _SLUG_CACHE[path]
    lines = FENCE_RE.sub("", path.read_text(encoding="utf-8")).split("\n")
    counts: dict[str, int] = {}
    slugs: set[str] = set()
    for i, line in enumerate(lines):
        m = HEADING_RE.match(line)
        if m:
            title = m.group(1)
        elif line.strip() and i + 1 < len(lines) and SETEXT_RE.match(lines[i + 1]):
            title = line.strip()
        else:
            continue
        base = _slug(title)
        if not base:
            continue
        n = counts.get(base, 0)
        counts[base] = n + 1
        slugs.add(base if n == 0 else f"{base}-{n}")
    _SLUG_CACHE[path] = slugs
    return slugs


def check_markdown_links(path: Path, text: str) -> list[Issue]:
    issues: list[Issue] = []
    for match in LINK_RE.finditer(_blank_fences(text)):
        raw = match.group(1)
        if not is_doc_link(raw):
            continue
        resolved = resolve_link(path, raw)
        if resolved is None:
            continue
        line = text.count("\n", 0, match.start()) + 1
        if not resolved.exists():
            issues.append(Issue(path, line, f"broken link: {raw}"))
            continue
        fragment = raw.strip().partition("#")[2]
        if fragment and resolved.suffix == ".md":
            if fragment.lower() not in heading_slugs(resolved):
                issues.append(Issue(path, line, f"dead anchor: {raw}"))
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


def _docref_module():
    """Import tests/test_doc_refs.py so the CLI and the pytest gate share one
    symbol inventory. Returns None if it cannot be loaded."""
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    checker = ROOT / "tests" / "test_doc_refs.py"
    if not checker.is_file():
        return None
    import importlib.util

    spec = importlib.util.spec_from_file_location("_gptoss_doc_refs", checker)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def check_coverage() -> list[Issue]:
    """Every public module-level symbol in COVERAGE_MODULES must be cited once.

    Returns [] when the shared inventory cannot be loaded, so a missing test
    helper degrades to "no coverage opinion" rather than a false pass claim.
    """
    docref = _docref_module()
    if docref is None:
        return []

    text = "\n".join(
        FENCE_RE.sub("", p.read_text(encoding="utf-8"))
        for p in sorted(set(iter_doc_files()) | {
            d for d in (ROOT / "README.md",) if d.is_file()
        })
    )
    anchored = {
        (m.group(1), m.group(2))
        for m in ANCHOR_RE.finditer(text)
        if f"{m.group(1)}:{m.group(2)}" not in PLACEHOLDER_ANCHORS
    }

    issues: list[Issue] = []
    for rel in COVERAGE_MODULES:
        if not (ROOT / rel).is_file():
            issues.append(Issue(README, 0, f"coverage module missing: {rel}"))
            continue
        for sym in docref.inventory_symbols(rel):
            if (rel, sym) not in anchored:
                issues.append(Issue(README, 0, f"uncited public symbol: {rel}:{sym}"))
    return issues


def line_counts() -> list[tuple[str, int]]:
    rows: list[tuple[str, int]] = []
    for path in iter_doc_files():
        if path.name == "README.md":
            continue
        label = f"{path.parent.name}/{path.name}" if path.parent.name in ("concepts", "references", "guides") else path.name
        rows.append((label, sum(1 for _ in path.open(encoding="utf-8"))))
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
        print("check_docs: could not find doc size table in docs/README.md", file=sys.stderr)
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
    parser.add_argument(
        "--coverage", action="store_true",
        help="require every public symbol in COVERAGE_MODULES to be cited >= 1x",
    )
    parser.add_argument(
        "--check-symbols", action="store_true",
        help="also run the doc-ref alignment checker (tests/test_doc_refs.py --strict-coverage)",
    )
    args = parser.parse_args()

    if args.update_sizes:
        if update_size_table():
            print("Updated docs/README.md doc size table")
        else:
            print("Doc size table already up to date")

    if args.stamp_footers:
        n = stamp_footers(git_short_head(), date.today().isoformat())
        print(f"Stamped {n} documentation file(s)")

    issues = collect_issues()
    if args.coverage:
        issues.extend(check_coverage())
    if issues:
        print(f"check_docs: {len(issues)} issue(s)", file=sys.stderr)
        for issue in issues:
            print(issue.format(), file=sys.stderr)
        return 1

    if args.check_symbols:
        checker = Path(__file__).resolve().parents[1] / "tests" / "test_doc_refs.py"
        result = subprocess.run(
            [sys.executable, str(checker), "--strict-coverage"],
            capture_output=True, text=True,
        )
        print(result.stdout, end="")
        if result.returncode != 0:
            print("check_docs: symbol-alignment FAILED", file=sys.stderr)
            return 1
        print("check_docs: symbol alignment OK")

    if not args.update_sizes and not args.stamp_footers:
        print(f"check_docs: OK ({len(iter_doc_files())} files)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
