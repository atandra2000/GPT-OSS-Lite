#!/usr/bin/env python3
"""Generate the code↔doc coverage map from `file.py:Symbol` anchors.

Reads every anchor in documentation/**/*.md + README.md and prints a per-module
table of documented vs undocumented public symbols (same inventory as
tests/test_doc_refs.py --coverage).

Usage:
  python3 scripts/generate_code_map.py            # summary + gaps
  python3 scripts/generate_code_map.py --md       # markdown table (for README)
"""
from __future__ import annotations

import argparse
import pathlib
import sys

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from tests.test_doc_refs import CORE_MODULES, ANCHOR_RE, FENCE_RE, iter_doc_files, inventory_symbols  # noqa: E402


def anchored_symbols() -> set[tuple[str, str]]:
    anchored: set[tuple[str, str]] = set()
    for path in iter_doc_files():
        text = FENCE_RE.sub("", path.read_text(encoding="utf-8"))
        for match in ANCHOR_RE.finditer(text):
            anchored.add((match.group(1), match.group(2)))
    return anchored


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--md", action="store_true", help="emit a markdown table")
    args = parser.parse_args()

    anchored = anchored_symbols()
    by_module: dict[str, list[tuple[str, bool]]] = {}
    for mod in CORE_MODULES:
        by_module[mod] = [(sym, (mod, sym) in anchored) for sym in inventory_symbols(mod)]

    total = sum(len(v) for v in by_module.values())
    covered = sum(1 for v in by_module.values() for _, ok in v if ok)

    if args.md:
        print("| Module | Symbols | Anchored |")
        print("|---|---:|---:|")
        for mod, syms in sorted(by_module.items()):
            n = len(syms)
            c = sum(1 for _, ok in syms if ok)
            print(f"| `{mod}` | {n} | {c} |")
        print(f"\n**Total:** {covered}/{total} public symbols anchored "
              f"({covered / total:.0%} coverage).")
        return 0

    print(f"Code↔doc coverage map: {covered}/{total} public symbols anchored "
          f"({covered / total:.0%})")
    for mod, syms in sorted(by_module.items()):
        missing = [sym for sym, ok in syms if not ok]
        if missing:
            print(f"\n{mod} — {len(missing)} unanchored:")
            for sym in missing:
                print(f"  - {sym}")
    return 0 if covered == total else 1


if __name__ == "__main__":
    raise SystemExit(main())
