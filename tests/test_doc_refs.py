"""Doc-code alignment checker: every `file.py:Symbol` anchor must resolve.

Usage:
  python3 tests/test_doc_refs.py                    # resolve anchors; exit 1 on stale
  python3 tests/test_doc_refs.py --coverage         # report public symbols with no anchor
  python3 tests/test_doc_refs.py --strict-coverage  # coverage gaps fail the run

Anchor format: `` `file.py:Symbol` `` or `` `file.py:Class.method` ``. Line
numbers (`file.py:123`) are NOT anchors and are rejected. Symbols defined
inside ``if HAS_TRITON:`` blocks (JIT kernels) are excluded from coverage —
always-defined host wrappers must be cited instead.
"""
from __future__ import annotations

import argparse
import ast
import importlib.util
import pathlib
import re
import sys

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
LLM_ROOT = REPO_ROOT.parent  # sibling workspace package (LLM/shared_data)
DOC_GLOBS = ("docs/**/*.md", "README.md")
CORE_MODULES = (
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
)

ANCHOR_RE = re.compile(r"([A-Za-z_][A-Za-z0-9_./-]*\.py):([A-Za-z_][A-Za-z0-9_.]*)")
FENCE_RE = re.compile(r"```.*?```", re.DOTALL)
# Literal metavariables from templates/contracts ("file.py:Symbol") are not anchors.
PLACEHOLDER_ANCHORS = {"file.py:Symbol", "file.py:Class.method", "file.py:function"}


def iter_doc_files() -> list[pathlib.Path]:
    files: list[pathlib.Path] = []
    for glob in DOC_GLOBS:
        files.extend(REPO_ROOT.glob(glob))
    return sorted(set(files))


def doc_text_without_fences() -> str:
    parts: list[str] = []
    for path in iter_doc_files():
        text = path.read_text(encoding="utf-8")
        parts.append(FENCE_RE.sub("", text))
    return "\n".join(parts)


def _load_module(rel_path: str):
    """Import a module by repo-relative path. Returns None if the file is missing."""
    for base in (REPO_ROOT, LLM_ROOT):
        path = base / rel_path
        if path.exists():
            mod_name = "docref_" + re.sub(r"[^A-Za-z0-9]", "_", rel_path)
            spec = importlib.util.spec_from_file_location(mod_name, path)
            if spec is None or spec.loader is None:
                return None
            mod = importlib.util.module_from_spec(spec)
            sys.modules[mod_name] = mod
            try:
                spec.loader.exec_module(mod)
            except Exception:
                return mod  # importable state unknown -> AST fallback below
            return mod
    return None


def _ast_has_symbol(tree: ast.Module, symbol: str) -> bool:
    """True if the AST defines `symbol` ('Class.method' / 'Class.attr' supported).

    Resolves module functions, classes, class methods, class-level attributes,
    and instance attributes assigned via ``self.attr = ...`` in ``__init__``.
    """
    parts = symbol.split(".")
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == parts[0]:
            return len(parts) == 1
        # Module-level constants (e.g. HAS_TRITON = ...) and annotated
        # module-level names (e.g. `SINK_CLAMP_MIN: float = -10.0`).
        if isinstance(node, (ast.Assign, ast.AnnAssign)) and _assigns_name(node, parts[0]):
            return len(parts) == 1
    # Constants assigned at module scope inside try/except or if/else blocks
    # (e.g. `try: HAS_TRITON = True / except ImportError: HAS_TRITON = False`).
    if len(parts) == 1 and _module_scope_assigns(tree, parts[0]):
        return True
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name == parts[0]:
            if len(parts) == 1:
                return True
            for sub in node.body:
                if isinstance(sub, (ast.FunctionDef, ast.AsyncFunctionDef)) and sub.name == parts[1]:
                    return True
                # Plain class attrs AND dataclass-annotated fields
                # (ast.AnnAssign, e.g. `vocab_size: int = 0`).
                if isinstance(sub, (ast.Assign, ast.AnnAssign)) and _assigns_name(sub, parts[1]):
                    return True
            for sub in node.body:
                if isinstance(sub, ast.FunctionDef) and sub.name == "__init__":
                    for stmt in ast.walk(sub):
                        if isinstance(stmt, ast.Assign):
                            for t in stmt.targets:
                                if (
                                    isinstance(t, ast.Attribute)
                                    and isinstance(t.value, ast.Name)
                                    and t.value.id == "self"
                                    and t.attr == parts[1]
                                ):
                                    return True
    return False


def _assigns_name(node: ast.AST, name: str) -> bool:
    """True if an Assign/AnnAssign node targets the module/class name `name`."""
    if isinstance(node, ast.AnnAssign):
        target = node.target
        return isinstance(target, ast.Name) and target.id == name
    if isinstance(node, ast.Assign):
        return any(isinstance(t, ast.Name) and t.id == name for t in node.targets)
    return False


def _module_scope_assigns(tree: ast.Module, name: str) -> bool:
    """True if `name` is assigned anywhere at module scope (incl. inside
    try/except, if/else, and with blocks, but not inside functions/classes)."""
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.Lambda)):
            continue
        if isinstance(node, (ast.Assign, ast.AnnAssign)) and _assigns_name(node, name):
            return True
    return False


def resolve_anchor(rel_path: str, symbol: str) -> str:
    """Return 'ok' or a reason the anchor is stale."""
    mod = _load_module(rel_path)
    if mod is None:
        return "missing-file"
    obj: object = mod
    for part in symbol.split("."):
        if not hasattr(obj, part):
            tree = ast.parse(pathlib.Path(REPO_ROOT / rel_path).read_text())
            return "ok" if _ast_has_symbol(tree, symbol) else f"missing-symbol:{part}"
        obj = getattr(obj, part)
    return "ok"


def check_resolution(verbose: bool = False) -> list[str]:
    errors: list[str] = []
    text = doc_text_without_fences()
    for match in ANCHOR_RE.finditer(text):
        rel_path, symbol = match.group(1), match.group(2)
        if f"{rel_path}:{symbol}" in PLACEHOLDER_ANCHORS:
            continue
        status = resolve_anchor(rel_path, symbol)
        if status != "ok":
            errors.append(f"{rel_path}:{symbol} -> {status}")
        elif verbose:
            print(f"  ok  {rel_path}:{symbol}")
    return errors


def inventory_symbols(rel_path: str) -> list[str]:
    """Public module-level functions/classes + public class methods."""
    tree = ast.parse(pathlib.Path(REPO_ROOT / rel_path).read_text(encoding="utf-8"))
    syms: list[str] = []
    for node in tree.body:
        if isinstance(node, ast.If):
            test = node.test
            if isinstance(test, ast.Name) and test.id == "HAS_TRITON":
                continue  # JIT kernels: never required, never cited
            continue
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and not node.name.startswith("_"):
            syms.append(node.name)
        elif isinstance(node, ast.ClassDef) and not node.name.startswith("_"):
            for sub in node.body:
                if isinstance(sub, (ast.FunctionDef, ast.AsyncFunctionDef)) and not sub.name.startswith("_"):
                    syms.append(f"{node.name}.{sub.name}")
    return syms


def check_coverage() -> list[str]:
    text = doc_text_without_fences()
    anchored: set[tuple[str, str]] = set()
    for match in ANCHOR_RE.finditer(text):
        anchored.add((match.group(1), match.group(2)))
    gaps: list[str] = []
    for mod in CORE_MODULES:
        for sym in inventory_symbols(mod):
            if (mod, sym) not in anchored:
                gaps.append(f"{mod}:{sym}")
    return sorted(gaps)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--coverage", action="store_true", help="report public symbols with no anchor")
    parser.add_argument("--strict-coverage", action="store_true", help="fail the run on coverage gaps")
    parser.add_argument("--verbose", action="store_true", help="print every resolved anchor")
    args = parser.parse_args()

    errors = check_resolution(args.verbose)
    for e in errors:
        print(f"STALE ANCHOR  {e}")

    if args.coverage or args.strict_coverage:
        gaps = check_coverage()
        print(f"\nCoverage: {len(gaps)} public symbol(s) without an anchor")
        for g in gaps:
            print(f"  UNANCHORED  {g}")

    fail = bool(errors)
    if args.strict_coverage and gaps:
        fail = True
    print(f"\ntest_doc_refs: {'FAIL' if fail else 'OK'} "
          f"({len(errors)} stale anchor(s))")
    return 1 if fail else 0


if __name__ == "__main__":
    raise SystemExit(main())
