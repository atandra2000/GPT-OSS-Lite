# Task 5 Report — Merge `configs.md` into `training.md`

**Status:** Complete

## Changes

1. **Slimmed Configuration Walkthrough** — replaced full YAML blocks with a headline-knobs table and link to Part B.
2. **Appended Part B — Configuration reference** (B.1–B.9): loading, file comparison, derived quantities, full `model`/`training`/`data` field tables (verbatim from `configs.md`), cross-field interactions, smoke rationale, CLI overrides, verify command.
3. **Deleted** `documentation/configs.md`.
4. **Retargeted links** in `documentation/{README,architecture,getting_started,inference,scripts,utils}.md`, root `README.md`, `CONTEXT.md` → `training.md#part-b--configuration-reference`.
5. **Content contract close** — Load-Bearing Invariants (NaN guard never disabled without consent), B.9 pytest verify, next link to `data_pipeline.md`.

## Verification

```bash
python3 -m pytest tests/test_training.py tests/test_validation.py -v
# 35 passed in 18.67s
```

```bash
rg -n 'configs\.md' documentation/ README.md AGENTS.md SKILLS.md CONTEXT.md
# 0 hits (only historical refs in docs/superpowers/)
```

## Commit

`docs: merge configs.md into training.md`

## Concerns

- `documentation/README.md` line-count table is approximate (`~1200`); run `python3 scripts/check_docs.py --update-sizes` when available.
- `transformer.md` links in Part B remain (not deleted until a later task).
