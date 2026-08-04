# Task 6 Report — Canonicalize data pipeline

**Status:** Complete  
**Date:** 2026-07-31

## Changes

| File | Action |
|------|--------|
| `documentation/data_pipeline.md` | Added § Vendored Copy (`data/shared_data/`), rsync update procedure, strengthened bit-identical shard note; updated TOC + Appendix C |
| `data/DATA_PIPELINE.md` | Replaced 131-line guide with 14-line stub pointing to canonical doc |
| `SKILLS.md` | Retargeted full-guide ref to `documentation/data_pipeline.md` |

## Verification

```text
$ wc -l data/DATA_PIPELINE.md documentation/data_pipeline.md
      14 data/DATA_PIPELINE.md
     783 documentation/data_pipeline.md

$ rg -n 'data/DATA_PIPELINE\.md' SKILLS.md README.md documentation/
SKILLS.md:139:   (`data/DATA_PIPELINE.md` is a short pointer).
```

- Stub ≤ 40 lines: **pass** (14)
- Single comprehensive source: **pass** (`documentation/data_pipeline.md`)
- Unique content folded: vendored layout, why vendored, rsync update, bit-identical shards

## Commit

`docs: canonicalize data pipeline under documentation/`

## Concerns

- ~~`data/shared_data/` is documented but not present in this clone~~ **Resolved** (see Path-alignment fix below).

---

## Path-alignment fix (review follow-up)

**Date:** 2026-07-31

### Problem

`documentation/data_pipeline.md` §5 claimed a vendored `data/shared_data/` and
`sys.path.insert(0, <project_root>/data)`. Actual `data/prepare_data.py`
prepends `_PROJECT_ROOT` (GPT-OSS-Lite/) and `_LLM_ROOT` (LLM/); there is no
`data/shared_data/` in this clone.

### Changes

| File | Action |
|------|--------|
| `documentation/data_pipeline.md` | Replaced §5 vendored-copy section with `LLM/shared_data/` shared-package docs; updated shim path-resolution table, code snippet, TOC, Appendix C |

### Verification

```text
$ rg -n 'shared_data|sys.path|vendored' documentation/data_pipeline.md data/prepare_data.py
# Docs: LLM/shared_data/ + _PROJECT_ROOT/_LLM_ROOT sys.path; explicit "no data/shared_data/"
# prepare_data.py: matches documented _PROJECT_ROOT / _LLM_ROOT inserts
```

- Bit-identical shard note (GPT-OSS + LLaMA-3, same tokenizer): **unchanged**
- `data/DATA_PIPELINE.md` stub: **unchanged** (14 lines, points to canonical doc)

### Commit

`docs: align data pipeline paths with prepare_data.py`
