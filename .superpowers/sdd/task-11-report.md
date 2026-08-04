# Task 11 Report — Update root README, AGENTS, SKILLS

**Status:** Complete

## Changes

- **README.md:** Documentation table expanded to all 10 surviving chapters; project tree lists 11 `documentation/*.md` files (10 chapters + index); `data/DATA_PIPELINE.md` marked stub; duplicate script lines removed; doc count fixed (was "16 component + ops docs").
- **AGENTS.md:** Added `documentation/moe.md` cross-ref on Triton path (path-only).
- **SKILLS.md:** Already retargeted (`documentation/data_pipeline.md`, `documentation/operations.md`) — no edits.
- **training.md:** Fixed stale `[transformer.md]` link → `architecture.md`.

## Verification

```bash
ls documentation/*.md | wc -l   # 11
rg stale patterns (excl. .superpowers/, docs/superpowers/, rope_yarn.md)  # 0 hits in documentation/
```

## Commit

`docs: update root README AGENTS SKILLS for consolidated chapters`

## Follow-ups

- Task 12: `scripts/check_docs.py` still references `CONTEXT.md`.
