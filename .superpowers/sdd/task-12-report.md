# Task 12 Report — Update check_docs.py; lint clean; vault sync

**Status:** Complete

## Changes

- **scripts/check_docs.py:** Extended `STALE_PATTERNS` per brief (merged-doc redirects, `when published`, test counts); `\byarn\.md\b` / `\brotary\.md\b` to avoid false positives on `rope_yarn.md`; `\battention\.md\b` unchanged (does not match `ATTENTION_SINKS.md`). Removed `CONTEXT.md` from root-doc iteration. Added `data/pretrain_smoke` and `data/shared_data` to `ALLOW_MISSING_PATHS`.
- **documentation/operations.md:** Removed stale "when published" placeholder on inference cross-link.
- **documentation/training.md:** Smoke `train_data_path` corrected to `data/pretrain_smoke` (matches `pretrain_gpu_smoke.yaml`).
- **documentation/README.md:** Size table refreshed via `--update-sizes`; all 11 chapter footers stamped.

## Verification

| Check | Result |
|-------|--------|
| `python3 scripts/check_docs.py` (final) | exit 0 |
| `python3 scripts/kv_cache_benchmark.py` | HEADLINE METRIC PASSED — 2.00× (≥1.8×) |
| Vault sync | 235 files mirrored |

## Success criteria (spec §7)

- [x] Only target files remain under `documentation/` (11 files)
- [x] `data/DATA_PIPELINE.md` is stub; full guide in documentation
- [x] `CONTEXT.md` removed
- [x] `ATTENTION_SINKS.md` still authoritative path
- [x] Chapters follow content contract
- [x] `check_docs.py` exits 0
- [x] Root README matches reality
- [x] Vault sync done

## Commit

`7fe1247` — `docs: finalize check_docs redirects and size table`
