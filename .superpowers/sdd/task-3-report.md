# Task 3 Report — `rope_yarn.md` merge

**Status:** Complete  
**Commit:** `4313d8f` — `docs: merge rotary.md and yarn.md into rope_yarn.md`

## Deliverables

| Action | Path |
|--------|------|
| Created | `documentation/rope_yarn.md` (753 lines) |
| Deleted | `documentation/rotary.md`, `documentation/yarn.md` |
| Link fixes | `README.md`, `ATTENTION_SINKS.md`, `OPTIMIZATIONS.md`, `getting_started.md`, `inference.md`, `configs.md` |

## TOC compliance

All 14 sections from brief present in locked order: Purpose → `apply_rope` → Frequency bases → YaRN theory → Production params → `compute_yarn_freqs`/`mscale` → `YaRNRoPE` → Pruned RoPE → Dtype/SDPA → Worked examples → Degenerate ramp → Debugging → Invariants → Verify.

## Content merge

- Rotary §§1–4, 8–12 ported (skipped standalone YaRN sections merged into §§4, 6, 7).
- Yarn §§1, 3–14 ported (skipped §2 RoPE Recap).
- Single end-to-end dataflow in §7.3 (merged Interaction with YaRN).
- Degenerate ramp `UserWarning` explicit in §11 with AGENTS rule reference.

## Verification

```bash
wc -l documentation/rope_yarn.md          # 753
test ! -f documentation/rotary.md         # OK
test ! -f documentation/yarn.md           # OK
python3 -m pytest tests/test_yarn.py -v   # 15 passed
```

## Parameters (honest)

θ=100000, scale=32, target=131072, original_max=4096, head_dim=96, prune=25% (24 pairs).

## Vault sync

`bash scripts/sync_to_vault.sh` — 222 files mirrored.

## Concerns / follow-ups

- `scripts/check_docs.py` redirect table for `rotary.md`/`yarn.md` not updated (Task 10 in overhaul plan).
- `documentation/README.md` size table uses `~730` estimate; run `--update-sizes` when lint task lands.
- Root `README.md` / `AGENTS.md` had no stale links (grep clean).
