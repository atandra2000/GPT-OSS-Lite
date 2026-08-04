# Task 4 Report — Merge triton_kernels.md into moe.md

**Status:** Complete  
**Date:** 2026-07-31

## Summary

Merged `documentation/triton_kernels.md` (628 lines) into `documentation/moe.md`
(920 lines). Added section **§12 Sanctioned Triton path (`moe_dispatch="triton_grouped"`)**
with: why fuse W1/W3+silu, public API, `HAS_TRITON` / hard `ImportError` (no silent
fallback), tiling `(16)×(32)×(32)`, `num_stages=1`, autograd (forward Triton /
backward PyTorch reference), `_dispatch_triton` integration, sm_75 vs sm_80, when to
enable, and verification commands. Retained MoE theory, aux loss α=0.01, stacked
dispatch. Deleted `triton_kernels.md`.

## Files changed

| Action | Path |
|--------|------|
| Modified | `documentation/moe.md` |
| Deleted | `documentation/triton_kernels.md` |
| Retargeted links | `configs.md`, `getting_started.md`, `OPTIMIZATIONS.md`, `README.md`, `architecture.md`, `training.md`, `scripts.md` |
| Modified | `scripts/check_docs.py` (stale-pattern: `triton_kernels.md` → merged into moe.md) |

## Verification

```bash
test ! -f documentation/triton_kernels.md                    # OK
rg 'triton_kernels\.md' documentation/ README.md AGENTS.md   # 0 hits (docs/)
python3 -m pytest tests/test_moe.py tests/test_moe_triton.py -v  # 23 passed, 2 skipped (GPU)
rg 'silent fallback|triton_grouped|moe_dispatch' documentation/moe.md  # explicit opt-in + no-fallback language
```

## Commit

`docs: merge triton_kernels.md into moe.md`

## Concerns

- `check_docs.py` still reports pre-existing broken `transformer.md` links (out of Task 4 scope).
- `documentation/README.md` size table row for `moe.md` is approximate (~900); run `check_docs.py --refresh-sizes` in a later task if exact counts matter.

## Constraints honored

- No Python source changes (only `scripts/check_docs.py` stale-pattern update).
- `moe_dispatch` explicit opt-in; no silent fallback documented and in invariants.
- Aux α=0.01 preserved throughout.
