# Task 9 Report — Tighten getting_started and foundations

**Commit:** `18a4c62` — `docs: tighten getting_started and foundations`

## Done

### getting_started.md (−175 lines net)
- Removed README-echo tables (full key-numbers grid, architecture bullet stack, pretrain internals list).
- Kept: install, layout, data prep, first commands, smoke train, headline metrics, pitfalls, next-chapter table.
- Retargeted links: `transformer.md` → `architecture.md`; added `rope_yarn.md`, `operations.md`; all surviving chapters referenced.
- OOM pitfall now points to `operations.md` (not `utils/memory.py` alone).

### foundations.md (+motivation)
- Deepened decoder-only “why” with 2.25 GB KV @128K, W=128 alternation, 502M budget.
- YaRN section: why train+decode at scale=32 / mscale ≈1.35 vs decode-only YaRN.
- MoE dispatch: why stacked default + hard-fail Triton opt-in (no silent fallback).
- Pruned RoPE: tied 24/48 dims to odd global layers vs even windowed layers.
- “Where to go next”: doc links to surviving chapters; **next read → ATTENTION_SINKS.md**.

## Verification

```bash
rg -n 'transformer\.md|attention\.md|rotary\.md|triton_kernels\.md|configs\.md|scripts\.md|utils\.md|OPTIMIZATIONS\.md' \
  documentation/getting_started.md documentation/foundations.md
# → no matches

rg -n '(?<![_/])yarn\.md' documentation/getting_started.md documentation/foundations.md
# → no matches (rope_yarn.md is intentional)
```

Vault sync: `bash scripts/sync_to_vault.sh` completed.
