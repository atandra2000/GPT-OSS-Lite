# Task 2 Report — Merge `attention.md` into `ATTENTION_SINKS.md`

**Status:** Complete  
**Commit:** `93e16ae` — `docs: merge attention.md into ATTENTION_SINKS.md`  
**Date:** 2026-07-31

---

## Summary

Merged `documentation/attention.md` into `documentation/ATTENTION_SINKS.md` as Part A
(theory, §§1–14) + Part B (implementation, B.1–B.10). Deleted `attention.md`.
Retargeted all `attention.md` links under `documentation/`, `README.md`, `AGENTS.md`,
and `SKILLS.md`.

---

## Brief checklist

| Step | Done | Notes |
|------|------|-------|
| 1. Restructure header | ✅ | New title + authoritative blurb per brief |
| 2. Part A dedupe §5 | ✅ | §5 shortened to pointer; impl detail in Part B only |
| 3. Append Part B | ✅ | B.1–B.10 with code blocks/tables from `attention.md` |
| 4. Delete + retarget links | ✅ | `attention.md` removed; 9 doc files updated |
| 5. Filename contract | ✅ | `ATTENTION_SINKS.md` exists; AGENTS.md unchanged path |

---

## Content contract

- **Filename:** `documentation/ATTENTION_SINKS.md` (unchanged — AGENTS hard rule)
- **Part A preserved:** historical context, math §4, BF16 clamp §6, alternation §7,
  KV-cache math §8 (≥ 1.8× at 128K), YaRN §9, prefill/decode, training, failure modes
- **Part B ported:** module overview, constants (`SINK_CLAMP_MIN/MAX` = [-10, 15]),
  `manual_causal_attention`, mask helpers, SDPA paths (incl. sink column), `repeat_kv`,
  `GPTOSSAttention`, forward trace, shapes, verification + pitfalls
- **Dedup:** Part B does not repeat KV math from Part A §8; Part A §5 no longer
  duplicates implementation walkthrough
- **No Python changes**

---

## Files changed

| File | Action |
|------|--------|
| `documentation/ATTENTION_SINKS.md` | Modified (+606/−~100 net; 1113 lines) |
| `documentation/attention.md` | Deleted |
| `documentation/architecture.md` | Link → Part B |
| `documentation/README.md` | TOC + size table updated |
| `documentation/yarn.md` | Links → ATTENTION_SINKS §B.7 / Part B |
| `documentation/rotary.md` | Links → Part B §B.8 |
| `documentation/inference.md` | Links consolidated |
| `documentation/getting_started.md` | Link → §7 alternation |
| `documentation/OPTIMIZATIONS.md` | Links consolidated |
| `documentation/utils.md` | Link → ATTENTION_SINKS |

---

## Verification

```bash
test -f documentation/ATTENTION_SINKS.md          # pass
test ! -f documentation/attention.md              # pass
rg -n 'attention\.md' documentation/ README.md AGENTS.md SKILLS.md  # no matches
rg -n 'ATTENTION_SINKS\.md' AGENTS.md             # lines 11, 74
rg -n '^### B\.' documentation/ATTENTION_SINKS.md  # B.1–B.10 present
```

Spot-checks:

- Header matches brief verbatim (title + blockquote)
- §5 ends with “Full walkthrough in Part B”
- B.10 includes `python3 -m pytest tests/test_attention.py -v` and
  `test_sliding_window_matches_full`
- Constants: `window_size=128`, clamp `[-10, 15]`, KV reduction ≈ 1.97× at 131072
  unchanged in Part A §8

---

## Self-review

**Strengths**

- Single canonical doc for sink theory + `models/attention.py` implementation
- Part B sections map 1:1 to brief TOC; SDPA backend notes folded into B.5,
  config knobs into B.7, inference integration into B.8, pitfalls into B.10
- Cross-doc links use Part B anchors where helpful (`#b5-…`, `#b8-…`)

**Minor notes (non-blocking)**

- `documentation/README.md` still lists `transformer.md` in size table (pre-existing;
  Task 1 merged transformer → architecture but README size table not fully updated)
- Line-count estimate `~1200` in README; actual 1113 lines

**Concerns:** None blocking. AGENTS contract intact.

---

## Vault sync

Run: `bash ~/Desktop/CoreProjects/scripts/sync_to_vault.sh`
