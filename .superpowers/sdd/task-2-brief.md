### Task 2: Merge `attention.md` into `ATTENTION_SINKS.md`

**Files:**
- Modify: `documentation/ATTENTION_SINKS.md`
- Delete: `documentation/attention.md`

**Interfaces:**
- Consumes: ATTENTION_SINKS theory §§1–9+; attention.md implementation §§1–10+
- Produces: one file with **Part A Theory** + **Part B Implementation**; filename unchanged

- [ ] **Step 1: Restructure header**

Change the opening to state both roles:

```markdown
# Attention Sinks & Alternating Attention — GPT-OSS-Lite

> **Authoritative** reference for learned sink bias, sliding-window / full
> alternation, and the `models/attention.py` implementation. Required reading
> before changing attention code (see `AGENTS.md`).
```

- [ ] **Step 2: Keep Part A (theory) — dedupe impl sketches**

Retain historical context, math (§4), BF16 clamp, alternation, KV-cache math, YaRN interaction. Where §5 already sketches `manual_causal_attention` / `causal_attention` / `GPTOSSAttention`, shorten §5 to a pointer: “Full walkthrough in Part B.”

- [ ] **Step 3: Append Part B from `attention.md`**

Add:

```markdown
## Part B — Implementation (`models/attention.py`)

### B.1 Module overview and public surface
### B.2 Constants (`SINK_BIAS_MIN`, `SINK_BIAS_MAX`)
### B.3 `manual_causal_attention` (FP32 scores, window, sink)
### B.4 Mask helpers (`_causal_mask`, `_window_mask`, cache key)
### B.5 `causal_attention` SDPA paths (fast / window / sink column)
### B.6 `repeat_kv` (`expand` + `reshape`, no `.contiguous()`)
### B.7 `GPTOSSAttention` construction (sink param, YaRN, pruned RoPE)
### B.8 Forward path trace (positions → out proj)
### B.9 Shape reference
### B.10 How to verify

Run: `python3 -m pytest tests/test_attention.py -v`
# Must include: test_sliding_window_matches_full
```

Port code blocks and tables from `attention.md`. Do not duplicate KV math already in Part A §8.

- [ ] **Step 4: Delete and retarget links**

```bash
rm documentation/attention.md
rg -n 'attention\.md' documentation/ README.md AGENTS.md SKILLS.md
```

Replace with `ATTENTION_SINKS.md` (or `#part-b-…` anchors where helpful).

- [ ] **Step 5: Verify filename contract**

```bash
test -f documentation/ATTENTION_SINKS.md
test ! -f documentation/attention.md
rg -n 'ATTENTION_SINKS\.md' AGENTS.md
```

Expected: AGENTS still points at `documentation/ATTENTION_SINKS.md`.

---

