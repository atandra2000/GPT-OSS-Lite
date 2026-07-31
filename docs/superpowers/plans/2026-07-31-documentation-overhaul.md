# GPT-OSS-Lite Documentation Overhaul Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Consolidate 18 documentation files into 11 denser chapters (textbook + ops), with a single data-pipeline source of truth and a lint-clean link graph.

**Architecture:** Merge related chapters in place (or into new names `rope_yarn.md` / `operations.md`), rewrite the book index, stub `data/DATA_PIPELINE.md`, delete absorbed files and `CONTEXT.md`, then update `check_docs.py` redirects and root indexes.

**Tech Stack:** Markdown under `documentation/`, `scripts/check_docs.py`, root `README.md` / `AGENTS.md` / `SKILLS.md`, Obsidian vault sync via `~/Desktop/CoreProjects/scripts/sync_to_vault.sh`.

**Spec:** `docs/superpowers/specs/2026-07-31-documentation-overhaul-design.md`

## Global Constraints

- Audience: textbook + ops (option C) — every chapter follows the §4 content contract in the spec.
- Keep filename `documentation/ATTENTION_SINKS.md` as the sink-bias authority (AGENTS hard rule).
- Canonical data guide: **only** `documentation/data_pipeline.md`; `data/DATA_PIPELINE.md` is a short stub.
- No model / training / inference Python changes.
- Do not invent training results; keep 187 tests, ≥1.8× KV, aux α=0.01, window=128, sink clamp `[-10, 15]`.
- After any `.md` change batch: run `bash ~/Desktop/CoreProjects/scripts/sync_to_vault.sh`.
- Commits: only when the user explicitly asks (do not auto-commit during execution unless requested).

---

## File map (locked)

| Action | Path |
|--------|------|
| Modify | `documentation/architecture.md` (+ absorb `transformer.md`) |
| Modify | `documentation/ATTENTION_SINKS.md` (+ absorb `attention.md`) |
| Create | `documentation/rope_yarn.md` (from `rotary.md` + `yarn.md`) |
| Modify | `documentation/moe.md` (+ absorb `triton_kernels.md`) |
| Modify | `documentation/training.md` (+ absorb `configs.md`) |
| Modify | `documentation/data_pipeline.md` (+ unique bits from `data/DATA_PIPELINE.md`) |
| Modify | `documentation/inference.md` (deepen MixedKVCache / verify section) |
| Create | `documentation/operations.md` (from `scripts.md` + `utils.md` + `OPTIMIZATIONS.md`) |
| Modify | `documentation/getting_started.md`, `documentation/foundations.md` |
| Rewrite | `documentation/README.md` |
| Stub | `data/DATA_PIPELINE.md` |
| Delete | `transformer.md`, `attention.md`, `rotary.md`, `yarn.md`, `triton_kernels.md`, `configs.md`, `scripts.md`, `utils.md`, `OPTIMIZATIONS.md`, root `CONTEXT.md` |
| Modify | root `README.md`, `AGENTS.md`, `SKILLS.md`, `scripts/check_docs.py` |

---

### Task 1: Merge `transformer.md` into `architecture.md`

**Files:**
- Modify: `documentation/architecture.md`
- Delete (end of task): `documentation/transformer.md`
- Verify: `python3 scripts/check_docs.py` (expect failures until later tasks update links — local grep is the gate for this task)

**Interfaces:**
- Consumes: existing architecture sections 1–13; full `transformer.md` sections 1–12
- Produces: single `architecture.md` that owns system map **and** `ModelConfig` / `RMSNorm` / `GPTOSSBlock` / `GPTOSS` / init / weight tying / checkpointing schedule

- [ ] **Step 1: Inventory overlap**

Run:

```bash
rg -n '^## ' documentation/architecture.md documentation/transformer.md
```

Note sections already covered in architecture (parameter accounting, ModelConfig wiring, MoE dispatch mention, MixedKVCache mention). Those must be **deduplicated**, not pasted twice.

- [ ] **Step 2: Append a Part B — Transformer stack**

After the existing system/invariants content (before the final “Where to go next”), insert a clearly marked part with this TOC (renumber as needed to fit the file):

```markdown
## Part B — Transformer stack (`models/transformer.py`)

### B.1 Module overview
### B.2 `ModelConfig` fields and `__post_init__` validation
### B.3 `moe_dispatch` values (`stacked` | `triton_grouped`)
### B.4 `RMSNorm`
### B.5 `GPTOSSBlock` construction and forward
### B.6 `GPTOSS` construction and submodule roles
### B.7 Weight initialization policy
### B.8 Forward pass, `positions`, return contract `(logits, aux_loss)`
### B.9 Gradient checkpointing schedule (`grad_ckpt_every`)
### B.10 `num_parameters` / `num_active_parameters` + 502M breakdown
### B.11 Weight tying
### B.12 Config validation edge cases
### B.13 How to verify

Run: `python3 -m pytest tests/test_models.py tests/test_smoke.py -v`
```

Copy concrete prose, tables, and code blocks from `transformer.md` into these subsections. Delete portfolio-comparison / “where to go next” fluff that duplicates architecture’s existing close.

- [ ] **Step 3: Apply content contract header**

Ensure the top of `architecture.md` has Purpose (≤5 lines), then mental model / file map early. End with one next-step pointer: `ATTENTION_SINKS.md` and `moe.md`.

- [ ] **Step 4: Delete `transformer.md` and fix in-file links**

```bash
rm documentation/transformer.md
rg -n 'transformer\.md' documentation/ README.md AGENTS.md SKILLS.md CONTEXT.md
```

Replace remaining hits that belong to this merge with `architecture.md` (leave hits inside yet-unmerged files if they will be rewritten in later tasks — but fix any you touch now).

- [ ] **Step 5: Spot-check**

```bash
wc -l documentation/architecture.md
test ! -f documentation/transformer.md
```

Expected: `architecture.md` grows (~650 → ~900–1100 lines); `transformer.md` gone.

---

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

### Task 3: Create `rope_yarn.md`; delete `rotary.md` and `yarn.md`

**Files:**
- Create: `documentation/rope_yarn.md`
- Delete: `documentation/rotary.md`, `documentation/yarn.md`

**Interfaces:**
- Consumes: full rotary + yarn chapters
- Produces: single position-encoding chapter in the order locked by the spec

- [ ] **Step 1: Write new file with this exact top-level TOC**

```markdown
# RoPE and YaRN — Position Encoding for 128K

> Purpose: end-to-end position encoding from pairwise RoPE geometry through
> YaRN extrapolation and pruned RoPE on global layers.
> Sources: `models/rotary.py`, `models/yarn.py`.

## Table of contents

1. Purpose and mental model
2. RoPE geometry and `apply_rope`
3. Frequency bases
4. YaRN theory (ramp, blend, mscale)
5. Production parameters (θ=100K, scale=32, target=131072)
6. `compute_yarn_freqs` / `compute_yarn_mscale`
7. `YaRNRoPE` module
8. Pruned RoPE on global layers (25% of dims)
9. Dtype / SDPA contract
10. Worked numerical examples
11. Degenerate ramp warning
12. Debugging long-context issues
13. Invariants and failure modes
14. How to verify
```

- [ ] **Step 2: Port content in order**

1. From `rotary.md`: §§1–4, 8–12 (`apply_rope`, geometry, freqs helpers, dtype, broadcasting, worked example).  
2. From `yarn.md`: §§1, 3–14 (skip yarn’s “RoPE Recap” if redundant with step 1).  
3. Merge “Interaction with YaRN” sections into one dataflow subsection.  
4. Keep degenerate-ramp `UserWarning` behavior explicit (AGENTS numerical-stability rule).

Verification commands for §14:

```bash
python3 -m pytest tests/test_yarn.py -v
```

- [ ] **Step 3: Delete old files and fix links**

```bash
rm documentation/rotary.md documentation/yarn.md
rg -n 'rotary\.md|yarn\.md' documentation/ README.md AGENTS.md SKILLS.md
```

Replace with `rope_yarn.md`.

- [ ] **Step 4: Size sanity**

```bash
wc -l documentation/rope_yarn.md
test ! -f documentation/rotary.md && test ! -f documentation/yarn.md
```

Expected: roughly 700–900 lines (not a naive concat of 396+474 with double TOCs).

---

### Task 4: Merge `triton_kernels.md` into `moe.md`

**Files:**
- Modify: `documentation/moe.md`
- Delete: `documentation/triton_kernels.md`

**Interfaces:**
- Consumes: moe theory + triton kernel contract
- Produces: one MoE chapter; Triton is an explicit opt-in section (no silent fallback)

- [ ] **Step 1: Keep MoE theory front matter**

Retain: Abstract, Why MoE, SwiGLU, routing, aux loss (α=0.01, FP32 softmax), topology, class references, stacked dispatch.

- [ ] **Step 2: Replace thin Triton subsection with full contract**

Where moe.md currently has a short “Triton grouped dispatch” section, expand into:

```markdown
## Sanctioned Triton path (`moe_dispatch="triton_grouped"`)

### Why fuse W1/W3+silu (not W2)
### Public API `triton_moe_w1w3_silu`
### `HAS_TRITON` import policy and hard failure (no silent fallback)
### Kernel tiling `(BLOCK_T=16)×(BLOCK_M=32)×(BLOCK_N=32)`, `num_stages=1`
### Autograd: forward Triton, backward PyTorch reference
### Integration: `MoELayer._dispatch_triton`
### sm_75 vs sm_80 notes
### When to enable
### How to verify

Run: `python3 -m pytest tests/test_moe.py tests/test_moe_triton.py -v`
# GPU-only cases: pytest -m gpu
```

Port the load-bearing content from `triton_kernels.md` (API contracts, error policy, tiling, caps). Drop duplicate abstracts and glossary padding.

- [ ] **Step 3: Delete and retarget**

```bash
rm documentation/triton_kernels.md
rg -n 'triton_kernels\.md|moe_triton\.md' documentation/ README.md AGENTS.md SKILLS.md scripts/
```

Replace with `moe.md`. Update AGENTS “add documentation/\<name\>.md” examples if they cite `triton_kernels.md`.

- [ ] **Step 4: Confirm opt-in language**

```bash
rg -n 'silent fallback|moe_dispatch|triton_grouped' documentation/moe.md
```

Expected: explicit opt-in; hard error if Triton unavailable when selected.

---

### Task 5: Merge `configs.md` into `training.md`

**Files:**
- Modify: `documentation/training.md`
- Delete: `documentation/configs.md`

**Interfaces:**
- Consumes: training loop chapter + YAML encyclopedia
- Produces: loop narrative first; YAML reference as a marked Part B

- [ ] **Step 1: Slim existing “Configuration Walkthrough”**

Keep a short selected-fields preview in the loop narrative. Remove any claim that configs live only in a separate chapter.

- [ ] **Step 2: Append Part B — YAML encyclopedia**

```markdown
## Part B — Configuration reference

### B.1 How configs are loaded (`yaml.safe_load` → `ModelConfig` / dicts)
### B.2 File comparison: `pretrain_a100_502m.yaml` vs `pretrain_gpu_smoke.yaml`
### B.3 Derived quantities (tokens/step, Chinchilla steps, active params)
### B.4 `model` block — every field
### B.5 `training` block — every field
### B.6 `data` block — every field
### B.7 Cross-field interactions
### B.8 Override patterns (CLI)
### B.9 How to verify

Run: `python3 -m pytest tests/test_training.py tests/test_validation.py -v`
```

Port field tables verbatim from `configs.md` (do not drop keys).

- [ ] **Step 3: Delete and retarget**

```bash
rm documentation/configs.md
rg -n 'configs\.md' documentation/ README.md AGENTS.md SKILLS.md CONTEXT.md
```

Replace with `training.md` (optionally `#part-b-configuration-reference`).

- [ ] **Step 4: Content-contract close**

Ensure training.md ends with invariants (NaN guard never disabled without consent) + verify commands + one next link (`data_pipeline.md` / `operations.md`).

---

### Task 6: Canonicalize data pipeline; stub `data/DATA_PIPELINE.md`

**Files:**
- Modify: `documentation/data_pipeline.md`
- Rewrite: `data/DATA_PIPELINE.md` (stub only)

**Interfaces:**
- Consumes: existing documentation chapter + unique stub-worthy bits from `data/DATA_PIPELINE.md`
- Produces: one full guide; discoverable stub under `data/`

- [ ] **Step 1: Fold missing unique content into `documentation/data_pipeline.md`**

Ensure these topics exist (add if missing):

- Vendored `data/shared_data/` layout and why vendored  
- Updating the vendored copy (pointer to workspace sync procedure from the old data doc)  
- Explicit note that GPT-OSS + LLaMA-3 shards are bit-identical  

Keep the existing stage walkthrough (download → clean → tokenize → pack) as the body.

- [ ] **Step 2: Replace `data/DATA_PIPELINE.md` with this stub**

```markdown
# Data Pipeline — GPT-OSS-Lite

> **Canonical guide:** [`documentation/data_pipeline.md`](../documentation/data_pipeline.md)
>
> This file is a short pointer only. Do not maintain a second full copy here.

## Quick start

```bash
python3 data/prepare_data.py --stage pretrain
```

Tokenizer: LLaMA-3 BPE, vocab **128,000**, EOS **128,009**.  
Full stages, mix weights, shard format, and `PretrainDataset` details live in the documentation chapter above.
```

- [ ] **Step 3: Retarget skill/readme refs that treated data/ as canonical**

```bash
rg -n 'data/DATA_PIPELINE\.md' SKILLS.md README.md documentation/
```

In `SKILLS.md`, change “See `data/DATA_PIPELINE.md` for the full guide” → `documentation/data_pipeline.md` (stub may remain as secondary pointer).

- [ ] **Step 4: Verify single source of truth**

```bash
wc -l data/DATA_PIPELINE.md documentation/data_pipeline.md
# stub should be << full guide (expect stub ≤ 40 lines)
```

---

### Task 7: Deepen `inference.md`

**Files:**
- Modify: `documentation/inference.md`

**Interfaces:**
- Consumes: existing inference chapter + MixedKVCache details from architecture/ATTENTION_SINKS if duplicated thinly
- Produces: sharper decode complexity + verify section

- [ ] **Step 1: Ensure these sections exist with real detail**

```markdown
## MixedKVCache
### Windowed layers — ring buffer (capacity = window_size)
### Global layers — exponential growth
### Why decode is O(1) per step vs O(T) naive cache rebuild

## generate()
### Prefill vs decode
### Sink clamp cache
### YaRN T=1 fast path

## Passkey retrieval (`inference/long_context.py` / `scripts/passkey_eval.py`)
### Stub behavior on untrained checkpoints
### Target ≥85% at 128K after training

## How to verify
```

Verification block:

```bash
python3 -m pytest tests/test_inference.py -v
python3 scripts/passkey_eval.py --help
```

- [ ] **Step 2: Remove stale “when published” language**

```bash
rg -n 'when published|TODO|TBD' documentation/inference.md
```

Expected: no matches.

- [ ] **Step 3: Cross-link only to `ATTENTION_SINKS.md`, `architecture.md`, `operations.md`**

---

### Task 8: Create `operations.md`; delete scripts/utils/OPTIMIZATIONS

**Files:**
- Create: `documentation/operations.md`
- Delete: `documentation/scripts.md`, `documentation/utils.md`, `documentation/OPTIMIZATIONS.md`

**Interfaces:**
- Consumes: three ops chapters
- Produces: one operations handbook

- [ ] **Step 1: Write `operations.md` with this structure**

```markdown
# Operations — Scripts, Utilities, and Optimizations

> Purpose: runbooks for benchmarks, checkpoints/logging/memory helpers, and the
> OPT-1…24 performance catalog.

## Part A — Scripts (`scripts/`)
### A.1 Selection guide (table)
### A.2 `_bootstrap.py`
### A.3 `check_docs.py`
### A.4 `kv_cache_benchmark.py` (headline ≥1.8×)
### A.5 `passkey_eval.py`
### A.6 `e2e_gpu_smoke.py`
### A.7 Profilers (`profile_components`, `profile_moe`, `profile_inference`)
### A.8 `microbench_a100.py` / `step_time_a100.py`

## Part B — Utilities (`utils/`)
### B.1 `CheckpointManager` (atomic safetensors protocol)
### B.2 `TrainingLogger` + WandB
### B.3 `estimate_model_memory_gb` / Mixed KV term / `assert_fits_in_available_gpu`
### B.4 Worked VRAM examples

## Part C — Optimization catalog (OPT-1 … OPT-24)
### C.1 How to read the catalog
### C.2 Attention and masks
### C.3 Tensor layout and dtype
### C.4 MoE dispatch
### C.5 Training loop
### C.6 Inference
### C.7 Numerical stability
### C.8 Compilation
### C.9 Quick reference table

## How to verify
```

Port content from the three sources. Keep OPT numbers stable (do not renumber). One quick-reference table only (drop triplicate indexes).

Verify commands:

```bash
python3 scripts/check_docs.py
python3 scripts/kv_cache_benchmark.py
python3 -m pytest tests/test_utils.py -v
```

- [ ] **Step 2: Delete absorbed files**

```bash
rm documentation/scripts.md documentation/utils.md documentation/OPTIMIZATIONS.md
```

- [ ] **Step 3: Retarget links**

```bash
rg -n 'scripts\.md|utils\.md|OPTIMIZATIONS\.md' documentation/ README.md AGENTS.md SKILLS.md
```

Replace with `operations.md`. In `SKILLS.md`, change OPTIMIZATIONS cross-refs to `documentation/operations.md` Part C.

---

### Task 9: Tighten `getting_started.md` and `foundations.md`

**Files:**
- Modify: `documentation/getting_started.md`
- Modify: `documentation/foundations.md`

- [ ] **Step 1: getting_started — cut README echo**

Remove long architecture/config tables that duplicate root README. Keep: install, layout, first commands, smoke train, headline metrics, pitfalls, next chapter link.

Update internal links: `attention.md`→`ATTENTION_SINKS.md`, `rotary.md`/`yarn.md`→`rope_yarn.md`, `configs.md`→`training.md`, etc.

- [ ] **Step 2: foundations — deepen thin spots**

For any section that is mostly a bullet list without “why”, add 1–2 paragraphs of motivation tied to GPT-OSS-Lite numbers (window=128, 6+6 layers, α=0.01). Do not paste implementation code (that belongs in later chapters).

Update “Where to go next” to: `architecture.md` → `ATTENTION_SINKS.md`.

- [ ] **Step 3: Grep stale paths**

```bash
rg -n 'transformer\.md|attention\.md|rotary\.md|yarn\.md|triton_kernels\.md|configs\.md|scripts\.md|utils\.md|OPTIMIZATIONS\.md' \
  documentation/getting_started.md documentation/foundations.md
```

Expected: no matches.

---

### Task 10: Rewrite `documentation/README.md`; delete `CONTEXT.md`

**Files:**
- Rewrite: `documentation/README.md`
- Delete: `CONTEXT.md`
- Modify: root files that mention CONTEXT

- [ ] **Step 1: Write the new index**

Include:

1. One-paragraph project blurb + headline metrics table  
2. Learning path table with **only** the surviving files  
3. Agent routing table (from `CONTEXT.md`):

```markdown
| Question type | Read first |
|---|---|
| How does this repo implement X? | `models/*.py` + matching chapter |
| Sink bias / SWA / YaRN theory + impl | `ATTENTION_SINKS.md` |
| RoPE / YaRN | `rope_yarn.md` |
| MoE / Triton | `moe.md` |
| YAML / train loop | `training.md` |
| Data | `data_pipeline.md` |
| Inference / KV cache | `inference.md` |
| Scripts / utils / OPT-* | `operations.md` |
| What must not break? | `../AGENTS.md` + `architecture.md` invariants |
| Onboarding | `getting_started.md` |
```

4. Maintaining docs (`check_docs.py` commands)  
5. Doc size reference placeholder table (regenerated in Task 12)

- [ ] **Step 2: Delete CONTEXT and fix refs**

```bash
rm CONTEXT.md
rg -n 'CONTEXT\.md' README.md AGENTS.md SKILLS.md documentation/
```

Remove from project trees / lists.

- [ ] **Step 3: Confirm surviving set**

```bash
ls documentation/*.md | sort
```

Expected exactly:

```
ATTENTION_SINKS.md
README.md
architecture.md
data_pipeline.md
foundations.md
getting_started.md
inference.md
moe.md
operations.md
rope_yarn.md
training.md
```

---

### Task 11: Update root `README.md`, `AGENTS.md`, `SKILLS.md`

**Files:**
- Modify: `README.md`, `AGENTS.md`, `SKILLS.md`

- [ ] **Step 1: Root README Documentation table**

Replace the docs table with the surviving set. Fix project structure tree:

- `documentation/…` new names  
- `data/DATA_PIPELINE.md` described as stub → documentation  
- Remove `CONTEXT.md`  
- Remove duplicate script lines if still present  

- [ ] **Step 2: AGENTS.md path-only edits**

Keep hard rules intact. Ensure architecture / ATTENTION_SINKS / moe (Triton) paths match. If it mentions `OPTIMIZATIONS.md` or `triton_kernels.md`, retarget.

- [ ] **Step 3: SKILLS.md**

Retarget:

- `data/DATA_PIPELINE.md` full-guide mentions → `documentation/data_pipeline.md`  
- `documentation/OPTIMIZATIONS.md` → `documentation/operations.md`

- [ ] **Step 4: Repo-wide stale path sweep**

```bash
rg -n 'transformer\.md|attention\.md(?!_sink)|rotary\.md|yarn\.md|triton_kernels\.md|configs\.md|scripts\.md|utils\.md|OPTIMIZATIONS\.md|CONTEXT\.md|moe_triton\.md' \
  --glob '*.md' --glob '*.py'
```

Fix every hit in-repo (except historical notes inside `docs/superpowers/` specs/plans, which may name old files as sources).

---

### Task 12: Update `check_docs.py`; lint clean; vault sync

**Files:**
- Modify: `scripts/check_docs.py`

- [ ] **Step 1: Update `STALE_PATTERNS`**

Replace/extend so old filenames are flagged if reintroduced:

```python
STALE_PATTERNS: list[tuple[str, str]] = [
    (r"\{,\}", "LaTeX thousand separator `{,}`"),
    (r"\b190 tests?\b", "stale test count (use 187)"),
    (r"\b185 tests?\b", "stale test count (use 187)"),
    (r"\b130 tests?\b", "stale test count (use 187)"),
    (r"\b600-line\b", "stale ATTENTION_SINKS line count"),
    (r"moe_triton\.md", "use moe.md (Triton section)"),
    (r"triton_kernels\.md", "merged into moe.md"),
    (r"transformer\.md", "merged into architecture.md"),
    (r"\battention\.md\b", "merged into ATTENTION_SINKS.md"),
    (r"rotary\.md", "merged into rope_yarn.md"),
    (r"yarn\.md", "merged into rope_yarn.md"),
    (r"configs\.md", "merged into training.md"),
    (r"scripts\.md", "merged into operations.md"),
    (r"utils\.md", "merged into operations.md"),
    (r"OPTIMIZATIONS\.md", "merged into operations.md"),
    (r"ENABLE_TRITON_KERNELS", "removed env-var gate; use moe_dispatch config"),
    (r"when published", "stale placeholder link language"),
]
```

Note: `\battention\.md\b` must not match `ATTENTION_SINKS.md` — use a pattern that requires exact `attention.md` (as above).

- [ ] **Step 2: Run linter and refresh tables**

```bash
python3 scripts/check_docs.py
python3 scripts/check_docs.py --update-sizes
python3 scripts/check_docs.py --stamp-footers
python3 scripts/check_docs.py
```

Expected: final command exits 0 with no issues.

- [ ] **Step 3: Headline metric still green (docs-only change, sanity)**

```bash
python3 scripts/kv_cache_benchmark.py
```

Expected: `HEADLINE METRIC PASSED` (or existing ≥1.8× pass string).

- [ ] **Step 4: Vault sync**

```bash
bash ~/Desktop/CoreProjects/scripts/sync_to_vault.sh
```

Expected: mirrors updated without error.

- [ ] **Step 5: Success criteria checklist**

Re-check spec §7:

- [ ] Only target files remain under `documentation/`  
- [ ] `data/DATA_PIPELINE.md` is stub; full guide in documentation  
- [ ] `CONTEXT.md` removed  
- [ ] `ATTENTION_SINKS.md` still authoritative path  
- [ ] Chapters follow content contract  
- [ ] `check_docs.py` exits 0  
- [ ] Root README matches reality  
- [ ] Vault sync done  

---

## Plan self-review

| Spec requirement | Task |
|------------------|------|
| Merge transformer → architecture | Task 1 |
| Merge attention → ATTENTION_SINKS | Task 2 |
| rope_yarn from rotary+yarn | Task 3 |
| moe + triton | Task 4 |
| training + configs | Task 5 |
| data canonical + stub | Task 6 |
| deepen inference | Task 7 |
| operations from scripts+utils+OPT | Task 8 |
| tighten getting_started + foundations | Task 9 |
| rewrite index; delete CONTEXT | Task 10 |
| root README / AGENTS / SKILLS | Task 11 |
| check_docs + vault | Task 12 |
| Content contract | Embedded in Tasks 1–9 |
| No code feature changes | Global Constraints |

No TBD/TODO placeholders remain in task steps.
