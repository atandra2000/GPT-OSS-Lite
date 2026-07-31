# Design: GPT-OSS-Lite Documentation Overhaul

**Date:** 2026-07-31  
**Status:** Approved (structure + quality + execution)  
**Audience:** Portfolio / interview reviewers **and** operators running experiments (option C)  
**Approach:** Hard consolidate into ~9 chapters; deepen weak spots; eliminate duplicate sources of truth

---

## 1. Problem

The `documentation/` tree is large (~10k lines across 17 files) but feels sub-standard:

- **Too many small chapters** — readers hop across `rotary` / `yarn`, `moe` / `triton_kernels`, `scripts` / `utils` / `OPTIMIZATIONS`, `configs` / `training`, `transformer` / `architecture`, `attention` / `ATTENTION_SINKS`.
- **Duplication** — root `CONTEXT.md`, root `README.md`, `getting_started.md`, and `data/DATA_PIPELINE.md` vs `documentation/data_pipeline.md` overlap.
- **Uneven depth** — heavy TOC / table / “where to go next” boilerplate; stale “when published” links; thin continuous narrative in places that should be textbook-grade.
- **Agent friction** — AGENTS hard-rules point at specific paths; redirects and stale filenames in `check_docs.py` lag renames.

---

## 2. Goals

1. **Fewer files, denser chapters** — one coherent story per topic.
2. **Textbook + ops** — foundations and architecture readable once; training / data / operations sharp as runbooks.
3. **Single sources of truth** — especially data pipeline and attention/sinks.
4. **Preserved agent contracts** — keep `documentation/ATTENTION_SINKS.md` as the sink-bias authority filename.
5. **Lint-clean** — `python3 scripts/check_docs.py` passes; old paths redirect or are updated.

### Non-goals

- No model / training code changes.
- No new architectural features (MLA, GDN, MTP, etc.).
- No rewrite of AGENTS hard rules beyond path updates.
- No inventing training results the repo does not have.

---

## 3. Target file layout

### 3.1 Surviving chapters (`documentation/`)

| # | File | Role | Absorbs |
|---|------|------|---------|
| 0 | `README.md` | Book index + learning path + agent routing (from `CONTEXT.md`) | `CONTEXT.md` routing table |
| 1 | `getting_started.md` | Onboarding; cut README echo | — |
| 2 | `foundations.md` | Why decoder-only, GQA, SWA, sinks, YaRN, MoE, Chinchilla | — (deepen thin spots) |
| 3 | `architecture.md` | System map, dataflow, invariants, `GPTOSS` / blocks / `ModelConfig` | `transformer.md` |
| 4 | `ATTENTION_SINKS.md` | Authoritative sink + SWA/full theory **and** implementation reference | `attention.md` |
| 5 | `rope_yarn.md` | RoPE → YaRN → pruned RoPE (global layers) as one story | `rotary.md`, `yarn.md` |
| 6 | `moe.md` | Top-2 routing, aux loss, grouped dispatch, Triton contract | `triton_kernels.md` |
| 7 | `training.md` | Pretrain loop, NaN guard, checkpoints, **full YAML encyclopedia** | `configs.md` |
| 8 | `data_pipeline.md` | Canonical data guide (only full source) | content from `data/DATA_PIPELINE.md` |
| 9 | `inference.md` | `MixedKVCache`, `generate()`, passkey eval | — (deepen) |
| 10 | `operations.md` | Scripts, utils, OPT-1…24 catalog | `scripts.md`, `utils.md`, `OPTIMIZATIONS.md` |

**Count:** 10 chapters + index = **11 files** in `documentation/` (was 18).  
Deletes 9 absorbed files; adds 2 new names (`rope_yarn.md`, `operations.md`).

### 3.2 Root / data

| Path | Action |
|------|--------|
| `CONTEXT.md` | Delete after routing table lands in `documentation/README.md` |
| `data/DATA_PIPELINE.md` | Replace with a **short stub** (≥ pointing to `documentation/data_pipeline.md`). No second full guide. |
| Root `README.md` | Update Documentation table + project tree to new names |
| `AGENTS.md` | Path updates only (architecture, ATTENTION_SINKS, ops/moe as needed) |
| `SKILLS.md` | Retarget `data/DATA_PIPELINE.md` / `OPTIMIZATIONS.md` / other absorbed paths |

### 3.3 Deleted after merge

`transformer.md`, `attention.md`, `rotary.md`, `yarn.md`, `triton_kernels.md`, `configs.md`, `scripts.md`, `utils.md`, `OPTIMIZATIONS.md`, `CONTEXT.md` (root).

---

## 4. Chapter content contract

Every surviving chapter (except the index README) uses this skeleton:

1. **Purpose** — ≤5 lines; what this chapter owns  
2. **Mental model** — equations / diagrams that matter for *this* repo  
3. **Implementation map** — symbols ↔ files ↔ config keys  
4. **Worked numbers** — ≥1 concrete A100 / 502M example (params, VRAM, KV bytes, tokens/step, etc.)  
5. **Invariants & failure modes** — what breaks headline metrics, NaN, routing collapse  
6. **How to verify** — exact `pytest` / script commands  
7. **Cross-links** — only adjacent chapters; no “when published”

### Quality bar

- Prefer continuous prose + one worked example over repeating the same comparison table in every file.
- Cut duplicated “portfolio comparison” and generic “where to go next” blocks; keep a single next-step pointer at the end.
- Numbers must match code/config/tests (187 tests, ≥1.8× KV, aux α=0.01, window=128, sink clamp `[-10, 15]`, etc.).
- Preserve math that already exists in foundations / sinks / YaRN; do not dilute when merging.

### Per-chapter merge notes

**`ATTENTION_SINKS.md` + `attention.md`**  
- Part A: theory (existing sinks / SWA / YaRN narrative).  
- Part B: implementation (`SlidingWindowAttention`, `FullAttention`, SDPA paths, mask cache, sink clamp).  
- Filename stays `ATTENTION_SINKS.md` (AGENTS hard rule).

**`architecture.md` + `transformer.md`**  
- System diagram and invariants stay front.  
- Append `GPTOSS`, `GPTOSSBlock`, `RMSNorm`, `ModelConfig` field wiring without duplicating MoE/attention deep dives.

**`rope_yarn.md`**  
- Order: RoPE geometry → `apply_rope` → YaRN freqs/mscale → pruned RoPE on global layers → dtype/SDPA contract.

**`moe.md` + `triton_kernels.md`**  
- Theory and aux loss first; Triton as an explicit opt-in section with the AGENTS kernel contract (no silent fallback).

**`training.md` + `configs.md`**  
- Loop narrative first; YAML field encyclopedia as a clearly marked reference section (every `model` / `training` / `data` key).

**`data_pipeline.md`**  
- Canonical under `documentation/` only.  
- Fold shim / tokenizer / quick-start bits from `data/DATA_PIPELINE.md`.  
- Stub at `data/DATA_PIPELINE.md` points here.

**`operations.md`**  
- Script selection guide + per-script invoke/interpret.  
- Checkpoint / logging / memory utils.  
- OPT catalog condensed but complete (OPT-1…24); drop triplicate indexes.

**`inference.md`**  
- Emphasize MixedKVCache ring vs exponential growth and O(1) decode claim; passkey stub vs trained checkpoint.

**`getting_started.md` / `foundations.md`**  
- Trim overlap with root README; deepen foundations where sections are table-only.

---

## 5. Link and tooling updates

### 5.1 `scripts/check_docs.py`

- Extend `STALE_PATTERNS` (or a redirect map) for old filenames:  
  `transformer.md`, `attention.md`, `rotary.md`, `yarn.md`, `triton_kernels.md`, `configs.md`, `scripts.md`, `utils.md`, `OPTIMIZATIONS.md`, `moe_triton.md`.  
- Point messages at the new canonical files (`architecture.md`, `ATTENTION_SINKS.md`, `rope_yarn.md`, `moe.md`, `training.md`, `operations.md`).  
- Refresh size table logic against the new file set.  
- Keep footer stamp support.

### 5.2 Cross-repo references inside this repo

Update all markdown and code comments that cite absorbed paths (README, AGENTS, SKILLS, chapter cross-links, any script docstrings).

### 5.3 Vault sync

After markdown changes: `bash ~/Desktop/CoreProjects/scripts/sync_to_vault.sh`.

---

## 6. Implementation order

1. Draft merged chapters on disk (new names first: `rope_yarn.md`, `operations.md`; expand survivors).  
2. Rewrite `documentation/README.md` learning path + routing.  
3. Stub `data/DATA_PIPELINE.md`; delete absorbed chapter files and `CONTEXT.md`.  
4. Update root README / AGENTS / SKILLS.  
5. Update `check_docs.py`; run until clean; stamp footers / size table.  
6. Vault sync.

Suggested batching for review:

- Batch A: architecture + ATTENTION_SINKS + rope_yarn + moe  
- Batch B: training + data_pipeline + inference + operations  
- Batch C: getting_started + foundations + indexes + tooling + deletes

---

## 7. Success criteria

- [ ] `documentation/` has the target set only (no absorbed leftovers).  
- [ ] `data/DATA_PIPELINE.md` is a stub; full guide only in `documentation/data_pipeline.md`.  
- [ ] `CONTEXT.md` removed; routing lives in `documentation/README.md`.  
- [ ] `ATTENTION_SINKS.md` remains the sink-bias authority path.  
- [ ] Each chapter follows the content contract (§4).  
- [ ] `python3 scripts/check_docs.py` exits 0.  
- [ ] Root README Documentation section and project tree match reality.  
- [ ] Vault sync completed for new/changed `.md` files.

---

## 8. Decisions locked

| Decision | Choice |
|----------|--------|
| Audience | C — textbook + ops |
| Consolidation level | Hard merge (10 chapters + README = 11 files) |
| Attention | Merge `attention.md` into `ATTENTION_SINKS.md` |
| Data guide | Canonical in `documentation/`; stub in `data/` |
| `CONTEXT.md` | Absorb into doc index; delete root file |
| Code changes | Out of scope |
