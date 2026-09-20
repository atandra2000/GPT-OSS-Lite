# GPT-OSS-Lite — Documentation & Codebase Audit

> **Scope.** Full audit of docs↔code alignment, verified against the working
> tree on 2026-09-21. Every claim below was checked by running the command or
> inspecting the cited source — nothing asserted from memory. This audit is
> part of the portfolio-wide upgrade to the LLaMA-3-Lite docs standard
> (four-track taxonomy + machine gates); it records the state after the
> learning-paths/glossary/audit addition.

---

## Verification runs (2026-09-21)

| Command | Result |
|---|---|
| `python3 tests/test_doc_refs.py --strict-coverage` | OK — all `file.py:Symbol` anchors resolve, strict public-symbol coverage holds, 0 line anchors |
| `python3 scripts/check_docs.py` | OK — 14 files linted (links, backtick paths, stale patterns) |
| `python3 -m pytest tests/ -q` | **203 passed, 2 skipped** (GPU-gated), ~54 s on macOS CPU |
| Anchor census | 426 `file.py:Symbol` citations across docs (165 distinct file↔symbol pairs) |
| Corpus size | 17 markdown files under `docs/`, ~91 000 words (measured `wc -w` 2026-09-21) |

---

## 1. State of the docs — what is already excellent

- **Corpus:** 17 markdown files, ~91 000 words. The `concepts/` track is
  seven self-contained docs (~6 800 lines) consolidating theory +
  implementation per topic; `training.md` (~2 000 lines) and
  `inference.md` (~770 lines) are the applied pipeline chapters.
- **Structure:** four-track taxonomy present — `concepts/` (7),
  `references/` (config-and-api), `guides/` (getting-started, operations,
  learning-paths, glossary), pipeline docs `training.md` + `inference.md`,
  nav map `README.md`, plus `RECEIPTS.md` verification receipts and the
  interactive visual atlas (4 Archify HTML maps).
- **Machine gates:** `tests/test_doc_refs.py --strict-coverage` (anchor
  resolution **and** every public symbol in `models/`,
  `training/pretrain.py`, `inference/`, `utils/` must be cited) and
  `scripts/check_docs.py` (link/path/stale-pattern lint) — both green
  before this audit. The strict-coverage mode already exceeds the
  portfolio's `--coverage` baseline; no gate code changes were needed.
- **Symbol coverage:** strict coverage was already at 0 gaps; this
  upgrade's new docs add citations without breaking the invariant.
- **Honesty discipline:** the headline KV-cache number is a measured
  1.94×–2.0× with the reproducing benchmark named; the 128K passkey target
  is explicitly marked "target, no run yet"; the 8.0B-token A100 run is
  "planned", with 16–20 h labeled a target estimate.
- **Navigation:** three complementary entries — the 13-section linear
  walkthrough in [guides/getting-started.md](guides/getting-started.md),
  the reading-order table in [README.md](README.md), and the new
  three-tier audience paths in
  [guides/learning-paths.md](guides/learning-paths.md).

## 2. Findings table — docs↔code misalignment

| ID | Severity | Finding | Status |
|---|---|---|---|
| A1 | medium | `README.md` size table claimed 14 481 total lines vs 12 264 measured — stale after the 2026-08-04 consolidation | **fixed 2026-09-21** — regenerated via `scripts/check_docs.py --update-sizes` |
| A2 | low | no audience-routed learning paths (only the linear walkthrough) | **fixed 2026-09-21** — `guides/learning-paths.md` added |
| A3 | low | no glossary; notation and config-key semantics scattered across concepts + reference | **fixed 2026-09-21** — `guides/glossary.md` added (notation, attention/YaRN/MoE/training/inference/data terms, acronyms) |
| A4 | low | no `AUDIT.md` (this file) | **fixed 2026-09-21** |
| A5 | low | six stale "190 passed / 2 skipped" mentions (AGENTS.md, SKILLS.md ×2, inference.md, kernels-and-checkpointing.md ×2, moe.md) vs the measured 203 | **fixed 2026-09-21** — all six corrected to the 2026-09-21 measurement |
| A6 | info | `scripts/check_docs.py` has no `--coverage` flag under that name; the equivalent gate is `--check-symbols` (delegates to `test_doc_refs.py --strict-coverage`) | accepted — naming differs; the strict mode is stronger (enforced in CI via pytest) |
| A7 | info | passkey retrieval at 128K (≥85%) and the full 8.0B-token run are unmeasured targets | accepted — docs correctly mark them as targets pending the A100 run |

## 3. From-scratch codebase explanation (condensed map)

The repo implements the GPT-OSS long-context architecture end-to-end in
raw PyTorch:

- **Attention** (`models/attention.py:GPTOSSAttention`): 12 layers
  alternating sliding-window (128) and full attention; GQA 8Q/4KV via
  `models/attention.py:repeat_kv`; per-head learned sink bias clamped to
  `[-10, 15]` (`models/attention.py:SINK_CLAMP_MIN`,
  `models/attention.py:SINK_CLAMP_MAX`); single backend
  `models/attention.py:causal_attention` with cached masks
  (`models/attention.py:_window_mask`).
- **Position encoding** (`models/yarn.py:YaRNRoPE`): YaRN stretches the
  4K training window to 128K (`models/rotary.py:compute_yarn_freqs`,
  `models/rotary.py:compute_yarn_mscale`); global layers prune the first
  25% of RoPE dims (`models/attention.py:GPTOSSAttention._n_pruned_dims`).
- **MoE** (`models/moe.py:MoELayer`): top-2 of 8 routed experts
  (`models/moe.py:MoERouter`) + 1 shared `models/moe.py:SwiGLUExpert`;
  balance via the Switch-style `models/moe.py:aux_load_balancing_loss`
  (α = 0.01, FP32 softmax) — deliberately not the aux-loss-free gate.
  Opt-in fused kernel `models/moe_triton.py:triton_moe_w1w3_silu` with
  pure-PyTorch reference backward.
- **Trunk** (`models/transformer.py:GPTOSS`): typed
  `models/transformer.py:ModelConfig`, `models/transformer.py:RMSNorm`,
  weight tying, `models/transformer.py:GPTOSSBlock`.
- **Training** (`training/pretrain.py:main`): AdamW + warmup→cosine
  (`training/pretrain.py:make_warmup_cosine_lambda`), NaN guard with
  rollback, chunked CE (`training/pretrain.py:chunked_cross_entropy`),
  mmap shards (`training/pretrain.py:PretrainDataset`), atomic
  checkpoints (`utils/checkpoint.py:CheckpointManager`).
- **Inference** (`inference/generate.py:generate`): the
  `inference/generate.py:MixedKVCache` — ring buffer on windowed layers,
  full prefix on global layers — delivers the measured 1.94×–2.0× KV-cache
  reduction (`scripts/kv_cache_benchmark.py`); long-context recall via
  `inference/long_context.py:PasskeyEvaluator`.
- **Data:** workspace `shared_data/` pipeline consumed via the
  `data/prepare_data.py` shim.

## 4. Modification plan (priority order)

1. ~~Refresh stale size table~~ — **done** (A1).
2. ~~Add learning-paths + glossary + this audit~~ — **done** (A2–A4).
3. ~~Correct stale test-count comments~~ — **done** (A5).
4. **On the A100 run:** after the 8.0B-token pretrain, replace the passkey
   target and all timing estimates with measured numbers and re-stamp the
   verification footers.
5. **Optional next:** fold the OPT-1…24 catalog's verification receipts
   into `RECEIPTS.md` as a machine-checked table.

## 5. Acceptance criteria for "audit complete"

- [x] `tests/test_doc_refs.py --strict-coverage` green (anchors + symbol coverage).
- [x] `scripts/check_docs.py` green.
- [x] Four-track taxonomy complete: concepts / references / guides (incl.
      learning-paths + glossary) / training.md + inference.md.
- [x] Nav map carries a measured, dated size table (regenerated A1).
- [x] No unmeasured headline presented as measured (passkey and run
      duration marked as targets, A7).
- [x] Full default pytest green on CPU (203 passed / 2 skipped).

<!-- docs:verified 2026-09-21 · ada1459 -->
