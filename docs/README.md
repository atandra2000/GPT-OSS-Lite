# GPT-OSS-Lite — Documentation Index

> **Start here.** This directory is the canonical reference for GPT-OSS-Lite: a faithful from-scratch PyTorch reproduction of OpenAI's GPT-OSS long-context architecture (~502M total / ~247M active parameters, 12-layer alternating sliding-window / full attention, YaRN 128K, top-2-of-8 MoE). For a one-page project overview see the root [README.md](../README.md).

The documentation is organized into four layers: **concept chapters** (`docs/concepts/`) that consolidate theory + implementation per topic, a **reference** (`docs/references/`) for config tables and API signatures, **guides** (`docs/guides/`) for onboarding and operations, and two top-level chapters for training and inference. Every code symbol is cited as a machine-verified `file.py:Symbol` anchor (see `tests/test_doc_refs.py`).

---

## Headline metrics

| Metric | Target | Script |
|--------|--------|--------|
| KV-cache reduction at 128K | ≥ 1.8× vs pure GQA full attention (**measured 2.00×**) | `scripts/kv_cache_benchmark.py` |
| Passkey retrieval at 128K | ≥ 85% (trained checkpoint — **target**, no run yet) | `scripts/passkey_eval.py` |

**Stack:** PyTorch 2.x, BF16, FA2 via SDPA, optional `torch.compile(max-autotune)`, opt-in Triton MoE via `moe_dispatch: triton_grouped`.

**Authoritative project rules:** [AGENTS.md](../AGENTS.md) (sink clamp, aux loss, sliding/full alternation, Triton contract).

---

## Learning path

Read in this order for a first pass.

| Step | Layer | Document | You'll learn |
|------|-------|----------|--------------|
| 0 | Guide | [getting-started.md](guides/getting-started.md) | Clone, env, first commands |
| 1 | Concept | [foundations-and-architecture.md](concepts/foundations-and-architecture.md) | Why decoder-only, GQA, SWA, sinks, YaRN, MoE; system diagram, `GPTOSS` / `ModelConfig` |
| 2 | Concept | [attention-sinks.md](concepts/attention-sinks.md) | SWA/full alternation, learned sink bias, SDPA paths |
| 3 | Concept | [attention-and-positional.md](concepts/attention-and-positional.md) | Attention math + position encoding (RoPE → YaRN 128K) |
| 4 | Concept | [moe.md](concepts/moe.md) | Top-2 routing, aux loss α=0.01, Triton opt-in |
| 5 | Chapter | [training.md](training.md) | `pretrain.py`, schedules, NaN guard, data pipeline, YAML reference |
| 6 | Chapter | [inference.md](inference.md) | `MixedKVCache`, `generate()`, KV-cache engineering |
| 7 | Reference | [config-and-api.md](references/config-and-api.md) | Config tables + key API signatures |
| 8 | Guide | [operations.md](guides/operations.md) | Scripts, utils, OPT-1…24 catalog |
| 9 | Concept | [kernels-and-checkpointing.md](concepts/kernels-and-checkpointing.md) | GPU execution model, Triton, gradient checkpointing |
| 10 | Concept | [optimizers-and-numerics.md](concepts/optimizers-and-numerics.md) | Optimizers, BF16/FP16/TF32 formats, sampling |
| 11 | Concept | [tokenization.md](concepts/tokenization.md) | BPE algorithm, 128K vocab economics |

---

## Agent routing

| Question type | Read first |
|---|---|
| How does this repo implement X? | `models/*.py` + matching concept chapter |
| Sink bias / SWA / YaRN theory + impl | [attention-sinks.md](concepts/attention-sinks.md) |
| RoPE / YaRN / attention math | [attention-and-positional.md](concepts/attention-and-positional.md) |
| MoE / Triton | [moe.md](concepts/moe.md) + [kernels-and-checkpointing.md](concepts/kernels-and-checkpointing.md) |
| YAML / train loop / data | [training.md](training.md) |
| Numerics / BF16 / clamps / optimizers / sampling | [optimizers-and-numerics.md](concepts/optimizers-and-numerics.md) |
| Tokenizer / BPE | [tokenization.md](concepts/tokenization.md) |
| Inference / KV cache / passkey | [inference.md](inference.md) |
| Scripts / utils / OPT-* | [operations.md](guides/operations.md) |
| Config tables / API signatures | [config-and-api.md](references/config-and-api.md) |
| What must not break? | [AGENTS.md](../AGENTS.md) + [foundations-and-architecture.md](concepts/foundations-and-architecture.md) invariants |
| Onboarding | [getting-started.md](guides/getting-started.md) |

---

## Component docs

| Component | Source | Chapter |
|-----------|--------|---------|
| Top-level model | `models/transformer.py` | [foundations-and-architecture.md](concepts/foundations-and-architecture.md) |
| Attention (SWA + full) | `models/attention.py` | [attention-sinks.md](concepts/attention-sinks.md) |
| RoPE helpers | `models/rotary.py` | [attention-and-positional.md](concepts/attention-and-positional.md) |
| YaRN scaling | `models/yarn.py` | [attention-and-positional.md](concepts/attention-and-positional.md) |
| MoE FFN | `models/moe.py` | [moe.md](concepts/moe.md) |
| Triton MoE kernel | `models/moe_triton.py` | [moe.md](concepts/moe.md) + [kernels-and-checkpointing.md](concepts/kernels-and-checkpointing.md) |
| Training loop | `training/pretrain.py` | [training.md](training.md) |
| Generation | `inference/generate.py` | [inference.md](inference.md) |
| Long-context eval | `inference/long_context.py` | [inference.md](inference.md) |
| Checkpoints / logging / VRAM | `utils/` | [operations.md](guides/operations.md) |
| Production config | `configs/pretrain_a100_502m.yaml` | [config-and-api.md](references/config-and-api.md) + [training.md](training.md) |
| Data pipeline | `data/` | [training.md](training.md) + [tokenization.md](concepts/tokenization.md) |

---

## Maintaining documentation

```bash
# Lint links, stale patterns, and control characters (all docs/, incl. subdirs)
python3 scripts/check_docs.py

# Enforce doc-code alignment: every `file.py:Symbol` anchor must resolve AND
# every public symbol in models/, training/, inference/, utils/ must be anchored
python3 scripts/check_docs.py --check-symbols
#   (equivalent to: python3 tests/test_doc_refs.py --strict-coverage)

# Refresh line counts in the size table below
python3 scripts/check_docs.py --update-sizes

# Stamp verification footers (git short hash)
python3 scripts/check_docs.py --stamp-footers
```

Rules for writers: anchors are `file.py:Symbol` (never line numbers); block
math uses `$$...$$`; every quantitative claim is derived or marked `[INFERENCE]`;
`tests/test_doc_refs.py` fails on any stale anchor. The 2026-08-04 massive expansion is complete and consolidated into this canonical layout.

Every chapter file ends with a verification footer:

```html
<!-- docs:verified YYYY-MM-DD · <commit> -->
```

---

## Doc size reference

| Doc | ~Lines | Status |
|---|---|---|
| training.md | 2,240 | Comprehensive |
| concepts/attention-and-positional.md | 2,224 | Comprehensive |
| concepts/foundations-and-architecture.md | 1,960 | Comprehensive |
| guides/operations.md | 1,652 | Comprehensive |
| concepts/moe.md | 1,408 | Comprehensive |
| concepts/optimizers-and-numerics.md | 1,237 | Comprehensive |
| inference.md | 1,157 | Comprehensive |
| concepts/attention-sinks.md | 1,126 | Comprehensive |
| concepts/kernels-and-checkpointing.md | 755 | Comprehensive |
| guides/getting-started.md | 320 | Comprehensive |
| concepts/tokenization.md | 236 | Comprehensive |
| references/config-and-api.md | 166 | Comprehensive |
| **Total** | **14,481** | |





*(Sizes are refreshed by `scripts/check_docs.py --update-sizes`.)*

---

## External index

| Resource | Location |
|----------|----------|
| Project README | [README.md](../README.md) |
| Agent rules | [AGENTS.md](../AGENTS.md) |
| Workflows | [SKILLS.md](../SKILLS.md) |
| LLM architecture skill | `../../.agents/skills/llm-architecture/SKILL.md` |

<!-- docs:verified 2026-08-05 · 6491066 -->
