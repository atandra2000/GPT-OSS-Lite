# GPT-OSS-Lite Documentation — Book Index

> **Start here.** This directory is the textbook-style reference for GPT-OSS-Lite:
> a faithful from-scratch PyTorch reproduction of OpenAI's GPT-OSS long-context
> architecture (~502M total / ~247M active parameters, 12-layer alternating
> sliding-window / full attention, YaRN 128K, top-2-of-8 MoE). For a one-page
> project overview see the root [README.md](../README.md).

The book has two layers: **concept chapters** (`documentation/*.md`) that walk
each component end-to-end, and **theory chapters** (`documentation/theory/*.md`)
that teach the underlying math from scratch — every equation derived, every
symbol anchored to `file.py:Symbol` in the codebase.

---

## Headline metrics

| Metric | Target | Script |
|--------|--------|--------|
| KV-cache reduction at 128K | ≥ 1.8× vs pure GQA full attention (**measured 2.00×**) | `scripts/kv_cache_benchmark.py` |
| Passkey retrieval at 128K | ≥ 85% (trained checkpoint — **target**, no run yet) | `scripts/passkey_eval.py` |

**Stack:** PyTorch 2.x, BF16, FA2 via SDPA, optional `torch.compile(max-autotune)`,
opt-in Triton MoE via `moe_dispatch: triton_grouped`.

**Authoritative project rules:** [AGENTS.md](../AGENTS.md) (sink clamp, aux loss,
sliding/full alternation, Triton contract).

---

## Learning path

Read in this order for a first pass. The theory layer is optional on a first
read — skip it, then come back per-topic.

| Step | Layer | Document | You'll learn |
|------|-------|----------|--------------|
| 0 | Guide | [getting_started.md](getting_started.md) | Clone, env, first commands |
| 1 | Concept | [foundations.md](foundations.md) | Why decoder-only, GQA, SWA, sinks, YaRN, MoE |
| 2 | Concept | [architecture.md](architecture.md) | 12-layer stack, `GPTOSS` / `ModelConfig`, file map |
| 3 | Concept | [ATTENTION_SINKS.md](ATTENTION_SINKS.md) | SWA/full alternation, learned sink bias, SDPA paths |
| 4 | Concept | [rope_yarn.md](rope_yarn.md) | Position encoding + 128K extrapolation |
| 5 | Concept | [moe.md](moe.md) | Top-2 routing, aux loss α=0.01, Triton opt-in |
| 6 | Concept | [training.md](training.md) | `pretrain.py`, schedules, NaN guard, YAML reference |
| 7 | Concept | [data_pipeline.md](data_pipeline.md) | Shards, tokenization, loader |
| 8 | Concept | [inference.md](inference.md) | `MixedKVCache`, `generate()` |
| 9 | Reference | [operations.md](operations.md) | Scripts, utils, OPT-1…24 catalog |
| 10 | Theory | [theory/attention_math.md](theory/attention_math.md) | Softmax, scaling, masks, SDPA/flash |
| 11 | Theory | [theory/positional_encodings.md](theory/positional_encodings.md) | Sinusoidal → RoPE → YaRN, from zero |
| 12 | Theory | [theory/moe_theory.md](theory/moe_theory.md) | Routing, aux-loss derivation, expert collapse |
| 13 | Theory | [theory/numerics.md](theory/numerics.md) | BF16/FP16/FP32/TF32 formats, autocast, clamps |
| 14 | Theory | [theory/optimizers.md](theory/optimizers.md) | Momentum → Adam → AdamW, bias correction, decay |
| 15 | Theory | [theory/autograd_checkpointing.md](theory/autograd_checkpointing.md) | Tape memory, recompute tradeoff |
| 16 | Theory | [theory/sampling.md](theory/sampling.md) | Temperature, top-k, top-p, entropy |
| 17 | Theory | [theory/kv_cache_engineering.md](theory/kv_cache_engineering.md) | Bandwidth, ring buffer, growth policy |
| 18 | Theory | [theory/tokenization_bpe.md](theory/tokenization_bpe.md) | BPE algorithm, 128K vocab economics |
| 19 | Theory | [theory/triton_programming.md](theory/triton_programming.md) | GPU tiling, Triton, the fused MoE kernel |

---

## Agent routing

| Question type | Read first |
|---|---|
| How does this repo implement X? | `models/*.py` + matching chapter |
| Sink bias / SWA / YaRN theory + impl | [ATTENTION_SINKS.md](ATTENTION_SINKS.md) |
| RoPE / YaRN | [rope_yarn.md](rope_yarn.md) + [theory/positional_encodings.md](theory/positional_encodings.md) |
| Attention math / SDPA / masks | [theory/attention_math.md](theory/attention_math.md) |
| MoE / Triton | [moe.md](moe.md) + [theory/moe_theory.md](theory/moe_theory.md) + [theory/triton_programming.md](theory/triton_programming.md) |
| YAML / train loop | [training.md](training.md) + [theory/optimizers.md](theory/optimizers.md) |
| Numerics / BF16 / clamps | [theory/numerics.md](theory/numerics.md) |
| Data / tokenizer / shards | [data_pipeline.md](data_pipeline.md) + [theory/tokenization_bpe.md](theory/tokenization_bpe.md) |
| Inference / KV cache / sampling | [inference.md](inference.md) + [theory/kv_cache_engineering.md](theory/kv_cache_engineering.md) + [theory/sampling.md](theory/sampling.md) |
| Scripts / utils / OPT-* | [operations.md](operations.md) |
| What must not break? | [../AGENTS.md](../AGENTS.md) + [architecture.md](architecture.md) invariants |
| Onboarding | [getting_started.md](getting_started.md) |

---

## Component docs

| Component | Source | Chapter | Theory |
|-----------|--------|---------|--------|
| Top-level model | `models/transformer.py` | [architecture.md](architecture.md) | [optimizers](theory/optimizers.md), [autograd_checkpointing](theory/autograd_checkpointing.md) |
| Attention (SWA + full) | `models/attention.py` | [ATTENTION_SINKS.md](ATTENTION_SINKS.md) | [attention_math](theory/attention_math.md) |
| RoPE helpers | `models/rotary.py` | [rope_yarn.md](rope_yarn.md) | [positional_encodings](theory/positional_encodings.md) |
| YaRN scaling | `models/yarn.py` | [rope_yarn.md](rope_yarn.md) | [positional_encodings](theory/positional_encodings.md) |
| MoE FFN | `models/moe.py` | [moe.md](moe.md) | [moe_theory](theory/moe_theory.md) |
| Triton MoE kernel | `models/moe_triton.py` | [moe.md](moe.md#sanctioned-triton-path-moe_dispatchtriton_grouped) | [triton_programming](theory/triton_programming.md) |
| Training loop | `training/pretrain.py` | [training.md](training.md) | [optimizers](theory/optimizers.md), [numerics](theory/numerics.md) |
| Generation | `inference/generate.py` | [inference.md](inference.md) | [sampling](theory/sampling.md), [kv_cache_engineering](theory/kv_cache_engineering.md) |
| Long-context eval | `inference/long_context.py` | [inference.md](inference.md) | [sampling](theory/sampling.md) |
| Checkpoints / logging / VRAM | `utils/` | [operations.md](operations.md) | [autograd_checkpointing](theory/autograd_checkpointing.md) |
| Production config | `configs/pretrain_a100_502m.yaml` | [training.md](training.md#part-b--configuration-reference) | — |
| Data pipeline | `data/` | [data_pipeline.md](data_pipeline.md) | [tokenization_bpe](theory/tokenization_bpe.md) |

---

## Maintaining documentation

```bash
# Lint links, stale patterns, and LaTeX separators (all documentation/, incl. theory/)
python3 scripts/check_docs.py

# Enforce doc-code alignment: every `file.py:Symbol` anchor must resolve AND
# every public symbol in models/, training/, inference/, utils/ must be anchored
python3 scripts/check_docs.py --check-symbols
#   (equivalent to: python3 tests/test_doc_refs.py --strict-coverage)

# Refresh line counts in the size table below
python3 scripts/check_docs.py --update-sizes

# Stamp verification footers (git short hash)
python3 scripts/check_docs.py --stamp-footers

# Coverage map (module -> anchored symbols) for the README component table
python3 scripts/generate_code_map.py
```

Rules for writers: anchors are `file.py:Symbol` (never line numbers); block
math uses `$$...$$`; every quantitative claim is derived or marked `[INFERENCE]`;
`tests/test_doc_refs.py` fails on any stale anchor. The 2026-08-04 massive
expansion is complete and stamped; this index is the in-repo writing contract.

Every chapter file ends with a verification footer:

```html
<!-- docs:verified YYYY-MM-DD · <commit> -->
```

---

## Doc size reference

| Doc | ~Lines | Status |
|---|---|---|
| operations.md | 1,643 | Comprehensive |
| training.md | 1,408 | Comprehensive |
| architecture.md | 1,240 | Comprehensive |
| moe.md | 1,134 | Comprehensive |
| ATTENTION_SINKS.md | 1,116 | Comprehensive |
| data_pipeline.md | 915 | Comprehensive |
| theory/positional_encodings.md | 822 | Comprehensive |
| rope_yarn.md | 763 | Comprehensive |
| foundations.md | 725 | Comprehensive |
| theory/attention_math.md | 664 | Comprehensive |
| theory/kv_cache_engineering.md | 649 | Comprehensive |
| inference.md | 517 | Comprehensive |
| theory/optimizers.md | 506 | Comprehensive |
| theory/autograd_checkpointing.md | 483 | Comprehensive |
| theory/sampling.md | 454 | Comprehensive |
| theory/moe_theory.md | 315 | Comprehensive |
| getting_started.md | 309 | Comprehensive |
| theory/numerics.md | 300 | Comprehensive |
| theory/triton_programming.md | 289 | Comprehensive |
| theory/tokenization_bpe.md | 229 | Comprehensive |
| **Total** | **14,481** | |





*(Theory chapters are listed here after `--update-sizes`.)*

---

## External index

| Resource | Location |
|----------|----------|
| Project README | [../README.md](../README.md) |
| Agent rules | [../AGENTS.md](../AGENTS.md) |
| Workflows | [../SKILLS.md](../SKILLS.md) |
| LLM architecture skill | `../../.agents/skills/llm-architecture/SKILL.md` |

---

<!-- docs:verified 2026-08-04 · 5da1a80 -->
