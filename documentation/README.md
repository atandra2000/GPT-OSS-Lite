# GPT-OSS-Lite Documentation — Book Index

> **Start here.** This directory is a textbook-style reference for the GPT-OSS-Lite
> reproduction: ~502M total parameters, ~247M active, 12-layer alternating
> sliding-window / full attention, YaRN 128K, top-2-of-8 MoE. For a one-page
> project overview see the root [README.md](../README.md).

---

## Overview

GPT-OSS-Lite is a faithful from-scratch PyTorch implementation of the GPT-OSS
long-context architecture. The documentation is organized as a **progressive
book**: foundations and motivation first, then system design, then component
deep-dives, then operations (training, data, scripts, utilities).

**Headline metrics** (verify with scripts, not prose):

| Metric | Target | Script |
|--------|--------|--------|
| KV-cache reduction at 128K | ≥ 1.8× vs pure GQA full attention | `scripts/kv_cache_benchmark.py` |
| Passkey retrieval at 128K | ≥ 85% (trained checkpoint) | `scripts/passkey_eval.py` |

**Stack:** PyTorch 2.x, BF16, FA2 via SDPA, optional `torch.compile(max-autotune)`,
opt-in Triton MoE via `moe_dispatch: triton_grouped`.

**Authoritative project rules:** [AGENTS.md](../AGENTS.md) (sink clamp, aux loss,
sliding/full alternation, Triton contract).

---

## Learning path

Read in this order for a first pass. Skip ahead if you already know transformers.

| Step | Chapter | Document | You'll learn |
|------|---------|----------|--------------|
| 0 | Onboarding | [getting_started.md](getting_started.md) | Clone, env, first commands |
| 1 | Foundations | [foundations.md](foundations.md) | Why decoder-only, GQA, SWA, sinks, YaRN, MoE |
| 2 | Architecture | [architecture.md](architecture.md) | 12-layer stack, file map, dataflow |
| 3 | Transformer | [transformer.md](transformer.md) | `GPTOSS`, `GPTOSSBlock`, `ModelConfig` |
| 4 | Attention | [ATTENTION_SINKS.md](ATTENTION_SINKS.md) | SWA/full alternation, sinks, SDPA paths |
| 4b | Sinks (deep) | [ATTENTION_SINKS.md](ATTENTION_SINKS.md) | Learned sink bias — read before tuning sinks |
| 5 | RoPE / YaRN | [rotary.md](rotary.md) → [yarn.md](yarn.md) | Position encoding + 128K extrapolation |
| 6 | MoE | [moe.md](moe.md) | Top-2 routing, aux loss α=0.01 |
| 6b | Triton (opt-in) | [triton_kernels.md](triton_kernels.md) | `moe_dispatch` kernel contract |
| 7 | Training | [training.md](training.md) | `pretrain.py`, schedules, NaN guard |
| 7b | Config | [configs.md](configs.md) | YAML reference |
| 7c | Data | [data_pipeline.md](data_pipeline.md) | Shards, tokenization, loader |
| 8 | Inference | [inference.md](inference.md) | `MixedKVCache`, `generate()` |
| 9 | Optimizations | [OPTIMIZATIONS.md](OPTIMIZATIONS.md) | OPT-1…24 catalog |
| 10 | Scripts | [scripts.md](scripts.md) | Benchmarks and profilers |
| 11 | Utilities | [utils.md](utils.md) | Checkpoints, logging, VRAM estimator |

---

## Doc tiers

### Tier 1 — Conceptual (read once)

- [foundations.md](foundations.md) — mathematical and architectural motivation
- [architecture.md](architecture.md) — system map tying modules together

### Tier 2 — Component reference (read as needed)

- [architecture.md](architecture.md), [ATTENTION_SINKS.md](ATTENTION_SINKS.md)
- [rotary.md](rotary.md), [yarn.md](yarn.md)
- [moe.md](moe.md), [triton_kernels.md](triton_kernels.md)

### Tier 3 — Operations (read when running experiments)

- [getting_started.md](getting_started.md), [configs.md](configs.md)
- [training.md](training.md), [data_pipeline.md](data_pipeline.md)
- [inference.md](inference.md)
- [scripts.md](scripts.md), [utils.md](utils.md)
- [OPTIMIZATIONS.md](OPTIMIZATIONS.md)

---

## Component docs

| Component | Source | Documentation |
|-----------|--------|---------------|
| Top-level model | `models/transformer.py` | [transformer.md](transformer.md) |
| Attention (SWA + full) | `models/attention.py` | [ATTENTION_SINKS.md](ATTENTION_SINKS.md) |
| RoPE helpers | `models/rotary.py` | [rotary.md](rotary.md) |
| YaRN scaling | `models/yarn.py` | [yarn.md](yarn.md) |
| MoE FFN | `models/moe.py` | [moe.md](moe.md) |
| Triton MoE kernel | `models/moe_triton.py` | [triton_kernels.md](triton_kernels.md) |
| Training loop | `training/pretrain.py` | [training.md](training.md) |
| Generation | `inference/generate.py` | [inference.md](inference.md) |
| Long-context eval | `inference/long_context.py` | [inference.md](inference.md) |
| Checkpoints | `utils/checkpoint.py` | [utils.md](utils.md) |
| Logging | `utils/logging.py` | [utils.md](utils.md) |
| VRAM estimator | `utils/memory.py` | [utils.md](utils.md) |
| Production config | `configs/pretrain_a100_502m.yaml` | [configs.md](configs.md) |
| Data pipeline | `data/` | [data_pipeline.md](data_pipeline.md) |

---

## Cross-cutting topics

| Question | Go to |
|----------|-------|
| Why alternating SWA/full? | [foundations.md](foundations.md) §4, [architecture.md](architecture.md) §4 |
| How does sink bias work? | [ATTENTION_SINKS.md](ATTENTION_SINKS.md) |
| What is `moe_dispatch`? | [moe.md](moe.md), [triton_kernels.md](triton_kernels.md) (YAML opt-in, not env vars) |
| Memory at B=8, T=4096? | [utils.md](utils.md), `scripts/microbench_a100.py` |
| All performance knobs? | [OPTIMIZATIONS.md](OPTIMIZATIONS.md) |
| How to run benchmarks? | [scripts.md](scripts.md) |
| Doc lint / stale patterns? | `scripts/check_docs.py` |

---

## Maintaining documentation

```bash
# Lint links and stale patterns
python3 scripts/check_docs.py

# Refresh line counts in the table below
python3 scripts/check_docs.py --update-sizes

# Stamp verification footers (git short hash)
python3 scripts/check_docs.py --stamp-footers
```

Every chapter file ends with a verification footer:

```html
<!-- docs:verified YYYY-MM-DD · <commit> -->
```

---

## Doc size reference

| Doc | ~Lines | Status |
|---|---|---|
| training.md | 834 | Comprehensive |
| moe.md | 754 | Comprehensive |
| data_pipeline.md | 735 | Comprehensive |
| ATTENTION_SINKS.md | ~1200 | Comprehensive (theory + implementation) |
| OPTIMIZATIONS.md | 712 | Comprehensive |
| architecture.md | 650 | Comprehensive |
| triton_kernels.md | 628 | Comprehensive |
| foundations.md | 606 | Comprehensive |
| inference.md | 549 | Comprehensive |
| transformer.md | 532 | Comprehensive |
| utils.md | 527 | Comprehensive |
| scripts.md | 520 | Comprehensive |
| yarn.md | 474 | Comprehensive |
| getting_started.md | 454 | Comprehensive |
| rotary.md | 396 | Comprehensive |
| configs.md | 379 | Comprehensive |
| **Total** | **10,134** | |




Run `python3 scripts/check_docs.py --update-sizes` to refresh this table and
add the **Total** row automatically.

---

## External index

| Resource | Location |
|----------|----------|
| Project README | [../README.md](../README.md) |
| Agent rules | [../AGENTS.md](../AGENTS.md) |
| Workflows | [../SKILLS.md](../SKILLS.md) |
| LLM architecture skill | `../../.agents/skills/llm-architecture/SKILL.md` |

---

<!-- docs:verified 2026-07-31 · fa6f918 -->
