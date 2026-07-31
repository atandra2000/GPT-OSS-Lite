# GPT-OSS-Lite Documentation — Book Index

> **Start here.** This directory is the textbook-style reference for GPT-OSS-Lite:
> a faithful from-scratch PyTorch reproduction of OpenAI's GPT-OSS long-context
> architecture (~502M total / ~247M active parameters, 12-layer alternating
> sliding-window / full attention, YaRN 128K, top-2-of-8 MoE). For a one-page
> project overview see the root [README.md](../README.md).

---

## Headline metrics

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
| 2 | Architecture | [architecture.md](architecture.md) | 12-layer stack, `GPTOSS` / `ModelConfig`, file map |
| 3 | Attention | [ATTENTION_SINKS.md](ATTENTION_SINKS.md) | SWA/full alternation, learned sink bias, SDPA paths |
| 4 | RoPE / YaRN | [rope_yarn.md](rope_yarn.md) | Position encoding + 128K extrapolation |
| 5 | MoE | [moe.md](moe.md) | Top-2 routing, aux loss α=0.01, Triton opt-in |
| 6 | Training | [training.md](training.md) | `pretrain.py`, schedules, NaN guard, YAML reference |
| 7 | Data | [data_pipeline.md](data_pipeline.md) | Shards, tokenization, loader |
| 8 | Inference | [inference.md](inference.md) | `MixedKVCache`, `generate()` |
| 9 | Operations | [operations.md](operations.md) | Scripts, utils, OPT-1…24 catalog |

---

## Agent routing

| Question type | Read first |
|---|---|
| How does this repo implement X? | `models/*.py` + matching chapter |
| Sink bias / SWA / YaRN theory + impl | [ATTENTION_SINKS.md](ATTENTION_SINKS.md) |
| RoPE / YaRN | [rope_yarn.md](rope_yarn.md) |
| MoE / Triton | [moe.md](moe.md) |
| YAML / train loop | [training.md](training.md) |
| Data | [data_pipeline.md](data_pipeline.md) |
| Inference / KV cache | [inference.md](inference.md) |
| Scripts / utils / OPT-* | [operations.md](operations.md) |
| What must not break? | [../AGENTS.md](../AGENTS.md) + [architecture.md](architecture.md) invariants |
| Onboarding | [getting_started.md](getting_started.md) |

---

## Component docs

| Component | Source | Documentation |
|-----------|--------|---------------|
| Top-level model | `models/transformer.py` | [architecture.md](architecture.md) |
| Attention (SWA + full) | `models/attention.py` | [ATTENTION_SINKS.md](ATTENTION_SINKS.md) |
| RoPE helpers | `models/rotary.py` | [rope_yarn.md](rope_yarn.md) |
| YaRN scaling | `models/yarn.py` | [rope_yarn.md](rope_yarn.md) |
| MoE FFN | `models/moe.py` | [moe.md](moe.md) |
| Triton MoE kernel | `models/moe_triton.py` | [moe.md](moe.md#sanctioned-triton-path-moe_dispatchtriton_grouped) |
| Training loop | `training/pretrain.py` | [training.md](training.md) |
| Generation | `inference/generate.py` | [inference.md](inference.md) |
| Long-context eval | `inference/long_context.py` | [inference.md](inference.md) |
| Checkpoints / logging / VRAM | `utils/` | [operations.md](operations.md) |
| Production config | `configs/pretrain_a100_502m.yaml` | [training.md](training.md#part-b--configuration-reference) |
| Data pipeline | `data/` | [data_pipeline.md](data_pipeline.md) |

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
| *(run `python3 scripts/check_docs.py --update-sizes` to regenerate)* | | |

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
