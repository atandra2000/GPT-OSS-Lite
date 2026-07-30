# GPT-OSS-Lite — Documentation Index

Educational technical references for every component of this project. Each doc follows a consistent structure: theory, math, implementation walkthrough, invariants, and cross-links.

**New here?** Start with [foundations.md](foundations.md) → [getting_started.md](getting_started.md) → [architecture.md](architecture.md).

---

## Learning Path

| Phase | Read | Learn |
|---|---|---|
| 0. Foundations | [foundations.md](foundations.md) | GQA, SWA, sinks, YaRN, Chinchilla |
| 1. Overview | [getting_started.md](getting_started.md) | Key numbers, smoke tests, pitfalls |
| 2. Big picture | [architecture.md](architecture.md) | How all components connect |
| 3. Attention | [ATTENTION_SINKS.md](ATTENTION_SINKS.md), [attention.md](attention.md) | Sinks, SWA/full alt, GQA |
| 4. MoE | [moe.md](moe.md) | Top-2-of-8, standard aux loss |
| 5. Long context | [yarn.md](yarn.md), [rotary.md](rotary.md) | YaRN extrapolation, pruned RoPE |
| 6. Wiring | [transformer.md](transformer.md) | Layer stack, weight tying |
| 7. Training | [training.md](training.md) | Pretrain loop, NaN guard |
| 8. Data | [data_pipeline.md](data_pipeline.md) | 8.0B-token corpus |
| 9. Inference | [inference.md](inference.md) | Mixed KV cache, passkey eval |
| 10. Ops | [configs.md](configs.md), [scripts.md](scripts.md), [utils.md](utils.md) | YAML, benchmarks, checkpoints |
| 11. Quality | [testing.md](testing.md) | Test corpus as oracle |
| 12. Advanced | [triton_kernels.md](triton_kernels.md), [OPTIMIZATIONS.md](OPTIMIZATIONS.md) | Fused MoE kernel, perf audit |

---

## Documentation tiers

| Tier | Files | When to read |
|---|---|---|
| **Essential** | [getting_started.md](getting_started.md), [architecture.md](architecture.md), [configs.md](configs.md), [testing.md](testing.md) | First run, debugging, YAML tuning |
| **Deep dives** | [ATTENTION_SINKS.md](ATTENTION_SINKS.md), [attention.md](attention.md), [moe.md](moe.md), [yarn.md](yarn.md), [foundations.md](foundations.md) | GPT-OSS mechanisms in depth |
| **Operations** | [training.md](training.md), [data_pipeline.md](data_pipeline.md), [inference.md](inference.md), [scripts.md](scripts.md), [utils.md](utils.md) | Full training run, data, checkpoints |
| **Advanced** | [triton_kernels.md](triton_kernels.md), [OPTIMIZATIONS.md](OPTIMIZATIONS.md), [transformer.md](transformer.md) | Kernel opt-in, perf, wiring internals |

---

## Component Docs

| File | Component(s) | Source |
|------|--------------|--------|
| [ATTENTION_SINKS.md](ATTENTION_SINKS.md) | Sink bias + SWA + YaRN theory | `models/attention.py` |
| [attention.md](attention.md) | SWA, full, GQA, sink implementation | `models/attention.py` |
| [moe.md](moe.md) | Top-2 MoE + aux loss | `models/moe.py` |
| [yarn.md](yarn.md) | YaRN RoPE scaling | `models/yarn.py` |
| [rotary.md](rotary.md) | RoPE helpers, prune | `models/rotary.py` |
| [transformer.md](transformer.md) | Top-level wiring | `models/transformer.py` |
| [training.md](training.md) | Pretrain loop | `training/pretrain.py` |
| [inference.md](inference.md) | Generate + passkey | `inference/` |
| [data_pipeline.md](data_pipeline.md) | 8.0B-token pipeline | `data/prepare_data.py` |
| [utils.md](utils.md) | Checkpoint, memory, logging | `utils/` |
| [triton_kernels.md](triton_kernels.md) | Fused MoE kernel | `models/moe_triton.py` |

---

## Operations Docs

| File | Purpose |
|------|---------|
| [getting_started.md](getting_started.md) | Onboarding, commands, pitfalls |
| [architecture.md](architecture.md) | System diagram, data flows, file map |
| [configs.md](configs.md) | YAML key reference |
| [scripts.md](scripts.md) | Benchmarks, smoke tests, headline metrics |
| [testing.md](testing.md) | Test suite guide + load-bearing tests |

---

## Configs

| Config | Purpose |
|---|---|
| `configs/pretrain_a100_502m.yaml` | Canonical Chinchilla-optimal recipe, ~502M params, 1× A100 80GB |

See [configs.md](configs.md) for every key.

---

## ATTENTION_SINKS reference

**[ATTENTION_SINKS.md](ATTENTION_SINKS.md)** is the single canonical doc for sink bias, sliding-window alternation, and YaRN theory. If prose and code disagree, **`models/attention.py` wins**.

---

## Load-bearing invariants (do not break)

| Invariant | Doc |
|---|---|
| Even layers = SWA(128), odd = full | [attention.md](attention.md) |
| Sink bias clamped `[-10, 15]` at forward | [ATTENTION_SINKS.md](ATTENTION_SINKS.md) |
| Standard aux loss (α=0.01), not aux-loss-free | [moe.md](moe.md) |
| Weight tying — head.weight = embed.weight | [transformer.md](transformer.md) |
| `moe_dispatch="stacked"` by default | [triton_kernels.md](triton_kernels.md) |
| NaN guard with checkpoint rollback | [training.md](training.md) |
| YaRN scale = 32 (128K / 4K) | [yarn.md](yarn.md) |

---

## Authoritative top-level references

- [`../AGENTS.md`](../AGENTS.md) — subagent rules and hard constraints
- [`../SKILLS.md`](../SKILLS.md) — project-local workflows
- [`../CONTEXT.md`](../CONTEXT.md) — agent working snapshot
- [`../README.md`](../README.md) — public project summary

---

## Doc size reference

| Doc | ~Lines | Status |
|---|---|---|
| ATTENTION_SINKS.md | 542 | Comprehensive |
| OPTIMIZATIONS.md | 521 | Comprehensive |
| moe.md | 426 | Comprehensive |
| training.md | 409 | Comprehensive |
| utils.md | 390 | Comprehensive |
| attention.md | 370 | Comprehensive |
| data_pipeline.md | 353 | Comprehensive |
| inference.md | 350 | Comprehensive |
| rotary.md | 266 | Comprehensive |
| yarn.md | 264 | Comprehensive |
| getting_started.md | 240 | Comprehensive |
| architecture.md | 214 | Comprehensive |
| foundations.md | 149 | Comprehensive |
| scripts.md | 147 | Comprehensive |
| configs.md | 131 | Comprehensive |
| transformer.md | 129 | Comprehensive |
| testing.md | 107 | Comprehensive |
| triton_kernels.md | 101 | Comprehensive |
| **Total** | **5,109** | |


<!-- docs:verified 2026-07-31 · fd4fe36 -->
