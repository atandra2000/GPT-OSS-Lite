# Architecture Overview

> **Purpose:** A single map of how every component in GPT-OSS-Lite fits together. Read this after [foundations.md](foundations.md) and before diving into component-specific docs.

---

## Prerequisites

- Causal language modeling — [foundations.md](foundations.md) §1
- Pre-norm residual blocks — [foundations.md](foundations.md) §3
- Chinchilla token budget — [foundations.md](foundations.md) §10

---

## Table of Contents

1. [Design Goals](#design-goals)
2. [System Diagram](#system-diagram)
3. [Layer Topology](#layer-topology)
4. [Data Flow — Training](#data-flow--training)
5. [Data Flow — Inference](#data-flow--inference)
6. [Parameter Budget](#parameter-budget)
7. [Memory Budget](#memory-budget)
8. [File Map](#file-map)
9. [Config → Code Routing](#config--code-routing)
10. [Load-Bearing Invariants](#load-bearing-invariants)
11. [Further Reading](#further-reading)

---

## Design Goals

1. **Faithful GPT-OSS reproduction** — sliding/full alternation, learned sinks, YaRN, top-2 MoE.
2. **Raw PyTorch** — no HuggingFace Trainer. Every line is inspectable.
3. **Single-GPU training** — 1× A100 80GB, Chinchilla-optimal 8.0B tokens.
4. **CPU-testable** — all correctness tests run without CUDA or Triton.
5. **Optional Triton** — fused MoE W1/W3+silu kernel (opt-in via config).

---

## System Diagram

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         TRAINING PATH                                   │
├─────────────────────────────────────────────────────────────────────────┤
│  data/pretrain_chinchilla/shard_*.bin (uint32 tokens)                   │
│         │                                                               │
│         ▼                                                               │
│  PretrainDataset ──► DataLoader ──► pretrain.py train loop              │
│                                         │                               │
│                    ┌────────────────────┼────────────────────┐          │
│                    ▼                    ▼                    ▼          │
│              GPTOSS.forward      AdamW (FP32 master)    NaN guard       │
│                    │                    │               rollback        │
│         ┌──────────┼──────────┐        LR scheduler                    │
│         ▼          ▼          ▼                                        │
│    Embedding   12× Block   LM Head (tied)                              │
│                    │                                                    │
│              ┌─────┴─────┐                                             │
│              ▼           ▼                                             │
│         Attention      MoE                                             │
│      (SWA / full alt)  top-2-of-8                                      │
│                                                                         │
│  CheckpointManager ──► model_step_N.safetensors + rng_step_N.pt         │
└─────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────┐
│                        INFERENCE PATH                                   │
├─────────────────────────────────────────────────────────────────────────┤
│  Prompt tokens ──► generate() with MixedKVCache                         │
│                         │                                               │
│                    ring buffer (SWA layers, window=128)                 │
│                    exponential buffer (global layers, O(T) growth)      │
│                         │                                               │
│                         ▼                                               │
│                    next-token logits                                    │
│                                                                         │
│  PasskeyEvaluator (long_context.py) ──► 128K retrieval metric           │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## Layer Topology

12 layers, **alternating** attention pattern:

| Layer index | Attention | KV cache at decode |
|---:|---|---|
| 0, 2, 4, 6, 8, 10 | Sliding window (128) | Ring buffer, O(window) |
| 1, 3, 5, 7, 9, 11 | Full causal | Full sequence, O(T) |

Set in `GPTOSSAttention.__init__`: `self.is_windowed = (layer_idx % 2 == 0)`.

Global (odd) layers optionally apply **pruned RoPE** (25% of dims zeroed) when `yarn_prune_rope_global=True`.

Every layer uses the same **MoELayer** (top-2 of 8 routed + 1 shared).

---

## Data Flow — Training

```
input_ids (B, T) ──► embed ──► for layer in blocks:
                                  x += attn(norm(x), positions)
                                  x += moe(norm(x))
                              ──► norm ──► head ──► logits (B, T, V)
aux_loss = mean over layers of MoE load-balancing loss
loss = chunked_CE(logits, targets) + α * aux_loss
```

Positions: `torch.arange(T)` during training (YaRN frequencies computed from absolute positions).

---

## Data Flow — Inference

```
prefill: full sequence through all layers, populate MixedKVCache (rotated K)
decode:  one token per step, append to per-layer cache
         windowed layers: ring buffer overwrite
         global layers: exponential growth buffer
```

See [inference.md](inference.md) for ring-buffer and rotated-K details.

---

## Parameter Budget

| Component | ~Params | Notes |
|---|---:|---|
| Embedding + tied head | ~98M | Counted once with weight tying |
| Attention (12 layers) | ~85M | GQA projections + sink bias |
| MoE FFN (12 layers) | ~310M | 8 experts × 3 matrices + shared |
| Norms + router | ~9M | |
| **Total** | **~502M** | |
| **Active / step** | **~247M** | Top-2 experts + shared only |

Verified by `test_anchor_metric_502m_total` and `test_anchor_metric_247m_active`.

---

## Memory Budget

At 128K context, BF16, batch=1:

| Layout | KV cache | Reduction |
|---|---:|---:|
| Pure GQA (all full) | ~2.25 GB | 1.00× |
| SWA(128)/full alt | ~1.13 GB | **2.00×** |

Formula in `utils/memory.py` and [inference.md](inference.md). Run `scripts/kv_cache_benchmark.py` to reproduce.

---

## File Map

| Path | Role |
|---|---|
| `models/transformer.py` | `ModelConfig`, `GPTOSS`, `GPTOSSBlock`, `RMSNorm` |
| `models/attention.py` | SWA, full attention, sink bias, GQA |
| `models/moe.py` | Router, experts, aux loss, dispatch |
| `models/moe_triton.py` | Opt-in fused MoE kernel |
| `models/yarn.py` | YaRN RoPE module |
| `models/rotary.py` | `apply_rope`, `compute_yarn_freqs`, `prune_rope` |
| `training/pretrain.py` | Full training loop |
| `inference/generate.py` | `MixedKVCache`, `generate()` |
| `inference/long_context.py` | `PasskeyEvaluator` |
| `utils/checkpoint.py` | Atomic safetensors |
| `utils/memory.py` | VRAM estimator |
| `configs/pretrain_a100_502m.yaml` | Canonical recipe |

---

## Config → Code Routing

```
configs/pretrain_a100_502m.yaml
    model.*  ──► ModelConfig ──► GPTOSS, GPTOSSAttention, MoELayer
    training.* ──► pretrain.py (lr, compile, nan_guard, aux_loss_alpha)
    data.*   ──► PretrainDataset paths
```

See [configs.md](configs.md) for every key.

---

## Load-Bearing Invariants

| Invariant | Doc |
|---|---|
| Even layers = SWA, odd = full | [attention.md](attention.md) |
| Sink bias clamped `[-10, 15]` at forward | [ATTENTION_SINKS.md](ATTENTION_SINKS.md) |
| Standard aux loss (α=0.01), not aux-loss-free | [moe.md](moe.md) |
| Weight tying — head.weight = embed.weight | [transformer.md](transformer.md) |
| `moe_dispatch="stacked"` by default | [triton_kernels.md](triton_kernels.md) |
| NaN guard with checkpoint rollback | [training.md](training.md) |
| YaRN scale = target / original = 32 | [yarn.md](yarn.md) |

---

## Further Reading

| Topic | Doc |
|---|---|
| Sink bias theory | [ATTENTION_SINKS.md](ATTENTION_SINKS.md) |
| Attention implementation | [attention.md](attention.md) |
| MoE + aux loss | [moe.md](moe.md) |
| YaRN extrapolation | [yarn.md](yarn.md) |
| Training loop | [training.md](training.md) |
| KV cache + passkey | [inference.md](inference.md) |
| Test invariants | [testing.md](testing.md) |

<!-- docs:verified 2026-07-31 · fd4fe36 -->
