# Getting Started — From Zero to a Running GPT-OSS-Lite

> **Purpose:** Onboard a strong ML student who has never seen this repo. You will learn *what* the project is, *why* each design choice exists, and *how* to verify your understanding with smoke tests — before diving into component chapters.

---

## Table of Contents

1. [What Problem Does This Project Solve?](#what-problem-does-this-project-solve)
2. [The GPT-OSS Architecture in One Page](#the-gpt-oss-architecture-in-one-page)
3. [Canonical Numbers — 502M Config](#canonical-numbers--502m-config)
4. [Mental Model — Three Execution Modes](#mental-model--three-execution-modes)
5. [Recommended Reading Order](#recommended-reading-order)
6. [Environment Setup](#environment-setup)
7. [Quick Smoke Test (CPU)](#quick-smoke-test-cpu)
8. [Quick GPU Smoke Test](#quick-gpu-smoke-test)
9. [Full Training Run (A100)](#full-training-run-a100)
10. [Headline Metrics](#headline-metrics)
11. [How to Read the Codebase](#how-to-read-the-codebase)
12. [Common Pitfalls — Theory and Fixes](#common-pitfalls--theory-and-fixes)
13. [FAQ](#faq)
14. [References](#references)

---

## What Problem Does This Project Solve?

Frontier models like GPT-OSS introduce **under-documented primitives** that show up constantly in interviews but are thin in the original model card:

| Primitive | One-line summary | Doc |
|---|---|---|
| **Learned attention sinks** | Per-head null-attention mass (StreamingLLM lineage) | [ATTENTION_SINKS.md](ATTENTION_SINKS.md) |
| **SWA(128) / full alternation** | 2× KV-cache reduction at 128K | [attention.md](attention.md) |
| **YaRN at training time** | 4K train → 128K extrapolation | [yarn.md](yarn.md) |
| **Top-2-of-8 MoE + standard aux loss** | Deliberate contrast to DeepSeek aux-loss-free | [moe.md](moe.md) |

**GPT-OSS-Lite** is a **pedagogical reproduction**: the full GPT-OSS attention + MoE + long-context stack at ~502M total / ~247M active parameters, implemented in raw PyTorch.

### What this project is NOT

- Not a production chatbot (502M is tiny by industry standards)
- Not a distributed training framework (single GPU only)
- Not a drop-in `transformers.AutoModel`

It **is** a complete, trainable, testable implementation you can run on one A100 in ~16–20 hours.

---

## The GPT-OSS Architecture in One Page

```
Token IDs (vocab=128,000)
    │
    ▼
Embedding (d=768) ───────────────────────────── weight-tied ──► LM Head
    │
    ▼
12 × GPTOSSBlock (gradient checkpoint every 3rd):
    ┌──────────────────────────────────────────────────────────┐
    │  RMSNorm → GPTOSSAttention (even=SWA128, odd=full)       │
    │         → Residual                                       │
    │  RMSNorm → MoELayer (top-2 of 8 + 1 shared)              │
    │         → Residual                                       │
    └──────────────────────────────────────────────────────────┘
    │
    ▼
Final RMSNorm → Chunked Cross-Entropy (chunk=4096) + aux loss (α=0.01)
```

**Key insight:** Attention (sliding/full + sink + YaRN) and FFN (MoE) are independent innovations glued by a standard pre-norm residual stack. Read [architecture.md](architecture.md) for the system map.

**Prerequisites:** If GQA, sliding-window attention, or YaRN are unfamiliar, read [foundations.md](foundations.md) first.

---

## Canonical Numbers — 502M Config

Memorise orders of magnitude; exact values live in `configs/pretrain_a100_502m.yaml` and [configs.md](configs.md).

| Quantity | Value | Formula / note |
|---|---|---|
| Total parameters | ~502M | MoE dominates |
| Active parameters | ~247M | Top-2 of 8 + 1 shared |
| Training tokens | 8.0B | Chinchilla: ~16 tokens/param |
| Layers | 12 | 6 SWA + 6 full (alternating) |
| Hidden dim $d$ | 768 | |
| Heads | 8 Q / 4 KV | GQA, head_dim=96 |
| MoE | 8 routed (top-2) + 1 shared | SwiGLU, ffn=1536 |
| Train seq len | 4,096 | YaRN active here |
| Eval seq len | 131,072 | 32× extrapolation |
| Window | 128 | Even layers only |

---

## Mental Model — Three Execution Modes

| Mode | When | Entry point |
|---|---|---|
| **CPU correctness** | Every PR, every arch change | `pytest tests/ -q` |
| **GPU smoke** | Before committing GPU time | `scripts/e2e_gpu_smoke.py` |
| **Production train** | Full 8B-token run | `training/pretrain.py --config configs/pretrain_a100_502m.yaml` |

All three share the same `GPTOSS` model class — no separate inference graph.

---

## Recommended Reading Order

See the full learning path in [README.md](README.md). Minimum path:

1. [foundations.md](foundations.md) — prerequisites
2. [architecture.md](architecture.md) — system map
3. [ATTENTION_SINKS.md](ATTENTION_SINKS.md) — headline primitive #1
4. [moe.md](moe.md) — headline primitive #2
5. [training.md](training.md) — run the loop

---

## Environment Setup

```bash
git clone https://github.com/atandra2000/GPT-OSS-Lite.git
cd GPT-OSS-Lite
pip install -r requirements.txt
```

Requirements: Python 3.10+, PyTorch 2.1+. CUDA optional for CPU tests.

---

## Quick Smoke Test (CPU)

```bash
python3 -m pytest tests/ -q
# Expected: 187 passed (~40s on a modern laptop)
```

Critical subsets after attention changes:

```bash
python3 -m pytest tests/test_attention.py -v
python3 -m pytest tests/test_validation.py -k anchor -v
```

---

## Quick GPU Smoke Test

```bash
python3 scripts/e2e_gpu_smoke.py
```

Covers Triton MoE path (if CUDA + Triton available), forward/backward, and KV-cache inference.

---

## Full Training Run (A100)

```bash
# 1. Prepare data (hours on first run)
python3 data/prepare_data.py --stage pretrain

# 2. Pre-flight VRAM + step time
python3 scripts/microbench_a100.py
python3 scripts/step_time_a100.py --steps 20 --warmup 5

# 3. Train (reproducible)
python3 training/pretrain.py \
    --config configs/pretrain_a100_502m.yaml \
    --seed 42
```

See [training.md](training.md) and [data/DATA_PIPELINE.md](../data/DATA_PIPELINE.md) for details.

---

## Headline Metrics

Both metrics have dedicated scripts — run them before claiming numbers in a README or interview:

```bash
# KV-cache reduction (analytical, CPU-friendly)
python3 scripts/kv_cache_benchmark.py

# Passkey retrieval (requires trained checkpoint)
python3 scripts/passkey_eval.py --checkpoint checkpoints/pretrain_a100/model_step_60000.safetensors
```

Untrained models: passkey eval runs as a stub (accuracy ≈ 0). See [inference.md](inference.md).

---

## How to Read the Codebase

| Question | Start here |
|---|---|
| Where is the model assembled? | `models/transformer.py` → [transformer.md](transformer.md) |
| Sliding window + sink bias? | `models/attention.py` → [ATTENTION_SINKS.md](ATTENTION_SINKS.md) |
| YaRN RoPE? | `models/yarn.py`, `models/rotary.py` |
| MoE routing + aux loss? | `models/moe.py` |
| Training loop? | `training/pretrain.py` |
| Decode + KV cache? | `inference/generate.py` |
| What must not break? | [testing.md](testing.md) load-bearing table |

---

## Common Pitfalls — Theory and Fixes

| Symptom | Likely cause | Fix |
|---|---|---|
| KV reduction < 1.8× | Wrong layer alternation | Verify `n_layers=12`, even=SWA |
| NaN at 128K | YaRN misconfigured | `scale_factor = target/original = 32` |
| Sink bias overflow in BF16 | Unclamped parameter | Forward clamp `[-10, 15]` — see [attention.md](attention.md) |
| Triton silently not used | Wrong dispatch key | Set `moe_dispatch="triton_grouped"` explicitly |
| Param count drift | Config change | `pytest tests/test_validation.py -k anchor` |
| Non-reproducible runs | Missing `--seed` | Always pass `--seed N` |

---

## FAQ

**Q: Why standard aux loss instead of DeepSeek's aux-loss-free gate?**
A: Deliberate portfolio distinction. GPT-OSS uses Switch/GShard-style load balancing (α=0.01). See [moe.md](moe.md) §1.

**Q: Why window=128 and not 4096?**
A: GPT-OSS default. Smaller window = more aggressive KV savings at long context.

**Q: Can I replace sliding-window layers with full attention?**
A: No — it breaks the headline KV-cache metric and violates AGENTS.md rule 2.

**Q: Has the full 8B-token run completed?**
A: Not yet (no GPU on dev machine). The 187-test suite validates all primitives on CPU.

---

## References

- [ATTENTION_SINKS.md](ATTENTION_SINKS.md) — authoritative sink-bias reference
- [README.md](../README.md) — public project summary
- [CONTEXT.md](../CONTEXT.md) — agent working snapshot

<!-- docs:verified 2026-07-31 · fd4fe36 -->
