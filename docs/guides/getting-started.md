# GPT-OSS-Lite — Getting Started

> **Chapter 0.** Onboarding: install, layout, first commands, smoke train, headline metrics, pitfalls. Math motivation: [foundations-and-architecture.md](../concepts/foundations-and-architecture.md). Layer stack: [foundations-and-architecture.md](../concepts/foundations-and-architecture.md).

---

## Table of contents

1. [What is GPT-OSS-Lite?](#1-what-is-gpt-oss-lite)
2. [Headline metrics](#2-headline-metrics)
3. [Prerequisites](#3-prerequisites)
4. [Installation](#4-installation)
5. [Repository layout](#5-repository-layout)
6. [Prepare training data](#6-prepare-training-data)
7. [Your first commands](#7-your-first-commands)
8. [Smoke training on a small GPU](#8-smoke-training-on-a-small-gpu)
9. [Reproduce the headline metrics](#9-reproduce-the-headline-metrics)
10. [Launch full pretraining](#10-launch-full-pretraining)
11. [Resume from checkpoint](#11-resume-from-checkpoint)
12. [Common pitfalls](#12-common-pitfalls)
13. [Where to go next](#13-where-to-go-next)

---

## 1. What is GPT-OSS-Lite?

From-scratch PyTorch reproduction of OpenAI's GPT-OSS (Apache 2.0) — not a HuggingFace or Lightning wrapper. Top-level model: `GPTOSS` in `models/transformer.py`; training via `training/pretrain.py`; decode via `inference/generate.py` with `MixedKVCache`.

Portfolio context and the sibling-project comparison table live in the root
[README](../README.md). Architectural primitives (GQA 8Q/4KV, sliding window
`W=128` on six layers, learned sink bias, YaRN 128K, top-2-of-8 MoE) are documented in [foundations-and-architecture.md](../concepts/foundations-and-architecture.md) and
[attention-sinks.md](../concepts/attention-sinks.md).

---

## 2. Headline metrics

KV reduction is **measured** (analytical benchmark, no GPU); the ≥85% passkey band is a **target** — no pretraining run has happened yet. Production YAML and derived arithmetic (~502M total, ~247M active, 8.0B tokens, 61k steps) are in
[`configs/pretrain_a100_502m.yaml`](../../configs/pretrain_a100_502m.yaml) and
[training.md](../training.md#part-b--configuration-reference).

| Metric | Target | Script |
|--------|--------|--------|
| KV-cache reduction at 128K | ≥ 1.8× vs pure GQA | `scripts/kv_cache_benchmark.py` |
| Passkey retrieval at 128K | ≥ 85% accuracy | `scripts/passkey_eval.py` |

KV benchmark is analytical (no GPU). Passkey eval needs a **trained** checkpoint — untrained models will not hit 85%.

---

## 3. Prerequisites

| Requirement | Minimum | Recommended |
|-------------|---------|-------------|
| Python | 3.10+ | 3.12+ |
| PyTorch | 2.1+ | 2.2+ with CUDA 12.x |
| GPU (full pretrain) | — | 1× A100 80GB |
| GPU (smoke / dev) | 4 GB VRAM | GTX 1650 or better |
| Disk (data + checkpoints) | ~50 GB | 100 GB+ for 8B-token shards |

CPU-only works for architecture tests and the KV analytical benchmark. Full pretrain and `torch.compile` need CUDA.

Optional: **Triton** only for `moe_dispatch: "triton_grouped"` ([moe.md](../concepts/moe.md)); default is `"stacked"`. **W&B** is optional (`wandb` in `requirements.txt`).

---

## 4. Installation

```bash
git clone https://github.com/atandra2000/GPT-OSS-Lite.git
cd GPT-OSS-Lite
pip install -r requirements.txt
```

Verify GPU visibility:

```bash
python3 -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

No `setup.py` step — scripts add the project root to `sys.path` automatically.

---

## 5. Repository layout

`configs/` (YAML), `models/` (`transformer.py`, `attention.py`, `moe.py`, `yarn.py`), `training/pretrain.py`, `inference/generate.py` + `long_context.py`, `data/prepare_data.py`, `scripts/kv_cache_benchmark.py`, `passkey_eval.py`, `utils/`. Reference chapters live in `docs/`; the from-scratch theory set is consolidated under `docs/concepts/` — attention math, positional encodings, MoE theory, numerics, optimizers, autograd checkpointing, sampling, KV-cache engineering, tokenization/BPE, Triton programming. Full map:
[foundations-and-architecture.md](../concepts/foundations-and-architecture.md).

---

## 6. Prepare training data

No pre-tokenized shards ship with the repo. Run before `pretrain.py`.

`data/prepare_data.py` delegates to the vendored CoreProjects pipeline under `LLM/shared_data/`. Defaults: LLaMA-3 BPE (`vocab_size=128000`), `gptoss-default` mix, 50M tokens per `shard_*.bin`, 8.0B total → `data/pretrain_chinchilla/` (matching the A100 config). `manifest.json` records `eos_token_id`, `vocab_size`, `total_tokens`, `shard_count`. The BPE merge algorithm and the 128K-vocab rationale: [tokenization.md](../concepts/tokenization.md).

```bash
python3 data/prepare_data.py
```

Smoke corpus flags: [training.md](../training.md). Missing data raises `FileNotFoundError` with an explicit `prepare_data.py` hint — do not point `train_data_path` at an empty directory.

---

## 7. Your first commands

### Step 1 — KV-cache headline metric (CPU, seconds)

```bash
python3 scripts/kv_cache_benchmark.py
```

Expected tail:

```
✅ HEADLINE METRIC PASSED: 2.00× KV-cache reduction at 128K (≥ 1.8×)
```

Analytical only — architecture constants, no weights loaded. Cache design:
[inference.md](../inference.md).

### Step 2 — Doc link checker (optional)

```bash
python3 scripts/check_docs.py
```

---

## 8. Smoke training on a small GPU

```bash
python3 training/pretrain.py \
    --config configs/pretrain_gpu_smoke.yaml \
    --seed 42
```

`pretrain_gpu_smoke.yaml` mirrors structural choices (SWA/full alt, sink bias, YaRN, MoE top-2) at 1/100th scale (`d_model=128`, `n_layers=4`, `max_seq_len=64`, `total_steps=5`, `compile=false`). MoE runs the `"stacked"` dispatch by default; the optional fused Triton path (`moe_dispatch: "triton_grouped"`) and its tile-level programming model:
[kernels-and-checkpointing.md](../concepts/kernels-and-checkpointing.md).

Broader GPU integration (forward, backward, checkpoint round-trip, generation, YaRN extrapolation):

```bash
python3 scripts/e2e_gpu_smoke.py
```

---

## 9. Reproduce the headline metrics

**KV reduction** — Step 1 above. At `T=131072`, `W=128`, ratio ≈ **2.0×**; at `T=4096` the windowed layers cache 128 tokens regardless of sequence length, so the ratio is ≈ **1.94×** — not 1.0× (measured by `scripts/kv_cache_benchmark.py`).

**Passkey at 128K** — after training:

```bash
python3 scripts/passkey_eval.py \
    --checkpoint checkpoints/pretrain_a100/model_step_61000.safetensors \
    --n-trials 100 \
    --context-lengths 4096 8192 32768 65536 131072
```

Protocol: [inference.md](../inference.md). Untrained weights → near-chance accuracy (exit 0, warning printed).

---

## 10. Launch full pretraining

```bash
python3 training/pretrain.py \
    --config configs/pretrain_a100_502m.yaml \
    --seed 42
```

Loop internals (compile, AdamW FP32 master, warmup 3000 steps, aux α=0.01, NaN guard, checkpoints every 2000 steps): [training.md](../training.md). Expected wall time **16–20 hours** on A100 80GB `[INFERENCE]` (`.benchmarks/` empty).

Debug override:

```bash
python3 training/pretrain.py \
    --config configs/pretrain_a100_502m.yaml \
    --seed 42 \
    --max-steps 10
```

---

## 11. Resume from checkpoint

```bash
python3 training/pretrain.py \
    --config configs/pretrain_a100_502m.yaml \
    --seed 42 \
    --resume-from 40000
```

Restores weights, optimizer/scheduler, and `rng_step_*.pt` when present. Reproducibility knobs: [training.md](../training.md).

---

## 12. Common pitfalls

### Missing training data

`FileNotFoundError` at startup → run `python3 data/prepare_data.py`; confirm `shard_*.bin` and `manifest.json` exist.

### NaN guard rollback loop

`[nan-guard] step N: non-finite loss` — LR too high, bad shard, or MoE edge case. Guard skips steps; after five consecutive failures reloads latest checkpoint. Lower `lr`, verify data, keep `aux_loss_alpha=0.01`. Do not disable `nan_guard` without intent — [training.md](../training.md).

### Replacing sliding-window layers with full attention

Breaks the ~2× KV headline at 128K. Even/odd `is_windowed` is load-bearing — read [attention-sinks.md](../concepts/attention-sinks.md) before editing `models/attention.py`.

### Triton opt-in confusion

`ImportError` on Mac/CPU when `moe_dispatch: "triton_grouped"` without Triton/CUDA. Fix: omit field (defaults `"stacked"`) or set explicitly. No env-var gate —
[moe.md](../concepts/moe.md#sanctioned-triton-path-moe_dispatchtriton_grouped).

### Passkey eval on untrained weights

~0% at 128K is expected — tests YaRN extrapolation **after** pretraining.

### `torch.compile` first-step latency

`max-autotune` benchmarks once; first step can take minutes. Set `compile: false` for smoke/debug.

### OOM on A100

Confirm `grad_checkpoint: true`, reduce `micro_batch_size`, or disable `compile`. VRAM estimates at startup: [operations.md](operations.md).

---

## 13. Where to go next

| Goal | Document |
|------|----------|
| Math behind sinks, SWA, YaRN | [foundations-and-architecture.md](../concepts/foundations-and-architecture.md) |
| System diagram, `GPTOSS`, `ModelConfig` | [foundations-and-architecture.md](../concepts/foundations-and-architecture.md) |
| Sink bias authoritative reference | [attention-sinks.md](../concepts/attention-sinks.md) |
| RoPE / YaRN 128K | [attention-and-positional.md](../concepts/attention-and-positional.md) |
| MoE routing, aux loss, Triton opt-in | [moe.md](../concepts/moe.md) |
| Training loop + YAML encyclopedia | [training.md](../training.md) |
| `MixedKVCache`, passkey eval | [inference.md](../inference.md) |
| Scripts, checkpoints, OPT catalog | [operations.md](operations.md) |
| Tokenization and shards | [training.md](../training.md) |

### 13.1 Theory read order

The from-scratch theory chapters build on each other; read them in this order:

1. [foundations-and-architecture.md](../concepts/foundations-and-architecture.md) — why decoder-only, GQA, SWA, sinks, YaRN, MoE.
2. [attention-and-positional.md](../concepts/attention-and-positional.md) — softmax, scaled dot product, mask semantics, SDPA backends.
3. [attention-and-positional.md](../concepts/attention-and-positional.md) — sinusoids → relative → RoPE, interpolation, YaRN ramp.
4. [moe.md](../concepts/moe.md) — top-k gating math, Switch/GShard aux loss, expert collapse.
5. [optimizers-and-numerics.md](../concepts/optimizers-and-numerics.md) — FP32/FP16/BF16/TF32 formats, epsilon, sink-clamp derivation.
6. [optimizers-and-numerics.md](../concepts/optimizers-and-numerics.md) — momentum → Adam → AdamW, bias correction, FP32 master weights.
7. [kernels-and-checkpointing.md](../concepts/kernels-and-checkpointing.md) — backward-graph memory, recompute tradeoff.
8. [optimizers-and-numerics.md](../concepts/optimizers-and-numerics.md) — temperature, top-k/top-p, entropy, why passkey runs greedy.
9. [inference.md](../inference.md) — arithmetic intensity, ring vs growth, GQA bandwidth, mixed-cache 2.00×.
10. [tokenization.md](../concepts/tokenization.md) — BPE merges, 128K vocab economics, EOS packing.
11. [kernels-and-checkpointing.md](../concepts/kernels-and-checkpointing.md) — GPU execution model, tiles, `tl` primitives, fused W1/W3+silu.

---

## References

- [`models/transformer.py:GPTOSS`](../../models/transformer.py) — top-level model
- [`configs/pretrain_a100_502m.yaml`](../../configs/pretrain_a100_502m.yaml) — canonical config
- [foundations-and-architecture.md](../concepts/foundations-and-architecture.md) — architecture primer
- [attention-sinks.md](../concepts/attention-sinks.md) — sink-bias deep-dive
- [moe.md](../concepts/moe.md) — Triton opt-in
- [training.md](../training.md) — training loop, data preparation
- [inference.md](../inference.md) — generation and passkey eval
- [operations.md](operations.md) — scripts and utilities

<!-- docs:verified 2026-08-05 · 6491066 -->
