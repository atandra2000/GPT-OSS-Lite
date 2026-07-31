# Getting Started — GPT-OSS-Lite

> **Chapter 0.** Onboarding: install, layout, first commands, smoke train, headline
> metrics, pitfalls. Math motivation: [foundations.md](foundations.md). Layer stack:
> [architecture.md](architecture.md).

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

From-scratch PyTorch reproduction of OpenAI's GPT-OSS (Apache 2.0) — not a
HuggingFace or Lightning wrapper. Top-level model: `GPTOSS` in
`models/transformer.py`; training via `training/pretrain.py`; decode via
`inference/generate.py` with `MixedKVCache`.

Portfolio context and the sibling-project comparison table live in the root
[README](../README.md). Architectural primitives (GQA 8Q/4KV, sliding window
`W=128` on six layers, learned sink bias, YaRN 128K, top-2-of-8 MoE) are
documented in [architecture.md](architecture.md) and
[ATTENTION_SINKS.md](ATTENTION_SINKS.md).

---

## 2. Headline metrics

Both targets are **measured**, not assumed. Production YAML and derived
arithmetic (~502M total, ~247M active, 8.0B tokens, 61k steps) are in
[`configs/pretrain_a100_502m.yaml`](../configs/pretrain_a100_502m.yaml) and
[training.md](training.md#part-b--configuration-reference).

| Metric | Target | Script |
|--------|--------|--------|
| KV-cache reduction at 128K | ≥ 1.8× vs pure GQA | `scripts/kv_cache_benchmark.py` |
| Passkey retrieval at 128K | ≥ 85% accuracy | `scripts/passkey_eval.py` |

KV benchmark is analytical (no GPU). Passkey eval needs a **trained**
checkpoint — untrained models will not hit 85%.

---

## 3. Prerequisites

| Requirement | Minimum | Recommended |
|-------------|---------|-------------|
| Python | 3.10+ | 3.12+ |
| PyTorch | 2.1+ | 2.2+ with CUDA 12.x |
| GPU (full pretrain) | — | 1× A100 80GB |
| GPU (smoke / dev) | 4 GB VRAM | GTX 1650 or better |
| Disk (data + checkpoints) | ~50 GB | 100 GB+ for 8B-token shards |

CPU-only works for architecture tests and the KV analytical benchmark. Full
pretrain and `torch.compile` need CUDA.

Optional: **Triton** only for `moe_dispatch: "triton_grouped"` ([moe.md](moe.md));
default is `"stacked"`. **W&B** is optional (`wandb` in `requirements.txt`).

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

`configs/` (YAML), `models/` (`transformer.py`, `attention.py`, `moe.py`,
`yarn.py`), `training/pretrain.py`, `inference/generate.py` +
`long_context.py`, `data/prepare_data.py`, `scripts/kv_cache_benchmark.py`,
`passkey_eval.py`, `utils/`. Full map: [architecture.md](architecture.md).

---

## 6. Prepare training data

No pre-tokenized shards ship with the repo. Run before `pretrain.py`.

`data/prepare_data.py` delegates to the vendored CoreProjects pipeline under
`LLM/shared_data/`. Defaults: LLaMA-3 BPE (`vocab_size=128000`), `gptoss-default`
mix, 50M tokens per `shard_*.bin`, 8.0B total → `data/pretrain_chinchilla/`
(matching the A100 config). `manifest.json` records `eos_token_id`, `vocab_size`,
`total_tokens`, `shard_count`.

```bash
python3 data/prepare_data.py
```

Smoke corpus flags: [data_pipeline.md](data_pipeline.md). Missing data raises
`FileNotFoundError` with an explicit `prepare_data.py` hint — do not point
`train_data_path` at an empty directory.

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
[inference.md](inference.md).

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

`pretrain_gpu_smoke.yaml` mirrors structural choices (SWA/full alt, sink bias,
YaRN, MoE top-2) at 1/100th scale (`d_model=128`, `n_layers=4`,
`max_seq_len=64`, `total_steps=5`, `compile=false`).

Broader GPU integration (forward, backward, checkpoint round-trip, generation,
YaRN extrapolation):

```bash
python3 scripts/e2e_gpu_smoke.py
```

---

## 9. Reproduce the headline metrics

**KV reduction** — Step 1 above. At `T=131072`, `W=128`, ratio ≈ **2.0×**; at
`T=4096` windowed layers see the full sequence, so ratio ≈ **1.0×** (expected).

**Passkey at 128K** — after training:

```bash
python3 scripts/passkey_eval.py \
    --checkpoint checkpoints/pretrain_a100/model_step_61000.safetensors \
    --n-trials 100 \
    --context-lengths 4096 8192 32768 65536 131072
```

Protocol: [inference.md](inference.md). Untrained weights → near-chance accuracy
(exit 0, warning printed).

---

## 10. Launch full pretraining

```bash
python3 training/pretrain.py \
    --config configs/pretrain_a100_502m.yaml \
    --seed 42
```

Loop internals (compile, AdamW FP32 master, warmup 3000 steps, aux α=0.01, NaN
guard, checkpoints every 2000 steps): [training.md](training.md). Expected wall
time **16–20 hours** on A100 80GB.

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

Restores weights, optimizer/scheduler, and `rng_step_*.pt` when present.
Reproducibility knobs: [training.md](training.md).

---

## 12. Common pitfalls

### Missing training data

`FileNotFoundError` at startup → run `python3 data/prepare_data.py`; confirm
`shard_*.bin` and `manifest.json` exist.

### NaN guard rollback loop

`[nan-guard] step N: non-finite loss` — LR too high, bad shard, or MoE edge case.
Guard skips steps; after five consecutive failures reloads latest checkpoint.
Lower `lr`, verify data, keep `aux_loss_alpha=0.01`. Do not disable `nan_guard`
without intent — [training.md](training.md).

### Replacing sliding-window layers with full attention

Breaks the ~2× KV headline at 128K. Even/odd `is_windowed` is load-bearing —
read [ATTENTION_SINKS.md](ATTENTION_SINKS.md) before editing `models/attention.py`.

### Triton opt-in confusion

`ImportError` on Mac/CPU when `moe_dispatch: "triton_grouped"` without Triton/CUDA.
Fix: omit field (defaults `"stacked"`) or set explicitly. No env-var gate —
[moe.md](moe.md#sanctioned-triton-path-moe_dispatchtriton_grouped).

### Passkey eval on untrained weights

~0% at 128K is expected — tests YaRN extrapolation **after** pretraining.

### `torch.compile` first-step latency

`max-autotune` benchmarks once; first step can take minutes. Set `compile: false`
for smoke/debug.

### OOM on A100

Confirm `grad_checkpoint: true`, reduce `micro_batch_size`, or disable `compile`.
VRAM estimates at startup: [operations.md](operations.md).

---

## 13. Where to go next

| Goal | Document |
|------|----------|
| Math behind sinks, SWA, YaRN | [foundations.md](foundations.md) |
| System diagram, `GPTOSS`, `ModelConfig` | [architecture.md](architecture.md) |
| Sink bias authoritative reference | [ATTENTION_SINKS.md](ATTENTION_SINKS.md) |
| RoPE / YaRN 128K | [rope_yarn.md](rope_yarn.md) |
| MoE routing, aux loss, Triton opt-in | [moe.md](moe.md) |
| Training loop + YAML encyclopedia | [training.md](training.md) |
| `MixedKVCache`, passkey eval | [inference.md](inference.md) |
| Scripts, checkpoints, OPT catalog | [operations.md](operations.md) |
| Tokenization and shards | [data_pipeline.md](data_pipeline.md) |

---

<!-- docs:verified 2026-07-31 · 123fd27 -->
