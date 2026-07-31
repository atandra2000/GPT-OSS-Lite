# Getting Started — GPT-OSS-Lite

> **Chapter 0 of the GPT-OSS-Lite documentation.** This chapter is the onboarding
> path: what the project is, how to install it, how to prepare data, and which
> commands to run first. For mathematical motivation, read
> [foundations.md](foundations.md). For the full layer stack, see
> [architecture.md](architecture.md).

---

## Table of contents

1. [What is GPT-OSS-Lite?](#1-what-is-gpt-oss-lite)
2. [Key numbers at a glance](#2-key-numbers-at-a-glance)
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

**GPT-OSS-Lite** is a from-scratch PyTorch reproduction of OpenAI's GPT-OSS
architecture (Apache 2.0). It is not a wrapper around HuggingFace Transformers or
Lightning — every primitive (attention, YaRN RoPE, MoE routing, training loop,
inference cache) is implemented directly in this repository.

The project exists for two overlapping audiences:

1. **Researchers** who want a faithful, testable reference for GPT-OSS-specific
   ideas: learned attention sinks, sliding-window / full-attention alternation,
   and YaRN applied at training time (not decode-only).
2. **Practitioners** who want a Chinchilla-optimal recipe that fits on a single
   A100 80GB and trains in roughly 16–20 hours.

Within the CoreProjects LLM portfolio, GPT-OSS-Lite is the first project to
combine **long-context MoE** with **learned sink bias**. Sibling repos cover
MLA (DeepSeek-v3-Lite), pure GQA (LLaMA-3-Lite), and SSMs (Mamba-3-Lite) —
see the comparison table in the root [README](../README.md).

### What you will build

A **12-layer decoder-only transformer** with:

- **GQA** — 8 query heads, 4 KV heads, `head_dim=96`
- **Alternating attention** — even layers use sliding window `W=128`; odd layers
  attend globally (see [attention.md](attention.md))
- **Learned sink bias** per head — authoritative theory in
  [ATTENTION_SINKS.md](ATTENTION_SINKS.md)
- **YaRN RoPE** — train at 4K, extrapolate to 128K ([yarn.md](yarn.md),
  [rotary.md](rotary.md))
- **MoE SwiGLU** — top-2 of 8 routed experts plus 1 shared ([moe.md](moe.md))
- **Weight-tied** embedding and LM head

The top-level model class is `GPTOSS` in `models/transformer.py`. Training runs
through `training/pretrain.py`. Autoregressive decoding uses
`inference/generate.py` with a mixed KV cache.

---

## 2. Key numbers at a glance

These are the production targets from
[`configs/pretrain_a100_502m.yaml`](../configs/pretrain_a100_502m.yaml). Every
field is explained in [configs.md](configs.md).

| Quantity | Value | Notes |
|----------|-------|-------|
| Total parameters | ~502M | `GPTOSS.num_parameters()` |
| Active parameters / token | ~247M | Top-2 routed + 1 shared expert |
| Sparsity | ~50.8% | Inactive expert weights per forward |
| Vocabulary | 128,000 | LLaMA-3 BPE tokenizer |
| `d_model` | 768 | `n_heads × head_dim` |
| Layers | 12 | 6 windowed + 6 global |
| Training context | 4,096 | `max_seq_len` |
| Eval / inference context | 131,072 | YaRN target (`eval_max_seq_len`) |
| Sliding window | 128 | GPT-OSS default |
| MoE aux loss α | 0.01 | Standard Switch Transformer scale |
| Training tokens | 8.0B | Chinchilla-optimal for ~502M |
| Optimizer steps | 61,000 | See derived arithmetic below |
| Warmup steps | 3,000 | ~4.9% of total — MoE stability |
| Peak learning rate | 4.0×10⁻⁴ | Cosine decay to 5% |
| Micro-batch × accum | 8 × 4 | Effective batch 32 sequences |
| Tokens per optimizer step | 131,072 | `8 × 4 × 4096` |
| Total tokens seen | ~8.0B | `61,000 × 131,072` |

### Headline metrics (measured, not assumed)

| Metric | Target | Script |
|--------|--------|--------|
| KV-cache reduction at 128K | ≥ 1.8× vs pure GQA | `scripts/kv_cache_benchmark.py` |
| Passkey retrieval at 128K | ≥ 85% accuracy | `scripts/passkey_eval.py` |

The KV benchmark is analytical (no GPU required). Passkey eval requires a
**trained** checkpoint — untrained models will not hit 85%.

---

## 3. Prerequisites

| Requirement | Minimum | Recommended |
|-------------|---------|-------------|
| Python | 3.10+ | 3.12+ |
| PyTorch | 2.1+ | 2.2+ with CUDA 12.x |
| GPU (full pretrain) | — | 1× A100 80GB |
| GPU (smoke / dev) | 4 GB VRAM | GTX 1650 or better |
| Disk (data + checkpoints) | ~50 GB | 100 GB+ for 8B-token shards |

CPU-only development is supported for architecture verification and the KV-cache
analytical benchmark. Full pretraining and `torch.compile` require CUDA.

Optional:

- **Triton** — only if you opt into `moe_dispatch: "triton_grouped"` (see
  [triton_kernels.md](triton_kernels.md)). Default is `"stacked"` (pure PyTorch).
- **Weights & Biases** — `wandb` is in `requirements.txt`; logging is optional.

---

## 4. Installation

Clone the repository and install dependencies:

```bash
git clone https://github.com/atandra2000/GPT-OSS-Lite.git
cd GPT-OSS-Lite
pip install -r requirements.txt
```

`requirements.txt` pins the core stack:

```
torch>=2.1
safetensors>=0.4
pyyaml>=6.0
tqdm>=4.65
wandb>=0.16
```

Verify PyTorch sees your GPU (if present):

```bash
python3 -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

No separate `setup.py` install step is required — scripts add the project root
to `sys.path` automatically.

---

## 5. Repository layout

Key paths: `configs/` (YAML recipes), `models/` (`transformer.py`, `attention.py`,
`moe.py`, `yarn.py`), `training/pretrain.py`, `inference/generate.py` +
`long_context.py`, `data/prepare_data.py`, `scripts/kv_cache_benchmark.py` and
`passkey_eval.py`. Full map: [architecture.md](architecture.md).

---

## 6. Prepare training data

GPT-OSS-Lite does not ship pre-tokenized shards. You must build them before
launching `pretrain.py`.

### What the pipeline produces

The shim at `data/prepare_data.py` delegates to the shared CoreProjects pipeline
under `LLM/shared_data/`. For GPT-OSS-Lite the defaults are:

- **Tokenizer:** LLaMA-3 BPE, `vocab_size=128000`
- **Corpus mix:** `gptoss-default` (FineWeb-Edu, FineWeb, The Stack Python,
  OpenMath, arXiv — see comments in the A100 config)
- **Shard size:** 50M tokens per `shard_*.bin`
- **Total budget:** 8.0B tokens

### Command

From the project root:

```bash
python3 data/prepare_data.py
```

This prints a banner with tokenizer and shard settings, then runs the universal
prepare script. Output lands in `data/pretrain_chinchilla/` (matching
`train_data_path` in the A100 config).

A `manifest.json` in that directory records `eos_token_id`, `vocab_size`,
`total_tokens`, and `shard_count`. The training `PretrainDataset` reads this
manifest when present.

### Smoke data

For `configs/pretrain_gpu_smoke.yaml`, point `train_data_path` at
`data/pretrain_chinchilla`. The universal pipeline can emit a tiny smoke corpus when
invoked with the smoke tokenizer — see [data_pipeline.md](data_pipeline.md) for
flags and directory layout.

### If data is missing

`training/pretrain.py` raises `FileNotFoundError` with an explicit message:

```
Training data not found at data/pretrain_chinchilla.
Run `python data/prepare_data.py` first.
```

Do not point `train_data_path` at an empty directory — sharded layout requires
at least one `shard_*.bin` file.

---

## 7. Your first commands

Run these in order after installation. Each step validates a different layer of
the stack.

### Step 1 — KV-cache headline metric (CPU, seconds)

```bash
python3 scripts/kv_cache_benchmark.py
```

Expected output ends with:

```
✅ HEADLINE METRIC PASSED: 2.00× KV-cache reduction at 128K (≥ 1.8×)
```

This script is **analytical** — it computes KV bytes from architecture constants
(12 layers, 6 windowed at `W=128`, 6 global, GQA with 4 KV heads). No model
weights are loaded. See [inference.md](inference.md) for how this relates to
`MixedKVCache`.

### Step 2 — Doc link checker (optional)

```bash
python3 scripts/check_docs.py
```

Flags broken cross-references in the documentation tree.

---

## 8. Smoke training on a small GPU

If you have a CUDA GPU with as little as 4 GB VRAM, use the smoke config:

```bash
python3 training/pretrain.py \
    --config configs/pretrain_gpu_smoke.yaml \
    --seed 42
```

`pretrain_gpu_smoke.yaml` mirrors the **structural** choices of the production
model (alternating SWA/full, sink bias, YaRN, MoE top-2) at 1/100th scale:

- `d_model=128`, `n_layers=4`, `max_seq_len=64`
- `total_steps=5`, `compile=false`
- Checkpoints under `checkpoints/gpu_smoke/`

For a broader GPU integration test (forward, backward, checkpoint round-trip,
`MixedKVCache` generation, YaRN extrapolation), run:

```bash
python3 scripts/e2e_gpu_smoke.py
```

This script exercises the full pipeline without pytest as the primary interface.

---

## 9. Reproduce the headline metrics

### Metric 1 — KV-cache reduction

Already covered in Step 1. The benchmark compares:

- **Pure GQA:** all 12 layers cache the full sequence length
- **SWA/Full mix:** 6 layers cache `min(W, T)` tokens; 6 layers cache `T`

At `T=131072` and `W=128`, the ratio approaches **2.0×**. At `T=4096`, the
window equals the sequence length on windowed layers, so reduction is **1.0×**
— this is expected.

### Metric 2 — Passkey retrieval at 128K

After training (or with a downloaded checkpoint):

```bash
python3 scripts/passkey_eval.py \
    --checkpoint checkpoints/pretrain_a100/model_step_61000.safetensors \
    --n-trials 100 \
    --context-lengths 4096 8192 32768 65536 131072
```

The evaluator (`inference/long_context.py`) embeds a random 5-digit passkey in
filler text, asks the model to recall it, and scores exact match. Protocol
details are in [inference.md](inference.md).

On an **untrained** model, accuracy will be near chance — the script still
exits 0 but prints a warning that ≥ 85% requires a trained checkpoint.

---

## 10. Launch full pretraining

Once `data/pretrain_chinchilla/` exists and you have an A100 80GB (or
equivalent):

```bash
python3 training/pretrain.py \
    --config configs/pretrain_a100_502m.yaml \
    --seed 42
```

What happens internally (full detail in [training.md](training.md)):

1. `ModelConfig` is built from YAML `model:` block
2. `GPTOSS` is constructed; param counts are printed
3. `torch.compile(mode="max-autotune")` on CUDA when `compile: true`
4. AdamW with FP32 master weights, `foreach=True`, `fused=True` on GPU
5. Linear warmup (3000 steps) → cosine decay to 5% of peak LR
6. BF16 autocast forward; chunked cross-entropy; aux loss scaled by α=0.01
7. Gradient checkpointing every 3rd block
8. Checkpoints every 2000 steps to `checkpoints/pretrain_a100/`
9. NaN guard with rollback after 5 consecutive non-finite losses

Expected wall time: **16–20 hours** at ~35–40% MFU on A100 80GB.

Override step count for debugging:

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

This restores:

- Model weights from `model_step_40000.safetensors`
- Optimizer and scheduler state
- RNG state from `rng_step_40000.pt` (when present)

Reproducibility knobs (`--seed`, `CUBLAS_WORKSPACE_CONFIG`, stable MoE
`argsort`) are documented in [training.md](training.md) and the root README.

---

## 12. Common pitfalls

### Missing training data

**Symptom:** `FileNotFoundError` at startup.

**Fix:** Run `python3 data/prepare_data.py`. Confirm `data/pretrain_chinchilla/shard_*.bin`
exists and `manifest.json` is present.

### NaN guard rollback loop

**Symptom:** Log spam `[nan-guard] step N: non-finite loss`.

**Causes:** Learning rate too high, bad data shard, or numerical edge case in
MoE routing. The guard skips the step and, after 5 consecutive failures, reloads
the latest checkpoint.

**Fix:** Lower `lr`, verify data integrity, ensure `aux_loss_alpha` stays at
0.01. Do **not** disable `nan_guard` in production without explicit intent —
see [training.md](training.md).

### Replacing sliding-window layers with full attention

**Symptom:** KV benchmark still passes analytically, but you changed
`models/attention.py` layer alternation.

**Impact:** Breaks the architectural claim of ~2× KV-cache reduction at 128K.
The even/odd `is_windowed` pattern is load-bearing. Read
[ATTENTION_SINKS.md](ATTENTION_SINKS.md) before editing attention code.

### Triton opt-in confusion

**Symptom:** `ImportError` mentioning Triton on a Mac or CPU machine.

**Cause:** `model.moe_dispatch: "triton_grouped"` in YAML without Triton/CUDA.

**Fix:** Omit `moe_dispatch` (defaults to `"stacked"`) or set explicitly:

```yaml
model:
  moe_dispatch: "stacked"
```

There is **no** environment variable gate — configuration is the only switch.
See [triton_kernels.md](triton_kernels.md).

### Passkey eval on untrained weights

**Symptom:** Accuracy ~0% at 128K.

**Expected.** Passkey retrieval tests whether YaRN extrapolation **learned**
something during pretraining. Run eval only after meaningful training steps.

### `torch.compile` first-step latency

**Symptom:** First optimizer step takes several minutes.

**Expected.** `max-autotune` benchmarks kernel variants once. Subsequent steps
are fast. Set `compile: false` in smoke configs or for debugging.

### OOM on A100

Confirm `grad_checkpoint: true`, reduce `micro_batch_size`, or disable `compile`
temporarily. See `utils/memory.py` estimates at startup.

---

## 13. Where to go next

| Goal | Document |
|------|----------|
| Math behind sinks, SWA, YaRN | [foundations.md](foundations.md) |
| System diagram and file map | [architecture.md](architecture.md) |
| Sink bias authoritative reference | [ATTENTION_SINKS.md](ATTENTION_SINKS.md) |
| `GPTOSS`, `GPTOSSBlock`, RMSNorm | [transformer.md](transformer.md) |
| `MixedKVCache`, `generate()`, passkey | [inference.md](inference.md) |
| Every YAML key | [configs.md](configs.md) |
| Training loop internals | [training.md](training.md) |
| Tokenization and shards | [data_pipeline.md](data_pipeline.md) |
| MoE routing and aux loss | [moe.md](moe.md) |
| Optional Triton MoE kernel | [triton_kernels.md](triton_kernels.md) |

---

<!-- docs:verified 2026-07-31 · fa6f918 -->
