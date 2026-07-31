# Scripts — Operational Reference for GPT-OSS-Lite

> **Chapter: tooling and benchmarks.** This chapter documents every script under
> `scripts/` — what it measures, how to invoke it, and how to interpret output.
> For training entry points see [training.md](training.md); for inference internals
> see [architecture.md](architecture.md) §9 and (when published)
> [inference.md](inference.md).

---

## Table of contents

1. [Overview](#1-overview)
2. [Shared bootstrap — `_bootstrap.py`](#2-shared-bootstrap--_bootstrappy)
3. [Documentation linter — `check_docs.py`](#3-documentation-linter--check_docspy)
4. [Headline metric #1 — `kv_cache_benchmark.py`](#4-headline-metric-1--kv_cache_benchmarkpy)
5. [Headline metric #2 — `passkey_eval.py`](#5-headline-metric-2--passkey_evalpy)
6. [End-to-end GPU smoke — `e2e_gpu_smoke.py`](#6-end-to-end-gpu-smoke--e2e_gpu_smokepy)
7. [Component profiler — `profile_components.py`](#7-component-profiler--profile_componentspy)
8. [MoE dispatch profiler — `profile_moe.py`](#8-moe-dispatch-profiler--profile_moepy)
9. [Inference profiler — `profile_inference.py`](#9-inference-profiler--profile_inferencepy)
10. [VRAM microbench — `microbench_a100.py`](#10-vram-microbench--microbench_a100py)
11. [Step-time / MFU — `step_time_a100.py`](#11-step-time--mfu--step_time_a100py)
12. [Script selection guide](#12-script-selection-guide)
13. [Related documentation](#13-related-documentation)

---

## 1. Overview

All scripts live in `scripts/` at the repository root. They are **not** installed
as console entry points; run them with Python from the project root:

```bash
cd LLM/GPT-OSS-Lite
python3 scripts/<name>.py [args]
```

Most profiling scripts import `_bootstrap.py`, which prepends the repo root to
`sys.path` and provides `micro_cfg()` plus `time_fn()` for repeatable timing.

| Script | GPU required | Primary purpose |
|--------|--------------|-----------------|
| `_bootstrap.py` | No | Shared helpers (import only) |
| `check_docs.py` | No | Lint docs, refresh size table |
| `kv_cache_benchmark.py` | No | Analytical KV-cache reduction (≥1.8×) |
| `passkey_eval.py` | Optional | Long-context passkey retrieval accuracy |
| `e2e_gpu_smoke.py` | Yes (~4 GB) | Full pipeline integration on tiny model |
| `profile_components.py` | Optional | Per-component forward latency |
| `profile_moe.py` | Optional | MoE forward + dispatch latency |
| `profile_inference.py` | Optional | `generate()` tokens/sec |
| `microbench_a100.py` | Optional | Peak VRAM vs threshold |
| `step_time_a100.py` | Optional | Training step time + MFU |

Training itself is **not** a script under `scripts/` — use
`python training/pretrain.py --config configs/pretrain_a100_502m.yaml`.
See [training.md](training.md) and [configs](configs.md) (when published).

---

## 2. Shared bootstrap — `_bootstrap.py`

**Purpose:** Deduplicate `sys.path` setup and micro-benchmark utilities across
profiling scripts.

**Exports:**

| Symbol | Description |
|--------|-------------|
| `time_fn(fn, n=20, warmup=3)` | Average milliseconds per call after warmup; CUDA-synced when available |
| `micro_cfg()` | `ModelConfig` with 4 layers, `d_model=64`, `max_seq_len=128` — fast CPU/GPU runs |

**Usage pattern:**

```python
from _bootstrap import micro_cfg, time_fn
```

Scripts that `import _bootstrap` must be run with `scripts/` as cwd **or** rely
on the path fix inside `_bootstrap` (parent of `scripts/` is added automatically).

**Expected output:** None — library module only.

---

## 3. Documentation linter — `check_docs.py`

**Purpose:** Validate all `documentation/*.md` files for broken links, stale
patterns, control characters, and missing backtick paths. Optionally refresh the
doc size table in [README.md](README.md) or stamp verification footers.

### Invocation

```bash
# Lint only (exit 0 = clean)
python3 scripts/check_docs.py

# Refresh ## Doc size reference table in documentation/README.md
python3 scripts/check_docs.py --update-sizes

# Stamp <!-- docs:verified YYYY-MM-DD · <git-short> --> on all docs
python3 scripts/check_docs.py --stamp-footers

# Both
python3 scripts/check_docs.py --update-sizes --stamp-footers
```

### Stale patterns rejected

The linter flags outdated documentation conventions:

| Pattern | Reason |
|---------|--------|
| LaTeX `{` `,` `}` thousand separators | Use plain `100000` or comma prose in docs |
| Hard-coded pytest totals in prose | Run `pytest` for the current count |
| Old MoE Triton standalone doc name | Use [triton_kernels.md](triton_kernels.md) |
| Removed Triton env-var gate | Use `moe_dispatch: triton_grouped` in YAML |

### Expected output

```
check_docs: OK (N files)
```

On failure, prints `documentation/<file>:<line>: <message>` to stderr and exits 1.

---

## 4. Headline metric #1 — `kv_cache_benchmark.py`

**Purpose:** Prove the **architectural KV-cache reduction** of the alternating
sliding-window / full-attention design without loading a model or GPU. This is
the analytical counterpart to `MixedKVCache` described in
[architecture.md](architecture.md) §9 and [OPTIMIZATIONS.md](OPTIMIZATIONS.md)
(OPT-11/12).

### Architecture constants (from `configs/pretrain_a100_502m.yaml`)

| Constant | Value |
|----------|-------|
| Layers | 12 (6 windowed + 6 global) |
| Window | 128 |
| GQA | 4 KV heads × 96 head_dim |
| Dtype | BF16 (2 bytes) |
| Batch | 1 |

### Formula

Per token per layer, KV storage is:

$$
\text{bytes}_{\text{token,layer}} = 2 \times n_{\text{kv\_heads}} \times d_{\text{head}} \times \text{dtype\_bytes}
$$

- **Pure GQA (all full):** $12 \times T \times \text{bytes}_{\text{token,layer}}$
- **Mixed SWA/full:** $(6 \times \min(128, T) + 6 \times T) \times \text{bytes}_{\text{token,layer}}$

### Invocation

```bash
python3 scripts/kv_cache_benchmark.py
```

No arguments. Runs in milliseconds on CPU.

### Expected output

```
GPT-OSS-Lite KV-cache benchmark (analytical, BF16, batch=1)
Architecture: 12 layers (6 SWA w=128 + 6 full)
GQA: 4 KV heads, head_dim=96

   Context      Pure GQA      SWA/Full   Reduction
--------------------------------------------------
     4,096        0.09 GB        0.06 GB       1.50×
    ...
   131,072        2.88 GB        1.50 GB       1.92×

✅ HEADLINE METRIC PASSED: 1.92× KV-cache reduction at 128K (≥ 1.8×)
```

Exit code **0** if reduction at 128K ≥ **1.8×**; otherwise exit **1**.

At long context the ratio approaches $12 / (6 + 6 \times 128/131072) \approx 1.92\times$.
See [foundations.md](foundations.md) §4 for the intuition.

---

## 5. Headline metric #2 — `passkey_eval.py`

**Purpose:** Measure **passkey retrieval accuracy** at long context lengths using
`inference/long_context.py::PasskeyEvaluator`. Target: **≥85%** at 128K on a
**trained** checkpoint.

### Invocation

```bash
python3 scripts/passkey_eval.py \
  --checkpoint checkpoints/pretrain_a100/model_step_60000.safetensors \
  --n-trials 10 \
  --context-lengths 4096 8192 32768 65536 131072 \
  --position middle \
  --seed 42
```

| Flag | Default | Description |
|------|---------|-------------|
| `--checkpoint` | (required) | Path to `.safetensors` weights |
| `--n-trials` | 10 | Trials per context length |
| `--context-lengths` | 4096 … 131072 | Space-separated list |
| `--position` | `middle` | Passkey placement: `start`, `middle`, `end` |
| `--seed` | 42 | Base RNG seed |

Loads `configs/pretrain_a100_502m.yaml` for `ModelConfig`. Uses a lightweight
char-level tokenizer stub for eval plumbing — swap for production BPE when the
data pipeline tokenizer is wired (see [data_pipeline.md](data_pipeline.md)).

### Expected output (trained model)

```
Passkey eval: checkpoint=model_step_60000.safetensors, device=cuda

   Context    Accuracy
------------------------
     4,096      100.0%
    ...
   131,072       87.5%

✅ HEADLINE METRIC PASSED: 87.5% at 131,072 (≥ 85%)
```

### Untrained model

On a random-init checkpoint, accuracy is near chance. The script still exits **0**
but prints a warning — full pretraining is required for the headline pass.
See [getting_started.md](getting_started.md) when available.

---

## 6. End-to-end GPU smoke — `e2e_gpu_smoke.py`

**Purpose:** Single-script integration test exercising every major subsystem on a
**tiny** model that fits in **~4 GB VRAM** (verified on sm_75). Run this after
any change to attention, MoE, checkpointing, or generation.

### What it tests (8 steps)

1. Build 4-layer GPU model (BF16)
2. Forward + backward; all params receive gradients
3. MoE `stacked` vs `triton_grouped` numerical equivalence (when Triton present)
4. `MoELayer` end-to-end with `moe_dispatch="triton_grouped"`
5. Five-step training loop (LR schedule, grad clip, AdamW)
6. `CheckpointManager` save → load round-trip
7. `MixedKVCache` generation on windowed/global split
8. YaRN forward at `eval_max_seq_len` > training `max_seq_len`

### Invocation

```bash
# Requires CUDA GPU with ≥4 GB VRAM
python3 scripts/e2e_gpu_smoke.py
```

### Expected output

Green checkmarks per step:

```
======================================================================
  Step 1: Build tiny GPU model (4 layers, d_model=128)
======================================================================
  ✓ Model built (0.xxxM params, fits in 4 GB)
...
======================================================================
  Step 7: MixedKVCache inference (windowed + global split)
======================================================================
  ✓ generate() with cache: output shape (1, T_prompt + 16)
...
ALL STEPS PASSED
```

Exit **0** on success; raises `SystemExit(1)` on any failure.

**Related:** [triton_kernels.md](triton_kernels.md), [utils.md](utils.md) (checkpoints).

---

## 7. Component profiler — `profile_components.py`

**Purpose:** Break down forward latency by subsystem on `micro_cfg()` — useful
when deciding which [OPTIMIZATIONS.md](OPTIMIZATIONS.md) entry to profile next.

### Invocation

```bash
python3 scripts/profile_components.py
```

Uses CUDA when available; otherwise CPU (slower, still valid for relative ordering).

### Components timed

| Label | What is measured |
|-------|------------------|
| `[model.forward]` | Full `GPTOSS` forward |
| `[attn.windowed]` | `GPTOSSAttention` layer 0 (SWA) |
| `[attn.global]` | `GPTOSSAttention` layer 1 (full) |
| `[manual_attn]` | O(T²) reference attention |
| `[sdpa_attn]` | `F.scaled_dot_product_attention` causal |
| `[swa_attn]` | `causal_attention` with window |
| `[moe.forward]` | Full `MoELayer` |
| `[moe.dispatch]` | `_dispatch_vectorized` only |
| `[apply_rope]` | `apply_rope` |
| `[repeat_kv]` | GQA head expansion |

### Expected output (example)

```
Total params: 0.42M
[model.forward]      12.34 ms/step
[attn.windowed]       1.23 ms/step
[attn.global]         2.45 ms/step
...
[repeat_kv]           0.01 ms/step
```

Absolute numbers vary by hardware. Compare **before/after** a change, not across machines.

---

## 8. MoE dispatch profiler — `profile_moe.py`

**Purpose:** Isolate MoE forward vs vectorized dispatch cost without YaRN prune
overhead (`yarn_prune_rope_global=False` via `dataclasses.replace`).

### Invocation

```bash
python3 scripts/profile_moe.py
```

### Expected output

```
moe.forward          3.45 ms
_dispatch_vectorized 2.10 ms
```

For Triton grouped dispatch timing, enable `moe_dispatch="triton_grouped"` in a
custom config and profile via `e2e_gpu_smoke.py` or a one-off script. See
[moe.md](moe.md) and [triton_kernels.md](triton_kernels.md).

---

## 9. Inference profiler — `profile_inference.py`

**Purpose:** Measure `inference/generate.py::generate()` throughput (tokens/sec)
for several prompt lengths with greedy decoding (`temperature=0.0`).

### Invocation

```bash
python3 scripts/profile_inference.py
```

### Expected output

```
prompt=8, new=64: 45.2 ms (1416 tok/s)
prompt=32, new=64: 52.1 ms (1228 tok/s)
prompt=128, new=64: 61.3 ms (1044 tok/s)
```

Decode cost is dominated by per-step `MixedKVCache` updates and MoE forward.
Longer prompts increase prefill time but decode tok/s should stabilize once the
cache is warm. See OPT-11/12/13/14/22 in [OPTIMIZATIONS.md](OPTIMIZATIONS.md).

---

## 10. VRAM microbench — `microbench_a100.py`

**Purpose:** Verify production model fits under a VRAM ceiling at
`batch_size=8`, `seq_len=4096`. Uses **actual** `torch.cuda.max_memory_allocated`
on GPU; falls back to `utils.memory.estimate_model_memory_gb` on CPU.

### Invocation

```bash
python3 scripts/microbench_a100.py \
  --config configs/pretrain_a100_502m.yaml \
  --batch-size 8 \
  --seq-len 4096 \
  --threshold-gb 25.0
```

| Flag | Default | Description |
|------|---------|-------------|
| `--config` | `configs/pretrain_a100_502m.yaml` | Model YAML |
| `--batch-size` | 8 | Micro-batch size |
| `--seq-len` | 4096 | Sequence length |
| `--threshold-gb` | 25.0 | Max acceptable peak GB (headroom for scaling) |

### Expected output (A100 80GB)

```
[microbench] Config: d_model=768, n_layers=12, vocab=128000, experts=8
[microbench] batch_size=8, seq_len=4096
[microbench] Peak VRAM (actual): 18.42 GB
[microbench] ✅ PASSED: peak < 25.0 GB
```

Estimator math is documented in [utils.md](utils.md).

---

## 11. Step-time / MFU — `step_time_a100.py`

**Purpose:** Measure training **tokens/sec** and approximate **MFU** (Model FLOPs
Utilization) on A100 BF16. Optional `torch.compile(max-autotune)` mirrors
production `training.compile: true`.

### Invocation

```bash
python3 scripts/step_time_a100.py \
  --config configs/pretrain_a100_502m.yaml \
  --batch-size 8 \
  --seq-len 4096 \
  --steps 20 \
  --warmup 5 \
  --compile
```

| Flag | Default | Description |
|------|---------|-------------|
| `--steps` | 20 | Measured steps after warmup |
| `--warmup` | 5 | Warmup steps (discarded) |
| `--compile` | off | Enable `torch.compile(mode="max-autotune")` |

Enables TF32 + cuDNN benchmark (see OPT-20 in [OPTIMIZATIONS.md](OPTIMIZATIONS.md)).

### Expected output (A100, with `--compile`)

```
[step_time] torch.compile enabled (mode=max-autotune)
[step_time] Config: 12L, batch=8, seq=4096
[step_time] Warmup: 5 steps, Measure: 20 steps
[step_time] 20 steps in 42.15s → 1,556,234 tokens/sec
[step_time] Approx MFU: 38.2% (achieved 119.2 TFLOPS BF16)
[step_time] ✅ MFU target (≥35%) met.
```

Exit **0** if MFU ≥ 35% on CUDA; CPU-only runs report tokens/sec without MFU.

**Note:** This script uses plain `cross_entropy`, not `chunked_cross_entropy` —
it measures raw step time, not the production CE path. For production parity see
[training.md](training.md).

---

## 12. Script selection guide

Use this decision tree when debugging or benchmarking:

```
Need to verify doc links / stale patterns?
  → check_docs.py

Need headline KV reduction proof (no GPU)?
  → kv_cache_benchmark.py  (target ≥1.8× at 128K)

Need long-context quality on trained weights?
  → passkey_eval.py  (target ≥85% at 128K)

Changed attention / MoE / inference / checkpoint code?
  → e2e_gpu_smoke.py  (4 GB GPU)

Which layer is slow?
  → profile_components.py

Is MoE dispatch the bottleneck?
  → profile_moe.py

Is decode slow?
  → profile_inference.py

Will training OOM at B=8, T=4096?
  → microbench_a100.py

What is training throughput / MFU?
  → step_time_a100.py --compile
```

### Recommended CI / pre-push sequence

```bash
python3 scripts/check_docs.py
python3 scripts/kv_cache_benchmark.py
pytest tests/ -q                    # see project README for test layout
python3 scripts/e2e_gpu_smoke.py  # when GPU available
```

---

## 13. Related documentation

| Topic | Document |
|-------|----------|
| Training loop CLI | [training.md](training.md) |
| Mixed KV cache design | [architecture.md](architecture.md) §9, [OPTIMIZATIONS.md](OPTIMIZATIONS.md) |
| Passkey eval internals | `inference/long_context.py`, [inference.md](inference.md) |
| MoE + Triton opt-in | [moe.md](moe.md), [triton_kernels.md](triton_kernels.md) |
| Memory estimator | [utils.md](utils.md) |
| Config reference | [configs](configs.md), `configs/pretrain_a100_502m.yaml` |
| Optimization catalog | [OPTIMIZATIONS.md](OPTIMIZATIONS.md) |
| Book index | [README.md](README.md) |

---

<!-- docs:verified 2026-07-31 · fa6f918 -->
