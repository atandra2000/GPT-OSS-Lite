# Scripts — Launch, Benchmark, and Smoke Tests

> **Covers:** `scripts/` — operational tooling for GPU validation and headline metrics.

---

## Table of Contents

1. [Overview](#overview)
2. [kv_cache_benchmark.py](#kv_cache_benchmarkpy)
3. [passkey_eval.py](#passkey_evalpy)
4. [microbench_a100.py](#microbench_a100py)
5. [step_time_a100.py](#step_time_a100py)
6. [e2e_gpu_smoke.py](#e2e_gpu_smokepy)
7. [profile_*.py](#profile-scripts)
8. [check_docs.py](#check_docspy)
9. [Recommended Workflow](#recommended-workflow)

---

## Overview

Scripts answer three questions before committing GPU-days:

1. **Does the headline metric hold?** (`kv_cache_benchmark.py`)
2. **Does it fit?** (`microbench_a100.py`)
3. **Is it fast enough?** (`step_time_a100.py`)

| Script | Purpose | Requires GPU |
|---|---|---|
| `kv_cache_benchmark.py` | Analytical KV-cache reduction table | No |
| `passkey_eval.py` | 128K passkey retrieval accuracy | Yes (for real eval) |
| `microbench_a100.py` | Peak VRAM measurement | Yes |
| `step_time_a100.py` | ms/step, tokens/sec, MFU | Yes |
| `e2e_gpu_smoke.py` | End-to-end GPU validation | Yes |
| `profile_components.py` | Per-component ms/op | Optional |
| `profile_moe.py` | MoE dispatch profiling | Optional |
| `profile_inference.py` | Generation throughput | Optional |
| `check_docs.py` | Lint documentation links/paths | No |

---

## kv_cache_benchmark.py

**Purpose:** Reproduce headline metric #1 — KV-cache reduction at 128K.

Computes analytical BF16 cache sizes for pure GQA vs SWA(128)/full alternation at context lengths 4K–128K. No model weights loaded.

```bash
python3 scripts/kv_cache_benchmark.py
# Expected: ✅ HEADLINE METRIC PASSED: ≥ 1.8× reduction at 128K
```

Formula: per layer, per token: `2 × n_kv_heads × head_dim × 2 bytes` (K+V, BF16). Windowed layers cap at `window_size`; global layers use full `seq_len`.

---

## passkey_eval.py

**Purpose:** Reproduce headline metric #2 — passkey retrieval at 128K.

Thin CLI over `inference/long_context.py:PasskeyEvaluator`.

```bash
python3 scripts/passkey_eval.py \
    --checkpoint checkpoints/pretrain_a100/model_step_60000.safetensors \
    --n-trials 100 \
    --context-lengths 4096 32768 65536 131072
```

Requires a **trained** checkpoint. Untrained models return ~0% accuracy (stub mode).

---

## microbench_a100.py

**Purpose:** Measure peak VRAM for production config at `max_seq_len=4096`.

```bash
python3 scripts/microbench_a100.py
```

Pre-flight before full training. Uses `utils/memory.py` estimator + measured peak.

---

## step_time_a100.py

**Purpose:** Measure training step time and estimate wall-clock for 61K steps.

```bash
python3 scripts/step_time_a100.py --steps 20 --warmup 5
```

Target: 35–40% MFU on A100 80GB.

---

## e2e_gpu_smoke.py

**Purpose:** Full GPU smoke test — Triton MoE (if available), forward/backward, KV-cache inference.

```bash
python3 scripts/e2e_gpu_smoke.py
```

Run after any change to `models/moe_triton.py` or inference path.

---

## profile_*.py

Debug aids for hotspot identification. Cross-reference with [OPTIMIZATIONS.md](OPTIMIZATIONS.md).

```bash
python3 scripts/profile_components.py
python3 scripts/profile_moe.py
python3 scripts/profile_inference.py
```

---

## check_docs.py

Validates documentation quality:

```bash
python3 scripts/check_docs.py
python3 scripts/check_docs.py --update-sizes --stamp-footers
```

Checks: internal links, backtick-quoted repo paths, stale test-count patterns.

---

## Recommended Workflow

```
CPU:  pytest tests/ -q
CPU:  python3 scripts/kv_cache_benchmark.py
CPU:  python3 scripts/check_docs.py
GPU:  python3 scripts/e2e_gpu_smoke.py
GPU:  python3 scripts/microbench_a100.py
GPU:  python3 scripts/step_time_a100.py --steps 20 --warmup 5
GPU:  python3 training/pretrain.py --config configs/pretrain_a100_502m.yaml --seed 42
POST: python3 scripts/passkey_eval.py --checkpoint <ckpt>
```

<!-- docs:verified 2026-07-31 · fd4fe36 -->
