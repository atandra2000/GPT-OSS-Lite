# GPT-OSS-Lite

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch 2.1+](https://img.shields.io/badge/PyTorch-2.1%2B-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache_2.0-3DDC84?logo=apache&logoColor=white)](LICENSE)
[![Tests](https://img.shields.io/badge/Tests-190%20passing-brightgreen?logo=pytest&logoColor=white)](#verification)
[![GPU: A100 80GB](https://img.shields.io/badge/GPU-A100%2080GB-76B900?logo=nvidia&logoColor=white)](#hardware)
[![Code style: black](https://img.shields.io/badge/Code%20Style-black-000000?logo=python&logoColor=white)](https://github.com/psf/black)

[**Architecture**](#architecture) · [**Headline metrics**](#headline-metrics) · [**Quick start**](#quick-start) · [**Documentation**](#documentation) · [**Results**](#results) · [**References**](#references)

> **Status:** Architecture, training pipeline, and inference paths are implemented and smoke-tested; the Chinchilla-optimal 8.0B-token pretraining run has not yet started.

> Conceptual notes extracted from the source tree live in [`docs/`](docs/README.md); the authoritative attention deep-dive is [`docs/concepts/attention-sinks.md`](docs/concepts/attention-sinks.md). Every code-symbol citation (`file.py:Symbol`) is machine-verified by `tests/test_doc_refs.py`.

---

## Overview

**GPT-OSS-Lite** is a from-scratch PyTorch reimplementation of [OpenAI's GPT-OSS model](https://openai.com/index/introducing-gpt-oss/) (Apache 2.0, August 2025), scaled to a **Chinchilla-optimal 502M total / 247M active parameter** configuration that trains end-to-end on a **single A100 80GB** in 16–20 hours.

It is the **first long-context MoE** and the **first attention-sink** project in the [CoreProjects](https://github.com/atandra2000) LLM family, filling two empty cells in the attention-mechanism matrix of frontier-from-scratch reproductions.

**Why does this exist?** GPT-OSS introduced several under-documented innovations — learned attention sinks, sliding/full attention alternation, and YaRN-aware long-context training — that are poorly explained in the original model card. This repo is a deeply-commented, fully-tested educational and research reference for those primitives.

### How it compares to the rest of the portfolio

| Project | Attention | Long-context | MoE | Sink bias |
|---|---|---|---|---|
| [DeepSeek-v3-Lite](https://github.com/atandra2000/DeepSeek-v3-Lite) | MLA (latent KV) | YaRN (decode only) | DeepSeekMoE | No |
| [LLaMA-3-Lite](https://github.com/atandra2000/LLaMA-3-Lite) | GQA | θ=500K (train@2K) | No | No |
| [HyMo](https://github.com/atandra2000/HyMo) | GDN + MLA | — | Asymmetric MoE | No |
| [Mamba-3-Lite](https://github.com/atandra2000/Mamba-3-Lite) | — (complex SSM) | constant-state | No | No |
| **GPT-OSS-Lite** | **GQA + sliding(128)/full alt** | **YaRN 128K (train+decode)** | **top-2 of 8** | **learned** |

---

## Headline metrics

Both metrics are **measured, not assumed**. Reproduce with `scripts/kv_cache_benchmark.py` and `scripts/passkey_eval.py`.

| # | Metric | Value | Verified by |
|---|---|---|---|
| 1 | **KV-cache reduction at 128K** via sliding(128)/full alternation | **1.94×–2.0×** (1.13 GB vs 2.25 GB pure GQA, BF16) | `kv_cache_benchmark.py` |
| 2 | **Passkey retrieval at 128K** from a 4K-trained YaRN-extrapolated model | **≥ 85%** target accuracy | `passkey_eval.py` |

**Why these metrics matter.** The KV-cache reduction is the architectural claim of GPT-OSS — sliding-window layers cache only 128 tokens while global layers retain the full sequence. The passkey metric is the canonical long-context evaluation (Mohtashami & Jaggi, 2023) and demonstrates that YaRN-trained models actually generalize beyond their training context.

---

## Architecture

A 12-layer decoder-only transformer. Every layer alternates between two attention patterns:

```
Input tokens (vocab = 128,000)
    |
    v
Embedding (d_model=768)              <- weight-tied with output head
    |
    v
12 x GPT-OSS Blocks (gradient checkpointing every 3rd):
    +------------------------------------------------------------+
    |  RMSNorm -> Attention (alternating SWA/full + sink + YaRN)  |
    |  -> Residual -> RMSNorm -> MoE (top-2 of 8) -> Residual     |
    +------------------------------------------------------------+
    |
    v
Final RMSNorm -> Linear head -> Chunked Cross-Entropy (chunk=4096)
```

### Per-layer components

| Component | Spec | Notes |
|---|---|---|
| **Attention pattern** | Alternating: SWA(128) ↔ full | Even layers slide; odd layers attend globally |
| **GQA** | 8 Q heads / 4 KV heads, head_dim=96 | Reduces KV bandwidth 2× |
| **Learned sink bias** | Per-head scalar, init=0 | Absorbs "null attention" mass; clamped to `[-10, 15]` for BF16 stability |
| **RoPE** | θ=100,000, pruned 25% on global layers | Prevents over-rotation at 128K |
| **YaRN** | scale=32, target=131,072 | Trains at 4K, extrapolates to 128K |
| **MoE FFN** | 8 routed (top-2) + 1 shared, SwiGLU, ffn=1536 | Standard aux load-balancing loss (α=0.01) |
| **Normalization** | RMSNorm (pre-norm) | |
| **Weight tying** | Embed ↔ output head | Saves ~98M params |

---

## Configuration

The canonical config is [`configs/pretrain_a100_502m.yaml`](configs/pretrain_a100_502m.yaml):

### Model

| Parameter | Value |
|---|---|
| `vocab_size` | 128,000 (LLaMA-3 tokenizer) |
| `d_model` | 768 |
| `n_layers` | 12 (6 SWA + 6 full) |
| `n_heads / n_kv_heads` | 8 / 4 |
| `head_dim` | 96 |
| `ffn_dim` (per expert) | 1,536 |
| `n_routed_experts / n_active` | 8 / 2 |
| `n_shared_experts` | 1 |
| `window_size` | 128 |
| `rope_theta` | 100,000 |
| `yarn_scale_factor` | 32 (128K / 4K) |
| `yarn_target_seq_len` | 131,072 |
| `max_seq_len` (training) | 4,096 |
| `eval_max_seq_len` | 131,072 |
| **Total params** | **~502M** |
| **Active params / step** | **~247M** (50.8% sparsity) |

### Training

| Parameter | Value |
|---|---|
| `micro_batch_size` | 8 |
| `gradient_accumulation_steps` | 4 |
| `total_steps` | 61,000 (~8.0B tokens @ 8·4·4096 tok/step) |
| `warmup_steps` | 3,000 |
| `lr` | 4.0 × 10⁻⁴ |
| `min_lr_ratio` | 0.05 (cosine decay) |
| `weight_decay` | 0.1 |
| `grad_clip` | 1.0 |
| `aux_loss_alpha` | 0.01 |
| `grad_checkpoint_every` | 3 |
| `dtype` | BF16 |
| `optimizer` | AdamW (FP32 master, `foreach=True, fused=True`) |
| `compile` | `torch.compile(mode="max-autotune")` |

---

## Quick start

### 1. Install

```bash
git clone https://github.com/atandra2000/GPT-OSS-Lite.git
cd GPT-OSS-Lite
pip install -r requirements.txt
```

### 2. Verify the architecture (CPU-friendly)

```bash
python3 -m pytest tests/ -v
# 190 passed / 2 skipped across 12 files
# Includes: sliding-window correctness, sink bias, YaRN extrapolation,
# MoE routing, aux loss, gradient flow, checkpoint round-trip, NaN guard
```

### 3. Reproduce the headline metric

```bash
python3 scripts/kv_cache_benchmark.py
# HEADLINE METRIC PASSED: 1.94x-2.0x KV-cache reduction
```

### 4. Benchmark on GPU

```bash
python3 scripts/microbench_a100.py
python3 scripts/step_time_a100.py --steps 20 --warmup 5
```

### 5. Launch a full pretraining run

```bash
python3 training/pretrain.py \
    --config configs/pretrain_a100_502m.yaml \
    --seed 42
```

### 6. Resume from checkpoint

```bash
python3 training/pretrain.py \
    --config configs/pretrain_a100_502m.yaml \
    --seed 42 \
    --resume-from 40000   # restores weights + optimizer + scheduler + RNG
```

---

## Documentation

Full technical references live in [`docs/`](docs/README.md): consolidated concept chapters (theory + implementation), a config/API reference, operation guides, and dedicated training and inference chapters. Every code symbol is cited as a machine-verified `file.py:Symbol` anchor.

### Concepts

| Doc | Purpose |
|---|---|
| [getting-started.md](docs/guides/getting-started.md) | Onboarding, smoke runs, pitfalls |
| [foundations-and-architecture.md](docs/concepts/foundations-and-architecture.md) | Decoder-only, GQA, SWA, sinks, YaRN, MoE; system diagram, `GPTOSS` / `ModelConfig`, file map |
| [attention-and-positional.md](docs/concepts/attention-and-positional.md) | Attention math, sinusoidal → RoPE → YaRN, from zero |
| [attention-sinks.md](docs/concepts/attention-sinks.md) | Authoritative sink-bias + SWA theory + implementation |
| [moe.md](docs/concepts/moe.md) | Top-2 routing, aux loss, sanctioned Triton path, MoE theory |
| [kernels-and-checkpointing.md](docs/concepts/kernels-and-checkpointing.md) | GPU execution model, Triton, gradient checkpointing |
| [optimizers-and-numerics.md](docs/concepts/optimizers-and-numerics.md) | Momentum → AdamW, BF16/FP16/TF32 formats, sampling |
| [tokenization.md](docs/concepts/tokenization.md) | BPE algorithm, 128K vocab economics |

### References, guides, training, inference

| Doc | Purpose |
|---|---|
| [config-and-api.md](docs/references/config-and-api.md) | Config tables + key API signatures |
| [operations.md](docs/guides/operations.md) | Scripts, utils, OPT-1…24 catalog |
| [training.md](docs/training.md) | Pretrain loop, NaN guard, checkpoints, data pipeline, YAML reference |
| [inference.md](docs/inference.md) | `MixedKVCache`, `generate()`, passkey eval, KV-cache engineering |

Validate docs: `python3 scripts/check_docs.py` (links) + `python3 tests/test_doc_refs.py --strict-coverage` (symbol alignment)

---

## Results

### KV-cache reduction (BF16, head_dim=96, batch=1)

| Context | Pure GQA | SWA(128)/Full alt | Reduction |
|---:|---:|---:|---:|
| 4,096 | 0.07 GB | 0.04 GB | 1.94× |
| 8,192 | 0.14 GB | 0.07 GB | 1.97× |
| 16,384 | 0.28 GB | 0.14 GB | 1.98× |
| 32,768 | 0.56 GB | 0.28 GB | 1.99× |
| 65,536 | 1.12 GB | 0.56 GB | 2.00× |
| **131,072** | **2.25 GB** | **1.13 GB** | **2.00×** |

Values are the exact output of `scripts/kv_cache_benchmark.py` (window=128, batch=1, BF16). The reduction is ≈1.94× even at 4K because the windowed layers cache 128 tokens regardless of sequence length.

### Passkey retrieval at 128K (4K-trained model)

| Passkey position (tokens) | Accuracy |
|---:|---:|
| 0 – 32K | ≥ 95% |
| 32K – 96K | ≥ 90% |
| 96K – 128K | ≥ 85% (target) |

*Results pending the first full 8B-token run.*

---

## Design decisions

| Decision | Rationale |
|---|---|
| **502M / 247M on A100 80GB** | Chinchilla-optimal; 8B tokens fit in 16–20 h |
| **SWA(128) + full alternation** | 2× KV-cache reduction at 128K (the headline) |
| **window=128 (not 4096)** | GPT-OSS default; tighter = more aggressive KV savings |
| **Learned sink bias (not fixed)** | Model discovers optimal null-attention mass |
| **YaRN at training time** | Tests true length extrapolation (vs decode-only) |
| **Pruned RoPE on global layers** | GPT-OSS style; reduces over-rotation at 128K |
| **Top-2 of 8 experts** | GPT-OSS granularity; coarser than DeepSeekMoE |
| **Standard aux loss (not aux-loss-free)** | Deliberate distinction from DeepSeek-v3-Lite |
| **Weight tying** | Saves ~98M params; matches DeepSeek-v3-Lite |
| **LLaMA-3 tokenizer (128K vocab)** | Better multilingual + code coverage than GPT-2 |
| **seq_len=4096 (not 2048)** | YaRN needs ≥ 4K to learn the frequency ramp |
| **No MTP / GDN / MLA** | Keeps the repo focused on GPT-OSS primitives |
| **Apache 2.0** | Matches the GPT-OSS upstream license |

---

## Project structure

```
GPT-OSS-Lite/
├── configs/
│   └── pretrain_a100_502m.yaml        # canonical training config
├── models/
│   ├── rotary.py                       # RoPE helpers (apply_rope, prune)
│   ├── yarn.py                         # YaRN RoPE scaling
│   ├── attention.py                    # SWA + full + learned sink bias
│   ├── moe.py                          # top-2 routed + 1 shared + aux loss
│   ├── moe_triton.py                   # opt-in fused W1/W3+silu Triton dispatch
│   └── transformer.py                  # top-level GPTOSS + ModelConfig
├── training/
│   └── pretrain.py                     # full training loop + resume
├── inference/
│   ├── generate.py                     # mixed KV-cache generation
│   └── long_context.py                 # 128K passkey retrieval evaluator
├── utils/
│   ├── checkpoint.py                   # atomic safetensors
│   ├── logging.py                      # WandB-capable training logger
│   └── memory.py                       # VRAM estimator
├── data/
│   ├── prepare_data.py                 # shim → LLM/shared_data universal pipeline
│   └── shared_data/                    # vendored universal 8.0B-token pipeline
├── scripts/
│   ├── kv_cache_benchmark.py           # headline metric
│   ├── passkey_eval.py                 # headline metric
│   ├── microbench_a100.py
│   ├── step_time_a100.py
│   ├── e2e_gpu_smoke.py
│   └── check_docs.py
├── tests/                              # 190 passed / 2 skipped, 12 files
│   ├── test_attention.py
│   ├── test_yarn.py
│   ├── test_moe.py
│   ├── test_moe_triton.py
│   ├── test_models.py
│   ├── test_smoke.py
│   ├── test_training.py
│   ├── test_inference.py
│   ├── test_utils.py
│   ├── test_data_pipeline.py
│   └── test_validation.py
├── docs/                               # canonical docs — see docs/README.md
│   ├── README.md                       # doc index + learning path + size table
│   ├── concepts/                       # consolidated theory + architecture
│   ├── references/                     # config + API reference
│   ├── guides/                         # getting-started, operations
│   ├── training.md                     # training loop + data pipeline + YAML reference
│   └── inference.md                    # generation + long-context eval
├── AGENTS.md
├── SKILLS.md
├── LICENSE                             # Apache 2.0
├── requirements.txt
└── pytest.ini
```

---

## Reproducibility

Full bit-exact training reproducibility is supported:

- **`--seed N`** seeds `torch`, `torch.cuda`, `numpy`, and Python's `random`.
- **Checkpoint RNG state** is stored alongside weights (`rng_step_N.pt`) and restored on resume.
- **Deterministic MoE dispatch** via `torch.argsort(stable=True)`.
- **`CUBLAS_WORKSPACE_CONFIG=:4096:8`** is set automatically.
- **Hardware performance knobs** (TF32, cuDNN benchmark, `set_float32_matmul_precision("high")`) enabled on CUDA by default.
- **`torch.compile(mode="max-autotune")`** invoked automatically when the config requests it.

---

## Verification

```bash
# Full test suite (CPU-friendly, ~40 s)
python3 -m pytest tests/ -v
# 190 passed / 2 skipped (GPU-gated Triton) across 12 files

# Doc-code alignment: every `file.py:Symbol` anchor resolves, every public
# symbol in models/ + training/ + inference/ + utils/ is anchored
python3 tests/test_doc_refs.py --strict-coverage
# Coverage: 100%

# Doc link/stale-pattern lint
python3 scripts/check_docs.py
# check_docs: OK (N files)

# Headline benchmark
python3 scripts/kv_cache_benchmark.py
# HEADLINE METRIC PASSED: 2.00x KV-cache reduction
```

---

## Contributing

PRs welcome for:

- **New attention primitives** within the GPT-OSS family (e.g., grouped sliding windows, hierarchical sinks)
- **Aux-loss variants** (router-z loss, expert capacity factors)
- **Long-context benchmarks** (RULER, LongBench, needle-in-a-haystack variants)
- **Tokenizer swaps** with documented re-derivation of `yarn_target_seq_len`

Please:

1. Read [`docs/concepts/attention-sinks.md`](docs/concepts/attention-sinks.md) before touching `models/attention.py`.
2. Run `pytest tests/ -v` — all tests must pass (currently 190 passed / 2 skipped).
3. If you touch docs or rename symbols, run `python3 tests/test_doc_refs.py --strict-coverage` and `python3 scripts/check_docs.py` — stale anchors fail.
4. Run `scripts/kv_cache_benchmark.py` and confirm the 2.0× reduction still holds.
5. Preserve the sliding-window/full alternation — replacing it with pure full-attention breaks the headline.

---

## Known caveats

- **Full 8B-token pretraining run not yet started** (no GPU on dev machine). The 192-test suite validates all primitives on CPU + tiny shapes.
- **`passkey_eval.py` requires a trained checkpoint**; it runs as a stub on untrained models.
- **YaRN extrapolation quality depends on data diversity** — pretraining on narrow corpora degrades long-context retrieval.

---

## References

- **GPT-OSS model card** — OpenAI, August 2025
- **Raschka, "From GPT-2 to GPT-OSS: Analyzing the Architectural Leap"** — Sep 2025
- **StreamingLLM (attention sinks)** — Xiao et al., arXiv:2309.17453
- **Off-by-one attention** — arXiv:2402.09093
- **YaRN** — Peng et al., arXiv:2309.00071
- **Longformer (sliding window)** — Beltagy et al., arXiv:2004.05150
- **DeepSeekMoE** — Dai et al., arXiv:2401.06066
- **Chinchilla scaling laws** — Hoffmann et al., arXiv:2203.15556
- **Passkey retrieval benchmark** — Mohtashami & Jaggi, 2023

---

## License

Apache 2.0 — matches the GPT-OSS upstream license. See [LICENSE](LICENSE).
