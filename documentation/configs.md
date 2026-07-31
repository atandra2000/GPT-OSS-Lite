# Configuration Reference — YAML Encyclopedia

> **Chapter on `configs/pretrain_a100_502m.yaml` and
> `configs/pretrain_gpu_smoke.yaml`.** Every `model`, `training`, and `data` key
> is documented with defaults, valid ranges, and interaction effects. For how
> configs connect to code, see [architecture.md](architecture.md) and
> [transformer.md](transformer.md). Training loop behavior:
> [training.md](training.md).

---

## Table of contents

1. [How configs are loaded](#1-how-configs-are-loaded)
2. [File comparison](#2-file-comparison)
3. [Derived quantities](#3-derived-quantities)
4. [`model` block — field reference](#4-model-block--field-reference)
5. [`training` block — field reference](#5-training-block--field-reference)
6. [`data` block — field reference](#6-data-block--field-reference)
7. [Cross-field interactions](#7-cross-field-interactions)
8. [Smoke config rationale](#8-smoke-config-rationale)
9. [Override patterns](#9-override-patterns)
10. [Where to go next](#10-where-to-go-next)

---

## 1. How configs are loaded

`training/pretrain.py` reads YAML with PyYAML:

```python
with open(config_path) as f:
    cfg = yaml.safe_load(f)
model_cfg = ModelConfig(**cfg["model"])
train_cfg = cfg["training"]
data_cfg = cfg["data"]
```

- **`model`** keys map to `ModelConfig` dataclass fields in
  `models/transformer.py`. Unknown keys raise `TypeError`.
- **`training`** and **`data`** are plain dicts — optional keys fall back to
  defaults inside `pretrain.py`.

CLI overrides:

| Flag | Effect |
|------|--------|
| `--config PATH` | Required YAML path |
| `--seed N` | Seeds all RNGs before model build |
| `--max-steps N` | Overrides `training.total_steps` |
| `--resume-from N` | Loads checkpoint at step N |

---

## 2. File comparison

| Aspect | `pretrain_a100_502m.yaml` | `pretrain_gpu_smoke.yaml` |
|--------|---------------------------|---------------------------|
| Purpose | Chinchilla 8B-token pretrain | E2E on 4 GB GPU |
| `d_model` | 768 | 128 |
| `n_layers` | 12 | 4 |
| `vocab_size` | 128000 | 4096 |
| `max_seq_len` | 4096 | 64 |
| `eval_max_seq_len` | 131072 | 256 |
| `window_size` | 128 | 32 |
| `total_steps` | 61000 | 5 |
| `compile` | true | false |
| `moe_dispatch` | omitted (stacked) | `"stacked"` explicit |
| `train_data_path` | `data/pretrain_chinchilla` | `data/pretrain_chinchilla` |
| Checkpoints | `checkpoints/pretrain_a100` | `checkpoints/gpu_smoke` |

Smoke config preserves **structural** invariants (alternation, sink, YaRN, MoE
top-k) at miniature scale — not Chinchilla token counts.

---

## 3. Derived quantities

### A100 production (`pretrain_a100_502m.yaml`)

**Tokens per micro-batch step (one forward):**

```
tokens_micro = micro_batch_size × max_seq_len
             = 8 × 4096 = 32,768
```

**Tokens per optimizer step (after gradient accumulation):**

```
tokens_step = tokens_micro × gradient_accumulation_steps
            = 32,768 × 4 = 131,072
```

**Total training tokens:**

```
total_tokens = total_steps × tokens_step
             = 61,000 × 131,072 = 7,995,392,000 ≈ 8.0 × 10⁹
```

**Chinchilla check:** ~502M total params × ~16 tokens/param ≈ 8B tokens ✓

**Parameter counts (from `GPTOSS` methods):**

| Method | Approx. value |
|--------|---------------|
| `num_parameters()` | ~502M |
| `num_active_parameters()` | ~247M |
| Sparsity | ~50.8% |

**Warmup fraction:**

```
3000 / 61000 ≈ 4.9%
```

Industry MoE recipes often use 2–5% warmup; 3000 steps was chosen for router
stability with top-2-of-8 routing.

**Minimum learning rate:**

```
lr_min = lr × min_lr_ratio = 4e-4 × 0.05 = 2e-5
```

**Checkpoint count (full run):**

```
floor(61000 / 2000) = 30 interval saves + 1 final
```

**Wall-time estimate (A100 80GB):**

At ~35–40% MFU and ~131K tokens/step, expect **16–20 hours** for 61K steps
(hardware-dependent).

### Smoke config

```
tokens_step = 2 × 1 × 64 = 128 tokens/step
total_tokens = 5 × 128 = 640 tokens
```

Not Chinchilla-optimal — sufficient to exercise the pipeline.

---

## 4. `model` block — field reference

Master table — **A100** = `pretrain_a100_502m.yaml`, **Smoke** = `pretrain_gpu_smoke.yaml`.
All keys map to `ModelConfig` in `models/transformer.py`; invalid combinations
raise in `__post_init__`.

### Core dimensions and GQA

| Key | A100 | Smoke | Notes |
|-----|------|-------|-------|
| `vocab_size` | `128000` | `4096` | Must match tokenizer ([data_pipeline.md](data_pipeline.md)) |
| `d_model` | `768` | `128` | `n_heads × head_dim` |
| `n_layers` | `12` | `4` | Even count → balanced SWA/global split |
| `n_heads` | `8` | `4` | Query heads |
| `n_kv_heads` | `4` | `2` | KV heads; `n_heads % n_kv_heads == 0` |
| `head_dim` | `96` | `32` | Even int; RoPE pair dimension |

### MoE

| Key | A100 | Smoke | Notes |
|-----|------|-------|-------|
| `ffn_dim` | `1536` | `256` | SwiGLU inner dim per expert |
| `n_routed_experts` | `8` | `4` | Router: `(d_model, n_routed)` |
| `n_activated_experts` | `2` | `2` | Top-k per token |
| `n_shared_experts` | `1` | `1` | Always-on expert(s) |
| `moe_dispatch` | *(default `stacked`)* | `stacked` | `"triton_grouped"` opt-in — [triton_kernels.md](triton_kernels.md) |

### Attention, YaRN, sequence limits

| Key | A100 | Smoke | Notes |
|-----|------|-------|-------|
| `window_size` | `128` | `32` | SWA span on even layers |
| `sink_bias` | `true` | `true` | Per-head sink — [ATTENTION_SINKS.md](ATTENTION_SINKS.md) |
| `rope_theta` | `100000` | `10000` | RoPE base frequency |
| `yarn_scale_factor` | `32` | `4` | `1` = plain RoPE |
| `yarn_original_max_seq_len` | `4096` | `64` | Training RoPE anchor |
| `yarn_target_seq_len` | `131072` | `256` | Extrapolation target (128K) |
| `yarn_beta_fast` | `32` | `4` | YaRN ramp — [yarn.md](yarn.md) |
| `yarn_beta_slow` | `1` | `1` | YaRN ramp slow boundary |
| `yarn_mscale` | `true` | `true` | Magnitude scaling during extrapolation |
| `yarn_prune_rope_global` | `true` | `true` | 25% dim freeze on global layers — [rotary.md](rotary.md) |
| `max_seq_len` | `4096` | `64` | Training window size |
| `eval_max_seq_len` | `131072` | `256` | Inference / passkey cap |

### Dtype and initialization

| Key | A100 | Smoke | Notes |
|-----|------|-------|-------|
| `dtype` | `bf16` | `bf16` | BF16 autocast on CUDA — no GradScaler |
| `weight_tying` | `true` | `true` | Embed ↔ head; saves ~98M at production scale |
| `rms_norm_eps` | `1.0e-5` | `1.0e-5` | RMSNorm ε |
| `init_std` | `0.02` | `0.02` | Linear/embed init; sink bias zero-init separately |

---

## 5. `training` block — field reference

| Key | A100 | Smoke | Notes |
|-----|------|-------|-------|
| `micro_batch_size` | `8` | `2` | Sequences per forward |
| `gradient_accumulation_steps` | `4` | `1` | Micro-batches per optimizer step |
| `total_steps` | `61000` | `5` | Override with `--max-steps` |
| `warmup_steps` | `3000` | `1` | ~4.9% of A100 total — MoE stability |
| `lr` | `4.0e-4` | `1.0e-3` | Peak LR after warmup |
| `min_lr_ratio` | `0.05` | `0.1` | Cosine floor fraction |
| `weight_decay` | `0.1` | `0.0` | AdamW; skipped for bias/norm/embed |
| `beta1` | `0.9` | `0.9` | Adam β₁ |
| `beta2` | `0.95` | `0.95` | Adam β₂; `eps=1e-6` hardcoded in code |
| `grad_clip` | `1.0` | `1.0` | Global norm clip; `0` disables |
| `aux_loss_alpha` | `0.01` | `0.01` | MoE load balance — [moe.md](moe.md) |
| `grad_checkpoint` | `true` | `true` | Calls `enable_gradient_checkpointing()` |
| `grad_checkpoint_every` | `3` | `2` | Checkpoint blocks where `idx % N == 0` |
| `compile` | `true` | `false` | `torch.compile` on CUDA only |
| `compile_mode` | `max-autotune` | — | Ignored when `compile: false` |
| `save_interval` | `2000` | `5` | Checkpoint every N optimizer steps |
| `log_interval` | `50` | `1` | Metrics logging cadence |
| `nan_guard` | `true` | `true` | Skip + rollback on non-finite loss |
| `nan_guard_max_consecutive` | `5` | `5` | NaN steps before reload latest ckpt |
| `save_dir` | `checkpoints/pretrain_a100` | `checkpoints/gpu_smoke` | Safetensors + optim + RNG |

**Implicit defaults** (not in YAML): `num_workers=4`, `pin_memory=true` on CUDA;
chunked CE `chunk_size=8192` in `pretrain.py`. Full loop detail:
[training.md](training.md).

---

## 6. `data` block — field reference

| Key | A100 | Smoke | Notes |
|-----|------|-------|-------|
| `train_data_path` | `data/pretrain_chinchilla` | `data/pretrain_chinchilla` | `shard_*.bin` dir; prepare first |
| `tokenizer` | `llama3` | `smoke` | Universal pipeline name |
| `shard_size_tokens` | `50000000` | `1024` | Tokens per shard at prepare time |
| `max_tokens` | `8000000000` | `8192` | Total corpus budget |
| `data_mix` | `gptoss-default` | `gptoss-default` | Mixture preset |

**`gptoss-default` mixture** (from A100 YAML comments): FineWeb-Edu 50%,
FineWeb 20%, The Stack Python 15%, OpenMath 10%, arXiv 5%. Includes 10%
long-context augmentation (4096 packed sequences). See
[data_pipeline.md](data_pipeline.md).

---

## 7. Cross-field interactions

### `max_seq_len` × batch × Chinchilla

Changing `micro_batch_size` or `gradient_accumulation_steps` changes
`tokens_step` and therefore total tokens for fixed `total_steps`. Re-derive
`total_steps` if you change batch while holding 8B tokens fixed:

```
total_steps = 8_000_000_000 / (micro_batch_size × accum × max_seq_len)
```

### YaRN triple consistency

These must align:

```
yarn_scale_factor ≈ yarn_target_seq_len / yarn_original_max_seq_len
yarn_original_max_seq_len == max_seq_len   (typically)
eval_max_seq_len <= yarn_target_seq_len    (recommended)
```

Mismatch triggers `ModelConfig` validation errors or degenerate ramps (see
[yarn.md](yarn.md)).

### `window_size` vs `max_seq_len`

Windowed layers cache `min(window_size, T)` tokens per layer. Reduction vs pure
GQA approaches **2×** only when `T >> window_size` (e.g. 128K). At `T=4096` with
`W=128`, `scripts/kv_cache_benchmark.py` still reports ~1.9× because six layers
store 128 slots while six store the full 4096.

### `moe_dispatch` × hardware

| Environment | Recommended |
|-------------|-------------|
| A100 + Triton installed | optional `triton_grouped` |
| CPU / Mac / smoke | `stacked` |

### `compile` × `grad_checkpoint`

Both reduce memory pressure differently — compile optimizes kernels; checkpointing
drops activations. Compatible together on A100.

### `sink_bias` × `dtype`

BF16 forward requires sink clamp `[-10, 15]` — always on when `sink_bias: true`.

---

## 8. Smoke config rationale

`pretrain_gpu_smoke.yaml` exists to answer: "Does the full stack run on my GPU
in seconds?"

Design choices:

| Field | Why |
|-------|-----|
| `n_layers: 4` | 2 SWA + 2 global — alternation preserved |
| `max_seq_len: 64` | Fits tiny VRAM |
| `yarn_target: 256` | Tests extrapolation > train len |
| `total_steps: 5` | Quick loss descent sanity |
| `compile: false` | Avoid compile latency in CI |
| `weight_decay: 0` | Simpler overfitting check on noise |
| `moe_dispatch: stacked` | No Triton dependency |

Pair with `scripts/e2e_gpu_smoke.py` for richer checks than 5 training steps.

---

## 9. Override patterns

### Debug 10 steps on A100 config

```bash
python3 training/pretrain.py \
    --config configs/pretrain_a100_502m.yaml \
    --seed 42 \
    --max-steps 10
```

### Enable Triton MoE (A100 only)

Add to YAML under `model:`:

```yaml
moe_dispatch: "triton_grouped"
```

Requires Triton installed. See [triton_kernels.md](triton_kernels.md).

### Smaller effective batch (OOM)

```yaml
training:
  micro_batch_size: 4
  gradient_accumulation_steps: 8   # keep tokens_step = 131072
```

### Resume

```bash
python3 training/pretrain.py \
    --config configs/pretrain_a100_502m.yaml \
    --seed 42 \
    --resume-from 40000
```

Use the **same** config file as the original run.

---

## 10. Where to go next

| Topic | Document |
|-------|----------|
| Onboarding commands | [getting_started.md](getting_started.md) |
| `ModelConfig` validation | [transformer.md](transformer.md) |
| Training loop | [training.md](training.md) |
| Data preparation | [data_pipeline.md](data_pipeline.md) |
| MoE + aux loss | [moe.md](moe.md) |
| YaRN details | [yarn.md](yarn.md) |
| Architecture map | [architecture.md](architecture.md) |

---

<!-- docs:verified 2026-07-31 · fa6f918 -->
