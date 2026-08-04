# GPT-OSS-Lite — Config and API Reference

> Quick reference for the canonical configuration and the key public API
> signatures. The full YAML encyclopedia with defaults, validation rules, and
> cross-field interactions lives in [training.md](../training.md) (Part B —
> Configuration reference); the class-by-class module map is in
> [foundations-and-architecture.md](../concepts/foundations-and-architecture.md).

## Model config (`ModelConfig`)

`models/transformer.py:ModelConfig` mirrors the YAML `model:` section. Canonical values from `configs/pretrain_a100_502m.yaml`:

| Field | A100 | Smoke | Notes |
|-------|------|-------|-------|
| `vocab_size` | 128000 | 4096 | Must match tokenizer ([training.md](../training.md)) |
| `d_model` | 768 | 128 | `n_heads × head_dim` |
| `n_layers` | 12 | 4 | Even count → balanced SWA/global split |
| `n_heads` | 8 | 4 | Query heads |
| `n_kv_heads` | 4 | 2 | KV heads; `n_heads % n_kv_heads == 0` |
| `head_dim` | 96 | 32 | Even int; RoPE pair dimension |
| `ffn_dim` | 1536 | 256 | SwiGLU inner dim per expert |
| `n_routed_experts` | 8 | 4 | Router: `(d_model, n_routed)` |
| `n_activated_experts` | 2 | 2 | Top-k per token |
| `n_shared_experts` | 1 | 1 | Always-on expert(s) |
| `moe_dispatch` | *(default `stacked`)* | `stacked` | `"triton_grouped"` opt-in — [moe.md](../concepts/moe.md) |
| `window_size` | 128 | 32 | SWA span on even layers |
| `sink_bias` | true | true | Per-head sink — [attention-sinks.md](../concepts/attention-sinks.md) |
| `rope_theta` | 100000 | 10000 | RoPE base frequency |
| `yarn_scale_factor` | 32 | 4 | `1` = plain RoPE |
| `yarn_original_max_seq_len` | 4096 | 64 | Training RoPE anchor |
| `yarn_target_seq_len` | 131072 | 256 | Extrapolation target (128K) |
| `yarn_beta_fast` | 32 | 4 | YaRN ramp — [attention-and-positional.md](../concepts/attention-and-positional.md) |
| `yarn_beta_slow` | 1 | 1 | YaRN ramp slow boundary |
| `yarn_mscale` | true | true | Magnitude scaling during extrapolation |
| `yarn_prune_rope_global` | true | true | 25% dim freeze on global layers |
| `max_seq_len` | 4096 | 64 | Training window size |
| `eval_max_seq_len` | 131072 | 256 | Inference / passkey cap |
| `dtype` | bf16 | bf16 | BF16 autocast on CUDA — no GradScaler |
| `weight_tying` | true | true | Embed ↔ head; saves ~98M at production scale |
| `rms_norm_eps` | 1.0e-5 | 1.0e-5 | RMSNorm ε |
| `init_std` | 0.02 | 0.02 | Linear/embed init; sink bias zero-init separately |

**Validation highlights** (`ModelConfig.__post_init__`):

- `n_heads % n_kv_heads == 0` (GQA)
- `n_heads * head_dim == d_model`
- `yarn_scale_factor >= 1`; if `> 1`, require `original < target`
- Warns if `eval_max_seq_len < max_seq_len`

**YAML → Python:**

```python
with open(config_path) as f:
    cfg = yaml.safe_load(f)
model_cfg = ModelConfig(**cfg["model"])
model = GPTOSS(model_cfg)
```

Training hyperparameters (`aux_loss_alpha`, `compile`, etc.) live under `training:` — not in `ModelConfig`.

---

## Training config

| Key | A100 | Smoke | Notes |
|-----|------|-------|-------|
| `micro_batch_size` | 8 | 2 | Sequences per forward |
| `gradient_accumulation_steps` | 4 | 1 | Micro-batches per optimizer step |
| `total_steps` | 61000 | 5 | Override with `--max-steps` |
| `warmup_steps` | 3000 | 1 | ~4.9% of A100 total — MoE stability |
| `lr` | 4.0e-4 | 1.0e-3 | Peak LR after warmup |
| `min_lr_ratio` | 0.05 | 0.1 | Cosine floor fraction |
| `weight_decay` | 0.1 | 0.0 | AdamW; skipped for bias/norm/embed |
| `beta1` | 0.9 | 0.9 | Adam β₁ |
| `beta2` | 0.95 | 0.95 | Adam β₂; `eps=1e-6` hardcoded in code |
| `grad_clip` | 1.0 | 1.0 | Global norm clip; `0` disables |
| `aux_loss_alpha` | 0.01 | 0.01 | MoE load balance — [moe.md](../concepts/moe.md) |
| `grad_checkpoint` | true | true | Calls `enable_gradient_checkpointing()` |
| `grad_checkpoint_every` | 3 | 2 | Checkpoint blocks where `idx % N == 0` |
| `compile` | true | false | `torch.compile` on CUDA only |
| `compile_mode` | max-autotune | — | Ignored when `compile: false` |
| `save_interval` | 2000 | 5 | Checkpoint every N optimizer steps |
| `log_interval` | 50 | 1 | Metrics logging cadence |
| `nan_guard` | true | true | Skip + rollback on non-finite loss |
| `nan_guard_max_consecutive` | 5 | 5 | NaN steps before reload latest ckpt |
| `save_dir` | checkpoints/pretrain_a100 | checkpoints/gpu_smoke | Safetensors + optim + RNG |
| `train_data_path` | data/pretrain_chinchilla | data/pretrain_smoke | `shard_*.bin` dir; prepare first |

**Implicit defaults** (not in YAML): `num_workers=4`, `pin_memory=true` on CUDA;
chunked CE `chunk_size=8192` in `training/pretrain.py`.

**`gptoss-default` mixture** (from A100 YAML comments): FineWeb-Edu 50%,
FineWeb 20%, The Stack Python 15%, OpenMath 10%, arXiv 5%. Includes 10%
long-context augmentation (4096 packed sequences). See
[training.md](../training.md) (Data Pipeline part).

---

## Key API signatures

### `GPTOSS` (models/transformer.py)

- `models/transformer.py:GPTOSS.forward(x, positions)` → `(logits, aux_loss)` — `logits` is `(B, T, vocab_size)` in model dtype (BF16 under autocast); `aux_loss` is a scalar mean MoE load-balancing loss.
- `models/transformer.py:GPTOSS.num_parameters()` → total param count (tracks seen `id()`s so the tied head is counted once; production: **501,836,640**).
- `models/transformer.py:GPTOSS.num_active_parameters()` → active count for top-2 routing (**247,032,672**, 50.8% sparsity).
- `models/transformer.py:GPTOSS.enable_gradient_checkpointing(every=3)` → wraps `GPTOSSBlock.forward` with `torch.utils.checkpoint.checkpoint` every Nth block.
- `models/transformer.py:ModelConfig` — dataclass with `__post_init__` validation (GQA divisibility, YaRN triple consistency).
- `models/transformer.py:GPTOSSBlock` — attention + MoE residual block.
- `models/transformer.py:RMSNorm` — pre-norm; FP32 RMS stats, native dtype output.

### Attention (models/attention.py)

- `models/attention.py:GPTOSSAttention` — projections, YaRN, alternation logic; clamps `sink_bias` to `[-10, 15]` at forward.
- `models/attention.py:SINK_CLAMP_MIN` / `models/attention.py:SINK_CLAMP_MAX` — `-10.0` / `15.0`.
- `models/attention.py:causal_attention` — SDPA path with optional window + sink.
- `models/attention.py:manual_causal_attention` — FP32 test oracle.
- `models/attention.py:repeat_kv` — GQA broadcast via `expand + reshape` (no `.contiguous()`).

### MoE (models/moe.py, models/moe_triton.py)

- `models/moe.py:MoELayer` — dispatch + shared expert.
- `models/moe.py:MoERouter.forward` — top-k gating with FP32 softmax; returns raw logits + renormalized weights.
- `models/moe.py:aux_load_balancing_loss` — Switch-style aux loss (FP32 internally).
- `models/moe.py:SwiGLUExpert` — W1/W2/W3.
- `models/moe_triton.py:triton_moe_w1w3_silu` — fused W1/W3+silu grouped-GEMM (opt-in `moe_dispatch="triton_grouped"`); W2 stays PyTorch.
- `models/moe_triton.py:HAS_TRITON` — import guard; no silent fallback.

### Rotary / YaRN (models/rotary.py, models/yarn.py)

- `models/rotary.py:apply_rope` — dtype-safe rotation.
- `models/rotary.py:compute_yarn_freqs` — ramp-blended inverse frequencies.
- `models/rotary.py:compute_yarn_mscale` — attention temperature correction.
- `models/yarn.py:YaRNRoPE` — buffer `inv_freq`, forward cos/sin + prune.

### Training (training/pretrain.py)

- `training/pretrain.py:main` — CLI entry; reads YAML, seeds RNGs, builds model/optimizer/scheduler, runs the loop with NaN guard + rollback.
- `training/pretrain.py:PretrainDataset` — mmap shard dataset; windowed reads with EOS-aware document boundaries.
- `training/pretrain.py:seed_everything` — seeds torch/cuda/numpy/random; sets `CUBLAS_WORKSPACE_CONFIG=:4096:8`.
- `training/pretrain.py:_set_hardware_perf_knobs` — TF32, cuDNN benchmark, `set_float32_matmul_precision("high")`.

### Inference (inference/generate.py, inference/long_context.py)

- `inference/generate.py:MixedKVCache` — per-layer cache: ring buffer (windowed layers) + exponential-growth array (global layers, 1.5× growth, 4M-token cap).
- `inference/generate.py:MixedKVCache.reset` / `inference/generate.py:MixedKVCache.append` / `inference/generate.py:MixedKVCache.get` / `inference/generate.py:MixedKVCache.seq_len` — lifecycle + chronologically ordered reads.
- `inference/generate.py:generate` — token-by-token decode; `use_cache=False` replays the prefix (O(T²) reference path).
- `inference/long_context.py:PasskeyEvaluator` — 128K passkey retrieval eval (greedy, `temperature=0.0`).

### Utilities (utils/)

- `utils/checkpoint.py:CheckpointManager` — atomic safetensors saves (`.tmp` → rename) + separate optim/sched/RNG files.
- `utils/memory.py:estimate_model_memory_gb` — VRAM estimator for training and eval.
- `utils/memory.py:assert_fits_in_available_gpu` — OOM pre-flight check.
- `utils/logging.py:TrainingLogger` — WandB-capable training logger.

---

## References

- [training.md](../training.md) — full YAML encyclopedia, derived quantities, cross-field interactions
- [foundations-and-architecture.md](../concepts/foundations-and-architecture.md) — module map, parameter accounting
- [moe.md](../concepts/moe.md) — MoE config semantics, Triton opt-in
- [attention-and-positional.md](../concepts/attention-and-positional.md) — YaRN parameter semantics
- [operations.md](../guides/operations.md) — CLI reference for scripts and utils

<!-- docs:verified 2026-08-05 · 6491066 -->
