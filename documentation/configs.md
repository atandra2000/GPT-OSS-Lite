# Configuration Reference

> **Purpose:** Reference for every YAML key in `configs/`, with theory for *why* each hyperparameter exists and *where* code consumes it.

> **Read this if** you're tuning hyperparameters. **Skip if** learning architecture → [architecture.md](architecture.md).

---

## Table of Contents

1. [Overview](#overview)
2. [pretrain_a100_502m.yaml — Canonical Recipe](#pretrain_a100_502myaml--canonical-recipe)
3. [Model Keys](#model-keys)
4. [Training Keys](#training-keys)
5. [Data Keys](#data-keys)
6. [Triton Dispatch](#triton-dispatch)
7. [Config Nesting](#config-nesting)

---

## Overview

Configs are YAML with three top-level sections:

```yaml
model:      # Architecture — ModelConfig → GPTOSS
training:   # Loop hyperparameters — pretrain.py
data:       # Paths — PretrainDataset
```

`training/pretrain.py` loads the YAML and constructs `ModelConfig` from `model.*`.

---

## pretrain_a100_502m.yaml — Canonical Recipe

**Target:** 1× A100 80GB, ~502M params, 8.0B tokens, 16–20 h wall.

| Section | Highlights |
|---|---|
| model | 12 layers, vocab 128000, d=768, 8 routed experts (top-2), YaRN 128K |
| training | 61000 steps, warmup 3000, aux_loss_alpha 0.01, compile, nan_guard |
| data | `data/pretrain_chinchilla`, LLaMA-3 tokenizer |

---

## Model Keys

| Key | Default (502M) | Consumed by | Notes |
|---|---|---|---|
| `vocab_size` | 128000 | `GPTOSS.embed`, `head` | LLaMA-3 BPE |
| `d_model` | 768 | All layers | Must equal `n_heads * head_dim` |
| `n_layers` | 12 | `GPTOSS.blocks` | Must be even for SWA/full alternation |
| `n_heads` | 8 | `GPTOSSAttention` | Query heads |
| `n_kv_heads` | 4 | `GPTOSSAttention` | GQA; must divide `n_heads` |
| `head_dim` | 96 | Attention | Must be even (RoPE pairs) |
| `ffn_dim` | 1536 | `MoELayer` experts | Per-expert SwiGLU width |
| `n_routed_experts` | 8 | `MoELayer` | |
| `n_activated_experts` | 2 | Router top-k | |
| `n_shared_experts` | 1 | Always-on expert | |
| `window_size` | 128 | Even-layer SWA | Headline KV metric depends on this |
| `sink_bias` | true | Per-head null logit | See ATTENTION_SINKS.md |
| `rope_theta` | 100000 | YaRN base freq | |
| `yarn_scale_factor` | 32 | 128K / 4K | Set to 1 for plain RoPE |
| `yarn_original_max_seq_len` | 4096 | Training context | |
| `yarn_target_seq_len` | 131072 | Eval extrapolation | |
| `yarn_beta_fast` | 32 | YaRN ramp bounds | |
| `yarn_beta_slow` | 1 | YaRN ramp bounds | |
| `yarn_mscale` | true | Attention temperature | Required for long context |
| `yarn_prune_rope_global` | true | Odd-layer RoPE | 25% dim pruning |
| `max_seq_len` | 4096 | Training | |
| `eval_max_seq_len` | 131072 | Inference eval | |
| `dtype` | bf16 | Autocast | |
| `weight_tying` | true | embed ↔ head | Saves ~98M params |
| `rms_norm_eps` | 1.0e-5 | RMSNorm | |
| `init_std` | 0.02 | Weight init | |
| `moe_dispatch` | stacked | MoE path | `triton_grouped` for fused kernel |

Validation: `ModelConfig.__post_init__` in `models/transformer.py`.

---

## Training Keys

| Key | Default | Notes |
|---|---|---|
| `micro_batch_size` | 8 | Tokens per micro-batch |
| `gradient_accumulation_steps` | 4 | Effective batch multiplier |
| `total_steps` | 61000 | ~8.0B tokens at 131K tok/step |
| `warmup_steps` | 3000 | 4.9% of total; MoE stability |
| `lr` | 4.0e-4 | Peak LR after warmup |
| `min_lr_ratio` | 0.05 | Cosine floor |
| `weight_decay` | 0.1 | AdamW |
| `beta1`, `beta2` | 0.9, 0.95 | AdamW |
| `grad_clip` | 1.0 | Global norm clip |
| `grad_checkpoint` | true | Activation checkpointing |
| `grad_checkpoint_every` | 3 | Every 3rd layer |
| `compile` | true | `torch.compile(max-autotune)` on CUDA |
| `compile_mode` | max-autotune | |
| `save_interval` | 2000 | Checkpoint frequency |
| `log_interval` | 50 | Console/WandB log |
| `nan_guard` | true | Rollback on NaN loss |
| `nan_guard_max_consecutive` | 5 | |
| `aux_loss_alpha` | 0.01 | Switch Transformer standard |
| `save_dir` | checkpoints/pretrain_a100 | |

---

## Data Keys

| Key | Default | Notes |
|---|---|---|
| `train_data_path` | data/pretrain_chinchilla | Sharded uint32 bins |
| `tokenizer` | llama3 | 128K vocab |
| `shard_size_tokens` | 50000000 | Per shard |
| `max_tokens` | 8000000000 | Chinchilla budget |
| `data_mix` | gptoss-default | See data_pipeline.md |

---

## Triton Dispatch

Set `model.moe_dispatch: triton_grouped` to enable the fused W1/W3+silu kernel. Default `stacked` uses pure PyTorch. Requires CUDA + Triton; raises `ImportError` on CPU if Triton unavailable.

See [triton_kernels.md](triton_kernels.md).

---

## Config Nesting

Tests may pass flat dicts; YAML uses nested `model:` / `training:` / `data:`. `pretrain.py` unwraps via `config.get("model", config)`.

<!-- docs:verified 2026-07-31 · fd4fe36 -->
