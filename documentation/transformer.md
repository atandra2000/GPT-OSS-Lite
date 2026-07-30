# Transformer — Top-Level Wiring

> **Prerequisites:** [foundations.md](foundations.md), [attention.md](attention.md), [moe.md](moe.md).

> **Covers:** `GPTOSS`, `GPTOSSBlock`, `ModelConfig`, `RMSNorm` in `models/transformer.py`.

---

## Table of Contents

1. [Abstract](#abstract)
2. [ModelConfig](#modelconfig)
3. [GPTOSSBlock](#gptossblock)
4. [GPTOSS](#gptoss)
5. [RMSNorm](#rmsnorm)
6. [Weight Tying](#weight-tying)
7. [Gradient Checkpointing](#gradient-checkpointing)
8. [Parameter Counting](#parameter-counting)
9. [Load-Bearing Invariants](#load-bearing-invariants)
10. [References](#references)

---

## Abstract

`GPTOSS` is the root `nn.Module`. It stacks 12 `GPTOSSBlock` layers — each with pre-norm attention (alternating SWA/full) and pre-norm MoE FFN. Forward returns `(logits, aux_loss)`.

---

## ModelConfig

Dataclass mirror of `model.*` YAML fields. `__post_init__` validates:

- `n_heads * head_dim == d_model`
- `n_heads % n_kv_heads == 0`
- YaRN: `scale_factor >= 1`, `original < target` when scaling
- MoE: `0 < n_activated <= n_routed`

27 fields total — `test_modelconfig_field_count_is_stable` guards against accidental additions.

---

## GPTOSSBlock

```python
class GPTOSSBlock(nn.Module):
    def forward(self, x, positions):
        x = x + self.attn(self.norm1(x), positions)
        moe_out, aux_loss = self.moe(self.norm2(x))
        x = x + moe_out
        return x, aux_loss
```

`layer_idx` passed to `GPTOSSAttention` determines SWA vs full: `is_windowed = (layer_idx % 2 == 0)`.

---

## GPTOSS

```python
class GPTOSS(nn.Module):
    def forward(self, idx, positions=None):
        # embed → blocks (accumulate aux_loss) → norm → head → logits
```

**Positions:** default `torch.arange(T)`. YaRN uses absolute positions for frequency lookup.

**Gradient checkpointing:** when `model.gradient_checkpointing = True`, every `grad_checkpoint_every` layers uses `torch.utils.checkpoint.checkpoint`.

---

## RMSNorm

Vectorized implementation — no FP32 copy of activations. Computes RMS in FP32 for stability, applies scale in native dtype:

```python
rms = x.detach().float().pow(2).mean(dim=-1, keepdim=True).add(self.eps).rsqrt()
return (x * (rms * self.weight.to(rms.dtype)).to(x.dtype))
```

See OPTIMIZATIONS.md §4.

---

## Weight Tying

When `weight_tying=True`:

```python
self.head.weight = self.embed.weight
```

`num_parameters()` and `num_active_parameters()` deduplicate tied weights via `id(p)` tracking. Verified by `test_active_params_correct_with_tied_weights`.

---

## Gradient Checkpointing

Enabled in `pretrain.py` via `enable_gradient_checkpointing(model, every=3)`. Recomputes activations during backward for every 3rd block — trades compute for VRAM.

---

## Parameter Counting

| Method | What it counts |
|---|---|
| `num_parameters()` | All unique params (tie-deduped) |
| `num_active_parameters()` | Embedding + attention + (top-k expert weights × 2 + shared) + norms + router |

Anchor ranges: 500M–504M total, 244M–250M active (`test_validation.py`).

---

## Load-Bearing Invariants

| Invariant | Why |
|---|---|
| `n_layers` even | SWA/full alternation requires pairs |
| Weight tying dedup in param count | README anchor metrics depend on it |
| `sink_bias` init zeros | `exp(0)=1` → neutral at start |
| `moe_dispatch` default `stacked` | No silent Triton (AGENTS rule 8) |

---

## References

- [architecture.md](architecture.md) — system diagram
- [transformer.py](../models/transformer.py) — source
- [configs.md](configs.md) — YAML keys

<!-- docs:verified 2026-07-31 · fd4fe36 -->
