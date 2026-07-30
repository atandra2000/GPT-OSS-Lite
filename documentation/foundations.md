# Foundations — Building Blocks of GPT-OSS-Lite

> **Purpose:** Prerequisites before reading component docs. Read this first if GQA, sliding-window attention, attention sinks, or YaRN are unfamiliar.

> **Skip if** you're ready to run smoke tests → [getting_started.md](getting_started.md).

---

## Table of Contents

1. [Causal Language Modeling](#causal-language-modeling)
2. [Pre-Norm Residual Blocks](#pre-norm-residual-blocks)
3. [RMSNorm](#rmsnorm)
4. [Grouped-Query Attention (GQA)](#grouped-query-attention-gqa)
5. [Sliding-Window Attention](#sliding-window-attention)
6. [Attention Sinks](#attention-sinks)
7. [Rotary Position Embeddings (RoPE)](#rotary-position-embeddings-rope)
8. [YaRN Length Extrapolation](#yarn-length-extrapolation)
9. [SwiGLU and MoE](#swiglu-and-moe)
10. [Chinchilla Scaling](#chinchilla-scaling)
11. [BF16 Training](#bf16-training)
12. [KV Caching](#kv-caching)
13. [How the Pieces Map to This Repo](#how-the-pieces-map-to-this-repo)
14. [References](#references)

---

## Causal Language Modeling

A causal LM predicts the next token given all prior tokens:

$$P(x_1, \ldots, x_T) = \prod_{t=1}^{T} P(x_t \mid x_1, \ldots, x_{t-1})$$

Training minimizes cross-entropy on every position. In this repo: $V = 128000$ (LLaMA-3 tokenizer), $d = 768$.

---

## Pre-Norm Residual Blocks

$$\mathbf{x}' = \mathbf{x} + \mathrm{Sublayer}(\mathrm{RMSNorm}(\mathbf{x}))$$

Each `GPTOSSBlock` applies attention then MoE, both pre-norm. Gradients flow through the residual highway directly.

---

## RMSNorm

$$\mathrm{RMSNorm}(\mathbf{x}) = \frac{\mathbf{x}}{\sqrt{\frac{1}{d}\sum_i x_i^2 + \epsilon}} \cdot \gamma$$

No mean subtraction (unlike LayerNorm). Implemented in `models/transformer.py:RMSNorm`.

---

## Grouped-Query Attention (GQA)

Standard MHA: $H$ query heads, $H$ KV heads.
GQA: $H$ query heads, $H_{\text{kv}} < H$ KV heads.

This repo: **8 Q / 4 KV**, head_dim=96. Each KV head is shared by 2 Q heads via `repeat_kv`. Halves KV bandwidth vs MHA.

---

## Sliding-Window Attention

Position $i$ may attend only to keys in $[\max(0, i - w + 1), i]$ (causal + window).

With $w = 128$ and $T \gg 128$, each query row has at most 128 finite attention entries. **KV cache is bounded by $w$** on windowed layers — the architectural headline.

Even layers (0, 2, …) use SWA; odd layers use full attention. See [ATTENTION_SINKS.md](ATTENTION_SINKS.md) §5.

---

## Attention Sinks

Softmax attention weights must sum to 1. When a head wants "no attention," mass collapses onto early tokens (StreamingLLM phenomenon).

**GPT-OSS solution:** per-head learned scalar $s_h$ added to the softmax denominator as a virtual null key:

$$\text{attn}[i,j] = \frac{\exp(\text{score}_{ij})}{\exp(s_h) + \sum_j \exp(\text{score}_{ij})}$$

Full treatment: [ATTENTION_SINKS.md](ATTENTION_SINKS.md).

---

## Rotary Position Embeddings (RoPE)

RoPE encodes relative position by rotating Q/K in the complex plane. Base frequency $\theta = 100000$ in this repo.

Implementation: `models/rotary.py:apply_rope`. See [rotary.md](rotary.md).

---

## YaRN Length Extrapolation

**Problem:** Model trains at $T_{\text{train}} = 4096$ but must work at $T_{\text{eval}} = 131072$.

**YaRN** (Peng et al.): interpolate low-frequency RoPE dimensions, leave high frequencies unchanged, apply mscale correction.

This repo: `scale_factor = 32 = 131072 / 4096`. YaRN is **active during training** (not decode-only). See [yarn.md](yarn.md).

---

## SwiGLU and MoE

**SwiGLU FFN:** $\text{SwiGLU}(x) = \text{silu}(W_1 x) \odot (W_3 x)$, then $W_2$.

**MoE:** Router selects top-2 of 8 experts per token + 1 always-on shared expert. **Standard aux load-balancing loss** (α=0.01) encourages uniform expert usage — distinct from DeepSeek's aux-loss-free bias.

---

## Chinchilla Scaling

Optimal tokens ≈ 20× parameter count. For ~502M params → ~8–10B tokens. This repo targets **8.0B tokens** in 61,000 steps at 131,072 tokens/step.

---

## BF16 Training

BF16 has FP32 exponent range — no `GradScaler` needed. AdamW keeps FP32 master weights. Forward runs in BF16 autocast on CUDA.

---

## KV Caching

At decode, recomputing attention over the full prefix is $O(T^2)$ per step. **KV cache** stores past K/V (or rotated K) so each new token is $O(T)$.

**Mixed cache:** windowed layers store only last 128 tokens (ring buffer); global layers store full history. Net effect: ~2× VRAM savings at 128K vs pure GQA.

---

## How the Pieces Map to This Repo

| Concept | File | Doc |
|---|---|---|
| GQA + SWA + sink | `models/attention.py` | [attention.md](attention.md) |
| YaRN RoPE | `models/yarn.py` | [yarn.md](yarn.md) |
| MoE + aux loss | `models/moe.py` | [moe.md](moe.md) |
| Block stack | `models/transformer.py` | [transformer.md](transformer.md) |
| Mixed KV cache | `inference/generate.py` | [inference.md](inference.md) |

---

## References

- StreamingLLM — Xiao et al., arXiv:2309.17453
- YaRN — Peng et al., arXiv:2309.00071
- Longformer — Beltagy et al., arXiv:2004.05150
- GQA — Ainslie et al., arXiv:2305.13245
- Chinchilla — Hoffmann et al., arXiv:2203.15556

<!-- docs:verified 2026-07-31 · fd4fe36 -->
