# Foundations — From First Principles to GPT-OSS-Lite

> **Chapter 1 of the GPT-OSS-Lite documentation.** This chapter explains *why* each architectural primitive exists before showing *how* GPT-OSS-Lite implements it. No prior exposure to this repository is required. For the concrete layer stack, parameter budget, and file map, continue to [architecture.md](architecture.md).

---

## Table of contents

1. [What is a decoder-only language model?](#1-what-is-a-decoder-only-language-model)
2. [Self-attention and the causal mask](#2-self-attention-and-the-causal-mask)
3. [Grouped-query attention (GQA)](#3-grouped-query-attention-gqa)
4. [Sliding-window attention and KV-cache memory](#4-sliding-window-attention-and-kv-cache-memory)
5. [Attention sinks — from StreamingLLM to learned bias](#5-attention-sinks--from-streamingllm-to-learned-bias)
6. [Rotary position embeddings (RoPE)](#6-rotary-position-embeddings-rope)
7. [YaRN — length extrapolation beyond training context](#7-yarn--length-extrapolation-beyond-training-context)
8. [Mixture-of-experts feed-forward layers](#8-mixture-of-experts-feed-forward-layers)
9. [Chinchilla scaling for the 502M / 247M budget](#9-chinchilla-scaling-for-the-502m--247m-budget)
10. [BF16 versus FP16 on modern GPUs](#10-bf16-versus-fp16-on-modern-gpus)
11. [What GPT-OSS-Lite uniquely combines](#11-what-gpt-oss-lite-uniquely-combines)
12. [Where to go next](#12-where-to-go-next)

---

## 1. What is a decoder-only language model?

A **language model** assigns a probability distribution over the next token given all prior tokens. Formally, for a token sequence $x_1, x_2, \ldots, x_T$ drawn from a vocabulary $\mathcal{V}$:

$$
P(x_1, x_2, \ldots, x_T) = \prod_{t=1}^{T} P(x_t \mid x_1, \ldots, x_{t-1})
$$

The model never conditions on future tokens. That **autoregressive** constraint is what makes generation possible: sample $x_T$, append it, sample $x_{T+1}$, and repeat.

A **decoder-only** transformer implements each conditional $P(x_t \mid x_{<t})$ with the same stack of layers applied to the prefix $x_{<t}$. There is no encoder, no cross-attention, and no separate source sequence. GPT, LLaMA, and GPT-OSS all follow this pattern.

### Why decoder-only for GPT-OSS-Lite?

OpenAI's GPT-OSS family is decoder-only. GPT-OSS-Lite reproduces that choice faithfully because:

1. **Long-context inference** is dominated by KV-cache size in the attention layers. Decoder-only stacks let us alternate sliding-window and full attention without an encoder bottleneck.
2. **MoE feed-forward** layers replace dense FFNs per block. Routing decisions are local to each token position — natural in a decoder stack.
3. **Weight tying** between the input embedding and output LM head is a well-understood ~98M-parameter savings at `vocab_size=128000`, `d_model=768`.

The training objective is standard next-token cross-entropy. Given logits $\ell_t \in \mathbb{R}^{|\mathcal{V}|}$ at position $t$:

$$
\mathcal{L}_{\text{CE}} = -\frac{1}{T}\sum_{t=1}^{T} \log \frac{e^{\ell_{t, x_t}}}{\sum_{v} e^{\ell_{t, v}}}
$$

GPT-OSS-Lite adds a small auxiliary MoE load-balancing term (Section 8) scaled by $\alpha = 0.01$.

### Pre-norm residual blocks

Each transformer block in GPT-OSS-Lite uses **pre-normalization**:

$$
\begin{aligned}
x' &= x + \mathrm{Attention}(\mathrm{RMSNorm}(x)) \\
x'' &= x' + \mathrm{MoE}(\mathrm{RMSNorm}(x'))
\end{aligned}
$$

Pre-norm stabilizes deep stacks because gradients flow through the residual highway before hitting sublayer parameters. GPT-OSS-Lite uses **RMSNorm** (no mean centering, no bias) with $\epsilon = 10^{-5}$.

---

## 2. Self-attention and the causal mask

### The attention mechanism

Given query, key, and value tensors $Q, K, V \in \mathbb{R}^{B \times H \times T \times D}$ (batch, heads, sequence, head dimension), scaled dot-product attention computes:

$$
\mathrm{Attention}(Q, K, V) = \mathrm{softmax}\left(\frac{QK^\top}{\sqrt{D}} + M\right) V
$$

where $M$ is an attention mask. For language modeling, $M$ must enforce **causality**: position $t$ may attend only to positions $\leq t$.

### Causal mask construction

Define a boolean mask $C \in \{0,1\}^{T \times T}$:

$$
C_{ij} = \begin{cases}
1 & \text{if } j \leq i \\
0 & \text{otherwise}
\end{cases}
$$

In additive mask form, disallowed positions receive $M_{ij} = -\infty$ (or a large negative finite value in practice) so softmax weights become zero.

GPT-OSS-Lite implements this in `models/attention.py` via `F.scaled_dot_product_attention` with `is_causal=True` for full attention, or explicit boolean masks for sliding-window variants.

### Why $\sqrt{D}$ scaling?

Without scaling, dot products grow with head dimension $D$, pushing softmax into saturated regions where gradients vanish. Dividing by $\sqrt{D}$ keeps score magnitudes stable across different head sizes. GPT-OSS-Lite uses `head_dim=96`, so $\sqrt{D} \approx 9.8$.

### Computational complexity

Naive attention is $O(T^2 \cdot H \cdot D)$ in time and $O(T^2)$ for the attention matrix. FlashAttention-style kernels (invoked via PyTorch SDPA on CUDA) reduce memory to $O(T)$ by tiling and never materializing the full $T \times T$ matrix. GPT-OSS-Lite relies on SDPA with FA2 backends when available.

### Manual reference path

`manual_causal_attention` in `models/attention.py` implements the naive $O(T^2)$ path in FP32 accumulation for test oracles. Production forward uses `causal_attention`, which routes to SDPA.

---

## 3. Grouped-query attention (GQA)

### Multi-head attention (MHA)

In MHA, each head has its own $Q$, $K$, and $V$ projections. With $H$ heads:

$$
Q, K, V \in \mathbb{R}^{B \times H \times T \times D}
$$

KV-cache memory per layer per token is $2 \cdot H \cdot D$ elements (K and V).

### Multi-query attention (MQA)

MQA shares a single $K$ and $V$ across all $H$ query heads. Memory drops to $2 \cdot D$ per token per layer, but quality often degrades because keys and values cannot specialize per head.

### GQA — the middle ground

**Grouped-query attention** uses $H$ query heads and $H_{\text{kv}}$ key-value heads, with $H_{\text{kv}} < H$. Each KV head is **repeated** (broadcast) to serve $g = H / H_{\text{kv}}$ query heads:

$$
H_{\text{kv}} = 4,\quad H = 8,\quad g = 2
$$

After projection:

$$
Q \in \mathbb{R}^{B \times 8 \times T \times 96},\quad K, V \in \mathbb{R}^{B \times 4 \times T \times 96}
$$

`repeat_kv` expands $K$ and $V$ to shape $(B, 8, T, 96)$ without `.contiguous()` — SDPA's flash path accepts the expanded layout.

### Why $H_{\text{kv}} < H$?

| Concern | MHA | GQA (8Q/4KV) |
|---------|-----|----------------|
| KV-cache per token | $2 \times 8 \times 96 = 1536$ el | $2 \times 4 \times 96 = 768$ el |
| Expressivity | Full per-head KV | Shared KV within groups |
| Bandwidth at long $T$ | Higher | **2× lower KV traffic** |

At 128K context, KV-cache dominates VRAM. Halving KV head count halves the per-layer KV footprint before any sliding-window trick.

GPT-OSS-Lite pairs GQA with sliding-window layers for a multiplicative reduction (Section 4).

### Parameter accounting for GQA projections

Per attention layer:

| Projection | Shape | Parameters |
|------------|-------|------------|
| `q_proj` | $(d_{\text{model}}, H \cdot D)$ | $768 \times 768 = 589824$ |
| `kv_proj` | $(d_{\text{model}}, 2 H_{\text{kv}} \cdot D)$ | $768 \times 768 = 589824$ |
| `o_proj` | $(H \cdot D, d_{\text{model}})$ | $768 \times 768 = 589824$ |

Total attention linear params per layer: $\approx 1.77 \times 10^6$, plus 8 sink-bias scalars.

---

## 4. Sliding-window attention and KV-cache memory

### The long-context memory problem

During autoregressive **decode**, each new token must attend to all prior keys and values. Storing $K$ and $V$ for every layer and every past position is the **KV-cache**. For one layer, one token, BF16:

$$
\text{bytes}_{\text{KV}} = 2 \times H_{\text{kv}} \times D \times \text{sizeof}(\text{bf16}) = 2 \times 4 \times 96 \times 2 = 1536 \text{ bytes}
$$

For 12 layers and sequence length $T$:

$$
\text{KV}_{\text{full}} = 12 \times T \times 1536 \text{ bytes}
$$

At $T = 131072$ (128K), that is $12 \times 131072 \times 1536 \approx 2.25$ GB in BF16 (batch=1).

### Sliding-window attention (SWA)

**Sliding-window attention** restricts each query position $i$ to keys within a window of width $W$:

$$
\text{allowed}(i, j) \iff (i - j < W) \land (j \leq i)
$$

GPT-OSS-Lite uses $W = 128$ on **even-indexed layers** (0, 2, 4, …). Odd layers use **full** causal attention.

Intuition: local layers capture recent syntactic and lexical patterns; global layers periodically integrate distant dependencies. GPT-OSS alternates the two.

### KV-cache size with alternation

Let $L = 12$ layers, $L_{\text{swa}} = 6$ windowed, $L_{\text{full}} = 6$ global, window $W = 128$.

Per token KV element count (K+V, all layers):

$$
\text{el}_{\text{mixed}} = L_{\text{swa}} \cdot \min(W, T) \cdot 2 H_{\text{kv}} D + L_{\text{full}} \cdot T \cdot 2 H_{\text{kv}} D
$$

For $T \gg W$:

$$
\text{el}_{\text{mixed}} \approx (6 \cdot 128 + 6 \cdot T) \cdot 2 H_{\text{kv}} D = (768 + 6T) \cdot 768
$$

Compared to all-full:

$$
\text{el}_{\text{full}} = 12 T \cdot 2 H_{\text{kv}} D = 12T \cdot 768
$$

**Reduction ratio** at large $T$:

$$
\frac{\text{el}_{\text{full}}}{\text{el}_{\text{mixed}}} \approx \frac{12T}{6T + 768} = \frac{2T}{T + 128}
$$

As $T \to \infty$, the ratio approaches **2.0×**. At $T = 131072$:

$$
\frac{2 \times 131072}{131072 + 128} = \frac{262144}{131200} \approx 2.00
$$

Measured by `scripts/kv_cache_benchmark.py`: **2.00×** (2.25 GB pure GQA vs 1.13 GB mixed, BF16, batch=1). At $T = 4096$ the ratio is **1.94×** because the window cap is not yet negligible relative to $T$.

This ≥1.8× headline metric is architectural — it holds even before training.

### Ring-buffer cache for windowed layers

Windowed layers need not grow the cache with $T$. `MixedKVCache` in `inference/generate.py` stores a **ring buffer** of size $W$ per windowed layer. Decode becomes $O(1)$ per step in cache size instead of $O(T)$.

Global layers use exponential growth caps for prefill efficiency, then append one token per decode step.

---

## 5. Attention sinks — from StreamingLLM to learned bias

### The streaming problem

Naive sliding-window attention **drops** tokens outside the window. StreamingLLM (Xiao et al., 2023) observed that many LLMs spontaneously allocate disproportionate attention mass to **initial tokens** — "attention sinks" — even when those tokens are semantically irrelevant.

If the first tokens fall out of the window, model quality collapses. StreamingLLM's fix: **always retain** the first few tokens in the KV-cache alongside the sliding window.

### Learned sink bias in GPT-OSS

GPT-OSS replaces the hard-coded "keep first $k$ tokens" heuristic with a **learnable per-head scalar** added as an extra attention logit. Conceptually, append a virtual **sink key** with zero value; the sink's logit is not computed from a dot product but from `sink_bias[h]`.

Softmax over augmented keys:

$$
\text{weights} = \mathrm{softmax}\left(\left[\frac{QK^\top}{\sqrt{D}},\; b_{\text{sink}}\right]\right)
$$

Because $V_{\text{sink}} = 0$, the sink column does not contribute to the output — it only **absorbs probability mass**, stabilizing the distribution over real keys.

### Implementation in `models/attention.py`

1. `sink_bias` is `nn.Parameter` shape $(H)$, **initialized to zero**.
2. At forward, bias is **clamped** to $[-10, 15]$ (`SINK_CLAMP_MIN`, `SINK_CLAMP_MAX`).
3. Keys and values are extended with zero columns; the clamped bias populates the mask column for the sink.

**Why clamp?** BF16 SDPA mask-add can overflow if trained bias grows very large. Clamping at forward preserves gradient flow through the uncapped parameter while keeping mask values representable.

**Why per-head?** Different heads specialize (local syntax vs long-range). Sink strength can vary per head.

For the authoritative implementation walkthrough, see [ATTENTION_SINKS.md](ATTENTION_SINKS.md); the constants live in `models/attention.py`.

### Contrast with StreamingLLM

| Approach | Mechanism | Trainable? |
|----------|-----------|------------|
| StreamingLLM | Retain first tokens in KV-cache | No |
| GPT-OSS sink bias | Virtual sink logit per head | **Yes** |
| GPT-OSS-Lite | Same as GPT-OSS | Yes, init 0, clamped |

---

## 6. Rotary position embeddings (RoPE)

### Absolute positions are awkward at scale

Adding a positional vector to embeddings couples position and content in a way that does not extrapolate cleanly to longer sequences than seen in training.

### RoPE intuition

Rotary Position Embedding (Su et al., 2021) encodes position by **rotating** pairs of dimensions in query and key space. For dimension pair $(x_{2i}, x_{2i+1})$ at position $m$:

$$
\begin{pmatrix} x'_{2i} \\ x'_{2i+1} \end{pmatrix} =
\begin{pmatrix} \cos m\theta_i & -\sin m\theta_i \\ \sin m\theta_i & \cos m\theta_i \end{pmatrix}
\begin{pmatrix} x_{2i} \\ x_{2i+1} \end{pmatrix}
$$

Frequencies $\theta_i$ decrease across dimension index $i$, giving high-frequency components for fine local discrimination and low-frequency components for coarse global structure.

Base inverse frequencies:

$$
\text{inv\_freq}_i = \frac{1}{\theta^{2i / D}}
$$

GPT-OSS-Lite uses $\theta = 100000$ (`rope_theta` in `ModelConfig`).

### Relative position property

Dot products $q_m^\top k_n$ depend on **relative** offset $m - n$ after rotation, which is exactly what causal attention needs.

### `apply_rope` in `models/rotary.py`

`apply_rope(x, cos, sin)`:

- Casts `cos`/`sin` to `x.dtype` before multiply (critical for BF16 SDPA — Q/K/V must share dtype).
- Uses the half-dimension `cos`/`sin` from YaRN, repeated across pairs.

### Pruned RoPE on global layers

On **full-attention (odd) layers**, GPT-OSS **prunes** the first $D/4 = 24$ frequency dimensions (of 48 half-dims) by setting their rotation to identity ($\cos=1, \sin=0$). Only the **global** layers prune; windowed layers use full RoPE.

Rationale: at 128K positions, the lowest-frequency components rotate through many cycles, causing **over-rotation** that hurts extrapolation. Neutralizing the slowest modes on layers that actually see the full sequence reduces that pathology.

In code: `_n_pruned_dims() = head_dim // 4` when `not is_windowed` and `yarn_prune_rope_global=True`.

---

## 7. YaRN — length extrapolation beyond training context

### The training vs deployment gap

GPT-OSS-Lite trains at `max_seq_len = 4096` but targets `eval_max_seq_len = 131072` (128K). Plain RoPE extrapolation degrades because frequencies calibrated for 4K behave poorly at 128K.

### YaRN mechanism (Yet another RoPE extension)

YaRN (Peng et al., 2023) **interpolates** inverse frequencies between:

- **Base** RoPE frequencies (good for short context)
- **Scaled** frequencies divided by `scale_factor` (stretched for long context)

A ramp across dimension index selects the blend. GPT-OSS-Lite:

| Parameter | Value |
|-----------|-------|
| `yarn_scale_factor` | 32 |
| `yarn_original_max_seq_len` | 4096 |
| `yarn_target_seq_len` | 131072 |
| `yarn_beta_fast` | 32 |
| `yarn_beta_slow` | 1 |
| `yarn_mscale` | true |

Per dimension $i$:

$$
\text{inv\_freq}'_i = \text{inv\_freq}_i \cdot (1 - r_i) + \frac{\text{inv\_freq}_i}{\text{scale\_factor}} \cdot r_i
$$

where $r_i \in [0,1]$ rises from low to high frequency bands (computed in `compute_yarn_freqs`).

### mscale — attention temperature correction

YaRN also applies an attention scaling factor:

$$
\text{mscale} = 0.1 \cdot \log(\text{scale\_factor}) + 1.0 \quad (\text{when scale} > 1)
$$

For `scale_factor=32`: $\text{mscale} \approx 1.35$. `cos` and `sin` are multiplied by mscale, effectively sharpening attention at long spans.

### Degenerate ramp warning

If `beta_fast`/`beta_slow` misconfigure the ramp (low ≥ high), `compute_yarn_freqs` emits a `UserWarning` and falls back to identity (no extrapolation). Production config is validated to avoid this.

### Train and decode both use YaRN

Unlike some reproductions that apply YaRN only at decode, GPT-OSS-Lite applies YaRN during **training** so the model learns representations consistent with 128K inference. Passkey retrieval at 128K (`inference/long_context.py`) is the canonical eval; target accuracy is **≥85%**.

---

## 8. Mixture-of-experts feed-forward layers

### Why MoE?

Dense FFN parameters scale as $O(d_{\text{model}} \cdot \text{ffn\_dim})$ per layer. MoE replaces one dense FFN with $E$ **experts** and activates only $k$ per token, decoupling **total capacity** from **per-token compute**.

GPT-OSS-Lite per layer:

| Setting | Value |
|---------|-------|
| Routed experts $E$ | 8 |
| Activated routed $k$ | 2 |
| Shared experts | 1 |
| FFN inner dim | 1536 |
| Activation | SwiGLU |

### SwiGLU expert

Each expert implements:

$$
\text{SwiGLU}(x) = W_2\big(\mathrm{silu}(W_1 x) \odot W_3 x\big)
$$

Three matrices per expert: $W_1, W_3 \in \mathbb{R}^{d \times f}$, $W_2 \in \mathbb{R}^{f \times d}$ with $d=768$, $f=1536$.

Expert parameter count:

$$
3 \cdot d \cdot f = 3 \times 768 \times 1536 = 3538944
$$

### Router — top-$k$ gating

Router logits $g(x) \in \mathbb{R}^E$. Probabilities:

$$
p_i = \frac{e^{g_i}}{\sum_j e^{g_j}}
$$

Select top-$k$ experts, renormalize weights:

$$
\tilde{p}_i = \frac{p_i}{\sum_{j \in \text{top-}k} p_j}
$$

Output:

$$
y = \sum_{i \in \text{top-}k} \tilde{p}_i \cdot \text{Expert}_i(x) + \text{Shared}(x)
$$

Softmax for routing uses **FP32** internally (`all_probs_f32`) to avoid BF16 underflow when logits saturate.

### Shared expert

The **shared** SwiGLU expert runs on **every** token regardless of routing. It provides a stable dense pathway so routing can specialize without carrying all baseline FFN duty.

### Auxiliary load-balancing loss (Switch Transformer style)

MoE routers tend toward **collapse** — routing all tokens to one expert. GPT-OSS-Lite uses the **standard** auxiliary loss from Switch Transformer / GShard, **not** DeepSeek's aux-loss-free gate:

$$
\mathcal{L}_{\text{aux}} = E \sum_{i=1}^{E} f_i \cdot P_i
$$

where $f_i$ is the fraction of (top-$k$) activations on expert $i$, and $P_i$ is the mean router probability for expert $i$. Implemented in `aux_load_balancing_loss` in `models/moe.py`.

Total training loss:

$$
\mathcal{L} = \mathcal{L}_{\text{CE}} + \alpha \cdot \mathcal{L}_{\text{aux}},\quad \alpha = 0.01
$$

This $\alpha$ is deliberate portfolio distinction from DeepSeek-v3-Lite.

### Dispatch paths — `moe_dispatch`

| Value | Path | When to use |
|-------|------|-------------|
| `"stacked"` (default) | PyTorch vectorized per-expert loop | CPU, default training |
| `"triton_grouped"` | Fused W1/W3+silu Triton kernel | Opt-in GPU hot path |

Set via `ModelConfig.moe_dispatch` or YAML `model.moe_dispatch`. Triton path **raises** if `triton` is unavailable — no silent fallback during a run configured for Triton.

### Active vs total parameters

**Total** params count all 8 routed experts per layer: $\approx 502$M.

**Active** params count only top-2 routed + 1 shared + router per layer, plus all non-MoE weights: $\approx 247$M.

Formula in `GPTOSS.num_active_parameters()`:

$$
N_{\text{active}} = N_{\text{non-moe}} + L \cdot \big( (k + n_{\text{shared}}) \cdot 3df + d \cdot E \big)
$$

Sparsity: $1 - 247/502 \approx 50.8\%$ of total parameters are inactive per forward pass.

---

## 9. Chinchilla scaling for the 502M / 247M budget

### Chinchilla optimal token count

Chinchilla (Hoffmann et al., 2022) established that compute-optimal training balances model size and token count. Rule of thumb:

$$
N_{\text{tokens}} \approx 20 \times N_{\text{params}}
$$

For $N_{\text{params}} \approx 5.02 \times 10^8$, optimal tokens $\approx 10$B. GPT-OSS-Lite targets **8.0B tokens** — slightly under Chinchilla-optimal but practical for a single-GPU 16–20h budget on A100 80GB.

### Production training schedule

From `configs/pretrain_a100_502m.yaml`:

| Setting | Value |
|---------|-------|
| `micro_batch_size` | 8 |
| `gradient_accumulation_steps` | 4 |
| `max_seq_len` | 4096 |
| Effective batch tokens/step | $8 \times 4 \times 4096 = 131072$ |
| `total_steps` | 61,000 |
| Total tokens | $61000 \times 131072 \approx 8.0 \times 10^9$ |

### Why this fits one A100 80GB

Memory stack (approximate):

- Model weights BF16: ~1.0 GB
- Optimizer FP32 master + moments: ~3× params
- Activations with grad checkpointing every 3rd layer
- Chunked CE (`chunk_size=4096`) avoids full $(B \cdot T \times |\mathcal{V}|)$ logits materialization

Training enables `torch.compile(max-autotune)`, TF32, FA2 via SDPA, fused AdamW, and BF16 autocast. Expected wall time **16–20 hours** at 35–40% MFU.

### Warmup and stability

- Linear warmup 3000 steps (~4.9% of training) — important for top-2-of-8 MoE routing stability
- Cosine decay to `min_lr_ratio=0.05`
- Gradient clip 1.0
- NaN guard with checkpoint rollback (never disable without explicit consent)

---

## 10. BF16 versus FP16 on modern GPUs

### Representable range

| Format | Exponent bits | Approx range |
|--------|---------------|--------------|
| FP32 | 8 | $\approx 10^{\pm38}$ |
| FP16 | 5 | $\approx 10^{\pm5}$ |
| BF16 | 8 | $\approx 10^{\pm38}$ |

BF16 trades mantissa precision for FP32-like range. On Ampere (A100) and Blackwell, BF16 tensor cores are first-class.

### Why GPT-OSS-Lite defaults to BF16

1. **No GradScaler** — FP16 often needs loss scaling; BF16 typically does not.
2. **Router softmax** — already promoted to FP32; activations stay BF16 without FP16 underflow in intermediate matmuls.
3. **Sink bias clamp** — mask values stay in a range BF16 handles when clamped to $[-10, 15]$.
4. **Portfolio consistency** — CoreProjects LLM stack standard on Ampere/Blackwell.

`ModelConfig.dtype = "bf16"` drives autocast in `training/pretrain.py`.

### Mixed precision discipline

- RMSNorm: activations stay native dtype (no silent FP32 copy in forward).
- `apply_rope`: explicit cast of `cos`/`sin` to `x.dtype` before SDPA.
- Manual attention reference: FP32 accumulation for scores only in test path.

---

## 11. What GPT-OSS-Lite uniquely combines

No other project in the CoreProjects LLM portfolio combines all of the following in one decoder-only stack:

| Primitive | GPT-OSS-Lite | DeepSeek-v3-Lite | LLaMA-3-Lite | HyMo | Mamba-3-Lite |
|-----------|--------------|------------------|--------------|------|--------------|
| Attention | GQA + SWA/full alt | MLA | GQA full | GDN + MLA | Complex SSD |
| Long context | YaRN train+decode 128K | YaRN decode | Extended θ | — | Constant state |
| MoE | Top-2/8 + shared | DeepSeekMoE | Dense | Asymmetric MoE | None |
| Sink bias | Learned per-head | None | None | None | None |
| Aux loss | Switch $\alpha=0.01$ | Aux-loss-free | — | — | — |

### The headline metrics (measured, not assumed)

1. **KV-cache reduction ≥1.8× at 128K** — sliding/full alternation + GQA. Measured **1.94–2.0×** (`scripts/kv_cache_benchmark.py`).
2. **Passkey retrieval ≥85% at 128K** — YaRN extrapolation from 4K training (`scripts/passkey_eval.py`, `inference/long_context.py`).

### Architectural invariants (do not break)

1. Even layers SWA ($W=128$), odd layers full — replacing with pure full attention destroys the KV headline.
2. Standard aux load-balancing — not aux-loss-free routing.
3. Learned sink bias with forward clamp — not hard-coded StreamingLLM retention only.
4. YaRN on both train and decode for 128K target.
5. `moe_dispatch` opt-in for Triton — no silent kernel fallback.

---

## 12. Where to go next

| Topic | Document / path |
|-------|-----------------|
| Layer stack, `GPTOSS.forward`, param budget | [architecture.md](architecture.md) |
| Sink bias implementation detail | `documentation/ATTENTION_SINKS.md` |
| Canonical training config | `configs/pretrain_a100_502m.yaml` |
| Attention masks and SDPA | `models/attention.py` |
| YaRN frequency math | `models/yarn.py`, `models/rotary.py` |
| MoE routing and aux loss | `models/moe.py` |
| Triton grouped dispatch | `models/moe_triton.py` |
| Training loop | `training/pretrain.py` |
| Mixed KV-cache generation | `inference/generate.py` |
| Passkey eval | `inference/long_context.py` |
| KV benchmark | `scripts/kv_cache_benchmark.py` |
| Portfolio overview | `README.md` |

Read [architecture.md](architecture.md) next for the system diagram, per-layer table for indices 0–11, and `ModelConfig` field wiring.

---

<!-- docs:verified 2026-07-31 · fa6f918 -->
