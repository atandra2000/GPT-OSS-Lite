# GPT-OSS-Lite — Foundations and Architecture

## Part A — Foundations (From First Principles)

> **Chapter 1 of the GPT-OSS-Lite documentation.** This chapter explains *why* each architectural primitive exists before showing *how* GPT-OSS-Lite implements it. No prior exposure to this repository is required. For the concrete layer stack, parameter budget, and file map, continue to [foundations-and-architecture.md](foundations-and-architecture.md).

---

---

### 1. What is a decoder-only language model?

A **language model** assigns a probability distribution over the next token given all prior tokens. Formally, for a token sequence $x_1, x_2, \ldots, x_T$ drawn from a vocabulary $\mathcal{V}$:

$$
P(x_1, x_2, \ldots, x_T) = \prod_{t=1}^{T} P(x_t \mid x_1, \ldots, x_{t-1})
$$

The model never conditions on future tokens. That **autoregressive** constraint is what makes generation possible: sample $x_T$, append it, sample $x_{T+1}$, and repeat.

A **decoder-only** transformer implements each conditional $P(x_t \mid x_{<t})$ with the same stack of layers applied to the prefix $x_{<t}$. There is no encoder, no cross-attention, and no separate source sequence. GPT, LLaMA, and GPT-OSS all follow this pattern.

### Why decoder-only for GPT-OSS-Lite?

OpenAI's GPT-OSS family is decoder-only. GPT-OSS-Lite reproduces that choice
because long-context **decode** is dominated by KV-cache bytes in attention, not
by FFN matmuls. With 12 layers, GQA 4 KV heads, `head_dim=96`, and BF16, a pure
full-attention cache at `T=131072` already costs ~2.25 GB before batching —
the architectural win is shrinking that footprint via six windowed layers at
`W=128`, not adding an encoder that would cache a second sequence.

MoE feed-forward replaces dense FFNs per block; routing is per token position,
which fits a decoder stack. Weight tying between embedding and LM head saves
~98M parameters at `vocab_size=128000`, `d_model=768` — meaningful on a 502M
budget where every million params trades against Chinchilla token count.

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

### 2. Self-attention and the causal mask

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

`models/attention.py:manual_causal_attention` implements the naive $O(T^2)$ path in FP32 accumulation for test oracles. Production forward uses `models/attention.py:causal_attention`, which routes to SDPA.

Softmax, the $\sqrt{D}$ temperature, mask-add versus mask-fill, and the fused SDPA backends are derived in [attention math](attention-and-positional.md).

---

### 3. Grouped-query attention (GQA)

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

`models/attention.py:repeat_kv` expands $K$ and $V$ to shape $(B, 8, T, 96)$ without `.contiguous()` — SDPA's flash path accepts the expanded layout.

### Why $H_{\text{kv}} < H$?

| Concern | MHA | GQA (8Q/4KV) |
|---------|-----|----------------|
| KV-cache per token | $2 \times 8 \times 96 = 1536$ el | $2 \times 4 \times 96 = 768$ el |
| Expressivity | Full per-head KV | Shared KV within groups |
| Bandwidth at long $T$ | Higher | **2× lower KV traffic** |

At 128K context, KV-cache dominates VRAM. Halving KV head count halves the per-layer KV footprint before any sliding-window trick.

GPT-OSS-Lite pairs GQA with sliding-window layers for a multiplicative reduction (Section 4).

### KV-parameter savings over MHA

GQA's second win is in the projection matrices. MHA with $H = 8$ heads needs a $d_{\text{model}} \times 2HD$ key-value projection; GQA needs only $d_{\text{model}} \times 2H_{\text{kv}}D$:

$$
N_{\text{kv,MHA}} = 2 H D\, d_{\text{model}} = 2 \times 8 \times 96 \times 768 = 1179648, \qquad
N_{\text{kv,GQA}} = 2 H_{\text{kv}} D\, d_{\text{model}} = 2 \times 4 \times 96 \times 768 = 589824
\tag{1}
$$

The per-layer saving is $2(H - H_{\text{kv}}) D\, d_{\text{model}} = 589824$ parameters; over 12 layers that is $7077888 \approx 7.1$M — about 1.4% of the 501.8M total budget — with no loss of query-side expressivity, because $Q$ still gets 8 independent heads. Only the $K/V$ side is shared.

### KV-cache bytes per head

The cache holds one copy of $K$ and one of $V$ per KV head. One head, one token, one layer costs $2D$ elements; in BF16 ($s = 2$ bytes/element):

$$
b_{\text{head}} = 2 D s = 2 \times 96 \times 2 = 384 \text{ bytes/token/layer}
\tag{2}
$$

MHA multiplies this by $H = 8$ → 3072 B/token/layer; GQA by $H_{\text{kv}} = 4$ → 1536 B/token/layer. The 2× factor is exactly the head-count ratio $H / H_{\text{kv}}$, and it applies to **every** layer and every cached token, so it multiplies the Section 4 sliding-window savings rather than adding to them.

### The cost of `repeat_kv` at matmul time

GQA does not buy FLOPs. After `models/attention.py:repeat_kv` broadcasts $K, V$ from $(B, 4, T, 96)$ to $(B, 8, T, 96)$, the score matmul costs exactly what MHA costs:

$$
C_{QK^\top} = 2 B H T^2 D
\tag{3}
$$

What GQA changes is *where bytes live and how often they are read*. `repeat_kv` first `expand`s to a stride-0 view (free), then `reshape`s — the merged head dimension cannot be a view over the stride-0 layout, so the reshape materializes one contiguous $(B, 8, T, 96)$ copy per layer per forward. That is a memory op, not a matmul, and it is dwarfed by the score matmul:

$$
\frac{\text{copy bytes}}{\text{QK}^\top \text{ FLOPs}} = \frac{2 H_{\text{kv}} T D s}{2 B H T^2 D} = \frac{H_{\text{kv}} s}{B H T} = \frac{4 \times 2}{1 \times 8 \times 4096} \approx 2.4 \times 10^{-4}
\tag{4}
$$

at prefill $T = 4096$, batch 1 ($\approx 7.6 \times 10^{-6}$ at 128K). The real savings show up at decode, where DRAM traffic per step is proportional to cached KV bytes — the bandwidth side of the 2× head-count reduction is derived in [kv cache engineering](../inference.md) §4.4.

### Parameter accounting for GQA projections

Per attention layer:

| Projection | Shape | Parameters |
|------------|-------|------------|
| `q_proj` | $(d_{\text{model}}, H \cdot D)$ | $768 \times 768 = 589824$ |
| `kv_proj` | $(d_{\text{model}}, 2 H_{\text{kv}} \cdot D)$ | $768 \times 768 = 589824$ |
| `o_proj` | $(H \cdot D, d_{\text{model}})$ | $768 \times 768 = 589824$ |

Total attention linear params per layer: $\approx 1.77 \times 10^6$, plus 8 sink-bias scalars.

---

### 4. Sliding-window attention and KV-cache memory

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

### Compute: windowed vs full attention

The mask also cuts prefill FLOPs, because attention cost is proportional to the number of *allowed* key positions, not $T^2$. Full causal attention allows the lower triangle, $N_{\text{full}} = T(T+1)/2$ pairs. A window of width $W$ allows row $i$ only $\min(i+1, W)$ keys:

$$
N_{\text{sw}}(T, W) = \sum_{i=0}^{T-1} \min(i+1, W) = \frac{W(W+1)}{2} + (T - W)\, W \;\approx\; W\, T \quad (T \gg W)
\tag{5}
$$

The first $W$ rows fill the triangle $\sum_{i=0}^{W-1}(i+1) = W(W+1)/2$; every later row sees exactly $W$ keys. Each allowed pair costs one $2D$-FLOP query-key dot product and one $2D$-FLOP value accumulation, so attention FLOPs per layer are

$$
C_{\text{attn}} = 4 H D \cdot N_{\text{allowed}} = 4 \times 8 \times 96 \cdot N_{\text{allowed}} = 3072\, N_{\text{allowed}}
\tag{6}
$$

and the full-to-windowed ratio (softmax and the output projection excluded — identical for both) is

$$
\frac{C_{\text{full}}}{C_{\text{sw}}} = \frac{T(T+1)}{W(W+1) + 2(T-W)W} \approx \frac{T}{2W} = \begin{cases} 16 & T = 4096 \\ 512 & T = 131072 \end{cases}
\tag{7}
$$

| $T$ | full FLOPs / layer | windowed FLOPs / layer | ratio |
|-----|--------------------|------------------------|-------|
| 4096 | $3072 \times 8.39 \times 10^6 \approx 25.8$ GFLOP | $3072 \times 5.16 \times 10^5 \approx 1.6$ GFLOP | 16.3× |
| 131072 | $3072 \times 8.59 \times 10^9 \approx 26.4$ TFLOP | $3072 \times 1.68 \times 10^7 \approx 51.5$ GFLOP | 512.3× |

Decode flips the bottleneck from FLOPs to bytes: each step reads the layer's cached $K, V$ — $b_{\text{head}} \cdot H_{\text{kv}} = 1536$ bytes per token (2). A full layer reads $1536 \cdot T$ bytes; a windowed layer reads only its ring buffer, $1536 \cdot W$:

$$
\frac{R_{\text{full}}}{R_{\text{sw}}} = \frac{T}{W} = \begin{cases} 32 & T = 4096 \\ 1024 & T = 131072 \end{cases}
\tag{8}
$$

At 128K one windowed decode step touches 192 KiB of KV instead of 192 MiB. The pair-count and FLOP arithmetic is derived in full in [attention math](attention-and-positional.md) §4.5; the cache-side accounting (ring buffers, exponential growth, the measured 2.00× mixed ratio) is in [kv cache engineering](../inference.md) §4.5–4.7.

### The window mask binds at prefill

Both mask paths are additive $0.0$ / $-\infty$ masks, never boolean casts. At prefill ($T_q = T_k$), `models/attention.py:_window_mask` returns the square mask $i - j < W$ ANDed with causality, and `models/attention.py:causal_attention` converts it with `torch.where(mask, 0.0, float("-inf"))` — blocked positions contribute exactly $-\infty$, so no softmax mass can cross the window boundary. The sink path uses the same construction on its causal slice. A 1.0/0.0 boolean cast would add nothing to the logits and leak future tokens; the fixed behavior is pinned by regression tests.

### Ring-buffer cache for windowed layers

Windowed layers need not grow the cache with $T$. `MixedKVCache` in `inference/generate.py` stores a **ring buffer** of size $W$ per windowed layer. Decode becomes $O(1)$ per step in cache size instead of $O(T)$.

Global layers use exponential growth caps for prefill efficiency, then append one token per decode step.

---

### 5. Attention sinks — from StreamingLLM to learned bias

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
3. Keys and values are extended with zero columns; the clamped bias populates the mask column for the sink. Blocked positions get a true additive $-\infty$ — `mask[:, :, :T_k] = torch.where(causal, 0.0, float("-inf"))` in `models/attention.py:causal_attention` — never a 1.0/0.0 boolean cast, which would add nothing to the logits and leave future tokens unmasked.

**Why clamp?** BF16 SDPA mask-add can overflow if trained bias grows very large. Clamping at forward preserves gradient flow through the uncapped parameter while keeping mask values representable.

**Why per-head?** Different heads specialize (local syntax vs long-range). Sink strength can vary per head.

For the authoritative implementation walkthrough, see [attention-sinks.md](attention-sinks.md); the constants live in `models/attention.py`.

### Contrast with StreamingLLM

| Approach | Mechanism | Trainable? |
|----------|-----------|------------|
| StreamingLLM | Retain first tokens in KV-cache | No |
| GPT-OSS sink bias | Virtual sink logit per head | **Yes** |
| GPT-OSS-Lite | Same as GPT-OSS | Yes, init 0, clamped |

---

### 6. Rotary position embeddings (RoPE)

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

On **full-attention (odd) layers**, GPT-OSS **prunes** the $D/4 = 24$ **fastest-rotating** pairs — the highest-frequency half-dims, $i = 0, \ldots, 23$ — by setting their rotation to identity ($\cos=1, \sin=0$). Only the **global** layers prune; windowed layers use full RoPE.

Rationale: at 128K positions, the **fastest**-rotating components (the first
$D/4 = 24$ pairs, `inv_freq[0] = 1.0` rad/token → ~20,861 full turns at position
131072) wrap many times, and a rotation of $\phi$ is indistinguishable from
$\phi + 2\pi k$ — **over-rotation** that aliases and hurts extrapolation.
Neutralizing the fastest modes on layers that actually see the full sequence
reduces that pathology. Only odd-indexed global layers prune (`head_dim=96` → 24
of 48 half-dims); even windowed layers keep full RoPE because they never attend
beyond `W=128` tokens.

In code: `_n_pruned_dims() = head_dim // 4` when `not is_windowed` and
`yarn_prune_rope_global=True`.

The rotation geometry, the over-rotation argument, and the full YaRN ramp are derived in [positional encodings](attention-and-positional.md) §4.5–4.8.

---

### 7. YaRN — length extrapolation beyond training context

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

Unlike reproductions that apply YaRN only at decode, GPT-OSS-Lite trains with
YaRN at `max_seq_len=4096` so gradients see the same frequency blend the model
will use at `eval_max_seq_len=131072`. With `yarn_scale_factor=32`, mscale
≈1.35 sharpens attention at long spans — if training stayed on plain RoPE, the
model would optimize for 4K geometry then face a different temperature at 128K
decode. Passkey retrieval (`inference/long_context.py`) is the canonical eval;
target **≥85%** after the 8.0B-token schedule.

---

### 8. Mixture-of-experts feed-forward layers

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

Set via `ModelConfig.moe_dispatch` or YAML `model.moe_dispatch`. Triton path
**raises** if `triton` is unavailable — no silent fallback during a run configured
for Triton.

**Why two paths?** Default `"stacked"` keeps every training run on the pure-PyTorch
oracle (`tests/test_moe.py`); Triton fuses W1/W3+silu for throughput on GPU but
W2 stays in PyTorch and backward uses the reference path. Opt-in via YAML avoids
silent kernel fallback when Triton fails to compile — a misconfigured
`triton_grouped` run must error loudly, not drift onto stacked mid-epoch and
invalidate throughput comparisons.

### Active vs total parameters

**Total** params count all 8 routed experts per layer: $\approx 502$M.

**Active** params count only top-2 routed + 1 shared + router per layer, plus all non-MoE weights: $\approx 247$M.

Formula in `GPTOSS.num_active_parameters()`:

$$
N_{\text{active}} = N_{\text{non-moe}} + L \cdot \big( (k + n_{\text{shared}}) \cdot 3df + d \cdot E \big)
$$

Sparsity: $1 - 247/502 \approx 50.8\%$ of total parameters are inactive per forward pass.

Routing as categorical selection, the Switch/GShard aux loss, the $\alpha = 0.01$ scale, and expert collapse are derived from scratch in [moe theory](moe.md) §5–7.

---

### 9. Chinchilla scaling for the 502M / 247M budget

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

AdamW's update rule, bias correction, warmup, and global-norm clipping are derived in [optimizers](optimizers-and-numerics.md); the checkpoint-every-3rd-layer memory/compute tradeoff is analyzed in [autograd checkpointing](kernels-and-checkpointing.md).

---

### 10. BF16 versus FP16 on modern GPUs

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

The IEEE-754 bit budget, TF32, and the FP32 accumulation islands inside the model are derived in [numerics](optimizers-and-numerics.md).

---

### 11. What GPT-OSS-Lite uniquely combines

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

### 12. Where to go next

| Topic | Document |
|-------|----------|
| Layer stack, `GPTOSS`, `ModelConfig`, param budget | [foundations-and-architecture.md](foundations-and-architecture.md) |
| Sink bias + SWA/full implementation | [attention-sinks.md](attention-sinks.md) |
| RoPE geometry, YaRN ramp, pruned global layers | [attention-and-positional.md](attention-and-positional.md) |
| Top-2 routing, aux α=0.01, Triton opt-in | [moe.md](moe.md) |
| Training loop, YAML encyclopedia | [training.md](../training.md) |
| `MixedKVCache`, passkey eval | [inference.md](../inference.md) |
| Scripts, checkpoints, OPT catalog | [operations.md](../guides/operations.md) |
| Canonical YAML | `configs/pretrain_a100_502m.yaml` |

Read [attention-sinks.md](attention-sinks.md) next — Part A theory plus Part B
`models/attention.py` walkthrough before changing masks, sink clamp, or the
even/odd `W=128` alternation.

---

## Part B — Architecture (System Design)

### Purpose

This chapter is the single system map for GPT-OSS-Lite: the 502M-total /
247M-active decoder, its module boundaries, config wiring, and the transformer
stack in `models/transformer.py`. Read [foundations-and-architecture.md](foundations-and-architecture.md) first
for the math behind each primitive; Part B below is the implementation guide
for `models/transformer.py:ModelConfig`, `models/transformer.py:RMSNorm`, `models/transformer.py:GPTOSSBlock`, and `models/transformer.py:GPTOSS`.

### Mental model

`models/transformer.py` is the **composition root** — it wires attention, MoE,
and norms into a 12-layer pre-norm decoder. `ModelConfig` (YAML → dataclass) is
the single source of truth for shapes and invariants. Even layers run
sliding-window attention (cheap KV); odd layers run full attention (global
context). Training returns `(logits, aux_loss)`; inference reuses block
submodules with `MixedKVCache` (see §9).

> **Chapter 2 of the GPT-OSS-Lite documentation.**

---

**Part A — System design**

1. [System overview](#1-system-overview)
2. [Layer stack and residual dataflow](#2-layer-stack-and-residual-dataflow)
3. [`GPTOSS.forward` dataflow](#3-gptossforward-dataflow)
4. [Alternating attention pattern (layers 0–11)](#4-alternating-attention-pattern-layers-0-11)
5. [Parameter accounting](#5-parameter-accounting)
6. [File map and module responsibilities](#6-file-map-and-module-responsibilities)
7. [`ModelConfig` — config to code wiring](#7-modelconfig--config-to-code-wiring)
8. [MoE dispatch and Triton opt-in](#8-moe-dispatch-and-triton-opt-in)
9. [Inference: `MixedKVCache` and generation](#9-inference-mixedkvcache-and-generation)
10. [Training pipeline integration](#10-training-pipeline-integration)
11. [Invariants and failure modes](#11-invariants-and-failure-modes)
12. [Comparison with sibling portfolio models](#12-comparison-with-sibling-portfolio-models)

**Part B — Transformer stack (`models/transformer.py`)**

- [B.1 Module overview](#b1-module-overview)
- [B.2 `ModelConfig` fields and `__post_init__` validation](#b2-modelconfig-fields-and-__post_init__-validation)
- [B.3 `moe_dispatch` values (`stacked` \| `triton_grouped`)](#b3-moe_dispatch-values-stacked--triton_grouped)
- [B.4 `RMSNorm`](#b4-rmsnorm)
- [B.5 `GPTOSSBlock` construction and forward](#b5-gptossblock-construction-and-forward)
- [B.6 `GPTOSS` construction and submodule roles](#b6-gptoss-construction-and-submodule-roles)
- [B.7 Weight initialization policy](#b7-weight-initialization-policy)
- [B.8 Forward pass, `positions`, return contract `(logits, aux_loss)`](#b8-forward-pass-positions-return-contract-logits-aux_loss)
- [B.9 Gradient checkpointing schedule (`grad_ckpt_every`)](#b9-gradient-checkpointing-schedule-grad_ckpt_every)
- [B.10 `num_parameters` / `num_active_parameters` + 502M breakdown](#b10-num_parameters--num_active_parameters--502m-breakdown)
- [B.11 Weight tying](#b11-weight-tying)
- [B.12 Config validation edge cases](#b12-config-validation-edge-cases)
- [B.13 How to verify](#b13-how-to-verify)

13. [Where to go next](#13-where-to-go-next)

---

### 1. System overview

GPT-OSS-Lite is a **12-layer decoder-only transformer** with:

- **GQA** attention (8 query heads, 4 KV heads, `head_dim=96`)
- **Alternating** sliding-window ($W=128$) and full attention
- **Learned per-head sink bias** (init 0, clamped $[-10, 15]$ at forward)
- **YaRN RoPE** ($\theta=100000$, scale 32, train 4096 → target 131072)
- **Pruned RoPE** on global (odd) layers — 25% of dims (`head_dim // 4`)
- **MoE SwiGLU** FFN: top-2 of 8 routed + 1 shared expert (`ffn_dim=1536`)
- **Standard aux load-balancing** loss ($\alpha = 0.01$)
- **Pre-norm RMSNorm**, **weight-tied** embed ↔ LM head
- **Vocab** 128000 (LLaMA-3 BPE tokenizer in data pipeline)

### ASCII system diagram

```
                    ┌─────────────────────────────────────────┐
                    │  input_ids  (B, T)  int64               │
                    └──────────────────┬──────────────────────┘
                                       │
                    ┌──────────────────▼──────────────────────┐
                    │  nn.Embedding(vocab=128000, d=768)        │
                    │  weight tied ↔ lm_head.weight             │
                    └──────────────────┬──────────────────────┘
                                       │  x  (B, T, 768)
           ┌───────────────────────────┼───────────────────────────┐
           │         repeat 12× GPTOSSBlock                       │
           │  ┌─────────────────────────────────────────────────┐ │
           │  │  RMSNorm → GPTOSSAttention → residual           │ │
           │  │  RMSNorm → MoELayer          → residual         │ │
           │  │           (+ aux_loss per layer)                │ │
           │  └─────────────────────────────────────────────────┘ │
           └───────────────────────────┬───────────────────────────┘
                                       │
                    ┌──────────────────▼──────────────────────┐
                    │  RMSNorm (final)                          │
                    └──────────────────┬──────────────────────┘
                                       │
                    ┌──────────────────▼──────────────────────┐
                    │  lm_head: Linear(768 → 128000, bias=False)│
                    └──────────────────┬──────────────────────┘
                                       │
                    ┌──────────────────▼──────────────────────┐
                    │  logits (B, T, 128000)                    │
                    │  aux_loss scalar (mean over layers)       │
                    └───────────────────────────────────────────┘
```

### Scale summary

| Metric | Value |
|--------|-------|
| Total parameters | ~502M (`501836640` counted) |
| Active parameters / token | ~247M (`247032672` counted) |
| Sparsity | ~50.8% inactive per forward |
| Training tokens | 8.0B Chinchilla-scale |
| Training wall time (target) | 16–20 h on 1× A100 80GB |
| Train sequence length | 4096 |
| Eval / deploy sequence length | 131072 (128K) |
| KV reduction at 128K | ≥1.8× target; measured **2.00×** |
| Passkey at 128K | ≥85% target accuracy |

---

### 2. Layer stack and residual dataflow

### `GPTOSSBlock`

Each block (`models/transformer.py`) contains:

```
x_in
  │
  ├─ norm1 ──► GPTOSSAttention ──► (+) ──► x_mid
  │                                      │
  ├─ norm2 ──► MoELayer ──► (+) ──► x_out
  │              │
  │              └── aux_loss (scalar)
```

**Pre-norm** means normalization precedes each sublayer. Residual connections are unscaled (no DeepSeek-style depth scaling).

### `GPTOSSAttention` internals

```
x (B,T,768)
  │
  ├─ q_proj  → Q  (B, 8, T, 96)
  ├─ kv_proj → K,V (B, 4, T, 96) each
  │
  ├─ YaRNRoPE(positions, n_pruned_dims) → cos, sin
  ├─ apply_rope(Q, cos, sin)
  ├─ apply_rope(K, cos, sin)
  ├─ repeat_kv(K,V) → (B, 8, T, 96)
  │
  ├─ causal_attention(
  │     window = 128 if layer_idx even else None,
  │     sink_bias = clamp(sink_bias, -10, 15)
  │  )
  │
  └─ o_proj → (B, T, 768)
```

Layer index parity sets `is_windowed = (layer_idx % 2 == 0)` in `GPTOSSAttention.__init__`.

### `MoELayer` internals

```
x (B,T,768) → flat (N, 768)
  │
  ├─ MoERouter → top-2 indices + weights + all_logits
  ├─ dispatch (stacked or triton_grouped)
  ├─ aux_load_balancing_loss(all_logits)
  ├─ + shared SwiGLU expert(s)
  │
  └─ view (B, T, 768)
```

---

### 3. `GPTOSS.forward` dataflow

Source: `models/transformer.py`, class `GPTOSS`.

### Inputs

| Argument | Shape | Default |
|----------|-------|---------|
| `idx` | $(B, T)$ int token ids | required |
| `positions` | $(T)$ or broadcastable position ids | `arange(T)` |

Positions matter for YaRN at eval lengths $T > \text{max\_seq\_len}$ during training — pass explicit position tensors during 128K inference.

### Forward steps

1. **Embed:** `x = embed(idx)` → $(B, T, 768)$.
2. **Blocks:** For each `GPTOSSBlock` in `blocks`:
   - Optional gradient checkpoint when `gradient_checkpointing` and `layer_idx % grad_ckpt_every == 0`.
   - `x, aux = block(x, positions)` — accumulates per-layer aux losses.
3. **Head:** `x = norm(x)`; `logits = head(x)` → $(B, T, 128000)$.
4. **Aux:** `aux_loss = mean(stack(aux_losses))` — scalar.

### Returns

```python
(logits, aux_loss)
# logits: (B, T, vocab_size)
# aux_loss: scalar — multiply by aux_loss_alpha in training loop
```

### Weight tying

When `cfg.weight_tying=True`:

```python
self.head.weight = self.embed.weight
```

`num_parameters()` deduplicates by parameter `id()` so embed/head are counted once. Savings: `vocab_size × d_model = 128000 × 768 = 98304000` parameters (~98M).

### Initialization

- Linear / Embedding: normal, `std=0.02`
- RMSNorm weight: ones
- `sink_bias`: **zeros** (per head, per windowed/full layer — all attention layers when `sink_bias=True`)

---

### 4. Alternating attention pattern (layers 0–11)

Even indices → **SWA** (`window_size=128`). Odd indices → **full** causal attention.

| Layer | Index | Attention | RoPE prune | KV cache growth (decode) |
|-------|-------|-----------|------------|--------------------------|
| 0 | even | SWA $W=128$ | no | Ring buffer, cap 128 |
| 1 | odd | Full | yes ($D/4=24$ dims) | Linear with $T$ |
| 2 | even | SWA | no | Ring buffer |
| 3 | odd | Full | yes | Linear with $T$ |
| 4 | even | SWA | no | Ring buffer |
| 5 | odd | Full | yes | Linear with $T$ |
| 6 | even | SWA | no | Ring buffer |
| 7 | odd | Full | yes | Linear with $T$ |
| 8 | even | SWA | no | Ring buffer |
| 9 | odd | Full | yes | Linear with $T$ |
| 10 | even | SWA | no | Ring buffer |
| 11 | odd | Full | yes | Linear with $T$ |

**Counts:** 6 SWA + 6 full = 12 layers.

### Why alternation instead of grouping?

Grouped patterns (e.g. 6 SWA then 6 full) create longer paths where no layer sees the full sequence. Alternation ensures every two layers, one global layer can integrate distant context while the next SWA layer refines local structure with a compact cache.

### `extra_repr` debugging

Each `GPTOSSAttention` reports mode in `extra_repr`:

```
layer=0 (SWA), H=8/4, D=96, window=128
layer=1 (Full, pruned=24), H=8/4, D=96, window=128
```

---

### 5. Parameter accounting

### Total parameters (`num_parameters`)

Production model count: **501,836,640** (~502M). Breakdown by component:

#### Embedding + tied head

| Component | Formula | Count |
|-----------|---------|-------|
| `embed` / `head` (tied) | $V \cdot d$ | $128000 \times 768 = 98304000$ |

Without tying, total would be $\approx 600$M — tying saves ~98M.

#### Per `GPTOSSBlock` (×12)

**Attention**

| Parameter | Count |
|-----------|-------|
| `q_proj` | $768^2 = 589824$ |
| `kv_proj` | $768^2 = 589824$ |
| `o_proj` | $768^2 = 589824$ |
| `sink_bias` | $8$ |
| Subtotal | $\approx 1769480$ |

YaRN buffers are non-persistent (`persistent=False`) — not counted in params.

**MoE**

| Parameter | Count |
|-----------|-------|
| Per SwiGLU expert | $3 \cdot 768 \cdot 1536 = 3538944$ |
| 8 routed experts | $28311552$ |
| 1 shared expert | $3538944$ |
| Router `gate` | $768 \times 8 = 6144$ |
| Subtotal | $31854640$ |

**Norms**

| Parameter | Count |
|-----------|-------|
| `norm1` + `norm2` | $2 \times 768 = 1536$ |

**Per block total** $\approx 33.6$M → ×12 $\approx 403$M MoE-heavy.

#### Final norm

768 parameters.

#### Sanity check

$$
98.3\text{M (embed)} + 12 \times 33.6\text{M} + 768 \approx 502\text{M}
$$

#### Tied-embedding accounting

The embedding table and the LM head are one tensor when `weight_tying=True`:
`models/transformer.py:GPTOSS.__init__` assigns `self.head.weight = self.embed.weight`, so
the head is a *view* of the embedding — one allocation, one gradient, one optimizer
state. A naive count would charge the vocabulary matrix twice. Its size is

$$
V \cdot d_{\text{model}} = 128000 \times 768 = 98304000. \tag{1}
$$

Because the tied head contributes zero *new* parameters, the untied counterfactual adds
one full matrix on top of the tied total:

$$
N_{\text{untied}} = 501836640 + 98304000 = 600140640, \tag{2}
$$

so tying saves 98.3M parameters — 16.4% of the untied total. The active count inherits
the same exclusion: `models/transformer.py:GPTOSS.num_active_parameters` walks
`named_parameters()` and deduplicates by `id()`, so the head never appears in the 247M
active figure. A plain sum over `nn.Module.parameters()` would not deduplicate — the
shared tensor is registered under both `embed.weight` and `head.weight` — which is why
`models/transformer.py:GPTOSS.num_parameters` tracks seen `id()`s (B.10).

### Active parameters (`num_active_parameters`)

**247,032,672** (~247M). MoE experts not routed on a given token are **inactive**.

Formula from `GPTOSS.num_active_parameters()`:

```python
non_moe = all parameters except names containing "experts" or "router"
expert_params = 3 * d_model * ffn_dim
moe_active_per_layer = (n_activated + n_shared) * expert_params + d_model * n_routed_experts
return non_moe + (moe_active_per_layer) * n_layers
```

Per layer MoE active:

$$
(2 + 1) \times 3 \times 768 \times 1536 + 768 \times 8 = 10622976
$$

Inactive routed experts per layer: $6 \times 3 \times 768 \times 1536 = 21233664$ not executed.

Expanding the formula with the production config ($d = 768$, $f = 1536$,
$k_{\text{act}} = 2$, $k_{\text{shared}} = 1$, $n_{\text{exp}} = 8$, $L = 12$):

$$
N_{\text{active}} = N_{\text{non-moe}} + L \left[ (k_{\text{act}} + k_{\text{shared}}) \cdot 3 d f + d \cdot n_{\text{exp}} \right], \tag{3}
$$

with $N_{\text{non-moe}} = 119556960$ (embed, attention, norms; router gates kept
out of the sweep) and the per-layer MoE term $10622976$ from above:

$$
N_{\text{active}} = 119556960 + 12 \times 10622976 = 247032672. \tag{4}
$$

**247,032,672** is the active figure: 49.2% of the 501,836,640 total is exercised per
token, i.e. 50.8% idle ($1 - 247032672 / 501836640 \approx 0.508$). The
router gate is deliberately excluded from `non_moe` — the sweep skips names containing
`"router"` as well as `"experts"` — and re-added exactly once via the
$d \cdot n_{\text{exp}}$ term in (3), so `models/transformer.py:GPTOSS.num_active_parameters`
and the derivation agree.

### KV-cache memory formula (BF16, batch=1)

Per layer per token (K+V):

$$
\text{bytes} = 2 \times H_{\text{kv}} \times D \times 2 = 2 \times 4 \times 96 \times 2 = 1536
$$

Mixed cache at sequence $T$:

$$
\text{KV}_{\text{mixed}} = (6 \cdot \min(128, T) + 6 \cdot T) \times 1536 \text{ bytes}
$$

At $T = 131072$: **1.13 GB** vs **2.25 GB** all-full → **2.00×**.

---

### 6. File map and module responsibilities

```
GPT-OSS-Lite/
├── models/
│   ├── transformer.py    # ModelConfig, RMSNorm, GPTOSSBlock, GPTOSS
│   ├── attention.py      # SWA/full, sink bias, causal_attention, GQA
│   ├── moe.py            # MoELayer, router, aux loss, stacked dispatch
│   ├── moe_triton.py     # Opt-in Triton W1/W3+silu grouped GEMM
│   ├── yarn.py           # YaRNRoPE module
│   └── rotary.py         # apply_rope, compute_yarn_freqs, mscale
├── training/
│   └── pretrain.py       # Main training loop, chunked CE, NaN guard
├── inference/
│   ├── generate.py       # MixedKVCache, autoregressive generate()
│   └── long_context.py   # PasskeyEvaluator for 128K eval
├── configs/
│   └── pretrain_a100_502m.yaml   # Canonical production config
└── scripts/
    ├── kv_cache_benchmark.py     # Analytical KV headline metric
    └── passkey_eval.py             # Passkey retrieval CLI
```

### `models/transformer.py`

| Symbol | Role |
|--------|------|
| `ModelConfig` | Dataclass + validation invariants |
| `RMSNorm` | Pre-norm; FP32 RMS stats, native dtype output |
| `GPTOSSBlock` | Attention + MoE residuals |
| `GPTOSS` | Top-level module, checkpointing, param counters |

### `models/attention.py`

| Symbol | Role |
|--------|------|
| `SINK_CLAMP_MIN/MAX` | $-10.0$, $15.0$ |
| `causal_attention` | SDPA path with optional window + sink |
| `manual_causal_attention` | Test oracle (FP32 scores) |
| `repeat_kv` | GQA broadcast without contiguous() |
| `GPTOSSAttention` | Projections, YaRN, alternation logic |

### `models/moe.py`

| Symbol | Role |
|--------|------|
| `SwiGLUExpert` | $W_1, W_2, W_3$ |
| `MoERouter` | Top-$k$ gating, FP32 softmax |
| `aux_load_balancing_loss` | Switch-style aux |
| `MoELayer` | Dispatch + shared expert |

### `models/moe_triton.py`

| Symbol | Role |
|--------|------|
| `triton_moe_w1w3_silu` | Fused gate+up+silu; W2 stays PyTorch |
| `HAS_TRITON` | Import guard |

### `models/yarn.py` + `models/rotary.py`

| Symbol | Role |
|--------|------|
| `YaRNRoPE` | Buffer `inv_freq`, forward cos/sin + prune |
| `compute_yarn_freqs` | Ramp-blended inverse frequencies |
| `compute_yarn_mscale` | Attention temperature correction |
| `apply_rope` | Dtype-safe rotation |

### `training/pretrain.py`

| Concern | Implementation |
|---------|----------------|
| Loss | Chunked CE + `aux_loss_alpha * aux_loss` |
| Optimizer | AdamW fused, FP32 master weights |
| Scheduler | Warmup + cosine |
| Compile | `torch.compile(max-autotune)` when CUDA |
| Stability | NaN guard, grad clip 1.0 |
| Reproducibility | `seed_everything`, `CUBLAS_WORKSPACE_CONFIG` |

### `inference/generate.py`

| Symbol | Role |
|--------|------|
| `MixedKVCache` | Ring (SWA) + growing (global) per layer |
| `generate()` | Token-by-token with cache |

### `inference/long_context.py`

| Symbol | Role |
|--------|------|
| `PasskeyEvaluator` | Build prompts, run generate, score accuracy |

---

### 7. `ModelConfig` — config to code wiring

`ModelConfig` in `models/transformer.py` mirrors YAML `model:` section. Canonical values from `configs/pretrain_a100_502m.yaml`:

| Field | Value | Consumed by |
|-------|-------|-------------|
| `vocab_size` | 128000 | `GPTOSS.embed`, `head` |
| `d_model` | 768 | All layers |
| `n_layers` | 12 | `GPTOSS.blocks` |
| `n_heads` | 8 | `GPTOSSAttention` |
| `n_kv_heads` | 4 | `GPTOSSAttention`, KV cache |
| `head_dim` | 96 | Projections, RoPE |
| `ffn_dim` | 1536 | MoE experts |
| `n_routed_experts` | 8 | `MoELayer` |
| `n_activated_experts` | 2 | Router top-$k$ |
| `n_shared_experts` | 1 | Shared SwiGLU |
| `window_size` | 128 | SWA layers |
| `sink_bias` | true | Per-head `sink_bias` param |
| `rope_theta` | 100000 | YaRN base |
| `yarn_scale_factor` | 32 | YaRN stretch |
| `yarn_original_max_seq_len` | 4096 | Train context |
| `yarn_target_seq_len` | 131072 | Extrapolation target |
| `yarn_beta_fast` | 32 | Ramp band |
| `yarn_beta_slow` | 1 | Ramp band |
| `yarn_mscale` | true | mscale enable |
| `yarn_prune_rope_global` | true | Prune on odd layers |
| `max_seq_len` | 4096 | Training windows |
| `eval_max_seq_len` | 131072 | Long-context eval |
| `dtype` | bf16 | Autocast in pretrain |
| `weight_tying` | true | Embed ↔ head |
| `rms_norm_eps` | 1e-5 | RMSNorm |
| `init_std` | 0.02 | Weight init |
| `moe_dispatch` | `"stacked"` | MoE path (opt-in `"triton_grouped"`) |

### Validation highlights (`__post_init__`)

- `n_heads % n_kv_heads == 0` (GQA)
- `n_heads * head_dim == d_model`
- `yarn_scale_factor >= 1`; if `> 1`, require `original < target`
- Warns if `eval_max_seq_len < max_seq_len`

### YAML → Python

```python
with open(config_path) as f:
    cfg = yaml.safe_load(f)
model_cfg = ModelConfig(**cfg["model"])
model = GPTOSS(model_cfg)
```

Training hyperparameters (`aux_loss_alpha`, `compile`, etc.) live under `training:` — not in `ModelConfig`.

---

### 8. MoE dispatch and Triton opt-in

### Default: `moe_dispatch = "stacked"`

`MoELayer._dispatch_vectorized`:

1. Flatten token-expert assignments.
2. `argsort` experts (`stable=True` for reproducibility).
3. Per-expert chunk loop with `index_add` weighted accumulation.

Pure PyTorch — runs on CPU and GPU.

### Opt-in: `moe_dispatch = "triton_grouped"`

`MoELayer._dispatch_triton`:

1. Same sort/grouping as stacked path.
2. `triton_moe_w1w3_silu` fuses W1, W3, silu, element-wise multiply.
3. W2 matmul per expert in PyTorch on sorted chunks.

**Contract:** If Triton unavailable and config requests `triton_grouped`, import raises with explicit message — **no silent fallback**.

Set in YAML:

```yaml
model:
  moe_dispatch: "triton_grouped"
```

Default in `pretrain_a100_502m.yaml` is `"stacked"`.

---

### 9. Inference: `MixedKVCache` and generation

### Cache types per layer

| Layer type | Storage | Max tokens stored |
|------------|---------|-------------------|
| SWA (even) | Ring buffer $(B, H, W, D)$ | $W = 128$ |
| Full (odd) | Growing tensor | $T$ (cap configurable) |

Cache stores **rotated** keys (post-RoPE) to avoid recomputing rotations during decode.

### Decode complexity

- SWA layer append: $O(1)$ cache size per step (ring update).
- Full layer append: $O(1)$ append amortized with growth strategy.
- Attention compute: SWA layers attend to $\leq 128$ keys; full layers attend to full history.

### Sink bias at inference

`inference/generate.py` imports `SINK_CLAMP_MIN/MAX` and applies the same clamp as training when building attention masks for cached decode.

### Long-context eval

`PasskeyEvaluator` (`inference/long_context.py`):

- Default context lengths: 4096, 8192, 32768, 65536, 131072
- Inserts 5-digit passkey in filler text
- Target: **≥85%** accuracy at 128K on trained checkpoint

---

### 10. Training pipeline integration

### Loss composition

```python
logits, aux_loss = model(input_ids, positions)
ce_loss = chunked_cross_entropy(logits, targets, chunk_size=4096)
loss = ce_loss + aux_loss_alpha * aux_loss  # alpha = 0.01
```

### Effective batch

$$
B_{\text{eff}} = \text{micro\_batch} \times \text{grad\_accum} = 8 \times 4 = 32 \text{ sequences}
$$

Tokens per optimizer step: $32 \times 4096 = 131072$.

### Gradient checkpointing

`enable_gradient_checkpointing(every=3)` checkpoints every third block — trades ~30% extra compute for materially lower activation memory.

### Hardware knobs (`_set_hardware_perf_knobs`)

- TF32 for matmul and cuDNN
- `cudnn.benchmark = True`, `benchmark_limit = 0`
- `preferred_blas_library = "cublaslt"`
- `set_float32_matmul_precision("high")`

### Checkpointing

Atomic saves via `utils/checkpoint.py`: write `.tmp`, rename. Includes optimizer, scheduler, RNG state in `rng_step_N.pt` when enabled.

---

### 11. Invariants and failure modes

### Must preserve (architectural contract)

| Invariant | Violation impact |
|-----------|------------------|
| Even=SWA, odd=full | KV reduction collapses toward 1× |
| `window_size=128` on SWA layers | Changes headline KV math |
| Standard aux loss, $\alpha=0.01$ | MoE collapse or portfolio mismatch |
| Sink bias clamp at forward | BF16 mask overflow risk |
| YaRN train+decode for 128K | Passkey metric fails |
| `moe_dispatch` explicit opt-in | Silent perf path switch forbidden |
| Weight tying default on | Param budget drifts +~98M |
| GQA 8/4 | KV bandwidth doubles if reverted to MHA |

### Common failure modes

| Symptom | Likely cause | Check |
|---------|--------------|-------|
| MoE routes to one expert | Aux loss disabled or $\alpha$ too low | `aux_loss_alpha`, aux loss logs |
| NaN loss | Router saturation, bad data shard | NaN guard rollback, FP32 softmax path |
| OOM at 4096 train | Checkpointing off, compile overhead | `grad_checkpoint`, micro batch |
| 128K gibberish | Positions not passed, YaRN misconfig | `eval_max_seq_len`, position ids |
| Triton crash on Mac | `triton_grouped` without CUDA | Use `moe_dispatch: stacked` |
| Sliding-window test fail | Mask bug in `attention.py` | `test_sliding_window_matches_full` |

### After any `attention.py` change

Run:

```bash
pytest tests/test_attention.py -v
```

Specifically `test_sliding_window_matches_full` must pass.

---

### 12. Comparison with sibling portfolio models

| Dimension | GPT-OSS-Lite | DeepSeek-v3-Lite | LLaMA-3-Lite | HyMo | Mamba-3-Lite |
|-----------|--------------|------------------|--------------|------|--------------|
| **Paradigm** | Decoder-only TF | Decoder-only TF | Decoder-only TF | Hybrid GDN/MLA | Pure SSM |
| **Attention** | GQA 8Q/4KV | MLA latent KV | GQA full | 3:1 GDN/MLA | Complex SSD |
| **Long context** | YaRN 128K train+decode | YaRN decode-focused | θ=500K, train 2K | — | Constant state |
| **Local/global** | SWA(128)/full alt | Full + MLA | Full | Alternating | Chunkwise SSD |
| **FFN** | MoE top-2/8 + shared | DeepSeekMoE | Dense SwiGLU | Asymmetric MoE | Dense |
| **MoE aux** | Switch $\alpha=0.01$ | **Aux-loss-free gate** | — | Custom | — |
| **Sink** | Learned per-head bias | None | None | None | None |
| **RoPE extras** | Prune 25% on global | YaRN | Extended base θ | — | — |
| **KV at 128K** | ~1.13 GB (mixed) | MLA-compressed | ~2.25 GB+ (GQA full) | Hybrid state | $O(1)$ state |
| **Scale (this repo)** | 502M / 247M active | Portfolio scale | Portfolio scale | Portfolio scale | Portfolio scale |
| **Primary headline** | 2× KV + passkey 128K | MTP, μP, speculative | 78% memory stack | GDN kernel | SSD throughput |

### Deliberate distinctions

1. **vs DeepSeek-v3-Lite:** Standard aux loss instead of aux-loss-free routing; GQA+SWA instead of MLA; learned sinks instead of none.
2. **vs LLaMA-3-Lite:** MoE instead of dense FFN; sliding/full alternation; YaRN with train-time 128K alignment.
3. **vs HyMo:** Pure attention stack (no GDN); standard MoE routing; long-context via YaRN not hybrid recurrence.
4. **vs Mamba-3-Lite:** Attention-based long context with KV cache (mitigated by SWA) vs constant-size SSM state; MoE vs dense.

GPT-OSS-Lite is the portfolio's **long-context MoE + attention sink** reference implementation.

---

### Part B — Transformer stack (`models/transformer.py`)

Implementation-level detail for `models/transformer.py`. Part A above covers
system dataflow and invariants; this part owns the composition root.

### B.1 Module overview

`models/transformer.py` defines:

| Symbol | Role |
|--------|------|
| `ModelConfig` | Dataclass mirror of YAML `model:` keys with validation |
| `RMSNorm` | Pre-norm normalization (no bias, no mean centering) |
| `GPTOSSBlock` | One transformer block: attention + MoE with residuals |
| `GPTOSS` | Full model: embed → blocks → final norm → LM head |

Lower-level primitives live in sibling modules:

- `models/attention.py` — `GPTOSSAttention` (SWA/full, sink, YaRN)
- `models/moe.py` — `MoELayer` (top-2 routed + shared, aux loss)
- `models/yarn.py` — YaRN frequency scaling for RoPE

The transformer file stays thin: it wires modules together and implements
cross-cutting concerns (init, checkpointing, param math).

### B.2 `ModelConfig` fields and `__post_init__` validation

`ModelConfig` is a `@dataclass` whose fields map 1:1 to the `model:` block in
YAML configs. Defaults match `configs/pretrain_a100_502m.yaml`.

```python
@dataclass
class ModelConfig:
    vocab_size: int = 128000
    d_model: int = 768
    n_layers: int = 12
    n_heads: int = 8
    n_kv_heads: int = 4
    head_dim: int = 96
    ffn_dim: int = 1536
    n_routed_experts: int = 8
    n_activated_experts: int = 2
    n_shared_experts: int = 1
    window_size: int = 128
    sink_bias: bool = True
    rope_theta: int = 100000
    yarn_scale_factor: int = 32
    yarn_original_max_seq_len: int = 4096
    yarn_target_seq_len: int = 131072
    # ... dtype, weight_tying, init_std, moe_dispatch, etc.
```

**Full field encyclopedia:** see [§7](#7-modelconfig--config-to-code-wiring) and
[training.md](../training.md#part-b--configuration-reference).

Construction **fails fast** on inconsistent configs:

| Rule | Rationale |
|------|-----------|
| `n_heads % n_kv_heads == 0` | GQA requires integer repeat factor |
| `n_heads * head_dim == d_model` | Projection shapes must align |
| `0 < n_activated_experts <= n_routed_experts` | Valid top-k routing |
| `yarn_scale_factor >= 1` | 1 means plain RoPE |
| If `yarn_scale_factor > 1`, `yarn_original_max_seq_len < yarn_target_seq_len` | YaRN needs extrapolation headroom |

Warnings (non-fatal):

- `yarn_prune_rope_global=True` with odd `n_layers` — final layer may be
  windowed (no pruning applied)
- `eval_max_seq_len < max_seq_len` — eval context shorter than training

### B.3 `moe_dispatch` values (`stacked` | `triton_grouped`)

Default `"stacked"` — pure PyTorch MoE loop. Opt-in `"triton_grouped"` enables
the fused kernel in `models/moe_triton.py`.

**Dispatch semantics and Triton contract:** see [§8](#8-moe-dispatch-and-triton-opt-in)
and [moe.md](moe.md#sanctioned-triton-path-moe_dispatchtriton_grouped).

### B.4 `RMSNorm`

Root Mean Square Layer Normalization replaces LayerNorm in GPT-OSS-Lite. Unlike
LayerNorm, RMSNorm **does not subtract the mean** — only scales by the RMS of
activations.

For input vector $x \in \mathbb{R}^d$ and learned scale $\gamma$:

$$
\mathrm{RMSNorm}(x) = \frac{x}{\sqrt{\frac{1}{d}\sum_i x_i^2 + \epsilon}} \cdot \gamma
$$

```python
class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rms = x.detach().float().pow(2).mean(dim=-1, keepdim=True).add(self.eps).rsqrt()
        return (x * (rms * self.weight.to(rms.dtype)).to(x.dtype))
```

Design notes:

1. **RMS computed in FP32** via `x.detach().float()` — stabilizes BF16 forward
   without keeping a persistent FP32 copy of activations.
2. **`detach()` on RMS** — norm statistics do not receive gradients (standard
   pre-norm practice).
3. **Learnable `weight`** initialized to ones in `_init_weights`.
4. **`rms_norm_eps`** from config (default `1e-5`) matches LLaMA-family recipes.

Each `GPTOSSBlock` has **two** RMSNorm layers (`norm1`, `norm2`) — one before
attention, one before MoE. A third RMSNorm (`GPTOSS.norm`) sits after all
blocks, before the LM head.

### B.5 `GPTOSSBlock` construction and forward

One block implements the GPT-OSS **pre-norm residual** pattern:

```
x ← x + Attention(RMSNorm(x))
x ← x + MoE(RMSNorm(x))
```

```python
class GPTOSSBlock(nn.Module):
    def __init__(self, cfg: ModelConfig, layer_idx: int):
        super().__init__()
        self.layer_idx = layer_idx
        self.attn = GPTOSSAttention(cfg, layer_idx)
        self.moe = MoELayer(cfg)
        self.norm1 = RMSNorm(cfg.d_model, cfg.rms_norm_eps)
        self.norm2 = RMSNorm(cfg.d_model, cfg.rms_norm_eps)
```

`layer_idx` drives attention alternation inside `GPTOSSAttention`:

- **Even** `layer_idx` → sliding-window attention (`is_windowed=True`)
- **Odd** `layer_idx` → full attention (`is_windowed=False`)

See [attention-sinks.md](attention-sinks.md#part-b--implementation-modelsattentionpy) for mask construction and sink bias.

```python
def forward(self, x, positions) -> tuple[torch.Tensor, torch.Tensor]:
    x = x + self.attn(self.norm1(x), positions)
    moe_out, aux_loss = self.moe(self.norm2(x))
    x = x + moe_out
    return x, aux_loss
```

Returns updated hidden states `(B, T, d_model)` and a per-layer `aux_loss`
scalar. `models/transformer.py:GPTOSS.forward` takes the mean across layers.

### B.6 `GPTOSS` construction and submodule roles

```python
class GPTOSS(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        self.embed = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.blocks = nn.ModuleList([
            GPTOSSBlock(cfg, i) for i in range(cfg.n_layers)
        ])
        self.norm = RMSNorm(cfg.d_model, cfg.rms_norm_eps)
        self.head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        if cfg.weight_tying:
            self.head.weight = self.embed.weight
        self._init_weights()
```

| Submodule | Shape / count | Purpose |
|-----------|---------------|---------|
| `embed` | `(vocab_size, d_model)` | Token → vector |
| `blocks` | `n_layers` × `GPTOSSBlock` | Transformer stack |
| `norm` | `d_model` | Final RMSNorm before logits |
| `head` | `(d_model, vocab_size)` | LM projection (tied to embed) |

`models/transformer.py:GPTOSS.extra_repr` prints a one-line summary of the active
configuration — `d_model`, `n_layers`, `vocab`, expert pattern, `window`, and the
tied-aware total from `models/transformer.py:GPTOSS.num_parameters` — for debugging
in notebooks. `models/transformer.py:GPTOSS.num_active_parameters` computes the
per-token active count with the tied head excluded ([§5](#5-parameter-accounting)).

### B.7 Weight initialization policy

`_init_weights()` runs after module construction:

```python
def _init_weights(self) -> None:
    std = self.cfg.init_std  # default 0.02
    for module in self.modules():
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=std)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=std)
        elif isinstance(module, RMSNorm):
            nn.init.ones_(module.weight)
    for block in self.blocks:
        if hasattr(block.attn, "sink_bias") and block.attn.sink_bias is not None:
            nn.init.zeros_(block.attn.sink_bias)
```

| Parameter group | Init | Notes |
|-----------------|------|-------|
| `Linear.weight` | $\mathcal{N}(0, 0.02^2)$ | `init_std=0.02` from config |
| `Embedding.weight` | $\mathcal{N}(0, 0.02^2)$ | Same std as linear |
| `RMSNorm.weight` | ones | Standard |
| `sink_bias` | **zeros** | Model learns sink mass from scratch |

Sink zero-init means early training behaves like standard causal attention;
sink mass emerges during optimization. Theory:
[attention-sinks.md](attention-sinks.md).

Attention and MoE linear layers use `bias=False` — no separate bias init rules.

**Why `init_std = 0.02`.** `models/transformer.py:GPTOSS._init_weights` draws every
`Linear` and `Embedding` weight from $\mathcal{N}(0, \sigma^2)$ with
$\sigma = \text{init\_std} = 0.02$. The value is a variance-budget argument. For a
layer $y = Wx$ with i.i.d. zero-mean weights of variance $\sigma^2$ and input
components of variance $\text{Var}(x)$, the output components have

$$
\text{Var}(y_j) = \sum_{i=1}^{F} \text{Var}(W_{ji} x_i) = F \sigma^2 \text{Var}(x),
\qquad \text{std}(y_j) = \sigma \sqrt{F}, \tag{5}
$$

where $F$ is the fan-in — the number of inputs summed into one output unit. Per output
unit, the projection layers give

$$
\sigma\sqrt{F} = 0.02 \sqrt{768} \approx 0.55
\quad (\text{q\_proj, kv\_proj, o\_proj, expert W1/W3}),
\qquad
0.02 \sqrt{1536} \approx 0.78 \quad (\text{expert W2}). \tag{6}
$$

Std 0.5–0.8 per unit is the sweet spot: large enough to keep signals alive across
BF16's $2^{-8}$ relative grid ([numerics](optimizers-and-numerics.md)), small enough that no
sublayer output saturates a softmax or approaches FP32's exponential overflow at
$z \approx 88.7$ (numerics.md §8.3). At $\sigma = 0.1$ the same layers would sit at
std 2.8–3.9 and, via (7)–(8), head-vector norms near 27 and score std ≈ 7.7 —
softmaxes pinned to a saturated corner and a residual stream swamping BF16's grid; at
$\sigma = 0.01$ (std 0.28–0.39) the score std drops to ≈ 0.08, the softmax is nearly
uniform, and the gradient signal through the score path is four times weaker
([optimizers](optimizers-and-numerics.md)). `init_std = 0.02` is the middle of that range.

The scale that actually enters attention is larger than the per-unit std. A query head
vector $q^h \in \mathbb{R}^{96}$ is a slice of the q-projection output: 96 components,
each of std $\sigma\sqrt{768}$ (fan-in of `q_proj`), so its typical L2 norm is

$$
\|q^h\| \approx \sigma \sqrt{d_{\text{model}} \cdot D}
= 0.02 \sqrt{768 \times 96} = 0.02 \sqrt{73728} \approx 5.4. \tag{7}
$$

This is the "large but controlled" scale: controlled by pre-norm (every sublayer input
is normalized to unit RMS, so $\text{Var}(x) = 1$ in (5)) and by the
$\frac{1}{\sqrt{D}}$ attention scaling, which brings the pre-softmax score
$s = q^h \cdot k^h / \sqrt{D}$ to

$$
\text{Var}(s) = \frac{1}{D} \sum_{i=1}^{D} \text{Var}(q_i k_i)
= (768 \cdot 0.02^2)^2 \approx 0.094,
\qquad \text{std}(s) \approx 0.31. \tag{8}
$$

Scores of std ≈ 0.3 sit far from both the flat tail and the saturated corner of the
softmax — the regime where the learned sink bias, clamped to $[-10, 15]$ (roughly
$-32\sigma$ to $+48\sigma$), can actually reshape the distribution
([attention-sinks.md](attention-sinks.md)). Without the $1/\sqrt{D}$ factor the score
std would be $\sqrt{96} \cdot 0.31 \approx 3.0$, and scores wandering over ±12 would
force softmaxes toward saturation before the sink can act.

**Residual-stream variance across 12 layers.** Write the pre-norm block as
$x_{l+1} = x_l + f^{\text{attn}}_l(\text{RMSNorm}(x_l)) + f^{\text{moe}}_l(\text{RMSNorm}(x_l))$.
At initialization the sublayer outputs are zero-mean and uncorrelated with the stream,
so variances add:

$$
\text{Var}(x_{l+1}) = \text{Var}(x_l) + \text{Var}(f^{\text{attn}}_l)
+ \text{Var}(f^{\text{moe}}_l). \tag{9}
$$

**Worst case — no normalization.** Each sublayer is approximately scale-homogeneous
(attention and experts are linear maps; SwiGLU is gated but bias-free), so its output
std scales with the input std, $\text{std}(f_l) \approx \alpha_l\, \text{std}(x_l)$
with gain $\alpha_l \approx \sigma\sqrt{F}$, and (9) becomes geometric:

$$
\text{Var}(x_L) = \text{Var}(x_0) \prod_{l=0}^{L-1} (1 + \alpha_l^2), \qquad
\alpha_l \approx \sigma \sqrt{F}. \tag{10}
$$

The growth is exponentially sensitive to $\sigma$: at $\sigma = 0.1$ the per-block
gain $\alpha \approx 3.8$ gives $(1 + 3.8^2)^{12} \approx 10^{14}$ — guaranteed
overflow; at $\sigma = 0.02$ the per-block gain is $\alpha \approx 0.76$ and the
stream *shrinks*, $\text{std}(x_{12}) = 0.02 \sqrt{(1.58)^{12}} \approx 0.31$ — the
signal decays toward the BF16 noise floor by layer 12. Either way the depth
dependence is exponential.

**With pre-norm.** RMSNorm pins every sublayer input to unit variance, so the sublayer
output variances $v^{\text{attn}}_l, v^{\text{moe}}_l$ are bounded by the fan-in gains
of (6) and no longer compound with the stream magnitude. The recursion becomes
arithmetic:

$$
\text{Var}(x_L) = \text{Var}(x_0) + \sum_{l=0}^{L-1} \left( v^{\text{attn}}_l
+ v^{\text{moe}}_l \right) = \text{Var}(x_0) + L\, v. \tag{11}
$$

Starting from the embedding rows ($\text{Var}(x_0) = \sigma^2 = 4\times10^{-4}$) with
per-block contribution $v \approx 0.58$ (dominated by the expert W2 fan-in, (6)):

$$
\text{std}(x_{12}) = \sqrt{0.0004 + 12 \times 0.58} \approx 2.6. \tag{12}
$$

The stream stays O(1) whether the stack has 12, 24, or 48 layers, and the final
`models/transformer.py:GPTOSS.norm` (RMSNorm) restores unit RMS before the tied head
projects to logits — the head always sees an O(1) input.

### B.8 Forward pass, `positions`, return contract `(logits, aux_loss)`

High-level dataflow: [§3](#3-gptossforward-dataflow). Source implementation:

```python
def forward(self, idx, positions=None) -> tuple[torch.Tensor, torch.Tensor]:
    B, T = idx.shape
    if positions is None:
        positions = torch.arange(T, device=idx.device)
    x = self.embed(idx)

    aux_losses = []
    use_grad_ckpt = (
        getattr(self, "gradient_checkpointing", False)
        and torch.is_grad_enabled()
    )
    grad_ckpt_every = max(1, getattr(self, "grad_ckpt_every", 3))

    for layer_idx, block in enumerate(self.blocks):
        if use_grad_ckpt and (layer_idx % grad_ckpt_every == 0):
            x, aux = torch.utils.checkpoint.checkpoint(
                block, x, positions, use_reentrant=False,
            )
        else:
            x, aux = block(x, positions)
        aux_losses.append(aux)

    aux_loss = torch.stack(aux_losses).mean() if aux_losses else torch.zeros(...)
    x = self.norm(x)
    logits = self.head(x)
    return logits, aux_loss
```

```
input_ids (B, T)
    │
    ▼
Embedding ──► x (B, T, 768)
    │
    ├──► Block 0 ──► aux_0
    ├──► Block 1 ──► aux_1
    │       ...
    └──► Block 11 ──► aux_11
    │
    ▼
RMSNorm ──► head ──► logits (B, T, vocab_size)
    │
aux_loss = mean(aux_0, ..., aux_11)
```

**`positions`:** shape `(T,)` or broadcastable; passed to YaRN RoPE inside each
attention layer. Default `torch.arange(T)` assumes contiguous positions starting
at 0. For inference with KV cache, `inference/generate.py` passes per-step
position indices — see [inference.md](../inference.md).

**Return contract:** `logits` is `(B, T, vocab_size)` in model dtype (BF16 under
autocast); `aux_loss` is a scalar mean MoE load-balancing loss. Loss
composition lives in `training/pretrain.py` — not in the model class.

### B.9 Gradient checkpointing schedule (`grad_ckpt_every`)

Memory-heavy activations are traded for extra backward recomputation:

```python
def enable_gradient_checkpointing(self, every: int = 3) -> None:
    self.gradient_checkpointing = True
    self.grad_ckpt_every = every
```

When enabled, blocks where `layer_idx % every == 0` are wrapped in
`torch.utils.checkpoint.checkpoint(..., use_reentrant=False)`.

With `every=3` (A100 config) and 12 layers, blocks **0, 3, 6, 9** are
checkpointed — four of twelve blocks, ~33% recomputation overhead for
materially lower activation memory.

`pretrain.py` calls `model.enable_gradient_checkpointing(every=...)` when
`grad_checkpoint: true`. Checkpointing is **disabled** when
`torch.is_grad_enabled()` is false (eval / `@torch.no_grad()` inference).

Training integration: [§10](#10-training-pipeline-integration).

### B.10 `num_parameters` / `num_active_parameters` + 502M breakdown

**502M / 247M breakdown and KV math:** [§5](#5-parameter-accounting).

`num_parameters()` deduplicates weight tying:

```python
def num_parameters(self) -> int:
    seen_ids = set()
    total = 0
    for p in self.parameters():
        if id(p) in seen_ids:
            continue
        seen_ids.add(id(p))
        total += p.numel()
    return total
```

Production config yields **501,836,640** total (~502M). The tied
embedding/head weight is counted once.

`num_active_parameters()` estimates parameters used per forward under top-k MoE
sparsity:

```python
def num_active_parameters(self) -> int:
    non_moe = ...  # all params except names containing "experts" or "router"
    expert_params = 3 * d_model * ffn_dim   # W1, W3, W2 per expert
    moe_active = (n_activated_experts + n_shared_experts) * expert_params
    router_params = d_model * n_routed_experts
    return non_moe + (moe_active + router_params) * n_layers
```

Production: **247,032,672** active (~247M), ~50.8% sparsity. This is an
**analytical estimate** aligned with Chinchilla active-param reporting — not a
runtime profiler. Always prefer `model.num_parameters()` over hand sums.

The `non_moe` sweep skips names containing `"router"` as well as `"experts"`, so the
router gate is counted exactly once via `router_params`; the counter returns the
derived figure directly. See [§5](#5-parameter-accounting) (eqs. 3–4) for the
derivation.

### B.11 Weight tying

When `weight_tying: true`:

```python
self.head.weight = self.embed.weight
```

The LM head and token embedding share one `(vocab_size, d_model)` matrix.

1. **98,304,000 parameter savings** — the head is a view of the embedding, so the
   untied total of 600,140,640 drops to 501,836,640 (derived in [§5](#5-parameter-accounting),
   eqs. 1–2)
2. **Consistent input/output token geometry** — standard in GPT-2/LLaMA families

Implications:

- Optimizer updates apply once to the shared tensor
- Checkpoint `state_dict` contains one key for the shared weight
- `num_parameters()` must deduplicate — counting both `embed` and `head` would
  double-count
- `models/transformer.py:GPTOSS.num_active_parameters` inherits the same exclusion:
  the tied head contributes nothing to the 247M active figure (derived in
  [§5](#5-parameter-accounting))

Set `weight_tying: false` only for ablation experiments.

### B.12 Config validation edge cases

**`head_dim` must be even** — RoPE rotates pairs of dimensions. Odd `head_dim`
raises `ValueError`.

**YaRN with `scale_factor=1`** — degenerate case, plain RoPE without length
extrapolation. Valid for smoke configs with small `eval_max_seq_len`.

**`sink_bias: false`** — attention layers omit learnable sink parameters. Forward
path skips clamp and sink-augmented softmax. Use only for ablations; production
GPT-OSS uses sinks.

**Layer count and alternation** — with `n_layers=12`, you get exactly 6 windowed
and 6 global layers. Changing `n_layers` without updating benchmarks invalidates
the 2× KV-cache headline unless you re-derive `N_WINDOWED = n_layers // 2`.

### B.13 How to verify

```bash
python3 -m pytest tests/test_models.py tests/test_smoke.py -v
```

Spot-check parameter counts on a fresh model:

```python
from models.transformer import ModelConfig, GPTOSS
cfg = ModelConfig()
m = GPTOSS(cfg)
assert m.num_parameters() == 501_836_640
assert m.num_active_parameters() == 247_032_672
```

---

### 13. Where to go next

| Goal | Resource |
|------|----------|
| Sink bias deep-dive | [attention-sinks.md](attention-sinks.md) |
| MoE routing and aux loss | [moe.md](moe.md) |
| Mathematical foundations | [foundations-and-architecture.md](foundations-and-architecture.md) |
| Run KV benchmark | `python3 scripts/kv_cache_benchmark.py` |
| Start training | `python3 training/pretrain.py --config configs/pretrain_a100_502m.yaml` |

---

## References

- [`models/transformer.py:GPTOSS`](../../models/transformer.py) — top-level model, `num_parameters`, `num_active_parameters`
- [`models/transformer.py:ModelConfig`](../../models/transformer.py) — config dataclass + validation
- [`models/transformer.py:GPTOSSBlock`](../../models/transformer.py) — attention + MoE residual block
- [`models/attention.py`](../../models/attention.py) — SWA/full attention, sink bias
- [`models/moe.py`](../../models/moe.py) — MoE layer, router, aux loss
- [attention-and-positional.md](attention-and-positional.md) — attention math, RoPE, YaRN derivations
- [attention-sinks.md](attention-sinks.md) — sink-bias deep-dive
- [moe.md](moe.md) — MoE implementation + Triton opt-in
- [training.md](../training.md) — training loop and config reference
- [inference.md](../inference.md) — `MixedKVCache`, generation

<!-- docs:verified 2026-08-05 · 6491066 -->
