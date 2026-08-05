# GPT-OSS-Lite — Attention and Positional Encoding

## Part A — Attention Math

> **Theory chapter T1.** From-scratch derivation of the arithmetic inside scaled dot-product attention, mapped onto `models/attention.py`. Assumes the primer level of [foundations](foundations-and-architecture.md) §2–§4; the sink mechanism is treated in depth in [ATTENTION_SINKS](attention-sinks.md) and the KV-cache consequences in [kv cache engineering](../inference.md).

---

### 1. 60-second summary

Scaled dot-product attention computes, for every position, a weighted average of value vectors. The weights are softmax over query-key dot products, divided by the square root of the head width so scores stay near unit variance. A mask (causal, sliding-window, learned sink logit) is added to the scores first so forbidden keys receive zero weight. GPT-OSS-Lite clamps its learned sink logits to `[-10, 15]` (`models/attention.py:SINK_CLAMP_MIN` / `models/attention.py:SINK_CLAMP_MAX`) so the mask arithmetic never overflows BF16. All production attention runs through `F.scaled_dot_product_attention` inside `models/attention.py:causal_attention`, which dispatches to a math, memory-efficient, or flash backend; flash never materializes the T-by-T score matrix. An FP32 oracle, `models/attention.py:manual_causal_attention`, is the test reference. This chapter derives each piece, then walks the code, including two verified behavioral quirks (the sink-path additive mask and the square-mask window) that the current test suite does not pin down.

### 2. Why it matters here

Attention is the arithmetic that everything else in GPT-OSS-Lite hangs off:

- **Alternating SWA/full.** Twelve layers, even indices sliding-window with
  $W = 128$, odd indices full causal (`models/transformer.py:GPTOSSBlock` constructs `models/attention.py:GPTOSSAttention` per layer; the alternation is decided in `models/attention.py:GPTOSSAttention.__init__` via `layer_idx % 2`). The mask math in §4.4–4.5 is what makes the two patterns differ — and, as §5.6 documents, where they currently do not differ during prefill.
- **Learned sink bias.** Each of the 8 heads carries one scalar logit that acts
  as an extra softmax column with a zero value vector. It exists to absorb attention mass that sliding-window eviction would otherwise scatter; the mathematics is derived in [ATTENTION_SINKS §4](attention-sinks.md), and the BF16 clamp rationale in [ATTENTION_SINKS §6](attention-sinks.md). This chapter supplies the underlying softmax and mask machinery.
- **GQA.** 8 query heads share 4 KV heads (`head_dim = 96`). The broadcast in
  §4.6 halves KV-cache bytes and KV traffic per token; combined with the six windowed layers it produces the measured 2.00× KV reduction at 128K ([ATTENTION_SINKS §8](attention-sinks.md), `scripts/kv_cache_benchmark.py`).
- **SDPA as the single execution path.** `models/attention.py:causal_attention`
  is the only attention entry point in training and inference (`inference/generate.py:_attn_forward_layer`). Which backend runs — math, memory-efficient, or flash — decides whether a $T \times T$ matrix is ever materialized, which is the difference between 128K evaluation fitting in VRAM or not.
- **Budget honesty.** The ≥85% passkey @128K figure is a **target**; no
  pretraining has run. The 2.00× / 1.94× KV figures are **measured**; every A100 throughput figure elsewhere is `[INFERENCE]`. This chapter contains no performance numbers that were not derived above.

### 3. Intuition

Think of each row of $K$ as a labeled point in $d$-dimensional space, and the query $q_i$ as a probe. The dot product $q_i \cdot k_j$ is (up to the lengths of the vectors) the projection of $k_j$ onto the direction of $q_i$ — a raw similarity score, unbounded and signed. Attention then answers: "among all keys, how should I split a unit budget of attention?" Three requirements fix the form of that split:

1. Weights must be non-negative and sum to one — the output is a convex
   combination of the value rows, an average, not a vector addition.
2. The split must be a smooth, differentiable function of the scores so
   gradients can flow through it.
3. The mapping should be "winner-take-most": a key with a slightly better score
   gets exponentially more weight, but not all of it.

Softmax is the canonical smooth relaxation of argmax that satisfies all three (§4.2). The $1/\sqrt{d}$ factor is a **temperature**: it controls how sharp the distribution is. Scores of unit variance (§4.3) keep the exponential from blowing up or degenerating. Masks are hard constraints that delete keys from the support set before normalization — and the learned sink (§4.4) is a clever softener: a dummy key with a learnable constant logit and a zero value vector, so the model can park attention mass where it does nothing, instead of being forced to redistribute it when the window evicts a token.

The fused kernels (§4.7) implement the same arithmetic but never build the $T \times T$ score matrix; they exploit that softmax normalization is *online* — you can accumulate it block by block as long as you rescale when a new block raises the running maximum.

### 4. Theory and derivation

### 4.1 The attention operation

Scaled dot-product attention is defined as

$$
\mathrm{Attn}(Q, K, V) = \mathrm{softmax}\!\left(\frac{QK^\top}{\sqrt{d}} + M\right) V
\tag{1}
$$

where $Q \in \mathbb{R}^{T_q \times d}$ holds one query per row, $K, V \in \mathbb{R}^{T_k \times d}$ hold keys and values, $d$ is the head dimension, and $M \in (\mathbb{R} \cup \{-\infty\})^{T_q \times T_k}$ is an additive mask. Writing the score matrix $S = QK^\top / \sqrt{d}$ and the weight matrix $A = \mathrm{softmax}(S + M)$ row-wise, row $i$ of the output is

$$
o_i = \sum_{j=1}^{T_k} \alpha_{ij}\, v_j, \qquad \alpha_{ij} \ge 0, \quad
\sum_j \alpha_{ij} = 1.
$$

The mask enters *before* softmax, never after: it modifies the support of the distribution, not the weights of already-normalized probabilities.

### 4.2 Softmax: definition, why it appears, numerical stability

For a vector of logits $z \in \mathbb{R}^n$, the softmax function is

$$
\sigma(z)_i = \frac{e^{z_i}}{\sum_{j=1}^{n} e^{z_j}}.
\tag{2}
$$

It appears in attention for three reasons, each of which is a design constraint of the operation in §4.1:

- **Normalization.** $\sigma(z)$ is a probability distribution over the $n$
  keys: non-negative, summing to one, so the output is a convex combination of values.
- **Max-entropy derivation.** Among all distributions $p$ over keys with a fixed
  expected score $\sum_j p_j z_j$, the one maximizing entropy $-\sum_j p_j \ln p_j$ is the exponential-family distribution $p_j \propto e^{\lambda z_j}$, of which (2) is the normalized form. It is the *least committed* distribution consistent with a given average similarity — a principled reason, not a convention.
- **Smoothness.** Unlike $\arg\max$, (2) is differentiable everywhere, with
  Jacobian $\partial \alpha_i / \partial z_j = \alpha_i (\delta_{ij} - \alpha_j)$; gradients flow to every key, weighted by current attention.

**Shift invariance and the stable form.** Softmax is invariant to adding a constant to every logit:

$$
\sigma(z - c \mathbf{1})_i = \frac{e^{z_i - c}}{\sum_j e^{z_j - c}}
= \frac{e^{z_i}}{\sum_j e^{z_j}} = \sigma(z)_i,
\tag{3}
$$

because the factor $e^{-c}$ cancels between numerator and denominator. The numerically stable form exploits (3) with $c$ equal to the row maximum:

$$
\sigma(z)_i = \frac{e^{z_i - m}}{\sum_j e^{z_j - m}}, \qquad m = \max_j z_j,
\qquad z_i - m \le 0 \;\; \forall i.
\tag{4}
$$

This is not optional in floating point. The largest finite value in both FP32 and BF16 is $3.4 \times 10^{38}$, and $e^x$ reaches it at $x \approx 88.7$ (derived in §4.4). If any logit exceeded that, $e^{z_i}$ would round to $+\infty$, the denominator would be $+\infty$, and every weight would collapse to NaN via $\infty / \infty$. Subtracting the max keeps every exponent argument $\le 0$, so all exponentials live in $(0, 1]$. PyTorch's softmax kernels do this internally; the danger is upstream, where a *mask value* (not a softmax argument) can be huge — see §4.4.

### 4.3 Scaled dot product: why $1/\sqrt{d}$

Take the entries of $q_i$ and $k_j$ to be independent draws with mean zero and unit variance, which is the regime the projections in `models/attention.py:GPTOSSAttention.forward` are initialized into (`init_std = 0.02` in `models/transformer.py:ModelConfig`, and the linear projections are weight-normalized so activation variance is $O(1)$). The raw score is a sum of $d$ products:

$$
\mathrm{Var}(s_{ij}) = \mathrm{Var}\!\left(\sum_{l=1}^{d} q_{il} k_{jl}\right)
= \sum_{l=1}^{d} \left(\mathbb{E}[q_{il}^2]\mathbb{E}[k_{jl}^2] -
\mathbb{E}[q_{il}]^2 \mathbb{E}[k_{jl}]^2 \right) = \sum_{l=1}^{d} 1 = d,
\tag{5}
$$

using independence of $q_{il}$ and $k_{jl}$ and $\mathbb{E}[q_{il}^2] = \mathrm{Var}(q_{il}) + \mathbb{E}[q_{il}]^2 = 1$. The score standard deviation is therefore $\sqrt{d}$, and dividing by it yields unit-variance scores:

$$
\tilde s_{ij} = \frac{s_{ij}}{\sqrt{d}}, \qquad
\mathrm{Var}(\tilde s_{ij}) = \frac{d}{d} = 1.
\tag{6}
$$

For this model, $d = 96$, so $\sqrt{d} \approx 9.8$. Why does the scaling matter? Without it, scores scale as $\sqrt{d}$: for $d = 96$ they spread to $\pm 10$, and with $T_k$ keys the row maximum grows like $\sqrt{2 \ln T_k}$ (the expected maximum of $n$ unit-variance Gaussians; at $T_k = 131072$, $\sqrt{2 \ln 131072} \approx 4.9$). The softmax of such scores concentrates almost all mass on one key: $\alpha_i \to 1$ for the argmax, and the Jacobian $\alpha_i(\delta_{ij} - \alpha_j)$ vanishes for every $j$ — gradients stop flowing to all non-winning keys. Unit-variance logits keep the distribution smooth across sequence lengths, which is exactly why this repo can train at $T = 4096$ and evaluate at $T = 131072$ without re-tuning the temperature. The same $\sqrt{d}$ appears as the "temperature" of the exponential family in §4.2: $\lambda = 1/\sqrt{d}$.

### 4.4 Causal masking: semantics, mask-add vs mask-fill, the BF16 trap

**Semantics.** A causal mask allows query $i$ to attend only to keys $j \le i$:

$$
M_{ij} = \begin{cases} 0 & j \le i \\ -\infty & j > i \end{cases},
\qquad
\alpha_{ij} = \frac{e^{\tilde s_{ij} + M_{ij}}}{\sum_{j'=1}^{T_k} e^{\tilde
s_{ij'} + M_{ij'}}} = \frac{e^{\tilde s_{ij}}}{\sum_{j' \le i} e^{\tilde
s_{ij'}}}.
\tag{7}
$$

Because $e^{-\infty} = 0$, forbidden keys contribute nothing to numerator or denominator; the surviving weights renormalize automatically. Two mechanical ways to apply $M$:

- **mask-fill**: write $-\infty$ into the score tensor itself
  (`scores.masked_fill(causal, float("-inf"))`), as the oracle `models/attention.py:manual_causal_attention` does;
- **mask-add**: keep a separate mask tensor and add it to scores
  (`scores + attn_mask`), as SDPA does with float masks. A *boolean* mask is sugar for mask-add: `True` means add $0$, `False` means add $-\infty$.

The two are mathematically identical; the difference is mechanical (in-place mutation vs composition) and numerical (where the $-\infty$ lives).

**The BF16 overflow trap.** Addition with true $-\infty$ is safe: $x + (-\infty) = -\infty$ for every finite $x$. The traps are the *other* directions, and they are the reason the sink bias is clamped:

- **exp overflow in normalization.** Softmax evaluates $e^{z_i - m}$ with $z_i -
  m \le 0$ (§4.2), so normalization itself is safe. But if the mask *adds a huge finite positive logit* — an unbounded learned sink, say $s_h = 1000$ — then the augmented row contains a logit of $+1000$, and $e^{1000} = +\infty$ even after max subtraction in low precision; the denominator becomes $\infty$ and every weight becomes NaN ($\infty / \infty$). The overflow threshold is

$$
e^{x} \le 3.4 \times 10^{38} \iff x \le \ln(3.4 \times 10^{38}) \approx 88.7.
\tag{8}
$$

- **finite-mask overflow on addition.** A mask built with a huge *finite*
  negative sentinel (e.g. `-1e38` in a low-precision tensor) can overflow to $\pm\infty$ when *added* to an opposite-sign score: $-10^{38} + 3 \times 10^{38} = +2 \times 10^{38} = +\infty$ in BF16, and $+\infty + (-\infty)$ later yields NaN. The rule: use true $-\infty$ (or $0$), never a large finite sentinel.

GPT-OSS-Lite's defense is exactly (8): `GPTOSSAttention.forward` clamps the sink parameter to $[SINK_CLAMP_MIN, SINK_CLAMP_MAX] = [-10, 15]$ before it enters the mask. With $s_h \le 15$, the sink logit never exceeds $e^{15} \approx 3.3 \times 10^6 \ll 3.4 \times 10^{38}$, and $-10$ is safe too: $e^{-10} \approx 4.5 \times 10^{-5}$, effectively "no sink". The upper bound 15 is also comfortably above the score spread derived in §4.3 (≈4.9 at 128K), so the sink can absorb essentially all mass when trained to do so. Full rationale and the gradient-through-clamp note are in
[ATTENTION_SINKS §6](attention-sinks.md).

### 4.5 Sliding-window masking as a banded mask

A sliding window of width $W$ restricts query $i$ to the most recent $W$ keys, intersected with causality:

$$
\mathcal{A}(i) = \{ j : j \le i \;\wedge\; i - j < W \}.
\tag{9}
$$

The resulting mask is **banded**: in row $i$, entries $j < i - W + 1$ are $-\infty$. The allowed set shrinks from a full triangle to a band of width $W$:

$$
N_{\mathrm{dense}} = \frac{T(T+1)}{2} \approx \frac{T^2}{2}, \qquad
N_{\mathrm{sw}} = \sum_{i=0}^{T-1} \min(i+1, W)
= \frac{W(W+1)}{2} + (T - W)\, W \approx W\, T \quad (T \gg W).
\tag{10}
$$

Attention FLOPs are proportional to the number of allowed pairs: each pair costs $2d$ FLOPs for the dot product and $2d$ FLOPs for the value accumulation ($4d$ total). The saving per windowed layer is therefore

$$
\frac{\mathrm{FLOPs}_{\mathrm{dense}}}{\mathrm{FLOPs}_{\mathrm{sw}}}
= \frac{4d\, N_{\mathrm{dense}}}{4d\, N_{\mathrm{sw}}}
\approx \frac{T}{2W}
\xrightarrow{T = 131072,\; W = 128} \frac{131072}{256} = 512.
\tag{11}
$$

The same ratio bounds the score-matrix memory: dense attention materializes $\approx T^2/2$ score elements, banded attention $\approx WT$. Neither the FLOP nor the memory saving is the repo's headline metric, though — the 2.00× number is the **KV-cache** reduction (windowed layers cache at most $W$ tokens), which is derived in [ATTENTION_SINKS §8](attention-sinks.md) and measured by `scripts/kv_cache_benchmark.py` (2.00× at 128K, 1.94× at 4K). The banded-mask arithmetic above is what *would* deliver the FLOP side of SWA; §5.6 documents where the current code actually applies it.

### 4.6 GQA: head broadcast and cache bytes

Grouped-query attention projects $H_{\mathrm{kv}} = 4$ key/value heads and repeats each one to serve $g = H / H_{\mathrm{kv}} = 8 / 4 = 2$ query heads. Formally, the key used by query head $h$ is

$$
\hat K_h = K_{\lfloor h / g \rfloor}, \qquad g = \frac{H}{H_{\mathrm{kv}}} = 2,
\tag{12}
$$

i.e. heads $\{0,1\}$ share $K_0$, heads $\{2,3\}$ share $K_1$, and so on. The gain is measured in bytes: with BF16 (2 bytes), $D = 96$:

$$
\text{KV bytes per token per layer} = 2 \cdot H_{\mathrm{kv}} \cdot D \cdot 2
= 2 \cdot 4 \cdot 96 \cdot 2 = 1536,
$$

against $2 \cdot 8 \cdot 96 \cdot 2 = 3072$ for full MHA — a 2× cut before any windowing, and the multiplicand in the 2.00× headline. In code the broadcast is `models/attention.py:repeat_kv`, discussed in §5.5.

### 4.7 SDPA backends, and flash attention's online softmax

`F.scaled_dot_product_attention` hides three implementations behind a dispatch heuristic (dtype, shape, mask type, device):

- **math backend**: the textbook algorithm of (1). Materializes the full
  $B \times H \times T_q \times T_k$ score matrix in memory, softmaxes it, then multiplies by $V$. $O(T^2)$ memory, always available (it is the CPU fallback).
- **memory-efficient backend** (xformers lineage): processes the output in
  blocks, computing a softmax per block with online rescaling (below). Never holds the full $T \times T$ matrix, but per-block $B_r \times B_c$ score tiles still materialize; supports limited mask shapes and requires aligned contiguous inputs.
- **flash backend** (FlashAttention-2 lineage): a fully fused kernel. Tiles
  $Q$, $K$, $V$ into SRAM-resident blocks, applies online softmax, and writes only the $B_r \times d$ output tile back to HBM. Nothing $O(T^2)$ ever exists — not the score matrix, not the weights. Uses tensor-core matrix-multiply units and is native-BF16, which is why it matters for this repo's BF16 GQA stack: KV heads are shared, so each KV block is read once per query block, and at $T = 131072$ the alternative (math backend) would require a $T^2$ matrix no GPU holds.

**Online softmax rescaling.** Fusing requires replacing the two-pass softmax (scan for max, then normalize) with a one-pass version that can absorb blocks in any order. Process key blocks left to right, maintaining a running row-max $m$ and an unnormalized accumulator $O$ (output) and $l$ (denominator). When a new score block $S'$ arrives with block maximum $m' = \max_j S'_{ij}$, every previously accumulated term was normalized by $e^{m_{\mathrm{old}}}$ and must be re-expressed in units of $e^{m}$ with $m = \max(m_{\mathrm{old}}, m')$:

$$
m \leftarrow \max(m_{\mathrm{old}}, m'), \qquad
O \leftarrow O\, e^{m_{\mathrm{old}} - m} + e^{S' - m}\, V', \qquad
l \leftarrow l\, e^{m_{\mathrm{old}} - m} + \mathbf{1}^\top e^{S' - m}.
\tag{13}
$$

The rescale factor $e^{m_{\mathrm{old}} - m}$ is $\le 1$ and only deviates from 1 when a new block actually raises the max — typically a few times per row. After the last block, the accumulated output is divided by the accumulated denominator:

$$
o_i = \frac{O_i}{l_i}.
\tag{14}
$$

The per-block weights are never stored, only the running pair $(m, l)$; that is the entire trick. (13)–(14) are exactly the FlashAttention paper's online softmax, and they make (1) memory-embarrassingly parallel across output blocks while remaining bitwise-equivalent-in-expectation to the two-pass version up to floating-point reassociation.

### 4.8 The sink column: one more key with a constant logit

The learned sink is an extra softmax column with logit $s_h$ (per head $h$) and a zero value vector. Augmenting the banded/causal score rows:

$$
\alpha_{ij} = \frac{e^{\tilde s_{ij}}}{\sum_{j' \in \mathcal{A}(i)} e^{\tilde
s_{ij'}} + e^{s_h}}, \qquad o_i = \sum_{j \in \mathcal{A}(i)} \alpha_{ij}\, v_j,
\tag{15}
$$

so the sink's weight $e^{s_h} / Z_i$ is absorbed without contributing to the output ($v_{\mathrm{sink}} = 0$). With $s_h \to +\infty$ the output tends to $\mathbf{0}$; with $s_h \to -\infty$ the sink vanishes. This is the formulation of [ATTENTION_SINKS §4.3](attention-sinks.md); §5.2 shows how the code implements it as a zero K/V column plus a mask column.

### 5. Code walkthrough

All symbols below live in `models/attention.py` unless noted.

### 5.1 The module: `GPTOSSAttention.forward` and `extra_repr`

`GPTOSSAttention` is constructed per layer by `models/transformer.py:GPTOSSBlock` with `cfg` from `models/transformer.py:ModelConfig`. Construction stores the alternation decision — `self.is_windowed = (layer_idx % 2 == 0)` — and the GQA ratio `self.n_rep = self.n_heads // self.n_kv_heads = 2`. Projections are `q_proj: (768, 8·96)`, `kv_proj: (768, 2·4·96)`, `o_proj: (8·96, 768)`, all bias-free; the sink is an `nn.Parameter` of shape `(n_heads,)` initialized to zero when `cfg.sink_bias` is set. `models/yarn.py:YaRNRoPE` is instantiated with the §4.3-relevant `head_dim=96` and the YaRN constants (θ=100K, scale 32, 4K→128K; see [rope_yarn](attention-and-positional.md)).

`models/attention.py:GPTOSSAttention.forward` implements (1) end to end:

```python
query_states = self.q_proj(x).view(B, T, self.n_heads, self.head_dim)
kv = self.kv_proj(x).view(B, T, 2, self.n_kv_heads, self.head_dim)
key_states, value_states = kv[:, :, 0], kv[:, :, 1]
```

One fused `kv_proj` produces both K and V for only $H_{\mathrm{kv}}$ heads — the GQA saving of §4.6 happens at projection time, before any attention arithmetic. The tensors are transposed to `(B, H, T, D)`, RoPE is applied (`models/rotary.py:apply_rope`, with `self._n_pruned_dims()` zeroing 25% of frequency pairs on global layers only), then the KV heads are broadcast:

```python
key_states = repeat_kv(key_states, self.n_rep)
value_states = repeat_kv(value_states, self.n_rep)
```

The sink clamp is applied here, *before* the mask is built — the parameter itself is never mutated:

```python
sink_bias_clamped = self.sink_bias.clamp(SINK_CLAMP_MIN, SINK_CLAMP_MAX)
```

Then `models/attention.py:causal_attention` is called with `window=self.window_size if self.is_windowed else None`, and the output is transposed, `contiguous()`, viewed back to `(B, T, H·D)`, and passed through `o_proj`. `models/attention.py:GPTOSSAttention.extra_repr` renders the layer identity used by `repr(model)`: `"layer={i} (SWA|Full[, pruned=n]), H=8/4, D=96, window=128"`, where the `pruned=` suffix appears only for global layers with `yarn_prune_rope_global` enabled.

### 5.2 `causal_attention`: three SDPA paths

`models/attention.py:causal_attention` is a dispatcher over four cases. With no sink and no window on a square input it takes the kernel-native fast path:

```python
if T_q == T_k:
    return F.scaled_dot_product_attention(query_states, key_states, value_states, is_causal=True)
```

`is_causal=True` lets the fused kernels use their optimized causal loop (only half the score blocks), which no explicit mask can match — this is the path the flash backend is designed for. With a window but no sink, the boolean masks are composed and converted to a proper additive mask:

```python
mask = _causal_mask(T_q, device, dtype) & _window_mask(T_q, T_k, window, device, dtype)
attn_mask = torch.where(mask, 0.0, float("-inf")).to(dtype).unsqueeze(0).unsqueeze(0)
```

`torch.where(mask, 0.0, -inf)` implements (7) exactly: allowed → $+0$, forbidden → true $-\infty$ (the safe form per §4.4). With a sink, the value side gains a zero column and the bias rides on the mask:

```python
sink_k = torch.zeros(B, H, 1, query_states.shape[-1], device=device, dtype=dtype)
sink_v = torch.zeros(B, H, 1, value_states.shape[-1], device=device, dtype=value_states.dtype)
k_ext = torch.cat([key_states, sink_k], dim=2)
v_ext = torch.cat([value_states, sink_v], dim=2)
```

The mask becomes shape `(H, T_q, T_k + 1)`: the first $T_k$ columns come from `causal.to(dtype)` and the last column carries the clamped per-head bias, implementing (15). See §5.6 for the exact float semantics of that mask, which differ from the window path above. Whether the mask forces a fallback from the fused kernels to math/mem-efficient depends on the torch version and tensor alignment; the code never forces a backend, and Q, K, V must share dtype for SDPA to dispatch at all.

### 5.3 Mask helpers: `_causal_mask`, `_window_mask`

`models/attention.py:_causal_mask` builds the lower triangle via broadcasting:

```python
idx = torch.arange(T, device=device)
return idx.unsqueeze(1) >= idx.unsqueeze(0)  # (T_q, T_k)
```

Entry $(i, j)$ is `idx[i] >= idx[j]`, i.e. `True` where $j \le i$ — the boolean form of (7). It is `lru_cache`d on `(T, device, dtype)` so prefill and decode reuse one allocation per shape. `models/attention.py:_window_mask` has two branches. For decode (`T_q = 1`, `T_k` growing), the query position is known to be `T_k - 1` and the mask is

```python
idx_q = torch.tensor([T_k - 1], device=device)
idx_k = torch.arange(T_k, device=device)
return (idx_q.unsqueeze(-1) - idx_k.unsqueeze(0) < window)
```

a single row that is `True` for the last `window` keys — a true band, matching (9) with $i = T_k - 1$. For square inputs the helper intersects its band with the causal mask (see §5.6 for the band's orientation).

### 5.4 `manual_causal_attention`: the FP32 oracle

`models/attention.py:manual_causal_attention` is the reference implementation for tests. Every score is computed in FP32 by explicit upcast:

```python
scores = (query_states.float() @ key_states.float().transpose(-2, -1)) / math.sqrt(D)
```

so the accumulation cannot be contaminated by BF16 rounding (the FP32 accumulation concern of §6). Causality uses a `triu` boolean fill, the window uses a masked fill with the same transposed orientation as §5.3, and the sink is appended as an actual score column:

```python
sink_logit = sink_bias.view(1, H, 1, 1).to(scores.dtype)
augmented = torch.cat([scores, sink_logit.expand(B, H, T, 1)], dim=-1)
attn_weights = F.softmax(augmented, dim=-1)
attn_weights = attn_weights[..., :T]
return (attn_weights.to(value_states.dtype) @ value_states)
```

The sink column is stripped before the value matmul (only real keys multiply $V$), exactly the "absorb, don't contribute" of (15). This is the oracle the equivalence tests compare SDPA against — the same math as §4.1, written out naively at $O(T^2)$.

### 5.5 `repeat_kv`: GQA broadcast without an explicit copy

`models/attention.py:repeat_kv` broadcasts $H_{\mathrm{kv}}$ heads to $H$:

```python
x = x[:, :, None, :, :]
x = x.expand(B, H_kv, n_rep, T, D)
return x.reshape(B, H_kv * n_rep, T, D)
```

`expand` is a stride-0 view (no data movement); `reshape` then merges the `(H_kv, n_rep)` dims into `(H,)`. The code never calls `.contiguous()` — and does not need to: `reshape` materializes a contiguous copy itself, because a stride-0 dimension cannot be merged without one. The result is therefore a real, contiguous `(B, 8, T, 96)` tensor (verified: `is_contiguous() == True`, no shared storage with the input), not a lazy view — but the copy is one pass over 4 heads' worth of data, whereas the alternative (projecting 8 KV heads) would cost double the projection FLOPs forever. `n_rep == 1` short-circuits to the input unchanged.

### 5.6 Verified behavioral notes (2026-08-04, by direct execution)

Two correctness bugs in the mask helpers were found and fixed on 2026-08-04 as part of the documentation-expansion audit (this chapter's first draft documented the buggy behavior; the fixes below are what the code does now). Both were silent: the 192-test baseline passed because the restriction tests were vacuous (Note 3).

**Note 1 — the square `_window_mask` band was vacuous under causality; fixed.** In the `T_q == T_k` branch the band condition was

```python
return (idx.unsqueeze(0) - idx.unsqueeze(1) < window) & _causal_mask(T_q, device, dtype)
```

Entry $(i, j)$ is $j - i$, so the condition was $j - i < W$; under causality $j \le i$ that is always true — the window added nothing, and windowed layers performed full causal attention during prefill. Fixed to $i - j < W$ (`idx.unsqueeze(1) - idx.unsqueeze(0)`): query $i$ now sees keys $\max(0, i - W + 1) \le j \le i$. `models/attention.py:manual_causal_attention` had the same transposed orientation and was likewise vacuous; fixed the same way. Verified post-fix: `_window_mask(T, T, 8) != _causal_mask(T)` (position 63 blocks 56 of 63 keys at $W=8$), SDPA-windowed ≡ manual-windowed at every position (allclose `1e-5`), and zeroing keys $j \le t - W$ leaves position $t$ unchanged. The decode branch was always a true last-$W$ band and is untouched; `inference/generate.py:MixedKVCache.append` still caps storage at $W$ tokens, and the KV-cache 2.00× metric is unaffected (a storage claim). With the fix, the banded-mask FLOP savings of (11) now materialize at prefill as well.

**Note 2 — the sink-path mask was additive `+1`/`0`, leaking future tokens; fixed.** The sink path previously wrote `mask[:, :, :T_k] = causal.to(dtype)` — `1.0` where allowed, `0.0` where forbidden — and SDPA *adds* float masks, so "forbidden" positions were not excluded and future tokens leaked into every position's softmax at prefill (verified: weight on future keys ≈ 0.77 for row 0 on random inputs). Fixed to `torch.where(causal, 0.0, -inf)`: allowed positions add `0`, blocked positions add `-∞`. Verified post-fix: sink-path SDPA ≡ the manual sink oracle at every position (allclose `1e-5`), and corrupting future keys leaves position 0 unchanged. Decode was unaffected in both versions (the cache contains no future keys; a uniform additive shift cancels in softmax).

**Note 3 — the SWA-restriction tests were vacuous at fixture sizes; fixed.** `tests/test_attention.py`'s fixtures produce `(B, H, T, D)` tensors (`tests/conftest.py` `_make_attn_inputs`), but the restriction tests unpacked them as `(B, T, H, D)`, so their `T` was the head count (4 or 8): the outside-the-window loop over `range(window, ...)` was empty and the inside-the-window loop visited at most 8 trivial positions. The unpacking was corrected and four regression tests were added: `test_sliding_window_sdpa_matches_manual`, `test_sliding_window_blocks_past_keys_at_prefill`, `test_sink_path_matches_manual_at_prefill`, `test_sink_path_is_causal`. `pytest tests/test_attention.py -v` now pins prefill windowing and sink-path causality directly (22 tests on CPU, 2026-08-04).

### 6. Pitfalls and verification

**FP32 accumulation in the oracle.** `models/attention.py:manual_causal_attention` upcasts to FP32 before the matmul; the equivalence tests feed it float64 inputs. This is deliberate — the oracle must be the *truth*, and BF16 scores would make "matches the reference" meaningless. The fused kernels also accumulate in FP32 internally but read BF16 inputs; when debugging an SDPA-vs-manual mismatch, first check dtype, then mask semantics, then backend.

**Mask arithmetic traps.** (a) Never substitute a large finite sentinel for $-\infty$ — §4.4's addition-overflow path turns it into NaN. (b) Boolean masks mean "allowed"; float masks are added verbatim — `causal.to(dtype)` would be `1.0`, not `0.0`, so blocked positions must be set to `-inf` explicitly (`torch.where(causal, 0.0, -inf)`, as the sink path now does). (c) `-inf + (-inf) = -inf` is safe; `+inf + (-inf) = NaN` is the overflow signature to grep for. The clamp `models/attention.py:SINK_CLAMP_MIN` / `models/attention.py:SINK_CLAMP_MAX` exists to keep the sink column out of the danger zone; the parameter retains its raw value for gradient flow.

**Regression guards.** A green `pytest tests/test_attention.py -v` now proves prefill windowing and sink-path causality (Notes 1–3). Any future change to the mask helpers must keep the four regression tests green — they exist precisely because the previous vacuous tests let two correctness bugs ship.

**Backend nondeterminism.** `F.scaled_dot_product_attention` chooses math, mem-efficient, or flash by heuristics; only the math backend is guaranteed. Do not assume a fused kernel ran — check `torch.backends.cuda` diagnostics — and keep Q, K, V in one dtype. Perf claims about flash on this repo's target hardware are `[INFERENCE]`; `.benchmarks/` is empty.

**Verification commands.** The guard for every equivalence claim in this chapter is

```
python3 -m pytest tests/test_attention.py -v
```

22 passed on CPU, 2026-08-04 (≈2.5 s; the 2 GPU-gated Triton skips live in `tests/test_moe_triton.py`, not here). The KV numbers cross-referenced in §4.5 are guarded by `python3 scripts/kv_cache_benchmark.py` (2.00× @128K, 1.94× @4K — measured). The passkey/quality bands are targets, not results, and are not guarded by any command yet.

## Part B — Positional Encodings

> **Theory chapter T2.** From-scratch derivation of position encoding: why attention needs it, the absolute sinusoidal construction, relative encodings, RoPE as a rotation, and the interpolation/extrapolation ladder (PI → NTK-aware → YaRN) that gets GPT-OSS-Lite from a 4,096-token training window to a 131,072-token evaluation window. Maps onto `models/rotary.py` and `models/yarn.py`. The implementation-focused companion is [rope_yarn](attention-and-positional.md) (worked numbers, dtype contract, SDPA interaction); this chapter supplies the derivations behind those numbers. Assumes the primer level of [foundations](foundations-and-architecture.md); the softmax arithmetic that consumes the encodings is derived in [attention math](attention-and-positional.md).

---

### 1. 60-second summary

Attention is a weighted average of value vectors; without position information the weights depend only on token content, so a transformer cannot distinguish "A B" from "B A". Every position-encoding scheme injects a sequence order signal, and the schemes form a ladder. Absolute sinusoidal encodings add a fixed vector of sines and cosines to each token embedding; the shift between two positions is a linear map (a rotation), which is the seed of everything after. Learned absolute embeddings replace the fixed table with parameters but extrapolate to nothing. Relative encodings parameterize attention scores by token distance, buying shift-invariance at the price of a bounded table or a fixed bias shape. RoPE achieves relative-position behavior with no table: it rotates each query and key in 2D subspaces, so the dot product between a query at position $p$ and a key at position $j$ depends only on $p - j$. RoPE fails beyond the trained length, so GPT-OSS-Lite applies YaRN: keep the fastest pairs' frequencies, divide the slowest pairs' frequencies by the scale factor 32, blend linearly between, and multiply every rotation by `mscale = 0.1·ln(32) + 1 ≈ 1.347` to restore attention sharpness. On global (full-attention) layers, the 24 fastest pairs are additionally frozen to identity (`cos=1, sin=0`) because at 131,072 tokens they have spun so many times that their phase is aliasing noise. The whole pipeline lives in `models/rotary.py:compute_yarn_freqs`, `models/rotary.py:compute_yarn_mscale`, `models/rotary.py:apply_rope`, and `models/yarn.py:YaRNRoPE`.

### 2. Why it matters here

Position handling is one of the few places where GPT-OSS-Lite's entire long-context story is decided by a handful of hyperparameters:

- **4K train, 128K eval.** Training runs at `max_seq_len = 4096` while
  evaluation targets `eval_max_seq_len = 131072` (`models/transformer.py:ModelConfig`) — a $32\times$ stretch. That stretch is the reason YaRN exists in this repo at all: plain RoPE extrapolation collapses far outside its training window (§4.6), and the target is 32× because the scale factor is literally the ratio of the two lengths (§4.7).
- **Alternating SWA/full.** Even layers are sliding-window ($W=128$), odd
  layers are full causal (`models/attention.py:GPTOSSAttention` decides via `layer_idx % 2`). The position-encoding treatment differs per branch: windowed layers keep all 48 frequency pairs; global layers prune the 24 fastest (§4.8). The mask half of this split is derived in
  [attention math](attention-and-positional.md).
- **GQA with `head_dim = 96`.** 8 query heads share 4 KV heads, so the
  rotary table is computed once per layer per sequence and broadcast across heads. `head_dim` must be even — the rotation lives in 2D subspaces — and `ModelConfig.__post_init__` enforces it.
- **Rotated K in the KV cache.** Keys are rotated *before* they enter
  `MixedKVCache`; at decode only the new query rotates. That is what makes the measured 2.00× KV reduction at 128K (1.94× at 4K, `scripts/kv_cache_benchmark.py`) compatible with RoPE at all — the cache math is in [kv cache engineering](../inference.md).
- **Budget honesty.** The ≥85% passkey @128K figure is a **target**; no
  pretraining run exists. The 2.00×/1.94× KV figures are **measured**. Everything in this chapter is derived arithmetic, tagged where it is a fitted constant or `[INFERENCE]`; no performance numbers are borrowed from `.benchmarks/` (it is empty).

### 3. Intuition

Picture each frequency pair of a position encoding as the hand of a clock with its own gear ratio. The pair $(\cos(p\omega_m), \sin(p\omega_m))$ is the hand's orientation after $p$ ticks. The fastest gears ($m$ small) complete thousands of turns over a 131,072-token sequence — they can tell you that token $p+3$ came after token $p$, but nothing about where in the sequence you are. The slowest gears barely move: after 4K tokens the slowest pair has rotated less than 4% of a turn, so its angle is a smooth, monotone "absolute position" meter. RoPE attaches one such geared hand to every 2D sub-vector of every query and key; the attention score between a query and a key compares the *angles between their hands*, which is why the score depends on the tick *difference* $p - j$ and not the absolute ticks.

Extending to 128K is a re-gearing problem. If you keep the original gear ratios, the slow hands sweep into angles never seen during 4K training (the network has no learned reading for them). Position Interpolation slows *all* gears by 32, which keeps every hand inside its trained arc but blurs the fast gears so badly that neighboring tokens become hard to tell apart. YaRN re-gears selectively: leave the fastest hands alone, divide the slowest hands' speeds by 32, blend in between — then stiffen the attention temperature (`mscale`) because compressed gears make every hand look alike, flattening the attention distribution. Finally, on the global layers, the fastest gears are disconnected entirely: by 128K they have spun so many times that their angle is effectively random noise.

### 4. Theory and derivation

### 4.1 Why positions at all: permutation invariance

Scaled dot-product attention (derived in [attention math](attention-and-positional.md)) computes, row by row, a convex combination of value rows:

$$
o_i = \sum_{j=1}^{T} \alpha_{ij}\, v_j, \qquad
\alpha_{ij} = \mathrm{softmax}_j\!\left(\frac{q_i^\top k_j}{\sqrt{d}}\right)
\tag{1}
$$

where $q_i = W_q x_i$, $k_j = W_k x_j$, $v_j = W_v x_j$ are linear maps of the input tokens $x_i \in \mathbb{R}^{d_{\text{model}}}$, and $d$ is the head width. Nothing in (1) references $i$ or $j$ except through the token contents $x_i, x_j$. If the input sequence is permuted by $\sigma$, the hidden states move with their tokens, and

$$
\mathrm{Attn}(x_{\sigma(1)}, \dots, x_{\sigma(T)}) =
\sigma\big(\mathrm{Attn}(x_1, \dots, x_T)\big),
\tag{2}
$$

i.e. the layer is *equivariant to permutation*: reordering the input merely reorders the output rows. The two sequences "the cat sat" and "sat cat the" produce the same multiset of row vectors, so no downstream layer can ever distinguish the orderings — the model would be a bag-of-tokens. Autoregressive language modeling is impossible without an explicit, learnable position signal injected somewhere in the forward pass. (Equation (2) holds for a single layer with no mask; the argument is unchanged by any mechanism that reads content only.) Everything in this chapter is a different way of injecting that signal.

### 4.2 Absolute sinusoidal encodings

The original Transformer adds a fixed, deterministic vector to each token embedding:

$$
\mathrm{PE}(p, 2m) = \sin(p\, \omega_m), \qquad
\mathrm{PE}(p, 2m+1) = \cos(p\, \omega_m), \qquad
\omega_m = \frac{1}{\theta^{2m/d}},
\tag{3}
$$

for position $p \in \{0, \dots, L-1\}$, pair index $m \in \{0, \dots, d/2-1\}$, base $\theta = 10000$, and head/model width $d$ (the construction is used with $d = d_{\text{model}}$). The input to the model becomes $x_p = e_{t_p} + \mathrm{PE}(p)$, where $e_{t_p}$ is the token embedding.

**Why geometric frequencies.** Pair $m$ has wavelength

$$
\lambda_m = \frac{2\pi}{\omega_m} = 2\pi\, \theta^{2m/d},
\tag{4}
$$

the number of positions per full rotation. The frequencies are *log-uniform*: $\omega_{m+1}/\omega_m = \theta^{-2/d}$ is constant, so every octave (factor-2 band) of wavelength contains exactly $d\cdot \ln 2 / (2 \ln \theta)$ pairs, regardless of scale. With $d = 768$ and $\theta = 10000$, wavelengths run from $\lambda_0 = 2\pi \approx 6.3$ tokens to $\lambda_{383} = 2\pi \cdot 10^{4\cdot 383/768} \approx 6.1\times 10^{4}$ tokens. A single sinusoid cannot resolve both "is token 3 vs 4" and "is token 3,000 vs 4,000" — those need phases that move quickly and slowly respectively — but a log-uniform bank of them covers every scale at once.

**Why a sin/cos pair.** One scalar $\sin(p\omega)$ cannot distinguish $p$ from $\pi/\omega - p$ and loses phase information at zero crossings. The pair $(\sin, \cos)$ is the unit phasor $e^{ip\omega}$, a point on the circle injective over one full period.

**Why it works: the shift is a linear map.** The decisive algebraic property is that moving a position by $k$ is a *fixed linear transformation* of its encoding. Using $\sin(a+b) = \sin a\cos b + \cos a\sin b$ and $\cos(a+b) = \cos a\cos b - \sin a\sin b$:

$$
\begin{pmatrix} \sin((p-k)\omega_m) \\ \cos((p-k)\omega_m) \end{pmatrix}
=
\underbrace{\begin{pmatrix} \cos(k\omega_m) & -\sin(k\omega_m) \\
\sin(k\omega_m) & \cos(k\omega_m) \end{pmatrix}}_{R(k\omega_m)}
\begin{pmatrix} \sin(p\omega_m) \\ \cos(p\omega_m) \end{pmatrix},
\tag{5}
$$

with the $2\times2$ rotation matrix $R(\phi)$ — the exact same matrix RoPE applies to queries and keys in §4.5. The set of encodings is therefore a *translation-structured* manifold: the encoding of every position $p+k$ is reachable from position $p$ by a rotation whose angle depends only on the offset $k$ (here $PE(p+k) = R(-k\omega)PE(p)$, using $R(-\phi) = R(\phi)^\top$). A linear layer can read "distance between two encodings" as "relative rotation angle", which is why relative structure is learnable at all — and it is the exact same rotation matrix that RoPE reuses on queries and keys in §4.5. The practical caveat, measured by Kazemnejad et al. (2023), is that this structure buys only *modest* extrapolation: sinusoidally-encoded models degrade gracefully a little past $L$ but not far.

### 4.3 Learned absolute embeddings

The alternative to a fixed table is to make the position vector trainable:

$$
x_p = e_{t_p} + E[p], \qquad E \in \mathbb{R}^{L_{\max} \times d}.
\tag{6}
$$

Each position gets an independent parameter vector. This is strictly more expressive than (3) — the model can learn whatever position geometry it wants — but it has no structure to exploit: there is no systematic relation between $E[p]$ and $E[p+1]$, and positions beyond $L_{\max}$ are undefined, so extrapolation is impossible by construction (the common hack is to clamp or train with longer windows). GPT-OSS-Lite uses no position embedding at all at the input level; its only position mechanism is RoPE inside attention, so `weight_tying` (`models/transformer.py:ModelConfig`) applies to a purely token-level embedding.

### 4.4 Relative encodings

The semantic role of a token is largely a function of its *distance* from the query: "the cat sat" and "a cat sat" have different absolute positions for "cat" but the same local structure. Absolute encodings force the model to learn a separate pattern for every absolute pair $(p, j)$; a relative scheme parameterizes the score by $\delta = j - i$ directly. The canonical forms are additive biases on the logit:

$$
s_{ij} = \frac{q_i^\top k_j}{\sqrt{d}} + b_{j-i},
\tag{7}
$$

with $b_\delta$ a learned bias table (Shaw et al. 2018; T5 buckets the offset and shares biases), or a fixed linear bias $b_\delta = -m|\delta|$ (ALiBi), which needs no table and extrapolates to any length — but imposes a fixed geometric prior the model cannot reshape. Relative encodings buy shift-invariance and length behavior at the cost of either a bounded table (offsets beyond the trained range must be clamped or bucketed) or a rigid bias shape. RoPE, next, gets relative behavior with no table and no length-dependent clamp.

### 4.5 RoPE: rotation, complex view, relative distance

Rotary Position Embedding (Su et al., 2021) applies position *inside* the attention score rather than at the input. Split the head dimension into $d/2$ pairs and rotate pair $m$ of the query at position $p$ and the key at position $j$ by the angles $p\omega_m$ and $j\omega_m$ respectively:

$$
q'_m = R(p\,\omega_m)\, q_m, \qquad k'_m = R(j\,\omega_m)\, k_m,
\qquad R(\phi) = \begin{pmatrix} \cos\phi & -\sin\phi \\
\sin\phi & \cos\phi \end{pmatrix}.
\tag{8}
$$

The frequencies $\omega_m = \theta^{-2m/d}$ are the same geometric grid as (3), and $R$ is the same matrix that appeared in (5). Because rotations preserve length, RoPE does not distort the query/key magnitudes.

**Complex view.** Identify each pair with a complex number, $(x_{2m}, x_{2m+1}) \leftrightarrow z_m = x_{2m} + i\,x_{2m+1}$. Rotation by $\phi$ is multiplication by the unit phasor $e^{i\phi}$, so RoPE is:

$$
z'_m = z_m\, e^{i p \omega_m} \qquad \text{(query)}, \qquad
w'_m = w_m\, e^{i j \omega_m} \qquad \text{(key)}.
\tag{9}
$$

The real inner product between two 2D pairs is the real part of the Hermitian inner product, so the score contribution of pair $m$ is:

$$
(q'_m)^\top k'_m = \operatorname{Re}\!\big[\, \bar z_m\, w_m\,
e^{i(p - j)\omega_m} \big].
\tag{10}
$$

Summing over pairs and writing the score with the usual $1/\sqrt{d}$ scaling gives the full attention logit:

$$
s_{ij} = \frac{1}{\sqrt{d}} \sum_{m=0}^{d/2-1}
\operatorname{Re}\!\big[\, \bar z_m w_m\, e^{i(p-j)\omega_m} \big],
\tag{11}
$$

which depends on the positions $p, j$ **only through the difference** $\delta = p - j$. Why this falls out: rotations compose ($e^{i a}e^{i b} = e^{i(a+b)}$) and are unitary, so

$$
(R(p\omega)\, q)^\top (R(j\omega)\, k) = q^\top R(p\omega)^\top R(j\omega)\, k
= q^\top R\big((j - p)\omega\big)\, k,
\tag{12}
$$

an orthogonal-matrix identity (the second equality uses $R(\phi)^\top = R(-\phi)$ and $R(a)R(b) = R(a+b)$). The relative offset $\delta$ is *built into the geometry* — no bias table, no bucketing, and the same frequency bank covers local syntax (fast pairs) and long-range structure (slow pairs) simultaneously. Position 0 rotates nothing ($R(0) = I$), so a query at the very first token is unmodified — a property the sink bias interacts with ([ATTENTION_SINKS](attention-sinks.md)).

A second practical consequence of (10)–(12): because a key's rotation depends only on its *own* absolute position, keys can be rotated once when they are produced and cached in rotated form; every future query recovers the relative offset automatically at score time. GPT-OSS-Lite relies on this exactly — `inference/generate.py:_attn_forward_layer` rotates `k_new` before inserting it into the cache, and only the new query is rotated at decode (§5.6).

### 4.6 Interpolation vs extrapolation

RoPE trained at length $L$ has seen only the phases $\phi_m(p) = p\omega_m$ for $p \in [0, L]$. At evaluation length $L' = sL$ there are two disjoint failure modes.

**Slow pairs extrapolate into unseen phases.** For a pair with wavelength $\lambda_m > L$ — the slow pairs — the trained phase range $[0, L\omega_m]$ is a *proper sub-arc* of the circle:

$$
\text{trained phase range: } [0, L\omega_m] \subset [0, 2\pi),
\qquad
\text{at } sL: [0, sL\omega_m].
\tag{13}
$$

The pair completes fewer than one rotation over the whole training sequence, so its phase is a monotone "absolute position" meter (§3); at $p > L$ the meter enters angles the network has never seen, and the learned map from phase to representation is undefined there. These are the pairs the YaRN paper calls absolute-position carriers — the model demonstrably uses them (they are why RoPE is not purely relative in practice).

**Fast pairs alias.** For a fast pair, $\phi_m(p) \bmod 2\pi$ repeats every $2\pi/\omega_m$ tokens, so absolute position is unreadable — but that was already true at training length, and every phase value is in-distribution, so fast pairs keep functioning as *relative* encoders. Their problem at extension is that they contribute nothing to the absolute-position signal that long-range attention needs; positions $p$ and $p + 2\pi/\omega_m$ produce identical phases. Frequency aliasing in the strict sense: $\cos (\phi) = \cos(\phi + 2\pi k)$, and for $\omega_0 = 1$ the period is $2\pi \approx 6.28$ tokens.

**Position Interpolation (PI)** (Chen et al., 2023) sidesteps the slow-pair failure by never leaving the trained phase envelope: evaluate the rotation at compressed positions, $f'_W(x_m, m, \theta) = f_W(x_m, mL/L', \theta)$, i.e. $\phi'_m(p) = p\omega_m/s$. Every phase stays inside $[0, L\omega_m]$. The cost: *all* frequencies drop by $s$, so adjacent tokens differ in phase by $\omega_m/s$ — the fastest pair moves $\sim 1/32$ rad per token instead of 1 rad, and fine-grained local ordering collapses ("loss of high-frequency information"; the NTK/Fourier-feature argument of Tancik et al., 2020). Empirically PI degrades beyond $s \approx 8$ even with fine-tuning.

**NTK-aware scaling** (bloc97, 2023) spreads the interpolation pressure across dimensions by changing the base instead of the positions. Choose the new base $\theta'$ so that the *slowest* pair is stretched by exactly $s$ (as PI would) while the *fastest* pair ($\omega_0 = 1$, base-independent) is untouched. The slowest pair has index $d/2 - 1$ (the pair with exponent $2(d/2 - 1)/d = (d-2)/d$), and requiring its wavelength to multiply by $s$:

$$
2\pi (\theta')^{(d-2)/d} = s \cdot 2\pi \theta^{(d-2)/d}
\quad\Longrightarrow\quad
\theta' = \theta \cdot s^{d/(d-2)}.
\tag{14}
$$

Every intermediate pair then scales by an intermediate factor, $\omega_m(\theta') = (\theta')^{-2m/d} = \omega_m(\theta)\cdot s^{-2m/(d-2)}$: high frequencies keep almost their original rate, low frequencies get almost full PI compression — no position-dependent gating, one hyperparameter $\theta'$. For GPT-OSS-Lite's geometry the new base is $\theta' = 10^{5} \cdot 32^{48/47} \approx 3.5\times 10^{6}$. The drawbacks (the paper's A.2): it is not a true interpolation — the fast pairs still extrapolate to slightly out-of-range phase values — and the right $\theta'$ for a target $s$ has to be found empirically. NTK-aware motivates the explicit per-pair policy that YaRN makes: decide, pair by pair, whether to keep, compress, or blend.

### 4.7 YaRN: the ramp, `mscale`, and why scale = 32

**Per-pair policy by rotation count.** Define the number of rotations pair $m$ completes over the training length:

$$
r_m = \frac{L}{\lambda_m} = \frac{L\, \omega_m}{2\pi}.
\tag{15}
$$

Large $r_m$ (fast pairs, $\lambda_m \ll L$) means the pair only ever encodes *relative* offsets — it must be left alone. Small $r_m$ (slow pairs, $\lambda_m \gtrsim L$) means the pair encodes *absolute* position — it must be compressed by $s$, never extrapolated. NTK-by-parts (bloc97, 2023; formalized in the YaRN paper) interpolates between a linear scale for $r_m < \alpha$ and no scaling for $r_m > \beta$ with a linear ramp in between:

$$
\gamma(r_m) = \mathrm{clamp}\!\left(\frac{r_m - \alpha}{\beta - \alpha},\,
0, 1\right), \qquad
\omega'_m = \omega_m\,(1 - \gamma_m) + \frac{\omega_m}{s}\,\gamma_m,
\tag{16}
$$

with recommended $\alpha = 1$, $\beta = 32$ (tuned on the Llama family). $\gamma_m = 0$ keeps the base frequency; $\gamma_m = 1$ applies full PI compression $\omega_m \to \omega_m/s$.

**The code's boundary closed form.** Instead of computing $r_m$ per pair, `models/rotary.py:compute_yarn_freqs` selects two dim indices directly:

$$
\mathrm{low} = \left\lfloor \frac{d/2}{\log_2\!\left(\frac{L}{\beta_{\mathrm{slow}}}\cdot \pi\right)} \right\rfloor, \qquad
\mathrm{high} = \left\lceil \frac{d/2}{\log_2\!\left(\frac{L}{\beta_{\mathrm{fast}}}\cdot \pi\right)} \right\rceil,
\tag{17}
$$

then ramps linearly in *dim index*: $\gamma_m = \mathrm{clamp}((m - \mathrm{low})/(\mathrm{high} - \mathrm{low}), 0, 1)$ and blends exactly as in (16), with $\beta_{\mathrm{slow}} = \alpha = 1$ and $\beta_{\mathrm{fast}} = \beta = 32$ as the defaults. Note the expression evaluates as $(L/\beta)\cdot\pi$ (division before multiplication). With the production values $L = 4096$, $d/2 = 48$, $\beta_{\mathrm{slow}} = 1$, $\beta_{\mathrm{fast}} = 32$:

$$
\log_2(4096\pi) = 13.65 \Rightarrow \mathrm{low} = \lfloor 48/13.65 \rfloor = 3,
\qquad
\log_2(4096\pi/32) = 8.65 \Rightarrow \mathrm{high} = \lceil 48/8.65 \rceil = 6.
\tag{18}
$$

So pairs $m \le 3$ keep their base frequencies ($\omega = 1.00, 0.79, 0.62, 0.49$), pairs $m \ge 6$ are divided by 32, and pairs 4–5 blend with $\gamma = 1/3, 2/3$. This closed form is the implementation lineage's boundary rule (the YaRN/jquesnelle → HF Transformers form); it is more aggressive than reading the paper's $\alpha,\beta$ as rotation-count thresholds directly — the r-threshold reading would put the ramp at dims $\approx 13$–27 for this geometry (derived from (15): $r_m = \beta = 32$ at $m = 48\cdot\ln(4096/64\pi)/\ln 10^5 \approx 12.6$; $r_m = \alpha = 1$ at $m \approx 27$). Both implement the same intent — preserve the fastest pairs, compress the slowest — and the boundary position is a hyperparameter of the implementation; the invariant that matters is that $\gamma$ is 0 below $\mathrm{low}$, 1 above $\mathrm{high}$, and linear between. A 32× stretch compresses 42 of 48 pairs; only the four fastest escape.

**`mscale`: the attention-temperature correction.** Compressing the slow pairs' frequencies makes their phase differences shrink by up to $s$, so the *contrast* of logits across keys drops (softmax is shift-invariant, so only contrast matters), the attention distribution flattens, and its entropy rises. YaRN compensates with a temperature $t$ on the logits (paper Eq. 14) and uses the "length scaling trick": scaling both $q$ and $k$ by $\sqrt{1/t}$ is equivalent to dividing the logits by $t$, and can be implemented by scaling the rotary embeddings alone. Fitting $\sqrt{1/t}$ against perplexity across LLaMA 7B–65B gives (paper Eq. 15):

$$
\sqrt{1/t} = \mathrm{mscale}(s) = 0.1\ln(s) + 1,
\qquad t = \frac{1}{\mathrm{mscale}(s)^2}.
\tag{19}
$$

The *form* $1 + c\ln s$ is what the log-uniform spectrum dictates: scaling by $s$ shifts every pair's effective frequency down by $s$, and the number of pairs whose phase behavior changes over the training span is proportional to the log-frequency width $\ln s$ (pairs are uniformly spaced in log-frequency, so a factor-$s$ shift crosses $\propto \ln s$ of them). The constant $c = 0.1$ is **fitted, not derived** — a fitted constant and the transferable observation that the entropy shift is roughly universal across models. GPT-OSS-Lite multiplies every `cos`/`sin` by `mscale` in `models/yarn.py:YaRNRoPE.forward`, which scales each of $q$ and $k$ by `mscale`, i.e. the logits by $\mathrm{mscale}^2 = 1/t$. For $s = 32$:

$$
\mathrm{mscale} = 0.1\ln 32 + 1 \approx 1.347, \qquad
\mathrm{mscale}^2 \approx 1.813, \qquad t \approx 0.552.
\tag{20}
$$

`models/rotary.py:compute_yarn_mscale` returns 1.0 for $s \le 1$, so plain RoPE is the $s=1$, zero-ramp limit of the whole mechanism.

**Why scale = 32.** The scale factor is defined as the ratio of the target to the original length:

$$
s = \frac{L'}{L} = \frac{131072}{4096} = 32,
\tag{21}
$$

and 32 is exactly the recipe the YaRN paper validates for 4k→128k extensions (its $s=32$ models were fine-tuned on 64k data and still extrapolated to 128k). `models/transformer.py:ModelConfig.__post_init__` enforces the sanity conditions: $s \ge 1$, and whenever $s > 1$ the original length must be strictly below the target.

### 4.8 Pruning on global layers: the over-rotation argument

At position $p$, pair $m$ completes

$$
\nu_m(p) = \frac{p\, \omega_m}{2\pi}
\tag{22}
$$

full turns. A pair's phase is *coherent* (usable for position) only while its total rotation over the span stays small — once $\nu_m \gg 1$, the phase $\cos(p\omega_m \bmod 2\pi)$ at any fixed query/key offset is effectively a pseudorandom function of the offset, oscillating with period $2\pi/\omega_m$ tokens and averaging to zero over a large key set: pure aliasing noise. The pair where $\nu = 1$ at 128K satisfies $\omega_m = 2\pi/131072 \approx 4.79\times10^{-5}$, which lands at $m \approx 27$ on the YaRN-scaled grid. Computed turn counts at $p = 131072$:

| pairs | turns $\nu$ at 128K | role |
|---|---|---|
| $m = 0$ (fastest, unscaled) | 20,861 | fully aliased |
| $m = 3$ (last unscaled) | 10,159 | fully aliased |
| $m = 23$ (last pruned) | 2.6 | aliased |
| $m = 24$ (first kept) | 2.1 | marginal |
| $m = 27$ | 1.0 | aliasing threshold |
| $m = 47$ (slowest) | 0.0083 | fully coherent |

Global (full-attention) layers score a query against *all* prior keys, so every aliased pair injects incoherent noise into the long-range logits — and the worst offenders are exactly the four unscaled pairs ($m = 0..3$), which YaRN deliberately preserves for local resolution that global layers do not need. GPT-OSS-Lite therefore freezes the 24 fastest pairs to identity on global layers: `models/attention.py:GPTOSSAttention._n_pruned_dims` returns `head_dim // 4 = 24` (in *pair* units) when the layer is not windowed and `yarn_prune_rope_global` is set, and `models/yarn.py:YaRNRoPE.forward` then overwrites the first 24 columns of `cos`/`sin` with 1.0/0.0. Since the frequency table is ordered fastest-first (§5.1), those are the pairs $m = 0..23$ — 24 pairs, i.e. 48 of the 96 scalar channels (half the head; "25%" is `head_dim // 4` expressed relative to `head_dim`). The pruning is applied *after* the `mscale` multiply, so a pruned pair contributes exactly $(\cos, \sin) = (1, 0)$: no rotation and no magnitude scaling.

Windowed layers keep all 48 pairs. They attend within 128 tokens, where pair $m$ still resolves offsets up to $\pi/\omega_m$ (phase below $\pi$): the compressed mid pairs resolve hundreds to tens of thousands of tokens — far more than the window — and the fast pairs carry the near-token order that sliding-window attention exists to provide. The 12-layer alternation (even = SWA, odd = full; [ATTENTION_SINKS](attention-sinks.md)) thus splits positional labor: windowed layers do fine-grained local positioning with full RoPE; global layers do long-range content matching using only the 32×-slowed, phase-coherent slow pairs. Because pruning is a pure function of layer parity, prefill and decode compute identical `cos`/`sin` for the same positions — required, since rotated keys are cached (§5.6).

### 5. Code walkthrough

### 5.1 `compute_yarn_freqs` — ramp math

`models/rotary.py:compute_yarn_freqs` builds the full YaRN frequency table in one pass:

```python
half = head_dim // 2
exponents = torch.arange(0, half, dtype=torch.float32) / half
base = 1.0 / (theta ** exponents)
```

`exponents` is $m / \mathrm{half}$, i.e. $2m/d$ (equivalent, since $\mathrm{half} = d/2$), so `base[m] = θ^{-2m/d} = ω_m` — the geometric grid of §4.2, in *descending* order (fastest pair first). The ramp bounds implement (17) verbatim:

```python
low = max(math.floor(half / math.log2(original_max_seq_len / beta_slow * math.pi)), 0)
high = min(math.ceil(half / math.log2(original_max_seq_len / beta_fast * math.pi)), half - 1)
```

Note the precedence: `original_max_seq_len / beta_slow * math.pi` is $(L/\beta)\cdot\pi$, not $L/(\beta\pi)$ — worth re-reading when comparing against other implementations, since the parenthesization shifts the ramp by several dims ([rope_yarn §10.2](attention-and-positional.md) reads it the other way). For production values the branch is:

```python
ramp = torch.clamp(
    (torch.arange(half, dtype=torch.float32) - low) / max(high - low, 1),
    0.0, 1.0,
)
inv_freq = base * (1.0 - ramp) + (base / scale_factor) * ramp
```

This is (16) exactly: $\gamma_m = 0$ for $m \le \mathrm{low}$ keeps `base`; $\gamma_m = 1$ for $m \ge \mathrm{high}$ yields `base / s`; between, a linear blend. If `high <= low` the ramp is degenerate — the function emits a `UserWarning` and falls back to `ramp = zeros`, i.e. identity RoPE with no length extension (guarded by `tests/test_yarn.py::test_compute_yarn_freqs_warns_on_degenerate_ramp`). The two `ValueError`s (odd `head_dim`, non-positive lengths) are the first line of defense for the even-`head_dim` invariant (§6). Note `target_seq_len` is accepted for API symmetry but unused: the extrapolation length is implicit in `scale_factor` and the ramp, not a separate clamp. The returned `inv_freq` has shape `(half,)` — `tests/test_yarn.py::test_yarn_freqs_shape` and `test_yarn_freqs_no_nan` (finite and positive at production scale) pin this.

### 5.2 `compute_yarn_mscale`

`models/rotary.py:compute_yarn_mscale` is (19) with the $s \le 1$ guard:

```python
if scale_factor <= 1.0:
    return 1.0
return 0.1 * math.log(scale_factor) + 1.0
```

`math.log` is the natural log, matching $\ln s$ in (19). `mscale` is a Python float, not a tensor — it is folded into `cos`/`sin` at forward time. `tests/test_yarn.py::test_yarn_mscale_basic` checks the $s=1$ identity and monotonicity ($s=32$ gives 1.347 > $s=4$ gives 1.139).

### 5.3 `YaRNRoPE` — construction

`models/yarn.py:YaRNRoPE.__init__` validates `head_dim % 2 == 0`, stores the hyperparameters, then calls `compute_yarn_freqs` and registers the result as a *non-persistent* buffer:

```python
self.register_buffer("inv_freq", inv_freq, persistent=False)
```

`persistent=False` means the table is not saved in `state_dict` — it is recomputed from hyperparameters at load time, so a checkpoint cannot carry a stale table. `self.mscale = compute_yarn_mscale(scale_factor)` when `mscale=True`, else `1.0`; the enabled flag is kept separately as `self.mscale_enabled` for `extra_repr`. One `YaRNRoPE` instance is created per attention layer (`models/attention.py:GPTOSSAttention.__init__`), each with identical hyperparameters — 48 floats per layer, negligible.

### 5.4 `YaRNRoPE.forward` — scalar fast path, outer product, pruning

`models/yarn.py:YaRNRoPE.forward(positions, n_pruned_dims=0)` computes $(\cos, \sin)$ tables of shape $(T, \mathrm{half})$:

```python
if positions.numel() == 1:
    inv_freq = self.inv_freq.to(positions.device)
    pos = positions.item() if positions.dim() == 0 else positions[0].item()
    freqs = inv_freq * float(pos)
    cos = freqs.cos().unsqueeze(0) * self.mscale
    sin = freqs.sin().unsqueeze(0) * self.mscale
else:
    freqs = torch.outer(positions.float(), self.inv_freq.to(positions.device))
    cos = freqs.cos() * self.mscale
    sin = freqs.sin() * self.mscale
```

The single-position branch is the decode fast path: it avoids materializing a $(1, \mathrm{half})$ outer product and, more importantly, avoids the `positions.float()` cast of a CPU int64 scalar. `inv_freq` is moved to `positions.device` but stays FP32; `cos`/`sin` are computed in FP32 and scaled by `mscale` — the phase is untouched, only the magnitude, which is exactly the length-scaling trick of §4.7. Position 0 therefore gives $\cos = \mathrm{mscale}$, $\sin = 0$ (`tests/test_yarn.py::test_yarn_module_zero_position_is_identity`), and $\cos^2 + \sin^2 = \mathrm{mscale}^2$ for every pair (`test_yarn_module_cos_sin_pair`). Pruning comes last:

```python
if n_pruned_dims > 0:
    cos = cos.clone()
    sin = sin.clone()
    cos[:, :n_pruned_dims] = 1.0
    sin[:, :n_pruned_dims] = 0.0
```

The `clone` keeps the non-pruned path free of in-place writes (safe for autograd) and the overwrite is exactly 1.0/0.0 — *after* the `mscale` multiply, so pruned pairs are pure identity (§4.8). The columns are the first `n_pruned_dims` of the fastest-first table, i.e. pairs $m = 0..n{-}1$. `models/yarn.py:YaRNRoPE.extra_repr` prints the full configuration (`head_dim, theta, scale_factor, original_max, target, mscale_enabled`), which is what appears in `repr(model)` for layer-level debugging.

### 5.5 `apply_rope` — repeat, rotate-half, dtype

`models/rotary.py:apply_rope(x, cos, sin)` implements the rotation (8) without ever constructing a complex tensor:

```python
T = x.size(-2)
half = x.size(-1) // 2

cos_full = cos.repeat_interleave(2, dim=-1).to(x.dtype)
sin_full = sin.repeat_interleave(2, dim=-1).to(x.dtype)
```

(`T`, the position length, is read but not used further; the broadcast below realigns `cos`/`sin` with the position axis.) Then:

```python
x_pairs = x.unflatten(-1, (-1, 2))
x_swapped = x_pairs.flip(-1)
x_swapped[..., 0] = -x_swapped[..., 0]
x_rotated = x_swapped.flatten(-2)
```

`cos`/`sin` carry one value per *pair*; `repeat_interleave(2)` fans each pair's value onto its two scalar channels $(2m, 2m+1)$, so `cos_full[..., 2m] == cos_full[..., 2m+1]`. The `unflatten`/`flip`/`negate` dance computes the "rotate-by-90°" twin: per pair, $(x_0, x_1) \to (-x_1, x_0)$, which is $R(\pi/2)\,x$ — the complex-multiply identity

$$
x \cdot e^{i\phi} = x\cos\phi + R(\pi/2)\,x \sin\phi,
\qquad
R(\pi/2)\begin{pmatrix}x_0\\ x_1\end{pmatrix} =
\begin{pmatrix}-x_1\\ x_0\end{pmatrix}.
\tag{23}
$$

Then:

```python
while cos_full.dim() < x.dim():
    cos_full = cos_full.unsqueeze(0)
    sin_full = sin_full.unsqueeze(0)

return x * cos_full + x_rotated * sin_full
```

The `while` broadcasts the $(T, d)$ tables over batch and head dimensions (positions sit at `x.size(-2)`), and the final line is (23) applied to every pair simultaneously. The `.to(x.dtype)` casts on `cos_full`/`sin_full` are the dtype contract: if `x` is BF16 and `cos` were FP32, the multiply would promote activations to FP32 and break `F.scaled_dot_product_attention`'s requirement that Q/K/V share a dtype. Rotation is norm-preserving when $\cos^2 + \sin^2 = 1$ (per-pair magnitudes unchanged — `tests/test_yarn.py::test_apply_rope_magnitude_preserved`); with `mscale` applied the preserved quantity is $\mathrm{mscale}^2$ instead, and `test_apply_rope_zero_rotation` pins the $\cos{=}1, \sin{=}0$ identity case that pruning produces.

### 5.6 Call sites: prefill vs decode, rotated K

`models/attention.py:GPTOSSAttention.forward` computes

```python
cos, sin = self.yarn(positions, n_pruned_dims=self._n_pruned_dims())
query_states = apply_rope(query_states, cos, sin)
key_states = apply_rope(key_states, cos, sin)
```

with `positions = torch.arange(T)` by default (prefill). At decode, `inference/generate.py:generate` passes `positions_step = torch.tensor([cur_pos - 1])` — a single-element tensor, which selects the scalar fast path of §5.4 — and `inference/generate.py:_attn_forward_layer` rotates the fresh key `k_new` *before* `MixedKVCache.append` stores it. Cached keys are thus stored pre-rotated; only the incoming query is rotated each step, and the relative offset is recovered inside the attention score by (12). Because `models/attention.py:GPTOSSAttention._n_pruned_dims` depends only on layer parity and the `yarn_prune_rope_global` flag, prefill and decode always agree on which pairs are frozen — a mismatch here would corrupt every cached key.

### 6. Pitfalls and verification

| Failure mode | Symptom | Guard |
|---|---|---|
| Odd `head_dim` | `ValueError: head_dim must be even` | `models/rotary.py:compute_yarn_freqs`, `models/yarn.py:YaRNRoPE.__init__`, `models/transformer.py:ModelConfig.__post_init__` — the rotation pairs (8) need $d$ even; 96 is fine |
| Degenerate ramp (`high <= low`) | `UserWarning: YaRN ramp degenerate` + identity fallback (no extrapolation) | `tests/test_yarn.py::test_compute_yarn_freqs_warns_on_degenerate_ramp`; fix `beta_fast`/`beta_slow` or lengths; `test_compute_yarn_freqs_no_warning_for_normal_params` pins the healthy path |
| Misread parenthesization in (17) | Ramp shifted by several dims vs other implementations | The code evaluates $(L/\beta)\cdot\pi$; `log2(4096π)=13.65` → `low=3`, `high=6` (not 4/9) — re-derive from `models/rotary.py:compute_yarn_freqs` directly |
| Mixed-dtype SDPA | `F.scaled_dot_product_attention` failure | `apply_rope` casts `cos`/`sin` to `x.dtype` before the multiply; `tests/test_yarn.py::test_apply_rope_*` + attention tests |
| Pruning vs `mscale` ordering | Pruned pairs scaled instead of identity | Pruning overwrites *after* the `mscale` multiply → exactly $(1, 0)$; `tests/test_yarn.py::test_yarn_module_pruned_dims` asserts the exact values |
| `mscale` correctness | Wrong temperature at extension | `cos^2+sin^2 = mscale^2` per pair, `mscale = 0.1·ln(32)+1 ≈ 1.347`; `tests/test_yarn.py::test_yarn_module_cos_sin_pair`, `test_yarn_mscale_basic` |
| Config inconsistencies | Silent wrong stretch | `ModelConfig.__post_init__`: $s \ge 1$; if $s > 1$ then original < target; positive lengths |
| Cache/table mismatch | Corrupt keys after decode step | `inv_freq` is a `persistent=False` buffer (recomputed at load); pruning is a function of layer parity only |

The single command that guards this entire chapter's arithmetic is

```bash
python3 -m pytest tests/test_yarn.py -v
```

It covers table shape/finiteness, the low/high spread (fastest pair unchanged at 1.0, slowest pair divided by ~32), `mscale` identity and monotonicity, the 4K-vs-128K distinctness of the rotation tables, position-0 identity, the $\cos^2+\sin^2$ invariant, monotone rotation of the fast dims, pruning, the degenerate-ramp warning, and all three `apply_rope` contracts (identity, shape, per-pair magnitude). Beyond it, the attention integration is exercised by the mask/regression tests referenced in
[attention math](attention-and-positional.md) and [ATTENTION_SINKS](attention-sinks.md).
No pretraining run exists yet, so the ≥85% passkey @128K figure remains a **target**, not a result; what is verifiable today is the arithmetic above and the measured 2.00×/1.94× KV reduction ([kv cache engineering](../inference.md), `scripts/kv_cache_benchmark.py`).

### Related documentation

- [rope_yarn](attention-and-positional.md) — implementation-focused companion: worked
  numerical examples, dtype/SDPA contract, debugging table, invariants.
- [attention math](attention-and-positional.md) — the softmax/mask arithmetic that
  consumes `cos`/`sin`.
- [ATTENTION_SINKS](attention-sinks.md) — sink bias, windowed vs global
  split, interaction with pruned RoPE.
- [kv cache engineering](../inference.md) — rotated-K caching and
  the measured KV reduction.
- [foundations](foundations-and-architecture.md) — primer: attention, GQA, SWA.

## Part C — RoPE and YaRN

> Purpose: end-to-end position encoding from pairwise RoPE geometry through YaRN extrapolation and pruned RoPE on global layers. Sources: `models/rotary.py`, `models/yarn.py`. Attention consumer: [attention-sinks.md](attention-sinks.md#part-b--implementation-modelsattentionpy).

---

### 1. Purpose and mental model

Rotary Position Embedding (RoPE; Su et al., 2021) encodes token position by **rotating** query and key vectors in two-dimensional subspaces. Attention score \(q_i^\top k_j\) depends on relative position \(i - j\) — a natural fit for causal autoregressive models.

GPT-OSS-Lite trains at **4,096 tokens** (`yarn_original_max_seq_len`) but targets **131,072 tokens** (`yarn_target_seq_len`) at inference — a 32× stretch.

Standard RoPE encodes position by rotating Q/K pairs at fixed frequencies derived from base \(\theta\). When sequences exceed the training length, high-frequency components complete many cycles per token — relative positions become ambiguous and attention quality degrades ("positional collapse").

**YaRN** (Peng et al., 2023) interpolates between:

- **High frequencies** (short wavelength): unchanged — preserve local structure.
- **Low frequencies** (long wavelength): scaled by factor \(s\) — stretch far positions.

GPT-OSS-Lite implements YaRN via precomputed `inv_freq` buffers and optional **pruned RoPE** on global-attention layers.

| Function / class | File | Role |
|------------------|------|------|
| `models/rotary.py:apply_rope` | `rotary.py` | Apply rotation to Q/K tensors |
| `models/rotary.py:compute_yarn_freqs` | `rotary.py` | Build YaRN-scaled inverse frequencies |
| `models/rotary.py:compute_yarn_mscale` | `rotary.py` | Attention temperature correction |
| `models/yarn.py:YaRNRoPE` | `yarn.py` | Module wrapping freq table + forward |

Standard RoPE is the `scale_factor=1`, zero-ramp limit of YaRN.

### Comparison with absolute PE

| Property | Absolute sinusoidal PE | RoPE |
|----------|------------------------|------|
| Applied to | Input embeddings | Q and K only |
| Position in score | Absolute | Relative |
| KV cache | Must store position offset | Rotate Q at decode; K pre-rotated |
| Extrapolation | Poor beyond train length | YaRN extends |

GPT-OSS caches **rotated** K in `MixedKVCache` — see
[attention-sinks.md §B.8](attention-sinks.md#b8-forward-path-trace-positions--out-proj).

---

### 2. RoPE geometry and `apply_rope`

```python
def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
```

Applies rotary embeddings to tensor `x` using precomputed `cos` and `sin`.

### 2.1 Contract

| Argument | Typical shape | Description |
|----------|---------------|-------------|
| `x` | `(B, H, T, D)` | Queries or keys (head-major) |
| `cos` | `(T, D/2)` | Cosine of rotation angles per pair |
| `sin` | `(T, D/2)` | Sine of rotation angles per pair |
| return | same as `x` | Rotated tensor, **same dtype as `x`** |

### 2.2 Pairwise rotation geometry

Head dimension \(D\) splits into \(D/2\) independent 2D rotations. Each pair \((x_{2m}, x_{2m+1})\) rotates in its own plane at frequency \(\omega_m\).

RoPE on queries at position \(i\) and keys at position \(j\):

\[
(R_{\theta_i} q)^\top (R_{\theta_j} k) = q^\top R_{\theta_i}^\top R_{\theta_j} k
= q^\top R_{\theta_j - \theta_i} k
\]

Attention scores depend on **relative** offset \(i - j\), not absolute positions individually.

### 2.3 Pair rotation via `unflatten` / `flip`

```python
x_pairs = x.unflatten(-1, (-1, 2))       # (..., D/2, 2)
x_swapped = x_pairs.flip(-1)              # swap pair elements
x_swapped[..., 0] = -x_swapped[..., 0]   # negate first of swapped
x_rotated = x_swapped.flatten(-2)
```

This implements the complex multiply formulation without explicit complex tensors:

\[
\begin{pmatrix} x'_0 \\ x'_1 \end{pmatrix}
=
\begin{pmatrix} \cos\theta & -\sin\theta \\ \sin\theta & \cos\theta \end{pmatrix}
\begin{pmatrix} x_0 \\ x_1 \end{pmatrix}
\]

Equivalent to:

\[
x' = x \odot \cos + x_{\text{rotate}} \odot \sin
\]

where \(x_{\text{rotate}}\) is the "rotate-half" transform: \((-x_1, x_0)\) per pair.

### 2.4 `repeat_interleave(2)`

`cos`/`sin` from `YaRNRoPE` have shape `(T, D/2)` — one value per pair. RoPE needs one value per scalar dimension:

```python
cos_full = cos.repeat_interleave(2, dim=-1)  # (T, D)
```

Pair \(m\) angles apply to both \(x_{2m}\) and \(x_{2m+1}\).

### 2.5 Broadcast over batch and heads

```python
while cos_full.dim() < x.dim():
    cos_full = cos_full.unsqueeze(0)
    sin_full = sin_full.unsqueeze(0)
```

`cos`/`sin` start as `(T, D)` after `repeat_interleave`; unsqueeze prepends dims until rank matches `x` (typically 4D). Position index aligns with `x.size(-2)`.

### 2.6 Final combine

```python
return x * cos_full + x_rotated * sin_full
```

No in-place ops — safe for autograd.

### 2.7 Broadcasting rules

`apply_rope` alignment requirements:

| `x` dim | `cos`/`sin` dim after unsqueeze |
|---------|--------------------------------|
| `(B, H, T, D)` | `(1, 1, T, D)` |

Position dimension `-2` of `x` must equal `T` in `cos`/`sin`.

Batch and head broadcast freely — same cos/sin table shared across all heads in a layer (head-agnostic positional encoding).

---

### 3. Frequency bases

### 3.1 Standard RoPE inverse frequencies

For base \(\theta\) (GPT-OSS: `rope_theta = 100000`):

\[
\text{inv\_freq}_m = \theta^{-2m/D}, \quad m \in \{0, \ldots, D/2 - 1\}
\]

In code (`compute_yarn_freqs`):

```python
exponents = torch.arange(0, half, dtype=torch.float32) / half
base = 1.0 / (theta ** exponents)
```

Note: exponent uses `m/half` not `2m/D` — equivalent because `half = D/2`.

### 3.2 Wavelength interpretation

Wavelength at pair \(m\) for position increment 1:

\[
\lambda_m = \frac{2\pi}{\omega_m}
\]

Low \(m\) → high frequency → short wavelength → local positional sensitivity. High \(m\) → low frequency → long wavelength → global positional sensitivity.

### 3.3 Position 0

At \(p = 0\): \(\theta_{0,m} = 0\) → \(\cos=1, \sin=0\) → identity rotation. Position 0 is unmodified — useful for sink-adjacent behaviour.

---

### 4. YaRN theory (ramp, blend, mscale)

### 4.1 Frequency blending

YaRN defines a ramp \(\gamma(m) \in [0, 1]\) over dimension index \(m\):

\[
\omega^{\text{YaRN}}_m = \omega^{\text{base}}_m \cdot (1 - \gamma_m) + \frac{\omega^{\text{base}}_m}{s} \cdot \gamma_m
\]

where \(s\) is `scale_factor` (default 32).

- \(\gamma_m = 0\): original frequency (local / high-freq dims).
- \(\gamma_m = 1\): frequency divided by \(s\) (extrapolation / low-freq dims).

### 4.2 Ramp boundaries

The ramp transitions between dimension indices `low` and `high`, derived from `original_max_seq_len`, `beta_fast`, and `beta_slow`:

\[
\text{low} = \left\lfloor \frac{d/2}{\log_2\!\left(\frac{L_{\text{orig}}}{\beta_{\text{slow}} \cdot \pi}\right)} \right\rfloor
\]

\[
\text{high} = \left\lceil \frac{d/2}{\log_2\!\left(\frac{L_{\text{orig}}}{\beta_{\text{fast}} \cdot \pi}\right)} \right\rceil
\]

Linear interpolation between `low` and `high`:

\[
\gamma_m = \mathrm{clamp}\!\left(\frac{m - \text{low}}{\text{high} - \text{low}},\; 0,\; 1\right)
\]

### 4.3 Attention temperature scaling (mscale)

YaRN optionally scales \(\cos/\sin\) by factor:

\[
\text{mscale} = 0.1 \cdot \ln(s) + 1
\]

This compensates for attention entropy change when frequencies are stretched. Implemented in `compute_yarn_mscale`.

---

### 5. Production parameters (θ=100K, scale=32, target=131072)

From `ModelConfig` defaults (`models/transformer.py`):

| Parameter | Default | Role |
|-----------|---------|------|
| `rope_theta` | 100000 | RoPE base \(\theta\) |
| `yarn_scale_factor` | 32 | Stretch factor \(s\) |
| `yarn_original_max_seq_len` | 4096 | Training context \(L_{\text{orig}}\) |
| `yarn_target_seq_len` | 131072 | Inference target (128K) |
| `yarn_beta_fast` | 32 | Fast ramp boundary control |
| `yarn_beta_slow` | 1 | Slow ramp boundary control |
| `yarn_mscale` | `True` | Enable mscale multiplier |
| `yarn_prune_rope_global` | `True` | Prune 25% dims on global layers |
| `head_dim` | 96 | Must be even |

With `head_dim=96`: 48 frequency pairs (`half = 48`).

---

### 6. `compute_yarn_freqs` / `compute_yarn_mscale`

Full implementation in `models/rotary.py`. Returns `inv_freq` tensor of shape `(head_dim // 2,)`.

```python
def compute_yarn_freqs(
    head_dim: int,
    theta: float,
    scale_factor: float,
    original_max_seq_len: int,
    target_seq_len: int,
    beta_fast: float = 32.0,
    beta_slow: float = 1.0,
) -> torch.Tensor:  # shape (head_dim // 2,)
```

### 6.1 Algorithm summary

1. Compute base RoPE frequencies `base`.
2. Compute ramp boundaries `low`, `high` from `original_max_seq_len`, `beta_fast`, `beta_slow`.
3. Build linear ramp \(\gamma_m\) from `low` to `high`.
4. Blend: `inv_freq = base * (1 - ramp) + (base / scale_factor) * ramp`.

### 6.2 Base inverse frequencies

```python
half = head_dim // 2
exponents = torch.arange(0, half, dtype=torch.float32) / half
base = 1.0 / (theta ** exponents)
```

Equivalent to \(\omega_m = \theta^{-2m/d}\).

Note: `target_seq_len` is accepted for API symmetry but **not used** in the frequency formula — extrapolation length is implicit in the chosen `scale_factor` and ramp, not a separate clamp.

### 6.3 Ramp indices

```python
low = max(math.floor(half / math.log2(original_max_seq_len / beta_slow * math.pi)), 0)
high = min(math.ceil(half / math.log2(original_max_seq_len / beta_fast * math.pi)), half - 1)
```

### 6.4 Ramp vector and blend

```python
if high <= low:
    warnings.warn("YaRN ramp degenerate: ...", UserWarning)
    ramp = torch.zeros(half, dtype=torch.float32)
else:
    ramp = torch.clamp(
        (torch.arange(half, dtype=torch.float32) - low) / max(high - low, 1),
        0.0, 1.0,
    )

inv_freq = base * (1.0 - ramp) + (base / scale_factor) * ramp
return inv_freq
```

Returned tensor registered as `YaRNRoPE.inv_freq` buffer.

### 6.5 Validation

```python
if head_dim % 2 != 0:
    raise ValueError(f"head_dim must be even, got {head_dim}")
if original_max_seq_len <= 0 or target_seq_len <= 0:
    raise ValueError(...)
```

### 6.6 `compute_yarn_mscale`

```python
def compute_yarn_mscale(scale_factor: float) -> float:
    if scale_factor <= 1.0:
        return 1.0
    return 0.1 * math.log(scale_factor) + 1.0
```

Multiplies all cos/sin values in `YaRNRoPE.forward`. Compensates for attention logit magnitude change when frequencies are compressed.

For `scale_factor = 32`:

\[
\text{mscale} = 0.1 \cdot \ln(32) + 1 \approx 0.1 \cdot 3.466 + 1 \approx 1.347
\]

Applied uniformly to all `cos` and `sin` values after computation. When `yarn_mscale=False` in config, `self.mscale = 1.0`.

---

### 7. `YaRNRoPE` module

```python
class YaRNRoPE(nn.Module):
    def __init__(self, head_dim, theta=100000.0, scale_factor=32.0, ...):
```

### 7.1 Construction

```python
inv_freq = compute_yarn_freqs(...)
self.register_buffer("inv_freq", inv_freq, persistent=False)

if mscale:
    self.mscale = compute_yarn_mscale(scale_factor)
else:
    self.mscale = 1.0
```

- `persistent=False`: not saved in `state_dict` — recomputed from hyperparameters.
- `mscale_enabled` stored separately from scalar `self.mscale`.

### 7.2 One module per attention layer

Each `GPTOSSAttention` instantiates its own `YaRNRoPE` with identical hyperparameters. Buffers are duplicated per layer (small memory — 48 floats each).

### 7.3 End-to-end data flow

```
compute_yarn_freqs()  ──► inv_freq buffer (48,)
        │
        ▼
YaRNRoPE.forward(positions, n_pruned_dims)
        │
        ├─ freqs = outer(positions, inv_freq)
        ├─ cos, sin = freqs.cos/sin * mscale
        ├─ [optional] prune first n_pruned_dims pairs
        │
        ▼
apply_rope(Q or K, cos, sin)
```

### 7.4 Forward pass — single position (decode)

```python
if positions.numel() == 1:
    inv_freq = self.inv_freq.to(positions.device)
    pos = positions.item() if positions.dim() == 0 else positions[0].item()
    freqs = inv_freq * float(pos)
    cos = freqs.cos().unsqueeze(0) * self.mscale
    sin = freqs.sin().unsqueeze(0) * self.mscale
```

Output shapes: `(1, half)`.

### 7.5 Forward pass — multiple positions (prefill)

```python
freqs = torch.outer(positions.float(), self.inv_freq.to(positions.device))
cos = freqs.cos() * self.mscale
sin = freqs.sin() * self.mscale
```

Output shapes: `(T, half)` where `T = len(positions)`.

### 7.6 Consumption in attention

```python
cos, sin = self.yarn(positions, n_pruned_dims=self._n_pruned_dims())
query_states = apply_rope(query_states, cos, sin)
key_states = apply_rope(key_states, cos, sin)
```

`cos`/`sin` broadcast over batch and head dimensions inside `apply_rope`.

### 7.7 Training vs inference positions

- **Prefill:** `positions = torch.arange(T)` → cos/sin shape `(T, half)`.
- **Decode:** `positions = torch.tensor([cur_pos - 1])` → shape `(1, half)`.

Same `inv_freq` table; only position values change.

---

### 8. Pruned RoPE on global layers (25% of dims)

### 8.1 Motivation

The `cos`/`sin` tables are ordered **fastest-first**: pair \(m\) has angular frequency \(\omega_m = \theta^{-m/\text{half}}\) (with YaRN blending), so \(m=0\) rotates fastest. At 128K the fastest pairs complete thousands of full rotations — \(\omega_0 = 1.0\) rad/token means ~20,861 turns at position 131072 — and a rotation of \(\phi\) is indistinguishable from \(\phi + 2\pi k\), so those channels **over-rotate and alias**. On **global** (full-attention) layers, freezing the fastest 25% of frequency pairs (\(m = 0 \ldots 23\) for `head_dim=96`, 48 of 96 scalar channels) to identity removes the aliasing channels while preserving the slow pairs that carry long-range position. Verified 2026-08-04 against `models/rotary.py:compute_yarn_freqs` + `models/yarn.py:YaRNRoPE.forward` (`inv_freq[0] = 1.0`, `inv_freq[47] ≈ 4.0e-7`; `cos[:, :24]` set to 1).

### 8.2 Selection rule

In `GPTOSSAttention._n_pruned_dims()`:

```python
if (not self.is_windowed) and self.prune_rope_global:
    return self.head_dim // 4   # pairs, not scalars
return 0
```

| Layer | `is_windowed` | `n_pruned_dims` |
|-------|---------------|-----------------|
| 0, 2, 4, 6, 8, 10 | `True` | 0 |
| 1, 3, 5, 7, 9, 11 | `False` | 24 (for `head_dim=96`) |

### 8.3 Application in `YaRNRoPE.forward`

```python
if n_pruned_dims > 0:
    cos = cos.clone()
    sin = sin.clone()
    cos[:, :n_pruned_dims] = 1.0
    sin[:, :n_pruned_dims] = 0.0
```

For the first `n_pruned_dims` **frequency pairs** (lowest indices \(m=0\ldots\)), i.e. the **fastest-rotating** channels:

- `cos → 1`, `sin → 0` → identity rotation (no positional encoding on those pairs).
- Remaining pairs use full YaRN-scaled rotation.

This is applied **after** mscale multiplication.

### 8.4 Interaction with `apply_rope`

`apply_rope` repeats each pair's cos/sin across the two scalars via `repeat_interleave(2, dim=-1)`. Pruning 24 pairs affects 48 of 96 head dimensions.

When `n_pruned_dims > 0` (global layers only), the fastest-rotating pairs become identity — equivalent to not rotating those subspaces. `apply_rope` is unaware of pruning; it receives modified cos/sin.

### 8.5 Interaction with sliding-window layers

| Layer type | YaRN | Pruning | Visible context |
|------------|------|---------|-----------------|
| Windowed (even) | Full YaRN table | None | Last 128 tokens |
| Global (odd) | Full YaRN table | First 24 pairs → identity | All prior tokens |

Windowed layers see only local context — they rely on **full** RoPE (no pruning) for fine-grained relative position within the 128-token window.

Global layers carry long-range dependencies — pruning the fastest pairs removes channels that would otherwise over-rotate (alias) across 131K positions.

YaRN and sink bias are orthogonal — see
[attention-sinks.md §9](attention-sinks.md#9-interaction-with-yarn-and-pruned-rope).

---

### 9. Dtype / SDPA contract

### 9.1 Dtype preservation in `apply_rope`

```python
cos_full = cos.repeat_interleave(2, dim=-1).to(x.dtype)
sin_full = sin.repeat_interleave(2, dim=-1).to(x.dtype)
```

`cos`/`sin` are computed in FP32 inside `YaRNRoPE`. Casting to `x.dtype` before multiply prevents implicit promotion to FP32, which would:

1. Break `torch.compile` fusion patterns.
2. Violate SDPA's requirement that Q, K, V share dtype.

### 9.2 Why BF16 matters

GPT-OSS trains in BF16. If `apply_rope` promoted Q/K to FP32:

```python
# BAD — would break SDPA dtype contract
return (x.float() * cos + x_rotated.float() * sin).to(x.dtype)
```

SDPA requires `query_states.dtype == key_states.dtype == value_states.dtype`.

### 9.3 cos/sin precision

Frequencies computed in FP32; cos/sin in FP32; cast to BF16 at multiply time. Sufficient precision for angles up to 128K positions with YaRN scaling.

---

### 10. Worked numerical examples

### 10.1 Small RoPE rotation (`D=4`, position 1)

**Setup:** `D=4` (2 pairs), `B=1`, `H=1`, `T=1`, position `p=1`, \(\theta=100000\).

Frequencies:

\[
\omega_0 = 1.0, \quad \omega_1 = 100000^{-1/2} \approx 0.00316
\]

Angles at position 1:

\[
\theta_0 = 1.0 \text{ rad}, \quad \theta_1 \approx 0.00316 \text{ rad}
\]

cos/sin (before mscale):

\[
\cos_0 \approx 0.540, \quad \sin_0 \approx 0.841
\]

Rotation of pair 0 for \((x_0, x_1)\):

\[
x'_0 = x_0 \cos\theta_0 - x_1 \sin\theta_0
\]
\[
x'_1 = x_0 \sin\theta_0 + x_1 \cos\theta_0
\]

`apply_rope` computes this via the `unflatten`/`flip` path — numerically equivalent.

### 10.2 YaRN ramp boundaries (defaults)

With `half=48`, `L_orig=4096`, `beta_slow=1`, `beta_fast=32`, the code evaluates `L_orig / beta * math.pi` (division first, then multiply):

\[
\log_2\!\left(\frac{4096}{1} \cdot \pi\right) \approx 13.65 \quad\Rightarrow\quad
\text{low} = \lfloor 48 / 13.65 \rfloor = 3
\]

\[
\log_2\!\left(\frac{4096}{32} \cdot \pi\right) \approx 8.65 \quad\Rightarrow\quad
\text{high} = \lceil 48 / 8.65 \rceil = 6
\]

Ramp transitions from \(m=3\) to \(m=6\). Dimensions 0–2 keep base freq; 7–47 use scaled freq; 3–6 blend. Verified against `models/rotary.py:compute_yarn_freqs` with the production config (2026-08-04).

### 10.3 Frequency at dimension 0

\[
\omega_0 = 100000^{-0/96} = 1.0
\]

With \(\gamma_0 = 0\): \(\omega^{\text{YaRN}}_0 = 1.0\).

### 10.4 Frequency at dimension 47

\[
\omega_{47} \approx 100000^{-47/48} \approx 1.58 \times 10^{-5}
\]

With \(\gamma_{47} = 1\): \(\omega^{\text{YaRN}}_{47} \approx \omega_{47} / 32\).

### 10.5 Position 131072 on pair 47

\[
\theta = 131072 \times \omega^{\text{YaRN}}_{47}
\]

Without YaRN scaling, this pair would complete \(\sim 2\) full rotations over 128K — with scaling, \(\sim 0.06\) rotations — much slower positional aliasing.

---

### 11. Degenerate ramp warning

When `high <= low`, the ramp cannot be constructed — typically from misconfigured `beta_fast`/`beta_slow` or very small `original_max_seq_len`.

```python
warnings.warn(
    f"YaRN ramp degenerate: low={low}, high={high} (head_dim={head_dim}, "
    f"original_max={original_max_seq_len}, beta_fast={beta_fast}, beta_slow={beta_slow}). "
    f"Falling back to identity (no length extrapolation). Check beta_fast/beta_slow.",
    UserWarning,
    stacklevel=2,
)
ramp = torch.zeros(half, dtype=torch.float32)
```

Effect: `inv_freq = base` — plain RoPE with no YaRN scaling. Long-context quality will suffer. This emits `UserWarning`, not silent failure (project numerical-stability rule from `AGENTS.md`).

---

### 12. Debugging long-context issues

| Symptom | Check |
|---------|-------|
| Identical outputs at 4K and 128K | YaRN disabled? `scale_factor=1`? |
| `UserWarning: degenerate ramp` | Fix `beta_fast`/`beta_slow` |
| Good at 4K, bad at 128K only on odd layers | Pruning too aggressive — try `yarn_prune_rope_global=False` |
| Position sensitivity wrong on even layers | Pruning should be 0 — verify `layer_idx % 2` |
| cos/sin dtype mismatch in SDPA | `apply_rope` must preserve `x.dtype` |

### Eval vs train sequence length

- Training: `max_seq_len=4096`
- Eval: `eval_max_seq_len=131072`

Ensure eval scripts pass positions up to `eval_max_seq_len`, not `max_seq_len`.

If YaRN ramp is degenerate (see [§11](#11-degenerate-ramp-warning)), positions beyond 4K remain ambiguous — extrapolation fails silently aside from the warning.

---

### 13. Invariants and failure modes

### 13.1 No learned RoPE parameters

`inv_freq` is a fixed buffer from hyperparameters. Position generalisation is entirely in the frequency table design (YaRN ramp), not learned embeddings.

### 13.2 `head_dim` must be even

Odd `head_dim` raises `ValueError` in both `compute_yarn_freqs` and `YaRNRoPE`. Production config uses `head_dim=96`.

### 13.3 Relation to standard `rope` in other repos

Some implementations use `torch.polar` or complex multiplication. This repo uses the rotate-half trick — fewer dependencies, identical math, better `torch.compile` compatibility.

### 13.4 Import paths

```python
from models.rotary import apply_rope, compute_yarn_freqs, compute_yarn_mscale
```

`models/attention.py` imports `apply_rope` from `models.rotary`. `models/yarn.py` imports freq helpers from `models.rotary`.

### 13.5 Config validation

`ModelConfig.__post_init__` enforces:

```python
if self.yarn_scale_factor < 1:
    raise ValueError(...)
if self.yarn_scale_factor > 1 and self.yarn_original_max_seq_len >= self.yarn_target_seq_len:
    raise ValueError(...)
if self.yarn_original_max_seq_len <= 0 or self.yarn_target_seq_len <= 0:
    raise ValueError(...)
```

`yarn_prune_rope_global=True` with odd `n_layers` emits a warning — final layer may be windowed (no pruning on last layer if it's even-indexed).

---

### 14. How to verify

```bash
python3 -m pytest tests/test_yarn.py -v
```

Additional checks:

- `head_dim % 2 == 0` — enforced at construction.
- Degenerate ramp emits `UserWarning` (not silent identity).
- `apply_rope` output dtype matches input dtype.
- Pruned global layers: `n_pruned_dims = head_dim // 4` pairs.

---

### Related Documentation

- [ATTENTION_SINKS.md §B.7](attention-sinks.md#b7-gptossattention-construction-sink-param-yarn-pruned-rope) — `GPTOSSAttention` integration
- [attention-sinks.md §B.8](attention-sinks.md#b8-forward-path-trace-positions--out-proj) — where RoPE sits in the forward path
- [attention-sinks.md](attention-sinks.md) — sinks independent of RoPE; YaRN at 128K

---

## References

- [`models/attention.py:GPTOSSAttention`](../../models/attention.py) — attention module consuming RoPE
- [`models/rotary.py:apply_rope`](../../models/rotary.py) — dtype-safe rotation
- [`models/rotary.py:compute_yarn_freqs`](../../models/rotary.py) — YaRN ramp-blended frequencies
- [`models/yarn.py:YaRNRoPE`](../../models/yarn.py) — YaRN module
- [attention-sinks.md](attention-sinks.md) — sink bias, SWA/full alternation, clamp rationale
- [foundations-and-architecture.md](foundations-and-architecture.md) — primer: attention, GQA, SWA
- [inference.md](../inference.md) — rotated-K caching in `MixedKVCache`

<!-- docs:verified 2026-08-05 · 6491066 -->
