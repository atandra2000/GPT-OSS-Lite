# Attention Math — Softmax, Scaling, Masks, and SDPA

> **Theory chapter T1.** From-scratch derivation of the arithmetic inside
> scaled dot-product attention, mapped onto `models/attention.py`. Assumes the
> primer level of [foundations](../foundations.md) §2–§4; the sink mechanism is
> treated in depth in [ATTENTION_SINKS](../ATTENTION_SINKS.md) and the KV-cache
> consequences in [kv cache engineering](kv_cache_engineering.md).

## Table of contents

1. [60-second summary](#1-60-second-summary)
2. [Why it matters here](#2-why-it-matters-here)
3. [Intuition](#3-intuition)
4. [Theory and derivation](#4-theory-and-derivation)
5. [Code walkthrough](#5-code-walkthrough)
6. [Pitfalls and verification](#6-pitfalls-and-verification)

---

## 1. 60-second summary

Scaled dot-product attention computes, for every position, a weighted average
of value vectors. The weights are softmax over query-key dot products, divided
by the square root of the head width so scores stay near unit variance. A mask
(causal, sliding-window, learned sink logit) is added to the scores first so
forbidden keys receive zero weight. GPT-OSS-Lite clamps its learned sink logits
to `[-10, 15]` (`models/attention.py:SINK_CLAMP_MIN` /
`models/attention.py:SINK_CLAMP_MAX`) so the mask arithmetic never overflows
BF16. All production attention runs through `F.scaled_dot_product_attention`
inside `models/attention.py:causal_attention`, which dispatches to a math,
memory-efficient, or flash backend; flash never materializes the T-by-T score
matrix. An FP32 oracle, `models/attention.py:manual_causal_attention`, is the
test reference. This chapter derives each piece, then walks the code, including
two verified behavioral quirks (the sink-path additive mask and the square-mask
window) that the current test suite does not pin down.

## 2. Why it matters here

Attention is the arithmetic that everything else in GPT-OSS-Lite hangs off:

- **Alternating SWA/full.** Twelve layers, even indices sliding-window with
  $W = 128$, odd indices full causal (`models/transformer.py:GPTOSSBlock`
  constructs `models/attention.py:GPTOSSAttention` per layer; the alternation is
  decided in `models/attention.py:GPTOSSAttention.__init__` via `layer_idx % 2`).
  The mask math in
  §4.4–4.5 is what makes the two patterns differ — and, as §5.6 documents, where
  they currently do not differ during prefill.
- **Learned sink bias.** Each of the 8 heads carries one scalar logit that acts
  as an extra softmax column with a zero value vector. It exists to absorb
  attention mass that sliding-window eviction would otherwise scatter; the
  mathematics is derived in [ATTENTION_SINKS §4](../ATTENTION_SINKS.md), and the
  BF16 clamp rationale in [ATTENTION_SINKS §6](../ATTENTION_SINKS.md). This
  chapter supplies the underlying softmax and mask machinery.
- **GQA.** 8 query heads share 4 KV heads (`head_dim = 96`). The broadcast in
  §4.6 halves KV-cache bytes and KV traffic per token; combined with the six
  windowed layers it produces the measured 2.00× KV reduction at 128K
  ([ATTENTION_SINKS §8](../ATTENTION_SINKS.md), `scripts/kv_cache_benchmark.py`).
- **SDPA as the single execution path.** `models/attention.py:causal_attention`
  is the only attention entry point in training and inference
  (`inference/generate.py:_attn_forward_layer`). Which backend runs — math,
  memory-efficient, or flash — decides whether a $T \times T$ matrix is ever
  materialized, which is the difference between 128K evaluation fitting in VRAM
  or not.
- **Budget honesty.** The ≥85% passkey @128K figure is a **target**; no
  pretraining has run. The 2.00× / 1.94× KV figures are **measured**; every A100
  throughput figure elsewhere is `[INFERENCE]`. This chapter contains no
  performance numbers that were not derived above.

## 3. Intuition

Think of each row of $K$ as a labeled point in $d$-dimensional space, and the
query $q_i$ as a probe. The dot product $q_i \cdot k_j$ is (up to the lengths of
the vectors) the projection of $k_j$ onto the direction of $q_i$ — a raw
similarity score, unbounded and signed. Attention then answers: "among all keys,
how should I split a unit budget of attention?" Three requirements fix the form
of that split:

1. Weights must be non-negative and sum to one — the output is a convex
   combination of the value rows, an average, not a vector addition.
2. The split must be a smooth, differentiable function of the scores so
   gradients can flow through it.
3. The mapping should be "winner-take-most": a key with a slightly better score
   gets exponentially more weight, but not all of it.

Softmax is the canonical smooth relaxation of argmax that satisfies all three
(§4.2). The $1/\sqrt{d}$ factor is a **temperature**: it controls how sharp the
distribution is. Scores of unit variance (§4.3) keep the exponential from
blowing up or degenerating. Masks are hard constraints that delete keys from the
support set before normalization — and the learned sink (§4.4) is a clever
softener: a dummy key with a learnable constant logit and a zero value vector,
so the model can park attention mass where it does nothing, instead of being
forced to redistribute it when the window evicts a token.

The fused kernels (§4.7) implement the same arithmetic but never build the
$T \times T$ score matrix; they exploit that softmax normalization is *online* —
you can accumulate it block by block as long as you rescale when a new block
raises the running maximum.

## 4. Theory and derivation

### 4.1 The attention operation

Scaled dot-product attention is defined as

$$
\mathrm{Attn}(Q, K, V) = \mathrm{softmax}\!\left(\frac{QK^\top}{\sqrt{d}} + M\right) V
\tag{1}
$$

where $Q \in \mathbb{R}^{T_q \times d}$ holds one query per row, $K, V \in
\mathbb{R}^{T_k \times d}$ hold keys and values, $d$ is the head dimension, and
$M \in (\mathbb{R} \cup \{-\infty\})^{T_q \times T_k}$ is an additive mask.
Writing the score matrix $S = QK^\top / \sqrt{d}$ and the weight matrix $A =
\mathrm{softmax}(S + M)$ row-wise, row $i$ of the output is

$$
o_i = \sum_{j=1}^{T_k} \alpha_{ij}\, v_j, \qquad \alpha_{ij} \ge 0, \quad
\sum_j \alpha_{ij} = 1.
$$

The mask enters *before* softmax, never after: it modifies the support of the
distribution, not the weights of already-normalized probabilities.

### 4.2 Softmax: definition, why it appears, numerical stability

For a vector of logits $z \in \mathbb{R}^n$, the softmax function is

$$
\sigma(z)_i = \frac{e^{z_i}}{\sum_{j=1}^{n} e^{z_j}}.
\tag{2}
$$

It appears in attention for three reasons, each of which is a design constraint
of the operation in §4.1:

- **Normalization.** $\sigma(z)$ is a probability distribution over the $n$
  keys: non-negative, summing to one, so the output is a convex combination of
  values.
- **Max-entropy derivation.** Among all distributions $p$ over keys with a fixed
  expected score $\sum_j p_j z_j$, the one maximizing entropy $-\sum_j p_j \ln
  p_j$ is the exponential-family distribution $p_j \propto e^{\lambda z_j}$, of
  which (2) is the normalized form. It is the *least committed* distribution
  consistent with a given average similarity — a principled reason, not a
  convention.
- **Smoothness.** Unlike $\arg\max$, (2) is differentiable everywhere, with
  Jacobian $\partial \alpha_i / \partial z_j = \alpha_i (\delta_{ij} -
  \alpha_j)$; gradients flow to every key, weighted by current attention.

**Shift invariance and the stable form.** Softmax is invariant to adding a
constant to every logit:

$$
\sigma(z - c \mathbf{1})_i = \frac{e^{z_i - c}}{\sum_j e^{z_j - c}}
= \frac{e^{z_i}}{\sum_j e^{z_j}} = \sigma(z)_i,
\tag{3}
$$

because the factor $e^{-c}$ cancels between numerator and denominator. The
numerically stable form exploits (3) with $c$ equal to the row maximum:

$$
\sigma(z)_i = \frac{e^{z_i - m}}{\sum_j e^{z_j - m}}, \qquad m = \max_j z_j,
\qquad z_i - m \le 0 \;\; \forall i.
\tag{4}
$$

This is not optional in floating point. The largest finite value in both FP32
and BF16 is $3.4 \times 10^{38}$, and $e^x$ reaches it at $x \approx 88.7$
(derived in §4.4). If any logit exceeded that, $e^{z_i}$ would round to
$+\infty$, the denominator would be $+\infty$, and every weight would collapse
to NaN via $\infty / \infty$. Subtracting the max keeps every exponent argument
$\le 0$, so all exponentials live in $(0, 1]$. PyTorch's softmax kernels do
this internally; the danger is upstream, where a *mask value* (not a softmax
argument) can be huge — see §4.4.

### 4.3 Scaled dot product: why $1/\sqrt{d}$

Take the entries of $q_i$ and $k_j$ to be independent draws with mean zero and
unit variance, which is the regime the projections in
`models/attention.py:GPTOSSAttention.forward` are initialized into
(`init_std = 0.02` in `models/transformer.py:ModelConfig`, and the linear
projections are weight-normalized so activation variance is $O(1)$). The raw
score is a sum of $d$ products:

$$
\mathrm{Var}(s_{ij}) = \mathrm{Var}\!\left(\sum_{l=1}^{d} q_{il} k_{jl}\right)
= \sum_{l=1}^{d} \left(\mathbb{E}[q_{il}^2]\mathbb{E}[k_{jl}^2] -
\mathbb{E}[q_{il}]^2 \mathbb{E}[k_{jl}]^2 \right) = \sum_{l=1}^{d} 1 = d,
\tag{5}
$$

using independence of $q_{il}$ and $k_{jl}$ and $\mathbb{E}[q_{il}^2] =
\mathrm{Var}(q_{il}) + \mathbb{E}[q_{il}]^2 = 1$. The score standard deviation
is therefore $\sqrt{d}$, and dividing by it yields unit-variance scores:

$$
\tilde s_{ij} = \frac{s_{ij}}{\sqrt{d}}, \qquad
\mathrm{Var}(\tilde s_{ij}) = \frac{d}{d} = 1.
\tag{6}
$$

For this model, $d = 96$, so $\sqrt{d} \approx 9.8$. Why does the scaling
matter? Without it, scores scale as $\sqrt{d}$: for $d = 96$ they spread to
$\pm 10$, and with $T_k$ keys the row maximum grows like $\sqrt{2 \ln T_k}$
(the expected maximum of $n$ unit-variance Gaussians; at $T_k = 131072$,
$\sqrt{2 \ln 131072} \approx 4.9$). The softmax of such scores concentrates
almost all mass on one key: $\alpha_i \to 1$ for the argmax, and the Jacobian
$\alpha_i(\delta_{ij} - \alpha_j)$ vanishes for every $j$ — gradients stop
flowing to all non-winning keys. Unit-variance logits keep the distribution
smooth across sequence lengths, which is exactly why this repo can train at
$T = 4096$ and evaluate at $T = 131072$ without re-tuning the temperature.
The same $\sqrt{d}$ appears as the "temperature" of the exponential family in
§4.2: $\lambda = 1/\sqrt{d}$.

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

Because $e^{-\infty} = 0$, forbidden keys contribute nothing to numerator or
denominator; the surviving weights renormalize automatically. Two mechanical
ways to apply $M$:

- **mask-fill**: write $-\infty$ into the score tensor itself
  (`scores.masked_fill(causal, float("-inf"))`), as the oracle
  `models/attention.py:manual_causal_attention` does;
- **mask-add**: keep a separate mask tensor and add it to scores
  (`scores + attn_mask`), as SDPA does with float masks. A *boolean* mask is
  sugar for mask-add: `True` means add $0$, `False` means add $-\infty$.

The two are mathematically identical; the difference is mechanical (in-place
mutation vs composition) and numerical (where the $-\infty$ lives).

**The BF16 overflow trap.** Addition with true $-\infty$ is safe: $x + (-\infty)
= -\infty$ for every finite $x$. The traps are the *other* directions, and they
are the reason the sink bias is clamped:

- **exp overflow in normalization.** Softmax evaluates $e^{z_i - m}$ with $z_i -
  m \le 0$ (§4.2), so normalization itself is safe. But if the mask *adds a huge
  finite positive logit* — an unbounded learned sink, say $s_h = 1000$ — then the
  augmented row contains a logit of $+1000$, and $e^{1000} = +\infty$ even after
  max subtraction in low precision; the denominator becomes $\infty$ and every
  weight becomes NaN ($\infty / \infty$). The overflow threshold is

$$
e^{x} \le 3.4 \times 10^{38} \iff x \le \ln(3.4 \times 10^{38}) \approx 88.7.
\tag{8}
$$

- **finite-mask overflow on addition.** A mask built with a huge *finite*
  negative sentinel (e.g. `-1e38` in a low-precision tensor) can overflow to
  $\pm\infty$ when *added* to an opposite-sign score: $-10^{38} + 3 \times 10^{38}
  = +2 \times 10^{38} = +\infty$ in BF16, and $+\infty + (-\infty)$ later yields
  NaN. The rule: use true $-\infty$ (or $0$), never a large finite sentinel.

GPT-OSS-Lite's defense is exactly (8): `GPTOSSAttention.forward` clamps the
sink parameter to $[SINK_CLAMP_MIN, SINK_CLAMP_MAX] = [-10, 15]$ before it
enters the mask. With $s_h \le 15$, the sink logit never exceeds
$e^{15} \approx 3.3 \times 10^6 \ll 3.4 \times 10^{38}$, and $-10$ is safe too:
$e^{-10} \approx 4.5 \times 10^{-5}$, effectively "no sink". The upper bound 15
is also comfortably above the score spread derived in §4.3 (≈4.9 at 128K), so
the sink can absorb essentially all mass when trained to do so. Full rationale
and the gradient-through-clamp note are in
[ATTENTION_SINKS §6](../ATTENTION_SINKS.md).

### 4.5 Sliding-window masking as a banded mask

A sliding window of width $W$ restricts query $i$ to the most recent $W$ keys,
intersected with causality:

$$
\mathcal{A}(i) = \{ j : j \le i \;\wedge\; i - j < W \}.
\tag{9}
$$

The resulting mask is **banded**: in row $i$, entries $j < i - W + 1$ are
$-\infty$. The allowed set shrinks from a full triangle to a band of width $W$:

$$
N_{\mathrm{dense}} = \frac{T(T+1)}{2} \approx \frac{T^2}{2}, \qquad
N_{\mathrm{sw}} = \sum_{i=0}^{T-1} \min(i+1, W)
= \frac{W(W+1)}{2} + (T - W)\, W \approx W\, T \quad (T \gg W).
\tag{10}
$$

Attention FLOPs are proportional to the number of allowed pairs: each pair costs
$2d$ FLOPs for the dot product and $2d$ FLOPs for the value accumulation
($4d$ total). The saving per windowed layer is therefore

$$
\frac{\mathrm{FLOPs}_{\mathrm{dense}}}{\mathrm{FLOPs}_{\mathrm{sw}}}
= \frac{4d\, N_{\mathrm{dense}}}{4d\, N_{\mathrm{sw}}}
\approx \frac{T}{2W}
\xrightarrow{T = 131072,\; W = 128} \frac{131072}{256} = 512.
\tag{11}
$$

The same ratio bounds the score-matrix memory: dense attention materializes
$\approx T^2/2$ score elements, banded attention $\approx WT$. Neither the FLOP
nor the memory saving is the repo's headline metric, though — the 2.00× number
is the **KV-cache** reduction (windowed layers cache at most $W$ tokens), which
is derived in [ATTENTION_SINKS §8](../ATTENTION_SINKS.md) and measured by
`scripts/kv_cache_benchmark.py` (2.00× at 128K, 1.94× at 4K). The banded-mask
arithmetic above is what *would* deliver the FLOP side of SWA; §5.6 documents
where the current code actually applies it.

### 4.6 GQA: head broadcast and cache bytes

Grouped-query attention projects $H_{\mathrm{kv}} = 4$ key/value heads and
repeats each one to serve $g = H / H_{\mathrm{kv}} = 8 / 4 = 2$ query heads.
Formally, the key used by query head $h$ is

$$
\hat K_h = K_{\lfloor h / g \rfloor}, \qquad g = \frac{H}{H_{\mathrm{kv}}} = 2,
\tag{12}
$$

i.e. heads $\{0,1\}$ share $K_0$, heads $\{2,3\}$ share $K_1$, and so on. The
gain is measured in bytes: with BF16 (2 bytes), $D = 96$:

$$
\text{KV bytes per token per layer} = 2 \cdot H_{\mathrm{kv}} \cdot D \cdot 2
= 2 \cdot 4 \cdot 96 \cdot 2 = 1536,
$$

against $2 \cdot 8 \cdot 96 \cdot 2 = 3072$ for full MHA — a 2× cut before any
windowing, and the multiplicand in the 2.00× headline. In code the broadcast is
`models/attention.py:repeat_kv`, discussed in §5.5.

### 4.7 SDPA backends, and flash attention's online softmax

`F.scaled_dot_product_attention` hides three implementations behind a dispatch
heuristic (dtype, shape, mask type, device):

- **math backend**: the textbook algorithm of (1). Materializes the full
  $B \times H \times T_q \times T_k$ score matrix in memory, softmaxes it, then
  multiplies by $V$. $O(T^2)$ memory, always available (it is the CPU fallback).
- **memory-efficient backend** (xformers lineage): processes the output in
  blocks, computing a softmax per block with online rescaling (below). Never
  holds the full $T \times T$ matrix, but per-block $B_r \times B_c$ score tiles
  still materialize; supports limited mask shapes and requires aligned
  contiguous inputs.
- **flash backend** (FlashAttention-2 lineage): a fully fused kernel. Tiles
  $Q$, $K$, $V$ into SRAM-resident blocks, applies online softmax, and writes
  only the $B_r \times d$ output tile back to HBM. Nothing $O(T^2)$ ever
  exists — not the score matrix, not the weights. Uses tensor-core
  matrix-multiply units and is native-BF16, which is why it matters for this
  repo's BF16 GQA stack: KV heads are shared, so each KV block is read once per
  query block, and at $T = 131072$ the alternative (math backend) would require
  a $T^2$ matrix no GPU holds.

**Online softmax rescaling.** Fusing requires replacing the two-pass softmax
(scan for max, then normalize) with a one-pass version that can absorb blocks
in any order. Process key blocks left to right, maintaining a running row-max
$m$ and an unnormalized accumulator $O$ (output) and $l$ (denominator). When a
new score block $S'$ arrives with block maximum $m' = \max_j S'_{ij}$, every
previously accumulated term was normalized by $e^{m_{\mathrm{old}}}$ and must be
re-expressed in units of $e^{m}$ with $m = \max(m_{\mathrm{old}}, m')$:

$$
m \leftarrow \max(m_{\mathrm{old}}, m'), \qquad
O \leftarrow O\, e^{m_{\mathrm{old}} - m} + e^{S' - m}\, V', \qquad
l \leftarrow l\, e^{m_{\mathrm{old}} - m} + \mathbf{1}^\top e^{S' - m}.
\tag{13}
$$

The rescale factor $e^{m_{\mathrm{old}} - m}$ is $\le 1$ and only deviates from
1 when a new block actually raises the max — typically a few times per row.
After the last block, the accumulated output is divided by the accumulated
denominator:

$$
o_i = \frac{O_i}{l_i}.
\tag{14}
$$

The per-block weights are never stored, only the running pair $(m, l)$; that is
the entire trick. (13)–(14) are exactly the FlashAttention paper's online
softmax, and they make (1) memory-embarrassingly parallel across output blocks
while remaining bitwise-equivalent-in-expectation to the two-pass version up to
floating-point reassociation.

### 4.8 The sink column: one more key with a constant logit

The learned sink is an extra softmax column with logit $s_h$ (per head $h$) and
a zero value vector. Augmenting the banded/causal score rows:

$$
\alpha_{ij} = \frac{e^{\tilde s_{ij}}}{\sum_{j' \in \mathcal{A}(i)} e^{\tilde
s_{ij'}} + e^{s_h}}, \qquad o_i = \sum_{j \in \mathcal{A}(i)} \alpha_{ij}\, v_j,
\tag{15}
$$

so the sink's weight $e^{s_h} / Z_i$ is absorbed without contributing to the
output ($v_{\mathrm{sink}} = 0$). With $s_h \to +\infty$ the output tends to
$\mathbf{0}$; with $s_h \to -\infty$ the sink vanishes. This is the formulation
of [ATTENTION_SINKS §4.3](../ATTENTION_SINKS.md); §5.2 shows how the code
implements it as a zero K/V column plus a mask column.

## 5. Code walkthrough

All symbols below live in `models/attention.py` unless noted.

### 5.1 The module: `GPTOSSAttention.forward` and `extra_repr`

`GPTOSSAttention` is constructed per layer by `models/transformer.py:GPTOSSBlock`
with `cfg` from `models/transformer.py:ModelConfig`. Construction stores the
alternation decision — `self.is_windowed = (layer_idx % 2 == 0)` — and the GQA
ratio `self.n_rep = self.n_heads // self.n_kv_heads = 2`. Projections are
`q_proj: (768, 8·96)`, `kv_proj: (768, 2·4·96)`, `o_proj: (8·96, 768)`, all
bias-free; the sink is an `nn.Parameter` of shape `(n_heads,)` initialized to
zero when `cfg.sink_bias` is set. `models/yarn.py:YaRNRoPE` is instantiated with
the §4.3-relevant `head_dim=96` and the YaRN constants (θ=100K, scale 32,
4K→128K; see [rope_yarn](../rope_yarn.md)).

`models/attention.py:GPTOSSAttention.forward` implements (1) end to end:

```python
query_states = self.q_proj(x).view(B, T, self.n_heads, self.head_dim)
kv = self.kv_proj(x).view(B, T, 2, self.n_kv_heads, self.head_dim)
key_states, value_states = kv[:, :, 0], kv[:, :, 1]
```

One fused `kv_proj` produces both K and V for only $H_{\mathrm{kv}}$ heads —
the GQA saving of §4.6 happens at projection time, before any attention
arithmetic. The tensors are transposed to `(B, H, T, D)`, RoPE is applied
(`models/rotary.py:apply_rope`, with `self._n_pruned_dims()` zeroing 25% of
frequency pairs on global layers only), then the KV heads are broadcast:

```python
key_states = repeat_kv(key_states, self.n_rep)
value_states = repeat_kv(value_states, self.n_rep)
```

The sink clamp is applied here, *before* the mask is built — the parameter
itself is never mutated:

```python
sink_bias_clamped = self.sink_bias.clamp(SINK_CLAMP_MIN, SINK_CLAMP_MAX)
```

Then `models/attention.py:causal_attention` is called with
`window=self.window_size if self.is_windowed else None`, and the output is
transposed, `contiguous()`, viewed back to `(B, T, H·D)`, and passed through
`o_proj`. `models/attention.py:GPTOSSAttention.extra_repr` renders the layer
identity used by `repr(model)`: `"layer={i} (SWA|Full[, pruned=n]), H=8/4,
D=96, window=128"`, where the `pruned=` suffix appears only for global layers
with `yarn_prune_rope_global` enabled.

### 5.2 `causal_attention`: three SDPA paths

`models/attention.py:causal_attention` is a dispatcher over four cases. With no
sink and no window on a square input it takes the kernel-native fast path:

```python
if T_q == T_k:
    return F.scaled_dot_product_attention(query_states, key_states, value_states, is_causal=True)
```

`is_causal=True` lets the fused kernels use their optimized causal loop (only
half the score blocks), which no explicit mask can match — this is the path the
flash backend is designed for. With a window but no sink, the boolean masks are
composed and converted to a proper additive mask:

```python
mask = _causal_mask(T_q, device, dtype) & _window_mask(T_q, T_k, window, device, dtype)
attn_mask = torch.where(mask, 0.0, float("-inf")).to(dtype).unsqueeze(0).unsqueeze(0)
```

`torch.where(mask, 0.0, -inf)` implements (7) exactly: allowed → $+0$, forbidden
→ true $-\infty$ (the safe form per §4.4). With a sink, the value side gains a
zero column and the bias rides on the mask:

```python
sink_k = torch.zeros(B, H, 1, query_states.shape[-1], device=device, dtype=dtype)
sink_v = torch.zeros(B, H, 1, value_states.shape[-1], device=device, dtype=value_states.dtype)
k_ext = torch.cat([key_states, sink_k], dim=2)
v_ext = torch.cat([value_states, sink_v], dim=2)
```

The mask becomes shape `(H, T_q, T_k + 1)`: the first $T_k$ columns come from
`causal.to(dtype)` and the last column carries the clamped per-head bias,
implementing (15). See §5.6 for the exact float semantics of that mask, which
differ from the window path above. Whether the mask forces a fallback from the
fused kernels to math/mem-efficient depends on the torch version and tensor
alignment; the code never forces a backend, and Q, K, V must share dtype for
SDPA to dispatch at all.

### 5.3 Mask helpers: `_causal_mask`, `_window_mask`

`models/attention.py:_causal_mask` builds the lower triangle via broadcasting:

```python
idx = torch.arange(T, device=device)
return idx.unsqueeze(1) >= idx.unsqueeze(0)  # (T_q, T_k)
```

Entry $(i, j)$ is `idx[i] >= idx[j]`, i.e. `True` where $j \le i$ — the boolean
form of (7). It is `lru_cache`d on `(T, device, dtype)` so prefill and decode
reuse one allocation per shape. `models/attention.py:_window_mask` has two
branches. For decode (`T_q = 1`, `T_k` growing), the query position is known to
be `T_k - 1` and the mask is

```python
idx_q = torch.tensor([T_k - 1], device=device)
idx_k = torch.arange(T_k, device=device)
return (idx_q.unsqueeze(-1) - idx_k.unsqueeze(0) < window)
```

a single row that is `True` for the last `window` keys — a true band, matching
(9) with $i = T_k - 1$. For square inputs the helper intersects its band with
the causal mask (see §5.6 for the band's orientation).

### 5.4 `manual_causal_attention`: the FP32 oracle

`models/attention.py:manual_causal_attention` is the reference implementation
for tests. Every score is computed in FP32 by explicit upcast:

```python
scores = (query_states.float() @ key_states.float().transpose(-2, -1)) / math.sqrt(D)
```

so the accumulation cannot be contaminated by BF16 rounding (the FP32
accumulation concern of §6). Causality uses a `triu` boolean fill, the window
uses a masked fill with the same transposed orientation as §5.3, and the sink
is appended as an actual score column:

```python
sink_logit = sink_bias.view(1, H, 1, 1).to(scores.dtype)
augmented = torch.cat([scores, sink_logit.expand(B, H, T, 1)], dim=-1)
attn_weights = F.softmax(augmented, dim=-1)
attn_weights = attn_weights[..., :T]
return (attn_weights.to(value_states.dtype) @ value_states)
```

The sink column is stripped before the value matmul (only real keys multiply
$V$), exactly the "absorb, don't contribute" of (15). This is the oracle the
equivalence tests compare SDPA against — the same math as §4.1, written out
naively at $O(T^2)$.

### 5.5 `repeat_kv`: GQA broadcast without an explicit copy

`models/attention.py:repeat_kv` broadcasts $H_{\mathrm{kv}}$ heads to $H$:

```python
x = x[:, :, None, :, :]
x = x.expand(B, H_kv, n_rep, T, D)
return x.reshape(B, H_kv * n_rep, T, D)
```

`expand` is a stride-0 view (no data movement); `reshape` then merges the
`(H_kv, n_rep)` dims into `(H,)`. The code never calls `.contiguous()` — and
does not need to: `reshape` materializes a contiguous copy itself, because a
stride-0 dimension cannot be merged without one. The result is therefore a real,
contiguous `(B, 8, T, 96)` tensor (verified: `is_contiguous() == True`, no
shared storage with the input), not a lazy view — but the copy is one pass over
4 heads' worth of data, whereas the alternative (projecting 8 KV heads) would
cost double the projection FLOPs forever. `n_rep == 1` short-circuits to the
input unchanged.

### 5.6 Verified behavioral notes (2026-08-04, by direct execution)

Two correctness bugs in the mask helpers were found and fixed on 2026-08-04 as
part of the documentation-expansion audit (this chapter's first draft
documented the buggy behavior; the fixes below are what the code does now).
Both were silent: the 192-test baseline passed because the restriction tests
were vacuous (Note 3).

**Note 1 — the square `_window_mask` band was vacuous under causality; fixed.**
In the `T_q == T_k` branch the band condition was

```python
return (idx.unsqueeze(0) - idx.unsqueeze(1) < window) & _causal_mask(T_q, device, dtype)
```

Entry $(i, j)$ is $j - i$, so the condition was $j - i < W$; under causality
$j \le i$ that is always true — the window added nothing, and windowed layers
performed full causal attention during prefill. Fixed to $i - j < W$
(`idx.unsqueeze(1) - idx.unsqueeze(0)`): query $i$ now sees keys
$\max(0, i - W + 1) \le j \le i$. `models/attention.py:manual_causal_attention`
had the same transposed orientation and was likewise vacuous; fixed the same
way. Verified post-fix: `_window_mask(T, T, 8) != _causal_mask(T)` (position 63
blocks 56 of 63 keys at $W=8$), SDPA-windowed ≡ manual-windowed at every
position (allclose `1e-5`), and zeroing keys $j \le t - W$ leaves position $t$
unchanged. The decode branch was always a true last-$W$ band and is untouched;
`inference/generate.py:MixedKVCache.append` still caps storage at $W$ tokens,
and the KV-cache 2.00× metric is unaffected (a storage claim). With the fix,
the banded-mask FLOP savings of (11) now materialize at prefill as well.

**Note 2 — the sink-path mask was additive `+1`/`0`, leaking future tokens;
fixed.** The sink path previously wrote `mask[:, :, :T_k] = causal.to(dtype)`
— `1.0` where allowed, `0.0` where forbidden — and SDPA *adds* float masks, so
"forbidden" positions were not excluded and future tokens leaked into every
position's softmax at prefill (verified: weight on future keys ≈ 0.77 for row 0
on random inputs). Fixed to `torch.where(causal, 0.0, -inf)`: allowed positions
add `0`, blocked positions add `-∞`. Verified post-fix: sink-path SDPA ≡ the
manual sink oracle at every position (allclose `1e-5`), and corrupting future
keys leaves position 0 unchanged. Decode was unaffected in both versions (the
cache contains no future keys; a uniform additive shift cancels in softmax).

**Note 3 — the SWA-restriction tests were vacuous at fixture sizes; fixed.**
`tests/test_attention.py`'s fixtures produce `(B, H, T, D)` tensors
(`tests/conftest.py` `_make_attn_inputs`), but the restriction tests unpacked
them as `(B, T, H, D)`, so their `T` was the head count (4 or 8): the
outside-the-window loop over `range(window, ...)` was empty and the
inside-the-window loop visited at most 8 trivial positions. The unpacking was
corrected and four regression tests were added:
`test_sliding_window_sdpa_matches_manual`, `test_sliding_window_blocks_past_keys_at_prefill`,
`test_sink_path_matches_manual_at_prefill`, `test_sink_path_is_causal`.
`pytest tests/test_attention.py -v` now pins prefill windowing and sink-path
causality directly (22 tests on CPU, 2026-08-04).

## 6. Pitfalls and verification

**FP32 accumulation in the oracle.** `models/attention.py:manual_causal_attention`
upcasts to FP32 before the matmul; the equivalence tests feed it float64 inputs.
This is deliberate — the oracle must be the *truth*, and BF16 scores would make
"matches the reference" meaningless. The fused kernels also accumulate in FP32
internally but read BF16 inputs; when debugging an SDPA-vs-manual mismatch,
first check dtype, then mask semantics, then backend.

**Mask arithmetic traps.** (a) Never substitute a large finite sentinel for
$-\infty$ — §4.4's addition-overflow path turns it into NaN. (b) Boolean masks
mean "allowed"; float masks are added verbatim — `causal.to(dtype)` would be
`1.0`, not `0.0`, so blocked positions must be set to `-inf` explicitly
(`torch.where(causal, 0.0, -inf)`, as the sink path now does). (c)
`-inf + (-inf) = -inf` is safe; `+inf + (-inf) = NaN` is the overflow signature
to grep for. The clamp `models/attention.py:SINK_CLAMP_MIN` /
`models/attention.py:SINK_CLAMP_MAX` exists to keep the sink column out of the
danger zone; the parameter retains its raw value for gradient flow.

**Regression guards.** A green `pytest tests/test_attention.py -v` now proves
prefill windowing and sink-path causality (Notes 1–3). Any future change to the
mask helpers must keep the four regression tests green — they exist precisely
because the previous vacuous tests let two correctness bugs ship.

**Backend nondeterminism.** `F.scaled_dot_product_attention` chooses math,
mem-efficient, or flash by heuristics; only the math backend is guaranteed.
Do not assume a fused kernel ran — check `torch.backends.cuda` diagnostics —
and keep Q, K, V in one dtype. Perf claims about flash on this repo's target
hardware are `[INFERENCE]`; `.benchmarks/` is empty.

**Verification commands.** The guard for every equivalence claim in this
chapter is

```
python3 -m pytest tests/test_attention.py -v
```

22 passed on CPU, 2026-08-04 (≈2.5 s; the 2 GPU-gated Triton skips live in
`tests/test_moe_triton.py`, not here). The KV numbers cross-referenced in §4.5
are guarded by `python3 scripts/kv_cache_benchmark.py` (2.00× @128K, 1.94×
@4K — measured). The passkey/quality bands are targets, not results, and are
not guarded by any command yet.

<!-- docs:verified 2026-08-04 · 5da1a80 -->
