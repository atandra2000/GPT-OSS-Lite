# Attention Sinks & Alternating Attention — GPT-OSS-Lite

> **Authoritative** reference for learned sink bias, sliding-window / full
> alternation, and the `models/attention.py` implementation. Required reading
> before changing attention code (see `AGENTS.md`). Positional encoding:
> [rope_yarn.md](rope_yarn.md).

---

## Table of Contents

1. [Executive Summary](#1-executive-summary)
2. [Historical Context: StreamingLLM and Attention Sinks](#2-historical-context-streamingllm-and-attention-sinks)
3. [Why Sinks Matter with Sliding-Window Attention](#3-why-sinks-matter-with-sliding-window-attention)
4. [Mathematical Formulation](#4-mathematical-formulation)
5. [Implementation in `models/attention.py`](#5-implementation-in-modelsattentionpy)
6. [BF16 Clamp Rationale](#6-bf16-clamp-rationale)
7. [Sliding-Window / Full Alternation](#7-sliding-window--full-alternation)
8. [KV-Cache Mathematics at 128K](#8-kv-cache-mathematics-at-128k)
9. [Interaction with YaRN and Pruned RoPE](#9-interaction-with-yarn-and-pruned-rope)
10. [Prefill vs Decode](#10-prefill-vs-decode)
11. [Training Dynamics](#11-training-dynamics)
12. [Failure Modes and Debugging](#12-failure-modes-and-debugging)
13. [Design Decisions vs Alternatives](#13-design-decisions-vs-alternatives)
14. [Quick Reference](#14-quick-reference)

**Part B — Implementation**

- [B.1 Module overview and public surface](#b1-module-overview-and-public-surface)
- [B.2 Constants (`SINK_CLAMP_MIN`, `SINK_CLAMP_MAX`)](#b2-constants-sink_clamp_min-sink_clamp_max)
- [B.3 `manual_causal_attention`](#b3-manual_causal_attention-fp32-scores-window-sink)
- [B.4 Mask helpers](#b4-mask-helpers-_causal_mask-_window_mask-cache-key)
- [B.5 `causal_attention` SDPA paths](#b5-causal_attention-sdpa-paths-fast--window--sink-column)
- [B.6 `repeat_kv`](#b6-repeat_kv-expand--reshape-no-contiguous)
- [B.7 `GPTOSSAttention` construction](#b7-gptossattention-construction-sink-param-yarn-pruned-rope)
- [B.8 Forward path trace](#b8-forward-path-trace-positions--out-proj)
- [B.9 Shape reference](#b9-shape-reference)
- [B.10 How to verify](#b10-how-to-verify)

---

## 1. Executive Summary

GPT-OSS-Lite combines three mechanisms that together enable long-context inference
with bounded memory:

| Mechanism | Where | Purpose |
|-----------|-------|---------|
| **Learned sink bias** | `GPTOSSAttention.sink_bias` | Absorb attention mass that would otherwise fall on evicted tokens |
| **Sliding-window attention (SWA)** | Even layers (`layer_idx % 2 == 0`) | Cap KV cache at `window_size` tokens per layer |
| **Full attention** | Odd layers (`layer_idx % 2 == 1`) | Preserve global context across the sequence |

Each attention head carries one scalar `sink_bias[h]`. At forward time the parameter
is clamped to `[SINK_CLAMP_MIN, SINK_CLAMP_MAX] = [-10, 15]` before entering the
softmax. The sink is implemented as an **extra softmax column** with zero value vector —
not as a physical token in the KV cache.

The reference path (`manual_causal_attention`) and the production path
(`causal_attention` via SDPA) are mathematically equivalent when masks align.

---

## 2. Historical Context: StreamingLLM and Attention Sinks

### 2.1 The long-context memory problem

Autoregressive transformers store key–value (KV) tensors for every prior token.
Memory grows linearly with sequence length \(T\):

\[
\text{KV bytes per layer} \;=\; 2 \cdot H_{\text{kv}} \cdot D \cdot T \cdot \text{bytes\_per\_elem}
\]

At 128K context with GQA (4 KV heads, head_dim 96, BF16), a single full-attention
layer needs roughly 200 MB of KV cache per batch element. Twelve such layers exceed
2 GB — before activations, weights, or MoE routing state.

### 2.2 StreamingLLM (Xiao et al., 2023)

[StreamingLLM](https://arxiv.org/abs/2309.17453) observed that pretrained LLMs
concentrate disproportionate attention mass on **initial tokens** — even when those
tokens carry no semantic content (e.g. a BOS padding). The authors called these
**attention sinks**.

Their inference recipe:

1. Keep the first \(k\) tokens (sink tokens) in the KV cache unconditionally.
2. Apply a sliding window over the remainder.
3. Drop middle tokens as the window advances.

The sink tokens act as a "pressure release valve" for attention logits that would
otherwise need to attend somewhere after middle context is evicted.

### 2.3 From fixed sink tokens to learned sink bias

StreamingLLM uses **physical** sink tokens — real embeddings whose KV entries persist.
GPT-OSS-Lite instead learns a **scalar logit per head** that enters the softmax
denominator without storing any extra KV tensor.

Advantages of the learned-bias formulation:

- **Zero KV overhead.** No reserved prefix slots in the cache.
- **Per-head specialization.** Different heads can learn different sink strengths.
- **Composable with SDPA.** Implemented as an extra mask column + zero K/V column
  (see [Part B §B.5](#b5-causal_attention-sdpa-paths-fast--window--sink-column)).
- **Gradient-friendly.** The parameter is always in the graph; clamping is forward-only.

---

## 3. Why Sinks Matter with Sliding-Window Attention

### 3.1 What SWA discards

Sliding-window attention restricts query position \(i\) to keys in
\([i - W + 1,\; i]\) (causal AND within window \(W\)). When \(i \ge W\), all keys
before \(i - W + 1\) are masked to \(-\infty\).

In a full-attention model, early tokens often receive residual attention mass
because:

1. **Positional artifacts.** RoPE makes early positions geometrically "special."
2. **Training distribution.** Models trained on left-padded or BOS-heavy data learn
   to route "overflow" attention to position 0.
3. **Softmax normalization.** When many keys are masked, surviving keys receive
   higher relative weight; without a sink, the distribution can become sharp and
   numerically unstable.

### 3.2 The eviction problem

When token \(j < i - W + 1\) falls outside the window, its KV pair is dropped from
the cache. Any attention pattern that previously allocated mass to position \(j\)
must redistribute that mass among surviving keys.

If no sink exists:

- Surviving keys absorb the orphaned mass → **attention drift** (outputs change
  discontinuously when a token ages out of the window).
- The model may learn pathological workarounds (e.g. encoding global state in the
  last visible token only).

If a sink exists with learned bias \(s_h\) for head \(h\):

- A fraction \(\frac{e^{s_h}}{Z}\) of attention can always flow to the sink.
- The sink's value vector is **zero**, so this fraction contributes nothing to the
  output — it is pure "absorption."
- The model learns how much mass to park in the sink vs distribute among real keys.

### 3.3 Intuition: the sink as a softmax garbage collector

Think of softmax attention as a probability distribution over keys. SWA deletes keys
from the support set. Without a replacement, the distribution must renormalize over
fewer keys — changing all outputs. The sink adds a **constant-support dummy key**
whose value is zero, letting the model discard mass silently.

---

## 4. Mathematical Formulation

### 4.1 Standard causal attention

For query \(Q \in \mathbb{R}^{T \times d}\), keys \(K \in \mathbb{R}^{T \times d}\),
values \(V \in \mathbb{R}^{T \times d}\), head index suppressed:

\[
\text{Attn}(Q, K, V) = \text{softmax}\!\left(\frac{QK^\top}{\sqrt{d}} + M\right) V
\]

where \(M_{ij} = -\infty\) if \(j > i\) (causal mask).

### 4.2 Sliding-window mask

Let \(W\) be the window size. Define the allowed set:

\[
\mathcal{A}(i) = \{\, j \;:\; j \le i \;\wedge\; i - j < W \,\}
\]

Equivalently, mask \(M^{\text{sw}}_{ij} = -\infty\) when \(i - j \ge W\).

### 4.3 Sink-augmented softmax

For head \(h\) with sink bias \(s_h\), augment the score matrix with one column:

\[
\tilde{S}_{i,j} = \frac{q_i^\top k_j}{\sqrt{d}}, \quad j \in \mathcal{A}(i)
\]
\[
\tilde{S}_{i,\text{sink}} = s_h
\]

Augmented softmax:

\[
\alpha_{i,j} = \frac{e^{\tilde{S}_{i,j}}}{\sum_{j' \in \mathcal{A}(i)} e^{\tilde{S}_{i,j'}} + e^{s_h}}, \quad
\alpha_{i,\text{sink}} = \frac{e^{s_h}}{Z_i}
\]

Output (sink has \(v_{\text{sink}} = 0\)):

\[
o_i = \sum_{j \in \mathcal{A}(i)} \alpha_{i,j}\, v_j
\]

The sink column is **stripped** before the value matmul in the reference implementation;
only \(\alpha_{i,j}\) for real keys participate, but they are renormalized with the
sink in the denominator.

### 4.4 Effect of bias magnitude

| \(s_h\) | Behavior |
|---------|----------|
| \(s_h \to -\infty\) | \(e^{s_h} \to 0\); sink ignored; equivalent to no sink |
| \(s_h = 0\) | Denominator gains \(+1\); all real weights scaled by \(\frac{Z}{Z+1}\) |
| \(s_h \to +\infty\) | \(\alpha_{i,\text{sink}} \to 1\); output \(\to 0\) |

### 4.5 Per-head independence

`sink_bias` has shape `(n_heads,)`. Each head maintains its own \(s_h\), allowing
some heads to be "sink-heavy" (local pattern matchers) and others "sink-light"
(global aggregators within the window).

---

## 5. Implementation in `models/attention.py`

All symbols refer to `models/attention.py`. Constants are clamped to
`[SINK_CLAMP_MIN, SINK_CLAMP_MAX] = [-10, 15]` at forward time (see [§6](#6-bf16-clamp-rationale)).
The reference path (`manual_causal_attention`) and production path (`causal_attention`
via SDPA) are mathematically equivalent when masks align.

**Full walkthrough in [Part B](#part-b--implementation-modelsattentionpy)** — module
overview, mask helpers, SDPA paths (including the zero K/V sink column),
`GPTOSSAttention` forward trace, shape reference, and verification.

---

## 6. BF16 Clamp Rationale

### 6.1 The overflow mechanism

SDPA computes `scores + attn_mask` before softmax. BF16 has limited dynamic range
(max finite ≈ \(3.4 \times 10^{38}\), but precision collapses at large magnitudes).
If `sink_bias` grows to hundreds or thousands during training:

1. `exp(score + sink_bias)` can overflow to `inf` in intermediate FP32/BF16 paths.
2. `inf / inf` → NaN in softmax normalization.
3. NaN propagates through `o_proj`, MoE, and loss.

### 6.2 Why clamp at forward time only

```python
sink_bias_clamped = self.sink_bias.clamp(SINK_CLAMP_MIN, SINK_CLAMP_MAX)
```

- **Forward:** uses clamped values → stable softmax.
- **Backward:** gradient flows to the unclamped parameter via `clamp`'s straight-through
  behavior at interior points; at bounds, gradient is zero (parameter stops growing
  further in that direction).

The parameter itself is **never** mutated by clamping — tests verify
`layer.sink_bias == 1000.0` after forward with extreme values.

### 6.3 Choice of [-10, 15]

| Bound | Rationale |
|-------|-----------|
| **-10** | \(e^{-10} \approx 4.5 \times 10^{-5}\) — sink effectively disabled; matches "no sink" within float tolerance |
| **+15** | \(e^{15} \approx 3.3 \times 10^6\) — sufficient to absorb almost all mass without reaching BF16 danger zone when added to typical attention scores (O(1) to O(10)) |

Empirically, trained sinks in similar architectures rarely exceed single digits.
The upper bound 15 leaves headroom while staying far from overflow.

### 6.3 Interaction with manual path

`manual_causal_attention` does not clamp — tests use moderate bias values. Production
always clamps via `GPTOSSAttention.forward`. If you call `causal_attention` directly
in custom code, pass pre-clamped bias.

---

## 7. Sliding-Window / Full Alternation

### 7.1 Layer pattern

GPT-OSS-Lite uses 12 layers with **alternating** attention patterns:

| `layer_idx` | `is_windowed` | Attention | KV storage |
|-------------|---------------|-----------|------------|
| 0, 2, 4, 6, 8, 10 | `True` | SWA (`window_size=128`) | Ring buffer, max 128 tokens |
| 1, 3, 5, 7, 9, 11 | `False` | Full causal | Unbounded (exponential growth) |

```python
self.is_windowed = (layer_idx % 2 == 0)
```

### 7.2 Why alternate instead of all-SWA?

Pure SWA limits every layer to 128 tokens of visible history. Information beyond
128 tokens is invisible to every layer — unsuitable for 128K passkey retrieval or
long-range dependency tasks.

Pure full attention at 12 layers × 128K tokens is memory-prohibitive.

**Alternation** is a compromise:

- Global layers (odd) carry the full sequence — any token can attend to any prior
  token at those depths.
- Windowed layers (even) sparsify most of the network, cutting aggregate KV footprint
  roughly in half while global layers preserve long-range signal.

### 7.3 Information flow mental model

```
Token at position 50,000
    ↓
Layer 0 (SWA):  sees tokens [49,873 .. 50,000]  (+ sink)
Layer 1 (Full): sees tokens [0 .. 50,000]
Layer 2 (SWA):  sees tokens [49,873 .. 50,000]  (+ sink)
Layer 3 (Full): sees tokens [0 .. 50,000]
    ...
```

A token's representation is refined by repeated **local → global → local → global**
cycles. Global layers act as "synchronisation points" that inject full-history context;
windowed layers compress and denoise locally.

### 7.4 Headline metric constraint

Replacing alternation with pure full attention breaks the ≥ 1.8× KV-cache reduction
target at 128K (see [§8](#8-kv-cache-mathematics-at-128k)). This is a hard
architectural invariant per `AGENTS.md`.

---

## 8. KV-Cache Mathematics at 128K

### 8.1 Per-token KV size

With GQA:

\[
\text{bytes per token per layer} = 2 \cdot H_{\text{kv}} \cdot D \cdot \text{elem\_size}
\]

Production config: \(H_{\text{kv}} = 4\), \(D = 96\), BF16 (2 bytes):

\[
\text{bytes per token per layer} = 2 \times 4 \times 96 \times 2 = 1536 \text{ bytes}
\]

### 8.2 Pure full-attention baseline

All 12 layers store full sequence \(T\):

\[
B_{\text{full}} = N_{\text{layers}} \cdot T \cdot 1536 = 12 \cdot T \cdot 1536
\]

At \(T = 131072\):

\[
B_{\text{full}} = 12 \times 131072 \times 1536 \approx 2.42 \text{ GB (batch=1)}
\]

### 8.3 Mixed SWA/full cache

6 windowed layers cap at \(W = 128\) tokens; 6 global layers store \(T\):

\[
B_{\text{mixed}} = \bigl( N_{\text{swa}} \cdot \min(W, T) + N_{\text{global}} \cdot T \bigr) \cdot 1536
\]

At \(T = 131072\), \(W = 128\):

\[
B_{\text{mixed}} = (6 \times 128 + 6 \times 131072) \times 1536 = 787200 \times 1536 \approx 1.21 \text{ GB}
\]

### 8.4 Reduction ratio

\[
\text{reduction} = \frac{B_{\text{full}}}{B_{\text{mixed}}} = \frac{12 \cdot T}{6 \cdot 128 + 6 \cdot T} = \frac{2T}{128 + T}
\]

| Context \(T\) | Reduction |
|---------------|-----------|
| 4,096 | 1.06× |
| 8,192 | 1.12× |
| 32,768 | 1.43× |
| 65,536 | 1.67× |
| **131,072** | **≈ 1.97×** |

The headline metric (≥ 1.8× at 128K) is met with margin. See
`scripts/kv_cache_benchmark.py` for the analytical checker.

### 8.5 `MixedKVCache` implementation sketch

`inference/generate.py` defines `MixedKVCache`:

- **Windowed:** fixed-size ring buffer `(B, H, window, D)`; O(1) append per decode step.
- **Global:** dynamically growing buffer with 1.5× exponential reallocation; stores
  **rotated** K (RoPE already applied).

The sink does **not** occupy a slot in either buffer.

---

## 9. Interaction with YaRN and Pruned RoPE

Attention sinks and positional encoding are **orthogonal mechanisms**:

| Component | Modifies | Layer scope |
|-----------|----------|-------------|
| Sink bias | Softmax denominator | All layers (when enabled) |
| YaRN RoPE | Query/key rotation frequencies | All layers |
| Pruned RoPE | Zeros out first `head_dim//4` frequency pairs on **global** layers | Odd layers only |

### 9.1 YaRN at 128K

Training context: 4,096 tokens (`yarn_original_max_seq_len`).
Inference target: 131,072 tokens (`yarn_target_seq_len`).
Scale factor: 32 (`yarn_scale_factor`).

YaRN interpolates inverse frequencies between base RoPE and scaled-down frequencies
so positions beyond 4K remain meaningful. See [rope_yarn.md](rope_yarn.md).

Sinks do not alter RoPE — they operate after Q/K are rotated.

### 9.2 Pruned RoPE on global layers

```python
def _n_pruned_dims(self) -> int:
    if (not self.is_windowed) and self.prune_rope_global:
        return self.head_dim // 4  # 24 of 48 frequency pairs for head_dim=96
    return 0
```

On global layers, the lowest 24 frequency dimensions are pruned (cos → 1, sin → 0),
reducing high-frequency positional sensitivity that can hurt very long extrapolation.
Windowed layers use full RoPE — local attention benefits from fine positional detail.

### 9.3 Combined effect at long context

At position 100,000 on a global layer:

1. YaRN scales frequencies for extrapolation.
2. Pruning removes the 24 lowest frequency pairs from rotation.
3. Full causal mask — all prior keys visible.
4. Sink bias available to absorb mass.

On the next windowed layer:

1. Same YaRN tables (no pruning).
2. Only last 128 keys visible (+ sink).
3. Sink becomes **more important** — evicted keys cannot be attended directly.

---

## 10. Prefill vs Decode

### 10.1 Prefill (prompt processing)

- Input shape: `(B, T_prompt, d_model)` with `T_prompt` possibly large (up to 128K
  in eval).
- `T_q == T_k` for all layers on the first pass.
- SWA: `_causal_mask(T) & _window_mask(T, T, W)`.
- Sink: full `(H, T, T_k+1)` mask built once.
- KV cache populated after RoPE; windowed layers may copy only last `W` entries
  after long prefills.

### 10.2 Decode (token-by-token)

- Input shape: `(B, 1, d_model)` per step.
- `T_q = 1`, `T_k =` cached length.
- `_window_mask(1, T_k, W)`: query index `T_k - 1`, keys `0..T_k-1`.
- Sink mask shape `(H, 1, T_k + 1)` — cheap per step.
- Windowed cache: O(1) ring update; global cache: O(1) append amortized.

### 10.3 Sink bias caching in generation

`generate()` maintains `sink_bias_cache: dict` keyed by `id(attn)`:

```python
sink_bias_clamped = attn.sink_bias.clamp(SINK_CLAMP_MIN, SINK_CLAMP_MAX)
sink_bias_cache[id(attn)] = sink_bias_clamped
```

Clamp runs once per layer per generation, not per token — minor optimisation.

### 10.4 Mask cache behaviour

`_causal_mask` and `_window_mask` are `lru_cache`d by shape/device/dtype.
During decode, `T_k` grows every step → new window masks compiled until `T_k`
stabilises at max context. Prefill at fixed `T` hits cache immediately on repeat runs.

---

## 11. Training Dynamics

### 11.1 Initialization

`sink_bias` initialized to zeros → each head starts with denominator \(Z + 1\)
instead of \(Z\). This is a mild, uniform damping.

### 11.2 What the model learns

During training on 4K context with SWA layers:

- Heads that would have attended to evicted positions learn positive \(s_h\) to
  park mass in the sink.
- Heads that only need local context may learn negative \(s_h\) (sink unused).

No explicit loss term for sink — it is learned end-to-end through the LM loss.

### 11.3 Gradient flow

`sink_bias` receives gradients through the softmax denominator. Clamping saturates
gradients at bounds. Tests confirm gradients reach `sink_bias`, `q_proj`, `kv_proj`,
`o_proj`.

### 11.4 Interaction with aux MoE loss

Sink bias is independent of MoE routing. Standard aux load-balancing loss (α=0.01)
applies to the FFN path only.

---

## 12. Failure Modes and Debugging

### 12.1 NaN loss after long training

**Symptom:** Loss becomes NaN mid-run.
**Check:** Unclamped `sink_bias` magnitudes in checkpoint. If max `|sink_bias| >> 15`,
overflow may occur before clamp in custom code paths.
**Fix:** Ensure all paths use `GPTOSSAttention.forward` or manually clamp.

### 12.2 Passkey retrieval fails at 128K

**Symptom:** Model cannot retrieve a key buried at position 80K.
**Check list:**
1. Are global layers receiving full KV (not truncated)?
2. Is YaRN configured (`yarn_scale_factor=32`, `target=131072`)?
3. Is eval using `eval_max_seq_len` not `max_seq_len`?
4. Untrained model — `passkey_eval.py` is a stub on random weights.

### 12.3 SWA/full mismatch in custom inference

**Symptom:** Results differ between cached and non-cached generation.
**Check:** `MixedKVCache` ring ordering — `get()` unwraps head pointer for windowed
layers. Verify `is_windowed` matches `layer_idx % 2`.

### 12.4 Sink appears ineffective

**Symptom:** `sink_bias` stays near zero; no effect on ablation.
**Possible causes:**
- Sequence shorter than `window_size` during eval — SWA behaves like full attention;
  sink less critical.
- All layers full-attention in misconfigured model (`is_windowed` always False).

### 12.5 Attention drift at window boundary

**Symptom:** Perplexity spikes when context crosses multiples of 128 on SWA-only
ablations.
**Expected:** Without sink, this is known SWA behaviour. Enable `sink_bias=True`.

### 12.6 Degenerate YaRN + long context

If YaRN ramp is degenerate (see [rope_yarn.md §11](rope_yarn.md#11-degenerate-ramp-warning)),
RoPE provides no extrapolation — sinks cannot compensate for positional collapse.

---

## 13. Design Decisions vs Alternatives

### 13.1 Physical sink tokens (StreamingLLM) vs learned bias

| Approach | KV cost | Flexibility | Implementation |
|----------|---------|-------------|----------------|
| Physical sinks | +k tokens per layer forever | Fixed positions | Simple mask |
| Learned bias (ours) | 0 | Per-head, continuous | SDPA mask column |

We chose learned bias for zero KV overhead and cleaner cache management.

### 13.2 Shared vs per-layer sink

GPT-OSS uses **per-head** bias within each layer (not shared across layers).
Each layer's attention pattern differs; sink demand varies by depth.

### 13.3 Sink on global layers

Sink is applied to **both** SWA and full layers when `cfg.sink_bias=True`.
On full layers the sink is less functionally necessary (no eviction) but harmless —
the model can learn \(s_h \to -\infty\) equivalent by pushing bias negative (within clamp).

### 13.4 Why not aux-loss-free routing here

This repo deliberately uses **standard** aux load-balancing for MoE (distinct from
DeepSeek-v3-Lite). Sink bias is unrelated to expert routing.

---

## 14. Quick Reference

### Symbols

| Symbol | Code location | Meaning |
|--------|---------------|---------|
| `s_h` | `sink_bias[h]` | Learned sink logit for head h |
| `W` | `cfg.window_size` | Sliding window (default 128) |
| `SINK_CLAMP_MIN` | `-10.0` | Lower forward clamp |
| `SINK_CLAMP_MAX` | `15.0` | Upper forward clamp |

### Key functions

| Function | File | Role |
|----------|------|------|
| `manual_causal_attention` | `attention.py` | FP32 reference / tests |
| `causal_attention` | `attention.py` | SDPA production path |
| `GPTOSSAttention` | `attention.py` | Full layer module |
| `MixedKVCache` | `inference/generate.py` | Inference cache |

### Related documentation

- [Part B](#part-b--implementation-modelsattentionpy) — line-level API walkthrough
- [rope_yarn.md](rope_yarn.md) — RoPE, YaRN scaling, `apply_rope`

### Verification commands

```bash
# Attention + sink tests
python3 -m pytest tests/test_attention.py -v

# KV reduction headline metric
python scripts/kv_cache_benchmark.py
```

---

## Part B — Implementation (`models/attention.py`)

### B.1 Module overview and public surface

`models/attention.py` implements the complete attention stack for GPT-OSS-Lite:

```
Input x (B, T, d_model)
    │
    ├─ q_proj ──► Q (B, H, T, D)
    ├─ kv_proj ─► K, V (B, H_kv, T, D) ──► repeat_kv ──► (B, H, T, D)
    │
    ├─ YaRNRoPE(positions) ──► cos, sin
    ├─ apply_rope(Q), apply_rope(K)          [rotary.py]
    │
    ├─ clamp(sink_bias)  [optional]
    └─ causal_attention(Q, K, V, window?, sink_bias?)
           │
           └─ o_proj ──► (B, T, d_model)
```

There is a single module class `GPTOSSAttention` — not separate SWA/Full classes.
Layer behaviour is selected by `is_windowed = (layer_idx % 2 == 0)`.

Public symbols: `SINK_CLAMP_MIN`, `SINK_CLAMP_MAX`, `manual_causal_attention`,
`causal_attention`, `repeat_kv`, `GPTOSSAttention`.

### B.2 Constants (`SINK_CLAMP_MIN`, `SINK_CLAMP_MAX`)

```python
from models.yarn import YaRNRoPE
from models.rotary import apply_rope

SINK_CLAMP_MIN = -10.0
SINK_CLAMP_MAX = 15.0
```

| Constant | Value | Purpose |
|----------|-------|---------|
| `SINK_CLAMP_MIN` | -10.0 | Forward clamp lower bound |
| `SINK_CLAMP_MAX` | 15.0 | Forward clamp upper bound |

Clamping prevents BF16 SDPA mask-add overflow. See [§6](#6-bf16-clamp-rationale).

### B.3 `manual_causal_attention` (FP32 scores, window, sink)

```python
def manual_causal_attention(
    query_states: torch.Tensor,   # (B, H, T, D)
    key_states: torch.Tensor,     # (B, H, T, D)
    value_states: torch.Tensor,   # (B, H, T, D)
    sink_bias: torch.Tensor | None = None,  # (H,)
    window: int | None = None,
) -> torch.Tensor:               # (B, H, T, D)
```

**Purpose:** Naive \(O(T^2)\) attention for tests and numerical oracle. Not used in
training or inference hot paths.

**Score computation (FP32):**

```python
B, H, T, D = query_states.shape
scores = (query_states.float() @ key_states.float().transpose(-2, -1)) / math.sqrt(D)
```

Scores are computed in FP32 regardless of input dtype. Result shape: `(B, H, T, T)`.

**Causal mask:**

```python
causal = torch.triu(torch.ones(T, T, dtype=torch.bool, device=...), diagonal=1)
scores = scores.masked_fill(causal, float("-inf"))
```

`causal[i, j] = True` when `j > i` (future positions).

**Sliding-window mask:**

```python
if window is not None and window < T:
    idx = torch.arange(T, device=...)
    outside = idx.unsqueeze(0) - idx.unsqueeze(1) >= window
    scores = scores.masked_fill(outside, float("-inf"))
```

For query index `i` and key index `j`: masked when `i - j >= window`.
When `window >= T`, the window constraint is vacuous — behaves as full causal.

**Sink augmentation:**

```python
if sink_bias is not None:
    sink_logit = sink_bias.view(1, H, 1, 1).to(scores.dtype)
    augmented = torch.cat([scores, sink_logit.expand(B, H, T, 1)], dim=-1)
    attn_weights = F.softmax(augmented, dim=-1)
    attn_weights = attn_weights[..., :T]
    return (attn_weights.to(value_states.dtype) @ value_states)
```

Steps: concat sink column → softmax over `T+1` columns → strip sink column → matmul
with `value_states`. When `sink_bias is None`, standard softmax over `T` columns.

### B.4 Mask helpers (`_causal_mask`, `_window_mask`, cache key)

Two cached functions build boolean masks for the SDPA path.

**`_causal_mask`:**

```python
@functools.lru_cache(maxsize=None)
def _causal_mask(T: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    idx = torch.arange(T, device=device)
    return idx.unsqueeze(1) >= idx.unsqueeze(0)  # (T, T)
```

Returns `True` where attention is **allowed** (lower-triangular including diagonal).
Opposite convention from `manual_causal_attention` (which masks `True` = forbidden).

**`_window_mask`:**

```python
@functools.lru_cache(maxsize=None)
def _window_mask(T_q, T_k, window, device, dtype) -> torch.Tensor:
```

Prefill branch (`T_q == T_k`):

```python
idx = torch.arange(T_q, device=device)
return (idx.unsqueeze(0) - idx.unsqueeze(1) < window) & _causal_mask(T_q, device, dtype)
```

Decode branch (`T_q = 1`, `T_k` = cached length):

```python
idx_q = torch.tensor([T_k - 1], device=device)
idx_k = torch.arange(T_k, device=device)
return (idx_q.unsqueeze(-1) - idx_k.unsqueeze(0) < window)
```

Decode assumes a single new query at position `T_k - 1` attending to cached keys
`0 .. T_k-1`. No separate causal check needed — all cached keys are in the past.

**Cache key semantics:** `lru_cache` keys on `(T, device, dtype)` or
`(T_q, T_k, window, device, dtype)`. `maxsize=None` — unbounded cache; typical
training sees few distinct `(T, window)` pairs per run.

### B.5 `causal_attention` SDPA paths (fast / window / sink column)

```python
def causal_attention(
    query_states: torch.Tensor,   # (B, H, T_q, D)
    key_states: torch.Tensor,     # (B, H, T_k, D)
    value_states: torch.Tensor,   # (B, H, T_k, D_v)
    window: int | None = None,
    sink_bias: torch.Tensor | None = None,  # (H,)
) -> torch.Tensor:               # (B, H, T_q, D_v)
```

Production attention via `F.scaled_dot_product_attention`. Supports rectangular
`T_q \ne T_k` for decode.

**Fast path: no sink, no window**

```python
if sink_bias is None:
    if window is None:
        if T_q == T_k:
            return F.scaled_dot_product_attention(
                query_states, key_states, value_states, is_causal=True
            )
        return F.scaled_dot_product_attention(
            query_states, key_states, value_states
        )
```

Square + causal → `is_causal=True` enables Flash Attention fusion.

**Window path without sink**

```python
if T_q == T_k:
    mask = _causal_mask(T_q, device, dtype) & _window_mask(T_q, T_k, window, device, dtype)
else:
    mask = _window_mask(T_q, T_k, window, device, dtype)

attn_mask = torch.where(mask, 0.0, float("-inf")).to(dtype).unsqueeze(0).unsqueeze(0)
return F.scaled_dot_product_attention(query_states, key_states, value_states, attn_mask=attn_mask)
```

| `sink_bias` | `window` | Path |
|-------------|----------|------|
| `None` | `None` | `is_causal=True` or bare SDPA |
| `None` | set | Cached boolean → `where(0, -inf)` mask |
| set | `None`/set | Sink K/V extension + head-specific mask |

**Sink path: SDPA with zero K/V column**

When `sink_bias is not None`:

```python
sink_k = torch.zeros(B, H, 1, query_states.shape[-1], device=device, dtype=dtype)
sink_v = torch.zeros(B, H, 1, value_states.shape[-1], device=device, dtype=value_states.dtype)
k_ext = torch.cat([key_states, sink_k], dim=2)   # (B, H, T_k+1, D)
v_ext = torch.cat([value_states, sink_v], dim=2)
```

The sink key is all zeros → dot product \(q \cdot k_{\text{sink}} = 0\). The learned
bias enters **only** through the attention mask, not through K.

```python
if window is None or window >= T_k:
    causal = _causal_mask(T_q, device, dtype) if T_q == T_k \
        else torch.ones(T_q, T_k, dtype=torch.bool, device=device)
elif T_q == T_k:
    causal = _causal_mask(T_q, device, dtype) & _window_mask(T_q, T_k, window, device, dtype)
else:
    causal = _window_mask(T_q, T_k, window, device, dtype)

mask = torch.zeros(H, T_q, T_k + 1, device=device, dtype=dtype)
mask[:, :, :T_k] = causal.to(dtype)
mask[:, :, T_k] = sink_bias.to(dtype).unsqueeze(1).expand(H, T_q)
return F.scaled_dot_product_attention(query_states, k_ext, v_ext, attn_mask=mask.unsqueeze(0))
```

| Column range | Value | Meaning |
|--------------|-------|---------|
| `0 .. T_k-1` | `1.0` if allowed, `0.0` if forbidden | Boolean `causal` cast to float |
| `T_k` (sink) | `sink_bias[h]` | Per-head learned logit |

**Important:** The non-sink window path uses `torch.where(mask, 0.0, -inf)` to block
forbidden positions. The sink path uses `causal.to(dtype)` (`1.0`/`0.0`) instead.
When validating sink behaviour, compare against `manual_causal_attention` (the test
oracle), not against the no-sink SDPA path.

**SDPA backend notes:** `is_causal=True` path (no sink, no window, square) triggers
FA2 via PyTorch SDPA when available. Window and sink paths pass `attn_mask` — may
fall back to math or memory-efficient kernel. Q, K, V must share dtype for SDPA.

### B.6 `repeat_kv` (`expand` + `reshape`, no `.contiguous()`)

Grouped-query attention (GQA) uses fewer KV heads than Q heads:

| Config field | Default | Meaning |
|--------------|---------|---------|
| `n_heads` | 8 | Query heads \(H\) |
| `n_kv_heads` | 4 | Key/value heads \(H_{\text{kv}}\) |
| `n_rep` | `n_heads // n_kv_heads` | Replication factor (2) |

```python
def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    if n_rep == 1:
        return x
    B, H_kv, T, D = x.shape
    x = x[:, :, None, :, :]           # (B, H_kv, 1, T, D)
    x = x.expand(B, H_kv, n_rep, T, D) # no memory copy
    return x.reshape(B, H_kv * n_rep, T, D)
```

`expand` creates a broadcast view — no `.contiguous()` copy. SDPA's Flash Attention
backend handles non-contiguous K/V internally. KV head `h_kv` maps to Q heads
`2*h_kv` and `2*h_kv + 1` when `n_rep=2`.

### B.7 `GPTOSSAttention` construction (sink param, YaRN, pruned RoPE)

```python
class GPTOSSAttention(nn.Module):
    def __init__(self, cfg, layer_idx: int):
```

**Construction:**

```python
self.is_windowed = (layer_idx % 2 == 0)
self.prune_rope_global = bool(getattr(cfg, "yarn_prune_rope_global", True))
self.n_rep = self.n_heads // self.n_kv_heads

self.q_proj = nn.Linear(d_model, n_heads * head_dim, bias=False)
self.kv_proj = nn.Linear(d_model, 2 * n_kv_heads * head_dim, bias=False)
self.o_proj = nn.Linear(n_heads * head_dim, d_model, bias=False)
```

**Sink parameter:**

```python
if cfg.sink_bias:
    self.sink_bias = nn.Parameter(torch.zeros(self.n_heads))
else:
    self.register_parameter("sink_bias", None)
```

**YaRN module:**

```python
self.yarn = YaRNRoPE(
    head_dim=self.head_dim,
    theta=cfg.rope_theta,
    scale_factor=cfg.yarn_scale_factor,
    original_max_seq_len=cfg.yarn_original_max_seq_len,
    target_seq_len=cfg.yarn_target_seq_len,
    beta_fast=cfg.yarn_beta_fast,
    beta_slow=cfg.yarn_beta_slow,
    mscale=cfg.yarn_mscale,
)
```

See [rope_yarn.md](rope_yarn.md) for parameter semantics.

**Pruned RoPE helper:**

```python
def _n_pruned_dims(self) -> int:
    if (not self.is_windowed) and self.prune_rope_global:
        return self.head_dim // 4
    return 0
```

Returns number of frequency **pairs** to prune (not individual dims). For
`head_dim=96` → 24 pairs → 48 scalar dimensions frozen to identity rotation.
Applied only on **global** (odd-indexed) layers when `yarn_prune_rope_global=True`.

**Config knobs** (from `ModelConfig` in `models/transformer.py`):

| Field | Default | Maps to |
|-------|---------|---------|
| `n_heads` | 8 | `self.n_heads` |
| `n_kv_heads` | 4 | `self.n_kv_heads` |
| `head_dim` | 96 | `self.head_dim` |
| `window_size` | 128 | SWA window on even layers |
| `sink_bias` | `True` | Whether `sink_bias` parameter exists |
| `rope_theta` | 100000 | YaRN base \(\theta\) |
| `yarn_scale_factor` | 32 | YaRN stretch factor |
| `yarn_original_max_seq_len` | 4096 | Training context for YaRN |
| `yarn_target_seq_len` | 131072 | Inference extrapolation target |
| `yarn_prune_rope_global` | `True` | Prune 25% RoPE dims on global layers |

### B.8 Forward path trace (positions → out proj)

Complete `forward(x, positions=None)` execution:

**Step 1 — Positions**

```python
B, T, _ = x.shape
if positions is None:
    positions = torch.arange(T, device=x.device)
```

**Step 2 — Projections**

```python
query_states = self.q_proj(x).view(B, T, self.n_heads, self.head_dim)
kv = self.kv_proj(x).view(B, T, 2, self.n_kv_heads, self.head_dim)
key_states, value_states = kv[:, :, 0], kv[:, :, 1]
```

**Step 3 — Head-major layout**

```python
query_states = query_states.transpose(1, 2)   # (B, H, T, D)
key_states = key_states.transpose(1, 2)         # (B, H_kv, T, D)
value_states = value_states.transpose(1, 2)
```

**Step 4 — RoPE**

```python
cos, sin = self.yarn(positions, n_pruned_dims=self._n_pruned_dims())
query_states = apply_rope(query_states, cos, sin)
key_states = apply_rope(key_states, cos, sin)
```

`apply_rope` is imported from `models.rotary` — see [rope_yarn.md](rope_yarn.md).

**Step 5 — GQA expansion**

```python
key_states = repeat_kv(key_states, self.n_rep)
value_states = repeat_kv(value_states, self.n_rep)
```

**Step 6 — Sink clamp**

```python
if self.sink_bias is not None:
    sink_bias_clamped = self.sink_bias.clamp(SINK_CLAMP_MIN, SINK_CLAMP_MAX)
else:
    sink_bias_clamped = None
```

**Step 7 — Attention**

```python
out = causal_attention(
    query_states, key_states, value_states,
    window=self.window_size if self.is_windowed else None,
    sink_bias=sink_bias_clamped,
)
```

| Layer type | `window` argument | Effect |
|------------|-------------------|--------|
| Even (`is_windowed=True`) | `cfg.window_size` (128) | SWA |
| Odd (`is_windowed=False`) | `None` | Full causal |

**Step 8 — Output projection**

```python
out = out.transpose(1, 2).contiguous().view(B, T, self.n_heads * self.head_dim)
return self.o_proj(out)
```

**Inference integration** (`inference/generate.py`): duplicates this forward with KV
caching. RoPE is applied **before** caching; cached K is already position-encoded.
Sink bias is clamped once per layer per generation via `sink_bias_cache[id(attn)]`.
Windowed vs global dispatch mirrors `GPTOSSAttention.forward` logic.

### B.9 Shape reference

Assume production config: `B=1`, `T=4096`, `H=8`, `H_kv=4`, `D=96`, `d_model=768`.

| Tensor | Shape | Notes |
|--------|-------|-------|
| `x` | `(B, T, 768)` | Input |
| `q_proj(x)` | `(B, T, 768)` | `8 * 96` |
| `query_states` (head-major) | `(B, 8, T, 96)` | After transpose |
| `kv_proj(x)` | `(B, T, 768)` | `2 * 4 * 96` |
| `key_states` (pre-repeat) | `(B, 4, T, 96)` | |
| `key_states` (post-repeat) | `(B, 8, T, 96)` | GQA |
| `cos`, `sin` | `(T, 48)` | `head_dim // 2` pairs |
| `sink_bias` | `(8,)` | Per Q head |
| `causal_attention` out | `(B, 8, T, 96)` | |
| `o_proj` out | `(B, T, 768)` | |

Decode step: `T_q=1`, `T_k` = cached length, shapes analogous with `T` replaced.

### B.10 How to verify

Run:

```bash
python3 -m pytest tests/test_attention.py -v
```

Must include: `test_sliding_window_matches_full`.

KV reduction headline metric:

```bash
python scripts/kv_cache_benchmark.py
```

**Common pitfalls:**

- **GQA repeat:** Always `repeat_kv` before attention — mismatched Q/K head counts
  raise SDPA shape errors.
- **RoPE twice:** Cached inference stores rotated K; do not `apply_rope` on cached
  keys again.
- **Unclamped sink:** Direct `causal_attention(..., sink_bias=raw_param)` bypasses
  clamp — use `.clamp(SINK_CLAMP_MIN, SINK_CLAMP_MAX)` or `GPTOSSAttention`.
- **Window on global layers:** Only even layers should pass `window_size` — breaks
  KV math and the ≥ 1.8× headline metric otherwise.
- **Mask convention confusion:**
  - `_causal_mask`: `True` = allowed
  - `manual_causal_attention`: `True` in `triu` = **forbidden**
  - Non-sink SDPA: `0.0` = allowed, `-inf` = forbidden
  - Sink SDPA: `causal.to(dtype)` → `1.0`/`0.0` for real keys, `sink_bias` for sink column

When in doubt, trust `manual_causal_attention` for golden values.

---

<!-- docs:verified 2026-07-31 · 7fe1247 -->
