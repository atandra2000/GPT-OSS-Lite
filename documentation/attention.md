# Attention Mechanisms in GPT-OSS-Lite

> **Source:** `models/attention.py`
> **Authoritative companion:** [`ATTENTION_SINKS.md`](ATTENTION_SINKS.md) — the
> 600-line theoretical deep-dive on the learned sink bias. This file documents the
> *implementation*; that file documents the *idea*. It now lives alongside this
> file in `documentation/`.

---

## 1. Overview

Attention is the load-bearing component of GPT-OSS-Lite — it is where the model's
two headline innovations live: **sliding-window / full attention alternation**
(which delivers the 2× KV-cache reduction at 128K context) and the **learned
attention-sink bias** (which stabilises long-context attention). This module
exposes three layers of API, from low-level reference kernels up to the full
`nn.Module`:

1. **`manual_causal_attention`** — a naive, explicit `O(T²)` reference
   implementation. It is *not* on any production path; it exists so the test suite
   can check the SDPA fast paths against ground truth that is obviously correct.
   It supports the sink bias and an optional sliding-window mask.
2. **`sliding_window_attention` / `full_causal_attention`** — the production
   paths, built on `F.scaled_dot_product_attention` (SDPA). Each supports the
   learned sink bias, and `sliding_window_attention` takes a bounded window.
3. **`GPTOSSAttention`** — the `nn.Module` a block actually instantiates. It
   wires together GQA projections, the YaRN RoPE, the per-head sink-bias
   parameter, and the **alternating layer pattern** (even layers slide, odd
   layers attend globally).

The split is deliberate: keeping the reference path separate from the
optimised path means correctness regressions in the SDPA code are caught by a
comparison against something a human can read line-by-line, not against another
fused kernel.

---

## 2. The math each path implements

### 2.1 Plain scaled dot-product attention

For query/key/value tensors `Q, K, V ∈ ℝ^{B×H×T×D}`:

```
scores  = (Q · Kᵀ) / √D                      # (B, H, T, T)
attn    = softmax(scores)                     # row-wise, over the key axis
out     = attn · V                            # (B, H, T, D)
```

Causal masking sets `scores[i, j] = -∞` for `j > i` so a position cannot attend
to the future. The `1/√D` factor keeps the pre-softmax scores in a sane range
as `D` grows (here `D = head_dim = 96`, so `√D ≈ 9.8`).

### 2.2 Sliding-window attention (SWA)

SWA adds a second mask on top of causality: position `i` may only attend to
keys in `[i − window + 1, i]`. For `window = 128` and `T ≫ 128`, each query row
has at most 128 finite entries. This is what makes the windowed layers' KV
cache bounded by `window` rather than `T` — the architectural claim of GPT-OSS.

```
mask[i, j] = -∞  if  j > i              (causality)
mask[i, j] = -∞  if  i − j ≥ window      (sliding window)
```

### 2.3 The learned sink bias (off-by-one / StreamingLLM trick)

The sink bias is a per-head scalar `s_h` (one float per query head, `H = 8`
values total). It is injected into the softmax *denominator* as if there were
an extra "virtual" key whose pre-softmax logit is `s_h`:

```
Z          = exp(s_h) + Σ_j exp(scores[i, j])     # augmented denominator
attn[i, j] = exp(scores[i, j]) / Z
```

Equivalently, append a synthetic `(sink_key, sink_value)` pair — with the
key's logit set to `s_h` and its value zeroed — and let the normal softmax
absorb it. The synthetic value being zero means the sink contributes mass to
the denominator (lowering all other weights uniformly) but adds nothing to the
numerator-weighted output. This is the StreamingLLM / "off-by-one" trick:
the sink absorbs the otherwise-unbound "null attention" mass that the
first few positions accumulate in a causal LM, which is what destabilises
naive long-context extrapolation. The GPT-OSS twist is that `s_h` is
**learned** (initialised to 0, trained by backprop) rather than fixed.

### 2.4 Grouped Query Attention (GQA)

`n_heads = 8` query heads share `n_kv_heads = 4` key/value heads, so each KV
head serves `n_rep = 8 / 4 = 2` query heads. This halves the KV-cache size and
the KV-bandwidth pressure of attention without measurably hurting quality at
this scale. `repeat_kv` broadcasts each KV head to its `n_rep` query heads.

---

## 3. `manual_causal_attention` — the reference path

```python
def manual_causal_attention(query_states, key_states, value_states,
                             sink_bias=None, window=None) -> torch.Tensor
```

A literal implementation of §2. It computes scores in **FP32** regardless of
the input dtype (`query_states.float() @ key_states.float().transpose(-2, -1)`),
builds the causal mask with `torch.triu`, optionally applies the sliding-window
mask, then either runs `F.softmax` directly or — when a sink bias is given —
concatenates a `(B, H, T, 1)` sink-logit column onto the scores, runs softmax
over the augmented `(T+1)` columns, and slices back to the first `T` columns
before multiplying by `V`.

**Why FP32 here?** Under BF16 inputs, the score matmul and softmax are the two
places where catastrophic cancellation and underflow bite. SDPA handles this
internally; the manual path does not, so it upcasts explicitly. The final
`attn · V` matmul is cast back to `value_states.dtype` so the output dtype
matches the production path. This is the reference path used by
`tests/test_attention.py` to certify the SDPA paths.

---

## 4. `sliding_window_attention` — the SWA production path

```python
def sliding_window_attention(query_states, key_states, value_states,
                              window=128, sink_bias=None) -> torch.Tensor
```

Built on `F.scaled_dot_product_attention` (SDPA), which dispatches to
FlashAttention / memory-efficient fused kernels on CUDA and a math fallback on
CPU. Two sub-paths:

### 4.1 No sink bias — pure SWA via cached mask

When `sink_bias is None` and `T_q == T_k` (training), the call reuses
`_get_sliding_window_mask(T, window, device, dtype)` and passes it as
`attn_mask`. SDPA's native `is_causal` path is *not* used here because the
sliding window is stricter than plain causality; instead the full
causal-and-window mask is precomputed once and cached (see OPT-1 in
[`OPTIMIZATIONS.md`](OPTIMIZATIONS.md)).

For the cross-attention case (`T_q < T_k`, as during cached decode), the query
position is treated as `T_k − 1` (the most recent key) and the window mask is
built on the fly — the cached mask key assumes self-attention.

### 4.2 With sink bias — virtual sink key + cached base mask

The sink case physically extends the K/V sequence with a synthetic
`(B, H, 1, D)` pair — `sink_k = 0`, `sink_v = 0` — so SDPA's softmax naturally
incorporates `exp(s_h)` into the denominator. The attention mask becomes
`(H, T_q, T_k + 1)`: the first `T_k` columns are the cached causal+window base
mask (zeros in the sink column), and the last column holds the per-head sink
logit, overwritten on each call from the live `sink_bias`:

```python
mask_with_sink = mask.clone()
mask_with_sink[:, :, T_k] = sink_bias.to(q_dtype).unsqueeze(1).expand(H, T_q)
```

The clone is the only per-call allocation; the base mask is reused across
calls (OPT-3). This is what makes the sink path only ~5% slower than the
no-sink path at production scale rather than ~30%.

---

## 5. `full_causal_attention` — the global-attention production path

```python
def full_causal_attention(query_states, key_states, value_states,
                           sink_bias=None) -> torch.Tensor
```

Three sub-paths, in order:
1. **No sink, self-attention** (`T_q == T_k`): the cheapest possible call —
   `SDPA(..., is_causal=True)` lets the kernel build the causal mask internally
   with no Python mask tensor at all.
2. **No sink, cross-attention** (`T_q < T_k`, cached decode): plain SDPA with no
   mask — the cache is already causal by construction, so no masking is needed.
3. **Sink bias (any shape):** delegates to `sliding_window_attention` with
   `window = max(T_q, T_k)`. With a window at least as large as the sequence, the
   sliding-window mask degenerates to plain causality, so this reuses the
   well-tested sink+mask code path instead of maintaining a second sink
   implementation.

This is why global (odd) layers are *more* expensive than windowed layers at
long context: they cache the full `T` keys, while windowed layers cap at
`window = 128`.

---

## 6. `repeat_kv` — the GQA broadcast helper

```python
def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor
```

Broadcasts each of the `H_kv` KV heads to `n_rep` query heads via
`expand + reshape`. The deliberately dropped `.contiguous()` (OPT-2) is the
whole point of this function's existence as a named helper: SDPA's flash path
operates on strided tensors and calls `.contiguous()` internally *only when
needed*, so the previous `expand + reshape + contiguous()` was allocating a
fresh `(B, H_kv × n_rep, T, D)` tensor on every forward — 12 times per step —
for nothing. The current version hands SDPA a view it can fuse into the
underlying matmul. A tiny `.contiguous()` is retained only on the no-sink
training path where SDPA's own internal copy would dominate anyway.

---

## 7. `GPTOSSAttention` — the layer module

```python
class GPTOSSAttention(nn.Module):
    def __init__(self, cfg: dict, layer_idx: int)
    def forward(self, x, positions=None) -> torch.Tensor
```

### 7.1 Projections

A fused `kv_proj` produces both K and V in one matmul
(`2 * n_kv_heads * head_dim` output features, then split along the feature
axis), and separate `q_proj` / `o_proj` linears. No biases anywhere — the
GPT-OSS convention, and a small parameter saving.

### 7.2 The alternating layer pattern

```python
self.is_windowed = (layer_idx % 2 == 0)
```

- **Even layers (0, 2, 4, …) → sliding-window attention** (`window = 128`).
- **Odd layers (1, 3, 5, …) → full causal attention.**

The alternation is the architectural headline: half the layers' KV caches are
bounded by `window`, the other half grow with `T`. At 128K context this is
exactly a 2× cache reduction versus pure full-attention GQA (verified by
`scripts/kv_cache_benchmark.py`). A pure-SWA stack would be cheaper still but
loses global information flow; the alternation gives every token a path to the
global layers within two hops, preserving long-range signal.

### 7.3 The sink-bias parameter and its forward-time clamp

```python
self._sink_clamp_min = -10.0
self._sink_clamp_max =  15.0
self.sink_bias = nn.Parameter(torch.zeros(self.n_heads))   # one scalar per head
```

The sink bias is initialised to **0** (so the sink starts disabled —
`exp(0) = 1` just adds a uniform constant to the denominator) and trained by
backprop. In `forward`, the *trained* parameter is left untouched (its gradient
flows normally) but a **detached, clamped copy** is what actually enters the
attention math:

```python
sink_bias_clamped = self.sink_bias.clamp(self._sink_clamp_min, self._sink_clamp_max)
```

The clamp bounds are chosen for BF16 safety:
- `exp(+15) ≈ 3.27 × 10⁶` — the largest power-of-e sink that stays comfortably
  inside BF16's `~3.4 × 10⁷` max representable value, with headroom for the
  `Σ exp(scores)` term that shares the denominator.
- `exp(−10) ≈ 4.5 × 10⁻⁵` — small enough to act as a *disabled* sink (its
  contribution to the denominator is negligible), the floor below which a
  learned sink is effectively off.

This is a **forward-only numerical-stability guard against BF16 SDPA mask-add
overflow**, not a regulariser — the unclamped `nn.Parameter` keeps its gradient
so the optimiser still sees the true value. See [`ATTENTION_SINKS.md`](ATTENTION_SINKS.md)
for the full theoretical treatment of *why* the sink needs this guard.

### 7.4 Pruned RoPE on global layers

```python
n_pruned_dims = 0
if (not self.is_windowed) and cfg.get("yarn_prune_rope_global", True):
    n_pruned_dims = self.head_dim // 4      # 25% of dims on global layers
```

Global (odd) layers zero out the lowest-frequency 25% of RoPE dimensions (set
`cos=1, sin=0` → identity rotation), reducing over-rotation at 128K context.
Windowed layers never prune. See [`yarn.md`](yarn.md) and [`rotary.md`](rotary.md).

### 7.5 Forward flow

```
x (B,T,d_model)
  │ q_proj → (B,T, n_heads·head_dim) → transpose → (B,H,T,D)   [Q]
  │ kv_proj → split → (B,H_kv,T,D) × 2                           [K,V]
  │ YaRNRoPE(positions, n_pruned) → (cos, sin)
  │ apply_rope(Q, cos, sin); apply_rope(K, cos, sin)
  │ repeat_kv(K, n_rep); repeat_kv(V, n_rep)                     [GQA broadcast]
  │ sink_bias.clamp(-10, 15)                                     [stability guard]
  │ → sliding_window_attention  (if is_windowed)
  │ → full_causal_attention      (else)
  ▼
  o_proj → (B,T,d_model)
```

---

## 8. Design rationale & rejected alternatives

| Decision | Rationale | Rejected alternative |
|---|---|---|
| Two attention paths (manual + SDPA) | Test the fused kernel against auditable ground truth | One path only — would make correctness bugs invisible |
| Cached sliding-window mask | Mask shape is invariant during training; rebuild is pure waste | Rebuild per forward — 40% of SWA call time on CPU |
| Sink bias = virtual key + value 0 | Lets SDPA's fused softmax absorb the sink natively | Manual denominator patch — defeats the point of SDPA |
| Forward-time clamp on a detached copy | BF16 safety without disturbing the gradient | Clamp the Parameter in place — kills the gradient signal |
| `repeat_kv` without `.contiguous()` | SDPA accepts strided tensors | Always-contiguous — wastes a full `(B,H,T,D)` copy per layer |
| Alternating SWA/full (not all-SWA) | Global layers preserve long-range signal | All-SWA stack — cheaper but loses global information |
| `kv_proj` fused | One matmul instead of two | Separate k_proj/v_proj — 2× the launch overhead |

---

## 9. Edge cases & pitfalls

- **`window >= T`**: the sliding-window mask degenerates to plain causality.
  `full_causal_attention`'s sink path relies on exactly this when it delegates
  to `sliding_window_attention` with `window = max(T_q, T_k)`.
- **`T_q < T_k` (cached decode)**: the windowed mask builder assumes the query
  sits at the *most recent* key position (`T_k − 1`). This is correct for
  autoregressive decode but would be wrong for arbitrary cross-attention — the
  module is only ever used autoregressively.
- **Sink-bias cache key**: the SWA+sink mask is keyed by
  `("sink_swa", T_q, T_k, H, window, device, dtype)`. The sink *value* is not
  in the key (it varies with training), so the cached base mask has zeros in
  the sink column and is overwritten per call. If you ever make the sink bias
  shape-dependent in a new way, update the key.
- **`attn_impl: "manual"`** in the config routes the production path through
  `manual_causal_attention` instead of SDPA. This is ~10× slower and exists
  only for debugging — never ship it.

---

## Implementation notes (extracted from code review)

- **Sink bias clamping at forward time**: `GPTOSSAttention.forward` clamps
  `sink_bias` to `[-10, 15]` on a detached copy before passing it into the
  attention path. The clamp bounds are chosen so `exp(15) ≈ 3.3e6` stays
  within BF16 range and `exp(-10) ≈ 4.5e-5` acts as a "disabled" sink. The
  unclamped `nn.Parameter` retains its gradient — the clamp is a forward-only
  numerical-stability guard against BF16 SDPA mask-add overflow when the
  trained parameter grows large. See `ATTENTION_SINKS.md` for the full
  theoretical treatment.
- **Mask caching**: the sliding-window causal mask is cached module-level in
  `_SLIDING_WINDOW_MASK_CACHE`, keyed by `(T, window, device, dtype)`. The
  sink-bias SWA mask is cached under the key `("sink_swa", T_q, T_k, H,
  window, device, dtype)` — the trailing sink-bias column is overwritten
  per-call (it varies with training) but the rest of the mask is reused.
  This amortises the per-forward mask build to one kernel launch.
- **FP32 accumulation in the manual path**: `manual_causal_attention` casts
  `query_states` and `key_states` to FP32 for the score matmul and runs
  softmax in FP32, casting back to `value_states.dtype` only for the final
  value matmul. This is the reference path used in tests; the SDPA path
  handles FP32 accumulation internally.
- **`repeat_kv` uses expand + reshape (no `.contiguous()`)**: SDPA's flash
  path operates on strided tensors and calls `.contiguous()` internally only
  when needed. Dropping the explicit `.contiguous()` avoids a fresh
  `(B, H_kv × n_rep, T, D)` allocation on every forward.
- **`MixedKVCache` ring + exponential growth**: windowed layers use a
  fixed-size ring buffer of length `window` (O(window) append, O(1) amortised
  decode step). Global layers use an exponentially-growing buffer
  (1.5× growth capped at `_GLOBAL_CAP_TOKENS = 4_000_000`), giving O(N)
  total work over an N-token generation instead of O(N²) from `torch.cat`
  on every step.
- **Pre-allocated `generate` output**: `generate()` allocates the full
  `(B, T_prompt + max_new_tokens)` output tensor once and writes new tokens
  in place, avoiding the O(T²) `torch.cat` per step.
- **Fast T=1 RoPE path**: `YaRNRoPE.forward` special-cases
  `positions.numel() == 1` to skip `torch.outer` and do a single scalar
  multiply — saves a kernel launch per decode step across all 12 layers.