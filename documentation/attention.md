# Attention Module — Implementation Walkthrough

> Line-level guide to `models/attention.py`. For the conceptual deep dive on sink
> bias and SWA/full alternation, see [ATTENTION_SINKS.md](ATTENTION_SINKS.md).
> Positional encoding is covered in [yarn.md](yarn.md) and [rotary.md](rotary.md).

---

## Table of Contents

1. [Module Overview](#1-module-overview)
2. [Constants and Imports](#2-constants-and-imports)
3. [`manual_causal_attention`](#3-manual_causal_attention)
4. [Mask Helpers](#4-mask-helpers)
5. [`causal_attention`](#5-causal_attention)
6. [Sink Path: SDPA with Zero K/V Column](#6-sink-path-sdpa-with-zero-kv-column)
7. [`repeat_kv` — GQA Head Expansion](#7-repeat_kv--gqa-head-expansion)
8. [`GPTOSSAttention`](#8-gptossattention)
9. [Forward Path Trace](#9-forward-path-trace)
10. [Shape Reference](#10-shape-reference)
11. [SDPA Backend Notes](#11-sdpa-backend-notes)
12. [Inference Integration](#12-inference-integration)
13. [Config Knobs](#13-config-knobs)
14. [Common Pitfalls](#14-common-pitfalls)

---

## 1. Module Overview

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

---

## 2. Constants and Imports

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

Clamping prevents BF16 SDPA mask-add overflow. See
[ATTENTION_SINKS.md §6](ATTENTION_SINKS.md#6-bf16-clamp-rationale).

---

## 3. `manual_causal_attention`

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

### 3.1 Score computation (FP32)

```python
B, H, T, D = query_states.shape
scores = (query_states.float() @ key_states.float().transpose(-2, -1)) / math.sqrt(D)
```

Scores are computed in FP32 regardless of input dtype. This is a project
numerical-stability rule — BF16 matmul on scores can accumulate error in long
sequences.

Result shape: `(B, H, T, T)`.

### 3.2 Causal mask

```python
causal = torch.triu(torch.ones(T, T, dtype=torch.bool, device=...), diagonal=1)
scores = scores.masked_fill(causal, float("-inf"))
```

`causal[i, j] = True` when `j > i` (future positions). Masked positions become
\(-\infty\) so softmax assigns zero weight.

### 3.3 Sliding-window mask

```python
if window is not None and window < T:
    idx = torch.arange(T, device=...)
    outside = idx.unsqueeze(0) - idx.unsqueeze(1) >= window
    scores = scores.masked_fill(outside, float("-inf"))
```

For query index `i` and key index `j`: masked when `i - j >= window`.
Combined with causal mask, allowed keys satisfy `j \le i` and `i - j < window`.

When `window >= T`, the window constraint is vacuous — behaves as full causal.

### 3.4 Sink augmentation

```python
if sink_bias is not None:
    sink_logit = sink_bias.view(1, H, 1, 1).to(scores.dtype)
    augmented = torch.cat([scores, sink_logit.expand(B, H, T, 1)], dim=-1)
    attn_weights = F.softmax(augmented, dim=-1)
    attn_weights = attn_weights[..., :T]
    return (attn_weights.to(value_states.dtype) @ value_states)
```

Steps:

1. Reshape `sink_bias` from `(H,)` to broadcastable `(1, H, 1, 1)`.
2. Concatenate along key dimension → shape `(B, H, T, T+1)`.
3. Softmax over all `T+1` columns.
4. **Strip** sink column → `(B, H, T, T)`.
5. Matmul with `value_states` — sink's zero value never enters because its weight
   was removed.

When `sink_bias is None`:

```python
attn_weights = F.softmax(scores, dim=-1)
return (attn_weights.to(value_states.dtype) @ value_states)
```

---

## 4. Mask Helpers

Two cached functions build boolean masks for the SDPA path.

### 4.1 `_causal_mask`

```python
@functools.lru_cache(maxsize=None)
def _causal_mask(T: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    idx = torch.arange(T, device=device)
    return idx.unsqueeze(1) >= idx.unsqueeze(0)  # (T, T)
```

Returns `True` where attention is **allowed** (lower-triangular including diagonal).
Opposite convention from `manual_causal_attention` (which masks `True` = forbidden).

Shape: `(T_q, T_k)` when square; used for prefill where `T_q == T_k`.

### 4.2 `_window_mask`

```python
@functools.lru_cache(maxsize=None)
def _window_mask(T_q, T_k, window, device, dtype) -> torch.Tensor:
```

**Prefill branch** (`T_q == T_k`):

```python
idx = torch.arange(T_q, device=device)
return (idx.unsqueeze(0) - idx.unsqueeze(1) < window) & _causal_mask(T_q, device, dtype)
```

**Decode branch** (`T_q = 1`, `T_k` = cached length):

```python
idx_q = torch.tensor([T_k - 1], device=device)
idx_k = torch.arange(T_k, device=device)
return (idx_q.unsqueeze(-1) - idx_k.unsqueeze(0) < window)
```

Decode assumes a single new query at position `T_k - 1` attending to cached keys
`0 .. T_k-1`. No separate causal check needed — all cached keys are in the past.

### 4.3 Cache key semantics

`lru_cache` keys on `(T, device, dtype)` or `(T_q, T_k, window, device, dtype)`.
Changing device or dtype misses cache (expected). `maxsize=None` — unbounded cache;
typical training sees few distinct `(T, window)` pairs per run.

---

## 5. `causal_attention`

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

### 5.1 Fast path: no sink, no window

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

- Square + causal → `is_causal=True` enables Flash Attention fusion.
- Rectangular without causal flag → caller responsible for mask semantics
  (decode with only past keys in cache).

### 5.2 Window path without sink

```python
if T_q == T_k:
    mask = _causal_mask(T_q, device, dtype) & _window_mask(T_q, T_k, window, device, dtype)
else:
    mask = _window_mask(T_q, T_k, window, device, dtype)

attn_mask = torch.where(mask, 0.0, float("-inf")).to(dtype).unsqueeze(0).unsqueeze(0)
return F.scaled_dot_product_attention(query_states, key_states, value_states, attn_mask=attn_mask)
```

Boolean allowed-mask → additive mask: `0.0` (pass) or `-inf` (block).
Batch/head dims added via `unsqueeze(0).unsqueeze(0)` → broadcast `(1, 1, T_q, T_k)`.

### 5.3 Branch summary

| `sink_bias` | `window` | Path |
|-------------|----------|------|
| `None` | `None` | `is_causal=True` or bare SDPA |
| `None` | set | Cached boolean → `where(0, -inf)` mask |
| set | `None`/set | Sink K/V extension + head-specific mask |

---

## 6. Sink Path: SDPA with Zero K/V Column

When `sink_bias is not None`:

### 6.1 Extend K and V

```python
sink_k = torch.zeros(B, H, 1, query_states.shape[-1], device=device, dtype=dtype)
sink_v = torch.zeros(B, H, 1, value_states.shape[-1], device=device, dtype=value_states.dtype)
k_ext = torch.cat([key_states, sink_k], dim=2)   # (B, H, T_k+1, D)
v_ext = torch.cat([value_states, sink_v], dim=2)
```

The sink key is all zeros → dot product \(q \cdot k_{\text{sink}} = 0\). The learned
bias enters **only** through the attention mask, not through K.

### 6.2 Build causal/window base

```python
if window is None or window >= T_k:
    causal = _causal_mask(T_q, device, dtype) if T_q == T_k \
        else torch.ones(T_q, T_k, dtype=torch.bool, device=device)
elif T_q == T_k:
    causal = _causal_mask(T_q, device, dtype) & _window_mask(T_q, T_k, window, device, dtype)
else:
    causal = _window_mask(T_q, T_k, window, device, dtype)
```

When `window >= T_k`, window constraint is inactive — only causal (or all-ones for decode).

### 6.3 Assemble head-specific mask

```python
mask = torch.zeros(H, T_q, T_k + 1, device=device, dtype=dtype)
mask[:, :, :T_k] = causal.to(dtype)
mask[:, :, T_k] = sink_bias.to(dtype).unsqueeze(1).expand(H, T_q)
return F.scaled_dot_product_attention(query_states, k_ext, v_ext, attn_mask=mask.unsqueeze(0))
```

Mask layout:

| Column range | Value | Meaning |
|--------------|-------|---------|
| `0 .. T_k-1` | `1.0` if allowed, `0.0` if forbidden | Boolean `causal` cast to float |
| `T_k` (sink) | `sink_bias[h]` | Per-head learned logit |

**Important:** The non-sink window path uses `torch.where(mask, 0.0, -inf)` to block
forbidden positions. The sink path uses `causal.to(dtype)` (`1.0`/`0.0`) instead.
These conventions differ — when validating sink behaviour, compare against
`manual_causal_attention` (the test oracle), not against the no-sink SDPA path.

The sink column receives `sink_bias[h]` directly — this is the logit added to
\(q \cdot k_{\text{sink}} / \sqrt{d} = 0\), equivalent to the concat-and-softmax
formulation in `manual_causal_attention`.

Output shape: `(B, H, T_q, D_v)` — unchanged from input value head dimension.

---

## 7. `repeat_kv` — GQA Head Expansion

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

### 7.1 Why `expand` not `repeat_interleave`

`expand` creates a broadcast view — no `.contiguous()` copy. SDPA's Flash Attention
backend handles non-contiguous K/V internally. This saves memory bandwidth during
both training and inference.

### 7.2 Head mapping

KV head `h_kv` maps to Q heads `2*h_kv` and `2*h_kv + 1` when `n_rep=2`.
Each pair shares identical K/V after `repeat_kv`.

---

## 8. `GPTOSSAttention`

```python
class GPTOSSAttention(nn.Module):
    def __init__(self, cfg, layer_idx: int):
```

### 8.1 Construction

```python
self.is_windowed = (layer_idx % 2 == 0)
self.prune_rope_global = bool(getattr(cfg, "yarn_prune_rope_global", True))
self.n_rep = self.n_heads // self.n_kv_heads

self.q_proj = nn.Linear(d_model, n_heads * head_dim, bias=False)
self.kv_proj = nn.Linear(d_model, 2 * n_kv_heads * head_dim, bias=False)
self.o_proj = nn.Linear(n_heads * head_dim, d_model, bias=False)
```

Fused `kv_proj` emits K and V in one matmul — halves KV projection kernel launches
vs separate `k_proj`/`v_proj`.

### 8.2 Sink parameter

```python
if cfg.sink_bias:
    self.sink_bias = nn.Parameter(torch.zeros(self.n_heads))
else:
    self.register_parameter("sink_bias", None)
```

When disabled, `causal_attention` takes the fast no-sink paths.

### 8.3 YaRN module

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

See [yarn.md](yarn.md) for parameter semantics.

### 8.4 Pruned RoPE helper

```python
def _n_pruned_dims(self) -> int:
    if (not self.is_windowed) and self.prune_rope_global:
        return self.head_dim // 4
    return 0
```

Returns number of frequency **pairs** to prune (not individual dims). For
`head_dim=96` → 24 pairs → 48 scalar dimensions frozen to identity rotation.

Applied only on **global** (odd-indexed) layers when `yarn_prune_rope_global=True`.

### 8.5 `extra_repr`

```python
def extra_repr(self) -> str:
    mode = "SWA" if self.is_windowed else "Full"
    ...
```

Example: `layer=0 (SWA), H=8/4, D=96, window=128`.

---

## 9. Forward Path Trace

Complete `forward(x, positions=None)` execution:

### Step 1 — Positions

```python
B, T, _ = x.shape
if positions is None:
    positions = torch.arange(T, device=x.device)
```

Default: contiguous positions `0, 1, ..., T-1`. Inference passes explicit positions
for decode (single position per step).

### Step 2 — Projections

```python
query_states = self.q_proj(x).view(B, T, self.n_heads, self.head_dim)
kv = self.kv_proj(x).view(B, T, 2, self.n_kv_heads, self.head_dim)
key_states, value_states = kv[:, :, 0], kv[:, :, 1]
```

### Step 3 — Head-major layout

```python
query_states = query_states.transpose(1, 2)   # (B, H, T, D)
key_states = key_states.transpose(1, 2)         # (B, H_kv, T, D)
value_states = value_states.transpose(1, 2)
```

SDPA expects `(B, H, T, D)`.

### Step 4 — RoPE

```python
cos, sin = self.yarn(positions, n_pruned_dims=self._n_pruned_dims())
query_states = apply_rope(query_states, cos, sin)
key_states = apply_rope(key_states, cos, sin)
```

`apply_rope` is imported from `models.rotary` — see [rotary.md](rotary.md).

### Step 5 — GQA expansion

```python
key_states = repeat_kv(key_states, self.n_rep)
value_states = repeat_kv(value_states, self.n_rep)
```

### Step 6 — Sink clamp

```python
if self.sink_bias is not None:
    sink_bias_clamped = self.sink_bias.clamp(SINK_CLAMP_MIN, SINK_CLAMP_MAX)
else:
    sink_bias_clamped = None
```

### Step 7 — Attention

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

### Step 8 — Output projection

```python
out = out.transpose(1, 2).contiguous().view(B, T, self.n_heads * self.head_dim)
return self.o_proj(out)
```

`.contiguous()` before `view` — attention output may be non-contiguous after SDPA.

---

## 10. Shape Reference

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

---

## 11. SDPA Backend Notes

### 11.1 Flash Attention eligibility

`is_causal=True` path (no sink, no window, square) triggers FA2 via PyTorch SDPA
when available. Requires CUDA + compatible head dims.

### 11.2 Explicit `attn_mask` paths

Window and sink paths pass `attn_mask` — may fall back to math or memory-efficient
kernel depending on GPU and PyTorch version. Still fused; not necessarily FA2.

### 11.3 Dtype contract

Q, K, V must share dtype for SDPA. `apply_rope` preserves `x.dtype` by casting
`cos`/`sin` to match — see [rotary.md §2](rotary.md#2-apply_rope).

### 11.4 `torch.compile`

When `training.compile: true`, `GPTOSSAttention` is compiled as part of the model.
`lru_cache` on mask helpers runs at trace time for static `T`; dynamic shapes may
graph-break on cache miss.

---

## 12. Inference Integration

`inference/generate.py` duplicates the attention forward with KV caching:

```python
from models.attention import SINK_CLAMP_MAX, SINK_CLAMP_MIN, causal_attention, repeat_kv
from models.rotary import apply_rope  # canonical import
```

Note: `generate.py` lists `apply_rope` in the `models.attention` import block in
some versions — the canonical definition is `models.rotary.apply_rope`.

### 12.1 Cache stores rotated K

```python
k_new_rot = apply_rope(k_new, cos, sin)
cache.append(layer_idx, k_new_rot, v_new, attn.is_windowed, attn.window_size)
```

RoPE is applied **before** caching. On decode, cached K is already position-encoded;
only the new query receives fresh RoPE at `positions_step`.

### 12.2 Sink bias cache

```python
sink_bias_clamped = attn.sink_bias.clamp(SINK_CLAMP_MIN, SINK_CLAMP_MAX)
sink_bias_cache[id(attn)] = sink_bias_clamped
```

Clamp once per layer per generation.

### 12.3 Windowed vs global dispatch

```python
if attn.is_windowed:
    out = causal_attention(q, k_for_q, v_for_q, window=attn.window_size, sink_bias=...)
else:
    out = causal_attention(q, k_for_q, v_for_q, sink_bias=...)
```

Mirrors `GPTOSSAttention.forward` logic.

---

## 13. Config Knobs

From `ModelConfig` in `models/transformer.py`:

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

---

## 14. Common Pitfalls

### 14.1 Forgetting GQA repeat

Calling `causal_attention` with mismatched Q/K head counts raises shape errors in
SDPA. Always `repeat_kv` before attention.

### 14.2 Applying RoPE twice

Cached inference stores rotated K. Do not call `apply_rope` on cached keys again.

### 14.3 Unclamped sink in custom code

Direct `causal_attention(..., sink_bias=raw_param)` bypasses clamp. Use
`.clamp(SINK_CLAMP_MIN, SINK_CLAMP_MAX)` or go through `GPTOSSAttention`.

### 14.4 Window on global layers

Passing `window=` on odd layers changes the architecture — breaks KV math and
headline metric. Only even layers should pass `window_size`.

### 14.5 `positions` dtype/device

`YaRNRoPE.forward` uses `positions.float()` for the outer product. Integer positions
on the correct device work; floats are fine too.

### 14.6 Mask convention confusion

- `_causal_mask`: `True` = allowed
- `manual_causal_attention`: `True` in `triu` = **forbidden**
- Non-sink SDPA: `0.0` = allowed, `-inf` = forbidden
- Sink SDPA: `causal.to(dtype)` → `1.0`/`0.0` for real keys, `sink_bias` for sink column

When in doubt, trust `manual_causal_attention` for golden values.

---

## Related Documentation

- [ATTENTION_SINKS.md](ATTENTION_SINKS.md) — sink theory, KV math, failure modes
- [yarn.md](yarn.md) — `YaRNRoPE` module
- [rotary.md](rotary.md) — `apply_rope` and frequency computation

---

<!-- docs:verified 2026-07-31 · fa6f918 -->
