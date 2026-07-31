# Rotary Position Embeddings (RoPE)

> Mathematical foundation and implementation of RoPE in GPT-OSS-Lite.
> Source: `models/rotary.py`. YaRN scaling wrapper: [yarn.md](yarn.md).
> Attention consumer: [ATTENTION_SINKS.md](ATTENTION_SINKS.md#part-b--implementation-modelsattentionpy).

---

## Table of Contents

1. [Introduction](#1-introduction)
2. [`apply_rope`](#2-apply_rope)
3. [Pairwise Rotation Geometry](#3-pairwise-rotation-geometry)
4. [Frequency Bases](#4-frequency-bases)
5. [`compute_yarn_freqs`](#5-compute_yarn_freqs)
6. [`compute_yarn_mscale`](#6-compute_yarn_mscale)
7. [Interaction with YaRN](#7-interaction-with-yarn)
8. [Dtype and SDPA Contract](#8-dtype-and-sdpa-contract)
9. [Broadcasting Rules](#9-broadcasting-rules)
10. [Worked Example](#10-worked-example)
11. [Comparison with Absolute PE](#11-comparison-with-absolute-pe)
12. [Implementation Notes](#12-implementation-notes)

---

## 1. Introduction

Rotary Position Embedding (RoPE; Su et al., 2021) encodes token position by
**rotating** query and key vectors in two-dimensional subspaces. Attention score
\(q_i^\top k_j\) depends on relative position \(i - j\) — a natural fit for causal
autoregressive models.

GPT-OSS-Lite uses:

| Function / class | File | Role |
|------------------|------|------|
| `apply_rope` | `rotary.py` | Apply rotation to Q/K tensors |
| `compute_yarn_freqs` | `rotary.py` | Build YaRN-scaled inverse frequencies |
| `compute_yarn_mscale` | `rotary.py` | Attention temperature correction |
| `YaRNRoPE` | `yarn.py` | Module wrapping freq table + forward |

Standard RoPE is the `scale_factor=1`, zero-ramp limit of YaRN.

---

## 2. `apply_rope`

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

### 2.2 Dtype preservation

```python
cos_full = cos.repeat_interleave(2, dim=-1).to(x.dtype)
sin_full = sin.repeat_interleave(2, dim=-1).to(x.dtype)
```

`cos`/`sin` are computed in FP32 inside `YaRNRoPE`. Casting to `x.dtype` before
multiply prevents implicit promotion to FP32, which would:

1. Break `torch.compile` fusion patterns.
2. Violate SDPA's requirement that Q, K, V share dtype.

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

### 2.4 Broadcast over batch and heads

```python
while cos_full.dim() < x.dim():
    cos_full = cos_full.unsqueeze(0)
    sin_full = sin_full.unsqueeze(0)
```

`cos`/`sin` start as `(T, D)` after `repeat_interleave`; unsqueeze prepends dims
until rank matches `x` (typically 4D). Position index aligns with `x.size(-2)`.

### 2.5 Final combine

```python
return x * cos_full + x_rotated * sin_full
```

No in-place ops — safe for autograd.

---

## 3. Pairwise Rotation Geometry

### 3.1 Why pairs

Head dimension \(D\) splits into \(D/2\) independent 2D rotations. Each pair
\((x_{2m}, x_{2m+1})\) rotates in its own plane at frequency \(\omega_m\).

### 3.2 Relative position property

RoPE on queries at position \(i\) and keys at position \(j\):

\[
(R_{\theta_i} q)^\top (R_{\theta_j} k) = q^\top R_{\theta_i}^\top R_{\theta_j} k
= q^\top R_{\theta_j - \theta_i} k
\]

Attention scores depend on **relative** offset \(i - j\), not absolute positions
individually.

### 3.3 `repeat_interleave(2)`

`cos`/`sin` from `YaRNRoPE` have shape `(T, D/2)` — one value per pair. RoPE needs
one value per scalar dimension:

```python
cos_full = cos.repeat_interleave(2, dim=-1)  # (T, D)
```

Pair \(m\) angles apply to both \(x_{2m}\) and \(x_{2m+1}\).

---

## 4. Frequency Bases

### 4.1 Standard RoPE inverse frequencies

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

### 4.2 Wavelength interpretation

Wavelength at pair \(m\) for position increment 1:

\[
\lambda_m = \frac{2\pi}{\omega_m}
\]

Low \(m\) → high frequency → short wavelength → local positional sensitivity.
High \(m\) → low frequency → long wavelength → global positional sensitivity.

### 4.3 Position 0

At \(p = 0\): \(\theta_{0,m} = 0\) → \(\cos=1, \sin=0\) → identity rotation.
Position 0 is unmodified — useful for sink-adjacent behaviour.

---

## 5. `compute_yarn_freqs`

Full implementation in `models/rotary.py`. Returns `inv_freq` tensor of shape
`(head_dim // 2,)`.

### 5.1 Algorithm summary

1. Compute base RoPE frequencies `base`.
2. Compute ramp boundaries `low`, `high` from `original_max_seq_len`, `beta_fast`, `beta_slow`.
3. Build linear ramp \(\gamma_m\) from `low` to `high`.
4. Blend: `inv_freq = base * (1 - ramp) + (base / scale_factor) * ramp`.

See [yarn.md §5](yarn.md#5-frequency-computation) for parameter table and degenerate
ramp handling.

### 5.2 Validation

```python
if head_dim % 2 != 0:
    raise ValueError(f"head_dim must be even, got {head_dim}")
if original_max_seq_len <= 0 or target_seq_len <= 0:
    raise ValueError(...)
```

---

## 6. `compute_yarn_mscale`

```python
def compute_yarn_mscale(scale_factor: float) -> float:
    if scale_factor <= 1.0:
        return 1.0
    return 0.1 * math.log(scale_factor) + 1.0
```

Multiplies all cos/sin values in `YaRNRoPE.forward`. Compensates for attention
logit magnitude change when frequencies are compressed.

For \(s = 32\): mscale \(\approx 1.347\).

---

## 7. Interaction with YaRN

### 7.1 Data flow

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

### 7.2 Pruning effect on rotation

When `n_pruned_dims > 0` (global layers only):

```python
cos[:, :n_pruned_dims] = 1.0
sin[:, :n_pruned_dims] = 0.0
```

Lowest-frequency pairs become identity — equivalent to not rotating those
subspaces. `apply_rope` is unaware of pruning; it receives modified cos/sin.

### 7.3 Training vs inference positions

- **Prefill:** `positions = torch.arange(T)` → cos/sin shape `(T, half)`.
- **Decode:** `positions = torch.tensor([cur_pos - 1])` → shape `(1, half)`.

Same `inv_freq` table; only position values change.

---

## 8. Dtype and SDPA Contract

### 8.1 Why BF16 matters

GPT-OSS trains in BF16. If `apply_rope` promoted Q/K to FP32:

```python
# BAD — would break SDPA dtype contract
return (x.float() * cos + x_rotated.float() * sin).to(x.dtype)
```

SDPA requires `query_states.dtype == key_states.dtype == value_states.dtype`.

### 8.2 cos/sin precision

Frequencies computed in FP32; cos/sin in FP32; cast to BF16 at multiply time.
Sufficient precision for angles up to 128K positions with YaRN scaling.

---

## 9. Broadcasting Rules

`apply_rope` alignment requirements:

| `x` dim | `cos`/`sin` dim after unsqueeze |
|---------|--------------------------------|
| `(B, H, T, D)` | `(1, 1, T, D)` |

Position dimension `-2` of `x` must equal `T` in `cos`/`sin`.

Batch and head broadcast freely — same cos/sin table shared across all heads in a
layer (head-agnostic positional encoding).

---

## 10. Worked Example

**Setup:** `D=4` (2 pairs), `B=1`, `H=1`, `T=1`, position `p=1`, \(\theta=100000\).

### 10.1 Frequencies

\[
\omega_0 = 1.0, \quad \omega_1 = 100000^{-1/2} \approx 0.00316
\]

### 10.2 Angles at position 1

\[
\theta_0 = 1.0 \text{ rad}, \quad \theta_1 \approx 0.00316 \text{ rad}
\]

### 10.3 cos/sin (before mscale)

\[
\cos_0 \approx 0.540, \quad \sin_0 \approx 0.841
\]

### 10.4 Rotation of pair 0

For \((x_0, x_1)\):

\[
x'_0 = x_0 \cos\theta_0 - x_1 \sin\theta_0
\]
\[
x'_1 = x_0 \sin\theta_0 + x_1 \cos\theta_0
\]

`apply_rope` computes this via the `unflatten`/`flip` path — numerically equivalent.

---

## 11. Comparison with Absolute PE

| Property | Absolute sinusoidal PE | RoPE |
|----------|------------------------|------|
| Applied to | Input embeddings | Q and K only |
| Position in score | Absolute | Relative |
| KV cache | Must store position offset | Rotate Q at decode; K pre-rotated |
| Extrapolation | Poor beyond train length | YaRN extends |

GPT-OSS caches **rotated** K in `MixedKVCache` — see
[ATTENTION_SINKS.md §B.8](ATTENTION_SINKS.md#b8-forward-path-trace-positions--out-proj).

---

## 12. Implementation Notes

### 12.1 No learned RoPE parameters

`inv_freq` is a fixed buffer from hyperparameters. Position generalisation is
entirely in the frequency table design (YaRN ramp), not learned embeddings.

### 12.2 `head_dim` must be even

Odd `head_dim` raises `ValueError` in both `compute_yarn_freqs` and `YaRNRoPE`.
Production config uses `head_dim=96`.

### 12.3 Relation to standard `rope` in other repos

Some implementations use `torch.polar` or complex multiplication. This repo uses
the rotate-half trick — fewer dependencies, identical math, better `torch.compile`
compatibility.

### 12.4 Import paths

```python
from models.rotary import apply_rope, compute_yarn_freqs, compute_yarn_mscale
```

`models/attention.py` imports `apply_rope` from `models.rotary`.
`models/yarn.py` imports freq helpers from `models.rotary`.

---

## Related Documentation

- [yarn.md](yarn.md) — `YaRNRoPE` module, pruning, config
- [ATTENTION_SINKS.md §B.8](ATTENTION_SINKS.md#b8-forward-path-trace-positions--out-proj) — where RoPE sits in the forward path
- [ATTENTION_SINKS.md](ATTENTION_SINKS.md) — sinks independent of RoPE

---

<!-- docs:verified 2026-07-31 · fa6f918 -->
