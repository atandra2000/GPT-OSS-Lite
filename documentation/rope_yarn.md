# RoPE and YaRN — Position Encoding for 128K

> Purpose: end-to-end position encoding from pairwise RoPE geometry through
> YaRN extrapolation and pruned RoPE on global layers.
> Sources: `models/rotary.py`, `models/yarn.py`.
> Attention consumer: [ATTENTION_SINKS.md](ATTENTION_SINKS.md#part-b--implementation-modelsattentionpy).

## Table of contents

1. [Purpose and mental model](#1-purpose-and-mental-model)
2. [RoPE geometry and `apply_rope`](#2-rope-geometry-and-apply_rope)
3. [Frequency bases](#3-frequency-bases)
4. [YaRN theory (ramp, blend, mscale)](#4-yarn-theory-ramp-blend-mscale)
5. [Production parameters (θ=100K, scale=32, target=131072)](#5-production-parameters-θ100k-scale32-target131072)
6. [`compute_yarn_freqs` / `compute_yarn_mscale`](#6-compute_yarn_freqs--compute_yarn_mscale)
7. [`YaRNRoPE` module](#7-yarnrope-module)
8. [Pruned RoPE on global layers (25% of dims)](#8-pruned-rope-on-global-layers-25-of-dims)
9. [Dtype / SDPA contract](#9-dtype--sdpa-contract)
10. [Worked numerical examples](#10-worked-numerical-examples)
11. [Degenerate ramp warning](#11-degenerate-ramp-warning)
12. [Debugging long-context issues](#12-debugging-long-context-issues)
13. [Invariants and failure modes](#13-invariants-and-failure-modes)
14. [How to verify](#14-how-to-verify)

---

## 1. Purpose and mental model

Rotary Position Embedding (RoPE; Su et al., 2021) encodes token position by
**rotating** query and key vectors in two-dimensional subspaces. Attention score
\(q_i^\top k_j\) depends on relative position \(i - j\) — a natural fit for causal
autoregressive models.

GPT-OSS-Lite trains at **4,096 tokens** (`yarn_original_max_seq_len`) but targets
**131,072 tokens** (`yarn_target_seq_len`) at inference — a 32× stretch.

Standard RoPE encodes position by rotating Q/K pairs at fixed frequencies derived
from base \(\theta\). When sequences exceed the training length, high-frequency
components complete many cycles per token — relative positions become ambiguous
and attention quality degrades ("positional collapse").

**YaRN** (Peng et al., 2023) interpolates between:

- **High frequencies** (short wavelength): unchanged — preserve local structure.
- **Low frequencies** (long wavelength): scaled by factor \(s\) — stretch far positions.

GPT-OSS-Lite implements YaRN via precomputed `inv_freq` buffers and optional
**pruned RoPE** on global-attention layers.

| Function / class | File | Role |
|------------------|------|------|
| `apply_rope` | `rotary.py` | Apply rotation to Q/K tensors |
| `compute_yarn_freqs` | `rotary.py` | Build YaRN-scaled inverse frequencies |
| `compute_yarn_mscale` | `rotary.py` | Attention temperature correction |
| `YaRNRoPE` | `yarn.py` | Module wrapping freq table + forward |

Standard RoPE is the `scale_factor=1`, zero-ramp limit of YaRN.

### Comparison with absolute PE

| Property | Absolute sinusoidal PE | RoPE |
|----------|------------------------|------|
| Applied to | Input embeddings | Q and K only |
| Position in score | Absolute | Relative |
| KV cache | Must store position offset | Rotate Q at decode; K pre-rotated |
| Extrapolation | Poor beyond train length | YaRN extends |

GPT-OSS caches **rotated** K in `MixedKVCache` — see
[ATTENTION_SINKS.md §B.8](ATTENTION_SINKS.md#b8-forward-path-trace-positions--out-proj).

---

## 2. RoPE geometry and `apply_rope`

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

Head dimension \(D\) splits into \(D/2\) independent 2D rotations. Each pair
\((x_{2m}, x_{2m+1})\) rotates in its own plane at frequency \(\omega_m\).

RoPE on queries at position \(i\) and keys at position \(j\):

\[
(R_{\theta_i} q)^\top (R_{\theta_j} k) = q^\top R_{\theta_i}^\top R_{\theta_j} k
= q^\top R_{\theta_j - \theta_i} k
\]

Attention scores depend on **relative** offset \(i - j\), not absolute positions
individually.

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

`cos`/`sin` from `YaRNRoPE` have shape `(T, D/2)` — one value per pair. RoPE needs
one value per scalar dimension:

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

`cos`/`sin` start as `(T, D)` after `repeat_interleave`; unsqueeze prepends dims
until rank matches `x` (typically 4D). Position index aligns with `x.size(-2)`.

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

Batch and head broadcast freely — same cos/sin table shared across all heads in a
layer (head-agnostic positional encoding).

---

## 3. Frequency bases

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

Low \(m\) → high frequency → short wavelength → local positional sensitivity.
High \(m\) → low frequency → long wavelength → global positional sensitivity.

### 3.3 Position 0

At \(p = 0\): \(\theta_{0,m} = 0\) → \(\cos=1, \sin=0\) → identity rotation.
Position 0 is unmodified — useful for sink-adjacent behaviour.

---

## 4. YaRN theory (ramp, blend, mscale)

### 4.1 Frequency blending

YaRN defines a ramp \(\gamma(m) \in [0, 1]\) over dimension index \(m\):

\[
\omega^{\text{YaRN}}_m = \omega^{\text{base}}_m \cdot (1 - \gamma_m) + \frac{\omega^{\text{base}}_m}{s} \cdot \gamma_m
\]

where \(s\) is `scale_factor` (default 32).

- \(\gamma_m = 0\): original frequency (local / high-freq dims).
- \(\gamma_m = 1\): frequency divided by \(s\) (extrapolation / low-freq dims).

### 4.2 Ramp boundaries

The ramp transitions between dimension indices `low` and `high`, derived from
`original_max_seq_len`, `beta_fast`, and `beta_slow`:

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

This compensates for attention entropy change when frequencies are stretched.
Implemented in `compute_yarn_mscale`.

---

## 5. Production parameters (θ=100K, scale=32, target=131072)

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

## 6. `compute_yarn_freqs` / `compute_yarn_mscale`

Full implementation in `models/rotary.py`. Returns `inv_freq` tensor of shape
`(head_dim // 2,)`.

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

Note: `target_seq_len` is accepted for API symmetry but **not used** in the
frequency formula — extrapolation length is implicit in the chosen `scale_factor`
and ramp, not a separate clamp.

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

Multiplies all cos/sin values in `YaRNRoPE.forward`. Compensates for attention
logit magnitude change when frequencies are compressed.

For `scale_factor = 32`:

\[
\text{mscale} = 0.1 \cdot \ln(32) + 1 \approx 0.1 \cdot 3.466 + 1 \approx 1.347
\]

Applied uniformly to all `cos` and `sin` values after computation.
When `yarn_mscale=False` in config, `self.mscale = 1.0`.

---

## 7. `YaRNRoPE` module

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

Each `GPTOSSAttention` instantiates its own `YaRNRoPE` with identical hyperparameters.
Buffers are duplicated per layer (small memory — 48 floats each).

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

## 8. Pruned RoPE on global layers (25% of dims)

### 8.1 Motivation

At very long context, the lowest-frequency RoPE dimensions rotate slowly — they
encode absolute position over huge spans. On **global** (full-attention) layers,
freezing the lowest 25% of frequency pairs to identity reduces spurious long-range
positional aliasing while preserving fine local structure in higher pairs.

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

For the first `n_pruned_dims` **frequency pairs** (lowest indices):

- `cos → 1`, `sin → 0` → identity rotation (no positional encoding on those pairs).
- Remaining pairs use full YaRN-scaled rotation.

This is applied **after** mscale multiplication.

### 8.4 Interaction with `apply_rope`

`apply_rope` repeats each pair's cos/sin across the two scalars via
`repeat_interleave(2, dim=-1)`. Pruning 24 pairs affects 48 of 96 head dimensions.

When `n_pruned_dims > 0` (global layers only), lowest-frequency pairs become
identity — equivalent to not rotating those subspaces. `apply_rope` is unaware of
pruning; it receives modified cos/sin.

### 8.5 Interaction with sliding-window layers

| Layer type | YaRN | Pruning | Visible context |
|------------|------|---------|-----------------|
| Windowed (even) | Full YaRN table | None | Last 128 tokens |
| Global (odd) | Full YaRN table | First 24 pairs → identity | All prior tokens |

Windowed layers see only local context — they rely on **full** RoPE (no pruning) for
fine-grained relative position within the 128-token window.

Global layers carry long-range dependencies — pruning lowest frequencies reduces
absolute-position interference at 128K.

YaRN and sink bias are orthogonal — see
[ATTENTION_SINKS.md §9](ATTENTION_SINKS.md#9-interaction-with-yarn-and-pruned-rope).

---

## 9. Dtype / SDPA contract

### 9.1 Dtype preservation in `apply_rope`

```python
cos_full = cos.repeat_interleave(2, dim=-1).to(x.dtype)
sin_full = sin.repeat_interleave(2, dim=-1).to(x.dtype)
```

`cos`/`sin` are computed in FP32 inside `YaRNRoPE`. Casting to `x.dtype` before
multiply prevents implicit promotion to FP32, which would:

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

Frequencies computed in FP32; cos/sin in FP32; cast to BF16 at multiply time.
Sufficient precision for angles up to 128K positions with YaRN scaling.

---

## 10. Worked numerical examples

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

With `half=48`, `L_orig=4096`, `beta_slow=1`, `beta_fast=32`:

\[
\log_2\!\left(\frac{4096}{\pi}\right) \approx 10.35 \quad\Rightarrow\quad
\text{low} = \lfloor 48 / 10.35 \rfloor = 4
\]

\[
\log_2\!\left(\frac{4096}{32\pi}\right) \approx 5.35 \quad\Rightarrow\quad
\text{high} = \lceil 48 / 5.35 \rceil = 9
\]

Ramp transitions from \(m=4\) to \(m=9\). Dimensions 0–3 keep base freq; 9–47 use
scaled freq; 4–8 blend.

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

Without YaRN scaling, this pair would complete \(\sim 2\) full rotations over 128K —
with scaling, \(\sim 0.06\) rotations — much slower positional aliasing.

---

## 11. Degenerate ramp warning

When `high <= low`, the ramp cannot be constructed — typically from misconfigured
`beta_fast`/`beta_slow` or very small `original_max_seq_len`.

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

Effect: `inv_freq = base` — plain RoPE with no YaRN scaling. Long-context quality
will suffer. This emits `UserWarning`, not silent failure (project numerical-stability
rule from `AGENTS.md`).

---

## 12. Debugging long-context issues

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

If YaRN ramp is degenerate (see [§11](#11-degenerate-ramp-warning)),
positions beyond 4K remain ambiguous — extrapolation fails silently aside from the
warning.

---

## 13. Invariants and failure modes

### 13.1 No learned RoPE parameters

`inv_freq` is a fixed buffer from hyperparameters. Position generalisation is
entirely in the frequency table design (YaRN ramp), not learned embeddings.

### 13.2 `head_dim` must be even

Odd `head_dim` raises `ValueError` in both `compute_yarn_freqs` and `YaRNRoPE`.
Production config uses `head_dim=96`.

### 13.3 Relation to standard `rope` in other repos

Some implementations use `torch.polar` or complex multiplication. This repo uses
the rotate-half trick — fewer dependencies, identical math, better `torch.compile`
compatibility.

### 13.4 Import paths

```python
from models.rotary import apply_rope, compute_yarn_freqs, compute_yarn_mscale
```

`models/attention.py` imports `apply_rope` from `models.rotary`.
`models/yarn.py` imports freq helpers from `models.rotary`.

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

`yarn_prune_rope_global=True` with odd `n_layers` emits a warning — final layer
may be windowed (no pruning on last layer if it's even-indexed).

---

## 14. How to verify

```bash
python3 -m pytest tests/test_yarn.py -v
```

Additional checks:

- `head_dim % 2 == 0` — enforced at construction.
- Degenerate ramp emits `UserWarning` (not silent identity).
- `apply_rope` output dtype matches input dtype.
- Pruned global layers: `n_pruned_dims = head_dim // 4` pairs.

---

## Related Documentation

- [ATTENTION_SINKS.md §B.7](ATTENTION_SINKS.md#b7-gptossattention-construction-sink-param-yarn-pruned-rope) — `GPTOSSAttention` integration
- [ATTENTION_SINKS.md §B.8](ATTENTION_SINKS.md#b8-forward-path-trace-positions--out-proj) — where RoPE sits in the forward path
- [ATTENTION_SINKS.md](ATTENTION_SINKS.md) — sinks independent of RoPE; YaRN at 128K

---

<!-- docs:verified 2026-07-31 · 263838e -->
