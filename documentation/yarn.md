# YaRN — Yet another RoPE extensioN

> Theory and implementation of YaRN positional scaling in GPT-OSS-Lite.
> Module: `models/yarn.py` (`YaRNRoPE`). Frequency math lives in
> `models/rotary.py` (`compute_yarn_freqs`, `compute_yarn_mscale`).
> Application: `models/rotary.py` (`apply_rope`). Attention integration:
> [attention.md](attention.md). Sink interaction: [ATTENTION_SINKS.md §9](ATTENTION_SINKS.md#9-interaction-with-yarn-and-pruned-rope).

---

## Table of Contents

1. [Problem Statement](#1-problem-statement)
2. [RoPE Recap](#2-rope-recap)
3. [YaRN Theory](#3-yarn-theory)
4. [Production Parameters](#4-production-parameters)
5. [Frequency Computation](#5-frequency-computation)
6. [`YaRNRoPE` Module](#6-yarnrope-module)
7. [Pruned RoPE on Global Layers](#7-pruned-rope-on-global-layers)
8. [Forward Pass Walkthrough](#8-forward-pass-walkthrough)
9. [Degenerate Ramp Warning](#9-degenerate-ramp-warning)
10. [mscale Attention Scaling](#10-mscale-attention-scaling)
11. [Interaction with Sliding-Window Layers](#11-interaction-with-sliding-window-layers)
12. [Config Validation](#12-config-validation)
13. [Numerical Examples](#13-numerical-examples)
14. [Debugging Long-Context Issues](#14-debugging-long-context-issues)

---

## 1. Problem Statement

GPT-OSS-Lite trains at **4,096 tokens** (`yarn_original_max_seq_len`) but targets
**131,072 tokens** (`yarn_target_seq_len`) at inference — a 32× stretch.

Standard RoPE (Rotary Position Embedding) encodes position by rotating Q/K pairs at
fixed frequencies derived from base \(\theta\). When sequences exceed the training
length, high-frequency components complete many cycles per token — relative positions
become ambiguous and attention quality degrades ("positional collapse").

**YaRN** (Peng et al., 2023) interpolates between:

- **High frequencies** (short wavelength): unchanged — preserve local structure.
- **Low frequencies** (long wavelength): scaled by factor \(s\) — stretch far positions.

GPT-OSS-Lite implements YaRN via precomputed `inv_freq` buffers and optional
**pruned RoPE** on global-attention layers.

---

## 2. RoPE Recap

For head dimension \(d\), RoPE operates on \(d/2\) dimension pairs. Pair \(m\) uses
frequency:

\[
\omega_m = \theta^{-2m/d}, \quad m \in \{0, 1, \ldots, d/2 - 1\}
\]

Position \(p\) contributes rotation angle \(\theta_{p,m} = p \cdot \omega_m\).

\[
\begin{pmatrix} x'_0 \\ x'_1 \end{pmatrix}
=
\begin{pmatrix} \cos\theta & -\sin\theta \\ \sin\theta & \cos\theta \end{pmatrix}
\begin{pmatrix} x_0 \\ x_1 \end{pmatrix}
\]

See [rotary.md](rotary.md) for `apply_rope` implementation details.

---

## 3. YaRN Theory

### 3.1 Frequency blending

YaRN defines a ramp \(\gamma(m) \in [0, 1]\) over dimension index \(m\):

\[
\omega^{\text{YaRN}}_m = \omega^{\text{base}}_m \cdot (1 - \gamma_m) + \frac{\omega^{\text{base}}_m}{s} \cdot \gamma_m
\]

where \(s\) is `scale_factor` (default 32).

- \(\gamma_m = 0\): original frequency (local / high-freq dims).
- \(\gamma_m = 1\): frequency divided by \(s\) (extrapolation / low-freq dims).

### 3.2 Ramp boundaries

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

### 3.3 Attention temperature scaling (mscale)

YaRN optionally scales \(\cos/\sin\) by factor:

\[
\text{mscale} = 0.1 \cdot \ln(s) + 1
\]

This compensates for attention entropy change when frequencies are stretched.
Implemented in `compute_yarn_mscale`.

---

## 4. Production Parameters

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

## 5. Frequency Computation

`compute_yarn_freqs` in `models/rotary.py`:

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

### 5.1 Base inverse frequencies

```python
half = head_dim // 2
exponents = torch.arange(0, half, dtype=torch.float32) / half
base = 1.0 / (theta ** exponents)
```

Equivalent to \(\omega_m = \theta^{-2m/d}\).

Note: `target_seq_len` is accepted for API symmetry but **not used** in the
frequency formula — extrapolation length is implicit in the chosen `scale_factor`
and ramp, not a separate clamp.

### 5.2 Ramp indices

```python
low = max(math.floor(half / math.log2(original_max_seq_len / beta_slow * math.pi)), 0)
high = min(math.ceil(half / math.log2(original_max_seq_len / beta_fast * math.pi)), half - 1)
```

### 5.3 Ramp vector

```python
if high <= low:
    warnings.warn("YaRN ramp degenerate: ...", UserWarning)
    ramp = torch.zeros(half, dtype=torch.float32)
else:
    ramp = torch.clamp(
        (torch.arange(half, dtype=torch.float32) - low) / max(high - low, 1),
        0.0, 1.0,
    )
```

### 5.4 Blended frequencies

```python
inv_freq = base * (1.0 - ramp) + (base / scale_factor) * ramp
return inv_freq
```

Returned tensor registered as `YaRNRoPE.inv_freq` buffer.

---

## 6. `YaRNRoPE` Module

```python
class YaRNRoPE(nn.Module):
    def __init__(self, head_dim, theta=100000.0, scale_factor=32.0, ...):
```

### 6.1 Construction

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

### 6.2 One module per attention layer

Each `GPTOSSAttention` instantiates its own `YaRNRoPE` with identical hyperparameters.
Buffers are duplicated per layer (small memory — 48 floats each).

---

## 7. Pruned RoPE on Global Layers

### 7.1 Motivation

At very long context, the lowest-frequency RoPE dimensions rotate slowly — they
encode absolute position over huge spans. On **global** (full-attention) layers,
freezing the lowest 25% of frequency pairs to identity reduces spurious long-range
positional aliasing while preserving fine local structure in higher pairs.

### 7.2 Selection rule

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

### 7.3 Application in `YaRNRoPE.forward`

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

### 7.4 Interaction with `apply_rope`

`apply_rope` repeats each pair's cos/sin across the two scalars via
`repeat_interleave(2, dim=-1)`. Pruning 24 pairs affects 48 of 96 head dimensions.

---

## 8. Forward Pass Walkthrough

```python
def forward(self, positions: torch.Tensor, n_pruned_dims: int = 0) -> tuple[Tensor, Tensor]:
```

### 8.1 Single position (decode)

```python
if positions.numel() == 1:
    inv_freq = self.inv_freq.to(positions.device)
    pos = positions.item() if positions.dim() == 0 else positions[0].item()
    freqs = inv_freq * float(pos)
    cos = freqs.cos().unsqueeze(0) * self.mscale
    sin = freqs.sin().unsqueeze(0) * self.mscale
```

Output shapes: `(1, half)`.

### 8.2 Multiple positions (prefill)

```python
freqs = torch.outer(positions.float(), self.inv_freq.to(positions.device))
cos = freqs.cos() * self.mscale
sin = freqs.sin() * self.mscale
```

Output shapes: `(T, half)` where `T = len(positions)`.

### 8.3 Pruning step

See [§7.3](#73-application-in-yarnropeforward).

### 8.4 Consumption

```python
cos, sin = self.yarn(positions, n_pruned_dims=self._n_pruned_dims())
query_states = apply_rope(query_states, cos, sin)
key_states = apply_rope(key_states, cos, sin)
```

`cos`/`sin` broadcast over batch and head dimensions inside `apply_rope`.

---

## 9. Degenerate Ramp Warning

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
rule).

---

## 10. mscale Attention Scaling

```python
def compute_yarn_mscale(scale_factor: float) -> float:
    if scale_factor <= 1.0:
        return 1.0
    return 0.1 * math.log(scale_factor) + 1.0
```

For `scale_factor = 32`:

\[
\text{mscale} = 0.1 \cdot \ln(32) + 1 \approx 0.1 \cdot 3.466 + 1 \approx 1.347
\]

Applied uniformly to all `cos` and `sin` values after computation.

When `yarn_mscale=False` in config, `self.mscale = 1.0`.

---

## 11. Interaction with Sliding-Window Layers

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

## 12. Config Validation

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

## 13. Numerical Examples

### 13.1 Ramp boundaries (defaults)

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

### 13.2 Frequency at dimension 0

\[
\omega_0 = 100000^{-0/96} = 1.0
\]

With \(\gamma_0 = 0\): \(\omega^{\text{YaRN}}_0 = 1.0\).

### 13.3 Frequency at dimension 47

\[
\omega_{47} \approx 100000^{-47/48} \approx 1.58 \times 10^{-5}
\]

With \(\gamma_{47} = 1\): \(\omega^{\text{YaRN}}_{47} \approx \omega_{47} / 32\).

### 13.4 Position 131072 on pair 47

\[
\theta = 131072 \times \omega^{\text{YaRN}}_{47}
\]

Without YaRN scaling, this pair would complete \(\sim 2\) full rotations over 128K —
with scaling, \(\sim 0.06\) rotations — much slower positional aliasing.

---

## 14. Debugging Long-Context Issues

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

---

## Related Documentation

- [rotary.md](rotary.md) — `apply_rope`, `compute_yarn_freqs` source
- [attention.md](attention.md) — `GPTOSSAttention` integration
- [ATTENTION_SINKS.md](ATTENTION_SINKS.md) — sinks + YaRN at 128K

---

<!-- docs:verified 2026-07-31 · fa6f918 -->
