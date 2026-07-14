# Rotary Position Embeddings (RoPE) in GPT-OSS-Lite

> **Source:** `models/rotary.py`
> **Companions:** [`yarn.md`](yarn.md) (YaRN scaling of these frequencies),
> [`attention.md`](attention.md) (where the rotation is applied).

---

## 1. Overview

RoPE encodes **relative** position into the attention scores by rotating the
query and key vectors in the complex plane *before* their dot product. Unlike
additive position embeddings (GPT-2 style), RoPE does not touch the input
stream at all — it acts on Q and K inside every attention layer, after the
projections. Because the rotation angle grows with position, the dot product
`Q·K` ends up depending on the *difference* of their positions, which is
exactly the relative-position structure attention wants.

This module exposes three functions, all operating on the last
(`head_dim`) axis:

1. **`apply_rope(x, cos, sin)`** — apply a precomputed rotation to a tensor of
   Q or K.
2. **`compute_yarn_freqs(...)`** — compute YaRN-scaled inverse frequencies
   (the per-dimension rotation-rate table). Lives here rather than in
   `yarn.py` because it is the bridge between "what is a frequency" and "how
   does YaRN stretch it".
3. **`compute_yarn_mscale(scale_factor)`** — the YaRN attention-temperature
   correction (a scalar, see [`yarn.md`](yarn.md)).
4. **`prune_rope(cos, sin, n_pruned_dims)`** — zero out the lowest-frequency
   dimensions on global layers (the GPT-OSS "pruned RoPE" trick).

All functions use the **half-dim convention**: `cos`/`sin` have shape
`(..., head_dim // 2)`, i.e. one angle per *pair* of feature dimensions, and are
duplicated to full `head_dim` inside `apply_rope`.

---

## 2. The math: why rotation encodes relative position

### 2.1 The 2-D rotation

For a single frequency dimension `d` with base frequency
`ω_d = 1 / θ^{2d / head_dim}`, RoPE rotates the `(x_{2d}, x_{2d+1})` feature
pair of the vector at position `m` by angle `m · ω_d`:

```
┌ x_{2d}   ┐     ┌ cos(m·ω_d)  −sin(m·ω_d) ┐ ┌ x_{2d}   ┐
│          │  =  │                          │ │          │
└ x_{2d+1} ┘     ┌ sin(m·ω_d)   cos(m·ω_d) ┘ └ x_{2d+1} ┘
```

i.e. the standard rotation. Written elementwise:

```
out[2d]     = x[2d]   · cos(m·ω_d)  − x[2d+1] · sin(m·ω_d)
out[2d+1]   = x[2d]   · sin(m·ω_d)  + x[2d+1] · cos(m·ω_d)
```

### 2.2 Why this gives relative position

The dot product of two rotated vectors depends only on the *difference* of
their rotation angles. For Q at position `m` and K at position `n`:

```
Q'_d · K'_d  =  f(x_Q, x_K,  (m − n) · ω_d)
```

So the attention score `Q'·K'` is a function of the relative offset `m − n`,
not the absolute positions — exactly what causal attention needs. No position
table to learn, no max-length baked in at construction; the position is
implicit in the rotation angle.

### 2.3 The frequency table

The frequencies span a geometric range from high (fast rotation, captures
local structure) to low (slow rotation, captures long-range structure):

```
ω_d = 1 / θ^{2d / head_dim},    d = 0 … head_dim/2 − 1
```

With `θ = 100 000` and `head_dim = 96` (so `head_dim/2 = 48` frequencies), the
fastest dimension rotates ~`100 000` radians over 128K tokens while the slowest
barely moves. YaRN (see [`yarn.md`](yarn.md)) stretches the *low*-frequency
dimensions to push the model's usable context from 4K to 128K.

---

## 3. `apply_rope` — the fused rotation

```python
def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor
```

`x` is `(B, H, T, head_dim)`; `cos`/`sin` are `(T, head_dim // 2)` and are
broadcast up. A naive implementation would `repeat_interleave` the half-dim
cos/sin to full head_dim, build the rotation matrix, and matmul — three
allocations and a small `O(T·H·D²)` matmul per layer. The implementation here
is fused into two elementwise ops:

### 3.1 The fused form

```
x_pairs   = x.unflatten(-1, (-1, 2))        # group (x_{2d}, x_{2d+1}) pairs
x_swapped = x_pairs.flip(-1)                # (x_{2d+1}, x_{2d})
x_swapped[..., 0] = -x_swapped[..., 0]      # (-x_{2d+1}, x_{2d})   =: x_rotated
out       = x * cos_full + x_rotated * sin_full
```

`x_rotated` is exactly the "swap and negate first slot" trick that produces
`(-x_{2d+1}, x_{2d})`, which is what multiplies `sin` in §2.1. The result is
the rotation, expressed as two fused elementwise multiply-adds instead of a
matmul. Only **one** `repeat_interleave` is needed (to expand the half-dim
`cos`/`sin` to full `head_dim`); everything else is a broadcast.

### 3.2 Why half-dim convention?

Storing `cos`/`sin` at `head_dim // 2` matches the "one angle per pair"
structure of §2.1. The alternative — full-dim `cos`/`sin` interleaved — wastes
half the storage on duplicated values. The half-dim form is the LLaMA / GPT-OSS
convention and is what `YaRNRoPE` produces.

---

## 4. `compute_yarn_freqs` — the YaRN frequency table

```python
def compute_yarn_freqs(head_dim, theta, scale_factor, original_max, target_max,
                        beta_fast=32.0, beta_slow=1.0, ...) -> torch.Tensor
```

This is the heart of YaRN. It produces a per-dimension inverse frequency that
**interpolates between two regimes**:

- **High-frequency dimensions** (large `ω_d`, fast rotation): *unchanged*.
  These encode local structure and would be destroyed by stretching.
- **Low-frequency dimensions** (small `ω_d`, slow rotation): *fully scaled* by
  `scale_factor` (here `32`, since `131072 / 4096 = 32`). These encode
  long-range structure and must be stretched to reach the longer context.
- **In between**: a smooth ramp blends the two, so no single dimension is
  abruptly switched.

### 4.1 The construction

```
base       = 1 / θ^{2d/head_dim}              # plain RoPE frequencies
low, high  = ramp bounds (from beta_fast, beta_slow — see below)
ramp[d]    = clamp((d − low) / (high − low), 0, 1)     # 0 → 1 across the ramp
inv_freq[d] = base[d] · (1 − ramp[d])  +  (base[d] / scale_factor) · ramp[d]
```

- When `ramp[d] = 0` (high-freq end): `inv_freq = base` — unchanged RoPE.
- When `ramp[d] = 1` (low-freq end): `inv_freq = base / scale_factor` —
  frequencies divided by 32, so the model "thinks" position 128K is really at
  4K, i.e. it interpolates the low frequencies into the longer range.

### 4.2 The ramp bounds

```
low  = floor( (head_dim/2) / log2(original_max / beta_slow · π) )
high = ceil ( (head_dim/2) / log2(original_max / beta_fast · π) )
```

`beta_fast` / `beta_slow` define the rotation counts at the ramp's two ends:
roughly, a dimension is "high-frequency" if it completes ≥ `beta_fast` full
rotations within `original_max` tokens, and "low-frequency" if it completes ≤
`beta_slow`. The defaults (`beta_fast=32`, `beta_slow=1`) come from the YaRN
paper. The ramp is the linear interpolation between those two thresholds.

### 4.3 Degenerate-ramp guard

If `high <= low` (extreme `head_dim`, `original_max`, or `beta` choices that
collapse the ramp), the function emits a `UserWarning` with the offending
values and returns a **zero ramp** — i.e. falls back to plain RoPE with *no*
length extrapolation. This is a loud failure rather than a silent one: a
degenerate ramp means YaRN is doing nothing, and the model will not extrapolate
to 128K. Check `beta_fast` / `beta_slow` if it fires. See [`yarn.md`](yarn.md).

---

## 5. `compute_yarn_mscale` — the attention-temperature correction

```python
def compute_yarn_mscale(scale_factor: float) -> float
    return 0.1 · log(scale_factor) + 1.0      # for scale_factor > 1
```

When the context grows, the softmax in attention sees more keys, so the
typical attention logit must be rescaled to keep the softmax temperature
well-conditioned. YaRN multiplies Q and K by `mscale` (equivalently, divides
the attention logits by `√mscale²`), which compensates for the longer
extrapolation context. For `scale_factor = 32`: `mscale ≈ 1.346`. The
`YaRNRoPE` module applies this by scaling `cos`/`sin` (and therefore Q/K) by
`mscale`. See [`yarn.md`](yarn.md) for the full rationale.

---

## 6. `prune_rope` — pruned RoPE on global layers

```python
def prune_rope(cos, sin, n_pruned_dims) -> (cos, sin)
    cos_pruned[..., :n_pruned_dims] = 1.0
    sin_pruned[..., :n_pruned_dims] = 0.0
```

**Pruned RoPE** zeroes out the leading `n_pruned_dims` *lowest-frequency*
dimensions by setting `cos = 1, sin = 0` — the identity rotation, so those
dimensions receive **no positional encoding at all**. GPT-OSS applies this on
global (full-attention) layers only (25% of dims, i.e. `head_dim // 4 = 24`
dims here).

The rationale: at 128K context the lowest-frequency dimensions have rotated so
far that their contribution to the attention score becomes noisy / saturated
("over-rotation"). Pruning them removes the noise without losing the
high-frequency local structure. Windowed layers never prune — their context
is bounded by `window = 128`, so over-rotation is not a concern. See
[`attention.md`](attention.md) §7.4 and [`yarn.md`](yarn.md).

---

## 7. Design rationale & rejected alternatives

| Decision | Rationale | Rejected alternative |
|---|---|---|
| RoPE (rotary), not additive | Relative position falls out of the dot product for free | Learned additive position embeddings — bake in a max length |
| Apply inside attention, on Q/K only | RoPE is a Q/K transform; V and the input stream are untouched | Apply to the residual stream — couples position to the FFN |
| Fused elementwise form | Two multiply-adds, one `repeat_interleave` | Build the `(D,D)` rotation matrix and matmul — `O(D²)` per layer |
| Half-dim `cos`/`sin` storage | Matches "one angle per pair"; halves storage | Full-dim interleaved — duplicated values |
| Pruned RoPE on global layers only | Removes over-rotation noise at 128K | Prune everywhere — loses local structure on windowed layers |
| Loud `UserWarning` on degenerate ramp | Silent fallback = silent loss of extrapolation | Silent identity — model trains, fails at 128K, mystery |

---

## 8. Edge cases & pitfalls

- **Odd `head_dim`**: `compute_yarn_freqs` raises `ValueError` — the half-dim
  convention requires an even `head_dim`. The model config validates this in
  `ModelConfig.__post_init__`.
- **`prune_rope` bounds**: passing `n_pruned_dims > head_dim // 2` raises
  `ValueError` (would prune more dims than exist).
- **`prune_rope` clones**: it `.clone()`s `cos`/`sin` before overwriting — the
  caller's tensors (often cached in `YaRNRoPE`) are not mutated. This is a
  per-call allocation on pruned layers; acceptable because only global layers
  prune and only on the prefill / decode-1 path.
- **Device consistency**: `cos`/`sin` are computed on the position tensor's
  device; `apply_rope` does not move anything. A device mismatch here would
  surface as an explicit broadcast error, not a silent wrong result.

---

## Implementation notes (extracted from code review)

- **`apply_rope` fused op**: the rotation is computed as
  `out = x * cos + x_rotated * sin` where `x_rotated` is built by
  pair-flipping `(a, b) → (b, a)` and negating the first slot. This avoids
  the `repeat_interleave` allocation for cos/sin on every call (the
  half-dim cos/sin is duplicated to full head_dim via a single
  `repeat_interleave` broadcast).
- **`compute_yarn_freqs` degenerate-ramp warning**: when `high <= low` the
  ramp is degenerate; the function emits a `UserWarning` and falls back to
  identity (zero ramp). See `documentation/yarn.md` for details.
- **`prune_rope` convention**: zeroes the leading `n_pruned_dims` cos/sin
  dims by setting `cos=1.0, sin=0.0` (identity rotation), so those
  dimensions receive no positional encoding. Used on global (full-attention)
  layers to reduce over-rotation at 128K.