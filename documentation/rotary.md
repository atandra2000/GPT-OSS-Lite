# RoPE Helpers

Three functions:
1. `apply_rope(x, cos, sin)` — standard RoPE application.
2. `compute_yarn_freqs(...)` — YaRN-scaled frequency computation with the high/low-frequency ramp interpolation between original and target lengths.
3. `prune_rope(cos, sin, n_pruned_dims)` — pruned RoPE for global layers (zeroes out a subset of frequency dimensions).

All functions operate on the last dimension (head_dim) and accept tensors of shape `(..., head_dim/2)` for cos/sin (half-dim RoPE convention).

## apply_rope
Convention: cos/sin have shape `(T, head_dim // 2)` (half-dim RoPE, with T = sequence length). They are duplicated to full `head_dim` and applied as a rotation in pairs.

**Notes:**
The standard RoPE rotation is:
`out[2k]   = x[2k]   * cos_k - x[2k+1] * sin_k`
`out[2k+1] = x[2k]   * sin_k + x[2k+1] * cos_k`

We implement it as two fused PyTorch ops:
1. `x_pairs = stack([x_even, x_odd])` → swap & negate first half.
2. `out = x * cos + x_rotated * sin`.

This avoids the `repeat_interleave` allocation for cos/sin by broadcasting the half-dim cos/sin via `repeat_interleave` only once.

## compute_yarn_freqs
Compute YaRN-scaled inverse frequencies for RoPE.
YaRN interpolates between identity (high frequencies, unchanged) and full scaling by `scale_factor` (low frequencies, stretched by `scale_factor`). The transition ramp is controlled by `beta_fast` and `beta_slow` — they define the per-dimension rotation counts that bound the ramp.

## compute_yarn_mscale
YaRN attention scaling factor (the mscale term).
Per YaRN, multiplying attention logits by `1 / sqrt(mscale^2)` (or, equivalently, scaling q,k by `mscale`) compensates for the longer extrapolation context.

## prune_rope
Prune (zero out) the first `n_pruned_dims` RoPE dimensions.
GPT-OSS uses pruned RoPE on full-attention (global) layers: a subset of the lowest-frequency dimensions receive no positional rotation, reducing over-rotation at very long contexts.

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
