# YaRN RoPE scaling module
Length extrapolation via frequency ramp interpolation.

YaRN (Peng et al. 2023, arXiv:2309.00071) trains at `original_max_seq_len` and extrapolates to `target_seq_len` by:
1. Computing scaled inverse frequencies via a ramp that preserves high frequencies (no scaling) and fully scales low frequencies by `scale_factor`.
2. Applying an attention logit scaling (`mscale`) so the softmax temperature stays well-conditioned at long contexts.

GPT-OSS-Lite uses YaRN at *training* time (not decode-only), enabling true 128K-context capability from a 4K-trained model.

---

## Implementation notes (extracted from code review)

- **Degenerate-ramp `UserWarning`**: `compute_yarn_freqs` computes the ramp
  bounds `low` and `high` from `head_dim`, `original_max`, `beta_fast`,
  `beta_slow`. When `high <= low` (extreme parameters), the ramp
  degenerates. Rather than silently falling back to identity, the function
  emits a `UserWarning` with the offending values and returns a zero ramp
  (no length extrapolation). Check `beta_fast`/`beta_slow` if this fires.
- **Pruned RoPE on global layers**: `GPTOSSAttention.__init__` sets
  `n_pruned_dims = head_dim // 4` (25% of dims) on global (odd) layers when
  `yarn_prune_rope_global=True`. `YaRNRoPE.forward` then zeros the leading
  `n_pruned_dims` cos/sin dims (cos=1, sin=0 → identity rotation), reducing
  over-rotation at 128K. Windowed (even) layers never prune.
- **Rotated-K caching for O(T) decode**: `inference/generate.py` applies
  RoPE to K *before* storing it in `MixedKVCache`, so each decode step only
  rotates the single new K rather than recomputing RoPE over the entire
  growing cache. This makes decode O(T) per token instead of O(T²).
- **Fast T=1 path in `YaRNRoPE.forward`**: during decode, `positions` is a
  (1,) tensor. The general `torch.outer(positions, inv_freq)` allocates a
  tiny matmul and incurs kernel-launch overhead. The T=1 path reads the
  position as a Python float, multiplies `inv_freq` by it directly, then
  takes `cos`/`sin` — saving one kernel launch per layer per decode step.
