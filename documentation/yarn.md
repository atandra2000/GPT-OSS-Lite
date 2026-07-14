# YaRN RoPE scaling in GPT-OSS-Lite

> **Source:** `models/yarn.py` (the `YaRNRoPE` module),
> `models/rotary.py::compute_yarn_freqs` (the frequency table).
> **Companion:** [`rotary.md`](rotary.md) for the underlying RoPE rotation.
> **Paper:** Peng et al., 2023 — arXiv:2309.00071.

---

## 1. Overview

**YaRN** (Yet another RoPE extensioN) is the method that lets a model trained
at `original_max_seq_len = 4 096` extrapolate to `target_seq_len = 131 072`
(128K) — a 32× context stretch — without being fine-tuned on long sequences. It
is the second headline capability of GPT-OSS-Lite after the sliding-window KV
reduction, and the `passkey_eval.py` benchmark exists specifically to measure it.

GPT-OSS-Lite uses YaRN **at training time** (not decode-only), which is the
stronger setting: the model never sees a 128K sequence during training, yet it
must generalise the attention pattern there. This is a deliberate test of true
length extrapolation rather than the easier "train short, decode short, then
stretch at inference" variant.

`YaRNRoPE` is an `nn.Module` that, given a set of positions, returns the
`(cos, sin)` tables that `apply_rope` consumes. It holds two pieces of
state:
- **`inv_freq`** — the YaRN-scaled frequency table (a buffer, not a
  parameter — it is fixed at construction).
- **`mscale`** — the attention-temperature correction scalar.

---

## 2. The problem YaRN solves

Plain RoPE bakes the training context length into the rotation frequencies.
Train at 4K and ask the model to attend at 128K, and two things break:

1. **Low-frequency over-rotation.** The slowest-rotating dimensions, which
   barely move over 4K tokens, complete many full rotations over 128K tokens.
   Their contribution to `Q·K` wraps around the circle and becomes noise
   ("over-rotation"). The model has never *seen* those rotation angles.
2. **Softsump temperature drift.** With 32× more keys in the softmax, the
   typical attention logit distribution shifts, and the softmax either
   sharpens or flattens in ways the model was not trained for.

Earlier fixes (PI, NTK-aware) interpolated or extrapolated the frequencies
globally, but each had a failure mode: pure interpolation makes the model
"slower" at local structure (it thinks 4K tokens are squeezed into a shorter
range), and pure extrapolation does not help the over-rotating low
frequencies. YaRN's insight is to treat **each frequency dimension
independently** and interpolate only where it is safe.

---

## 3. The YaRN method, step by step

### 3.1 Per-dimension frequency ramp

For each of the `head_dim / 2 = 48` frequency dimensions, YaRN picks one of
two fates based on how fast that dimension rotates:

- **High-frequency dimensions** (fast rotation, encode *local* structure):
  **leave unchanged.** Interpolating these would blur local token
  distinctions and hurt short-range quality.
- **Low-frequency dimensions** (slow rotation, encode *long-range*
  structure): **divide the frequency by `scale_factor = 32`.** This stretches
  the dimension's rotation over a 32× longer range, so the model sees, at
  position 128K, the same rotation angle it would have seen at position 4K.
- **Mid-frequency dimensions**: blend the two with a smooth ramp so no
  dimension is switched abruptly.

Concretely, the inverse frequency for dimension `d` is
(see [`rotary.md`](rotary.md) §4 for the full formula):

```
inv_freq[d] = base[d] · (1 − ramp[d])  +  (base[d] / scale_factor) · ramp[d]
```

where `ramp[d] ∈ [0, 1]` goes from 0 (high-freq, unchanged) to 1 (low-freq,
fully scaled). The ramp bounds come from `beta_fast = 32` and `beta_slow = 1`
— the rotation-count thresholds that separate "local" from "global"
dimensions.

### 3.2 The mscale temperature correction

```
mscale = 0.1 · log(scale_factor) + 1.0     # ≈ 1.346 for scale_factor = 32
```

`YaRNRoPE.forward` multiplies `cos` and `sin` by `mscale`. Since `apply_rope`
computes `out = x · cos + x_rotated · sin`, scaling `cos`/`sin` by `mscale` is
equivalent to scaling both Q and K by `mscale`, which scales the attention
logit `Q·K` by `mscale²`. The net effect — dividing attention logits by
`√mscale²`'s neighbourhood — keeps the softmax temperature well-conditioned at
the longer context (§2, problem 2). For `scale_factor ≤ 1`, `mscale` is 1.0
(no correction — the model is not stretching).

### 3.3 Pruned RoPE on global layers (the GPT-OSS addition)

On top of YaRN, GPT-OSS *prunes* the lowest-frequency 25% of dimensions on
**global (full-attention) layers** — sets `cos = 1, sin = 0` (identity rotation)
so those dimensions get *no* positional encoding. This removes the residual
over-rotation noise in the lowest frequencies that even YaRN's full scaling
does not fully tame at 128K. Windowed layers never prune (their context is
bounded by `window = 128`, so over-rotation is a non-issue). See
[`rotary.md`](rotary.md) §6 and [`attention.md`](attention.md) §7.4.

---

## 4. `YaRNRoPE` — the module

```python
class YaRNRoPE(nn.Module):
    def __init__(self, head_dim, theta=100000.0, scale_factor=32.0,
                 original_max_seq_len=4096, target_seq_len=131072,
                 beta_fast=32.0, beta_slow=1.0, mscale=True)
    def forward(self, positions, n_pruned_dims=0) -> (cos, sin)
```

### 4.1 Construction

`__init__` computes the frequency table once via `compute_yarn_freqs` and
registers it as a **non-persistent buffer** (`persistent=False` — it is
re-derived from the config on load, so it does not bloat the checkpoint). It
also computes `mscale` (or sets it to 1.0 if `mscale=False`).

### 4.2 `forward` — two paths

```python
if positions.numel() == 1:
    # Fast T=1 decode path
    pos = positions.item()
    freqs = inv_freq * float(pos)
    cos = freqs.cos().unsqueeze(0) * mscale
    sin = freqs.sin().unsqueeze(0) * mscale
else:
    # General (prefill / multi-token) path
    freqs = torch.outer(positions.float(), inv_freq)
    cos = freqs.cos() * mscale
    sin = freqs.sin() * mscale
```

The general path is the textbook YaRN: `torch.outer(positions, inv_freq)` gives
the per-position, per-dimension rotation angle, and `cos`/`sin` are the
rotation table of shape `(T, head_dim // 2)`.

### 4.3 The fast T=1 path (OPT-15)

During autoregressive decode, `positions` is a single-element tensor — the
position of the one new token. The general path would still call `torch.outer`,
which allocates a tiny `(1, 48)` matmul and incurs a full kernel-launch's
worth of overhead. The T=1 path reads the position as a Python float, scales
`inv_freq` by it directly (an elementwise multiply, no matmul), and takes
`cos`/`sin`. That saves one kernel launch per layer per decode step — across
12 layers and thousands of decode steps, this is a measurable decode
speedup. The two paths are mathematically identical; the fast path just
special-cases the shape that dominates decode.

### 4.4 Pruning in `forward`

When `n_pruned_dims > 0` (global layers), `forward` clones `cos`/`sin` and
zeros the leading `n_pruned_dims` columns. The clone is necessary because the
unpruned `cos`/`sin` from the T=1 / general path are computed fresh each call,
but `prune_rope` must not mutate the `inv_freq`-derived base — keeping them
separate makes the prune logic a pure function of its inputs.

---

## 5. Rotated-K caching (decode-side YaRN win)

`inference/generate.py` applies RoPE to K **before** storing it in the
`MixedKVCache`, not at attention time. This is a YaRN-specific optimisation:

- Without it, each decode step would have to recompute RoPE over the *entire*
  growing K cache (O(T) RoPE applications per step → O(T²) over a generation),
  because YaRN's frequencies are position-dependent.
- With it, each decode step rotates only the *one new K* (O(1) RoPE per step),
  stores the already-rotated K, and attention reads cached rotated K directly.

This makes decode **O(T) per token instead of O(T²)** — the same asymptotic win
as the ring buffer / exponential-growth KV cache, and it composes with them
(see [`inference.md`](inference.md)). It is only correct because RoPE is a
*linear* rotation of K that commutes with the cache append — a property that
would not hold for additive position embeddings.

---

## 6. The training-time vs decode-only distinction

Most YaRN deployments train at the original context and apply YaRN only at
inference. GPT-OSS-Lite trains *with* YaRN-scale frequencies from step 0
(the frequencies are baked into `inv_freq` at construction and used in every
forward). This means:

- The model's weights are optimised *for the stretched frequencies*, not
  against plain RoPE then stretched — a harder optimisation target but a
  stronger extrapolator.
- `passkey_eval.py` is a genuine extrapolation test: the model trained at
  4K is evaluated at up to 128K, and the ≥85% target at 128K is the proof that
  YaRN + pruned RoPE actually generalise.
- This is why `seq_len = 4096` (not 2048): YaRN needs ≥4K of training context
  to learn the frequency ramp meaningfully. Too short and the ramp has nothing
  to interpolate.

---

## 7. Design rationale & rejected alternatives

| Decision | Rationale | Rejected alternative |
|---|---|---|
| YaRN (per-dimension ramp) | Treats each frequency independently; preserves local structure | PI (global interpolation) — blurs local distinctions; NTK (global extrapolation) — does not fix low-freq over-rotation |
| Train-time YaRN (not decode-only) | Tests *true* extrapolation; stronger model | Decode-only YaRN — easier but not a real extrapolation claim |
| `mscale` temperature correction | Keeps softmax conditioned at 32× context | Ignore the softmax drift — model saturates / collapses at long context |
| Pruned RoPE on global layers (25%) | Removes residual low-freq noise at 128K | Prune everywhere — hurts windowed layers' local structure |
| Fast T=1 path | One fewer kernel launch per layer per decode step | Always `torch.outer` — wastes a launch on the common decode case |
| `inv_freq` as non-persistent buffer | Re-derived from config on load; no checkpoint bloat | Persistent buffer — stored in every checkpoint for no reason |
| Rotated-K caching | O(T) decode, not O(T²) | Recompute RoPE over whole cache each step |

---

## 8. Edge cases & pitfalls

- **Degenerate ramp** (`high <= low`): `compute_yarn_freqs` warns and returns a
  zero ramp → plain RoPE, *no* extrapolation. The model will then train fine at
  4K but fail the 128K passkey eval with no obvious cause. The warning is the
  only signal — do not suppress it.
- **`scale_factor > 1` requires `original < target`**: `ModelConfig.__post_init__`
  raises if `yarn_original_max_seq_len >= yarn_target_seq_len` while
  `scale_factor > 1`, because the ramp is meaningless when the model is not
  stretching.
- **`n_layers` odd with `yarn_prune_rope_global=True`**: the alternating
  pattern expects even `n_layers`; an odd count means the last layer is
  windowed (no pruning) and the pruning is asymmetric. `__post_init__` warns.
- **`inv_freq` device**: `forward` moves `inv_freq` to `positions.device` each
  call — cheap, but means the buffer is *not* moved with the model. If you
  ever cache `inv_freq` on a fixed device, update this.
- **Pruning clones**: `forward` clones `cos`/`sin` before pruning, so the
  unpruned base is not mutated. This is a per-call allocation on global layers
  only — acceptable.

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