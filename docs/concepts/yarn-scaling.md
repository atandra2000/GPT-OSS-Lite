# GPT-OSS-Lite — YaRN: Scaling 4K RoPE to 128K

> **Audience:** intermediate. Builds the RoPE background it needs in the first
> section, then derives the three moving parts of YaRN exactly as this repo
> implements them. The authoritative deep dive — full derivations, geometry,
> and the NTK lineage — is [attention-and-positional.md](attention-and-positional.md)
> (Parts B and C).

---

## RoPE in three sentences

Rotary position embedding gives every token position $p$ a rotation:
`models/rotary.py:apply_rope` rotates each consecutive pair of a head's
dimensions by an angle $p\,\omega_m$, where pair $m$ has frequency
$\omega_m = \theta^{-m/(d/2)}$ (with rope head dim $d$ and
`models/transformer.py:ModelConfig.rope_theta` $\theta = 10^5$). Because a
dot product of two rotated vectors depends only on the *difference* of their
rotation angles, attention scores become a function of relative position. The
48 pairs of a 96-dim head (48 = $d/2$) form a spectrum: fast pairs ($\omega$
near 1 rad/token) complete a rotation every few tokens and encode local order;
slow pairs barely rotate over a whole 4096-token sequence and act as
absolute-position carriers.

## The problem YaRN solves

This model trains at `max_seq_len = 4096` but evaluates at
`eval_max_seq_len = 131072` (both `models/transformer.py:ModelConfig` fields).
Feed position $p > 4096$ to plain RoPE and the two regimes fail differently:

- **Slow pairs extrapolate.** Over training length $L$ the slow pair has seen
  only the phase arc $[0, L\omega_m]$ — a proper sub-arc of the circle. At
  $p > L$ the rotation enters phases the network has never mapped to
  representations. Undefined behavior, in the most literal sense.
- **Fast pairs alias.** Their phase repeats every $2\pi/\omega_m$ tokens; they
  carry no absolute-position information and never did. Extending the context
  does nothing for them, yet they are the pairs that encode local order.

Position Interpolation (PI) rescales *every* frequency by $1/s$ so no phase
ever leaves the trained arc — but then adjacent tokens differ by
$\omega_m/s$ in phase, and fine-grained local ordering collapses. YaRN's move
is to make the policy **per pair**: keep the fast, compress the slow, blend in
between.

## Knob 1 — the interpolation ramp

`models/rotary.py:compute_yarn_freqs` blends each pair between its base
frequency and the fully interpolated one:

$$
\tilde\omega_m \;=\; \omega_m\,(1 - \gamma_m) + \frac{\omega_m}{s}\,\gamma_m,
\qquad
\gamma_m = \mathrm{clamp}\!\left(\frac{m - \texttt{low}}{\texttt{high} - \texttt{low}},\, 0,\, 1\right)
\tag{1}
$$

with the boundary pair indices computed in closed form from the training
length $L$ and the two ramp hyperparameters ($\beta_{\text{fast}} = 32$,
$\beta_{\text{slow}} = 1$, i.e. `ModelConfig.yarn_beta_fast` /
`ModelConfig.yarn_beta_slow`):

$$
\texttt{low} = \left\lfloor \frac{d/2}{\log_2\!\big(\tfrac{L}{\beta_{\text{slow}}}\,\pi\big)} \right\rfloor = 3,
\qquad
\texttt{high} = \left\lceil \frac{d/2}{\log_2\!\big(\tfrac{L}{\beta_{\text{fast}}}\,\pi\big)} \right\rceil = 6
\tag{2}
$$

Re-verified by direct execution on 2026-09-21 at the production configuration
($L = 4096$, $d/2 = 48$, $s = 32$): `inv_freq[0..3]` = 1.0, 0.787, 0.619,
0.487 — exactly the unscaled base $\theta^{-m/48}$, so the four fastest pairs
keep base frequencies; pairs 4–5 blend ($\gamma = 1/3,\, 2/3$); pairs 6–47 sit
at exactly $\omega_m/32$ — **42 of 48 pairs fully interpolated**, only the four
fastest escape. A degenerate ramp ($\texttt{high} \le \texttt{low}$, wrong
$\beta$s) degrades to a warning plus identity frequencies rather than a silent
mis-scale (`tests/test_yarn.py:test_compute_yarn_freqs_warns_on_degenerate_ramp`).

## Knob 2 — the attention temperature (`mscale`)

Compressing the slow pairs' frequencies by $s$ shrinks the phase differences
they can express between distant tokens, so the logit contrast across keys
drops, the softmax flattens, and attention entropy rises. YaRN counteracts
with a temperature on the logits — the logits are divided by
$t = 1/\mathrm{mscale}^2$ where

$$
\mathrm{mscale}(s) = 0.1 \ln s + 1
\tag{3}
$$

(`models/rotary.py:compute_yarn_mscale`; 1.0 for $s \le 1$, so plain RoPE is
the $s = 1$ limit). At $s = 32$: mscale $= 1.3466$, $t \approx 0.552$
(re-verified by execution 2026-09-21). The implementation multiplies **both**
the cos and sin tables by `mscale` in `models/yarn.py:YaRNRoPE.forward` —
every query and key passes through rope once, so each picks up one factor of
mscale and the logits end up scaled by mscale$^2 = 1/t$. The algebraic
invariant $\cos^2 + \sin^2 = \mathrm{mscale}^2$ per (position, dim) is pinned
by `tests/test_yarn.py:test_yarn_module_cos_sin_pair`.

## Knob 3 — where YaRN applies, and where it is pruned

Every attention layer builds its own `models/yarn.py:YaRNRoPE` with the same
config (`models/attention.py:GPTOSSAttention.__init__`), so **both** windowed
and full layers use YaRN-scaled positions. What differs is pruning:
`models/attention.py:GPTOSSAttention._n_pruned_dims` returns
`head_dim // 4 = 24` for full layers when
`ModelConfig.yarn_prune_rope_global` is on, and 0 for windowed layers. In
`YaRNRoPE.forward` the first `n_pruned_dims` table entries are forced to
cos = 1, sin = 0 — the 12 fastest pairs of a full layer carry **no** positional
encoding at all (re-verified: the stack's pruning values are exactly {0, 24};
pinned by `tests/test_yarn.py:test_yarn_module_pruned_dims`).

The asymmetry is the over-rotation argument: at $p = 131072$ the fastest
(unscaled) pair completes on the order of $10^4$ full turns, so its phase is
pseudorandom at long range — useless and noise-like to a full layer that must
aggregate over the entire context (derivation in
[attention-and-positional.md §4.8](attention-and-positional.md)). A windowed
layer never looks further back than `ModelConfig.window_size` = 128 tokens, so
its fastest pairs remain meaningful local-order encoders and nothing is
pruned.

The scale factor itself is the length ratio, $s = 131072/4096 = 32$, and
`models/transformer.py:ModelConfig.__post_init__` refuses invalid settings:
$s \ge 1$, and whenever $s > 1$ the original length must be strictly below the
target.

## What is verified vs. inherited

**Verified here** (tests + 2026-09-21 execution): frequency values and the
3/6 ramp boundary; mscale 1.3466; 42-of-48 compression; pruning {0, 24}; no
NaN at position 131072
(`tests/test_yarn.py:test_yarn_module_no_nan_128k`); 4K and 128K rotations are
distinct (`tests/test_yarn.py:test_yarn_module_4k_vs_128k_distinct`).
**Inherited** [INFERENCE]: the claim that this scheme *preserves quality* at
128K — the repo's passkey target (≥ 85% at 128K) is explicitly a target with
no measured run yet, and the s = 32 recipe's track record comes from the
YaRN-lineage models summarized in
[attention-and-positional.md](attention-and-positional.md).

## Related documentation

- [attention-and-positional.md](attention-and-positional.md) — authoritative
  deep dive: RoPE geometry, PI/NTK lineage, mscale derivation, pruning math.
- [sliding-full-alternation.md](sliding-full-alternation.md) — why windowed and
  full layers need different positional treatment.
- [learned-sinks.md](learned-sinks.md) — the attention-stability half of the
  long-context design.
- [../references/config-and-api.md](../references/config-and-api.md) — every
  `yarn_*` config key in table form.

<!-- docs:verified 2026-09-21 · 41a8c94 -->
