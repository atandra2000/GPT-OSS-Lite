# Positional Encodings — From Sinusoids to YaRN-Scaled RoPE

> **Theory chapter T2.** From-scratch derivation of position encoding: why
> attention needs it, the absolute sinusoidal construction, relative
> encodings, RoPE as a rotation, and the interpolation/extrapolation ladder
> (PI → NTK-aware → YaRN) that gets GPT-OSS-Lite from a 4,096-token training
> window to a 131,072-token evaluation window. Maps onto
> `models/rotary.py` and `models/yarn.py`. The implementation-focused
> companion is [rope_yarn](../rope_yarn.md) (worked numbers, dtype contract,
> SDPA interaction); this chapter supplies the derivations behind those
> numbers. Assumes the primer level of [foundations](../foundations.md); the
> softmax arithmetic that consumes the encodings is derived in
> [attention math](attention_math.md).

## Table of contents

1. [60-second summary](#1-60-second-summary)
2. [Why it matters here](#2-why-it-matters-here)
3. [Intuition](#3-intuition)
4. [Theory and derivation](#4-theory-and-derivation)
5. [Code walkthrough](#5-code-walkthrough)
6. [Pitfalls and verification](#6-pitfalls-and-verification)

---

## 1. 60-second summary

Attention is a weighted average of value vectors; without position
information the weights depend only on token content, so a transformer cannot
distinguish "A B" from "B A". Every position-encoding scheme injects a
sequence order signal, and the schemes form a ladder. Absolute sinusoidal
encodings add a fixed vector of sines and cosines to each token embedding;
the shift between two positions is a linear map (a rotation), which is the
seed of everything after. Learned absolute embeddings replace the fixed
table with parameters but extrapolate to nothing. Relative encodings
parameterize attention scores by token distance, buying shift-invariance at
the price of a bounded table or a fixed bias shape. RoPE achieves
relative-position behavior with no table: it rotates each query and key in
2D subspaces, so the dot product between a query at position $p$ and a key
at position $j$ depends only on $p - j$. RoPE fails beyond the trained
length, so GPT-OSS-Lite applies YaRN: keep the fastest pairs' frequencies,
divide the slowest pairs' frequencies by the scale factor 32, blend linearly
between, and multiply every rotation by `mscale = 0.1·ln(32) + 1 ≈ 1.347` to
restore attention sharpness. On global (full-attention) layers, the 24
fastest pairs are additionally frozen to identity (`cos=1, sin=0`) because at
131,072 tokens they have spun so many times that their phase is aliasing
noise. The whole pipeline lives in `models/rotary.py:compute_yarn_freqs`,
`models/rotary.py:compute_yarn_mscale`, `models/rotary.py:apply_rope`, and
`models/yarn.py:YaRNRoPE`.

## 2. Why it matters here

Position handling is one of the few places where GPT-OSS-Lite's entire
long-context story is decided by a handful of hyperparameters:

- **4K train, 128K eval.** Training runs at `max_seq_len = 4096` while
  evaluation targets `eval_max_seq_len = 131072` (`models/transformer.py:ModelConfig`)
  — a $32\times$ stretch. That stretch is the reason YaRN exists in this
  repo at all: plain RoPE extrapolation collapses far outside its training
  window (§4.6), and the target is 32× because the scale factor is literally
  the ratio of the two lengths (§4.7).
- **Alternating SWA/full.** Even layers are sliding-window ($W=128$), odd
  layers are full causal (`models/attention.py:GPTOSSAttention` decides via
  `layer_idx % 2`). The position-encoding treatment differs per branch:
  windowed layers keep all 48 frequency pairs; global layers prune the 24
  fastest (§4.8). The mask half of this split is derived in
  [attention math](attention_math.md).
- **GQA with `head_dim = 96`.** 8 query heads share 4 KV heads, so the
  rotary table is computed once per layer per sequence and broadcast across
  heads. `head_dim` must be even — the rotation lives in 2D subspaces — and
  `ModelConfig.__post_init__` enforces it.
- **Rotated K in the KV cache.** Keys are rotated *before* they enter
  `MixedKVCache`; at decode only the new query rotates. That is what makes
  the measured 2.00× KV reduction at 128K (1.94× at 4K,
  `scripts/kv_cache_benchmark.py`) compatible with RoPE at all — the cache
  math is in [kv cache engineering](kv_cache_engineering.md).
- **Budget honesty.** The ≥85% passkey @128K figure is a **target**; no
  pretraining run exists. The 2.00×/1.94× KV figures are **measured**.
  Everything in this chapter is derived arithmetic, tagged where it is a
  fitted constant or `[INFERENCE]`; no performance numbers are borrowed from
  `.benchmarks/` (it is empty).

## 3. Intuition

Picture each frequency pair of a position encoding as the hand of a clock
with its own gear ratio. The pair $(\cos(p\omega_m), \sin(p\omega_m))$ is
the hand's orientation after $p$ ticks. The fastest gears ($m$ small)
complete thousands of turns over a 131,072-token sequence — they can tell
you that token $p+3$ came after token $p$, but nothing about where in the
sequence you are. The slowest gears barely move: after 4K tokens the
slowest pair has rotated less than 4% of a turn, so its angle is a smooth,
monotone "absolute position" meter. RoPE attaches one such geared hand to
every 2D sub-vector of every query and key; the attention score between a
query and a key compares the *angles between their hands*, which is why the
score depends on the tick *difference* $p - j$ and not the absolute ticks.

Extending to 128K is a re-gearing problem. If you keep the original gear
ratios, the slow hands sweep into angles never seen during 4K training (the
network has no learned reading for them). Position Interpolation slows *all*
gears by 32, which keeps every hand inside its trained arc but blurs the
fast gears so badly that neighboring tokens become hard to tell apart. YaRN
re-gears selectively: leave the fastest hands alone, divide the slowest
hands' speeds by 32, blend in between — then stiffen the attention
temperature (`mscale`) because compressed gears make every hand look alike,
flattening the attention distribution. Finally, on the global layers, the
fastest gears are disconnected entirely: by 128K they have spun so many
times that their angle is effectively random noise.

## 4. Theory and derivation

### 4.1 Why positions at all: permutation invariance

Scaled dot-product attention (derived in [attention math](attention_math.md))
computes, row by row, a convex combination of value rows:

$$
o_i = \sum_{j=1}^{T} \alpha_{ij}\, v_j, \qquad
\alpha_{ij} = \mathrm{softmax}_j\!\left(\frac{q_i^\top k_j}{\sqrt{d}}\right)
\tag{1}
$$

where $q_i = W_q x_i$, $k_j = W_k x_j$, $v_j = W_v x_j$ are linear maps of the
input tokens $x_i \in \mathbb{R}^{d_{\text{model}}}$, and $d$ is the head
width. Nothing in (1) references $i$ or $j$ except through the token
contents $x_i, x_j$. If the input sequence is permuted by $\sigma$, the
hidden states move with their tokens, and

$$
\mathrm{Attn}(x_{\sigma(1)}, \dots, x_{\sigma(T)}) =
\sigma\big(\mathrm{Attn}(x_1, \dots, x_T)\big),
\tag{2}
$$

i.e. the layer is *equivariant to permutation*: reordering the input merely
reorders the output rows. The two sequences "the cat sat" and "sat cat the"
produce the same multiset of row vectors, so no downstream layer can ever
distinguish the orderings — the model would be a bag-of-tokens. Autoregressive
language modeling is impossible without an explicit, learnable position
signal injected somewhere in the forward pass. (Equation (2) holds for a
single layer with no mask; the argument is unchanged by any mechanism that
reads content only.) Everything in this chapter is a different way of
injecting that signal.

### 4.2 Absolute sinusoidal encodings

The original Transformer adds a fixed, deterministic vector to each token
embedding:

$$
\mathrm{PE}(p, 2m) = \sin(p\, \omega_m), \qquad
\mathrm{PE}(p, 2m+1) = \cos(p\, \omega_m), \qquad
\omega_m = \frac{1}{\theta^{2m/d}},
\tag{3}
$$

for position $p \in \{0, \dots, L-1\}$, pair index $m \in \{0, \dots, d/2-1\}$,
base $\theta = 10000$, and head/model width $d$ (the construction is used
with $d = d_{\text{model}}$). The input to the model becomes
$x_p = e_{t_p} + \mathrm{PE}(p)$, where $e_{t_p}$ is the token embedding.

**Why geometric frequencies.** Pair $m$ has wavelength

$$
\lambda_m = \frac{2\pi}{\omega_m} = 2\pi\, \theta^{2m/d},
\tag{4}
$$

the number of positions per full rotation. The frequencies are *log-uniform*:
$\omega_{m+1}/\omega_m = \theta^{-2/d}$ is constant, so every octave
(factor-2 band) of wavelength contains exactly $d\cdot \ln 2 / (2 \ln
\theta)$ pairs, regardless of scale. With $d = 768$ and $\theta = 10000$,
wavelengths run from $\lambda_0 = 2\pi \approx 6.3$ tokens to
$\lambda_{383} = 2\pi \cdot 10^{4\cdot 383/768} \approx 6.1\times 10^{4}$
tokens. A single sinusoid cannot resolve both "is token 3 vs 4" and "is
token 3,000 vs 4,000" — those need phases that move quickly and slowly
respectively — but a log-uniform bank of them covers every scale at once.

**Why a sin/cos pair.** One scalar $\sin(p\omega)$ cannot distinguish $p$
from $\pi/\omega - p$ and loses phase information at zero crossings. The
pair $(\sin, \cos)$ is the unit phasor $e^{ip\omega}$, a point on the circle
injective over one full period.

**Why it works: the shift is a linear map.** The decisive algebraic
property is that moving a position by $k$ is a *fixed linear transformation*
of its encoding. Using $\sin(a+b) = \sin a\cos b + \cos a\sin b$ and
$\cos(a+b) = \cos a\cos b - \sin a\sin b$:

$$
\begin{pmatrix} \sin((p-k)\omega_m) \\ \cos((p-k)\omega_m) \end{pmatrix}
=
\underbrace{\begin{pmatrix} \cos(k\omega_m) & -\sin(k\omega_m) \\
\sin(k\omega_m) & \cos(k\omega_m) \end{pmatrix}}_{R(k\omega_m)}
\begin{pmatrix} \sin(p\omega_m) \\ \cos(p\omega_m) \end{pmatrix},
\tag{5}
$$

with the $2\times2$ rotation matrix $R(\phi)$ — the exact same matrix RoPE
applies to queries and keys in §4.5. The set of encodings is
therefore a *translation-structured* manifold: the encoding of every
position $p+k$ is reachable from position $p$ by a rotation whose angle
depends only on the offset $k$ (here $PE(p+k) = R(-k\omega)PE(p)$, using
$R(-\phi) = R(\phi)^\top$). A linear layer can read "distance between
two encodings" as "relative rotation angle", which is why relative structure
is learnable at all — and it is the exact same rotation matrix that RoPE
reuses on queries and keys in §4.5. The practical caveat, measured by
Kazemnejad et al. (2023), is that this structure buys only *modest*
extrapolation: sinusoidally-encoded models degrade gracefully a little past
$L$ but not far.

### 4.3 Learned absolute embeddings

The alternative to a fixed table is to make the position vector
trainable:

$$
x_p = e_{t_p} + E[p], \qquad E \in \mathbb{R}^{L_{\max} \times d}.
\tag{6}
$$

Each position gets an independent parameter vector. This is strictly more
expressive than (3) — the model can learn whatever position geometry it
wants — but it has no structure to exploit: there is no systematic relation
between $E[p]$ and $E[p+1]$, and positions beyond $L_{\max}$ are undefined,
so extrapolation is impossible by construction (the common hack is to clamp
or train with longer windows). GPT-OSS-Lite uses no position embedding at
all at the input level; its only position mechanism is RoPE inside
attention, so `weight_tying` (`models/transformer.py:ModelConfig`) applies
to a purely token-level embedding.

### 4.4 Relative encodings

The semantic role of a token is largely a function of its *distance* from
the query: "the cat sat" and "a cat sat" have different absolute positions
for "cat" but the same local structure. Absolute encodings force the model
to learn a separate pattern for every absolute pair $(p, j)$; a relative
scheme parameterizes the score by $\delta = j - i$ directly. The canonical
forms are additive biases on the logit:

$$
s_{ij} = \frac{q_i^\top k_j}{\sqrt{d}} + b_{j-i},
\tag{7}
$$

with $b_\delta$ a learned bias table (Shaw et al. 2018; T5 buckets the
offset and shares biases), or a fixed linear bias $b_\delta = -m|\delta|$
(ALiBi), which needs no table and extrapolates to any length — but imposes a
fixed geometric prior the model cannot reshape. Relative encodings buy
shift-invariance and length behavior at the cost of either a bounded table
(offsets beyond the trained range must be clamped or bucketed) or a rigid
bias shape. RoPE, next, gets relative behavior with no table and no
length-dependent clamp.

### 4.5 RoPE: rotation, complex view, relative distance

Rotary Position Embedding (Su et al., 2021) applies position *inside* the
attention score rather than at the input. Split the head dimension into
$d/2$ pairs and rotate pair $m$ of the query at position $p$ and the key at
position $j$ by the angles $p\omega_m$ and $j\omega_m$ respectively:

$$
q'_m = R(p\,\omega_m)\, q_m, \qquad k'_m = R(j\,\omega_m)\, k_m,
\qquad R(\phi) = \begin{pmatrix} \cos\phi & -\sin\phi \\
\sin\phi & \cos\phi \end{pmatrix}.
\tag{8}
$$

The frequencies $\omega_m = \theta^{-2m/d}$ are the same geometric grid as
(3), and $R$ is the same matrix that appeared in (5). Because rotations
preserve length, RoPE does not distort the query/key magnitudes.

**Complex view.** Identify each pair with a complex number,
$(x_{2m}, x_{2m+1}) \leftrightarrow z_m = x_{2m} + i\,x_{2m+1}$. Rotation
by $\phi$ is multiplication by the unit phasor $e^{i\phi}$, so RoPE is:

$$
z'_m = z_m\, e^{i p \omega_m} \qquad \text{(query)}, \qquad
w'_m = w_m\, e^{i j \omega_m} \qquad \text{(key)}.
\tag{9}
$$

The real inner product between two 2D pairs is the real part of the
Hermitian inner product, so the score contribution of pair $m$ is:

$$
(q'_m)^\top k'_m = \operatorname{Re}\!\big[\, \bar z_m\, w_m\,
e^{i(p - j)\omega_m} \big].
\tag{10}
$$

Summing over pairs and writing the score with the usual $1/\sqrt{d}$
scaling gives the full attention logit:

$$
s_{ij} = \frac{1}{\sqrt{d}} \sum_{m=0}^{d/2-1}
\operatorname{Re}\!\big[\, \bar z_m w_m\, e^{i(p-j)\omega_m} \big],
\tag{11}
$$

which depends on the positions $p, j$ **only through the difference**
$\delta = p - j$. Why this falls out: rotations compose ($e^{i a}e^{i b} =
e^{i(a+b)}$) and are unitary, so

$$
(R(p\omega)\, q)^\top (R(j\omega)\, k) = q^\top R(p\omega)^\top R(j\omega)\, k
= q^\top R\big((j - p)\omega\big)\, k,
\tag{12}
$$

an orthogonal-matrix identity (the second equality uses $R(\phi)^\top =
R(-\phi)$ and $R(a)R(b) = R(a+b)$). The relative offset $\delta$ is
*built into the geometry* — no bias table, no bucketing, and the same
frequency bank covers local syntax (fast pairs) and long-range structure
(slow pairs) simultaneously. Position 0 rotates nothing ($R(0) = I$), so a
query at the very first token is unmodified — a property the sink bias
interacts with ([ATTENTION_SINKS](../ATTENTION_SINKS.md)).

A second practical consequence of (10)–(12): because a key's rotation
depends only on its *own* absolute position, keys can be rotated once when
they are produced and cached in rotated form; every future query recovers
the relative offset automatically at score time. GPT-OSS-Lite relies on this
exactly — `inference/generate.py:_attn_forward_layer` rotates `k_new`
before inserting it into the cache, and only the new query is rotated at
decode (§5.6).

### 4.6 Interpolation vs extrapolation

RoPE trained at length $L$ has seen only the phases $\phi_m(p) = p\omega_m$
for $p \in [0, L]$. At evaluation length $L' = sL$ there are two disjoint
failure modes.

**Slow pairs extrapolate into unseen phases.** For a pair with wavelength
$\lambda_m > L$ — the slow pairs — the trained phase range $[0, L\omega_m]$
is a *proper sub-arc* of the circle:

$$
\text{trained phase range: } [0, L\omega_m] \subset [0, 2\pi),
\qquad
\text{at } sL: [0, sL\omega_m].
\tag{13}
$$

The pair completes fewer than one rotation over the whole training
sequence, so its phase is a monotone "absolute position" meter (§3); at
$p > L$ the meter enters angles the network has never seen, and the learned
map from phase to representation is undefined there. These are the pairs the
YaRN paper calls absolute-position carriers — the model demonstrably uses
them (they are why RoPE is not purely relative in practice).

**Fast pairs alias.** For a fast pair, $\phi_m(p) \bmod 2\pi$ repeats every
$2\pi/\omega_m$ tokens, so absolute position is unreadable — but that was
already true at training length, and every phase value is in-distribution,
so fast pairs keep functioning as *relative* encoders. Their problem at
extension is that they contribute nothing to the absolute-position signal
that long-range attention needs; positions $p$ and $p + 2\pi/\omega_m$
produce identical phases. Frequency aliasing in the strict sense: $\cos
(\phi) = \cos(\phi + 2\pi k)$, and for $\omega_0 = 1$ the period is $2\pi
\approx 6.28$ tokens.

**Position Interpolation (PI)** (Chen et al., 2023) sidesteps the slow-pair
failure by never leaving the trained phase envelope: evaluate the rotation
at compressed positions, $f'_W(x_m, m, \theta) = f_W(x_m, mL/L', \theta)$,
i.e. $\phi'_m(p) = p\omega_m/s$. Every phase stays inside $[0, L\omega_m]$.
The cost: *all* frequencies drop by $s$, so adjacent tokens differ in phase
by $\omega_m/s$ — the fastest pair moves $\sim 1/32$ rad per token instead
of 1 rad, and fine-grained local ordering collapses ("loss of high-frequency
information"; the NTK/Fourier-feature argument of Tancik et al., 2020).
Empirically PI degrades beyond $s \approx 8$ even with fine-tuning.

**NTK-aware scaling** (bloc97, 2023) spreads the interpolation pressure
across dimensions by changing the base instead of the positions. Choose the
new base $\theta'$ so that the *slowest* pair is stretched by exactly $s$
(as PI would) while the *fastest* pair ($\omega_0 = 1$, base-independent) is
untouched. The slowest pair has index $d/2 - 1$ (the pair with exponent
$2(d/2 - 1)/d = (d-2)/d$), and requiring its wavelength to multiply by $s$:

$$
2\pi (\theta')^{(d-2)/d} = s \cdot 2\pi \theta^{(d-2)/d}
\quad\Longrightarrow\quad
\theta' = \theta \cdot s^{d/(d-2)}.
\tag{14}
$$

Every intermediate pair then scales by an intermediate factor,
$\omega_m(\theta') = (\theta')^{-2m/d} = \omega_m(\theta)\cdot
s^{-2m/(d-2)}$: high frequencies keep almost their original rate, low
frequencies get almost full PI compression — no position-dependent gating,
one hyperparameter $\theta'$. For GPT-OSS-Lite's geometry the new base is
$\theta' = 10^{5} \cdot 32^{48/47} \approx 3.5\times 10^{6}$. The drawbacks
(the paper's A.2): it is not a true interpolation — the fast pairs still
extrapolate to slightly out-of-range phase values — and the right $\theta'$
for a target $s$ has to be found empirically. NTK-aware motivates the
explicit per-pair policy that YaRN makes: decide, pair by pair, whether to
keep, compress, or blend.

### 4.7 YaRN: the ramp, `mscale`, and why scale = 32

**Per-pair policy by rotation count.** Define the number of rotations pair
$m$ completes over the training length:

$$
r_m = \frac{L}{\lambda_m} = \frac{L\, \omega_m}{2\pi}.
\tag{15}
$$

Large $r_m$ (fast pairs, $\lambda_m \ll L$) means the pair only ever encodes
*relative* offsets — it must be left alone. Small $r_m$ (slow pairs,
$\lambda_m \gtrsim L$) means the pair encodes *absolute* position — it must
be compressed by $s$, never extrapolated. NTK-by-parts (bloc97, 2023;
formalized in the YaRN paper) interpolates between a linear scale for
$r_m < \alpha$ and no scaling for $r_m > \beta$ with a linear ramp in
between:

$$
\gamma(r_m) = \mathrm{clamp}\!\left(\frac{r_m - \alpha}{\beta - \alpha},\,
0, 1\right), \qquad
\omega'_m = \omega_m\,(1 - \gamma_m) + \frac{\omega_m}{s}\,\gamma_m,
\tag{16}
$$

with recommended $\alpha = 1$, $\beta = 32$ (tuned on the Llama family).
$\gamma_m = 0$ keeps the base frequency; $\gamma_m = 1$ applies full PI
compression $\omega_m \to \omega_m/s$.

**The code's boundary closed form.** Instead of computing $r_m$ per pair,
`models/rotary.py:compute_yarn_freqs` selects two dim indices directly:

$$
\mathrm{low} = \left\lfloor \frac{d/2}{\log_2\!\left(\frac{L}{\beta_{\mathrm{slow}}}\cdot \pi\right)} \right\rfloor, \qquad
\mathrm{high} = \left\lceil \frac{d/2}{\log_2\!\left(\frac{L}{\beta_{\mathrm{fast}}}\cdot \pi\right)} \right\rceil,
\tag{17}
$$

then ramps linearly in *dim index*: $\gamma_m = \mathrm{clamp}((m -
\mathrm{low})/(\mathrm{high} - \mathrm{low}), 0, 1)$ and blends exactly as
in (16), with $\beta_{\mathrm{slow}} = \alpha = 1$ and
$\beta_{\mathrm{fast}} = \beta = 32$ as the defaults. Note the expression
evaluates as $(L/\beta)\cdot\pi$ (division before multiplication). With the
production values $L = 4096$, $d/2 = 48$, $\beta_{\mathrm{slow}} = 1$,
$\beta_{\mathrm{fast}} = 32$:

$$
\log_2(4096\pi) = 13.65 \Rightarrow \mathrm{low} = \lfloor 48/13.65 \rfloor = 3,
\qquad
\log_2(4096\pi/32) = 8.65 \Rightarrow \mathrm{high} = \lceil 48/8.65 \rceil = 6.
\tag{18}
$$

So pairs $m \le 3$ keep their base frequencies ($\omega = 1.00, 0.79, 0.62,
0.49$), pairs $m \ge 6$ are divided by 32, and pairs 4–5 blend with
$\gamma = 1/3, 2/3$. This closed form is the implementation lineage's
boundary rule (the YaRN/jquesnelle → HF Transformers form); it is more
aggressive than reading the paper's $\alpha,\beta$ as rotation-count
thresholds directly — the r-threshold reading would put the ramp at dims
$\approx 13$–27 for this geometry (derived from (15): $r_m = \beta = 32$ at
$m = 48\cdot\ln(4096/64\pi)/\ln 10^5 \approx 12.6$; $r_m = \alpha = 1$ at
$m \approx 27$). Both implement the same intent — preserve the fastest
pairs, compress the slowest — and the boundary position is a hyperparameter
of the implementation; the invariant that matters is that $\gamma$ is 0
below $\mathrm{low}$, 1 above $\mathrm{high}$, and linear between. A 32×
stretch compresses 42 of 48 pairs; only the four fastest escape.

**`mscale`: the attention-temperature correction.** Compressing the slow
pairs' frequencies makes their phase differences shrink by up to $s$, so
the *contrast* of logits across keys drops (softmax is shift-invariant, so
only contrast matters), the attention distribution flattens, and its
entropy rises. YaRN compensates with a temperature $t$ on the logits
(paper Eq. 14) and uses the "length scaling trick": scaling both $q$ and
$k$ by $\sqrt{1/t}$ is equivalent to dividing the logits by $t$, and can be
implemented by scaling the rotary embeddings alone. Fitting
$\sqrt{1/t}$ against perplexity across LLaMA 7B–65B gives (paper Eq. 15):

$$
\sqrt{1/t} = \mathrm{mscale}(s) = 0.1\ln(s) + 1,
\qquad t = \frac{1}{\mathrm{mscale}(s)^2}.
\tag{19}
$$

The *form* $1 + c\ln s$ is what the log-uniform spectrum dictates: scaling
by $s$ shifts every pair's effective frequency down by $s$, and the number
of pairs whose phase behavior changes over the training span is
proportional to the log-frequency width $\ln s$ (pairs are uniformly
spaced in log-frequency, so a factor-$s$ shift crosses $\propto \ln s$ of
them). The constant $c = 0.1$ is **fitted, not derived** — a fitted
constant and the transferable observation that the entropy shift is
roughly universal across models. GPT-OSS-Lite multiplies every `cos`/`sin`
by `mscale` in `models/yarn.py:YaRNRoPE.forward`, which scales each of $q$
and $k$ by `mscale`, i.e. the logits by $\mathrm{mscale}^2 = 1/t$. For
$s = 32$:

$$
\mathrm{mscale} = 0.1\ln 32 + 1 \approx 1.347, \qquad
\mathrm{mscale}^2 \approx 1.813, \qquad t \approx 0.552.
\tag{20}
$$

`models/rotary.py:compute_yarn_mscale` returns 1.0 for $s \le 1$, so plain
RoPE is the $s=1$, zero-ramp limit of the whole mechanism.

**Why scale = 32.** The scale factor is defined as the ratio of the target
to the original length:

$$
s = \frac{L'}{L} = \frac{131072}{4096} = 32,
\tag{21}
$$

and 32 is exactly the recipe the YaRN paper validates for 4k→128k
extensions (its $s=32$ models were fine-tuned on 64k data and still
extrapolated to 128k). `models/transformer.py:ModelConfig.__post_init__`
enforces the sanity conditions: $s \ge 1$, and whenever $s > 1$ the
original length must be strictly below the target.

### 4.8 Pruning on global layers: the over-rotation argument

At position $p$, pair $m$ completes

$$
\nu_m(p) = \frac{p\, \omega_m}{2\pi}
\tag{22}
$$

full turns. A pair's phase is *coherent* (usable for position) only while
its total rotation over the span stays small — once $\nu_m \gg 1$, the
phase $\cos(p\omega_m \bmod 2\pi)$ at any fixed query/key offset is
effectively a pseudorandom function of the offset, oscillating with period
$2\pi/\omega_m$ tokens and averaging to zero over a large key set: pure
aliasing noise. The pair where $\nu = 1$ at 128K satisfies $\omega_m =
2\pi/131072 \approx 4.79\times10^{-5}$, which lands at $m \approx 27$ on
the YaRN-scaled grid. Computed turn counts at $p = 131072$:

| pairs | turns $\nu$ at 128K | role |
|---|---|---|
| $m = 0$ (fastest, unscaled) | 20,861 | fully aliased |
| $m = 3$ (last unscaled) | 10,159 | fully aliased |
| $m = 23$ (last pruned) | 2.6 | aliased |
| $m = 24$ (first kept) | 2.1 | marginal |
| $m = 27$ | 1.0 | aliasing threshold |
| $m = 47$ (slowest) | 0.0083 | fully coherent |

Global (full-attention) layers score a query against *all* prior keys, so
every aliased pair injects incoherent noise into the long-range logits —
and the worst offenders are exactly the four unscaled pairs ($m = 0..3$),
which YaRN deliberately preserves for local resolution that global layers
do not need. GPT-OSS-Lite therefore freezes the 24 fastest pairs to identity
on global layers: `models/attention.py:GPTOSSAttention._n_pruned_dims`
returns `head_dim // 4 = 24` (in *pair* units) when the layer is not
windowed and `yarn_prune_rope_global` is set, and
`models/yarn.py:YaRNRoPE.forward` then overwrites the first 24 columns of
`cos`/`sin` with 1.0/0.0. Since the frequency table is ordered fastest-first
(§5.1), those are the pairs $m = 0..23$ — 24 pairs, i.e. 48 of the 96
scalar channels (half the head; "25%" is `head_dim // 4` expressed relative
to `head_dim`). The pruning is applied *after* the `mscale` multiply, so a
pruned pair contributes exactly $(\cos, \sin) = (1, 0)$: no rotation and no
magnitude scaling.

Windowed layers keep all 48 pairs. They attend within 128 tokens, where
pair $m$ still resolves offsets up to $\pi/\omega_m$ (phase below $\pi$):
the compressed mid pairs resolve hundreds to tens of thousands of tokens —
far more than the window — and the fast pairs carry the near-token order
that sliding-window attention exists to provide. The 12-layer alternation
(even = SWA, odd = full; [ATTENTION_SINKS](../ATTENTION_SINKS.md)) thus
splits positional labor: windowed layers do fine-grained local positioning
with full RoPE; global layers do long-range content matching using only the
32×-slowed, phase-coherent slow pairs. Because pruning is a pure function of
layer parity, prefill and decode compute identical `cos`/`sin` for the same
positions — required, since rotated keys are cached (§5.6).

## 5. Code walkthrough

### 5.1 `compute_yarn_freqs` — ramp math

`models/rotary.py:compute_yarn_freqs` builds the full YaRN frequency table
in one pass:

```python
half = head_dim // 2
exponents = torch.arange(0, half, dtype=torch.float32) / half
base = 1.0 / (theta ** exponents)
```

`exponents` is $m / \mathrm{half}$, i.e. $2m/d$ (equivalent, since $\mathrm{half}
= d/2$), so `base[m] = θ^{-2m/d} = ω_m` — the geometric grid of §4.2, in
*descending* order (fastest pair first). The ramp bounds implement (17)
verbatim:

```python
low = max(math.floor(half / math.log2(original_max_seq_len / beta_slow * math.pi)), 0)
high = min(math.ceil(half / math.log2(original_max_seq_len / beta_fast * math.pi)), half - 1)
```

Note the precedence: `original_max_seq_len / beta_slow * math.pi` is
$(L/\beta)\cdot\pi$, not $L/(\beta\pi)$ — worth re-reading when comparing
against other implementations, since the parenthesization shifts the ramp by
several dims ([rope_yarn §10.2](../rope_yarn.md) reads it the other way).
For production values the branch is:

```python
ramp = torch.clamp(
    (torch.arange(half, dtype=torch.float32) - low) / max(high - low, 1),
    0.0, 1.0,
)
inv_freq = base * (1.0 - ramp) + (base / scale_factor) * ramp
```

This is (16) exactly: $\gamma_m = 0$ for $m \le \mathrm{low}$ keeps
`base`; $\gamma_m = 1$ for $m \ge \mathrm{high}$ yields `base / s`;
between, a linear blend. If `high <= low` the ramp is degenerate — the
function emits a `UserWarning` and falls back to `ramp = zeros`, i.e.
identity RoPE with no length extension (guarded by
`tests/test_yarn.py::test_compute_yarn_freqs_warns_on_degenerate_ramp`). The
two `ValueError`s (odd `head_dim`, non-positive lengths) are the first line
of defense for the even-`head_dim` invariant (§6). Note `target_seq_len` is
accepted for API symmetry but unused: the extrapolation length is implicit
in `scale_factor` and the ramp, not a separate clamp. The returned
`inv_freq` has shape `(half,)` — `tests/test_yarn.py::test_yarn_freqs_shape`
and `test_yarn_freqs_no_nan` (finite and positive at production scale) pin
this.

### 5.2 `compute_yarn_mscale`

`models/rotary.py:compute_yarn_mscale` is (19) with the $s \le 1$ guard:

```python
if scale_factor <= 1.0:
    return 1.0
return 0.1 * math.log(scale_factor) + 1.0
```

`math.log` is the natural log, matching $\ln s$ in (19). `mscale` is a
Python float, not a tensor — it is folded into `cos`/`sin` at forward time.
`tests/test_yarn.py::test_yarn_mscale_basic` checks the $s=1$ identity and
monotonicity ($s=32$ gives 1.347 > $s=4$ gives 1.139).

### 5.3 `YaRNRoPE` — construction

`models/yarn.py:YaRNRoPE.__init__` validates `head_dim % 2 == 0`, stores the
hyperparameters, then calls `compute_yarn_freqs` and registers the result as
a *non-persistent* buffer:

```python
self.register_buffer("inv_freq", inv_freq, persistent=False)
```

`persistent=False` means the table is not saved in `state_dict` — it is
recomputed from hyperparameters at load time, so a checkpoint cannot carry a
stale table. `self.mscale = compute_yarn_mscale(scale_factor)` when
`mscale=True`, else `1.0`; the enabled flag is kept separately as
`self.mscale_enabled` for `extra_repr`. One `YaRNRoPE` instance is created
per attention layer (`models/attention.py:GPTOSSAttention.__init__`), each
with identical hyperparameters — 48 floats per layer, negligible.

### 5.4 `YaRNRoPE.forward` — scalar fast path, outer product, pruning

`models/yarn.py:YaRNRoPE.forward(positions, n_pruned_dims=0)` computes
$(\cos, \sin)$ tables of shape $(T, \mathrm{half})$:

```python
if positions.numel() == 1:
    inv_freq = self.inv_freq.to(positions.device)
    pos = positions.item() if positions.dim() == 0 else positions[0].item()
    freqs = inv_freq * float(pos)
    cos = freqs.cos().unsqueeze(0) * self.mscale
    sin = freqs.sin().unsqueeze(0) * self.mscale
else:
    freqs = torch.outer(positions.float(), self.inv_freq.to(positions.device))
    cos = freqs.cos() * self.mscale
    sin = freqs.sin() * self.mscale
```

The single-position branch is the decode fast path: it avoids materializing
a $(1, \mathrm{half})$ outer product and, more importantly, avoids the
`positions.float()` cast of a CPU int64 scalar. `inv_freq` is moved to
`positions.device` but stays FP32; `cos`/`sin` are computed in FP32 and
scaled by `mscale` — the phase is untouched, only the magnitude, which is
exactly the length-scaling trick of §4.7. Position 0 therefore gives
$\cos = \mathrm{mscale}$, $\sin = 0$ (`tests/test_yarn.py::test_yarn_module_zero_position_is_identity`),
and $\cos^2 + \sin^2 = \mathrm{mscale}^2$ for every pair
(`test_yarn_module_cos_sin_pair`). Pruning comes last:

```python
if n_pruned_dims > 0:
    cos = cos.clone()
    sin = sin.clone()
    cos[:, :n_pruned_dims] = 1.0
    sin[:, :n_pruned_dims] = 0.0
```

The `clone` keeps the non-pruned path free of in-place writes (safe for
autograd) and the overwrite is exactly 1.0/0.0 — *after* the `mscale`
multiply, so pruned pairs are pure identity (§4.8). The columns are the
first `n_pruned_dims` of the fastest-first table, i.e. pairs $m =
0..n{-}1$. `models/yarn.py:YaRNRoPE.extra_repr` prints the full
configuration (`head_dim, theta, scale_factor, original_max, target,
mscale_enabled`), which is what appears in `repr(model)` for layer-level
debugging.

### 5.5 `apply_rope` — repeat, rotate-half, dtype

`models/rotary.py:apply_rope(x, cos, sin)` implements the rotation (8)
without ever constructing a complex tensor:

```python
T = x.size(-2)
half = x.size(-1) // 2

cos_full = cos.repeat_interleave(2, dim=-1).to(x.dtype)
sin_full = sin.repeat_interleave(2, dim=-1).to(x.dtype)
```

(`T`, the position length, is read but not used further; the broadcast
below realigns `cos`/`sin` with the position axis.) Then:

```python
x_pairs = x.unflatten(-1, (-1, 2))
x_swapped = x_pairs.flip(-1)
x_swapped[..., 0] = -x_swapped[..., 0]
x_rotated = x_swapped.flatten(-2)
```

`cos`/`sin` carry one value per *pair*; `repeat_interleave(2)` fans each
pair's value onto its two scalar channels $(2m, 2m+1)$, so
`cos_full[..., 2m] == cos_full[..., 2m+1]`. The `unflatten`/`flip`/`negate`
dance computes the "rotate-by-90°" twin: per pair, $(x_0, x_1) \to
(-x_1, x_0)$, which is $R(\pi/2)\,x$ — the complex-multiply identity

$$
x \cdot e^{i\phi} = x\cos\phi + R(\pi/2)\,x \sin\phi,
\qquad
R(\pi/2)\begin{pmatrix}x_0\\ x_1\end{pmatrix} =
\begin{pmatrix}-x_1\\ x_0\end{pmatrix}.
\tag{23}
$$

Then:

```python
while cos_full.dim() < x.dim():
    cos_full = cos_full.unsqueeze(0)
    sin_full = sin_full.unsqueeze(0)

return x * cos_full + x_rotated * sin_full
```

The `while` broadcasts the $(T, d)$ tables over batch and head dimensions
(positions sit at `x.size(-2)`), and the final line is (23) applied to every
pair simultaneously. The `.to(x.dtype)` casts on `cos_full`/`sin_full` are
the dtype contract: if `x` is BF16 and `cos` were FP32, the multiply would
promote activations to FP32 and break `F.scaled_dot_product_attention`'s
requirement that Q/K/V share a dtype. Rotation is norm-preserving when
$\cos^2 + \sin^2 = 1$ (per-pair magnitudes unchanged —
`tests/test_yarn.py::test_apply_rope_magnitude_preserved`); with `mscale`
applied the preserved quantity is $\mathrm{mscale}^2$ instead, and
`test_apply_rope_zero_rotation` pins the $\cos{=}1, \sin{=}0$ identity
case that pruning produces.

### 5.6 Call sites: prefill vs decode, rotated K

`models/attention.py:GPTOSSAttention.forward` computes

```python
cos, sin = self.yarn(positions, n_pruned_dims=self._n_pruned_dims())
query_states = apply_rope(query_states, cos, sin)
key_states = apply_rope(key_states, cos, sin)
```

with `positions = torch.arange(T)` by default (prefill). At decode,
`inference/generate.py:generate` passes `positions_step = torch.tensor([cur_pos - 1])`
— a single-element tensor, which selects the scalar fast path of §5.4 — and
`inference/generate.py:_attn_forward_layer` rotates the fresh key
`k_new` *before* `MixedKVCache.append` stores it. Cached keys are thus
stored pre-rotated; only the incoming query is rotated each step, and the
relative offset is recovered inside the attention score by (12). Because
`models/attention.py:GPTOSSAttention._n_pruned_dims` depends only on layer
parity and the `yarn_prune_rope_global` flag, prefill and decode always
agree on which pairs are frozen — a mismatch here would corrupt every cached
key.

## 6. Pitfalls and verification

| Failure mode | Symptom | Guard |
|---|---|---|
| Odd `head_dim` | `ValueError: head_dim must be even` | `models/rotary.py:compute_yarn_freqs`, `models/yarn.py:YaRNRoPE.__init__`, `models/transformer.py:ModelConfig.__post_init__` — the rotation pairs (8) need $d$ even; 96 is fine |
| Degenerate ramp (`high <= low`) | `UserWarning: YaRN ramp degenerate` + identity fallback (no extrapolation) | `tests/test_yarn.py::test_compute_yarn_freqs_warns_on_degenerate_ramp`; fix `beta_fast`/`beta_slow` or lengths; `test_compute_yarn_freqs_no_warning_for_normal_params` pins the healthy path |
| Misread parenthesization in (17) | Ramp shifted by several dims vs other implementations | The code evaluates $(L/\beta)\cdot\pi$; `log2(4096π)=13.65` → `low=3`, `high=6` (not 4/9) — re-derive from `models/rotary.py:compute_yarn_freqs` directly |
| Mixed-dtype SDPA | `F.scaled_dot_product_attention` failure | `apply_rope` casts `cos`/`sin` to `x.dtype` before the multiply; `tests/test_yarn.py::test_apply_rope_*` + attention tests |
| Pruning vs `mscale` ordering | Pruned pairs scaled instead of identity | Pruning overwrites *after* the `mscale` multiply → exactly $(1, 0)$; `tests/test_yarn.py::test_yarn_module_pruned_dims` asserts the exact values |
| `mscale` correctness | Wrong temperature at extension | `cos^2+sin^2 = mscale^2` per pair, `mscale = 0.1·ln(32)+1 ≈ 1.347`; `tests/test_yarn.py::test_yarn_module_cos_sin_pair`, `test_yarn_mscale_basic` |
| Config inconsistencies | Silent wrong stretch | `ModelConfig.__post_init__`: $s \ge 1$; if $s > 1$ then original < target; positive lengths |
| Cache/table mismatch | Corrupt keys after decode step | `inv_freq` is a `persistent=False` buffer (recomputed at load); pruning is a function of layer parity only |

The single command that guards this entire chapter's arithmetic is

```bash
python3 -m pytest tests/test_yarn.py -v
```

It covers table shape/finiteness, the low/high spread (fastest pair
unchanged at 1.0, slowest pair divided by ~32), `mscale` identity and
monotonicity, the 4K-vs-128K distinctness of the rotation tables, position-0
identity, the $\cos^2+\sin^2$ invariant, monotone rotation of the fast
dims, pruning, the degenerate-ramp warning, and all three `apply_rope`
contracts (identity, shape, per-pair magnitude). Beyond it, the attention
integration is exercised by the mask/regression tests referenced in
[attention math](attention_math.md) and [ATTENTION_SINKS](../ATTENTION_SINKS.md).
No pretraining run exists yet, so the ≥85% passkey @128K figure remains a
**target**, not a result; what is verifiable today is the arithmetic above
and the measured 2.00×/1.94× KV reduction
([kv cache engineering](kv_cache_engineering.md), `scripts/kv_cache_benchmark.py`).

## Related documentation

- [rope_yarn](../rope_yarn.md) — implementation-focused companion: worked
  numerical examples, dtype/SDPA contract, debugging table, invariants.
- [attention math](attention_math.md) — the softmax/mask arithmetic that
  consumes `cos`/`sin`.
- [ATTENTION_SINKS](../ATTENTION_SINKS.md) — sink bias, windowed vs global
  split, interaction with pruned RoPE.
- [kv cache engineering](kv_cache_engineering.md) — rotated-K caching and
  the measured KV reduction.
- [foundations](../foundations.md) — primer: attention, GQA, SWA.

<!-- docs:verified 2026-08-04 · 5da1a80 -->
