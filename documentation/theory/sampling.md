# Sampling — Autoregressive Decode and Token Selection

> **Chapter on `inference/generate.py`.** How a trained decoder turns logits into
> tokens: autoregressive factorization, greedy decoding, temperature (Boltzmann)
> scaling, top-k / top-p truncation, entropy and perplexity, and the determinism
> rules that make the passkey benchmark meaningful. Engine-level decode (KV
> cache, ring buffers) is [inference.md](../inference.md) and
> [kv cache engineering](kv_cache_engineering.md); the softmax that underlies
> every equation here is derived in [attention math](attention_math.md).

---

## 60-second summary

A transformer's final layer emits one **logit** $z_i$ per vocabulary token $i$ — an
unnormalized score. Sampling is the layer that turns those scores into an actual
next token. The default recipe in this repo: scale logits by $1/T$ (temperature),
take a softmax to get a probability distribution over the 128,000-token vocabulary,
truncate the distribution to the smallest high-probability set whose cumulative
mass reaches `top_p`, then draw one token from that truncated distribution.
`temperature <= 0` skips all of it and just takes the argmax (greedy).

Why it matters: greedy decode is deterministic and reproducible, but it loops and
degrades on open-ended text; sampling adds controlled stochasticity. GPT-OSS-Lite's
headline evaluation — passkey retrieval at up to 131,072 tokens — runs **greedy**
(`temperature=0.0`), because a retrieval task has one correct answer and sampling
would add variance to the accuracy estimate. Everything lives in
`inference/generate.py:generate`; the benchmark harness is
`inference/long_context.py:PasskeyEvaluator.evaluate`.

## 1. Why it matters here

- **Generation quality.** Greedy decoding maximizes per-step probability, not
  sequence quality: at every step it re-picks the same high-mass regions, which
  produces repetitive loops on open-ended generation. Sampling from the
  temperature-scaled distribution lets the model explore lower-ranked continuations.
- **The headline metric is a retrieval task.** The passkey benchmark inserts a
  five-digit number into filler text and asks the model to repeat it
  ([inference.md](../inference.md)). There is exactly one right answer, so the
  eval uses greedy decoding — see §5 for the variance argument.
- **A 502M-parameter budget makes decode cost visible.** The 12-layer stack
  (alternating SWA(128)/full attention) was chosen partly so long-context decode
  stays cheap: the mixed cache stores only 128 tokens on six layers
  ([foundations.md](../foundations.md), [ATTENTION_SINKS.md](../ATTENTION_SINKS.md)).
  The sampling layer itself is O(V) per token ($V = 128000$), negligible next to
  attention — but it runs once per generated token, so its semantics (and any
  truncation) are the only thing standing between logits and the emitted text.
- **Honesty boundary.** No pretraining run has happened yet; the ≥85% passkey
  accuracy at 128K is a **target**, not a result. What is measured today: the
  2.00× KV reduction at 128K ([operations.md](../operations.md)) and the 192-test
  CPU suite.

## 2. Intuition

Think of the vocabulary as a terrain over which the model has placed one "altitude"
per token — the logit. A softmax is a camera looking at that terrain: it turns
altitudes into a probability distribution whose mass concentrates near the peaks.
**Temperature** is the zoom knob: $T < 1$ magnifies altitude differences so the
highest peak dominates (at $T \to 0$ you see only the summit — greedy); $T > 1$
flattens the terrain so foothills get a fair share (at $T \to \infty$ everything is
flat — uniform). **Top-p** then says "ignore everything below this elevation
contour": keep only the smallest set of peaks that together hold a fraction $p$ of
the total mass, and renormalize. Finally, instead of always planting the flag on
the tallest remaining peak (greedy), **sampling** drops a ball on the terrain and
lets the mass distribution decide where it lands — usually the peak, occasionally
a shoulder. That occasional "shoulder" is what prevents degenerate repetition.

## 3. Theory and derivation

### 3.1 Autoregressive factorization

A language model assigns a probability to a token sequence $x_1, \ldots, x_T$ from
a vocabulary $\mathcal{V}$ of size $V$ by the chain rule of probability, conditioned
only on the past:

$$
P(x_1, \ldots, x_T) = \prod_{t=1}^{T} P(x_t \mid x_{<t}), \qquad x_{<t} = (x_1, \ldots, x_{t-1})
\tag{1}
$$

The decoder-only transformer (causal mask, [foundations.md](../foundations.md))
implements each conditional $P(x_t \mid x_{<t})$ as a softmax over logits
$z^{(t)} \in \mathbb{R}^{V}$ produced by the shared stack. This factorization is
what makes generation a loop: sample $x_T$, append, sample $x_{T+1}$, repeat — the
prefix never needs to be re-sampled.

### 3.2 Greedy decoding

Greedy decoding picks the most probable token at every step:

$$
\hat{x}_t = \arg\max_{v \in \mathcal{V}} P(v \mid x_{<t}) = \arg\max_v z^{(t)}_v
\tag{2}
$$

The second equality holds because softmax is monotonic in its argument. Greedy is
the mode of each conditional — but the mode of each conditional is not generally the
mode of the joint (1): a low-probability token can be the right choice if it opens a
long high-probability continuation. That gap is the theoretical justification for
sampling. Greedy is also exactly the $T \to 0$ limit of temperature sampling
(§3.3), and `inference/generate.py:generate` treats every `temperature <= 0` as
greedy.

### 3.3 Temperature — Boltzmann scaling

The sampling distribution is the softmax of logits scaled by $1/T$:

$$
p_i = \frac{\exp(z_i / T)}{\sum_{j=1}^{V} \exp(z_j / T)}, \qquad T > 0
\tag{3}
$$

This is the Boltzmann distribution of statistical mechanics with $T$ in the role of
temperature: states (tokens) with higher energy-like score $z_i$ get exponentially
more mass. To derive the limits, shift by the maximum logit
$z_{\max} = \max_j z_j$ (numerically stable, probabilities unchanged):

$$
p_i = \frac{\exp\big((z_i - z_{\max})/T\big)}{\sum_{j=1}^{V} \exp\big((z_j - z_{\max})/T\big)}
\tag{4}
$$

**Limit $T \to 0^+$.** Let $A = \{ i : z_i = z_{\max} \}$ be the set of argmax
indices. For $i \notin A$, the exponent $(z_i - z_{\max})/T \to -\infty$, so
$\exp(\cdot) \to 0$; for $i \in A$ the exponent is $0$ and $\exp(0) = 1$. Hence

$$
\lim_{T \to 0^+} p_i = \begin{cases} 1/|A| & i \in A \\ 0 & i \notin A \end{cases}
\tag{5}
$$

i.e. a uniform draw over the tied argmax tokens — greedy when $|A| = 1$ (the usual
case), which is exactly the code path `next_token_logits.argmax(dim=-1)`.

**Limit $T \to \infty$.** Every exponent in (4) tends to $0$, so every $\exp$ tends
to $1$ and

$$
\lim_{T \to \infty} p_i = \frac{1}{V},
\tag{6}
$$

the uniform distribution. Greedy and uniform are the two endpoints of the family
(3); every finite $T$ interpolates between them.

**What $T = 0.7$ does.** Compare two tokens by probability ratio — a scale-free
quantity:

$$
\frac{p_i}{p_j} = \exp\!\left(\frac{z_i - z_j}{T}\right).
\tag{7}
$$

At $T = 1$ the ratio is $\exp(z_i - z_j)$ (plain softmax). At $T = 0.7 < 1$ the
exponent is multiplied by $1/T \approx 1.43$, so the ratio between any two tokens is
amplified: $p_{\text{high}}$ grows, $p_{\text{low}}$ shrinks, and the distribution
becomes **sharper** than the raw softmax while remaining non-degenerate — it leans
toward the mode without committing to it. That is the default in
`inference/generate.py:generate`. (Training uses plain cross-entropy on the $T=1$
softmax, [training.md](../training.md); temperature is purely a decode-time knob.)

### 3.4 Top-k truncation

Top-k keeps only the $k$ largest logits (equivalently, the $k$ largest
probabilities) and renormalizes the rest to zero:

$$
\tilde{p}_i = \begin{cases} p_i \big/ \displaystyle\sum_{j \in S_k} p_j & i \in S_k \\[6pt] 0 & i \notin S_k \end{cases}
\qquad S_k = \{ \text{the } k \text{ indices with largest } z_i \}
\tag{8}
$$

Effect: the tail of the distribution — thousands of near-zero probabilities that
together hold real mass — is cut off, so the draw cannot land on a "random" token.
The cost is a hard cutoff: when the model is confident, top-k may discard tokens
that the model still considers plausible; when it is uncertain (near-uniform), top-k
discards genuinely competing options. This repo does **not** implement top-k in
`inference/generate.py:generate` — it is the classic baseline that top-p
generalizes, and `top_p` is the tunable here.

### 3.5 Top-p (nucleus) sampling

Top-p keeps the *smallest* set of top-ranked tokens whose cumulative probability
mass reaches $p$, then renormalizes. Order probabilities
$p_{(1)} \ge p_{(2)} \ge \cdots \ge p_{(V)}$ and define

$$
S_p = \left\{ (1), \ldots, (m) \right\}, \qquad
m = \min\left\{ m' : \sum_{i=1}^{m'} p_{(i)} \ge p \right\},
\qquad
\tilde{p}_i = \frac{p_i}{\sum_{j \in S_p} p_j} \cdot \mathbb{1}[i \in S_p]
\tag{9}
$$

The nucleus size $m$ adapts to confidence: a peaked distribution needs $m \approx 1$
(a few tokens cover 90% of the mass), a flat one needs $m \approx V$. This is the
property top-k lacks — instead of "always keep 50 tokens," it is "keep however many
tokens are actually plausible."

**Interaction with temperature — order matters.** In
`inference/generate.py:generate` the pipeline is strictly sequential:
**temperature first, then softmax, then top-p on the resulting probabilities**:

```python
probs = F.softmax(next_token_logits / temperature, dim=-1)
sorted_probs, sorted_idx = probs.sort(dim=-1, descending=True)
cumsum = sorted_probs.cumsum(dim=-1)
mask = cumsum - sorted_probs > top_p
sorted_probs[mask] = 0.0
sorted_probs = sorted_probs / sorted_probs.sum(dim=-1, keepdim=True).clamp(min=1e-10)
```

Because temperature reshapes the distribution before the nucleus is selected, the
truncation set depends on $T$: at $T = 0.7$ the distribution is sharper, so the
nucleus is smaller and more aggressive than it would be at $T = 1$. (Some libraries
apply top-p to the raw logits before softmax — a different, temperature-agnostic
set.) The mask implements (9) exactly: token $k$ in sorted order is kept iff the
mass strictly *before* it satisfies $\sum_{i<k} p_{(i)} \le p$, i.e. the kept
prefix is the smallest one with cumulative mass $\ge p$; `top_p = 1.0` never masks
anything, and `clamp(min=1e-10)` guards the renormalization against an empty nucleus
(e.g. degenerate inputs). The final draw is a single categorical sample per batch
row via `torch.multinomial(sorted_probs, 1)`, un-permuted with
`sorted_idx.gather(-1, next_id)`.

### 3.6 Entropy and perplexity

The per-token entropy of the predictive distribution quantifies how much the model
"knows" at that position:

$$
H(p) = -\sum_{i=1}^{V} p_i \log p_i \quad (\text{nats}), \qquad
H(x_1, \ldots, x_T) = \sum_{t=1}^{T} H\!\big(P(\cdot \mid x_{<t})\big)
\tag{10}
$$

The second identity follows from (1): the entropy of the joint factors into the sum
of conditional entropies. Entropy is $0$ for a degenerate (one-hot) distribution
and maximal for uniform. The uniform bound, derived from (10) by symmetry
($p_i = 1/V$):

$$
H_{\text{uniform}} = \log V = \log 128000 \approx 11.76 \text{ nats} \approx 16.97 \text{ bits}
\tag{11}
$$

Perplexity is the exponential of the average per-token cross-entropy, which at eval
equals entropy of the model's own distribution:

$$
\text{PPL} = \exp\!\left(-\frac{1}{T}\sum_{t=1}^{T} \log P(x_t \mid x_{<t})\right)
= \exp\!\left(\frac{1}{T}\sum_{t=1}^{T} H_t\right)
\tag{12}
$$

For the uniform distribution over this vocabulary, $\text{PPL} = V = 128000$ — a
useful sanity floor: a model that beats 128K perplexity has learned *something*.
Intuition: perplexity is "the effective number of equally likely choices the model
feels at each step." High entropy $\Rightarrow$ many plausible continuations
$\Rightarrow$ sampling matters and greedy is brittle; low entropy $\Rightarrow$ the
mode is a safe bet. This is precisely the passkey case: after training, the
distribution at the answer position should be sharply peaked on the true five-digit
string, so greedy is both the right estimator and the low-variance one.

## 4. Code walkthrough — `inference/generate.py:generate`

```python
@torch.no_grad()
def generate(
    model: GPTOSS,
    input_ids: torch.Tensor,
    max_new_tokens: int = 64,
    temperature: float = 0.7,
    top_p: float = 0.9,
    use_cache: bool = True,
) -> torch.Tensor:
```

**Parameter semantics** (`inference/generate.py:generate`):

- `max_new_tokens` — number of decode steps; the return tensor has shape
  `(B, T_prompt + max_new_tokens)`.
- `temperature` — the $T$ of (3). `temperature <= 0` (including exactly `0.0`)
  switches to the greedy branch (5); strictly positive values use the full
  sampling pipeline. `top_p` is **ignored** in the greedy branch.
- `top_p` — the nucleus mass $p$ of (9), applied after temperature/softmax.
- `use_cache` — when `True`, each decode step runs one token through the model with
  `inference/generate.py:MixedKVCache`; when `False`, the full prefix is replayed
  every step (an $O(T^2)$ reference path used by the equivalence test, §6).

**Determinism.** `@torch.no_grad()` disables autograd graph construction, and
`model.eval()` disables dropout, so the only stochasticity left is
`torch.multinomial`, which consumes the global PyTorch RNG: for reproducible
sampling runs, call `torch.manual_seed(seed)` before `generate`. The greedy branch
consumes no RNG and is deterministic for a fixed model and prompt. (Dropout is the
only other RNG consumer in the forward pass, and eval disables it; see
[training.md](../training.md) for the training-time dropout contract.)

**Prefill.** Prompt tokens are embedded and run through all 12 blocks in one pass
per layer, caching rotated K/V per layer via `inference/generate.py:_attn_forward_layer`
into a fresh `inference/generate.py:MixedKVCache` (windowed layers store only the
last 128 tokens, global layers everything — [inference.md](../inference.md)):

```python
x = model.embed(input_ids)
positions = torch.arange(T_prompt, device=dev)
for layer_idx, block in enumerate(model.blocks):
    x = _attn_forward_layer(block, layer_idx, x, positions, cache, sink_bias_cache)
x = model.norm(x)
next_token_logits = model.head(x)[:, -1, :]
```

`models/transformer.py:GPTOSS.head` is the final linear to $V = 128000$ logits,
weight-tied to `models/transformer.py:GPTOSS.embed` (saves ~98M parameters on the
502M budget, [architecture.md](../architecture.md)).

**Decode loop.** One iteration per new token:

```python
for step in range(max_new_tokens):
    if temperature <= 0:
        next_id = next_token_logits.argmax(dim=-1, keepdim=True)
    else:
        probs = F.softmax(next_token_logits / temperature, dim=-1)
        sorted_probs, sorted_idx = probs.sort(dim=-1, descending=True)
        cumsum = sorted_probs.cumsum(dim=-1)
        mask = cumsum - sorted_probs > top_p
        sorted_probs[mask] = 0.0
        sorted_probs = sorted_probs / sorted_probs.sum(dim=-1, keepdim=True).clamp(min=1e-10)
        next_id = torch.multinomial(sorted_probs, 1)
        next_id = sorted_idx.gather(-1, next_id)
    output[:, T_prompt + step : T_prompt + step + 1] = next_id
    cur_pos += 1
```

- Greedy branch: `argmax` over logits — (2)/(5).
- Sampling branch: temperature scaling (3) → softmax → sort descending →
  cumulative mass → mask (9) → renormalize with an epsilon floor → one
  multinomial draw per row → map sorted indices back to vocabulary indices.
- The new token is written into a pre-allocated `output` buffer of shape
  `(B, T_prompt + max_new_tokens)`.

With `use_cache=True`, the next logits come from embedding only the new token at its
**absolute** position `positions_step = torch.tensor([cur_pos - 1])` and running one
decode step per layer (append length-1 K/V to the cache, attend, MoE):

```python
x_step = model.embed(next_id)
positions_step = torch.tensor([cur_pos - 1], device=dev)
for layer_idx, block in enumerate(model.blocks):
    x_step = _attn_forward_layer(block, layer_idx, x_step, positions_step, cache, sink_bias_cache)
x_step = model.norm(x_step)
next_token_logits = model.head(x_step)[:, -1, :]
```

With `use_cache=False` the same logits are recomputed by re-embedding the whole
prefix `output[:, :T_prompt + step + 1]` and re-running every block with `cache=None`
— the two paths agree by construction, and `tests/test_inference.py` asserts they
produce identical output for `max_new_tokens=1`. Absolute positions matter here:
YaRN extrapolates from the trained 4,096 to 131,072 via scale-32 RoPE
([rope_yarn.md](../rope_yarn.md)), and a wrong relative offset would silently
corrupt every attention layer's positional signal at long context.

## 5. The passkey eval path — why `temperature=0.0`

`inference/long_context.py:PasskeyEvaluator.evaluate` loops over context lengths
`(4096, 8192, 32768, 65536, 131072)`, samples 100 distinct five-digit passkeys per
length, builds a prompt via `inference/long_context.py:PasskeyEvaluator.build_prompt`
over deterministic filler (`inference/long_context.py:make_filler_text`), and calls:

```python
output_ids = generate(
    self.model,
    input_ids,
    max_new_tokens=16,
    temperature=0.0,
    top_p=1.0,
    use_cache=True,
)
```

`temperature=0.0` routes to the argmax branch; `top_p=1.0` is inert. The first
five-digit number in the decoded continuation is extracted by
`inference/long_context.py:PasskeyEvaluator.extract_passkey_from_output` and matched
against the ground truth.

**Why greedy is the right eval choice.** Per trial, the model either answers
correctly (a Bernoulli variable with success probability $q$). With $n = 100$ trials
the estimated accuracy has standard deviation

$$
\sigma = \sqrt{\frac{q(1-q)}{n}} \approx 0.036 \quad \text{at } q = 0.85,\ n = 100
\tag{13}
$$

if the draws were i.i.d. Greedy decoding removes the sampling layer from the loop
entirely: for a fixed prompt the completion is a deterministic function of the
weights, so the only variation left across trials is the passkey/filler content —
the estimator measures *the model*, not the sampler's luck. Sampling at the answer
position would occasionally draw a non-mode token (a wrong digit), depressing and
noising the measured accuracy. Since the task is a copy/retrieval task — one correct
answer, sharply peaked distribution — the mode *is* the answer, and greedy is both
the highest-accuracy and lowest-variance choice (§3.6).

**Caveat:** the ≥85% target at 128K is a **target**. No pretraining run has
completed; `scripts/passkey_eval.py` on an untrained checkpoint produces near-chance
accuracy (the passkey is one of 100,000 uniformly sampled values, so chance is
$10^{-5}$ per trial) and exits 0 with a warning, not an error.

## 6. Pitfalls and verification

| Pitfall | Symptom | Guard |
|---------|---------|-------|
| Assuming `top_p` applies under `temperature <= 0` | Greedy ignores `top_p` entirely | Read the branch in `inference/generate.py:generate`; passkey eval sets `top_p=1.0` for exactly this reason |
| Temperature/top-p ordering confusion | Truncation set changes if top-p runs on logits pre-softmax | This repo: temperature → softmax → top-p (code order in §3.5) |
| Non-reproducible sampling runs | Different output per launch with `temperature > 0` | `torch.manual_seed(seed)` before `generate`; greedy (`temperature=0.0`) is seed-independent |
| `top_p=0.0` with `temperature > 0` | Only the single top token survives the mask | Not a crash — `clamp(min=1e-10)` prevents division by zero, but the result is effectively greedy with tie-breaking by `multinomial` |
| Using sampling in passkey eval | Accuracy becomes a noisy random variable (13) | Keep `temperature=0.0` in `inference/long_context.py:PasskeyEvaluator.evaluate` |
| Breaking cache vs no-cache equivalence | Subtle positional drift at long context | `tests/test_inference.py` asserts `use_cache=False` matches `use_cache=True` for one token |
| Reading the ≥85% number as a result | It is a target; untrained checkpoints score ~0% | `scripts/passkey_eval.py` prints the warning line and exits 0 |

**Verify the claims:**

```bash
pytest tests/test_inference.py -v
```

covers: generation output shape (`(1, 4 + 8)` for `max_new_tokens=8`,
`temperature=0.0`), greedy no-crash, the `use_cache` equivalence property, the KV
ring-buffer order invariants, and passkey prompt construction / regex extraction.
The sampling branch itself is exercised through the shape and no-crash tests with
`temperature=0.0`; stochastic-draw behavior is deterministic given a seed and can be
checked by calling `generate` twice with `torch.manual_seed(0)` and comparing
outputs. The full passkey pipeline needs a trained checkpoint:

```bash
python3 scripts/passkey_eval.py --checkpoint path/to/model.safetensors --n-trials 100
```

On an untrained model this runs end-to-end, prints an accuracy table at
~0% per length, and warns "needs trained checkpoint for ≥ 85% target" — expected,
not a bug ([getting_started.md](../getting_started.md) §12).

## Where to go next

| Topic | Document |
|-------|----------|
| Softmax derivation, causal mask, scaled dot product | [attention math](attention_math.md) |
| Decode loop mechanics: mixed KV cache, ring buffer, O(1) append | [kv cache engineering](kv_cache_engineering.md), [inference.md](../inference.md) |
| Cross-entropy objective that shapes the logits | [training.md](../training.md) |
| Position handling (absolute positions, YaRN 4K→128K) | [rope_yarn.md](../rope_yarn.md) |
| Vocabulary size and tokenization | [tokenization_bpe.md](tokenization_bpe.md) |
| BF16 precision of logits/softmax | [numerics](numerics.md) |

<!-- docs:verified 2026-08-04 · 5da1a80 -->
