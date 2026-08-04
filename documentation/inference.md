# Inference — Mixed KV Cache and Long-Context Evaluation

> **Chapter on `inference/generate.py` and `inference/long_context.py`.** Autoregressive
> decode with a heterogeneous per-layer KV cache, sink-bias clamp caching, and the
> passkey retrieval headline metric. Architecture overview:
> [architecture.md §9](architecture.md#9-inference-mixedkvcache-and-generation).
> Sink bias: [ATTENTION_SINKS.md](ATTENTION_SINKS.md). Scripts and benchmarks:
> [operations.md](operations.md).

---

## Table of contents

1. [Why a custom inference path?](#why-a-custom-inference-path)
2. [MixedKVCache](#mixedkvcache)
3. [`generate()`](#generate)
4. [Passkey retrieval (`inference/long_context.py` / `scripts/passkey_eval.py`)](#passkey-retrieval-inferencelong_contextpy--scriptspasskey_evalpy)
5. [KV-cache benchmark meaning](#kv-cache-benchmark-meaning)
6. [Failure modes](#failure-modes)
7. [How to verify](#how-to-verify)

---

## Why a custom inference path?

`GPTOSS.forward` in `models/transformer.py` is optimized for **training**: full-sequence
forward, gradient checkpointing, aux-loss aggregation. Generation needs incremental
decode with KV reuse and heterogeneous per-layer storage:

| Layer index | Attention | KV stored |
|-------------|-----------|-----------|
| 0, 2, 4, 6, 8, 10 | Sliding window `W=128` | Last `min(T, 128)` tokens |
| 1, 3, 5, 7, 9, 11 | Full (global) | All `T` tokens |

`inference/generate.py` calls block submodules (`norm1`, `attn`, `moe`, `norm2`) through
`_attn_forward_layer` without modifying the training forward. Keys are **RoPE-rotated
before caching**; stored tensors are already position-rotated. GQA: 8 query heads, 4 KV
heads (`models/attention.py:repeat_kv` expands at attention matmul time).

---

## MixedKVCache

`MixedKVCache` in `inference/generate.py` maintains two parallel storage systems indexed
by `layer_idx`:

| Storage | Layers | Semantics |
|---------|--------|-----------|
| `windowed_kv` | `is_windowed=True` | Fixed-size ring, capacity `window` |
| `global_kv` | `is_windowed=False` | Dynamic array, amortized O(1) append |

`inference/generate.py:MixedKVCache.reset` clears all state between independent requests.
`inference/generate.py:MixedKVCache.append` accepts **already-rotated** K and V;
`inference/generate.py:MixedKVCache.get` returns chronologically ordered tensors for attention.
`inference/generate.py:MixedKVCache.seq_len` reports the effective cached length of one
layer — the ring buffer's `count` field (capped at `window`) for windowed layers, the
tracked `global_lengths[layer_idx]` for global layers, and `0` before the first append;
`tests/test_inference.py:test_kv_cache_seq_len_helper` pins both behaviors.

### Windowed layers — ring buffer (capacity = window_size)

Each windowed layer stores `[buf_k, buf_v, head, count]`:

| Field | Meaning |
|-------|---------|
| `buf_k`, `buf_v` | `(B, H_kv, window, head_dim)` tensors |
| `head` | Next write index in the ring (`0 ≤ head < window`) |
| `count` | Effective history length (`≤ window`) |

**Prompt prefill** (`T_new` tokens at once):

- If `T_new >= window`: copy only the **last `window`** K/V; set `head=0`, `count=window`
- If `T_new < window`: fill from the start; `count = T_new`

**Decode** (one token per step):

1. Write K/V slice at `head`
2. Advance `head = (head + 1) % window`
3. `count = min(window, count + 1)`

**Read ordering:** when `head != 0`, `get()` unrotates the ring:

```python
k_ordered = torch.cat([k[:, :, head:, :], k[:, :, :head, :]], dim=2)
```

Attention always sees the last `count` keys in temporal order. Storage is **O(window)**
regardless of total sequence length.

### Global layers — exponential growth

Global layers keep the **entire prefix** for full attention:

1. **First write:** allocate `(B, H_kv, new_cap, D)` with `new_cap = max(needed, 1)`
2. **Growth:** when `cur_len + T_new > cur_cap`, reallocate with
   `new_cap = max(needed, int(cur_cap * 1.5) + 1)`
3. **Cap:** `new_cap = min(new_cap, _GLOBAL_CAP_TOKENS)` (default **4,000,000** tokens)

`global_lengths[layer_idx]` tracks valid prefix length; `global_caps[layer_idx]` tracks
allocated capacity (may exceed length). The 1.5× growth factor gives amortized O(1)
append — standard dynamic-array strategy with a gentler constant than doubling. 128K
context is well under the 4M safety cap.

### Why decode is O(1) per step vs O(T) naive cache rebuild

Two distinct costs matter: **cache maintenance** and **attention compute**.

**Cache maintenance (what `MixedKVCache` optimizes):**

| Strategy | Per decode step | At T = 131072 |
|----------|-----------------|---------------|
| Naive rebuild | Allocate/copy entire `(B, H, T, D)` K and V tensors | O(T) memory copy per layer per step |
| `MixedKVCache` windowed | In-place write at `head`, wrap index | O(1) — fixed `W=128` buffer |
| `MixedKVCache` global | Append into pre-allocated slack or rare 1.5× realloc | O(1) amortized append |

A naive implementation that concatenates `torch.cat([old_k, k_new], dim=2)` every step
forces an O(T) copy of all prior keys and values **per layer per token**. At 128K that
is ~131K × 12 layers × 1536 B/token ≈ 2.4 GB copied per generated token — unusable.

`MixedKVCache` instead:

- **Windowed:** overwrites one ring slot; no reallocation, no prefix copy
- **Global:** writes into existing capacity; reallocation copies the prefix only when
  capacity is exhausted (amortized over ~1.5× growth intervals)

**Attention compute (unchanged by cache design):**

- Windowed layers: O(W) per step (W = 128)
- Global layers: O(T) per step — 6 global layers dominate at long context

The headline **~2× KV memory reduction** at 128K comes from storing only `W` tokens on
half the layers. Analytical proof: `scripts/kv_cache_benchmark.py` (see
[operations.md](operations.md)).

`use_cache=False` in `generate()` deliberately replays the full prefix each step
(O(T²) total) as a correctness reference — production long-context eval always sets
`use_cache=True`.

---

## `generate()`

```python
@torch.no_grad()
def generate(model, input_ids, max_new_tokens=64,
             temperature=0.7, top_p=0.9, use_cache=True) -> Tensor
```

Entry point `inference/generate.py:generate`; `use_cache=False` replays the full
prefix each step as a correctness reference (see §3).

Returns `(B, T_prompt + max_new_tokens)` token ids. `model.eval()` at entry; `model.to(dev)`
enforces the model↔input device contract (no-op when already aligned).

### Parameter semantics

Four knobs control the decode loop; only two change the distribution, and only when
`temperature > 0`:

| Parameter | Default | Effect |
|-----------|---------|--------|
| `max_new_tokens` | 64 | Number of decode steps; the return tensor has shape `(B, T_prompt + max_new_tokens)` |
| `temperature` | 0.7 | Boltzmann scale $T$ of (1); `temperature <= 0` switches to the greedy argmax branch and makes `top_p` inert |
| `top_p` | 0.9 | Nucleus mass $p$ of (2), applied after temperature scaling and softmax |
| `use_cache` | True | `True`: per-step decode against `inference/generate.py:MixedKVCache`; `False`: replay the whole prefix every step (the $O(T^2)$ reference path below) |

**Boltzmann scaling.** Sampling draws the next token from the softmax of logits
scaled by $1/T$ — the Boltzmann distribution with temperature in the exponent, over
the vocabulary $\mathcal{V}$ of size $V = 128000$:

$$
p_i = \frac{\exp(z_i / T)}{\sum_{j=1}^{V} \exp(z_j / T)}, \qquad T > 0
\tag{1}
$$

$T$ interpolates between the two extremes of the family: as $T \to 0^+$ the
probability ratio $\exp((z_i - z_j)/T)$ of any two unequal logits diverges and the
mass concentrates on the argmax (uniform over ties) — which is exactly the
`next_token_logits.argmax(dim=-1)` branch `inference/generate.py:generate` takes
for `temperature <= 0`; as $T \to \infty$ every exponent tends to 0 and
$p_i \to 1/V$, the uniform distribution. The default `temperature=0.7 < 1`
sharpens the distribution toward the mode without committing to it.

**Top-p (nucleus) truncation.** Order the probabilities
$p_{(1)} \ge p_{(2)} \ge \cdots \ge p_{(V)}$ and keep the smallest prefix whose
cumulative mass reaches $p$, zeroing the rest and renormalizing:

$$
m = \min\left\{ m' : \sum_{i=1}^{m'} p_{(i)} \ge p \right\}, \qquad
\tilde{p}_i = \frac{p_i}{\sum_{j=1}^{m} p_{(j)}} \cdot \mathbb{1}\!\left[i \in S_p\right]
\tag{2}
$$

The nucleus size adapts to confidence: a peaked distribution needs $m \approx 1$
token, a flat one all $V$. Ordering matters — temperature is applied **before**
the nucleus is selected, so a sharper (lower-$T$) distribution yields a smaller
truncation set. The full derivation, both limits, and the entropy/perplexity
framing live in [sampling theory](theory/sampling.md) §3; this chapter keeps only
the semantics. The only RNG consumer in the loop is the final
`torch.multinomial(sorted_probs, 1)` draw, so `torch.manual_seed(seed)` before
`generate` makes a sampling run reproducible; the greedy branch consumes no RNG,
and dropout — the only other stochasticity in the forward — is disabled by
`model.eval()`.

### Prefill vs decode

**Phase A — Prompt prefill** processes all prompt tokens in one parallel pass per layer:

```python
cache = MixedKVCache() if use_cache else None
sink_bias_cache = {}
x = model.embed(input_ids)
positions = torch.arange(T_prompt, device=dev)
for layer_idx, block in enumerate(model.blocks):
    x = _attn_forward_layer(block, layer_idx, x, positions, cache, sink_bias_cache)
x = model.norm(x)
next_token_logits = model.head(x)[:, -1, :]
```

KV cache is populated for both windowed and global layers. Per-layer work is
$O(T_{\text{prompt}}^2 D)$ for the attention matmuls and $O(T_{\text{prompt}} D^2)$
for the projections/MoE — attention dominates at long context (derivation below).

**Phase B — Token-by-token decode** runs `max_new_tokens` steps:

1. Sample `next_id` from `next_token_logits` (greedy argmax when `temperature <= 0`)
2. Append to pre-allocated `output` buffer
3. Embed single token; `positions_step = tensor([cur_pos - 1])` — **absolute** index of
   the new token (not relative offset within the decode window)
4. Run all layers via `_attn_forward_layer` (append length-1 K/V to cache)
5. `next_token_logits = head(norm(x_step))[:, -1, :]`

Each decode step touches only the new token's activations plus cached K/V — no full-prefix
re-forward when `use_cache=True`.

### Prefill vs decode — FLOP asymmetry and bandwidth

Let $H = 8$ be the query-head count, $H_{\text{kv}} = 4$ the KV-head count (expanded
at attention time by `models/attention.py:repeat_kv`), $D = 96$ the head dimension,
and $L = 12$ the layer count. One attention over $T$ keys costs two matmuls per head
— $QK^\top$ and the probability-vector-times-$V$, each $2TD$ FLOPs. Masking never
reduces these counts: `models/attention.py:causal_attention` computes a dense score
matrix and zeroes disallowed entries — `models/attention.py:_window_mask` binding at
prefill changes correctness, not FLOPs.

**Prefill.** All $T$ prompt tokens are processed in one parallel pass. Per layer the
score and output matmuls are $T \times D$ against $D \times T$, so per layer

$$
F_{\text{prefill}}(T) = 4 H T^2 D
\tag{3}
$$

$O(T^2 D)$ per layer — quadratic in prompt length — but amortized over $T$ tokens in
one pass, and the $L$ layers together pay $O(T^2 D L)$ exactly once per generation.

**Decode.** One new token attends against the $T$ cached keys. The query is
$1 \times D$ against $D \times T$ keys, so per step per layer

$$
F_{\text{decode}}(T) = 4 H T D
\tag{4}
$$

a factor $1/T$ of the prefill pass, but paid serially once per generated token:
generating $N$ tokens at cache length $T$ costs $\sum_{t=1}^{N} 4 H D (T + t)
\approx 4 H D (T N + N^2/2)$ — quadratic in the *generated* length, dominated by
the late steps. Windowed layers dodge the $T$ on the key side:
`inference/generate.py:MixedKVCache.get` returns at most `window = 128` cached
tokens, so their decode attention is $4 H W D$ — a $T/W = 1024\times$ reduction at
$T = 131072$ on six of the twelve layers. That is the asymmetry the mixed cache
exploits: prefill pays $O(T^2)$ on windowed and global layers alike, but decode
attention on half the layers is constant in $T$.

**Why decode is memory-bound.** A decode step performs ~0.3 GFLOP at short context
(≈ 2.7 GFLOP at 128K, [kv cache engineering](theory/kv_cache_engineering.md) §4.3)
but must stream the stored keys and values from HBM. Per token per layer the cache
holds $K$ and $V$, each an $H_{\text{kv}} \times D$ tensor of $s$-byte elements
($s = 2$ for BF16):

$$
b = 2 \cdot H_{\text{kv}} \cdot D \cdot s = 2 \times 4 \times 96 \times 2 = 1536 \text{ bytes per token per layer}
\tag{5}
$$

At step $T$ the attention matmuls move $T b$ bytes of KV per layer against
$4 H T D$ FLOPs, so the arithmetic intensity — FLOPs per byte — collapses to the
head-count ratio (the byte factor $s$ folds into the denominator):

$$
\text{AI}_{\text{attn}} = \frac{4 H T D}{T \cdot 2 \cdot H_{\text{kv}} D s} = \frac{H}{H_{\text{kv}}} = \frac{8}{4} = 2 \text{ FLOP/byte}
\tag{6}
$$

The A100 80GB ridge point — where compute time equals memory time — is
$\Pi/\beta \approx 312\text{ TFLOP/s} / 2.04\text{ TB/s} \approx 153$
FLOP/byte `[INFERENCE]` (published specs; `.benchmarks/` is empty and no A100 run
has happened). At 2 FLOP/byte the attention kernels sit ~75× under the ridge, so
per-token decode latency is set by bandwidth: $M_{\text{step}} / \beta$ for
$M_{\text{step}}$ bytes streamed per step, and token throughput
$\approx \beta / M_{\text{step}}$. The full derivation, including the whole-model
intensity and the ~1,200 tokens/s `[INFERENCE]` ceiling, is in
[kv cache engineering](theory/kv_cache_engineering.md) §4.3.

### Sink clamp cache

Training clamps sink bias every forward:

```python
sink_bias_clamped = attn.sink_bias.clamp(SINK_CLAMP_MIN, SINK_CLAMP_MAX)
# SINK_CLAMP_MIN = -10.0, SINK_CLAMP_MAX = 15.0
```

During multi-step `generate()`, recomputing the clamp on every layer every step is wasted
work. `generate()` passes a `sink_bias_cache: dict` keyed by `id(attn)`:

```python
if id(attn) in sink_bias_cache:
    sink_bias_clamped = sink_bias_cache[id(attn)]
else:
    sink_bias_clamped = attn.sink_bias.clamp(SINK_CLAMP_MIN, SINK_CLAMP_MAX)
    sink_bias_cache[id(attn)] = sink_bias_clamped
```

Properties:

- **Unclamped parameter** still trains normally — clamp is forward-only
- Cache is per `generate()` call, not global — reflects current parameter values
- Identical numerics to training forward for a given `sink_bias` parameter

See [ATTENTION_SINKS.md §6](ATTENTION_SINKS.md#6-bf16-clamp-rationale) for why clamp
prevents BF16 mask-add overflow.

### YaRN T=1 fast path

During decode, `positions_step` has shape `(1,)` — a single absolute position. In
`YaRNRoPE.forward` (`models/yarn.py`), `positions.numel() == 1` triggers a scalar fast
path that avoids `torch.outer(positions, inv_freq)` and its `(1, d/2)` temporaries:

```python
if positions.numel() == 1:
    inv_freq = self.inv_freq.to(positions.device)
    pos = positions.item() if positions.dim() == 0 else positions[0].item()
    freqs = inv_freq * float(pos)
    cos = freqs.cos().unsqueeze(0) * self.mscale
    sin = freqs.sin().unsqueeze(0) * self.mscale
```

Prefill uses the full-sequence `torch.outer` path (`T_prompt` positions). Pruned RoPE on
global layers (`n_pruned_dims`) zeroes the first 25% of sin/cos dims on both paths.
YaRN extrapolation parameters (`yarn_scale_factor=32`, `yarn_target_seq_len=131072`) are
documented in [rope_yarn.md §5](rope_yarn.md#5-production-parameters-θ100k-scale32-target131072) and
[architecture.md](architecture.md).

---

## Passkey retrieval (`inference/long_context.py` / `scripts/passkey_eval.py`)

`inference/long_context.py:PasskeyEvaluator` implements the needle-in-a-haystack protocol (Mohtashami & Jaggi, 2023).

### Prompt construction

1. Generate deterministic filler text (`inference/long_context.py:make_filler_text`) of roughly `context_length` words
2. Insert `"The passkey is {passkey}."` at `start`, `middle`, or `end` via `inference/long_context.py:PasskeyEvaluator.build_prompt`
3. Append question template asking the model to recall the 5-digit passkey

### Scoring

1. Tokenize prompt
2. `generate(..., max_new_tokens=16, temperature=0.0, use_cache=True)`
3. Decode only **new** tokens after the prompt
4. Extract first 5-digit number via regex `r"\b(\d{5})\b"` in `inference/long_context.py:PasskeyEvaluator.extract_passkey_from_output`
5. Match against ground-truth passkey

Default context lengths: `4096, 8192, 32768, 65536, 131072` with `n_trials=100` distinct
random passkeys per length. Returns `{ctx_len: accuracy}` via `inference/long_context.py:PasskeyEvaluator.evaluate`.

### Why greedy is deterministic — the variance argument

The eval runs `temperature=0.0` (the full argument is
[sampling theory](theory/sampling.md) §5) so that each trial's outcome measures the
model, not the sampler. With greedy decode the completion is a deterministic function
of the weights and the prompt on a given device: the argmax branch never reaches
`torch.multinomial` (the only RNG consumer), and dropout — the only other
stochasticity — is disabled by `model.eval()`, so a fixed checkpoint and prompt
produce bit-identical output on every run. Per trial the outcome is therefore a
Bernoulli variable $X_i \in \{0, 1\}$ with success probability $q$ — the model's
true retrieval accuracy at that context length — and the reported estimate is the
sample mean over $n$ independent trials:

$$
\hat{q} = \frac{1}{n} \sum_{i=1}^{n} X_i, \qquad
\operatorname{Var}(\hat{q}) = \frac{q(1-q)}{n}, \qquad
\sigma(\hat{q}) = \sqrt{\frac{q(1-q)}{n}} \approx 0.036 \ \ (q = 0.85,\ n = 100)
\tag{7}
$$

At the headline target $q = 0.85$ with `n_trials=100`, the 95% interval is
$\pm 1.96\,\sigma \approx \pm 7.1$ percentage points — the sampling noise floor of
the estimator itself. Sampling with `temperature > 0` would stack a second noise
source on top: at the answer position a non-mode draw occasionally emits a wrong
digit, depressing the measured accuracy and inflating its variance. Because the
passkey task has exactly one correct answer, the mode *is* the answer, so greedy is
simultaneously the highest-accuracy and the lowest-variance choice.

**Trial independence.** `inference/long_context.py:PasskeyEvaluator.evaluate` seeds
a fresh `random.Random(base_seed + ctx_len)` per context length (`base_seed=42`),
samples `n_trials` distinct 5-digit passkeys without replacement from the 100,000
possible values, and fixes the filler text per length
(`inference/long_context.py:make_filler_text` seeds on `context_length`), so the
$n$ trials per length vary only in the passkey — the quantity the model must
retrieve. The distinct passkeys make the trials genuinely different experiments; the
Bernoulli model (7) is the i.i.d. approximation behind the ±7.1-point interval.

### Stub behavior on untrained checkpoints

`scripts/passkey_eval.py` uses `_CharTokenizer` — `ord(char) mod vocab_size` — for CPU
portability without loading the LLaMA-3 tokenizer. This is a **harness convenience**,
not the training distribution.

On an **untrained** checkpoint:

- Random logits produce garbage completions; accuracy is **near zero** at all context lengths
- The script still runs end-to-end and prints the accuracy table
- Exit code is 0 even when below target — the final line reports
  `⚠️ Accuracy … — needs trained checkpoint for ≥ 85% target`

For production eval, swap in the real LLaMA-3 tokenizer to match training. E2E GPU smoke
(`scripts/e2e_gpu_smoke.py`) exercises `MixedKVCache` generation on a tiny model without
passkey scoring.

### Target ≥85% at 128K after training

Headline metric (trained checkpoint with YaRN extrapolation from 4K → 128K):

| Position range | Target |
|----------------|--------|
| 0 – 32K | ≥ 95% |
| 32K – 96K | ≥ 90% |
| 96K – 128K | ≥ 85% |

`passkey_eval.py` checks the **maximum** `--context-lengths` entry (default 131072) and
prints `✅ HEADLINE METRIC PASSED` when accuracy ≥ 85%.

CLI:

```bash
python3 scripts/passkey_eval.py \
    --checkpoint path/to/model.safetensors \
    --n-trials 100 \
    --context-lengths 4096 8192 32768 65536 131072 \
    --position middle \
    --seed 42
```

Loads `ModelConfig` from `configs/pretrain_a100_502m.yaml`, builds `GPTOSS`, loads
safetensors weights (`strict=False`).

---

## KV-cache benchmark meaning

`scripts/kv_cache_benchmark.py` is an **analytical** script — no GPU, no model load. It
computes KV bytes from architecture constants (12 layers, 6 windowed + 6 global, GQA 4 KV
heads, `head_dim=96`, `W=128`, BF16).

At T = 131072, batch = 1: **~1.13 GB** SWA/full mix vs **~2.25 GB** pure full attention
(**2.00×** reduction). Pass threshold: **≥ 1.8×** at 128K. See [operations.md](operations.md)
for the full command reference.

`MixedKVCache` adds ring metadata and capacity slack — real memory is slightly higher but
same asymptotic scaling. The benchmark validates the **architectural claim**, not runtime
allocator behavior.

---

## Failure modes

| Symptom | Likely cause | Fix |
|---------|--------------|-----|
| Garbage long-context output | Untrained model or wrong tokenizer | Train; use LLaMA-3 tokenizer |
| OOM at 128K prefill | Batch > 1 or insufficient VRAM | `batch=1`; BF16; shorter eval |
| Slower than expected decode | `use_cache=False` | Enable cache |
| Position bugs / repeated tokens | Wrong `positions` during decode | Use absolute index `cur_pos-1` |
| Global cap hit | Sequence > 4M tokens | Raise `global_cap_tokens` |
| Sink overflow / NaN logits | Missing clamp | Ensure `SINK_CLAMP_*` path used |

---

## How to verify

Unit tests cover `MixedKVCache` ring/global semantics, `generate()` shape and cache
equivalence, and passkey prompt construction:

```bash
python3 -m pytest tests/test_inference.py -v
```

Passkey CLI loads a checkpoint and prints the accuracy table:

```bash
python3 scripts/passkey_eval.py --help
```

Analytical KV headline metric (CPU, no checkpoint):

```bash
python3 scripts/kv_cache_benchmark.py
```

After any change to `inference/generate.py`, re-run `tests/test_inference.py`. After
attention mask changes, also run `pytest tests/test_attention.py -v` (sliding-window
oracle). Operational runbook: [operations.md](operations.md).

---

<!-- docs:verified 2026-08-04 · 5da1a80 -->
