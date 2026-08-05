# GPT-OSS-Lite — Inference

> **Chapter on `inference/generate.py` and `inference/long_context.py`.** Autoregressive decode with a heterogeneous per-layer KV cache, sink-bias clamp caching, and the passkey retrieval headline metric. Architecture overview: [architecture.md §9](concepts/foundations-and-architecture.md#9-inference-mixedkvcache-and-generation). Sink bias: [attention-sinks.md](concepts/attention-sinks.md). Scripts and benchmarks: [operations.md](guides/operations.md).

---

---

## Why a custom inference path?

`GPTOSS.forward` in `models/transformer.py` is optimized for **training**: full-sequence forward, gradient checkpointing, aux-loss aggregation. Generation needs incremental decode with KV reuse and heterogeneous per-layer storage:

| Layer index | Attention | KV stored |
|-------------|-----------|-----------|
| 0, 2, 4, 6, 8, 10 | Sliding window `W=128` | Last `min(T, 128)` tokens |
| 1, 3, 5, 7, 9, 11 | Full (global) | All `T` tokens |

`inference/generate.py` calls block submodules (`norm1`, `attn`, `moe`, `norm2`) through `_attn_forward_layer` without modifying the training forward. Keys are **RoPE-rotated before caching**; stored tensors are already position-rotated. GQA: 8 query heads, 4 KV heads (`models/attention.py:repeat_kv` expands at attention matmul time).

---

## MixedKVCache

`MixedKVCache` in `inference/generate.py` maintains two parallel storage systems indexed by `layer_idx`:

| Storage | Layers | Semantics |
|---------|--------|-----------|
| `windowed_kv` | `is_windowed=True` | Fixed-size ring, capacity `window` |
| `global_kv` | `is_windowed=False` | Dynamic array, amortized O(1) append |

`inference/generate.py:MixedKVCache.reset` clears all state between independent requests. `inference/generate.py:MixedKVCache.append` accepts **already-rotated** K and V; `inference/generate.py:MixedKVCache.get` returns chronologically ordered tensors for attention. `inference/generate.py:MixedKVCache.seq_len` reports the effective cached length of one layer — the ring buffer's `count` field (capped at `window`) for windowed layers, the tracked `global_lengths[layer_idx]` for global layers, and `0` before the first append; `tests/test_inference.py:test_kv_cache_seq_len_helper` pins both behaviors.

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

Attention always sees the last `count` keys in temporal order. Storage is **O(window)** regardless of total sequence length.

### Global layers — exponential growth

Global layers keep the **entire prefix** for full attention:

1. **First write:** allocate `(B, H_kv, new_cap, D)` with `new_cap = max(needed, 1)`
2. **Growth:** when `cur_len + T_new > cur_cap`, reallocate with
   `new_cap = max(needed, int(cur_cap * 1.5) + 1)`
3. **Cap:** `new_cap = min(new_cap, _GLOBAL_CAP_TOKENS)` (default **4,000,000** tokens)

`global_lengths[layer_idx]` tracks valid prefix length; `global_caps[layer_idx]` tracks allocated capacity (may exceed length). The 1.5× growth factor gives amortized O(1) append — standard dynamic-array strategy with a gentler constant than doubling. 128K context is well under the 4M safety cap.

### Why decode is O(1) per step vs O(T) naive cache rebuild

Two distinct costs matter: **cache maintenance** and **attention compute**.

**Cache maintenance (what `MixedKVCache` optimizes):**

| Strategy | Per decode step | At T = 131072 |
|----------|-----------------|---------------|
| Naive rebuild | Allocate/copy entire `(B, H, T, D)` K and V tensors | O(T) memory copy per layer per step |
| `MixedKVCache` windowed | In-place write at `head`, wrap index | O(1) — fixed `W=128` buffer |
| `MixedKVCache` global | Append into pre-allocated slack or rare 1.5× realloc | O(1) amortized append |

A naive implementation that concatenates `torch.cat([old_k, k_new], dim=2)` every step forces an O(T) copy of all prior keys and values **per layer per token**. At 128K that is ~131K × 12 layers × 1536 B/token ≈ 2.4 GB copied per generated token — unusable.

`MixedKVCache` instead:

- **Windowed:** overwrites one ring slot; no reallocation, no prefix copy
- **Global:** writes into existing capacity; reallocation copies the prefix only when
  capacity is exhausted (amortized over ~1.5× growth intervals)

**Attention compute (unchanged by cache design):**

- Windowed layers: O(W) per step (W = 128)
- Global layers: O(T) per step — 6 global layers dominate at long context

The headline **~2× KV memory reduction** at 128K comes from storing only `W` tokens on half the layers. Analytical proof: `scripts/kv_cache_benchmark.py` (see
[operations.md](guides/operations.md)).

`use_cache=False` in `generate()` deliberately replays the full prefix each step (O(T²) total) as a correctness reference — production long-context eval always sets `use_cache=True`.

---

## `generate()`

```python
@torch.no_grad()
def generate(model, input_ids, max_new_tokens=64,
             temperature=0.7, top_p=0.9, use_cache=True) -> Tensor
```

Entry point `inference/generate.py:generate`; `use_cache=False` replays the full prefix each step as a correctness reference (see §3).

Returns `(B, T_prompt + max_new_tokens)` token ids. `model.eval()` at entry; `model.to(dev)` enforces the model↔input device contract (no-op when already aligned).

### Parameter semantics

Four knobs control the decode loop; only two change the distribution, and only when `temperature > 0`:

| Parameter | Default | Effect |
|-----------|---------|--------|
| `max_new_tokens` | 64 | Number of decode steps; the return tensor has shape `(B, T_prompt + max_new_tokens)` |
| `temperature` | 0.7 | Boltzmann scale $T$ of (1); `temperature <= 0` switches to the greedy argmax branch and makes `top_p` inert |
| `top_p` | 0.9 | Nucleus mass $p$ of (2), applied after temperature scaling and softmax |
| `use_cache` | True | `True`: per-step decode against `inference/generate.py:MixedKVCache`; `False`: replay the whole prefix every step (the $O(T^2)$ reference path below) |

**Boltzmann scaling.** Sampling draws the next token from the softmax of logits scaled by $1/T$ — the Boltzmann distribution with temperature in the exponent, over the vocabulary $\mathcal{V}$ of size $V = 128000$:

$$
p_i = \frac{\exp(z_i / T)}{\sum_{j=1}^{V} \exp(z_j / T)}, \qquad T > 0
\tag{1}
$$

$T$ interpolates between the two extremes of the family: as $T \to 0^+$ the probability ratio $\exp((z_i - z_j)/T)$ of any two unequal logits diverges and the mass concentrates on the argmax (uniform over ties) — which is exactly the `next_token_logits.argmax(dim=-1)` branch `inference/generate.py:generate` takes for `temperature <= 0`; as $T \to \infty$ every exponent tends to 0 and $p_i \to 1/V$, the uniform distribution. The default `temperature=0.7 < 1` sharpens the distribution toward the mode without committing to it.

**Top-p (nucleus) truncation.** Order the probabilities $p_{(1)} \ge p_{(2)} \ge \cdots \ge p_{(V)}$ and keep the smallest prefix whose cumulative mass reaches $p$, zeroing the rest and renormalizing:

$$
m = \min\left\{ m' : \sum_{i=1}^{m'} p_{(i)} \ge p \right\}, \qquad
\tilde{p}_i = \frac{p_i}{\sum_{j=1}^{m} p_{(j)}} \cdot \mathbb{1}\!\left[i \in S_p\right]
\tag{2}
$$

The nucleus size adapts to confidence: a peaked distribution needs $m \approx 1$ token, a flat one all $V$. Ordering matters — temperature is applied **before** the nucleus is selected, so a sharper (lower-$T$) distribution yields a smaller truncation set. The full derivation, both limits, and the entropy/perplexity framing live in [sampling theory](concepts/optimizers-and-numerics.md) §3; this chapter keeps only the semantics. The only RNG consumer in the loop is the final `torch.multinomial(sorted_probs, 1)` draw, so `torch.manual_seed(seed)` before `generate` makes a sampling run reproducible; the greedy branch consumes no RNG, and dropout — the only other stochasticity in the forward — is disabled by `model.eval()`.

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

KV cache is populated for both windowed and global layers. Per-layer work is $O(T_{\text{prompt}}^2 D)$ for the attention matmuls and $O(T_{\text{prompt}} D^2)$ for the projections/MoE — attention dominates at long context (derivation below).

**Phase B — Token-by-token decode** runs `max_new_tokens` steps:

1. Sample `next_id` from `next_token_logits` (greedy argmax when `temperature <= 0`)
2. Append to pre-allocated `output` buffer
3. Embed single token; `positions_step = tensor([cur_pos - 1])` — **absolute** index of
   the new token (not relative offset within the decode window)
4. Run all layers via `_attn_forward_layer` (append length-1 K/V to cache)
5. `next_token_logits = head(norm(x_step))[:, -1, :]`

Each decode step touches only the new token's activations plus cached K/V — no full-prefix re-forward when `use_cache=True`.

### Prefill vs decode — FLOP asymmetry and bandwidth

Let $H = 8$ be the query-head count, $H_{\text{kv}} = 4$ the KV-head count (expanded at attention time by `models/attention.py:repeat_kv`), $D = 96$ the head dimension, and $L = 12$ the layer count. One attention over $T$ keys costs two matmuls per head — $QK^\top$ and the probability-vector-times-$V$, each $2TD$ FLOPs. Masking never reduces these counts: `models/attention.py:causal_attention` computes a dense score matrix and zeroes disallowed entries — `models/attention.py:_window_mask` binding at prefill changes correctness, not FLOPs.

**Prefill.** All $T$ prompt tokens are processed in one parallel pass. Per layer the score and output matmuls are $T \times D$ against $D \times T$, so per layer

$$
F_{\text{prefill}}(T) = 4 H T^2 D
\tag{3}
$$

$O(T^2 D)$ per layer — quadratic in prompt length — but amortized over $T$ tokens in one pass, and the $L$ layers together pay $O(T^2 D L)$ exactly once per generation.

**Decode.** One new token attends against the $T$ cached keys. The query is $1 \times D$ against $D \times T$ keys, so per step per layer

$$
F_{\text{decode}}(T) = 4 H T D
\tag{4}
$$

a factor $1/T$ of the prefill pass, but paid serially once per generated token: generating $N$ tokens at cache length $T$ costs $\sum_{t=1}^{N} 4 H D (T + t) \approx 4 H D (T N + N^2/2)$ — quadratic in the *generated* length, dominated by the late steps. Windowed layers dodge the $T$ on the key side: `inference/generate.py:MixedKVCache.get` returns at most `window = 128` cached tokens, so their decode attention is $4 H W D$ — a $T/W = 1024\times$ reduction at $T = 131072$ on six of the twelve layers. That is the asymmetry the mixed cache exploits: prefill pays $O(T^2)$ on windowed and global layers alike, but decode attention on half the layers is constant in $T$.

**Why decode is memory-bound.** A decode step performs ~0.3 GFLOP at short context (≈ 2.7 GFLOP at 128K, [kv cache engineering](inference.md) §4.3) but must stream the stored keys and values from HBM. Per token per layer the cache holds $K$ and $V$, each an $H_{\text{kv}} \times D$ tensor of $s$-byte elements ($s = 2$ for BF16):

$$
b = 2 \cdot H_{\text{kv}} \cdot D \cdot s = 2 \times 4 \times 96 \times 2 = 1536 \text{ bytes per token per layer}
\tag{5}
$$

At step $T$ the attention matmuls move $T b$ bytes of KV per layer against $4 H T D$ FLOPs, so the arithmetic intensity — FLOPs per byte — collapses to the head-count ratio (the byte factor $s$ folds into the denominator):

$$
\text{AI}_{\text{attn}} = \frac{4 H T D}{T \cdot 2 \cdot H_{\text{kv}} D s} = \frac{H}{H_{\text{kv}}} = \frac{8}{4} = 2 \text{ FLOP/byte}
\tag{6}
$$

The A100 80GB ridge point — where compute time equals memory time — is $\Pi/\beta \approx 312\text{ TFLOP/s} / 2.04\text{ TB/s} \approx 153$ FLOP/byte `[INFERENCE]` (published specs; `.benchmarks/` is empty and no A100 run has happened). At 2 FLOP/byte the attention kernels sit ~75× under the ridge, so per-token decode latency is set by bandwidth: $M_{\text{step}} / \beta$ for $M_{\text{step}}$ bytes streamed per step, and token throughput $\approx \beta / M_{\text{step}}$. The full derivation, including the whole-model intensity and the ~1,200 tokens/s `[INFERENCE]` ceiling, is in
[kv cache engineering](inference.md) §4.3.

### Sink clamp cache

Training clamps sink bias every forward:

```python
sink_bias_clamped = attn.sink_bias.clamp(SINK_CLAMP_MIN, SINK_CLAMP_MAX)
# SINK_CLAMP_MIN = -10.0, SINK_CLAMP_MAX = 15.0
```

During multi-step `generate()`, recomputing the clamp on every layer every step is wasted work. `generate()` passes a `sink_bias_cache: dict` keyed by `id(attn)`:

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

See [attention-sinks.md §6](concepts/attention-sinks.md#6-bf16-clamp-rationale) for why clamp prevents BF16 mask-add overflow.

### YaRN T=1 fast path

During decode, `positions_step` has shape `(1,)` — a single absolute position. In `YaRNRoPE.forward` (`models/yarn.py`), `positions.numel() == 1` triggers a scalar fast path that avoids `torch.outer(positions, inv_freq)` and its `(1, d/2)` temporaries:

```python
if positions.numel() == 1:
    inv_freq = self.inv_freq.to(positions.device)
    pos = positions.item() if positions.dim() == 0 else positions[0].item()
    freqs = inv_freq * float(pos)
    cos = freqs.cos().unsqueeze(0) * self.mscale
    sin = freqs.sin().unsqueeze(0) * self.mscale
```

Prefill uses the full-sequence `torch.outer` path (`T_prompt` positions). Pruned RoPE on global layers (`n_pruned_dims`) zeroes the first 25% of sin/cos dims on both paths. YaRN extrapolation parameters (`yarn_scale_factor=32`, `yarn_target_seq_len=131072`) are documented in [attention-and-positional.md §5](concepts/attention-and-positional.md#5-production-parameters-θ100k-scale32-target131072) and
[foundations-and-architecture.md](concepts/foundations-and-architecture.md).

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

Default context lengths: `4096, 8192, 32768, 65536, 131072` with `n_trials=100` distinct random passkeys per length. Returns `{ctx_len: accuracy}` via `inference/long_context.py:PasskeyEvaluator.evaluate`.

### Why greedy is deterministic — the variance argument

The eval runs `temperature=0.0` (the full argument is
[sampling theory](concepts/optimizers-and-numerics.md) §5) so that each trial's outcome measures the
model, not the sampler. With greedy decode the completion is a deterministic function of the weights and the prompt on a given device: the argmax branch never reaches `torch.multinomial` (the only RNG consumer), and dropout — the only other stochasticity — is disabled by `model.eval()`, so a fixed checkpoint and prompt produce bit-identical output on every run. Per trial the outcome is therefore a Bernoulli variable $X_i \in \{0, 1\}$ with success probability $q$ — the model's true retrieval accuracy at that context length — and the reported estimate is the sample mean over $n$ independent trials:

$$
\hat{q} = \frac{1}{n} \sum_{i=1}^{n} X_i, \qquad
\operatorname{Var}(\hat{q}) = \frac{q(1-q)}{n}, \qquad
\sigma(\hat{q}) = \sqrt{\frac{q(1-q)}{n}} \approx 0.036 \ \ (q = 0.85,\ n = 100)
\tag{7}
$$

At the headline target $q = 0.85$ with `n_trials=100`, the 95% interval is $\pm 1.96\,\sigma \approx \pm 7.1$ percentage points — the sampling noise floor of the estimator itself. Sampling with `temperature > 0` would stack a second noise source on top: at the answer position a non-mode draw occasionally emits a wrong digit, depressing the measured accuracy and inflating its variance. Because the passkey task has exactly one correct answer, the mode *is* the answer, so greedy is simultaneously the highest-accuracy and the lowest-variance choice.

**Trial independence.** `inference/long_context.py:PasskeyEvaluator.evaluate` seeds a fresh `random.Random(base_seed + ctx_len)` per context length (`base_seed=42`), samples `n_trials` distinct 5-digit passkeys without replacement from the 100,000 possible values, and fixes the filler text per length (`inference/long_context.py:make_filler_text` seeds on `context_length`), so the $n$ trials per length vary only in the passkey — the quantity the model must retrieve. The distinct passkeys make the trials genuinely different experiments; the Bernoulli model (7) is the i.i.d. approximation behind the ±7.1-point interval.

### Stub behavior on untrained checkpoints

`scripts/passkey_eval.py` uses `_CharTokenizer` — `ord(char) mod vocab_size` — for CPU portability without loading the LLaMA-3 tokenizer. This is a **harness convenience**, not the training distribution.

On an **untrained** checkpoint:

- Random logits produce garbage completions; accuracy is **near zero** at all context lengths
- The script still runs end-to-end and prints the accuracy table
- Exit code is 0 even when below target — the final line reports
  `⚠️ Accuracy … — needs trained checkpoint for ≥ 85% target`

For production eval, swap in the real LLaMA-3 tokenizer to match training. E2E GPU smoke (`scripts/e2e_gpu_smoke.py`) exercises `MixedKVCache` generation on a tiny model without passkey scoring.

### Target ≥85% at 128K after training

Headline metric (trained checkpoint with YaRN extrapolation from 4K → 128K):

| Position range | Target |
|----------------|--------|
| 0 – 32K | ≥ 95% |
| 32K – 96K | ≥ 90% |
| 96K – 128K | ≥ 85% |

`passkey_eval.py` checks the **maximum** `--context-lengths` entry (default 131072) and prints `✅ HEADLINE METRIC PASSED` when accuracy ≥ 85%.

CLI:

```bash
python3 scripts/passkey_eval.py \
    --checkpoint path/to/model.safetensors \
    --n-trials 100 \
    --context-lengths 4096 8192 32768 65536 131072 \
    --position middle \
    --seed 42
```

Loads `ModelConfig` from `configs/pretrain_a100_502m.yaml`, builds `GPTOSS`, loads safetensors weights (`strict=False`).

---

## KV-cache benchmark meaning

`scripts/kv_cache_benchmark.py` is an **analytical** script — no GPU, no model load. It computes KV bytes from architecture constants (12 layers, 6 windowed + 6 global, GQA 4 KV heads, `head_dim=96`, `W=128`, BF16).

At T = 131072, batch = 1: **~1.13 GB** SWA/full mix vs **~2.25 GB** pure full attention (**2.00×** reduction). Pass threshold: **≥ 1.8×** at 128K. See [operations.md](guides/operations.md) for the full command reference.

`MixedKVCache` adds ring metadata and capacity slack — real memory is slightly higher but same asymptotic scaling. The benchmark validates the **architectural claim**, not runtime allocator behavior.

---

## Failure modes

| Symptom | Likely cause | Fix |
|---------|--------------|-----|
| Garbage long-context output | Untrained model or wrong tokenizer | Train; use LLaMA-3 tokenizer |
| OOM at 128K prefill | Batch > 1 or insufficient VRAM | `batch=1`; BF16; shorter eval |
| Slower than expected decode | `use_cache=False` | Enable cache |
| Position bugs / repeated tokens | Wrong `positions` during decode | Use absolute index `cur_pos-1` |
| Global cap hit | Sequence > 4M tokens | Use a longer-context model or reduce sequence length |
| Sink overflow / NaN logits | Missing clamp | Ensure `SINK_CLAMP_*` path used |

---

## How to verify

Unit tests cover `MixedKVCache` ring/global semantics, `generate()` shape and cache equivalence, and passkey prompt construction:

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

After any change to `inference/generate.py`, re-run `tests/test_inference.py`. After attention mask changes, also run `pytest tests/test_attention.py -v` (sliding-window oracle). Operational runbook: [operations.md](guides/operations.md).

---

## KV-Cache Engineering — Reuse, Bandwidth, and the Mixed Cache

> **Theory chapter.** Why a KV cache exists, how big it must be, why decode is bandwidth-bound, and how GPT-OSS-Lite's `MixedKVCache` combines GQA head sharing, six ring-buffered windowed layers, and six exponentially grown global layers into the measured 2.00× KV reduction at 128K. Companion to the practical walkthrough in [inference](inference.md); the worked 128K byte arithmetic lives in [ATTENTION_SINKS §8](concepts/attention-sinks.md). Assumes the primer level of [foundations](concepts/foundations-and-architecture.md) §2–§4. Mask and softmax machinery: [attention math](concepts/attention-and-positional.md). YaRN parameters: [rope_yarn](concepts/attention-and-positional.md).

---

---

### 1. 60-second summary

A KV cache stores the key and value tensors of every processed token so that later decode steps can reuse them instead of recomputing the whole history. The cache exists because of a reuse identity: in causal autoregressive inference, a token's key and value never change after they are first computed, so each token's KV pair can be computed exactly once and appended to a buffer.

The size math is simple: per token per layer, a KV cache stores two tensors of `H_kv × head_dim` elements, which is 1536 bytes here (4 KV heads × 96 dims × 2 bytes × 2 tensors). Twelve full layers at 131,072 tokens is about 2.25 GB.

Decode is memory-bound, not compute-bound: every step re-reads the KV history plus the model weights, and the ratio of FLOPs to bytes moved sits far below the machine's ridge point. Reducing bytes is therefore the only lever that moves token throughput, and GPT-OSS-Lite pulls it twice: GQA halves KV bytes by using 4 KV heads for 8 query heads, and the mixed cache halves them again at long context by giving six of the twelve layers a 128-token sliding-window ring buffer instead of unbounded growth. The combined measured reduction at 128K is 2.00× (`scripts/kv_cache_benchmark.py`).

The implementation is `inference/generate.py:MixedKVCache`, consumed by `inference/generate.py:generate` through `inference/generate.py:_attn_forward_layer`.

---

### 2. Why it matters here

- **The architecture is built around the cache.** Twelve layers alternate
  sliding-window (even layers, `W = 128`) and full attention (odd layers), with `inference/generate.py:MixedKVCache` mirroring that split exactly: windowed layers get a fixed ring buffer, global layers get a growing array. The alternation is what produces the 2.00× headline; replacing it with pure full attention breaks the ≥ 1.8× target (see [ATTENTION_SINKS §7.4](concepts/attention-sinks.md)).
- **GQA sets the byte floor.** 8 query heads share 4 KV heads
  (`head_dim = 96`), so every cached token costs half what MHA would cost.
- **The eval target is 128K from a 4K training context.** YaRN extrapolation
  (`scale = 32`, `target_seq_len = 131072`; [rope_yarn](concepts/attention-and-positional.md)) makes long prompts *possible*; the cache makes them *affordable*. The 4M global cap is two orders of magnitude above any planned eval length.
- **Sink bias makes eviction safe.** Windowed layers drop tokens, and the
  learned per-head sink (`models/attention.py:GPTOSSAttention.sink_bias`) absorbs the attention mass that would otherwise scatter — the cache design and the sink design are one mechanism. See
  [ATTENTION_SINKS §3](concepts/attention-sinks.md).
- **Budget discipline.** 501.8M total / 247.0M active parameters means decode
  streams ~494 MB of active weights per token; at 128K the KV history adds ~1.2 GB. Both numbers matter for the bandwidth argument in §4.3.

---

### 3. Intuition

**The sticky-note wall.** During decode, each token pastes its key and value onto a wall as soon as it is computed. Later tokens never re-derive what is on the wall; they only read it and paste their own note. Without the wall, every step would have to recompute every previous token's key and value from scratch — work that grows quadratically in the number of generated tokens.

**The carousel.** A windowed layer's cache is a carousel with `W = 128` seats. New tokens sit down at the current front-of-line pointer; when the carousel is full, the oldest rider is simply overwritten. Reading it back in chronological order requires walking the loop from the oldest occupied seat to the newest — a two-slice concatenation.

**The growing shelf.** A global layer's cache is a shelf that is occasionally replaced by a bigger one. Every time the shelf fills, you buy a shelf 1.5× larger and move the books over. The moves are expensive but rare, and the geometric series of move costs converges, so each book is moved a constant number of times on average.

**The bottleneck is the pantry conveyor, not the chef.** A decode step is a tiny amount of arithmetic (hundreds of MFLOP) but a huge amount of data movement (hundreds of MB to GB). The GPU's compute units finish long before the memory system finishes delivering bytes — so per-token latency is set by bandwidth, and every byte saved is a byte of latency saved.

---

### 4. Theory and derivation

### 4.1 The reuse identity — why a cache exists

Let $x_i^{(l)}$ be the input to layer $l$ at position $i$, and let $W_K^{(l)}, W_V^{(l)}$ be that layer's key and value projections. Then

$$
K_i^{(l)} = W_K^{(l)} x_i^{(l)}, \qquad V_i^{(l)} = W_V^{(l)} x_i^{(l)} \tag{1}
$$

where $K_i^{(l)}, V_i^{(l)} \in \mathbb{R}^{H_{\text{kv}} \times D}$ with $H_{\text{kv}} = 4$ KV heads and $D = 96$ head dimensions here. The input $x_i^{(l)}$ is a function of tokens $1 \dots i$ only: causal masking means position $i$ never receives information from positions $> i$, and within a fixed-weight inference pass the forward computation at position $i$ is deterministic.

Now consider two consecutive decode steps, $t$ and $t+1$. Step $t+1$ computes the hidden state of the new token $t+1$, but it does not recompute the hidden states of tokens $1 \dots t$ — autoregressive generation is strictly sequential, and those states already exist. Hence $x_i^{(l)}$, and therefore $K_i^{(l)}, V_i^{(l)}$, are bit-identical between the two steps:

$$
K_i^{(l)}(t+1) = K_i^{(l)}(t), \qquad V_i^{(l)}(t+1) = V_i^{(l)}(t),
\quad \forall i \le t \tag{2}
$$

The **reuse identity** follows: the key/value set the next query needs is the previous set plus exactly one new pair,

$$
\left\{K_1^{(l)}, \dots, K_{t+1}^{(l)}\right\} =
\left\{K_1^{(l)}, \dots, K_t^{(l)}\right\} \cup \left\{K_{t+1}^{(l)}\right\} \tag{3}
$$

and likewise for $V$. A cache is precisely the data structure that makes (3) operational: compute each pair once at its own step, store it, append the new pair per step. (2) is why this is *correct* and not merely an optimization.

Without the cache, step $t+1$ re-projects all $t+1$ tokens. Let $c = 2 \cdot d_{\text{model}} \cdot (2 H_{\text{kv}} D)$ be the FLOPs of one token's KV projection (the single `kv_proj` matmul; here $2 \cdot 768 \cdot 768 \approx 1.18$ MFLOP). A $T$-token generation then costs $c \sum_{t=1}^{T} t = c\, T(T+1)/2$ KV-projection FLOPs; the cached version costs $c\, T$. The ratio

$$
\frac{c\, T(T+1)/2}{c\, T} = \frac{T+1}{2} \tag{4}
$$

is the recompute factor: at $T = 131072$ it is $\approx 65537$×. The quadratic blowup of (4) is the fundamental reason every transformer inference stack, including this one, caches.

### 4.2 Cache size — bytes per token per layer

One cached token holds $K$ and $V$, each an $H_{\text{kv}} \times D$ tensor of $s$-byte elements ($s = 2$ for BF16). Per token per layer:

$$
b = 2 \cdot H_{\text{kv}} \cdot D \cdot s \tag{5}
$$

With the production numbers ($H_{\text{kv}} = 4$, $D = 96$, $s = 2$):

$$
b = 2 \times 4 \times 96 \times 2 = 1536 \text{ bytes per token per layer} \tag{6}
$$

For $L = 12$ layers at sequence length $T$, batch 1, all layers full:

$$
B_{\text{full}}(T) = L \cdot T \cdot b = 12 \cdot T \cdot 1536 \tag{7}
$$

At $T = 131072$ that is $12 \times 131072 \times 1536 = 2415919104$ bytes ≈ 2.25 GiB — larger than every model weight tensor in the network. The KV cache, not the weights, is the dominant memory consumer at long context.

### 4.3 Decode is memory-bound — arithmetic intensity

A decode step feeds one query against $T$ cached keys/values per layer. The two attention matmuls cost, per layer, per head group:

$$
F_{\text{attn}} = 2 \cdot H \cdot T \cdot D \;(\text{QK}^\top) + 2 \cdot H \cdot T \cdot D \;(\text{PV})
= 4 \cdot H \cdot T \cdot D \tag{8}
$$

(one query × $T$ keys, and the $1 \times T$ probability vector × $T$ values; $H = 8$ query heads after GQA expansion). The bytes moved to feed those matmuls are the stored K and V:

$$
M_{\text{attn}} = 2 \cdot H_{\text{kv}} \cdot T \cdot D \cdot s = 4 \cdot H_{\text{kv}} \cdot T \cdot D \tag{9}
$$

with $s = 2$ folded in. The arithmetic intensity — FLOPs per byte — is their ratio, and the byte factor cancels:

$$
\text{AI}_{\text{attn}} = \frac{4 \cdot H \cdot T \cdot D}{4 \cdot H_{\text{kv}} \cdot T \cdot D}
= \frac{H}{H_{\text{kv}}} = \frac{8}{4} = 2 \text{ FLOP/byte} \tag{10}
$$

This is the *ideal* broadcast figure: it assumes each stored KV element is fetched once and reused by all $H/H_{\text{kv}} = 2$ query-head groups that share it. The current implementation does not achieve that — see §5.5 — which only lowers the intensity further.

The projections are no better. Per token per layer, `q_proj`, `kv_proj`, `o_proj` each multiply $d_{\text{model}} \times d_{\text{model}}$ ($768^2$): $F_{\text{proj}} = 3 \cdot 2 \cdot 768^2 \approx 3.54$ MFLOP against $3 \cdot 768^2 \cdot 2 = 3.54$ MB of weight bytes streamed from HBM (weights do not fit in on-chip cache at 494 MB total), so $\text{AI}_{\text{proj}} \approx 1$ FLOP/byte. The MoE block is identical per active expert ($3 \cdot 2 \cdot 768 \cdot 1536 \approx 7.08$ MFLOP against $3 \cdot 768 \cdot 1536 \cdot 2 \approx 7.08$ MB).

The ridge point — where compute time equals memory time — is

$$
R = \frac{\Pi}{\beta} \tag{11}
$$

with $\Pi$ peak FLOPs and $\beta$ HBM bandwidth. On the A100 80GB target (published specs: $\Pi \approx 312$ TFLOP/s BF16 dense, $\beta \approx 2.04$ TB/s), $R \approx 153$ FLOP/byte. Every intensity above sits 75–150× below the ridge, so decode kernels are bandwidth-saturated: per-token latency $\approx M_{\text{step}} / \beta$ where $M_{\text{step}}$ is bytes moved per step, and token throughput $\approx \beta / M_{\text{step}}$.

Whole-model numbers, derived from the 247.0M active parameters and the layer count: per-token compute ≈ 12 × (3.54 + 21.2) ≈ 297 MFLOP; active weights ≈ 247,032,672 × 2 ≈ 494 MB. Even at zero KV history the intensity is 297e6/494e6 ≈ 0.6 FLOP/byte. At $T = 131072$ the six global layers add $6 \cdot 1536 \cdot 131072 \approx 1.21$ GB of KV reads per step; the step then moves ≈ 1.7 GB and performs ≈ 2.7 GFLOP (the attention matmuls now dominate compute), for ≈ 1.6 FLOP/byte — still ~100× under the ridge. At 2.04 TB/s that implies ≈ 0.85 ms/step, i.e. roughly 1,200 tokens/s single-stream `[INFERENCE]` — plausible upper bound, not a measurement; `.benchmarks/` is empty and no A100 run has happened.

### 4.4 GQA — the 2× bandwidth reduction

Multi-head attention with $H = 8$ heads stores $2 \cdot H \cdot D \cdot s$ bytes per token per layer; GQA with $H_{\text{kv}} = 4$ stores $2 \cdot H_{\text{kv}} \cdot D \cdot s$. The ratio is the head-count ratio:

$$
\frac{B_{\text{MHA}}}{B_{\text{GQA}}} = \frac{2 \cdot H \cdot D \cdot s \cdot L \cdot T}
{2 \cdot H_{\text{kv}} \cdot D \cdot s \cdot L \cdot T} = \frac{H}{H_{\text{kv}}} = 2 \tag{12}
$$

Because the cache holds only $H_{\text{kv}}$ copies and the query count $H$ is unchanged, GQA simultaneously halves stored bytes (memory), halves per-step KV read bytes (bandwidth, the binding constraint from §4.3), and doubles $\text{AI}_{\text{attn}}$ from 1 to 2 FLOP/byte. This is a pure win: the query side still has 8 distinct heads, so representational capacity is not reduced — only the duplicated storage is. The mechanism at attention time is `models/attention.py:repeat_kv`, which expands the 4-head cached tensors to 8 heads per call.

### 4.5 Ring buffers — O(1) append vs O(T) concat

The naive incremental cache appends with `torch.cat([old, new], dim=2)`, which allocates a new tensor and copies the entire history. At step $t$ that copies $t$ stored tokens; over $T$ steps, per layer:

$$
\sum_{t=1}^{T} t = \frac{T(T+1)}{2} \approx \frac{T^2}{2} \tag{13}
$$

token-copies. At $T = 131072$: $131072^2/2 \approx 8.6 \times 10^9$ copies × 1536 B ≈ 13 TB of copying per layer, ~158 TB across 12 layers, for a single generation. This is not a performance question; it is a disqualifier.

A ring buffer of $W$ slots replaces the history copy with an in-place overwrite: the append writes at the `head` pointer and advances $\text{head} \leftarrow (\text{head} + 1) \bmod W$, costing $O(1)$ regardless of $T$ — $O(T)$ total over the whole generation. The price is that reading back in chronological order may cross the wrap boundary: when the buffer is full and $\text{head} \neq 0$, the oldest token sits at index `head`, so the ordered tensor is the concatenation of two slices, `[head:]` followed by `[:head]`. That materialization copies $W$ tokens (1536·$W$ bytes) per call. It runs on most steps (all but the one where `head == 0`), so its amortized cost is $O(W)$ per step — versus $O(T)$ for the naive copy. At $T = 131072$ that is a factor $W/T = 128/131072 \approx 0.1\%$, and it is also ~0.1% of one global layer's KV read, so the wrap read is a constant fraction of the step's traffic, not a scaling term.

### 4.6 Exponential growth — amortized O(1) append

Global layers cannot ring-buffer: full attention needs every token. The dynamic array grows by factor $g = 1.5$: when `needed > cur_cap`, allocate `new_cap = max(needed, int(cur_cap * 1.5) + 1)`, copy the old prefix, continue. Let $C_k$ be the capacity after the $k$-th growth; $C_k \approx C_0 g^k$. The total elements copied over all growths is the geometric series

$$
\sum_{k=0}^{m-1} C_k = C_0 \frac{g^m - 1}{g - 1} \approx \frac{C_m}{g - 1}
\tag{14}
$$

With final capacity $C_m \approx n$, the number of appended elements, the amortized copy work is $1/(g-1)$ element-copies per append: 2 for $g = 1.5$, 1 for doubling. Either is $O(1)$ amortized — appends are constant-time on average because the growing gaps between reallocations absorb the copies. The 1.5× policy deliberately trades the extra amortized copy for tighter memory: right after a growth the buffer is at least $2/3$ full, so the worst-case slack is

$$
\frac{C - n}{C} \le 1 - \frac{1}{g} = \frac{1}{3} \quad (g = 1.5), \qquad
\frac{1}{2} \quad (g = 2) \tag{15}
$$

A $1/3$ waste ceiling on a multi-GB structure is worth one extra amortized copy per token.

The cap `inference/generate.py:MixedKVCache._GLOBAL_CAP_TOKENS = 4000000` bounds the worst case: a full-capacity global layer holds $4 \times 10^6 \times 1536 \approx 6.1$ GiB. Appending past the cap does not truncate silently — `new_cap` is clamped below `needed`, the growth copy's destination slice is shorter than the source chunk, and `copy_` raises a shape `RuntimeError` (verified 2026-08-04). The cap turns an unbounded-allocation bug into a loud error.

### 4.7 The mixed cache — 6·min(W, T) + 6·T and the 2.00× headline

The two policies above are assigned by layer parity. Six windowed layers store $\min(W, T)$ tokens each; six global layers store $T$:

$$
B_{\text{mixed}}(T) = \left(6 \cdot \min(W, T) + 6 \cdot T\right) \cdot 1536 \tag{16}
$$

against the all-full baseline (7). For $T \ge W$ the reduction ratio is

$$
\frac{B_{\text{full}}}{B_{\text{mixed}}} =
\frac{12 \cdot T}{6 \cdot W + 6 \cdot T} =
\frac{2T}{T + W} =
\frac{2T}{T + 128} \tag{17}
$$

which rises monotonically toward the asymptote 2 as $T \to \infty$: 1.94× at 4K, 1.97× at 8K, 1.99× at 32K, 2.00× at 128K — measured by `scripts/kv_cache_benchmark.py` (2.25 GB vs 1.13 GB at 131072). The full byte-level arithmetic and the GB table are in
[ATTENTION_SINKS §8](concepts/attention-sinks.md); (16)–(17) are the condensed
derivation. Note the boundary condition: at $T \le W$ every layer stores $T$, so the ratio is 1 — the mixed cache only pays off once sequences outgrow the window, which is exactly the regime the 128K eval target cares about. The per-step bandwidth story mirrors the memory story: windowed layers read $1536 \cdot 128 \approx 197$ KB per step regardless of $T$, global layers read $1536 \cdot T$, so at 128K the six windowed layers contribute ~0.1% of KV traffic.

### 4.8 Rotated-K caching — apply once at append

RoPE is a block-diagonal rotation: each dimension pair $(2i, 2i+1)$ of a vector at position $m$ is rotated by angle $m \theta_i$, with $\theta_i$ the $i$-th frequency. Collecting pairs into $R_m$ (block-diagonal, orthogonal), a query and key at positions $m, n$ score as

$$
(R_m q_m)^\top (R_n k_n) = q_m^\top R_m^\top R_n k_n = q_m^\top R_{n-m} k_n \tag{18}
$$

since the blocks satisfy $R_m^\top R_n = R_{n-m}$ — scores depend only on the relative offset $n - m$, which is the property that lets a 4K-trained model generalize to 128K positions under YaRN. Equation (18) shows the rotation can be applied to either operand; the cache's design choice is to store $k'_n = R_n k_n$, the already-rotated key.

Why rotate at append rather than at attention time? The alternative — store raw $k_n$ and rotate on every attention call — would require knowing each key's absolute position (so a positions array must be cached too) and would re-rotate the entire history each step: $2 \cdot H_{\text{kv}} \cdot T \cdot D$ FLOPs plus a fresh $(B, H_{\text{kv}}, T, D)$ materialization per layer per step, recreating exactly the $1536 \cdot T$ bytes/step of traffic the cache exists to avoid. Rotating at append time costs each key exactly one rotation, performed at the one moment its position is known — during prefill from the `arange` position tensor, during decode from the single absolute position `cur_pos - 1` passed by `inference/generate.py:generate`. The stored tensor bakes in the YaRN table (`θ = 100K`, `scale = 32`, `mscale`) and the pruned dims: on global layers the first 24 of 48 frequency pairs are identity (`cos = 1`, `sin = 0` in `models/yarn.py:YaRNRoPE.forward`), so those dims are stored unrotated and the decode query, rotated with the same table, matches. Because stored keys are never re-rotated as positions advance, decode positions must be absolute — a relative offset would apply the wrong rotation (§6).

---

### 5. Code walkthrough

### 5.1 Storage layout

`inference/generate.py:MixedKVCache.__init__` builds four parallel per-layer structures:

- `windowed_kv` — one entry per windowed layer; each entry is
  `[buf_k, buf_v, head, count]` with `buf_k`/`buf_v` of shape `(B, H_kv, window, D)`.
- `global_kv` — one `[buf_k, buf_v]` per global layer, capacity ≥ length.
- `global_lengths` — valid prefix length per global layer (`≤` capacity).
- `global_caps` — allocated capacity per global layer.

The class constant `_GLOBAL_CAP_TOKENS = 4_000_000` is the §4.6 cap. `inference/generate.py:MixedKVCache.__len__` returns `max(len(windowed_kv), len(global_kv))` — the number of layers that have written anything.

### 5.2 `append`

`inference/generate.py:MixedKVCache.append` takes `(layer_idx, k_rot, v, is_windowed, window)` — it accepts the **already-rotated** K (contract: the caller rotated it, §4.8) and the raw V, both shaped `(B, H_kv, T_new, D)`.

**Windowed branch** — first write allocates the ring:

```python
buf_k = torch.zeros(B, H, window, D, dtype=k_rot.dtype, device=k_rot.device)
if T_new >= window:
    buf_k[:, :, :window, :] = k_rot[:, :, -window:, :]
    head = 0
    count = window
else:
    buf_k[:, :, :T_new, :] = k_rot
    head = T_new % window
    count = T_new
```

The `T_new >= window` path is the long-prompt prefill case: a 131072-token prompt keeps only its final 128 tokens on this layer. Subsequent appends write at `head`; when `head + T_new > window` the chunk is split into two slices that wrap around the ring, then `head = (head + T_new) % window` and `count = min(window, count + T_new)`. Every path is an in-place `copy_` into the fixed buffer — no allocation, no history copy, the $O(1)$ append of §4.5.

**Global branch** — first write allocates `max(needed, 1)`; otherwise:

```python
if needed > cur_cap:
    new_cap = max(needed, int(cur_cap * 1.5) + 1)
    new_cap = min(new_cap, self._global_cap_tokens)
    buf_k = torch.empty(B, H, new_cap, D, ...)
    buf_k[:, :, :cur_len, :].copy_(old_k[:, :, :cur_len, :])
    buf_k[:, :, cur_len:needed, :].copy_(k_rot)
    ...
else:
    old_k[:, :, cur_len:needed, :].copy_(k_rot)
```

The 1.5× growth of §4.6, capped at 4M tokens, with `global_lengths` advanced to `needed`. When capacity suffices, the append is a pure in-place write.

### 5.3 `get`

`inference/generate.py:MixedKVCache.get` takes `(layer_idx, is_windowed)` and returns `(K, V)` in **chronological order**, or `(None, None)` for an empty layer.

Windowed: after the empty guards, three cases. `count < k.size(2)` means the ring has never wrapped — tokens occupy slots `[0, count)`, sliced directly. `head == 0` with a full buffer means slot 0 is the oldest, in order, returned as-is. Otherwise the wrap read of §4.5:

```python
k_ordered = torch.cat([k[:, :, head:, :], k[:, :, :head, :]], dim=2)
v_ordered = torch.cat([v[:, :, head:, :], v[:, :, :head, :]], dim=2)
```

Global: `k[:, :, :cur_len, :]` — a view onto the valid prefix; capacity slack beyond `cur_len` is invisible to attention.

### 5.4 `reset` and `seq_len`

`inference/generate.py:MixedKVCache.reset` empties all four lists — required between independent generations, and the reason `generate` builds a fresh cache per call rather than reusing one. `inference/generate.py:MixedKVCache.seq_len` reports the effective history: the ring's `count` for windowed layers (capped at `window`), `global_lengths[layer_idx]` for global layers, and 0 for layers never written. The distinction matters: a windowed layer's *buffer* is always `window` slots, but its *valid* length is `count`; attention needs the latter.

### 5.5 `_attn_forward_layer` — the cache meets attention

`inference/generate.py:_attn_forward_layer` runs one block without touching `models/transformer.py:GPTOSS.forward` (training stays untouched). The relevant sequence:

```python
q = attn.q_proj(x_norm).view(B, T, attn.n_heads, attn.head_dim).transpose(1, 2)
kv = attn.kv_proj(x_norm).view(B, T, 2, attn.n_kv_heads, attn.head_dim)
k_new, v_new = kv[:, :, 0].transpose(1, 2), kv[:, :, 1].transpose(1, 2)
cos, sin = attn.yarn(positions, n_pruned_dims=attn._n_pruned_dims())
q = apply_rope(q, cos, sin)
k_new_rot = apply_rope(k_new, cos, sin)
if cache is not None:
    cache.append(layer_idx, k_new_rot, v_new, attn.is_windowed, attn.window_size)
    k_for_q, v_for_q = cache.get(layer_idx, attn.is_windowed)
else:
    k_for_q, v_for_q = k_new_rot, v_new
k_for_q = repeat_kv(k_for_q, attn.n_rep)
v_for_q = repeat_kv(v_for_q, attn.n_rep)
```

The `kv_proj` produces both K and V from one matmul (`2 * n_kv_heads * head_dim` outputs), so the per-token projection cost of §4.3 is a single 768×768 matmul. The rotation happens exactly once, at append time (`k_new_rot` is what gets stored) — never re-applied to history. When `cache is None` (training-style or `use_cache=False`), the just-computed chunk serves as the full history, which is correct because there is no earlier history to prepend.

KV is stored at $H_{\text{kv}} = 4$ heads; `models/attention.py:repeat_kv` expands to 8 heads only at attention time:

```python
x = x[:, :, None, :, :].expand(B, H_kv, n_rep, T, D)
return x.reshape(B, H_kv * n_rep, T, D)
```

with `n_rep = n_heads // n_kv_heads = 2` from `models/attention.py:GPTOSSAttention`. Semantically this is the GQA broadcast of §4.4 — the cache never stores the duplicated heads. Mechanically, the `reshape` after a stride-0 `expand` materializes a new tensor in current PyTorch (verified: `y.untyped_storage().data_ptr() != x.untyped_storage().data_ptr()` for `n_rep = 2`), so the ideal broadcast bandwidth of (10) is not achieved at the matmul: the attention call pays an extra copy proportional to $(H - H_{\text{kv}}) \cdot T \cdot D$ per tensor per layer per step. This is an implementation cost of the attention call, not of the cache — all headline numbers (§4.7) are measured on *stored* bytes — and it only strengthens the memory-bound conclusion of §4.3. A fused kernel that broadcasts K/V heads in-register would recover the factor-2 intensity.

`models/attention.py:causal_attention` then runs SDPA with the windowed/full mask and the clamped sink bias (`sink_bias_cache` in `generate` memoizes the clamp per layer across steps). Two correctness properties matter here, reflecting the fixed behavior of 2026-08-04: the sliding-window mask binds at prefill (`models/attention.py:_window_mask` square branch is `(idx_q - idx_k < window) & _causal_mask`, so windowed layers attend only to the last `window` keys in both phases), and the sink path masks with additive `torch.where(causal, 0.0, float("-inf"))` rather than a bool→float cast, so future tokens stay blocked.

### 5.6 `generate` — prefill vs decode, `use_cache` on/off

`inference/generate.py:generate(model, input_ids, max_new_tokens, temperature, top_p, use_cache)` has two phases.

**Prefill** — one parallel pass over the whole prompt per layer:

```python
cache = MixedKVCache() if use_cache else None
x = model.embed(input_ids)
positions = torch.arange(T_prompt, device=dev)
for layer_idx, block in enumerate(model.blocks):
    x = _attn_forward_layer(block, layer_idx, x, positions, cache, sink_bias_cache)
```

Windowed layers ingest the full prompt in one `append` (`T_new = T_prompt`): long prompts take the `T_new >= window` path and keep the last 128 tokens.

**Decode** — `max_new_tokens` steps of: sample (argmax for `temperature <= 0`, else top-p softmax + multinomial), append to the pre-allocated `output` buffer, embed the single new token, and run all layers with

```python
positions_step = torch.tensor([cur_pos - 1], device=dev)
```

The absolute position of the new token — the §4.8 contract. Each layer's `append` writes one `(B, H_kv, 1, D)` slice, `get` returns the full history, and attention runs. Total per step: one forward pass over 12 layers on one token, reading the cached history — never re-forwarding the prefix.

`use_cache=False` is the correctness reference: every step re-embeds the full prefix `output[:, :T_prompt + step + 1]`, re-runs all layers over the whole history with `cache=None` (positions = `arange(full_len)`), and uses the last token's logits. This is the $O(T^2)$ recompute of §4.1, and it is deliberately slow; the regression test `test_generate_use_cache_false_matches_cache_for_one_token` asserts the two paths agree token-for-token on the first decoded token.

---

### 6. Pitfalls and verification

**Capacity slack is real memory.** `get` exposes only `cur_len`, but the underlying buffers are pre-allocated: ring buffers always hold `window` slots, global layers hold up to $1.5\times$ their length. The measured 1.13 GB at 128K (§4.7) is the *analytical* floor on stored tokens; resident memory is a bit higher, same asymptotics. `scripts/kv_cache_benchmark.py` is honest about this ("analytical, BF16, batch=1").

**The 4M cap is a loud cliff, not a truncation.** Past `_GLOBAL_CAP_TOKENS` tokens on one global layer, `new_cap` is clamped below `needed` and the growth copy raises a shape `RuntimeError` (verified 2026-08-04). Sequences must fit under the cap (128K eval uses ~3% of it), and `reset()` must run between independent generations — the tests exercise both contracts: `test_kv_cache_reset` and the cap's absence in normal flow.

**Ring order is the classic bug.** The `get` unrotate is easy to get wrong (write order is not read order once the ring wraps). Guarded by `test_kv_cache_windowed_preserves_order_after_rollover`, which appends 7 tokens to a `window=3` ring and asserts the returned K is `[5, 6, 7]`, and by `test_kv_cache_append_windowed`, which asserts `seq_len` caps at `window`.

**Absolute positions at decode.** `positions_step = [cur_pos - 1]` — never a relative offset — because stored keys keep their original rotations (§4.8). A relative position would rotate the query wrong and silently corrupt scores. This is a listed failure mode in [inference.md](inference.md) §6.

**Mask correctness at the window edge.** Two regression tests pin the fixed behavior: `test_sliding_window_blocks_past_keys_at_prefill` (the square `_window_mask` branch, `i - j < window`) and `test_sink_path_matches_manual_at_prefill` (the additive `torch.where` sink mask matches `models/attention.py:manual_causal_attention`). The decode branch (`T_q = 1`) derives its window from `idx_q = T_k - 1`, so the last `window` keys are visible in both phases.

**Verification commands.**

```bash
python3 scripts/kv_cache_benchmark.py     # 2.00× at 128K, 1.94× at 4K — measured 2026-08-04
python3 -m pytest tests/test_inference.py -v   # ring/global semantics, generate, cache equivalence
python3 -m pytest tests/ -q               # full suite: 190 passed / 2 GPU-gated skips
```

All byte and FLOP figures in §4 are derived from architecture constants and reproduce the benchmark's output; the A100 throughput figures in §4.3 are `[INFERENCE]` — derived from published hardware specs, not measured on this repo (`.benchmarks/` is empty). The ≥ 85% passkey target at 128K is a target, not a result: no pretraining run has completed.

---

## References

- [`inference/generate.py:MixedKVCache`](../inference/generate.py) — ring + growing cache
- [`inference/generate.py:generate`](../inference/generate.py) — autoregressive decode
- [`inference/long_context.py:PasskeyEvaluator`](../inference/long_context.py) — 128K passkey eval
- [`utils/memory.py:estimate_model_memory_gb`](../utils/memory.py) — VRAM estimates
- [attention-and-positional.md](concepts/attention-and-positional.md) — attention math, RoPE/YaRN
- [attention-sinks.md](concepts/attention-sinks.md) — sink bias, clamp rationale
- [foundations-and-architecture.md](concepts/foundations-and-architecture.md) — architecture, KV math
- [operations.md](guides/operations.md) — benchmark commands, OPT catalog

<!-- docs:verified 2026-08-05 · 6491066 -->
