# Inference — GPT-OSS-Lite

> **Source:** `inference/generate.py`, `inference/long_context.py`
> **Companion:** [`attention.md`](attention.md) (the attention paths the cache feeds),
> [`yarn.md`](yarn.md) (rotated-K caching), [`OPTIMIZATIONS.md`](OPTIMIZATIONS.md) §11–15.

---

## 1. Overview

The inference path is where the model's two architectural choices pay off in
*wall-clock terms*, not just parameter counts:

1. **The mixed windowed/global KV cache** (`MixedKVCache`) delivers the 2×
   VRAM savings at long context (the first headline metric) by giving
   windowed layers an O(window) cache and global layers an O(T) but
   amortised-O(1)-per-step cache.
2. **Rotated-K caching** turns decode from O(T²) per token into O(T) per token
   by applying RoPE once at insertion rather than recomputing over the whole
   cache.

On top of that, `generate()` pre-allocates its output and avoids `torch.cat`
per step, and `long_context.PasskeyEvaluator` provides the second headline
metric: 128K passkey retrieval from a 4K-trained model.

---

## 2. `MixedKVCache` — the per-layer mixed cache

```python
class MixedKVCache:
    _GLOBAL_CAP_TOKENS = 4_000_000
    def append(self, layer_idx, k_rot, v, is_windowed, window) -> None
    def get(self, layer_idx, is_windowed) -> (K, V)
    def seq_len(self, layer_idx, is_windowed) -> int
```

A single cache object holds **two storage strategies in parallel**, one per
layer type — this is the literal implementation of the GPT-OSS
sliding/full alternation at the cache level. It stores **rotated K** (RoPE is
applied at insertion, not at attention time — see §4 and [`yarn.md`](yarn.md)).

### 2.1 Windowed layers — fixed-size ring buffer

Even-indexed layers cache only the last `window = 128` tokens. Storage is a
**ring buffer**: a single `(B, H_kv, window, D)` tensor allocated once at first
append, plus a `head` pointer and a `count`.

- **Append**: if the new chunk fits in the remaining tail
  (`head + T_new <= window`), copy it in place; otherwise wrap around,
  copying the first part to the tail and the second part to the head, then
  advance `head = (head + T_new) % window`. This is **O(window) memcpy per
  step, independent of T** — the whole cache is never reallocated.
- **Get**: when `count < window`, return the leading `count` slots (still
  filling up). When full, if `head == 0` the buffer is already in temporal
  order; otherwise return `cat(buf[head:], buf[:head])` — a single
  `torch.cat` to reconstruct temporal order from the ring. The result is a
  view (zero-copy slices where possible), so attention reads it without an
  extra copy.

This is OPT-11: the previous implementation did `torch.cat([old_k, k_rot])` on
every step, allocating and copying the *entire* cache each time — O(T) per
step, O(T²) total. The ring buffer makes per-step work O(window). For a 64K
context with `window = 128`, that is a **500× reduction** in per-step KV
work.

### 2.2 Global layers — exponentially-growing buffer

Odd-indexed layers cache the full sequence. Storage is a contiguous buffer
that **grows by 1.5× on demand**, capped at `_GLOBAL_CAP_TOKENS = 4 000 000`
per layer (a safety bound, ~4M tokens ≈ far beyond the 128K target):

```python
new_cap = max(needed, int(cur_cap * 1.5) + 1)
new_cap = min(new_cap, self._global_cap_tokens)
buf = torch.empty(B, H, new_cap, D, ...)
buf[:, :, :cur_len, :].copy_(old[:, :, :cur_len, :])     # copy existing
buf[:, :, cur_len:needed, :].copy_(k_rot)               # append new
```

- **Append** (no grow needed): in-place copy into the existing buffer's tail —
  O(T_new), no allocation.
- **Append** (grow needed): allocate a 1.5× buffer, copy the old prefix + the
  new chunk. This happens ~`log_{1.5}(T)` times over a generation, so the
  **amortised cost per step is O(1)** even though a single grow step is O(T).
  This is OPT-12: the previous `torch.cat`-every-step was O(T) per step and
  O(T²) total (~2 billion tokens of memory traffic over a 64K generation); the
  exponential buffer is O(N) total.

`get()` returns `buf[:, :, :cur_len, :]` — a zero-copy view.

### 2.3 Why the split?

A naive "one buffer per layer of size T" cache would (a) need to know T up
front — fine for fixed-length decode but not for open-ended generation, and
(b) over-allocate for windowed layers that only ever use `window` slots. The
mixed cache gives windowed layers their tight O(window) bound and global layers
an amortised-O(1) append, matching each layer's actual access pattern. This is
what produces the measured **1.94×–2.0× KV-cache reduction at 128K** in
`scripts/kv_cache_benchmark.py` — half the layers (the windowed ones) hold
`window = 128` tokens instead of `T = 131 072`.

---

## 3. `generate()` — the decode loop

```python
@torch.no_grad()
def generate(model, input_ids, max_new_tokens=64, temperature=0.7,
             top_p=0.9, use_cache=True) -> torch.Tensor
```

### 3.1 Pre-allocated output (OPT-13)

```python
output = torch.empty(B, T_prompt + max_new_tokens, dtype=input_ids.dtype, device=dev)
output[:, :T_prompt] = input_ids
...
output[:, T_prompt + step : T_prompt + step + 1] = next_id      # in-place write
```

The full output tensor is allocated once; each new token is written in place.
The previous `torch.cat([generated, next_id])` per step was O(T) per step and
O(T²) total over a generation — pure waste, since the output length is known
up front.

### 3.2 The prefill + decode structure

```
# Prefill: process the whole prompt at once, populate the cache.
x = model.embed(input_ids)
positions = arange(T_prompt)
for layer_idx, block in enumerate(model.blocks):
    x = _attn_forward_layer(block, layer_idx, x, positions, cache, sink_bias_cache)
next_token_logits = model.head(model.norm(x))[:, -1, :]

# Decode: one token at a time, reading from the cache.
for step in range(max_new_tokens):
    sample next_id from next_token_logits (greedy or top-p)
    output[:, T_prompt + step] = next_id
    x_step = model.embed(next_id)
    positions_step = tensor([cur_pos - 1])
    for layer_idx, block in enumerate(model.blocks):
        x_step = _attn_forward_layer(block, layer_idx, x_step, positions_step, cache, sink_bias_cache)
    next_token_logits = model.head(model.norm(x_step))[:, -1, :]
```

The prefill processes all `T_prompt` tokens in one forward pass and seeds the
cache; each decode step processes a *single* token and appends one (rotated) K
and one V per layer. This is the standard prefill/decode split; the GPT-OSS
twist is that the cache is *mixed*, so windowed and global layers behave
differently inside `_attn_forward_layer`.

### 3.3 Per-call sink-bias clamp cache (OPT-14)

```python
if sink_bias_cache is not None and id(attn) in sink_bias_cache:
    sink_bias_clamped = sink_bias_cache[id(attn)]
else:
    sink_bias_clamped = attn.sink_bias.clamp(attn._sink_clamp_min, attn._sink_clamp_max)
    if sink_bias_cache is not None:
        sink_bias_cache[id(attn)] = sink_bias_clamped
```

A `sink_bias_cache: dict` keyed by `id(attn)` is threaded through the decode
loop. The first call per layer computes the clamped sink bias; subsequent
calls reuse the cached tensor. With 12 layers × N decode steps, this avoids
12N redundant clamps. Small but free.

### 3.4 Sampling: greedy vs top-p

- `temperature <= 0` → greedy `argmax` (deterministic; used by `passkey_eval`).
- `temperature > 0` → top-p (nucleus) sampling: softmax with temperature, sort
  descending, zero out the tokens above the cumulative `top_p` mass
  (`cumsum - sorted_probs > top_p`), renormalise, `multinomial`, gather back.
  The `cumsum - sorted_probs > top_p` form keeps the highest-probability token
  always eligible even when it alone exceeds `top_p`.

### 3.5 `use_cache=False` — the correctness replay path

When the cache is disabled, each step re-runs the *full* prompt + history
through the model with no cache. This is correct but O(T) per step — useless
for production but invaluable for testing: `generate(..., use_cache=False,
max_new_tokens=1)` produces the *same logits* as `model(input_ids)`, so the
test suite can certify that the cached fast path matches the uncached ground
truth without any tolerance.

### 3.6 Device contract

```python
model.to(dev)   # no-op if already on dev
```

`generate` forces the model onto the input's device so embed/head/matmuls
never cross devices. `.to(dev)` is a no-op (no copy) when the model is already
on `dev`, so this is cheap to call per-generation inside an eval loop (e.g. the
passkey retriever calls `generate` hundreds of times).

---

## 4. Rotated-K caching — why decode is O(T), not O(T²)

`_attn_forward_layer` applies RoPE to the *new* K before appending it:

```python
cos, sin = attn.yarn(positions, n_pruned_dims=attn.n_pruned_dims)
q = apply_rope(q, cos, sin)
k_new_rot = apply_rope(k_new, cos, sin)
if cache is not None:
    cache.append(layer_idx, k_new_rot, v_new, ...)     # store ROTATED K
    cached_k, cached_v = cache.get(layer_idx, ...)
    k_for_q = cached_k                                  # already rotated
```

The cache stores `k_new_rot`, the *already-RoPE'd* key. Attention then reads
cached rotated K directly — no re-application of RoPE over the growing cache.
Without this, each decode step would recompute RoPE over the entire `T`-token
cache (O(T) RoPE applications per step → O(T²) over a generation), because
YaRN's frequencies are position-dependent. With it, each step rotates only the
*one* new K (O(1) per step). This composes with the ring buffer (windowed) and
exponential buffer (global) to make the *whole* decode loop O(T) per token
rather than O(T²).

This is only correct because RoPE is a *linear* rotation that commutes with
cache append — the rotated K of an old token does not change when a new token
is appended. A nonlinear or additive position encoding would not have this
property.

---

## 5. `long_context.PasskeyEvaluator` — the 128K benchmark

```python
class PasskeyEvaluator:
    def evaluate(self, context_lengths=(4096, 8192, 32768, 65536, 131072),
                 n_trials=100, passkey_position="middle", base_seed=42) -> dict[int, float]
```

The **passkey retrieval** benchmark (Mohtashami & Jaggi, 2023) is the canonical
long-context eval: hide a 5-digit passkey in a long filler-text context, ask
the model to retrieve it, measure accuracy as a function of where the passkey
sits and how long the context is. It is the second headline metric of this
repo: a 4K-trained model retrieving passkeys at 128K is the proof that YaRN +
pruned RoPE actually extrapolate.

### 5.1 Prompt construction

`build_prompt(passkey, context_length, passkey_position, seed)`:
1. `make_filler_text(target_tokens=context_length, seed=context_length)` —
   deterministic filler (a fixed 16-word vocabulary sampled with a seeded
   RNG). The filler seed is `context_length`, *not* the trial index, so every
   trial at the same context length sees the same filler — only the passkey
   is per-trial randomness.
2. Insert `"The passkey is {passkey}."` at `start` / `middle` / `end` of the
   filler word list.
3. Append the question template:
   `"There is an important info in the context above. Find it and remember
   it. The passkey is {passkey}. Now answer: what is the passkey?"`

### 5.2 Reproducibility design

- **Per-context-length RNG**: `rng = random.Random(base_seed + ctx_len)` —
  different context lengths are statistically independent (a fix at one length
  does not perturb the passkey set at another).
- **Passkeys without replacement**: `rng.sample(range(100_000), n_distinct)` —
  no duplicate passkeys within a context length's trial set.
- **Filler seeded by context length** (not trial): each context length has
  its own deterministic filler; only the passkey varies per trial. This
  isolates the variable under test (passkey position vs retrieval) from
  filler-text variation.

### 5.3 Evaluation

For each `(context_length, trial)`:
1. Tokenise the prompt, run `generate(model, input_ids, max_new_tokens=16,
   temperature=0.0, top_p=1.0, use_cache=True)` — greedy decode, 16 tokens
   (enough for a 5-digit number plus surrounding text).
2. Decode the *new* tokens only, regex-extract the first `\b\d{5}\b`, compare
   to the true passkey.

Expected behaviour: **≥ 95% at 4K** (training context — should be near-perfect),
**≥ 85% at 128K** (the YaRN extrapolation target). `scripts/passkey_eval.py` is
the CLI entrypoint; it falls back to a **stub** when no trained checkpoint is
provided (it still tests prompt construction and the cache mechanics, just not
retrieval accuracy — there are no learned weights to retrieve *with*).

---

## 6. Design rationale & rejected alternatives

| Decision | Rationale | Rejected alternative |
|---|---|---|
| Mixed cache (ring + exponential) | Matches each layer type's access pattern | One buffer per layer of size T — needs T up front, over-allocates windowed layers |
| Ring buffer for windowed layers | O(window) per step, fixed memory | `torch.cat` per step — O(T²) total |
| Exponential 1.5× growth for global | Amortised O(1) append | Fixed-size T buffer — needs T known; `torch.cat` — O(T²) |
| Store rotated K | O(1) RoPE per decode step | Recompute RoPE over whole cache — O(T²) |
| Pre-allocated output | No O(T²) `torch.cat` | `torch.cat` per step |
| `use_cache=False` replay path | Certifies the fast path against uncached ground truth | Trust the fast path — bugs hide |
| Sink-bias clamp cache | Avoids 12N redundant clamps | Recompute every layer every step |
| Passkey filler seeded by ctx_len | Isolates passkey-position as the variable | Per-trial filler — confounds filler variance with position |

---

## 7. Edge cases & pitfalls

- **`generate` mutates `model`'s device** via `model.to(dev)`. If you hold a
  reference to the model expecting it on a specific device, be aware it may
  move. The `.to(dev)` is a no-op when already correct.
- **`use_cache=False` is O(T²)** — never use it for long generation; it exists
  for correctness testing only.
- **Ring buffer `get()` allocates on wrap**: when `head != 0` and the buffer is
  full, `get()` does a `torch.cat` to reconstruct temporal order. This is one
  allocation per windowed-layer attention call in the wrapped state —
  unavoidable without a double-buffer, and tiny relative to the matmul.
- **`_GLOBAL_CAP_TOKENS = 4_000_000`** caps global-layer buffers. At 128K
  target context this is a 30× safety margin; a generation that exceeds it
  would silently stop growing (cache would drop new tokens). If you ever push
  past 4M tokens per layer, raise this.
- **`PasskeyEvaluator` needs a trained checkpoint**: on an untrained model the
  stub path runs prompt construction + decode but retrieval is random
  (~`1e-5` for a random 5-digit guess). Do not interpret stub accuracy as a
  model property.

---

## Implementation notes (extracted from code review)

- **`MixedKVCache` ring + exponential growth**: windowed layers use a
  fixed-size ring buffer of length `window` (O(window) append, O(1) amortised
  decode step). Global layers use an exponentially-growing buffer
  (1.5× growth capped at `_GLOBAL_CAP_TOKENS = 4_000_000`), giving O(N)
  total work over an N-token generation instead of O(N²) from `torch.cat`
  on every step.
- **Pre-allocated `generate` output**: `generate()` allocates the full
  `(B, T_prompt + max_new_tokens)` output tensor once and each new token is
  written in place. This avoids the O(T²) `torch.cat` per step that the
  original implementation had.
- **Per-call sink-bias clamp cache**: a `sink_bias_cache: dict` keyed by
  `id(attn)` is threaded through the decode loop. The first call per
  layer computes `attn.sink_bias.clamp(min, max)`; subsequent calls
  reuse the cached tensor. Avoids 12N redundant clamps over N decode
  steps.
- **`use_cache=False` replay path**: when the cache is disabled, the
  full prompt + history is replayed on every step (correct but slow).
  Useful for testing — `generate(..., use_cache=False, max_new_tokens=1)`
  gives the same logits as `model(input_ids)`.
- **Rotated-K caching**: `_attn_forward_layer` applies RoPE to the new K
  once and stores `k_new_rot` in the cache, so each decode step only
  rotates the single new K rather than recomputing RoPE over the
  growing cache. Decode is O(T) per token, not O(T²).