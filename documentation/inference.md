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
heads (`repeat_kv` expands at attention matmul time).

---

## MixedKVCache

`MixedKVCache` in `inference/generate.py` maintains two parallel storage systems indexed
by `layer_idx`:

| Storage | Layers | Semantics |
|---------|--------|-----------|
| `windowed_kv` | `is_windowed=True` | Fixed-size ring, capacity `window` |
| `global_kv` | `is_windowed=False` | Dynamic array, amortized O(1) append |

`reset()` clears all state between independent requests. `append(layer_idx, k_rot, v,
is_windowed, window)` accepts **already-rotated** K and V; `get(layer_idx, is_windowed)`
returns chronologically ordered tensors for attention.

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

Returns `(B, T_prompt + max_new_tokens)` token ids. `model.eval()` at entry; `model.to(dev)`
enforces the model↔input device contract (no-op when already aligned).

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

KV cache is populated for both windowed and global layers. Work is O(T_prompt) per layer
for attention (standard prefill).

**Phase B — Token-by-token decode** runs `max_new_tokens` steps:

1. Sample `next_id` from `next_token_logits` (greedy argmax when `temperature <= 0`)
2. Append to pre-allocated `output` buffer
3. Embed single token; `positions_step = tensor([cur_pos - 1])` — **absolute** index of
   the new token (not relative offset within the decode window)
4. Run all layers via `_attn_forward_layer` (append length-1 K/V to cache)
5. `next_token_logits = head(norm(x_step))[:, -1, :]`

Each decode step touches only the new token's activations plus cached K/V — no full-prefix
re-forward when `use_cache=True`.

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
documented in [architecture.md](architecture.md).

---

## Passkey retrieval (`inference/long_context.py` / `scripts/passkey_eval.py`)

`PasskeyEvaluator` implements the needle-in-a-haystack protocol (Mohtashami & Jaggi, 2023).

### Prompt construction

1. Generate deterministic filler text (`make_filler_text`) of roughly `context_length` words
2. Insert `"The passkey is {passkey}."` at `start`, `middle`, or `end`
3. Append question template asking the model to recall the 5-digit passkey

### Scoring

1. Tokenize prompt
2. `generate(..., max_new_tokens=16, temperature=0.0, use_cache=True)`
3. Decode only **new** tokens after the prompt
4. Extract first 5-digit number via regex `r"\b(\d{5})\b"`
5. Match against ground-truth passkey

Default context lengths: `4096, 8192, 32768, 65536, 131072` with `n_trials=100` distinct
random passkeys per length. Returns `{ctx_len: accuracy}`.

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

<!-- docs:verified 2026-07-31 · 7fe1247 -->
