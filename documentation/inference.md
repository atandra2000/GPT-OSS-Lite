# Inference — Mixed KV Cache and Long-Context Evaluation

> **Chapter on `inference/generate.py` and `inference/long_context.py`.** This
> chapter explains how GPT-OSS-Lite decodes autoregressively with a heterogeneous
> per-layer KV cache, how sink-bias clamping is cached for decode, the passkey
> retrieval protocol, and what the KV-cache benchmark measures. Prerequisites:
> [ATTENTION_SINKS.md](ATTENTION_SINKS.md),
> [transformer.md](transformer.md).

---

## Table of contents

1. [Why a custom inference path?](#1-why-a-custom-inference-path)
2. [Architecture recap for inference](#2-architecture-recap-for-inference)
3. [`MixedKVCache` design](#3-mixedkvcache-design)
4. [Windowed layer: ring buffer](#4-windowed-layer-ring-buffer)
5. [Global layer: exponential growth](#5-global-layer-exponential-growth)
6. [The `append` / `get` contract](#6-the-append--get-contract)
7. [`_attn_forward_layer`](#7-_attn_forward_layer)
8. [Sink bias clamp cache](#8-sink-bias-clamp-cache)
9. [The `generate()` loop](#9-the-generate-loop)
10. [Top-p sampling](#10-top-p-sampling)
11. [Decode without cache](#11-decode-without-cache)
12. [Passkey evaluation protocol](#12-passkey-evaluation-protocol)
13. [`scripts/passkey_eval.py`](#13-scriptspasskey_evalpy)
14. [KV-cache benchmark meaning](#14-kv-cache-benchmark-meaning)
15. [Memory and complexity](#15-memory-and-complexity)
16. [Operational commands](#16-operational-commands)
17. [Failure modes](#17-failure-modes)
18. [Where to go next](#18-where-to-go-next)

---

## 1. Why a custom inference path?

`GPTOSS.forward` in `models/transformer.py` is optimized for **training**:
full-sequence forward, gradient checkpointing, aux loss aggregation. Autoregressive
generation needs different behavior:

1. **Incremental decode** — after the prompt, only one new token per step
2. **KV reuse** — past keys and values must not be recomputed
3. **Heterogeneous caches** — even layers (sliding window) store at most `W`
   tokens; odd layers (global) store the full prefix
4. **Rotated keys** — RoPE is applied before caching; stored tensors are
   already position-rotated

`inference/generate.py` implements this without modifying the training forward.
It calls block submodules (`norm1`, `attn`, `moe`, `norm2`) through
`_attn_forward_layer`.

---

## 2. Architecture recap for inference

For the 502M production config:

| Layer index | Attention type | KV stored |
|-------------|----------------|-----------|
| 0, 2, 4, 6, 8, 10 | Sliding window `W=128` | Last `min(T, 128)` tokens |
| 1, 3, 5, 7, 9, 11 | Full (global) | All `T` tokens |

GQA: queries use 8 heads; K/V use 4 heads (`repeat_kv` expands K/V to 8 at
attention matmul time).

Sink bias: per-head scalar, clamped to `[-10, 15]` at forward — see
[ATTENTION_SINKS.md](ATTENTION_SINKS.md).

YaRN: position-dependent RoPE frequencies; `positions` tensor must reflect
**absolute** token index during decode (0, 1, …, T-1 for prompt; then
`cur_pos-1` for each new token).

---

## 3. `MixedKVCache` design

```python
class MixedKVCache:
  _GLOBAL_CAP_TOKENS = 4_000_000

  def __init__(self, global_cap_tokens: int | None = None):
      self.windowed_kv: List[...] = []   # ring buffers per SWA layer
      self.global_kv: List[...] = []    # growable buffers per global layer
      self.global_lengths: List[int] = []
      self.global_caps: List[int] = []
```

Two parallel storage systems indexed by `layer_idx`:

| Storage | Layers | Semantics |
|---------|--------|-----------|
| `windowed_kv` | `is_windowed=True` | Fixed-size ring, capacity `window` |
| `global_kv` | `is_windowed=False` | Dynamic array, amortized O(1) append |

`reset()` clears all state — call between independent generation requests.

`__len__` returns `max(len(windowed_kv), len(global_kv))` — useful for
debugging how many layers have been touched.

### Why "mixed"?

A single flat cache cannot capture GPT-OSS memory behavior. At sequence length
131072:

- 6 global layers × 131072 tokens × KV footprint
- 6 windowed layers × 128 tokens × KV footprint

The headline **~2× KV reduction** comes entirely from this split. Analytical
proof: `scripts/kv_cache_benchmark.py` (Section 14).

---

## 4. Windowed layer: ring buffer

Each windowed layer stores `[buf_k, buf_v, head, count]`:

| Field | Meaning |
|-------|---------|
| `buf_k`, `buf_v` | `(B, H_kv, window, head_dim)` tensors |
| `head` | Next write index in the ring (0 ≤ head < window) |
| `count` | Effective history length (`≤ window`) |

### Prompt phase (`T_new` tokens at once)

When the prompt fits in one forward (typical):

- If `T_new >= window`: copy only the **last `window`** K/V into the buffer;
  set `head=0`, `count=window`
- If `T_new < window`: fill from the start; `count = T_new`

### Decode phase (one token per step)

Each step appends one K/V slice:

1. Write at `head` position
2. Wrap with `(head + 1) % window`
3. `count = min(window, count + 1)`

### Chronological ordering on read

`get()` for windowed layers may need to **unrotate** the ring when `head != 0`:

```python
k_ordered = torch.cat([k[:, :, head:, :], k[:, :, :head, :]], dim=2)
```

Attention always sees K/V in temporal order for the last `count` positions.

---

## 5. Global layer: exponential growth

Global layers keep the **entire prefix** for full attention. Storage strategy:

1. **First write:** allocate `(B, H_kv, new_cap, D)` with `new_cap = max(needed, 1)`
2. **Growth:** when `cur_len + T_new > cur_cap`, reallocate with
   `new_cap = max(needed, int(cur_cap * 1.5) + 1)`
3. **Cap:** `new_cap = min(new_cap, _GLOBAL_CAP_TOKENS)` where default cap is
   **4,000,000 tokens** per layer

The 1.5× growth factor gives amortized O(1) append cost per token — standard
dynamic array doubling strategy with a gentler constant.

`global_lengths[layer_idx]` tracks valid prefix length; `global_caps[layer_idx]`
tracks allocated capacity (may be larger than length).

### Why cap at 4M?

Safety rail against runaway allocation on bugs or adversarially long generation.
128K context is well under this cap. For research beyond 4M, pass a custom
`global_cap_tokens` to `MixedKVCache(...)`.

---

## 6. The `append` / `get` contract

### `append(layer_idx, k_rot, v, is_windowed, window)`

- `k_rot`: **already RoPE-rotated** keys, shape `(B, H_kv, T_new, D)`
- `v`: values, same shape
- Appends along sequence dim for this layer only

### `get(layer_idx, is_windowed) -> (K, V)`

Returns cached tensors for attention, or `(None, None)` if empty.

- Windowed: at most `window` tokens, chronologically ordered
- Global: `K[:, :, :cur_len, :]`

### `seq_len(layer_idx, is_windowed)`

Introspection helper — current cached length at a layer.

---

## 7. `_attn_forward_layer`

Core per-layer inference step (simplified flow):

```
x_norm = block.norm1(x)
q, k_new, v_new = projections + RoPE on q and k_new
cache.append(layer_idx, k_new_rot, v_new, is_windowed, window)
k, v = cache.get(...) or k_new_rot, v_new
k, v = repeat_kv(k, v)  # GQA
out = causal_attention(q, k, v, window=..., sink_bias=clamped)
x = x + o_proj(out)
x = x + moe(block.norm2(x))
return x
```

Differences from training `GPTOSSAttention.forward`:

- Uses cached K/V instead of full-sequence matmul over recomputed history
- MoE runs on every token (no aux loss needed at inference — discarded as `_`)
- Same sink clamp and window mask logic as training

Rotary: `cos, sin = attn.yarn(positions, n_pruned_dims=...)` then
`apply_rope` — matches [yarn.md](yarn.md) training behavior.

---

## 8. Sink bias clamp cache

Training clamps sink bias every forward:

```python
sink_bias_clamped = self.sink_bias.clamp(SINK_CLAMP_MIN, SINK_CLAMP_MAX)
# SINK_CLAMP_MIN = -10.0, SINK_CLAMP_MAX = 15.0
```

During multi-step `generate()`, the same clamped tensor would be recomputed
every layer every step. `generate()` passes a `sink_bias_cache: dict` keyed by
`id(attn)`:

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

See [ATTENTION_SINKS.md §6](ATTENTION_SINKS.md#6-bf16-clamp-rationale) for why clamp prevents BF16 mask-add overflow.

---

## 9. The `generate()` loop

```python
@torch.no_grad()
def generate(model, input_ids, max_new_tokens=64,
             temperature=0.7, top_p=0.9, use_cache=True) -> Tensor
```

Returns `(B, T_prompt + max_new_tokens)` token ids.

### Phase A — Prompt prefill

```python
model.to(dev)  # no-op if already on device
cache = MixedKVCache() if use_cache else None
sink_bias_cache = {}

x = model.embed(input_ids)
positions = torch.arange(T_prompt, device=dev)
for layer_idx, block in enumerate(model.blocks):
    x = _attn_forward_layer(block, layer_idx, x, positions, cache, sink_bias_cache)
x = model.norm(x)
next_token_logits = model.head(x)[:, -1, :]
```

All prompt tokens processed in **one parallel pass** per layer (standard
prefill). KV cache populated for both windowed and global layers.

### Phase B — Token-by-token decode

For each of `max_new_tokens` steps:

1. Sample `next_id` from `next_token_logits` (greedy if `temperature <= 0`)
2. Append to `output` buffer
3. If `use_cache`:
   - Embed single token `x_step`
   - `positions_step = tensor([cur_pos - 1])` — absolute index of new token
   - Run all layers with `_attn_forward_layer` (append length-1 K/V)
   - `next_token_logits = head(norm(x_step))[:, -1, :]`
4. If not `use_cache`: re-forward entire prefix (correctness reference, slow)

`model.eval()` is set at entry; dropout is not used in GPT-OSS-Lite.

---

## 10. Top-p (nucleus) sampling

When `temperature > 0`:

```python
probs = softmax(logits / temperature)
sorted_probs, sorted_idx = probs.sort(descending=True)
cumsum = sorted_probs.cumsum(dim=-1)
mask = cumsum - sorted_probs > top_p
sorted_probs[mask] = 0.0
sorted_probs /= sorted_probs.sum(dim=-1, keepdim=True)
next_id = multinomial(sorted_probs, 1)
next_id = sorted_idx.gather(-1, next_id)
```

Default `top_p=0.9`, `temperature=0.7`. Passkey eval uses `temperature=0.0`
(greedy argmax) for deterministic scoring.

---

## 11. Decode without cache

`use_cache=False` recomputes the full prefix every step:

```python
full_input = output[:, : T_prompt + step + 1]
x_full = model.embed(full_input)
positions_full = torch.arange(full_input.size(1), device=dev)
for layer_idx, block in enumerate(model.blocks):
    x_full = _attn_forward_layer(..., cache=None, ...)
```

Complexity: O(T²) per step — usable only for short sequences and correctness
checks. Production long-context eval always sets `use_cache=True`.

---

## 12. Passkey evaluation protocol

`PasskeyEvaluator` in `inference/long_context.py` implements the **needle-in-a-haystack**
variant popularized by Mohtashami & Jaggi (2023).

### Prompt construction

1. Generate deterministic filler text (`make_filler_text`) of roughly
   `context_length` words
2. Insert `"The passkey is {passkey}."` at `start`, `middle`, or `end`
3. Append question template:

```
There is an important info in the context above.
Find it and remember it. The passkey is {passkey}.
Now answer: what is the passkey?
```

### Scoring

1. Tokenize prompt (eval script uses a char-level stub tokenizer for portability)
2. `generate(..., max_new_tokens=16, temperature=0.0)`
3. Decode only **new** tokens after prompt
4. Extract first 5-digit number via regex `r"\b(\d{5})\b"`
5. Match against ground-truth passkey

### Default context lengths

```python
context_lengths = [4096, 8192, 32768, 65536, 131072]
n_trials = 100  # distinct random passkeys per length
```

Returns `{ctx_len: accuracy}` dict.

### Target accuracy (README headline)

| Position range | Target |
|----------------|--------|
| 0 – 32K | ≥ 95% |
| 32K – 96K | ≥ 90% |
| 96K – 128K | ≥ 85% |

These require a model trained with YaRN at 4K that successfully extrapolates —
see [yarn.md](yarn.md). Untrained models score near zero.

---

## 13. `scripts/passkey_eval.py`

CLI wrapper:

```bash
python3 scripts/passkey_eval.py \
    --checkpoint path/to/model.safetensors \
    --n-trials 100 \
    --context-lengths 4096 8192 32768 65536 131072 \
    --position middle \
    --seed 42
```

Behavior:

1. Loads `ModelConfig` from `configs/pretrain_a100_502m.yaml`
2. Builds `GPTOSS`, loads safetensors weights (`strict=False` for flexibility)
3. Uses `_CharTokenizer` stub — ord(char) mod vocab for CPU portability
4. Runs `PasskeyEvaluator.evaluate`
5. Prints accuracy table; checks ≥ 85% at max context length

For production eval, swap in the real LLaMA-3 tokenizer to match training
distribution — the stub is documented as a harness convenience.

---

## 14. KV-cache benchmark meaning

`scripts/kv_cache_benchmark.py` is an **analytical** script — no GPU, no model
load. It computes KV bytes from architecture constants:

```python
N_LAYERS = 12
N_WINDOWED = 6
N_GLOBAL = 6
N_KV_HEADS = 4
HEAD_DIM = 96
WINDOW = 128
DTYPE_BYTES = 2  # BF16
```

### Per-token KV size (one layer)

```
kv_bytes_per_token = 2 * N_KV_HEADS * HEAD_DIM * DTYPE_BYTES
                   = 2 * 4 * 96 * 2 = 1536 bytes
```

Factor 2 = K and V.

### Total cache bytes at sequence length `T`

**Pure GQA (all full):**

```
bytes = N_LAYERS * T * kv_bytes_per_token
```

**SWA/Full mix:**

```
windowed_tokens = N_WINDOWED * min(WINDOW, T)
global_tokens   = N_GLOBAL * T
bytes = (windowed_tokens + global_tokens) * kv_bytes_per_token
```

### Example table (batch=1, BF16)

| Context | Pure GQA | SWA/Full | Reduction |
|--------:|---------:|---------:|----------:|
| 4,096 | 72 MB | 72 MB | 1.00× |
| 16,384 | 288 MB | 144 MB | 2.00× |
| 131,072 | 2.25 GB | 1.13 GB | 2.00× |

Pass threshold: **≥ 1.8×** at 128K (`THRESHOLD = 1.8`).

### Relationship to `MixedKVCache`

The benchmark assumes ideal caching (exactly `W` windowed tokens, no overhead).
`MixedKVCache` adds ring metadata and capacity slack — real memory is slightly
higher but same asymptotic scaling.

The benchmark validates the **architectural claim**, not runtime allocator behavior.

---

## 15. Memory and complexity

### Per decode step (with cache)

| Layer type | Work per new token | KV growth |
|------------|-------------------|-----------|
| Windowed | O(W) attention | O(1) storage |
| Global | O(T) attention | O(1) amortized append |

Dominant cost at 128K: **6 global layers** each attend over full `T`.

### KV memory at T=131072 (production)

~1.13 GB BF16 (batch=1) from benchmark — before activations, MoE weights, or
framework overhead.

### Sink + window interaction

Windowed layers apply both causal mask **and** sliding mask plus sink column —
attention never sees more than `W` prior keys plus sink semantics. See
[ATTENTION_SINKS.md](ATTENTION_SINKS.md).

---

## 16. Operational commands

```bash
# Analytical KV headline metric (CPU)
python3 scripts/kv_cache_benchmark.py

# Passkey eval (GPU + checkpoint)
python3 scripts/passkey_eval.py --checkpoint checkpoints/pretrain_a100/model_step_61000.safetensors

# Programmatic generation
python3 -c "
import torch, yaml
from models.transformer import GPTOSS, ModelConfig
from inference.generate import generate
with open('configs/pretrain_a100_502m.yaml') as f:
    cfg = ModelConfig(**yaml.safe_load(f)['model'])
m = GPTOSS(cfg).eval()
ids = torch.randint(0, 1000, (1, 32))
out = generate(m, ids, max_new_tokens=8, use_cache=True)
print(out.shape)
"
```

E2E GPU smoke (`scripts/e2e_gpu_smoke.py`) includes a `MixedKVCache` generation
step on a tiny model.

---

## 17. Failure modes

| Symptom | Likely cause | Fix |
|---------|--------------|-----|
| Garbage long-context output | Untrained model or wrong tokenizer | Train; use LLaMA-3 tokenizer |
| OOM at 128K prefill | Batch > 1 or insufficient VRAM | batch=1; BF16; shorter eval |
| Slower than expected decode | `use_cache=False` | Enable cache |
| Position bugs / repeated tokens | Wrong `positions` during decode | Use absolute index `cur_pos-1` |
| Global cap hit | Sequence > 4M tokens | Raise `global_cap_tokens` |
| Sink overflow / NaN logits | Missing clamp | Ensure `SINK_CLAMP_*` path used |

---

## 18. Where to go next

| Topic | Document |
|-------|----------|
| Attention implementation | [ATTENTION_SINKS.md](ATTENTION_SINKS.md#part-b--implementation-modelsattentionpy) |
| Sink theory | [ATTENTION_SINKS.md](ATTENTION_SINKS.md) |
| YaRN / RoPE | [yarn.md](yarn.md), [rotary.md](rotary.md) |
| Model composition | [transformer.md](transformer.md) |
| Training (no KV cache) | [training.md](training.md) |
| Config limits (`eval_max_seq_len`) | [configs.md](configs.md) |
| Onboarding commands | [getting_started.md](getting_started.md) |

---

<!-- docs:verified 2026-07-31 · fa6f918 -->
