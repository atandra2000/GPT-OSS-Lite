# GPT-OSS-Lite — Sliding-Window / Full Attention Alternation

> **Audience:** beginner. Assumes only that attention averages value vectors with
> softmax weights that sum to 1; everything else is built from first principles
> below. For the authoritative deep dive with the full math and the SDPA
> walkthrough, read [attention-sinks.md](attention-sinks.md).

---

Attention has two costs. The familiar one is compute: every query scores every
key, O(T²) per layer. The one that hurts long-context inference is memory: to
generate token T+1, every layer must keep the keys and values of all T past
tokens, so the KV cache grows linearly with context on **every** layer. At 128K
tokens a full-attention cache is gigabytes before the weights are even loaded.

Sliding-window attention (SWA) fixes the memory half by refusing to look
further back than a fixed window W: a query at position *i* sees exactly the
keys in {j : j ≤ i and i − j < W} — the last W tokens. The cache for such a
layer is capped at W entries no matter how long the context grows. The price is
obvious: the layer is blind beyond W tokens back, and any dependency longer
than W must be carried some other way.

GPT-OSS-Lite refuses to pay either full price. It alternates the two modes,
layer by layer, with a period of 2:

| `layer_idx` | Mode | Window | KV storage |
|---|---|---|---|
| 0, 2, 4, 6, 8, 10 | sliding window | 128 | ring buffer, ≤ 128 rows |
| 1, 3, 5, 7, 9, 11 | full causal | — | grows with context |

## Which layers slide, and why that pattern

The choice is made per layer in `models/attention.py:GPTOSSAttention`:
`self.is_windowed = (layer_idx % 2 == 0)` — even layers slide, odd layers keep
full history. The block builder `models/transformer.py:GPTOSS` hands each
`models/transformer.py:GPTOSSBlock` its index, and the window itself is the
`models/transformer.py:ModelConfig.window_size` field (128 in
`configs/pretrain_a100_502m.yaml`). The pattern is test-pinned in
`tests/test_attention.py:test_attention_module_alternating_pattern`, and was
re-verified by direct execution on 2026-09-21: windowed = {0, 2, 4, 6, 8, 10},
full = {1, 3, 5, 7, 9, 11}.

Why even layers and not odd? Nothing about evenness matters — the requirement
is only that the two modes alternate with period 2, so the two layer kinds
appear equally often and no two windowed layers are ever adjacent. Layer 0 is
windowed, so the very first projection into the residual stream is cheap; every
full layer has a windowed layer immediately before it to compact recent context
into the residual stream.

## The mask, from first principles

Full causal attention allows every key with j ≤ i. Sliding-window attention
adds the second condition, i − j < W. `models/attention.py:_window_mask`
computes exactly this conjunction:

$$
\mathcal{A}(i) = \{\, j : j \le i \;\wedge\; i - j < W \,\}
\tag{1}
$$

and `models/attention.py:causal_attention` selects between the modes with one
argument — `window=None` means full, an integer selects the banded mask. The
production module passes the decision straight through:

```python
# illustrative — trimmed from models/attention.py:GPTOSSAttention.forward
out = causal_attention(
    query_states, key_states, value_states,
    window=self.window_size if self.is_windowed else None,
    sink_bias=sink_bias_clamped,
)
```

The window arithmetic was re-verified by execution (2026-09-21): with T = 6 and
W = 3, query 5 attends to keys {3, 4, 5} — the last three positions, no more.

## Why alternation preserves long range

A stack of *only* windowed layers would not be blind at long range —
information still walks forward one window-step per layer — but it would be
*cache-cheap and context-poor*: nothing in the stack keeps a memory of a token
after it leaves the window. A stack of only full layers is the opposite trade.
Alternation gets both properties: on a windowed layer, a token's influence can
advance up to W − 1 positions; the very next full layer stores K/V for **all**
past tokens, so whatever the windowed layers mixed into recent positions stays
addressable forever after. Long-range detail is therefore carried by half the
layers, at half the cache cost, and the two kinds of layers alternate so the
compaction → broadcast cycle repeats every two layers. This is an argument
about information flow, not a measurement — the quality claim itself is
inherited from the GPT-OSS lineage it reproduces [INFERENCE].

## The KV-cache math, measured

Per token per layer the cache holds 2 × n_kv_heads × head_dim values in BF16:
2 × 4 × 96 × 2 = 1536 bytes. A pure-GQA full model stores 12·T such rows; the
alternating model stores 6·min(W, T) + 6·T. The ratio at context T is

$$
\frac{12T}{6W + 6T} = \frac{2T}{W + T} \;\xrightarrow{T \gg W}\; 2.0\times
\tag{2}
$$

`scripts/kv_cache_benchmark.py:cache_bytes` evaluates this analytically and was
re-run on 2026-09-21: 2.25 GB (pure GQA) vs 1.13 GB (alternating) at 128K —
**2.00×** reduction, above the repo's 1.8× headline threshold. At inference the
cap is enforced by the ring buffer in `inference/generate.py:MixedKVCache`:
windowed layers overwrite a fixed 128-row buffer while full layers keep the
whole prefix.

## The partner mechanisms

Two other pieces of the architecture exist *because* of this alternation.
First, evicted tokens take their attention mass with them — the learned sink
bias exists so the orphaned softmax mass has somewhere silent to go
([learned-sinks.md](learned-sinks.md)). Second, the two layer kinds diverge in
positional encoding: windowed layers never look more than 128 tokens back so
they keep all RoPE frequencies, while full layers prune the fastest quarter of
RoPE dims to avoid over-rotation at 128K
([yarn-scaling.md](yarn-scaling.md)).

## Related documentation

- [attention-sinks.md](attention-sinks.md) — authoritative deep dive: sink
  theory, the SDPA sink column, and the alternation pattern.
- [learned-sinks.md](learned-sinks.md) — what the sink bias is and why
  windowed eviction needs it.
- [yarn-scaling.md](yarn-scaling.md) — YaRN scaling and per-layer RoPE pruning.
- [../inference.md](../inference.md) — `MixedKVCache` ring buffers in decode.
- [attention-and-positional.md](attention-and-positional.md) — attention math
  and the sliding-window banded mask, from scratch.

<!-- docs:verified 2026-09-21 · 41a8c94 -->
