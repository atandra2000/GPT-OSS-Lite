# GPT-OSS-Lite — Learned Attention Sinks

> **Audience:** beginner→intermediate. Builds softmax attention from a one-line
> definition, then derives the sink mechanism from what softmax forces when
> tokens are evicted. The authoritative deep dive — full history, math, and the
> per-path implementation — is [attention-sinks.md](attention-sinks.md).

---

## The one fact softmax forces on you

Attention output for query *i* is a weighted average of value vectors, where
the weights are a softmax over that query's scores against every *visible*
key:

$$
o_i \;=\; \sum_{j \in \mathcal{A}(i)} \alpha_{i,j}\, v_j,
\qquad
\alpha_{i,\cdot} = \mathrm{softmax}\big(\text{scores over } \mathcal{A}(i)\big),
\tag{1}
$$

and $\mathcal{A}(i)$ is the visible key set (causal; plus the window on
windowed layers, see [sliding-full-alternation.md](sliding-full-alternation.md)).
Softmax weights sum to 1 **over whatever set they are computed on**. That is
the whole problem in one sentence: if the visible set shrinks, the remaining
weights must grow. The model is never allowed to say "none of the above."

## The eviction problem, and what breaks without a sink

In a full-attention layer every past token stays visible, so the weights are
computed over a fixed, growing support. In a windowed layer the support is
not fixed: when token $j$ ages out of the window ($i - j \ge W$), its key and
value are dropped — at inference literally overwritten in
`inference/generate.py:MixedKVCache`'s ring buffer. Whatever attention mass
the model had been routing to token $j$ has to go somewhere: the surviving
keys absorb it, and every downstream output shifts discontinuously at the
moment of eviction. StreamingLLM (Xiao et al., 2023) identified the companion
symptom: strong-attention models park a large, roughly constant share of mass
on the first tokens, so eviction rips out the anchor, not a random token
(history and citations in [attention-sinks.md §2](attention-sinks.md)).

A **sink** gives the model a legal "none of the above": one extra key per head
whose logit is a learned constant and whose **value vector is exactly zero**.
Mass routed to the sink participates in the denominator — so the real keys'
weights still sum to less than 1 — but contributes nothing to the output:

$$
\alpha_{i,\text{sink}} = \frac{e^{s_h}}{\sum_{j \in \mathcal{A}(i)} e^{\tilde S_{i,j}} + e^{s_h}},
\qquad o_i = \sum_{j \in \mathcal{A}(i)} \alpha_{i,j} v_j
\tag{2}
$$

The sink is a softmax garbage collector: mass can disappear from the visible
support without touching any surviving key's contribution.

## How the sink is learned here

Each attention layer owns one bias per query head — a
`models/attention.py:GPTOSSAttention` parameter of shape `(n_heads,)`, created
when `models/transformer.py:ModelConfig.sink_bias` is true (it is, in
`configs/pretrain_a100_502m.yaml`). Across the 12-layer stack that is 8 × 12 =
96 scalars, initialized to zero by `models/transformer.py:GPTOSS._init_weights`
(re-verified by execution on 2026-09-21: 96 sink parameters). Zero init means
training starts from "denominator gains a constant +1" — the sink is inert but
present — and gradient descent learns per-head how much mass to park.

Each head learns independently: a pattern-matching head can learn a
sink-heavy profile, an aggregator head a sink-light one. During decode the
bias enters as an **additive** mask column, not a real key:

```python
# illustrative — trimmed from models/attention.py:causal_attention
sink_k = torch.zeros(B, H, 1, D)          # zero key: never matches anything
sink_v = torch.zeros(B, H, 1, D)          # zero value: contributes nothing
k_ext = torch.cat([key_states, sink_k], dim=2)
mask[:, :, T_k] = sink_bias.to(dtype).unsqueeze(1).expand(H, T_q)  # additive logit
return F.scaled_dot_product_attention(q, k_ext, v_ext, attn_mask=mask)
```

The appended K/V are zeros, so the sink column's score is *purely* the learned
bias; the mask column is the bias broadcast to every query row. The reference
implementation `models/attention.py:manual_causal_attention` computes the same
thing explicitly — concatenate the bias as an extra score column, softmax,
strip the sink weight before the value matmul — and
`tests/test_attention.py:test_sink_path_matches_manual_at_prefill` pins the two
paths to agree.

## What the learned value means

The bias $s_h$ is a logit, so its magnitude sets the fraction of mass the head
can hide:

| $s_h$ | Sink mass when scores are O(1) | Behavior |
|---|---|---|
| $s_h \to -\infty$ | $\to 0$ | sink off; equivalent to no sink |
| $s_h = 0$ | $\approx 1/(Z+1)$ | constant +1 in the denominator |
| $s_h = 15$ | up to $e^{15} \approx 3.3\times10^{6}$ vs $Z$ | nearly all mass absorbable |

The clamp bounds the parameter's *effective* value at forward time to
$[-10, 15]$ via `models/attention.py:SINK_CLAMP_MIN` /
`models/attention.py:SINK_CLAMP_MAX` ($e^{-10} \approx 4.5\times10^{-5}$: sink
effectively disabled). The clamp exists because SDPA adds the mask to the
scores before softmax, and an unclamped bias drifting to large magnitudes
overflows BF16's range to `inf`, which softmax turns into NaN (mechanism and
the choice of bounds in [attention-sinks.md §6](attention-sinks.md); the
numeric behavior is pinned by
`tests/test_attention.py:test_sink_bias_clamped_at_forward`). The clamp is applied to a temporary, never the parameter itself: gradient
flows at interior values and stops at the bounds, so the parameter cannot
grow further in a clamped direction but is never mutated.

## Why windowed layers make sinks non-optional

On a full layer, a sink is a mild convenience. On a windowed layer it is the
stability mechanism: eviction happens continuously during decode, and without
a sink every eviction redistributes mass onto surviving keys *and* the sink
cannot absorb the orphaned share. With it, the model can route a steady fraction
of mass to the sink and let window eviction silently reclaim it. This is why
the sink parameter and the alternating pattern are designed together in this
repo — the sink lives in the same module that owns
`models/attention.py:GPTOSSAttention.is_windowed`, and the two concepts share
one implementation path in `models/attention.py:causal_attention`
(`tests/test_attention.py:test_sink_and_window_compose_high_bias_dampens`
exercises them jointly).

> Evidence note: the *mechanism* — zero value column, additive logit,
> per-head parameter, clamp — is read directly from the cited code and pinned
> by the named tests. The *training-dynamics* claims (what trained heads
> learn, drift magnitudes without a sink) come from the StreamingLLM
> literature summarized in [attention-sinks.md](attention-sinks.md); no
> trained checkpoint here measures them yet [INFERENCE].

## Related documentation

- [attention-sinks.md](attention-sinks.md) — authoritative deep dive (history,
  full math, SDPA walkthrough, clamp rationale).
- [sliding-full-alternation.md](sliding-full-alternation.md) — the eviction
  schedule that creates the problem.
- [yarn-scaling.md](yarn-scaling.md) — the positional-encoding half of the
  long-context design.
- [../inference.md](../inference.md) — where eviction happens at decode time.

<!-- docs:verified 2026-09-21 · 41a8c94 -->
