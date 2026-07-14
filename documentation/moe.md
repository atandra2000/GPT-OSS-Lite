# Mixture-of-Experts FFN for GPT-OSS-Lite

> **Source:** `models/moe.py`
> **Companion:** [`../AGENTS.md`](../AGENTS.md) — the "do not replace the
> standard aux loss" rule lives here.

---

## 1. Overview

Every GPT-OSS-Lite block replaces the single dense FFN of a classic
transformer with a **Mixture-of-Experts** (MoE) FFN: `8` routed SwiGLU
experts of which the top-`2` are activated per token, plus `1` shared expert
that is always on. This is what gives the model its **50.8% sparsity** — 502 M
total parameters but only ~247 M active per forward pass — and it is the
second architectural pillar after sliding-window attention.

The implementation is deliberately distinct from the sibling
[DeepSeek-v3-Lite](https://github.com/atandra2000/DeepSeek-v3-Lite) project in
three ways, each chosen to keep this repo a clean educational reference for the
*GPT-OSS* MoE rather than a copy of DeepSeek's:

| Aspect | GPT-OSS-Lite (this repo) | DeepSeek-v3-Lite |
|---|---|---|
| Aux loss | **Standard** Switch/GShard load-balancing (α=0.01) | Aux-loss-free bias trick |
| Routing granularity | **Top-2 of 8** (coarse) | Top-4 of 20 (fine) |
| Dispatch | **Grouped / vectorized** (per-expert loop, CPU-friendly) | Stacked `bmm` |
| Shared expert | 1 (same) | 1 (same) |

---

## 2. The SwiGLU expert

```python
class SwiGLUExpert(nn.Module):
    """Single SwiGLU expert: W2(silu(W1(x)) * W3(x))."""
    def forward(self, x):
        return self.w2(F.silu(self.w1(x)) * self.w3(x))
```

Each expert is a **SwiGLU** (Swish-Gated Linear Unit) FFN with three
weight matrices:

```
gate  = W1 x          ∈ ℝ^{ffn_dim}      (dim → ffn_dim = 768 → 1536)
up    = W3 x          ∈ ℝ^{ffn_dim}
h     = silu(gate) * up                   Swish gating: silu(z)=z·σ(z)
out   = W2 h          ∈ ℝ^{dim}           (ffn_dim → dim)
```

SwiGLU is the standard GLU variant in modern LLMs (LLaMA, DeepSeek, GPT-OSS):
the `silu` gating is smooth and empirically outperforms ReLU/GLU variants. The
three-matrix form means each expert holds `3 · d_model · ffn_dim =
3 · 768 · 1536 ≈ 3.54 M` parameters.

---

## 3. The router — top-k gating

```python
class MoERouter(nn.Module):
    def __init__(self, d_model, n_experts, n_activated)
    def forward(self, x) -> (indices, weights, all_logits)
```

Routing is a single linear projection `gate: d_model → n_experts` followed by
softmax and top-k selection:

```
logits   = W_gate x                       ∈ ℝ^{n_experts}
probs    = softmax(logits)                ∈ ℝ^{n_experts}    (FP32!)
topk_w, topk_idx = probs.topk(n_activated)
topk_w   = topk_w / topk_w.sum()           renormalise the k selected weights
```

The router returns three things:
- **`indices`** `(B·T, k)` — which expert each token goes to (here `k = 2`).
- **`weights`** `(B·T, k)` — the renormalised routing weights, summed to 1 per
  token. These multiply each expert's output before scattering back.
- **`all_logits`** `(B·T, n_experts)` — the *full* pre-topk logits, kept for the
  auxiliary loss (which needs the mean probability of *every* expert, not just
  the chosen ones).

Two numerical-stability details:
1. The softmax runs in **FP32** (`F.softmax(logits.float(), dim=-1)`) even when
   the input is BF16 — small per-expert probabilities underflow to zero in BF16.
2. The renormalisation denominator is clamped at `1e-6` so a degenerate
   all-equal softmax does not divide by zero.

---

## 4. The auxiliary load-balancing loss

```python
def aux_load_balancing_loss(all_logits, n_experts, n_activated) -> torch.Tensor
```

This is the **standard** Switch Transformer / GShard load-balancing loss —
*not* the aux-loss-free bias trick that DeepSeek-v3 uses. GPT-OSS-Lite keeps it
deliberately, both as an educational contrast and because at this scale
(`8` experts, top-`2`) the standard loss is simple and stable.

### 4.1 The math

The loss penalises two kinds of imbalance simultaneously:

```
f_i = (fraction of tokens routed to expert i)         # from top-k selection
P_i = (mean softmax probability of expert i)          # over all tokens
L   = n_experts · Σ_i f_i · P_i
```

- **`f_i`** is the *dispatch* frequency — how often expert `i` was actually
  chosen. Computed via `torch.bincount(topk_idx)` over the flattened
  `(N·k,)` selection indices, divided by `N·k`.
- **`P_i`** is the *average router probability* — how much the router *wants*
  to send tokens to expert `i`, regardless of whether it did. Computed as
  `probs.mean(dim=0)`.

The product `f_i · P_i` is what makes this work: a collapsed router that sends
everything to one expert has high `f_i` and high `P_i` for that expert, so the
product (and the loss) is large. A perfectly balanced router has
`f_i = P_i = 1/n_experts` for all `i`, giving
`L = n_experts · n_experts · (1/n_experts)² = 1` — the minimum. The
`n_experts` prefactor normalises the loss to be scale-invariant to expert
count.

### 4.2 Why both `f` and `P`?

Using `f` alone is non-differentiable (top-k selection is discrete). Using
`P` alone is differentiable but weak — the router can keep `P` uniform while
dispatch collapses (the argmax is what matters, not the softmax shape). The
product couples a differentiable signal (`P`) to the actual dispatch outcome
(`f`): gradients flow through `P`, but the loss only goes *down* when `f` also
balances. This is the GShard insight.

### 4.3 FP32 softmax + `bincount`

The loss upcasts logits to FP32 for the softmax and mean (the same BF16
underflow concern as the router), and uses `torch.bincount` for the per-expert
fraction — a single vectorised histogram pass, no Python loop over experts. The
result is cast back to the input dtype before returning. The training loop
adds `α · L` to the CE loss with `α = 0.01` (see [`training.md`](training.md)).

---

## 5. Shared expert

```python
if self.n_shared > 0:
    self.shared_experts = nn.ModuleList([SwiGLUExpert(...) for _ in range(n_shared)])
...
shared_out = sum(e(flat) for e in self.shared_experts)
out = out + shared_out
```

The `1` shared expert is **always active** — every token passes through it in
addition to its top-2 routed experts. Its role is to carry the
"all-tokens-always-need-this" knowledge (syntactic primitives, common
functionality) so the routed experts can specialise without each having to
redundantly relearn the common case. DeepSeek-v3 introduced this; GPT-OSS
adopts it. Because it is dense, its parameters count toward the *active* total,
not the sparsity.

---

## 6. Vectorized dispatch

```python
def _dispatch_vectorized(self, flat, indices, weights) -> torch.Tensor
```

The fast path. Given `N = B·T` tokens each assigned to `k = 2` experts, the
algorithm groups tokens by expert so each expert's matmul is one batched call:

### 6.1 Algorithm

```
1. Flatten (token, expert, weight) triples into (N·k,) vectors.
2. argsort by expert id (stable) → contiguous per-expert runs.
3. bincount → per-expert counts; cumsum → per-expert offsets.
4. For each expert e (E = 8):
       chunk_tokens  = sorted_token_ids[offset_e : offset_e + count_e]
       expert_in     = flat[chunk_tokens]
       gate          = F.linear(expert_in, W1_stack[e])     # SwiGLU
       up            = F.linear(expert_in, W3_stack[e])
       expert_out    = F.linear(silu(gate) * up, W2_stack[e])
       out.index_add_(0, chunk_tokens, expert_out * chunk_weights)
```

### 6.2 Why this is fast

- **One `torch.argsort`** over `(N·k,)` entries — the only `O(N·k log(N·k))`
  step, and it is a single kernel.
- **`E = 8` expert matmul launches per layer**, not `N·k` — the per-expert loop
  is Python but it iterates 8 times, not thousands.
- **`F.linear(...)` bypasses `nn.Module.__call__`** — the stacked weights (see
  §7) let each expert's three matmuls go straight to ATEN, skipping the
  `nn.Linear` forward / autograd-hook machinery.
- **`index_add_`** scatters the weighted expert outputs back to their source
  token positions in one fused kernel.

### 6.3 Determinism

`torch.argsort(flat_idx, stable=True)` is load-bearing for reproducibility:
with `stable=True` the same input *always* produces the same permutation, so
two `--seed 42` runs route identically. Without it, ties in `flat_idx` could
break arbitrarily across runs and break bit-exact resume.

---

## 7. The stacked-expert weight cache

```python
def _ensure_stacked(self):
    version = sum(e.w1.weight._version for e in self.experts)
    if self._stacked_cache is not None and self._stacked_version == version:
        return self._stacked_cache
    W1 = torch.stack([e.w1.weight for e in self.experts], dim=0)   # (E, F, D)
    W2 = torch.stack([e.w2.weight for e in self.experts], dim=0)
    W3 = torch.stack([e.w3.weight for e in self.experts], dim=0)
    self._stacked_cache = (W1, W2, W3)
    self._stacked_version = version
    return self._stacked_cache
```

The vectorized dispatch wants each expert's weights as a single stack so it
can index `W1_stack[e]`. Building that stack on every forward would cost
`3 · E` `torch.stack` allocations. Instead the stack is built **once** and
cached, with cache invalidation keyed on the sum of each parameter's
`_version` counter.

`tensor._version` is PyTorch's in-place-write counter: every in-place op
(including the optimiser step) bumps it. So `sum(e.w1.weight._version)` changes
exactly when the weights change — the cache is invalidated automatically on
every optimiser step and rebuilt on the next forward, with zero manual
plumbing. This is the same mechanism autograd uses to detect in-place bugs.

Memory cost: `E · 3 · F · D · 2 bytes = 8 · 3 · 1536 · 768 · 2 ≈ 57 MB` at
production scale — negligible on an A100 80GB.

---

## 8. Forward flow and parameter accounting

```
x (B,T,d_model)
  │ router(x.view(-1,D)) → indices, weights, all_logits
  │ _dispatch_vectorized(flat, indices, weights)   → routed_out (B·T, D)
  │ aux_load_balancing_loss(all_logits, ...)        → aux_loss scalar
  │ shared_experts(flat)                            → shared_out
  │ out = routed_out + shared_out
  ▼
  (out.view(B,T,D), aux_loss)
```

### Active-parameter accounting (`GPTOSS.num_active_parameters`)

```
non_moe_params                              # embedding, attn, norms, head, router(ish)
  + n_layers · (n_activated + n_shared) · (3 · d_model · ffn_dim)   # active experts
  + n_layers · (d_model · n_routed_experts)                         # router
```

With `n_activated=2`, `n_shared=1`, `n_routed=8`, `ffn_dim=1536`,
`d_model=768`, `n_layers=12`: each layer activates `3` experts' worth of FFN
parameters (2 routed + 1 shared) instead of all `8` — the source of the 50.8%
sparsity. The full `8` routed experts still count toward *total* parameters
(they all live in memory and receive gradients); only their *forward* cost is
sparse.

---

## 9. Design rationale & rejected alternatives

| Decision | Rationale | Rejected alternative |
|---|---|---|
| Standard aux loss (α=0.01) | Simple, stable at 8 experts; educational contrast | Aux-loss-free bias — DeepSeek-v3's choice, explicitly *not* used here |
| Top-2 of 8 | GPT-OSS granularity; coarse routing is easier to balance | Top-4 of 20 (DeepSeek) — more experts, finer routing, more memory |
| Grouped/vectorized dispatch | CPU-friendly, simple to read, `stable argsort`-reproducible | Stacked `bmm` over a padded `(E, max_tokens, D)` tensor — wastes memory on padding |
| Stacked-weight cache via `_version` | Automatic invalidation on every optimiser step | Manual `clear_cache()` call — easy to forget, causes stale-weight bugs |
| FP32 softmax in router & aux loss | BF16 underflows small per-expert probabilities | BF16 softmax — silently zeroes the loss when the router saturates |
| 1 shared expert | Common-knowledge carrier, lets routed experts specialise | 0 shared — each routed expert must relearn primitives; 2 shared — too dense |

---

## 10. Edge cases & pitfalls

- **Empty expert**: if no token routes to expert `e` in a step (`count_e == 0`),
  the dispatch loop `continue`s past it — no matmul, no allocation. This is
  the normal early-training state before the aux loss balances routing.
- **All-equal logits**: the router's renormalisation clamps the denominator at
  `1e-6`; the aux loss's `bincount` handles a uniform selection fine.
- **`stable=True` is mandatory, not cosmetic**: drop it and two seeded runs
  can disagree on tie-breaking, breaking `--resume-from` bit-exactness.
- **Do not "optimise" the aux loss away**. Replacing it with the aux-loss-free
  bias trick would make this repo a duplicate of DeepSeek-v3-Lite and loses the
  deliberate architectural distinction. `AGENTS.md` calls this out explicitly.

---

## Implementation notes (extracted from code review)

- **FP32 softmax in the aux loss**: `aux_load_balancing_loss` upcasts the
  router logits to FP32 before softmax and mean. Under BF16, small
  per-expert probabilities can underflow to zero, silently zeroing out the
  loss when the router saturates early in training. The FP32 path keeps the
  loss numerically meaningful; the result is cast back to the input dtype
  before returning.
- **Stable argsort for reproducible dispatch**: both `_dispatch_vectorized`
  and `_dispatch_grouped` call `torch.argsort(flat_idx, stable=True)`. With
  `stable=True`, the same input always produces the same permutation across
  runs (required for bit-exact reproducibility under `--seed`).
- **Cached `(W1, W2, W3)` stacks via `F.linear`**: `MoELayer._ensure_stacked`
  builds per-expert weight stacks once and caches them keyed on
  `sum(e.w1.weight._version)` so the cache is invalidated automatically on
  every optimizer step (in-place writes bump `_version`). The dispatch loop
  then calls `F.linear(expert_in, W1_stack[e])` instead of
  `self.experts[e](expert_in)`, bypassing the `nn.Module.__call__` Python
  overhead. Bit-exact equivalent of the grouped path; verified by
  `test_moe_dispatch_correct` and `test_moe_dispatch_is_deterministic`.
- **Standard aux loss vs aux-loss-free**: GPT-OSS-Lite deliberately uses the
  Switch-Transformer / GShard standard auxiliary load-balancing loss
  (α=0.01), NOT the aux-loss-free bias trick. This is an intentional
  distinction from DeepSeek-v3-Lite; do not "optimise" it away.