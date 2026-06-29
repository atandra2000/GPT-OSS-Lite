# MoE FFN for GPT-OSS-Lite

## Overview
Top-2 of 8 routed experts + 1 shared expert.

Deliberately distinct from DeepSeek-v3-Lite's MoE:
- Standard auxiliary load-balancing loss (not the aux-loss-free bias trick).
- Top-2 of 8 (vs DeepSeek's top-4 of 20) — coarser routing granularity.
- Grouped dispatch (vs DeepSeek's stacked bmm) — simpler & CPU-friendly.
- 1 shared expert (same as DeepSeek) — always active, contributes to every token.

## Performance
This implementation supports two dispatch modes:
- **grouped** (default): per-expert Python loop, deterministic, very simple.
- **vectorized**: one fused bmm over all (E) experts — much faster on GPU and on CPU for batched inputs, but requires the experts to expose their weights as stacked parameters. The fast path is selected automatically when the expert parameters are already stacked.

## Router
Top-k gating network: linear projection → softmax → top-k.

Returns:
- `indices`: `(B*T, k)` — selected expert indices per token.
- `weights`: `(B*T, k)` — softmax-normalized routing weights.
- `all_logits`: `(B*T, n_experts)` — full logits (for aux loss).

## Auxiliary Load-Balancing Loss
Standard MoE load-balancing loss (Switch Transformer / GShard style).
Encourages uniform routing across experts. Penalises both (a) imbalance in how often each expert is selected and (b) imbalance in the average routing probability per expert.

`L = n_experts * sum_i(f_i * P_i)`
where `f_i` = fraction of tokens routed to expert i (top-k selection), and `P_i` = mean probability of expert i across all tokens.

**Implementation notes:**
- Computes softmax + mean in **float32** to avoid BF16 underflow on small probabilities.
- Uses `bincount` for the per-expert fraction (vectorised).

## Vectorized Dispatch
Algorithm:
1. Sort (token_id, expert_id, weight) triples by expert_id (stable).
2. Compute per-expert token ranges via cumsum.
3. For each expert, gather its tokens, run bmm with its weights, and scatter the results back.

Why this is fast:
- One `torch.argsort` over (N*k,) entries.
- Per-expert matmul launched in Python, but only `E = 8` launches per layer.
- Single fused bmm over `(E, F, D)` weights.

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
