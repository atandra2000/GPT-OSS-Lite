# Attention Mechanisms in GPT-OSS-Lite

## Overview
Sliding-window + full attention alternation with learned attention-sink bias.

The load-bearing component of GPT-OSS-Lite. Two distinct attention paths:

1. `manual_causal_attention` — O(T²) reference implementation (ground truth for tests; supports sink bias and optional sliding-window mask).
2. `sliding_window_attention` — production path via SDPA, with sliding-window mask, learned sink bias, and YaRN RoPE.
3. `GPTOSSAttention` — `nn.Module` wrapping both paths with GQA, sink bias, and the alternating layer pattern (even layers = SWA, odd = full).

## Manual Causal Attention
Naive O(T²) causal attention — the reference implementation.

**Notes:**
The sink bias implements the off-by-one / StreamingLLM trick: the softmax denominator becomes `Z = exp(sink_bias) + sum(exp(scores))`. Equivalently, we add an extra "virtual" key with logit `sink_bias` to the attention logits before softmax.

All softmax computations are done in FP32 internally for numerical stability under BF16 inputs (the BF16 cast only happens at the end).

## Sliding Window Attention
Sliding-window causal attention via SDPA with optional sink bias. Supports both self-attention (T_q == T_k) and cross-attention (T_q < T_k, as during cached decode).

**Implementation:**
For the no-sink case, we use the efficient `is_causal` path with a precomputed sliding-window mask applied via `attn_mask`.

For the sink case, we extend the K/V sequence with a virtual "sink key" whose logit (along the key axis) is the per-head sink bias. SDPA softmax then naturally incorporates `exp(sink_bias)` into the denominator.

## Full Causal Attention
Full causal attention via SDPA with optional sink bias. Supports both self-attention and cross-attention.

## GQA (Grouped Query Attention) Helper
`repeat_kv` repeats KV heads to match the number of query heads.

**Notes:**
The previous implementation used `expand + reshape + contiguous()` which allocates a fresh contiguous tensor on every forward. For SDPA, the input doesn't need to be contiguous — SDPA's flash path operates on strided tensors via `.contiguous()` internally as needed. We use `expand + reshape` here, which gives SDPA a view it can fuse with the underlying matmul.

However, PyTorch's reshape after expand will materialise the data. For correctness when callers need to compare (e.g. tests), we keep `.contiguous()` only in the no-sink case (where SDPA's internal copy dominates anyway), and drop it in the cached decode path where SDPA accepts non-contiguous input.

## GPTOSSAttention (Layer Module)
GPT-OSS attention layer: GQA + YaRN RoPE + learned sink bias + alternating SWA/full.

**Layer pattern:**
- Even layers (0, 2, 4, ...) → sliding-window attention (window=128).
- Odd layers (1, 3, 5, ...) → full causal attention.
- Global (full-attention) layers additionally apply pruned RoPE.

---

## Implementation notes (extracted from code review)

- **Sink bias clamping at forward time**: `GPTOSSAttention.forward` clamps
  `sink_bias` to `[-10, 15]` on a detached copy before passing it into the
  attention path. The clamp bounds are chosen so `exp(15) ≈ 3.3e6` stays
  within BF16 range and `exp(-10) ≈ 4.5e-5` acts as a "disabled" sink. The
  unclamped `nn.Parameter` retains its gradient — the clamp is a forward-only
  numerical-stability guard against BF16 SDPA mask-add overflow when the
  trained parameter grows large. See `ATTENTION_SINKS.md` for the full
  theoretical treatment.
- **Mask caching**: the sliding-window causal mask is cached module-level in
  `_SLIDING_WINDOW_MASK_CACHE`, keyed by `(T, window, device, dtype)`. The
  sink-bias SWA mask is cached under the key `("sink_swa", T_q, T_k, H,
  window, device, dtype)` — the trailing sink-bias column is overwritten
  per-call (it varies with training) but the rest of the mask is reused.
  This amortises the per-forward mask build to one kernel launch.
- **FP32 accumulation in the manual path**: `manual_causal_attention` casts
  `query_states` and `key_states` to FP32 for the score matmul and runs
  softmax in FP32, casting back to `value_states.dtype` only for the final
  value matmul. This is the reference path used in tests; the SDPA path
  handles FP32 accumulation internally.
- **`repeat_kv` uses expand + reshape (no `.contiguous()`)**: SDPA's flash
  path operates on strided tensors and calls `.contiguous()` internally only
  when needed. Dropping the explicit `.contiguous()` avoids a fresh
  `(B, H_kv × n_rep, T, D)` allocation on every forward.
- **`MixedKVCache` ring + exponential growth**: windowed layers use a
  fixed-size ring buffer of length `window` (O(window) append, O(1) amortised
  decode step). Global layers use an exponentially-growing buffer
  (1.5× growth capped at `_GLOBAL_CAP_TOKENS = 4_000_000`), giving O(N)
  total work over an N-token generation instead of O(N²) from `torch.cat`
  on every step.
- **Pre-allocated `generate` output**: `generate()` allocates the full
  `(B, T_prompt + max_new_tokens)` output tensor once and writes new tokens
  in place, avoiding the O(T²) `torch.cat` per step.
- **Fast T=1 RoPE path**: `YaRNRoPE.forward` special-cases
  `positions.numel() == 1` to skip `torch.outer` and do a single scalar
  multiply — saves a kernel launch per decode step across all 12 layers.
