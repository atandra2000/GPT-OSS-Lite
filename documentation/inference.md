# Inference — GPT-OSS-Lite

> **Source:** `inference/generate.py`, `inference/long_context.py`

## Overview

Token-by-token generation with a mixed-shape KV cache that delivers the
2× VRAM savings at long context, plus 128K passkey retrieval evaluation.

## `MixedKVCache`

Per-layer mixed KV cache storing **rotated** K (RoPE is applied at
insertion time, not at attention time). Two storage strategies:

- **Windowed layers (even idx)**: a fixed-size ring buffer of length
  `window`. The buffer is allocated once at first append; subsequent
  appends are O(window) memcpy (in-place `cat + slice` of the last
  `T_new` slots), not O(T). When the buffer is full, `get()` returns the
  keys in temporal order (head → end → 0 → head) via a single `torch.cat`.
- **Global layers (odd idx)**: an exponentially-growing contiguous
  buffer. Grows by 1.5× on demand up to a per-layer cap
  (`_GLOBAL_CAP_TOKENS = 4_000_000`). New keys are copied in-place into
  the buffer. Total work for an N-token generation is O(N) instead of
  O(N²) from `torch.cat` on every step.

`get()` returns views (zero-copy slices) of the underlying buffer;
attention is free to operate on them without an extra copy.

## `generate()`

- **Pre-allocated output**: the `(B, T_prompt + max_new_tokens)` output
  tensor is allocated once and each new token is written in place
  (`output[:, T_prompt + step] = next_id`). This avoids the O(T²)
  `torch.cat` per step that the original implementation had.
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

## `long_context.PasskeyEvaluator`

128K passkey retrieval evaluation. A 5-digit passkey is inserted at a
configurable position (start/middle/end) in a long filler-text context,
and the model is prompted to retrieve it. Reproducibility:

- Each context-length uses a distinct RNG seeded by `base_seed + ctx_len`
  so different lengths are statistically independent.
- Passkeys are drawn **without replacement** within a context-length's
  trial set (no duplicates).
- The filler corpus is seeded by `context_length` (not by trial), so
  each context length has its own deterministic filler; only the
  passkey is per-trial randomness.

Expected behaviour: ≥ 95% retrieval at 4K (training context), ≥ 85% at
128K (YaRN extrapolation target). `scripts/passkey_eval.py` is the CLI
entrypoint; it falls back to a stub when no trained checkpoint is
provided (only tests prompt construction).