# GPT-OSS-Lite Documentation

Index of the design / implementation docs for GPT-OSS-Lite. Each file
captures the rationale that was extracted from the codebase during the
cleanup pass — the code itself is kept clean and the "why" lives here.

## Architecture & components

- [`attention.md`](attention.md) — Sliding-window + full attention
  alternation, learned sink bias, mask caching, FP32 accumulation,
  `repeat_kv` expand-not-contiguous, `MixedKVCache` ring/exponential
  growth, pre-allocated `generate` output, fast T=1 RoPE path.
- [`moe.md`](moe.md) — Top-2 of 8 routed + 1 shared expert, standard
  aux load-balancing loss (NOT aux-loss-free), FP32 softmax aux loss,
  stable argsort, cached `(W1, W2, W3)` stacks via `F.linear`.
- [`rotary.md`](rotary.md) — `apply_rope`, `compute_yarn_freqs`,
  `compute_yarn_mscale`, `prune_rope` (pruned RoPE on global layers).
- [`yarn.md`](yarn.md) — YaRN RoPE scaling: degenerate-ramp
  `UserWarning`, pruned-RoPE on global layers, rotated-K caching for
  O(T) decode, fast T=1 path.
- [`training.md`](training.md) — BF16 + `torch.compile(max-autotune)`
  + TF32 + FP32 AdamW master + grad-ckpt every 3rd layer + NaN guard
  rollback + chunked CE chunk=4096 + foreach/fused AdamW +
  `CUBLAS_WORKSPACE_CONFIG` + RNG seeding.
- [`inference.md`](inference.md) — `MixedKVCache` ring + exponential
  growth, rotated-K caching, pre-allocated output, long-context
  passkey eval.
- [`data_pipeline.md`](data_pipeline.md) — 4-stage pipeline (download
  → clean → tokenize → pack), mmap zero-copy slices,
  num_workers/pin_memory/persistent_workers, uint32 storage.
- [`utils.md`](utils.md) — CheckpointManager (atomic safetensors,
  RNG state sibling file), distributed, logging, memory estimator.
- [`OPTIMIZATIONS.md`](OPTIMIZATIONS.md) — Audit of every performance
  optimisation applied (problem, fix, impact, risk, test coverage).

## Authoritative top-level references

- [`../ATTENTION_SINKS.md`](../ATTENTION_SINKS.md) — 600-line
  technical deep-dive on the learned attention-sink bias. **This is
  the authoritative reference for sink-bias questions**; the
  `attention.md` notes above supplement it with implementation
  details but do not duplicate it.
- [`../AGENTS.md`](../AGENTS.md) — Subagent definition and project
  rules (Numerical-stability / Reproducibility / Performance rules).
- [`../SKILLS.md`](../SKILLS.md) — Project-local skill workflows
  (smoke tests, KV-cache benchmark, YaRN debug, pretraining,
  passkey eval, reproducible runs, profiling).