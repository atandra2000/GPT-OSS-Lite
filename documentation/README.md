# GPT-OSS-Lite Documentation

Index of the design / implementation docs for GPT-OSS-Lite. The code itself is
kept clean and readable; the *why* — the math, the design rationale, the
rejected alternatives, the edge cases — lives here. Each file is a
self-contained deep-dive on one subsystem, cross-linked to its neighbours.

## How to read these docs

- **New to the codebase?** Start with [`attention.md`](attention.md) and
  [`moe.md`](moe.md) — they cover the two architectural pillars (sliding-window
  attention and the MoE FFN). Then [`yarn.md`](yarn.md) for the long-context
  story, [`training.md`](training.md) for the loop, and
  [`inference.md`](inference.md) for how the architecture pays off at decode
  time.
- **Debugging a numerical issue?** Go straight to
  [`training.md`](training.md) §3 (stability knobs) and §5 (NaN guard), then
  [`attention.md`](attention.md) §7.3 (sink-bias clamp).
- **Tuning for a different GPU?** [`utils.md`](utils.md) §5 (memory
  estimator) and [`training.md`](training.md) §3 (perf knobs).
- **Understanding the headline metrics?** [`inference.md`](inference.md)
  (2× KV cache) and [`yarn.md`](yarn.md) (128K passkey extrapolation).

---

## Architecture & components

- [`ATTENTION_SINKS.md`](ATTENTION_SINKS.md) — **★ the authoritative sink-bias
  deep-dive.** The 600-line theoretical treatment of the learned attention-sink
  bias (the off-by-one / StreamingLLM trick): why it exists, why GPT-OSS makes it
  *learned*, the BF16 numerical-stability clamps, and the relationship to
  streaming long-context attention. **This is the authoritative reference for
  sink-bias questions**; [`attention.md`](attention.md) supplements it with
  implementation details but does not duplicate the theory.
- [`attention.md`](attention.md) — **The load-bearing component.** Sliding-window
  + full attention alternation, the learned sink bias (off-by-one /
  StreamingLLM trick), GQA, mask caching, FP32 accumulation in the reference
  path, `repeat_kv` expand-not-contiguous, the forward-time sink-bias clamp,
  and the alternating even-SWA / odd-full layer pattern. Includes the math for
  each attention variant and the rationale for every rejected alternative.
- [`moe.md`](moe.md) — **The second pillar.** Top-2 of 8 routed + 1 shared
  SwiGLU expert, the standard Switch/GShard auxiliary load-balancing loss
  (deliberately *not* the aux-loss-free bias trick — a contrast with
  DeepSeek-v3-Lite), the router's FP32 softmax, stable-argsort reproducible
  dispatch, and the cached `(W1, W2, W3)` weight stacks invalidated via
  `tensor._version`. Includes the aux-loss derivation and active-parameter
  accounting.
- [`rotary.md`](rotary.md) — RoPE fundamentals: why rotation in the complex
  plane encodes *relative* position for free, the half-dim convention, the
  fused two-op `apply_rope` implementation, the YaRN frequency table
  (`compute_yarn_freqs`), the `mscale` temperature correction, and `prune_rope`
  (pruned RoPE on global layers).
- [`yarn.md`](yarn.md) — **YaRN length extrapolation.** How a 4K-trained model
  reaches 128K: the per-dimension frequency ramp (interpolate low frequencies,
  leave high frequencies alone), the mscale softmax correction, pruned RoPE on
  global layers, the degenerate-ramp `UserWarning`, rotated-K caching for O(T)
  decode, and the fast T=1 path. Explains the *training-time* YaRN choice
  (true extrapolation, not decode-only).
- [`training.md`](training.md) — **The training loop.** BF16 autocast (no
  `GradScaler`), `torch.compile(max-autotune)`, TF32 + cuDNN knobs, FP32 AdamW
  master weights, gradient checkpointing every 3rd layer, the NaN-guard state
  machine with checkpoint rollback, chunked cross-entropy (chunk=4096),
  `foreach`/`fused` AdamW, `CUBLAS_WORKSPACE_CONFIG`, the warmup→cosine LR
  schedule, and the full RNG-chain reproducibility story.
- [`inference.md`](inference.md) — **The decode path.** `MixedKVCache` (ring
  buffer for windowed layers, exponential-growth buffer for global layers),
  rotated-K caching (O(T) decode not O(T²)), pre-allocated output, per-call
  sink-bias clamp cache, the `use_cache=False` correctness replay path, and
  the 128K `PasskeyEvaluator` (the second headline metric) with its
  reproducibility design.
- [`data_pipeline.md`](data_pipeline.md) — **The 8 B-token pipeline.** Four
  stages (download → clean + dedup → tokenize → pack shards), SHA-256
  hash-sharded constant-memory dedup, EOS-separated token streams, atomic
  shard writes, round-robin packing, the manifest schema, mmap zero-copy
  slices, and `uint32` storage. Includes the five-source mixture table and
  validation rules.
- [`utils.md`](utils.md) — **Infrastructure.** `CheckpointManager` (atomic
  safetensors with shared-tensor dedup for weight tying, `.tmp → os.replace`,
  complete-checkpoint step discovery), the RNG sibling file, the one-line
  `distributed` device helper, the sync-free rolling-window `TrainingLogger`
  with optional WandB, and the two-regime `memory` estimator + pre-flight
  VRAM check.
- [`OPTIMIZATIONS.md`](OPTIMIZATIONS.md) — **The perf audit.** Every
  optimisation applied (problem → fix → impact → risk → test coverage), with
  a TL;DR results table and CPU vs A100 projected gains. The single source of
  truth for "why is this line of code written this way."

---

## Authoritative top-level references

- [`../AGENTS.md`](../AGENTS.md) — subagent definitions and project rules
  (numerical-stability / reproducibility / performance; the "do not replace the
  standard aux loss" rule; the "no MLA/GDN/MTP" rule).
- [`../SKILLS.md`](../SKILLS.md) — project-local skill workflows (smoke tests,
  KV-cache benchmark, YaRN debug, pretraining, passkey eval, reproducible runs,
  profiling).
- [`../README.md`](../README.md) — the public project summary: headline metrics,
  quick start, results tables, design-decision table, project structure.