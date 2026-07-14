# AGENTS.md — GPT-OSS-Lite

> Read root `AGENTS.md` and `self.md` first. Workspace rules are
> authoritative; this file adds project-specific rules only.

> **Project:** `LLM/GPT-OSS-Lite/` · **Type:** faithful GPT-OSS reproduction
> **Scale:** ~502M total / ~247M active · 8.0B tokens planned · 16–20h on A100 80GB
> **Stack:** PyTorch 2.x, BF16, `torch.compile(max-autotune)`, FA2 via SDPA
> **Architecture detail:** see `README.md §7`; cross-architecture explainer
> at `.agents/skills/llm-architecture/SKILL.md §2, §5`; **authoritative
> sink-bias deep-dive at `documentation/ATTENTION_SINKS.md`.**

## 1. Subagent: `gptoss-long-context-engineer`

**Triggers:** "Explain the sliding-window/full alternation", "How does the
learned sink bias work?", "Debug YaRN extrapolation at 128K", "Why is my
MoE routing collapsing to one expert?", "Tune window_size for KV cache."

**Knows cold:**
- **12 layers**: 6 sliding-window (window=128) + 6 full-attention,
  **alternating**. GQA (8 Q / 4 KV, head_dim=96). vocab 128,000, d_model
  768. **Learned attention-sink bias per head.** **YaRN RoPE** (θ=100K,
  scale=32, target=128K). **Pruned RoPE on global layers (25% of dims).**
  Top-2 of 8 routed experts + 1 shared (SwiGLU, ffn=1536). **Standard aux
  load-balancing loss (α=0.01)** — deliberately distinct from
  DeepSeek-v3-Lite's `AuxLossFreeGate`.
- `models/attention.py` — `SlidingWindowAttention` + `FullAttention` +
  learned sink bias.
- `models/moe.py` — top-2 of 8 routed + 1 shared, standard aux loss, grouped
  dispatch.
- `models/yarn.py` — YaRN RoPE scaling + pruned RoPE.
- Training: BF16 + `torch.compile(max-autotune)` + TF32 + FA2 via SDPA, FP32
  AdamW master weights + gradient checkpointing (every 3rd layer), NaN guard
  with rollback, aux load-balancing loss (α=0.01), chunked cross-entropy
  (chunk=4096).
- Inference: `MixedKVCache` (windowed = ring buffer, global = exponential
  growth → decode is O(1) per step instead of O(T)). `inference/long_context.py`
  runs 128K passkey retrieval eval.

## 2. Hard rules

1. **Always** preserve the sliding-window / full-attention alternation —
   replacing it with pure full-attention breaks the headline metric
   (≥ 1.8× KV-cache reduction at 128K).
2. **Always** read `documentation/ATTENTION_SINKS.md` before answering sink-bias questions.
3. **Always** verify `test_sliding_window_matches_full` passes after any
   change to `models/attention.py` (`pytest tests/test_attention.py -v`).
4. **Always** use the standard aux load-balancing loss (not the aux-loss-free
   bias trick) — this is a deliberate distinction from DeepSeek-v3-Lite.
5. **Never** disable the NaN guard without explicit user consent.
6. **Never** suggest adding MLA, GDN, or MTP — this is a GPT-OSS repo
   (avoids FusionLLM / DeepSeek-v3-Lite overlap).

## 3. Numerical-stability rules

- **Sink bias clamped** to `[-10, 15]` at forward time — prevents BF16 SDPA
  mask-add overflow when the trained parameter grows large. Unclamped
  parameter retains gradient flow.
- **Aux loss uses FP32 softmax** internally to avoid BF16 underflow when the
  router saturates.
- **Manual attention uses FP32 accumulation** for the score matmul.
- **YaRN degenerate ramp** emits a `UserWarning` (not silent identity).

## 4. Reproducibility rules

- All RNGs (torch / cuda / numpy / python random) seeded when `--seed` is
  passed. Without `--seed`, runs are NOT reproducible.
- `torch.argsort` uses `stable=True` in MoE dispatch.
- Checkpoints include RNG state in `rng_step_N.pt`.
- `CUBLAS_WORKSPACE_CONFIG = :4096:8`.

## 5. Performance rules (applied in code)

- `torch.compile(max-autotune)` auto-invoked on CUDA when
  `training.compile: true` in YAML.
- TF32 + cuDNN benchmark + `set_float32_matmul_precision("high")`.
- Gradient checkpointing applies `torch.utils.checkpoint.checkpoint` per
  `grad_ckpt_every` layers.
- DataLoader: `num_workers=4`, `pin_memory=True`, `persistent_workers=True`.
- Sharded dataset: `mmap=True` → zero-copy `__getitem__`.
- AdamW: `foreach=True, fused=True` (1.5–2× faster than default loop on
  A100/H100).
- Sliding-window attention mask cached by `(T, window, device, dtype)`.
- `repeat_kv` uses `expand + reshape` (no `.contiguous()`) — SDPA's flash
  path handles non-contiguous K/V internally.
- `RMSNorm` keeps activations in native dtype (no FP32 copy).

## 6. Known caveats

- Full 8B-token pretraining run not yet started (no GPU on dev machine).
- `passkey_eval.py` requires a trained checkpoint; runs as a stub on
  untrained models.
- `channels_last` is **not** applied to this code (LLM matmuls don't
  benefit enough). Mandatory on Blackwell only applies to Blackwell-targeted
  projects like StableDiffusion; see root AGENTS.md §1.9.
