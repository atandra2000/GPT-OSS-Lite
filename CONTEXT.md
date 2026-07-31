# GPT-OSS-Lite — Working Context

> Build target: `LLM/GPT-OSS-Lite/`
> Status snapshot: 2026-07-31.

## Scoping note

**Attention documentation:** [`documentation/ATTENTION_SINKS.md`](documentation/ATTENTION_SINKS.md) is the single canonical reference for sink bias, sliding-window alternation, and YaRN. For routine questions, derive answers from `models/attention.py` first; read the doc for the full deep-dive.

### Doc routing (agents)

| Question type | Read first |
|---|---|
| How does this repo implement X? | `models/*.py` + matching component doc §Implementation |
| Sink bias / SWA / YaRN theory | `documentation/ATTENTION_SINKS.md` |
| YAML / hyperparameters | `documentation/configs.md` |
| Train loop / NaN / checkpoints | `documentation/training.md` + `tests/test_training.py` |
| What must not break? | `AGENTS.md` hard rules + `documentation/architecture.md` invariants |
| Onboarding | `documentation/getting_started.md` |

Everything in this file is derived from code/configs/tests.

## Project snapshot

| Field | Value |
|---|---|
| Repo | GPT-OSS-Lite (single-A100 80GB faithful GPT-OSS reproduction) |
| Scale | ~502M total / ~247M active, 8.0B Chinchilla-optimal tokens |
| Wall budget | 16–20 h A100 80GB, 35–40% MFU target |
| Stack | PyTorch 2.x, BF16, SDPA/FA2, `torch.compile(max-autotune)`, safetensors ckpt |
| Vocab | 128,000 (LLaMA-3 BPE) |
| Topology | d=768, n_layers=12 (6 SWA + 6 full), 8 Q / 4 KV heads |
| Attention | Learned sink bias, YaRN (θ=100K, scale=32, target=128K), pruned RoPE on global layers |
| MoE | 8 routed (top-2), 1 shared, ffn=1536, standard aux loss α=0.01 |
| Steps | 61,000 total, warmup=3000, cosine to 5% |
| Tests | 187 across 11 files (CPU-friendly) |

## Hard rules (never violate)

1. Preserve sliding-window / full-attention alternation — breaks headline KV metric.
2. Use standard aux load-balancing loss — not aux-loss-free (DeepSeek distinction).
3. Read `ATTENTION_SINKS.md` before answering sink-bias questions.
4. Never disable NaN guard without explicit user consent.
5. Never suggest adding MLA, GDN, or MTP.
6. Triton opt-in is explicit via `moe_dispatch="triton_grouped"` — no silent fallback.

## Directory map

```
AGENTS.md, SKILLS.md, README.md, CONTEXT.md, documentation/
configs/pretrain_a100_502m.yaml
models/
  transformer.py    # GPTOSS, ModelConfig, GPTOSSBlock, RMSNorm
  attention.py      # SlidingWindowAttention, FullAttention, sink bias
  moe.py            # MoELayer, aux loss, grouped dispatch
  moe_triton.py     # sanctioned fused W1/W3+silu kernel (opt-in)
  yarn.py, rotary.py
training/pretrain.py
inference/generate.py, long_context.py
utils/checkpoint.py, logging.py, memory.py
scripts/            # microbench, step_time, e2e_gpu_smoke, kv_cache_benchmark, passkey_eval, check_docs
tests/              # 187 tests, 11 files
```

## Headline metrics

| Metric | Target | Verified by |
|---|---|---|
| KV-cache reduction at 128K | ≥ 1.8× (measured ~2.0×) | `scripts/kv_cache_benchmark.py` |
| Passkey retrieval at 128K | ≥ 85% | `scripts/passkey_eval.py` (needs trained ckpt) |
