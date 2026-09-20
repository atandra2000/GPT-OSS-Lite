# Learning Paths — How to Read the GPT-OSS-Lite Docs

> Audience: all levels. This guide is the navigational meta layer for the
> documentation tree — it teaches no model topic itself; it tells you which
> doc to read next depending on what you already know and what you want
> from the codebase. The linear 13-section walkthrough in
> [getting-started.md](getting-started.md) is the alternative presentation
> of the same corpus; the three paths below assume prior steps and end with
> a working mental model of the whole stack.

The docs are organized into three tracks: **concepts**
([concepts/](../concepts/foundations-and-architecture.md), theory and
implementation consolidated per topic), **references**
([references/config-and-api.md](../references/config-and-api.md), config
tables and API signatures), and **guides** (this folder — onboarding and
operations), plus the two pipeline docs [training.md](../training.md) and
[inference.md](../inference.md). Every citation to code uses symbol anchors
(`models/attention.py:GPTOSSAttention` style), verified by
`tests/test_doc_refs.py`.

---

## Beginner path — What this model is and how it works

Who this is for: you know basic PyTorch and what next-token prediction is,
and want to understand the GPT-OSS long-context architecture from the
ground up. No prior sliding-window-attention or MoE background required.

| Step | Doc | What you will know after |
|------|-----|--------------------------|
| 1 | [getting-started.md](getting-started.md) (§1–§2) | What GPT-OSS-Lite is, the headline 1.94×–2.0× KV-cache number, and the ~502M total / ~247M active parameter split. |
| 2 | [foundations-and-architecture.md](../concepts/foundations-and-architecture.md) | Why decoder-only, the full 12-layer topology, the GQA 8Q/4KV head layout, and the config→code routing through `models/transformer.py:ModelConfig` and `models/transformer.py:GPTOSS`. |
| 3 | [attention-sinks.md](../concepts/attention-sinks.md) | The alternating sliding-window/full design (`models/attention.py:GPTOSSAttention`), why even layers keep only 128 tokens, and the learned sink bias with its clamp. |
| 4 | [attention-and-positional.md](../concepts/attention-and-positional.md) | Attention math end-to-end: QK^T scaling, masking, and the RoPE → YaRN path that stretches a 4K training window to 128K. |
| 5 | [moe.md](../concepts/moe.md) | Top-2-of-8 routing (`models/moe.py:MoERouter`), the auxiliary load-balancing loss (`models/moe.py:aux_load_balancing_loss`), and the shared expert. |
| 6 | [inference.md](../inference.md) (KV-cache chapters) | How `inference/generate.py:MixedKVCache` keeps ring buffers on windowed layers and full prefixes on global layers — the mechanism behind the headline metric. |
| 7 | [config-and-api.md](../references/config-and-api.md) + `models/transformer.py:GPTOSS` | The code tour: every block with its shape contract. |

## Intermediate path — Train it and understand the numerics

Who this is for: you have read the beginner path and want to run, tune, or
resume training — including the long-context augmentation and the A100
runbook.

| Step | Doc | What you will know after |
|------|-----|--------------------------|
| 1 | [foundations-and-architecture.md](../concepts/foundations-and-architecture.md) (config sections) | Every `models/transformer.py:ModelConfig` field's meaning and where it is consumed. |
| 2 | [config-and-api.md](../references/config-and-api.md) | The full YAML schema for `configs/pretrain_a100_502m.yaml`: warmup 3000, lr 4.0e-4, aux loss α=0.01, and why. |
| 3 | [training.md](../training.md) | The applied pretrain loop: AdamW, gradient accumulation, warmup→cosine decay (`training/pretrain.py:make_warmup_cosine_lambda`), NaN guard with rollback, chunked cross-entropy (`training/pretrain.py:chunked_cross_entropy`). |
| 4 | [getting-started.md](getting-started.md) (§6) | Preparing the corpus via the `data/prepare_data.py` shim into the workspace `shared_data/` pipeline. |
| 5 | [tokenization.md](../concepts/tokenization.md) | BPE algorithm, 128K vocab economics, and byte-fallback behavior. |
| 6 | [optimizers-and-numerics.md](../concepts/optimizers-and-numerics.md) | BF16/FP16/TF32 trade-offs, sampling, and why the sink clamp bounds exist. |
| 7 | [getting-started.md](getting-started.md) (§10–§11) | Launching full pretraining and resuming from an atomic checkpoint (`utils/checkpoint.py:CheckpointManager`). |

## Expert path — Operate, optimize, and measure

Who this is for: you want the kernels, the long-context evaluation, the
optimization catalog, and the measurement discipline.

| Step | Doc | What you will know after |
|------|-----|--------------------------|
| 1 | [kernels-and-checkpointing.md](../concepts/kernels-and-checkpointing.md) | GPU execution model, gradient checkpointing cadence, and the opt-in Triton MoE kernel (`models/moe_triton.py:triton_moe_w1w3_silu`). |
| 2 | [attention-and-positional.md](../concepts/attention-and-positional.md) (YaRN half) | Frequency interpolation internals: `models/rotary.py:compute_yarn_freqs`, `models/rotary.py:compute_yarn_mscale`, and global-layer RoPE pruning. |
| 3 | [operations.md](operations.md) (Part A) | Every script: benchmarks, smoke runs, e2e GPU check, doc gates. |
| 4 | [inference.md](../inference.md) | Incremental decode, sampling, and the `inference/long_context.py:PasskeyEvaluator` long-context test. |
| 5 | [operations.md](operations.md) (Part C) | The OPT-1…24 optimization catalog — what each lever buys and how it was verified. |
| 6 | [operations.md](operations.md) (Part B) | `utils/memory.py:estimate_model_memory_gb` VRAM budgeting and `utils/logging.py:TrainingLogger` telemetry. |
| 7 | [RECEIPTS.md](../RECEIPTS.md) | Verification receipts: which claims are machine-reproduced and when. |

---

## What each track is for

- **Concepts** build the mental model; they are self-contained and can be
  read without the code open.
- **References** are the code-keyed counterparts; keep the cited file open
  beside them.
- **Guides** assume the concepts and give procedures; this folder's
  getting-started covers onboarding, operations covers scripts/utils, and
  `tests/test_doc_refs.py` + `scripts/check_docs.py` are the two gates
  every doc change must keep green.