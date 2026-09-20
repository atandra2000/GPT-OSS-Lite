# Glossary — GPT-OSS-Lite

> Notation, component terms, and the config keys that appear throughout the
> documentation. Full config semantics live in
> [config-and-api.md](../references/config-and-api.md) and
> [foundations-and-architecture.md](../concepts/foundations-and-architecture.md);
> this page is the quick lookup. Every code citation uses symbol anchors
> verified by `tests/test_doc_refs.py`.

---

## Notation

| Symbol | Meaning | Canonical value |
|--------|---------|-----------------|
| `d_model` | Residual-stream width | 768 |
| `n_layers` | Decoder blocks | 12 (6 SWA + 6 full, alternating) |
| `n_heads` | Query heads | 8 |
| `n_kv_heads` | Key/value heads (GQA) | 4 (`n_rep` = 2) |
| `head_dim` | Per-head QK/V width | 96 |
| `ffn_dim` | Dense/shared FFN width | 1536 |
| `n_routed_experts` / `n_activated_experts` | Routed / per-token active experts | 8 / 2 |
| `n_shared_experts` | Always-on shared experts | 1 |
| `window_size` | Sliding-window width on even layers | 128 |
| `max_seq_len` | Training context window | 4096 |
| `yarn_target_seq_len` | YaRN-extended context | 131072 (128K) |
| `vocab_size` | Tokenizer rows | 128 000 |
| `aux_loss_alpha` | Load-balancing loss weight | 0.01 |
| `[B, T, d]` | Batch × sequence × hidden tensor shape | — |

## Attention: windows, GQA, and sinks

| Term | Definition | Where implemented |
|------|------------|-------------------|
| **SWA (sliding-window attention)** | Even-indexed layers attend only to the last `window_size` = 128 tokens, capping their KV memory at a constant. | `models/attention.py:GPTOSSAttention` (`is_windowed` = `layer_idx % 2 == 0`) |
| **Window/full alternation** | The 12 layers strictly alternate SWA and full attention; odd layers retain the complete prefix so information propagates beyond one window. | `models/attention.py:GPTOSSAttention.forward` |
| **GQA** | Grouped-query attention: 8 query heads share 4 KV heads via `models/attention.py:repeat_kv` (expand + reshape, no `.contiguous()`). | `models/attention.py:repeat_kv` |
| **Learned sink bias** | A per-head scalar added to the attention scores of every position — models' attention "sink" tokens — trained as a parameter. | `models/attention.py:GPTOSSAttention.sink_bias` |
| **Sink clamp** | The bias is clamped to `[-10, 15]` at forward time (BF16 SDPA mask-add overflow guard); the raw parameter keeps gradient flow. | `models/attention.py:SINK_CLAMP_MIN`, `models/attention.py:SINK_CLAMP_MAX` |
| **`causal_attention`** | The single attention backend used by both windowed and full layers — window is `None` (full) or an int (SWA). | `models/attention.py:causal_attention` |
| **`manual_causal_attention`** | FP32-accumulating non-SDPA reference used for tests and debugging. | `models/attention.py:manual_causal_attention` |
| **`_window_mask` / `_causal_mask`** | Cached masks keyed by `(T, window, device, dtype)` so decode never re-materializes them. | `models/attention.py:_window_mask`, `models/attention.py:_causal_mask` |

## Position encoding: RoPE → YaRN

| Term | Definition | Where implemented |
|------|------------|-------------------|
| **RoPE** | Rotary position embedding — rotates Q/K pairs by position-dependent angles. | `models/rotary.py:apply_rope` |
| **YaRN** | Yet another RoPE extension: frequency interpolation from a 4K training window to 128K, with β-fast/β-slow ramp bands. | `models/yarn.py:YaRNRoPE` |
| **Frequency interpolation** | Per-dimension frequency blend computed by `compute_yarn_freqs` from `scale_factor` = 32 and the ramp bands. | `models/rotary.py:compute_yarn_freqs` |
| **mscale** | Attention-scale correction (`compute_yarn_mscale`) that rescales softmax temperature under YaRN. | `models/rotary.py:compute_yarn_mscale` |
| **RoPE pruning (global layers)** | On full-attention layers, the first 25% of RoPE dims are pinned to angle 0 (`cos=1, sin=0`) — global layers carry no position on pruned dims. | `models/attention.py:GPTOSSAttention._n_pruned_dims`, `models/yarn.py:YaRNRoPE.forward` |
| **`rope_theta`** | Base frequency (100 000 canonical). | `models/transformer.py:ModelConfig` |

## Mixture of Experts

| Term | Definition | Where implemented |
|------|------------|-------------------|
| **Top-2-of-8 routing** | Each token activates 2 of 8 routed experts, chosen by softmax router logits. | `models/moe.py:MoERouter` |
| **Shared expert** | One always-on `SwiGLU` FFN applied to every token, independent of routing. | `models/moe.py:SwiGLUExpert`, `models/moe.py:MoELayer` |
| **Aux load-balancing loss** | Switch-Transformer-style objective (α = 0.01): penalizes disagreement between routing frequency `f` and mean gate probability `P`, computed in FP32 to avoid BF16 underflow. Deliberate contrast with DeepSeek-v3-Lite's aux-loss-free bias gate. | `models/moe.py:aux_load_balancing_loss` |
| **`MoELayer`** | The full block: router → dispatch → routed experts + shared expert → combine. | `models/moe.py:MoELayer` |
| **Triton grouped-GEMM** | Opt-in fused W1/W3+silu kernel (`moe_dispatch = "triton_grouped"`); W2 stays in PyTorch; autograd backward uses the pure-PyTorch reference. | `models/moe_triton.py:triton_moe_w1w3_silu`, `models/moe_triton.py:_MoEW1W3SiluFunction` |

## Training and precision

| Term | Definition | Where implemented |
|------|------------|-------------------|
| **Warmup→cosine schedule** | 3000-step linear ramp (4.9% of 61 000 steps — MoE top-2 stability), cosine decay to `min_lr_ratio` = 0.05. | `training/pretrain.py:make_warmup_cosine_lambda` |
| **NaN guard** | Loss finite-check; on repeated failure rolls back to the last checkpoint (`nan_guard_max_consecutive` = 5). | [training.md](../training.md) |
| **Chunked cross-entropy** | Computes the CE loss in `chunk_size` = 4096-token slices so full logits never materialize. | `training/pretrain.py:chunked_cross_entropy` |
| **Gradient checkpointing** | Re-activates instead of storing intermediates, applied every 3rd layer. | [kernels-and-checkpointing.md](../concepts/kernels-and-checkpointing.md) |
| **BF16 autocast + FP32 AdamW** | Compute in BF16 (autocast); optimizer state stays FP32; TF32 matmuls on CUDA. | [optimizers-and-numerics.md](../concepts/optimizers-and-numerics.md) |
| **Weight tying** | Embedding and LM head share weights. | `models/transformer.py:GPTOSS` |
| **`RMSNorm`** | Root-mean-square norm, native-dtype (no FP32 copy). | `models/transformer.py:RMSNorm` |
| **`torch.compile`** | `max-autotune` mode, auto-invoked on CUDA when `training.compile: true`. | `training/pretrain.py:_set_hardware_perf_knobs` |
| **Atomic checkpoint** | Model/optimizer/RNG state written via `CheckpointManager` so a crash never leaves a torn checkpoint set. | `utils/checkpoint.py:CheckpointManager` |
| **`PretrainDataset`** | mmap-backed sharded dataset — zero-copy `__getitem__`. | `training/pretrain.py:PretrainDataset` |

## Inference and the KV cache

| Term | Definition | Where implemented |
|------|------------|-------------------|
| **`MixedKVCache`** | Per-layer cache: windowed layers hold a `window` = 128 ring buffer; global layers hold the full prefix — the mechanism behind the 1.94×–2.0× headline cut. | `inference/generate.py:MixedKVCache` |
| **Ring buffer** | Fixed-size windowed cache with a rotating write head; old tokens are overwritten, never accumulated. | `inference/generate.py:MixedKVCache.append` |
| **KV-cache reduction** | Measured vs pure-GQA full attention: ≈1.94× at 4K (constant-window floor), rising to 2.00× at 128K. | `scripts/kv_cache_benchmark.py`, [inference.md](../inference.md) |
| **`generate`** | Incremental decode loop with the mixed cache; decode is O(1) per step per windowed layer. | `inference/generate.py:generate` |
| **Passkey eval** | Long-context retrieval test — inserts a random key in filler text and asks the model to recall it at up to 128K. | `inference/long_context.py:PasskeyEvaluator` |

## Data pipeline

| Term | Definition |
|------|------------|
| **8.0B-token universal corpus** | Prepared once in workspace `shared_data/` (7-source mixture); consumed via this repo's shim. |
| **Shim** | `data/prepare_data.py` — per-project entry that imports the workspace pipeline (not vendored). |
| **mmap shard** | Packed binary shard read via memory map by `training/pretrain.py:PretrainDataset`. |
| **Long-context augmentation** | 10% of sequences packed to 4096 with document-boundary awareness and passkey-style inserts, keeping eval readiness during pretraining. |
| **Chinchilla-optimal run** | `configs/pretrain_a100_502m.yaml`: ~61 000 steps at ~502M total / ~247M active params (8B tokens ≈ 16–20 h on one A100 80 GB — target, not yet run). |

## Acronyms

| Acronym | Expansion |
|---------|-----------|
| SWA | Sliding-Window Attention |
| GQA | Grouped-Query Attention |
| MoE | Mixture of Experts |
| FFN | Feed-Forward Network |
| RoPE | Rotary Position Embedding |
| YaRN | Yet another RoPE extension (context-length scaling) |
| KV cache | Key/Value cache |
| SDPA | Scaled-Dot-Product Attention (`torch.nn.functional.scaled_dot_product_attention`) |
| FA2 | Flash Attention 2 (used here via SDPA's flash backend) |
| MFU | Model FLOPs Utilization |
| BPE | Byte-Pair Encoding |
| CE | Cross-Entropy (loss) |

<!-- docs:verified 2026-09-21 · ada1459 -->
