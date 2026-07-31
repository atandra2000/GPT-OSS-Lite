# Architecture — GPT-OSS-Lite System Design

> **Chapter 2 of the GPT-OSS-Lite documentation.** This chapter maps the production 502M-total / 247M-active model to concrete modules, dataflow, and configuration. For the mathematical motivation behind each primitive, read [foundations.md](foundations.md) first.

---

## Table of contents

1. [System overview](#1-system-overview)
2. [Layer stack and residual dataflow](#2-layer-stack-and-residual-dataflow)
3. [`GPTOSS.forward` dataflow](#3-gptossforward-dataflow)
4. [Alternating attention pattern (layers 0–11)](#4-alternating-attention-pattern-layers-0-11)
5. [Parameter accounting](#5-parameter-accounting)
6. [File map and module responsibilities](#6-file-map-and-module-responsibilities)
7. [`ModelConfig` — config to code wiring](#7-modelconfig--config-to-code-wiring)
8. [MoE dispatch and Triton opt-in](#8-moe-dispatch-and-triton-opt-in)
9. [Inference: `MixedKVCache` and generation](#9-inference-mixedkvcache-and-generation)
10. [Training pipeline integration](#10-training-pipeline-integration)
11. [Invariants and failure modes](#11-invariants-and-failure-modes)
12. [Comparison with sibling portfolio models](#12-comparison-with-sibling-portfolio-models)
13. [Where to go next](#13-where-to-go-next)

---

## 1. System overview

GPT-OSS-Lite is a **12-layer decoder-only transformer** with:

- **GQA** attention (8 query heads, 4 KV heads, `head_dim=96`)
- **Alternating** sliding-window ($W=128$) and full attention
- **Learned per-head sink bias** (init 0, clamped $[-10, 15]$ at forward)
- **YaRN RoPE** ($\theta=100000$, scale 32, train 4096 → target 131072)
- **Pruned RoPE** on global (odd) layers — 25% of dims (`head_dim // 4`)
- **MoE SwiGLU** FFN: top-2 of 8 routed + 1 shared expert (`ffn_dim=1536`)
- **Standard aux load-balancing** loss ($\alpha = 0.01$)
- **Pre-norm RMSNorm**, **weight-tied** embed ↔ LM head
- **Vocab** 128000 (LLaMA-3 BPE tokenizer in data pipeline)

### ASCII system diagram

```
                    ┌─────────────────────────────────────────┐
                    │  input_ids  (B, T)  int64               │
                    └──────────────────┬──────────────────────┘
                                       │
                    ┌──────────────────▼──────────────────────┐
                    │  nn.Embedding(vocab=128000, d=768)        │
                    │  weight tied ↔ lm_head.weight             │
                    └──────────────────┬──────────────────────┘
                                       │  x  (B, T, 768)
           ┌───────────────────────────┼───────────────────────────┐
           │         repeat 12× GPTOSSBlock                       │
           │  ┌─────────────────────────────────────────────────┐ │
           │  │  RMSNorm → GPTOSSAttention → residual           │ │
           │  │  RMSNorm → MoELayer          → residual         │ │
           │  │           (+ aux_loss per layer)                │ │
           │  └─────────────────────────────────────────────────┘ │
           └───────────────────────────┬───────────────────────────┘
                                       │
                    ┌──────────────────▼──────────────────────┐
                    │  RMSNorm (final)                          │
                    └──────────────────┬──────────────────────┘
                                       │
                    ┌──────────────────▼──────────────────────┐
                    │  lm_head: Linear(768 → 128000, bias=False)│
                    └──────────────────┬──────────────────────┘
                                       │
                    ┌──────────────────▼──────────────────────┐
                    │  logits (B, T, 128000)                    │
                    │  aux_loss scalar (mean over layers)       │
                    └───────────────────────────────────────────┘
```

### Scale summary

| Metric | Value |
|--------|-------|
| Total parameters | ~502M (`501836640` counted) |
| Active parameters / token | ~247M (`247106400` counted) |
| Sparsity | ~50.8% inactive per forward |
| Training tokens | 8.0B Chinchilla-scale |
| Training wall time (target) | 16–20 h on 1× A100 80GB |
| Train sequence length | 4096 |
| Eval / deploy sequence length | 131072 (128K) |
| KV reduction at 128K | ≥1.8× target; measured **2.00×** |
| Passkey at 128K | ≥85% target accuracy |

---

## 2. Layer stack and residual dataflow

### `GPTOSSBlock`

Each block (`models/transformer.py`) contains:

```
x_in
  │
  ├─ norm1 ──► GPTOSSAttention ──► (+) ──► x_mid
  │                                      │
  ├─ norm2 ──► MoELayer ──► (+) ──► x_out
  │              │
  │              └── aux_loss (scalar)
```

**Pre-norm** means normalization precedes each sublayer. Residual connections are unscaled (no DeepSeek-style depth scaling).

### `GPTOSSAttention` internals

```
x (B,T,768)
  │
  ├─ q_proj  → Q  (B, 8, T, 96)
  ├─ kv_proj → K,V (B, 4, T, 96) each
  │
  ├─ YaRNRoPE(positions, n_pruned_dims) → cos, sin
  ├─ apply_rope(Q, cos, sin)
  ├─ apply_rope(K, cos, sin)
  ├─ repeat_kv(K,V) → (B, 8, T, 96)
  │
  ├─ causal_attention(
  │     window = 128 if layer_idx even else None,
  │     sink_bias = clamp(sink_bias, -10, 15)
  │  )
  │
  └─ o_proj → (B, T, 768)
```

Layer index parity sets `is_windowed = (layer_idx % 2 == 0)` in `GPTOSSAttention.__init__`.

### `MoELayer` internals

```
x (B,T,768) → flat (N, 768)
  │
  ├─ MoERouter → top-2 indices + weights + all_logits
  ├─ dispatch (stacked or triton_grouped)
  ├─ aux_load_balancing_loss(all_logits)
  ├─ + shared SwiGLU expert(s)
  │
  └─ view (B, T, 768)
```

---

## 3. `GPTOSS.forward` dataflow

Source: `models/transformer.py`, class `GPTOSS`.

### Inputs

| Argument | Shape | Default |
|----------|-------|---------|
| `idx` | $(B, T)$ int token ids | required |
| `positions` | $(T)$ or broadcastable position ids | `arange(T)` |

Positions matter for YaRN at eval lengths $T > \text{max\_seq\_len}$ during training — pass explicit position tensors during 128K inference.

### Forward steps

1. **Embed:** `x = embed(idx)` → $(B, T, 768)$.
2. **Blocks:** For each `GPTOSSBlock` in `blocks`:
   - Optional gradient checkpoint when `gradient_checkpointing` and `layer_idx % grad_ckpt_every == 0`.
   - `x, aux = block(x, positions)` — accumulates per-layer aux losses.
3. **Head:** `x = norm(x)`; `logits = head(x)` → $(B, T, 128000)$.
4. **Aux:** `aux_loss = mean(stack(aux_losses))` — scalar.

### Returns

```python
(logits, aux_loss)
# logits: (B, T, vocab_size)
# aux_loss: scalar — multiply by aux_loss_alpha in training loop
```

### Weight tying

When `cfg.weight_tying=True`:

```python
self.head.weight = self.embed.weight
```

`num_parameters()` deduplicates by parameter `id()` so embed/head are counted once. Savings: `vocab_size × d_model = 128000 × 768 = 98304000` parameters (~98M).

### Initialization

- Linear / Embedding: normal, `std=0.02`
- RMSNorm weight: ones
- `sink_bias`: **zeros** (per head, per windowed/full layer — all attention layers when `sink_bias=True`)

---

## 4. Alternating attention pattern (layers 0–11)

Even indices → **SWA** (`window_size=128`). Odd indices → **full** causal attention.

| Layer | Index | Attention | RoPE prune | KV cache growth (decode) |
|-------|-------|-----------|------------|--------------------------|
| 0 | even | SWA $W=128$ | no | Ring buffer, cap 128 |
| 1 | odd | Full | yes ($D/4=24$ dims) | Linear with $T$ |
| 2 | even | SWA | no | Ring buffer |
| 3 | odd | Full | yes | Linear with $T$ |
| 4 | even | SWA | no | Ring buffer |
| 5 | odd | Full | yes | Linear with $T$ |
| 6 | even | SWA | no | Ring buffer |
| 7 | odd | Full | yes | Linear with $T$ |
| 8 | even | SWA | no | Ring buffer |
| 9 | odd | Full | yes | Linear with $T$ |
| 10 | even | SWA | no | Ring buffer |
| 11 | odd | Full | yes | Linear with $T$ |

**Counts:** 6 SWA + 6 full = 12 layers.

### Why alternation instead of grouping?

Grouped patterns (e.g. 6 SWA then 6 full) create longer paths where no layer sees the full sequence. Alternation ensures every two layers, one global layer can integrate distant context while the next SWA layer refines local structure with a compact cache.

### `extra_repr` debugging

Each `GPTOSSAttention` reports mode in `extra_repr`:

```
layer=0 (SWA), H=8/4, D=96, window=128
layer=1 (Full, pruned=24), H=8/4, D=96, window=128
```

---

## 5. Parameter accounting

### Total parameters (`num_parameters`)

Production model count: **501,836,640** (~502M). Breakdown by component:

#### Embedding + tied head

| Component | Formula | Count |
|-----------|---------|-------|
| `embed` / `head` (tied) | $V \cdot d$ | $128000 \times 768 = 98304000$ |

Without tying, total would be $\approx 600$M — tying saves ~98M.

#### Per `GPTOSSBlock` (×12)

**Attention**

| Parameter | Count |
|-----------|-------|
| `q_proj` | $768^2 = 589824$ |
| `kv_proj` | $768^2 = 589824$ |
| `o_proj` | $768^2 = 589824$ |
| `sink_bias` | $8$ |
| Subtotal | $\approx 1769480$ |

YaRN buffers are non-persistent (`persistent=False`) — not counted in params.

**MoE**

| Parameter | Count |
|-----------|-------|
| Per SwiGLU expert | $3 \cdot 768 \cdot 1536 = 3538944$ |
| 8 routed experts | $28311552$ |
| 1 shared expert | $3538944$ |
| Router `gate` | $768 \times 8 = 6144$ |
| Subtotal | $31854640$ |

**Norms**

| Parameter | Count |
|-----------|-------|
| `norm1` + `norm2` | $2 \times 768 = 1536$ |

**Per block total** $\approx 33.6$M → ×12 $\approx 403$M MoE-heavy.

#### Final norm

768 parameters.

#### Sanity check

$$
98.3\text{M (embed)} + 12 \times 33.6\text{M} + 768 \approx 502\text{M}
$$

### Active parameters (`num_active_parameters`)

**247,106,400** (~247M). MoE experts not routed on a given token are **inactive**.

Formula from `GPTOSS.num_active_parameters()`:

```python
non_moe = all parameters except names containing "experts"
expert_params = 3 * d_model * ffn_dim
moe_active_per_layer = (n_activated + n_shared) * expert_params + d_model * n_routed_experts
return non_moe + (moe_active_per_layer) * n_layers
```

Per layer MoE active:

$$
(2 + 1) \times 3 \times 768 \times 1536 + 768 \times 8 = 10622976
$$

Inactive routed experts per layer: $5 \times 3 \times 768 \times 1536 = 17694720$ not executed.

### KV-cache memory formula (BF16, batch=1)

Per layer per token (K+V):

$$
\text{bytes} = 2 \times H_{\text{kv}} \times D \times 2 = 2 \times 4 \times 96 \times 2 = 1536
$$

Mixed cache at sequence $T$:

$$
\text{KV}_{\text{mixed}} = (6 \cdot \min(128, T) + 6 \cdot T) \times 1536 \text{ bytes}
$$

At $T = 131072$: **1.13 GB** vs **2.25 GB** all-full → **2.00×**.

---

## 6. File map and module responsibilities

```
GPT-OSS-Lite/
├── models/
│   ├── transformer.py    # ModelConfig, RMSNorm, GPTOSSBlock, GPTOSS
│   ├── attention.py      # SWA/full, sink bias, causal_attention, GQA
│   ├── moe.py            # MoELayer, router, aux loss, stacked dispatch
│   ├── moe_triton.py     # Opt-in Triton W1/W3+silu grouped GEMM
│   ├── yarn.py           # YaRNRoPE module
│   └── rotary.py         # apply_rope, compute_yarn_freqs, mscale
├── training/
│   └── pretrain.py       # Main training loop, chunked CE, NaN guard
├── inference/
│   ├── generate.py       # MixedKVCache, autoregressive generate()
│   └── long_context.py   # PasskeyEvaluator for 128K eval
├── configs/
│   └── pretrain_a100_502m.yaml   # Canonical production config
└── scripts/
    ├── kv_cache_benchmark.py     # Analytical KV headline metric
    └── passkey_eval.py             # Passkey retrieval CLI
```

### `models/transformer.py`

| Symbol | Role |
|--------|------|
| `ModelConfig` | Dataclass + validation invariants |
| `RMSNorm` | Pre-norm; FP32 RMS stats, native dtype output |
| `GPTOSSBlock` | Attention + MoE residuals |
| `GPTOSS` | Top-level module, checkpointing, param counters |

### `models/attention.py`

| Symbol | Role |
|--------|------|
| `SINK_CLAMP_MIN/MAX` | $-10.0$, $15.0$ |
| `causal_attention` | SDPA path with optional window + sink |
| `manual_causal_attention` | Test oracle (FP32 scores) |
| `repeat_kv` | GQA broadcast without contiguous() |
| `GPTOSSAttention` | Projections, YaRN, alternation logic |

### `models/moe.py`

| Symbol | Role |
|--------|------|
| `SwiGLUExpert` | $W_1, W_2, W_3$ |
| `MoERouter` | Top-$k$ gating, FP32 softmax |
| `aux_load_balancing_loss` | Switch-style aux |
| `MoELayer` | Dispatch + shared expert |

### `models/moe_triton.py`

| Symbol | Role |
|--------|------|
| `triton_moe_w1w3_silu` | Fused gate+up+silu; W2 stays PyTorch |
| `HAS_TRITON` | Import guard |

### `models/yarn.py` + `models/rotary.py`

| Symbol | Role |
|--------|------|
| `YaRNRoPE` | Buffer `inv_freq`, forward cos/sin + prune |
| `compute_yarn_freqs` | Ramp-blended inverse frequencies |
| `compute_yarn_mscale` | Attention temperature correction |
| `apply_rope` | Dtype-safe rotation |

### `training/pretrain.py`

| Concern | Implementation |
|---------|----------------|
| Loss | Chunked CE + `aux_loss_alpha * aux_loss` |
| Optimizer | AdamW fused, FP32 master weights |
| Scheduler | Warmup + cosine |
| Compile | `torch.compile(max-autotune)` when CUDA |
| Stability | NaN guard, grad clip 1.0 |
| Reproducibility | `seed_everything`, `CUBLAS_WORKSPACE_CONFIG` |

### `inference/generate.py`

| Symbol | Role |
|--------|------|
| `MixedKVCache` | Ring (SWA) + growing (global) per layer |
| `generate()` | Token-by-token with cache |

### `inference/long_context.py`

| Symbol | Role |
|--------|------|
| `PasskeyEvaluator` | Build prompts, run generate, score accuracy |

---

## 7. `ModelConfig` — config to code wiring

`ModelConfig` in `models/transformer.py` mirrors YAML `model:` section. Canonical values from `configs/pretrain_a100_502m.yaml`:

| Field | Value | Consumed by |
|-------|-------|-------------|
| `vocab_size` | 128000 | `GPTOSS.embed`, `head` |
| `d_model` | 768 | All layers |
| `n_layers` | 12 | `GPTOSS.blocks` |
| `n_heads` | 8 | `GPTOSSAttention` |
| `n_kv_heads` | 4 | `GPTOSSAttention`, KV cache |
| `head_dim` | 96 | Projections, RoPE |
| `ffn_dim` | 1536 | MoE experts |
| `n_routed_experts` | 8 | `MoELayer` |
| `n_activated_experts` | 2 | Router top-$k$ |
| `n_shared_experts` | 1 | Shared SwiGLU |
| `window_size` | 128 | SWA layers |
| `sink_bias` | true | Per-head `sink_bias` param |
| `rope_theta` | 100000 | YaRN base |
| `yarn_scale_factor` | 32 | YaRN stretch |
| `yarn_original_max_seq_len` | 4096 | Train context |
| `yarn_target_seq_len` | 131072 | Extrapolation target |
| `yarn_beta_fast` | 32 | Ramp band |
| `yarn_beta_slow` | 1 | Ramp band |
| `yarn_mscale` | true | mscale enable |
| `yarn_prune_rope_global` | true | Prune on odd layers |
| `max_seq_len` | 4096 | Training windows |
| `eval_max_seq_len` | 131072 | Long-context eval |
| `dtype` | bf16 | Autocast in pretrain |
| `weight_tying` | true | Embed ↔ head |
| `rms_norm_eps` | 1e-5 | RMSNorm |
| `init_std` | 0.02 | Weight init |
| `moe_dispatch` | `"stacked"` | MoE path (opt-in `"triton_grouped"`) |

### Validation highlights (`__post_init__`)

- `n_heads % n_kv_heads == 0` (GQA)
- `n_heads * head_dim == d_model`
- `yarn_scale_factor >= 1`; if `> 1`, require `original < target`
- Warns if `eval_max_seq_len < max_seq_len`

### YAML → Python

```python
with open(config_path) as f:
    cfg = yaml.safe_load(f)
model_cfg = ModelConfig(**cfg["model"])
model = GPTOSS(model_cfg)
```

Training hyperparameters (`aux_loss_alpha`, `compile`, etc.) live under `training:` — not in `ModelConfig`.

---

## 8. MoE dispatch and Triton opt-in

### Default: `moe_dispatch = "stacked"`

`MoELayer._dispatch_vectorized`:

1. Flatten token-expert assignments.
2. `argsort` experts (`stable=True` for reproducibility).
3. Per-expert chunk loop with `index_add` weighted accumulation.

Pure PyTorch — runs on CPU and GPU.

### Opt-in: `moe_dispatch = "triton_grouped"`

`MoELayer._dispatch_triton`:

1. Same sort/grouping as stacked path.
2. `triton_moe_w1w3_silu` fuses W1, W3, silu, element-wise multiply.
3. W2 matmul per expert in PyTorch on sorted chunks.

**Contract:** If Triton unavailable and config requests `triton_grouped`, import raises with explicit message — **no silent fallback**.

Set in YAML:

```yaml
model:
  moe_dispatch: "triton_grouped"
```

Default in `pretrain_a100_502m.yaml` is `"stacked"`.

---

## 9. Inference: `MixedKVCache` and generation

### Cache types per layer

| Layer type | Storage | Max tokens stored |
|------------|---------|-------------------|
| SWA (even) | Ring buffer $(B, H, W, D)$ | $W = 128$ |
| Full (odd) | Growing tensor | $T$ (cap configurable) |

Cache stores **rotated** keys (post-RoPE) to avoid recomputing rotations during decode.

### Decode complexity

- SWA layer append: $O(1)$ cache size per step (ring update).
- Full layer append: $O(1)$ append amortized with growth strategy.
- Attention compute: SWA layers attend to $\leq 128$ keys; full layers attend to full history.

### Sink bias at inference

`inference/generate.py` imports `SINK_CLAMP_MIN/MAX` and applies the same clamp as training when building attention masks for cached decode.

### Long-context eval

`PasskeyEvaluator` (`inference/long_context.py`):

- Default context lengths: 4096, 8192, 32768, 65536, 131072
- Inserts 5-digit passkey in filler text
- Target: **≥85%** accuracy at 128K on trained checkpoint

---

## 10. Training pipeline integration

### Loss composition

```python
logits, aux_loss = model(input_ids, positions)
ce_loss = chunked_cross_entropy(logits, targets, chunk_size=4096)
loss = ce_loss + aux_loss_alpha * aux_loss  # alpha = 0.01
```

### Effective batch

$$
B_{\text{eff}} = \text{micro\_batch} \times \text{grad\_accum} = 8 \times 4 = 32 \text{ sequences}
$$

Tokens per optimizer step: $32 \times 4096 = 131072$.

### Gradient checkpointing

`enable_gradient_checkpointing(every=3)` checkpoints every third block — trades ~30% extra compute for materially lower activation memory.

### Hardware knobs (`_set_hardware_perf_knobs`)

- TF32 for matmul and cuDNN
- `cudnn.benchmark = True`, `benchmark_limit = 0`
- `preferred_blas_library = "cublaslt"`
- `set_float32_matmul_precision("high")`

### Checkpointing

Atomic saves via `utils/checkpoint.py`: write `.tmp`, rename. Includes optimizer, scheduler, RNG state in `rng_step_N.pt` when enabled.

---

## 11. Invariants and failure modes

### Must preserve (architectural contract)

| Invariant | Violation impact |
|-----------|------------------|
| Even=SWA, odd=full | KV reduction collapses toward 1× |
| `window_size=128` on SWA layers | Changes headline KV math |
| Standard aux loss, $\alpha=0.01$ | MoE collapse or portfolio mismatch |
| Sink bias clamp at forward | BF16 mask overflow risk |
| YaRN train+decode for 128K | Passkey metric fails |
| `moe_dispatch` explicit opt-in | Silent perf path switch forbidden |
| Weight tying default on | Param budget drifts +~98M |
| GQA 8/4 | KV bandwidth doubles if reverted to MHA |

### Common failure modes

| Symptom | Likely cause | Check |
|---------|--------------|-------|
| MoE routes to one expert | Aux loss disabled or $\alpha$ too low | `aux_loss_alpha`, aux loss logs |
| NaN loss | Router saturation, bad data shard | NaN guard rollback, FP32 softmax path |
| OOM at 4096 train | Checkpointing off, compile overhead | `grad_checkpoint`, micro batch |
| 128K gibberish | Positions not passed, YaRN misconfig | `eval_max_seq_len`, position ids |
| Triton crash on Mac | `triton_grouped` without CUDA | Use `moe_dispatch: stacked` |
| Sliding-window test fail | Mask bug in `attention.py` | `test_sliding_window_matches_full` |

### After any `attention.py` change

Run:

```bash
pytest tests/test_attention.py -v
```

Specifically `test_sliding_window_matches_full` must pass.

---

## 12. Comparison with sibling portfolio models

| Dimension | GPT-OSS-Lite | DeepSeek-v3-Lite | LLaMA-3-Lite | HyMo | Mamba-3-Lite |
|-----------|--------------|------------------|--------------|------|--------------|
| **Paradigm** | Decoder-only TF | Decoder-only TF | Decoder-only TF | Hybrid GDN/MLA | Pure SSM |
| **Attention** | GQA 8Q/4KV | MLA latent KV | GQA full | 3:1 GDN/MLA | Complex SSD |
| **Long context** | YaRN 128K train+decode | YaRN decode-focused | θ=500K, train 2K | — | Constant state |
| **Local/global** | SWA(128)/full alt | Full + MLA | Full | Alternating | Chunkwise SSD |
| **FFN** | MoE top-2/8 + shared | DeepSeekMoE | Dense SwiGLU | Asymmetric MoE | Dense |
| **MoE aux** | Switch $\alpha=0.01$ | **Aux-loss-free gate** | — | Custom | — |
| **Sink** | Learned per-head bias | None | None | None | None |
| **RoPE extras** | Prune 25% on global | YaRN | Extended base θ | — | — |
| **KV at 128K** | ~1.13 GB (mixed) | MLA-compressed | ~2.25 GB+ (GQA full) | Hybrid state | $O(1)$ state |
| **Scale (this repo)** | 502M / 247M active | Portfolio scale | Portfolio scale | Portfolio scale | Portfolio scale |
| **Primary headline** | 2× KV + passkey 128K | MTP, μP, speculative | 78% memory stack | GDN kernel | SSD throughput |

### Deliberate distinctions

1. **vs DeepSeek-v3-Lite:** Standard aux loss instead of aux-loss-free routing; GQA+SWA instead of MLA; learned sinks instead of none.
2. **vs LLaMA-3-Lite:** MoE instead of dense FFN; sliding/full alternation; YaRN with train-time 128K alignment.
3. **vs HyMo:** Pure attention stack (no GDN); standard MoE routing; long-context via YaRN not hybrid recurrence.
4. **vs Mamba-3-Lite:** Attention-based long context with KV cache (mitigated by SWA) vs constant-size SSM state; MoE vs dense.

GPT-OSS-Lite is the portfolio's **long-context MoE + attention sink** reference implementation.

---

## 13. Where to go next

| Goal | Resource |
|------|----------|
| Mathematical foundations | [foundations.md](foundations.md) |
| Sink bias deep-dive | `documentation/ATTENTION_SINKS.md` |
| Run KV benchmark | `python3 scripts/kv_cache_benchmark.py` |
| Run passkey eval | `python3 scripts/passkey_eval.py` |
| Start training | `python3 training/pretrain.py --config configs/pretrain_a100_502m.yaml` |
| Portfolio rules | `AGENTS.md` |
| Cross-architecture skill | `.agents/skills/llm-architecture/SKILL.md` |

---

<!-- docs:verified 2026-07-31 · fa6f918 -->
