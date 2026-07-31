# Performance Optimizations — Catalog OPT-1 through OPT-24

> **Chapter: engineering decisions.** This document is a numbered catalog of
> deliberate performance and stability optimizations in GPT-OSS-Lite. Each entry
> states the **problem**, the **fix** (with file references), and **expected
> impact**. For architectural context see [architecture.md](architecture.md);
> for training integration see [training.md](training.md).

---

## Table of contents

1. [How to read this catalog](#1-how-to-read-this-catalog)
2. [Attention and masks (OPT-1 – OPT-3, OPT-22)](#2-attention-and-masks-opt-1--opt-3-opt-22)
3. [Tensor layout and dtype (OPT-2, OPT-4, OPT-23)](#3-tensor-layout-and-dtype-opt-2-opt-4-opt-23)
4. [MoE dispatch (OPT-5, OPT-24)](#4-moe-dispatch-opt-5-opt-24)
5. [Training loop (OPT-8 – OPT-10, OPT-17 – OPT-21)](#5-training-loop-opt-8--opt-10-opt-17--opt-21)
6. [Inference (OPT-11 – OPT-15, OPT-22)](#6-inference-opt-11--opt-15-opt-22)
7. [Numerical stability (OPT-6, OPT-7)](#7-numerical-stability-opt-6-opt-7)
8. [Compilation (OPT-16)](#8-compilation-opt-16)
9. [Quick reference table](#9-quick-reference-table)
10. [Verification commands](#10-verification-commands)
11. [Related documentation](#11-related-documentation)

---

## 1. How to read this catalog

| Field | Meaning |
|-------|---------|
| **Problem** | What was slow, memory-heavy, or numerically fragile before the fix |
| **Fix** | Current implementation — note when `functools.lru_cache` replaced older dict caches |
| **Impact** | Measured or estimated gain; always verify on your hardware |
| **Files** | Primary source locations |

**Numbering gaps:** OPT-6, OPT-7, and OPT-16 cover stability and compilation
entries from the same audit pass. All 24 IDs are assigned; there are no reserved
slots beyond that.

**Stale patterns to avoid in docs and configs:**

- Removed Triton env-var gate — use `moe_dispatch: triton_grouped` in YAML
- Old standalone MoE Triton doc — see [triton_kernels.md](triton_kernels.md) instead
- Manual dict-based mask caches — replaced by `@functools.lru_cache` (OPT-1)

---

## 2. Attention and masks (OPT-1 – OPT-3, OPT-22)

### OPT-1 — Mask cache via `functools.lru_cache`

**Problem:** Building causal and sliding-window boolean masks every forward pass
allocates $O(T^2)$ tensors and burns GPU time. At $T = 4096$ training and at
decode with growing $T_k$, mask construction was a measurable fraction of attention
latency.

**Fix:** Cache masks by shape/device/dtype signature:

```python
@functools.lru_cache(maxsize=None)
def _causal_mask(T: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    idx = torch.arange(T, device=device)
    return idx.unsqueeze(1) >= idx.unsqueeze(0)

@functools.lru_cache(maxsize=None)
def _window_mask(T_q: int, T_k: int, window: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    ...
```

`causal_attention()` composes `_causal_mask & _window_mask` for prefill and
calls `_window_mask` alone for decode ($T_q = 1$, $T_k$ growing).

**Impact:**

- First call at a given $(T, device)$ pays allocation cost once per process.
- Subsequent forwards at the same $T$ reuse cached masks — typical training sees
  fixed $T = 4096$ → **one** causal mask build per GPU for the entire run.
- Decode builds one mask per distinct $T_k$ — amortized over 128K context.

**Files:** `models/attention.py` (`_causal_mask`, `_window_mask`, `causal_attention`)

**Related:** [attention.md](attention.md), [ATTENTION_SINKS.md](ATTENTION_SINKS.md)

---

### OPT-2 — `repeat_kv` without `.contiguous()`

**Problem:** GQA expands $n_{\text{kv\_heads}}$ to $n_{\text{heads}}$ by repeating
each KV head. A naive `repeat_interleave` + `.contiguous()` forces a full tensor
copy every layer — $12 \times$ per forward at 12 layers.

**Fix:** `expand` + `reshape` preserves a non-contiguous view SDPA accepts:

```python
def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    if n_rep == 1:
        return x
    B, H_kv, T, D = x.shape
    x = x[:, :, None, :, :]
    x = x.expand(B, H_kv, n_rep, T, D)
    return x.reshape(B, H_kv * n_rep, T, D)
```

FlashAttention / SDPA flash path handles non-contiguous K/V internally.

**Impact:** Eliminates redundant memcpy on every attention layer. Profiled in
`scripts/profile_components.py` as sub-millisecond but scales with layer count
and batch.

**Files:** `models/attention.py`, `inference/generate.py`

**Verify:** `pytest tests/test_attention.py::test_repeat_kv_identity -v`

---

### OPT-3 — Sink path via extended K/V + mask column

**Problem:** Learned attention-sink bias must enter the softmax denominator
without changing output dimensionality. A separate manual attention path would
bypass SDPA/FA2.

**Fix:** Append a synthetic sink key/value (zeros) and put clamped sink bias on
the mask's last column:

```python
sink_k = torch.zeros(B, H, 1, D, device=device, dtype=dtype)
sink_v = torch.zeros(B, H, 1, D_v, device=device, dtype=value_states.dtype)
k_ext = torch.cat([key_states, sink_k], dim=2)
v_ext = torch.cat([value_states, sink_v], dim=2)
mask[:, :, T_k] = sink_bias.to(dtype).unsqueeze(1).expand(H, T_q)
return F.scaled_dot_product_attention(query_states, k_ext, v_ext, attn_mask=mask.unsqueeze(0))
```

Sink receives zero value — only the bias logit affects weights.

**Impact:** Single SDPA code path for both sink and non-sink layers; enables FA2
for the bulk of the computation. See [ATTENTION_SINKS.md](ATTENTION_SINKS.md) for
the full mathematical treatment.

**Files:** `models/attention.py` (`causal_attention` sink branch)

**Numerical note:** Sink bias clamped to $[-10, 15]$ before mask add (OPT-14).

---

### OPT-22 — Decode-specific window mask ($T_q = 1$)

**Problem:** During autoregressive decode, $T_q = 1$ but $T_k$ grows. Building a
full $(T_k, T_k)$ causal mask each step is $O(T_k^2)$ — catastrophic at 128K.

**Fix:** `_window_mask` special-cases $T_q \neq T_k$:

```python
if T_q == T_k:
    idx = torch.arange(T_q, device=device)
    return (idx.unsqueeze(0) - idx.unsqueeze(1) < window) & _causal_mask(T_q, device, dtype)
# Decode: T_q=1, T_k grows. Query position is T_k - 1.
idx_q = torch.tensor([T_k - 1], device=device)
idx_k = torch.arange(T_k, device=device)
return (idx_q.unsqueeze(-1) - idx_k.unsqueeze(0) < window)
```

Combined with `MixedKVCache` ring buffer (OPT-11), effective KV length for
windowed layers stays $\leq W$ regardless of global position.

**Impact:** Decode attention mask is $O(T_k)$ not $O(T_k^2)$. Essential for 128K
inference throughput.

**Files:** `models/attention.py` (`_window_mask`), `inference/generate.py`

**Related:** [inference.md](inference.md) when published

---

## 3. Tensor layout and dtype (OPT-2, OPT-4, OPT-23)

### OPT-4 — RMSNorm native dtype activations

**Problem:** Naive RMSNorm upcasts the full activation tensor to FP32 for
`mean` + `rsqrt`, doubling memory bandwidth and breaking BF16 end-to-end paths.

**Fix:** Compute RMS statistics in FP32 on a **detached** slice, multiply back
in native dtype:

```python
def forward(self, x: torch.Tensor) -> torch.Tensor:
    rms = x.detach().float().pow(2).mean(dim=-1, keepdim=True).add(self.eps).rsqrt()
    return (x * (rms * self.weight.to(rms.dtype)).to(x.dtype))
```

Only the reduction runs in FP32; `x` stays BF16 throughout.

**Impact:** ~2× less activation memory traffic in norm layers; preserves BF16
through SDPA without dtype promotion surprises.

**Files:** `models/transformer.py` (`RMSNorm`)

**Verify:** `tests/test_validation.py` (RMSNorm BF16 stability)

---

### OPT-23 — `apply_rope` dtype preservation

**Problem:** If `cos`/`sin` stay FP32 while `q`/`k` are BF16, PyTorch promotes
the RoPE output to FP32. SDPA then rejects mixed dtypes across Q/K/V or silently
upcasts — breaking FA2 eligibility.

**Fix:** Explicit cast before multiply:

```python
cos_full = cos.repeat_interleave(2, dim=-1).to(x.dtype)
sin_full = sin.repeat_interleave(2, dim=-1).to(x.dtype)
return x * cos_full + x_rotated * sin_full
```

**Impact:** Q and K remain BF16 through attention; required for production
`dtype: bf16` config. Documented in [rotary.md](rotary.md).

**Files:** `models/rotary.py` (`apply_rope`)

---

## 4. MoE dispatch (OPT-5, OPT-24)

### OPT-5 — Stacked expert dispatch (default `moe_dispatch="stacked"`)

**Problem:** Per-expert Python loops over tokens scatter memory access and launch
many small GEMMs — poor GPU utilization at MoE scale.

**Fix:** `MoELayer._dispatch_vectorized`:

1. Flatten tokens; router produces top-2 indices + weights.
2. `argsort(flat_idx, stable=True)` groups tokens by expert.
3. For each expert with assigned tokens, run `SwiGLUExpert` on the gathered chunk.
4. `index_add` weighted outputs back to token positions.

Stable sort ensures reproducible routing (required by `AGENTS.md` §4).

**Impact:** Default path for all training runs. Groups tokens so each expert GEMM
has reasonable $M$ dimension. See [moe.md](moe.md) for dispatch diagrams.

**Files:** `models/moe.py` (`_dispatch_vectorized`, `MoELayer.forward`)

**Profile:** `scripts/profile_moe.py`

---

### OPT-24 — Triton grouped GEMM (`moe_dispatch="triton_grouped"`)

**Problem:** Even vectorized dispatch launches separate W1/W3/silu kernels per
expert chunk. Fusing W1+W3+silu into one Triton kernel reduces launch overhead.

**Fix:** Opt-in via config — **not** enabled by default:

```yaml
model:
  moe_dispatch: triton_grouped   # default is "stacked"
```

`models/moe_triton.py` implements `triton_moe_w1w3_silu` with tile shape
$(B_T=16) \times (B_M=32) \times (B_N=32)$. W2 stays PyTorch. Backward uses
pure-PyTorch reference path.

**Impact:** Verified end-to-end on 4 GB GPU (sm_75) via `e2e_gpu_smoke.py`.
Expected 5–15% MoE forward speedup on sm_80+ depending on batch/token count.

**Contract:**

- Must **not** silently fall back during default-config training.
- If Triton unavailable and explicitly requested → clear error.
- Requires unit tests in `tests/test_moe_triton.py` (CPU reference path).

**Files:** `models/moe_triton.py`, `models/moe.py` (`_dispatch_triton`)

**Related:** [triton_kernels.md](triton_kernels.md) (replaces the old standalone MoE Triton doc)

---

## 5. Training loop (OPT-8 – OPT-10, OPT-17 – OPT-21)

### OPT-8 — Gradient clip with `foreach=True`

**Problem:** Default `clip_grad_norm_` loops parameter-by-parameter on large models.

**Fix:**

```python
nn.utils.clip_grad_norm_(model.parameters(), grad_clip, foreach=True)
```

Falls back to non-foreach on older PyTorch (`TypeError` catch).

**Impact:** Modest CPU-side speedup on gradient norm computation; more noticeable
with 500M+ parameters and `grad_clip=1.0` every step.

**Files:** `training/pretrain.py`

---

### OPT-9 — AdamW `foreach=True`, `fused=True` on CUDA

**Problem:** Scalar AdamW loop is Python-bound; each param update is a separate kernel.

**Fix:**

```python
optim = AdamW(
    [...],
    foreach=True,
    fused=(dev.type == "cuda"),
)
```

**Impact:** 1.5–2× faster optimizer step vs default loop on A100/H100 (workspace
rule of thumb). Pairs with FP32 master weights under BF16 autocast.

**Files:** `training/pretrain.py`

**Related:** [training.md](training.md) §11

---

### OPT-10 — Bisect shard lookup in `PretrainDataset`

**Problem:** Multi-shard token datasets need $O(\log S)$ shard resolution vs linear
scan over shard metadata for every `__getitem__` call.

**Fix:** Precompute `shard_offsets` prefix array; lookup with:

```python
shard_idx = self._bisect.bisect_right(self.shard_offsets, start) - 1
```

Plus single-shard fast path when the entire window fits in one shard.

**Impact:** Negligible at small $S$; essential at production shard counts (50M
tokens per shard × hundreds of shards). See [data_pipeline.md](data_pipeline.md).

**Files:** `training/pretrain.py` (`PretrainDataset._get_window_sharded`)

---

### OPT-17 — AdamW `eps=1e-6` (not `1e-8`)

**Problem:** BF16 has 7 mantissa bits. Adam's second moment with `eps=1e-8`
underflows to denormal/zero, silently stalling late-stage convergence.

**Fix:** `eps=1e-6` in production AdamW — matches DeepSeek-V3 and LLaMA-3 practice.

**Impact:** Stability improvement, not throughput. Prevents loss plateau artifacts
after ~30K steps on MoE models.

**Files:** `training/pretrain.py`, `configs/pretrain_a100_502m.yaml`

---

### OPT-18 — Warmup 3000 steps

**Problem:** Top-2-of-8 MoE routing is sensitive to early high learning rates;
insufficient warmup causes expert collapse.

**Fix:**

```yaml
warmup_steps: 3000   # 4.9% of 61000 total steps
```

Linear warmup → cosine decay to `min_lr_ratio=0.05` via `make_warmup_cosine_lambda`.

**Impact:** Routing entropy remains healthier in first 5% of training. Industry
MoE standard is 2–5% warmup; 3000 steps replaces an earlier 2000-step default.

**Files:** `configs/pretrain_a100_502m.yaml`, `training/pretrain.py`

**Verify:** `tests/test_training.py::test_lr_schedule_at_warmup_boundary`

---

### OPT-19 — Aux load-balancing `alpha=0.01`

**Problem:** Without aux loss, top-2 routing collapses to one expert — wasted
capacity and training instability.

**Fix:** Standard Switch Transformer aux loss (FP32 softmax internally — OPT-6):

```python
aux_alpha = train_cfg.get("aux_loss_alpha", 0.01)
loss = (ce + aux_alpha * aux_loss) / accum
```

**Deliberate distinction:** DeepSeek-v3-Lite uses aux-loss-free gating; GPT-OSS-Lite
does **not** (AGENTS.md rule 5).

**Impact:** Keeps expert utilization near uniform; $\alpha=0.01$ is small enough
not to dominate CE loss.

**Files:** `models/moe.py` (`aux_load_balancing_loss`), `training/pretrain.py`

**Related:** [moe.md](moe.md)

---

### OPT-20 — `cudnn.benchmark_limit=0` + `preferred_blas_library="cublaslt"`

**Problem:** Default cuDNN autotune evaluates only 10 algorithms; BF16 matmul may
not pick the fastest sm_80 kernel.

**Fix** in `_set_hardware_perf_knobs()`:

```python
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.benchmark_limit = 0      # exhaustive search (one-time cost)
torch.backends.cuda.preferred_blas_library = "cublaslt"
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
```

**Impact:** ~3–5% end-to-end on A100 after first-step warmup. Bit-exact numerics
(same dtype, different kernel selection).

**Files:** `training/pretrain.py` (`_set_hardware_perf_knobs`)

**Related:** [training.md](training.md) §7

---

### OPT-21 — Chunked cross-entropy `chunk_size=8192`

**Problem:** Full-vocab CE at $|\mathcal{V}| = 128000$ materializes large
log-softmax buffers. Smaller chunks reduce peak activation memory; larger chunks
reduce kernel launch count.

**Fix:**

```python
ce = chunked_cross_entropy(logits, target_ids, chunk_size=8192)
```

Loops over flattened $(B \times T)$ positions in 8192-token chunks (was 4096).

**Impact:** At $B=8$, $T=4096$: 4 CE kernel launches instead of 8 (~20 μs/step
saved). Peak CE intermediate ~16 GB — well under A100 80 GB budget.

**Files:** `training/pretrain.py` (`chunked_cross_entropy`)

---

## 6. Inference (OPT-11 – OPT-15, OPT-22)

### OPT-11 — `MixedKVCache` ring buffer (windowed layers)

**Problem:** Storing full $T$ KV for all 12 layers at 128K exceeds GPU memory.

**Fix:** Windowed layers (even indices) use a fixed-size ring buffer of length
$W = 128$:

- Pre-allocate `(B, H, window, D)` tensors once.
- Track `head` pointer and `count` for wrap-around.
- On `get()`, reorder chronologically if `head != 0`.

**Impact:** Windowed KV memory is $O(W)$ not $O(T)$ per layer — 6 layers × 128
tokens vs 6 × 131072 at 128K. Headline **≥1.8×** total KV reduction with global
layers (OPT-12). Verified analytically by `scripts/kv_cache_benchmark.py`.

**Files:** `inference/generate.py` (`MixedKVCache.append`, `get`)

---

### OPT-12 — `MixedKVCache` exponential growth (global layers)

**Problem:** Global layers need full history; reallocating every decode step is
$O(T^2)$ memcpy.

**Fix:** Global layer buffers grow by factor 1.5× when `needed > cur_cap`:

```python
new_cap = max(needed, int(cur_cap * 1.5) + 1)
new_cap = min(new_cap, self._global_cap_tokens)  # default 4M token cap
```

Append in-place when capacity suffices.

**Impact:** Amortized $O(1)$ append per token after occasional growth events.
Decode is $O(1)$ per step per layer (not $O(T)$ recompute).

**Files:** `inference/generate.py` (`MixedKVCache.append` global branch)

---

### OPT-13 — Pre-allocated generation output tensor

**Problem:** `torch.cat` on each decode step allocates a new $(B, T+1)$ tensor.

**Fix:**

```python
output = torch.empty(B, out_total_len, dtype=input_ids.dtype, device=dev)
output[:, :T_prompt] = input_ids
# each step: output[:, T_prompt + step : T_prompt + step + 1] = next_id
```

**Impact:** Removes $O(\text{new\_tokens})$ allocations during generation; improves
decode latency stability in eval loops (`passkey_eval.py`).

**Files:** `inference/generate.py` (`generate`)

---

### OPT-14 — Sink bias clamp cache in `generate()`

**Problem:** `sink_bias.clamp(-10, 15)` every decode step × 12 layers adds
redundant element-wise ops.

**Fix:** Per-attention-module dict keyed by `id(attn)`:

```python
sink_bias_cache: dict = {}
if id(attn) in sink_bias_cache:
    sink_bias_clamped = sink_bias_cache[id(attn)]
else:
    sink_bias_clamped = attn.sink_bias.clamp(SINK_CLAMP_MIN, SINK_CLAMP_MAX)
    sink_bias_cache[id(attn)] = sink_bias_clamped
```

Clamp runs once per generation, not once per token.

**Impact:** Small but free — matters in tight passkey eval loops at 128K.

**Files:** `inference/generate.py` (`_attn_forward_layer`)

**Related:** [ATTENTION_SINKS.md](ATTENTION_SINKS.md) (clamp rationale)

---

### OPT-15 — YaRN $T=1$ fast path

**Problem:** `torch.outer(positions, inv_freq)` for single decode position builds
unnecessary $(1, d/2)$ temporaries.

**Fix:** In `YaRNRoPE.forward`:

```python
if positions.numel() == 1:
    inv_freq = self.inv_freq.to(positions.device)
    pos = positions.item() if positions.dim() == 0 else positions[0].item()
    freqs = inv_freq * float(pos)
    cos = freqs.cos().unsqueeze(0) * self.mscale
    sin = freqs.sin().unsqueeze(0) * self.mscale
```

**Impact:** Faster per-token RoPE during decode; full-sequence path unchanged for
training prefill.

**Files:** `models/yarn.py` (`YaRNRoPE.forward`)

**Related:** [yarn.md](yarn.md), [rotary.md](rotary.md)

---

## 7. Numerical stability (OPT-6, OPT-7)

### OPT-6 — FP32 softmax in MoE router and aux loss

**Problem:** BF16 `softmax` on router logits underflows when one expert dominates;
gradients to cold experts vanish.

**Fix:**

```python
all_probs_f32 = F.softmax(logits.float(), dim=-1)
# aux_load_balancing_loss:
probs_f32 = F.softmax(all_logits.float(), dim=-1)
```

Top-k weights cast back to activation dtype after normalization.

**Impact:** Prevents routing collapse in early training; required companion to
OPT-19 ($\alpha = 0.01$). Documented in AGENTS.md §3.

**Files:** `models/moe.py` (`MoERouter.forward`, `aux_load_balancing_loss`)

---

### OPT-7 — Gradient checkpointing every 3rd layer

**Problem:** Full activation storage for 12 layers × $B=8$ × $T=4096$ × MoE
width exceeds A100 budget.

**Fix:**

```python
model.enable_gradient_checkpointing(every=3)
# GPTOSS.forward:
if use_grad_ckpt and (layer_idx % grad_ckpt_every == 0):
    x, aux = torch.utils.checkpoint.checkpoint(block, x, positions, use_reentrant=False)
```

**Impact:** ~30% extra compute, ~33% lower activation memory (see
`utils/memory.py` `store_factor`). Enables production micro-batch without OOM.

**Files:** `models/transformer.py`, `training/pretrain.py`, `configs/pretrain_a100_502m.yaml`

**Related:** [utils.md](utils.md) §10

---

## 8. Compilation (OPT-16)

### OPT-16 — `torch.compile(max-autotune)` when `compile: true`

**Problem:** Eager PyTorch leaves fusion opportunities on the table for the
12-layer + MoE graph.

**Fix:** Production config:

```yaml
training:
  compile: true
  compile_mode: "max-autotune"
```

Applied in `pretrain.py` after model construction on CUDA. `step_time_a100.py`
mirrors with `--compile` flag for MFU measurement.

**Impact:** First-step compile latency (minutes); steady-state **10–25%** tokens/sec
improvement on A100 depending on CUDA/PyTorch version. Target MFU ≥35%.

**Files:** `training/pretrain.py`, `configs/pretrain_a100_502m.yaml`, `scripts/step_time_a100.py`

**Caveat:** Interacts with gradient checkpointing — both enabled in production;
debug compile issues by temporarily setting `compile: false`.

---

## 9. Quick reference table

| ID | Name | Category | Primary file |
|----|------|----------|--------------|
| OPT-1 | `lru_cache` attention masks | Attention | `models/attention.py` |
| OPT-2 | `repeat_kv` expand+reshape | GQA | `models/attention.py` |
| OPT-3 | Sink via extended K/V + mask | Attention | `models/attention.py` |
| OPT-4 | RMSNorm native dtype | Norm | `models/transformer.py` |
| OPT-5 | Stacked MoE dispatch | MoE | `models/moe.py` |
| OPT-6 | FP32 MoE softmax | Stability | `models/moe.py` |
| OPT-7 | Grad checkpoint every 3 | Memory | `models/transformer.py` |
| OPT-8 | `clip_grad` foreach | Training | `training/pretrain.py` |
| OPT-9 | AdamW foreach+fused | Training | `training/pretrain.py` |
| OPT-10 | Bisect shard lookup | Data | `training/pretrain.py` |
| OPT-11 | KV ring buffer | Inference | `inference/generate.py` |
| OPT-12 | KV exponential growth | Inference | `inference/generate.py` |
| OPT-13 | Prealloc output | Inference | `inference/generate.py` |
| OPT-14 | Sink clamp cache | Inference | `inference/generate.py` |
| OPT-15 | YaRN T=1 fast path | RoPE | `models/yarn.py` |
| OPT-16 | `torch.compile` | Compile | `training/pretrain.py` |
| OPT-17 | AdamW eps=1e-6 | Stability | `training/pretrain.py` |
| OPT-18 | Warmup 3000 steps | Schedule | `configs/pretrain_a100_502m.yaml` |
| OPT-19 | aux_alpha=0.01 | MoE | `training/pretrain.py` |
| OPT-20 | cuDNN limit=0 + cuBLASLt | Hardware | `training/pretrain.py` |
| OPT-21 | CE chunk 8192 | Loss | `training/pretrain.py` |
| OPT-22 | Decode window mask | Attention | `models/attention.py` |
| OPT-23 | `apply_rope` dtype | RoPE | `models/rotary.py` |
| OPT-24 | Triton MoE opt-in | MoE | `models/moe_triton.py` |

---

## 10. Verification commands

```bash
# Attention equivalence (SWA vs reference)
pytest tests/test_attention.py -v

# MoE Triton vs stacked (CPU reference + optional GPU)
pytest tests/test_moe_triton.py -v

# KV headline metric (≥1.8× at 128K)
python3 scripts/kv_cache_benchmark.py

# Full GPU integration
python3 scripts/e2e_gpu_smoke.py

# Component timings
python3 scripts/profile_components.py

# Memory estimator
pytest tests/test_utils.py -v

# Doc lint (no stale patterns)
python3 scripts/check_docs.py
```

After changing `models/attention.py`, **always** run
`test_sliding_window_matches_full` per AGENTS.md rule 4.

---

## 11. Related documentation

| Topic | Document |
|-------|----------|
| Attention masks + sinks | [attention.md](attention.md), [ATTENTION_SINKS.md](ATTENTION_SINKS.md) |
| YaRN / RoPE | [yarn.md](yarn.md), [rotary.md](rotary.md) |
| MoE routing | [moe.md](moe.md) |
| Triton kernel contract | [triton_kernels.md](triton_kernels.md) |
| Training loop | [training.md](training.md) |
| Memory math | [utils.md](utils.md) |
| Benchmark scripts | [scripts.md](scripts.md) |
| System design | [architecture.md](architecture.md) |
| Book index | [README.md](README.md) |

---

<!-- docs:verified 2026-07-31 · fa6f918 -->
