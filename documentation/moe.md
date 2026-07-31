# Mixture-of-Experts — GPT-OSS-Lite

## A From-Scratch Technical Reference

> **Prerequisites:** [architecture.md](architecture.md) (transformer block layout),
> [foundations.md](foundations.md) if present.

> **Implementation:** [`models/moe.py`](../models/moe.py) · optional Triton path
> [`models/moe_triton.py`](../models/moe_triton.py) · integration in
> [`models/transformer.py`](../models/transformer.py).

> **Related:** [training.md](training.md) (aux loss weight α=0.01 in the objective),
> [`AGENTS.md`](../AGENTS.md) (Triton kernel contract §1).

---

## Table of Contents

1. [Abstract](#abstract)
2. [Why MoE in GPT-OSS-Lite](#why-moe-in-gpt-oss-lite)
3. [SwiGLU — The Expert Building Block](#swiglu--the-expert-building-block)
4. [Sparse Routing — Theory](#sparse-routing--theory)
5. [Auxiliary Load-Balancing Loss](#auxiliary-load-balancing-loss)
6. [GPT-OSS-Lite MoE Topology](#gpt-oss-lite-moe-topology)
7. [Class Reference — `SwiGLUExpert`](#class-reference--swigluexpert)
8. [Class Reference — `MoERouter`](#class-reference--moerouter)
9. [Function Reference — `aux_load_balancing_loss`](#function-reference--aux_load_balancing_loss)
10. [Class Reference — `MoELayer`](#class-reference--moelayer)
11. [Dispatch Paths — Stacked vs Triton Grouped](#dispatch-paths--stacked-vs-triton-grouped)
12. [Sanctioned Triton path (`moe_dispatch="triton_grouped"`)](#sanctioned-triton-path-moe_dispatchtriton_grouped)
13. [Shared Experts](#shared-experts)
14. [Integration in `GPTOSS`](#integration-in-gptoss)
15. [Parameter and FLOP Accounting](#parameter-and-flop-accounting)
16. [Numerical Stability](#numerical-stability)
17. [Comparison with DeepSeek-v3-Lite](#comparison-with-deepseek-v3-lite)
18. [Debugging Checklist](#debugging-checklist)
19. [Appendix A — Worked routing example](#appendix-a--worked-routing-example)
20. [Appendix B — Dispatch layout diagram](#appendix-b--dispatch-layout-diagram)
21. [Appendix C — Gradient flow](#appendix-c--gradient-flow)
22. [Appendix D — Glossary](#appendix-d--glossary)
23. [Load-Bearing Invariants](#load-bearing-invariants)
24. [References](#references)

---

## Abstract

GPT-OSS-Lite replaces the dense feed-forward network (FFN) in every transformer
block with a **Mixture-of-Experts (MoE)** layer. Each token is routed to
**top-2 of 8 routed experts** plus **1 shared expert** that always runs. The
router uses **softmax gating in FP32**, renormalises the top-k weights to sum
to 1, and trains with a **standard Switch/GShard auxiliary load-balancing
loss** (α = 0.01) — deliberately *not* DeepSeek-V3's auxiliary-loss-free gate.

The implementation is raw PyTorch in [`models/moe.py`](../models/moe.py). An
optional Triton fused kernel ([`models/moe_triton.py`](../models/moe_triton.py))
accelerates the W1/W3+silu stage when `moe_dispatch="triton_grouped"` is set
in [`ModelConfig`](../models/transformer.py).

---

## Why MoE in GPT-OSS-Lite

A dense SwiGLU FFN at `d_model=768`, `ffn_dim=1536` stores:

```
Params per layer  ≈ 3 × d × d_ff = 3 × 768 × 1536 ≈ 3.5 M
Active FLOPs/token ≈ 6 × d × d_ff  (three matmuls at full width)
```

With 12 layers, dense FFNs would dominate parameter count. MoE trades **stored
capacity** for **sparse activation**:

| Quantity | Dense (hypothetical) | GPT-OSS MoE (actual) |
|---|---|---|
| Routed experts stored | 1 | 8 |
| Routed experts active per token | 1 | 2 |
| Shared experts (always on) | 0 | 1 |
| Expert width `ffn_dim` | 1536 | 1536 |
| Router params per layer | 0 | `d × 8` |

**Headline benefit:** ~502 M total parameters with ~247 M **active** per
forward pass — the model can specialise experts without paying full dense FFN
cost on every token.

MoE also pairs naturally with GPT-OSS's long-context design: sliding-window /
full-attention alternation saves KV-cache memory; MoE saves FFN compute while
retaining capacity for reasoning-heavy corpora (code + math in the data mix).

---

## SwiGLU — The Expert Building Block

SwiGLU (Shazeer, 2020) is a gated linear unit used in LLaMA, PaLM, and most
modern LLMs. For input `x ∈ ℝ^d`:

```
SwiGLU(x) = W2 · (silu(W1 · x) ⊙ (W3 · x))
```

where `silu(t) = t · σ(t)` is the sigmoid linear unit.

### Why three matrices?

A standard GLU uses two projections (gate and value). SwiGLU splits the "up"
projection into **W1** (gate) and **W3** (value), then multiplies after the
nonlinearity:

```
gate  = silu(W1 x)     ∈ ℝ^{d_ff}
value = W3 x           ∈ ℝ^{d_ff}
hidden = gate ⊙ value  ∈ ℝ^{d_ff}
out    = W2 hidden     ∈ ℝ^d
```

This matches the LLaMA/PaLM convention and gives slightly better quality than
ReLU-gated variants at similar cost.

### Parameter layout in code

[`SwiGLUExpert`](../models/moe.py) stores three `nn.Linear` layers, all
**bias=False** (RMSNorm handles scaling; bias is omitted for parameter
efficiency):

| Layer | Shape | Role |
|---|---|---|
| `w1` | `(ffn_dim, d_model)` | Gate projection |
| `w2` | `(d_model, ffn_dim)` | Down projection |
| `w3` | `(ffn_dim, d_model)` | Value projection |

Forward (verbatim from source):

```python
return self.w2(F.silu(self.w1(x)) * self.w3(x))
```

### FLOPs per expert forward

For one token, one expert:

```
FLOPs ≈ 2 × d × d_ff   (W1, W3 matmuls)
      + 2 × d_ff × d   (W2 matmul)
      ≈ 4 × d × d_ff   (elementwise ops negligible)
```

With `d=768`, `d_ff=1536`: ~4.7 M FLOPs per expert per token.

---

## Sparse Routing — Theory

### The routing problem

Given hidden state `h_t ∈ ℝ^d` for token `t`, the router must:

1. Score all `N` routed experts.
2. Select `k` experts (`k = n_activated_experts`).
3. Produce weights `w_{t,i}` for the weighted sum of expert outputs.

### Softmax gating (GPT-OSS-Lite)

Unlike DeepSeek-V3 (sigmoid + bias buffer), GPT-OSS-Lite uses **softmax over
all routed experts**:

```
logits_t = h_t · W_gate^T        ∈ ℝ^N
probs_t  = softmax(logits_t)     ∈ ℝ^N   (computed in FP32)
ℐ_t      = TopK(probs_t, k)      indices of k largest
w_{t,i}  = probs_{t,i} / Σ_{j∈ℐ_t} probs_{t,j}   (renormalised on top-k)
```

**Renormalisation** ensures `Σ_{i∈ℐ_t} w_{t,i} = 1` even though softmax
originally summed over all `N` experts. Only the top-k slice is used in the
forward pass, but weights are **re-scaled** so the routed contribution has unit
mass.

### Why FP32 softmax?

Router logits are produced in the model's native dtype (typically BF16 during
training). BF16 has only 7 mantissa bits; when one expert's logit dominates,
`softmax` in BF16 can **underflow** smaller probabilities to zero, starving
gradients through the gate. Computing `F.softmax(logits.float(), dim=-1)` in
FP32 before top-k selection is a standard MoE stability fix (also used in
Switch Transformer training recipes).

### Top-k selection

`topk_weights, topk_indices = all_probs_f32.topk(n_activated, dim=-1)`

For GPT-OSS-Lite: `N=8`, `k=2`. Each token activates exactly two routed
experts (unless numerical edge cases in topk — indices are always in
`[0, N-1]`).

### Routed output

For flattened tokens `t = 1…N_tokens`:

```
y_t^routed = Σ_{i∈ℐ_t} w_{t,i} · Expert_i(h_t)
```

Implementation gathers tokens per expert (see [Dispatch Paths](#dispatch-paths--stacked-vs-triton-grouped)),
runs the expert forward on contiguous chunks, scales by routing weights, and
accumulates back into the token output buffer with `index_add_`.

---

## Auxiliary Load-Balancing Loss

### The collapse problem

Without balancing, routing positive feedback collapses capacity:

```
Expert 0 gets more tokens → adapts faster → gate routes more to Expert 0
→ other experts starve → effective model shrinks to 1–2 experts
```

### Switch / GShard formulation

GPT-OSS-Lite implements the **standard auxiliary loss** from Switch
Transformer and GShard (Fedus et al., 2021; Lepikhin et al., 2020):

```
f_e = (1 / (N_tokens × k)) × count_e     fraction of top-k slots assigned to expert e
P_e = mean_t probs_{t,e}                   mean router probability for expert e
L_aux = N × Σ_e f_e × P_e
```

where `N = n_routed_experts` (8 in the default config).

### Implementation walkthrough

From [`aux_load_balancing_loss`](../models/moe.py):

```python
probs_f32 = F.softmax(all_logits.float(), dim=-1)
N = probs_f32.size(0)
topk_idx = probs_f32.topk(n_activated, dim=-1).indices.flatten()
f = torch.bincount(topk_idx, minlength=n_experts).to(torch.float32) / float(N * n_activated)
P = probs_f32.mean(dim=0)
return (n_experts * (f * P).sum()).to(all_logits.dtype)
```

**Interpretation:**

- `f_e` measures **actual dispatch frequency** (hard assignment).
- `P_e` measures **router's soft preference** for expert `e`.
- The product `f_e × P_e` is high when an expert is both **selected often**
  and **preferred by the gate** — the loss pushes this toward uniformity.
- Scaling by `N` keeps magnitude comparable across different expert counts.

### Training objective

From [`training/pretrain.py`](../training/pretrain.py):

```
L_total = L_CE + α × L_aux
```

with `α = aux_loss_alpha = 0.01` (Switch Transformer default for top-k MoE).

Per micro-batch, the loss is **divided by gradient accumulation steps** before
`backward()`:

```python
loss = (ce + aux_alpha * aux_loss) / accum
```

`aux_loss` returned from the model is the **mean across all 12 MoE layers**
(see [Integration in `GPTOSS`](#integration-in-gptoss)).

### FP32 internal computation

Like the router forward, `aux_load_balancing_loss` computes `softmax` and
`bincount` statistics in FP32, then casts the scalar result back to the
activation dtype. This prevents BF16 underflow when the router saturates on
one expert during early training.

---

## GPT-OSS-Lite MoE Topology

Default config ([`configs/pretrain_a100_502m.yaml`](../configs/pretrain_a100_502m.yaml)):

| Field | Value | Meaning |
|---|---|---|
| `d_model` | 768 | Hidden size |
| `ffn_dim` | 1536 | Expert intermediate width |
| `n_routed_experts` | 8 | Router pool size |
| `n_activated_experts` | 2 | Top-k per token |
| `n_shared_experts` | 1 | Always-on experts |
| `n_layers` | 12 | Every block is MoE (no dense FFN layers) |
| `moe_dispatch` | `"stacked"` (default) | PyTorch loop dispatch |

### Per-token expert activation

Each token through one MoE layer executes:

- **2 routed SwiGLU experts** (weighted sum)
- **1 shared SwiGLU expert** (unweighted sum into output)
- **Router matmul** `768 → 8`

Effective SwiGLU executions per token per layer: **3** (2 routed + 1 shared).

### Layer placement

Every [`GPTOSSBlock`](../models/transformer.py) is:

```
x = x + Attention(RMSNorm(x))
x = x + MoE(RMSNorm(x))     → returns (moe_out, aux_loss)
```

There is no dense FFN alternate — MoE is universal across all 12 layers.

---

## Class Reference — `SwiGLUExpert`

**File:** [`models/moe.py`](../models/moe.py)

```python
class SwiGLUExpert(nn.Module):
    def __init__(self, dim: int, inter_dim: int):
        self.w1 = nn.Linear(dim, inter_dim, bias=False)
        self.w2 = nn.Linear(inter_dim, dim, bias=False)
        self.w3 = nn.Linear(dim, inter_dim, bias=False)
```

| Argument | Default (502M config) | Role |
|---|---|---|
| `dim` | 768 | `d_model` |
| `inter_dim` | 1536 | `ffn_dim` |

**Shapes:** input `(B, T, D)` or `(N, D)` → output same shape.

**Gradients:** all three weight matrices and the input receive gradients on
`backward()`. No special routing mask — shared experts are dense paths.

---

## Class Reference — `MoERouter`

**File:** [`models/moe.py`](../models/moe.py)

```python
class MoERouter(nn.Module):
    def __init__(self, d_model: int, n_experts: int, n_activated: int):
        self.gate = nn.Linear(d_model, n_experts, bias=False)
```

### Forward return tuple

```python
def forward(self, x) -> tuple[Tensor, Tensor, Tensor]:
    # returns (topk_indices, topk_weights, all_logits)
```

| Return | Shape | Dtype | Description |
|---|---|---|---|
| `topk_indices` | `(N, k)` | int64 | Expert indices per token |
| `topk_weights` | `(N, k)` | matches `x` | Renormalised routing weights |
| `all_logits` | `(N, n_experts)` | matches `x` | Pre-softmax logits (for aux loss) |

### Renormalisation

```python
topk_weights = topk_weights / topk_weights.sum(dim=-1, keepdim=True).clamp(min=1e-6)
```

The `clamp(min=1e-6)` guards against a degenerate all-zero top-k slice (should
not occur with softmax, but prevents division by zero).

---

## Function Reference — `aux_load_balancing_loss`

**Signature:**

```python
def aux_load_balancing_loss(
    all_logits: torch.Tensor,   # (N_tokens, n_experts)
    n_experts: int,
    n_activated: int,
) -> torch.Tensor:               # scalar
```

**Inputs:** `all_logits` from the router (not softmaxed — function recomputes
softmax internally in FP32).

**Output:** scalar loss term, same dtype as `all_logits`.

**Differentiability:** gradients flow through `all_logits` via `probs_f32` and
the `P` term. The `f` term uses hard top-k indices (Switch Transformer
standard — straight-through on the soft part).

---

## Class Reference — `MoELayer`

**File:** [`models/moe.py`](../models/moe.py)

### Construction

```python
class MoELayer(nn.Module):
    def __init__(self, cfg):
        self.router = MoERouter(d_model, n_routed, n_activated)
        self.experts = ModuleList([SwiGLUExpert(...) for _ in range(n_routed)])
        self.shared_experts = ModuleList([SwiGLUExpert(...) ...])  # if n_shared > 0
        self.moe_dispatch = getattr(cfg, "moe_dispatch", "stacked")
```

`moe_dispatch` is read from [`ModelConfig.moe_dispatch`](../models/transformer.py)
(default `"stacked"`). Set `"triton_grouped"` to enable the fused kernel path
(see [Sanctioned Triton path](#sanctioned-triton-path-moe_dispatchtriton_grouped)).

### Forward

```python
def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
    # x: (B, T, D)
    # returns (out (B, T, D), aux_loss scalar)
```

**Steps:**

1. Flatten `(B, T, D) → (N, D)` where `N = B × T`.
2. Router → `(indices, weights, all_logits)`.
3. Dispatch routed experts via `_dispatch_vectorized` or `_dispatch_triton`.
4. `aux_loss = aux_load_balancing_loss(all_logits, ...)`.
5. Add shared expert outputs (if configured).
6. Reshape to `(B, T, D)`.

---

## Dispatch Paths — Stacked vs Triton Grouped

Both paths share the same **sort-by-expert** layout. The difference is how
W1/W3+silu is computed inside each expert chunk.

### Common preprocessing

Given `indices (N, k)` and `weights (N, k)`:

```python
flat_idx = indices.reshape(-1)           # N × k slots
flat_w = weights.reshape(-1)
token_ids = arange(N).repeat_interleave(k)

order = argsort(flat_idx, stable=True)    # group slots by expert id
sorted_token_ids = token_ids[order]
sorted_weights = flat_w[order]
sorted_expert_ids = flat_idx[order]

expert_counts = bincount(flat_idx, minlength=n_routed)
expert_offsets = cumsum(counts) with leading zero
```

**Stable argsort** is mandatory for reproducibility ([`AGENTS.md`](../AGENTS.md)
§4).

### Stacked dispatch (`moe_dispatch="stacked"`)

Method: [`_dispatch_vectorized`](../models/moe.py)

For each expert `e` with `cnt > 0` tokens:

```python
chunk_tokens = sorted_token_ids[start:end]
expert_out = self.experts[e](flat[chunk_tokens])   # full SwiGLU
out.index_add(0, chunk_tokens, expert_out * chunk_weights)
```

**Characteristics:**

- Pure PyTorch — runs on CPU, Mac, and CUDA without Triton.
- Each expert chunk calls the full `SwiGLUExpert.forward` (W1, W3, silu, W2).
- Correct reference path for tests and for machines without Triton.

### Triton grouped dispatch (`moe_dispatch="triton_grouped"`)

Method: [`_dispatch_triton`](../models/moe.py) — same sort-by-expert layout as
stacked, but W1/W3+silu runs through the fused Triton kernel. Raises
`ImportError` if Triton is missing (**no silent fallback**). Full kernel
contract: [Sanctioned Triton path](#sanctioned-triton-path-moe_dispatchtriton_grouped).

### Dispatch path selection

| Scenario | Recommended `moe_dispatch` |
|---|---|
| Default training / CPU dev | `"stacked"` |
| A100 production with Triton installed | `"triton_grouped"` |
| Mac / no CUDA Triton | `"stacked"` (required) |
| Debugging routing correctness | `"stacked"` (easier to breakpoint) |

Set in YAML under `model.moe_dispatch` or in `ModelConfig`. There is **no**
environment-variable gate — only the explicit config string.

---

## Sanctioned Triton path (`moe_dispatch="triton_grouped"`)

GPT-OSS-Lite is **raw PyTorch first**. The only sanctioned custom Triton path is
fused grouped-GEMM for MoE W1/W3+silu in
[`models/moe_triton.py`](../models/moe_triton.py). It is **opt-in** via
`ModelConfig.moe_dispatch = "triton_grouped"` (default `"stacked"`). If Triton
is unavailable and `triton_grouped` is requested, the code **raises
`ImportError`** — never silently falls back to PyTorch during a configured
Triton run ([`AGENTS.md`](../AGENTS.md) rule 8).

| File | Entry point | Fuses | Opt-in key |
|---|---|---|---|
| [`models/moe_triton.py`](../models/moe_triton.py) | `triton_moe_w1w3_silu` | W1, W3, silu, mul | `moe_dispatch="triton_grouped"` |

### Why fuse W1/W3+silu (not W2)

Stacked dispatch runs four ops per expert chunk: `W1@x`, `W3@x`, `silu(g)*u`,
`W2@h`. For `B=8`, `T=4096`, top-2 routing yields 65,536 expert slots per
layer — many small forwards even after sorting.

The fused kernel combines W1 matmul + W3 matmul + silu + multiply into one
Triton grid per `(expert, token-tile, d_ff-tile)`:

1. Fewer kernel launches (one grid replaces four PyTorch ops per chunk).
2. Better memory locality — `g` and `u` never materialised as full
   `(tokens, d_ff)` buffers.
3. FP32 accumulation on matmul tiles before silu (matches reference numerics).

W2 is **not fused**: down-projection `d_ff → d_model` has different tiling;
fusing would add complexity for modest gain. Split: Triton up+gate, PyTorch W2.

### Activation and configuration

```yaml
model:
  moe_dispatch: "triton_grouped"   # default is "stacked"
```

```python
if self.moe_dispatch == "triton_grouped":
    out = self._dispatch_triton(flat, indices, weights)
else:
    out = self._dispatch_vectorized(flat, indices, weights)
```

### Public API — `triton_moe_w1w3_silu`

```python
def triton_moe_w1w3_silu(
    x_sorted: torch.Tensor,           # (n_slots, d_model)
    expert_ids_sorted: torch.Tensor,  # (n_slots,)
    counts: torch.Tensor,             # (n_experts,)
    offsets: torch.Tensor,            # (n_experts,)
    W1_stack: torch.Tensor,           # (n_experts, d_ff, d_model)
    W3_stack: torch.Tensor,           # (n_experts, d_ff, d_model)
) -> torch.Tensor:                    # (n_slots, d_ff) = silu(W1@x) * (W3@x)
```

`N_slots = N_tokens × k`. W2 is applied outside by `_dispatch_triton`.

### `HAS_TRITON` import policy and hard failure

```python
try:
    import triton
    HAS_TRITON = True
except ImportError:
    HAS_TRITON = False
```

| `HAS_TRITON` | `moe_dispatch` | Result |
|---|---|---|
| `False` | `"stacked"` | Normal PyTorch dispatch |
| `False` | `"triton_grouped"` | **`ImportError`** at first `triton_moe_w1w3_silu` call |
| `True` | `"stacked"` | Triton never imported by MoE forward |
| `True` | `"triton_grouped"` | Triton forward kernel runs |

Error intent:

```python
raise ImportError(
    "triton_moe_w1w3_silu requires the `triton` package. "
    "Install with `pip install triton` (Linux + CUDA only). "
    "For CPU/Mac, use moe_dispatch='stacked' in your config."
)
```

### Kernel tiling and launcher

| Constant | Value | Role |
|---|---|---|
| `BLOCK_T` | **16** | Tokens per program along expert chunk |
| `BLOCK_M` | **32** | `d_model` reduction tile (K dimension) |
| `BLOCK_N` | **32** | `d_ff` output tile |

Grid: `(n_experts, ceil(max_tokens/BLOCK_T), ceil(d_ff/BLOCK_N))`. Programs
for shorter experts exit early via `tok_mask`. Launcher:

```python
_moe_w1w3_silu_kernel[(n_experts, n_tiles_t, n_tiles_n)](
    ..., BLOCK_T=16, BLOCK_M=32, BLOCK_N=32,
    num_warps=4, num_stages=1,
)
```

Matmul tiles use `allow_tf32=False` inside the kernel for test agreement;
global TF32 still applies to PyTorch W2 outside.

**Hard caps:** `d_ff` and `d_model` must be ≤ 8192 or forward raises
`ValueError` before launch (defaults 768/1536 are well within).

### Autograd — forward Triton, backward PyTorch reference

`_MoEW1W3SiluFunction` runs Triton in `forward`, saves tensors, and in
`backward` recomputes via `_moe_w1w3_silu_reference` (pure PyTorch loop over
experts) with `torch.enable_grad()`. Gradients for `counts`/`offsets`/`ids`
are `None` (discrete routing metadata). Trade-off: backward slower than a
fused backward kernel, but MoE backward is dominated by W2 and attention at
`T=4096`.

### Integration — `MoELayer._dispatch_triton`

```
1. Sort tokens by expert (stable argsort)
2. x_sorted = flat[sorted_token_ids]
3. gated_sorted = triton_moe_w1w3_silu(...)     ← Triton
4. For each expert e:
       out_sorted[slice] = gated_sorted[slice] @ W2_e.T   ← PyTorch
5. out_sorted *= sorted_weights
6. out.index_add_(0, sorted_token_ids, out_sorted)
```

Shared experts still use full `SwiGLUExpert.forward` in PyTorch after step 6.

Even with `triton_grouped`, router, aux loss (α=0.01), W2, routing
`index_add_`, shared experts, attention, RoPE, norms, and loss stay PyTorch.

### Hardware — sm_75 vs sm_80

| GPU | Arch | `num_stages` | Status |
|---|---|---|---|
| GTX 1650 | sm_75 | 1 | Verified via [`scripts/e2e_gpu_smoke.py`](../scripts/e2e_gpu_smoke.py) |
| A100 80GB | sm_80 | 1 (default) | Production target; `num_stages=2` possible via launcher tweak |
| RTX 5090 | sm_120 | — | Use `stacked` until verified |

Triton requires **Linux + CUDA**. macOS and CPU-only machines must use
`moe_dispatch="stacked"`.

### When to enable

**Enable** on Linux+CUDA with Triton, sm_75+, and MoE dispatch is a profiling
hotspot. **Keep `stacked`** on Mac/CPU, when debugging routing/aux loss, or
when kernel correctness is unverified on your GPU. Default
[`pretrain_a100_502m.yaml`](../configs/pretrain_a100_502m.yaml) leaves
`moe_dispatch` unset → `"stacked"`. Expected speedup modest (~5–15% on
MoE-heavy steps); profile before assuming gains.

### How to verify

```bash
python3 -m pytest tests/test_moe.py tests/test_moe_triton.py -v
# GPU-only cases:
pytest -m gpu tests/test_moe_triton.py -v
```

| Test | CPU? | Purpose |
|---|---|---|
| `test_reference_matches_naive_per_expert_loop` | Yes | Reference correctness |
| `test_triton_moe_raises_when_triton_missing` | Yes | ImportError policy (no fallback) |
| `test_triton_forward_matches_reference` | GPU | Triton vs reference |
| `test_moe_triton_grouped_matches_stacked` | GPU | End-to-end dispatch parity |

---

## Shared Experts

When `n_shared_experts > 0`, [`MoELayer.forward`](../models/moe.py) adds:

```python
shared_out = sum(e(flat) for e in self.shared_experts)
out = out + shared_out
```

**Properties:**

- **No router** — shared experts run on every token.
- **No routing weights** — output is summed directly (not scaled by gate).
- **Always added** after routed dispatch, before reshape.
- Default config: `n_shared_experts = 1`.

**Rationale:** Shared experts capture ubiquitous patterns (syntax, common
function words, formatting) so routed experts can specialise on harder
sub-domains (math, code, long-form prose).

**Gradient flow:** full dense backward through shared SwiGLU — typically
~33% of MoE FFN FLOPs per token (1 of 3 effective expert executions).

---

## Integration in `GPTOSS`

### Block forward

[`GPTOSSBlock`](../models/transformer.py):

```python
x = x + self.attn(self.norm1(x), positions)
moe_out, aux_loss = self.moe(self.norm2(x))
x = x + moe_out
return x, aux_loss
```

### Model-level aux aggregation

[`GPTOSS.forward`](../models/transformer.py) collects per-layer aux losses and
returns the **mean**:

```python
aux_loss = torch.stack(aux_losses).mean()
return logits, aux_loss
```

So `aux_loss` in the training loop is one scalar representing average load-
balancing pressure across all 12 MoE layers.

### Active parameter estimate

[`num_active_parameters`](../models/transformer.py) counts:

- All non-expert params (embed, attention, norms, head, routers).
- Per layer: `(n_activated + n_shared) × 3 × d × d_ff` expert weights +
  `d × n_routed` router weights.

Default: ~247 M active of ~502 M total.

---

## Parameter and FLOP Accounting

### Per-layer MoE parameters

| Component | Formula | Value (768/1536/8/2/1) |
|---|---|---|
| Router `gate` | `d × N` | 6,144 |
| Routed experts (8) | `8 × 3 × d × d_ff` | ~28.3 M |
| Shared experts (1) | `3 × d × d_ff` | ~3.5 M |
| **Total MoE per layer** | | ~31.8 M |

× 12 layers ≈ **382 M** of the ~502 M total (remainder: attention, embed, norms).

### Per-token active FLOPs (one MoE layer, approximate)

| Path | FLOPs |
|---|---|
| Router | `2 × d × N` |
| 2 routed SwiGLUs | `2 × 4 × d × d_ff` |
| 1 shared SwiGLU | `4 × d × d_ff` |
| **Total** | `≈ 12 × d × d_ff + 2dN` |

At `d=768`, `d_ff=1536`: ~14.2 MFLOPs/token/layer for MoE FFN.

---

## Numerical Stability

| Mechanism | Where | Why |
|---|---|---|
| FP32 softmax in router | `MoERouter.forward` | BF16 underflow on saturated gates |
| FP32 aux loss internals | `aux_load_balancing_loss` | Stable `f` and `P` statistics |
| Top-k weight renorm + clamp | `MoERouter.forward` | Unit sum; no div-by-zero |
| `eps=1e-6` in AdamW | `pretrain.py` | BF16-safe optimizer (see [training.md](training.md)) |

MoE-specific: watch **expert histograms** during warmup (first 3000 steps).
Healthy training shows all 8 routed experts receiving >5% of top-k slots by
step ~5000.

---

## Comparison with DeepSeek-v3-Lite

| Feature | GPT-OSS-Lite | DeepSeek-v3-Lite |
|---|---|---|
| Gate activation | Softmax (FP32) | Sigmoid |
| Load balancing | **Aux loss** (α=0.01) | **Aux-loss-free bias buffer** |
| Routed experts | 8, top-2 | 20, top-4 |
| Shared experts | 1 | 1 |
| Expert width | 1536 | 384 (fine-grained) |
| Dense FFN layers | None (all MoE) | 2 dense + 16 MoE |

This distinction is **deliberate** ([`AGENTS.md`](../AGENTS.md) rule 5). Do not
port `AuxLossFreeGate` into GPT-OSS-Lite without an explicit design change.

---

## Debugging Checklist

| Symptom | Likely cause | Fix |
|---|---|---|
| One expert >80% of tokens | Router collapse | Check aux loss is enabled; increase warmup |
| `aux_loss` → 0 instantly | α too low or router frozen | Verify `aux_loss_alpha=0.01` |
| `ImportError: triton` | `triton_grouped` without Triton | Use `moe_dispatch="stacked"` or install Triton |
| Triton forward mismatch | Kernel vs reference drift | `pytest tests/test_moe_triton.py -v` (GPU for Triton tests) |
| Routed vs shared imbalance | Data mix too narrow | Broaden web/code fraction |
| NaN after MoE step | BF16 overflow in attention mask | Check sink bias clamp (attention, not MoE) |
| Dispatch mismatch | Stable argsort disabled | Ensure `stable=True` in argsort |

**Routing histogram** (manual debug snippet):

```python
with torch.no_grad():
    flat = x.view(-1, D)
    idx, w, _ = moe.router(flat)
    hist = torch.bincount(idx.reshape(-1), minlength=8)
    print(hist.float() / hist.sum())
```

---

## Appendix A — Worked routing example

**Setup:** `N_tokens=1`, `d=4`, `n_experts=4`, `k=2`.

```
h = [1.0, 0.0, -1.0, 0.5]
W_gate produces logits = [2.0, 0.5, -1.0, 1.0]
softmax(logits) = [0.576, 0.105, 0.014, 0.305]
top-2 indices = [0, 3]
top-2 weights (raw) = [0.576, 0.305]
renormalised = [0.576/0.881, 0.305/0.881] = [0.654, 0.346]

y = 0.654 × Expert_0(h) + 0.346 × Expert_3(h) + Shared(h)
```

---

## Appendix B — Dispatch layout diagram

```
Tokens:     T0  T1  T2  T3
Top-k=2:    e2  e0  e2  e1   (expert ids)
            e5  e3  e7  e2

Flat slots (N×k = 8):
  slot:  0   1   2   3   4   5   6   7
  tok:   T0  T0  T1  T1  T2  T2  T3  T3
  exp:   e2  e5  e0  e3  e2  e7  e1  e2

After stable argsort by expert:
  exp:   e0  e2  e2  e2  e2  e3  e5  e7
  tok:   T1  T0  T2  T3  T0  T1  T0  T2

Expert loops process contiguous [start:end] ranges per expert id.
```

---

## Appendix C — Gradient flow

**Router:** gradients through renormed top-k weights → selected softmax
entries → `gate` weights → input `h`.

**Routed experts:** gradients through `index_add_` scatter → expert W1/W2/W3
for tokens routed to that expert only.

**Shared experts:** dense gradients on every token.

**Aux loss:** gradients into `all_logits` → `gate` (encourages uniform `P`).

**Checkpointing:** MoE layers inside gradient-checkpointed blocks recompute
forward on backward — router + dispatch run twice per step for checkpointed
layers.

---

## Appendix D — Glossary

| Term | Definition |
|---|---|
| **Routed expert** | Expert selected by top-k router |
| **Shared expert** | Expert that runs on every token |
| **Top-k** | Number of routed experts per token (`n_activated_experts`) |
| **Dispatch** | Grouping tokens by assigned expert for batched execution |
| **Aux loss** | Load-balancing penalty on router statistics |
| **`f_e`** | Hard dispatch frequency for expert `e` |
| **`P_e`** | Mean soft probability for expert `e` |
| **SwiGLU** | `W2(silu(W1x) * W3x)` activation |
| **`moe_dispatch`** | Config key: `"stacked"` or `"triton_grouped"` |
| **HAS_TRITON** | Module-level flag after import attempt |
| **Grouped GEMM** | Batched matmul where groups share weights but not batch index |
| **Sanctioned path** | Kernel listed in AGENTS.md — allowed custom Triton |

---

## Load-Bearing Invariants

1. **Top-k weights sum to 1** per token after renorm.
2. **Stable argsort** in dispatch (`stable=True`).
3. **Shared experts always added** when `n_shared_experts > 0`.
4. **Aux loss uses FP32 softmax** — do not cast to BF16 inside the loss.
5. **`triton_grouped` must not silently fall back** — raise on missing Triton.
6. **Triton backward via PyTorch reference** in v1 — do not skip grad tests.
7. **Do not replace aux loss with bias-buffer balancing** without project-level
   approval (GPT-OSS vs DeepSeek distinction).

---

## References

- Shazeer, *GLU Variants Improve Transformer* (2020) — SwiGLU.
- Fedus et al., *Switch Transformers* (2021) — top-k routing + aux loss.
- Lepikhin et al., *GShard* (2020) — load-balanced MoE at scale.
- [`models/moe.py`](../models/moe.py) — implementation.
- [`models/moe_triton.py`](../models/moe_triton.py) — fused kernel (opt-in).
- [`tests/test_moe_triton.py`](../tests/test_moe_triton.py) — Triton contract tests.
- [training.md](training.md) — α=0.01 in the training loop.

<!-- docs:verified 2026-07-31 · fa6f918 -->
