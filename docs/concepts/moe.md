# GPT-OSS-Lite — Mixture-of-Experts

## A From-Scratch Technical Reference

> **Prerequisites:** [foundations-and-architecture.md](foundations-and-architecture.md) (transformer block layout),
> [foundations-and-architecture.md](foundations-and-architecture.md) if present.

> **Implementation:** [`models/moe.py`](../../models/moe.py) · optional Triton path
> [`models/moe_triton.py`](../../models/moe_triton.py) · integration in
> [`models/transformer.py`](../../models/transformer.py).

> **Related:** [training.md](../training.md) (aux loss weight α=0.01 in the objective),
> [`AGENTS.md`](../../AGENTS.md) (Triton kernel contract §1).

---

---

## Abstract

GPT-OSS-Lite replaces the dense feed-forward network (FFN) in every transformer
block with a **Mixture-of-Experts (MoE)** layer. Each token is routed to
**top-2 of 8 routed experts** plus **1 shared expert** that always runs. The
router uses **softmax gating in FP32**, renormalises the top-k weights to sum
to 1, and trains with a **standard Switch/GShard auxiliary load-balancing
loss** (α = 0.01) — deliberately *not* DeepSeek-V3's auxiliary-loss-free gate.

The implementation is raw PyTorch in [`models/moe.py`](../../models/moe.py). An
optional Triton fused kernel ([`models/moe_triton.py`](../../models/moe_triton.py))
accelerates the W1/W3+silu stage when `moe_dispatch="triton_grouped"` is set
in [`ModelConfig`](../../models/transformer.py).

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

[`models/moe.py:SwiGLUExpert`](../../models/moe.py) stores three `nn.Linear` layers, all
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

### Derivation — softmax, top-2 selection, renormalisation

The router is a single bias-free linear map followed by a softmax
([`models/moe.py:MoERouter.forward`](../../models/moe.py)). Write the gate weight
as $W_g \in \mathbb{R}^{E \times d}$, where $E = n\_routed\_experts = 8$ is
the routed pool size and $d = d\_model = 768$. For token $t$ with hidden state
$x_t \in \mathbb{R}^d$ the logits are $z_t = x_t W_g^{\top} \in \mathbb{R}^E$,
and the softmax turns them into a probability distribution over experts:

$$
p_{t,i} = \frac{e^{z_{t,i}}}{\sum_{j=1}^{E} e^{z_{t,j}}}, \qquad
\sum_{i=1}^{E} p_{t,i} = 1, \tag{1}
$$

computed in FP32 in `MoERouter.forward` (`F.softmax(logits.float(), dim=-1)`)
so that BF16 logits cannot underflow the small probabilities (see
[numerics §8.3](optimizers-and-numerics.md)). Top-k selection takes the
$k = 2$ largest probabilities, defining the selected set
$\mathcal{I}_t = \{ i : p_{t,i} \text{ in top-}k(p_t) \}$. The renormalised
weight of selected expert $i$ is its softmax mass divided by the mass of the
selected set:

$$
w_{t,i} = \frac{p_{t,i}}{S_t}, \qquad S_t = \sum_{j \in \mathcal{I}_t} p_{t,j}, \tag{2}
$$

so $w_{t,\cdot}$ is the softmax distribution $p_{t,\cdot}$ *conditioned on*
$\mathcal{I}_t$: it sums to 1 over the selected experts by construction,
$\sum_{i \in \mathcal{I}_t} w_{t,i} = S_t / S_t = 1$ (guarded by
`tests/test_moe.py:test_router_weights_sum_to_one`). The normalisation is not
cosmetic: $S_t < 1$ in general, so without it the routed output
$y_t^{\text{routed}} = \sum_{i \in \mathcal{I}_t} w_{t,i} E_i(x_t)$ would
shrink or grow with how peaked $p_t$ happens to be. The `clamp(min=1e-6)` in
`MoERouter.forward` guards the degenerate $S_t \to 0$ case — impossible for an
exact softmax (every $p_{t,i} > 0$) but cheap insurance against FP edge cases.
The full derivation of this gating scheme, including why $k = 2$ rather than
$k = 1$, is in [moe theory §5](moe.md).

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

From [`models/moe.py:aux_load_balancing_loss`](../../models/moe.py):

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

From [`training/pretrain.py`](../../training/pretrain.py):

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

### Derivation — f_i, P_i, and why the loss is ≥ 1

Over a micro-batch of $N$ flattened tokens ($N = B \times T$) the two
statistics of the Switch/GShard loss are the hard routing frequency and the
soft mean probability. Let $\mathcal{I}_t$ be token $t$'s top-2 set; the
$N k$ routing slots ($k = n\_activated\_experts = 2$) are each owned by one
expert, and $P_i$ is the batch mean of the gate's soft preferences
([`models/moe.py:aux_load_balancing_loss`](../../models/moe.py)):

$$
f_i = \frac{1}{N k} \sum_{t=1}^{N} \mathbb{1}[i \in \mathcal{I}_t], \qquad
P_i = \frac{1}{N} \sum_{t=1}^{N} p_{t,i}, \qquad
\sum_i f_i = \sum_i P_i = 1. \tag{3}
$$

The code computes these exactly: `torch.bincount(topk_idx, minlength=n_experts) / float(N * n_activated)` is $f_i$ (slot counts over the $Nk$ slots), and `probs_f32.mean(dim=0)` is $P_i$. The auxiliary loss is the $E$-scaled inner product,

$$
\mathcal{L}_{\text{aux}} = E \sum_{i=1}^{E} f_i P_i, \qquad E = n\_routed\_experts, \tag{4}
$$

large exactly when an expert is both frequently selected ($f_i$ high) and
strongly preferred ($P_i$ high) — the collapse signature. The gradient of (4)
flows only through $P$ (the top-k indices are discrete), and with
$\partial P_j / \partial z_{t,i} = \tfrac{1}{N} p_{t,i}(\delta_{ij} - p_{t,j})$
the chain rule gives

$$
\frac{\partial \mathcal{L}_{\text{aux}}}{\partial z_{t,i}} = \frac{E}{N}\, p_{t,i}\Big(f_i - \sum_{j} f_j p_{t,j}\Big), \tag{5}
$$

an automatic negative-feedback loop: an over-loaded expert ($f_i$ above the
probability-weighted average $\langle f \rangle_p$) is pushed *down*, an
under-loaded one up — every token, every step.

**The floor.** The loss trains the router toward $f \approx P$ (hard assignment
tracking soft preference, by (5)), so evaluate (4) on that coupled slice:
$\sum_i f_i P_i = \sum_i f_i^2 = \lVert f \rVert_2^2$, and Cauchy–Schwarz
against the all-ones vector gives

$$
1 = \Big(\sum_i f_i\Big)^2 \le E \sum_i f_i^2 = \mathcal{L}_{\text{aux}}, \tag{6}
$$

with equality iff $f_i = 1/E$ for every $i$. The loss is therefore **≥ 1, and
exactly 1 at uniform routing** — the $E$ scaling normalises the floor. The
tests bracket the span: uniform logits give $f = (\tfrac12, \tfrac12, 0, \dots)$
and $P = (\tfrac18, \dots, \tfrac18)$, so $\mathcal{L}_{\text{aux}} = 8 \cdot
\tfrac18 = 1.0$, while saturated logits ($P_1 \to 1$, top-2 tie-breaking
forces $f_1 = f_2 = \tfrac12$) give $\mathcal{L}_{\text{aux}} \to 8 \cdot
\tfrac12 = 4$ (`tests/test_moe.py:test_aux_loss_low_for_uniform` pins 1.0 vs
~4.0). The full derivation, including the collapse regime and its absorbing
state, is in [moe theory §6](moe.md).

**Why α = 0.01.** With $\mathcal{L}_{\text{aux}} \in [1, 4]$ over the collapse
spectrum, the weighted term is $\alpha \mathcal{L}_{\text{aux}} \in [0.01,
0.04]$ against an initial cross-entropy of $\ln 128000 \approx 11.76$ per
token (a uniform model over the 128K vocabulary): the aux term is **0.085% of
the CE at balance and 0.34% at full collapse**. The gradient scale matches:
from (5), each token's aux gradient on a router logit is bounded by
$\alpha \tfrac{E}{N}\,|p_{t,i}(f_i - \langle f\rangle_p)| \le \alpha E/N =
0.08/N$; at the aggregate level this is the $\mathcal{O}(\alpha E\, p\,
\Delta f) \approx 0.08\, p\, \Delta f$ force per token-event that
[moe theory §6.3](moe.md) compares with the
$\mathcal{O}(1)$ per-token CE gradient. So α = 0.01 keeps the balancing
pressure at roughly one percent of the task signal — large enough to arrest
drift over 61,000 steps, small enough not to fight legitimate specialisation
(an expert that is genuinely best for a region should keep receiving its
tokens; the aux gradient opposes imbalance, not specialisation).

---

## GPT-OSS-Lite MoE Topology

Default config ([`configs/pretrain_a100_502m.yaml`](../../configs/pretrain_a100_502m.yaml)):

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

Every [`GPTOSSBlock`](../../models/transformer.py) is:

```
x = x + Attention(RMSNorm(x))
x = x + MoE(RMSNorm(x))     → returns (moe_out, aux_loss)
```

There is no dense FFN alternate — MoE is universal across all 12 layers.

---

## Class Reference — `SwiGLUExpert`

**File:** [`models/moe.py`](../../models/moe.py)

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

**File:** [`models/moe.py`](../../models/moe.py)

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

**File:** [`models/moe.py`](../../models/moe.py)

### Construction

```python
class MoELayer(nn.Module):
    def __init__(self, cfg):
        self.router = MoERouter(d_model, n_routed, n_activated)
        self.experts = ModuleList([SwiGLUExpert(...) for _ in range(n_routed)])
        self.shared_experts = ModuleList([SwiGLUExpert(...) ...])  # if n_shared > 0
        self.moe_dispatch = getattr(cfg, "moe_dispatch", "stacked")
```

`moe_dispatch` is read from [`ModelConfig.moe_dispatch`](../../models/transformer.py)
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

**Stable argsort** is mandatory for reproducibility ([`AGENTS.md`](../../AGENTS.md)
§4).

### Stacked dispatch (`moe_dispatch="stacked"`)

Method: [`_dispatch_vectorized`](../../models/moe.py)

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

Method: [`_dispatch_triton`](../../models/moe.py) — same sort-by-expert layout as
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
[`models/moe_triton.py`](../../models/moe_triton.py). It is **opt-in** via
`ModelConfig.moe_dispatch = "triton_grouped"` (default `"stacked"`). If Triton
is unavailable and `triton_grouped` is requested, the code **raises
`ImportError`** — never silently falls back to PyTorch during a configured
Triton run ([`AGENTS.md`](../../AGENTS.md) rule 8).

| File | Entry point | Fuses | Opt-in key |
|---|---|---|---|
| [`models/moe_triton.py`](../../models/moe_triton.py) | `triton_moe_w1w3_silu` | W1, W3, silu, mul | `moe_dispatch="triton_grouped"` |

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
| GTX 1650 | sm_75 | 1 | Verified via [`scripts/e2e_gpu_smoke.py`](../../scripts/e2e_gpu_smoke.py) |
| A100 80GB | sm_80 | 1 (default) | Production target; `num_stages=2` possible via launcher tweak |
| RTX 5090 | sm_120 | — | Use `stacked` until verified |

Triton requires **Linux + CUDA**. macOS and CPU-only machines must use
`moe_dispatch="stacked"`.

### When to enable

**Enable** on Linux+CUDA with Triton, sm_75+, and MoE dispatch is a profiling
hotspot. **Keep `stacked`** on Mac/CPU, when debugging routing/aux loss, or
when kernel correctness is unverified on your GPU. Default
[`pretrain_a100_502m.yaml`](../../configs/pretrain_a100_502m.yaml) leaves
`moe_dispatch` unset → `"stacked"`. Expected speedup modest (~5–15% on
MoE-heavy steps); profile before assuming gains.

### How to verify

```bash
python3 -m pytest tests/test_moe.py tests/test_moe_triton.py -v
# GPU tests auto-skip on CPU (skipif gpu_required); run the same command on
# an sm_75+ GPU to exercise the Triton kernel-parity tests
```

| Test | CPU? | Purpose |
|---|---|---|
| `test_reference_matches_naive_per_expert_loop` | Yes | Reference correctness |
| `test_triton_moe_raises_when_triton_missing` | Yes | ImportError policy (no fallback) |
| `test_triton_forward_matches_reference` | GPU | Triton vs reference |
| `test_moe_triton_grouped_matches_stacked` | GPU | End-to-end dispatch parity |

---

## Shared Experts

When `n_shared_experts > 0`, [`models/moe.py:MoELayer.forward`](../../models/moe.py) adds:

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

[`GPTOSSBlock`](../../models/transformer.py):

```python
x = x + self.attn(self.norm1(x), positions)
moe_out, aux_loss = self.moe(self.norm2(x))
x = x + moe_out
return x, aux_loss
```

### Model-level aux aggregation

[`GPTOSS.forward`](../../models/transformer.py) collects per-layer aux losses and
returns the **mean**:

```python
aux_loss = torch.stack(aux_losses).mean()
return logits, aux_loss
```

So `aux_loss` in the training loop is one scalar representing average load-
balancing pressure across all 12 MoE layers.

### Active parameter estimate

[`num_active_parameters`](../../models/transformer.py) counts:

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

### Derivation — stored vs active parameters

Each expert is a bias-free SwiGLU with three matrices of $d \times d_{\text{ff}}$
elements ([`models/moe.py:SwiGLUExpert`](../../models/moe.py): `w1`, `w3` of shape
$(d_{\text{ff}}, d)$, `w2` of shape $(d, d_{\text{ff}})$), so

$$
P_{\text{expert}} = 3\, d\, d_{\text{ff}} = 3 \times 768 \times 1536 = 3538944. \tag{7}
$$

A layer stores $E = 8$ routed experts, $s = 1$ shared expert, and the router's
$d \times E$ gate matrix ([`models/moe.py:MoERouter`](../../models/moe.py)); a
forward pass executes only the $k = 2$ selected routed experts, the $s$ shared
ones, and that same gate:

$$
P_{\text{stored}} = (E + s)\, 3\, d\, d_{\text{ff}} + dE = 9 \times 3538944 + 6144 = 31856640, \tag{8}
$$

$$
P_{\text{active}} = (k + s)\, 3\, d\, d_{\text{ff}} + dE = 3 \times 3538944 + 6144 = 10622976. \tag{9}
$$

The gate is counted in both because it runs on every token and scores every
expert — its $dE = 6144$ parameters are never idle. Twelve layers give
$382279680$ stored / $127475712$ active MoE parameters; the
always-active attention, embedding, and norm machinery adds $119556960$
to each side, reproducing the verified totals of 501,836,640 stored and
247,032,672 active parameters — sparsity

$$
1 - \frac{247032672}{501836640} = 0.5077 \approx 50.8\%. \tag{10}
$$

These are exactly the terms
[`models/transformer.py:GPTOSS.num_active_parameters`](../../models/transformer.py)
counts: per layer $(n\_activated + n\_shared) \times 3 d d_{\text{ff}}$ expert
weights plus $d \times n\_routed$ router weights, on top of all non-expert
parameters.

### Derivation — the dense/sparse FLOP ratio

Counting each multiply-accumulate as 2 FLOPs, one expert forward is three
matmuls of cost $2\, d\, d_{\text{ff}}$ each, i.e. $6\, d\, d_{\text{ff}} =
7077888 \approx 7.1$M FLOPs/token — the same factor-6 convention that
counts parameters. The MoE layer then costs
$(k + s) \cdot 6\, d\, d_{\text{ff}} + 2\, d\, E$ (the router adds $2 d E$ for
one $d \to E$ matmul) versus $(E + s) \cdot 6\, d\, d_{\text{ff}}$ for a dense
FFN of equal stored capacity:

$$
\frac{C_{\text{moe}}}{C_{\text{dense}}^{\text{eq}}} = \frac{(k+s)\, 6\, d\, d_{\text{ff}} + 2\, d\, E}{(E+s)\, 6\, d\, d_{\text{ff}}}
= \frac{k+s}{E+s} + O\Big(\frac{1}{d_{\text{ff}}}\Big)
= \frac{3}{9} \approx \frac{1}{3}, \tag{11}
$$

numerically $21245952 / 63700992 = 0.3335$ — the router's
$12288$ FLOPs are 0.06% of the layer cost. At the model level, forward
FLOPs per token scale as roughly $2N$ for $N$ parameters, so the ratio is
exactly the active-parameter fraction (the always-on attention/embedding
machinery is identical in both counts and cancels):

$$
\frac{2 \times 247032672}{2 \times 501836640} = \frac{247032672}{501836640} = 0.4923, \tag{12}
$$

≈ 494M vs ≈ 1,004M FLOPs/token forward — 49.2% of the dense-equivalent
compute, the same ratio as the parameter split. (Reconciliation with the
tables above: the per-expert FLOP table sums the two up-projections at
$2\, d\, d_{\text{ff}}$ — one FLOP per MAC on W1+W3 — giving $4\, d\, d_{\text{ff}}$
per expert and the ~14.2M layer total; under the uniform 2-FLOP-per-MAC
convention used here and in [moe theory §4](moe.md),
each expert is $6\, d\, d_{\text{ff}}$ and the layer total is ~21.2M. Every
ratio in this section is identical under either convention.)

---

## Numerical Stability

| Mechanism | Where | Why |
|---|---|---|
| FP32 softmax in router | `models/moe.py:MoERouter.forward` | BF16 underflow on saturated gates |
| FP32 aux loss internals | `models/moe.py:aux_load_balancing_loss` | Stable `f` and `P` statistics |
| Top-k weight renorm + clamp | `models/moe.py:MoERouter.forward` | Unit sum; no div-by-zero |
| `eps=1e-6` in AdamW | `pretrain.py` | BF16-safe optimizer (see [training.md](../training.md)) |

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

This distinction is **deliberate** ([`AGENTS.md`](../../AGENTS.md) rule 5). Do not
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

### Derivation — gradients through top-k routing

Let $g_t := \partial \mathcal{L} / \partial y_t$ be the task-loss gradient
with respect to the MoE layer's output. (The shared expert's term is
$z$-independent, so it enters only through $g_t$.) The routed output
$y_t = \sum_{j \in \mathcal{I}_t} w_{t,j} E_j(x_t)$ depends on the logits only
through the renormalised weights of (2). The Jacobian of that renormalisation
([`models/moe.py:MoERouter.forward`](../../models/moe.py)) is the softmax
Jacobian restricted to the selected set:

$$
\frac{\partial w_{t,j}}{\partial z_{t,i}} = w_{t,j}\,(\delta_{ij} - w_{t,i}) \quad (i, j \in \mathcal{I}_t), \qquad
\frac{\partial w_{t,j}}{\partial z_{t,i}} = 0 \quad (i \notin \mathcal{I}_t), \tag{13}
$$

the first identity from $\partial p_{t,j}/\partial z_{t,i} = p_{t,j}(\delta_{ij} - p_{t,i})$ and $w = p/S_t$; the second because neither the numerator $p_{t,j}$ nor the denominator $S_t$ depends on $z_{t,i}$ when $i$ is unselected. Hence the router's task gradient for a *selected* expert is the weight times the advantage over the weight-averaged expert credit,

$$
\frac{\partial \mathcal{L}}{\partial z_{t,i}} = w_{t,i}\Big( g_t^{\top} E_i(x_t) - \sum_{j \in \mathcal{I}_t} w_{t,j}\, g_t^{\top} E_j(x_t) \Big), \qquad i \in \mathcal{I}_t, \tag{14}
$$

and **exactly zero for unselected experts** — the top-k op is discrete, so a token never routes task gradient into an expert it did not select. That is precisely where the aux loss matters for idle experts: (5) gives $\partial \mathcal{L}_{\text{aux}} / \partial z_{t,i} = \tfrac{E}{N} p_{t,i}(f_i - \langle f\rangle_p) \neq 0$ for *every* expert, because softmax probabilities are strictly positive — unselected experts still receive router gradient through the gate weights, even though their own matrices get none:

$$
\frac{\partial \mathcal{L}}{\partial W^{(i)}} = \sum_{t:\, i \in \mathcal{I}_t} w_{t,i}\, \Big(\frac{\partial E_i(x_t)}{\partial W^{(i)}}\Big)^{\!\top} g_t, \tag{15}
$$

a sum over exactly the tokens routed to expert $i$ — autograd scatters the gradient through the `index_add_` in
[`models/moe.py:MoELayer._dispatch_vectorized`](../../models/moe.py) (and its Triton twin
[`models/moe.py:MoELayer._dispatch_triton`](../../models/moe.py)) to only those rows. Two consequences. (i) With $k = 1$ the renormalised weight is the constant $1$ and (13) vanishes: the router would learn nothing from the task loss, a core reason for $k = 2$ (derived in [moe theory §5.3](moe.md), guarded by `tests/test_moe.py:test_router_grad_flow`). (ii) A dead expert ($f_i = 0$, $P_i \to 0$) receives neither term — the absorbing state of [moe theory §7](moe.md). The FP32 softmax keeps (13)–(15) alive when router logits saturate (`tests/test_moe.py:test_aux_loss_robust_to_bf16_saturation`).

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

## Mixture-of-Experts Theory

> **Chapter 3 of the GPT-OSS-Lite theory series.** A dense FFN runs every parameter on every token; a mixture of experts runs a *subset*. This chapter derives the theory of conditional computation, top-k routing, and the Switch/GShard load-balancing loss from first principles, then maps each result onto the implementation in `models/moe.py`. The implementation reference (class-by-class, parameters, FLOP accounting, dispatch layouts) is [moe.md](moe.md); the fused Triton grouped-GEMM is covered in [triton programming](kernels-and-checkpointing.md); the FP32 stability story is [numerics](optimizers-and-numerics.md).

---

---

### 1. 60-second summary

A Mixture-of-Experts (MoE) layer replaces one large dense FFN with a *pool* of smaller experts plus a learned *router* that picks which experts process each token. GPT-OSS-Lite stores 9 SwiGLU experts per layer (8 routed + 1 shared) but activates only 3 per token: the router's top-2 choices plus the always-on shared expert. Storing 9× the expert capacity while computing 3× the expert FLOPs is the whole point — the model keeps dense-scale *capacity* (501.8M parameters) at roughly half the *compute* of a same-size dense model (247.0M active, 50.8% sparsity).

The router is a bias-free linear layer producing 8 logits per token, softmaxed in FP32, truncated to the top-2, and renormalised. Because unconstrained routing collapses onto a few experts (rich-get-richer), training adds the standard Switch/GShard auxiliary loss — the product of each expert's *routing frequency* and *mean gate probability*, scaled by the expert count — weighted by α = 0.01. The code is `models/moe.py` (router, aux loss, layer, dispatch), with an opt-in Triton grouped-GEMM for the W1/W3+SiLU stage (`models/moe_triton.py:triton_moe_w1w3_silu`).

### 2. Why it matters here

GPT-OSS-Lite is budgeted as a ~502M-parameter model that must train on a single A100 80GB (`configs/pretrain_a100_502m.yaml`). Every transformer block is MoE — there is no dense FFN anywhere in `models/transformer.py:GPTOSSBlock` — so the MoE design *is* the model's parameter/FLOP identity, not an add-on. The specific choices matter jointly:

- **Top-2 of 8 + 1 shared** (k=2, E=8, s=1) fixes both the 50.8% sparsity ratio and the per-token FFN FLOP count (section 4).
- **Softmax gating with FP32 statistics** keeps the router trainable and the aux loss honest under BF16 autocast ([numerics](optimizers-and-numerics.md) §8.3).
- **α = 0.01 aux loss** is a deliberate, documented deviation from DeepSeek-V3-Lite's aux-loss-free gate (config comment, AGENTS.md rule 5); it is the mechanism that prevents expert collapse over 61,000 steps.
- The aux loss is returned from every layer and averaged in `models/transformer.py:GPTOSS.forward`, then folded into the objective in `training/pretrain.py:main` as `loss = (ce + aux_alpha * aux_loss) / accum`.

Quality claims are targets, not measurements: no pretraining run has happened, so the ≥85% passkey @128K and the 16–20 h / 35–40% MFU A100 figures are `[INFERENCE]`. What is derived here — parameter counts, FLOP ratios, the aux-loss lower bound, the dispatch layout — is exact and checked against the source.

### 3. Intuition

Think of a consulting firm with eight specialists and one generalist. Every client (token) is handed to the generalist plus the two specialists whose names appear on the client's intake form (the router's top-2). The firm pays salaries (parameters) for all nine, but only three people do billable work on any given client. If the intake desk always routes to the same two specialists, the other six atrophy — the firm is paying for capacity it never uses. The load-balancing loss is a management rule that penalises the intake desk whenever a specialist is both over-booked and over-preferred, keeping the whole pool exercised.

Geometrically: expert outputs live in $\mathbb{R}^{d_{\text{model}}}$; the router partitions the input manifold into regions, each region sending its tokens to a different pair of experts. A dense FFN is the degenerate case with one region and one expert. MoE trades away the smoothness of a single learned function for a *piecewise* function assembled from specialists — more capacity, same per-token compute, at the price of a discrete selection problem that must itself be learned and kept balanced.

### 4. Conditional computation: sparse vs dense FFN economics

### 4.1 FLOPs of one dense SwiGLU FFN

A dense SwiGLU FFN with input width $d$ and intermediate width $d_{\text{ff}}$ computes $W_2(\text{silu}(W_1 x) \odot (W_3 x))$ — three matrix-vector products. Each product of a $d_{\text{ff}} \times d$ matrix with a $d$-vector costs $2 d\, d_{\text{ff}}$ FLOPs (multiply-accumulate counts twice), and $W_2$ costs $2 d_{\text{ff}} d$:

$$
C_{\text{dense}} = \underbrace{2\,d\,d_{\text{ff}}}_{W_1} + \underbrace{2\,d\,d_{\text{ff}}}_{W_3} + \underbrace{2\,d_{\text{ff}}\,d}_{W_2} = 6\,d\,d_{\text{ff}} \quad \text{FLOPs/token/layer}. \tag{1}
$$

For GPT-OSS-Lite's dense-equivalent width ($d = 768$, $d_{\text{ff}} = 1536$): $C_{\text{dense}} = 6 \cdot 768 \cdot 1536 = 7077888 \approx 7.08$M FLOPs per token per layer. The same factor 6 counts parameters: three matrices of $d \times d_{\text{ff}}$ elements (bias-free, [moe.md](moe.md) §3), so each expert stores $3 d\, d_{\text{ff}} = 3538944 \approx 3.54$M parameters.

### 4.2 The MoE budget: stored vs active

The layer stores $E$ routed experts of width $d_{\text{ff}}$ plus $s$ shared experts plus the router's gate matrix $W_g \in \mathbb{R}^{E \times d}$:

$$
P_{\text{stored}} = (E + s) \cdot 3\,d\,d_{\text{ff}} + d\,E, \qquad
P_{\text{active}} = (k + s) \cdot 3\,d\,d_{\text{ff}} + d\,E, \tag{2}
$$

where $k$ is the number of routed experts selected per token. Only the $k$ selected routed experts and the $s$ shared experts execute; the other $E - k$ routed experts contribute zero FLOPs. With $E = 8$, $k = 2$, $s = 1$: per layer, 9 stored experts ($31856640 \approx 31.9$M params including the router) versus 3 active ($10622976 \approx 10.6$M). Across 12 layers plus the always-active attention/embedding machinery, the verified totals are 501,836,640 stored and 247,032,672 active parameters — sparsity $1 - 247032672/501836640 = 0.508$. Note that attention, norms, and embeddings are *always* active; the sparsity lives entirely in the FFN blocks.

### 4.3 The honest comparison: equal capacity, fewer FLOPs

The correct baseline is not the MoE layer against one expert — it is the MoE layer against a *dense FFN of equal stored capacity*. Set the dense width $d_{\text{ff}}' = (E+s)\,d_{\text{ff}}$ so both hold the same number of expert parameters. Then by (1),

$$
\frac{C_{\text{moe}}}{C_{\text{dense}}^{\text{eq}}} = \frac{(k+s) \cdot 6\,d\,d_{\text{ff}}}{6\,d\,(E+s)\,d_{\text{ff}}} = \frac{k+s}{E+s} = \frac{3}{9} = \frac{1}{3}. \tag{3}
$$

The MoE layer performs one-third of the FFN FLOPs of a same-capacity dense FFN: 21,233,664 ≈ 21.2M versus 63,700,992 ≈ 63.7M FLOPs per token per layer. The router adds only $2 d E = 12288$ FLOPs/token — 0.06% of the layer's cost. (If the shared expert were counted as part of the pool, the ratio would be $(k+s)/(E+s)$ with $E$ the routed pool — identical expression, and the point stands: idle experts are free at inference, not at storage.)

At the model level, forward FLOPs per token scale as roughly $2N$ for a transformer with $N$ parameters (two FLOPs per parameter per token, forward; attention is a small correction at these sizes). So the MoE model runs $\approx 2 \times 247$M $= 494$M FLOPs/token forward versus $\approx 2 \times 502$M $= 1004$M for a dense model with the same total parameter count — the active parameter ratio (49.2%) is also the FLOP ratio. The dense-equivalent-quality claim is conditional: capacity is what buys quality, and MoE buys it at 49–67% of the FLOPs, at the cost of the balancing machinery that keeps all 8 routed experts usable (sections 6–7).

### 5. Routing as learned categorical selection

### 5.1 The gate

The router is a single bias-free linear map from the token's residual-stream state to $E$ logits (`models/moe.py:MoERouter`). For token $t$ with hidden state $x_t \in \mathbb{R}^d$ and gate weight $W_g \in \mathbb{R}^{E \times d}$:

$$
z_t = x_t W_g^{\top} \in \mathbb{R}^{E}, \qquad
p_{t,i} = \frac{e^{z_{t,i}}}{\sum_{j=1}^{E} e^{z_{t,j}}}, \tag{4}
$$

so $p_t$ is a categorical distribution over experts. The selection is a learned *soft* preference: $W_g$ is trained by gradient descent, and every token votes for every expert with weight $p_{t,i}$ — but only the top-$k$ experts actually compute.

### 5.2 Top-k truncation and renormalisation

The forward pass keeps the $k$ largest probabilities, then renormalises them to a probability distribution over the selected set $\mathcal{I}_t = \{i : p_{t,i} \in \text{top-}k(p_t)\}$:

$$
w_{t,i} = \frac{p_{t,i}}{\sum_{j \in \mathcal{I}_t} p_{t,j}}, \quad i \in \mathcal{I}_t, \qquad
y_t^{\text{routed}} = \sum_{i \in \mathcal{I}_t} w_{t,i}\, E_i(x_t), \tag{5}
$$

with $E_i$ the $i$-th expert's SwiGLU map. Renormalisation is not cosmetic: $\sum_{j \in \mathcal{I}_t} p_{t,j} < 1$ in general, so without it the routed output's scale would wander with how peaked $p_t$ is. Normalising fixes $\sum_i w_{t,i} = 1$ and keeps the routed contribution at unit mass (guarded by `tests/test_moe.py:test_router_weights_sum_to_one`).

### 5.3 Why top-k (k=2), not top-1

Two distinct failure modes motivate $k \geq 2$.

**Vanishing gate gradient.** The selection $\mathcal{I}_t = \text{argmax}_k$ is a discrete op — no gradient flows through the *index*. All gate learning must flow through the *weights*. With $k = 1$ and renormalisation, $w_{t,i} = p_{t,i}/p_{t,i} = 1$ is a constant, so $\partial y_t^{\text{routed}} / \partial z_t = 0$: the router would receive **zero** gradient from the task loss and could learn only from the aux loss. With $k = 2$, $w_{t,i} = p_{t,i}/(p_{t,i} + p_{t,j})$ genuinely depends on the logits, so $\partial w_{t,i}/\partial z_{t,i} \neq 0$ and the gate is trained by the language-modeling objective itself (`tests/test_moe.py:test_router_grad_flow` asserts exactly this). More selected experts = smoother, lower-variance gate gradients: each token's assignment is spread over $k$ experts, so the per-step gradient of the router is an average over $k$ independent draws instead of one hard bet — the categorical-selection analogue of reducing Monte-Carlo gradient noise by stratified sampling.

**Winner-take-all collapse.** With top-1, the runner-up expert receives no forward gradient; only the single argmax winner updates on each token. Experts that lose the argmax never improve, so they keep losing — the selection degenerates to one or two experts (section 7). Top-2 keeps the runner-up inside the gradient path, so every token trains two experts and no expert needs to *win* to improve.

GPT-OSS-Lite fixes $k = 2$ of $E = 8$ (config fields `n_activated_experts`, `n_routed_experts` in `models/transformer.py:ModelConfig`).

### 6. Load-balancing theory: the Switch/GShard aux loss

### 6.1 The two statistics

Over a batch of $N$ tokens (flattened across batch and sequence), define the **routing frequency** — the fraction of all $Nk$ top-k slots assigned to expert $i$ — and the **mean gate probability**:

$$
f_i = \frac{1}{Nk} \sum_{t=1}^{N} \mathbb{1}[i \in \mathcal{I}_t], \qquad
P_i = \frac{1}{N} \sum_{t=1}^{N} p_{t,i}. \tag{6}
$$

Both are probability vectors: $\sum_i f_i = 1$ (each of the $Nk$ slots has one owner) and $\sum_i P_i = 1$ (each $p_t$ sums to 1, so the batch mean does too). $f$ measures what the router *did* (hard assignment); $P$ measures what the router *prefers* (soft assignment). They are different objects: with saturated logits the router can strongly prefer expert 1 ($P_1 \approx 1$) while top-2 ties force slots onto expert 2 ($f_2 = 1/2$).

The auxiliary loss (Switch Transformer, Fedus et al. 2021; GShard, Lepikhin et al. 2020) is

$$
\mathcal{L}_{\text{aux}} = E \sum_{i=1}^{E} f_i\, P_i, \tag{7}
$$

exactly as implemented in `models/moe.py:aux_load_balancing_loss` (where the code's `N` variable is the token count and `n_experts` is $E$).

### 6.2 Why $E \sum_i f_i P_i$ measures imbalance

**Product structure.** $f_i P_i$ is large exactly when expert $i$ is both *frequently selected* and *strongly preferred*. Two experts that both carry mass contribute; the coupling matters: by the rearrangement inequality, $\sum_i f_i P_i$ is maximised over permutations when $f$ and $P$ are sorted in the same order — i.e. when the same experts lead on both axes, which is precisely the collapse regime. It is minimised when the router prefers experts it does not select, a transient the gradient (below) actively discourages.

**The lower bound (Cauchy–Schwarz).** In the coupled equilibrium the router is trained toward $f \approx P$: hard assignment tracks soft preference (section 6.3 shows the aux gradient drives $f$ and $P$ together). On that slice, $\sum_i f_i P_i = \sum_i f_i^2 = \lVert f \rVert_2^2$, and Cauchy–Schwarz on $f$ against the all-ones vector gives

$$
1 = \Big(\sum_i f_i\Big)^2 \le E \sum_i f_i^2 \quad\Longrightarrow\quad
\mathcal{L}_{\text{aux}} = E \sum_i f_i^2 \ge 1, \tag{8}
$$

with equality iff $f_i = 1/E$ for all $i$. Hence: **the aux loss is ≥ 1, and its minimum, exactly 1, sits at uniform routing.** The scaling by $E$ is what normalises the floor to 1, making the loss's magnitude comparable across expert counts. At the opposite extreme — every token routed to the same two experts with $P$ concentrated on one — $f = (\tfrac12, \tfrac12, 0, \dots)$, $P = (1, 0, \dots)$, and $\mathcal{L}_{\text{aux}} = E \cdot \tfrac12 = 4$ for $E = 8$; in general the collapsed value is $E/k \cdot \max_i P_i$'s dominant term, i.e. the loss spans $[1, E/2]$ here. The test `tests/test_moe.py:test_aux_loss_low_for_uniform` pins exactly this ordering: uniform logits score $\mathcal{L}_{\text{aux}} = 1.0$, collapsed logits score $\approx 4.0$.

**Gradient: negative feedback.** The hard indices make $f$ non-differentiable, so the loss's gradient flows only through $P$ (the `topk().indices` in `models/moe.py:aux_load_balancing_loss` is detached by construction). With $\partial P_j / \partial z_{t,i} = \tfrac{1}{N} p_{t,i}(\delta_{ij} - p_{t,j})$ (the softmax Jacobian, averaged over the batch),

$$
\frac{\partial \mathcal{L}_{\text{aux}}}{\partial z_{t,i}} = \frac{E}{N}\, p_{t,i}\Big(f_i - \sum_j f_j\, p_{t,j}\Big). \tag{9}
$$

The bracketed term is the deviation of expert $i$'s frequency from the probability-weighted average frequency $\langle f \rangle_p$. If $i$ is over-loaded ($f_i > \langle f \rangle_p$), gradient descent pushes $z_{t,i}$ *down* — less soft mass on the busy expert; if under-loaded, $z_{t,i}$ rises. Equation (9) is a per-token, per-step negative-feedback loop on imbalance: it does not wait for collapse to develop, and it pulls $f$ toward the shape of $P$ while $P$ is pulled toward uniformity. (The $\mathcal{L}_{\text{aux}} \ge 1$ floor of (8) is exactly the fixed point of (9): at $f = P = $ uniform, the bracket vanishes.)

### 6.3 Why α = 0.01

The total objective is $\mathcal{L} = \mathcal{L}_{\text{CE}} + \alpha \mathcal{L}_{\text{aux}}$ with $\alpha = 0.01$ (`training/pretrain.py:main`, `aux_loss_alpha` in `configs/pretrain_a100_502m.yaml`). Two quantitative facts justify the scale:

1. **Magnitude budget.** $\mathcal{L}_{\text{aux}} \in [1, 4]$ over the whole collapse spectrum, so the aux term contributes $\alpha \mathcal{L}_{\text{aux}} \in [0.01, 0.04]$. Against the initial cross-entropy — a uniform model over the 128,000-token vocabulary scores $\ln 128000 = 11.76$ per token — the aux term is 0.085% of the CE at balance and 0.34% at full collapse. It is a regulariser, not a competing objective.
2. **Counterforce sizing.** The collapse force is *cumulative*: it grows as experts diverge (section 7). The aux gradient (9) is $\mathcal{O}(\alpha E p \Delta f) \approx 0.08 \cdot p \cdot \Delta f$ per token versus an $\mathcal{O}(1)$ token-level CE gradient — a nudge that acts on *every* token, every step, in a direction the task loss is blind to. α = 0.01 is the Switch Transformer default for top-k MoE, large enough to arrest the drift, small enough not to fight legitimate specialisation (an expert that is genuinely best for a region of input space should keep receiving its tokens — the CE gradient says so; the aux gradient only opposes *imbalance*, not *specialisation*).

The alternative design — DeepSeek-V3's auxiliary-loss-free gate with per-expert bias buffers — is deliberately not used here (AGENTS.md rule 5, noted in the config): it is a more complex mechanism, and the aux-loss route is the documented, test-guarded choice.

### 7. Expert collapse

Without balancing, routing is a rich-get-richer feedback loop. Let $n_i = Nk f_i$ be the number of slots expert $i$ receives in a step. Each slot contributes a full update to $W_1^{(i)}, W_2^{(i)}, W_3^{(i)}$, so expert $i$'s gradient norm scales with $n_i$:

$$
\lVert \nabla_{W^{(i)}} \mathcal{L} \rVert \approx n_i \cdot \lVert g_i \rVert, \tag{10}
$$

where $g_i$ is the per-token gradient of one expert. Experts with high $n_i$ improve faster; improved experts produce lower loss for their regions; the gate, trained to minimise loss, raises $p_{t,i}$ for them; higher $P_i$ and better top-k odds raise $n_i$ further. The loop closes: $\lVert \nabla W^{(i)} \rVert \propto n_i$ is the multiplicative engine, and (10) has no counterterm.

The fixed points are degenerate: a small subset of experts absorbs all tokens while the rest starve toward zero routing probability — *dead experts*. In the worst case the effective model is 2 of 8 experts per layer, i.e. the stored capacity collapses to $k/(E+s) = 2/9 \approx 22\%$ of the FFN parameters actually used, yet **all** 9 experts' parameters still occupy memory and are updated (with ~zero gradient) — wasted capacity and wasted optimizer state. Worse, dead experts never recover: with $f_i = 0$ and $P_i \approx 0$, both the task gradient and the aux gradient (9) vanish for expert $i$, a stable absorbing state.

The aux loss is the counterterm to (10): its gradient (9) is proportional to $p_{t,i}(f_i - \langle f\rangle_p)$ and is *largest exactly when the loop is strongest* — an over-loaded expert gets pushed down every step, not just when collapse is complete. The balancing force also has the right symmetry: it opposes imbalance regardless of which experts are good, so it cannot accidentally shield a genuinely poor expert. The interplay is a saddle: the CE gradient wants concentration on the best experts (specialisation), the aux gradient wants uniformity; α sets where the saddle sits, and the warmup (3,000 of 61,000 steps, 4.9%, deliberately above the 2–5% MoE-standard band per the config comment) lets the router find stable structure before the balancing pressure matters. The guard for the *observable* claim — routing must not collapse over a large batch — is `tests/test_moe.py:test_moe_layer_routes_to_all_experts_over_batch`.

### 8. The shared expert

One expert ($s = 1$) is outside the routing pool entirely: `models/moe.py:MoELayer` adds its output to every token unconditionally, and `tests/test_moe.py:test_moe_layer_shared_expert_active` proves it by zeroing its weights and observing a change. Three reasons:

1. **Guaranteed capacity.** Every token, no matter what the gate does, receives a full $3.54$M-parameter transformation. The routed contribution can be thin (near-uniform $p_t$, small top-2 mass) without the FFN output vanishing; the shared expert is the floor under the layer's representational power. It also bounds the worst-case effect of a bad router: a mistrusted gate can starve routed experts, but not the shared one.
2. **Routing-noise reduction.** The shared output is deterministic in $x_t$ — no gate variance. Writing $y_t = S(x_t) + \sum_{i \in \mathcal{I}_t} w_{t,i} E_i(x_t)$, the stochastic part (the routed sum, whose weights and membership vary across tokens) is superimposed on a stable baseline rather than being the whole signal. This damps the per-token gradient variance that discrete routing injects into the FFN path.
3. **Universal-feature offload.** Features that nearly every token needs (syntactic scaffolding, scale calibration) would otherwise be replicated across all 8 routed experts — 8 redundant copies consuming capacity that could hold distinct specialisations. Routing them through the gate would also waste top-2 slots on low-information choices. One always-on expert absorbs the "average" transform; the routed pool specialises in the residual.

The shared expert costs FLOPs on every token — that is exactly why it is counted in the active budget of section 4 and why the sparsity ratio is 49.2% active rather than 25% (which would be 2-of-8 with no shared expert). DeepSeek-V3-style "1 shared + top-k" is the lineage.

### 9. Grouped and stacked dispatch

### 9.1 The gather layout

Routing produces, per token, $k$ (expert, weight) pairs. Execution needs each expert's input tokens *contiguous* so expert $e$'s weight matrix multiplies one block $X_e$ rather than $k$ scattered rows. The standard layout (both dispatch paths in `models/moe.py:MoELayer`):

1. Flatten the $Nk$ slots: `flat_idx = indices.reshape(-1)`, with `token_ids = arange(N).repeat_interleave(k)` remembering each slot's token.
2. Stable-sort slots by expert id: `order = torch.argsort(flat_idx, stable=True)`. The `stable=True` is load-bearing — equal expert ids keep their token order, so the layout is a deterministic function of the routing (guarded by `tests/test_moe.py:test_moe_dispatch_is_deterministic`).
3. Count and offset: `expert_counts = bincount(flat_idx, minlength=E)`; `expert_offsets` from the cumulative sum. Expert $e$'s chunk is `x_sorted[off_e : off_e + cnt_e]` where `x_sorted = flat[sorted_token_ids]`.
4. Compute per expert, scale by the sorted weights, and scatter back: `out.index_add(0, sorted_token_ids, weighted)` — an exact scatter (each token receives exactly $k$ contributions).

### 9.2 Why not a single bmm

The chunked form is a *grouped GEMM*: $E$ matrix products $X_e W_e^{\top}$ with variable row counts $n_e$. A single `torch.bmm` requires equal batch dimensions, so the stacked-tensor formulation would have to pad every chunk to $\max_e n_e$ rows. The FLOP cost is then

$$
C_{\text{bmm}} = 6\,d\,d_{\text{ff}} \sum_{e=1}^{E} \max_e n_e
\;\ge\; 6\,d\,d_{\text{ff}} \sum_e n_e = 6\,d\,d_{\text{ff}}\, Nk = C_{\text{exact}}, \tag{11}
$$

with equality only when routing is perfectly balanced. Padding waste scales with imbalance — precisely the regime the aux loss fights. The per-expert loop (`models/moe.py:MoELayer._dispatch_vectorized`) is exact: $\sum_e n_e = Nk$ always, because every slot has an owner; the price is $E$ small kernel launches per layer.

### 9.3 Launch-bound economics and the Triton path

The small-chunk regime is GPU-hostile: a GEMM's compute time is $t_c = 6\,d\,d_{\text{ff}}\,n_e / P$ (achieved rate $P$), and with per-launch latency $t_l$ the overhead fraction is

$$
\frac{t_l}{t_c} = \frac{t_l\, P}{6\,d\,d_{\text{ff}}\, n_e} \propto \frac{1}{n_e}. \tag{12}
$$

Halving the chunk size doubles the relative overhead; small chunks are the MoE norm (4096 tokens over 8 experts ≈ 512-token chunks before top-2 splitting). The W1/W3+SiLU stage is two of the expert's three GEMMs, so `models/moe.py:MoELayer._dispatch_triton` hands that stage to a single fused grouped-GEMM launch, `models/moe_triton.py:triton_moe_w1w3_silu`, which consumes the same counts/offsets layout natively: per-expert masked tiles of 16 tokens (no padding, no per-expert launch), W2 staying in PyTorch. The numeric launch-vs-compute ratios, the activation-traffic savings ($\approx 96$ MiB/layer at 4096 tokens), and the tile-shape derivation are in [triton programming](kernels-and-checkpointing.md) §6–7, with A100 rates marked `[INFERENCE]` there (`.benchmarks/` is empty).

The two paths are numerically interchangeable — the Triton path is opt-in via `ModelConfig.moe_dispatch` (default `"stacked"`) and its parity is pinned by the GPU-gated tests in `tests/test_moe_triton.py`; on CPU-only machines those tests skip (repo-wide: 190 passed / 2 skipped).

### 10. Code walkthrough

### 10.1 The expert — `models/moe.py:SwiGLUExpert` / `models/moe.py:SwiGLUExpert.forward`

Three bias-free `nn.Linear`s: `w1` ($d \to d_{\text{ff}}$), `w3` ($d \to d_{\text{ff}}$), `w2` ($d_{\text{ff}} \to d$). The forward is the equation of section 4.1 verbatim:

```python
def forward(self, x: torch.Tensor) -> torch.Tensor:
    return self.w2(F.silu(self.w1(x)) * self.w3(x))
```

`models/moe.py:SwiGLUExpert.forward` is the only place the gated activation is computed in the stacked path; `tests/test_moe.py:test_swiglu_expert_shape` and `test_swiglu_expert_grad_flow` pin its shape and its three weight gradients.

### 10.2 The router — `models/moe.py:MoERouter` / `models/moe.py:MoERouter.forward`

```python
def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    logits = self.gate(x)
    all_probs_f32 = F.softmax(logits.float(), dim=-1)
    topk_weights, topk_indices = all_probs_f32.topk(self.n_activated, dim=-1)
    topk_weights = topk_weights / topk_weights.sum(dim=-1, keepdim=True).clamp(min=1e-6)
    return topk_indices, topk_weights.to(x.dtype), logits
```

Line by line against section 5: `self.gate` is the bias-free $\mathbb{R}^{E \times d}$ map of (4); `.float()` before `softmax` is the FP32-island rule ([numerics](optimizers-and-numerics.md) §8.3); `topk` implements $\mathcal{I}_t$; the division implements the renormalisation of (5) with `clamp(min=1e-6)` guarding the (theoretical) zero-mass case; `topk_weights.to(x.dtype)` casts the FP32 weights back to the activation dtype (BF16 under autocast) while the *indices* stay integer. The raw `logits` are returned so `models/moe.py:aux_load_balancing_loss` recomputes the same distribution from the un-softmaxed values — the aux path never depends on which weights the forward consumed. `tests/test_moe.py:test_router_topk_indices`, `test_router_weights_sum_to_one`, `test_router_indices_in_range`, `test_router_grad_flow` guard each contract.

### 10.3 The aux loss — `models/moe.py:aux_load_balancing_loss`

```python
probs_f32 = F.softmax(all_logits.float(), dim=-1)
N = probs_f32.size(0)
topk_idx = probs_f32.topk(n_activated, dim=-1).indices.flatten()
f = torch.bincount(topk_idx, minlength=n_experts).to(torch.float32) / float(N * n_activated)
P = probs_f32.mean(dim=0)
return (n_experts * (f * P).sum()).to(all_logits.dtype)
```

This is equation (6)–(7) line for line: `bincount` over the flattened top-2 indices counts the $Nk$ slots per expert, divided by $Nk$ → $f_i$; the mean over the batch dimension → $P_i$; the product-sum scaled by `n_experts` ($E$) → $\mathcal{L}_{\text{aux}}$. Two details matter. First, `topk_idx` is produced from `probs_f32.topk(...).indices` — indices are non-differentiable, so the loss's gradient flows only through $P$, exactly the mechanism of (9). Second, everything is computed in FP32 and cast back at the end, so saturated BF16 logits cannot zero out the $P_i$ statistics (`tests/test_moe.py:test_aux_loss_robust_to_bf16_saturation` builds logits of $\pm 100$ and requires a nonzero loss; `test_aux_loss_finite_and_nonneg` and `test_aux_loss_grad_flow` pin finiteness and differentiability).

### 10.4 The layer — `models/moe.py:MoELayer` / `models/moe.py:MoELayer.forward`

```python
def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    B, T, D = x.shape
    flat = x.view(-1, D)
    N = flat.size(0)

    indices, weights, all_logits = self.router(flat)
    if self.moe_dispatch == "triton_grouped":
        out = self._dispatch_triton(flat, indices, weights)
    else:
        out = self._dispatch_vectorized(flat, indices, weights)
    aux_loss = aux_load_balancing_loss(all_logits, self.n_routed, self.n_activated)
    if self.shared_experts is not None:
        shared_out = sum(e(flat) for e in self.shared_experts)
        out = out + shared_out

    return out.view(B, T, D), aux_loss
```

The sequence is exactly the theory: flatten tokens (the aux loss and router operate on the *whole* batch, as (6) requires — batch-level statistics, not per-sequence); route; dispatch (section 9); compute the aux loss from the same logits; add the shared expert's output to every token (section 8); return the aux loss as a second output. The layer is constructed from the config in `models/moe.py:MoELayer.__init__` (`n_routed`, `n_activated`, `n_shared` read off `ModelConfig`). Integration: `models/transformer.py:GPTOSSBlock.forward` returns `(x + moe_out, aux_loss)` and `models/transformer.py:GPTOSS.forward` stacks the 12 per-layer aux losses and takes the mean, so the scalar that reaches `training/pretrain.py:main` is the layer average; `models/transformer.py:GPTOSS.num_parameters` reproduces the section-4 accounting analytically (it computes active expert params as $(k + s) \cdot 3 d d_{\text{ff}}$ per layer, skipping stored-but-idle experts).

### 10.5 Dispatch — `models/moe.py:MoELayer._dispatch_vectorized` and `_dispatch_triton`

Both implement the layout of section 9.1: flatten slots, `torch.argsort(flat_idx, stable=True)`, `bincount`/`cumsum` for `expert_counts`/`expert_offsets`, per-expert contiguous chunks. The vectorized path runs `self.experts[e](expert_in)` per chunk and scatters with `out.index_add(0, chunk_tokens, expert_out * chunk_weights)`. The Triton path replaces the per-expert W1/W3+SiLU with one `triton_moe_w1w3_silu` call over stacked weights and the same counts/offsets, then finishes W2, weight-scaling, and `index_add_` in PyTorch. `tests/test_moe.py:test_moe_layer_dispatch_correct` recomputes the forward by hand per token and requires `allclose(atol=1e-4)` — the ground truth that both dispatch paths and the Triton kernel must match.

### 11. Pitfalls + verify

| Pitfall | Symptom | Guard |
|---|---|---|
| BF16 router softmax | Small $p_{t,i}$ underflow to 0; renormalised weights and $P_i$ statistics distorted; aux loss goes blind to imbalance | FP32 softmax in `models/moe.py:MoERouter.forward` and `models/moe.py:aux_load_balancing_loss`; `pytest tests/test_moe.py -v` → `test_aux_loss_robust_to_bf16_saturation` |
| Renormalisation divide-by-zero | NaN routed output if top-k mass is 0 | `clamp(min=1e-6)` in `MoERouter.forward`; `test_router_weights_sum_to_one` |
| Top-1 gate | Router gets zero task gradient (weights are constant 1); winner-take-all collapse | $k = 2$ keeps logit dependence in the weights; `test_router_grad_flow`, `test_moe_layer_grad_flow` |
| No aux loss (α = 0) | Rich-get-richer: routing concentrates, dead experts, effective capacity $\to k/(E{+}s) = 22\%$ | α = 0.01 objective in `training/pretrain.py:main`; `test_aux_loss_low_for_uniform` (1.0 vs ~4.0), `test_moe_layer_routes_to_all_experts_over_batch` (mass fraction < 0.9) |
| Non-deterministic dispatch | Same input, different output across runs (unordered gather of tied expert ids) | `torch.argsort(flat_idx, stable=True)` in both dispatch paths; `test_moe_dispatch_is_deterministic` (`torch.equal`) |
| Non-finite aux loss | NaN statistics poison the total loss | FP32 `bincount`/`mean`; `test_aux_loss_finite_and_nonneg`, `test_aux_loss_grad_flow` |
| Triton missing on CPU | Kernel call fails | No silent fallback: `triton_grouped` raises `ImportError` (`models/moe_triton.py:triton_moe_w1w3_silu`); CPU suite uses `"stacked"` and skips only the 2 GPU-gated Triton tests |
| Shared expert accidentally routed | Every token loses its guaranteed baseline | Shared output added unconditionally in `MoELayer.forward`; `test_moe_layer_shared_expert_active` (zeroing shared weights must change output) |

**Verification** (CPU-runnable, no GPU required):

```
pytest tests/test_moe.py -v
```

covers every row above — 190 tests pass repo-wide / 2 skipped (the skips are the GPU-gated Triton parity tests, expected on CPU). Numerical-equality with the Triton path is additionally pinned by `pytest tests/test_moe_triton.py -v` on an sm_75+ GPU (GPU tests auto-skip on CPU). Everything derived in this chapter — (1)–(12) — is exact arithmetic on the code and config, except where marked `[INFERENCE]`: no pretraining run exists yet, so the ≥85% passkey @128K and the A100 time/MFU figures are targets and estimates, not measurements.

---

Related: implementation reference [moe.md](moe.md); fused kernel [triton programming](kernels-and-checkpointing.md); FP32 islands [numerics](optimizers-and-numerics.md); training objective and warmup [training.md](../training.md); model layout [foundations-and-architecture.md](foundations-and-architecture.md); hardware budget [foundations-and-architecture.md](foundations-and-architecture.md).

## References


- Shazeer, *GLU Variants Improve Transformer* (2020) — SwiGLU.
- Fedus et al., *Switch Transformers* (2021) — top-k routing + aux loss.
- Lepikhin et al., *GShard* (2020) — load-balanced MoE at scale.
- [`models/moe.py`](../../models/moe.py) — implementation.
- [`models/moe_triton.py`](../../models/moe_triton.py) — fused kernel (opt-in).
- [`tests/test_moe_triton.py`](../../tests/test_moe_triton.py) — Triton contract tests.
- [training.md](../training.md) — α=0.01 in the training loop.

<!-- docs:verified 2026-08-05 · 6491066 -->
