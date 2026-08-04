# Autograd & Gradient Checkpointing — Trading Compute for Memory

> Purpose: from the autograd tape and the memory of the backward graph, to the
> recompute-for-memory tradeoff of gradient checkpointing, and why a 502M-parameter
> model budgets its activations the way it does.
> Sources: `models/transformer.py`, `utils/memory.py`, `training/pretrain.py`.
> Related: [training](../training.md), [architecture](../architecture.md),
> [foundations](../foundations.md), [attention math](attention_math.md),
> [numerics](numerics.md), [kv cache](kv_cache_engineering.md).

## Table of contents

1. [60-second summary](#1-60-second-summary)
2. [Why it matters here](#2-why-it-matters-here)
3. [Intuition](#3-intuition)
4. [The autograd tape](#4-the-autograd-tape)
5. [Backward mechanics: why saved tensors dominate](#5-backward-mechanics-why-saved-tensors-dominate)
6. [Gradient checkpointing: the memory/compute tradeoff](#6-gradient-checkpointing-the-memorycompute-tradeoff)
7. [`use_reentrant=False` semantics](#7-use_reentrantfalse-semantics)
8. [Why `grad_ckpt_every=3` for 12 layers](#8-why-grad_ckpt_every3-for-12-layers)
9. [Code walkthrough: `GPTOSS.forward`](#9-code-walkthrough-gptossforward)
10. [Pitfalls + verify](#10-pitfalls--verify)

---

## 1. 60-second summary

Training needs two passes: a forward pass that computes the loss and a backward
pass that computes gradients. Between them, PyTorch's autograd engine must keep
every tensor a backward formula needs — the *saved tensors*. In a transformer
those saved tensors are the activations: residual streams, norm inputs,
attention weights, MoE intermediates. Their bytes dominate the memory line that
grows with sequence length, and gradient checkpointing attacks exactly that
line.

Gradient checkpointing (also *activation checkpointing*, *recomputation*)
discards a region's activations right after forward, then re-runs that region's
forward inside backward when the gradients need the tensors back. It is a pure
memory/compute trade: fewer live activations in exchange for one extra forward
pass per checkpointed region.

In GPT-OSS-Lite the mechanism lives in `models/transformer.py:GPTOSS.forward`:
every block with `layer_idx % grad_ckpt_every == 0` is wrapped in
`torch.utils.checkpoint.checkpoint(block, x, positions, use_reentrant=False)`.
The training script enables it via `training/pretrain.py:main`, and
`utils/memory.py:estimate_model_memory_gb` accounts for it in the VRAM budget
that `utils/memory.py:assert_fits_in_available_gpu` enforces before training.

## 2. Why it matters here

The GPT-OSS-Lite design concentrates the memory problem. The 501,836,640
parameters (247,032,672 active, 50.8% sparsity — see
[architecture](../architecture.md) for the accounting) occupy a fixed
footprint: FP32 weights at 4 B/param plus the AdamW state (moments $m$, $v$,
FP32 master) at 12 B/param give $501\,836\,640 \times 16$ B ≈ **7.48 GB** that
does not change with sequence length or batch size. Everything else in the
training budget — activations and the KV cache — scales with the context.

Three repo decisions make the activation term the binding one:

- **MoE feed-forward.** Each layer routes each token to top-2 of 8 experts plus
  1 shared expert (see [MoE theory](moe_theory.md)). The estimator in
  `utils/memory.py:estimate_model_memory_gb` models the per-layer activation
  cost as one $d_{model}$-wide residual tensor plus $3 \times 3 = 9$
  $f_{ffn}$-wide MoE intermediates (3 active experts × the three SwiGLU
  matmuls `models/moe.py:SwiGLUExpert.forward` performs). Since
  $f_{ffn} = 1536 = 2 \times d_{model}$, the MoE intermediates are ~18× the
  residual-stream width per token: **~95% of the activation footprint is MoE
  feed-forward, not attention** (§4.2, eq. (10)).
- **Micro-batched gradient accumulation.** Training uses micro-batch 8,
  accumulation 4 (effective 131,072 tokens/step); activations peak at the
  *micro*-batch size because only gradients accumulate across micro-steps
  ([training](../training.md)). Checkpointing keeps that micro-batch peak low
  enough that accumulation stays memory-neutral.
- **128K eval target.** Eval at 131,072 tokens runs under `torch.no_grad` —
  checkpointing is disabled there by design — so eval memory is the KV cache
  story (the measured 2.00× mixed-cache reduction, [attention sinks](../ATTENTION_SINKS.md),
  [kv cache](kv_cache_engineering.md)), while checkpointing is the training-time
  story.

The honest quantitative claim: at the default training config (S = 4,096,
B = 8) running `utils/memory.py:estimate_model_memory_gb` on the real model
reports **20.73 GB without checkpointing vs 17.17 GB with `grad_ckpt_every=3`**
(2.0 GB overhead term; with the estimator's A100 overhead model of 13.6 GB the
same run reports 32.33 GB vs 28.77 GB — the 13.6 GB overhead figure is the
estimator's heuristic `min(13.7, max(2.0, total·0.17))`, not a measurement;
`.benchmarks/` is empty). The model fits an 80 GB A100 either way at the
default config — the strong claim is about scaling: activations are the only
training term linear in $S \times B$ (§4.2), so checkpointing is what keeps
longer-context or larger-batch runs inside the card. Evaluating the same
estimator formula at S = 16,384, B = 8 gives ≈ 66 GB without checkpointing
(~12 GB of slack under the 2 GB safety margin) vs ≈ 52 GB with it.

## 3. Intuition

Think of the backward pass as an auditor re-deriving every intermediate
gradient, and the autograd tape as the evidence locker. Each op in the forward
pass files a receipt — the tensors its backward formula needs. Full training
parks every receipt: the locker holds one activation per layer, and the
auditor walks the stack from the loss back to the input without re-doing any
work. The cost is that the locker holds $L$ activations at once.

Checkpointing changes the audit protocol, not the math. For a checkpointed
segment you throw the receipts away as soon as forward ends. When the auditor
reaches the segment, she re-enacts the forward — recomputes the receipts — and
then proceeds. The locker holds only one segment's receipts at a time; the
price is the re-enactment itself, i.e. one extra forward pass per segment.

A physical picture: $L$ rooms in a hallway, each holding a suitcase of size
$A$. Normal training parks all $L$ suitcases ($L \cdot A$ bytes). Checkpointing
parks one suitcase per checkpoint wall and, on the way back, re-packs the
in-between rooms one corridor at a time — fewer suitcases parked, more walking.

The same idea already exists *inside* attention: the fused SDPA backends do not
save the $T \times T$ attention weights; they recompute them during backward
([attention math](attention_math.md)). Gradient checkpointing is that principle
applied one level up, at block granularity.

## 4. The autograd tape

### 4.1 What the backward graph stores per op

Autograd records the forward as a DAG: each node is a tensor, each edge is an
op, and every op carries a backward function. The backward function needs
specific inputs to compute gradients — the *saved tensors* — and those are
copied into the node at forward time. The rule is per-op and small. For a
matmul $y = xW$ with loss $\mathcal{L}$:

$$
\frac{\partial \mathcal{L}}{\partial W} = x^\top \frac{\partial \mathcal{L}}{\partial y}, \qquad
\frac{\partial \mathcal{L}}{\partial x} = \frac{\partial \mathcal{L}}{\partial y} W^\top \tag{1}
$$

The node saves $x$ (the pre-activation). $W$ is a parameter — not saved
per-op, it lives once in global memory. For an elementwise add, backward is
identity and nothing is saved. For softmax, the backward formula needs the
*output*, not the input:

$$
\frac{\partial \mathcal{L}}{\partial s_i} = p_i \left(\frac{\partial \mathcal{L}}{\partial p_i} - \sum_j p_j \frac{\partial \mathcal{L}}{\partial p_j}\right), \qquad p = \mathrm{softmax}(s) \tag{2}
$$

so a softmax node saves its output $p$. This is the classic example: the
attention-weights tensor is $T \times T$ per head, the quadratic term that
fused SDPA backends avoid (§5). In general: **an op saves exactly the tensors
its backward formula needs that are not parameters.**

### 4.2 Memory accounting: $O(L \cdot S \cdot D)$ activations vs $O(P)$ weights

Per layer, the saved set is a handful of $D$-wide (or $f_{ffn}$-wide) tensors
per token: the layer input, norm outputs, attention pre-activations, and MoE
intermediates. If $c$ is the number of such tensors saved per layer per token
and $\beta$ the bytes per element, activation memory is

$$
M_{\mathrm{act}} = L \cdot c \cdot S \cdot B \cdot D \cdot \beta \;\in\; O(L \cdot S \cdot D) \tag{3}
$$

with $L$ layers, $S$ sequence length, $B$ batch, $D = d_{model} = 768$,
$\beta = 2$ B/elt under BF16 autocast ([numerics](numerics.md)). Weights and
optimizer state are a function of the parameter count $P$ alone:

$$
M_{\mathrm{weights}} = \underbrace{4\,P}_{\text{FP32 weights}} + \underbrace{12\,P}_{\text{AdamW } m,\, v,\, \text{master}} = 16\,P \;\in\; O(P) \tag{4}
$$

The asymmetry is the whole story: doubling the context doubles $M_{\mathrm{act}}$
and leaves $M_{\mathrm{weights}}$ untouched. For this model at B = 8, S = 4,096,
the estimator's activation model decomposes per token as one $D$-wide residual
tensor plus nine $f_{ffn}$-wide MoE intermediates per layer:

$$
m_{\mathrm{token}} = L \left(D + 9 f_{ffn}\right) \beta = 12 (768 + 9 \cdot 1536) \cdot 2 = 350\,208\ \text{B/token} \approx 342\ \text{KB/token} \tag{5}
$$

of which $12 \cdot 9 \cdot 1536 \cdot 2 = 331\,776$ B ≈ 324 KB (95%) is MoE
feed-forward. At S = 4,096, B = 8 that is $350\,208 \times 32\,768 =
11\,475\,615\,744$ B ≈ 10.69 GB (GiB) raw, of which the estimator attributes
the residual term
(0.5625 GB) and the MoE term (10.125 GB) — 10.69 GB total at store factor 1.0.
The $O(L \cdot S \cdot D)$ term is 1.4× the fixed $O(P)$ budget at the training
context, and it is the term that scales.

## 5. Backward mechanics: why saved tensors dominate

The backward pass walks the graph in reverse, applying the chain rule. A tensor
saved by op $k$ cannot be freed until op $k$'s backward runs, so the peak
saved-tensor population is reached *before backward starts* — every layer's
saved set is alive simultaneously. The optimizer only touches gradients
($O(P)$), so backward's memory profile is entirely the saved tensors plus the
incoming gradient tensors (also $O(S \cdot B \cdot D)$ per layer, transient).

Which tensors dominate in this repo:

- **Softmax outputs.** Equation (2) forces the attention-weights tensor to be
  saved. The naive path `models/attention.py:manual_causal_attention` (the
  test oracle) materializes scores in FP32 and saves $p$ of shape
  $T \times T \times B \times H$ (β = 4 B/elt, FP32):

  $$
  |P| = T^2 \cdot B \cdot H \cdot \beta = 4096^2 \cdot 8 \cdot 8 \cdot 4 \approx 4\ \text{GB per layer} \tag{6}
  $$

  — 48 GB across the stack, larger than the whole weight+optimizer budget.
  This is why production routes through `models/attention.py:causal_attention`,
  which calls `F.scaled_dot_product_attention`: on CUDA the fused
  flash/mem-efficient backends never materialize $p$, recomputing it in
  backward instead. That is checkpointing at attention granularity.
- **Norm inputs.** `models/transformer.py:RMSNorm.forward` computes its
  statistics from `x.detach().float()`, so the graph's backward needs the raw
  input $x$ (and the scaled weight) for the final multiply — roughly two
  $S \cdot B \cdot D$ tensors per norm, ~96 MB per block at the training
  config.
- **MoE intermediates.** Each active expert runs three matmuls
  (`models/moe.py:SwiGLUExpert.forward`); the backward of each needs its
  input, so the saved set holds ~9 $f_{ffn}$-wide tensors per token per layer
  — the 10.125 GB term of eq. (5). The router logits and top-k indices
  (`models/moe.py:MoERouter.forward`) are cheap by comparison.

The practical ranking at S = 4,096, B = 8: MoE intermediates (~10.1 GB) ≫
attention (≈0 under fused SDPA, 4 GB/layer on the naive path) > norms and
residual stream (~0.6 GB). "Saved tensors dominate" because every one of these
is sequence-length-proportional, and the MoE design multiplies the per-token
width by 18.

## 6. Gradient checkpointing: the memory/compute tradeoff

### 6.1 The segment model and the $\sqrt{L}$ optimum

Partition the $L$ layers into $G$ contiguous segments of $\ell = L/G$ layers
and checkpoint every segment: store only each segment's *input* activation
($G$ tensors), and during backward recompute a segment's forward to regenerate
its saved tensors one segment at a time. If $A$ is the activation footprint of
one layer, peak activation memory is

$$
M(G) = \underbrace{G \cdot A}_{\text{segment inputs}} + \underbrace{\frac{L}{G} \cdot A}_{\text{one segment's regenerated tensors}} = \left(G + \frac{L}{G}\right) A \tag{7}
$$

Minimizing over $G$ (treating $G$ as continuous):

$$
\frac{dM}{dG} = A\left(1 - \frac{L}{G^2}\right) = 0 \;\Rightarrow\; G^\ast = \sqrt{L}, \qquad M_{\min} = 2\sqrt{L}\,A \tag{8}
$$

Memory drops from $O(L)$ to $O(\sqrt{L})$ layer-units. For $L = 12$:
$G^\ast \approx 3.5$, so $G = 3$ or $4$ gives $7A$ — about 58% of the full
$12A$. This is the classic "checkpoint everything" scheme, and its compute cost
is the worst case: every segment's forward runs twice.

### 6.2 Compute: $1 + 1/g$

A matmul's backward is exactly twice its forward (it computes both $dW$ and
$dx$, each a full matmul), and matmuls dominate transformer FLOPs, so the
step FLOP budget is roughly

$$
F_{\mathrm{fwd}} + F_{\mathrm{bwd}} = 3\,F_{\mathrm{fwd}} \tag{9}
$$

Now checkpoint with period $g$ — layers $0, g, 2g, \dots$ are wrapped, i.e. the
fraction $1/g$ of the stack is recomputed. Recomputation adds $L/g$
layer-passes, exactly $(1/g) \times$ one forward pass. Total forward-layer work
and step FLOPs become

$$
T(g) = \underbrace{\left(1 + \frac{1}{g}\right) F_{\mathrm{fwd}}}_{\text{forward work}} + 2\,F_{\mathrm{fwd}} = \left(3 + \frac{1}{g}\right) F_{\mathrm{fwd}} \tag{10}
$$

The worst case is full per-layer checkpointing, $g = 1$: forward work doubles
("compute rises by one extra forward pass"), total step FLOPs rise by 33%.
With granularity, $g = 3$: forward work is $1.333\times$ and total step FLOPs
rise by $1/9 \approx 11\%$. These are derived layer-pass counts, not measured
wall-clock times — no pretraining run exists.

### 6.3 The estimator's model

`utils/memory.py:estimate_model_memory_gb` encodes a coarser heuristic with the
same shape. Its `store_factor` weights checkpointed layers at 1.0 (they keep
their segment input) and stored layers at 0.5 (they keep half their footprint):

$$
s(g) = \frac{1}{g}\cdot 1 + \left(1 - \frac{1}{g}\right)\frac{1}{2} = \frac{1}{g} + \frac{1}{2}\left(1 - \frac{1}{g}\right), \qquad s(3) = \frac{2}{3} \tag{11}
$$

so the estimator's total training memory is

$$
M_{\mathrm{est}}(g) = \underbrace{16P}_{\text{weights + AdamW}} + \underbrace{\mathrm{KV}}_{\text{conservative full-length term}} + s(g)\,M_{\mathrm{act}} + \underbrace{\mathrm{overhead}}_{\text{2.0 GB CPU / estimator's A100 model}} \tag{12}
$$

At S = 4,096, B = 8: $16P = 7.477$ GB, KV = 0.5625 GB (the estimator sizes
windowed layers at $\max(128, S)$ pre-steady-state, so all 12 layers at $S$),
$M_{\mathrm{act}} = 10.6875$ GB. With $s(3) = 2/3$: activation block 10.69 →
7.13 GB (−33%), total 20.73 → 17.17 GB (−3.56 GB, −17%). The 0.5 weight on
stored layers is a fixed heuristic, not a profiler result.

## 7. `use_reentrant=False` semantics

`torch.utils.checkpoint` has two implementations. The legacy *reentrant* path
runs the function under `torch.no_grad` on detached inputs and re-attaches it
with a custom `torch.autograd.Function` whose backward re-invokes the function.
It works but carries restrictions: at least one input and output must require
grad; detached tensors *inside* the region raise on backward; it recomputes the
whole function even if only part is needed; and it breaks double-backward and
`autograd.backward(..., inputs=...)`.

GPT-OSS-Lite passes `use_reentrant=False`, the recommended implementation
(since torch 2.9 the parameter must be explicit). The non-reentrant path uses
`torch.autograd.graph.saved_tensors_hooks`: the forward records the real graph,
and the hooks trigger segment recomputation lazily when a saved tensor is first
requested during backward. Consequences, all relevant here:

- **No detached-tensor requirement.** Every block contains a `detach()`:
  `models/transformer.py:RMSNorm.forward` computes `x.detach().float().pow(2)`
  for its statistics. Under the reentrant implementation this is precisely the
  forbidden "detached tensor inside the checkpointed region" pattern; the
  non-reentrant path tolerates it.
- **No double-backward issues.** The graph is recorded, so backward can run on
  the region repeatedly and `autograd.backward` with an `inputs` argument
  works.
- **Nested outputs and non-grad args.** `positions` is a plain int64 tensor
  with no gradient, and the block returns a tuple `(x, aux)`; the non-reentrant
  path handles both without the "at least one input must require grad" rule.
- **Early stop.** Recompute stops once the tensors backward actually needs are
  produced, instead of re-running the whole block.
- **State preservation.** Both paths capture the autocast state and RNG state
  at forward and re-enter them during recomputation, so the recomputed forward
  sees the same BF16 policy as the original. This model has no dropout, so
  determinism does not depend on RNG restoration.

The contract in `models/transformer.py:GPTOSS.forward` is therefore: checkpoint
the block as a pure function `(x, positions) -> (x', aux)` with
`use_reentrant=False`, and let autograd decide when to replay it.

## 8. Why `grad_ckpt_every=3` for 12 layers

The effective memory curve is $s(g)$ from eq. (11) against the compute overhead
$1/g$ from eq. (10):

| $g$ | stored layers | $s(g)$ | activation memory (GB) | forward-work overhead |
|---:|---:|---:|---:|---:|
| 1 | 0 | 1.000 | 10.69 | +100% |
| 2 | 6 | 0.750 | 8.02 | +50% |
| **3** | **8** | **0.667** | **7.13** | **+33%** |
| 4 | 9 | 0.625 | 6.68 | +25% |
| 6 | 10 | 0.583 | 6.23 | +17% |
| 12 | 11 | 0.542 | 5.79 | +8% |

Three arguments select 3:

1. **The $\sqrt{L}$ rule.** The uniform-segment optimum is $G^\ast = \sqrt{12}
   \approx 3.5$ segments (eq. (8)), i.e. a period of $12/3.5 \approx 3.4$
   layers. Both 3 and 4 are on the optimum plateau; 3 divides 12 evenly,
   giving exactly 4 checkpointed blocks — layers 0, 3, 6, 9 — which alternate
   windowed/full attention (even layers are SWA(128), odd layers full).
2. **The compute budget.** $g = 3$ caps recomputation at one-third of a forward
   pass, matching the "~33% recomputation overhead" convention in
   [architecture](../architecture.md) §B.9. The marginal gains beyond it shrink:
   $g = 6$ saves only 0.9 GB more than $g = 3$ while pushing stored runs to
   5 layers between checkpoints, and $g = 12$ checkpoints only layer 0 — a
   single boundary with almost no recompute budget spent anywhere else.
3. **Honesty about the heuristic.** By eq. (11), $g = 4$ scores marginally
   better on both axes ($s = 5/8$, +25%). The 0.5 stored-layer weight does not
   grow with stored-run length, so the heuristic under-counts longer runs; 3
   keeps more margin and is the production default (`grad_checkpoint_every: 3`
   in the training config, consumed by `training/pretrain.py:main`). The small
   test config uses `every=2`.

## 9. Code walkthrough: `GPTOSS.forward`

The entire mechanism is in `models/transformer.py:GPTOSS.forward`:

```python
use_grad_ckpt = (
    getattr(self, "gradient_checkpointing", False)
    and torch.is_grad_enabled()
)
grad_ckpt_every = max(1, getattr(self, "grad_ckpt_every", 3))

for layer_idx, block in enumerate(self.blocks):
    if use_grad_ckpt and (layer_idx % grad_ckpt_every == 0):
        x, aux = torch.utils.checkpoint.checkpoint(
            block,
            x,
            positions,
            use_reentrant=False,
        )
    else:
        x, aux = block(x, positions)
    aux_losses.append(aux)

if aux_losses:
    aux_loss = torch.stack(aux_losses).mean()
```

Walkthrough:

- **The double guard.** `gradient_checkpointing` and `grad_ckpt_every` are
  runtime attributes injected by `models/transformer.py:GPTOSS.enable_gradient_checkpointing`
  — they are not `models/transformer.py:ModelConfig` fields, and `getattr` defaults make the model
  trainable as-is: `False` → no checkpointing, `3` → the production period.
  `torch.is_grad_enabled()` turns the path off under eval and
  `torch.no_grad()` inference, where there is nothing to save. The
  `max(1, ...)` guards the modulo.
- **The checkpoint call.** `block` is a `models/transformer.py:GPTOSSBlock`
  whose forward returns the tuple `(x, aux)`; the checkpoint wrapper accepts
  multi-output callables. `x` is the residual stream — the segment input the
  wrapper keeps alive — and `positions` rides along as a non-grad argument.
  `use_reentrant=False` is explicit (§7).
- **The aux-loss list.** Every block's aux loss is appended, checkpointed or
  not, and the 12 losses are stacked and meaned. The aux tensors produced by
  the *original* forward are the ones entering `aux_losses`; the recomputed
  forward's aux is only consumed internally by the backward. The aux loss
  therefore receives gradients through the checkpointed path like any other
  tensor — no special handling needed with `use_reentrant=False`.
- **BF16 autocast.** The forward runs under
  `with autocast(device_type=dev.type, dtype=torch.bfloat16, ...)` in
  `training/pretrain.py:main` (§6 of [training](../training.md)). The
  checkpoint wrapper records that autocast state and re-enters it during
  recomputation, so the replay sees the identical mixed-precision policy and
  produces identical tensors (the region is deterministic — no dropout, no
  in-place ops).

The wiring lives in `training/pretrain.py:main`:

```python
grad_ckpt = train_cfg.get("grad_checkpoint", True)
if grad_ckpt:
    model.enable_gradient_checkpointing(every=train_cfg.get("grad_checkpoint_every", 3))
```

which sets `self.gradient_checkpointing = True` and `self.grad_ckpt_every =
every` on the model instance. The VRAM guard runs the estimator
(`utils/memory.py:estimate_model_memory_gb` with `grad_checkpoint=True`) and
raises if the estimate exceeds the available GPU minus the 2 GB safety margin
(`utils/memory.py:assert_fits_in_available_gpu`).

## 10. Pitfalls + verify

- **Silent no-op when grad is disabled.** `torch.is_grad_enabled()` is part of
  the guard, so eval and inference never checkpoint. Do not budget eval memory
  with checkpointing — eval memory is KV-cache-dominated ([kv cache](kv_cache_engineering.md)).
- **Runtime attributes, not config.** `grad_ckpt_every` is not a
  `ModelConfig` field; a caller that builds `GPTOSS(cfg)` and never calls
  `models/transformer.py:GPTOSS.enable_gradient_checkpointing` trains without
  checkpointing silently (this was a real regression, guarded by the
  invocation-count tests below).
- **Reentrant traps.** `use_reentrant=True` would break on this model: every
  block contains `x.detach()` inside `models/transformer.py:RMSNorm.forward`,
  the documented reentrant failure mode. Keep `use_reentrant=False` explicit
  (torch ≥ 2.9 raises if omitted).
- **Mutation inside the region.** In-place ops or global-state changes make
  the recomputed forward differ from the original → silently wrong gradients.
  `GPTOSSBlock` is pure; keep it that way.
- **The estimator is a heuristic.** The 0.5 stored-layer weight (eq. (11)) and
  the overhead model (2.0 GB CPU / `min(13.7, max(2.0, total·0.17))` on GPU)
  are budgeting heuristics, not profiler output; the estimator also assumes
  fused SDPA backends (no $T^2$ term). `.benchmarks/` is empty — treat every
  derived GB figure as a model, and the 16–20 h / 35–40% MFU A100 figures
  elsewhere as `[INFERENCE]`.
- **`torch.compile` interplay.** `training/pretrain.py:main` enables
  `torch.compile` (`max-autotune`) on CUDA. Dynamo treats checkpointed regions
  specially and may inline recomputation; keep the region deterministic so the
  compiled replay matches. This is version-sensitive — re-verify after a torch
  upgrade.

Verification commands (all CPU-runnable):

- `pytest tests/test_models.py -v` — `test_gradient_checkpointing_runs`
  (forward + backward with checkpointing on), `test_gradient_checkpointing_actually_checkpoints`
  (spies on `torch.utils.checkpoint.checkpoint`; `every=1` must wrap all 4
  layers of the small config), `test_gradient_checkpointing_skip_layers`
  (`every=2` must wrap exactly half).
- `pytest tests/test_utils.py -v` — `test_estimate_with_grad_ckpt_every`
  (every=2 estimate must exceed every=3).
- `pytest tests/test_training.py -v` — training-loop integration: chunked CE
  under autocast, aux-loss accumulation, NaN guard, checkpoint round-trip.
- Full suite baseline: 190 passed / 2 skipped (GPU-gated Triton).
- `python3 scripts/check_docs.py` and `python3 tests/test_doc_refs.py` validate
  this chapter's links and anchors; run after any edit.

<!-- docs:verified 2026-08-04 · 5da1a80 -->
