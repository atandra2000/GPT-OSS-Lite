# GPT-OSS-Lite — Optimizers, Numerics, and Sampling

## Part A — Optimizers (From Gradient Descent to AdamW)

> **Chapter T4.** The optimizer is the only component that touches every parameter of a 502M-parameter model on every one of 61,000 steps. This chapter derives the machinery — momentum, Adam's bias correction, decoupled weight decay, mixed-precision state, gradient clipping, warmup — and then reads the exact construction in `training/pretrain.py`.
>
> **Related:** [training.md](../training.md) §11 (optimizer section) · [foundations-and-architecture.md](foundations-and-architecture.md) §10 (BF16 vs FP16) · [numerics](optimizers-and-numerics.md) (IEEE-754 bit layouts) · [moe.md](moe.md) §16 (numerical stability) · [moe theory](moe.md) (routing instability).

---

### 1. Sixty-second summary

The training loop optimizes a ~502M-parameter (247M active) top-2-of-8 MoE decoder with AdamW: first and second moment estimates with bias correction, decoupled weight decay, FP32 optimizer state under a BF16 forward pass, global gradient-norm clipping at 1.0, and a 3000-step linear warmup followed by cosine decay to 5% of peak LR. The optimizer is constructed in `training/pretrain.py:main` with two parameter groups (decay vs no-decay), `foreach=True`, `fused=True` on CUDA, and `eps=1e-6`. Every design choice — β₁=0.9, β₂=0.95, wd=0.1, lr=4e-4, clip=1.0, warmup — exists because this is a short (8B-token), single-pass pretraining run over a sparsely routed model whose gradients mix two objectives (cross-entropy and an auxiliary load-balance loss). No pretraining run has completed yet; all runtime figures are targets.

### 2. Why it matters here

GPT-OSS-Lite budgets exactly 61,000 steps × 131,072 tokens ≈ 8.0B tokens — a Chinchilla-optimal single pass over the corpus. There is no second epoch to recover from a bad optimizer setting. Three properties of this model make the optimizer load-bearing:

- **MoE routing noise.** The router in `models/moe.py:MoERouter` selects 2 of 8
  experts per token. Early in training the gate is near-random, so per-batch gradients through the router are large and the auxiliary loss `models/moe.py:aux_load_balancing_loss` (α=0.01) injects a second gradient signal roughly an order of magnitude smaller than the CE gradient. Adam's per-coordinate scaling and the 3000-step warmup are what keep the gate from collapsing onto one expert (see [moe theory](moe.md)).
- **Weight decay must not touch everything.** RMSNorm gains are scale
  parameters, the sink bias is a learned offset clamped to $[-10, 15]$ (`models/attention.py:SINK_CLAMP_MIN/MAX`), and the embedding is tied to the head. Decaying them is either meaningless or harmful, so the optimizer splits parameters into decay / no-decay groups.
- **BF16 arithmetic.** The forward pass runs in BF16 (7 mantissa bits) but the
  optimizer state runs in FP32. Getting that split wrong silently rounds away most updates (derived in §4.6).

### 3. Intuition

Think of the loss surface as terrain and the optimizer as a hiker with three improvements over naive hill descent. First, a **ball** (momentum): it keeps rolling in the direction it was already going, so it averages out noisy measurements of the slope. Second, a **per-axis ruler** (Adam): instead of one step size for every direction, it measures the typical slope magnitude along each coordinate and takes a step of *relative* size — this is what makes training work when one coordinate's gradient is 1000× another's. Third, a **speed limit and a launch ramp** (clipping + warmup): the hiker never takes a step longer than a fixed bound, and starts at rest, because the first slope measurements are the noisiest ones.

### 4. Theory + derivation

### 4.1 Gradient descent and the fixed-step limit

Let $L(\theta)$ be the training objective and $g_t = \nabla_\theta L(\theta_t)$ the gradient at step $t$. Gradient descent with learning rate $\eta$:

$$
\theta_{t+1} = \theta_t - \eta g_t. \tag{1}
$$

Expand the loss after the step to second order, with Hessian $H_t = \nabla^2_\theta L(\theta_t)$:

$$
L(\theta_t - \eta g_t) = L(\theta_t) - \eta\, g_t^{\top} g_t
+ \tfrac{1}{2}\eta^2 g_t^{\top} H_t g_t + O(\eta^3). \tag{2}
$$

The step decreases the loss only while $\eta < 2\, g^{\top}g / (g^{\top}Hg)$, and the step size that maximizes the decrease is $\eta^* = g^{\top}g / (g^{\top}Hg)$. The trouble is visible in a coordinate-separable quadratic $L = \tfrac{1}{2}\sum_i h_i \theta_i^2$ (hessian eigenvalues $h_i$): gradient descent gives $\theta_i \leftarrow (1-\eta h_i)\theta_i$, which converges only for $\eta < 2/h_{\max}$, and the convergence rate is set by the condition number $\kappa = h_{\max}/h_{\min}$. In contrast, a *per-coordinate* step $\eta_i = 1/h_i$ converges in one step. Real transformer losses have a huge spread of $h_i$ (attention logits vs embedding rows), so a single global $\eta$ is forced to be small enough for the stiffest direction. This is the entire motivation for the adaptive scaling Adam implements.

### 4.2 Momentum: the EMA of gradients

Momentum replaces the raw gradient with an exponential moving average $m_t$ ("velocity"), with decay $\beta_1 \in (0,1)$:

$$
m_t = \beta_1 m_{t-1} + (1-\beta_1) g_t, \qquad m_0 = 0. \tag{3}
$$

Unrolling (3) expresses $m_t$ as a weighted sum of the past gradients:

$$
m_t = (1-\beta_1) \sum_{i=0}^{t-1} \beta_1^{i}\, g_{t-i}. \tag{4}
$$

The weights sum to $\sum_{i=0}^{t-1} (1-\beta_1)\beta_1^i = 1 - \beta_1^t < 1$: at small $t$ the estimate is systematically *too small* — biased toward zero, because the missing early terms are absent rather than zero. For a stationary gradient $g$, $\mathbb{E}[m_t] = (1-\beta_1^t)\,g$, so dividing by $1-\beta_1^t$ removes the bias:

$$
\hat{m}_t = \frac{m_t}{1-\beta_1^t}. \tag{5}
$$

The window of the EMA: the coefficient of $g_{t-k}$ is $\beta_1^k$, which halves at $k = \ln 2 / \ln(1/\beta_1)$ — about 6.6 steps for $\beta_1 = 0.9$, i.e. the velocity averages roughly the last seven gradients.

### 4.3 Adam: per-coordinate adaptive steps

Adam (Kingma & Ba, 2015) maintains a second EMA over *squared* gradients $v_t$, with its own decay $\beta_2$:

$$
m_t = \beta_1 m_{t-1} + (1-\beta_1) g_t, \qquad
v_t = \beta_2 v_{t-1} + (1-\beta_2) g_t^2. \tag{6}
$$

If the gradient components are stationary with $\mathbb{E}[g_i^2] = \nu_i$, the same unrolling argument gives $\mathbb{E}[v_t] = (1-\beta_2^t)\,\nu$, so the *second* moment needs its own correction — this is why the correction exponent must be the *age* $t$, not the step number modulo anything:

$$
\hat{m}_t = \frac{m_t}{1-\beta_1^t}, \qquad
\hat{v}_t = \frac{v_t}{1-\beta_2^t}. \tag{7}
$$

The parameter update is then:

$$
\theta_{t+1} = \theta_t - \eta\, \frac{\hat{m}_t}{\sqrt{\hat{v}_t} + \varepsilon}, \tag{8}
$$

where $\varepsilon > 0$ prevents division by zero and sets the noise floor. Per coordinate, $\sqrt{\hat{v}_{t,i}}$ is the RMS of recent gradients, so (8) implements exactly the per-coordinate scaling $\eta_i \propto 1/\sqrt{h_i}$ of §4.1 — the effective preconditioner is $\text{diag}(\hat{v})^{-1/2}$.

Two consequences are worth deriving because they explain design choices later. First, **scale invariance**: if every gradient were multiplied by $c > 0$ (e.g. rescaling the whole loss), then $\hat{m} \to c\hat{m}$ and $\sqrt{\hat{v}} \to c\sqrt{\hat{v}}$, so the ratio — and the step — is unchanged:

$$
\frac{c\hat{m}}{c\sqrt{\hat{v}}} = \frac{\hat{m}}{\sqrt{\hat{v}}}. \tag{9}
$$

This is why one optimizer can absorb the mixed CE + α·aux objective of this repo: the two terms differ by an order of magnitude in gradient scale, but Adam is invariant to the overall scale of each coordinate's gradient history. Second, **early-steps behavior**: for $t = 1$ the correction gives $\hat{m}_1 = g_1$ and $\hat{v}_1 = g_1^2$, hence $\hat{m}_1 / \sqrt{\hat{v}_1} = \operatorname{sign}(g_1)$ — the very first Adam step moves *every* coordinate by exactly $\pm \eta$, regardless of gradient magnitude. This is the mathematical reason warmup exists (§4.8).

### 4.4 AdamW: decoupled weight decay

L2 regularization adds $\tfrac{1}{2}\lambda\|\theta\|^2$ to the loss, so the gradient becomes $g + \lambda\theta$ and Adam's update becomes (absorbing the regularizer into the moments):

$$
\theta_{t+1} = \theta_t - \eta\, \frac{\hat{m}_t + \lambda\theta_t}
{\sqrt{\hat{v}_t} + \varepsilon}
\;\approx\; \theta_t - \eta\, \frac{\lambda\theta_t}{\sqrt{\hat{v}_t}+\varepsilon}
- \eta\, \frac{\hat{m}_t}{\sqrt{\hat{v}_t}+\varepsilon}. \tag{10}
$$

The approximation keeps the dominant term: because $\hat{v}$ is an EMA, the decay term $\lambda\theta_t$ enters the update *divided by* $\sqrt{\hat{v}_t}$, and it is also smeared through the momentum history. The decay is therefore per-coordinate adaptive: coordinates with large past gradients (large $\sqrt{\hat{v}}$) get *less* decay, so the effective regularization varies by orders of magnitude across a weight matrix. AdamW (Loshchilov & Hutter, 2019) moves the decay out of the moments entirely:

$$
\theta_{t+1} = (1 - \eta\lambda)\, \theta_t
- \eta\, \frac{\hat{m}_t}{\sqrt{\hat{v}_t} + \varepsilon}. \tag{11}
$$

In (11) the shrinkage factor $(1 - \eta\lambda)$ is identical for every coordinate and every step; with no gradient the weight decays exactly geometrically, $\theta_t = (1-\eta\lambda)^t \theta_0$. Comparing (10) and (11), the difference is the placement of $\lambda\theta$:

$$
\text{L2-Adam: decay} = \eta\lambda\theta\,/\,(\sqrt{\hat v}+\varepsilon)
\qquad
\text{AdamW: decay} = \eta\lambda\theta. \tag{12}
$$

Why decoupling wins for transformers: transformer weight matrices are reparameterization-redundant with the following norm layer (RMSNorm rescales its input), so what matters is the *magnitude* of the weights feeding activations, not the loss contribution of their norm; a uniform, predictable shrinkage (11) keeps weight norms in a steady state, while the adaptive decay of (10) lets coordinates with long gradient histories escape regularization exactly where drift is likeliest (large matrices). Every mainstream LLM recipe (GPT-3, LLaMA, DeepSeek-V3) uses the decoupled form.

### 4.5 Hyperparameters for this run

From `configs/pretrain_a100_502m.yaml` (training section): `beta1: 0.9`, `beta2: 0.95`, `weight_decay: 0.1`, `lr: 4.0e-4`, `min_lr_ratio: 0.05`, `warmup_steps: 3000`, `total_steps: 61000`, `grad_clip: 1.0`, `eps: 1e-6`.

- **β₁ = 0.9.** Half-life ≈ 6.6 steps (§4.2): the velocity averages roughly
  one gradient-accumulation window (4 micro-batches) plus a bit. Lower values track noise, higher values lag schedule changes.
- **β₂ = 0.95.** The variance window is deliberately short (half-life ≈ 13.5
  steps, effective sample count $1/(1-\beta_2) = 20$) versus the Adam default 0.999 (1000 steps). Pretraining here runs 61K steps over a cosine schedule; a 1000-step variance window would feed stale $\hat{v}$ to the preconditioner after every LR change. LLaMA-style recipes use the same 0.95.
- **wd = 0.1.** Per-step relative shrinkage $\eta\lambda = 4\times10^{-5}$ at
  peak LR: small enough that gradients dominate, large enough to counteract slow drift of expert matrices and router logits. It is a standard pretraining value (GPT-3, LLaMA), not a tuned one.
- **lr = 4e-4 with init_std = 0.02.** A step of size $\eta$ is 2% of the
  typical weight magnitude set by `init_std` — large enough to reorganize weight norms within a few hundred steps, small enough not to scramble the residual stream. Adam's step per coordinate is $O(\eta)$ (§4.3), so the LR *is* the step size; 4e-4 at 131K tokens/step over 8B tokens is the standard Chinchilla-scale recipe.
- **min_lr_ratio = 0.05.** Cosine floor at $2\times10^{-5}$; the schedule never
  reaches zero, so the final steps still refine rather than add noise.
- **eps = 1e-6.** See §6 (pitfall 2); it is the BF16-safe floor.

### 4.6 Vectorized updates and FP32 master weights

BF16 stores 1 sign + 8 exponent + 7 mantissa bits; FP32 stores 23 mantissa bits. For a value in $[training.md](../training.md) is an **`[INFERENCE]` estimate** — `.benchmarks/` is empty. `fused` requires all parameters of a group to share a dtype and be contiguous, which the all-FP32 state here satisfies; the code gates it on `dev.type == "cuda"` and falls back to `foreach` on CPU.

### 4.7 Global-norm gradient clipping

Let $g$ be the flattened gradient of all $N$ parameters and $G$ its Euclidean norm:

$$
G = \|g\|_2 = \left( \sum_{i=1}^{N} g_i^2 \right)^{1/2}. \tag{14}
$$

Clipping projects $g$ onto the $\ell_2$ ball of radius $c$ — the closest vector to $g$ with norm $\le c$, found by rescaling the whole gradient by one scalar:

$$
g_{\text{clipped}} = g \cdot \min\!\left(1, \frac{c}{G}\right), \qquad
\|g_{\text{clipped}}\|_2 \le c. \tag{15}
$$

The direction of $g$ is untouched and every coordinate is scaled by the *same* factor, so relative magnitudes — which Adam respects via (9) — are preserved. The bound on the update: because $|g_i| \le G \le c$ after clipping, the per-coordinate step in (8) is bounded by $O(\eta)$ rather than $\eta \times (\text{spike}/\text{typical})$. With $N \approx 5\times10^8$ dimensions, an unclipped norm of $G \approx \sqrt{N}\,\sigma$ (for per-coordinate std $\sigma$) exceeds $c = 1.0$ whenever $\sigma \gtrsim 4.5\times10^{-5}$ [derived estimate]; so clipping is a safety valve that binds during warmup and routing instability, not a constant brake.

### 4.8 Warmup: protecting the biased moments

The schedule multiplies the entire update: $\eta \to \eta\lambda(t)$ with $\lambda(t) = t/w$ for $t < w$ (linear warmup), then cosine decay to the floor. Section 4.3 showed the first Adam step is $\pm\eta$ in *every* coordinate because one sample makes $\hat m/\sqrt{\hat v}$ a pure sign. With warmup the effective step is:

$$
\theta_{t+1} = \theta_t - \eta\,\lambda(t)\, \frac{\hat m_t}{\sqrt{\hat v_t}+\varepsilon}, \qquad
\lambda(t) = \min\!\left(\frac{t}{w},\, \text{cosine}\right), \tag{16}
$$

so the early steps grow linearly from zero while $\hat v$ accumulates its effective $1/(1-\beta_2) = 20$ samples. The relative error of the variance estimate after $n$ samples scales as $\sim 1/\sqrt{n}$: at $n = 20$ that is ~22%, at $n = 1$ it is 100% — warmup does not fix the bias (the correction (7) already does), it bounds the *variance* of early steps. The repo's choice of $w = 3000$ (4.9% of the run) sits in the 2–5% band recommended for MoE training per the YAML comment, because router gradients are largest and noisiest exactly during warmup. Note also that $\lambda(t)$ multiplies the decoupled decay in (11) too — weight decay ramps up with the learning rate, which is desirable: don't shrink weights before gradients have set their scale.

### 5. Code walkthrough

### 5.1 Optimizer construction in `training/pretrain.py:main`

`training/pretrain.py:main` builds two parameter groups by name:

```python
no_decay = ["bias", "norm", "embed"]
decay_params = [p for n, p in model.named_parameters() if not any(nd in n.lower() for nd in no_decay)]
no_decay_params = [p for n, p in model.named_parameters() if any(nd in n.lower() for nd in no_decay)]
```

Substring matching: any parameter whose name contains `bias`, `norm`, or `embed` is decay-free. This covers the RMSNorm gains and biases everywhere, and the tied embedding — the word embedding and the output head are the *same* `Parameter` tensor (`weight_tying: true` in the config), and `named_parameters()` lists the shared tensor once under the embedding's name, so excluding `embed` excludes the tied head as well. The two groups then go into a single `AdamW`:

```python
optim = AdamW(
    [
        {"params": decay_params, "weight_decay": train_cfg["weight_decay"]},
        {"params": no_decay_params, "weight_decay": 0.0},
    ],
    lr=train_cfg["lr"],
    betas=(train_cfg.get("beta1", 0.9), train_cfg.get("beta2", 0.95)),
    eps=1e-6,
    foreach=True,
    fused=(dev.type == "cuda"),
)
```

Every hyperparameter comes from `configs/pretrain_a100_502m.yaml` (or its default), so the schedule and the optimizer cannot drift out of sync with the recipe. `fused=(dev.type == "cuda")` is the only hardware-dependent flag: on CPU the same code path runs with `foreach`.

### 5.2 Schedule interaction

The schedule is a `LambdaLR` wrapping the lambda returned by `training/pretrain.py:make_warmup_cosine_lambda`:

```python
sched = LambdaLR(optim, make_warmup_cosine_lambda(
    train_cfg["warmup_steps"],
    train_cfg["total_steps"],
    train_cfg.get("min_lr_ratio", 0.05),
))
```

`make_warmup_cosine_lambda` returns `lr_lambda(step)` where `step / warmup_steps` for `step < warmup_steps`, the cosine `min_lr_ratio + (1 - min_lr_ratio) * 0.5 * (1 + cos(π · progress))` afterwards, and the constant `min_lr_ratio` past `total_steps` — the exact schedule of (16). `LambdaLR` multiplies the group's base LR by this factor, which is why warmup scales the *whole* AdamW update including decay (§4.8). The schedule's three boundary invariants — zero at step 0, peak at the warmup boundary, floor at the end — are pinned by `tests/test_training.py::test_lr_schedule_at_warmup_boundary`, `test_lr_schedule_at_end`, and `test_lr_schedule_monotonic_decay_after_warmup`.

### 5.3 The step: clipping, NaN guard, and optimizer state

Inside the accumulation loop (after `loss = (ce + aux_alpha * aux_loss) / accum` in `training/pretrain.py:main`), the optimizer runs only on accumulation boundaries:

```python
if grad_clip > 0:
    nn.utils.clip_grad_norm_(model.parameters(), grad_clip, foreach=True)
optim.step()
sched.step()
optim.zero_grad(set_to_none=True)
```

`clip_grad_norm_` implements (14)–(15) over all parameters before `step()`; the `try/except TypeError` around it is a version-compat shim for the `foreach` kwarg. Dividing the loss by `accum` before `backward()` keeps the accumulated gradient at single-micro-batch scale, so the clip threshold is meaningful across `accum=4`. `zero_grad(set_to_none=True)` frees the FP32 gradient buffers (2.0 GB) after each step instead of zeroing them.

The NaN guard sits *before* the backward/step block:

```python
if not torch.isfinite(loss):
    if nan_guard:
        nan_count += 1
        optim.zero_grad(set_to_none=True)
        micro_step = 0
        if nan_count >= nan_max_consec:
            ckpt.load(model, step=latest, device=str(dev), optimizer=optim, scheduler=sched)
        continue
```

When the loss is non-finite the loop **skips `optim.step()` entirely** — it never lets a poisoned gradient touch Adam state. That is load-bearing for a running-moment optimizer: in (6), if any $g_i$ is NaN then $v_t = \beta_2 v_{t-1} + (1-\beta_2)\text{NaN}^2$ is NaN, and because $\beta_2 v_t$ keeps the NaN in the running sum,

$$
v_{t+k} = \beta_2^k\, \text{NaN} + \sum_{j=1}^{k} \beta_2^{k-j}(1-\beta_2)\, g_{t+j}^2
= \text{NaN} \tag{17}
$$

for *every* future $k$ — one NaN gradient permanently poisons Adam's second moment; the state cannot heal itself. The guard prevents the poison from entering, and the rollback branch (after `nan_guard_max_consecutive: 5` consecutive NaNs) restores $m, v$ from the last checkpoint (`tests/test_training.py::test_checkpoint_round_trip` covers the optimizer-state round trip, and `test_nan_guard_detection` pins the `torch.isfinite` check). Reproducibility of the whole trajectory rests on `training/pretrain.py:seed_everything`, which seeds Python, NumPy, and PyTorch RNGs and sets `CUBLAS_WORKSPACE_CONFIG=:4096:8` for deterministic cuBLAS before the model — and therefore before Adam's zero-initialized moments — are created.

### 6. Pitfalls + verify

1. **NaN through the optimizer state is permanent.** One NaN gradient makes
   $v_t$ NaN forever ((17)); a NaN *loss* is caught and skipped, but a finite loss with a NaN gradient (e.g. overflow inside the router softmax) would not be. Guard: `tests/test_training.py::test_nan_guard_detection` plus watching the `[nan-guard]` log lines during a smoke run (`scripts/e2e_gpu_smoke.py`).
2. **eps=1e-6, not 1e-8.** The classical eps=1e-8 fails outright under FP16
   (min normal $6.1\times10^{-5}$, min subnormal $6\times10^{-8}$ — 1e-8 rounds to zero in the second moment). Under BF16 the range is FP32-like, so nothing underflows, but with only 7 mantissa bits (13) a variance stored near 1e-8 carries ~0.4% relative error, and when $\sqrt{\hat v} \ll \varepsilon$ the denominator of (8) is dominated by $\varepsilon$: the step becomes $\eta\hat m/\varepsilon$, amplifying gradient noise instead of normalizing it. 1e-6 is a safe floor given typical BF16 gradient magnitudes; this is the setting DeepSeek-V3 and LLaMA-3 use, and it is pinned in the config comment and [moe.md](moe.md) §16.
3. **MoE: the optimizer sees all 501.8M parameters.** Unrouted experts receive
   zero gradient, but AdamW still applies the decoupled decay (11) to them and keeps their moments alive for when they are next selected. The 4.0 GB optimizer state is over *total* parameters — sparsity does not reduce it. Consequence: an expert starved for many steps still shrinks at $\eta\lambda$ per step, which is part of why the aux loss (α=0.01) must keep routing balanced (see [moe theory](moe.md)).
4. **Warmup is not the bias correction.** The correction (7) makes the moments
   unbiased from step 1; warmup (16) bounds the *variance* of the early sign-like steps (§4.3, §4.8). Cutting warmup on this model means the first step moves every coordinate by $\pm\eta$ — with 8 routed experts' gates at init, that is a reliable way to seed expert collapse. Verify the boundary behavior with the three schedule tests in `tests/test_training.py`.
5. **`fused` silently requires uniformity.** `fused=True` demands one dtype per
   group and contiguous tensors; if a future change casts some parameters to BF16 in place, the fused path either errors or (worse) falls back without telling you. The A100 speedup claims (1.5–2×, 35–40% MFU) are **`[INFERENCE]`** until `.benchmarks/` is populated.

---

*Verified 2026-08-04 against `training/pretrain.py`, `configs/pretrain_a100_502m.yaml`, `tests/test_training.py`, and [training.md](../training.md). No pretraining run has completed; all performance figures are targets or `[INFERENCE]`.*

## Part B — Numerics (Precision, Range, and the Bit Budget)

> **Where in the code:** `training/pretrain.py` (autocast, TF32 knobs, NaN guard), `models/transformer.py:RMSNorm`, `models/attention.py` (sink clamp, FP32 scores), `models/moe.py` (FP32 router/aux softmax). Companion derivations live in [attention math](attention-and-positional.md), [optimizers](optimizers-and-numerics.md), and [triton programming](kernels-and-checkpointing.md); the primer-level version of §6 is [foundations](foundations-and-architecture.md) §10.

### 1. 60-second summary

Every number in this model is stored in one of a handful of binary formats, and the choice between them decides whether a 61,000-step run converges or diverges. GPT-OSS-Lite trains in **BF16** (bfloat16): the same 8-bit exponent as FP32, so the *range* of representable magnitudes matches FP32, at the cost of a 7-bit mantissa (machine epsilon $2^{-7}$). That choice removes the single most common pretraining failure mode — FP16 gradient underflow and its band-aid, loss scaling — and lets the loop skip `GradScaler` entirely. Precision is recovered in the places that need it by deliberate **FP32 islands**: RMSNorm statistics, attention scores, and the router/auxiliary softmax are promoted to FP32 explicitly. Two guards keep the run alive: the sink bias is clamped to $[-10, 15]$ so the learned mask addend can never overflow the softmax exponential, and a run-level NaN guard skips poisoned steps and rolls back after five consecutive non-finite losses. `training/pretrain.py:_set_hardware_perf_knobs` additionally enables TF32 on A100 tensor cores so the remaining FP32 matmuls are fast.

### 2. Why it matters here

The design decisions of this repo are, at bottom, numerical decisions:

- **`dtype: "bf16"`** in `configs/pretrain_a100_502m.yaml` drives `_amp_dtype` and the autocast region in `training/pretrain.py:main`. BF16 is the load-bearing invariant that makes "no GradScaler" safe ([training](../training.md) lists it as invariant #1).
- **Vocab 128,000** makes the per-class cross-entropy gradient $p_i \approx 1/V \approx 7.8\times10^{-6}$ — below FP16's minimum normal (see §6.1). The format choice is decided by this number.
- **Top-2-of-8 routing** computes a softmax over 8 expert logits whose *saturation* (logits $\gtrsim 88$) would overflow BF16's exponential; `models/moe.py:MoERouter.forward` and `models/moe.py:aux_load_balancing_loss` promote to FP32.
- **The learned sink bias** is an additive logit riding inside the SDPA mask; `models/attention.py:GPTOSSAttention.forward` clamps it to $[-10, 15]$ so the mask addend and the resulting softmax stay finite at every one of the 12 layers (`models/attention.py:SINK_CLAMP_MIN` / `SINK_CLAMP_MAX`).
- **A 61,000-step, 8×4096-token schedule** means roughly $2\times10^9$ token-events per run. Even a once-in-a-million numerical accident *will* occur; the NaN guard in `training/pretrain.py:main` exists because no format choice can prevent every NaN source.

No pretraining run has completed yet; this chapter derives the numerics contract the run depends on, and everything quantitative below is either derived here or marked `[INFERENCE]` (`.benchmarks/` is empty, so all A100 timing figures are estimates).

### 3. Intuition — the logarithmic ruler

Think of a floating-point format as a ruler marked in scientific notation with a fixed number of significant digits. The exponent field decides *where on the number line* the ruler sits — which orders of magnitude are reachable at all. The mantissa field decides *how finely* it is marked — how many significant digits each mark carries. FP16 is a short ruler covering only $10^{\pm5}$; FP32 and BF16 are long rulers covering $10^{\pm38}$. The catch: BF16's ruler is long but coarsely marked (about 2 significant decimal digits), while FP16's is short but finer (about 3). Training gradients are tiny numbers that live at the far bottom of the scale — so range, not fineness, is the survival constraint, and BF16 wins. Precision is then spent where it matters by switching the computation to FP32's fine ruler for a single op (softmax, norm statistics), then switching back.

### 4. IEEE-754: sign, exponent, mantissa

### 4.1 The value formula

Every IEEE-754 binary format stores three fields: a sign bit $s$, an exponent field of $k$ bits holding a biased exponent $E$, and a fraction field of $p$ bits holding the mantissa $f$. For normal numbers (the only ones needed for the range and epsilon arguments) the value is

$$
x = (-1)^s \times (1.f)_2 \times 2^{E - B}, \qquad B = 2^{k-1} - 1, \qquad 1 \le E \le 2^k - 2, \tag{1}
$$

where $(1.f)_2 = 1 + \sum_{j=1}^{p} f_j 2^{-j}$ is the mantissa in $[1, 2)$, and $B$ is the exponent bias. The exponent field is biased so that small numbers get negative true exponents; the all-zeros field is reserved for subnormals and zero, the all-ones field for infinities and NaN. Three formats matter here:

| Format | $s$ | $k$ (exponent) | $p$ (fraction) | Bias $B$ |
|---|---|---|---|---|
| FP32 | 1 | 8 | 23 | 127 |
| FP16 | 1 | 5 | 10 | 15 |
| BF16 | 1 | 8 | 7 | 127 |

### 4.2 Range: maximum and minimum normal

The largest normal stored exponent is $E_{\max} = 2^k - 2$, so the largest true exponent is $e_{\max} = (2^k - 2) - (2^{k-1} - 1) = 2^{k-1} - 1$. The largest mantissa is $(2 - 2^{-p})$. The maximum finite value is therefore

$$
x_{\max} = (2 - 2^{-p})\, 2^{\,2^{k-1} - 1}. \tag{2}
$$

The smallest normal has $E = 1$, true exponent $e_{\min} = 1 - B = 2 - 2^{k-1}$, and mantissa exactly $1.0$:

$$
x_{\min} = 2^{\,2 - 2^{k-1}}. \tag{3}
$$

Plugging the table into (2) and (3):

- **FP32** ($k=8, p=23$): $x_{\max} = (2 - 2^{-23})\cdot 2^{127} \approx 3.403\times10^{38}$; $x_{\min} = 2^{-126} \approx 1.175\times10^{-38}$.
- **FP16** ($k=5, p=10$): $x_{\max} = (2 - 2^{-10})\cdot 2^{15} = 65504$; $x_{\min} = 2^{-14} \approx 6.104\times10^{-5}$.
- **BF16** ($k=8, p=7$): $x_{\max} = (2 - 2^{-7})\cdot 2^{127} \approx 3.389\times10^{38}$; $x_{\min} = 2^{-126} \approx 1.175\times10^{-38}$.

The two 16-bit formats could hardly differ more: FP16 tops out at $6.55\times10^{4}$, BF16 at $3.39\times10^{38}$ — the FP32 ceiling. Below the minimum normal lie subnormals (FP16 down to $2^{-24} \approx 5.96\times10^{-8}$), with reduced precision and, on some hardware, slower paths; values below that flush to zero.

### 4.3 Precision: machine epsilon

Numbers in $[2^{e}, 2^{e+1})$ are spaced by one unit in the last place, $\mathrm{ULP} = 2^{e-p}$: the mantissa is fixed at $p$ bits, so adjacent values differ in the last bit of the fraction. At $e = 0$ (the interval $[1, 2)$) the spacing is $2^{-p}$, which defines machine epsilon:

$$
\varepsilon = \mathrm{ULP}(1.0) = 2^{-p}, \qquad \text{round-to-nearest error} \le \frac{\varepsilon}{2} = 2^{-(p+1)}. \tag{4}
$$

The second half follows because any value in $[2^e, 2^{e+1})$ is within half an ULP of a representable value, and the ULP relative to the value is at most $2^{e-p}/2^e = 2^{-p}$. So $\varepsilon$ is both the spacing at 1.0 and the relative-precision scale. Numerically:

- FP32: $\varepsilon = 2^{-23} \approx 1.19\times10^{-7}$ (about 7 decimal digits)
- FP16: $\varepsilon = 2^{-10} \approx 9.77\times10^{-4}$ (about 3 decimal digits)
- BF16: $\varepsilon = 2^{-7} \approx 7.81\times10^{-3}$ (about 2 decimal digits)

Every arithmetic operation rounds its exact result to this grid; the accumulated errors are what the rest of this chapter manages.

### 4.4 The three formats at a glance

| Format | $x_{\max}$ | $x_{\min}$ (normal) | $\varepsilon$ | Meaningful digits |
|---|---|---|---|---|
| FP32 | $3.40\times10^{38}$ | $1.18\times10^{-38}$ | $1.19\times10^{-7}$ | ~7 |
| FP16 | $6.55\times10^{4}$ | $6.10\times10^{-5}$ | $9.77\times10^{-4}$ | ~3 |
| BF16 | $3.39\times10^{38}$ | $1.18\times10^{-38}$ | $7.81\times10^{-3}$ | ~2 |

### 5. TF32 — the Ampere matmul compromise

A100 tensor cores accept three input precisions: FP16/BF16, and TF32 — a 19-bit format, $1 + 8 + 10$: the FP32 exponent field (full FP32 range) with a 10-bit mantissa, $\varepsilon_{\mathrm{TF32}} = 2^{-10} \approx 9.77\times10^{-4}$ — the same precision as FP16 but the range of FP32. The tensor core rounds the FP32 inputs to TF32, computes the products exactly, and accumulates in FP32, roughly doubling FP32 matmul throughput on the A100 relative to the FP32 CUDA-core path.

`training/pretrain.py:_set_hardware_perf_knobs` opts into this path explicitly:

```python
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.set_float32_matmul_precision("high")
```

It also sets `torch.backends.cuda.preferred_blas_library = "cublaslt"` (cuBLASLt, A100-tuned; the code comment claims "2–5% faster" — unmeasured, `[INFERENCE]`) and `cudnn.benchmark_limit = 0` (exhaustive cuDNN algorithm search, "~3–5% on A100" per the comment — also `[INFERENCE]`). The subtlety: TF32 only changes *FP32* matmuls. This repo's production path runs the model in BF16 via autocast, so TF32 matters for whatever FP32 matmuls remain — mostly the FP32 islands of §8 and any un-autocast code. It is not a source of precision loss for the BF16 path, because BF16 matmuls already run at $\varepsilon = 2^{-7}$ by design. The one caution: enabling TF32 makes FP32 matmuls round their inputs to 10-bit mantissas, so bit-exact reproducibility across GPU vendors is not guaranteed — a reproducibility note, not a correctness issue.

### 6. Why BF16 for pretraining (and not FP16)

### 6.1 The gradient magnitudes this model actually produces

The loss is a cross-entropy over $V = 128\,000$ classes (see `training/pretrain.py:chunked_cross_entropy`). The gradient of the per-token loss with respect to the logits is

$$
\frac{\partial \mathrm{CE}}{\partial z_i} = p_i - \mathbf{1}_{i = y}, \qquad p_i = \frac{e^{z_i}}{\sum_{j=1}^{V} e^{z_j}}, \tag{5}
$$

where $p_i$ is the softmax probability of class $i$ and $y$ the target class. Early in training, before the model separates classes, the probabilities are near-uniform:

$$
p_i \approx \frac{1 - p_y}{V - 1} \approx \frac{1}{V} = \frac{1}{128\,000} \approx 7.81\times10^{-6}. \tag{6}
$$

So the *majority* of logit gradients — one per vocabulary class per token — are of order $10^{-5}$ or smaller at the start of training, and products of such terms through the network's layers are far smaller still.

### 6.2 FP16 underflow and loss scaling

Compare (6) with FP16's minimum normal from (3): $x_{\min} = 2^{-14} \approx 6.10\times10^{-5}$. The per-class gradient $7.81\times10^{-6}$ is *below* the FP16 normal floor by a factor of ~8 — it lands in the subnormal band $[2^{-24}, 2^{-14})$ and any value below $2^{-24} \approx 5.96\times10^{-8}$ flushes to zero. Products like $10^{-5} \times 10^{-5} = 10^{-10}$ — which appear routinely in deep paths and rarely-updated parameters — are hard zeros in FP16. Those silent zeros are not noise; they delete the gradient signal for precisely the parameters that need small updates.

The classical remedy is loss scaling: multiply the loss by a scale $S$ before backpropagation so that $S \cdot g$ fits inside the FP16 normal window,

$$
2^{-14} \;\le\; S\,|g| \;\le\; 65504 \qquad\Longleftrightarrow\qquad \frac{2^{-14}}{|g|} \;\le\; S \;\le\; \frac{65504}{|g|}. \tag{7}
$$

For a gradient component of magnitude $10^{-8}$, (7) demands $S \ge 6.1\times10^{3} \approx 2^{13}$; for a component of $10^{-3}$, it demands $S \le 6.6\times10^{7}$. Any single $S$ works only if all gradients sit within its window, so real frameworks track overflow (skip the step, halve $S$) and underflow (double $S$) dynamically — the `GradScaler` machinery. It is correct, but it is one more moving part, and every bug in it corrupts training silently.

### 6.3 What BF16 changes

BF16's minimum normal is $2^{-126} \approx 1.18\times10^{-38}$ — identical to FP32's. The bottom of the range is

$$
\frac{x_{\min}^{\mathrm{BF16}}}{x_{\min}^{\mathrm{FP16}}} = \frac{2^{-126}}{2^{-14}} = 2^{-112} \approx 1.9\times10^{-34}. \tag{8}
$$

Any gradient magnitude that the FP32 optimizer state can hold (down to $1.18\times10^{-38}$; see [optimizers](optimizers-and-numerics.md) for the FP32 master-state story) survives the BF16 storage round-trip. The per-class gradient $7.81\times10^{-6}$ from (6) is a perfectly ordinary BF16 number — there are 32 orders of magnitude of headroom below it. Hence **no `GradScaler`**: the scale $S$ in (7) can be fixed at 1 for every step, and the optimizer consumes unscaled gradients, which keeps the loss, the learning rate, and the weight decay all in their natural units.

What BF16 gives up is precision: $\varepsilon = 2^{-7}$ versus FP16's $2^{-10}$ — 8× coarser rounding per operation. That is acceptable for three reasons, each true in this repo:

1. **FP32 accumulation** — tensor-core matmuls accumulate products in FP32, so the rounding happens once at the input (relative error $\le 2^{-8}$) rather than once per product-add.
2. **FP32 optimizer state** — AdamW keeps FP32 moment estimates (and FP32 master weights in the fused path), so the *updates* are precise even though the forward pass is coarse; this is also why `training/pretrain.py:main` uses `eps=1e-6` rather than $10^{-8}$: a $10^{-8}$ epsilon added to a BF16-valued second moment would itself underflow (see [optimizers](optimizers-and-numerics.md)).
3. **FP32 islands** — the numerically fragile ops (softmax, norm statistics, router probabilities) are explicitly promoted to FP32, as the next two sections show.

### 7. Autocast — what runs where

`training/pretrain.py:main` wraps the forward and loss in

```python
_amp_dtype = {"bf16": torch.bfloat16, "fp16": torch.float16}.get(model_cfg.dtype, torch.bfloat16)
...
with autocast(device_type=dev.type, dtype=_amp_dtype, enabled=(dev.type == "cuda")):
    logits, aux_loss = model(input_ids)
    ce = chunked_cross_entropy(logits, target_ids, chunk_size=8192)
    loss = (ce + aux_alpha * aux_loss) / accum
```

Autocast is a per-op dtype policy, not a global cast:

- **Matmul-class ops** (`linear`, `addmm`, `bmm` — the QKV projections, the attention output projection, the expert SwiGLU matmuls) execute in the autocast dtype, BF16 here, with FP32 accumulation on tensor cores. This is where the ~2× throughput gain lives.
- **A small set of fragile ops is widened to FP32 on CUDA** — softmax, log-softmax, layer norm, and the loss reductions built on them. This is why the CE over BF16 logits is still computed stably: `chunked_cross_entropy` reduces per-chunk `F.cross_entropy` sums whose internal log-softmax runs in FP32.
- **Everything else** — elementwise ops, reductions, casts — runs in the input's dtype, which is why the explicit `.float()` promotions in §8 exist at all: autocast does *not* touch them.

`enabled=(dev.type == "cuda")` is deliberate: CPU runs (smoke tests, CI) skip autocast and stay in FP32, so CPU and GPU results differ only in the BF16 quantization — which the tests accommodate with tolerances.

### 8. FP32 accumulation islands inside the model

Three spots inside the model promote to FP32 explicitly rather than trusting autocast or the 16-bit formats.

### 8.1 RMSNorm statistics — `models/transformer.py:RMSNorm`

```python
def forward(self, x: torch.Tensor) -> torch.Tensor:
    rms = x.detach().float().pow(2).mean(dim=-1, keepdim=True).add(self.eps).rsqrt()
    return (x * (rms * self.weight.to(rms.dtype)).to(x.dtype))
```

The norm scale is $\big(\frac{1}{d}\sum_{j=1}^{d} x_j^2 + \epsilon\big)^{-1/2}$ over $d = 768$ features, applied at every block via `models/transformer.py:GPTOSSBlock.forward` (`x + self.attn(self.norm1(x), ...)`). If the sum were accumulated in BF16, each of the 768 additions would round with relative error up to $2^{-8}$, and the accumulated variance estimate would carry relative error on the order of

$$
\frac{\Delta \overline{x^2}}{\overline{x^2}} \;\sim\; \sqrt{d}\, 2^{-8} \approx 27.7 \times 3.9\times10^{-3} \approx 1.1\times10^{-1}, \tag{9}
$$

i.e. ~10% error in the variance — hence ~5% in the RMS scale — per layer per step, a structured perturbation 25× larger than the BF16 grid it is applied to. The explicit `.float()` moves the squares, the mean, the epsilon add, and the rsqrt into FP32, where the accumulation error is $\sqrt{768}\cdot 2^{-24} \approx 1.4\times10^{-6}$ and the residual error is dominated by the input rounding, $2\cdot 2^{-8}/\sqrt{768} \approx 2.8\times10^{-4}$. The scale is then applied in FP32 and the result cast back to `x.dtype`: the *statistics* are FP32, but the *activation* stays BF16 — no FP32 copy of the hidden state is materialized, so the memory footprint of the norm is unchanged.

### 8.2 Attention scores — `models/attention.py:manual_causal_attention`

The reference/test attention path computes

```python
scores = (query_states.float() @ key_states.float().transpose(-2, -1)) / math.sqrt(D)
```

The dot product over $D = 96$ dims feeds directly into softmax, which exponentiates. If the score accumulation ran in BF16, the relative error of each score would be $\sqrt{96}\cdot 2^{-8} \approx 0.12$ — a 12% perturbation of the exponent argument, i.e. ~12% multiplicative noise on the softmax weights. FP32 accumulation brings it to $\sqrt{96}\cdot 2^{-24} \approx 6\times10^{-7}$, so the score error is dominated by the BF16 quantization of the inputs themselves ($\le 2^{-8}$ relative, a 0.4% perturbation of the exponent argument — harmless). The softmax then runs in FP32 and the resulting weights are cast to the value dtype for the final matmul, mirroring what the fused SDPA path does internally. (The production path is `models/attention.py:causal_attention`, whose kernel handles the accumulation; the FP32 reference exists precisely to give the tests a numerically clean target — see the full derivation in [attention math](attention-and-positional.md).)

### 8.3 Router and aux-loss softmax — `models/moe.py`

Both the router and the load-balancing loss promote their softmax to FP32:

```python
# MoERouter.forward
all_probs_f32 = F.softmax(logits.float(), dim=-1)
# aux_load_balancing_loss
probs_f32 = F.softmax(all_logits.float(), dim=-1)
```

Why this matters numerically: softmax over BF16 logits overflows as soon as a logit exceeds the point where its exponential leaves the format's range. In FP32 the exponential overflows at

$$
e^{z} = \infty \quad\Longleftrightarrow\quad z > \ln x_{\max}^{\mathrm{FP32}} = \ln(3.4\times10^{38}) \approx 88.7, \tag{10}
$$

and in BF16 at a similar magnitude (BF16's max is the same order). Saturated router logits are not hypothetical — `tests/test_moe.py:test_aux_loss_robust_to_bf16_saturation` constructs exactly this case. An `inf` in one softmax entry makes the output NaN (inf/inf), which then poisons every expert output and the aux loss. The `.float()` widens the safe logit range to $(-\infty, 88.7]$ and keeps the per-expert probabilities $P_i$ and routing fractions $f_i$ accurate; the aux loss $N \sum_i f_i P_i$ multiplies small probabilities, and BF16's $2^{-8}$ relative rounding on near-zero $P_i$ would swamp the gradient signal for under-used experts (full derivation in [moe theory](moe.md)).

### 9. The sink-bias clamp — a finite mask addend

`models/attention.py:causal_attention` adds the learned sink bias to the attention computation through the SDPA mask: the mask is a float tensor in the activation dtype (BF16), the sink occupies its final column, and SDPA *adds* the mask to the pre-softmax scores,

$$
s'_{i,\mathrm{sink}} = s_{i,\mathrm{sink}} + b_h, \qquad \sigma(z)_i = \frac{e^{z_i}}{\sum_j e^{z_j}}, \tag{11}
$$

where $b_h$ is the per-head bias parameter and the softmax runs in FP32. Two failure modes make an unbounded $b_h$ fatal:

1. **Overflow.** From (10), once $s' > 88.7$ the FP32 exponential overflows to `inf`, the softmax becomes inf/inf = NaN, and the NaN propagates to the layer output and every downstream gradient. A learned bias is a free parameter — nothing stops it drifting past 88 — so an unclamped sink is a guaranteed eventual NaN.
2. **Underflow / death.** If $b_h \to -\infty$, then $e^{b_h} \to 0$: the sink term vanishes from the denominator, the output stops depending on $b_h$, and its gradient $\partial \sigma/\partial b_h \to 0$ — the parameter dies and can never recover.

The clamp `models/attention.py:SINK_CLAMP_MIN = -10.0` / `SINK_CLAMP_MAX = 15.0` bounds the exponential to

$$
e^{-10} \;\le\; e^{b_h} \;\le\; e^{15} \qquad\Longleftrightarrow\qquad 4.54\times10^{-5} \;\le\; e^{b_h} \;\le\; 3.27\times10^{6}, \tag{12}
$$

a dynamic range of $\approx 7.2\times10^{10} \approx 2^{36}$ — comfortably inside both FP32's and BF16's representable ranges (from §4.4: $[1.18\times10^{-38}, 3.39\times10^{38}]$), and far from the softmax overflow threshold of (10). The upper bound is a *design* bound, not the FP32 limit: $e^{15} \approx 3.3\times10^{6}$ makes the sink a strong but surmountable attractor against a window of 128 scores, whereas $b_h = 50$ would give $e^{50} \approx 5.2\times10^{21}$ and collapse essentially all attention mass onto the sink's zero value vector (this is the behavior `tests/test_attention.py:test_sink_bias_high_value_collapses_attention` probes; the sink's role is derived in [ATTENTION_SINKS](attention-sinks.md)). The lower bound keeps $e^{b_h}$ and its gradient alive: $e^{-10}$ is 33 orders of magnitude above the BF16 underflow floor.

**Gradient flow.** `models/attention.py:GPTOSSAttention.forward` applies the clamp at forward time on a *copy*:

```python
sink_bias_clamped = self.sink_bias.clamp(SINK_CLAMP_MIN, SINK_CLAMP_MAX)
```

The `nn.Parameter` itself is never mutated, so AdamW's moment estimates stay consistent with the parameter's true value, and while $b_h$ is inside $[-10, 15]$ the backward pass flows through the clamp with derivative 1. Outside the interval the local derivative is 0 — a deliberate stop that prevents the parameter from being pushed further out — and the parameter can still be pulled back in by the optimizer. `tests/test_attention.py:test_sink_bias_clamped_at_forward` pins both properties: after a forward with `sink_bias` filled with 1000.0, the parameter is still 1000.0 and the output is finite.

### 10. The NaN guard — run-level defense

BF16 removes the *underflow* failure class, but NaN sources remain: exponential overflow from (10) in any softmax (router logits, attention scores, the tail of a diverging weight matrix), inf − inf and 0 × inf in loss reductions, loader corruption, a bad checkpoint, or an optimizer step that overshoots into diverged weights. NaN propagates deterministically: one NaN element in a matmul output contaminates the entire output tensor, so a single bad step corrupts every downstream weight by the next step. With $2\times10^9$ token-events per planned run, "rare" is not rare enough — which is why `training/pretrain.py:main` checks the loss every step:

```python
if not torch.isfinite(loss):
    if nan_guard:
        nan_count += 1
        optim.zero_grad(set_to_none=True)
        micro_step = 0
        if nan_count >= nan_max_consec:
            ckpt.load(model, step=latest, device=str(dev), optimizer=optim, scheduler=sched)
            step = latest
            nan_count = 0
        continue
```

The design handles the two attack surfaces:

- **A single bad step is discarded**, not accumulated: gradients are zeroed and the micro-step counter resets, so a NaN that appears mid-accumulation (before the `accum`-boundary optimizer step) cannot poison the accumulated gradient buffer of the next real step.
- **Five consecutive non-finite losses mean the model itself is corrupted**, not the batch. The loop rolls the model, optimizer, and scheduler back to the latest complete checkpoint (`CheckpointManager.latest_step`), or raises `RuntimeError` if no checkpoint exists. Five is the tolerance for a transient (e.g., a corrupted batch that repeats); beyond that, retrying forward steps against a poisoned weight matrix is guaranteed to keep producing NaNs.

The guard is run-level, not format-level: no dtype choice can prevent 0 × inf or a corrupted loader, and with gradient accumulation the cost of a missed NaN is not one step but the whole accumulated gradient. Note the honesty caveat: no pretraining run has completed yet, so the guard is exercised by code review and the training-loop tests, not by a production run — its thresholds (`nan_guard: true`, `nan_guard_max_consecutive: 5` in `configs/pretrain_a100_502m.yaml`) are the planned, untested configuration.

### 11. Pitfalls and verification

| Trap | Consequence | Guard |
|---|---|---|
| Training in FP16 without scaling | Per-class gradients (6) flush to subnormal/zero; silent dead parameters | BF16 default + no GradScaler; `training.md` invariant #1 |
| Trusting autocast for softmax/norm | Autocast widens only the fragile ops on CUDA; elementwise ops stay BF16 | Explicit `.float()` in `models/transformer.py:RMSNorm.forward`, `models/moe.py:MoERouter.forward`, `models/moe.py:aux_load_balancing_loss` |
| Unclamped sink bias | `exp` overflow at $s' > 88.7$ → NaN, or bias death at $b_h \to -\infty$ | Forward-time clamp $[-10, 15]$, `models/attention.py:SINK_CLAMP_MIN/MAX`; `pytest tests/test_attention.py -v` (`test_sink_bias_clamped_at_forward`) |
| BF16 router softmax | Saturated logits → `exp` overflow → NaN aux/expert output | FP32 softmax; `pytest tests/test_moe.py -v` (`test_aux_loss_robust_to_bf16_saturation`) |
| AdamW `eps=1e-8` with BF16 params | Epsilon below the BF16 grid underflows the second moment; late-stage convergence stalls | `eps=1e-6` in `training/pretrain.py:main` (see [optimizers](optimizers-and-numerics.md)) |
| Ignoring TF32 non-determinism | FP32 matmuls round inputs to 10-bit mantissas; results differ across vendors | Seed for data/model/optimizer; treat FP32-path numerics as platform-dependent |

**Verification commands** (CPU-runnable, no GPU required):

```bash
pytest tests/test_attention.py -v    # includes test_sink_bias_clamped_at_forward
pytest tests/test_moe.py -v          # includes test_aux_loss_robust_to_bf16_saturation
```

**Format constants** (regenerate the §4.4 table on any torch version):

```bash
python3 -c "import torch; [print(n, torch.finfo(getattr(torch, n)).max, torch.finfo(getattr(torch, n)).tiny, torch.finfo(getattr(torch, n)).eps) for n in ('float32','float16','bfloat16')]"
```

The full suite baseline is 192 tests (190 pass + 2 GPU-gated Triton skips on CPU). Timing claims in `training/pretrain.py:_set_hardware_perf_knobs` comments ("~3–5%", "2–5%") are code-comment estimates, not measurements — `.benchmarks/` is empty — and the 61,000-step A100 schedule ($\approx$16–20 h at $\approx$35–40% MFU) is `[INFERENCE]` until a run completes.

---

## Part C — Sampling (Autoregressive Decode and Token Selection)

> **Chapter on `inference/generate.py`.** How a trained decoder turns logits into tokens: autoregressive factorization, greedy decoding, temperature (Boltzmann) scaling, top-k / top-p truncation, entropy and perplexity, and the determinism rules that make the passkey benchmark meaningful. Engine-level decode (KV cache, ring buffers) is [inference.md](../inference.md) and [kv cache engineering](../inference.md); the softmax that underlies every equation here is derived in [attention math](attention-and-positional.md).

---

### 60-second summary

A transformer's final layer emits one **logit** $z_i$ per vocabulary token $i$ — an unnormalized score. Sampling is the layer that turns those scores into an actual next token. The default recipe in this repo: scale logits by $1/T$ (temperature), take a softmax to get a probability distribution over the 128,000-token vocabulary, truncate the distribution to the smallest high-probability set whose cumulative mass reaches `top_p`, then draw one token from that truncated distribution. `temperature <= 0` skips all of it and just takes the argmax (greedy).

Why it matters: greedy decode is deterministic and reproducible, but it loops and degrades on open-ended text; sampling adds controlled stochasticity. GPT-OSS-Lite's headline evaluation — passkey retrieval at up to 131,072 tokens — runs **greedy** (`temperature=0.0`), because a retrieval task has one correct answer and sampling would add variance to the accuracy estimate. Everything lives in `inference/generate.py:generate`; the benchmark harness is `inference/long_context.py:PasskeyEvaluator.evaluate`.

### 1. Why it matters here

- **Generation quality.** Greedy decoding maximizes per-step probability, not
  sequence quality: at every step it re-picks the same high-mass regions, which produces repetitive loops on open-ended generation. Sampling from the temperature-scaled distribution lets the model explore lower-ranked continuations.
- **The headline metric is a retrieval task.** The passkey benchmark inserts a
  five-digit number into filler text and asks the model to repeat it ([inference.md](../inference.md)). There is exactly one right answer, so the eval uses greedy decoding — see §5 for the variance argument.
- **A 502M-parameter budget makes decode cost visible.** The 12-layer stack
  (alternating SWA(128)/full attention) was chosen partly so long-context decode stays cheap: the mixed cache stores only 128 tokens on six layers ([foundations-and-architecture.md](foundations-and-architecture.md), [attention-sinks.md](attention-sinks.md)). The sampling layer itself is O(V) per token ($V = 128000$), negligible next to attention — but it runs once per generated token, so its semantics (and any truncation) are the only thing standing between logits and the emitted text.
- **Honesty boundary.** No pretraining run has happened yet; the ≥85% passkey
  accuracy at 128K is a **target**, not a result. What is measured today: the 2.00× KV reduction at 128K ([operations.md](../guides/operations.md)) and the 192-test CPU suite.

### 2. Intuition

Think of the vocabulary as a terrain over which the model has placed one "altitude" per token — the logit. A softmax is a camera looking at that terrain: it turns altitudes into a probability distribution whose mass concentrates near the peaks. **Temperature** is the zoom knob: $T < 1$ magnifies altitude differences so the highest peak dominates (at $T \to 0$ you see only the summit — greedy); $T > 1$ flattens the terrain so foothills get a fair share (at $T \to \infty$ everything is flat — uniform). **Top-p** then says "ignore everything below this elevation contour": keep only the smallest set of peaks that together hold a fraction $p$ of the total mass, and renormalize. Finally, instead of always planting the flag on the tallest remaining peak (greedy), **sampling** drops a ball on the terrain and lets the mass distribution decide where it lands — usually the peak, occasionally a shoulder. That occasional "shoulder" is what prevents degenerate repetition.

### 3. Theory and derivation

### 3.1 Autoregressive factorization

A language model assigns a probability to a token sequence $x_1, \ldots, x_T$ from a vocabulary $\mathcal{V}$ of size $V$ by the chain rule of probability, conditioned only on the past:

$$
P(x_1, \ldots, x_T) = \prod_{t=1}^{T} P(x_t \mid x_{<t}), \qquad x_{<t} = (x_1, \ldots, x_{t-1})
\tag{1}
$$

The decoder-only transformer (causal mask, [foundations-and-architecture.md](foundations-and-architecture.md)) implements each conditional $P(x_t \mid x_{<t})$ as a softmax over logits $z^{(t)} \in \mathbb{R}^{V}$ produced by the shared stack. This factorization is what makes generation a loop: sample $x_T$, append, sample $x_{T+1}$, repeat — the prefix never needs to be re-sampled.

### 3.2 Greedy decoding

Greedy decoding picks the most probable token at every step:

$$
\hat{x}_t = \arg\max_{v \in \mathcal{V}} P(v \mid x_{<t}) = \arg\max_v z^{(t)}_v
\tag{2}
$$

The second equality holds because softmax is monotonic in its argument. Greedy is the mode of each conditional — but the mode of each conditional is not generally the mode of the joint (1): a low-probability token can be the right choice if it opens a long high-probability continuation. That gap is the theoretical justification for sampling. Greedy is also exactly the $T \to 0$ limit of temperature sampling (§3.3), and `inference/generate.py:generate` treats every `temperature <= 0` as greedy.

### 3.3 Temperature — Boltzmann scaling

The sampling distribution is the softmax of logits scaled by $1/T$:

$$
p_i = \frac{\exp(z_i / T)}{\sum_{j=1}^{V} \exp(z_j / T)}, \qquad T > 0
\tag{3}
$$

This is the Boltzmann distribution of statistical mechanics with $T$ in the role of temperature: states (tokens) with higher energy-like score $z_i$ get exponentially more mass. To derive the limits, shift by the maximum logit $z_{\max} = \max_j z_j$ (numerically stable, probabilities unchanged):

$$
p_i = \frac{\exp\big((z_i - z_{\max})/T\big)}{\sum_{j=1}^{V} \exp\big((z_j - z_{\max})/T\big)}
\tag{4}
$$

**Limit $T \to 0^+$.** Let $A = \{ i : z_i = z_{\max} \}$ be the set of argmax indices. For $i \notin A$, the exponent $(z_i - z_{\max})/T \to -\infty$, so $\exp(\cdot) \to 0$; for $i \in A$ the exponent is $0$ and $\exp(0) = 1$. Hence

$$
\lim_{T \to 0^+} p_i = \begin{cases} 1/|A| & i \in A \\ 0 & i \notin A \end{cases}
\tag{5}
$$

i.e. a uniform draw over the tied argmax tokens — greedy when $|A| = 1$ (the usual case), which is exactly the code path `next_token_logits.argmax(dim=-1)`.

**Limit $T \to \infty$.** Every exponent in (4) tends to $0$, so every $\exp$ tends to $1$ and

$$
\lim_{T \to \infty} p_i = \frac{1}{V},
\tag{6}
$$

the uniform distribution. Greedy and uniform are the two endpoints of the family (3); every finite $T$ interpolates between them.

**What $T = 0.7$ does.** Compare two tokens by probability ratio — a scale-free quantity:

$$
\frac{p_i}{p_j} = \exp\!\left(\frac{z_i - z_j}{T}\right).
\tag{7}
$$

At $T = 1$ the ratio is $\exp(z_i - z_j)$ (plain softmax). At $T = 0.7 < 1$ the exponent is multiplied by $1/T \approx 1.43$, so the ratio between any two tokens is amplified: $p_{\text{high}}$ grows, $p_{\text{low}}$ shrinks, and the distribution becomes **sharper** than the raw softmax while remaining non-degenerate — it leans toward the mode without committing to it. That is the default in `inference/generate.py:generate`. (Training uses plain cross-entropy on the $T=1$ softmax, [training.md](../training.md); temperature is purely a decode-time knob.)

### 3.4 Top-k truncation

Top-k keeps only the $k$ largest logits (equivalently, the $k$ largest probabilities) and renormalizes the rest to zero:

$$
\tilde{p}_i = \begin{cases} p_i \big/ \displaystyle\sum_{j \in S_k} p_j & i \in S_k \\[6pt] 0 & i \notin S_k \end{cases}
\qquad S_k = \{ \text{the } k \text{ indices with largest } z_i \}
\tag{8}
$$

Effect: the tail of the distribution — thousands of near-zero probabilities that together hold real mass — is cut off, so the draw cannot land on a "random" token. The cost is a hard cutoff: when the model is confident, top-k may discard tokens that the model still considers plausible; when it is uncertain (near-uniform), top-k discards genuinely competing options. This repo does **not** implement top-k in `inference/generate.py:generate` — it is the classic baseline that top-p generalizes, and `top_p` is the tunable here.

### 3.5 Top-p (nucleus) sampling

Top-p keeps the *smallest* set of top-ranked tokens whose cumulative probability mass reaches $p$, then renormalizes. Order probabilities $p_{(1)} \ge p_{(2)} \ge \cdots \ge p_{(V)}$ and define

$$
S_p = \left\{ (1), \ldots, (m) \right\}, \qquad
m = \min\left\{ m' : \sum_{i=1}^{m'} p_{(i)} \ge p \right\},
\qquad
\tilde{p}_i = \frac{p_i}{\sum_{j \in S_p} p_j} \cdot \mathbb{1}[i \in S_p]
\tag{9}
$$

The nucleus size $m$ adapts to confidence: a peaked distribution needs $m \approx 1$ (a few tokens cover 90% of the mass), a flat one needs $m \approx V$. This is the property top-k lacks — instead of "always keep 50 tokens," it is "keep however many tokens are actually plausible."

**Interaction with temperature — order matters.** In `inference/generate.py:generate` the pipeline is strictly sequential: **temperature first, then softmax, then top-p on the resulting probabilities**:

```python
probs = F.softmax(next_token_logits / temperature, dim=-1)
sorted_probs, sorted_idx = probs.sort(dim=-1, descending=True)
cumsum = sorted_probs.cumsum(dim=-1)
mask = cumsum - sorted_probs > top_p
sorted_probs[mask] = 0.0
sorted_probs = sorted_probs / sorted_probs.sum(dim=-1, keepdim=True).clamp(min=1e-10)
```

Because temperature reshapes the distribution before the nucleus is selected, the truncation set depends on $T$: at $T = 0.7$ the distribution is sharper, so the nucleus is smaller and more aggressive than it would be at $T = 1$. (Some libraries apply top-p to the raw logits before softmax — a different, temperature-agnostic set.) The mask implements (9) exactly: token $k$ in sorted order is kept iff the mass strictly *before* it satisfies $\sum_{i<k} p_{(i)} \le p$, i.e. the kept prefix is the smallest one with cumulative mass $\ge p$; `top_p = 1.0` never masks anything, and `clamp(min=1e-10)` guards the renormalization against an empty nucleus (e.g. degenerate inputs). The final draw is a single categorical sample per batch row via `torch.multinomial(sorted_probs, 1)`, un-permuted with `sorted_idx.gather(-1, next_id)`.

### 3.6 Entropy and perplexity

The per-token entropy of the predictive distribution quantifies how much the model "knows" at that position:

$$
H(p) = -\sum_{i=1}^{V} p_i \log p_i \quad (\text{nats}), \qquad
H(x_1, \ldots, x_T) = \sum_{t=1}^{T} H\!\big(P(\cdot \mid x_{<t})\big)
\tag{10}
$$

The second identity follows from (1): the entropy of the joint factors into the sum of conditional entropies. Entropy is $0$ for a degenerate (one-hot) distribution and maximal for uniform. The uniform bound, derived from (10) by symmetry ($p_i = 1/V$):

$$
H_{\text{uniform}} = \log V = \log 128000 \approx 11.76 \text{ nats} \approx 16.97 \text{ bits}
\tag{11}
$$

Perplexity is the exponential of the average per-token cross-entropy, which at eval equals entropy of the model's own distribution:

$$
\text{PPL} = \exp\!\left(-\frac{1}{T}\sum_{t=1}^{T} \log P(x_t \mid x_{<t})\right)
= \exp\!\left(\frac{1}{T}\sum_{t=1}^{T} H_t\right)
\tag{12}
$$

For the uniform distribution over this vocabulary, $\text{PPL} = V = 128000$ — a useful sanity floor: a model that beats 128K perplexity has learned *something*. Intuition: perplexity is "the effective number of equally likely choices the model feels at each step." High entropy $\Rightarrow$ many plausible continuations $\Rightarrow$ sampling matters and greedy is brittle; low entropy $\Rightarrow$ the mode is a safe bet. This is precisely the passkey case: after training, the distribution at the answer position should be sharply peaked on the true five-digit string, so greedy is both the right estimator and the low-variance one.

### 4. Code walkthrough — `inference/generate.py:generate`

```python
@torch.no_grad()
def generate(
    model: GPTOSS,
    input_ids: torch.Tensor,
    max_new_tokens: int = 64,
    temperature: float = 0.7,
    top_p: float = 0.9,
    use_cache: bool = True,
) -> torch.Tensor:
```

**Parameter semantics** (`inference/generate.py:generate`):

- `max_new_tokens` — number of decode steps; the return tensor has shape
  `(B, T_prompt + max_new_tokens)`.
- `temperature` — the $T$ of (3). `temperature <= 0` (including exactly `0.0`)
  switches to the greedy branch (5); strictly positive values use the full sampling pipeline. `top_p` is **ignored** in the greedy branch.
- `top_p` — the nucleus mass $p$ of (9), applied after temperature/softmax.
- `use_cache` — when `True`, each decode step runs one token through the model with
  `inference/generate.py:MixedKVCache`; when `False`, the full prefix is replayed every step (an $O(T^2)$ reference path used by the equivalence test, §6).

**Determinism.** `@torch.no_grad()` disables autograd graph construction, and `model.eval()` disables dropout, so the only stochasticity left is `torch.multinomial`, which consumes the global PyTorch RNG: for reproducible sampling runs, call `torch.manual_seed(seed)` before `generate`. The greedy branch consumes no RNG and is deterministic for a fixed model and prompt. (Dropout is the only other RNG consumer in the forward pass, and eval disables it; see
[training.md](../training.md) for the training-time dropout contract.)

**Prefill.** Prompt tokens are embedded and run through all 12 blocks in one pass per layer, caching rotated K/V per layer via `inference/generate.py:_attn_forward_layer` into a fresh `inference/generate.py:MixedKVCache` (windowed layers store only the last 128 tokens, global layers everything — [inference.md](../inference.md)):

```python
x = model.embed(input_ids)
positions = torch.arange(T_prompt, device=dev)
for layer_idx, block in enumerate(model.blocks):
    x = _attn_forward_layer(block, layer_idx, x, positions, cache, sink_bias_cache)
x = model.norm(x)
next_token_logits = model.head(x)[:, -1, :]
```

`models/transformer.py:GPTOSS.head` is the final linear to $V = 128000$ logits, weight-tied to `models/transformer.py:GPTOSS.embed` (saves ~98M parameters on the 502M budget, [foundations-and-architecture.md](foundations-and-architecture.md)).

**Decode loop.** One iteration per new token:

```python
for step in range(max_new_tokens):
    if temperature <= 0:
        next_id = next_token_logits.argmax(dim=-1, keepdim=True)
    else:
        probs = F.softmax(next_token_logits / temperature, dim=-1)
        sorted_probs, sorted_idx = probs.sort(dim=-1, descending=True)
        cumsum = sorted_probs.cumsum(dim=-1)
        mask = cumsum - sorted_probs > top_p
        sorted_probs[mask] = 0.0
        sorted_probs = sorted_probs / sorted_probs.sum(dim=-1, keepdim=True).clamp(min=1e-10)
        next_id = torch.multinomial(sorted_probs, 1)
        next_id = sorted_idx.gather(-1, next_id)
    output[:, T_prompt + step : T_prompt + step + 1] = next_id
    cur_pos += 1
```

- Greedy branch: `argmax` over logits — (2)/(5).
- Sampling branch: temperature scaling (3) → softmax → sort descending →
  cumulative mass → mask (9) → renormalize with an epsilon floor → one multinomial draw per row → map sorted indices back to vocabulary indices.
- The new token is written into a pre-allocated `output` buffer of shape
  `(B, T_prompt + max_new_tokens)`.

With `use_cache=True`, the next logits come from embedding only the new token at its **absolute** position `positions_step = torch.tensor([cur_pos - 1])` and running one decode step per layer (append length-1 K/V to the cache, attend, MoE):

```python
x_step = model.embed(next_id)
positions_step = torch.tensor([cur_pos - 1], device=dev)
for layer_idx, block in enumerate(model.blocks):
    x_step = _attn_forward_layer(block, layer_idx, x_step, positions_step, cache, sink_bias_cache)
x_step = model.norm(x_step)
next_token_logits = model.head(x_step)[:, -1, :]
```

With `use_cache=False` the same logits are recomputed by re-embedding the whole prefix `output[:, :T_prompt + step + 1]` and re-running every block with `cache=None` — the two paths agree by construction, and `tests/test_inference.py` asserts they produce identical output for `max_new_tokens=1`. Absolute positions matter here: YaRN extrapolates from the trained 4,096 to 131,072 via scale-32 RoPE ([attention-and-positional.md](attention-and-positional.md)), and a wrong relative offset would silently corrupt every attention layer's positional signal at long context.

### 5. The passkey eval path — why `temperature=0.0`

`inference/long_context.py:PasskeyEvaluator.evaluate` loops over context lengths `(4096, 8192, 32768, 65536, 131072)`, samples 100 distinct five-digit passkeys per length, builds a prompt via `inference/long_context.py:PasskeyEvaluator.build_prompt` over deterministic filler (`inference/long_context.py:make_filler_text`), and calls:

```python
output_ids = generate(
    self.model,
    input_ids,
    max_new_tokens=16,
    temperature=0.0,
    top_p=1.0,
    use_cache=True,
)
```

`temperature=0.0` routes to the argmax branch; `top_p=1.0` is inert. The first five-digit number in the decoded continuation is extracted by `inference/long_context.py:PasskeyEvaluator.extract_passkey_from_output` and matched against the ground truth.

**Why greedy is the right eval choice.** Per trial, the model either answers correctly (a Bernoulli variable with success probability $q$). With $n = 100$ trials the estimated accuracy has standard deviation

$$
\sigma = \sqrt{\frac{q(1-q)}{n}} \approx 0.036 \quad \text{at } q = 0.85,\ n = 100
\tag{13}
$$

if the draws were i.i.d. Greedy decoding removes the sampling layer from the loop entirely: for a fixed prompt the completion is a deterministic function of the weights, so the only variation left across trials is the passkey/filler content — the estimator measures *the model*, not the sampler's luck. Sampling at the answer position would occasionally draw a non-mode token (a wrong digit), depressing and noising the measured accuracy. Since the task is a copy/retrieval task — one correct answer, sharply peaked distribution — the mode *is* the answer, and greedy is both the highest-accuracy and lowest-variance choice (§3.6).

**Caveat:** the ≥85% target at 128K is a **target**. No pretraining run has completed; `scripts/passkey_eval.py` on an untrained checkpoint produces near-chance accuracy (the passkey is one of 100,000 uniformly sampled values, so chance is $10^{-5}$ per trial) and exits 0 with a warning, not an error.

### 6. Pitfalls and verification

| Pitfall | Symptom | Guard |
|---------|---------|-------|
| Assuming `top_p` applies under `temperature <= 0` | Greedy ignores `top_p` entirely | Read the branch in `inference/generate.py:generate`; passkey eval sets `top_p=1.0` for exactly this reason |
| Temperature/top-p ordering confusion | Truncation set changes if top-p runs on logits pre-softmax | This repo: temperature → softmax → top-p (code order in §3.5) |
| Non-reproducible sampling runs | Different output per launch with `temperature > 0` | `torch.manual_seed(seed)` before `generate`; greedy (`temperature=0.0`) is seed-independent |
| `top_p=0.0` with `temperature > 0` | Only the single top token survives the mask | Not a crash — `clamp(min=1e-10)` prevents division by zero, but the result is effectively greedy with tie-breaking by `multinomial` |
| Using sampling in passkey eval | Accuracy becomes a noisy random variable (13) | Keep `temperature=0.0` in `inference/long_context.py:PasskeyEvaluator.evaluate` |
| Breaking cache vs no-cache equivalence | Subtle positional drift at long context | `tests/test_inference.py` asserts `use_cache=False` matches `use_cache=True` for one token |
| Reading the ≥85% number as a result | It is a target; untrained checkpoints score ~0% | `scripts/passkey_eval.py` prints the warning line and exits 0 |

**Verify the claims:**

```bash
pytest tests/test_inference.py -v
```

covers: generation output shape (`(1, 4 + 8)` for `max_new_tokens=8`, `temperature=0.0`), greedy no-crash, the `use_cache` equivalence property, the KV ring-buffer order invariants, and passkey prompt construction / regex extraction. The sampling branch itself is exercised through the shape and no-crash tests with `temperature=0.0`; stochastic-draw behavior is deterministic given a seed and can be checked by calling `generate` twice with `torch.manual_seed(0)` and comparing outputs. The full passkey pipeline needs a trained checkpoint:

```bash
python3 scripts/passkey_eval.py --checkpoint path/to/model.safetensors --n-trials 100
```

On an untrained model this runs end-to-end, prints an accuracy table at ~0% per length, and warns "needs trained checkpoint for ≥ 85% target" — expected, not a bug ([getting-started.md](../guides/getting-started.md) §12).

### Where to go next

| Topic | Document |
|-------|----------|
| Softmax derivation, causal mask, scaled dot product | [attention math](attention-and-positional.md) |
| Decode loop mechanics: mixed KV cache, ring buffer, O(1) append | [kv cache engineering](../inference.md), [inference.md](../inference.md) |
| Cross-entropy objective that shapes the logits | [training.md](../training.md) |
| Position handling (absolute positions, YaRN 4K→128K) | [attention-and-positional.md](attention-and-positional.md) |
| Vocabulary size and tokenization | [tokenization.md](tokenization.md) |
| BF16 precision of logits/softmax | [numerics](optimizers-and-numerics.md) |

## References

- [`training/pretrain.py`](../../training/pretrain.py) — optimizer construction, autocast, NaN guard
- [`models/attention.py:SINK_CLAMP_MIN`](../../models/attention.py) — sink clamp bounds
- [`models/attention.py:SINK_CLAMP_MAX`](../../models/attention.py) — sink clamp bounds
- [`models/moe.py:MoERouter.forward`](../../models/moe.py) — FP32 router softmax
- [`inference/generate.py:generate`](../../inference/generate.py) — sampling entry point
- [training.md](../training.md) — AdamW settings, BF16 autocast, chunked CE
- [attention-and-positional.md](attention-and-positional.md) — attention softmax derivation
- [moe.md](moe.md) — routing instability, FP32 islands

<!-- docs:verified 2026-08-05 · 6491066 -->
