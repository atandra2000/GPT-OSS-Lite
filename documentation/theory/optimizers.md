# Optimizers — From Gradient Descent to AdamW

> **Chapter T4.** The optimizer is the only component that touches every
> parameter of a 502M-parameter model on every one of 61,000 steps. This
> chapter derives the machinery — momentum, Adam's bias correction, decoupled
> weight decay, mixed-precision state, gradient clipping, warmup — and then
> reads the exact construction in `training/pretrain.py`.
>
> **Related:** [training.md](../training.md) §11 (optimizer section) ·
> [foundations.md](../foundations.md) §10 (BF16 vs FP16) ·
> [numerics](numerics.md) (IEEE-754 bit layouts) ·
> [moe.md](../moe.md) §16 (numerical stability) ·
> [moe theory](moe_theory.md) (routing instability).

---

## 1. Sixty-second summary

The training loop optimizes a ~502M-parameter (247M active) top-2-of-8 MoE
decoder with AdamW: first and second moment estimates with bias correction,
decoupled weight decay, FP32 optimizer state under a BF16 forward pass, global
gradient-norm clipping at 1.0, and a 3000-step linear warmup followed by cosine
decay to 5% of peak LR. The optimizer is constructed in
`training/pretrain.py:main` with two parameter groups (decay vs no-decay),
`foreach=True`, `fused=True` on CUDA, and `eps=1e-6`. Every design choice —
β₁=0.9, β₂=0.95, wd=0.1, lr=4e-4, clip=1.0, warmup — exists because this is a
short (8B-token), single-pass pretraining run over a sparsely routed model
whose gradients mix two objectives (cross-entropy and an auxiliary load-balance
loss). No pretraining run has completed yet; all runtime figures are targets.

## 2. Why it matters here

GPT-OSS-Lite budgets exactly 61,000 steps × 131,072 tokens ≈ 8.0B tokens — a
Chinchilla-optimal single pass over the corpus. There is no second epoch to
recover from a bad optimizer setting. Three properties of this model make the
optimizer load-bearing:

- **MoE routing noise.** The router in `models/moe.py:MoERouter` selects 2 of 8
  experts per token. Early in training the gate is near-random, so per-batch
  gradients through the router are large and the auxiliary loss
  `models/moe.py:aux_load_balancing_loss` (α=0.01) injects a second gradient
  signal roughly an order of magnitude smaller than the CE gradient. Adam's
  per-coordinate scaling and the 3000-step warmup are what keep the gate from
  collapsing onto one expert (see [moe theory](moe_theory.md)).
- **Weight decay must not touch everything.** RMSNorm gains are scale
  parameters, the sink bias is a learned offset clamped to $[-10, 15]$
  (`models/attention.py:SINK_CLAMP_MIN/MAX`), and the embedding is tied to the
  head. Decaying them is either meaningless or harmful, so the optimizer splits
  parameters into decay / no-decay groups.
- **BF16 arithmetic.** The forward pass runs in BF16 (7 mantissa bits) but the
  optimizer state runs in FP32. Getting that split wrong silently rounds away
  most updates (derived in §4.6).

## 3. Intuition

Think of the loss surface as terrain and the optimizer as a hiker with three
improvements over naive hill descent. First, a **ball** (momentum): it keeps
rolling in the direction it was already going, so it averages out noisy
measurements of the slope. Second, a **per-axis ruler** (Adam): instead of one
step size for every direction, it measures the typical slope magnitude along
each coordinate and takes a step of *relative* size — this is what makes
training work when one coordinate's gradient is 1000× another's. Third, a
**speed limit and a launch ramp** (clipping + warmup): the hiker never takes a
step longer than a fixed bound, and starts at rest, because the first slope
measurements are the noisiest ones.

## 4. Theory + derivation

### 4.1 Gradient descent and the fixed-step limit

Let $L(\theta)$ be the training objective and $g_t = \nabla_\theta L(\theta_t)$
the gradient at step $t$. Gradient descent with learning rate $\eta$:

$$
\theta_{t+1} = \theta_t - \eta g_t. \tag{1}
$$

Expand the loss after the step to second order, with Hessian
$H_t = \nabla^2_\theta L(\theta_t)$:

$$
L(\theta_t - \eta g_t) = L(\theta_t) - \eta\, g_t^{\top} g_t
+ \tfrac{1}{2}\eta^2 g_t^{\top} H_t g_t + O(\eta^3). \tag{2}
$$

The step decreases the loss only while $\eta < 2\, g^{\top}g / (g^{\top}Hg)$,
and the step size that maximizes the decrease is
$\eta^* = g^{\top}g / (g^{\top}Hg)$. The trouble is visible in a
coordinate-separable quadratic $L = \tfrac{1}{2}\sum_i h_i \theta_i^2$ (hessian
eigenvalues $h_i$): gradient descent gives $\theta_i \leftarrow (1-\eta h_i)\theta_i$,
which converges only for $\eta < 2/h_{\max}$, and the convergence rate is set by
the condition number $\kappa = h_{\max}/h_{\min}$. In contrast, a *per-coordinate*
step $\eta_i = 1/h_i$ converges in one step. Real transformer losses have a
huge spread of $h_i$ (attention logits vs embedding rows), so a single global
$\eta$ is forced to be small enough for the stiffest direction. This is the
entire motivation for the adaptive scaling Adam implements.

### 4.2 Momentum: the EMA of gradients

Momentum replaces the raw gradient with an exponential moving average
$m_t$ ("velocity"), with decay $\beta_1 \in (0,1)$:

$$
m_t = \beta_1 m_{t-1} + (1-\beta_1) g_t, \qquad m_0 = 0. \tag{3}
$$

Unrolling (3) expresses $m_t$ as a weighted sum of the past gradients:

$$
m_t = (1-\beta_1) \sum_{i=0}^{t-1} \beta_1^{i}\, g_{t-i}. \tag{4}
$$

The weights sum to $\sum_{i=0}^{t-1} (1-\beta_1)\beta_1^i = 1 - \beta_1^t < 1$:
at small $t$ the estimate is systematically *too small* — biased toward zero,
because the missing early terms are absent rather than zero. For a stationary
gradient $g$, $\mathbb{E}[m_t] = (1-\beta_1^t)\,g$, so dividing by
$1-\beta_1^t$ removes the bias:

$$
\hat{m}_t = \frac{m_t}{1-\beta_1^t}. \tag{5}
$$

The window of the EMA: the coefficient of $g_{t-k}$ is $\beta_1^k$, which
halves at $k = \ln 2 / \ln(1/\beta_1)$ — about 6.6 steps for $\beta_1 = 0.9$,
i.e. the velocity averages roughly the last seven gradients.

### 4.3 Adam: per-coordinate adaptive steps

Adam (Kingma & Ba, 2015) maintains a second EMA over *squared* gradients
$v_t$, with its own decay $\beta_2$:

$$
m_t = \beta_1 m_{t-1} + (1-\beta_1) g_t, \qquad
v_t = \beta_2 v_{t-1} + (1-\beta_2) g_t^2. \tag{6}
$$

If the gradient components are stationary with $\mathbb{E}[g_i^2] = \nu_i$, the
same unrolling argument gives $\mathbb{E}[v_t] = (1-\beta_2^t)\,\nu$, so the
*second* moment needs its own correction — this is why the correction exponent
must be the *age* $t$, not the step number modulo anything:

$$
\hat{m}_t = \frac{m_t}{1-\beta_1^t}, \qquad
\hat{v}_t = \frac{v_t}{1-\beta_2^t}. \tag{7}
$$

The parameter update is then:

$$
\theta_{t+1} = \theta_t - \eta\, \frac{\hat{m}_t}{\sqrt{\hat{v}_t} + \varepsilon}, \tag{8}
$$

where $\varepsilon > 0$ prevents division by zero and sets the noise floor.
Per coordinate, $\sqrt{\hat{v}_{t,i}}$ is the RMS of recent gradients, so (8)
implements exactly the per-coordinate scaling $\eta_i \propto 1/\sqrt{h_i}$ of
§4.1 — the effective preconditioner is $\text{diag}(\hat{v})^{-1/2}$.

Two consequences are worth deriving because they explain design choices later.
First, **scale invariance**: if every gradient were multiplied by $c > 0$
(e.g. rescaling the whole loss), then $\hat{m} \to c\hat{m}$ and
$\sqrt{\hat{v}} \to c\sqrt{\hat{v}}$, so the ratio — and the step — is
unchanged:

$$
\frac{c\hat{m}}{c\sqrt{\hat{v}}} = \frac{\hat{m}}{\sqrt{\hat{v}}}. \tag{9}
$$

This is why one optimizer can absorb the mixed CE + α·aux objective of this
repo: the two terms differ by an order of magnitude in gradient scale, but Adam
is invariant to the overall scale of each coordinate's gradient history.
Second, **early-steps behavior**: for $t = 1$ the correction gives
$\hat{m}_1 = g_1$ and $\hat{v}_1 = g_1^2$, hence
$\hat{m}_1 / \sqrt{\hat{v}_1} = \operatorname{sign}(g_1)$ — the very first Adam
step moves *every* coordinate by exactly $\pm \eta$, regardless of gradient
magnitude. This is the mathematical reason warmup exists (§4.8).

### 4.4 AdamW: decoupled weight decay

L2 regularization adds $\tfrac{1}{2}\lambda\|\theta\|^2$ to the loss, so the
gradient becomes $g + \lambda\theta$ and Adam's update becomes (absorbing the
regularizer into the moments):

$$
\theta_{t+1} = \theta_t - \eta\, \frac{\hat{m}_t + \lambda\theta_t}
{\sqrt{\hat{v}_t} + \varepsilon}
\;\approx\; \theta_t - \eta\, \frac{\lambda\theta_t}{\sqrt{\hat{v}_t}+\varepsilon}
- \eta\, \frac{\hat{m}_t}{\sqrt{\hat{v}_t}+\varepsilon}. \tag{10}
$$

The approximation keeps the dominant term: because $\hat{v}$ is an EMA, the
decay term $\lambda\theta_t$ enters the update *divided by* $\sqrt{\hat{v}_t}$,
and it is also smeared through the momentum history. The decay is therefore
per-coordinate adaptive: coordinates with large past gradients (large
$\sqrt{\hat{v}}$) get *less* decay, so the effective regularization varies by
orders of magnitude across a weight matrix. AdamW (Loshchilov & Hutter, 2019)
moves the decay out of the moments entirely:

$$
\theta_{t+1} = (1 - \eta\lambda)\, \theta_t
- \eta\, \frac{\hat{m}_t}{\sqrt{\hat{v}_t} + \varepsilon}. \tag{11}
$$

In (11) the shrinkage factor $(1 - \eta\lambda)$ is identical for every
coordinate and every step; with no gradient the weight decays exactly
geometrically, $\theta_t = (1-\eta\lambda)^t \theta_0$. Comparing (10) and
(11), the difference is the placement of $\lambda\theta$:

$$
\text{L2-Adam: decay} = \eta\lambda\theta\,/\,(\sqrt{\hat v}+\varepsilon)
\qquad
\text{AdamW: decay} = \eta\lambda\theta. \tag{12}
$$

Why decoupling wins for transformers: transformer weight matrices are
reparameterization-redundant with the following norm layer (RMSNorm rescales
its input), so what matters is the *magnitude* of the weights feeding
activations, not the loss contribution of their norm; a uniform, predictable
shrinkage (11) keeps weight norms in a steady state, while the adaptive decay
of (10) lets coordinates with long gradient histories escape regularization
exactly where drift is likeliest (large matrices). Every mainstream LLM recipe
(GPT-3, LLaMA, DeepSeek-V3) uses the decoupled form.

### 4.5 Hyperparameters for this run

From `configs/pretrain_a100_502m.yaml` (training section): `beta1: 0.9`,
`beta2: 0.95`, `weight_decay: 0.1`, `lr: 4.0e-4`, `min_lr_ratio: 0.05`,
`warmup_steps: 3000`, `total_steps: 61000`, `grad_clip: 1.0`, `eps: 1e-6`.

- **β₁ = 0.9.** Half-life ≈ 6.6 steps (§4.2): the velocity averages roughly
  one gradient-accumulation window (4 micro-batches) plus a bit. Lower values
  track noise, higher values lag schedule changes.
- **β₂ = 0.95.** The variance window is deliberately short (half-life ≈ 13.5
  steps, effective sample count $1/(1-\beta_2) = 20$) versus the Adam default
  0.999 (1000 steps). Pretraining here runs 61K steps over a cosine schedule;
  a 1000-step variance window would feed stale $\hat{v}$ to the
  preconditioner after every LR change. LLaMA-style recipes use the same 0.95.
- **wd = 0.1.** Per-step relative shrinkage $\eta\lambda = 4\times10^{-5}$ at
  peak LR: small enough that gradients dominate, large enough to counteract
  slow drift of expert matrices and router logits. It is a standard
  pretraining value (GPT-3, LLaMA), not a tuned one.
- **lr = 4e-4 with init_std = 0.02.** A step of size $\eta$ is 2% of the
  typical weight magnitude set by `init_std` — large enough to reorganize
  weight norms within a few hundred steps, small enough not to scramble the
  residual stream. Adam's step per coordinate is $O(\eta)$ (§4.3), so the LR
  *is* the step size; 4e-4 at 131K tokens/step over 8B tokens is the standard
  Chinchilla-scale recipe.
- **min_lr_ratio = 0.05.** Cosine floor at $2\times10^{-5}$; the schedule never
  reaches zero, so the final steps still refine rather than add noise.
- **eps = 1e-6.** See §6 (pitfall 2); it is the BF16-safe floor.

### 4.6 Vectorized updates and FP32 master weights

BF16 stores 1 sign + 8 exponent + 7 mantissa bits; FP32 stores 23 mantissa
bits. For a value in $[2^e, 2^{e+1})$, the spacing between representable
numbers is $2^{e-7}$ (BF16) or $2^{e-23}$ (FP32), giving *relative* spacings:

$$
\text{BF16: } 2^{-8} \approx 3.9\times10^{-3}, \qquad
\text{FP32: } 2^{-23} \approx 1.2\times10^{-7}. \tag{13}
$$

A weight near the init scale $|\theta| \approx 0.02$ therefore has a BF16
rounding quantum of $\approx 7.8\times10^{-5}$, while a typical Adam update
has magnitude $|\Delta\theta| \approx \eta\,|\hat m/\sqrt{\hat v}| \sim 10^{-4}$
— the same order as the quantum. If weights were stored and updated in BF16, a
large fraction of updates would round to zero. FP32 at the same magnitude has a
quantum of $\approx 2.4\times10^{-9}$, four orders of magnitude finer. Hence
the split used here: parameters stay FP32 ("master weights"), the optimizer
state ($m, v$) is FP32, and autocast casts to BF16 *only at matmul boundaries*
inside the forward pass. Note that with BF16 autocast, PyTorch never casts
parameters themselves — so the master weights come for free. The optimizer
state costs $2 \times 501836640 \times 4\text{B} \approx 4.0$ GB, plus
$501836640 \times 4\text{B} \approx 2.0$ GB of FP32 gradients during
accumulation. Crucially, the state covers **all** 501.8M parameters: MoE
sparsity saves FLOPs and activations, but every expert occasionally receives
gradients, so Adam must maintain moments for the experts not routed in a given
step too (and AdamW still decays them, §6 pitfall 3).

**foreach / fused.** A naive optimizer loops over parameter tensors (~400 here:
embedding, 12 layers × attention/moe/norm matrices, gate, experts) and launches
several elementwise kernels per tensor — roughly 6 ops × 400 tensors ≈ 2,400
kernel launches per step, each paying a fixed launch cost. `foreach=True` packs
a parameter group's tensors into one kernel that iterates the tensor list
inside the kernel with vectorized loads, amortizing launches and reaching
memory bandwidth; `fused=True` goes further with a single kernel per group that
computes the whole AdamW step in one pass over flattened state. The
~1.5–2× speedup versus the default Python loop on A100 cited in
[training.md](../training.md) is an **`[INFERENCE]` estimate** — `.benchmarks/`
is empty. `fused` requires all parameters of a group to share a dtype and be
contiguous, which the all-FP32 state here satisfies; the code gates it on
`dev.type == "cuda"` and falls back to `foreach` on CPU.

### 4.7 Global-norm gradient clipping

Let $g$ be the flattened gradient of all $N$ parameters and $G$ its Euclidean
norm:

$$
G = \|g\|_2 = \left( \sum_{i=1}^{N} g_i^2 \right)^{1/2}. \tag{14}
$$

Clipping projects $g$ onto the $\ell_2$ ball of radius $c$ — the closest vector
to $g$ with norm $\le c$, found by rescaling the whole gradient by one scalar:

$$
g_{\text{clipped}} = g \cdot \min\!\left(1, \frac{c}{G}\right), \qquad
\|g_{\text{clipped}}\|_2 \le c. \tag{15}
$$

The direction of $g$ is untouched and every coordinate is scaled by the *same*
factor, so relative magnitudes — which Adam respects via (9) — are preserved.
The bound on the update: because $|g_i| \le G \le c$ after clipping, the
per-coordinate step in (8) is bounded by $O(\eta)$ rather than
$\eta \times (\text{spike}/\text{typical})$. With $N \approx 5\times10^8$
dimensions, an unclipped norm of $G \approx \sqrt{N}\,\sigma$ (for
per-coordinate std $\sigma$) exceeds $c = 1.0$ whenever $\sigma \gtrsim
4.5\times10^{-5}$ [derived estimate]; so clipping is a safety valve that binds
during warmup and routing instability, not a constant brake.

### 4.8 Warmup: protecting the biased moments

The schedule multiplies the entire update: $\eta \to \eta\lambda(t)$ with
$\lambda(t) = t/w$ for $t < w$ (linear warmup), then cosine decay to the floor.
Section 4.3 showed the first Adam step is $\pm\eta$ in *every* coordinate
because one sample makes $\hat m/\sqrt{\hat v}$ a pure sign. With warmup the
effective step is:

$$
\theta_{t+1} = \theta_t - \eta\,\lambda(t)\, \frac{\hat m_t}{\sqrt{\hat v_t}+\varepsilon}, \qquad
\lambda(t) = \min\!\left(\frac{t}{w},\, \text{cosine}\right), \tag{16}
$$

so the early steps grow linearly from zero while $\hat v$ accumulates its
effective $1/(1-\beta_2) = 20$ samples. The relative error of the variance
estimate after $n$ samples scales as $\sim 1/\sqrt{n}$: at $n = 20$ that is
~22%, at $n = 1$ it is 100% — warmup does not fix the bias (the correction
(7) already does), it bounds the *variance* of early steps. The repo's choice
of $w = 3000$ (4.9% of the run) sits in the 2–5% band recommended for MoE
training per the YAML comment, because router gradients are largest and noisiest
exactly during warmup. Note also that $\lambda(t)$ multiplies the decoupled
decay in (11) too — weight decay ramps up with the learning rate, which is
desirable: don't shrink weights before gradients have set their scale.

## 5. Code walkthrough

### 5.1 Optimizer construction in `training/pretrain.py:main`

`training/pretrain.py:main` builds two parameter groups by name:

```python
no_decay = ["bias", "norm", "embed"]
decay_params = [p for n, p in model.named_parameters() if not any(nd in n.lower() for nd in no_decay)]
no_decay_params = [p for n, p in model.named_parameters() if any(nd in n.lower() for nd in no_decay)]
```

Substring matching: any parameter whose name contains `bias`, `norm`, or
`embed` is decay-free. This covers the RMSNorm gains and biases everywhere,
and the tied embedding — the word embedding and the output head are the *same*
`Parameter` tensor (`weight_tying: true` in the config), and
`named_parameters()` lists the shared tensor once under the embedding's name,
so excluding `embed` excludes the tied head as well. The two groups then go
into a single `AdamW`:

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

Every hyperparameter comes from `configs/pretrain_a100_502m.yaml` (or its
default), so the schedule and the optimizer cannot drift out of sync with the
recipe. `fused=(dev.type == "cuda")` is the only hardware-dependent flag: on
CPU the same code path runs with `foreach`.

### 5.2 Schedule interaction

The schedule is a `LambdaLR` wrapping the lambda returned by
`training/pretrain.py:make_warmup_cosine_lambda`:

```python
sched = LambdaLR(optim, make_warmup_cosine_lambda(
    train_cfg["warmup_steps"],
    train_cfg["total_steps"],
    train_cfg.get("min_lr_ratio", 0.05),
))
```

`make_warmup_cosine_lambda` returns `lr_lambda(step)` where
`step / warmup_steps` for `step < warmup_steps`, the cosine
`min_lr_ratio + (1 - min_lr_ratio) * 0.5 * (1 + cos(π · progress))` afterwards,
and the constant `min_lr_ratio` past `total_steps` — the exact schedule of
(16). `LambdaLR` multiplies the group's base LR by this factor, which is why
warmup scales the *whole* AdamW update including decay (§4.8). The schedule's
three boundary invariants — zero at step 0, peak at the warmup boundary, floor
at the end — are pinned by `tests/test_training.py::test_lr_schedule_at_warmup_boundary`,
`test_lr_schedule_at_end`, and `test_lr_schedule_monotonic_decay_after_warmup`.

### 5.3 The step: clipping, NaN guard, and optimizer state

Inside the accumulation loop (after `loss = (ce + aux_alpha * aux_loss) / accum`
in `training/pretrain.py:main`), the optimizer runs only on accumulation
boundaries:

```python
if grad_clip > 0:
    nn.utils.clip_grad_norm_(model.parameters(), grad_clip, foreach=True)
optim.step()
sched.step()
optim.zero_grad(set_to_none=True)
```

`clip_grad_norm_` implements (14)–(15) over all parameters before `step()`;
the `try/except TypeError` around it is a version-compat shim for the
`foreach` kwarg. Dividing the loss by `accum` before `backward()` keeps the
accumulated gradient at single-micro-batch scale, so the clip threshold is
meaningful across `accum=4`. `zero_grad(set_to_none=True)` frees the FP32
gradient buffers (2.0 GB) after each step instead of zeroing them.

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

When the loss is non-finite the loop **skips `optim.step()` entirely** — it
never lets a poisoned gradient touch Adam state. That is load-bearing for a
running-moment optimizer: in (6), if any $g_i$ is NaN then
$v_t = \beta_2 v_{t-1} + (1-\beta_2)\text{NaN}^2$ is NaN, and because $\beta_2
v_t$ keeps the NaN in the running sum,

$$
v_{t+k} = \beta_2^k\, \text{NaN} + \sum_{j=1}^{k} \beta_2^{k-j}(1-\beta_2)\, g_{t+j}^2
= \text{NaN} \tag{17}
$$

for *every* future $k$ — one NaN gradient permanently poisons Adam's second
moment; the state cannot heal itself. The guard prevents the poison from
entering, and the rollback branch (after `nan_guard_max_consecutive: 5`
consecutive NaNs) restores $m, v$ from the last checkpoint
(`tests/test_training.py::test_checkpoint_round_trip` covers the
optimizer-state round trip, and `test_nan_guard_detection` pins the
`torch.isfinite` check). Reproducibility of the whole trajectory rests on
`training/pretrain.py:seed_everything`, which seeds Python, NumPy, and PyTorch
RNGs and sets `CUBLAS_WORKSPACE_CONFIG=:4096:8` for deterministic cuBLAS
before the model — and therefore before Adam's zero-initialized moments — are
created.

## 6. Pitfalls + verify

1. **NaN through the optimizer state is permanent.** One NaN gradient makes
   $v_t$ NaN forever ((17)); a NaN *loss* is caught and skipped, but a finite
   loss with a NaN gradient (e.g. overflow inside the router softmax) would
   not be. Guard: `tests/test_training.py::test_nan_guard_detection` plus
   watching the `[nan-guard]` log lines during a smoke run
   (`scripts/e2e_gpu_smoke.py`).
2. **eps=1e-6, not 1e-8.** The classical eps=1e-8 fails outright under FP16
   (min normal $6.1\times10^{-5}$, min subnormal $6\times10^{-8}$ — 1e-8
   rounds to zero in the second moment). Under BF16 the range is FP32-like, so
   nothing underflows, but with only 7 mantissa bits (13) a variance stored
   near 1e-8 carries ~0.4% relative error, and when $\sqrt{\hat v} \ll \varepsilon$
   the denominator of (8) is dominated by $\varepsilon$: the step becomes
   $\eta\hat m/\varepsilon$, amplifying gradient noise instead of normalizing
   it. 1e-6 is a safe floor given typical BF16 gradient magnitudes; this is
   the setting DeepSeek-V3 and LLaMA-3 use, and it is pinned in the config
   comment and [moe.md](../moe.md) §16.
3. **MoE: the optimizer sees all 501.8M parameters.** Unrouted experts receive
   zero gradient, but AdamW still applies the decoupled decay (11) to them and
   keeps their moments alive for when they are next selected. The 4.0 GB
   optimizer state is over *total* parameters — sparsity does not reduce it.
   Consequence: an expert starved for many steps still shrinks at
   $\eta\lambda$ per step, which is part of why the aux loss (α=0.01) must
   keep routing balanced (see [moe theory](moe_theory.md)).
4. **Warmup is not the bias correction.** The correction (7) makes the moments
   unbiased from step 1; warmup (16) bounds the *variance* of the early
   sign-like steps (§4.3, §4.8). Cutting warmup on this model means the first
   step moves every coordinate by $\pm\eta$ — with 8 routed experts' gates at
   init, that is a reliable way to seed expert collapse. Verify the boundary
   behavior with the three schedule tests in `tests/test_training.py`.
5. **`fused` silently requires uniformity.** `fused=True` demands one dtype per
   group and contiguous tensors; if a future change casts some parameters to
   BF16 in place, the fused path either errors or (worse) falls back without
   telling you. The A100 speedup claims (1.5–2×, 35–40% MFU) are
   **`[INFERENCE]`** until `.benchmarks/` is populated.

---

*Verified 2026-08-04 against `training/pretrain.py`, `configs/pretrain_a100_502m.yaml`,
`tests/test_training.py`, and [training.md](../training.md). No pretraining run
has completed; all performance figures are targets or `[INFERENCE]`.*

<!-- docs:verified 2026-08-04 · 5da1a80 -->
