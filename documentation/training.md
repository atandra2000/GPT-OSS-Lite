# Pre-Training Loop — GPT-OSS-Lite

## A Complete Book Chapter on `training/pretrain.py`

> **Config reference:** [`configs/pretrain_a100_502m.yaml`](../configs/pretrain_a100_502m.yaml)

> **Related:** [data_pipeline.md](data_pipeline.md) (corpus and loader),
> [moe.md](moe.md) (aux loss α=0.01, Triton opt-in)
> (optional `moe_dispatch`).

---

## Table of Contents

1. [Abstract](#abstract)
2. [Training Objective](#training-objective)
3. [Chinchilla Budget](#chinchilla-budget)
4. [Configuration Walkthrough](#configuration-walkthrough)
5. [Entry Point and CLI](#entry-point-and-cli)
6. [Reproducibility — `seed_everything`](#reproducibility--seed_everything)
7. [Hardware Performance Knobs](#hardware-performance-knobs)
8. [Model Construction](#model-construction)
9. [`torch.compile`](#torchcompile)
10. [Memory Estimation](#memory-estimation)
11. [Optimizer — AdamW with Parameter Groups](#optimizer--adamw-with-parameter-groups)
12. [Learning Rate Schedule](#learning-rate-schedule)
13. [Data Loading](#data-loading)
14. [The Training Step — Inner Loop](#the-training-step--inner-loop)
15. [Mixed Precision — BF16 Autocast](#mixed-precision--bf16-autocast)
16. [Chunked Cross-Entropy](#chunked-cross-entropy)
17. [Auxiliary MoE Loss Integration](#auxiliary-moe-loss-integration)
18. [Gradient Accumulation](#gradient-accumulation)
19. [Gradient Clipping](#gradient-clipping)
20. [Gradient Checkpointing](#gradient-checkpointing)
21. [NaN Guard and Checkpoint Rollback](#nan-guard-and-checkpoint-rollback)
22. [Logging](#logging)
23. [Checkpointing — `CheckpointManager`](#checkpointing--checkpointmanager)
24. [Resume Training](#resume-training)
25. [RNG State Persistence](#rng-state-persistence)
26. [End-to-End Timeline](#end-to-end-timeline)
27. [Operational Commands](#operational-commands)
28. [Debugging Checklist](#debugging-checklist)
29. [Appendix A — Step arithmetic](#appendix-a--step-arithmetic)
30. [Appendix B — LR schedule samples](#appendix-b--lr-schedule-samples)
31. [Appendix C — File map](#appendix-c--file-map)
32. [Part B — Configuration reference](#part-b--configuration-reference)
33. [Load-Bearing Invariants](#load-bearing-invariants)
34. [References](#references)

---

## Abstract

[`training/pretrain.py`](../training/pretrain.py) is the sole pre-training script
for GPT-OSS-Lite. It implements a **from-scratch PyTorch loop** — no HuggingFace
Trainer, no Lightning. The default A100 recipe trains a **~502 M total /
~247 M active** model for **61,000 optimizer steps** on an **8.0 B-token**
Chinchilla-optimal corpus at **131,072 tokens per step**.

Stability features: **3000-step warmup**, cosine decay to **5% of peak LR**,
**gradient clipping at 1.0**, **NaN guard with rollback** after 5 consecutive
non-finite losses, **BF16 autocast**, **chunked cross-entropy** (chunk_size
8192), and **auxiliary MoE load-balancing loss** (α = 0.01).

Performance features: **`torch.compile(max-autotune)`**, **TF32**, **cuDNN
benchmark_limit=0**, **cuBLASLt** preferred BLAS, **AdamW foreach+fused**,
**gradient checkpointing every 3 layers**.

---

## Training Objective

Per optimizer step (after gradient accumulation):

```
L = L_CE + α · L_aux
```

| Term | Source | Description |
|---|---|---|
| `L_CE` | Chunked cross-entropy on LM head | Next-token prediction |
| `L_aux` | Mean aux loss across 12 MoE layers | Load-balancing (Switch/GShard) |
| `α` | `aux_loss_alpha` = **0.01** | Switch Transformer default |

The model returns `(logits, aux_loss)` from [`GPTOSS.forward`](../models/transformer.py).
`aux_loss` is already averaged across layers.

During micro-batches, the **scalar backward target** is:

```python
loss = (ce + aux_alpha * aux_loss) / accum
```

Division by `accum` keeps gradient magnitude equivalent to one large batch.

---

## Chinchilla Budget

From [`configs/pretrain_a100_502m.yaml`](../configs/pretrain_a100_502m.yaml):

| Knob | Value |
|---|---|
| `micro_batch_size` | 8 |
| `gradient_accumulation_steps` | 4 |
| `max_seq_len` (model) | 4096 |
| **Tokens per optimizer step** | **8 × 4 × 4096 = 131,072** |
| `total_steps` | 61,000 |
| **Total training tokens** | **61,000 × 131,072 ≈ 8.0 × 10⁹** |

This matches the **8.0 B-token** corpus prepared by the data pipeline
(Chinchilla-optimal for ~500 M-param models).

---

## Configuration Walkthrough

The A100 recipe lives in
[`configs/pretrain_a100_502m.yaml`](../configs/pretrain_a100_502m.yaml). A smoke
config [`pretrain_gpu_smoke.yaml`](../configs/pretrain_gpu_smoke.yaml) exercises
the full stack on 4 GB GPUs.

| Block | Headline knobs |
|---|---|
| `model` | 12 layers, GQA 8/4, `max_seq_len: 4096`, YaRN 128K target |
| `training` | `micro_batch_size: 8`, `accum: 4`, 61K steps, `nan_guard: true` |
| `data` | `data/pretrain_chinchilla`, 8B tokens, Llama-3 tokenizer |

Every `model`, `training`, and `data` key is documented in
[Part B — Configuration reference](#part-b--configuration-reference).
See [data_pipeline.md](data_pipeline.md) for corpus details.

---

## Entry Point and CLI

```bash
python training/pretrain.py \
  --config configs/pretrain_a100_502m.yaml \
  --seed 42 \
  --resume-from 4000
```

| Flag | Purpose |
|---|---|
| `--config` | YAML path (required) |
| `--seed` | Seed all RNGs + set `CUBLAS_WORKSPACE_CONFIG` |
| `--max-steps` | Override `total_steps` (debug runs) |
| `--resume-from` | Load checkpoint at step N |

`training/pretrain.py:main` flow:

1. Optional `training/pretrain.py:seed_everything`
2. `_set_hardware_perf_knobs()`
3. Load YAML → `ModelConfig`, training dict, data dict
4. Build model, optimizer, scheduler, DataLoader
5. Optional resume
6. Training loop until `total_steps`
7. Final checkpoint + RNG save

---

## Reproducibility — `seed_everything`

```python
def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if os.environ.get("CUBLAS_WORKSPACE_CONFIG") is None:
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
```

### Rules

| Condition | Reproducible? |
|---|---|
| `--seed` passed | Yes (with same hardware/software) |
| No `--seed` | **Not** reproducible |
| Checkpoint resume + RNG file | Restores exact continuation |

`CUBLAS_WORKSPACE_CONFIG=:4096:8` selects deterministic cuBLAS workspace
when seed is set. Without `--seed`, the env var is still set if unset, but
RNGs are not seeded.

Checkpoints include RNG state in `rng_step_N.pt` (see
[RNG State Persistence](#rng-state-persistence)).

---

## Hardware Performance Knobs

`_set_hardware_perf_knobs()` in [`pretrain.py`](../training/pretrain.py):

```python
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.benchmark_limit = 0      # exhaustive algo search
torch.backends.cuda.preferred_blas_library = "cublaslt"
torch.set_float32_matmul_precision("high")
```

| Knob | Effect |
|---|---|
| TF32 matmul | ~8× faster FP32 accum on Ampere+ |
| cuDNN benchmark | Autotune convolutions (minimal here) |
| `benchmark_limit=0` | Full cuDNN search — one-time cost, ~3–5% gain on A100 |
| `cublaslt` | BF16 matmul via cuBLASLt hand-tuned sm_80 kernels |
| `high` matmul precision | Allows TF32 internal accum |

These apply globally before the first forward pass.

---

## Model Construction

```python
dev = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = GPTOSS(model_cfg).to(dev)
```

[`GPTOSS`](../models/transformer.py) prints parameter counts:

```
[model] total params: 502.xxM, active: 247.xxM
```

`ModelConfig` is built from YAML `model:` block via dataclass constructor.
Invalid configs fail in `ModelConfig.__post_init__` (GQA divisibility, MoE
counts, YaRN lengths).

AMP dtype from config:

```python
_amp_dtype = {"bf16": torch.bfloat16, "fp16": torch.float16}.get(model_cfg.dtype, torch.bfloat16)
```

Default and recommended: **BF16** — no `GradScaler` needed on Ampere/Blackwell.

---

## `torch.compile`

```python
compile_enabled = train_cfg.get("compile", False) and dev.type == "cuda"
compile_mode = train_cfg.get("compile_mode", "max-autotune")
if compile_enabled:
    model = torch.compile(model, mode=compile_mode, fullgraph=False)
```

| Setting | Value |
|---|---|
| Enabled when | `compile: true` in YAML **and** CUDA available |
| Mode | `"max-autotune"` |
| `fullgraph` | `False` — allows graph breaks |

First steps incur compile/autotune latency. Failure prints warning and
continues without compile.

---

## Memory Estimation

Before training:

```python
est = estimate_model_memory_gb(model, seq_len=..., batch_size=micro_bs, grad_checkpoint=True)
assert_fits_in_available_gpu(est)
```

From [`utils/memory.py`](../utils/memory.py). OOM risk prints WARNING but does
not abort (allows smoke tests on small GPUs).

Gradient checkpointing (`grad_checkpoint: true`, every 3 layers) trades ~30%
extra compute for ~40% activation memory savings at `T=4096`.

### Memory accounting derivation

Let $P = 501836640$ be the total parameter count (deduplicated for
weight tying). Serialized BF16 weights cost 2 bytes per parameter:

$$
W_{\mathrm{bf16}} = 2P = 1003673280\ \text{B} \approx 0.94\ \text{GB}, \tag{1}
$$

the size of a `model_step_N.safetensors` file (≈ 1.00 × 10⁹ B in decimal
units). In GPU memory the parameters live as FP32 master weights — autocast
casts to BF16 only at matmul boundaries ([optimizers theory](theory/optimizers.md) §4.6)
— and AdamW keeps two more FP32 tensors per parameter, the moments $m$ and
$v$. Weights + optimizer state + FP32 gradient buffers:

$$
M_{\mathrm{optim}} = (4 + 4 + 4)\,P = 12P = 6.02 \times 10^{9}\ \text{B} \approx 5.6\ \text{GiB}, \tag{2}
$$

$$
M_{\mathrm{fixed}} = (4 + 12)\,P = 16P \approx 7.5\ \text{GiB}. \tag{3}
$$

Equation (3) is exactly the `param_bytes + optim_bytes` sum in
`utils/memory.py:estimate_model_memory_gb`: `element_size() × P` for the FP32
model (4P) plus 12P for the FP32 masters + moments. MoE sparsity does not
reduce this floor: every expert occasionally receives gradients, so all
501.8M parameters carry state.

**Activations with gradient checkpointing.** The estimator scales saved
activations by a store factor `s(g) = 1/g + (1 − 1/g)·½` for checkpointing
every $g$-th block: at `grad_checkpoint_every=3`, one of every three blocks
saves no forward activations and the other two save about half, so
$s(3) = 2/3$ — a 33% saving (the ~40% rough figure above). The activation
block is the attention activations plus the MoE intermediates (3 live experts
per layer × 3 matrices each):

$$
M_{\mathrm{act}} = s(3)\,\big[\,12 \cdot T \cdot B \cdot d_{\mathrm{model}} + 12 \cdot 3 \cdot 3 \cdot T \cdot B \cdot f_{\mathrm{ffn}}\big] \cdot 2\ \text{B} \approx 7.1\ \text{GiB} \quad \text{at } T{=}4096,\ B{=}8. \tag{4}
$$

The KV cache during training sizes windowed layers at $\max(W, T) = 4096$
(`win_len = window if steady_state else max(window, seq_len)`), so no sliding
savings before steady-state inference: 6 windowed + 6 global layers × 4096
token-slots × 8 batch × 1536 B per slot ≈ 0.56 GiB.

**Total vs the microbench threshold.** The full estimate — eq. (3) + KV +
eq. (4) + ~2 GiB runtime overhead — is ≈ 17.2 GiB, comfortably under the
**25 GB ceiling** of `scripts/microbench_a100.py` (`--threshold-gb 25.0`,
chosen to leave ~55 GB of the 80 GB A100 for batch scaling). The 17.2 GiB
figure is *derived* from the estimator's formula; the measured
`torch.cuda.max_memory_allocated` peak on an A100 is **`[INFERENCE]`** —
`.benchmarks/` is empty. Confirm with
`python3 scripts/microbench_a100.py --config configs/pretrain_a100_502m.yaml --batch-size 8 --seq-len 4096`.

---

## Optimizer — AdamW with Parameter Groups

### Parameter groups

```python
no_decay = ["bias", "norm", "embed"]
decay_params = [p for n, p in model.named_parameters() if not any(nd in n.lower() for nd in no_decay)]
no_decay_params = [p for n, p in model.named_parameters() if any(nd in n.lower() for nd in no_decay)]
```

| Group | Weight decay |
|---|---|
| Matrices (attention, MoE, head if not tied) | `weight_decay: 0.1` |
| bias, norm, embed | **0.0** |

Embedding is excluded from decay because it is tied to the LM head weight.

### Parameter-group derivation

AdamW applies the decoupled update ([optimizers theory](theory/optimizers.md)
eq. 11) to every parameter of a group with that group's decay $\lambda$:

$$
\theta \leftarrow (1 - \eta\lambda)\,\theta - \eta\,\frac{\hat{m}_t}{\sqrt{\hat{v}_t} + \varepsilon}. \tag{5}
$$

The two groups are built by a substring filter over parameter names. Let
$K = \{\text{bias}, \text{norm}, \text{embed}\}$ and $n(p)$ the parameter
name; the partition is

$$
\text{decay} = \{\,p : n(p) \text{ contains no keyword of } K\,\}, \qquad
\text{no-decay} = \{\,p : n(p) \text{ contains a keyword of } K\,\}. \tag{6}
$$

Every named parameter satisfies exactly one predicate, so the groups are
exhaustive and disjoint — no tensor is decayed twice and none is missed. Two
details make the weight-tying exclusion sound:

- `model.named_parameters()` deduplicates by tensor identity
  (`remove_duplicate=True`, the default). Because
  `models/transformer.py:GPTOSS.__init__` assigns
  `self.head.weight = self.embed.weight`, the shared tensor is yielded once,
  under its first-registered name `embed.weight`, which the keyword `embed`
  catches — the `head.weight` alias never reaches the decay group, so the tied
  tensor has a single group entry.
- The split at production scale (verified): decay holds **403,513,344**
  parameters (372 tensors), no-decay holds **98,323,296** (38 tensors), and
  the two sum to 501,836,640 exactly. The no-decay total decomposes as
  $128000 \times 768 = 98304000$ (tied embedding) +
  $25 \times 768 = 19200$ (24 per-block RMSNorm gains + final norm) +
  $12 \times 8 = 96$ (per-head sink biases) = 98,323,296.

Why these three classes: the embedding is tied to the LM head, so its rows
serve both the input projection and the logit projection — decaying it would
directly shrink the scale the head weight applies to the final hidden state.
RMSNorm gains are reparameterization-redundant scales (the norm is
scale-invariant in its input), and biases are additive offsets with no
overfitting risk; decaying either is at best meaningless, at worst harmful to
the residual stream. This mirrors LLaMA/GPT-3 practice.

### AdamW settings

```python
optim = AdamW(
    [
        {"params": decay_params, "weight_decay": 0.1},
        {"params": no_decay_params, "weight_decay": 0.0},
    ],
    lr=4e-4,
    betas=(0.9, 0.95),
    eps=1e-6,
    foreach=True,
    fused=(dev.type == "cuda"),
)
```

| Hyperparameter | Value | Notes |
|---|---|---|
| `lr` | **4e-4** | Peak after warmup |
| `beta1` | 0.9 | |
| `beta2` | 0.95 | |
| `eps` | **1e-6** | BF16-safe (not 1e-8) |
| `foreach` | True | Faster multi-tensor update |
| `fused` | True on CUDA | ~1.5–2× vs default loop on A100 |

**Why eps=1e-6?** BF16 has 7 mantissa bits; `1e-8` underflows in Adam's
second moment, silently stalling late training. Matches DeepSeek-V3 and
LLaMA-3 recipes.

---

## Learning Rate Schedule

`training/pretrain.py:make_warmup_cosine_lambda(warmup_steps=3000, total_steps=61000, min_lr_ratio=0.05)`:

```
step < warmup:     lr_mult = step / warmup_steps
step >= total:     lr_mult = min_lr_ratio (0.05)
else:              cosine from 1.0 → 0.05 over (total - warmup) steps
```

### Schedule parameters

| Phase | Steps | LR multiplier |
|---|---|---|
| Warmup | 0 → 3000 | 0 → 1.0 linear |
| Cosine decay | 3000 → 61000 | 1.0 → 0.05 |
| After 61000 | — | 0.05 (floor) |

Peak LR = **4e-4**; floor LR = **2e-5** (5% of peak).

Warmup **3000 steps** ≈ 4.9% of total — industry MoE standard 2–5% for
top-k routing stability.

### Closed form and LR budget

`training/pretrain.py:make_warmup_cosine_lambda` defines the LR multiplier
$\lambda(s)$ as an exact piecewise function of the optimizer step $s$, with
warmup $W = 3000$, total steps $T = 61000$, and floor ratio
$\lambda_{\min} = 0.05$:

$$
\lambda(s) = \begin{cases}
\dfrac{s}{W} & 0 \le s < W, \\[10pt]
\lambda_{\min} + (1 - \lambda_{\min})\,\dfrac{1}{2}\Big(1 + \cos \pi \dfrac{s - W}{T - W}\Big) & W \le s < T, \\[10pt]
\lambda_{\min} & s \ge T.
\end{cases} \tag{7}
$$

The warmup branch is the linear ramp $s/W$; the cosine branch interpolates
$\lambda$ from 1.0 down to $\lambda_{\min}$ with **zero slope at both
endpoints** (the cosine's derivative is $\propto \sin$, which vanishes at
$s = W$ and $s = T$), avoiding the kinks of a linear decay. The third branch
clamps the floor: the schedule never reaches zero, so the final steps refine
rather than add noise.

**Effective LR area.** The total LR budget of the run is the integral of
$\lambda$ over the schedule. Warmup contributes a triangle of area $W/2$; the
cosine half-wave integrates to exactly half its rectangle (a full period of
$\cos$ integrates to zero), so the decay contributes
$(T - W)(1 + \lambda_{\min})/2$:

$$
A = \int_0^W \frac{s}{W}\,ds + \int_W^T \Big[\lambda_{\min}
+ (1 - \lambda_{\min})\,\tfrac{1}{2}\bigl(1 + \cos \pi \tfrac{s-W}{T-W}\bigr)\Big]\,ds
= \frac{W}{2} + \frac{T - W}{2}\,(1 + \lambda_{\min}). \tag{8}
$$

With the production numbers:

$$
A = \frac{3000}{2} + \frac{58000}{2}\,(1.05) = 1500 + 30450 = 31950, \qquad
\bar\lambda = \frac{A}{T} = \frac{31950}{61000} \approx 0.52. \tag{9}
$$

The run's total LR budget equals **31,950 optimizer steps at constant peak
LR** — about 52% of the nominal 61,000 steps. Two consequences: (i)
$\bar\lambda \approx 0.52$ is the mean multiplier, the right comparator
against a constant-LR baseline of equal step count; (ii) warmup contributes
only $1500 / 31950 \approx 4.7\%$ of the area despite occupying 4.9% of the
steps — the triangle shape halves its budget.

**Why 3000 warmup steps for MoE.** At initialization the router logits are
Gaussian-scaled by `models/transformer.py:ModelConfig.init_std` = 0.02, so the
routing softmax is nearly uniform and top-2 selection is effectively random;
router gradients are correspondingly noisy, and the aux loss (α = 0.01,
[moe.md](moe.md)) injects a second, weaker signal that can collapse the gate
onto one expert before the CE signal stabilizes. Warmup also neutralizes the
worst Adam pathology: at $t = 1$ the bias-corrected step is
$\hat{m}_1 / \sqrt{\hat{v}_1} = \operatorname{sign}(g_1)$ — every coordinate
moves by exactly $\pm\eta$ ([optimizers theory](theory/optimizers.md) §4.3) —
so an immediate peak LR would apply full-magnitude sign-noise steps to every
weight. Ramping $\eta$ from 0 damps both effects. The 3000-step ramp spans
$3000 \times 131072 \approx 0.39$B tokens, ~4.9% of the 8B-token corpus;
$W/T = 4.9\%$ sits inside the 2–5% band the config comment calls the MoE
industry standard (2000 steps, 3.3%, "was on the low end").

```python
sched = LambdaLR(optim, lr_lambda)
# sched.step() called once per optimizer step (after accum boundary)
```

---

## Data Loading

```python
ds = PretrainDataset(data_cfg["train_data_path"], model_cfg.max_seq_len)
loader = DataLoader(
    ds,
    batch_size=micro_bs,          # 8
    shuffle=True,
    num_workers=4,                # default from YAML or 4
    pin_memory=True,              # on CUDA
    persistent_workers=True,      # when num_workers > 0
    drop_last=True,
)
```

See [data_pipeline.md](data_pipeline.md) for `training/pretrain.py:PretrainDataset` internals.

Default path: `data/pretrain_chinchilla` — directory of `shard_*.bin` + optional
`manifest.json`.

---

## The Training Step — Inner Loop

```python
while step < total_steps:
    for input_ids, target_ids in loader:
        input_ids = input_ids.to(dev, non_blocking=True)
        target_ids = target_ids.to(dev, non_blocking=True)

        with autocast(device_type=dev.type, dtype=_amp_dtype, enabled=cuda):
            logits, aux_loss = model(input_ids)
            ce = chunked_cross_entropy(logits, target_ids, chunk_size=8192)
            loss = (ce + aux_alpha * aux_loss) / accum

        if not torch.isfinite(loss):
            # NaN guard branch ...
            continue

        loss.backward()
        micro_step += 1

        if micro_step % accum == 0:
            clip_grad_norm_(..., 1.0)
            optim.step()
            sched.step()
            optim.zero_grad(set_to_none=True)
            step += 1
            # log / save checkpoints
```

### Step counter semantics

| Counter | Increments when |
|---|---|
| `micro_step` | Every forward-backward |
| `step` | Every `accum` micro-batches (optimizer step) |
| `pbar` | Tracks optimizer `step` |

---

## Mixed Precision — BF16 Autocast

```python
with autocast(device_type=dev.type, dtype=torch.bfloat16, enabled=(dev.type == "cuda")):
    ...
```

- **Forward** runs in BF16 for matmuls/convolutions where beneficial.
- **Loss** computed inside autocast region (CE in FP32 internally via PyTorch).
- **Master weights:** AdamW updates FP32 master copies when using fused AdamW
  on CUDA (standard PyTorch behaviour).
- **No GradScaler** — BF16 exponent range matches FP32.

RMSNorm in the model keeps activations in native dtype (no FP32 copy in norm).

---

## Chunked Cross-Entropy

`training/pretrain.py:chunked_cross_entropy` avoids materialising the full
`(B·T, vocab)` softmax by summing chunk-local losses:

```python
def chunked_cross_entropy(logits, targets, chunk_size=4096):
    flat_logits = logits.view(-1, vocab_size)
    flat_targets = targets.view(-1)
    total_loss = 0
    for start in range(0, n_total, chunk_size):
        chunk_loss = F.cross_entropy(flat_logits[start:end], flat_targets[start:end], reduction="sum")
        total_loss += chunk_loss
    return total_loss / n_total
```

### Production chunk size

Pretrain uses **`chunk_size=8192`** (not the function default 4096):

```python
ce = chunked_cross_entropy(logits, target_ids, chunk_size=8192)
```

At `B=8`, `T=4096`: `n_total = 32,768` tokens → 4 CE chunks (was 8 at 4096).

**Memory:** avoids materialising full `(B×T, vocab)` softmax — peak intermediate
~`chunk_size × vocab_size` instead of `B×T × vocab_size`.

With `vocab=128000`, chunk 8192 ≈ 1 GB BF16 logits slice — well under 80 GB
with model + activations.

### Equivalence proof

Let $N = B \cdot T = 32768$ be the number of tokens in the batch and
$V = 128000$ the vocabulary. The per-token cross-entropy for logits
$z_t \in \mathbb{R}^{V}$ and target token $y_t$ is

$$
\ell_t = -\log \operatorname{softmax}(z_t)_{y_t} = -z_{t,y_t} + \log \sum_{v=1}^{V} e^{z_{t,v}}, \tag{10}
$$

and the full-batch loss is the mean $\mathcal{L} = \tfrac{1}{N}\sum_{t=1}^{N}\ell_t$.
The softmax normalizes over the **vocabulary axis only** — each token's
probability is computed independently of the other $N-1$ tokens. That is what
makes chunking free: cross-entropy is additive over tokens, so for any
partition of $\{1,\dots,N\}$ into chunks $S_1,\dots,S_C$,

$$
\frac{1}{N}\sum_{c=1}^{C}\sum_{t \in S_c} \ell_t
= \frac{1}{N}\sum_{t=1}^{N} \ell_t = \mathcal{L}. \tag{11}
$$

`training/pretrain.py:chunked_cross_entropy` computes exactly the left-hand
side: each `F.cross_entropy(..., reduction="sum")` yields
$\sum_{t \in S_c}\ell_t$, the chunk sums accumulate, and the final
`total_loss / n_total` divides by $N$. The only cross-token operation in the
entire loss is that final mean, and summing chunk-local sums before dividing
recovers it identically. If the loss had any normalization over the *token*
axis, chunking would change it — it does not. Chunked and full CE are equal in
real arithmetic; in floating point they differ only by summation order and the
dtype of the accumulator (`total_loss` lives in the logits' dtype, BF16 under
autocast), a rounding effect that is negligible for losses $O(1)$ over 4
chunks ([numerics](theory/numerics.md)).

**Why the memory drops.** A single `F.cross_entropy` over all $N$ tokens
materializes both the logits and the log-softmax output, each $N \times V \times
2$ bytes; chunking bounds the live slice to $C = \min(\text{chunk\_size}, N)$
tokens:

$$
M_{\mathrm{full}} \approx 2 \cdot N \cdot V \cdot 2\ \text{B}
= 2 \cdot 32768 \cdot 128000 \cdot 2 \approx 16.8\ \text{GB}, \qquad
M_{\mathrm{chunk}} \approx 2 \cdot C \cdot V \cdot 2\ \text{B}. \tag{12}
$$

At `chunk_size=8192`, $C = 8192$ gives $M_{\mathrm{chunk}} \approx 4.2$ GB — a
$N/C = 4\times$ reduction. (Each live slice alone is
$8192 \times 128000 \times 2 = 2.1 \times 10^9$ B ≈ 2.1 GB; the "≈ 1 GB"
figure above underestimates the slice, though the conclusion — well under the
80 GB budget — is unchanged.)

---

## Auxiliary MoE Loss Integration

```python
aux_alpha = train_cfg.get("aux_loss_alpha", 0.01)
loss = (ce + aux_alpha * aux_loss) / accum
```

Logging every `log_interval`:

```python
logger.log(step, ce_val, metrics={"aux": aux_val}, lr=lr)
pbar.set_postfix(ce=f"{ce.item():.4f}", aux=f"{aux_loss.item():.4f}")
```

Healthy training: `aux` starts ~1–4, decreases toward ~0.5–1.5 as routing
balances. If `aux → 0` while one expert dominates, check router gradients.

**Distinct from DeepSeek-v3-Lite:** GPT-OSS uses standard aux loss, not
aux-loss-free bias updates ([moe.md](moe.md)).

---

## Gradient Accumulation

```yaml
gradient_accumulation_steps: 4
micro_batch_size: 8
```

Effective batch = **32 sequences × 4096 tokens = 131,072 tokens/step**.

```python
optim.zero_grad(set_to_none=True)   # once before loop
# ...
loss = (ce + aux_alpha * aux_loss) / accum
loss.backward()
if micro_step % accum == 0:
    optim.step()
    optim.zero_grad(set_to_none=True)
```

`set_to_none=True` frees gradient tensors instead of zeroing — lower peak
memory.

### Equivalence derivation

Gradient accumulation with $k = 4$ micro-batches of $m$ sequences each is
exact, not an approximation. Let the full-batch objective over all
$N = k \cdot m$ samples be

$$
\mathcal{L}(\theta) = \frac{1}{km}\sum_{i=1}^{k}\sum_{j=1}^{m} \ell_{i,j}(\theta), \qquad
\nabla\mathcal{L} = \frac{1}{km}\sum_{i,j} \nabla\ell_{i,j}, \tag{13}
$$

and let $\mathcal{L}_i$ be micro-batch $i$'s loss, a mean over its $m$
samples. The sum of micro-batch gradients, each scaled by $1/k$, is

$$
\frac{1}{k}\sum_{i=1}^{k} \nabla\mathcal{L}_i
= \frac{1}{k}\sum_{i=1}^{k} \frac{1}{m}\sum_{j=1}^{m} \nabla\ell_{i,j}
= \frac{1}{km}\sum_{i,j} \nabla\ell_{i,j}
= \nabla\mathcal{L}, \tag{14}
$$

because gradients accumulate linearly — each sample appears in exactly one
micro-batch, and averaging commutes with the gradient. PyTorch's
`loss.backward()` implements the left-hand side: it adds each scaled
micro-batch gradient into `.grad`, so after $k$ micro-batches `.grad` holds
$\nabla\mathcal{L}$ exactly.

The combined objective follows the same rule term by term. With
`loss = (ce + aux_alpha * aux_loss) / accum` in `training/pretrain.py:main`:

$$
\sum_{i=1}^{k} \nabla\Big(\frac{\text{ce}_i + \alpha\,\text{aux}_i}{k}\Big)
= \frac{1}{k}\sum_{i=1}^{k} \nabla\text{ce}_i
+ \frac{\alpha}{k}\sum_{i=1}^{k} \nabla\text{aux}_i
= \nabla\mathcal{L}_{\mathrm{CE}} + \alpha\,\nabla\mathcal{L}_{\mathrm{aux}}, \tag{15}
$$

so both the CE and the aux term inherit the same $1/k$ scale and the relative
weight $\alpha = 0.01$ is preserved. The division happens *before*
`backward()`, scaling the gradients as they are accumulated. Why not divide
afterward? Adam's update ratio $\hat{m}/\sqrt{\hat{v}}$ is scale-invariant
([optimizers theory](theory/optimizers.md) eq. 9), but `clip_grad_norm_` at
1.0 is not — an un-scaled accumulated gradient would have a $k \times$ larger
norm and would trip the clip far more often — and the decoupled weight decay
is also un-scaled, so the decay/update balance would shift with $k$. Dividing
keeps behavior identical for any $k$: `accum=1` and `accum=4` produce the same
update scale, which is why the optimizer step and the LR scheduler tick only
at the `micro_step % accum == 0` boundary, and
`optim.zero_grad(set_to_none=True)` then releases the accumulated buffers.

---

## Gradient Clipping

```python
grad_clip = 1.0
nn.utils.clip_grad_norm_(model.parameters(), grad_clip, foreach=True)
```

Applied **only at optimizer step boundary** (after `accum` micro-batches).

`foreach=True` for faster multi-tensor norm on CUDA; falls back if unsupported.

If `grad_clip <= 0`, clipping is skipped.

---

## Gradient Checkpointing

```python
model.enable_gradient_checkpointing(every=train_cfg.get("grad_checkpoint_every", 3))
```

[`GPTOSS.forward`](../models/transformer.py):

```python
if use_grad_ckpt and (layer_idx % grad_ckpt_every == 0):
    x, aux = torch.utils.checkpoint.checkpoint(block, x, positions, use_reentrant=False)
```

| Setting | Value |
|---|---|
| `grad_checkpoint` | `true` |
| `grad_checkpoint_every` | **3** |

Every 3rd block (layers 0, 3, 6, 9) recomputes forward on backward. Layers
1, 2, 4, 5, 7, 8, 10, 11 store activations normally.

---

## NaN Guard and Checkpoint Rollback

```yaml
nan_guard: true
nan_guard_max_consecutive: 5
```

```python
if not torch.isfinite(loss):
    nan_count += 1
    optim.zero_grad(set_to_none=True)
    micro_step = 0
    if nan_count >= nan_max_consec:
        latest = ckpt.latest_step()
        ckpt.load(model, step=latest, optimizer=optim, scheduler=sched)
        step = latest
        nan_count = 0
    continue
```

| Event | Action |
|---|---|
| Single non-finite loss | Skip batch, zero grad, reset micro_step |
| 5 consecutive non-finite | Roll back to **latest complete checkpoint** |
| No checkpoint available | `RuntimeError` |
| `nan_guard: false` | Immediate `RuntimeError` on non-finite |

**Never disable** NaN guard in production without explicit approval
([`AGENTS.md`](../AGENTS.md) rule 6).

---

## Logging

[`utils/logging.py:TrainingLogger`](../utils/logging.py) with:

```yaml
log_interval: 50
```

`utils/logging.py:TrainingLogger.log` emits CE loss, aux metric, LR every 50
optimizer steps; `utils/logging.py:TrainingLogger.finish` closes the optional
WandB run. Seq len passed for tokens/sec estimation.

---

## Checkpointing — `CheckpointManager`

[`utils/checkpoint.py:CheckpointManager`](../utils/checkpoint.py)

### Save (every `save_interval=2000` + final)

```python
ckpt.save(model, optim, step, scheduler=sched, extra_meta={...})
```

`utils/checkpoint.py:CheckpointManager.save` writes atomically;
`utils/checkpoint.py:CheckpointManager.load` restores weights and returns meta.

| File | Format | Contents |
|---|---|---|
| `model_step_N.safetensors` | safetensors | Model weights (deduped shared tensors) |
| `optim_step_N.pt` | torch.save | AdamW state |
| `sched_step_N.pt` | torch.save | LR scheduler state |
| `meta_step_N.json` | JSON | Step, optional aux_loss, final flag |

### Atomic write pattern

```
write to .tmp in save_dir → os.replace to final path
```

Safetensors save **clones** duplicate `data_ptr` tensors (weight tying) to
avoid safetensors duplicate-key errors.

### Completeness check

`latest_step()` returns highest step where model + optim + meta all exist.

Step discovery and retention both run through the same completeness filter.
`utils/checkpoint.py:CheckpointManager.list_checkpoints` returns every complete
step and `utils/checkpoint.py:CheckpointManager.latest_step` returns the newest
of them; the NaN guard resolves its rollback point through `latest_step`
rather than trusting a running step counter, and `list_checkpoints` shows which
steps exist before an interactive `--resume-from`. Retention is explicit:
`utils/checkpoint.py:CheckpointManager.keep_last_n` keeps the newest `n`
complete checkpoints and `utils/checkpoint.py:CheckpointManager.delete_checkpoint`
removes all four files (`model`/`optim`/`sched`/`meta`) of one step, so a
partial or obsolete step can be cleaned without disturbing its neighbours.

---

## Resume Training

```bash
python training/pretrain.py \
  --config configs/pretrain_a100_502m.yaml \
  --resume-from 4000 \
  --seed 42
```

```python
meta = ckpt.load(model, step=resume_from, optimizer=optim, scheduler=sched)
start_step = meta["step"]
```

RNG restoration from `rng_step_{resume_from}.pt` if present:

```python
random.setstate(rng_state["python"])
np.random.set_state(rng_state["numpy"])
torch.set_rng_state(rng_state["torch"])
torch.cuda.set_rng_state_all(rng_state["cuda"])
```

Resume **without** `--seed` still loads weights/optimizer; RNG only restored
if rng file exists.

---

## RNG State Persistence

At end of training:

```python
rng_state = {
    "python": random.getstate(),
    "numpy": np.random.get_state(),
    "torch": torch.get_rng_state(),
    "cuda": torch.cuda.get_rng_state_all() if cuda else None,
}
torch.save(rng_state, ckpt.save_dir / f"rng_step_{step}.pt")
```

Saved alongside final checkpoint. Enables bit-exact continuation of data
order (with same DataLoader worker config).

---

## End-to-End Timeline

| Step range | Phase |
|---|---|
| 0 – 3000 | Linear warmup, routing stabilising |
| 3000 – 20000 | Main learning, CE rapid drop |
| 20000 – 50000 | Cosine decay, aux loss settling |
| 50000 – 61000 | Low LR refinement |
| Every 2000 | Checkpoint |
| 61000 | Final save + RNG |

**Wall clock (A100 80GB):** target ~16–20 h at 35–40% MFU (config header estimate).

---

## Operational Commands

```bash
# Full A100 run
python training/pretrain.py \
  --config configs/pretrain_a100_502m.yaml \
  --seed 42

# Short smoke (override steps in code path)
python training/pretrain.py \
  --config configs/pretrain_gpu_smoke.yaml \
  --max-steps 10 \
  --seed 0

# Resume
python training/pretrain.py \
  --config configs/pretrain_a100_502m.yaml \
  --resume-from 2000 \
  --seed 42
```

Ensure data exists:

```bash
python data/prepare_data.py --stage pretrain
```

---

## Debugging Checklist

| Symptom | Check |
|---|---|
| OOM at T=4096 | `grad_checkpoint: true`; reduce `micro_batch_size` |
| Loss NaN early | NaN guard logs; sink bias clamp in attention |
| `aux` stuck high | MoE routing collapse — see [moe.md](moe.md) |
| Slow step 0 | `torch.compile` + cuDNN autotune — normal |
| No reproducibility | Pass `--seed`; set `CUBLAS_WORKSPACE_CONFIG` |
| CE dominates, aux tiny | Expected — α=0.01 scales aux down |
| Checkpoint won't load | `list_checkpoints()` — need complete triple |

---

## Appendix A — Step arithmetic

```
tokens_per_micro = micro_bs × max_seq_len = 8 × 4096 = 32,768
tokens_per_step  = tokens_per_micro × accum = 32,768 × 4 = 131,072
total_tokens     = 61,000 × 131,072 = 7,995,392,000 ≈ 8.0B
```

---

## Appendix B — LR schedule samples

| Step | LR multiplier | LR (peak 4e-4) |
|---|---|---|
| 0 | 0.0 | 0 |
| 1500 | 0.5 | 2e-4 |
| 3000 | 1.0 | 4e-4 |
| 20000 | ~0.85 | ~3.4e-4 |
| 40000 | ~0.45 | ~1.8e-4 |
| 61000 | 0.05 | 2e-5 |

---

## Appendix C — File map

| File | Role |
|---|---|
| [`training/pretrain.py`](../training/pretrain.py) | Main loop |
| [`configs/pretrain_a100_502m.yaml`](../configs/pretrain_a100_502m.yaml) | A100 recipe |
| [`models/transformer.py`](../models/transformer.py) | `GPTOSS`, checkpointing |
| [`utils/checkpoint.py`](../utils/checkpoint.py) | Safetensors I/O |
| [`utils/logging.py`](../utils/logging.py) | Training logger |
| [`utils/memory.py`](../utils/memory.py) | VRAM estimator |

---

## Part B — Configuration reference

> **YAML encyclopedia** for `configs/pretrain_a100_502m.yaml` and
> `configs/pretrain_gpu_smoke.yaml`. Every `model`, `training`, and `data` key
> is documented with defaults, valid ranges, and interaction effects. For how
> configs connect to code, see [architecture.md](architecture.md).

### B.1 How configs are loaded (`yaml.safe_load` → `ModelConfig` / dicts)

`training/pretrain.py` reads YAML with PyYAML:

```python
with open(config_path) as f:
    cfg = yaml.safe_load(f)
model_cfg = ModelConfig(**cfg["model"])
train_cfg = cfg["training"]
data_cfg = cfg["data"]
```

- **`model`** keys map to `ModelConfig` dataclass fields in
  `models/transformer.py`. Unknown keys raise `TypeError`.
- **`training`** and **`data`** are plain dicts — optional keys fall back to
  defaults inside `pretrain.py`.

CLI overrides:

| Flag | Effect |
|------|--------|
| `--config PATH` | Required YAML path |
| `--seed N` | Seeds all RNGs before model build |
| `--max-steps N` | Overrides `training.total_steps` |
| `--resume-from N` | Loads checkpoint at step N |

### B.2 File comparison: `pretrain_a100_502m.yaml` vs `pretrain_gpu_smoke.yaml`

| Aspect | `pretrain_a100_502m.yaml` | `pretrain_gpu_smoke.yaml` |
|--------|---------------------------|---------------------------|
| Purpose | Chinchilla 8B-token pretrain | E2E on 4 GB GPU |
| `d_model` | 768 | 128 |
| `n_layers` | 12 | 4 |
| `vocab_size` | 128000 | 4096 |
| `max_seq_len` | 4096 | 64 |
| `eval_max_seq_len` | 131072 | 256 |
| `window_size` | 128 | 32 |
| `total_steps` | 61000 | 5 |
| `compile` | true | false |
| `moe_dispatch` | omitted (stacked) | `"stacked"` explicit |
| `train_data_path` | `data/pretrain_chinchilla` | `data/pretrain_smoke` |
| Checkpoints | `checkpoints/pretrain_a100` | `checkpoints/gpu_smoke` |

Smoke config preserves **structural** invariants (alternation, sink, YaRN, MoE
top-k) at miniature scale — not Chinchilla token counts.

#### Smoke config rationale

`pretrain_gpu_smoke.yaml` exists to answer: "Does the full stack run on my GPU
in seconds?"

| Field | Why |
|-------|-----|
| `n_layers: 4` | 2 SWA + 2 global — alternation preserved |
| `max_seq_len: 64` | Fits tiny VRAM |
| `yarn_target: 256` | Tests extrapolation > train len |
| `total_steps: 5` | Quick loss descent sanity |
| `compile: false` | Avoid compile latency in CI |
| `weight_decay: 0` | Simpler overfitting check on noise |
| `moe_dispatch: stacked` | No Triton dependency |

Pair with `scripts/e2e_gpu_smoke.py` for richer checks than 5 training steps.

### B.3 Derived quantities (tokens/step, Chinchilla steps, active params)

#### A100 production (`pretrain_a100_502m.yaml`)

**Tokens per micro-batch step (one forward):**

```
tokens_micro = micro_batch_size × max_seq_len
             = 8 × 4096 = 32,768
```

**Tokens per optimizer step (after gradient accumulation):**

```
tokens_step = tokens_micro × gradient_accumulation_steps
            = 32,768 × 4 = 131,072
```

**Total training tokens:**

```
total_tokens = total_steps × tokens_step
             = 61,000 × 131,072 = 7,995,392,000 ≈ 8.0 × 10⁹
```

**Chinchilla check:** ~502M total params × ~16 tokens/param ≈ 8B tokens ✓

**Parameter counts (from `GPTOSS` methods):**

| Method | Approx. value |
|--------|---------------|
| `num_parameters()` | ~502M |
| `num_active_parameters()` | ~247M |
| Sparsity | ~50.8% |

**Warmup fraction:**

```
3000 / 61000 ≈ 4.9%
```

Industry MoE recipes often use 2–5% warmup; 3000 steps was chosen for router
stability with top-2-of-8 routing.

**Minimum learning rate:**

```
lr_min = lr × min_lr_ratio = 4e-4 × 0.05 = 2e-5
```

**Checkpoint count (full run):**

```
floor(61000 / 2000) = 30 interval saves + 1 final
```

**Wall-time estimate (A100 80GB):**

At ~35–40% MFU and ~131K tokens/step, expect **16–20 hours** for 61K steps
(hardware-dependent).

#### Smoke config

```
tokens_step = 2 × 1 × 64 = 128 tokens/step
total_tokens = 5 × 128 = 640 tokens
```

Not Chinchilla-optimal — sufficient to exercise the pipeline.

### B.4 `model` block — every field

Master table — **A100** = `pretrain_a100_502m.yaml`, **Smoke** = `pretrain_gpu_smoke.yaml`.
All keys map to `ModelConfig` in `models/transformer.py`; invalid combinations
raise in `__post_init__`.

#### Core dimensions and GQA

| Key | A100 | Smoke | Notes |
|-----|------|-------|-------|
| `vocab_size` | `128000` | `4096` | Must match tokenizer ([data_pipeline.md](data_pipeline.md)) |
| `d_model` | `768` | `128` | `n_heads × head_dim` |
| `n_layers` | `12` | `4` | Even count → balanced SWA/global split |
| `n_heads` | `8` | `4` | Query heads |
| `n_kv_heads` | `4` | `2` | KV heads; `n_heads % n_kv_heads == 0` |
| `head_dim` | `96` | `32` | Even int; RoPE pair dimension |

#### MoE

| Key | A100 | Smoke | Notes |
|-----|------|-------|-------|
| `ffn_dim` | `1536` | `256` | SwiGLU inner dim per expert |
| `n_routed_experts` | `8` | `4` | Router: `(d_model, n_routed)` |
| `n_activated_experts` | `2` | `2` | Top-k per token |
| `n_shared_experts` | `1` | `1` | Always-on expert(s) |
| `moe_dispatch` | *(default `stacked`)* | `stacked` | `"triton_grouped"` opt-in — [moe.md](moe.md#sanctioned-triton-path-moe_dispatchtriton_grouped) |

#### Attention, YaRN, sequence limits

| Key | A100 | Smoke | Notes |
|-----|------|-------|-------|
| `window_size` | `128` | `32` | SWA span on even layers |
| `sink_bias` | `true` | `true` | Per-head sink — [ATTENTION_SINKS.md](ATTENTION_SINKS.md) |
| `rope_theta` | `100000` | `10000` | RoPE base frequency |
| `yarn_scale_factor` | `32` | `4` | `1` = plain RoPE |
| `yarn_original_max_seq_len` | `4096` | `64` | Training RoPE anchor |
| `yarn_target_seq_len` | `131072` | `256` | Extrapolation target (128K) |
| `yarn_beta_fast` | `32` | `4` | YaRN ramp — [rope_yarn.md](rope_yarn.md) |
| `yarn_beta_slow` | `1` | `1` | YaRN ramp slow boundary |
| `yarn_mscale` | `true` | `true` | Magnitude scaling during extrapolation |
| `yarn_prune_rope_global` | `true` | `true` | 25% dim freeze on global layers — [rope_yarn.md](rope_yarn.md) |
| `max_seq_len` | `4096` | `64` | Training window size |
| `eval_max_seq_len` | `131072` | `256` | Inference / passkey cap |

#### Dtype and initialization

| Key | A100 | Smoke | Notes |
|-----|------|-------|-------|
| `dtype` | `bf16` | `bf16` | BF16 autocast on CUDA — no GradScaler |
| `weight_tying` | `true` | `true` | Embed ↔ head; saves ~98M at production scale |
| `rms_norm_eps` | `1.0e-5` | `1.0e-5` | RMSNorm ε |
| `init_std` | `0.02` | `0.02` | Linear/embed init; sink bias zero-init separately |

### B.5 `training` block — every field

| Key | A100 | Smoke | Notes |
|-----|------|-------|-------|
| `micro_batch_size` | `8` | `2` | Sequences per forward |
| `gradient_accumulation_steps` | `4` | `1` | Micro-batches per optimizer step |
| `total_steps` | `61000` | `5` | Override with `--max-steps` |
| `warmup_steps` | `3000` | `1` | ~4.9% of A100 total — MoE stability |
| `lr` | `4.0e-4` | `1.0e-3` | Peak LR after warmup |
| `min_lr_ratio` | `0.05` | `0.1` | Cosine floor fraction |
| `weight_decay` | `0.1` | `0.0` | AdamW; skipped for bias/norm/embed |
| `beta1` | `0.9` | `0.9` | Adam β₁ |
| `beta2` | `0.95` | `0.95` | Adam β₂; `eps=1e-6` hardcoded in code |
| `grad_clip` | `1.0` | `1.0` | Global norm clip; `0` disables |
| `aux_loss_alpha` | `0.01` | `0.01` | MoE load balance — [moe.md](moe.md) |
| `grad_checkpoint` | `true` | `true` | Calls `enable_gradient_checkpointing()` |
| `grad_checkpoint_every` | `3` | `2` | Checkpoint blocks where `idx % N == 0` |
| `compile` | `true` | `false` | `torch.compile` on CUDA only |
| `compile_mode` | `max-autotune` | — | Ignored when `compile: false` |
| `save_interval` | `2000` | `5` | Checkpoint every N optimizer steps |
| `log_interval` | `50` | `1` | Metrics logging cadence |
| `nan_guard` | `true` | `true` | Skip + rollback on non-finite loss |
| `nan_guard_max_consecutive` | `5` | `5` | NaN steps before reload latest ckpt |
| `save_dir` | `checkpoints/pretrain_a100` | `checkpoints/gpu_smoke` | Safetensors + optim + RNG |

**Implicit defaults** (not in YAML): `num_workers=4`, `pin_memory=true` on CUDA;
chunked CE `chunk_size=8192` in `pretrain.py`. Full loop detail: see sections
above in this document.

### B.6 `data` block — every field

| Key | A100 | Smoke | Notes |
|-----|------|-------|-------|
| `train_data_path` | `data/pretrain_chinchilla` | `data/pretrain_smoke` | `shard_*.bin` dir; prepare first |

**`gptoss-default` mixture** (from A100 YAML comments): FineWeb-Edu 50%,
FineWeb 20%, The Stack Python 15%, OpenMath 10%, arXiv 5%. Includes 10%
long-context augmentation (4096 packed sequences). See
[data_pipeline.md](data_pipeline.md).

### B.7 Cross-field interactions

#### `max_seq_len` × batch × Chinchilla

Changing `micro_batch_size` or `gradient_accumulation_steps` changes
`tokens_step` and therefore total tokens for fixed `total_steps`. Re-derive
`total_steps` if you change batch while holding 8B tokens fixed:

```
total_steps = 8_000_000_000 / (micro_batch_size × accum × max_seq_len)
```

#### YaRN triple consistency

These must align:

```
yarn_scale_factor ≈ yarn_target_seq_len / yarn_original_max_seq_len
yarn_original_max_seq_len == max_seq_len   (typically)
eval_max_seq_len <= yarn_target_seq_len    (recommended)
```

Mismatch triggers `ModelConfig` validation errors or degenerate ramps (see
[rope_yarn.md](rope_yarn.md)).

#### `window_size` vs `max_seq_len`

Windowed layers cache `min(window_size, T)` tokens per layer. Reduction vs pure
GQA approaches **2×** only when `T >> window_size` (e.g. 128K). At `T=4096` with
`W=128`, `scripts/kv_cache_benchmark.py` still reports ~1.9× because six layers
store 128 slots while six store the full 4096.

#### `moe_dispatch` × hardware

| Environment | Recommended |
|-------------|-------------|
| A100 + Triton installed | optional `triton_grouped` |
| CPU / Mac / smoke | `stacked` |

#### `compile` × `grad_checkpoint`

Both reduce memory pressure differently — compile optimizes kernels; checkpointing
drops activations. Compatible together on A100.

#### `sink_bias` × `dtype`

BF16 forward requires sink clamp `[-10, 15]` — always on when `sink_bias: true`.

### B.8 Override patterns (CLI)

#### Debug 10 steps on A100 config

```bash
python3 training/pretrain.py \
    --config configs/pretrain_a100_502m.yaml \
    --seed 42 \
    --max-steps 10
```

#### Enable Triton MoE (A100 only)

Add to YAML under `model:`:

```yaml
moe_dispatch: "triton_grouped"
```

Requires Triton installed. See [moe.md](moe.md#sanctioned-triton-path-moe_dispatchtriton_grouped).

#### Smaller effective batch (OOM)

```yaml
training:
  micro_batch_size: 4
  gradient_accumulation_steps: 8   # keep tokens_step = 131072
```

#### Resume

```bash
python3 training/pretrain.py \
    --config configs/pretrain_a100_502m.yaml \
    --seed 42 \
    --resume-from 40000
```

Use the **same** config file as the original run.

### B.9 How to verify

```bash
python3 -m pytest tests/test_training.py tests/test_validation.py -v
```

---

## Load-Bearing Invariants

1. **BF16 autocast** — no FP16 GradScaler.
2. **AdamW eps=1e-6** for BF16 stability.
3. **Aux loss α=0.01** — standard Switch, not aux-loss-free.
4. **NaN guard** enabled by default — **never disable in production without
   explicit user consent** ([`AGENTS.md`](../AGENTS.md) rule 6).
5. **Atomic checkpoints** — safetensors + separate optim/sched files.
6. **`--seed` required** for reproducibility claims.
7. **Chunked CE** — never materialise full `(B×T, V)` softmax.

---

## References

- [`training/pretrain.py`](../training/pretrain.py)
- [`configs/pretrain_a100_502m.yaml`](../configs/pretrain_a100_502m.yaml)
- [moe.md](moe.md) — aux loss theory
- [data_pipeline.md](data_pipeline.md) — `PretrainDataset`, shard preparation
- Hoffmann et al., *Training Compute-Optimal LLMs* (Chinchilla)

**Next:** [data_pipeline.md](data_pipeline.md) — corpus, tokenization, and
`PretrainDataset` internals.

<!-- docs:verified 2026-08-04 · 5da1a80 -->
