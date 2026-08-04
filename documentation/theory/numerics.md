# Numerics — Precision, Range, and the Bit Budget of a 502M Run

> **Where in the code:** `training/pretrain.py` (autocast, TF32 knobs, NaN guard), `models/transformer.py:RMSNorm`, `models/attention.py` (sink clamp, FP32 scores), `models/moe.py` (FP32 router/aux softmax). Companion derivations live in [attention math](attention_math.md), [optimizers](optimizers.md), and [triton programming](triton_programming.md); the primer-level version of §6 is [foundations](../foundations.md) §10.

## 1. 60-second summary

Every number in this model is stored in one of a handful of binary formats, and the choice between them decides whether a 61,000-step run converges or diverges. GPT-OSS-Lite trains in **BF16** (bfloat16): the same 8-bit exponent as FP32, so the *range* of representable magnitudes matches FP32, at the cost of a 7-bit mantissa (machine epsilon $2^{-7}$). That choice removes the single most common pretraining failure mode — FP16 gradient underflow and its band-aid, loss scaling — and lets the loop skip `GradScaler` entirely. Precision is recovered in the places that need it by deliberate **FP32 islands**: RMSNorm statistics, attention scores, and the router/auxiliary softmax are promoted to FP32 explicitly. Two guards keep the run alive: the sink bias is clamped to $[-10, 15]$ so the learned mask addend can never overflow the softmax exponential, and a run-level NaN guard skips poisoned steps and rolls back after five consecutive non-finite losses. `training/pretrain.py:_set_hardware_perf_knobs` additionally enables TF32 on A100 tensor cores so the remaining FP32 matmuls are fast.

## 2. Why it matters here

The design decisions of this repo are, at bottom, numerical decisions:

- **`dtype: "bf16"`** in `configs/pretrain_a100_502m.yaml` drives `_amp_dtype` and the autocast region in `training/pretrain.py:main`. BF16 is the load-bearing invariant that makes "no GradScaler" safe ([training](../training.md) lists it as invariant #1).
- **Vocab 128,000** makes the per-class cross-entropy gradient $p_i \approx 1/V \approx 7.8\times10^{-6}$ — below FP16's minimum normal (see §6.1). The format choice is decided by this number.
- **Top-2-of-8 routing** computes a softmax over 8 expert logits whose *saturation* (logits $\gtrsim 88$) would overflow BF16's exponential; `models/moe.py:MoERouter.forward` and `models/moe.py:aux_load_balancing_loss` promote to FP32.
- **The learned sink bias** is an additive logit riding inside the SDPA mask; `models/attention.py:GPTOSSAttention.forward` clamps it to $[-10, 15]$ so the mask addend and the resulting softmax stay finite at every one of the 12 layers (`models/attention.py:SINK_CLAMP_MIN` / `SINK_CLAMP_MAX`).
- **A 61,000-step, 8×4096-token schedule** means roughly $2\times10^9$ token-events per run. Even a once-in-a-million numerical accident *will* occur; the NaN guard in `training/pretrain.py:main` exists because no format choice can prevent every NaN source.

No pretraining run has completed yet; this chapter derives the numerics contract the run depends on, and everything quantitative below is either derived here or marked `[INFERENCE]` (`.benchmarks/` is empty, so all A100 timing figures are estimates).

## 3. Intuition — the logarithmic ruler

Think of a floating-point format as a ruler marked in scientific notation with a fixed number of significant digits. The exponent field decides *where on the number line* the ruler sits — which orders of magnitude are reachable at all. The mantissa field decides *how finely* it is marked — how many significant digits each mark carries. FP16 is a short ruler covering only $10^{\pm5}$; FP32 and BF16 are long rulers covering $10^{\pm38}$. The catch: BF16's ruler is long but coarsely marked (about 2 significant decimal digits), while FP16's is short but finer (about 3). Training gradients are tiny numbers that live at the far bottom of the scale — so range, not fineness, is the survival constraint, and BF16 wins. Precision is then spent where it matters by switching the computation to FP32's fine ruler for a single op (softmax, norm statistics), then switching back.

## 4. IEEE-754: sign, exponent, mantissa

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

## 5. TF32 — the Ampere matmul compromise

A100 tensor cores accept three input precisions: FP16/BF16, and TF32 — a 19-bit format, $1 + 8 + 10$: the FP32 exponent field (full FP32 range) with a 10-bit mantissa, $\varepsilon_{\mathrm{TF32}} = 2^{-10} \approx 9.77\times10^{-4}$ — the same precision as FP16 but the range of FP32. The tensor core rounds the FP32 inputs to TF32, computes the products exactly, and accumulates in FP32, roughly doubling FP32 matmul throughput on the A100 relative to the FP32 CUDA-core path.

`training/pretrain.py:_set_hardware_perf_knobs` opts into this path explicitly:

```python
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.set_float32_matmul_precision("high")
```

It also sets `torch.backends.cuda.preferred_blas_library = "cublaslt"` (cuBLASLt, A100-tuned; the code comment claims "2–5% faster" — unmeasured, `[INFERENCE]`) and `cudnn.benchmark_limit = 0` (exhaustive cuDNN algorithm search, "~3–5% on A100" per the comment — also `[INFERENCE]`). The subtlety: TF32 only changes *FP32* matmuls. This repo's production path runs the model in BF16 via autocast, so TF32 matters for whatever FP32 matmuls remain — mostly the FP32 islands of §8 and any un-autocast code. It is not a source of precision loss for the BF16 path, because BF16 matmuls already run at $\varepsilon = 2^{-7}$ by design. The one caution: enabling TF32 makes FP32 matmuls round their inputs to 10-bit mantissas, so bit-exact reproducibility across GPU vendors is not guaranteed — a reproducibility note, not a correctness issue.

## 6. Why BF16 for pretraining (and not FP16)

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

Any gradient magnitude that the FP32 optimizer state can hold (down to $1.18\times10^{-38}$; see [optimizers](optimizers.md) for the FP32 master-state story) survives the BF16 storage round-trip. The per-class gradient $7.81\times10^{-6}$ from (6) is a perfectly ordinary BF16 number — there are 32 orders of magnitude of headroom below it. Hence **no `GradScaler`**: the scale $S$ in (7) can be fixed at 1 for every step, and the optimizer consumes unscaled gradients, which keeps the loss, the learning rate, and the weight decay all in their natural units.

What BF16 gives up is precision: $\varepsilon = 2^{-7}$ versus FP16's $2^{-10}$ — 8× coarser rounding per operation. That is acceptable for three reasons, each true in this repo:

1. **FP32 accumulation** — tensor-core matmuls accumulate products in FP32, so the rounding happens once at the input (relative error $\le 2^{-8}$) rather than once per product-add.
2. **FP32 optimizer state** — AdamW keeps FP32 moment estimates (and FP32 master weights in the fused path), so the *updates* are precise even though the forward pass is coarse; this is also why `training/pretrain.py:main` uses `eps=1e-6` rather than $10^{-8}$: a $10^{-8}$ epsilon added to a BF16-valued second moment would itself underflow (see [optimizers](optimizers.md)).
3. **FP32 islands** — the numerically fragile ops (softmax, norm statistics, router probabilities) are explicitly promoted to FP32, as the next two sections show.

## 7. Autocast — what runs where

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

## 8. FP32 accumulation islands inside the model

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

The dot product over $D = 96$ dims feeds directly into softmax, which exponentiates. If the score accumulation ran in BF16, the relative error of each score would be $\sqrt{96}\cdot 2^{-8} \approx 0.12$ — a 12% perturbation of the exponent argument, i.e. ~12% multiplicative noise on the softmax weights. FP32 accumulation brings it to $\sqrt{96}\cdot 2^{-24} \approx 6\times10^{-7}$, so the score error is dominated by the BF16 quantization of the inputs themselves ($\le 2^{-8}$ relative, a 0.4% perturbation of the exponent argument — harmless). The softmax then runs in FP32 and the resulting weights are cast to the value dtype for the final matmul, mirroring what the fused SDPA path does internally. (The production path is `models/attention.py:causal_attention`, whose kernel handles the accumulation; the FP32 reference exists precisely to give the tests a numerically clean target — see the full derivation in [attention math](attention_math.md).)

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

and in BF16 at a similar magnitude (BF16's max is the same order). Saturated router logits are not hypothetical — `tests/test_moe.py:test_aux_loss_robust_to_bf16_saturation` constructs exactly this case. An `inf` in one softmax entry makes the output NaN (inf/inf), which then poisons every expert output and the aux loss. The `.float()` widens the safe logit range to $(-\infty, 88.7]$ and keeps the per-expert probabilities $P_i$ and routing fractions $f_i$ accurate; the aux loss $N \sum_i f_i P_i$ multiplies small probabilities, and BF16's $2^{-8}$ relative rounding on near-zero $P_i$ would swamp the gradient signal for under-used experts (full derivation in [moe theory](moe_theory.md)).

## 9. The sink-bias clamp — a finite mask addend

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

a dynamic range of $\approx 7.2\times10^{10} \approx 2^{36}$ — comfortably inside both FP32's and BF16's representable ranges (from §4.4: $[1.18\times10^{-38}, 3.39\times10^{38}]$), and far from the softmax overflow threshold of (10). The upper bound is a *design* bound, not the FP32 limit: $e^{15} \approx 3.3\times10^{6}$ makes the sink a strong but surmountable attractor against a window of 128 scores, whereas $b_h = 50$ would give $e^{50} \approx 5.2\times10^{21}$ and collapse essentially all attention mass onto the sink's zero value vector (this is the behavior `tests/test_attention.py:test_sink_bias_high_value_collapses_attention` probes; the sink's role is derived in [ATTENTION_SINKS](../ATTENTION_SINKS.md)). The lower bound keeps $e^{b_h}$ and its gradient alive: $e^{-10}$ is 33 orders of magnitude above the BF16 underflow floor.

**Gradient flow.** `models/attention.py:GPTOSSAttention.forward` applies the clamp at forward time on a *copy*:

```python
sink_bias_clamped = self.sink_bias.clamp(SINK_CLAMP_MIN, SINK_CLAMP_MAX)
```

The `nn.Parameter` itself is never mutated, so AdamW's moment estimates stay consistent with the parameter's true value, and while $b_h$ is inside $[-10, 15]$ the backward pass flows through the clamp with derivative 1. Outside the interval the local derivative is 0 — a deliberate stop that prevents the parameter from being pushed further out — and the parameter can still be pulled back in by the optimizer. `tests/test_attention.py:test_sink_bias_clamped_at_forward` pins both properties: after a forward with `sink_bias` filled with 1000.0, the parameter is still 1000.0 and the output is finite.

## 10. The NaN guard — run-level defense

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

## 11. Pitfalls and verification

| Trap | Consequence | Guard |
|---|---|---|
| Training in FP16 without scaling | Per-class gradients (6) flush to subnormal/zero; silent dead parameters | BF16 default + no GradScaler; `training.md` invariant #1 |
| Trusting autocast for softmax/norm | Autocast widens only the fragile ops on CUDA; elementwise ops stay BF16 | Explicit `.float()` in `models/transformer.py:RMSNorm.forward`, `models/moe.py:MoERouter.forward`, `models/moe.py:aux_load_balancing_loss` |
| Unclamped sink bias | `exp` overflow at $s' > 88.7$ → NaN, or bias death at $b_h \to -\infty$ | Forward-time clamp $[-10, 15]$, `models/attention.py:SINK_CLAMP_MIN/MAX`; `pytest tests/test_attention.py -v` (`test_sink_bias_clamped_at_forward`) |
| BF16 router softmax | Saturated logits → `exp` overflow → NaN aux/expert output | FP32 softmax; `pytest tests/test_moe.py -v` (`test_aux_loss_robust_to_bf16_saturation`) |
| AdamW `eps=1e-8` with BF16 params | Epsilon below the BF16 grid underflows the second moment; late-stage convergence stalls | `eps=1e-6` in `training/pretrain.py:main` (see [optimizers](optimizers.md)) |
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

<!-- docs:verified 2026-08-04 · 5da1a80 -->
