# Training — GPT-OSS-Lite

> **Source:** `training/pretrain.py` · **Config:** `configs/pretrain_a100_502m.yaml`
> **Companion:** [`data_pipeline.md`](data_pipeline.md) (where the tokens come from),
> [`utils.md`](utils.md) (checkpointing, memory), [`OPTIMIZATIONS.md`](OPTIMIZATIONS.md).

---

## 1. Overview

The pretraining script trains the 502 M-parameter (247 M active) GPT-OSS-Lite
model on a Chinchilla-optimal 8 B-token corpus on a single A100 80 GB in
**~16–20 hours**. The loop is intentionally conventional — a standard
micro-batch + gradient-accumulation + cosine-LR decoder-only LM pretraining
loop — so that every unusual decision (BF16 over FP16, the NaN guard, the
chunked cross-entropy, the RNG-state resume) is visible and auditable rather
than hidden behind a framework.

Reproducibility is **opt-in via `--seed N`**: without it, runs are not
reproducible (and deliberately so — forcing determinism costs throughput).
With it, two seeded runs match to within the BF16 determinism floor (~1e-4).

---

## 2. The training loop, end to end

```
seed_everything(seed)              # torch / numpy / python / cuda + CUBLAS_WORKSPACE_CONFIG
_set_hardware_perf_knobs()         # TF32, cuDNN benchmark, float32 matmul precision
build model → .to(dev)
optional torch.compile(max-autotune)
estimate_model_memory_gb → assert_fits_in_available_gpu   # pre-flight VRAM check
build AdamW(decay / no-decay param groups)
build LambdaLR(warmup → cosine → constant min_lr)
build PretrainDataset + DataLoader(num_workers=4, pin_memory, persistent_workers)
(optionally) resume from checkpoint: weights + optim + sched + RNG state
enable_gradient_checkpointing(every=3)

for step in range(total_steps):                       # 61 000 optimizer steps
    for micro in range(grad_accum):                   # 4 micro-batches / step
        with autocast(bf16):
            logits, aux_loss = model(input_ids)
            ce = chunked_cross_entropy(logits, targets, chunk=4096)
            loss = (ce + 0.01 * aux_loss) / accum
        if not isfinite(loss): NaN-guard rollback      # see §6
        loss.backward()
    clip_grad_norm_(foreach=True)
    optim.step(); sched.step(); optim.zero_grad()
    if step % 50   == 0: logger.log(...)
    if step % 2000 == 0: ckpt.save(model, optim, sched, extra_meta) + save RNG state
```

Effective batch = `micro_batch_size · grad_accum · n_tokens = 8 · 4 · 4096 =
131 072 tokens/step`; over 61 000 steps that is ~8.0 B tokens — the
Chinchilla-optimal count for a 502 M-param model (≈16 tokens/parameter).

---

## 3. Numerical-stability & performance knobs

### 3.1 BF16 autocast (no `GradScaler`)

```python
with autocast(device_type=dev.type, dtype=torch.bfloat16, enabled=(dev.type == "cuda")):
    logits, aux_loss = model(input_ids)
    ce = chunked_cross_entropy(logits, target_ids, chunk_size=4096)
    loss = (ce + aux_alpha * aux_loss) / accum
```

Forward + loss run in BF16 on CUDA. The key point: **BF16 does not need a
`GradScaler`** — only FP16 does. BF16 has the same exponent range as FP32
(8 bits), so it does not overflow the way FP16 does; the underflow that
`GradScaler` works around is handled by the FP32 master weights in AdamW. This
is the workspace-wide rule (BF16 on Ampere/Blackwell, no `GradScaler`) applied
here. The loss is divided by `accum` *before* backward so the accumulated
gradient is the correct mean over micro-batches.

### 3.2 `torch.compile(mode="max-autotune")`

Auto-invoked on CUDA when `training.compile: true`. Uses
`fullgraph=False` because the NaN-guard control flow (data-dependent branch
on `torch.isfinite(loss)`) breaks full-graph capture. `max-autotune` runs the
autotuner to pick the best fused kernels — a one-time compile cost, then
A100-specific kernels for the rest of the run. If `torch.compile` fails to
apply (older PyTorch, unsupported op), the script catches the exception and
continues without it rather than aborting a 16-hour run.

### 3.3 Hardware performance knobs

`_set_hardware_perf_knobs()` (called before model construction on CUDA):
- `torch.backends.cuda.matmul.allow_tf32 = True` — TF32 matmuls on Ampere
  (19-bit mantissa matmul, FP32 accumulate). ~3× faster, indistinguishable
  quality for LM training.
- `torch.backends.cudnn.allow_tf32 = True` + `cudnn.benchmark = True` — TF32
  convs + cuDNN autotune.
- `torch.set_float32_matmul_precision("high")` — allows TF32 for the FP32
  paths (the FP32 master-weight updates).

These are all no-ops on CPU, so the same script runs in both environments.

### 3.4 FP32 master weights via AdamW

```python
optim = AdamW(
    [{"params": decay_params, "weight_decay": weight_decay},
     {"params": no_decay_params, "weight_decay": 0.0}],
    lr=lr, betas=(0.9, 0.95), eps=1e-8,
    foreach=True,
    fused=(dev.type == "cuda"),
)
```

Two parameter groups: **decay** (matmuls, embeddings) and **no-decay** (biases,
norms, embeddings — matched by substring on the parameter name). AdamW keeps
**FP32 momentum/variance internally** even when the model params are BF16, so
the optimizer state is the numerical-stability anchor. `foreach=True` batches
the per-parameter update into one kernel; `fused=True` uses the CUDA-only fused
AdamW kernel (1.5–2× faster on A100/H100), falling back to `foreach` on CPU /
older CUDA. The betas `(0.9, 0.95)` and `eps=1e-8` are the GPT-OSS / LLaMA
convention.

### 3.5 Gradient checkpointing every 3rd layer

```python
model.enable_gradient_checkpointing(every=3)
# → checkpoint(block, x, positions, use_reentrant=False) on layers 0, 3, 6, 9
```

Layers `0, 3, 6, 9` recompute their activations in the backward pass instead of
storing them; the other 8 layers keep activations. This is a memory/throughput
trade tuned to "fit on one A100 80GB without dropping batch":
- Memory: ~`2/3` of layer activations are *not* stored (see `utils.md` /
  `_activation_bytes` `store_factor`).
- Throughput: the 4 checkpointed layers pay a ~1.3× forward-cost penalty in
  backward (recompute), the other 8 do not.
- `use_reentrant=False` is the modern recommended path (the reentrant variant
  is deprecated and has footguns around nested autograd).

### 3.6 Chunked cross-entropy (chunk = 4096)

```python
def chunked_cross_entropy(logits, targets, chunk_size=4096):
    flat_logits  = logits.view(-1, logits.size(-1))    # (B·T, vocab=128000)
    flat_targets = targets.view(-1)
    total_loss = torch.zeros((), device=..., dtype=...)
    n_total = flat_logits.size(0)                        # Python int, not a tensor
    for start in range(0, n_total, chunk_size):
        end = min(start + chunk_size, n_total)
        chunk_loss = F.cross_entropy(flat_logits[start:end], flat_targets[start:end],
                                     reduction="sum")
        total_loss = total_loss + chunk_loss
    return total_loss / max(1, n_total)
```

The final logits are `(B·T, vocab) = (32K, 128K)` per micro-batch — a full
softmax over that is ~16 GB of intermediate in HBM. Chunking to 4096 rows keeps
the peak at `4096 · 128K · 4 bytes ≈ 2 GB` and accumulates the *sum* of
per-chunk losses into a single scalar, dividing by `n_total` (a Python int —
not a tensor — once at the end). This avoids materialising the full softmax
and saves `2 · n_chunks` kernel launches versus the old two-scalar
accumulator (OPT-7).

### 3.7 `clip_grad_norm_(foreach=True)`

```python
try:
    nn.utils.clip_grad_norm_(model.parameters(), grad_clip, foreach=True)
except TypeError:
    nn.utils.clip_grad_norm_(model.parameters(), grad_clip)   # older PyTorch
```

Batches the per-parameter norm computation into one kernel (~2× faster on A100
with hundreds of parameter tensors), with a `try/except TypeError` fallback
for PyTorch < 2.1 where `foreach` did not exist on this function.

### 3.8 `CUBLAS_WORKSPACE_CONFIG=:4096:8`

Set in `seed_everything` only if not already present. Required for *full*
CUDA determinism (cuBLAS non-determinism otherwise breaks bit-exactness across
seeds). Harmless when cuBLAS is not used (CPU).

### 3.9 RNG seeding

`seed_everything(seed)` seeds Python `random`, NumPy, `torch`, and
`torch.cuda` (when available). It must be called **before** model construction
(so weight init is reproducible) and **before** DataLoader creation (so shuffle
order is reproducible). Without `--seed`, none of this is set and runs are
deliberately non-deterministic.

---

## 4. The LR schedule

```python
def make_warmup_cosine_lambda(warmup_steps, total_steps, min_lr_ratio=0.05):
    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(1, warmup_steps)                       # linear warmup
        if step >= total_steps:
            return min_lr_ratio                                      # hold at min
        progress = (step - warmup_steps) / (total_steps - warmup_steps)
        return min_lr_ratio + (1 - min_lr_ratio) * 0.5 * (1 + cos(π · progress))  # cosine
    return lr_lambda
```

Three phases, all expressed as a multiplier on the base `lr = 4e-4`:
1. **Linear warmup** over 2 000 steps — ramps from 0 to `lr` to avoid
   early-training instability (large LR + random weights = divergence).
2. **Cosine decay** from `lr` down to `min_lr_ratio · lr = 0.05 · 4e-4 = 2e-5`
   over the remaining 59 000 steps.
3. **Constant at min** once `step >= total_steps` (the `>=` branch handles
   training past `total_steps` gracefully).

The schedule is a `LambdaLR` so it composes with the optimiser's own LR
handling and saves/restores cleanly in the checkpoint.

---

## 5. The NaN guard with checkpoint rollback

The training loop's safety net. On every micro-step:

```python
if not torch.isfinite(loss):
    if nan_guard:
        nan_count += 1
        print(f"[nan-guard] step {step}: non-finite loss ({nan_count}/{nan_max_consec})")
        optim.zero_grad(set_to_none=True)
        micro_step = 0
        if nan_count >= nan_max_consec:           # default 5 consecutive NaNs
            latest = ckpt.latest_step()
            if latest is not None:
                ckpt.load(model, optim, sched, step=latest)   # full rollback
                step = latest
                nan_count = 0
            else:
                raise RuntimeError("NaN guard triggered with no checkpoint to roll back to.")
        continue
    else:
        raise RuntimeError(f"Non-finite loss at step {step}: {loss.item()}")

nan_count = 0          # reset on any *good* step
loss.backward()
```

The state machine:
- A single non-finite loss is treated as a transient spike — zero the grads,
  reset the micro-step counter, skip the step, keep going. This survives the
  occasional BF16 spike without wasting a checkpoint.
- After `nan_guard_max_consecutive` (5) **consecutive** NaNs, the run rolls
  back to the latest *complete* checkpoint (`latest_step()` requires all three
  of `model_step_N.safetensors`, `optim_step_N.pt`, `meta_step_N.json` — see
  [`utils.md`](utils.md)), resyncs the step counter, and continues. This
  recovers from a sustained divergence.
- If there is no checkpoint to roll back to (early in the run), it raises —
  better to fail loudly than to silently spin on NaNs forever.
- `nan_count` is reset to 0 on every *good* step, so the counter measures
  *consecutive* NaNs, not total.

This is why checkpoint atomicity (see [`utils.md`](utils.md)) matters: the
rollback must land on a known-good state, never a half-written one.

---

## 6. Reproducibility — the full RNG chain

A seeded run is reproducible end-to-end because every source of randomness is
captured and restored:

1. **At start**: `seed_everything(seed)` seeds all four RNGs.
2. **At every save**: the RNG state is written to a sibling
   `rng_step_N.pt` file alongside the checkpoint, capturing the `{python,
   numpy, torch, cuda}` states at that step.
3. **On `--resume-from N`**: the script loads weights + optimiser + scheduler
   from the checkpoint *and* restores the RNG state from `rng_step_N.pt`, so the
   resumed run is **bit-identical to a non-interrupted run** from that step
   onward.
4. **MoE dispatch** uses `torch.argsort(stable=True)` so routing ties break
   identically across runs (see [`moe.md`](moe.md) §6.3).

The determinism floor under BF16 is **~1e-4** — BF16 matmuls are not
bit-reproducible even with `CUBLAS_WORKSPACE_CONFIG` set, because the
reduction order is not fully deterministic. Two seeded runs should match
*within* that tolerance; demanding bit-exact equality under BF16 is not
achievable.

---

## 7. The dataset / DataLoader

`PretrainDataset` (in `pretrain.py`) reads the packed-token shards produced by
the data pipeline (see [`data_pipeline.md`](data_pipeline.md)). Each
`__getitem__` returns a `(input_ids, target_ids)` window of length
`max_seq_len + 1`, sliced as `chunk[:-1]` / `chunk[1:]` — the standard
next-token LM shift. Shards are mmap'd (`torch.from_file(..., shared=True)` or
`torch.load(..., mmap=True)`) so the 32 GB corpus is never loaded into RAM;
the last-loaded shard is cached to amortise the mmap cost across consecutive
windows. Cross-shard windows are stitched with a multi-shard scan, with
`bisect.bisect_right` giving `O(log N)` shard lookup (OPT-10).

The DataLoader uses `num_workers=4`, `pin_memory=True` (on CUDA), and
`persistent_workers=True` — async H2D transfer plus worker reuse across epochs
(avoids the per-epoch re-import cost). `drop_last=True` keeps the
gradient-accumulation math exact.

---

## 8. Pre-flight VRAM check

Before the loop starts, `estimate_model_memory_gb(model, seq_len, batch,
grad_checkpoint=...)` sums params + FP32 optimiser state (12 bytes/param for
AdamW m, v, master) + KV cache + activations, adds an auto-detected CUDA
overhead (~17% of total GPU memory, capped at 13.7 GB), and
`assert_fits_in_available_gpu(est)` raises *before* training if the estimate
exceeds `total_memory − 2 GB` margin. A 16-hour run that OOMs at step 5 000 is
far worse than a clear error at step 0. On CPU this is a no-op. See
[`utils.md`](utils.md).

---

## 9. Design rationale & rejected alternatives

| Decision | Rationale | Rejected alternative |
|---|---|---|
| BF16 autocast, no `GradScaler` | BF16 has FP32 exponent range; scaler is FP16-only | FP16 + GradScaler — more moving parts, overflow risk |
| `fullgraph=False` compile | NaN-guard branch breaks full graph | `fullgraph=True` — would force removing the safety net |
| FP32 AdamW master weights | Optimiser state is the numerical anchor under BF16 | BF16 optimiser state — momentum/variance underflow |
| Checkpoint every 3rd layer | Fits 502 M on one A100 80GB without dropping batch | Every layer — too slow; none — OOM |
| Chunked CE (chunk=4096) | Peak softmax 2 GB, not 16 GB | Full softmax — OOM at large B·T |
| NaN guard with rollback (5 consec) | Survives transient spikes, recovers from divergence | Hard abort on first NaN — wastes a 16-h run on a spike |
| RNG sibling file + restore on resume | Bit-identical resume | Re-seed on resume — resume ≠ fresh run |
| Opt-in `--seed` | Determinism costs throughput; let the user choose | Always-deterministic — slower for everyone |

---

## 10. Edge cases & pitfalls

- **`gradient_accumulation_steps < 1`**: raises `ValueError` — guarded in
  `main()`. Same for `micro_batch_size < 1`.
- **`max_steps` override**: `--max-steps N` rewrites `total_steps` in the
  config before the schedule is built, so the cosine decay still ends cleanly
  at `N`.
- **Resume into a differently-shaped model**: `ckpt.load` uses
  `strict=False` and warns on missing/unexpected keys, so a config change
  produces a loud warning, not a silent shape mismatch.
- **`compile` failure is non-fatal**: caught and the run continues without
  compile — do not assume the run is compiled just because the config says so.
- **`grad_checkpoint_every` vs `n_layers`**: `enable_gradient_checkpointing(every=3)`
  checkpoints layers `0, 3, 6, 9`. With `n_layers = 12` this is exactly 4
  layers; a different `n_layers` changes the count and the memory savings.

---

## Implementation notes (extracted from code review)

- **BF16 autocast**: `torch.amp.autocast(device_type=dev.type,
  dtype=torch.bfloat16)` wraps the forward + loss. No `GradScaler` is
  needed for BF16 (only FP16 requires it).
- **`torch.compile(max-autotune)`**: auto-invoked on CUDA when
  `training.compile: true` in the YAML. `fullgraph=False` is used because
  the NaN-guard control flow breaks full-graph capture.
- **TF32 + cuDNN benchmark + `set_float32_matmul_precision("high")`**:
  set on CUDA before model construction via
  `_set_hardware_perf_knobs()`. These are the standard A100 performance
  knobs; harmless on CPU.
- **FP32 AdamW master weights**: `AdamW(foreach=True, fused=(dev.type ==
  "cuda"))` keeps FP32 momentum/variance internally even when the model
  params are BF16. `foreach=True` batches the per-param update into one
  kernel; `fused=True` uses the CUDA-only fused kernel (1.5–2× faster on
  A100/H100). Falls back to `foreach` on CPU/older CUDA.
- **Gradient checkpointing every 3rd layer**: `GPTOSS.enable_gradient_checkpointing(every=3)`
  applies `torch.utils.checkpoint.checkpoint(block, x, positions,
  use_reentrant=False)` on layers `0, 3, 6, 9`. The other layers keep
  their activations for the backward pass. `use_reentrant=False` is the
  recommended modern path.
- **NaN guard with checkpoint rollback**: if `torch.isfinite(loss)` is
  False, the optimiser is zeroed, the micro-step counter is reset, and
  after `nan_guard_max_consecutive` (default 5) consecutive NaNs the
  latest checkpoint is reloaded and the step counter is resynced. Without
  a checkpoint to roll back to, the run raises.
- **Chunked cross-entropy (chunk=4096)**: `chunked_cross_entropy` flattens
  logits and targets, then runs `F.cross_entropy(..., reduction="sum")`
  over 4096-row chunks, accumulating into a single `total_loss` scalar
  and dividing by `n_total` (a Python int) once at the end. This avoids
  materialising the full `(B*T, vocab)` softmax in HBM and saves
  `2 × n_chunks` kernel launches vs the old two-scalar accumulator.
- **`clip_grad_norm_(foreach=True)`**: batches the per-param norm into
  one kernel (~2× faster on A100); falls back to the loop on older
  PyTorch via `try/except TypeError`.
- **`CUBLAS_WORKSPACE_CONFIG=:4096:8`**: set in `seed_everything` when
  not already present. Required for full CUDA determinism; harmless when
  cuBLAS is not used.
- **RNG seeding**: `seed_everything(seed)` seeds Python `random`, NumPy,
  `torch`, and `torch.cuda` (when available). Must be called BEFORE model
  construction (so weight init is reproducible) and BEFORE DataLoader
  creation (so shuffle order is reproducible). Without `--seed`, runs are
  NOT reproducible.

## Reproducibility

- Checkpoints include RNG state in a sibling `rng_step_N.pt` file
  (`{python, numpy, torch, cuda}` states). On `--resume-from N`, the
  script restores the RNG state from that file so the resumed run is
  bit-identical to a non-interrupted run.
- `torch.argsort(stable=True)` in MoE dispatch ensures the same input
  always produces the same permutation across runs (see
  `documentation/moe.md`).
- Determinism floor under BF16 is ~1e-4 (BF16 non-determinism); two
  seeded runs should match within that tolerance.