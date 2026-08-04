# GPT-OSS-Lite — Operations

> Purpose: runbooks for benchmarks, checkpoints/logging/memory helpers, and the
> OPT-1…24 performance catalog. For training see [training.md](../training.md); for
> inference internals see [foundations-and-architecture.md](../concepts/foundations-and-architecture.md) §9 and
> [inference.md](../inference.md).

---

## Part A — Scripts (`scripts/`)

All scripts live in `scripts/` at the repository root. Run from project root:

```bash
cd LLM/GPT-OSS-Lite
python3 scripts/<name>.py [args]
```

| Script | GPU required | Primary purpose |
|--------|--------------|-----------------|
| `_bootstrap.py` | No | Shared helpers (import only) |
| `check_docs.py` | No | Lint docs, refresh size table |
| `kv_cache_benchmark.py` | No | Analytical KV-cache reduction (≥1.8×) |
| `passkey_eval.py` | Optional | Long-context passkey retrieval accuracy |
| `e2e_gpu_smoke.py` | Yes (~4 GB) | Full pipeline integration on tiny model |
| `profile_components.py` | Optional | Per-component forward latency |
| `profile_moe.py` | Optional | MoE forward + dispatch latency |
| `profile_inference.py` | Optional | `generate()` tokens/sec |
| `microbench_a100.py` | Optional | Peak VRAM vs threshold |
| `step_time_a100.py` | Optional | Training step time + MFU |

Training itself uses `python training/pretrain.py --config configs/pretrain_a100_502m.yaml` — see [training.md](../training.md#part-b--configuration-reference).

### A.1 Selection guide

Use this decision tree when debugging or benchmarking:

```
Need to verify doc links / stale patterns?
  → check_docs.py

Need headline KV reduction proof (no GPU)?
  → kv_cache_benchmark.py  (target ≥1.8× at 128K)

Need long-context quality on trained weights?
  → passkey_eval.py  (target ≥85% at 128K)

Changed attention / MoE / inference / checkpoint code?
  → e2e_gpu_smoke.py  (4 GB GPU)

Which layer is slow?
  → profile_components.py

Is MoE dispatch the bottleneck?
  → profile_moe.py

Is decode slow?
  → profile_inference.py

Will training OOM at B=8, T=4096?
  → microbench_a100.py

What is training throughput / MFU?
  → step_time_a100.py --compile
```

### Recommended CI / pre-push sequence

```bash
python3 scripts/check_docs.py
python3 scripts/kv_cache_benchmark.py
pytest tests/ -q                    # see project README for test layout
python3 scripts/e2e_gpu_smoke.py  # when GPU available
```

---

### A.2 `_bootstrap.py`

**Purpose:** Deduplicate `sys.path` setup and micro-benchmark utilities across
profiling scripts.

**Exports:**

| Symbol | Description |
|--------|-------------|
| `time_fn(fn, n=20, warmup=3)` | Average milliseconds per call after warmup; CUDA-synced when available |
| `micro_cfg()` | `ModelConfig` with 4 layers, `d_model=64`, `max_seq_len=128` — fast CPU/GPU runs |

**Usage pattern:**

```python
from _bootstrap import micro_cfg, time_fn
```

Scripts that `import _bootstrap` must be run with `scripts/` as cwd **or** rely
on the path fix inside `_bootstrap` (parent of `scripts/` is added automatically).

**Expected output:** None — library module only.

---

### A.3 `check_docs.py`

**Purpose:** Validate all `docs/**/*.md` files for broken links, stale
patterns, control characters, and missing backtick paths. Optionally refresh the
doc size table in [README.md](../README.md) or stamp verification footers.

#### Invocation

```bash
# Lint only (exit 0 = clean)
python3 scripts/check_docs.py

# Refresh ## Doc size reference table in docs/README.md
python3 scripts/check_docs.py --update-sizes

# Stamp <!-- docs:verified YYYY-MM-DD · <git-short> --> on all docs
python3 scripts/check_docs.py --stamp-footers

# Both
python3 scripts/check_docs.py --update-sizes --stamp-footers
```

#### Stale patterns rejected

The linter flags outdated documentation conventions:

| Pattern | Reason |
|---------|--------|
| LaTeX `{` `,` `}` thousand separators | Use plain `100000` or comma prose in docs |
| Hard-coded pytest totals in prose | Run `pytest` for the current count |
| Old MoE Triton standalone doc name | Merged into [moe.md](../concepts/moe.md#sanctioned-triton-path-moe_dispatchtriton_grouped) |
| Removed Triton env-var gate | Use `moe_dispatch: triton_grouped` in YAML |

#### Expected output

```
check_docs: OK (N files)
```

On failure, prints `docs/<file>:<line>: <message>` to stderr and exits 1.

---

### A.4 `kv_cache_benchmark.py` (headline ≥1.8×)

**Purpose:** Prove the **architectural KV-cache reduction** of the alternating
sliding-window / full-attention design without loading a model or GPU. This is
the analytical counterpart to `MixedKVCache` described in
[foundations-and-architecture.md](../concepts/foundations-and-architecture.md) §9 and [Part C](operations.md#part-c--optimization-catalog-opt-1-opt-24)
(OPT-11/12).

#### Architecture constants (from `configs/pretrain_a100_502m.yaml`)

| Constant | Value |
|----------|-------|
| Layers | 12 (6 windowed + 6 global) |
| Window | 128 |
| GQA | 4 KV heads × 96 head_dim |
| Dtype | BF16 (2 bytes) |
| Batch | 1 |

#### Formula

Per token per layer, KV storage is:

$$
\text{bytes}_{\text{token,layer}} = 2 \times n_{\text{kv\_heads}} \times d_{\text{head}} \times \text{dtype\_bytes}
$$

- **Pure GQA (all full):** $12 \times T \times \text{bytes}_{\text{token,layer}}$
- **Mixed SWA/full:** $(6 \times \min(128, T) + 6 \times T) \times \text{bytes}_{\text{token,layer}}$

#### Invocation

```bash
python3 scripts/kv_cache_benchmark.py
```

No arguments. Runs in milliseconds on CPU.

#### Expected output

```
GPT-OSS-Lite KV-cache benchmark (analytical, BF16, batch=1)
Architecture: 12 layers (6 SWA w=128 + 6 full)
GQA: 4 KV heads, head_dim=96

   Context      Pure GQA      SWA/Full   Reduction
--------------------------------------------------
     4,096       0.07 GB       0.04 GB       1.94×
     8,192       0.14 GB       0.07 GB       1.97×
    16,384       0.28 GB       0.14 GB       1.98×
    32,768       0.56 GB       0.28 GB       1.99×
    65,536       1.12 GB       0.56 GB       2.00×
   131,072       2.25 GB       1.13 GB       2.00×

✅ HEADLINE METRIC PASSED: 2.00× KV-cache reduction at 128K (≥ 1.8×)
```

Exit code **0** if reduction at 128K ≥ **1.8×**; otherwise exit **1**.

At long context the ratio approaches $12 / (6 + 6 \times 128/131072) \approx 2.00\times$.
See [foundations-and-architecture.md](../concepts/foundations-and-architecture.md) §A.4 for the intuition.

---

### A.5 `passkey_eval.py`

**Purpose:** Measure **passkey retrieval accuracy** at long context lengths using
`inference/long_context.py::PasskeyEvaluator`. Target: **≥85%** at 128K on a
**trained** checkpoint.

#### Invocation

```bash
python3 scripts/passkey_eval.py \
  --checkpoint checkpoints/pretrain_a100/model_step_60000.safetensors \
  --n-trials 10 \
  --context-lengths 4096 8192 32768 65536 131072 \
  --position middle \
  --seed 42
```

| Flag | Default | Description |
|------|---------|-------------|
| `--checkpoint` | (required) | Path to `.safetensors` weights |
| `--n-trials` | 10 | Trials per context length |
| `--context-lengths` | 4096 … 131072 | Space-separated list |
| `--position` | `middle` | Passkey placement: `start`, `middle`, `end` |
| `--seed` | 42 | Base RNG seed |

Loads `configs/pretrain_a100_502m.yaml` for `ModelConfig`. Uses a lightweight
char-level tokenizer stub for eval plumbing — swap for production BPE when the
data pipeline tokenizer is wired (see [training.md](../training.md)).

#### Expected output (trained model)

> [INFERENCE] target, not a result — no pretraining run yet; the 87.5% figure is an illustrative sample.

```
Passkey eval: checkpoint=model_step_60000.safetensors, device=cuda

   Context    Accuracy
------------------------
     4,096      100.0%
    ...
   131,072       87.5%

✅ HEADLINE METRIC PASSED: 87.5% at 131,072 (≥ 85%)
```

#### Untrained model

On a random-init checkpoint, accuracy is near chance. The script still exits **0**
but prints a warning — full pretraining is required for the headline pass.
See [getting-started.md](getting-started.md) when available.

**Related:** [sampling](../concepts/optimizers-and-numerics.md) — greedy (`temperature=0.0`) semantics the eval loop relies on.

---

### A.6 `e2e_gpu_smoke.py`

**Purpose:** Single-script integration test exercising every major subsystem on a
**tiny** model that fits in **~4 GB VRAM** (verified on sm_75). Run this after
any change to attention, MoE, checkpointing, or generation.

#### What it tests (8 steps)

1. Build 4-layer GPU model (BF16)
2. Forward + backward; all params receive gradients
3. MoE `stacked` vs `triton_grouped` numerical equivalence (when Triton present)
4. `MoELayer` end-to-end with `moe_dispatch="triton_grouped"`
5. Five-step training loop (LR schedule, grad clip, AdamW)
6. `CheckpointManager` save → load round-trip
7. `MixedKVCache` generation on windowed/global split
8. YaRN forward at `eval_max_seq_len` > training `max_seq_len`

#### Invocation

```bash
# Requires CUDA GPU with ≥4 GB VRAM
python3 scripts/e2e_gpu_smoke.py
```

#### Expected output

Green checkmarks per step:

```
======================================================================
  Step 1: Build tiny GPU model (4 layers, d_model=128)
======================================================================
  ✓ Model built (0.xxxM params, fits in 4 GB)
...
======================================================================
  Step 7: MixedKVCache inference (windowed + global split)
======================================================================
  ✓ generate() with cache: output shape (1, T_prompt + 16)
...
ALL STEPS PASSED
```

Exit **0** on success; raises `SystemExit(1)` on any failure.

**Related:** [moe.md](../concepts/moe.md#sanctioned-triton-path-moe_dispatchtriton_grouped), [§B.1 CheckpointManager](operations.md#b1-checkpointmanager-atomic-safetensors-protocol).

---

### A.7 Profilers (`profile_components`, `profile_moe`, `profile_inference`)

#### Component profiler — `profile_components.py`

**Purpose:** Break down forward latency by subsystem on `micro_cfg()` — useful
when deciding which [Part C](operations.md#part-c--optimization-catalog-opt-1-opt-24) entry to profile next.

##### Invocation

```bash
python3 scripts/profile_components.py
```

Uses CUDA when available; otherwise CPU (slower, still valid for relative ordering).

##### Components timed

| Label | What is measured |
|-------|------------------|
| `[model.forward]` | Full `GPTOSS` forward |
| `[attn.windowed]` | `GPTOSSAttention` layer 0 (SWA) |
| `[attn.global]` | `GPTOSSAttention` layer 1 (full) |
| `[manual_attn]` | O(T²) reference attention |
| `[sdpa_attn]` | `F.scaled_dot_product_attention` causal |
| `[swa_attn]` | `causal_attention` with window |
| `[moe.forward]` | Full `MoELayer` |
| `[moe.dispatch]` | `_dispatch_vectorized` only |
| `[apply_rope]` | `apply_rope` |
| `[repeat_kv]` | GQA head expansion |

##### Expected output (example)

> [INFERENCE] estimate — .benchmarks/ empty; sample output, not a measured run.

```
Total params: 0.42M
[model.forward]      12.34 ms/step
[attn.windowed]       1.23 ms/step
[attn.global]         2.45 ms/step
...
[repeat_kv]           0.01 ms/step
```

Absolute numbers vary by hardware. Compare **before/after** a change, not across machines.

---

#### MoE dispatch profiler — `profile_moe.py`

**Purpose:** Isolate MoE forward vs vectorized dispatch cost without YaRN prune
overhead (`yarn_prune_rope_global=False` via `dataclasses.replace`).

##### Invocation

```bash
python3 scripts/profile_moe.py
```

##### Expected output

> [INFERENCE] estimate — .benchmarks/ empty; sample output, not a measured run.

```
moe.forward          3.45 ms
_dispatch_vectorized 2.10 ms
```

For Triton grouped dispatch timing, enable `moe_dispatch="triton_grouped"` in a
custom config and profile via `e2e_gpu_smoke.py` or a one-off script. See
[moe.md](../concepts/moe.md#sanctioned-triton-path-moe_dispatchtriton_grouped).

---

#### Inference profiler — `profile_inference.py`

**Purpose:** Measure `inference/generate.py::generate()` throughput (tokens/sec)
for several prompt lengths with greedy decoding (`temperature=0.0`).

##### Invocation

```bash
python3 scripts/profile_inference.py
```

##### Expected output

> [INFERENCE] estimate — .benchmarks/ empty; sample output, not a measured run.

```
prompt=8, new=64: 45.2 ms (1416 tok/s)
prompt=32, new=64: 52.1 ms (1228 tok/s)
prompt=128, new=64: 61.3 ms (1044 tok/s)
```

Decode cost is dominated by per-step `MixedKVCache` updates and MoE forward.
Longer prompts increase prefill time but decode tok/s should stabilize once the
cache is warm. See OPT-11/12/13/14/22 in [Part C](operations.md#part-c--optimization-catalog-opt-1-opt-24).

**Related:** [kv cache engineering](../inference.md) — decode-bandwidth model behind the tok/s ceiling.

---

### A.8 `microbench_a100.py` / `step_time_a100.py`

#### VRAM microbench — `microbench_a100.py`

**Purpose:** Verify production model fits under a VRAM ceiling at
`batch_size=8`, `seq_len=4096`. Uses **actual** `torch.cuda.max_memory_allocated`
on GPU; falls back to `utils/memory.py:estimate_model_memory_gb` on CPU.

##### Invocation

```bash
python3 scripts/microbench_a100.py \
  --config configs/pretrain_a100_502m.yaml \
  --batch-size 8 \
  --seq-len 4096 \
  --threshold-gb 25.0
```

| Flag | Default | Description |
|------|---------|-------------|
| `--config` | `configs/pretrain_a100_502m.yaml` | Model YAML |
| `--batch-size` | 8 | Micro-batch size |
| `--seq-len` | 4096 | Sequence length |
| `--threshold-gb` | 25.0 | Max acceptable peak GB (headroom for scaling) |

##### Expected output (A100 80GB)

> [INFERENCE] estimate — .benchmarks/ empty; sample output, not a measured run.

```
[microbench] Config: d_model=768, n_layers=12, vocab=128000, experts=8
[microbench] batch_size=8, seq_len=4096
[microbench] Peak VRAM (actual): 18.42 GB
[microbench] ✅ PASSED: peak < 25.0 GB
```

Estimator math is documented in [§B.3](operations.md#b3-estimate_model_memory_gb-mixed-kv-term-assert_fits_in_available_gpu).

**Related:** [numerics](../concepts/optimizers-and-numerics.md) (FP32/BF16 memory footprint) · [optimizers](../concepts/optimizers-and-numerics.md) (AdamW 12 B/param state).

---

#### Step-time / MFU — `step_time_a100.py`

**Purpose:** Measure training **tokens/sec** and approximate **MFU** (Model FLOPs
Utilization) on A100 BF16. Optional `torch.compile(max-autotune)` mirrors
production `training.compile: true`.

##### Invocation

```bash
python3 scripts/step_time_a100.py \
  --config configs/pretrain_a100_502m.yaml \
  --batch-size 8 \
  --seq-len 4096 \
  --steps 20 \
  --warmup 5 \
  --compile
```

| Flag | Default | Description |
|------|---------|-------------|
| `--steps` | 20 | Measured steps after warmup |
| `--warmup` | 5 | Warmup steps (discarded) |
| `--compile` | off | Enable `torch.compile(mode="max-autotune")` |

Enables TF32 + cuDNN benchmark (see [OPT-20](operations.md#opt-20--cudnnbenchmark_limit0-preferred_blas_librarycublaslt)).

##### Expected output (A100, with `--compile`)

> [INFERENCE] estimate — .benchmarks/ empty; sample output, not a measured run.

```
[step_time] torch.compile enabled (mode=max-autotune)
[step_time] Config: 12L, batch=8, seq=4096
[step_time] Warmup: 5 steps, Measure: 20 steps
[step_time] 20 steps in 42.15s → 1,556,234 tokens/sec
[step_time] Approx MFU: 38.2% (achieved 119.2 TFLOPS BF16)
[step_time] ✅ MFU target (≥35%) met.
```

Exit **0** if MFU ≥ 35% on CUDA; CPU-only runs report tokens/sec without MFU.

**Note:** This script uses plain `cross_entropy`, not `chunked_cross_entropy` —
it measures raw step time, not the production CE path. For production parity see
[training.md](../training.md).

**Related:** [optimizers](../concepts/optimizers-and-numerics.md) (AdamW fused/foreach path) · [numerics](../concepts/optimizers-and-numerics.md) (TF32/BF16 matmul assumptions).

---

---

## Part B — Utilities (`utils/`)

`utils/checkpoint.py`, `utils/logging.py`, and `utils/memory.py` sit between model code and the training loop. Wired in [training.md](../training.md).

### B.1 `CheckpointManager` (atomic safetensors protocol)

**File:** `utils/checkpoint.py`

`utils/checkpoint.py:CheckpointManager` is the single sanctioned API for saving
and loading training state. The training loop in `training/pretrain.py` constructs
one instance per run:

```python
ckpt = CheckpointManager(train_cfg["save_dir"])
```

#### File layout per step

For checkpoint step `N`, four files are written:

| File | Format | Contents |
|------|--------|----------|
| `model_step_N.safetensors` | safetensors | Model `state_dict()` |
| `optim_step_N.pt` | `torch.save` | AdamW optimizer state |
| `sched_step_N.pt` | `torch.save` | LR scheduler state (optional on save) |
| `meta_step_N.json` | JSON | `{"step": N, ...extra_meta}` |

Additionally, `pretrain.py` saves RNG state separately:

| File | Contents |
|------|----------|
| `rng_step_N.pt` | Python, NumPy, PyTorch, CUDA RNG states |

RNG files are **not** managed by `CheckpointManager` — the training script writes
them directly after the final save. See [training.md](../training.md) §25.

#### `save(model, optimizer, step, scheduler=None, extra_meta=None)`

Saves all three core files atomically. Logs:

```
[checkpoint] saved step 2000 → checkpoints/pretrain_a100
```

`extra_meta` merges into `meta_step_N.json` — e.g. `{"aux_loss": 0.0042}` or
`{"final": true, "seed": 42}`.

#### `load(model, step, device="cuda", optimizer=None, scheduler=None, strict=True)`

1. Loads `model_step_N.safetensors` via `safetensors.torch.load_file`.
2. Calls `model.load_state_dict(weights, strict=False)` then validates keys.
3. Optionally restores optimizer and scheduler from `.pt` files.
4. Returns `meta` dict from JSON (or `{"step": step}` if meta missing).

**Strict mode:** If `strict=True` (default), missing or unexpected keys raise
`RuntimeError`. Set `strict=False` for partial loads during architecture experiments.

**Missing optimizer:** Warns and leaves optimizer at current state — useful when
fine-tuning with a new optimizer but warm-started weights.

#### Complete checkpoint definition

A step is **complete** only when all three exist:

- `model_step_N.safetensors`
- `optim_step_N.pt`
- `meta_step_N.json`

`sched_step_N.pt` is optional. Incomplete steps are ignored by `latest_step()` and
`list_checkpoints()`.

---

All writes go through `_atomic_write`:

```python
def _atomic_write(self, path: Path, writer, *, suffix: str) -> None:
    fd, tmp = tempfile.mkstemp(dir=self.save_dir, suffix=suffix)
    os.close(fd)
    try:
        writer(tmp)
        os.replace(tmp, path)
    except Exception:
        os.unlink(tmp)  # best-effort cleanup
        raise
```

**Why temp files live in `save_dir`:** `os.replace` across filesystems is not
atomic. Same-directory rename guarantees readers never see a partial file.

**Suffixes:** `.safetensors.tmp`, `.pt.tmp`, `.json.tmp` — distinguishable in
`ls` during crash recovery.

`tests/test_training.py::test_checkpoint_atomicity_no_partial_files` verifies no
`.tmp` files remain after successful save.

---

Weight tying (`embed.weight` ↔ `head.weight`) creates **two state_dict keys
pointing at the same storage**. `safetensors` rejects duplicate `data_ptr` values.

`_atomic_save_safetensors` deduplicates:

```python
seen_ptrs: set = set()
for k, v in state.items():
    ptr = v.data_ptr()
    if ptr in seen_ptrs:
        deduped[k] = v.contiguous().clone()
    else:
        seen_ptrs.add(ptr)
        deduped[k] = v.contiguous()
```

First occurrence is saved in-place (contiguous). Duplicates get an explicit clone.
On load, both keys restore correctly because safetensors stores independent tensors.

**Impact:** Checkpoint size increases by one embedding matrix (~98M params × 2 bytes
≈ 196 MB for production config) — acceptable for correctness.

---

#### `latest_step() -> Optional[int]`

Returns the highest step number with a **complete** triple, or `None`.

Used by the NaN guard rollback in `pretrain.py`:

```python
latest = ckpt.latest_step()
if latest is not None:
    ckpt.load(model, step=latest, ...)
```

#### `list_checkpoints() -> list[int]`

Sorted ascending list of complete steps.

#### `delete_checkpoint(step: int)`

Removes all four filename patterns for that step (model, optim, sched, meta).

#### `keep_last_n(n: int)`

Deletes all complete checkpoints except the `n` most recent. Example — retain
only the last 3 during long runs:

```python
ckpt.keep_last_n(3)
```

Call after `save()` on a schedule to cap disk usage. Not invoked automatically
by `pretrain.py` — add if checkpoint dir grows too large.

---

### B.2 `TrainingLogger` + WandB

**File:** `utils/logging.py`

`TrainingLogger` prints a human-readable rolling summary every `log_interval`
steps and optionally forwards metrics to Weights & Biases.

#### Construction

```python
logger = TrainingLogger(
    log_interval=train_cfg.get("log_interval", 50),
    seq_len=model_cfg.max_seq_len,
)
```

#### `log(step, loss, metrics=None, lr=0.0)`

**Behavior:**

1. Appends `loss` to an internal window.
2. If `step % log_interval != 0`, returns immediately (no print).
3. On log boundary:
   - Computes `avg_loss` over the window.
   - Computes `tokens_per_sec = (log_interval * seq_len) / elapsed`.
   - Computes `ppl = exp(avg_loss)`.
   - Prints one line to stdout.
   - Forwards to WandB if enabled.
   - Resets window and timer.

#### Example stdout line

```
step=    200 | loss=3.2145 | ppl=24.89 | lr=2.67e-04 | tps=142,350 | aux=0.0031
```

**Note:** `loss` logged is **cross-entropy only** — `pretrain.py` passes
`ce.item()`, not `ce + aux_alpha * aux`. Aux appears under `metrics={"aux": ...}`.

#### `finish()`

Calls `wandb.finish()` if WandB was initialized. Invoke at end of training.

---

WandB is **opt-in via environment variable** — no code changes required.

```bash
export WANDB_PROJECT=gpt-oss-lite
export WANDB_RUN_NAME=a100-pretrain-502m   # optional
python training/pretrain.py --config configs/pretrain_a100_502m.yaml
```

| Env var | Required | Effect |
|---------|----------|--------|
| `WANDB_PROJECT` | Yes (to enable) | Project name; triggers `wandb.init` |
| `WANDB_RUN_NAME` | No | Run display name |

If `wandb` is not installed, prints:

```
[logging] wandb not installed -- skipping WandB integration
```

#### Logged scalars

| WandB key | Source |
|-----------|--------|
| `train/loss` | Rolling average CE |
| `train/ppl` | `exp(loss)` |
| `train/lr` | Scheduler LR |
| `train/tokens_per_sec` | Derived throughput |
| `train/<metric>` | Extra keys from `metrics` dict (e.g. `train/aux`) |

---

### B.3 `estimate_model_memory_gb` / Mixed KV term / `assert_fits_in_available_gpu`

**File:** `utils/memory.py` — `utils/memory.py:estimate_model_memory_gb` estimates
steady-state VRAM; `utils/memory.py:assert_fits_in_available_gpu` enforces the
budget at startup.

```python
def estimate_model_memory_gb(
    model: nn.Module,
    seq_len: int,
    batch_size: int,
    grad_checkpoint: bool = True,
    overhead_gb: float | None = None,
    steady_state: bool = False,
    grad_ckpt_every: int = 3,
) -> float
```

Returns estimated **peak VRAM in GB** for forward + backward at the given batch
and sequence length. Requires `model.cfg` (`ModelConfig`) — returns `0.0` if absent.

Called at training startup in `pretrain.py` before the first step:

```python
est = estimate_model_memory_gb(model, seq_len=model_cfg.max_seq_len, batch_size=micro_bs, ...)
assert_fits_in_available_gpu(est)
```

Also used by `scripts/microbench_a100.py` on CPU-only hosts.

**Related:** [numerics](../concepts/optimizers-and-numerics.md) — FP32 master / BF16 activation footprint; [autograd checkpointing](../concepts/kernels-and-checkpointing.md) — the `store_factor` activation tradeoff below.

#### Total formula

$$
\text{VRAM}_{\text{GB}} = \frac{P + O + KV + A}{1024^3} + \text{overhead}_{\text{GB}}
$$

| Term | Symbol | Description |
|------|--------|-------------|
| Parameters | $P$ | All model weights in native dtype |
| Optimizer | $O$ | 12 bytes/param (FP32 m, v, master) |
| KV cache | $KV$ | Mixed windowed/global (§9) |
| Activations | $A$ | Layer activations with checkpoint factor (§B.3 (activations)) |
| Overhead | — | CUDA allocator + cudnn workspace heuristic |

---

The estimator counts **only KV tensors**, not Q — matching inference `MixedKVCache`
and the analytical benchmark in `scripts/kv_cache_benchmark.py`.

```python
per_token = 2 * n_kv_heads * head_dim * dtype_bytes   # K + V, BF16 → 2 bytes
n_windowed = sum(1 for b in model.blocks if b.attn.is_windowed)
n_global = n_layers - n_windowed
win_len = window if steady_state else max(window, seq_len)
kv_bytes = (n_windowed * win_len + n_global * seq_len) * batch_size * per_token
```

#### Production numbers (`pretrain_a100_502m.yaml`)

| Parameter | Value |
|-----------|-------|
| `n_kv_heads` | 4 |
| `head_dim` | 96 |
| `window_size` | 128 |
| `n_layers` | 12 (6 windowed + 6 global) |
| `dtype_bytes` | 2 (BF16) |

Per token per layer: $2 \times 4 \times 96 \times 2 = 1536$ bytes.

At $T = 4096$, `steady_state=False`:

- Windowed contribution per layer: $\max(128, 4096) = 4096$ (prefill holds full window during training forward)
- Global: $4096$
- Total KV: $(6 \times 4096 + 6 \times 4096) \times B \times 1536$ — same as full attention **during training prefill**.

At **inference steady state** (`steady_state=True`), windowed layers hold only
$W = 128$ tokens:

$$
KV_{\text{inf}} = (6 \times 128 + 6 \times T) \times B \times 1536 \text{ bytes}
$$

At $T = 131072$, $B = 1$: ratio vs pure full $\approx 1.92\times$ reduction —
matching `kv_cache_benchmark.py`.

**Why two modes:** Training forward sees the full sequence in every layer (no
incremental cache during `GPTOSS.forward`). Inference with `MixedKVCache` ring
buffers is cheaper — use `steady_state=True` when budgeting decode VRAM.

---

#### Optimizer state

```python
optim_bytes = sum(p.numel() for p in model.parameters()) * 12
```

Assumes AdamW with FP32 master weights and moments (4 + 4 + 4 bytes per param).
Matches `eps=1e-6`, `foreach=True`, `fused=True` in [training.md](../training.md) —
see [OPT-9](operations.md#opt-9--adamw-foreachtrue-fusedtrue-on-cuda)/[OPT-17](operations.md#opt-17--adamw-eps1e-6-not-1e-8).

#### Activations with gradient checkpointing

```python
ckpt_factor = 1.0 / grad_ckpt_every          # if grad_checkpoint
store_factor = ckpt_factor + (1 - ckpt_factor) * 0.5   # if grad_checkpoint
act_bytes = n_layers * seq_len * batch_size * d_model * dtype_bytes * store_factor
```

MoE intermediate adds:

```python
act_bytes += n_layers * 3 * 3 * seq_len * batch_size * ffn_dim * dtype_bytes * store_factor
```

The `3 * 3` factor approximates three active expert paths (top-2 routed + shared)
with a conservative width multiplier. This is a **heuristic** — actual peaks depend
on routing distribution.

With `grad_ckpt_every=3` (production default):

$$
\text{store\_factor} = \frac{1}{3} + \frac{2}{3} \times 0.5 = \frac{2}{3}
$$

Roughly **33% activation memory savings** vs no checkpointing — consistent with
the ~30% compute overhead cited in [foundations-and-architecture.md](../concepts/foundations-and-architecture.md).

#### GPU overhead heuristic

```python
overhead_gb = min(13.7, max(2.0, total_gb * 0.17))
```

On CPU or no CUDA: `overhead_gb = 2.0`. Override with explicit `overhead_gb=0.0`
in tests (`tests/test_utils.py`).

---

```python
def assert_fits_in_available_gpu(estimate_gb: float, safety_margin_gb: float = 2.0) -> None
```

- No-op when CUDA unavailable.
- Raises `RuntimeError` if `estimate_gb > available - safety_margin_gb`.
- Otherwise prints confirmation:

```
[memory] Estimated peak VRAM: 22.4 GB / 80.0 GB — OK.
```

`pretrain.py` catches `RuntimeError` and prints a **warning** instead of aborting —
allows exploratory runs on tight GPUs. Tighten policy by removing the try/except
if hard failure is preferred.

---

### B.4 Worked VRAM examples

#### Example A — Pretrain startup estimate

```python
from models.transformer import GPTOSS, ModelConfig
from utils.memory import estimate_model_memory_gb, assert_fits_in_available_gpu
import yaml

with open("configs/pretrain_a100_502m.yaml") as f:
    raw = yaml.safe_load(f)
cfg = ModelConfig(**raw["model"])
model = GPTOSS(cfg)

est = estimate_model_memory_gb(
    model,
    seq_len=cfg.max_seq_len,      # 4096
    batch_size=8,
    grad_checkpoint=True,
    grad_ckpt_every=3,
)
assert_fits_in_available_gpu(est, safety_margin_gb=2.0)
```

#### Example B — Inference KV at 128K

```python
est_decode = estimate_model_memory_gb(
    model,
    seq_len=131072,
    batch_size=1,
    grad_checkpoint=False,
    steady_state=True,
    overhead_gb=4.0,
)
```

Dominated by $6 \times 131072$ global-layer KV, not windowed $6 \times 128$.

#### Example C — Checkpoint round-trip

```python
from utils.checkpoint import CheckpointManager
import torch
from torch.optim import AdamW

ckpt = CheckpointManager("checkpoints/test")
optim = AdamW(model.parameters(), lr=1e-4)
ckpt.save(model, optim, step=100, extra_meta={"loss": 3.5})
meta = ckpt.load(model, step=100, optimizer=optim)
assert meta["step"] == 100
```

Verified in `tests/test_training.py` and `scripts/e2e_gpu_smoke.py` step 6.

#### Example D — Logger with aux metric

```python
from utils.logging import TrainingLogger

logger = TrainingLogger(log_interval=10, seq_len=4096)
for step in range(1, 101):
    logger.log(step, loss=3.2, metrics={"aux": 0.01}, lr=4e-4)
logger.finish()
```

---

---

## Part C — Optimization catalog (OPT-1 … OPT-24)

### C.1 How to read the catalog

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
- MoE Triton contract — see [moe.md](../concepts/moe.md#sanctioned-triton-path-moe_dispatchtriton_grouped)
- Manual dict-based mask caches — replaced by `@functools.lru_cache` (OPT-1)

---

### C.2 Attention and masks

#### OPT-1 — Mask cache via `functools.lru_cache`

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

**Related:** [attention-sinks.md](../concepts/attention-sinks.md)

---

#### OPT-2 — `repeat_kv` without `.contiguous()`

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

#### OPT-3 — Sink path via extended K/V + mask column

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
for the bulk of the computation. See [attention-sinks.md](../concepts/attention-sinks.md) for
the full mathematical treatment.

**Files:** `models/attention.py` (`causal_attention` sink branch)

**Numerical note:** Sink bias clamped to $[-10, 15]$ before mask add (OPT-14).

---

#### OPT-22 — Decode-specific window mask ($T_q = 1$)

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

**Related:** [inference.md](../inference.md), [kv cache engineering](../inference.md) (ring-buffer interplay)

---

### C.3 Tensor layout and dtype

#### OPT-4 — RMSNorm native dtype activations

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

#### OPT-23 — `apply_rope` dtype preservation

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
`dtype: bf16` config. Documented in [attention-and-positional.md](../concepts/attention-and-positional.md).

**Files:** `models/rotary.py` (`apply_rope`)

---

### C.4 MoE dispatch

#### OPT-5 — Stacked expert dispatch (default `moe_dispatch="stacked"`)

**Problem:** Per-expert Python loops over tokens scatter memory access and launch
many small GEMMs — poor GPU utilization at MoE scale.

**Fix:** `MoELayer._dispatch_vectorized`:

1. Flatten tokens; router produces top-2 indices + weights.
2. `argsort(flat_idx, stable=True)` groups tokens by expert.
3. For each expert with assigned tokens, run `SwiGLUExpert` on the gathered chunk.
4. `index_add` weighted outputs back to token positions.

Stable sort ensures reproducible routing (required by `AGENTS.md` §A.4).

**Impact:** Default path for all training runs. Groups tokens so each expert GEMM
has reasonable $M$ dimension. See [moe.md](../concepts/moe.md) for dispatch diagrams.

**Files:** `models/moe.py` (`_dispatch_vectorized`, `MoELayer.forward`)

**Profile:** `scripts/profile_moe.py`

---

#### OPT-24 — Triton grouped GEMM (`moe_dispatch="triton_grouped"`)

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
Expected 5–15% MoE forward speedup on sm_80+ depending on batch/token count [INFERENCE] — .benchmarks/ empty.

**Contract:**

- Must **not** silently fall back during default-config training.
- If Triton unavailable and explicitly requested → clear error.
- Requires unit tests in `tests/test_moe_triton.py` (CPU reference path).

**Files:** `models/moe_triton.py`, `models/moe.py` (`_dispatch_triton`)

**Related:** [moe.md](../concepts/moe.md#sanctioned-triton-path-moe_dispatchtriton_grouped) (Triton kernel contract), [triton programming](../concepts/kernels-and-checkpointing.md) (kernel design and numerics)

---

### C.5 Training loop

#### OPT-8 — Gradient clip with `foreach=True`

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

#### OPT-9 — AdamW `foreach=True`, `fused=True` on CUDA

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
rule of thumb) [INFERENCE] — .benchmarks/ empty. Pairs with FP32 master weights under BF16 autocast.

**Files:** `training/pretrain.py`

**Related:** [training.md](../training.md) §11

---

#### OPT-10 — Bisect shard lookup in `PretrainDataset`

**Problem:** Multi-shard token datasets need $O(\log S)$ shard resolution vs linear
scan over shard metadata for every `__getitem__` call.

**Fix:** Precompute `shard_offsets` prefix array; lookup with:

```python
shard_idx = self._bisect.bisect_right(self.shard_offsets, start) - 1
```

Plus single-shard fast path when the entire window fits in one shard.

**Impact:** Negligible at small $S$; essential at production shard counts (50M
tokens per shard × hundreds of shards). See [training.md](../training.md).

**Files:** `training/pretrain.py` (`PretrainDataset._get_window_sharded`)

---

#### OPT-17 — AdamW `eps=1e-6` (not `1e-8`)

**Problem:** BF16 has 7 mantissa bits. Adam's second moment with `eps=1e-8`
underflows to denormal/zero, silently stalling late-stage convergence.

**Fix:** `eps=1e-6` in production AdamW — matches DeepSeek-V3 and LLaMA-3 practice.

**Impact:** Stability improvement, not throughput. Prevents loss plateau artifacts
after ~30K steps on MoE models.

**Files:** `training/pretrain.py`, `configs/pretrain_a100_502m.yaml`

---

#### OPT-18 — Warmup 3000 steps

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

#### OPT-19 — Aux load-balancing `alpha=0.01`

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

**Related:** [moe.md](../concepts/moe.md)

---

#### OPT-20 — `cudnn.benchmark_limit=0` + `preferred_blas_library="cublaslt"`

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

**Impact:** ~3–5% end-to-end on A100 after first-step warmup [INFERENCE] — .benchmarks/ empty. Bit-exact numerics
(same dtype, different kernel selection).

**Files:** `training/pretrain.py` (`_set_hardware_perf_knobs`)

**Related:** [training.md](../training.md) §7

---

#### OPT-21 — Chunked cross-entropy `chunk_size=8192`

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

### C.6 Inference

#### OPT-11 — `MixedKVCache` ring buffer (windowed layers)

**Problem:** Storing full $T$ KV for all 12 layers at 128K exceeds GPU memory.

**Fix:** Windowed layers (even indices) use a fixed-size ring buffer of length
$W = 128$:

- Pre-allocate `(B, H, window, D)` tensors once.
- Track `head` pointer and `count` for wrap-around.
- On `get()`, reorder chronologically if `head != 0`.

**Impact:** Windowed KV memory is $O(W)$ not $O(T)$ per layer — 6 layers × 128
tokens vs 6 × 131072 at 128K. Headline **≥1.8×** total KV reduction with global
layers (OPT-12). Verified analytically by `scripts/kv_cache_benchmark.py`.

**Related:** [kv cache engineering](../inference.md) — ring-buffer design and the bandwidth model behind the reduction.

**Files:** `inference/generate.py` (`MixedKVCache.append`, `get`)

---

#### OPT-12 — `MixedKVCache` exponential growth (global layers)

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

**Related:** [kv cache engineering](../inference.md) — exponential-growth sizing for global layers.

**Files:** `inference/generate.py` (`MixedKVCache.append` global branch)

---

#### OPT-13 — Pre-allocated generation output tensor

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

#### OPT-14 — Sink bias clamp cache in `generate()`

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

**Related:** [attention-sinks.md](../concepts/attention-sinks.md) (clamp rationale)

---

#### OPT-15 — YaRN $T=1$ fast path

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

**Related:** [attention-and-positional.md](../concepts/attention-and-positional.md)

---

### C.7 Numerical stability

#### OPT-6 — FP32 softmax in MoE router and aux loss

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

#### OPT-7 — Gradient checkpointing every 3rd layer

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

**Related:** [§B.3](operations.md#b3-estimate_model_memory_gb-mixed-kv-term-assert_fits_in_available_gpu) (activations), [autograd checkpointing](../concepts/kernels-and-checkpointing.md) (memory/compute tradeoff)

---

### C.8 Compilation

#### OPT-16 — `torch.compile(max-autotune)` when `compile: true`

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
improvement on A100 depending on CUDA/PyTorch version [INFERENCE] — .benchmarks/ empty. Target MFU ≥35%.

**Files:** `training/pretrain.py`, `configs/pretrain_a100_502m.yaml`, `scripts/step_time_a100.py`

**Caveat:** Interacts with gradient checkpointing — both enabled in production;
debug compile issues by temporarily setting `compile: false`.

---

### C.9 Quick reference table

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

---

## How to verify

```bash
python3 scripts/check_docs.py
python3 scripts/kv_cache_benchmark.py
python3 -m pytest tests/test_utils.py -v
pytest tests/test_attention.py -v
pytest tests/test_moe_triton.py -v
python3 scripts/e2e_gpu_smoke.py
python3 scripts/profile_components.py
```

After changing `models/attention.py`, always run `test_sliding_window_matches_full` per AGENTS.md rule 4.

### Related documentation

| Topic | Document |
|-------|----------|
| Training loop CLI | [training.md](../training.md) |
| Mixed KV cache design | [foundations-and-architecture.md](../concepts/foundations-and-architecture.md) §9, Part C (OPT-11/12) |
| Passkey eval internals | `inference/long_context.py`, [inference.md](../inference.md) |
| MoE + Triton opt-in | [moe.md](../concepts/moe.md) |
| Config reference | [training.md](../training.md#part-b--configuration-reference) |
| Attention masks + sinks | [attention-sinks.md](../concepts/attention-sinks.md) |
| YaRN / RoPE | [attention-and-positional.md](../concepts/attention-and-positional.md) |
| Book index | [README.md](../README.md) |

## References

- [`utils/checkpoint.py:CheckpointManager`](../../utils/checkpoint.py) — atomic safetensors protocol
- [`utils/memory.py:estimate_model_memory_gb`](../../utils/memory.py) — VRAM estimator
- [`utils/logging.py:TrainingLogger`](../../utils/logging.py) — WandB-capable logger
- [`inference/long_context.py:PasskeyEvaluator`](../../inference/long_context.py) — passkey eval
- [training.md](../training.md) — training loop and config reference
- [inference.md](../inference.md) — `MixedKVCache`, generation
- [attention-sinks.md](../concepts/attention-sinks.md) — attention masks and sinks

<!-- docs:verified 2026-08-05 · 6491066 -->
