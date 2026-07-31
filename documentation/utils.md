# Utilities — Checkpointing, Logging, and Memory Estimation

> **Chapter: production helpers.** This chapter documents `utils/checkpoint.py`,
> `utils/logging.py`, and `utils/memory.py` — the three modules that sit between
> the model code and the training loop. For how they are wired into pretraining,
> see [training.md](training.md) §22–24. For script-level VRAM checks see
> [scripts.md](scripts.md) §10.

---

## Table of contents

1. [Package overview](#1-package-overview)
2. [CheckpointManager](#2-checkpointmanager)
3. [Atomic write protocol](#3-atomic-write-protocol)
4. [Shared-tensor deduplication](#4-shared-tensor-deduplication)
5. [Checkpoint discovery and retention](#5-checkpoint-discovery-and-retention)
6. [TrainingLogger](#6-traininglogger)
7. [WandB integration](#7-wandb-integration)
8. [Memory estimation — `estimate_model_memory_gb`](#8-memory-estimation--estimate_model_memory_gb)
9. [Mixed KV-cache term](#9-mixed-kv-cache-term)
10. [Activation and optimizer terms](#10-activation-and-optimizer-terms)
11. [`assert_fits_in_available_gpu`](#11-assert_fits_in_available_gpu)
12. [Worked examples](#12-worked-examples)
13. [Related documentation](#13-related-documentation)

---

## 1. Package overview

```
utils/
├── __init__.py       # re-exports package docstring
├── checkpoint.py     # CheckpointManager — safetensors I/O
├── logging.py        # TrainingLogger — stdout + optional WandB
└── memory.py         # estimate_model_memory_gb, assert_fits_in_available_gpu
```

Design principles (from workspace `AGENTS.md`):

- **No pickle for model weights** — `safetensors` for tensors, `torch.save` for
  optimizer/scheduler state only.
- **Atomic writes** — write to temp file in `save_dir`, then `os.replace`.
- **Mixed KV accounting** — memory estimator reflects the alternating SWA/full
  architecture, not naive $12 \times T$ full attention.

---

## 2. CheckpointManager

**File:** `utils/checkpoint.py`

`CheckpointManager` is the single sanctioned API for saving and loading training
state. The training loop in `training/pretrain.py` constructs one instance per run:

```python
ckpt = CheckpointManager(train_cfg["save_dir"])
```

### File layout per step

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
them directly after the final save. See [training.md](training.md) §25.

### `save(model, optimizer, step, scheduler=None, extra_meta=None)`

Saves all three core files atomically. Logs:

```
[checkpoint] saved step 2000 → checkpoints/pretrain_a100
```

`extra_meta` merges into `meta_step_N.json` — e.g. `{"aux_loss": 0.0042}` or
`{"final": true, "seed": 42}`.

### `load(model, step, device="cuda", optimizer=None, scheduler=None, strict=True)`

1. Loads `model_step_N.safetensors` via `safetensors.torch.load_file`.
2. Calls `model.load_state_dict(weights, strict=False)` then validates keys.
3. Optionally restores optimizer and scheduler from `.pt` files.
4. Returns `meta` dict from JSON (or `{"step": step}` if meta missing).

**Strict mode:** If `strict=True` (default), missing or unexpected keys raise
`RuntimeError`. Set `strict=False` for partial loads during architecture experiments.

**Missing optimizer:** Warns and leaves optimizer at current state — useful when
fine-tuning with a new optimizer but warm-started weights.

### Complete checkpoint definition

A step is **complete** only when all three exist:

- `model_step_N.safetensors`
- `optim_step_N.pt`
- `meta_step_N.json`

`sched_step_N.pt` is optional. Incomplete steps are ignored by `latest_step()` and
`list_checkpoints()`.

---

## 3. Atomic write protocol

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

## 4. Shared-tensor deduplication

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

## 5. Checkpoint discovery and retention

### `latest_step() -> Optional[int]`

Returns the highest step number with a **complete** triple, or `None`.

Used by the NaN guard rollback in `pretrain.py`:

```python
latest = ckpt.latest_step()
if latest is not None:
    ckpt.load(model, step=latest, ...)
```

### `list_checkpoints() -> list[int]`

Sorted ascending list of complete steps.

### `delete_checkpoint(step: int)`

Removes all four filename patterns for that step (model, optim, sched, meta).

### `keep_last_n(n: int)`

Deletes all complete checkpoints except the `n` most recent. Example — retain
only the last 3 during long runs:

```python
ckpt.keep_last_n(3)
```

Call after `save()` on a schedule to cap disk usage. Not invoked automatically
by `pretrain.py` — add if checkpoint dir grows too large.

---

## 6. TrainingLogger

**File:** `utils/logging.py`

`TrainingLogger` prints a human-readable rolling summary every `log_interval`
steps and optionally forwards metrics to Weights & Biases.

### Construction

```python
logger = TrainingLogger(
    log_interval=train_cfg.get("log_interval", 50),
    seq_len=model_cfg.max_seq_len,
)
```

### `log(step, loss, metrics=None, lr=0.0)`

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

### Example stdout line

```
step=    200 | loss=3.2145 | ppl=24.89 | lr=2.67e-04 | tps=142,350 | aux=0.0031
```

**Note:** `loss` logged is **cross-entropy only** — `pretrain.py` passes
`ce.item()`, not `ce + aux_alpha * aux`. Aux appears under `metrics={"aux": ...}`.

### `finish()`

Calls `wandb.finish()` if WandB was initialized. Invoke at end of training.

---

## 7. WandB integration

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

### Logged scalars

| WandB key | Source |
|-----------|--------|
| `train/loss` | Rolling average CE |
| `train/ppl` | `exp(loss)` |
| `train/lr` | Scheduler LR |
| `train/tokens_per_sec` | Derived throughput |
| `train/<metric>` | Extra keys from `metrics` dict (e.g. `train/aux`) |

---

## 8. Memory estimation — `estimate_model_memory_gb`

**File:** `utils/memory.py`

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

### Total formula

$$
\text{VRAM}_{\text{GB}} = \frac{P + O + KV + A}{1024^3} + \text{overhead}_{\text{GB}}
$$

| Term | Symbol | Description |
|------|--------|-------------|
| Parameters | $P$ | All model weights in native dtype |
| Optimizer | $O$ | 12 bytes/param (FP32 m, v, master) |
| KV cache | $KV$ | Mixed windowed/global (§9) |
| Activations | $A$ | Layer activations with checkpoint factor (§10) |
| Overhead | — | CUDA allocator + cudnn workspace heuristic |

---

## 9. Mixed KV-cache term

The estimator counts **only KV tensors**, not Q — matching inference `MixedKVCache`
and the analytical benchmark in `scripts/kv_cache_benchmark.py`.

```python
per_token = 2 * n_kv_heads * head_dim * dtype_bytes   # K + V, BF16 → 2 bytes
n_windowed = sum(1 for b in model.blocks if b.attn.is_windowed)
n_global = n_layers - n_windowed
win_len = window if steady_state else max(window, seq_len)
kv_bytes = (n_windowed * win_len + n_global * seq_len) * batch_size * per_token
```

### Production numbers (`pretrain_a100_502m.yaml`)

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

## 10. Activation and optimizer terms

### Optimizer state

```python
optim_bytes = sum(p.numel() for p in model.parameters()) * 12
```

Assumes AdamW with FP32 master weights and moments (4 + 4 + 4 bytes per param).
Matches `eps=1e-6`, `foreach=True`, `fused=True` in [training.md](training.md) —
see OPT-9/17 in [OPTIMIZATIONS.md](OPTIMIZATIONS.md).

### Activations with gradient checkpointing

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
the ~30% compute overhead cited in [architecture.md](architecture.md).

### GPU overhead heuristic

```python
overhead_gb = min(13.7, max(2.0, total_gb * 0.17))
```

On CPU or no CUDA: `overhead_gb = 2.0`. Override with explicit `overhead_gb=0.0`
in tests (`tests/test_utils.py`).

---

## 11. `assert_fits_in_available_gpu`

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

## 12. Worked examples

### Example A — Pretrain startup estimate

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

### Example B — Inference KV at 128K

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

### Example C — Checkpoint round-trip

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

### Example D — Logger with aux metric

```python
from utils.logging import TrainingLogger

logger = TrainingLogger(log_interval=10, seq_len=4096)
for step in range(1, 101):
    logger.log(step, loss=3.2, metrics={"aux": 0.01}, lr=4e-4)
logger.finish()
```

---

## 13. Related documentation

| Topic | Document |
|-------|----------|
| Training loop wiring | [training.md](training.md) |
| KV cache architecture | [architecture.md](architecture.md) §9, [attention.md](attention.md) |
| KV benchmark script | [scripts.md](scripts.md) §4 |
| Optimization catalog | [OPTIMIZATIONS.md](OPTIMIZATIONS.md) |
| MoE memory heuristic | [moe.md](moe.md) |
| Config fields | [configs](configs.md), `configs/pretrain_a100_502m.yaml` |
| Book index | [README.md](README.md) |

---

<!-- docs:verified 2026-07-31 · fa6f918 -->
