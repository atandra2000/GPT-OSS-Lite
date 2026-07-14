# Utils — GPT-OSS-Lite

> **Source:** `utils/checkpoint.py`, `utils/distributed.py`, `utils/logging.py`,
> `utils/memory.py`
> **Companion:** [`training.md`](training.md) (how these are wired into the loop),
> [`data_pipeline.md`](data_pipeline.md) (atomic writes, the same pattern).

---

## 1. Overview

The `utils/` package holds the four pieces of infrastructure that the training
loop *and* the inference path lean on, none of which are model-specific:

1. **`CheckpointManager`** — atomic safetensors checkpointing with
   shared-tensor dedup and step-discovery. The reliability anchor: a crash
   never produces a half-written checkpoint, so the NaN-guard rollback always
   lands on a known-good state.
2. **`distributed`** — a one-line device helper. Intentionally minimal; this
   repo is single-GPU.
3. **`TrainingLogger`** — a rolling-window logger with optional WandB. Step-driven,
   never forces a CPU/GPU sync per step.
4. **`memory`** — VRAM budgeting for the mixed windowed/global KV cache.
   Pre-flight check that aborts *before* a 16-hour run, not at step 5 000.

The design philosophy across all four: **fail loudly, never silently**, and
**never block the training hot path**.

---

## 2. `checkpoint.CheckpointManager` — atomic safetensors

```python
class CheckpointManager:
    def __init__(self, save_dir: str)
    def save(self, model, optimizer, step, scheduler=None, extra_meta=None, state_dict=None)
    def load(self, model, step, device="cuda", optimizer=None, scheduler=None, strict=True) -> dict
    def latest_step(self) -> Optional[int]
    def list_checkpoints(self) -> list
    def delete_checkpoint(self, step) / keep_last_n(self, n)
```

### 2.1 Files per step

A single logical checkpoint is **four physical files** in `save_dir`:

| File | Contents | Writer |
|---|---|---|
| `model_step_N.safetensors` | weights (deduped by `data_ptr()`) | `safetensors.save_file` |
| `optim_step_N.pt` | optimiser state (AdamW m, v, step) | `torch.save` |
| `sched_step_N.pt` | scheduler state (optional) | `torch.save` |
| `meta_step_N.json` | step + extra metadata (`aux_loss`, `seed`, …) | `json.dump` |

The split matters: safetensors for the *weights* (portable, no arbitrary-code
execution, mmap-friendly) and `torch.save` for the *optimiser/scheduler* (which
have Python-object state that safetensors cannot represent). A checkpoint is
"complete" only when all three of `model_step_N.safetensors`,
`optim_step_N.pt`, and `meta_step_N.json` exist — `latest_step()` and
`list_checkpoints()` enforce this.

### 2.2 Atomic writes — the `.tmp` → `os.replace` pattern

```python
def _atomic_write(self, path, writer, *, suffix):
    fd, tmp = tempfile.mkstemp(dir=self.save_dir, suffix=suffix)
    os.close(fd)
    try:
        writer(tmp)
        os.replace(tmp, path)        # atomic on POSIX, same filesystem
    except Exception:
        try: os.unlink(tmp)
        except OSError: pass
        raise
```

Every file is written to a `tempfile.mkstemp` temp path in the *same directory*
(atomic `os.replace` requires same-filesystem), then `os.replace`d into place.
`os.replace` is atomic on POSIX — at every instant, the path either points to
the old file or the new file, never a half-written one. On any exception the
temp is unlinked and the exception re-raised. This is the **same pattern used by
`data/common.py::atomic_write_bytes`** for the data shards, and it is the
reason the NaN-guard in [`training.md`](training.md) §5 can trust its rollback
target.

### 2.3 Shared-tensor dedup (weight tying)

```python
def _atomic_save_safetensors(self, state, path):
    seen_ptrs: set = set()
    deduped: dict = {}
    for k, v in state.items():
        ptr = v.data_ptr()
        if ptr in seen_ptrs:
            deduped[k] = v.contiguous().clone()      # break the share
        else:
            seen_ptrs.add(ptr)
            deduped[k] = v.contiguous()
    ...
```

GPT-OSS-Lite uses **weight tying**: the output `head.weight` *is*
`embed.weight` (the same tensor, same `data_ptr()`). safetensors refuses to
save two tensors sharing storage — it would double-count and corrupt on load.
The dedup walks `state_dict()`, tracks `data_ptr()`s, and for any duplicate
*clones to a fresh contiguous tensor* so the tied weight is stored exactly
once (under its first-seen key) and the duplicate gets an independent copy.
This preserves the tie on load (`model.load_state_dict` re-shares them) while
keeping safetensors happy. Without the `.clone()`, safetensors would reject the
save; without the dedup entirely, the file would be silently corrupt.

### 2.4 Strict vs non-strict load

`load(..., strict=True)` raises on any missing/unexpected keys (the safe
default); `strict=False` warns and continues (used by `--resume-from` into a
slightly-changed model). The warning prints up to 5 keys with a `…` truncator
so a large mismatch does not flood the log.

### 2.5 Step discovery and `keep_last_n`

- `_list_steps()` globs `model_step_*.safetensors` and parses the integer step
  from each stem — robust to non-numeric files in the dir.
- `latest_step()` returns the highest step whose checkpoint is *complete*
  (all three core files present) — a half-written step (only `model` saved
  before a crash) is skipped.
- `keep_last_n(n)` deletes all complete checkpoints except the newest `n` —
  disk-budget management over a 61 000-step run.

### 2.6 The RNG sibling file (owned by the training script, not the manager)

`training/pretrain.py` saves RNG state to a sibling `rng_step_N.pt` file
alongside each checkpoint (`{python, numpy, torch, cuda}` states). On
`--resume-from N`, the script restores the RNG state from that file so the
resumed run is bit-identical to a non-interrupted run. **The `CheckpointManager`
itself does NOT manage RNG state** — the training script owns it, because RNG
state is a *training-loop* concern (it must be captured at the right point in
the loop), not a model-serialisation concern. See [`training.md`](training.md)
§6.

---

## 3. `distributed` — the one-line device helper

```python
DEVICE: torch.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
def device() -> torch.device: return DEVICE
```

`DEVICE` is evaluated **once at import time** and cached; `device()` returns
it. This is deliberately the entire "distributed" module — this repo is
single-GPU, and a fake DDP wrapper would be exactly the kind of speculative
abstraction that does not belong here. If multi-GPU is ever added, the right
place is to grow this module, not to sprinkle `if dist.is_initialized()`
checks through the training loop.

---

## 4. `logging.TrainingLogger` — step-driven, sync-free

```python
class TrainingLogger:
    def __init__(self, log_interval=10, seq_len=1024)
    def log(self, step, loss, metrics=None, lr=0.0)
    def finish(self)
```

### 4.1 The rolling window

`log()` appends `loss` to `_loss_window` on every call. Every `log_interval`
steps, it computes the window average, the elapsed wall time, the derived
metrics, prints one line, and clears the window:

```
step= 12345 | loss=2.8421 | ppl=17.15 | lr=3.21e-04 | tps=12,345 | aux=0.0123
```

The key design choice: **the logger is called every step but only prints every
`log_interval` steps**. The per-step append is a Python-list `append` — no
`.item()` sync, no print. This avoids the two things that wreck training
throughput if done naively:
- A per-step `print` (I/O syscall every step).
- A per-step `.item()` on the loss tensor (forces a CPU/GPU sync — the GPU
  stalls waiting for the scalar transfer). The training loop calls
  `ce.item()` / `aux_loss.item()` *only at log boundaries*, not per step.

### 4.2 Derived metrics

- `ppl = exp(avg_loss)` — perplexity, the interpretable LM metric.
- `tps = (log_interval · seq_len) / elapsed` — tokens/sec from the window's
  wall time. People-sec is a throughput number, not a quality number; it
  catches a silently-slowing run (e.g. KV-cache pathology) early.

### 4.3 Optional WandB

```python
wandb_project = os.environ.get("WANDB_PROJECT")
if wandb_project:
    import wandb; wandb.init(project=wandb_project, name=..., reinit=True)
    self._wandb = wandb
```

WandB is **opt-in via the `WANDB_PROJECT` env var** — no code change, no
hard dependency (`wandb` is imported lazily; a missing import prints a
skip-notice, not a crash). On each log boundary, the same metrics are
forwarded to WandB under `train/*` keys. `finish()` calls `wandb.finish()`.

---

## 5. `memory` — VRAM budgeting for the mixed KV cache

```python
def estimate_model_memory_gb(model, seq_len, batch_size, grad_checkpoint=True,
                             overhead_gb=None, steady_state=False,
                             grad_ckpt_every=3) -> float
def assert_fits_in_available_gpu(estimate_gb, safety_margin_gb=2.0) -> None
```

The estimator is what powers the **pre-flight VRAM check** in
`training.pretrain.main`: build the model, estimate peak VRAM, abort *before*
the loop if it would OOM. The estimate sums four terms:

### 5.1 Parameters

`_parameter_bytes(model)` — `Σ numel · element_size`. For BF16 params, 2
bytes/param; the 502 M model is ~1 GB.

### 5.2 Optimiser (FP32 master)

`_optimiser_bytes(model)` — `Σ numel · 12`. AdamW stores `m`, `v`, and the
FP32 master copy, all in FP32 = 12 bytes/param. This is the largest single
term for a 502 M model (~6 GB) and the reason "BF16 model, FP32 optimiser" is
the standard memory shape.

### 5.3 The mixed KV cache — two regimes

`_mixed_kv_cache_bytes(model, seq_len, batch_size, steady_state)`:

```
per_token      = 2 · n_kv_heads · head_dim · dtype_bytes        # K + V
windowed_layers = count of even layers
global_layers   = count of odd layers
if steady_state:                       # inference-time decode
    windowed_len = window              # windowed layers hold exactly window
else:                                  # training-time prefill peak
    windowed_len = max(window, seq_len)
windowed_bytes = windowed_layers · windowed_len · batch · per_token
global_bytes   = global_layers   · seq_len      · batch · per_token
```

The two regimes reflect the cache's actual behaviour (see
[`inference.md`](inference.md) §2):
- **Prefill / training peak** (`steady_state=False`): windowed layers hold
  `max(window, seq_len)` — during training (`seq_len = 4096 > window = 128`)
  the windowed layers are still filling and hold the full `seq_len`.
- **Steady-state decode** (`steady_state=True`): windowed layers hold exactly
  `window = 128`. This is the inference regime where the 2× savings show up.

Global layers always hold `seq_len` (they are unbounded by design).

### 5.4 Activations

`_activation_bytes(...)`:
- `hidden_bytes = n_layers · seq_len · batch · hidden_dim · dtype_bytes · store_factor`
- `moe_bytes    = n_layers · 3(active experts) · 3(SwiGLU matrices) · seq_len · batch · ffn_dim · dtype_bytes · store_factor`
- `attn_bytes   = (only for `attn_impl == "manual"`) n_full_layers · seq_len² · batch · n_heads · dtype_bytes · store_factor`

`store_factor` models gradient checkpointing:
```
ckpt_factor   = 1 / grad_ckpt_every                       # fully-ckpt fraction
store_factor  = ckpt_factor + (1 - ckpt_factor) · 0.5    # ckpt layers recompute (store ~0.5)
```
At `grad_ckpt_every = 3`, `store_factor ≈ 0.667` — about 2/3 of activations
are stored, matching the "checkpoint 4 of 12 layers" choice in
[`training.md`](training.md) §3.5. The manual-attention `seq_len²` term is
only counted for the `manual` path (the SDPA path does not materialise the
full score matrix) — another reason to never ship `attn_impl: "manual"`.

### 5.5 Overhead and the safety check

`_detect_overhead_gb()` returns ~17% of total GPU memory (capped at 13.7 GB,
floored at 2 GB on CPU) — a rough model of PyTorch's allocator fragmentation
and workspace. `estimate_model_memory_gb` adds this to the four terms above.

`assert_fits_in_available_gpu(est, safety_margin_gb=2.0)`:
- **No-op on CPU** — there is no GPU to fit in.
- On CUDA, raises `RuntimeError` if `est > total_memory − 2 GB`. The 2 GB
  margin is the safety slack for the allocator's own working set.

The training script wraps this in a `try/except RuntimeError` that *warns*
rather than aborts — a tight estimate that is slightly over is often fine in
practice (the estimate is conservative), so the user gets a loud warning but
can still proceed. The hard abort is reserved for a true overflow.

---

## 6. Design rationale & rejected alternatives

| Decision | Rationale | Rejected alternative |
|---|---|---|
| Atomic `.tmp → os.replace` | A crash leaves old-or-new, never half | Direct write — half-written checkpoint corrupts rollback |
| safetensors for weights, `torch.save` for optim | Portable + safe for weights; optim needs Python-object state | All-safetensors (cannot store optim state) / all-torch.save (arbitrary-code-exec risk) |
| Shared-tensor dedup by `data_ptr()` | Weight tying would corrupt safetensors | Skip dedup — save fails or corrupts |
| RNG sibling file owned by the training script | RNG capture timing is a loop concern | Put RNG in CheckpointManager — couples serialisation to loop state |
| `latest_step()` requires all 3 files | A half-written step is not a valid rollback target | Take the max step present — rollback into a corrupt checkpoint |
| Logger prints every `log_interval`, not every step | No per-step print, no per-step `.item()` sync | Print every step — I/O + GPU sync stalls the loop |
| WandB opt-in via env var, lazy import | No hard dep, no code change to enable | Hard `import wandb` — crashes on a CI box without it |
| Memory estimator two-regime KV | Matches the cache's actual prefill vs decode behaviour | One regime — over- or under-estimates one path |
| `assert_fits` raises *before* training | Fail at step 0, not step 5 000 of a 16-h run | Let it OOM — wastes hours |
| `distributed` is one line | Single-GPU repo; no fake DDP | A DDP wrapper with one rank — speculative abstraction |

---

## 7. Edge cases & pitfalls

- **Checkpoint completeness**: `latest_step()` skips any step missing one of
  the three core files. If you manually delete `optim_step_N.pt`, step `N`
  silently drops out of `list_checkpoints()` — intended, but worth knowing if
  you debug "where did my checkpoint go."
- **`keep_last_n` deletes the older half**: it operates on *complete*
  checkpoints only and keeps the newest `n`. A crashed run with many
  half-written steps is unaffected (those are already ignored).
- **`state_dict=` override in `save`**: pass a custom state dict to save a
  subset (e.g. for inference export). The dedup still applies to whatever you
  pass.
- **`overhead_gb` override**: `estimate_model_memory_gb(..., overhead_gb=0)`
  to get the pure model estimate without the 17% allocator fudge — useful for
  comparing against a profiled actual peak.
- **`steady_state` flag**: the default (`False`) gives the *training* peak
  (prefill). For inference budgeting pass `steady_state=True` to reflect the
  decode cache regime.
- **`attn_impl="manual"` blows up the activation estimate** via the `seq_len²`
  score term — this is a feature: it warns you the manual path is
  memory-hostile at long context, not a bug.

---

## Implementation notes (extracted from code review)

- **mmap zero-copy slices**: `PretrainDataset._load_shard` mmaps each shard
  via `torch.from_file(path, dtype=raw_dtype, shared=True, size=...)` (raw
  bytes layout) or `torch.load(path, mmap=True)` (legacy torch.save layout).
  Each `__getitem__` then returns a zero-copy slice of the mmap'd tensor —
  the OS pages in the requested range on demand and the page cache warms
  for repeat accesses. This is critical for the 8B-token corpus: we cannot
  afford to load 32 GB into RAM just to slice it.
- **DataLoader prefetch knobs**: `training/pretrain.py` constructs the
  DataLoader with `num_workers=4`, `pin_memory=True` (on CUDA), and
  `persistent_workers=(num_workers > 0)`. This enables async H2D transfer
  and keeps the worker processes alive across epochs (avoids the per-epoch
  re-import cost).
- **`uint32` storage**: the manifest's `dtype` field records the shard
  element dtype. GPT-OSS-Lite uses `uint32` (4 bytes/token) because the
  LLaMA-3 vocab is 128K (fits in uint16 numerically, but uint32 is the
  safe default for multi-trillion-token corpora with reserved special
  tokens up to vocab+256). `select_token_dtype(vocab_size)` picks the
  smallest dtype automatically.

## `distributed.device`

Single-GPU device helper. `device()` returns the cached
`torch.device("cuda:0" if torch.cuda.is_available() else "cpu")`. The
module-level `DEVICE` constant is evaluated at import time.

## `logging.TrainingLogger`

Step-driven logger that prints a rolling-window summary every
`log_interval` steps and optionally forwards to WandB (enabled via the
`WANDB_PROJECT` env var). The rolling window avoids per-step `print`
overhead and the `.item()` sync that would force a CPU/GPU sync every
step. People/sec is computed from the elapsed wall time over the window.

## `memory`

VRAM budgeting for the mixed windowed/global KV cache.

- `_mixed_kv_cache_bytes(model, seq_len, batch_size, steady_state)`:
  two regimes — prefill (windowed layers hold `max(window, seq_len)`;
  the training-time peak) and steady-state decode (windowed layers hold
  exactly `window`; the inference-time peak).
- `_activation_bytes(...)`: estimates hidden-state + MoE-intermediate +
  (manual-only) attention-score memory. Gradient checkpointing with
  `every=N` reduces activation memory by a factor interpolated between
  `1/N` (fully checkpointed) and `1.0` (no checkpointing) — at `every=3`
  ~2/3 of layers' activations are stored.
- `estimate_model_memory_gb(...)`: sums params + optimiser (FP32 m, v,
  master = 12 bytes/param) + KV cache + activations, plus an
  auto-detected CUDA overhead (~17% of total GPU memory, capped at
  13.7 GB).
- `assert_fits_in_available_gpu(est, safety_margin_gb=2.0)`: no-op on
  CPU; on CUDA raises `RuntimeError` if the estimate exceeds
  `total_memory - safety_margin_gb`.