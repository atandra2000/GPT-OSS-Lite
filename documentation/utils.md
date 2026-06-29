# Utils — GPT-OSS-Lite

> **Source:** `utils/checkpoint.py`, `utils/distributed.py`,
> `utils/logging.py`, `utils/memory.py`

## `checkpoint.CheckpointManager`

Atomic safetensors checkpoint manager with shared-tensor dedup and
step discovery. Files per step:

- `model_step_N.safetensors` — weights (deduped by `data_ptr()` so
  weight-tied params are stored once; duplicates are cloned to a fresh
  contiguous tensor to avoid safetensors rejecting shared storage).
- `optim_step_N.pt` — optimiser state (torch.save).
- `sched_step_N.pt` — scheduler state (optional, when a scheduler is
  passed).
- `meta_step_N.json` — step metadata (incl. `aux_loss`, `seed`, etc.).

All writes go via `tempfile.mkstemp` + `os.replace` so a crash mid-write
leaves either the old file or the new one — never a half-written one.
This is the same pattern used by `data/common.py::atomic_write_bytes`
and is critical for the NaN-guard in `training/pretrain.py` to roll back
to a known-good state.

### RNG state sibling file

`training/pretrain.py` saves RNG state to a sibling `rng_step_N.pt` file
alongside each checkpoint (`{python, numpy, torch, cuda}` states). On
`--resume-from N`, the script restores the RNG state from that file so
the resumed run is bit-identical to a non-interrupted run. The
CheckpointManager itself does NOT manage RNG state — the training script
owns it (see `documentation/training.md`).

`latest_step()` returns the highest step for which all three of
`model_step_N.safetensors`, `optim_step_N.pt`, and `meta_step_N.json`
exist (a checkpoint is "complete" only when all three are present).

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