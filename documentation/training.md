# Training — GPT-OSS-Lite

> **Source:** `training/pretrain.py` · **Config:** `configs/pretrain_a100_502m.yaml`

## Overview

The pretraining script trains the 502M-param (247M active) GPT-OSS-Lite
model on a Chinchilla-optimal 8B-token corpus on a single A100 80GB in
~16–20 hours. Reproducibility is opt-in via `--seed N`.

## Numerical-stability & performance knobs

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