# MoE Triton kernel — design + implementation

> **Source:** `models/moe_triton.py`
> **Sanctioned:** Yes (first sanctioned kernel in the project).
> **Opt-in via:** `moe_dispatch="triton_grouped"` in `ModelConfig`.

This is the design document for the only sanctioned Triton kernel in
GPT-OSS-Lite. It supplements [`moe.md`](moe.md) §11 (the dispatch-level
description) with the kernel-level detail: the algorithm, the launch
geometry, the numerical choices, and the test surface.

## 1. Scope

We fuse the **W1·x (gating) + W3·x (up-projection) + silu(g) * u** chain
into a single Triton kernel. W2 (down-projection) is a plain matmul in
PyTorch because it has no activation to fuse and the ragged token counts
after the sort are awkward to express in Triton.

```python
# What the kernel computes:
gate = W1[e] @ x_chunk          # (chunk, d_model) × (d_model, d_ff) → (chunk, d_ff)
up   = W3[e] @ x_chunk          # (chunk, d_model) × (d_model, d_ff) → (chunk, d_ff)
out[start:end] = silu(gate) * up  # silu is x * sigmoid(x); fused here
```

The reference path (`_moe_w1w3_silu_reference` in the same file) computes
the same thing in pure PyTorch for testing.

## 2. Algorithm

### 2.1 Inputs

- `x_sorted: (N, d_model)` — input tokens, sorted by expert assignment.
- `expert_ids_sorted: (N,)` — integer expert id for each row of `x_sorted`.
- `counts: (E,)` — number of rows per expert.
- `offsets: (E,)` — starting row index per expert.
- `W1_stack: (E, d_ff, d_model)` — gated-projection weights.
- `W3_stack: (E, d_ff, d_model)` — up-projection weights.

### 2.2 Output

- `gated_sorted: (N, d_ff)` — `silu(W1[e] @ x_chunk) * (W3[e] @ x_chunk)`.

### 2.3 Launch geometry

```
grid = (n_experts, ceil(max_per_expert_tokens / BLOCK_T), ceil(d_ff / BLOCK_N))
```

The third grid dim parallelises over the d_ff output axis. A single
program computes a `(BLOCK_T, BLOCK_N)` tile of the output for one expert.

### 2.4 Inner loop

For each program, the K-loop accumulates the W1·x and W3·x products
in **fp32** over `d_model` chunks of size `BLOCK_M = 32`:

```python
g_acc = tl.zeros((BLOCK_T, BLOCK_N), dtype=tl.float32)
u_acc = tl.zeros((BLOCK_T, BLOCK_N), dtype=tl.float32)
for k0 in range(0, d_model, BLOCK_M):
    k_offsets = k0 + tl.arange(0, BLOCK_M)
    k_mask = k_offsets < d_model
    x_tile = tl.load(x_row_base + k_offsets[None, :] * stride_xd,
                     mask=tok_mask & k_mask, other=0.0)
    w1_tile = tl.load(w1_row_base + k_offsets[None, :] * stride_w1d,
                      mask=n_mask & k_mask, other=0.0)
    w3_tile = tl.load(w3_row_base + k_offsets[None, :] * stride_w3d,
                      mask=n_mask & k_mask, other=0.0)
    g_acc += tl.dot(x_tile, tl.trans(w1_tile), allow_tf32=False)
    u_acc += tl.dot(x_tile, tl.trans(w3_tile), allow_tf32=False)

# silu in fp32 (Triton's tl.sigmoid requires fp32/fp64).
silu = g_acc * tl.sigmoid(g_acc)
fused = (silu * u_acc).to(out_ptr.dtype.element_ty)
tl.store(out_row, fused, mask=tok_mask & n_mask)
```

The `allow_tf32=False` is deliberate: BF16 inputs would otherwise
down-convert to TF32 with a precision loss larger than the BF16 ULP we're
trying to preserve.

## 3. Numerical choices

| Decision | Why |
|---|---|
| fp32 accumulator in the K-loop | The product of two BF16 numbers summed K times would round to BF16 at the end; fp32 keeps the accumulation tight. The output is cast to the input dtype on store, so memory traffic stays the same. |
| silu in fp32 (not BF16) | `tl.sigmoid` in Triton 3.7 accepts only fp32/fp64 inputs. Computing silu in fp32 is also strictly more accurate. |
| `allow_tf32=False` on `tl.dot` | TF32 has 10 mantissa bits — wider than BF16's 7. With `allow_tf32=True` the matmul silently rounds to a different format than the rest of the pipeline expects, producing BF16 ULP-level differences that complicate the cross-check against the reference. |
| `BLOCK_T=16, BLOCK_M=32, BLOCK_N=32` | Tile size chosen to fit in 64 KB shared memory on sm_75 (GTX 1650): 16 × 32 × 4 bytes (input) + 2 × 32 × 32 × 4 bytes (W tiles) ≈ 10 KB. Plenty of headroom. `num_stages=1` (no pipelining) keeps the live set small. |
| `num_stages=1` | With `num_stages=2` the kernel spills shared memory on sm_75 (`OutOfResources: shared memory`). On sm_80+ with 164 KB shared, `num_stages=2` can be re-enabled by editing the launcher. |

## 4. Backward (autograd)

The forward is on Triton. The backward uses the **pure-PyTorch reference
path** (`_moe_w1w3_silu_reference`) inside `torch.autograd.grad`. This is
intentional: the forward is the hot path (one fused launch vs two separate
matmuls), but the backward is two matmuls per expert where the Triton
gain is small relative to autograd setup cost, and writing a backward
that matches the fp32-accumulating forward in Triton is a non-trivial
extra surface area.

This is the same `forward-Triton, backward-reference` split used by
many other first-party kernels (vLLM's `fused_moe`, FasterTransformer's
fused attention). The gradient is bit-equivalent to a fully-PyTorch
forward+backward because the reference path is bit-equivalent to the
forward for inputs in the supported range.

## 5. Hard caps

| Variable | Hard cap | Rationale |
|---|---|---|
| `d_ff` | 8192 | `BLOCK_N` is at most 8192 (`triton.next_power_of_2(d_ff)`), and the W1/W3 tile size is `BLOCK_N × BLOCK_M` which would overflow 64 KB shared memory above this. |
| `d_model` | 8192 | Same reasoning for the K-loop input tile. |

The kernel raises `ValueError` (not silent fallback) if either is
exceeded, per AGENTS.md rule #8: *"Never let a Triton kernel silently
fall back to the raw-PyTorch path during a default-config training run."*

## 6. When to use

- **Production A100 (sm_80, 164 KB shared)**: enable via
  `moe_dispatch="triton_grouped"`. The launch-count saving is the
  primary win. Optional: bump `num_stages=2` in the launcher.
- **Dev box (sm_75, 64 KB shared)**: enable. The kernel runs at
  `num_stages=1`. Speedup is smaller than on A100 but the launch-count
  reduction still helps, and it verifies the path works end-to-end on
  small hardware before the production run.
- **CPU / Mac**: keep the default `"stacked"` path. The Triton import
  is gated on `try/except ImportError`; `MoELayer._dispatch_triton`
  raises `ImportError` if `HAS_TRITON` is False.

## 7. Test surface

`tests/test_moe_triton.py` covers:

- **Reference cross-checks** (always run, CPU-runnable):
  - `test_reference_matches_naive_per_expert_loop` — reference matches a
    hand-rolled per-expert matmul.
  - `test_reference_handles_empty_experts` — experts with `count=0`
    produce a well-defined output row.
  - `test_reference_matches_existing_moe_dispatch_shape` — output shape
    matches what `_dispatch_triton` expects.
  - `test_triton_moe_raises_when_triton_missing` — public function
    raises `ImportError` when `HAS_TRITON=False` (monkeypatched).
  - `test_triton_moe_raises_on_hard_cap_violation` — `ValueError` on
    `d_ff > 8192` or `d_model > 8192`.
  - `test_MoELayer_default_moe_dispatch_is_stacked` — default
    dispatch is `"stacked"` (no silent Triton opt-in).
  - `test_MoELayer_triton_dispatch_raises_when_triton_missing` —
    `MoELayer` with `moe_dispatch="triton_grouped"` raises
    `ImportError` when `HAS_TRITON=False` (monkeypatched).
- **GPU-gated tests** (`@gpu_required`, skip on CPU-only machines):
  - `test_kernel_forward_matches_reference_fp32` — kernel matches
    reference within `atol=rtol=1e-3` at FP32.
  - `test_kernel_forward_matches_reference_bf16` — kernel matches
    reference within `atol=rtol=2e-2` at BF16.

`scripts/e2e_gpu_smoke.py` adds end-to-end coverage:
- Step 3: same FP32 + BF16 cross-checks but on `n_tokens=128,
  d_model=128, d_ff=256` (closer to production ratio).
- Step 4: full `MoELayer` forward with `moe_dispatch="triton_grouped"`
  vs `moe_dispatch="stacked"`, weights shared via `load_state_dict`.
  Confirms the integration with the rest of the model.

## 8. Future work

- **Triton W2 kernel**: a second kernel that handles W2 as a grouped
  GEMM would eliminate the remaining per-expert Python loop. Deferred
  until a profile shows the W2 stage is the bottleneck.
- **Backward Triton kernel**: would speed up training. The current
  pure-PyTorch backward is correct but slow; a Triton backward that
  matches the fp32-accumulating forward would close the gap.
- **`num_stages=2` on sm_80+**: trivial change to the launcher once
  the project runs on production hardware.
- **Larger tile sizes on A100**: `BLOCK_T=32, BLOCK_M=64, BLOCK_N=64`
  fits comfortably in 164 KB shared; would give ~2× the matmul
  throughput per program.
