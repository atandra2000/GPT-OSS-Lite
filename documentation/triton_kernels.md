# Triton Kernels — GPT-OSS-Lite

## Sanctioned Custom Kernel Contract

> **Scope:** This document covers the **one sanctioned Triton path** in
> GPT-OSS-Lite: fused grouped-GEMM for MoE W1/W3+silu in
> [`models/moe_triton.py`](../models/moe_triton.py).

> **Related:** [moe.md](moe.md) (MoE theory and dispatch),
> [`AGENTS.md`](../AGENTS.md) (Triton kernel contract §1),
> [`tests/test_moe_triton.py`](../tests/test_moe_triton.py).

---

## Table of Contents

1. [Abstract](#abstract)
2. [Why a Custom Kernel Here](#why-a-custom-kernel-here)
3. [Sanctioned Kernel Inventory](#sanctioned-kernel-inventory)
4. [Activation and Configuration](#activation-and-configuration)
5. [Public API — `triton_moe_w1w3_silu`](#public-api--triton_moe_w1w3_silu)
6. [HAS_TRITON Gate and Import Policy](#has_triton-gate-and-import-policy)
7. [Kernel Architecture — `_moe_w1w3_silu_kernel`](#kernel-architecture--_moe_w1w3_silu_kernel)
8. [Tiling Strategy](#tiling-strategy)
9. [Launcher Configuration](#launcher-configuration)
10. [PyTorch Reference Path](#pytorch-reference-path)
11. [Autograd — Forward Triton, Backward PyTorch](#autograd--forward-triton-backward-pytorch)
12. [Integration in `MoELayer._dispatch_triton`](#integration-in-moelayer_dispatch_triton)
13. [What Stays in PyTorch](#what-stays-in-pytorch)
14. [Hardware Notes — sm_75 vs sm_80](#hardware-notes--sm_75-vs-sm_80)
15. [Hard Caps and Shape Limits](#hard-caps-and-shape-limits)
16. [When to Enable `triton_grouped`](#when-to-enable-triton_grouped)
17. [Adding a New Sanctioned Kernel](#adding-a-new-sanctioned-kernel)
18. [Testing Contract](#testing-contract)
19. [Debugging Checklist](#debugging-checklist)
20. [Appendix A — Program ID layout](#appendix-a--program-id-layout)
21. [Appendix B — Memory traffic comparison](#appendix-b--memory-traffic-comparison)
22. [Appendix C — Glossary](#appendix-c--glossary)
23. [Load-Bearing Invariants](#load-bearing-invariants)
24. [References](#references)

---

## Abstract

GPT-OSS-Lite follows a **raw PyTorch first** policy: attention, YaRN RoPE,
RMSNorm, loss, and the default MoE dispatch path are pure PyTorch. One hot path
— the **grouped GEMM that fuses W1, W3, silu, and the gate×value multiply**
for routed MoE experts — has an optional Triton implementation.

The kernel is **opt-in** via `ModelConfig.moe_dispatch = "triton_grouped"`.
It is **not** enabled by default. If Triton is unavailable and
`triton_grouped` is requested, the code **raises `ImportError`** — there is
no silent fallback to PyTorch during a configured Triton run.

Backward passes use a **pure-PyTorch reference** recomputation inside
`torch.autograd.Function` — the Triton kernel is forward-only.

---

## Why a Custom Kernel Here

### The MoE dispatch bottleneck

In stacked dispatch (`moe_dispatch="stacked"`), each expert chunk runs:

```
g = W1 @ x
u = W3 @ x
h = silu(g) * u
out = W2 @ h
```

For top-2 of 8 experts, a batch of `B=8`, `T=4096` yields `N = 32,768` tokens
and `2 × 32,768 = 65,536` expert slots per layer. Even after sorting by
expert, each layer launches many small expert forwards.

The fused kernel combines **W1 matmul + W3 matmul + silu + multiply** into
one Triton program grid per (expert, token-tile, d_ff-tile). Benefits:

1. **Fewer kernel launches** — one grid replaces 4 PyTorch ops per chunk.
2. **Better memory locality** — `g` and `u` never materialised as full
   `(tokens, d_ff)` tensors in global memory.
3. **FP32 accumulation** on the matmul tiles before silu (matches reference
   numerics).

W2 is **not fused** — it is a separate down-projection with no following
nonlinearity to pair with in the same kernel pass.

### Why not fuse W2?

Down-projection maps `d_ff → d_model`. Fusing W2 would require either:

- A second kernel launch anyway (different output dimension tiling), or
- A much larger kernel with divergent tile shapes.

The current split (Triton up+gate, PyTorch W2) keeps the kernel simple and
still removes the dominant intermediate writes (`g`, `u`, `silu(g)`).

---

## Sanctioned Kernel Inventory

From [`AGENTS.md`](../AGENTS.md) §1 — **only these Triton paths are sanctioned:**

| File | Entry point | Fuses | Opt-in key |
|---|---|---|---|
| [`models/moe_triton.py`](../models/moe_triton.py) | `triton_moe_w1w3_silu` | W1, W3, silu, mul | `moe_dispatch="triton_grouped"` |

No other module in GPT-OSS-Lite should import Triton without updating
`AGENTS.md` and adding a `documentation/<name>.md` plan.

**Not sanctioned (stay PyTorch):**

- Attention (FA2 via `torch.nn.functional.scaled_dot_product_attention`)
- YaRN RoPE ([`models/yarn.py`](../models/yarn.py))
- RMSNorm, embeddings, LM head
- Chunked cross-entropy ([`training/pretrain.py`](../training/pretrain.py))
- MoE W2 down-projection
- Full stacked SwiGLU expert (default path)

---

## Activation and Configuration

### YAML config

```yaml
model:
  moe_dispatch: "triton_grouped"   # default is "stacked"
```

### `ModelConfig`

```python
@dataclass
class ModelConfig:
    moe_dispatch: str = "stacked"
```

[`MoELayer`](../models/moe.py) reads:

```python
self.moe_dispatch = getattr(cfg, "moe_dispatch", "stacked")
```

### Dispatch branch

```python
if self.moe_dispatch == "triton_grouped":
    out = self._dispatch_triton(flat, indices, weights)
else:
    out = self._dispatch_vectorized(flat, indices, weights)
```

There is **no** environment-variable gate for kernels. The only
switch is the explicit `moe_dispatch` config string.

---

## Public API — `triton_moe_w1w3_silu`

**Module:** [`models/moe_triton.py`](../models/moe_triton.py)

```python
def triton_moe_w1w3_silu(
    x_sorted: torch.Tensor,           # (n_slots, d_model)
    expert_ids_sorted: torch.Tensor,  # (n_slots,) — expert id per row
    counts: torch.Tensor,             # (n_experts,) — tokens per expert
    offsets: torch.Tensor,            # (n_experts,) — start index per expert
    W1_stack: torch.Tensor,           # (n_experts, d_ff, d_model)
    W3_stack: torch.Tensor,           # (n_experts, d_ff, d_model)
) -> torch.Tensor:                    # (n_slots, d_ff)
```

### Tensor contracts

| Tensor | Shape | Notes |
|---|---|---|
| `x_sorted` | `(N_slots, d_model)` | Tokens sorted by expert assignment |
| `expert_ids_sorted` | `(N_slots,)` | Redundant with sort order but passed for kernel indexing |
| `counts` | `(E,)` | `bincount` of expert assignments |
| `offsets` | `(E,)` | Cumulative start indices into sorted layout |
| `W1_stack` | `(E, d_ff, d_model)` | Stacked `expert.w1.weight` |
| `W3_stack` | `(E, d_ff, d_model)` | Stacked `expert.w3.weight` |
| **return** | `(N_slots, d_ff)` | `silu(W1@x) * (W3@x)` per row |

`N_slots = N_tokens × k` (e.g. 32,768 × 2 = 65,536 for default batch).

### Mathematical definition

For each row `r` assigned to expert `e`:

```
g_r = W1_e · x_r
u_r = W3_e · x_r
out_r = silu(g_r) ⊙ u_r
```

W2 is applied **outside** this function by [`MoELayer._dispatch_triton`](../models/moe.py).

---

## HAS_TRITON Gate and Import Policy

At module import:

```python
try:
    import triton
    import triton.language as tl
    HAS_TRITON = True
except ImportError:
    HAS_TRITON = False
```

### Behaviour matrix

| `HAS_TRITON` | `moe_dispatch` | Result |
|---|---|---|
| `False` | `"stacked"` | Normal PyTorch dispatch |
| `False` | `"triton_grouped"` | **`ImportError`** at first `triton_moe_w1w3_silu` call |
| `True` | `"stacked"` | Triton never imported by MoE forward |
| `True` | `"triton_grouped"` | Triton forward kernel runs |

### Error message (verbatim intent)

```python
raise ImportError(
    "triton_moe_w1w3_silu requires the `triton` package. "
    "Install with `pip install triton` (Linux + CUDA only). "
    "For CPU/Mac, use moe_dispatch='stacked' in your config."
)
```

**Critical rule** ([`AGENTS.md`](../AGENTS.md) rule 8): a default-config training
run must never silently fall back. Opt-in is per-kernel config key; failure
must surface clearly.

---

## Kernel Architecture — `_moe_w1w3_silu_kernel`

Triton JIT kernel in [`models/moe_triton.py`](../models/moe_triton.py).

### Program grid

**One program per** `(expert_id, token_tile, d_ff_tile)`:

```
program_id(0) = expert index e
program_id(1) = token block index t_blk
program_id(2) = d_ff block index n_blk
```

Early exit if `counts[e] == 0` (expert received no tokens this batch).

### Algorithm per program

1. Load token indices for this expert's slice: `off + tok_in_blk`.
2. Initialise `g_acc`, `u_acc` accumulators `(BLOCK_T, BLOCK_N)` in FP32.
3. **K-loop** over `d_model` in steps of `BLOCK_M`:
   - Load `x` tile `(BLOCK_T, BLOCK_M)`
   - Load `W1`, `W3` tiles `(BLOCK_N, BLOCK_M)`
   - `g_acc += dot(x_tile, trans(w1_tile))`
   - `u_acc += dot(x_tile, trans(w3_tile))`
4. `silu = g_acc * sigmoid(g_acc)` (FP32 — Triton sigmoid requirement)
5. `fused = silu * u_acc`, cast to output dtype
6. Store to `out` at matching token/ff positions

### TF32 policy

```python
tl.dot(x_tile, tl.trans(w1_tile), allow_tf32=False)
```

Matmul tiles disable TF32 inside the kernel for bit-closer agreement with
the FP64 reference in tests. Global TF32 (enabled in [`pretrain.py`](../training/pretrain.py))
still applies to PyTorch W2 matmuls outside the kernel.

---

## Tiling Strategy

### Tile dimensions (production defaults)

| Constant | Value | Role |
|---|---|---|
| `BLOCK_T` | **16** | Tokens per program along expert chunk |
| `BLOCK_M` | **32** | `d_model` reduction tile (K dimension) |
| `BLOCK_N` | **32** | `d_ff` output tile |

These are **hardcoded in `_MoEW1W3SiluFunction.forward`**, not autotuned at
runtime in v1.

### Grid sizing

```python
max_tokens = counts.max()
n_tiles_t = ceil(max_tokens / BLOCK_T)
n_tiles_n = ceil(d_ff / BLOCK_N)
grid = (n_experts, n_tiles_t, n_tiles_n)
```

**Note:** `n_tiles_t` uses the **maximum** expert count, not per-expert count.
Programs for shorter experts exit early via `tok_mask` — simple grid, some idle
programs when load is imbalanced.

### Grouped GEMM interpretation

Tokens assigned to the same expert are **contiguous** in `x_sorted` after
stable argsort. Each expert program reads a contiguous slice
`[offsets[e], offsets[e] + counts[e])` — classic grouped GEMM layout used in
Megatron-LM and vLLM MoE paths.

---

## Launcher Configuration

```python
_moe_w1w3_silu_kernel[(n_experts, n_tiles_t, n_tiles_n)](
    ...,
    BLOCK_T=16, BLOCK_M=32, BLOCK_N=32,
    N_EXPERTS=n_experts,
    num_warps=4,
    num_stages=1,
)
```

| Parameter | Value | Rationale |
|---|---|---|
| `num_warps` | 4 | Standard for 32×32 output tiles |
| `num_stages` | **1** | sm_75 (GTX 1650, 64 KB shared) — `num_stages=2` spills |
| | | sm_80 (A100, 164 KB) can use `num_stages=2` if launcher updated |

Verified end-to-end on **4 GB GPU (sm_75)** via
[`scripts/e2e_gpu_smoke.py`](../scripts/e2e_gpu_smoke.py).

### sm_80 tuning note

For A100 production, a future launcher change may re-enable `num_stages=2`
without changing the kernel body — only the launch kwargs in
`_MoEW1W3SiluFunction.forward`. Document any change in this file and
`AGENTS.md`.

---

## PyTorch Reference Path

### `_moe_w1w3_silu_reference`

Pure PyTorch loop over experts — **always available**, no Triton required:

```python
for e in range(n_experts):
    cnt = counts[e]
    if cnt == 0:
        continue
    start, end = offsets[e], offsets[e] + cnt
    chunk = x_sorted[start:end]
    g = F.linear(chunk, W1_stack[e])
    u = F.linear(chunk, W3_stack[e])
    out[start:end] = F.silu(g) * u
```

Used by:

- [`tests/test_moe_triton.py`](../tests/test_moe_triton.py) on CPU
- `_MoEW1W3SiluFunction.backward` for gradient computation
- Numerical cross-checks against the Triton forward

---

## Autograd — Forward Triton, Backward PyTorch

### `_MoEW1W3SiluFunction`

```python
class _MoEW1W3SiluFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x_sorted, ..., W1_stack, W3_stack):
        out = _moe_w1w3_silu_kernel[...](...)   # Triton
        ctx.save_for_backward(...)
        return out

    @staticmethod
    def backward(ctx, grad_outputs):
        with torch.enable_grad():
            x_d = x_sorted.detach().requires_grad_(True)
            W1_d = W1_stack.detach().requires_grad_(True)
            W3_d = W3_stack.detach().requires_grad_(True)
            y_ref = _moe_w1w3_silu_reference(x_d, ..., W1_d, W3_d)
        g_x = autograd.grad(y_ref, x_d, grad_outputs)[0]
        g_W1 = autograd.grad(y_ref, W1_d, grad_outputs)[0]
        g_W3 = autograd.grad(y_ref, W3_d, grad_outputs)[0]
        return g_x, None, None, None, g_W1, g_W3
```

### Design rationale

| Choice | Reason |
|---|---|
| Forward = Triton | Speed on the hot path |
| Backward = PyTorch reference | Correctness without writing backward Triton |
| `enable_grad()` recomputation | Standard autograd.Function pattern |
| No grad for counts/offsets/ids | Discrete routing metadata |

**Trade-off:** backward is slower than a fused backward kernel would be, but
MoE backward is dominated by W2 and attention anyway at `T=4096`.

### Gradient paths to expert weights

`g_W1` and `g_W3` have shape `(E, d_ff, d_model)` — must be consumed when
stacking weights from `nn.Linear` parameters. The Triton path stacks weights
for forward; gradients flow back into individual `expert.w1.weight` and
`expert.w3.weight` through the autograd graph when using stacked tensors
derived from parameters (the dispatch path stacks from live parameters each
forward).

---

## Integration in `MoELayer._dispatch_triton`

Full path in [`models/moe.py`](../models/moe.py):

```
1. Sort tokens by expert (stable argsort)
2. x_sorted = flat[sorted_token_ids]
3. gated_sorted = triton_moe_w1w3_silu(...)     ← Triton
4. For each expert e:
       out_sorted[slice] = gated_sorted[slice] @ W2_e.T   ← PyTorch
5. out_sorted *= sorted_weights
6. out.index_add_(0, sorted_token_ids, out_sorted)
```

Shared experts still use full `SwiGLUExpert.forward` in PyTorch after step 6.

---

## What Stays in PyTorch

Even with `triton_grouped`:

| Component | Implementation |
|---|---|
| MoE router + softmax | PyTorch |
| Aux load-balancing loss | PyTorch |
| W2 down-projection | PyTorch `matmul` per expert |
| Routing weight multiply + index_add | PyTorch |
| Shared experts (full SwiGLU) | PyTorch |
| Attention, RoPE, norms, head | PyTorch |
| Loss, optimizer | PyTorch |

---

## Hardware Notes — sm_75 vs sm_80

| GPU | Arch | Shared mem | `num_stages` | Status |
|---|---|---|---|---|
| GTX 1650 | sm_75 | 64 KB | 1 | Verified (smoke test) |
| A100 80GB | sm_80 | 164 KB | 1 (default) | Production target |
| RTX 5090 | sm_120 | — | TBD | Use `stacked` until verified |

Triton requires **Linux + CUDA** for the fused path. macOS and CPU-only
machines must use `moe_dispatch="stacked"`.

---

## Hard Caps and Shape Limits

```python
_MOE_FFN_HARD_CAP = 8192
_MOE_DMODEL_HARD_CAP = 8192
```

If `d_ff > 8192` or `d_model > 8192`, forward raises `ValueError` before
kernel launch. GPT-OSS-Lite defaults (`768`, `1536`) are well within caps.

---

## When to Enable `triton_grouped`

### Enable when

- Training on **Linux + CUDA** with Triton installed
- GPU is **sm_75+** with sufficient shared memory for tile config
- MoE dispatch is a profiling hotspot (profile before enabling)
- Running long A100 pretrain ([`configs/pretrain_a100_502m.yaml`](../configs/pretrain_a100_502m.yaml))

### Keep `stacked` when

- Developing on **Mac or CPU**
- Debugging routing / aux loss issues
- Running unit tests (tests use both paths but reference is always PyTorch)
- Triton version mismatch / compile failures
- Any scenario where kernel correctness is unverified on your GPU

### Config example

```yaml
model:
  moe_dispatch: "triton_grouped"
```

Default [`pretrain_a100_502m.yaml`](../configs/pretrain_a100_502m.yaml) leaves
`moe_dispatch` unset → `"stacked"`. Add the key explicitly for Triton runs.

### Expected speedup

Modest (~5–15% on MoE-heavy steps) — W2 and attention dominate at `T=4096`.
Measure with `torch.profiler` on your batch shape before assuming gains.

---

## Adding a New Sanctioned Kernel

Per [`AGENTS.md`](../AGENTS.md):

1. Place in `models/<name>_triton.py`
2. Gate on `import triton` with `try/except` → `HAS_TRITON = False`
3. Wrap in `torch.autograd.Function` (forward kernel, backward reference or kernel)
4. Add `tests/test_<name>_triton.py` with CPU-runnable PyTorch reference
5. GPU-only tests behind `@pytest.mark.gpu`
6. Update `AGENTS.md` sanctioned list
7. Add `documentation/<name>.md` plan
8. Opt-in via explicit config key — **never** silent fallback

---

## Testing Contract

[`tests/test_moe_triton.py`](../tests/test_moe_triton.py):

| Test | Runs on CPU? | Purpose |
|---|---|---|
| `test_reference_matches_naive_per_expert_loop` | Yes | Reference correctness |
| `test_reference_handles_empty_experts` | Yes | Zero-count experts |
| `test_triton_moe_raises_when_triton_missing` | Yes | ImportError policy |
| `test_triton_forward_matches_reference` | GPU only | Triton vs reference |
| `test_moe_triton_grouped_matches_stacked` | GPU only | End-to-end dispatch parity |

**Rule:** every new Triton path must have a CPU-runnable reference test without
`triton` installed ([`AGENTS.md`](../AGENTS.md) rule 9).

---

## Debugging Checklist

| Symptom | Check |
|---|---|
| `ImportError: triton` | Install Triton or switch to `stacked` |
| `ValueError: exceeds hard cap` | Absurd `d_ff`/`d_model` in config |
| Triton compile error | CUDA/Triton version mismatch |
| Forward mismatch vs stacked | Run `pytest tests/test_moe_triton.py -v` on GPU |
| NaN only with Triton | Compare to reference; check input dtypes |
| OOM on sm_75 | Confirm `num_stages=1`; reduce batch if needed |

---

## Appendix A — Program ID layout

```
Expert 0: tokens [0, cnt0)     → programs (e=0, t_blk=0.., n_blk=0..)
Expert 1: tokens [cnt0, cnt0+cnt1)
...
Expert 7: tokens [..., N_slots)

For each (e, t_blk, n_blk):
  tok_in_blk = t_blk * BLOCK_T + [0..BLOCK_T-1]
  n_in_blk   = n_blk * BLOCK_N + [0..BLOCK_N-1]
  accumulate over k in [0, d_model) step BLOCK_M
```

---

## Appendix B — Memory traffic comparison

**Stacked path per expert chunk** (conceptual writes):

```
x → W1x (write g) → silu(g) (write) → W3x (write u) → mul (write h)
```

**Triton path:**

```
x → fused directly to h (one write per (token, d_ff) element)
```

For `d_ff=1536`, eliminates ~3 intermediate `(tokens, 1536)` buffers per
expert chunk.

---

## Appendix C — Glossary

| Term | Definition |
|---|---|
| **Grouped GEMM** | Batched matmul where groups share weights but not batch index |
| **BLOCK_T/M/N** | Triton tile sizes for tokens, d_model, d_ff |
| **HAS_TRITON** | Module-level flag after import attempt |
| **Reference backward** | PyTorch recomputation for autograd |
| **`moe_dispatch`** | Config switch: `stacked` vs `triton_grouped` |
| **Sanctioned path** | Kernel listed in AGENTS.md — allowed custom Triton |

---

## Load-Bearing Invariants

1. **`HAS_TRITON` gate** at import — no unconditional Triton import.
2. **`triton_grouped` + no Triton → ImportError** — never silent fallback.
3. **Backward via PyTorch reference** in v1 — do not skip grad tests.
4. **Tile sizes documented** — `BLOCK_T=16, BLOCK_M=32, BLOCK_N=32`.
5. **`num_stages=1`** default for sm_75 compatibility.
6. **New kernels** require AGENTS.md + tests + documentation update.

---

## References

- [`models/moe_triton.py`](../models/moe_triton.py) — kernel source
- [`models/moe.py`](../models/moe.py) — dispatch integration
- [moe.md](moe.md) — MoE theory
- [training.md](training.md) — compile and TF32 knobs
- [`AGENTS.md`](../AGENTS.md) — workspace kernel contract
- Triton docs: https://triton-lang.org/

<!-- docs:verified 2026-07-31 · fa6f918 -->
