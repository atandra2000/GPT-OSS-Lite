# Triton Kernels — GPT-OSS-Lite

> **Companion:** [moe.md](moe.md) §11, [AGENTS.md](../AGENTS.md) Triton kernel contract.

> **Read this if** you're enabling or modifying the fused MoE kernel. **Skip if** default PyTorch path is sufficient.

## Status (2026-07-31)

### Implemented

| Item | Location | Notes |
|---|---|---|
| MoE fused W1/W3+silu grouped-GEMM | `models/moe_triton.py` | `moe_dispatch: triton_grouped` |
| CPU reference path | `models/moe_triton.py` | `_moe_w1w3_silu_reference` |
| MoE Triton tests | `tests/test_moe_triton.py` | CPU reference + `@pytest.mark.gpu` |
| Default-off PyTorch path | `models/moe.py` | `stacked` dispatch remains default |
| E2E GPU smoke | `scripts/e2e_gpu_smoke.py` | sm_75 verified |

### Not implemented (deferred)

| Item | Notes |
|---|---|
| Triton W2 down-projection kernel | W2 stays PyTorch matmul |
| Triton backward kernel | Backward uses PyTorch reference |
| Fused cross-entropy | Chunked CE stays PyTorch |
| Attention Triton | SDPA/FA2 sufficient |

### Resolved decisions

- **Opt-in:** `moe_dispatch="triton_grouped"` in `ModelConfig` (no env-var gate).
- **MoE backward:** re-compute via PyTorch reference inside autograd.
- **Test baseline:** **187** pytest tests pass with Triton disabled (default config).

---

# MoE Triton kernel — design + implementation

> **Source:** `models/moe_triton.py`
> **Sanctioned:** Yes (only sanctioned kernel in the project).
> **Opt-in via:** `moe_dispatch="triton_grouped"` in `ModelConfig`.

This document supplements [`moe.md`](moe.md) with kernel-level detail: algorithm, launch geometry, numerical choices, and test surface.

## 1. Scope

We fuse the **W1·x (gating) + W3·x (up-projection) + silu(g) * u** chain into a single Triton kernel. W2 (down-projection) is a plain matmul in PyTorch.

```python
gate = W1[e] @ x_chunk
up   = W3[e] @ x_chunk
out[start:end] = silu(gate) * up
```

## 2. Algorithm

### 2.1 Launch geometry

```
grid = (n_experts, ceil(max_per_expert_tokens / BLOCK_T), ceil(d_ff / BLOCK_N))
```

### 2.2 Numerical choices

| Decision | Why |
|---|---|
| fp32 accumulator in K-loop | Tight BF16 matmul accumulation |
| silu in fp32 | `tl.sigmoid` requires fp32/fp64 |
| `allow_tf32=False` on `tl.dot` | Preserve BF16 ULP vs reference |
| `BLOCK_T=16, BLOCK_M=32, BLOCK_N=32` | sm_75 friendly (64 KB shared) |
| `num_stages=1` | sm_75 shared memory limit |

## 3. Backward

Forward on Triton; backward via pure-PyTorch reference in `torch.autograd.grad`. Same pattern as vLLM fused MoE.

## 4. Hard caps

| Variable | Cap |
|---|---|
| `d_ff` | 8192 |
| `d_model` | 8192 |

Raises `ValueError` if exceeded — no silent fallback (AGENTS.md rule 8).

## 5. When to use

- **A100 (sm_80):** enable `moe_dispatch="triton_grouped"`. Optional `num_stages=2`.
- **Dev (sm_75):** works at `num_stages=1`.
- **CPU / Mac:** keep `"stacked"`. `_dispatch_triton` raises `ImportError` if Triton missing.

## 6. Test surface

See [testing.md](testing.md) — `test_moe_triton.py` (9 tests).

`scripts/e2e_gpu_smoke.py` steps 3–4 cover kernel + full `MoELayer` integration.

## 7. Future work

- Triton W2 grouped GEMM
- Triton backward matching fp32-accumulating forward
- Larger tiles on A100 (`BLOCK_T=32, BLOCK_M=64, BLOCK_N=64`)

<!-- docs:verified 2026-07-31 · fd4fe36 -->
