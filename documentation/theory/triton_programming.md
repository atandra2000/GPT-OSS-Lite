# Triton Programming and the Fused MoE Kernel

> **Chapter 10 of the GPT-OSS-Lite theory series.** GPU execution models are usually taught as CUDA C: threads, blocks, and explicit `__syncthreads()`. Triton inverts that — you program *tiles*, and the compiler assigns warps, schedules loads, and inserts barriers. This chapter builds the GPU mental model from first principles, then reads the one sanctioned Triton kernel in the repo: the fused W1/W3+silu MoE grouped-GEMM in `models/moe_triton.py`. For the MoE design itself see [moe_theory.md](moe_theory.md) and [moe.md](../moe.md); for the surrounding hardware budget see [foundations.md](../foundations.md).

---

## Table of contents

1. [60-second summary](#1-60-second-summary)
2. [Why it matters here](#2-why-it-matters-here)
3. [Intuition](#3-intuition)
4. [The GPU execution model](#4-the-gpu-execution-model)
5. [Triton: programming tiles, not threads](#5-triton-programming-tiles-not-threads)
6. [Why fuse W1/W3+silu — the launch/FLOP tradeoff](#6-why-fuse-w1w3silu--the-launchflop-tradeoff)
7. [The tile shape: BLOCK_T=16 × BLOCK_M=32 × BLOCK_N=32](#7-the-tile-shape-block_t16--block_m32--block_n32)
8. [Autograd contract: forward Triton, backward reference](#8-autograd-contract-forward-triton-backward-reference)
9. [No-silent-fallback rule](#9-no-silent-fallback-rule)
10. [Code walkthrough](#10-code-walkthrough)
11. [Pitfalls + verify](#11-pitfalls--verify)

---

## 1. 60-second summary

GPUs are throughput machines: thousands of threads run the same instruction in lockstep (SIMT), and every byte they touch must travel through a strict memory hierarchy — registers, then shared memory, then HBM. Tiled matmuls are mandatory because a naive GEMM re-reads its inputs from HBM for every output element; tiling reuses each tile many times before it is evicted.

Triton is a tile-oriented language: you write `tl.load` / `tl.store` / `tl.dot` on blocks of tensors with masks, and Triton lowers that to warps, shared-memory buffers, and software-pipelined loads. GPT-OSS-Lite uses exactly one Triton kernel, the sanctioned exception to its raw-PyTorch rule ([AGENTS.md](../../AGENTS.md) §1): `models/moe_triton.py:triton_moe_w1w3_silu` fuses the gating GEMM (W1), up-projection GEMM (W3), SiLU, and elementwise multiply of one MoE expert's dispatch into a single launch, tiled 16×32×32 with `num_stages=1`. It is opt-in (`moe_dispatch="triton_grouped"`), raises `ImportError` when Triton is absent — never silently falls back — and its autograd backward re-runs a pure-PyTorch reference. W2 stays in PyTorch.

## 2. Why it matters here

GPT-OSS-Lite is a 12-layer, 501.8M-parameter transformer with 247.0M active parameters (50.8% sparsity): every MoE layer routes each token to the top-2 of 8 routed experts plus a shared expert ([moe.md](../moe.md)). Each expert is a SwiGLU block `W2(silu(W1 x) * W3 x)` with `d_model=768`, `d_ff=1536` ([moe.md](../moe.md#parameter-and-flop-accounting)). Per token, an activated expert's W1 and W3 GEMMs are

$$
2 \times 2 \cdot d_{\text{model}} \cdot d_{\text{ff}} = 4 \cdot 768 \cdot 1536 = 4718592 \text{ FLOPs},
$$

i.e. two-thirds of the expert's three GEMMs. With 32,768 tokens per step (micro_bs=8 × max_seq_len=4096) and top-2 routing, the W1/W3 stage alone is ≈309 GFLOP per layer per step — and it is executed as 8 separate small per-expert chunks. Small chunks are the pathological regime for GPU kernels: launch latency (~µs) is comparable to the compute time, so per-op PyTorch dispatch wastes a large fraction of the time. This is the *dispatch hot path* the kernel exists to fix, and it is the single sanctioned Triton path per the AGENTS.md kernel contract ([AGENTS.md](../../AGENTS.md)).

The rest of the model — attention, RMSNorm, RoPE/YaRN, embeddings, LM head, aux loss, W2, the router — stays raw PyTorch, per hard rule 1. The kernel is not a modernization project; it is a targeted, test-guarded intervention on one measured bottleneck.

## 3. Intuition

Think of the GPU as a warehouse: one foreman (the host CPU) issues instructions, but the actual work is done by thousands of workers (threads) organized into teams (warps). Every worker on a team must do the same task at the same time, or the stragglers idle. The workers' workbenches are registers — tiny, per-worker, and fast. The shared lunch table (shared memory) is visible to one team at a time and is the only way teammates hand materials to each other. The warehouse shelves (HBM) hold everything but are far away: fetching from shelves costs ~100–400× a register access, so the entire art of GPU programming is *fetching a pile of material to the table once, then using it many times before putting it back*.

A matmul is exactly a job where the same numbers get reused: every output element needs an entire row of one matrix and an entire column of another. Naively that is one shelf trip per output element. Tiling says: carry a rectangular slab (a tile) to the table, compute a whole patch of outputs from it, and only then go back to the shelves. Triton makes the warehouse analogy literal: `tl.dot` computes a tile of outputs, `tl.load` hauls a tile from shelves to the table, and `num_stages` decides how many slabs the foreman has stacked up ahead of time so workers never wait on a trip.

## 4. The GPU execution model

### 4.1 SIMT: warps, not threads

A GPU streaming multiprocessor (SM) executes *warps* — groups of 32 threads that run one instruction at a time in lockstep (SIMT: single instruction, multiple threads). Divergence (an `if` where some lanes take one branch) serializes: the warp executes both paths, with the other lanes masked off. There is no branch prediction that hides this; you pay both sides.

Latency hiding replaces caches-as-speed: an SM keeps many warps resident and switches to another warp whenever one stalls on memory. Occupancy — the number of resident warps — is bounded by the scarcest resource per thread: registers (A100: 65,536 32-bit registers per SM, 255 per thread) and shared memory. This is why the fused kernel's accumulator choice matters: two fp32 accumulators of 16×32 floats = 1,024 registers per program; with 4 warps (128 threads) that is 8 registers per thread just for `g_acc` and `u_acc`, leaving headroom for operand tiles and addresses.

### 4.2 Memory hierarchy and coalescing

The hierarchy, in increasing latency and capacity:

| Level | Scope | A100 (sm_80) | GTX 1650 (sm_75) |
|---|---|---|---|
| Registers | per thread | ~256/thread, 64K/SM | same class |
| Shared memory | per block | 164 KB/SM | 64 KB/SM |
| L2 | chip-wide | ~40 MB | ~1.5 MB |
| HBM | device | ~2 TB/s | ~192 GB/s |

HBM bytes are the budget that matters: a 768×1536 BF16 weight is 2.36 MB and cannot be re-fetched casually. *Coalescing* is the rule that makes HBM traffic efficient: when the 32 lanes of a warp load consecutive 4-byte addresses, the memory system serves them with one 128-byte transaction; scattered addresses cost 32 transactions. Triton guarantees coalescing for you *when the innermost stride of a `tl.load` is the contiguous one* — which is why the kernel passes explicit row strides (`stride_xt`, `stride_xd`, …) rather than reshuffling tensors.

### 4.3 Why tiling is mandatory

Consider one expert's GEMM $y = xW^\top$ with $x \in \mathbb{R}^{T \times K}$, $W \in \mathbb{R}^{N \times K}$, $K = d_{\text{model}} = 768$, $N = d_{\text{ff}} = 1536$, element size $b$ bytes (2 for BF16). Untiled, each output element $y_{ij}$ reads row $i$ of $x$ ($K$ elements) and column $j$ of $W$ ($K$ elements):

$$
\text{naive: } \frac{\text{bytes}}{\text{output}} = 2Kb = 2 \cdot 768 \cdot 2 = 3072 \text{ B}.
$$

Tiled with an output tile of $T \times N$ (the fused kernel uses $T = \text{BLOCK\_T} = 16$, $N = \text{BLOCK\_N} = 32$) and two weight matrices (W1 and W3 share the same $x$ tile), one program reads $T \times K$ x-elements and $2 \times N \times K$ weight elements to produce $T \times N$ outputs:

$$
\text{tiled: } \frac{\text{bytes}}{\text{output}} = Kb\left(\frac{2}{T} + \frac{1}{N}\right) = 768 \cdot 2 \cdot \left(\frac{2}{16} + \frac{1}{32}\right) = 240 \text{ B},
$$

a 12.8× reduction in HBM traffic at the kernel's own tile size — before counting that the same $x$-tile is re-read from L2 by the 48 programs that share it (the $d_{\text{ff}}$ direction). The arithmetic intensity (FLOPs per HBM byte) is the quotient of $4K$ FLOPs per output element (two GEMMs) and the traffic above:

$$
I_{\text{tiled}} = \frac{4K}{Kb(2/T + 1/N)} = \frac{4}{b(2/T + 1/N)} = 12.8 \text{ FLOP/B},
$$

versus $I_{\text{naive}} = 2/b = 1$ FLOP/B. Against A100's ~2 TB/s HBM, an intensity of 1 FLOP/B caps the GEMM at ~2 TFLOP/s; intensity 12.8 caps it near ~26 TFLOP/s — the difference between a kernel that is bandwidth-bound and one that is compute-bound. Larger tiles (e.g. $T=64$, $N=64$) would push intensity to $4/(2 \cdot (2/64+1/64)) = 42.7$ FLOP/B, which is why the launcher comments note production can grow tiles on A100; the shipped 16×32 shape is the sm_75-compatible default (section 7).

## 5. Triton: programming tiles, not threads

### 5.1 Primitives

A Triton kernel is `@triton.jit`-decorated Python executed at compile time by the Triton compiler; the "variables" are *tiles* (block tensors), not scalars. The four primitives used by `_moe_w1w3_silu_kernel`:

- `tl.arange(0, N)` — a 1-D block of consecutive integers $0..N-1$; the only way to build index vectors, and its length must be a power of two.
- `tl.load(ptr + offsets, mask=..., other=0.0)` — load a tile from global memory; lanes failing `mask` get `other`. Masked-out lanes do not fault and do not contribute.
- `tl.dot(a, b, allow_tf32=False)` — a tile matmul; the reduction dimension is the last dim of `a` and must match the first dim of `b`. On sm_80 `allow_tf32=False` forces IEEE fp32 (10-bit TF32 mantissa would break the fp32 test tolerance).
- `tl.store(ptr + offsets, val, mask=...)` — write a tile back; the `mask` prevents writes to out-of-bounds or padded rows.

There is no `for` over threads anywhere — the kernel body describes one tile-shaped unit of work; `tl.program_id(axis)` identifies which unit this program is. The compiler maps tiles onto warps, chooses shared-memory layouts, and inserts synchronization.

### 5.2 Grid, masks, and the tile contract

Launching is `kernel[(grid_x, grid_y, grid_z)](args...)`: one program per grid cell, each running the same body. In this kernel the grid is `(n_experts, n_tiles_t, n_tiles_n)` — one program per (expert, token-tile, d_ff-tile). Each program accumulates a full output tile over the reduction dimension with a Python loop:

```python
for k0 in range(0, d_model, BLOCK_M):
    k_offsets = k0 + tl.arange(0, BLOCK_M)
    k_mask = k_offsets < d_model
    x_tile = tl.load(x_row_base + k_offsets[None, :] * stride_xd,
                     mask=tok_mask[:, None] & k_mask[None, :], other=0.0)
    g_acc += tl.dot(x_tile, tl.trans(w1_tile), allow_tf32=False)
```

Masks are mandatory because tiles are powers of two but real dimensions are not: `d_ff = 1536 = 48 × 32` divides evenly, but `max_tokens` (the largest expert's count, which sizes `n_tiles_t`) generally does not, so the final token tile of most experts is partially masked by `tok_mask`, and `n_mask`/`k_mask` keep the kernel general to non-divisible `d_ff`/`d_model`. The `other=0.0` on masked loads means padded rows contribute zero to the accumulators; the store is masked with the same predicate, so garbage rows never leave the kernel.

### 5.3 num_stages and software pipelining

`num_stages` tells Triton how many iterations of the loop to overlap: while program $i$ computes `tl.dot` on the tiles already in shared memory, the hardware asynchronously prefetches the tiles for iteration $i+1$ (on sm_80+ via `cp.async`; sm_90+ via TMA). With `num_stages=1` the load of iteration $i+1$ cannot begin until iteration $i$'s dot finishes — the SM idles on every memory trip. Each extra stage costs one more set of shared-memory buffers, so the trade is

$$
\text{shared} \approx \text{num\_stages} \times \underbrace{\big(\text{BLOCK\_T}\cdot\text{BLOCK\_M} + 2\cdot\text{BLOCK\_M}\cdot\text{BLOCK\_N}\big)b}_{\text{tile buffers per stage}}.
$$

For the shipped shape, per stage: $x$-tile $16 \times 32$ elements plus W1 and W3 tiles $2 \times 32 \times 32$ = 2,560 elements, 5 KB in BF16 (10 KB in fp32). On sm_75 (GTX 1650-class) shared memory is 64 KB/SM, and Triton's pre-Ampere pipeliner has no `cp.async`; it allocates conservative staging buffers per stage on top of the tile buffers, and `num_stages=2` overruns the 64 KB budget and spills to local memory. That is why the launcher pins `num_stages=1` with the comment "GTX 1650 (sm_75) has 64 KB shared; num_stages=2 spills", verified end-to-end via `scripts/e2e_gpu_smoke.py`. A100 (sm_80) has 164 KB and `cp.async`; the launcher comment notes production can re-enable `num_stages=2`. `num_warps=4` (128 threads) sizes the warp-level parallelization of each tile.

## 6. Why fuse W1/W3+silu — the launch/FLOP tradeoff

### 6.1 Launch-bound regimes

Fusion changes zero FLOPs — `silu(W1 x) * (W3 x)` is the same arithmetic in one kernel or three. The win is launch overhead and intermediate traffic. Per expert chunk in the vectorized PyTorch path, the W1/W3/silu/mul stage costs four kernel launches (W1 linear, W3 linear, silu, mul); with 8 routed experts that is 32 launches per layer for the stage the fused kernel replaces with one. Define the per-chunk compute time and the per-launch latency:

$$
t_c = \frac{2\,d_{\text{model}}\,d_{\text{ff}}\,n_c}{P}, \qquad \frac{t_l}{t_c} = \frac{t_l \, P}{2\,d_{\text{model}}\,d_{\text{ff}}\,n_c} = \frac{21.2}{n_c},
$$

where $n_c$ is the chunk's token count and $P$ the achieved FLOP rate. The numeric form uses $t_l = 5\,\mu\text{s}$ launch latency `[INFERENCE]` and $P = 10^{13}$ FLOP/s `[INFERENCE]` (a conservative A100 BF16 rate, since `.benchmarks/` is empty). At $n_c = 32$ tokens per expert the launch overhead is ~66% of compute; at $n_c = 128$ it is ~17%. MoE chunks are exactly this small — with 8 experts and balanced routing (the aux-loss α=0.01 objective encourages it, [moe.md](../moe.md)), 4096 tokens split into ≈512-token chunks *before* top-2 slots, and each routed slot's chunk is a fraction of that. Collapsing 32 launches into 1 removes 31/32 of the overhead. The 5–15% MoE-forward speedup expected on sm_80+ ([operations.md](../operations.md#opt-24--triton-grouped-gemm-moe_dispatchtriton_grouped)) is `[INFERENCE]`, not measured.

### 6.2 Activation traffic

The second, larger win is HBM traffic on the activations. Unfused, the pipeline writes `g` to HBM, writes `u`, re-reads both for `silu(g) * u`, and writes the product `y` — five activation streams. Fused, `g` and `u` live in registers and only `y` is stored. With $N_{\text{slots}} = N_{\text{tokens}} \times 2$ routed slots:

$$
\Delta T = 4\,N_{\text{slots}}\,d_{\text{ff}}\,b = 4 \cdot 8192 \cdot 1536 \cdot 2 = 100663296 \text{ B} \approx 96 \text{ MiB}
$$

saved per MoE layer forward (8192 slots = 4096 tokens × top-2, BF16) — ~1.15 GiB across 12 layers per forward, before backward. The weight reads are identical in both paths; fusion only removes the activation round-trips.

### 6.3 Why W2 stays in PyTorch

W2 is the down-projection $d_{\text{ff}} \to d_{\text{model}}$ applied after the fused activation. Fusing it would require a second kernel with a different tiling (reduction over $d_{\text{ff}}$, output into the residual stream) plus the routing-weight scaling and the `index_add_` scatter back into the token buffer — a different problem from the grouped-GEMM, for a stage that is one GEMM instead of two and that must interact with the router's per-token weights. The sanctioned split is deliberate: Triton up+gate, PyTorch W2 ([moe.md](../moe.md#why-fuse-w1w3silu-not-w2)).

## 7. The tile shape: BLOCK_T=16 × BLOCK_M=32 × BLOCK_N=32

Three constants, three distinct roles:

- **BLOCK_T = 16** — tokens per program. The token direction is variable (`max_tokens` per expert), so it is the *grid* dimension that absorbs variance ($n_{\text{tiles\_t}} = \lceil \text{max\_tokens}/16 \rceil$), not the tile. 16 keeps the output tile $16 \times 32 = 512$ elements: two fp32 accumulators = 1,024 registers per program (section 4.1).
- **BLOCK_M = 32** — the reduction (K) block over $d_{\text{model}} = 768$, giving 24 loop iterations. It is the middle dimension of `tl.dot`: `(16, 32) @ (32, 32) → (16, 32)`. Small enough that the x-tile and both weight tiles fit the per-stage shared budget of equation (7).
- **BLOCK_N = 32** — outputs ($d_{\text{ff}}$) per program; $d_{\text{ff}} = 1536 = 48 \times 32$ tiles exactly. The x-tile is shared across these 48 programs through L2, which is where the $2/T$ term of equation (4) gets its cross-program reuse.

The grid and its size (per layer):

$$
\text{grid} = (n_{\text{experts}},\; \lceil \text{max\_tokens}/16 \rceil,\; \lceil d_{\text{ff}}/32 \rceil) = (8,\; \lceil \text{max\_tokens}/16 \rceil,\; 48).
$$

With balanced routing, max_tokens ≈ 1024 → 64 token tiles → ≈ 24,576 programs, each doing 24 k-steps × 32,768 FLOPs ≈ 786K FLOPs. Masking semantics are exact: `tok_mask` zeroes rows beyond expert $e$'s count (every expert except the busiest has a partially masked last token tile); `n_mask` zeroes columns beyond $d_{\text{ff}}$ (never false at $d_{\text{ff}}=1536$, but required for the hard-cap-general kernel); both are ANDed into every load and the store.

## 8. Autograd contract: forward Triton, backward reference

`models/moe_triton.py:_MoEW1W3SiluFunction` wraps the kernel in `torch.autograd.Function`. `forward` runs the Triton kernel and saves the six inputs; `backward` discards nothing and instead recomputes the entire forward graph through the pure-PyTorch reference under `torch.enable_grad()`, then calls `torch.autograd.grad` three times (x, W1, W3) against the incoming `grad_output`. Gradients for `expert_ids_sorted`, `counts`, `offsets` are `None` — they are discrete routing metadata, not differentiable.

Why this is *correct*: the fused function and its Jacobians factor cleanly. With $g = W_1 x$, $u = W_3 x$, $\sigma$ the logistic sigmoid, and $\odot$ elementwise multiplication,

$$
y = \text{silu}(g) \odot u, \qquad \text{silu}(z) = z\,\sigma(z), \qquad \sigma(z) = \frac{1}{1 + e^{-z}},
$$

so the vector-Jacobian products decompose as

$$
\frac{\partial \ell}{\partial x} = W_1^\top\big(\text{silu}'(g) \odot u \odot \bar y\big) + W_3^\top\big(\text{silu}(g) \odot \bar y\big), \qquad
\text{silu}'(z) = \sigma(z)\big(1 + z(1-\sigma(z))\big),
$$

with $\bar y = \partial \ell/\partial y$ the incoming gradient, and $W_1^\top, W_3^\top$ products and the outer products $\partial \ell/\partial W_1 = \bar g \otimes x$, $\partial \ell/\partial W_3 = \bar u \otimes x$ handled by autograd on the recomputed graph. The reference recomputes exactly the graph whose forward equals the kernel's output (guarded by `test_kernel_forward_matches_reference_*`), so the gradients are the gradients of the function the kernel computes.

Why it is *acceptable for a v1*: a hand-written backward would need three more kernels (two $W^\top$ GEMMs and the two weight outer-products) plus a fused activation-derivative kernel — roughly doubling the kernel surface for a 2×-forward backward. The reference backward costs 6G FLOPs/token (2G recompute + 2G for $x$-gradients + G + G for the weight grads, with $G = 2\,d_{\text{model}}\,d_{\text{ff}}$) against 4G for a cached custom backward:

$$
B_{\text{ref}} = 6G = 3 \times F_{\text{forward}}, \qquad B_{\text{custom}} = 4G = 2 \times F_{\text{forward}},
$$

a 50% backward-FLOP premium paid in exchange for zero kernel gradient code and provably-exact gradients. Memory stays minimal: only the six input tensors are saved, no activations. The trade is documented in [moe.md](../moe.md#autograd--forward-triton-backward-pytorch-reference); a v2 may add the dual kernels.

## 9. No-silent-fallback rule

AGENTS.md hard rule 8: *Never* let a Triton kernel silently fall back to the raw-PyTorch path during a default-config training run ([AGENTS.md](../../AGENTS.md) §2.8). The machinery, in order:

1. `models/moe_triton.py:HAS_TRITON` — module-level flag from `try: import triton … except ImportError`. Import failure is *visible*, not swallowed into a fallback path.
2. The public entry `models/moe_triton.py:triton_moe_w1w3_silu` checks the flag first and raises `ImportError` with install guidance and the escape hatch ("For CPU/Mac, use `moe_dispatch='stacked'`").
3. `models/moe.py:MoELayer.forward` only enters `_dispatch_triton` when `moe_dispatch == "triton_grouped"`; the default is `"stacked"`. Selection is explicit per-run configuration (`models/transformer.py:ModelConfig.moe_dispatch`), so a configured Triton run failing at *compile* time (a Triton `CompilationError`/`RuntimeError`) propagates up uncaught — nothing catches it and reroutes to PyTorch.
4. `_MoEW1W3SiluFunction.forward` additionally hard-fails with `ValueError` if `d_ff` or `d_model` exceeds the hard caps (8,192) — oversized shapes must not silently degrade.

The behavior matrix (also in [moe.md](../moe.md#has_triton-import-policy-and-hard-failure)): `HAS_TRITON=False` + `"triton_grouped"` → `ImportError` at first call; `True` + `"stacked"` → Triton never imported by the MoE forward. CPU-only machines simply never see the kernel: the CPU suite runs green with only the 2 GPU-gated Triton tests skipping.

## 10. Code walkthrough

### 10.1 Host entry — `models/moe_triton.py:triton_moe_w1w3_silu`

The public function is deliberately small: check the flag, delegate to `torch.autograd.Function.apply`. The `forward` of `models/moe_triton.py:_MoEW1W3SiluFunction` does the real orchestration — shape checks, hard caps, tile constants, grid math, launch:

```python
BLOCK_T = 16
BLOCK_M = 32
BLOCK_N = 32

max_tokens = int(counts.max().item()) if counts.numel() else 0
n_tiles_t = (max_tokens + BLOCK_T - 1) // BLOCK_T
n_tiles_n = (d_ff + BLOCK_N - 1) // BLOCK_N

_moe_w1w3_silu_kernel[(n_experts, n_tiles_t, n_tiles_n)](
    x_sorted, expert_ids_sorted, counts, offsets,
    W1_stack, W3_stack, out,
    n_tokens, d_model, d_ff,
    x_sorted.stride(0), x_sorted.stride(1),
    0, 0,
    W1_stack.stride(0), W1_stack.stride(1), W1_stack.stride(2),
    W3_stack.stride(0), W3_stack.stride(1), W3_stack.stride(2),
    out.stride(0), out.stride(1),
    BLOCK_T=BLOCK_T, BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N,
    N_EXPERTS=n_experts,
    num_warps=4, num_stages=1,
)
```

Notes: strides are passed explicitly so the kernel is layout-generic; `stride_e1, stride_e2 = 0, 0` because the sorted layout means `tl.program_id(0)` alone identifies the expert (the `eid_ptr` argument is part of the signature but unused — `expert_ids_sorted` is consumed by the sort, not the kernel). `counts`/`offsets` are the per-expert boundaries computed by `_dispatch_triton`; the kernel indexes the stacked weights as `w1_ptr + e*stride_w1e + n*stride_w1f + k*stride_w1d` where `W1_stack[e]` is `(d_ff, d_model)`, which is why weight tiles are loaded in `(N, K)` orientation and transposed before `tl.dot`.

The JIT kernel `_moe_w1w3_silu_kernel` itself (defined inside `if HAS_TRITON:`) loads `cnt`/`off` for its expert, builds `tok_mask`/`n_mask`, accumulates `g_acc`/`u_acc` in fp32 over `d_model` in BLOCK_M steps, applies `silu(g) * u` in fp32 (sigmoid is fp32-only in Triton), casts to the output dtype, and stores masked. Its correctness is anchored here through the host wrapper, per the CI symbol rule.

### 10.2 The CPU-runnable oracle — `models/moe_triton.py:_moe_w1w3_silu_reference`

The reference is the spec the kernel must match, and the backward engine (section 8):

```python
for e in range(n_experts):
    cnt = int(counts[e].item())
    if cnt == 0:
        continue
    start = int(offsets[e].item())
    end = start + cnt
    chunk = x_sorted[start:end]
    g = torch.nn.functional.linear(chunk, W1_stack[e])
    u = torch.nn.functional.linear(chunk, W3_stack[e])
    out[start:end] = torch.nn.functional.silu(g) * u
```

It runs on CPU with no Triton installed, handles empty experts (skips them; `out` is pre-allocated so every row is defined), and produces exactly the `(N_slots, d_ff)` shape the dispatch expects. It is the oracle for `test_reference_matches_naive_per_expert_loop` (1e-10 tolerance), `test_reference_handles_empty_experts`, and `test_reference_matches_existing_moe_dispatch_shape`, and — via the autograd wrapper — the gradient source of truth.

### 10.3 Integration — `models/moe.py:MoELayer._dispatch_triton`

`MoELayer.forward` picks `_dispatch_triton` only when `moe_dispatch == "triton_grouped"`; the method itself owns the sort-by-expert layout:

1. Router (`models/moe.py:MoERouter`) emits top-2 `(indices, weights)`; flatten to slots and stable-argsort by expert id so every expert's tokens are contiguous — the same layout both dispatch paths share.
2. `x_sorted = flat[sorted_token_ids]`; `expert_counts`/`expert_offsets` from `torch.bincount` + cumulative sum; the 8 expert weights stacked into `(8, 1536, 768)` `W1_stack`/`W3_stack`/`W2_stack` tensors.
3. `gated_sorted = triton_moe_w1w3_silu(x_sorted, sorted_expert_ids, expert_counts, expert_offsets, W1_stack, W3_stack)` — one launch, all experts.
4. W2 in PyTorch: per-expert loop `out_sorted[start:end] = gated_sorted[start:end] @ W2_stack[e].T`, then scale by `sorted_weights` and scatter back with `index_add_`.

Everything except step 3 is the same code the stacked path uses — the Triton path replaces only the two GEMMs + activation inside each chunk, exactly the sanctioned scope.

## 11. Pitfalls + verify

| Pitfall | Symptom | Guard |
|---|---|---|
| Silent fallback on missing Triton | Run proceeds on PyTorch despite `triton_grouped` | `pytest tests/test_moe_triton.py -v` → `test_triton_moe_raises_when_triton_missing`, `test_MoELayer_triton_dispatch_raises_when_triton_missing` (both CPU) |
| Oversized shapes | Kernel tile counts explode / shared overflow | `test_triton_moe_raises_on_hard_cap_violation` (d_ff, d_model > 8192 → `ValueError`) |
| TF32 rounding in `tl.dot` | fp32 kernel diverges from reference beyond 1e-3 | `allow_tf32=False` in the kernel; `test_kernel_forward_matches_reference_fp32` (GPU, atol/rtol 1e-3) |
| BF16 accumulation error | bf16 mismatch vs reference | fp32 accumulators (`g_acc`/`u_acc`); `test_kernel_forward_matches_reference_bf16` (GPU, 2e-2) |
| Shared-memory spill on sm_75 | Compile fail or local-memory spill on GTX 1650 | `num_stages=1`, tiles 16×32×32; `python3 scripts/e2e_gpu_smoke.py` on a ≥4 GB GPU |
| Masked rows writing garbage | Out-of-bounds store faults or NaN rows | `tok_mask`/`n_mask` ANDed into loads *and* store; `other=0.0` on loads |
| Backward drift | Gradients disagree with an autograd reference | Backward *is* the reference; end-to-end parity: `pytest tests/test_moe_triton.py -v` → `test_moe_triton_grouped_matches_stacked` (GPU tests auto-skip on CPU) |
| Wrong dispatch selected | `triton_grouped` ignored | `ModelConfig.moe_dispatch` default `"stacked"`; set explicitly in YAML (`configs/pretrain_a100_502m.yaml`) |

CPU verification (no Triton): `pytest tests/test_moe_triton.py -v` — runs the reference-correctness, ImportError-policy, and hard-cap tests; the two GPU-gated kernel-parity tests (skipif `gpu_required`) skip on CPU-only machines (190 passed / 2 skipped repo-wide). GPU verification: `pytest tests/test_moe_triton.py -v` (GPU tests auto-skip on CPU) and `python3 scripts/e2e_gpu_smoke.py`, which exercises `stacked` vs `triton_grouped` numerical equivalence plus the five-step training loop on sm_75 (operations catalog entry A.6, [operations.md](../operations.md#a6-e2e_gpu_smokepy)). Performance claims beyond these are `[INFERENCE]`: `.benchmarks/` is empty, and the 5–15% MoE-forward speedup on sm_80+ is an estimate, not a measurement.

For the surrounding machinery: routing and aux-loss math in [moe_theory.md](moe_theory.md), optimizer numerics in [optimizers.md](optimizers.md), and the softmax/attention side of GPU numerics in [attention_math.md](attention_math.md). The kernel's place in the MoE contract is [moe.md](../moe.md#sanctioned-triton-path-moe_dispatchtriton_grouped).

<!-- docs:verified 2026-08-04 · 5da1a80 -->
