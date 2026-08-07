"""Opt-in Triton fusion for the routed MoE input projections.

The kernel computes ``silu(W1(x)) * W3(x)`` for expert-sorted tokens in one
launch. W2 remains in PyTorch, and the reference implementation supplies the
CPU-safe behaviour and autograd backward.
"""
from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    import triton
    import triton.language as tl

try:
    import triton
    import triton.language as tl
    HAS_TRITON = True
except ImportError:
    HAS_TRITON = False

_MOE_FFN_HARD_CAP = 8192
_MOE_DMODEL_HARD_CAP = 8192


def _moe_w1w3_silu_reference(
    x_sorted: torch.Tensor,
    expert_ids_sorted: torch.Tensor,
    counts: torch.Tensor,
    offsets: torch.Tensor,
    W1_stack: torch.Tensor,
    W3_stack: torch.Tensor,
) -> torch.Tensor:
    """Compute the grouped gated projection in PyTorch for reference/backward."""
    n_experts = W1_stack.shape[0]
    out = torch.empty(
        x_sorted.shape[0], W1_stack.shape[1],
        dtype=x_sorted.dtype, device=x_sorted.device,
    )
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
    return out


if HAS_TRITON:

    @triton.jit
    def _moe_w1w3_silu_kernel(
        x_ptr, eid_ptr, cnt_ptr, off_ptr,
        w1_ptr, w3_ptr, out_ptr,
        n_tokens, d_model, d_ff,
        stride_xt, stride_xd,
        stride_e1, stride_e2,
        stride_w1e, stride_w1f, stride_w1d,
        stride_w3e, stride_w3f, stride_w3d,
        stride_ot, stride_of,
        BLOCK_T: tl.constexpr,  # token block
        BLOCK_M: tl.constexpr,  # reduction (d_model) block
        BLOCK_N: tl.constexpr,  # output (d_ff) block
        N_EXPERTS: tl.constexpr,
    ):
        """Fuse both input projections and the SwiGLU gate for one tile.

        Programs are grouped by expert and tile tokens/output features while
        accumulating the reduction dimension in FP32.
        """
        e = tl.program_id(0)
        t_blk = tl.program_id(1)
        n_blk = tl.program_id(2)

        cnt = tl.load(cnt_ptr + e)
        if cnt == 0:
            return
        off = tl.load(off_ptr + e)

        tok_in_blk = t_blk * BLOCK_T + tl.arange(0, BLOCK_T)
        tok_mask = tok_in_blk < cnt

        n_in_blk = n_blk * BLOCK_N + tl.arange(0, BLOCK_N)
        n_mask = n_in_blk < d_ff

        # Base pointers stay fixed while the reduction loop advances through d_model.
        x_row_base = x_ptr + (off + tok_in_blk)[:, None] * stride_xt
        w1_row_base = (
            w1_ptr + e * stride_w1e
            + n_in_blk[:, None] * stride_w1f
        )
        w3_row_base = (
            w3_ptr + e * stride_w3e
            + n_in_blk[:, None] * stride_w3f
        )
        out_row = out_ptr + (off + tok_in_blk)[:, None] * stride_ot + n_in_blk[None, :] * stride_of

        # FP32 accumulation limits error across the input-feature reduction.
        g_acc = tl.zeros((BLOCK_T, BLOCK_N), dtype=tl.float32)
        u_acc = tl.zeros((BLOCK_T, BLOCK_N), dtype=tl.float32)
        for k0 in range(0, d_model, BLOCK_M):
            k_offsets = k0 + tl.arange(0, BLOCK_M)
            k_mask = k_offsets < d_model
            x_tile = tl.load(
                x_row_base + k_offsets[None, :] * stride_xd,
                mask=tok_mask[:, None] & k_mask[None, :],
                other=0.0,
            )
            w1_tile = tl.load(
                w1_row_base + k_offsets[None, :] * stride_w1d,
                mask=n_mask[:, None] & k_mask[None, :],
                other=0.0,
            )
            w3_tile = tl.load(
                w3_row_base + k_offsets[None, :] * stride_w3d,
                mask=n_mask[:, None] & k_mask[None, :],
                other=0.0,
            )
            g_acc += tl.dot(x_tile, tl.trans(w1_tile), allow_tf32=False)
            u_acc += tl.dot(x_tile, tl.trans(w3_tile), allow_tf32=False)

        # Triton's sigmoid path is most portable in FP32.
        g32 = g_acc
        u32 = u_acc
        silu = g32 * tl.sigmoid(g32)
        fused = (silu * u32).to(out_ptr.dtype.element_ty)

        tl.store(out_row, fused, mask=tok_mask[:, None] & n_mask[None, :])


class _MoEW1W3SiluFunction(torch.autograd.Function):
    """Use Triton for forward and the differentiable PyTorch reference backward."""

    @staticmethod
    def forward(
        ctx,
        x_sorted: torch.Tensor,
        expert_ids_sorted: torch.Tensor,
        counts: torch.Tensor,
        offsets: torch.Tensor,
        W1_stack: torch.Tensor,
        W3_stack: torch.Tensor,
    ) -> torch.Tensor:
        n_tokens, d_model = x_sorted.shape
        d_ff = W1_stack.shape[1]
        n_experts = W1_stack.shape[0]
        out = torch.empty(n_tokens, d_ff, dtype=x_sorted.dtype, device=x_sorted.device)

        if d_ff > _MOE_FFN_HARD_CAP or d_model > _MOE_DMODEL_HARD_CAP:
            raise ValueError(
                f"triton_moe_w1w3_silu: d_ff={d_ff} or d_model={d_model} "
                f"exceeds hard cap ({_MOE_FFN_HARD_CAP} / {_MOE_DMODEL_HARD_CAP})."
            )

        # These conservative tiles keep the sanctioned sm_75 path within shared memory.
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
            # A single stage avoids shared-memory spills on the minimum supported GPU.
            num_warps=4, num_stages=1,
        )

        ctx.save_for_backward(
            x_sorted, expert_ids_sorted, counts, offsets, W1_stack, W3_stack,
        )
        return out

    @staticmethod
    def backward(ctx, grad_outputs: torch.Tensor):
        (x_sorted, expert_ids_sorted, counts, offsets, W1_stack, W3_stack) = \
            ctx.saved_tensors
        with torch.enable_grad():
            x_d = x_sorted.detach().requires_grad_(True)
            W1_d = W1_stack.detach().requires_grad_(True)
            W3_d = W3_stack.detach().requires_grad_(True)
            y_ref = _moe_w1w3_silu_reference(
                x_d, expert_ids_sorted, counts, offsets, W1_d, W3_d,
            )
        g_x = torch.autograd.grad(y_ref, x_d, grad_outputs)[0]
        g_W1 = torch.autograd.grad(y_ref, W1_d, grad_outputs)[0]
        g_W3 = torch.autograd.grad(y_ref, W3_d, grad_outputs)[0]
        return g_x, None, None, None, g_W1, g_W3


def triton_moe_w1w3_silu(
    x_sorted: torch.Tensor,
    expert_ids_sorted: torch.Tensor,
    counts: torch.Tensor,
    offsets: torch.Tensor,
    W1_stack: torch.Tensor,
    W3_stack: torch.Tensor,
) -> torch.Tensor:
    """Run the fused projection, failing clearly when Triton is unavailable."""
    if not HAS_TRITON:
        raise ImportError(
            "triton_moe_w1w3_silu requires the `triton` package. "
            "Install with `pip install triton` (Linux + CUDA only). "
            "For CPU/Mac, use moe_dispatch='stacked' in your config."
        )
    return _MoEW1W3SiluFunction.apply(
        x_sorted, expert_ids_sorted, counts, offsets, W1_stack, W3_stack,
    )
