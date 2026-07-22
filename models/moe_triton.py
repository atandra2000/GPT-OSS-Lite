"""Fused grouped-GEMM Triton kernel for GPT-OSS-Lite MoE dispatch.

Fuses W1 / W3 projections + silu(g)*u for one expert into a single launch.
W2 stays in PyTorch (no activation to fuse). Public entry point:
`triton_moe_w1w3_silu(x_sorted, expert_ids_sorted, counts, offsets, W1_stack, W3_stack)`.
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
    """Pure-PyTorch reference: `silu(W1[e] @ x_chunk) * (W3[e] @ x_chunk)`; W2 stays in the caller."""
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
        BLOCK_D: tl.constexpr,
        BLOCK_F: tl.constexpr,
        N_EXPERTS: tl.constexpr,
    ):
        """One program per (expert, token-tile) — fuses W1, W3, silu, mul."""
        e = tl.program_id(0)
        t_blk = tl.program_id(1)

        cnt = tl.load(cnt_ptr + e)
        if cnt == 0:
            return
        off = tl.load(off_ptr + e)

        tok_in_blk = t_blk * BLOCK_D + tl.arange(0, BLOCK_D)
        tok_mask = tok_in_blk < cnt

        x_row = x_ptr + (off + tok_in_blk)[:, None] * stride_xt
        x_cols = tl.arange(0, BLOCK_F)
        x_blk = tl.load(
            x_row + x_cols[None, :] * stride_xd,
            mask=tok_mask[:, None] & (x_cols[None, :] < d_model),
            other=0.0,
        )

        w1_row = w1_ptr + e * stride_w1e + tl.arange(0, BLOCK_F)[:, None] * stride_w1f
        w1_cols = tl.arange(0, BLOCK_D)
        w1_blk = tl.load(
            w1_row + w1_cols[None, :] * stride_w1d,
            mask=(tl.arange(0, BLOCK_F)[:, None] < d_ff)
                  & (w1_cols[None, :] < d_model),
            other=0.0,
        )
        w3_blk = tl.load(
            w3_ptr + e * stride_w3e
            + tl.arange(0, BLOCK_F)[:, None] * stride_w3f
            + w1_cols[None, :] * stride_w3d,
            mask=(tl.arange(0, BLOCK_F)[:, None] < d_ff)
                  & (w1_cols[None, :] < d_model),
            other=0.0,
        )

        g = tl.dot(tl.trans(x_blk), w1_blk)
        u = tl.dot(tl.trans(x_blk), w3_blk)
        silu = g * tl.sigmoid(g)
        fused = (silu * u).to(out_ptr.dtype.element_ty)

        out_row = out_ptr + (off + tok_in_blk)[:, None] * stride_ot
        out_cols = tl.arange(0, BLOCK_F)
        out_mask = tok_mask[:, None] & (out_cols[None, :] < d_ff)
        tl.store(out_row + out_cols[None, :] * stride_of, fused, mask=out_mask)


class _MoEW1W3SiluFunction(torch.autograd.Function):
    """v1 reference-stub autograd: forward = Triton kernel, backward = ref."""

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

        BLOCK_D = triton.next_power_of_2(min(64, max(16, d_model)))
        BLOCK_F = triton.next_power_of_2(min(128, max(64, d_ff)))
        BLOCK_D = min(BLOCK_D, _MOE_DMODEL_HARD_CAP)
        BLOCK_F = min(BLOCK_F, _MOE_FFN_HARD_CAP)

        max_tokens = int(counts.max().item()) if counts.numel() else 0
        n_tiles = (max_tokens + BLOCK_D - 1) // BLOCK_D

        _moe_w1w3_silu_kernel[(n_experts, n_tiles)](
            x_sorted, expert_ids_sorted, counts, offsets,
            W1_stack, W3_stack, out,
            n_tokens, d_model, d_ff,
            x_sorted.stride(0), x_sorted.stride(1),
            0, 0,
            W1_stack.stride(0), W1_stack.stride(1), W1_stack.stride(2),
            W3_stack.stride(0), W3_stack.stride(1), W3_stack.stride(2),
            out.stride(0), out.stride(1),
            BLOCK_D=BLOCK_D, BLOCK_F=BLOCK_F,
            N_EXPERTS=n_experts,
            num_warps=4, num_stages=2,
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
    """Public entry point. Raises ImportError if Triton is unavailable."""
    if not HAS_TRITON:
        raise ImportError(
            "triton_moe_w1w3_silu requires the `triton` package. "
            "Install with `pip install triton` (Linux + CUDA only). "
            "For CPU/Mac, use moe_dispatch='stacked' in your config."
        )
    return _MoEW1W3SiluFunction.apply(
        x_sorted, expert_ids_sorted, counts, offsets, W1_stack, W3_stack,
    )
