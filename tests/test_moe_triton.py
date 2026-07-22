"""Fused Triton MoE grouped-GEMM kernel tests."""
import pytest
import torch
import torch.nn.functional as F
from dataclasses import replace

from models.moe import MoELayer
from models.moe_triton import (
    HAS_TRITON,
    _moe_w1w3_silu_reference,
    triton_moe_w1w3_silu,
)
from models.transformer import ModelConfig


# Reference cross-check (always CPU-runnable)

def test_reference_matches_naive_per_expert_loop():
    """The pure-PyTorch reference must match a naive per-expert matmul + silu loop."""
    torch.manual_seed(0)
    n_tokens, d_model, d_ff, n_experts = 8, 16, 32, 4
    x_sorted = torch.randn(n_tokens, d_model, dtype=torch.float64)
    counts = torch.tensor([2, 2, 2, 2])
    offsets = torch.tensor([0, 2, 4, 6])
    eids = torch.repeat_interleave(torch.arange(n_experts), counts)
    W1 = torch.randn(n_experts, d_ff, d_model, dtype=torch.float64)
    W3 = torch.randn(n_experts, d_ff, d_model, dtype=torch.float64)
    out = _moe_w1w3_silu_reference(x_sorted, eids, counts, offsets, W1, W3)
    for e in range(n_experts):
        s, c = int(offsets[e]), int(counts[e])
        if c == 0:
            continue
        chunk = x_sorted[s:s+c]
        g = chunk @ W1[e].T
        u = chunk @ W3[e].T
        assert torch.allclose(out[s:s+c], F.silu(g) * u, atol=1e-10)


def test_reference_handles_empty_experts():
    """An expert with count=0 must produce a well-defined row in the output buffer."""
    torch.manual_seed(0)
    n_tokens, d_model, d_ff, n_experts = 4, 8, 16, 4
    x_sorted = torch.randn(n_tokens, d_model, dtype=torch.float64)
    counts = torch.tensor([2, 0, 2, 0])
    offsets = torch.tensor([0, 2, 2, 4])
    eids = torch.tensor([0, 0, 2, 2])
    W1 = torch.randn(n_experts, d_ff, d_model, dtype=torch.float64)
    W3 = torch.randn(n_experts, d_ff, d_model, dtype=torch.float64)
    out = _moe_w1w3_silu_reference(x_sorted, eids, counts, offsets, W1, W3)
    assert out.shape == (n_tokens, d_ff)
    assert torch.isfinite(out).all()


def test_reference_matches_existing_moe_dispatch_shape(small_cfg):
    """The reference must produce the (N, d_ff) shape that the dispatch expects."""
    cfg = small_cfg
    moe = MoELayer(cfg)
    flat = torch.randn(8, cfg.d_model)
    indices, weights, _ = moe.router(flat)
    N = flat.size(0)
    flat_idx = indices.reshape(-1)
    token_ids = torch.arange(N).repeat_interleave(indices.size(1))
    order = torch.argsort(flat_idx, stable=True)
    sorted_token_ids = token_ids[order]
    x_sorted = flat[sorted_token_ids]
    counts = torch.bincount(flat_idx, minlength=moe.n_routed)
    offsets = torch.cat([torch.zeros(1, dtype=counts.dtype), counts.cumsum(0)[:-1]])
    W1 = torch.stack([e.w1.weight for e in moe.experts], 0)
    W3 = torch.stack([e.w3.weight for e in moe.experts], 0)
    out = _moe_w1w3_silu_reference(x_sorted, flat_idx[order], counts, offsets, W1, W3)
    assert out.shape == (N * moe.n_activated, cfg.ffn_dim)


# Public function surface

def test_triton_moe_raises_when_triton_missing(monkeypatch):
    """When triton is not installed, the public function must raise ImportError."""
    import models.moe_triton as mod
    monkeypatch.setattr(mod, "HAS_TRITON", False)
    with pytest.raises(ImportError, match="triton"):
        triton_moe_w1w3_silu(
            torch.randn(4, 8), torch.zeros(4, dtype=torch.long),
            torch.tensor([2, 2]), torch.tensor([0, 2]),
            torch.randn(2, 16, 8), torch.randn(2, 16, 8),
        )


def test_triton_moe_raises_on_hard_cap_violation(monkeypatch):
    """d_ff > 8192 or d_model > 8192 must hard-fail with ValueError."""
    import models.moe_triton as mod
    monkeypatch.setattr(mod, "HAS_TRITON", True)
    monkeypatch.setattr(mod, "_MOE_FFN_HARD_CAP", 32)
    with pytest.raises(ValueError, match="hard cap"):
        triton_moe_w1w3_silu(
            torch.randn(2, 8), torch.zeros(2, dtype=torch.long),
            torch.tensor([1, 1]), torch.tensor([0, 1]),
            torch.randn(2, 64, 8), torch.randn(2, 64, 8),
        )


# MoELayer dispatch wiring

def test_MoELayer_default_moe_dispatch_is_stacked(small_cfg):
    """Default moe_dispatch must be 'stacked' (the existing path)."""
    cfg = small_cfg
    moe = MoELayer(cfg)
    assert moe.moe_dispatch == "stacked"


def test_MoELayer_triton_dispatch_raises_when_triton_missing(small_cfg):
    """moe_dispatch='triton_grouped' must raise ImportError when triton not installed."""
    cfg = replace(small_cfg, moe_dispatch="triton_grouped")
    moe = MoELayer(cfg)
    with pytest.raises(ImportError, match="triton"):
        moe(torch.randn(2, cfg.max_seq_len, cfg.d_model))


# GPU-gated tests — exercise the actual Triton kernel on CUDA

gpu_required = pytest.mark.skipif(
    not (HAS_TRITON and torch.cuda.is_available()),
    reason="requires triton + CUDA",
)


@gpu_required
def test_kernel_forward_matches_reference_fp32():
    """On GPU + fp32, the kernel must match the reference within 1e-3."""
    torch.manual_seed(0)
    n_tokens, d_model, d_ff, n_experts = 64, 64, 128, 4
    x_sorted = torch.randn(n_tokens, d_model, device="cuda", dtype=torch.float32)
    counts = torch.tensor([16, 16, 16, 16], device="cuda")
    offsets = torch.tensor([0, 16, 32, 48], device="cuda")
    eids = torch.repeat_interleave(torch.arange(n_experts, device="cuda"), torch.tensor([16, 16, 16, 16], device="cuda"))
    W1 = torch.randn(n_experts, d_ff, d_model, device="cuda", dtype=torch.float32)
    W3 = torch.randn(n_experts, d_ff, d_model, device="cuda", dtype=torch.float32)
    y_ref = _moe_w1w3_silu_reference(x_sorted, eids, counts, offsets, W1, W3)
    y_tri = triton_moe_w1w3_silu(x_sorted, eids, counts, offsets, W1, W3)
    assert torch.allclose(y_ref, y_tri, atol=1e-3, rtol=1e-3)


@gpu_required
def test_kernel_forward_matches_reference_bf16():
    """On GPU + bf16, the kernel must match the reference within 1e-2."""
    torch.manual_seed(0)
    n_tokens, d_model, d_ff, n_experts = 64, 64, 128, 4
    x_sorted = torch.randn(n_tokens, d_model, device="cuda", dtype=torch.bfloat16)
    counts = torch.tensor([16, 16, 16, 16], device="cuda")
    offsets = torch.tensor([0, 16, 32, 48], device="cuda")
    eids = torch.repeat_interleave(torch.arange(n_experts, device="cuda"), torch.tensor([16, 16, 16, 16], device="cuda"))
    W1 = torch.randn(n_experts, d_ff, d_model, device="cuda", dtype=torch.bfloat16)
    W3 = torch.randn(n_experts, d_ff, d_model, device="cuda", dtype=torch.bfloat16)
    y_ref = _moe_w1w3_silu_reference(x_sorted, eids, counts, offsets, W1, W3)
    y_tri = triton_moe_w1w3_silu(x_sorted, eids, counts, offsets, W1, W3)
    assert torch.allclose(y_ref, y_tri, atol=1e-2, rtol=1e-2)
