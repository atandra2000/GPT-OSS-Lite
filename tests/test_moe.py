"""MoE routing + dispatch + aux load-balancing loss correctness tests."""
import pytest
import torch

from models.moe import (
    MoELayer,
    MoERouter,
    SwiGLUExpert,
    aux_load_balancing_loss,
)


# SwiGLU expert

def test_swiglu_expert_shape():
    """SwiGLU expert must produce same-shape output as input."""
    expert = SwiGLUExpert(dim=16, inter_dim=64)
    x = torch.randn(2, 8, 16)
    y = expert(x)
    assert y.shape == (2, 8, 16)


def test_swiglu_expert_grad_flow():
    """Gradients must flow through W1, W2, W3."""
    expert = SwiGLUExpert(dim=8, inter_dim=32)
    x = torch.randn(2, 4, 8, requires_grad=True)
    y = expert(x)
    y.sum().backward()
    assert expert.w1.weight.grad is not None
    assert expert.w2.weight.grad is not None
    assert expert.w3.weight.grad is not None
    assert x.grad is not None


# Router

def test_router_topk_indices():
    """Top-k routing must return exactly k indices per token."""
    router = MoERouter(d_model=16, n_experts=8, n_activated=2)
    x = torch.randn(4, 16)
    indices, weights, logits = router(x)
    assert indices.shape == (4, 2)
    assert weights.shape == (4, 2)
    assert logits.shape == (4, 8)


def test_router_weights_sum_to_one():
    """Top-k weights must sum to 1.0 per token."""
    router = MoERouter(d_model=16, n_experts=8, n_activated=2)
    x = torch.randn(4, 16)
    _, weights, _ = router(x)
    sums = weights.sum(dim=-1)
    assert torch.allclose(sums, torch.ones_like(sums), atol=1e-5)


def test_router_indices_in_range():
    """Top-k indices must be in [0, n_experts)."""
    router = MoERouter(d_model=16, n_experts=8, n_activated=2)
    x = torch.randn(4, 16)
    indices, _, _ = router(x)
    assert (indices >= 0).all()
    assert (indices < 8).all()


def test_router_grad_flow():
    """Gradients must flow through the router gate."""
    router = MoERouter(d_model=16, n_experts=8, n_activated=2)
    x = torch.randn(4, 16, requires_grad=True)
    _, weights, _ = router(x)
    weights.sum().backward()
    assert router.gate.weight.grad is not None
    assert x.grad is not None


# Aux load-balancing loss

def test_aux_loss_finite_and_nonneg():
    """Aux loss must be finite and non-negative."""
    torch.manual_seed(0)
    logits = torch.randn(100, 8)
    loss = aux_load_balancing_loss(logits, n_experts=8, n_activated=2)
    assert torch.isfinite(loss)
    assert loss >= 0.0


def test_aux_loss_low_for_uniform():
    """Uniform routing logits should yield low aux loss."""
    # All logits equal → softmax is uniform → P_i = 1/n_experts for all i.
    # Routing fraction f_i depends on tie-breaking but is ~ uniform as well.
    logits = torch.zeros(1000, 8)  # uniform logits
    loss_uniform = aux_load_balancing_loss(logits, n_experts=8, n_activated=2)
    # Collapsed: huge bias toward one expert
    logits_collapsed = torch.zeros(1000, 8)
    logits_collapsed[:, 0] = 10.0
    loss_collapsed = aux_load_balancing_loss(logits_collapsed, n_experts=8, n_activated=2)
    # Uniform should have lower aux loss than collapsed.
    assert loss_uniform < loss_collapsed


def test_aux_loss_grad_flow():
    """Aux loss must be differentiable w.r.t. logits."""
    logits = torch.randn(10, 8, requires_grad=True)
    loss = aux_load_balancing_loss(logits, n_experts=8, n_activated=2)
    loss.backward()
    assert logits.grad is not None
    assert torch.isfinite(logits.grad).all()


# Full MoELayer

def test_moe_layer_shapes(small_cfg):
    """MoELayer forward must return (output, aux_loss) with correct shapes."""
    cfg = small_cfg
    moe = MoELayer(cfg)
    B, T = 2, cfg["max_seq_len"]
    x = torch.randn(B, T, cfg["d_model"])
    out, aux = moe(x)
    assert out.shape == (B, T, cfg["d_model"])
    assert aux.shape == ()
    assert aux >= 0.0
    assert torch.isfinite(aux)


def test_moe_layer_shared_expert_active(small_cfg):
    """The shared expert must always be active (contributes to every token)."""
    cfg = small_cfg
    moe = MoELayer(cfg)
    # Forward pass should use shared expert unconditionally.
    x = torch.randn(1, 4, cfg["d_model"])
    # If we set shared expert to zero, the output should differ from baseline.
    base_out, _ = moe(x)
    with torch.no_grad():
        for p in moe.shared_experts.parameters():
            p.zero_()
    zeroed_out, _ = moe(x)
    # When shared expert is zeroed, output must change (proves it was active).
    assert not torch.allclose(base_out, zeroed_out, atol=1e-4)


def test_moe_layer_grad_flow(small_cfg):
    """Gradients must flow through router, routed experts, and shared expert."""
    cfg = small_cfg
    moe = MoELayer(cfg)
    x = torch.randn(2, cfg["max_seq_len"], cfg["d_model"], requires_grad=True)
    out, aux = moe(x)
    (out.sum() + aux).backward()
    # Router
    assert moe.router.gate.weight.grad is not None
    # All routed experts
    for i, expert in enumerate(moe.experts):
        assert expert.w1.weight.grad is not None, f"expert {i} w1 has no grad"
    # Shared expert
    assert moe.shared_experts[0].w1.weight.grad is not None
    assert x.grad is not None


def test_moe_layer_dispatch_correct(small_cfg):
    """Tokens routed to the same expert should be combined with the right weights."""
    cfg = small_cfg
    moe = MoELayer(cfg)
    moe.eval()  # disable any dropout / mode switches
    B, T = 1, cfg["max_seq_len"]
    torch.manual_seed(123)
    x = torch.randn(B, T, cfg["d_model"])

    with torch.no_grad():
        # Manually compute the MoE output
        flat = x.view(-1, cfg["d_model"])
        indices, weights, _ = moe.router(flat)
        # Build expected output
        expected = torch.zeros_like(flat)
        for t in range(flat.size(0)):
            for k in range(2):
                e = indices[t, k].item()
                w = weights[t, k].item()
                expected[t] = expected[t] + w * moe.experts[e](flat[t])
        # Add shared expert
        for s in moe.shared_experts:
            expected = expected + s(flat)
        expected = expected.view(B, T, cfg["d_model"])

    actual, _ = moe(x)
    assert torch.allclose(actual, expected, atol=1e-4)


def test_moe_layer_routes_to_all_experts_over_batch(small_cfg):
    """Over a large batch, routing should reach multiple experts (not collapse to one).

    Note: with random initialization and only an aux loss (no bias-update), it's
    normal for 3-5 experts to dominate initially. We just check that the
    routing is not collapsed to a single expert.
    """
    cfg = small_cfg
    moe = MoELayer(cfg)
    # Large enough batch that some probability mass reaches each expert.
    x = torch.randn(64, cfg["max_seq_len"], cfg["d_model"])
    _, _, all_logits = moe.router(x.view(-1, cfg["d_model"]))
    probs = torch.softmax(all_logits, dim=-1)
    expert_mass = probs.sum(dim=0)
    # At least 2 experts should receive meaningful mass (not collapsed to 1).
    n_meaningful = (expert_mass > expert_mass.mean() * 0.25).sum().item()
    assert n_meaningful >= 2, f"Routing collapsed to {n_meaningful}/8 experts"
    # The maximum mass fraction should be < 0.9 (otherwise it's a near-collapse).
    max_frac = (expert_mass / expert_mass.sum()).max().item()
    assert max_frac < 0.9, f"Near-collapse: max expert mass = {max_frac:.3f}"


# FP32 stability + stable sort determinism

def test_aux_loss_robust_to_bf16_saturation():
    """Aux loss must remain non-zero even when router logits saturate (BF16 issue)."""
    from models.moe import aux_load_balancing_loss
    # Saturated logits: one expert dominates heavily. In naive BF16, the small
    # probabilities for non-dominant experts underflow to 0, silently zeroing
    # the aux loss. The FP32 softmax path must still report a non-zero loss.
    logits = torch.tensor([
        [100.0, -100.0, -100.0, -100.0, -100.0, -100.0, -100.0, -100.0],
        [100.0, -100.0, -100.0, -100.0, -100.0, -100.0, -100.0, -100.0],
        [100.0, -100.0, -100.0, -100.0, -100.0, -100.0, -100.0, -100.0],
        [100.0, -100.0, -100.0, -100.0, -100.0, -100.0, -100.0, -100.0],
    ])
    loss = aux_load_balancing_loss(logits, n_experts=8, n_activated=2)
    assert loss > 0.0, f"Aux loss vanished under saturation (got {loss.item()})"


def test_moe_dispatch_is_deterministic(small_cfg):
    """Two forward passes with the same input must produce identical output."""
    torch.manual_seed(123)
    cfg = small_cfg
    moe = MoELayer(cfg)
    moe.eval()
    x = torch.randn(8, cfg["max_seq_len"], cfg["d_model"])
    with torch.no_grad():
        out1, _ = moe(x)
        out2, _ = moe(x)
    assert torch.equal(out1, out2), "MoE dispatch is non-deterministic across identical inputs"