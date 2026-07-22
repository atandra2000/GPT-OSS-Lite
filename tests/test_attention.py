"""Sliding-window attention + learned sink bias correctness tests."""
import dataclasses
import math

import pytest
import torch
import torch.nn.functional as F

from models.attention import (
    GPTOSSAttention,
    full_causal_attention,
    manual_causal_attention,
    repeat_kv,
    sliding_window_attention,
)
from models.transformer import ModelConfig


# Sliding-window vs. full causal attention equivalence

def test_sliding_window_matches_full_small(attn_inputs_small, attn_small):
    """SWA must match full causal attention for positions within the window."""
    q, k, v = attn_inputs_small
    B, T, H, D = q.shape
    window = attn_small["window"]

    # Ground truth: full causal attention
    full = manual_causal_attention(q, k, v)

    # Sliding-window attention via SDPA
    sw = sliding_window_attention(q, k, v, window=window)

    # For positions where full causal mask fits in window (t < window), they must match.
    # The positions t >= window should differ (SW restricts to last `window` keys).
    for t in range(min(window, T)):
        assert torch.allclose(full[:, :, t], sw[:, :, t], atol=1e-5), \
            f"Mismatch at position t={t} (window={window}, T={T})"


def test_sliding_window_zeros_outside_window(attn_inputs_small, attn_small):
    """SWA must not attend to positions outside the window — verify via attn weights."""
    q, k, v = attn_inputs_small
    B, T, H, D = q.shape
    window = attn_small["window"]

    # Use manual_causal_attention with window to expose weights
    out = manual_causal_attention(q, k, v, window=window)

    # For position t > window-1, it must NOT see position j where j < t - window + 1
    # We can verify by zeroing those K positions and checking the output is unchanged.
    for t in range(window, min(T, window + 4)):
        # Zero out keys outside the window
        k_masked = k.clone()
        k_masked[:, :, :t - window + 1, :] = 0.0
        # Re-run with zeroed early keys
        out_masked = manual_causal_attention(q, k_masked, v, window=window)
        # If SWA correctly ignores early keys, output must match the unmasked run
        assert torch.allclose(out[:, :, t], out_masked[:, :, t], atol=1e-5), \
            f"SWA attended to keys outside window at t={t}"


def test_sliding_window_window_matches_full_inside(attn_inputs_small, attn_small):
    """For t < window, SWA should match full causal exactly (no information loss)."""
    q, k, v = attn_inputs_small
    T = q.shape[2]
    window = attn_small["window"]

    full = manual_causal_attention(q, k, v)
    sw = manual_causal_attention(q, k, v, window=window)

    # All positions t < window: full mask fits in window
    assert torch.allclose(full[:, :, :window], sw[:, :, :window], atol=1e-5)


# Learned sink bias tests

def test_sink_bias_absorbs_attention(attn_inputs_tiny):
    """Sink bias must add exp(sink_bias) to the softmax denominator."""
    q, k, v = attn_inputs_tiny

    # Case 1: no sink bias — pure causal softmax
    out_no_sink = manual_causal_attention(q, k, v)

    # Case 2: zero sink bias — denominator gains exp(0)=1 extra mass; attention softens.
    sink_zero = torch.zeros(q.shape[1])
    out_zero_sink = manual_causal_attention(q, k, v, sink_bias=sink_zero)

    # With sink=0, output should be slightly attenuated (more denominator mass).
    # Verify the outputs are *different* but both finite.
    assert torch.isfinite(out_no_sink).all()
    assert torch.isfinite(out_zero_sink).all()
    assert not torch.allclose(out_no_sink, out_zero_sink, atol=1e-6)


def test_sink_bias_high_value_collapses_attention(attn_inputs_tiny):
    """When sink_bias is large positive, attention mass goes mostly to the sink → output ≈ 0."""
    q, k, v = attn_inputs_tiny

    # sink_bias = +100 → exp(100) ≫ any reasonable attention score
    sink_huge = torch.full((q.shape[1],), 100.0)
    out = manual_causal_attention(q, k, v, sink_bias=sink_huge)
    # Output should be ~0 because V terms get weight exp(scores)/Z ≈ 0
    assert out.abs().max() < 1e-3, f"Sink did not absorb attention, max output = {out.abs().max()}"


def test_sink_bias_negative_value_behaves_like_no_sink(attn_inputs_tiny):
    """When sink_bias is very negative, exp(sink) → 0 → sink has no effect → matches no-sink."""
    q, k, v = attn_inputs_tiny

    out_no_sink = manual_causal_attention(q, k, v)
    sink_neg = torch.full((q.shape[1],), -100.0)
    out_neg = manual_causal_attention(q, k, v, sink_bias=sink_neg)
    assert torch.allclose(out_no_sink, out_neg, atol=1e-4)


def test_sink_bias_per_head_differs(attn_inputs_tiny):
    """Different per-head biases must produce different outputs per head."""
    q, k, v = attn_inputs_tiny

    sink_mixed = torch.tensor([0.0, 5.0, -2.0, 0.0])[:q.shape[1]]
    out = manual_causal_attention(q, k, v, sink_bias=sink_mixed)

    # Different heads should give different magnitudes if their biases differ
    head_mags = out.abs().mean(dim=(0, 2, 3))  # (H,)
    assert head_mags.std() > 1e-4, "Per-head biases had no effect on per-head outputs"


# GPTOSSAttention nn.Module tests

def test_attention_module_forward_shape(small_cfg, device):
    """GPTOSSAttention must produce same-shape output as input."""
    cfg = dataclasses.replace(small_cfg)
    layer = GPTOSSAttention(cfg, layer_idx=0).to(device).to(torch.float64)
    B, T = 2, cfg.max_seq_len
    x = torch.randn(B, T, cfg.d_model, device=device, dtype=torch.float64)
    out = layer(x)
    assert out.shape == (B, T, cfg.d_model)


def test_attention_module_grad_flow(small_cfg, device):
    """Gradients must flow through Q, KV, O, sink_bias parameters."""
    cfg = dataclasses.replace(small_cfg)
    layer = GPTOSSAttention(cfg, layer_idx=0).to(device).to(torch.float64)
    B, T = 1, cfg.max_seq_len
    x = torch.randn(B, T, cfg.d_model, device=device, dtype=torch.float64, requires_grad=True)
    out = layer(x)
    out.sum().backward()
    assert layer.q_proj.weight.grad is not None
    assert layer.kv_proj.weight.grad is not None
    assert layer.o_proj.weight.grad is not None
    assert layer.sink_bias.grad is not None
    assert x.grad is not None


def test_attention_module_alternating_pattern(small_cfg, device):
    """Even layers are SWA, odd layers are full attention."""
    cfg = dataclasses.replace(small_cfg)
    layer_even = GPTOSSAttention(cfg, layer_idx=0)
    layer_odd = GPTOSSAttention(cfg, layer_idx=1)
    assert layer_even.is_windowed is True
    assert layer_odd.is_windowed is False
    # Global layer must have non-zero n_pruned_dims (if prune_rope enabled)
    assert layer_odd._n_pruned_dims() > 0
    assert layer_even._n_pruned_dims() == 0


def test_attention_module_sink_learned(small_cfg, device):
    """Sink bias must be a learnable nn.Parameter."""
    cfg = dataclasses.replace(small_cfg)
    layer = GPTOSSAttention(cfg, layer_idx=0)
    assert isinstance(layer.sink_bias, torch.nn.Parameter)
    assert layer.sink_bias.shape == (cfg.n_heads,)
    # Initialized to 0 (logit=1 → standard softmax denominator)
    assert torch.allclose(layer.sink_bias.detach(), torch.zeros(cfg.n_heads))


def test_attention_sink_affects_output(small_cfg, device):
    """Changing sink_bias after init must change the output."""
    cfg = dataclasses.replace(small_cfg)
    layer = GPTOSSAttention(cfg, layer_idx=0).to(device).to(torch.float64)
    B, T = 1, cfg.max_seq_len
    x = torch.randn(B, T, cfg.d_model, device=device, dtype=torch.float64)
    out1 = layer(x).clone()

    with torch.no_grad():
        layer.sink_bias.fill_(2.0)
    out2 = layer(x).clone()

    assert not torch.allclose(out1, out2, atol=1e-4), \
        "Sink bias perturbation had no effect on output"


# GQA repeat_kv test

def test_repeat_kv_identity():
    """n_rep=1 should return input unchanged."""
    x = torch.randn(2, 4, 16, 32)
    y = repeat_kv(x, n_rep=1)
    assert y.shape == x.shape
    assert torch.equal(x, y)


def test_repeat_kv_doubles():
    """n_rep=2 should double the head dim, replicating each head twice."""
    x = torch.randn(2, 4, 16, 32)
    y = repeat_kv(x, n_rep=2)
    assert y.shape == (2, 8, 16, 32)
    # Each pair of consecutive heads must be a duplicate of the same source head.
    for h in range(4):
        assert torch.allclose(y[:, 2 * h], y[:, 2 * h + 1]), \
            f"Heads {2*h} and {2*h+1} should be identical (GQA replicate)"
    # Adjacent pairs must NOT be equal (different source heads).
    assert not torch.allclose(y[:, 0], y[:, 2])


# Production-scale (slow on CPU)

@pytest.mark.slow
def test_sliding_window_matches_full_large(attn_large):
    """SWA must match full causal at production-scale dims."""
    B, T, H, D, window = (attn_large[k] for k in ("B", "T", "H", "D", "window"))
    torch.manual_seed(0)
    q = torch.randn(B, H, T, D)
    k = torch.randn(B, H, T, D)
    v = torch.randn(B, H, T, D)
    full = manual_causal_attention(q, k, v)
    sw = sliding_window_attention(q, k, v, window=window)
    assert torch.allclose(full[:, :, :window], sw[:, :, :window], atol=1e-5)


# Sink bias + sliding window combination (the load-bearing integration)

def test_sink_and_window_compose_zero_bias():
    """When sink_bias=0, SWA-with-sink must behave similarly to SWA-without-sink."""
    torch.manual_seed(0)
    q = torch.randn(2, 4, 32, 16)
    k = torch.randn(2, 4, 32, 16)
    v = torch.randn(2, 4, 32, 16)
    sink_zero = torch.zeros(4)
    out_no_sink = sliding_window_attention(q, k, v, window=8, sink_bias=None)
    out_zero_sink = sliding_window_attention(q, k, v, window=8, sink_bias=sink_zero)
    # The magnitude should be reduced (by factor Z/(Z+1)) but the sign preserved.
    # Verify output_damped == output_normal * (something < 1, > 0).
    # Magnitudes ratio should be in (0, 1).
    ratio = out_zero_sink.abs().mean() / out_no_sink.abs().mean()
    assert 0.0 < ratio < 1.0, f"Magnitude ratio {ratio} outside (0, 1)"


def test_sink_and_window_compose_high_bias_dampens():
    """sink_bias=20 inside window must dampen attention (mass goes to sink)."""
    q = torch.randn(1, 2, 16, 8)
    k = torch.randn(1, 2, 16, 8)
    v = torch.randn(1, 2, 16, 8)
    sink_high = torch.full((2,), 20.0)
    out_normal = sliding_window_attention(q, k, v, window=8, sink_bias=None)
    out_damped = sliding_window_attention(q, k, v, window=8, sink_bias=sink_high)
    # With huge sink, exp(20) dominates → attention mass goes to sink → output ≈ 0.
    assert out_damped.abs().max() < out_normal.abs().max() * 0.5


def test_sink_bias_clamped_at_forward(small_cfg, device):
    """GPTOSSAttention must clamp sink_bias before use (preventing BF16 overflow)."""
    cfg = dataclasses.replace(small_cfg)
    layer = GPTOSSAttention(cfg, layer_idx=0).to(device).to(torch.float64)
    # Set sink_bias to a value beyond the clamp range.
    with torch.no_grad():
        layer.sink_bias.fill_(1000.0)
    # Verify the parameter itself is unchanged (clamp is at forward time).
    assert (layer.sink_bias == 1000.0).all()
    # Forward should NOT produce NaN even with this extreme value (clamp to SINK_MAX=15).
    x = torch.randn(1, cfg.max_seq_len, cfg.d_model, device=device, dtype=torch.float64)
    out = layer(x)
    assert torch.isfinite(out).all(), "Forward produced non-finite output with extreme sink_bias"