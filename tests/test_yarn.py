"""YaRN RoPE scaling + pruned RoPE correctness tests."""
import math

import pytest
import torch

from models.rotary import (
    apply_rope,
    compute_yarn_freqs,
    compute_yarn_mscale,
)
from models.yarn import YaRNRoPE


# YaRN frequency computation

def test_yarn_freqs_shape(yarn_cfg_small):
    """YaRN freqs must be (head_dim // 2,)."""
    inv_freq = compute_yarn_freqs(**yarn_cfg_small)
    assert inv_freq.shape == (yarn_cfg_small["head_dim"] // 2,)


def test_yarn_freqs_low_high_spread(yarn_cfg_small):
    """Low-frequency dims should be scaled; high-frequency dims should be unchanged."""
    inv_freq = compute_yarn_freqs(**yarn_cfg_small)
    half = inv_freq.shape[0]
    # Without YaRN scaling, the spread is enormous (1 → tiny).
    # After YaRN scaling, the lowest-frequency dims (idx near half-1) are divided
    # by scale_factor, making them even smaller. The high-frequency end (idx 0)
    # is unchanged.
    assert inv_freq[0] > inv_freq[-1]
    # inv_freq[0] (high freq) should equal the base: 1.0 (since theta^0 = 1).
    assert abs(inv_freq[0] - 1.0) < 1e-4
    # inv_freq[-1] (low freq) should be much smaller than the unscaled base,
    # because it has been divided by ~scale_factor.
    base_low = 1.0 / (yarn_cfg_small["theta"] ** ((half - 1) * 2.0 / yarn_cfg_small["head_dim"]))
    assert inv_freq[-1] < base_low * 0.5


def test_yarn_freqs_no_nan(yarn_cfg):
    """YaRN frequencies must be finite at production scale (128K)."""
    inv_freq = compute_yarn_freqs(**yarn_cfg)
    assert torch.isfinite(inv_freq).all()
    assert (inv_freq > 0).all()


def test_yarn_mscale_basic():
    """mscale should be 1.0 when scale_factor=1, and increase with scale_factor."""
    assert compute_yarn_mscale(1.0) == 1.0
    m1 = compute_yarn_mscale(4.0)
    m2 = compute_yarn_mscale(32.0)
    assert m1 > 1.0
    assert m2 > m1  # larger scale_factor → larger mscale


# YaRNRoPE module — forward pass

def test_yarn_module_no_nan_128k(yarn_cfg):
    """YaRN at position 131072 must produce no NaN."""
    rope = YaRNRoPE(**yarn_cfg)
    cos, sin = rope(torch.tensor([131072]))
    assert torch.isfinite(cos).all()
    assert torch.isfinite(sin).all()


def test_yarn_module_4k_vs_128k_distinct(yarn_cfg):
    """Positions 4K and 128K must produce distinct rotations."""
    rope = YaRNRoPE(**yarn_cfg)
    cos_4k, sin_4k = rope(torch.tensor([4096]))
    cos_128k, sin_128k = rope(torch.tensor([131072]))
    assert not torch.allclose(cos_4k, cos_128k, atol=1e-3)
    assert not torch.allclose(sin_4k, sin_128k, atol=1e-3)


def test_yarn_module_zero_position_is_identity(yarn_cfg_small):
    """Position 0 must give cos=mscale, sin=0 (no rotation at start, mscale applied)."""
    rope = YaRNRoPE(**yarn_cfg_small)
    cos, sin = rope(torch.tensor([0]))
    assert torch.allclose(cos, torch.full_like(cos, rope.mscale), atol=1e-5)
    assert torch.allclose(sin, torch.zeros_like(sin), atol=1e-5)


def test_yarn_module_cos_sin_pair(yarn_cfg_small):
    """cos^2 + sin^2 must equal mscale^2 for every (position, dim)."""
    rope = YaRNRoPE(**yarn_cfg_small)
    cos, sin = rope(torch.tensor([10, 100, 200]))
    mscale_sq = rope.mscale ** 2
    assert torch.allclose(cos ** 2 + sin ** 2, torch.full_like(cos, mscale_sq), atol=1e-5)


def test_yarn_module_position_monotonic(yarn_cfg_small):
    """High-frequency dims should rotate visibly across positions."""
    rope = YaRNRoPE(**yarn_cfg_small)
    positions = torch.arange(0, 64)
    cos, _ = rope(positions)
    # High-frequency dims (small idx) should show significant rotation over 64 positions.
    # We check that the max std across dims is meaningful (at least the high-freq dims rotate).
    max_std = cos.std(dim=0).max()
    assert max_std > 0.1, f"No high-frequency dim showed visible rotation (max std = {max_std})"


def test_yarn_module_pruned_dims(yarn_cfg):
    """YaRN with n_pruned_dims must zero out those dims."""
    rope = YaRNRoPE(**yarn_cfg)
    n_pruned = yarn_cfg["head_dim"] // 4  # 24 of 96
    cos, sin = rope(torch.tensor([100]), n_pruned_dims=n_pruned)
    assert torch.allclose(cos[:, :n_pruned], torch.ones(n_pruned), atol=1e-5)
    assert torch.allclose(sin[:, :n_pruned], torch.zeros(n_pruned), atol=1e-5)
    # Other dims unchanged
    assert not torch.allclose(cos[:, n_pruned:], torch.ones(cos.shape[1] - n_pruned))


# apply_rope

def test_apply_rope_zero_rotation():
    """cos=1, sin=0 must leave input unchanged."""
    x = torch.randn(2, 4, 8, 16)         # (B, H, T=8, head_dim=16)
    cos = torch.ones(8, 8)               # (T=8, head_dim//2=8)
    sin = torch.zeros(8, 8)
    y = apply_rope(x, cos, sin)
    assert torch.allclose(x, y, atol=1e-5)


def test_apply_rope_shape_preserved():
    """apply_rope must preserve shape."""
    x = torch.randn(2, 4, 16, 32)        # (B, H, T=16, head_dim=32)
    cos = torch.randn(16, 16)            # (T=16, head_dim//2=16)
    sin = torch.randn(16, 16)
    y = apply_rope(x, cos, sin)
    assert y.shape == x.shape


def test_apply_rope_magnitude_preserved():
    """Rotations preserve vector magnitude per-pair when cos^2+sin^2=1."""
    x = torch.randn(2, 4, 8, 16)         # (B, H, T=8, head_dim=16)
    # Use cos/sin sampled on the unit circle so cos^2 + sin^2 = 1 per dim.
    angles = torch.randn(8, 8)
    cos = angles.cos()
    sin = angles.sin()
    y = apply_rope(x, cos, sin)
    # Reshape to pairs and verify magnitude preserved per pair (rotations preserve norm).
    x_pairs = x.unflatten(-1, (-1, 2))
    y_pairs = y.unflatten(-1, (-1, 2))
    assert torch.allclose(x_pairs.pow(2).sum(-1), y_pairs.pow(2).sum(-1), atol=1e-5)


# Degenerate ramp warning

def test_compute_yarn_freqs_warns_on_degenerate_ramp():
    """Extreme beta_fast/beta_slow must emit a UserWarning (not silently fail)."""
    from models.rotary import compute_yarn_freqs
    # Pick parameters such that high <= low (the degenerate condition).
    # With original_max_seq_len=8 and beta_slow=64, the log2 ratio becomes tiny → low >= half.
    with pytest.warns(UserWarning, match="YaRN ramp degenerate"):
        # Use small head_dim and large beta_slow to force degenerate ramp
        compute_yarn_freqs(
            head_dim=8, theta=10000, scale_factor=4,
            original_max_seq_len=8, target_seq_len=32,
            beta_fast=64, beta_slow=64,
        )


def test_compute_yarn_freqs_no_warning_for_normal_params():
    """Normal YaRN params must NOT emit a warning."""
    from models.rotary import compute_yarn_freqs
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("error")  # treat warnings as errors
        compute_yarn_freqs(
            head_dim=64, theta=10000, scale_factor=4,
            original_max_seq_len=128, target_seq_len=512,
            beta_fast=4, beta_slow=1,
        )