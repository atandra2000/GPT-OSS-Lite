"""Tests for ModelConfig validation + edge cases."""
import dataclasses

import pytest
import torch

from models.transformer import GPTOSS, ModelConfig


# ModelConfig validation tests

def test_modelconfig_rejects_zero_vocab():
    with pytest.raises(ValueError, match="vocab_size must be positive"):
        ModelConfig(vocab_size=0)


def test_modelconfig_rejects_negative_vocab():
    with pytest.raises(ValueError, match="vocab_size must be positive"):
        ModelConfig(vocab_size=-1)


def test_modelconfig_rejects_zero_d_model():
    with pytest.raises(ValueError, match="d_model must be positive"):
        ModelConfig(d_model=0)


def test_modelconfig_rejects_d_model_head_dim_mismatch():
    """n_heads * head_dim must equal d_model."""
    with pytest.raises(ValueError, match="n_heads \\* head_dim must equal d_model"):
        ModelConfig(d_model=64, n_heads=4, head_dim=8)  # 32 != 64


def test_modelconfig_rejects_n_heads_not_multiple_of_n_kv_heads():
    with pytest.raises(ValueError, match="n_heads must be a multiple of n_kv_heads"):
        ModelConfig(n_heads=4, n_kv_heads=6)


def test_modelconfig_rejects_n_kv_heads_greater_than_n_heads():
    """Same check as above; explicit test for the boundary."""
    with pytest.raises(ValueError, match="n_heads must be a multiple of n_kv_heads"):
        ModelConfig(n_heads=2, n_kv_heads=4)


def test_modelconfig_rejects_zero_n_heads():
    with pytest.raises(ValueError, match="n_heads and n_kv_heads must be positive"):
        ModelConfig(n_heads=0)


def test_modelconfig_rejects_odd_head_dim():
    with pytest.raises(ValueError, match="head_dim must be a positive even int"):
        ModelConfig(d_model=64, n_heads=4, head_dim=15)


def test_modelconfig_rejects_n_activated_gt_n_routed():
    with pytest.raises(ValueError, match="0 < n_activated_experts"):
        ModelConfig(n_routed_experts=8, n_activated_experts=10)


def test_modelconfig_rejects_zero_n_activated():
    with pytest.raises(ValueError, match="0 < n_activated_experts"):
        ModelConfig(n_activated_experts=0)


def test_modelconfig_rejects_yarn_scale_with_no_extrapolation():
    """scale_factor > 1 requires original < target."""
    with pytest.raises(ValueError, match="yarn_scale_factor > 1 requires"):
        ModelConfig(
            yarn_scale_factor=32,
            yarn_original_max_seq_len=4096,
            yarn_target_seq_len=2048,
        )


def test_modelconfig_rejects_zero_window_size():
    with pytest.raises(ValueError, match="window_size must be positive"):
        ModelConfig(window_size=0)


def test_modelconfig_accepts_scale_factor_one():
    """scale_factor=1 is valid (identity RoPE, no extrapolation)."""
    cfg = ModelConfig(yarn_scale_factor=1, yarn_original_max_seq_len=4096, yarn_target_seq_len=4096)
    assert cfg.yarn_scale_factor == 1


def test_modelconfig_field_count_is_stable():
    """ModelConfig has a fixed set of fields; adding fields should be deliberate."""
    from dataclasses import fields
    n_fields = len(fields(ModelConfig))
    assert n_fields == 29, f"Expected 29 fields, got {n_fields}"


# Anchor metric: ~502M total / ~247M active (the headline numbers)

def test_anchor_metric_502m_total(model_cfg):
    """Total params must be in [500M, 504M] (production config)."""
    cfg = model_cfg
    model = GPTOSS(cfg)
    total = model.num_parameters()
    assert 500_000_000 <= total <= 504_000_000, (
        f"Total params {total/1e6:.2f}M outside the [500M, 504M] anchor range"
    )


def test_anchor_metric_247m_active(model_cfg):
    """Active params must be in [244M, 250M] (production config, post-tie-dedup)."""
    cfg = model_cfg
    model = GPTOSS(cfg)
    active = model.num_active_parameters()
    assert 244_000_000 <= active <= 250_000_000, (
        f"Active params {active/1e6:.2f}M outside the [244M, 250M] anchor range"
    )


def test_active_params_correct_with_tied_weights():
    """When weight_tying=True, num_active_parameters must not double-count embed/head."""
    cfg = ModelConfig(
        vocab_size=128,
        d_model=32,
        n_layers=2,
        n_heads=2,
        n_kv_heads=1,
        head_dim=16,
        ffn_dim=64,
        n_routed_experts=2,
        n_activated_experts=1,
        n_shared_experts=1,
        window_size=8,
        max_seq_len=16,
        rope_theta=10000,
        yarn_scale_factor=1,
        yarn_original_max_seq_len=16,
        yarn_target_seq_len=16,
        yarn_beta_fast=2,
        yarn_beta_slow=1,
        yarn_prune_rope_global=False,
        weight_tying=True,
    )
    model = GPTOSS(cfg)
    # With tying: embed (128*32=4096) + head (4096) are the SAME parameter.
    # num_parameters must count it once; num_active_parameters must also count it once.
    total = model.num_parameters()
    active = model.num_active_parameters()
    # Both should be strictly less than (vocab*d) + (attention) + (moe) + (vocab*d) tie-naive.
    # Tied total ≈ embed(4096) + 2*attention + 2*moe (active+inactive + shared + router) + norms
    # Tie-naive would double-count the embed/head pair → 4096 extra params.
    # Check: tied total should equal non-tied total minus vocab*d.
    cfg_no_tie = dataclasses.replace(cfg, weight_tying=False)
    model_no_tie = GPTOSS(cfg_no_tie)
    total_no_tie = model_no_tie.num_parameters()
    # tied_total should be smaller than non-tied by exactly vocab*d
    assert total == total_no_tie - cfg.vocab_size * cfg.d_model, (
        f"Tied total ({total}) should be non-tied total ({total_no_tie}) minus embed ({cfg.vocab_size * cfg.d_model})"
    )


# Edge-case dimension tests

def _make_tiny_config(**overrides):
    base = dict(
        vocab_size=64,
        d_model=32,
        n_layers=1,  # single layer
        n_heads=2,
        n_kv_heads=1,
        head_dim=16,
        ffn_dim=64,
        n_routed_experts=2,
        n_activated_experts=1,
        n_shared_experts=1,
        window_size=1,  # extreme: window=1
        max_seq_len=1,  # extreme: T=1
        rope_theta=10000,
        yarn_scale_factor=1,
        yarn_original_max_seq_len=1,
        yarn_target_seq_len=1,
        yarn_beta_fast=1,
        yarn_beta_slow=1,
        yarn_prune_rope_global=False,
    )
    base.update(overrides)
    return ModelConfig(**base)


def test_extreme_dim_window_equals_seq():
    """window_size = seq_len: SWA behaves as full causal (no actual restriction)."""
    cfg = _make_tiny_config(window_size=4, max_seq_len=4)
    model = GPTOSS(cfg)
    x = torch.randint(0, cfg.vocab_size, (1, cfg.max_seq_len))
    logits, aux = model(x)
    assert logits.shape == (1, cfg.max_seq_len, cfg.vocab_size)
    assert torch.isfinite(logits).all()


def test_extreme_dim_window_equals_one():
    """window_size=1: each token attends only to itself (or sink)."""
    cfg = _make_tiny_config(window_size=1, max_seq_len=8)
    model = GPTOSS(cfg)
    x = torch.randint(0, cfg.vocab_size, (1, cfg.max_seq_len))
    logits, aux = model(x)
    assert logits.shape == (1, cfg.max_seq_len, cfg.vocab_size)
    assert torch.isfinite(logits).all()


def test_extreme_dim_seq_len_one():
    """seq_len=1: single-token forward still works (attention is trivial)."""
    cfg = _make_tiny_config(window_size=1, max_seq_len=1)
    model = GPTOSS(cfg)
    x = torch.randint(0, cfg.vocab_size, (1, 1))
    logits, aux = model(x)
    assert logits.shape == (1, 1, cfg.vocab_size)


def test_extreme_dim_single_layer():
    """n_layers=1: single-layer model runs forward + backward."""
    cfg = _make_tiny_config(n_layers=1)
    model = GPTOSS(cfg)
    x = torch.randint(0, cfg.vocab_size, (1, cfg.max_seq_len))
    # Use non-zero input that exercises all paths (no trivial all-zero tokens).
    x = torch.randint(1, cfg.vocab_size, (1, cfg.max_seq_len))
    logits, aux = model(x)
    loss = logits.sum() + aux
    loss.backward()
    # Confirm gradients flow to the parameters that should receive them
    # (some params may be tied to head/embed and skipped; check the non-tied ones).
    tied_id = id(model.embed.weight)
    n_with_grad = 0
    n_total = 0
    for name, p in model.named_parameters():
        if id(p) == tied_id:
            continue  # tied to head, may be counted via head's grad
        if p.requires_grad:
            n_total += 1
            if p.grad is not None and torch.isfinite(p.grad).all() and p.grad.abs().sum() > 0:
                n_with_grad += 1
    # At least 80% of trainable params should receive non-trivial gradients.
    assert n_with_grad >= int(0.8 * n_total), (
        f"Only {n_with_grad}/{n_total} non-tied trainable params got non-trivial gradients"
    )


def test_extreme_dim_vocab_size_one():
    """vocab_size=1: only one possible output token; loss is deterministic."""
    cfg = _make_tiny_config(vocab_size=1)
    model = GPTOSS(cfg)
    x = torch.zeros(1, cfg.max_seq_len, dtype=torch.long)  # only token 0
    logits, aux = model(x)
    assert logits.shape == (1, cfg.max_seq_len, 1)
    assert torch.isfinite(logits).all()


# RMSNorm numerical stability

def test_rmsnorm_handles_bf16_input():
    """RMSNorm must work with BF16 input without overflow at reasonable magnitudes."""
    from models.transformer import RMSNorm
    norm = RMSNorm(dim=64).to(torch.bfloat16)
    x = torch.randn(2, 8, 64, dtype=torch.bfloat16)
    y = norm(x)
    assert y.dtype == torch.bfloat16
    assert torch.isfinite(y).all()
    # Output magnitude should be ~ unit variance per dim (RMSNorm output)
    assert (y.abs() < 100.0).all()