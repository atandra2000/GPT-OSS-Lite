"""Tiny smoke tests — CPU-only, fast (<5s), no data download."""
import pytest
import torch

from models.transformer import GPTOSS, ModelConfig


def test_tiny_forward_cpu():
    """Tiny model (vocab=256, d=64, 2 layers) must run forward on CPU."""
    cfg = ModelConfig(
        vocab_size=256,
        d_model=64,
        n_layers=2,
        n_heads=4,
        n_kv_heads=2,
        head_dim=16,
        ffn_dim=128,
        n_routed_experts=2,
        n_activated_experts=1,
        n_shared_experts=1,
        window_size=16,
        max_seq_len=32,
        rope_theta=10000,
        yarn_scale_factor=2,
        yarn_original_max_seq_len=32,
        yarn_target_seq_len=64,
        yarn_beta_fast=2,
        yarn_beta_slow=1,
        yarn_prune_rope_global=False,
    )
    model = GPTOSS(cfg)
    model.eval()  # disable any randomness from training-mode hooks
    idx = torch.randint(0, cfg.vocab_size, (1, cfg.max_seq_len))
    with torch.no_grad():
        logits, aux_loss = model(idx)
    assert logits.shape == (1, cfg.max_seq_len, cfg.vocab_size)
    assert torch.isfinite(logits).all()
    assert aux_loss.item() >= 0.0


def test_tiny_backward_cpu():
    """Tiny model must complete a backward pass on CPU."""
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
        yarn_scale_factor=2,
        yarn_original_max_seq_len=16,
        yarn_target_seq_len=32,
        yarn_beta_fast=2,
        yarn_beta_slow=1,
        yarn_prune_rope_global=False,
    )
    model = GPTOSS(cfg)
    idx = torch.randint(0, cfg.vocab_size, (1, cfg.max_seq_len))
    logits, aux_loss = model(idx)
    loss = logits.sum() + aux_loss
    loss.backward()
    # Confirm every param has a grad.
    for name, p in model.named_parameters():
        assert p.grad is not None, f"No grad for {name}"


def test_model_config_defaults():
    """ModelConfig() with no args should produce the production 502M config."""
    cfg = ModelConfig()
    assert cfg.vocab_size == 128000
    assert cfg.d_model == 768
    assert cfg.n_layers == 12
    assert cfg.n_routed_experts == 8
    assert cfg.n_activated_experts == 2
    assert cfg.window_size == 128


def test_model_repr_includes_param_count():
    """GPTOSS.__repr__ should include the parameter count (millions)."""
    cfg = ModelConfig(
        vocab_size=256,
        d_model=64,
        n_layers=2,
        n_heads=4,
        n_kv_heads=2,
        head_dim=16,
        ffn_dim=128,
        n_routed_experts=2,
        n_activated_experts=1,
        n_shared_experts=1,
        window_size=16,
        max_seq_len=32,
        rope_theta=10000,
        yarn_scale_factor=2,
        yarn_original_max_seq_len=32,
        yarn_target_seq_len=64,
        yarn_beta_fast=2,
        yarn_beta_slow=1,
        yarn_prune_rope_global=False,
    )
    model = GPTOSS(cfg)
    s = repr(model)
    assert "M" in s  # param count in millions
    assert "d_model=64" in s
    assert "window=16" in s