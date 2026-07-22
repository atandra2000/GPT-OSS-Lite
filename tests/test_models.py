"""GPT-OSS-Lite full-model tests: shape, param count, weight tying, grad flow, overfit."""
import dataclasses

import pytest
import torch

from models.transformer import GPTOSS, ModelConfig


# Forward shape + smoke test

def test_forward_shape_small(small_cfg):
    """GPTOSS forward must return (logits, aux_loss) with correct shapes."""
    cfg = small_cfg
    model = GPTOSS(cfg)
    B, T = 2, cfg.max_seq_len
    idx = torch.randint(0, cfg.vocab_size, (B, T))
    logits, aux_loss = model(idx)
    assert logits.shape == (B, T, cfg.vocab_size)
    assert aux_loss.shape == ()
    assert torch.isfinite(logits).all()
    assert torch.isfinite(aux_loss)


def test_forward_returns_aux_loss(small_cfg):
    """Forward must return a non-zero aux loss (from MoE load-balancing)."""
    cfg = small_cfg
    model = GPTOSS(cfg)
    idx = torch.randint(0, cfg.vocab_size, (1, cfg.max_seq_len))
    _, aux_loss = model(idx)
    assert aux_loss > 0.0
    assert aux_loss.requires_grad


# Parameter count test (the ~502M budget)

def test_param_count_production(model_cfg):
    """Param count for the production config must be ~502M (±5%)."""
    cfg = model_cfg
    model = GPTOSS(cfg)
    n_params = model.num_parameters()
    # DESIGN §3 target: ~502M total
    assert 4.8e8 < n_params < 5.2e8, f"Param count {n_params/1e6:.1f}M outside 480-520M range"


def test_active_params_smaller_than_total(model_cfg):
    """Active params (top-2 + 1 shared) must be significantly smaller than total."""
    cfg = model_cfg
    model = GPTOSS(cfg)
    total = model.num_parameters()
    active = model.num_active_parameters()
    # MoE sparsity: 2 of 8 routed active + 1 shared → roughly 3/8 of expert params active.
    # Active should be ~50% of total (matches design spec ~247M / ~502M).
    ratio = active / total
    assert 0.40 < ratio < 0.60, f"Active/total ratio {ratio:.2f} outside expected range [0.40, 0.60]"


def test_param_count_small(small_cfg):
    """Smoke test: small_cfg should have a tiny param count."""
    cfg = small_cfg
    model = GPTOSS(cfg)
    n_params = model.num_parameters()
    # Sanity: small model shouldn't have more than 5M params
    assert n_params < 5e6, f"Small model has {n_params/1e6:.1f}M params (too many)"


# Weight tying

def test_weight_tying(small_cfg):
    """Embedding and head must share the same parameter (data_ptr equal)."""
    cfg = small_cfg
    cfg.weight_tying = True
    model = GPTOSS(cfg)
    assert model.head.weight.data_ptr() == model.embed.weight.data_ptr()


def test_weight_tying_disabled(small_cfg):
    """When weight_tying=False, embed and head must have separate params."""
    cfg = small_cfg
    cfg.weight_tying = False
    model = GPTOSS(cfg)
    assert model.head.weight.data_ptr() != model.embed.weight.data_ptr()


# Alternating layer pattern

def test_alternating_layer_pattern(small_cfg):
    """Even layers must be SWA; odd layers must be full attention."""
    cfg = small_cfg
    model = GPTOSS(cfg)
    for i, block in enumerate(model.blocks):
        if i % 2 == 0:
            assert block.attn.is_windowed, f"Layer {i} should be SWA"
        else:
            assert not block.attn.is_windowed, f"Layer {i} should be full"


def test_global_layers_have_pruned_rope(small_cfg):
    """Full-attention layers must have non-zero n_pruned_dims (pruned RoPE)."""
    cfg = small_cfg
    cfg = dataclasses.replace(cfg, yarn_prune_rope_global=True)
    model = GPTOSS(cfg)
    for i, block in enumerate(model.blocks):
        if i % 2 == 1:  # full attention layer
            assert block.attn._n_pruned_dims() > 0, \
                f"Full-attention layer {i} should have pruned RoPE"


# Grad flow + overfit

def test_grad_flow_all_params(small_cfg):
    """Backward must populate gradients on all learnable params."""
    cfg = small_cfg
    model = GPTOSS(cfg)
    idx = torch.randint(0, cfg.vocab_size, (1, cfg.max_seq_len))
    logits, aux_loss = model(idx)
    loss = logits.sum() + aux_loss
    loss.backward()
    n_with_grad = sum(1 for p in model.parameters() if p.grad is not None)
    n_total = sum(1 for _ in model.parameters())
    assert n_with_grad == n_total, f"Only {n_with_grad}/{n_total} params got gradients"


def test_two_step_overfit(small_cfg):
    """Loss must decrease on a single batch over 2 optimizer steps (sanity for grad flow)."""
    cfg = small_cfg
    model = GPTOSS(cfg)
    optim = torch.optim.AdamW(model.parameters(), lr=1e-3)
    idx = torch.randint(0, cfg.vocab_size, (2, cfg.max_seq_len))
    target = torch.randint(0, cfg.vocab_size, (2, cfg.max_seq_len))
    import torch.nn.functional as F
    losses = []
    for _ in range(2):
        logits, aux_loss = model(idx)
        ce_loss = F.cross_entropy(logits.view(-1, cfg.vocab_size), target.view(-1))
        loss = ce_loss + 0.01 * aux_loss
        optim.zero_grad()
        loss.backward()
        optim.step()
        losses.append(loss.item())
    assert losses[1] < losses[0], f"Loss did not decrease: {losses[0]:.3f} → {losses[1]:.3f}"


# Gradient checkpointing

def test_gradient_checkpointing_runs(small_cfg):
    """Model with gradient checkpointing enabled must still run forward + backward."""
    cfg = small_cfg
    model = GPTOSS(cfg)
    model.enable_gradient_checkpointing(every=2)
    idx = torch.randint(0, cfg.vocab_size, (2, cfg.max_seq_len))
    logits, aux_loss = model(idx)
    loss = logits.sum() + aux_loss
    loss.backward()
    assert model.gradient_checkpointing is True


def test_gradient_checkpointing_actually_checkpoints(small_cfg):
    """Verify gradient checkpointing is INVOKED, not just enabled (the regression C1)."""
    import torch.utils.checkpoint as cp
    cfg = small_cfg
    model = GPTOSS(cfg)
    model.enable_gradient_checkpointing(every=1)  # checkpoint every layer
    # Spy on torch.utils.checkpoint.checkpoint to count invocations.
    original = cp.checkpoint
    call_count = [0]
    def spy(*args, **kwargs):
        call_count[0] += 1
        return original(*args, **kwargs)
    cp.checkpoint = spy
    try:
        idx = torch.randint(0, cfg.vocab_size, (1, cfg.max_seq_len))
        logits, aux_loss = model(idx)
        (logits.sum() + aux_loss).backward()
    finally:
        cp.checkpoint = original
    # 4 layers, every=1 → all 4 should be checkpointed.
    assert call_count[0] == cfg.n_layers, (
        f"Expected {cfg.n_layers} checkpoint calls, got {call_count[0]}. "
        f"Gradient checkpointing is set but not applied!"
    )


def test_gradient_checkpointing_skip_layers(small_cfg):
    """every=2 should checkpoint half the layers, not all of them."""
    import torch.utils.checkpoint as cp
    cfg = small_cfg
    model = GPTOSS(cfg)
    model.enable_gradient_checkpointing(every=2)
    original = cp.checkpoint
    call_count = [0]
    def spy(*args, **kwargs):
        call_count[0] += 1
        return original(*args, **kwargs)
    cp.checkpoint = spy
    try:
        idx = torch.randint(0, cfg.vocab_size, (1, cfg.max_seq_len))
        logits, aux_loss = model(idx)
        (logits.sum() + aux_loss).backward()
    finally:
        cp.checkpoint = original
    # 4 layers, every=2 → 2 should be checkpointed (idx 0, 2).
    assert call_count[0] == cfg.n_layers // 2, (
        f"Expected {cfg.n_layers // 2} checkpoint calls (every=2), got {call_count[0]}"
    )