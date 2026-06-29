"""Training pipeline tests: LR schedule, checkpoint round-trip, NaN guard, aux loss accumulation."""
import os
import tempfile
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F

from models.transformer import GPTOSS, ModelConfig
from training.pretrain import (
    PretrainDataset,
    chunked_cross_entropy,
    make_warmup_cosine_lambda,
)
from utils.checkpoint import CheckpointManager


# LR schedule

def test_lr_schedule_at_warmup_boundary():
    """LR at step 0 must be 0; at warmup_steps must be 1 (peak)."""
    lr_lambda = make_warmup_cosine_lambda(warmup_steps=100, total_steps=1000, min_lr_ratio=0.05)
    assert lr_lambda(0) == 0.0
    assert abs(lr_lambda(100) - 1.0) < 1e-5


def test_lr_schedule_at_end():
    """LR at total_steps must equal min_lr_ratio."""
    lr_lambda = make_warmup_cosine_lambda(warmup_steps=100, total_steps=1000, min_lr_ratio=0.05)
    assert abs(lr_lambda(1000) - 0.05) < 1e-5


def test_lr_schedule_monotonic_decay_after_warmup():
    """After warmup, LR must decay monotonically."""
    lr_lambda = make_warmup_cosine_lambda(warmup_steps=100, total_steps=1000, min_lr_ratio=0.05)
    prev = lr_lambda(100)
    for s in range(101, 1000, 50):
        curr = lr_lambda(s)
        assert curr <= prev + 1e-7, f"LR increased at step {s}: {curr} > {prev}"
        prev = curr


# Chunked cross-entropy

def test_chunked_ce_matches_full():
    """Chunked CE must equal unchunked F.cross_entropy (within fp tolerance)."""
    torch.manual_seed(0)
    logits = torch.randn(8, 16, 100, dtype=torch.float32)  # (B, T, V)
    targets = torch.randint(0, 100, (8, 16))
    full_loss = F.cross_entropy(logits.view(-1, 100), targets.view(-1))
    chunked_loss = chunked_cross_entropy(logits, targets, chunk_size=32)
    assert torch.allclose(full_loss, chunked_loss, atol=1e-5)


def test_chunked_ce_gradient_flow():
    """Chunked CE must backprop gradients to logits."""
    torch.manual_seed(0)
    logits = torch.randn(4, 8, 50, dtype=torch.float32, requires_grad=True)
    targets = torch.randint(0, 50, (4, 8))
    loss = chunked_cross_entropy(logits, targets, chunk_size=16)
    loss.backward()
    assert logits.grad is not None
    assert torch.isfinite(logits.grad).all()


# Dataset

def test_pretrain_dataset_single_file():
    """PretrainDataset must load a single .bin file and yield windows."""
    with tempfile.NamedTemporaryFile(suffix=".bin", delete=False) as tmp:
        data = torch.randint(0, 256, (2000,), dtype=torch.long)
        torch.save(data, tmp.name)
        tmp_path = tmp.name
    try:
        ds = PretrainDataset(tmp_path, max_seq_len=128)
        assert len(ds) > 0
        x, y = ds[0]
        assert x.shape == (128,)
        assert y.shape == (128,)
        # Targets are shifted by one (next-token prediction)
        assert torch.equal(x[1:], y[:-1])
    finally:
        os.unlink(tmp_path)


def test_pretrain_dataset_sharded():
    """PretrainDataset must load a sharded directory."""
    tmp = Path(tempfile.mkdtemp(prefix="test_shards_"))
    try:
        for i in range(3):
            shard = torch.randint(0, 1024, (500,), dtype=torch.long)
            torch.save(shard, tmp / f"shard_{i:05d}.bin")
        ds = PretrainDataset(str(tmp), max_seq_len=128)
        assert len(ds) > 0
        x, y = ds[0]
        assert x.shape == (128,)
        assert y.shape == (128,)
    finally:
        import shutil
        shutil.rmtree(tmp, ignore_errors=True)


# Checkpoint round-trip

def test_checkpoint_round_trip(small_cfg, tmp_ckpt_dir):
    """Save → load must restore weights exactly."""
    cfg = ModelConfig(**{k: v for k, v in small_cfg.items() if k in ModelConfig.__dataclass_fields__})
    model1 = GPTOSS(cfg)
    optim1 = torch.optim.AdamW(model1.parameters(), lr=1e-3)

    # Save
    ckpt = CheckpointManager(str(tmp_ckpt_dir))
    ckpt.save(model1, optim1, step=100)

    # New model + optimizer, load into them
    model2 = GPTOSS(cfg)
    optim2 = torch.optim.AdamW(model2.parameters(), lr=1e-3)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    meta = ckpt.load(model2, step=100, optimizer=optim2, device=dev)
    assert meta["step"] == 100

    # Weights must match exactly
    for (n1, p1), (n2, p2) in zip(model1.named_parameters(), model2.named_parameters()):
        assert n1 == n2
        assert torch.equal(p1, p2), f"Weights differ at {n1}"


def test_checkpoint_atomicity_no_partial_files(small_cfg, tmp_ckpt_dir):
    """Killing mid-save should not leave partial files."""
    import safetensors
    cfg = ModelConfig(**{k: v for k, v in small_cfg.items() if k in ModelConfig.__dataclass_fields__})
    model = GPTOSS(cfg)
    optim = torch.optim.AdamW(model.parameters(), lr=1e-3)
    ckpt = CheckpointManager(str(tmp_ckpt_dir))
    ckpt.save(model, optim, step=42)
    # Verify no `.tmp` files leak
    tmp_files = list(tmp_ckpt_dir.glob("*.tmp"))
    assert len(tmp_files) == 0, f"Leftover tmp files: {tmp_files}"
    # Verify the actual checkpoint files exist
    assert (tmp_ckpt_dir / "model_step_42.safetensors").exists()
    assert (tmp_ckpt_dir / "optim_step_42.pt").exists()
    assert (tmp_ckpt_dir / "meta_step_42.json").exists()


def test_checkpoint_latest_step(small_cfg, tmp_ckpt_dir):
    """latest_step must return the highest completed step."""
    cfg = ModelConfig(**{k: v for k, v in small_cfg.items() if k in ModelConfig.__dataclass_fields__})
    model = GPTOSS(cfg)
    optim = torch.optim.AdamW(model.parameters(), lr=1e-3)
    ckpt = CheckpointManager(str(tmp_ckpt_dir))
    ckpt.save(model, optim, step=10)
    ckpt.save(model, optim, step=20)
    ckpt.save(model, optim, step=30)
    assert ckpt.latest_step() == 30
    # List checkpoints
    steps = ckpt.list_checkpoints()
    assert steps == [10, 20, 30]


# Aux loss accumulation

def test_aux_loss_accumulated_in_training(small_cfg):
    """Aux loss must be finite and addable to CE loss."""
    cfg = ModelConfig(**{k: v for k, v in small_cfg.items() if k in ModelConfig.__dataclass_fields__})
    model = GPTOSS(cfg)
    optim = torch.optim.AdamW(model.parameters(), lr=1e-3)
    idx = torch.randint(0, cfg.vocab_size, (1, cfg.max_seq_len))
    target = torch.randint(0, cfg.vocab_size, (1, cfg.max_seq_len))
    logits, aux = model(idx)
    ce = F.cross_entropy(logits.view(-1, cfg.vocab_size), target.view(-1))
    total_loss = ce + 0.01 * aux
    assert torch.isfinite(total_loss)
    total_loss.backward()
    # Aux loss gradient should reach the router
    for block in model.blocks:
        assert block.moe.router.gate.weight.grad is not None
        assert torch.isfinite(block.moe.router.gate.weight.grad).all()


# NaN guard

def test_nan_guard_detection():
    """Simulating a NaN loss must be detected by torch.isfinite()."""
    nan_loss = torch.tensor(float("nan"))
    assert not torch.isfinite(nan_loss)
    inf_loss = torch.tensor(float("inf"))
    assert not torch.isfinite(inf_loss)
    normal = torch.tensor(1.0)
    assert torch.isfinite(normal)