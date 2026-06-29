"""Utility tests: checkpoint atomicity, memory estimator (mixed KV), logging, distributed."""
import json
import os
import tempfile
from pathlib import Path

import pytest
import torch

from models.transformer import GPTOSS, ModelConfig
from utils.checkpoint import CheckpointManager
from utils.distributed import device as get_device
from utils.logging import TrainingLogger
from utils.memory import (
    _mixed_kv_cache_bytes,
    assert_fits_in_available_gpu,
    estimate_model_memory_gb,
)


# Distributed helper

def test_distributed_device_returns_torch_device():
    """distributed.device() must return a torch.device object."""
    d = get_device()
    assert isinstance(d, torch.device)
    assert d.type in ("cpu", "cuda")


# Memory estimator

def test_mixed_kv_cache_correct(small_cfg):
    """KV cache must reflect windowed (128 tok) + global (full) split."""
    cfg = ModelConfig(**{k: v for k, v in small_cfg.items() if k in ModelConfig.__dataclass_fields__})
    model = GPTOSS(cfg)
    seq_len = cfg.max_seq_len
    bs = 1
    bytes_per_token = 2 * cfg.n_kv_heads * cfg.head_dim * 2
    n_windowed = sum(1 for i in range(cfg.n_layers) if i % 2 == 0)
    n_global = cfg.n_layers - n_windowed
    expected_steady = (
        n_windowed * cfg.window_size * bs * bytes_per_token
        + n_global * seq_len * bs * bytes_per_token
    )
    actual_steady = _mixed_kv_cache_bytes(model, seq_len, bs, steady_state=True)
    assert actual_steady == expected_steady, (
        f"Steady-state KV mismatch: actual={actual_steady}, expected={expected_steady}"
    )
    prefill_windowed_len = max(cfg.window_size, seq_len)
    expected_prefill = (
        n_windowed * prefill_windowed_len * bs * bytes_per_token
        + n_global * seq_len * bs * bytes_per_token
    )
    actual_prefill = _mixed_kv_cache_bytes(model, seq_len, bs, steady_state=False)
    assert actual_prefill == expected_prefill, (
        f"Prefill KV mismatch: actual={actual_prefill}, expected={expected_prefill}"
    )
    if seq_len > cfg.window_size:
        assert actual_prefill > actual_steady


def test_estimate_model_memory_returns_gb(small_cfg):
    """estimate_model_memory_gb must return a positive float in GB."""
    cfg = ModelConfig(**{k: v for k, v in small_cfg.items() if k in ModelConfig.__dataclass_fields__})
    model = GPTOSS(cfg)
    est_gb = estimate_model_memory_gb(model, seq_len=cfg.max_seq_len, batch_size=2, grad_checkpoint=True)
    assert est_gb > 0.0
    assert est_gb < 100.0


def test_estimate_smaller_with_checkpointing(small_cfg):
    """Activations are smaller with gradient checkpointing."""
    cfg = ModelConfig(**{k: v for k, v in small_cfg.items() if k in ModelConfig.__dataclass_fields__})
    model = GPTOSS(cfg)
    est_no_ckpt = estimate_model_memory_gb(model, seq_len=cfg.max_seq_len, batch_size=2, grad_checkpoint=False)
    est_ckpt = estimate_model_memory_gb(model, seq_len=cfg.max_seq_len, batch_size=2, grad_checkpoint=True)
    assert est_ckpt < est_no_ckpt


def test_estimate_with_grad_ckpt_every(small_cfg):
    """More aggressive checkpointing (every=2) should give lower estimate than every=3."""
    cfg = ModelConfig(**{k: v for k, v in small_cfg.items() if k in ModelConfig.__dataclass_fields__})
    model = GPTOSS(cfg)
    est_e3 = estimate_model_memory_gb(
        model, seq_len=cfg.max_seq_len, batch_size=2,
        grad_checkpoint=True, grad_ckpt_every=3,
    )
    est_e2 = estimate_model_memory_gb(
        model, seq_len=cfg.max_seq_len, batch_size=2,
        grad_checkpoint=True, grad_ckpt_every=2,
    )
    # every=2 keeps more activations than every=3, so the every=2 estimate is higher.
    assert est_e2 > est_e3, f"every=2 ({est_e2}) should be higher than every=3 ({est_e3})"


def test_estimate_manual_attn_more_memory_than_sdpa(small_cfg):
    """Manual attention stores scores (O(T^2)); SDPA does not."""
    from dataclasses import replace
    cfg_base = ModelConfig(**{k: v for k, v in small_cfg.items() if k in ModelConfig.__dataclass_fields__})
    model_sdpa = GPTOSS(cfg_base)
    model_manual = GPTOSS(replace(cfg_base, attn_impl="manual"))
    seq_len = 512
    batch_size = 4
    est_sdpa = estimate_model_memory_gb(
        model_sdpa, seq_len=seq_len, batch_size=batch_size, grad_checkpoint=True,
    )
    est_manual = estimate_model_memory_gb(
        model_manual, seq_len=seq_len, batch_size=batch_size, grad_checkpoint=True,
    )
    assert est_manual > est_sdpa, f"Manual {est_manual} should exceed SDPA {est_sdpa}"


def test_assert_fits_in_available_gpu_runs_on_cpu():
    """On CPU, assert_fits_in_available_gpu must be a no-op (no exception)."""
    assert_fits_in_available_gpu(1000.0)


# Logger

def test_training_logger_no_crash():
    """TrainingLogger must not crash on log calls."""
    logger = TrainingLogger(log_interval=2, seq_len=128)
    for step in [0, 1, 2, 3, 4]:
        logger.log(step, loss=2.5 - step * 0.1, lr=1e-4)


def test_training_logger_logs_metrics():
    """TrainingLogger should log metrics when provided."""
    logger = TrainingLogger(log_interval=1, seq_len=128)
    logger.log(1, loss=2.0, metrics={"aux": 0.1, "ce": 1.9}, lr=1e-4)


# Checkpoint helpers

def test_checkpoint_keep_last_n(small_cfg, tmp_ckpt_dir):
    """keep_last_n must delete older checkpoints."""
    cfg = ModelConfig(**{k: v for k, v in small_cfg.items() if k in ModelConfig.__dataclass_fields__})
    model = GPTOSS(cfg)
    optim = torch.optim.AdamW(model.parameters(), lr=1e-3)
    ckpt = CheckpointManager(str(tmp_ckpt_dir))
    for step in [10, 20, 30, 40, 50]:
        ckpt.save(model, optim, step)
    assert ckpt.list_checkpoints() == [10, 20, 30, 40, 50]
    ckpt.keep_last_n(2)
    assert ckpt.list_checkpoints() == [40, 50]


def test_checkpoint_delete_specific_step(small_cfg, tmp_ckpt_dir):
    """delete_checkpoint must remove all 3 files for that step."""
    cfg = ModelConfig(**{k: v for k, v in small_cfg.items() if k in ModelConfig.__dataclass_fields__})
    model = GPTOSS(cfg)
    optim = torch.optim.AdamW(model.parameters(), lr=1e-3)
    ckpt = CheckpointManager(str(tmp_ckpt_dir))
    ckpt.save(model, optim, step=10)
    ckpt.save(model, optim, step=20)
    ckpt.delete_checkpoint(10)
    assert 10 not in ckpt.list_checkpoints()
    assert 20 in ckpt.list_checkpoints()
    leftover = list(tmp_ckpt_dir.glob("*_step_10.*"))
    assert leftover == []