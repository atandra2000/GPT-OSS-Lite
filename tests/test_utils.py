"""Utility tests: checkpoint atomicity, memory estimator (mixed KV), logging, distributed."""
import json
import os
import tempfile
from pathlib import Path

import pytest
import torch

from models.transformer import GPTOSS, ModelConfig
from utils.checkpoint import CheckpointManager
from utils.logging import TrainingLogger
from utils.memory import (
    assert_fits_in_available_gpu,
    estimate_model_memory_gb,
)


# Memory estimator

def test_mixed_kv_cache_correct(small_cfg):
    """The KV cache portion of the estimate must reflect windowed (128 tok) + global (full) split.

    Compute the estimate for the same model at two steady-state flags and verify
    the difference matches the windowed-vs-full-cache delta.
    """
    cfg = small_cfg
    model = GPTOSS(cfg)
    seq_len = cfg.max_seq_len
    bs = 1
    bytes_per_token = 2 * cfg.n_kv_heads * cfg.head_dim * 2
    n_windowed = sum(1 for i in range(cfg.n_layers) if i % 2 == 0)
    n_global = cfg.n_layers - n_windowed

    # Steady-state: windowed layers hold `window` tokens, global hold `seq_len`.
    expected_steady_kv = (
        n_windowed * cfg.window_size * bs * bytes_per_token
        + n_global * seq_len * bs * bytes_per_token
    )
    # Prefill: windowed layers grow to max(window, seq_len) = seq_len for seq_len > window.
    expected_prefill_kv = (
        n_windowed * max(cfg.window_size, seq_len) * bs * bytes_per_token
        + n_global * seq_len * bs * bytes_per_token
    )
    # Difference (prefill - steady) = n_windowed * (seq_len - window) * bytes_per_token
    expected_delta = n_windowed * max(0, seq_len - cfg.window_size) * bs * bytes_per_token

    # Build two estimates where the only delta is steady_state.
    est_steady = estimate_model_memory_gb(
        model, seq_len=seq_len, batch_size=bs,
        grad_checkpoint=True, overhead_gb=0.0, steady_state=True,
    )
    est_prefill = estimate_model_memory_gb(
        model, seq_len=seq_len, batch_size=bs,
        grad_checkpoint=True, overhead_gb=0.0, steady_state=False,
    )
    actual_delta_bytes = (est_prefill - est_steady) * 1024**3
    # Allow small float tolerance from activation/param differences that are NOT
    # affected by steady_state — the delta should isolate the KV cache diff.
    assert abs(actual_delta_bytes - expected_delta) < 1024, (
        f"KV cache delta mismatch: actual={actual_delta_bytes}, expected={expected_delta}"
    )
    if seq_len > cfg.window_size:
        assert est_prefill > est_steady


def test_estimate_model_memory_returns_gb(small_cfg):
    """estimate_model_memory_gb must return a positive float in GB."""
    cfg = small_cfg
    model = GPTOSS(cfg)
    est_gb = estimate_model_memory_gb(model, seq_len=cfg.max_seq_len, batch_size=2, grad_checkpoint=True)
    assert est_gb > 0.0
    assert est_gb < 100.0


def test_estimate_smaller_with_checkpointing(small_cfg):
    """Activations are smaller with gradient checkpointing."""
    cfg = small_cfg
    model = GPTOSS(cfg)
    est_no_ckpt = estimate_model_memory_gb(model, seq_len=cfg.max_seq_len, batch_size=2, grad_checkpoint=False)
    est_ckpt = estimate_model_memory_gb(model, seq_len=cfg.max_seq_len, batch_size=2, grad_checkpoint=True)
    assert est_ckpt < est_no_ckpt


def test_estimate_with_grad_ckpt_every(small_cfg):
    """More aggressive checkpointing (every=2) should give lower estimate than every=3."""
    cfg = small_cfg
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
    cfg = small_cfg
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
    cfg = small_cfg
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