"""Shared bootstrap for scripts/ — sys.path fix, time_fn, micro_cfg.

Import this first thing in every script:
    sys.path is fixed, time_fn + micro_cfg() are in scope.
"""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch  # noqa: E402

from models.transformer import ModelConfig  # noqa: E402


def time_fn(fn, n: int = 20, warmup: int = 3) -> float:
    """Average ms per fn() call over n runs after warmup."""
    for _ in range(warmup):
        fn()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(n):
        fn()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    return (time.perf_counter() - t0) / n * 1000


def micro_cfg() -> ModelConfig:
    """Small 4-layer / 64-dim config for fast CPU benchmarks."""
    return ModelConfig(
        vocab_size=128,
        d_model=64,
        n_layers=4,
        n_heads=4,
        n_kv_heads=2,
        head_dim=16,
        ffn_dim=128,
        n_routed_experts=4,
        n_activated_experts=2,
        n_shared_experts=1,
        window_size=8,
        max_seq_len=128,
        rope_theta=10000,
        yarn_scale_factor=2,
        yarn_original_max_seq_len=128,
        yarn_target_seq_len=256,
        yarn_beta_fast=2,
        yarn_beta_slow=1,
        yarn_prune_rope_global=True,
    )
