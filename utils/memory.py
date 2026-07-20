"""VRAM budgeting for GPT-OSS-Lite (mixed windowed/global KV cache)."""
from __future__ import annotations

import torch
import torch.nn as nn


def _parameter_bytes(model: nn.Module) -> int:
    return sum(p.numel() * p.element_size() for p in model.parameters())


def _optimiser_bytes(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters()) * 12


def _mixed_kv_cache_bytes(
    model: nn.Module,
    seq_len: int,
    batch_size: int,
    dtype_bytes: int = 2,
    steady_state: bool = False,
) -> int:
    """Estimate the mixed (windowed + global) KV-cache size in bytes."""
    cfg = getattr(model, "cfg", None)
    if cfg is None:
        return 0
    n_kv_heads = cfg.n_kv_heads
    head_dim = cfg.head_dim
    window = cfg.window_size
    windowed_layers = sum(1 for b in model.blocks if b.attn.is_windowed)
    global_layers = len(model.blocks) - windowed_layers
    per_token = 2 * n_kv_heads * head_dim * dtype_bytes
    if steady_state:
        windowed_len = window
    else:
        windowed_len = max(window, seq_len)
    windowed_bytes = windowed_layers * windowed_len * batch_size * per_token
    global_bytes = global_layers * seq_len * batch_size * per_token
    return windowed_bytes + global_bytes


def _activation_bytes(
    seq_len: int,
    batch_size: int,
    hidden_dim: int,
    n_layers: int,
    grad_checkpoint: bool,
    dtype_bytes: int = 2,
    ffn_dim: int = 0,
    n_heads: int = 0,
    grad_ckpt_every: int = 3,
) -> int:
    """Estimate activation memory (bytes) for forward + backward."""
    if grad_checkpoint:
        ckpt_factor = 1.0 / max(1, grad_ckpt_every)
        store_factor = ckpt_factor + (1.0 - ckpt_factor) * 0.5
    else:
        store_factor = 1.0

    hidden_bytes = n_layers * seq_len * batch_size * hidden_dim * dtype_bytes * store_factor
    moe_bytes = 0
    if ffn_dim > 0:
        n_active_per_layer = 3
        moe_bytes = n_layers * n_active_per_layer * 3 * seq_len * batch_size * ffn_dim * dtype_bytes * store_factor
    return hidden_bytes + moe_bytes


def _infer_dim_n_layers(model: nn.Module) -> tuple[int, int]:
    hd = getattr(model.cfg, "d_model", 0) if hasattr(model, "cfg") else 0
    nl = len(model.blocks) if hasattr(model, "blocks") else 0
    return hd, nl


def _detect_overhead_gb() -> float:
    if not torch.cuda.is_available():
        return 2.0
    total_gb = torch.cuda.get_device_properties(0).total_memory / 1024**3
    return min(13.7, max(2.0, total_gb * 0.17))


def estimate_model_memory_gb(
    model: nn.Module,
    seq_len: int,
    batch_size: int,
    grad_checkpoint: bool = True,
    overhead_gb: float | None = None,
    steady_state: bool = False,
    grad_ckpt_every: int = 3,
) -> float:
    """Estimate peak VRAM (GB) for forward + backward at the given batch/seq."""
    params_b = _parameter_bytes(model)
    optim_b = _optimiser_bytes(model)
    kv_b = _mixed_kv_cache_bytes(model, seq_len, batch_size, steady_state=steady_state)
    hd, nl = _infer_dim_n_layers(model)
    cfg = getattr(model, "cfg", None)
    ffn_dim = getattr(cfg, "ffn_dim", 0) if cfg is not None else 0
    n_heads = getattr(cfg, "n_heads", 0) if cfg is not None else 0
    act_b = _activation_bytes(
        seq_len, batch_size, hidden_dim=hd, n_layers=nl,
        grad_checkpoint=grad_checkpoint,
        ffn_dim=ffn_dim, n_heads=n_heads,
        grad_ckpt_every=grad_ckpt_every,
    )
    total = params_b + optim_b + kv_b + act_b
    return total / 1024**3 + (overhead_gb if overhead_gb is not None else _detect_overhead_gb())


def assert_fits_in_available_gpu(estimate_gb: float, safety_margin_gb: float = 2.0) -> None:
    if not torch.cuda.is_available():
        return
    try:
        available = torch.cuda.get_device_properties(0).total_memory / 1024**3
    except Exception:
        return
    if estimate_gb > available - safety_margin_gb:
        raise RuntimeError(
            f"Estimated peak VRAM ({estimate_gb:.1f} GB) exceeds available GPU memory "
            f"({available:.1f} GB, {safety_margin_gb:.1f} GB margin)."
        )
    print(f"[memory] Estimated peak VRAM: {estimate_gb:.1f} GB / {available:.1f} GB — OK.")