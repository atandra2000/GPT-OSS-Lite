"""Conservative GPU-memory estimates for training and mixed KV caching."""
import torch
import torch.nn as nn


def estimate_model_memory_gb(
    model: nn.Module,
    seq_len: int,
    batch_size: int,
    grad_checkpoint: bool = True,
    overhead_gb: float | None = None,
    steady_state: bool = False,
    grad_ckpt_every: int = 3,
) -> float:
    """Estimate peak GB from parameters, optimizer state, activations, KV, and overhead.

    The estimate is a planning guard rather than an allocator measurement;
    ``steady_state`` models decode after windowed caches have filled.
    """
    cfg = getattr(model, "cfg", None)
    if cfg is None:
        return 0.0

    # AdamW keeps two moments plus an FP32 master copy.
    param_bytes = sum(p.numel() * p.element_size() for p in model.parameters())
    optim_bytes = sum(p.numel() for p in model.parameters()) * 12  # 4 (m) + 4 (v) + 4 (master)

    # Local layers retain a window; global layers retain the full context.
    n_kv_heads, head_dim, window = cfg.n_kv_heads, cfg.head_dim, cfg.window_size
    dtype_bytes = 2  # BF16
    per_token = 2 * n_kv_heads * head_dim * dtype_bytes  # K + V
    n_layers = len(model.blocks)
    n_windowed = sum(1 for b in model.blocks if b.attn.is_windowed)
    n_global = n_layers - n_windowed
    win_len = window if steady_state else max(window, seq_len)
    kv_bytes = (n_windowed * win_len + n_global * seq_len) * batch_size * per_token

    # Approximate saved activations; checkpointing reduces the retained fraction.
    ckpt_factor = 1.0 / max(1, grad_ckpt_every) if grad_checkpoint else 0.0
    store_factor = (ckpt_factor + (1.0 - ckpt_factor) * 0.5) if grad_checkpoint else 1.0
    act_bytes = n_layers * seq_len * batch_size * cfg.d_model * dtype_bytes * store_factor
    if cfg.ffn_dim > 0:
        n_active_per_layer = 3
        act_bytes += n_layers * n_active_per_layer * 3 * seq_len * batch_size * cfg.ffn_dim * dtype_bytes * store_factor

    # Reserve allocator and framework overhead not represented above.
    if overhead_gb is None:
        if torch.cuda.is_available():
            total_gb = torch.cuda.get_device_properties(0).total_memory / 1024**3
            overhead_gb = min(13.7, max(2.0, total_gb * 0.17))
        else:
            overhead_gb = 2.0

    total = (param_bytes + optim_bytes + kv_bytes + act_bytes) / 1024**3 + overhead_gb
    return total


def assert_fits_in_available_gpu(estimate_gb: float, safety_margin_gb: float = 2.0) -> None:
    """Raise when the estimate leaves less than ``safety_margin_gb`` available."""
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
