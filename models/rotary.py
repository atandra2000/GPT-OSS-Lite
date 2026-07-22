"""RoPE helpers: standard apply_rope, YaRN frequency computation."""
import math
import torch


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """Apply rotary position embeddings to ``x``."""
    T = x.size(-2)
    half = x.size(-1) // 2

    cos_full = cos.repeat_interleave(2, dim=-1)
    sin_full = sin.repeat_interleave(2, dim=-1)

    x_pairs = x.unflatten(-1, (-1, 2))
    x_swapped = x_pairs.flip(-1)
    x_swapped[..., 0] = -x_swapped[..., 0]
    x_rotated = x_swapped.flatten(-2)

    while cos_full.dim() < x.dim():
        cos_full = cos_full.unsqueeze(0)
        sin_full = sin_full.unsqueeze(0)

    return x * cos_full + x_rotated * sin_full


def compute_yarn_freqs(
    head_dim: int,
    theta: float,
    scale_factor: float,
    original_max_seq_len: int,
    target_seq_len: int,
    beta_fast: float = 32.0,
    beta_slow: float = 1.0,
) -> torch.Tensor:
    """Compute YaRN-scaled inverse frequencies for RoPE."""
    if head_dim % 2 != 0:
        raise ValueError(f"head_dim must be even, got {head_dim}")
    if original_max_seq_len <= 0 or target_seq_len <= 0:
        raise ValueError(
            f"yarn sequence lengths must be positive, "
            f"got original={original_max_seq_len}, target={target_seq_len}"
        )

    half = head_dim // 2
    exponents = torch.arange(0, half, dtype=torch.float32) / half
    base = 1.0 / (theta ** exponents)

    low = max(math.floor(half / math.log2(original_max_seq_len / beta_slow * math.pi)), 0)
    high = min(math.ceil(half / math.log2(original_max_seq_len / beta_fast * math.pi)), half - 1)
    if high <= low:
        import warnings
        warnings.warn(
            f"YaRN ramp degenerate: low={low}, high={high} (head_dim={head_dim}, "
            f"original_max={original_max_seq_len}, beta_fast={beta_fast}, beta_slow={beta_slow}). "
            f"Falling back to identity (no length extrapolation). Check beta_fast/beta_slow.",
            UserWarning,
            stacklevel=2,
        )
        ramp = torch.zeros(half, dtype=torch.float32)
    else:
        ramp = torch.clamp(
            (torch.arange(half, dtype=torch.float32) - low) / max(high - low, 1),
            0.0, 1.0,
        )
    inv_freq = base * (1.0 - ramp) + (base / scale_factor) * ramp
    return inv_freq


def compute_yarn_mscale(scale_factor: float) -> float:
    """YaRN attention scaling factor (the mscale term)."""
    if scale_factor <= 1.0:
        return 1.0
    return 0.1 * math.log(scale_factor) + 1.0