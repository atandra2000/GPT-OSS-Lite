"""YaRN RoPE scaling module."""
import math
import torch
import torch.nn as nn

from models.rotary import compute_yarn_freqs, compute_yarn_mscale


class YaRNRoPE(nn.Module):
    """YaRN-scaled RoPE: trains at ``original_max_seq_len``, extrapolates to ``target_seq_len``."""

    def __init__(
        self,
        head_dim: int,
        theta: float = 100000.0,
        scale_factor: float = 32.0,
        original_max_seq_len: int | None = None,
        target_seq_len: int = 131072,
        beta_fast: float = 32.0,
        beta_slow: float = 1.0,
        mscale: bool = True,
        original_max: int | None = None,
        target_max: int | None = None,
    ):
        super().__init__()
        if head_dim % 2 != 0:
            raise ValueError(f"head_dim must be even, got {head_dim}")
        if original_max_seq_len is None:
            if original_max is None:
                raise ValueError("Either original_max_seq_len or original_max must be provided.")
            original_max_seq_len = original_max
        if target_max is not None:
            target_seq_len = target_max
        self.head_dim = head_dim
        self.theta = theta
        self.scale_factor = scale_factor
        self.original_max_seq_len = original_max_seq_len
        self.target_seq_len = target_seq_len
        self.mscale_enabled = mscale

        inv_freq = compute_yarn_freqs(
            head_dim=head_dim,
            theta=theta,
            scale_factor=scale_factor,
            original_max=original_max_seq_len,
            target_max=target_seq_len,
            beta_fast=beta_fast,
            beta_slow=beta_slow,
        )
        self.register_buffer("inv_freq", inv_freq, persistent=False)

        if mscale:
            self.mscale = compute_yarn_mscale(scale_factor)
        else:
            self.mscale = 1.0

    def forward(
        self,
        positions: torch.Tensor,
        n_pruned_dims: int = 0,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute ``(cos, sin)`` for the given positions; each ``(T, head_dim // 2)``."""
        if positions.numel() == 1:
            inv_freq = self.inv_freq.to(positions.device)
            pos = positions.item() if positions.dim() == 0 else positions[0].item()
            freqs = inv_freq * float(pos)
            cos = freqs.cos().unsqueeze(0) * self.mscale
            sin = freqs.sin().unsqueeze(0) * self.mscale
        else:
            freqs = torch.outer(positions.float(), self.inv_freq.to(positions.device))
            cos = freqs.cos() * self.mscale
            sin = freqs.sin() * self.mscale

        if n_pruned_dims > 0:
            cos = cos.clone()
            sin = sin.clone()
            cos[:, :n_pruned_dims] = 1.0
            sin[:, :n_pruned_dims] = 0.0

        return cos, sin

    def extra_repr(self) -> str:
        return (
            f"head_dim={self.head_dim}, theta={self.theta}, "
            f"scale_factor={self.scale_factor}, "
            f"original_max={self.original_max_seq_len}, "
            f"target={self.target_seq_len}, mscale={self.mscale_enabled}"
        )