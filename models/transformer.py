"""GPT-OSS-Lite top-level model: embedding + 12 alternating-attention/MoE blocks + head."""
import math
from dataclasses import dataclass, field
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.attention import GPTOSSAttention
from models.moe import MoELayer


@dataclass
class ModelConfig:
    """Dataclass mirror of the model.* fields in the YAML config."""
    vocab_size: int = 128000
    d_model: int = 768
    n_layers: int = 12
    n_heads: int = 8
    n_kv_heads: int = 4
    head_dim: int = 96
    ffn_dim: int = 1536
    n_routed_experts: int = 8
    n_activated_experts: int = 2
    n_shared_experts: int = 1
    window_size: int = 128
    attention_pattern: str = "alternating"
    sink_bias: bool = True
    rope_theta: int = 100000
    yarn_scale_factor: int = 32
    yarn_original_max_seq_len: int = 4096
    yarn_target_seq_len: int = 131072
    yarn_beta_fast: int = 32
    yarn_beta_slow: int = 1
    yarn_mscale: bool = True
    yarn_prune_rope_global: bool = True
    max_seq_len: int = 4096
    eval_max_seq_len: int = 131072
    dtype: str = "bf16"
    weight_tying: bool = True
    rms_norm_eps: float = 1e-5
    init_std: float = 0.02
    attn_impl: str = "sdpa"

    def __post_init__(self):
        """Validate config invariants. Fails fast on misconfiguration."""
        if self.vocab_size <= 0:
            raise ValueError(f"vocab_size must be positive, got {self.vocab_size}")
        if self.d_model <= 0:
            raise ValueError(f"d_model must be positive, got {self.d_model}")
        if self.n_layers <= 0:
            raise ValueError(f"n_layers must be positive, got {self.n_layers}")
        if self.n_heads <= 0 or self.n_kv_heads <= 0:
            raise ValueError(
                f"n_heads and n_kv_heads must be positive, "
                f"got n_heads={self.n_heads}, n_kv_heads={self.n_kv_heads}"
            )
        if self.n_heads % self.n_kv_heads != 0:
            raise ValueError(
                f"n_heads must be a multiple of n_kv_heads for GQA, "
                f"got n_heads={self.n_heads}, n_kv_heads={self.n_kv_heads}"
            )
        if self.head_dim <= 0 or self.head_dim % 2 != 0:
            raise ValueError(f"head_dim must be a positive even int, got {self.head_dim}")
        if self.n_heads * self.head_dim != self.d_model:
            raise ValueError(
                f"n_heads * head_dim must equal d_model, "
                f"got n_heads*head_dim={self.n_heads * self.head_dim}, d_model={self.d_model}"
            )
        if self.n_routed_experts <= 0:
            raise ValueError(f"n_routed_experts must be positive, got {self.n_routed_experts}")
        if not (0 < self.n_activated_experts <= self.n_routed_experts):
            raise ValueError(
                f"0 < n_activated_experts <= n_routed_experts required, "
                f"got n_activated={self.n_activated_experts}, n_routed={self.n_routed_experts}"
            )
        if self.n_shared_experts < 0:
            raise ValueError(f"n_shared_experts must be >= 0, got {self.n_shared_experts}")
        if self.window_size <= 0:
            raise ValueError(f"window_size must be positive, got {self.window_size}")
        if self.ffn_dim <= 0:
            raise ValueError(f"ffn_dim must be positive, got {self.ffn_dim}")
        if self.rope_theta <= 0:
            raise ValueError(f"rope_theta must be positive, got {self.rope_theta}")
        if self.yarn_scale_factor < 1:
            raise ValueError(
                f"yarn_scale_factor must be >= 1; for plain RoPE without scaling use 1, "
                f"got {self.yarn_scale_factor}"
            )
        if self.yarn_scale_factor > 1 and self.yarn_original_max_seq_len >= self.yarn_target_seq_len:
            raise ValueError(
                f"yarn_scale_factor > 1 requires yarn_original_max_seq_len < yarn_target_seq_len; "
                f"got original={self.yarn_original_max_seq_len}, target={self.yarn_target_seq_len}"
            )
        if self.yarn_prune_rope_global and self.n_layers % 2 != 0:
            import warnings
            warnings.warn(
                f"yarn_prune_rope_global=True with n_layers={self.n_layers}: "
                f"the alternating pattern expects even n_layers; the final "
                f"layer may be a windowed layer (no pruning)."
            )
        if self.yarn_original_max_seq_len <= 0 or self.yarn_target_seq_len <= 0:
            raise ValueError(
                f"yarn sequence lengths must be positive, "
                f"got original={self.yarn_original_max_seq_len}, target={self.yarn_target_seq_len}"
            )
        if self.rms_norm_eps <= 0:
            raise ValueError(f"rms_norm_eps must be positive, got {self.rms_norm_eps}")
        if self.max_seq_len <= 0:
            raise ValueError(f"max_seq_len must be positive, got {self.max_seq_len}")
        if self.eval_max_seq_len < self.max_seq_len:
            import warnings
            warnings.warn(
                f"eval_max_seq_len ({self.eval_max_seq_len}) < max_seq_len "
                f"({self.max_seq_len}); eval context will be shorter than training."
            )

    def as_dict(self) -> dict:
        """Return a shallow copy of the config as a dict (safe for callers)."""
        return dict(self.__dict__)


class RMSNorm(nn.Module):
    """Root Mean Square LayerNorm (no bias, no mean subtraction)."""

    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rms = x.detach().float().pow(2).mean(dim=-1, keepdim=True).add(self.eps).rsqrt()
        return (x * (rms * self.weight.to(rms.dtype)).to(x.dtype))


class GPTOSSBlock(nn.Module):
    """One GPT-OSS block: pre-norm attention + residual, pre-norm MoE + residual."""

    def __init__(self, cfg: ModelConfig, layer_idx: int):
        super().__init__()
        self.layer_idx = layer_idx
        self.attn = GPTOSSAttention(cfg.as_dict(), layer_idx)
        self.moe = MoELayer(cfg.as_dict())
        self.norm1 = RMSNorm(cfg.d_model, cfg.rms_norm_eps)
        self.norm2 = RMSNorm(cfg.d_model, cfg.rms_norm_eps)

    def forward(self, x: torch.Tensor, positions: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x = x + self.attn(self.norm1(x), positions)
        moe_out, aux_loss = self.moe(self.norm2(x))
        x = x + moe_out
        return x, aux_loss


class GPTOSS(nn.Module):
    """GPT-OSS-Lite: top-level model returning ``(logits, aux_loss)``."""

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        self.embed = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.blocks = nn.ModuleList([
            GPTOSSBlock(cfg, i) for i in range(cfg.n_layers)
        ])
        self.norm = RMSNorm(cfg.d_model, cfg.rms_norm_eps)
        self.head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        if cfg.weight_tying:
            self.head.weight = self.embed.weight
        self._init_weights()

    def _init_weights(self) -> None:
        """Standard small-init scheme (DeepSeek-V3 style: std=0.02 for most params)."""
        std = self.cfg.init_std
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=std)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0.0, std=std)
            elif isinstance(module, RMSNorm):
                nn.init.ones_(module.weight)
        for block in self.blocks:
            if hasattr(block.attn, "sink_bias") and block.attn.sink_bias is not None:
                nn.init.zeros_(block.attn.sink_bias)

    def forward(
        self,
        idx: torch.Tensor,
        positions: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward returning ``(logits (B,T,vocab), aux_loss scalar)``."""
        B, T = idx.shape
        if positions is None:
            positions = torch.arange(T, device=idx.device)
        x = self.embed(idx)

        aux_losses: list[torch.Tensor] = []
        use_grad_ckpt = (
            getattr(self, "gradient_checkpointing", False)
            and torch.is_grad_enabled()
        )
        grad_ckpt_every = max(1, getattr(self, "grad_ckpt_every", 3))

        for layer_idx, block in enumerate(self.blocks):
            if use_grad_ckpt and (layer_idx % grad_ckpt_every == 0):
                x, aux = torch.utils.checkpoint.checkpoint(
                    block,
                    x,
                    positions,
                    use_reentrant=False,
                )
            else:
                x, aux = block(x, positions)
            aux_losses.append(aux)

        if aux_losses:
            aux_loss = torch.stack(aux_losses).mean()
        else:
            aux_loss = torch.zeros((), device=x.device, dtype=x.dtype)

        x = self.norm(x)
        logits = self.head(x)
        return logits, aux_loss

    def num_parameters(self, only_trainable: bool = False) -> int:
        """Count parameters (excludes duplicates from weight tying)."""
        if only_trainable:
            return sum(p.numel() for p in self.parameters() if p.requires_grad)
        seen_ids = set()
        total = 0
        for p in self.parameters():
            if id(p) in seen_ids:
                continue
            seen_ids.add(id(p))
            total += p.numel()
        return total

    def num_active_parameters(self) -> int:
        """Estimate active parameters per token (top-2 routed + 1 shared)."""
        seen_ids: set[int] = set()
        non_moe = 0
        for name, p in self.named_parameters():
            if "experts" in name:
                continue
            if id(p) in seen_ids:
                continue
            seen_ids.add(id(p))
            non_moe += p.numel()
        d_model = self.cfg.d_model
        ffn_dim = self.cfg.ffn_dim
        n_layers = self.cfg.n_layers
        expert_params = 3 * d_model * ffn_dim
        moe_active = (self.cfg.n_activated_experts + self.cfg.n_shared_experts) * expert_params
        router_params = d_model * self.cfg.n_routed_experts
        return non_moe + (moe_active + router_params) * n_layers

    def enable_gradient_checkpointing(self, every: int = 3) -> None:
        """Enable gradient checkpointing on every Nth block to save memory."""
        self.gradient_checkpointing = True
        self.grad_ckpt_every = every

    def extra_repr(self) -> str:
        return (
            f"d_model={self.cfg.d_model}, n_layers={self.cfg.n_layers}, "
            f"vocab={self.cfg.vocab_size}, "
            f"experts={self.cfg.n_routed_experts}×top{self.cfg.n_activated_experts}"
            f"+{self.cfg.n_shared_experts}shared, "
            f"window={self.cfg.window_size}, "
            f"params={self.num_parameters() / 1e6:.1f}M"
        )