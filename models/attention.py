"""Sliding-window + full attention alternation with learned attention-sink bias."""
import math
import functools
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.yarn import YaRNRoPE
from models.rotary import apply_rope

# Sink bias is clamped at forward time to keep the SDPA mask-add within
# BF16's representable range. Unclamped parameter retains gradient flow.
SINK_CLAMP_MIN = -10.0
SINK_CLAMP_MAX = 15.0


def manual_causal_attention(
    query_states: torch.Tensor,
    key_states: torch.Tensor,
    value_states: torch.Tensor,
    sink_bias: torch.Tensor | None = None,
    window: int | None = None,
) -> torch.Tensor:
    """Naive O(T²) causal attention (reference path for tests only)."""
    B, H, T, D = query_states.shape
    scores = (query_states.float() @ key_states.float().transpose(-2, -1)) / math.sqrt(D)

    causal = torch.triu(torch.ones(T, T, dtype=torch.bool, device=query_states.device), diagonal=1)
    scores = scores.masked_fill(causal, float("-inf"))

    if window is not None and window < T:
        idx = torch.arange(T, device=query_states.device)
        outside = idx.unsqueeze(0) - idx.unsqueeze(1) >= window
        scores = scores.masked_fill(outside, float("-inf"))

    if sink_bias is not None:
        sink_logit = sink_bias.view(1, H, 1, 1).to(scores.dtype)
        augmented = torch.cat([scores, sink_logit.expand(B, H, T, 1)], dim=-1)
        attn_weights = F.softmax(augmented, dim=-1)
        attn_weights = attn_weights[..., :T]
        return (attn_weights.to(value_states.dtype) @ value_states)
    attn_weights = F.softmax(scores, dim=-1)
    return (attn_weights.to(value_states.dtype) @ value_states)


@functools.lru_cache(maxsize=None)
def _causal_mask(T: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    """Causal mask of shape ``(T, T)``: True where attention is allowed."""
    idx = torch.arange(T, device=device)
    return idx.unsqueeze(1) >= idx.unsqueeze(0)  # (T_q, T_k)


@functools.lru_cache(maxsize=None)
def _window_mask(T_q: int, T_k: int, window: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    """Sliding-window mask of shape ``(T_q, T_k)``: True where attention is allowed."""
    if T_q == T_k:
        idx = torch.arange(T_q, device=device)
        return (idx.unsqueeze(0) - idx.unsqueeze(1) < window) & _causal_mask(T_q, device, dtype)
    # Decode: T_q=1, T_k grows. Query position is T_k - 1.
    idx_q = torch.tensor([T_k - 1], device=device)
    idx_k = torch.arange(T_k, device=device)
    return (idx_q.unsqueeze(-1) - idx_k.unsqueeze(0) < window)


def causal_attention(
    query_states: torch.Tensor,
    key_states: torch.Tensor,
    value_states: torch.Tensor,
    window: int | None = None,
    sink_bias: torch.Tensor | None = None,
) -> torch.Tensor:
    """Causal attention via SDPA. ``window=None`` is full causal; else sliding window.

    When ``sink_bias`` is provided, an extra "sink" key with value 0 is added
    so the learned bias contributes to the softmax denominator. The output
    dimension is unchanged (sink is appended, then stripped).
    """
    T_q = query_states.shape[2]
    T_k = key_states.shape[2]
    B, H = query_states.shape[:2]
    device, dtype = query_states.device, query_states.dtype

    if sink_bias is None:
        if window is None:
            if T_q == T_k:
                return F.scaled_dot_product_attention(query_states, key_states, value_states, is_causal=True)
            return F.scaled_dot_product_attention(query_states, key_states, value_states)
        # window < T_k path
        if T_q == T_k:
            mask = _causal_mask(T_q, device, dtype) & _window_mask(T_q, T_k, window, device, dtype)
        else:
            mask = _window_mask(T_q, T_k, window, device, dtype)
        attn_mask = torch.where(mask, 0.0, float("-inf")).to(dtype).unsqueeze(0).unsqueeze(0)
        return F.scaled_dot_product_attention(query_states, key_states, value_states, attn_mask=attn_mask)

    # sink path: extend K with a sink column (zero K, zero V) and let the bias
    # ride on the mask's last column.
    sink_k = torch.zeros(B, H, 1, query_states.shape[-1], device=device, dtype=dtype)
    sink_v = torch.zeros(B, H, 1, value_states.shape[-1], device=device, dtype=value_states.dtype)
    k_ext = torch.cat([key_states, sink_k], dim=2)
    v_ext = torch.cat([value_states, sink_v], dim=2)

    if window is None or window >= T_k:
        # Full causal mask + sink column
        causal = _causal_mask(T_q, device, dtype) if T_q == T_k else torch.ones(T_q, T_k, dtype=torch.bool, device=device)
    elif T_q == T_k:
        causal = _causal_mask(T_q, device, dtype) & _window_mask(T_q, T_k, window, device, dtype)
    else:
        causal = _window_mask(T_q, T_k, window, device, dtype)

    mask = torch.zeros(H, T_q, T_k + 1, device=device, dtype=dtype)
    mask[:, :, :T_k] = causal.to(dtype)
    mask[:, :, T_k] = sink_bias.to(dtype).unsqueeze(1).expand(H, T_q)
    return F.scaled_dot_product_attention(query_states, k_ext, v_ext, attn_mask=mask.unsqueeze(0))


def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    """Repeat KV heads ``n_rep`` times to match the number of query heads (GQA)."""
    if n_rep == 1:
        return x
    B, H_kv, T, D = x.shape
    x = x[:, :, None, :, :]
    x = x.expand(B, H_kv, n_rep, T, D)
    return x.reshape(B, H_kv * n_rep, T, D)


# Back-compat thin wrappers — production code uses causal_attention directly.
def sliding_window_attention(q, k, v, window: int = 128, sink_bias=None):
    return causal_attention(q, k, v, window=window, sink_bias=sink_bias)


def full_causal_attention(q, k, v, sink_bias=None):
    return causal_attention(q, k, v, window=None, sink_bias=sink_bias)


class GPTOSSAttention(nn.Module):
    """GPT-OSS attention layer: GQA + YaRN RoPE + learned sink bias + alternating SWA/full."""

    def __init__(self, cfg, layer_idx: int):
        super().__init__()
        self.cfg = cfg
        self.layer_idx = layer_idx
        self.d_model = cfg.d_model
        self.n_heads = cfg.n_heads
        self.n_kv_heads = cfg.n_kv_heads
        self.head_dim = cfg.head_dim
        self.window_size = cfg.window_size
        self.n_rep = self.n_heads // self.n_kv_heads
        self.is_windowed = (layer_idx % 2 == 0)
        self.prune_rope_global = bool(getattr(cfg, "yarn_prune_rope_global", True))

        self.q_proj = nn.Linear(self.d_model, self.n_heads * self.head_dim, bias=False)
        self.kv_proj = nn.Linear(self.d_model, 2 * self.n_kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.n_heads * self.head_dim, self.d_model, bias=False)

        if cfg.sink_bias:
            self.sink_bias = nn.Parameter(torch.zeros(self.n_heads))
        else:
            self.register_parameter("sink_bias", None)

        self.yarn = YaRNRoPE(
            head_dim=self.head_dim,
            theta=cfg.rope_theta,
            scale_factor=cfg.yarn_scale_factor,
            original_max_seq_len=cfg.yarn_original_max_seq_len,
            target_seq_len=cfg.yarn_target_seq_len,
            beta_fast=cfg.yarn_beta_fast,
            beta_slow=cfg.yarn_beta_slow,
            mscale=cfg.yarn_mscale,
        )

    def _n_pruned_dims(self) -> int:
        if (not self.is_windowed) and self.prune_rope_global:
            return self.head_dim // 4
        return 0

    def forward(
        self,
        x: torch.Tensor,
        positions: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Forward pass returning ``(B, T, d_model)`` attended output."""
        B, T, _ = x.shape
        if positions is None:
            positions = torch.arange(T, device=x.device)

        query_states = self.q_proj(x).view(B, T, self.n_heads, self.head_dim)
        kv = self.kv_proj(x).view(B, T, 2, self.n_kv_heads, self.head_dim)
        key_states, value_states = kv[:, :, 0], kv[:, :, 1]

        query_states = query_states.transpose(1, 2)
        key_states = key_states.transpose(1, 2)
        value_states = value_states.transpose(1, 2)

        cos, sin = self.yarn(positions, n_pruned_dims=self._n_pruned_dims())
        query_states = apply_rope(query_states, cos, sin)
        key_states = apply_rope(key_states, cos, sin)

        key_states = key_states.repeat_interleave(self.n_rep, dim=1)
        value_states = value_states.repeat_interleave(self.n_rep, dim=1)

        if self.sink_bias is not None:
            sink_bias_clamped = self.sink_bias.clamp(SINK_CLAMP_MIN, SINK_CLAMP_MAX)
        else:
            sink_bias_clamped = None

        out = causal_attention(
            query_states, key_states, value_states,
            window=self.window_size if self.is_windowed else None,
            sink_bias=sink_bias_clamped,
        )
        out = out.transpose(1, 2).contiguous().view(B, T, self.n_heads * self.head_dim)
        return self.o_proj(out)

    def extra_repr(self) -> str:
        mode = "SWA" if self.is_windowed else "Full"
        n_pruned = self._n_pruned_dims()
        n_pruned_s = f", pruned={n_pruned}" if n_pruned > 0 else ""
        return f"layer={self.layer_idx} ({mode}{n_pruned_s}), H={self.n_heads}/{self.n_kv_heads}, D={self.head_dim}, window={self.window_size}"