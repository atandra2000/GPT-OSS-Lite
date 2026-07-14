"""Sliding-window + full attention alternation with learned attention-sink bias."""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.yarn import YaRNRoPE
from models.rotary import apply_rope


def manual_causal_attention(
    query_states: torch.Tensor,
    key_states: torch.Tensor,
    value_states: torch.Tensor,
    sink_bias: torch.Tensor | None = None,
    window: int | None = None,
) -> torch.Tensor:
    """Naive O(T²) causal attention (reference path)."""
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
    else:
        attn_weights = F.softmax(scores, dim=-1)
        return (attn_weights.to(value_states.dtype) @ value_states)


_SLIDING_WINDOW_MASK_CACHE: dict = {}


def _get_sliding_window_mask(T: int, window: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    """Get a cached sliding-window causal mask of shape ``(T, T)``."""
    key = (T, window, device, dtype)
    cached = _SLIDING_WINDOW_MASK_CACHE.get(key)
    if cached is not None:
        return cached
    idx = torch.arange(T, device=device)
    outside = idx.unsqueeze(0) - idx.unsqueeze(1) >= window
    causal = idx.unsqueeze(1) - idx.unsqueeze(0) < 0
    mask = outside | causal
    out = torch.zeros(T, T, device=device, dtype=torch.float32)
    out = out.masked_fill(mask, float("-inf"))
    out = out.to(dtype)
    _SLIDING_WINDOW_MASK_CACHE[key] = out
    return out


def _get_decode_window_mask(T_k: int, window: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    """Get a cached decode-time (T_q=1) sliding-window mask of shape ``(1, T_k)``.

    During autoregressive decode, the query tensor is always T_q=1 (single new
    token) and the key tensor grows by 1 each step. The mask shape is therefore
    fixed at ``(1, T_k)`` but its *contents* change as T_k grows. The expensive
    part — the ``arange(T_k)``, the broadcast, the ``masked_fill`` — only
    depends on ``(T_k, window)``, not on which step we're at, so we cache it.

    Used by the no-sink path in ``sliding_window_attention`` (decode branch
    at lines ~117-122). Saves ~50μs per decode token per layer.
    """
    key = ("decode", T_k, window, device, dtype)
    cached = _SLIDING_WINDOW_MASK_CACHE.get(key)
    if cached is not None:
        return cached
    # Query position is always (T_k - 1) during decode: the single new token
    # attends to itself and the previous (T_k - 1) cached tokens.
    idx_q = torch.tensor([T_k - 1], device=device, dtype=torch.long)
    idx_k = torch.arange(T_k, device=device)
    outside = (idx_q.unsqueeze(-1) - idx_k.unsqueeze(0)) >= window
    out = torch.zeros(1, T_k, device=device, dtype=torch.float32)
    out = out.masked_fill(outside, float("-inf"))
    out = out.to(dtype)
    _SLIDING_WINDOW_MASK_CACHE[key] = out
    return out


def _build_sink_sliding_window_mask(
    T_q: int,
    T_k: int,
    H: int,
    window: int,
    sink_bias: torch.Tensor,
    q_device: torch.device,
    q_dtype: torch.dtype,
) -> torch.Tensor:
    """Build the cached attention mask for sliding-window + sink bias path."""
    key = ("sink_swa", T_q, T_k, H, window, q_device, q_dtype)
    cached = _SLIDING_WINDOW_MASK_CACHE.get(key)
    if cached is not None:
        return cached
    if T_q == T_k:
        idx = torch.arange(T_k, device=q_device)
        causal_real = torch.zeros(T_k, T_k, device=q_device, dtype=q_dtype)
        causal_real = causal_real.masked_fill(
            idx.unsqueeze(1) - idx.unsqueeze(0) < 0, float("-inf")
        )
        if window < T_k:
            causal_real = causal_real.masked_fill(
                (idx.unsqueeze(0) - idx.unsqueeze(1) >= window),
                float("-inf"),
            )
    else:
        idx_q = torch.full((T_q,), T_k - 1, device=q_device, dtype=torch.long)
        idx_k = torch.arange(T_k, device=q_device)
        causal_real = torch.zeros(T_q, T_k, device=q_device, dtype=q_dtype)
        if window < T_k:
            causal_real = causal_real.masked_fill(
                (idx_q.unsqueeze(-1) - idx_k.unsqueeze(0)) >= window,
                float("-inf"),
            )
    mask = torch.zeros(H, T_q, T_k + 1, device=q_device, dtype=q_dtype)
    mask[:, :, :T_k] = causal_real.unsqueeze(0)
    _SLIDING_WINDOW_MASK_CACHE[key] = mask
    return mask


def sliding_window_attention(
    query_states: torch.Tensor,
    key_states: torch.Tensor,
    value_states: torch.Tensor,
    window: int = 128,
    sink_bias: torch.Tensor | None = None,
) -> torch.Tensor:
    """Sliding-window causal attention via SDPA with optional sink bias."""
    B, H, T_q, D = query_states.shape
    T_k = key_states.shape[2]

    if sink_bias is None:
        if T_q == T_k:
            mask = _get_sliding_window_mask(T_k, window, query_states.device, query_states.dtype)
            attn_mask = mask.unsqueeze(0).unsqueeze(0)
        else:
            # Decode path: T_q=1, T_k grows. Use the decode-specific cache
            # keyed by T_k to avoid rebuilding the (1, T_k) mask every token.
            mask = _get_decode_window_mask(T_k, window, query_states.device, query_states.dtype)
            attn_mask = mask.unsqueeze(0).unsqueeze(0)
        return F.scaled_dot_product_attention(query_states, key_states, value_states, attn_mask=attn_mask)

    sink_k = torch.zeros(B, H, 1, D, device=query_states.device, dtype=query_states.dtype)
    sink_v = torch.zeros(B, H, 1, D, device=query_states.device, dtype=value_states.dtype)
    k_ext = torch.cat([key_states, sink_k], dim=2)
    v_ext = torch.cat([value_states, sink_v], dim=2)

    mask = _build_sink_sliding_window_mask(
        T_q, T_k, H, window, sink_bias, query_states.device, query_states.dtype
    )
    mask_with_sink = mask.clone()
    sink_col = sink_bias.to(query_states.dtype).unsqueeze(1).expand(H, T_q)
    mask_with_sink[:, :, T_k] = sink_col
    return F.scaled_dot_product_attention(query_states, k_ext, v_ext, attn_mask=mask_with_sink.unsqueeze(0))


def full_causal_attention(
    query_states: torch.Tensor,
    key_states: torch.Tensor,
    value_states: torch.Tensor,
    sink_bias: torch.Tensor | None = None,
) -> torch.Tensor:
    """Full causal attention via SDPA with optional sink bias."""
    T_q = query_states.shape[2]
    T_k = key_states.shape[2]
    if sink_bias is None and T_q == T_k:
        return F.scaled_dot_product_attention(query_states, key_states, value_states, is_causal=True)
    if sink_bias is None and T_q < T_k:
        return F.scaled_dot_product_attention(query_states, key_states, value_states)
    return sliding_window_attention(query_states, key_states, value_states, window=max(T_q, T_k), sink_bias=sink_bias)


def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    """Repeat KV heads ``n_rep`` times to match the number of query heads (GQA)."""
    if n_rep == 1:
        return x
    B, H_kv, T, D = x.shape
    x = x[:, :, None, :, :]
    x = x.expand(B, H_kv, n_rep, T, D)
    return x.reshape(B, H_kv * n_rep, T, D)


class GPTOSSAttention(nn.Module):
    """GPT-OSS attention layer: GQA + YaRN RoPE + learned sink bias + alternating SWA/full."""

    def __init__(self, cfg: dict, layer_idx: int):
        super().__init__()
        self.cfg = cfg
        self.layer_idx = layer_idx
        self.d_model = cfg["d_model"]
        self.n_heads = cfg["n_heads"]
        self.n_kv_heads = cfg["n_kv_heads"]
        self.head_dim = cfg["head_dim"]
        self.window_size = cfg["window_size"]
        self.n_rep = self.n_heads // self.n_kv_heads
        self.is_windowed = (layer_idx % 2 == 0)

        self.q_proj = nn.Linear(self.d_model, self.n_heads * self.head_dim, bias=False)
        self.kv_proj = nn.Linear(self.d_model, 2 * self.n_kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.n_heads * self.head_dim, self.d_model, bias=False)

        self._sink_clamp_min = -10.0
        self._sink_clamp_max = 15.0
        if cfg.get("sink_bias", True):
            self.sink_bias = nn.Parameter(torch.zeros(self.n_heads))
        else:
            self.register_parameter("sink_bias", None)

        n_pruned_dims = 0
        if (not self.is_windowed) and cfg.get("yarn_prune_rope_global", True):
            n_pruned_dims = self.head_dim // 4
        self.n_pruned_dims = n_pruned_dims
        self.yarn = YaRNRoPE(
            head_dim=self.head_dim,
            theta=cfg["rope_theta"],
            scale_factor=cfg["yarn_scale_factor"],
            original_max_seq_len=cfg["yarn_original_max_seq_len"],
            target_seq_len=cfg["yarn_target_seq_len"],
            beta_fast=cfg["yarn_beta_fast"],
            beta_slow=cfg["yarn_beta_slow"],
            mscale=cfg["yarn_mscale"],
        )

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

        cos, sin = self.yarn(positions, n_pruned_dims=self.n_pruned_dims)
        query_states = apply_rope(query_states, cos, sin)
        key_states = apply_rope(key_states, cos, sin)

        key_states = repeat_kv(key_states, self.n_rep)
        value_states = repeat_kv(value_states, self.n_rep)

        if self.sink_bias is not None:
            sink_bias_clamped = self.sink_bias.clamp(self._sink_clamp_min, self._sink_clamp_max)
        else:
            sink_bias_clamped = None

        if self.cfg.get("attn_impl", "sdpa") == "manual":
            out = manual_causal_attention(
                query_states, key_states, value_states,
                sink_bias=sink_bias_clamped,
                window=self.window_size if self.is_windowed else None,
            )
        else:
            if self.is_windowed:
                out = sliding_window_attention(query_states, key_states, value_states, window=self.window_size, sink_bias=sink_bias_clamped)
            else:
                out = full_causal_attention(query_states, key_states, value_states, sink_bias=sink_bias_clamped)

        out = out.transpose(1, 2).contiguous().view(B, T, self.n_heads * self.head_dim)
        return self.o_proj(out)

    def extra_repr(self) -> str:
        mode = "SWA" if self.is_windowed else "Full"
        n_pruned = f", pruned={self.n_pruned_dims}" if self.n_pruned_dims > 0 else ""
        return f"layer={self.layer_idx} ({mode}{n_pruned}), H={self.n_heads}/{self.n_kv_heads}, D={self.head_dim}, window={self.window_size}"