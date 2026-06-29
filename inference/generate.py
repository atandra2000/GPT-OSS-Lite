"""Mixed KV-cache generation for GPT-OSS-Lite."""
from typing import List, Optional, Tuple

import torch
import torch.nn.functional as F

from models.attention import (
    apply_rope,
    full_causal_attention,
    manual_causal_attention,
    repeat_kv,
    sliding_window_attention,
)
from models.transformer import GPTOSS


class MixedKVCache:
    """Per-layer mixed KV cache (windowed + global). Stores *rotated* K."""

    _GLOBAL_CAP_TOKENS = 4_000_000

    def __init__(self, global_cap_tokens: int | None = None):
        self.windowed_kv: List[List[Optional[torch.Tensor]]] = []
        self.global_kv: List[List[Optional[torch.Tensor]]] = []
        self.global_lengths: List[int] = []
        self.global_caps: List[int] = []
        self._global_cap_tokens = (
            global_cap_tokens if global_cap_tokens is not None
            else self._GLOBAL_CAP_TOKENS
        )

    def reset(self) -> None:
        """Empty the cache (call between independent generations)."""
        self.windowed_kv = []
        self.global_kv = []
        self.global_lengths = []
        self.global_caps = []

    def __len__(self) -> int:
        return max(len(self.windowed_kv), len(self.global_kv))

    def append(
        self,
        layer_idx: int,
        k_rot: torch.Tensor,
        v: torch.Tensor,
        is_windowed: bool,
        window: int,
    ) -> None:
        """Append a new *rotated* K and the corresponding V for layer ``layer_idx``."""
        if is_windowed:
            target = self.windowed_kv
            while len(target) <= layer_idx:
                target.append([None, None, 0, 0])
            entry = target[layer_idx]
            B, H, T_new, D = k_rot.shape

            if entry[0] is None:
                buf_k = torch.zeros(B, H, window, D, dtype=k_rot.dtype, device=k_rot.device)
                buf_v = torch.zeros(B, H, window, D, dtype=v.dtype, device=v.device)
                if T_new >= window:
                    buf_k[:, :, :window, :] = k_rot[:, :, -window:, :]
                    buf_v[:, :, :window, :] = v[:, :, -window:, :]
                    head = 0
                    count = window
                else:
                    buf_k[:, :, :T_new, :] = k_rot
                    buf_v[:, :, :T_new, :] = v
                    head = T_new % window
                    count = T_new
                target[layer_idx] = [buf_k, buf_v, head, count]
            else:
                old_k, old_v, head, count = entry
                if T_new >= window:
                    old_k.copy_(k_rot[:, :, -window:, :])
                    old_v.copy_(v[:, :, -window:, :])
                    target[layer_idx] = [old_k, old_v, 0, window]
                else:
                    if head + T_new <= window:
                        old_k[:, :, head:head+T_new, :] = k_rot
                        old_v[:, :, head:head+T_new, :] = v
                    else:
                        first_chunk = window - head
                        second_chunk = T_new - first_chunk
                        old_k[:, :, head:, :] = k_rot[:, :, :first_chunk, :]
                        old_v[:, :, head:, :] = v[:, :, :first_chunk, :]
                        old_k[:, :, :second_chunk, :] = k_rot[:, :, first_chunk:, :]
                        old_v[:, :, :second_chunk, :] = v[:, :, first_chunk:, :]

                    new_head = (head + T_new) % window
                    new_count = min(window, count + T_new)
                    target[layer_idx] = [old_k, old_v, new_head, new_count]
        else:
            target = self.global_kv
            while len(target) <= layer_idx:
                target.append([None, None])
                self.global_lengths.append(0)
                self.global_caps.append(0)
            entry = target[layer_idx]
            B, H, T_new, D = k_rot.shape
            cur_len = self.global_lengths[layer_idx]
            cur_cap = self.global_caps[layer_idx]
            needed = cur_len + T_new
            if entry[0] is None:
                new_cap = max(needed, 1)
                buf_k = torch.empty(B, H, new_cap, D, dtype=k_rot.dtype, device=k_rot.device)
                buf_v = torch.empty(B, H, new_cap, D, dtype=v.dtype, device=v.device)
                buf_k[:, :, :T_new, :].copy_(k_rot)
                buf_v[:, :, :T_new, :].copy_(v)
                target[layer_idx] = [buf_k, buf_v]
                self.global_lengths[layer_idx] = T_new
                self.global_caps[layer_idx] = new_cap
            else:
                if needed > cur_cap:
                    new_cap = max(needed, int(cur_cap * 1.5) + 1)
                    new_cap = min(new_cap, self._global_cap_tokens)
                    old_k, old_v = entry
                    buf_k = torch.empty(B, H, new_cap, D, dtype=old_k.dtype, device=old_k.device)
                    buf_v = torch.empty(B, H, new_cap, D, dtype=old_v.dtype, device=old_v.device)
                    buf_k[:, :, :cur_len, :].copy_(old_k[:, :, :cur_len, :])
                    buf_v[:, :, :cur_len, :].copy_(old_v[:, :, :cur_len, :])
                    buf_k[:, :, cur_len:needed, :].copy_(k_rot)
                    buf_v[:, :, cur_len:needed, :].copy_(v)
                    target[layer_idx] = [buf_k, buf_v]
                    self.global_caps[layer_idx] = new_cap
                else:
                    old_k, old_v = entry
                    old_k[:, :, cur_len:needed, :].copy_(k_rot)
                    old_v[:, :, cur_len:needed, :].copy_(v)
                self.global_lengths[layer_idx] = needed

    def get(self, layer_idx: int, is_windowed: bool) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        """Return ``(K, V)`` for a given layer; ``(None, None)`` if empty."""
        if is_windowed:
            target = self.windowed_kv
            if layer_idx >= len(target) or target[layer_idx][0] is None:
                return None, None

            k, v, head, count = target[layer_idx]
            if count == 0:
                return None, None

            if count < k.size(2):
                return k[:, :, :count, :], v[:, :, :count, :]

            if head == 0:
                return k, v

            k_ordered = torch.cat([k[:, :, head:, :], k[:, :, :head, :]], dim=2)
            v_ordered = torch.cat([v[:, :, head:, :], v[:, :, :head, :]], dim=2)
            return k_ordered, v_ordered
        else:
            target = self.global_kv
            if layer_idx >= len(target):
                return None, None
            k, v = target[layer_idx]
            cur_len = self.global_lengths[layer_idx]
            if k is None or cur_len == 0:
                return None, None
            return k[:, :, :cur_len, :], v[:, :, :cur_len, :]

    def seq_len(self, layer_idx: int, is_windowed: bool) -> int:
        """Return the current sequence length cached at ``layer_idx``."""
        if is_windowed:
            target = self.windowed_kv
            if layer_idx >= len(target) or target[layer_idx][0] is None:
                return 0
            return target[layer_idx][3]
        if layer_idx >= len(self.global_kv):
            return 0
        return self.global_lengths[layer_idx]


def _attention_for_layer(
    attn,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
) -> torch.Tensor:
    """Dispatch to the right attention function based on layer type + cfg."""
    is_windowed = attn.is_windowed
    window = attn.window_size
    attn_impl = attn.cfg.get("attn_impl", "sdpa")
    sink_bias = (
        attn.sink_bias.clamp(attn._sink_clamp_min, attn._sink_clamp_max)
        if attn.sink_bias is not None else None
    )
    if attn_impl == "manual":
        return manual_causal_attention(
            q, k, v, sink_bias=sink_bias,
            window=window if is_windowed else None,
        )
    if is_windowed:
        return sliding_window_attention(q, k, v, window=window, sink_bias=sink_bias)
    return full_causal_attention(q, k, v, sink_bias=sink_bias)


def _attn_forward_with_cache(
    model: GPTOSS,
    x: torch.Tensor,
    positions: torch.Tensor,
    cache: Optional[MixedKVCache],
    full_history_for_nocache: Optional[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = None,
):
    """Use _attn_forward_layer instead."""
    raise NotImplementedError("Use _attn_forward_layer instead.")


def _attn_forward_layer(
    block,
    layer_idx: int,
    x: torch.Tensor,
    positions: torch.Tensor,
    cache: Optional[MixedKVCache],
    sink_bias_cache: Optional[dict] = None,
) -> torch.Tensor:
    """Run one GPTOSSBlock; if cache is provided, append rotated K/V to it."""
    attn = block.attn
    B, T, _ = x.shape
    x_norm = block.norm1(x)
    q = attn.q_proj(x_norm).view(B, T, attn.n_heads, attn.head_dim).transpose(1, 2)
    kv = attn.kv_proj(x_norm).view(B, T, 2, attn.n_kv_heads, attn.head_dim)
    k_new = kv[:, :, 0].transpose(1, 2)
    v_new = kv[:, :, 1].transpose(1, 2)
    cos, sin = attn.yarn(positions, n_pruned_dims=attn.n_pruned_dims)
    q = apply_rope(q, cos, sin)
    k_new_rot = apply_rope(k_new, cos, sin)

    if cache is not None:
        cache.append(layer_idx, k_new_rot, v_new, attn.is_windowed, attn.window_size)
        cached_k, cached_v = cache.get(layer_idx, attn.is_windowed)
        k_for_q = cached_k
        v_for_q = cached_v
    else:
        k_for_q = k_new_rot
        v_for_q = v_new

    k_for_q = repeat_kv(k_for_q, attn.n_rep)
    v_for_q = repeat_kv(v_for_q, attn.n_rep)

    if attn.sink_bias is not None:
        if sink_bias_cache is not None and id(attn) in sink_bias_cache:
            sink_bias_clamped = sink_bias_cache[id(attn)]
        else:
            sink_bias_clamped = attn.sink_bias.clamp(
                attn._sink_clamp_min, attn._sink_clamp_max
            )
            if sink_bias_cache is not None:
                sink_bias_cache[id(attn)] = sink_bias_clamped
    else:
        sink_bias_clamped = None

    attn_impl = attn.cfg.get("attn_impl", "sdpa")
    if attn_impl == "manual":
        out = manual_causal_attention(
            q, k_for_q, v_for_q, sink_bias=sink_bias_clamped,
            window=attn.window_size if attn.is_windowed else None,
        )
    elif attn.is_windowed:
        out = sliding_window_attention(
            q, k_for_q, v_for_q, window=attn.window_size, sink_bias=sink_bias_clamped,
        )
    else:
        out = full_causal_attention(q, k_for_q, v_for_q, sink_bias=sink_bias_clamped)

    out = out.transpose(1, 2).contiguous().view(B, T, attn.n_heads * attn.head_dim)
    out = attn.o_proj(out)
    x = x + out
    moe_out, _ = block.moe(block.norm2(x))
    x = x + moe_out
    return x


@torch.no_grad()
def generate(
    model: GPTOSS,
    input_ids: torch.Tensor,
    max_new_tokens: int = 64,
    temperature: float = 0.7,
    top_p: float = 0.9,
    use_cache: bool = True,
) -> torch.Tensor:
    """Token-by-token generation with mixed KV cache; returns ``(B, T + max_new_tokens)``."""
    model.eval()
    dev = input_ids.device
    B, T_prompt = input_ids.shape
    cache = MixedKVCache() if use_cache else None
    sink_bias_cache: dict = {}

    out_total_len = T_prompt + max_new_tokens
    output = torch.empty(B, out_total_len, dtype=input_ids.dtype, device=dev)
    output[:, :T_prompt] = input_ids

    x = model.embed(input_ids)
    positions = torch.arange(T_prompt, device=dev)
    for layer_idx, block in enumerate(model.blocks):
        x = _attn_forward_layer(block, layer_idx, x, positions, cache, sink_bias_cache)
    x = model.norm(x)
    next_token_logits = model.head(x)[:, -1, :]

    cur_pos = T_prompt
    for step in range(max_new_tokens):
        if temperature <= 0:
            next_id = next_token_logits.argmax(dim=-1, keepdim=True)
        else:
            probs = F.softmax(next_token_logits / temperature, dim=-1)
            sorted_probs, sorted_idx = probs.sort(dim=-1, descending=True)
            cumsum = sorted_probs.cumsum(dim=-1)
            mask = cumsum - sorted_probs > top_p
            sorted_probs[mask] = 0.0
            sorted_probs = sorted_probs / sorted_probs.sum(dim=-1, keepdim=True).clamp(min=1e-10)
            next_id = torch.multinomial(sorted_probs, 1)
            next_id = sorted_idx.gather(-1, next_id)
        output[:, T_prompt + step : T_prompt + step + 1] = next_id
        cur_pos += 1

        if not use_cache:
            full_input = output[:, : T_prompt + step + 1]
            x_full = model.embed(full_input)
            positions_full = torch.arange(full_input.size(1), device=dev)
            for layer_idx, block in enumerate(model.blocks):
                x_full = _attn_forward_layer(block, layer_idx, x_full, positions_full, None, sink_bias_cache)
            x_step = model.norm(x_full)[:, -1:, :]
            next_token_logits = model.head(x_step)[:, -1, :]
            continue

        x_step = model.embed(next_id)
        positions_step = torch.tensor([cur_pos - 1], device=dev)
        for layer_idx, block in enumerate(model.blocks):
            x_step = _attn_forward_layer(block, layer_idx, x_step, positions_step, cache, sink_bias_cache)
        x_step = model.norm(x_step)
        next_token_logits = model.head(x_step)[:, -1, :]
    return output