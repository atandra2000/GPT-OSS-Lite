"""Profile GPT-OSS-Lite components to identify bottlenecks."""
import sys
import time
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

import torch

from models.attention import (
    GPTOSSAttention,
    full_causal_attention,
    manual_causal_attention,
    repeat_kv,
    sliding_window_attention,
)
from models.moe import MoELayer, MoERouter, SwiGLUExpert, aux_load_balancing_loss
from models.rotary import apply_rope, compute_yarn_freqs
from models.transformer import GPTOSS, ModelConfig


def time_fn(fn, n=20, warmup=3):
    """Time fn() in ms; average over n runs after warmup."""
    for _ in range(warmup):
        fn()
    t0 = time.perf_counter()
    for _ in range(n):
        fn()
    elapsed = (time.perf_counter() - t0) / n * 1000
    return elapsed


def main():
    device = torch.device("cpu")

    cfg = ModelConfig(
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
    model = GPTOSS(cfg)
    if torch.cuda.is_available():
        model = model.to(memory_format=torch.channels_last)
    print(f"Total params: {model.num_parameters()/1e6:.2f}M")

    B, T = 2, cfg.max_seq_len
    idx = torch.randint(0, cfg.vocab_size, (B, T))

    def model_forward():
        with torch.no_grad():
            model(idx)
    t = time_fn(model_forward)
    print(f"[model.forward]      {t:.2f} ms/step")

    attn_win = GPTOSSAttention(cfg.as_dict(), layer_idx=0)
    x = torch.randn(B, T, cfg.d_model)
    def attn_win_fwd():
        with torch.no_grad():
            attn_win(x)
    t = time_fn(attn_win_fwd)
    print(f"[attn.windowed]      {t:.2f} ms/step")

    attn_full = GPTOSSAttention(cfg.as_dict(), layer_idx=1)
    def attn_full_fwd():
        with torch.no_grad():
            attn_full(x)
    t = time_fn(attn_full_fwd)
    print(f"[attn.global]        {t:.2f} ms/step")

    q = torch.randn(B, cfg.n_heads, T, cfg.head_dim)
    k = torch.randn(B, cfg.n_heads, T, cfg.head_dim)
    v = torch.randn(B, cfg.n_heads, T, cfg.head_dim)
    def manual_attn():
        manual_causal_attention(q, k, v)
    t = time_fn(manual_attn)
    print(f"[manual_attn]        {t:.2f} ms/step")

    def sdpa_attn():
        torch.nn.functional.scaled_dot_product_attention(q, k, v, is_causal=True)
    t = time_fn(sdpa_attn)
    print(f"[sdpa_attn]          {t:.2f} ms/step")

    def swa_attn():
        sliding_window_attention(q, k, v, window=cfg.window_size)
    t = time_fn(swa_attn)
    print(f"[swa_attn]           {t:.2f} ms/step")

    moe = MoELayer(cfg.as_dict())
    def moe_fwd():
        with torch.no_grad():
            moe(x)
    t = time_fn(moe_fwd)
    print(f"[moe.forward]        {t:.2f} ms/step")

    N = B * T
    flat = x.view(-1, cfg.d_model)
    indices, weights, _ = moe.router(flat)
    def moe_dispatch():
        with torch.no_grad():
            moe._dispatch_grouped(flat, indices, weights)
    t = time_fn(moe_dispatch)
    print(f"[moe.dispatch]       {t:.2f} ms/step")

    freqs = torch.randn(T, cfg.head_dim // 2)
    cos = torch.randn(T, cfg.head_dim // 2)
    sin = torch.randn(T, cfg.head_dim // 2)
    def rope_apply():
        apply_rope(q, cos, sin)
    t = time_fn(rope_apply)
    print(f"[apply_rope]         {t:.2f} ms/step")

    k_kv = torch.randn(B, cfg.n_kv_heads, T, cfg.head_dim)
    def repeat():
        repeat_kv(k_kv, cfg.n_heads // cfg.n_kv_heads)
    t = time_fn(repeat)
    print(f"[repeat_kv]          {t:.2f} ms/step")


if __name__ == "__main__":
    main()