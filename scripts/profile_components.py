"""Profile GPT-OSS-Lite components to identify bottlenecks."""
import torch

from _bootstrap import micro_cfg, time_fn
from models.attention import (
    GPTOSSAttention,
    full_causal_attention,
    manual_causal_attention,
    repeat_kv,
    sliding_window_attention,
)
from models.moe import MoELayer
from models.rotary import apply_rope
from models.transformer import GPTOSS


def main():
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg = micro_cfg()
    model = GPTOSS(cfg).to(dev)
    print(f"Total params: {model.num_parameters()/1e6:.2f}M")

    B, T = 2, cfg.max_seq_len
    idx = torch.randint(0, cfg.vocab_size, (B, T), device=dev)

    def model_forward():
        with torch.no_grad():
            model(idx)
    t = time_fn(model_forward)
    print(f"[model.forward]      {t:.2f} ms/step")

    attn_win = GPTOSSAttention(cfg, layer_idx=0).to(dev)
    x = torch.randn(B, T, cfg.d_model, device=dev)
    def attn_win_fwd():
        with torch.no_grad():
            attn_win(x)
    t = time_fn(attn_win_fwd)
    print(f"[attn.windowed]      {t:.2f} ms/step")

    attn_full = GPTOSSAttention(cfg, layer_idx=1).to(dev)
    def attn_full_fwd():
        with torch.no_grad():
            attn_full(x)
    t = time_fn(attn_full_fwd)
    print(f"[attn.global]        {t:.2f} ms/step")

    q = torch.randn(B, cfg.n_heads, T, cfg.head_dim, device=dev)
    k = torch.randn(B, cfg.n_heads, T, cfg.head_dim, device=dev)
    v = torch.randn(B, cfg.n_heads, T, cfg.head_dim, device=dev)
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

    moe = MoELayer(cfg).to(dev)
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
            moe._dispatch_vectorized(flat, indices, weights)
    t = time_fn(moe_dispatch)
    print(f"[moe.dispatch]       {t:.2f} ms/step")

    freqs = torch.randn(T, cfg.head_dim // 2, device=dev)
    cos = torch.randn(T, cfg.head_dim // 2, device=dev)
    sin = torch.randn(T, cfg.head_dim // 2, device=dev)
    def rope_apply():
        apply_rope(q, cos, sin)
    t = time_fn(rope_apply)
    print(f"[apply_rope]         {t:.2f} ms/step")

    k_kv = torch.randn(B, cfg.n_kv_heads, T, cfg.head_dim, device=dev)
    def repeat():
        repeat_kv(k_kv, cfg.n_heads // cfg.n_kv_heads)
    t = time_fn(repeat)
    print(f"[repeat_kv]          {t:.2f} ms/step")


if __name__ == "__main__":
    main()
