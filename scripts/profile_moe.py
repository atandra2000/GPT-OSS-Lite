"""Focused MoE dispatch benchmark."""
import sys
import time
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

import torch

from models.moe import MoELayer
from models.transformer import ModelConfig


def time_fn(fn, n=20, warmup=3):
    for _ in range(warmup):
        fn()
    t0 = time.perf_counter()
    for _ in range(n):
        fn()
    return (time.perf_counter() - t0) / n * 1000


def main():
    cfg = ModelConfig(
        vocab_size=128,
        d_model=64,
        n_layers=2,
        n_heads=4,
        n_kv_heads=2,
        head_dim=16,
        ffn_dim=128,
        n_routed_experts=8,
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
        yarn_prune_rope_global=False,
    )

    moe = MoELayer(cfg.as_dict())
    B, T = 2, cfg.max_seq_len
    x = torch.randn(B, T, cfg.d_model)

    # MoE forward
    def moe_fwd():
        with torch.no_grad():
            moe(x)
    t = time_fn(moe_fwd)
    print(f"moe.forward         {t:.2f} ms")

    # Just dispatch
    flat = x.view(-1, cfg.d_model)
    indices, weights, _ = moe.router(flat)
    def dispatch():
        with torch.no_grad():
            moe._dispatch_vectorized(flat, indices, weights)
    t = time_fn(dispatch)
    print(f"_dispatch_vectorized {t:.2f} ms")

    def dispatch_old():
        with torch.no_grad():
            moe._dispatch_grouped(flat, indices, weights)
    t = time_fn(dispatch_old)
    print(f"_dispatch_grouped    {t:.2f} ms")

    # Test correctness
    out1 = moe._dispatch_vectorized(flat, indices, weights)
    out2 = moe._dispatch_grouped(flat, indices, weights)
    print(f"max diff: {(out1 - out2).abs().max().item():.2e}")


if __name__ == "__main__":
    main()