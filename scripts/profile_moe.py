"""Focused MoE dispatch benchmark."""
import torch

from _bootstrap import micro_cfg, time_fn
from models.moe import MoELayer
from dataclasses import replace


def main():
    cfg = micro_cfg()
    # Drop yarn_prune_rope_global for MoE-only timing (avoids YaRN init overhead)
    cfg = replace(cfg, yarn_prune_rope_global=False)

    moe = MoELayer(cfg)
    B, T = 2, cfg.max_seq_len
    x = torch.randn(B, T, cfg.d_model)

    def moe_fwd():
        with torch.no_grad():
            moe(x)
    t = time_fn(moe_fwd)
    print(f"moe.forward         {t:.2f} ms")

    flat = x.view(-1, cfg.d_model)
    indices, weights, _ = moe.router(flat)
    def dispatch():
        with torch.no_grad():
            moe._dispatch_vectorized(flat, indices, weights)
    t = time_fn(dispatch)
    print(f"_dispatch_vectorized {t:.2f} ms")


if __name__ == "__main__":
    main()
