"""End-to-end training step benchmark for GPT-OSS-Lite."""
import sys
import time
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

import torch
import torch.nn.functional as F

from models.transformer import GPTOSS, ModelConfig
from training.pretrain import chunked_cross_entropy


def time_fn(fn, n=20, warmup=3):
    for _ in range(warmup):
        fn()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(n):
        fn()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    return (time.perf_counter() - t0) / n * 1000


def main():
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
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = GPTOSS(cfg).to(dev)
    optim = torch.optim.AdamW(model.parameters(), lr=1e-4)
    B, T = 2, cfg.max_seq_len
    idx = torch.randint(0, cfg.vocab_size, (B, T), device=dev)
    target = torch.randint(0, cfg.vocab_size, (B, T), device=dev)

    def step():
        optim.zero_grad(set_to_none=True)
        logits, aux = model(idx)
        ce = chunked_cross_entropy(logits, target, chunk_size=4096)
        loss = ce + 0.01 * aux
        loss.backward()
        optim.step()

    t = time_fn(step, n=10, warmup=3)
    print(f"training step (4L, d=64): {t:.2f} ms/step")

    cfg_big = ModelConfig(
        vocab_size=512,
        d_model=128,
        n_layers=8,
        n_heads=8,
        n_kv_heads=4,
        head_dim=16,
        ffn_dim=256,
        n_routed_experts=8,
        n_activated_experts=2,
        n_shared_experts=1,
        window_size=16,
        max_seq_len=256,
        rope_theta=10000,
        yarn_scale_factor=2,
        yarn_original_max_seq_len=256,
        yarn_target_seq_len=512,
        yarn_beta_fast=2,
        yarn_beta_slow=1,
        yarn_prune_rope_global=True,
    )
    model_big = GPTOSS(cfg_big).to(dev)
    optim_big = torch.optim.AdamW(model_big.parameters(), lr=1e-4)
    B, T = 2, cfg_big.max_seq_len
    idx = torch.randint(0, cfg_big.vocab_size, (B, T), device=dev)
    target = torch.randint(0, cfg_big.vocab_size, (B, T), device=dev)

    def step_big():
        optim_big.zero_grad(set_to_none=True)
        logits, aux = model_big(idx)
        ce = chunked_cross_entropy(logits, target, chunk_size=4096)
        loss = ce + 0.01 * aux
        loss.backward()
        optim_big.step()

    t = time_fn(step_big, n=10, warmup=3)
    print(f"training step (8L, d=128, T=256): {t:.2f} ms/step")
    print(f"  model params: {model_big.num_parameters()/1e6:.2f}M")


if __name__ == "__main__":
    main()