"""Long-context inference benchmark for GPT-OSS-Lite."""
import sys
import time
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

import torch

from inference.generate import generate
from models.transformer import GPTOSS, ModelConfig


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
    model = GPTOSS(cfg)
    if torch.cuda.is_available():
        model = model.to(memory_format=torch.channels_last)
    model.eval()

    for prompt_len in [8, 32, 128]:
        B = 1
        input_ids = torch.randint(0, cfg.vocab_size, (B, prompt_len))
        n_new = 64
        with torch.no_grad():
            generate(model, input_ids, max_new_tokens=4, temperature=0.0)
        t0 = time.perf_counter()
        n_runs = 3
        for _ in range(n_runs):
            with torch.no_grad():
                generate(model, input_ids, max_new_tokens=n_new, temperature=0.0)
        elapsed = (time.perf_counter() - t0) / n_runs
        print(f"prompt={prompt_len}, new={n_new}: {elapsed*1000:.1f} ms ({n_new/elapsed:.0f} tok/s)")


if __name__ == "__main__":
    main()