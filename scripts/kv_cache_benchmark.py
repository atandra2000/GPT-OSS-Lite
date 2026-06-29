#!/usr/bin/env python3
"""KV-cache reduction benchmark — verifies the 2× reduction headline metric."""
import argparse
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent.parent))

from models.transformer import ModelConfig


def measure_kv_cache(cfg: ModelConfig, context_len: int) -> tuple[float, float]:
    """Measure KV cache size (in GB, BF16) for pure GQA vs alternating SWA/full."""
    n_layers = cfg.n_layers
    n_kv_heads = cfg.n_kv_heads
    head_dim = cfg.head_dim
    window = cfg.window_size
    n_windowed = sum(1 for i in range(n_layers) if i % 2 == 0)
    n_global = n_layers - n_windowed
    bytes_per_token = 2 * n_kv_heads * head_dim * 2

    pure_gqa = n_layers * context_len * bytes_per_token
    alternating = (
        n_windowed * window * bytes_per_token
        + n_global * context_len * bytes_per_token
    )

    return pure_gqa / 1024**3, alternating / 1024**3


def main():
    parser = argparse.ArgumentParser(description="GPT-OSS-Lite KV-cache benchmark")
    parser.add_argument("--config", default="configs/pretrain_a100_502m.yaml", type=str)
    parser.add_argument("--threshold", default=1.8, type=float, help="Min reduction ratio to claim headline")
    args = parser.parse_args()

    import yaml
    with open(args.config) as f:
        full = yaml.safe_load(f)
    cfg = ModelConfig(**full["model"])

    print(f"GPT-OSS-Lite KV-cache benchmark")
    print(f"  Model: {cfg.n_layers} layers, {cfg.n_kv_heads} KV heads, head_dim={cfg.head_dim}")
    print(f"  Alternation: 6 sliding-window(window={cfg.window_size}) + 6 full attention")
    print(f"  Precision: BF16 (2 bytes/element)")
    print()
    print(f"  {'Context':>10} | {'Pure GQA':>10} | {'SWA+Full':>10} | {'Reduction':>10}")
    print(f"  {'-'*10}-+-{'-'*10}-+-{'-'*10}-+-{'-'*10}")

    context_lengths = [4096, 8192, 32768, 65536, 131072]
    min_reduction = float("inf")
    for ctx in context_lengths:
        pure, alt = measure_kv_cache(cfg, ctx)
        reduction = pure / alt
        min_reduction = min(min_reduction, reduction)
        print(f"  {ctx//1024:>7}K  | {pure:>8.2f}GB | {alt:>8.2f}GB | {reduction:>9.1f}×")

    print()
    print(f"  Min reduction: {min_reduction:.2f}× (headline requires ≥ {args.threshold}×)")
    if min_reduction >= args.threshold:
        print(f"  ✅ HEADLINE METRIC PASSED: alternating SWA/full delivers ≥ {args.threshold}× KV-cache reduction.")
        return 0
    else:
        print(f"  ❌ HEADLINE METRIC FAILED: reduction {min_reduction:.2f}× < {args.threshold}×")
        return 1


if __name__ == "__main__":
    sys.exit(main())