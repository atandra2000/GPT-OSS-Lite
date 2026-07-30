#!/usr/bin/env python3
"""Analytical KV-cache size benchmark for GPT-OSS-Lite headline metric."""

from __future__ import annotations

# Production architecture constants (configs/pretrain_a100_502m.yaml)
N_LAYERS = 12
N_WINDOWED = 6
N_GLOBAL = 6
N_KV_HEADS = 4
HEAD_DIM = 96
WINDOW = 128
DTYPE_BYTES = 2  # BF16
BATCH = 1
THRESHOLD = 1.8


def kv_bytes_per_layer_per_token() -> int:
    """K + V for one token, one layer."""
    return 2 * N_KV_HEADS * HEAD_DIM * DTYPE_BYTES


def cache_bytes(seq_len: int, *, all_full: bool) -> int:
    per_tok = kv_bytes_per_layer_per_token()
    if all_full:
        return N_LAYERS * seq_len * BATCH * per_tok
    windowed = N_WINDOWED * min(WINDOW, seq_len)
    global_ = N_GLOBAL * seq_len
    return (windowed + global_) * BATCH * per_tok


def fmt_gb(nbytes: int) -> str:
    return f"{nbytes / 1024**3:.2f} GB"


def main() -> int:
    contexts = [4096, 8192, 16384, 32768, 65536, 131072]
    print("GPT-OSS-Lite KV-cache benchmark (analytical, BF16, batch=1)")
    print(f"Architecture: {N_LAYERS} layers ({N_WINDOWED} SWA w={WINDOW} + {N_GLOBAL} full)")
    print(f"GQA: {N_KV_HEADS} KV heads, head_dim={HEAD_DIM}\n")
    print(f"{'Context':>10}  {'Pure GQA':>12}  {'SWA/Full':>12}  {'Reduction':>10}")
    print("-" * 50)

    reduction_at_128k = 0.0
    for ctx in contexts:
        pure = cache_bytes(ctx, all_full=True)
        mixed = cache_bytes(ctx, all_full=False)
        ratio = pure / mixed if mixed else 0.0
        if ctx == 131072:
            reduction_at_128k = ratio
        print(f"{ctx:>10,}  {fmt_gb(pure):>12}  {fmt_gb(mixed):>12}  {ratio:>9.2f}×")

    print()
    if reduction_at_128k >= THRESHOLD:
        print(f"✅ HEADLINE METRIC PASSED: {reduction_at_128k:.2f}× KV-cache reduction at 128K (≥ {THRESHOLD}×)")
        return 0
    print(f"❌ HEADLINE METRIC FAILED: {reduction_at_128k:.2f}× at 128K (need ≥ {THRESHOLD}×)")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
