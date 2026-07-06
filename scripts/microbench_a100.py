#!/usr/bin/env python3
"""Microbenchmark: peak VRAM for GPT-OSS-Lite at production batch/seq."""
import argparse
import os
import sys
from pathlib import Path

import torch

sys.path.append(str(Path(__file__).parent.parent))
from models.transformer import GPTOSS, ModelConfig
from utils.memory import assert_fits_in_available_gpu, estimate_model_memory_gb


def main():
    parser = argparse.ArgumentParser(description="GPT-OSS-Lite microbench")
    parser.add_argument("--config", default="configs/pretrain_a100_502m.yaml", type=str)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--seq-len", type=int, default=4096)
    parser.add_argument("--threshold-gb", type=float, default=25.0,
                        help="Max acceptable peak VRAM (GB). A100 has 80GB; "
                             "we leave 55GB headroom for batch scaling.")
    args = parser.parse_args()

    import yaml
    with open(args.config) as f:
        full = yaml.safe_load(f)
    cfg = ModelConfig(**full["model"])

    print(f"[microbench] Config: d_model={cfg.d_model}, n_layers={cfg.n_layers}, "
          f"vocab={cfg.vocab_size}, experts={cfg.n_routed_experts}")
    print(f"[microbench] batch_size={args.batch_size}, seq_len={args.seq_len}")

    if torch.cuda.is_available():
        model = GPTOSS(cfg).to("cuda")
        if cfg.dtype == "bf16":
            model = model.to(torch.bfloat16)
        torch.cuda.reset_peak_memory_stats()
        x = torch.randint(0, cfg.vocab_size, (args.batch_size, args.seq_len), device="cuda")
        with torch.no_grad():
            _ = model(x)
        torch.cuda.synchronize()
        peak_gb = torch.cuda.max_memory_allocated() / 1024**3
        print(f"[microbench] Peak VRAM (actual): {peak_gb:.2f} GB")
        passed = peak_gb < args.threshold_gb
    else:
        model = GPTOSS(cfg)
        est_gb = estimate_model_memory_gb(
            model,
            seq_len=args.seq_len,
            batch_size=args.batch_size,
            grad_checkpoint=True,
        )
        print(f"[microbench] Peak VRAM (analytical): {est_gb:.2f} GB (CPU-only)")
        passed = est_gb < args.threshold_gb

    if passed:
        print(f"[microbench] ✅ PASSED: peak < {args.threshold_gb} GB")
        return 0
    else:
        print(f"[microbench] ❌ FAILED: peak ≥ {args.threshold_gb} GB")
        return 1


if __name__ == "__main__":
    sys.exit(main())