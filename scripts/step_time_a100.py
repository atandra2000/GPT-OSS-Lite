#!/usr/bin/env python3
"""Step-time benchmark — measures tokens/sec and reports MFU."""
import argparse
import sys
import time
from pathlib import Path

import torch

sys.path.append(str(Path(__file__).parent.parent))
from models.transformer import GPTOSS, ModelConfig


def main():
    parser = argparse.ArgumentParser(description="GPT-OSS-Lite step-time benchmark")
    parser.add_argument("--config", default="configs/pretrain_a100_502m.yaml", type=str)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--seq-len", type=int, default=4096)
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--compile", action="store_true", help="Enable torch.compile for fair MFU measurement")
    args = parser.parse_args()

    import yaml
    with open(args.config) as f:
        full = yaml.safe_load(f)
    cfg = ModelConfig(**full["model"])

    # Hardware performance knobs (TF32, cuDNN benchmark) for accurate A100 measurement
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = GPTOSS(cfg).to(dev)
    if dev.type == "cuda":
        model = model.to(memory_format=torch.channels_last)
    if cfg.dtype == "bf16" and dev.type == "cuda":
        model = model.to(torch.bfloat16)
    if args.compile and dev.type == "cuda":
        model = torch.compile(model, mode="max-autotune")
        print("[step_time] torch.compile enabled (mode=max-autotune)")
    optim = torch.optim.AdamW(model.parameters(), lr=1e-4)

    print(f"[step_time] Config: {cfg.n_layers}L, batch={args.batch_size}, seq={args.seq_len}")
    print(f"[step_time] Warmup: {args.warmup} steps, Measure: {args.steps} steps")

    def step():
        x = torch.randint(0, cfg.vocab_size, (args.batch_size, args.seq_len), device=dev)
        y = torch.randint(0, cfg.vocab_size, (args.batch_size, args.seq_len), device=dev)
        logits, aux = model(x)
        loss = torch.nn.functional.cross_entropy(logits.view(-1, cfg.vocab_size), y.view(-1))
        loss = loss + 0.01 * aux
        optim.zero_grad()
        loss.backward()
        optim.step()
        return loss.item()

    # Warmup
    for _ in range(args.warmup):
        _ = step()
    if dev.type == "cuda":
        torch.cuda.synchronize()

    t0 = time.time()
    for _ in range(args.steps):
        _ = step()
    if dev.type == "cuda":
        torch.cuda.synchronize()
    elapsed = time.time() - t0
    tokens = args.steps * args.batch_size * args.seq_len
    tps = tokens / elapsed
    print(f"[step_time] {args.steps} steps in {elapsed:.2f}s → {tps:,.0f} tokens/sec")

    if dev.type == "cuda":
        n_active = model.num_active_parameters()
        achieved_tflops = 6 * n_active * tps / 1e12
        mfu = achieved_tflops / 312 * 100
        print(f"[step_time] Approx MFU: {mfu:.1f}% (achieved {achieved_tflops:.1f} TFLOPS BF16)")
        if mfu >= 35:
            print(f"[step_time] ✅ MFU target (≥35%) met.")
            return 0
        else:
            print(f"[step_time] ❌ MFU target (<35%).")
            return 1
    else:
        print(f"[step_time] (CPU-only — MFU undefined; reported tokens/sec as proxy)")
        return 0


if __name__ == "__main__":
    sys.exit(main())