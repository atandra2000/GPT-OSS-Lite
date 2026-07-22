"""Inference + long-context benchmark for GPT-OSS-Lite."""
import time
import torch

from _bootstrap import micro_cfg, time_fn
from inference.generate import generate
from models.transformer import GPTOSS


def main():
    cfg = micro_cfg()
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = GPTOSS(cfg).to(dev)
    model.eval()
    B = 1
    n_new = 64

    for prompt_len in [8, 32, 128]:
        input_ids = torch.randint(0, cfg.vocab_size, (B, prompt_len), device=dev)
        with torch.no_grad():
            generate(model, input_ids, max_new_tokens=4, temperature=0.0)
        if dev.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        n_runs = 3
        for _ in range(n_runs):
            with torch.no_grad():
                generate(model, input_ids, max_new_tokens=n_new, temperature=0.0)
        if dev.type == "cuda":
            torch.cuda.synchronize()
        elapsed = (time.perf_counter() - t0) / n_runs
        print(f"prompt={prompt_len}, new={n_new}: {elapsed*1000:.1f} ms ({n_new/elapsed:.0f} tok/s)")


if __name__ == "__main__":
    main()
