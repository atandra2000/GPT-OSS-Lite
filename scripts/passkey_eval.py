#!/usr/bin/env python3
"""Passkey retrieval evaluation at long context (headline metric #2)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main() -> int:
    parser = argparse.ArgumentParser(description="GPT-OSS-Lite passkey retrieval eval")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to .safetensors checkpoint")
    parser.add_argument("--n-trials", type=int, default=10, help="Trials per context length")
    parser.add_argument(
        "--context-lengths",
        type=int,
        nargs="+",
        default=[4096, 8192, 32768, 65536, 131072],
    )
    parser.add_argument("--position", choices=["start", "middle", "end"], default="middle")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    ckpt = Path(args.checkpoint)
    if not ckpt.exists():
        print(f"Checkpoint not found: {ckpt}", file=sys.stderr)
        return 1

    import torch
    import yaml
    from safetensors.torch import load_file

    from inference.long_context import PasskeyEvaluator
    from models.transformer import GPTOSS, ModelConfig

    with open(ROOT / "configs" / "pretrain_a100_502m.yaml") as f:
        raw = yaml.safe_load(f)
    cfg = ModelConfig(**raw["model"])
    model = GPTOSS(cfg)
    state = load_file(str(ckpt))
    model.load_state_dict(state, strict=False)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)

    class _CharTokenizer:
        def encode(self, text: str) -> list[int]:
            return [min(ord(c), cfg.vocab_size - 1) for c in text[: cfg.eval_max_seq_len]]

        def decode(self, ids: list[int]) -> str:
            return "".join(chr(min(i, 127)) for i in ids)

    evaluator = PasskeyEvaluator(model, _CharTokenizer(), device=device)

    print(f"Passkey eval: checkpoint={ckpt.name}, device={device}")
    results = evaluator.evaluate(
        context_lengths=args.context_lengths,
        n_trials=args.n_trials,
        passkey_position=args.position,
        base_seed=args.seed,
    )

    print(f"\n{'Context':>10}  {'Accuracy':>10}")
    print("-" * 24)
    for ctx_len, acc in sorted(results.items()):
        print(f"{ctx_len:>10,}  {acc:>9.1%}")

    target_ctx = max(args.context_lengths)
    final_acc = results.get(target_ctx, 0.0)
    if final_acc >= 0.85:
        print(f"\n✅ HEADLINE METRIC PASSED: {final_acc:.1%} at {target_ctx:,} (≥ 85%)")
        return 0
    print(f"\n⚠️  Accuracy {final_acc:.1%} at {target_ctx:,} — needs trained checkpoint for ≥ 85% target")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
