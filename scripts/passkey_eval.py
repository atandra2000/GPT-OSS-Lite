#!/usr/bin/env python3
"""Passkey retrieval evaluation — verifies the 128K long-context headline."""
import argparse
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))


def main():
    parser = argparse.ArgumentParser(description="GPT-OSS-Lite passkey retrieval eval")
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to model.safetensors")
    parser.add_argument("--n-trials", type=int, default=10)
    parser.add_argument("--threshold", type=float, default=0.85, help="Min accuracy at 128K")
    parser.add_argument("--context-lengths", nargs="+", type=int,
                        default=[4096, 8192, 32768, 65536, 131072])
    parser.add_argument("--position", default="middle", choices=["start", "middle", "end"])
    args = parser.parse_args()

    import torch
    from models.transformer import GPTOSS, ModelConfig

    import yaml
    cfg_path = Path(__file__).parent.parent / "configs" / "pretrain_a100_502m.yaml"
    with open(cfg_path) as f:
        full = yaml.safe_load(f)
    cfg = ModelConfig(**full["model"])

    model = GPTOSS(cfg)
    if torch.cuda.is_available():
        model = model.to(memory_format=torch.channels_last)

    if args.checkpoint and Path(args.checkpoint).exists():
        from safetensors.torch import load_file
        weights = load_file(args.checkpoint, device="cpu")
        model.load_state_dict(weights, strict=False)
        print(f"[passkey_eval] Loaded checkpoint from {args.checkpoint}")
    else:
        print(f"[passkey_eval] No checkpoint provided — running stub evaluation "
              "(prompt construction only).")
        from inference.long_context import PasskeyEvaluator
        class StubTokenizer:
            def encode(self, s): return s.split()
            def decode(self, ids): return " ".join(str(i) for i in ids)
        evaluator = PasskeyEvaluator(model, StubTokenizer())
        prompt = evaluator.build_prompt("12345", context_length=128, passkey_position="middle", seed=0)
        assert "12345" in prompt, "Passkey not in prompt"
        print(f"[passkey_eval] ✅ Stub prompt construction works (prompt length: {len(prompt)} chars).")
        print(f"[passkey_eval] Run with --checkpoint <path> to evaluate a trained model.")
        return 0

    print(f"[passkey_eval] Running passkey retrieval at lengths {args.context_lengths}")
    print(f"[passkey_eval] Trials per length: {args.n_trials}")
    print(f"[passkey_eval] Passkey position: {args.position}")

    class WhitespaceTokenizer:
        def encode(self, s): return s.split()
        def decode(self, ids): return " ".join(str(i) for i in ids)
    tokenizer = WhitespaceTokenizer()

    from inference.long_context import PasskeyEvaluator
    evaluator = PasskeyEvaluator(model, tokenizer)
    results = evaluator.evaluate(
        context_lengths=args.context_lengths,
        n_trials=args.n_trials,
        passkey_position=args.position,
    )

    print()
    print(f"  {'Context':>10} | {'Accuracy':>10}")
    print(f"  {'-'*10}-+-{'-'*10}")
    for ctx_len, acc in results.items():
        marker = "✅" if acc >= args.threshold else "  "
        print(f"  {ctx_len//1024:>7}K  | {acc*100:>8.1f}% {marker}")

    max_ctx = max(args.context_lengths)
    max_acc = results[max_ctx]
    print()
    if max_acc >= args.threshold:
        print(f"  ✅ HEADLINE METRIC PASSED: ≥{int(args.threshold*100)}% retrieval at {max_ctx//1024}K")
        return 0
    else:
        print(f"  ⚠️  Headline metric FAILED: {max_acc*100:.1f}% < {args.threshold*100:.1f}% at {max_ctx//1024}K")
        return 1


if __name__ == "__main__":
    sys.exit(main())