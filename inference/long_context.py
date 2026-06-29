"""Long-context evaluation: passkey retrieval at increasing context lengths."""
import random
import re
from typing import Optional


def make_filler_text(target_tokens: int, seed: int = 0) -> str:
    """Generate ~target_tokens of filler text (deterministic)."""
    rng = random.Random(seed)
    vocab = ["the", "quick", "brown", "fox", "jumps", "over", "lazy", "dog", "and",
             "runs", "through", "forest", "while", "watching", "birds", "in", "sky"]
    words = []
    while len(words) < target_tokens:
        words.append(rng.choice(vocab))
    return " ".join(words)


PASSKEY_PROMPT_TEMPLATE = (
    "There is an important info in the context above. "
    "Find it and remember it. The passkey is {passkey}. "
    "Now answer: what is the passkey?"
)


class PasskeyEvaluator:
    """Evaluates passkey retrieval at varying context lengths."""

    def __init__(self, model, tokenizer, device: Optional[str] = None):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device or ("cuda" if __import__("torch").cuda.is_available() else "cpu")

    def build_prompt(self, passkey: str, context_length: int, passkey_position: str, seed: int = 0) -> str:
        """Build a (context, question) prompt with the passkey at the given position."""
        filler = make_filler_text(target_tokens=context_length, seed=context_length)
        words = filler.split()
        if passkey_position == "start":
            insert_idx = 0
        elif passkey_position == "end":
            insert_idx = len(words)
        else:
            insert_idx = len(words) // 2
        words.insert(insert_idx, f"The passkey is {passkey}.")
        context = " ".join(words)
        prompt = context + "\n\n" + PASSKEY_PROMPT_TEMPLATE.format(passkey=passkey)
        return prompt

    def extract_passkey_from_output(self, output: str) -> Optional[str]:
        """Extract a 5-digit passkey from the model output."""
        m = re.search(r"\b(\d{5})\b", output)
        return m.group(1) if m else None

    def evaluate(
        self,
        context_lengths: list[int] = (4096, 8192, 32768, 65536, 131072),
        n_trials: int = 100,
        passkey_position: str = "middle",
        base_seed: int = 42,
    ) -> dict[int, float]:
        """Evaluate passkey retrieval; returns ``{ctx_len: accuracy}``."""
        import torch
        from inference.generate import generate

        self.model.eval()
        results = {}
        for ctx_len in context_lengths:
            rng = random.Random(base_seed + ctx_len)
            n_distinct = min(n_trials, 100_000)
            passkeys = [f"{p:05d}" for p in rng.sample(range(100_000), n_distinct)]
            n_correct = 0
            for trial, passkey in enumerate(passkeys):
                prompt = self.build_prompt(passkey, ctx_len, passkey_position, seed=trial)
                input_ids = torch.tensor(
                    self.tokenizer.encode(prompt),
                    dtype=torch.long,
                    device=self.device,
                ).unsqueeze(0)
                output_ids = generate(
                    self.model,
                    input_ids,
                    max_new_tokens=16,
                    temperature=0.0,
                    top_p=1.0,
                    use_cache=True,
                )
                output_text = self.tokenizer.decode(output_ids[0, input_ids.size(1):].tolist())
                extracted = self.extract_passkey_from_output(output_text)
                if extracted == passkey:
                    n_correct += 1
            accuracy = n_correct / len(passkeys)
            results[ctx_len] = accuracy
        return results