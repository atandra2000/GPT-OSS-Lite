"""GPT-OSS-Lite data-preparation entry point.

The shared pipeline owns argument parsing and tokenization; this shim prints the
GPT-OSS corpus settings before delegating to it.
"""
import sys
from pathlib import Path


_PROJECT_ROOT = Path(__file__).resolve().parents[1]
_LLM_ROOT = Path(__file__).resolve().parents[2]  # .../LLM/ — shared_data lives here
for _p in (_PROJECT_ROOT, _LLM_ROOT):
    _p = str(_p)
    if _p not in sys.path:
        sys.path.insert(0, _p)


def main() -> int:
    # Report the shared tokenizer before handing control to the universal pipeline.
    from shared_data.config import UNIVERSAL_TOTAL_TOKENS, load_universal_data_config
    cfg = load_universal_data_config()
    tok = cfg["pipeline"]["tokenizer"]
    print(f"[data/gptoss] universal corpus: {UNIVERSAL_TOTAL_TOKENS:,} tokens")
    print(f"[data/gptoss] tokenizer: {tok['name']} "
          f"(vocab={tok['vocab_size']:,}, EOS={tok['eos_token_id']})")
    print(f"[data/gptoss] shard size: "
          f"{cfg['pipeline']['sharding']['shard_size_tokens']:,} tokens")

    from shared_data.prepare_data import main as shared_main
    return shared_main()


if __name__ == "__main__":
    sys.exit(main())