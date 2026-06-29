"""GPT-OSS-Lite data preparation: thin shim over the universal pipeline."""
import argparse
import sys
from pathlib import Path


_PROJECT_ROOT = Path(__file__).resolve().parents[1]
_LLM_ROOT = _PROJECT_ROOT.parent.parent  # .../CoreProjects/
for _p in (_PROJECT_ROOT, _LLM_ROOT):
    _p = str(_p)
    if _p not in sys.path:
        sys.path.insert(0, _p)


def _apply_gptoss_defaults() -> None:
    """Configure the universal pipeline with GPT-OSS-Lite's tokenizer."""
    from shared_data.config import (
        UNIVERSAL_TOTAL_TOKENS,
        load_universal_data_config,
    )
    cfg = load_universal_data_config()
    tok = cfg["pipeline"]["tokenizer"]
    print(f"[data/gptoss] universal corpus: {UNIVERSAL_TOTAL_TOKENS:,} tokens")
    print(f"[data/gptoss] tokenizer: {tok['name']} "
          f"(vocab={tok['vocab_size']:,}, EOS={tok['eos_token_id']})")
    print(f"[data/gptoss] shard size: "
          f"{cfg['pipeline']['sharding']['shard_size_tokens']:,} tokens")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="GPT-OSS-Lite data prep (delegates to universal pipeline)",
    )
    parser.add_argument("--stage", choices=["pretrain"], default="pretrain")
    parser.add_argument("--mixture", default=None,
                        help="Override mixture.yaml path (defaults to universal)")
    parser.add_argument("--data-config", default=None,
                        help="Override data_config.yaml path (defaults to universal)")
    parser.add_argument("--data-root", default=None,
                        help="Override DATA_ROOT (default: $LLM_DATA_ROOT or $PWD/data)")
    parser.add_argument("--source", default=None,
                        help="Restrict to a single source id")
    parser.add_argument("--skip-download", action="store_true")
    parser.add_argument("--skip-clean", action="store_true")
    parser.add_argument("--skip-tokenize", action="store_true")
    parser.add_argument("--skip-pack", action="store_true")
    args = parser.parse_args()

    _apply_gptoss_defaults()

    from shared_data.config import UNIVERSAL_MIXTURE_PATH, UNIVERSAL_DATA_CONFIG_PATH
    from shared_data.prepare_data import run_pipeline

    return run_pipeline(
        mixture_path=Path(args.mixture) if args.mixture else UNIVERSAL_MIXTURE_PATH,
        data_config_path=Path(args.data_config) if args.data_config else UNIVERSAL_DATA_CONFIG_PATH,
        source=args.source,
        skip_download=args.skip_download,
        skip_clean=args.skip_clean,
        skip_tokenize=args.skip_tokenize,
        skip_pack=args.skip_pack,
        data_root=Path(args.data_root) if args.data_root else None,
    )


if __name__ == "__main__":
    sys.exit(main())