# Data Pipeline — GPT-OSS-Lite

> **Canonical guide:** [`documentation/data_pipeline.md`](../documentation/data_pipeline.md)
>
> This file is a short pointer only. Do not maintain a second full copy here.

## Quick start

```bash
python3 data/prepare_data.py --stage pretrain
```

Tokenizer: LLaMA-3 BPE, vocab **128,000**, EOS **128,009**.  
Full stages, mix weights, shard format, and `PretrainDataset` details live in the documentation chapter above.
