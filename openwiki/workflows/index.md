# Files

- [Incremental Generation and Long-Context Evaluation](inference-and-evaluation.md) - Traces prompt prefill, absolute-positioned token-by-token decoding, heterogeneous KV-cache lifecycle, and sampling behavior. Documents cache-disabled correctness replay and the configurable passkey evaluation workflow, including checkpoint, tokenizer, determinism, and 128K target requirements.
- [Pretraining Runtime Workflow](pretraining.md) - Traces GPT-OSS-Lite pretraining from CLI and YAML loading through reproducible setup, model and device construction, memory checks, packed-token input, mixed-precision optimization, checkpoint recovery, and final state persistence. Highlights optimizer-step boundaries, the exact combined loss contract, and non-finite-loss failure semantics.
