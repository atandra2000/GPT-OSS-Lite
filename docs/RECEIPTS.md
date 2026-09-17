# GPT-OSS-Lite Archify evidence

Refreshed 2026-09-17. Scope: four specs, their delivered artifacts and the [guide](gpt_oss_visual_guide.html). No model or shared-data code changed.

## Source baseline

Repository revision: `3c9675deaf163b7bfa3d8d8f7d0fae6883fdedc0`. Configuration: `configs/pretrain_a100_502m.yaml` plus `ModelConfig` defaults (notably `moe_dispatch="stacked"`), no CLI overrides. Shared pipeline files are outside this Git repository, so their inspected bytes are pinned below rather than assigned a fictitious commit.

| Input | SHA-256 |
|---|---|
| `configs/pretrain_a100_502m.yaml` | `8220587d00c2e149821958a2900b117bfad51207aaad06a90819c35a2bf03ebe` |
| `models/transformer.py` | `9cb3243805cb86ecfe9e063a7af7b740ed8b6a224bc705115744255bd5a4fc2b` |
| `../shared_data/config/data_config.yaml` | `c6c57d274a737ca34ecfc33f3c117e6a8a4f187e56a8327362a405c617cc5ac1` |
| `../shared_data/config/mixture.yaml` | `fefd48c045af197cb18da1e0d710dee4aaf7aebf63af544633f89848364820e0` |
| `../shared_data/shard_writer.py` | `35a81ffc5d268af96009b68c6ea17cf82c6ea5a2b5a83c225399ee9e78c5b3a8` |
| `../shared_data/scripts/pack_shards.py` | `0f90101375437b5155260897ce7478f02f71bc0476f635465bb98e7fe21d8f31` |

## What the sources establish

- `models/transformer.py:GPTOSSBlock.forward` and `models/transformer.py:GPTOSS.__init__`: one pre-norm Attention + residual, then pre-norm MoE + residual, repeated 12 times.
- `models/attention.py:GPTOSSAttention._n_pruned_dims` returns 96//4 = 24. `models/yarn.py:YaRNRoPE.forward` sets 24 of its 48 frequency-pair columns to identity on global layers. This is 50% of pairs, not 25%.
- `inference/generate.py:_attn_forward_layer` appends rotated K and unchanged V through `inference/generate.py:MixedKVCache.append`. The cache ratio counts logical entries, not physical allocation or transient reordering copies.
- Shared `TokenStream.write_doc` validates document tokens then appends EOS 128009. The configured embedding has 128000 rows (IDs 0–127999). `training/pretrain.py:PretrainDataset.__getitem__` supplies windows without remapping that ID. This is an unresolved compatibility problem, not repaired here.
- Shared `interleave_sources` is document round-robin and does not decrement its remaining quota. Declared mixture weights are not evidence of enforced output proportions or corpus completion.
- `training/pretrain.py:main` computes CE in 8192-token chunks, then `(CE + .01*aux)/4`, then tests finiteness before backward. Four finite micros commit one optimizer/scheduler update. Nonfinite loss clears partial gradients and micro count. Five failures restore the latest checkpoint or abort if none exists.
- `utils/checkpoint.py:CheckpointManager.save` writes files atomically individually. Periodic saves include model/optimizer/scheduler/meta, but no RNG or sampler position. Final save adds RNG separately using torch.save. `utils/checkpoint.py:CheckpointManager._checkpoint_complete` checks model/optimizer/meta only, not scheduler or RNG. Rollback does not restore RNG or data position. Loading may fail and is not certified by this diagram.
- Compilation is requested only on CUDA with the YAML flag. Immediate setup exceptions print and continue eager, not a guaranteed compiled execution. SDPA backend eligibility is conditional. Optional Triton dispatch is separate. 16–20 h on A100 80GB remains an unverified target.

## Independent gates

All four delivered HTML artifacts passed **9/9 showcase checks, zero composition errors and warnings**, then fresh `visual-check` commands returned exit 0 and `status: pass` on those exact HTML hashes. Final validate was also rerun synchronously.

Automated browser coverage: light-theme containment at 1440×900, 1600×1000, 1920×1080 and 2048×1320. Screenshots at both endpoints in light and dark. This does not assert intermediate dark-theme measurement.

**Perceptual review: skipped (image reader unavailable).** The image-read tool located the new model screenshot, but the provider explicitly omitted image input. No aesthetic, contrast or visual-balance sign-off is claimed. Automated receipts intentionally retain `visualReview: pending`. Perceptual correction rounds: 0. Deterministic authoring repairs are separate from perceptual rounds.

A separate attempt to open the guide via the built-in browser failed with a broken-pipe tool error. Guide mobile containment, diagram focus/search/reset/navigation and representative exports are **not browser-verified** in this session. This does not replace or downgrade the successful packaged diagram checks.

## Readability limits

The table uses the observed `readability.viewports[].minimumProjectedNodeTextPx`, not the receipt's top-level 6px acceptance threshold. These secondary-label minima are unchanged across the four measured light-theme viewports. **The proposed 12px essential / 11px secondary standard is not met.** Passing the packaged 6px floor is not readability or accessibility certification. Node labels were not shrunk to force containment.

| Diagram | Previous audit minimum | Fresh measured minimum (CSS px) |
|---|---:|---:|
| model-architecture | 6.40 | 9.00 |
| data-pipeline | 7.00 | 7.00 |
| training-workflow | 7.07 | 8.00 |
| optimization-stack | 6.00 | 9.00 |

## Exact artifact bindings

### model-architecture

- Type: `architecture`. [HTML](gpt-oss-lite-model-architecture.html) · [specification](gpt-oss-lite-model-architecture.archify.json) · [delivery receipt](gpt-oss-lite-model-architecture.delivery.json).
- Specification SHA-256: `dd5f904082fb0cd5264b2cf620f558f0b90c12567ed1a53c973b7d2d02a6981b` (6967 bytes).
- HTML SHA-256: `632f1cd9b84271d8d4c69da9fa623d31fd557daccb53f1ffed2b08fd1713fc10` (715968 bytes).
- [Automated browser receipt](gpt-oss-lite-model-architecture.visual-check.json) · [endpoint contact sheet](gpt-oss-lite-model-architecture.visual-check.html).
- `browser_evidence: passed`. `visual_review: skipped (image reader unavailable)`. `correction_rounds: 0`.

### data-pipeline

- Type: `dataflow`. [HTML](gpt-oss-lite-data-pipeline.html) · [specification](gpt-oss-lite-data-pipeline.archify.json) · [delivery receipt](gpt-oss-lite-data-pipeline.delivery.json).
- Specification SHA-256: `1ec1f8fe422c479c2d8472ef29e0c8053f096db38d44203e5848aefb5355c668` (5911 bytes).
- HTML SHA-256: `79ee1da4b1e64357d264cf115faf803ef555312ef60ca40ac6e2499177fdfb35` (714822 bytes).
- [Automated browser receipt](gpt-oss-lite-data-pipeline.visual-check.json) · [endpoint contact sheet](gpt-oss-lite-data-pipeline.visual-check.html).
- `browser_evidence: passed`. `visual_review: skipped (image reader unavailable)`. `correction_rounds: 0`.

### training-workflow

- Type: `workflow`. [HTML](gpt-oss-lite-training-workflow.html) · [specification](gpt-oss-lite-training-workflow.archify.json) · [delivery receipt](gpt-oss-lite-training-workflow.delivery.json).
- Specification SHA-256: `99e758febd117447165c3eadcd690414191b6258a506bd4b4c7c83cbce0b3031` (5868 bytes).
- HTML SHA-256: `396a615b33a8c614edc7590404b963b8fb68822cf875b713a7897e96eedd1a5f` (719036 bytes).
- [Automated browser receipt](gpt-oss-lite-training-workflow.visual-check.json) · [endpoint contact sheet](gpt-oss-lite-training-workflow.visual-check.html).
- `browser_evidence: passed`. `visual_review: skipped (image reader unavailable)`. `correction_rounds: 0`.

### optimization-stack

- Type: `architecture`. [HTML](gpt-oss-lite-optimization-stack.html) · [specification](gpt-oss-lite-optimization-stack.archify.json) · [delivery receipt](gpt-oss-lite-optimization-stack.delivery.json).
- Specification SHA-256: `1b1eafda34df4dba431f79025bb735d5d3dac7d02274cd70e6643b8dee7ef6b3` (4612 bytes).
- HTML SHA-256: `28fe1e481ffd50f6e2ae17243b9d18621961c5f08f582d2268caf47bb45c6d06` (710291 bytes).
- [Automated browser receipt](gpt-oss-lite-optimization-stack.visual-check.json) · [endpoint contact sheet](gpt-oss-lite-optimization-stack.visual-check.html).
- `browser_evidence: passed`. `visual_review: skipped (image reader unavailable)`. `correction_rounds: 0`.

## Project checks

- `python3 -m pytest tests/ -q`: 203 passed, 2 skipped (CPU, 51.01 s).
- `python3 scripts/kv_cache_benchmark.py`: analytical 2.00× at 128K, exceeds 1.8× threshold. Not an allocated-VRAM benchmark.
- `python3 tests/test_doc_refs.py --strict-coverage`: no stale anchors, no uncovered public symbols.
- `python3 scripts/check_docs.py`: OK (14 Markdown files). This gate does not certify HTML prose or JavaScript.
- Guide calculator checked with a synthetic DOM harness against 12 layers, head_dim 96, 8 MHA / 4 GQA heads, window 128 and BF16 storage. This is arithmetic/script evidence, not a real browser or a model runtime test.
- Internal guide links and HTML Python symbol references are checked separately from the Markdown-only gates.

No CUDA training, corpus build, model/data compatibility repair, exact-resume verification or trained 128K retrieval evaluation was performed.
