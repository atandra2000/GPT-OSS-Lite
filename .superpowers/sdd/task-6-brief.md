### Task 6: Canonicalize data pipeline; stub `data/DATA_PIPELINE.md`

**Files:**
- Modify: `documentation/data_pipeline.md`
- Rewrite: `data/DATA_PIPELINE.md` (stub only)

**Interfaces:**
- Consumes: existing documentation chapter + unique stub-worthy bits from `data/DATA_PIPELINE.md`
- Produces: one full guide; discoverable stub under `data/`

- [ ] **Step 1: Fold missing unique content into `documentation/data_pipeline.md`**

Ensure these topics exist (add if missing):

- Vendored `data/shared_data/` layout and why vendored  
- Updating the vendored copy (pointer to workspace sync procedure from the old data doc)  
- Explicit note that GPT-OSS + LLaMA-3 shards are bit-identical  

Keep the existing stage walkthrough (download → clean → tokenize → pack) as the body.

- [ ] **Step 2: Replace `data/DATA_PIPELINE.md` with this stub**

```markdown
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
```

- [ ] **Step 3: Retarget skill/readme refs that treated data/ as canonical**

```bash
rg -n 'data/DATA_PIPELINE\.md' SKILLS.md README.md documentation/
```

In `SKILLS.md`, change “See `data/DATA_PIPELINE.md` for the full guide” → `documentation/data_pipeline.md` (stub may remain as secondary pointer).

- [ ] **Step 4: Verify single source of truth**

```bash
wc -l data/DATA_PIPELINE.md documentation/data_pipeline.md
# stub should be << full guide (expect stub ≤ 40 lines)
```

---

