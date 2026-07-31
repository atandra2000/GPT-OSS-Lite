# Data Pipeline — GPT-OSS-Lite

## From Raw Text to `PretrainDataset`

> **Shim:** [`data/prepare_data.py`](../data/prepare_data.py) delegates to
> `LLM/shared_data/` universal pipeline.

> **Training consumer:** [`training/pretrain.py`](../training/pretrain.py)
> `PretrainDataset` class.

> **Related:** [training.md](training.md) (loader knobs, `train_data_path`).

---

## Table of Contents

1. [Abstract](#abstract)
2. [Design Goals](#design-goals)
3. [Quick Start](#quick-start)
4. [The Shim — `data/prepare_data.py`](#the-shim--dataprepare_datapy)
5. [Pipeline Stages Overview](#pipeline-stages-overview)
6. [Stage 1 — Download (`download_raw`)](#stage-1--download-download_raw)
7. [Stage 2 — Clean (`clean`)](#stage-2--clean-clean)
8. [Stage 3 — Tokenize (`tokenize`)](#stage-3--tokenize-tokenize)
9. [Stage 4 — Pack Shards (`pack_shards`)](#stage-4--pack-shards-pack_shards)
10. [Corpus Mix — `gptoss-default`](#corpus-mix--gptoss-default)
11. [Tokenizer — LLaMA-3 BPE](#tokenizer--llama-3-bpe)
12. [Shard Format](#shard-format)
13. [Manifest Schema](#manifest-schema)
14. [Output Layout — `data/pretrain_chinchilla`](#output-layout--datapretrain_chinchilla)
15. [Training Loader — `PretrainDataset`](#training-loader--pretraindataset)
16. [DataLoader Configuration](#dataloader-configuration)
17. [Cross-Project Sharing](#cross-project-sharing)
18. [Idempotency and Resume](#idempotency-and-resume)
19. [Disk and Time Budgets](#disk-and-time-budgets)
20. [Validation Checklist](#validation-checklist)
21. [Appendix A — Token arithmetic](#appendix-a--token-arithmetic)
22. [Appendix B — Window crossing shards](#appendix-b--window-crossing-shards)
23. [Appendix C — Directory tree](#appendix-c--directory-tree)
24. [Appendix D — Source dataset reference](#appendix-d--source-dataset-reference)
25. [Load-Bearing Invariants](#load-bearing-invariants)
26. [References](#references)

---

## Abstract

GPT-OSS-Lite trains on an **8.0 billion-token** Chinchilla-optimal corpus
assembled from quality-filtered web, code, math, and scientific prose sources.
The corpus is prepared by a **four-stage pipeline** (download → clean →
tokenize → pack) implemented in the shared `LLM/shared_data/` package and
invoked through a thin project shim at [`data/prepare_data.py`](../data/prepare_data.py).

Tokenised output is stored as **uint32 shards** of **50 million tokens** each,
with **EOS-separated documents** and a JSON **manifest**. Training reads shards
via mmap through [`PretrainDataset`](../training/pretrain.py) in
[`training/pretrain.py`](../training/pretrain.py), yielding `(input_ids,
target_ids)` windows of length `max_seq_len` (4096).

---

## Design Goals

| Goal | How achieved |
|---|---|
| Chinchilla-optimal scale | 8.0 B tokens for ~502 M-param model |
| Reproducible dedup | SHA-256 exact dedup with persisted seen-set |
| Zero-copy training I/O | mmap `shard_*.bin` + `torch.from_file` |
| Cross-project comparability | Universal mixture + shard format in `shared_data` |
| Crash safety | Atomic shard writes; per-stage resume state |
| Document integrity | EOS after every doc; no cross-shard doc splits |

---

## Quick Start

```bash
# From GPT-OSS-Lite project root
python data/prepare_data.py --stage pretrain

# Skip download if raw data already exists
python data/prepare_data.py --stage pretrain --skip-download

# Re-pack only (after shard config change)
python data/prepare_data.py --stage pretrain \
  --skip-download --skip-clean --skip-tokenize

# Single source debug
python data/prepare_data.py --stage pretrain --source fineweb-edu
```

Expected output directory for training (config default):

```
data/pretrain_chinchilla/
  manifest.json
  shard_00000.bin
  shard_00001.bin
  ...
```

Then train:

```bash
python training/pretrain.py --config configs/pretrain_a100_502m.yaml --seed 42
```

---

## The Shim — `data/prepare_data.py`

GPT-OSS-Lite does **not** duplicate the pipeline. The shim:

1. Adds `LLM/` (parent of `GPT-OSS-Lite/`) to `sys.path`.
2. Prints an info banner (corpus size, tokenizer, shard size).
3. Calls `shared_data.prepare_data.main()`.

```python
from shared_data.config import UNIVERSAL_TOTAL_TOKENS, load_universal_data_config
cfg = load_universal_data_config()
tok = cfg["pipeline"]["tokenizer"]
print(f"[data/gptoss] universal corpus: {UNIVERSAL_TOTAL_TOKENS:,} tokens")
print(f"[data/gptoss] tokenizer: {tok['name']} (vocab={tok['vocab_size']:,}, EOS={tok['eos_token_id']})")

from shared_data.prepare_data import main as shared_main
return shared_main()
```

### Path resolution

| Path | Role |
|---|---|
| `GPT-OSS-Lite/data/prepare_data.py` | Project shim |
| `LLM/shared_data/` | Universal pipeline package |
| `LLM/shared_data/config/mixture.yaml` | Source weights |
| `LLM/shared_data/config/data_config.yaml` | Tokenizer, shard, dedup knobs |

The shim uses **universal defaults** — no project-local `data_config.yaml`
override is required for GPT-OSS-Lite.

### CLI flags (delegated)

All flags are parsed by `shared_data.prepare_data`:

| Flag | Purpose |
|---|---|
| `--stage pretrain` | Full pretrain pipeline |
| `--mixture PATH` | Override mixture YAML |
| `--data-config PATH` | Override pipeline YAML |
| `--data-root PATH` | Override output root (`$LLM_DATA_ROOT` or project `data/`) |
| `--source ID` | Process one mixture source only |
| `--skip-download` | Skip HF download |
| `--skip-clean` | Skip quality + dedup |
| `--skip-tokenize` | Skip BPE tokenisation |
| `--skip-pack` | Skip shard packing |
| `--train-tokenizer` | Train custom BPE (HyMo path; not default) |

---

## Pipeline Stages Overview

```
┌─────────────┐    ┌─────────────┐    ┌─────────────┐    ┌─────────────┐
│  download   │───▶│    clean    │───▶│  tokenize   │───▶│ pack_shards │
│  download_  │    │  quality +  │    │  LLaMA-3    │    │  uint32     │
│  raw.py     │    │  SHA dedup  │    │  BPE        │    │  50M tokens │
└─────────────┘    └─────────────┘    └─────────────┘    └─────────────┘
     │                  │                  │                  │
 data/raw/         data/clean/        data/tokens/       data/shards/
 <source>/         <source>/          <source>/          shard_*.bin
 data.jsonl        data.jsonl         tokens.bin         manifest.json
```

Each stage runs as a **subprocess** (`_run_module`) so OOM in tokenisation
does not kill the orchestrator.

**Pipeline version:** `PIPELINE_VERSION = "1.0.0"` in
`shared_data/config.py`.

**Corpus target:** `UNIVERSAL_TOTAL_TOKENS = 8_000_000_000`.

---

## Stage 1 — Download (`download_raw`)

**Script:** `shared_data/scripts/download_raw.py`

**Input:** `mixture.yaml` source definitions.

**Output:** `data/raw/<source_id>/data.jsonl` — one JSON object per line with
a `text` field (or configured `text_field`).

**Behaviour:**

- Streams from HuggingFace `datasets` using each source's `dataset`, `config`,
  `split`, and `text_field`.
- OpenMath concatenates `problem` + `generated_solution` with separator
  `\n\n### Solution\n\n` when `extra_text_field` is set.
- Resumable via per-source state in `data/state/`.

**Example sources** (see [Corpus Mix](#corpus-mix--gptoss-default)):

| Source id | HF dataset |
|---|---|
| `fineweb-edu` | `HuggingFaceFW/fineweb-edu` |
| `fineweb` | `HuggingFaceFW/fineweb` |
| `the-stack-python` | `bigcode/the-stack-python` |
| `openmath` | `nvidia/OpenMathInstruct-2` |
| `arxiv` | `cdv/arxiv-classification` |

---

## Stage 2 — Clean (`clean`)

**Script:** `shared_data/scripts/clean.py`

**Input:** `data/raw/<source>/data.jsonl`

**Output:** `data/clean/<source>/data.jsonl`

### Quality filters

From `data_config.yaml` `pipeline.quality`:

| Filter | Default | Purpose |
|---|---|---|
| `drop_empty` | true | Remove blank docs |
| `min_unique_chars_ratio` | 0.05 | Repetition detector |
| `max_digit_ratio` | 0.50 | OCR garbage |
| `max_punct_ratio` | 0.50 | Symbol spam |
| `max_whitespace_ratio` | 0.50 | Whitespace spam |

Per-source overrides in mixture YAML (`min_chars`, `max_chars`, `lang`).

### Dedup

```yaml
dedup:
  enabled: true
  method: sha256
  n_hash_buckets: 256
  bloom_capacity_per_bucket: 200000
  bloom_error_rate: 0.001
```

- **SHA-256** exact dedup on normalised text.
- Seen-set persisted to `data/state/dedup_<source>.json` every 100k docs —
  crash mid-clean does not lose dedup memory.

---

## Stage 3 — Tokenize (`tokenize`)

**Script:** `shared_data/scripts/tokenize.py`

**Input:** `data/clean/<source>/data.jsonl`

**Output:** `data/tokens/<source>/tokens.bin` — raw uint32 token stream

### Tokenizer config (universal default)

```yaml
tokenizer:
  name: llama3
  vocab_size: 128000
  eos_token_id: 128009
  pad_token_id: 128002
  add_eos: true
```

### Tokenisation rules

- **LLaMA-3 BPE** via HuggingFace tokenizer (`name: llama3`).
- `add_special_tokens: false` during encode — EOS appended manually.
- **EOS after every document** when `add_eos: true`.
- Batch size 1024 docs; prefetch depth 16 for producer/consumer overlap.

### Per-source token stream format

`TokenStream` writes:

```
Header: version (uint32), eos_token_id (uint32)
Body:   token_ids as uint32 little-endian, EOS after each doc
```

---

## Stage 4 — Pack Shards (`pack_shards`)

**Script:** `shared_data/scripts/pack_shards.py`

**Input:** All `data/tokens/<source>/tokens.bin`

**Output:**

- `data/shards/shard_NNNNN.bin`
- `data/manifest.json`

### Packing rules

```yaml
pack:
  docs_per_shard_target: 50000000
  cross_document_boundary_ok: false   # CRITICAL
```

- **`cross_document_boundary_ok: false`** — a document never spans two shards.
  Downstream `PretrainDataset` can align windows on EOS without stitching
  unrelated text.
- Target **50 M tokens per shard** (~190 MB as uint32).
- `ShardWriter` atomic flush + `shard_writer_state.json` for crash resume.

### Shard count estimate

```
8.0e9 tokens / 50e6 ≈ 160 shards
```

---

## Corpus Mix — `gptoss-default`

[`configs/pretrain_a100_502m.yaml`](../configs/pretrain_a100_502m.yaml) documents:

```yaml
data_mix: "gptoss-default"
# mix: fineweb-edu 0.5 / fineweb 0.2 / the-stack-python 0.15 / openmath 0.1 / arxiv 0.05
```

### GPT-OSS designated weights

| Source | Weight | Tokens (of 8.0 B) | Role |
|---|---:|---:|---|
| FineWeb-Edu (`HuggingFaceFW/fineweb-edu`) | **0.50** | 4.00 B | Quality-gated educational web backbone |
| FineWeb (`HuggingFaceFW/fineweb`) | **0.20** | 1.60 B | Raw web diversity |
| the-stack-python (`bigcode/the-stack-python`) | **0.15** | 1.20 B | Python code reasoning |
| OpenMathInstruct-2 (`nvidia/OpenMathInstruct-2`) | **0.10** | 0.80 B | Worked math solutions |
| arxiv (`cdv/arxiv-classification`) | **0.05** | 0.40 B | Long scientific prose |
| **Total** | **1.00** | **8.00 B** | |

### Mix design rationale

- **Web 70%** (edu + raw): language modelling backbone; edu filter improves
  sample efficiency on small models.
- **Code 15%**: strongest lever for reasoning at ~500 M scale.
- **Math 10%**: full problem+solution pairs for derivations.
- **Arxiv 5%**: long documents — supports YaRN 128K extrapolation and
  passkey-style eval readiness.

### Long-context augmentation (training config note)

The A100 config comments note **10% of sequences packed to 4096** with
document-boundary awareness and passkey-style inserts for eval readiness.
This is a **training-time packing policy** (when building the chinchilla
subset path) — the universal pipeline produces EOS-separated shards; project
pack scripts may further curate `data/pretrain_chinchilla/`.

### Code + math combined

Code (0.15) + math (0.10) = **25%** reasoning-heavy tokens — below the 30%
"reasoning diet" ceiling but aligned with GPT-OSS long-context focus (more web
+ long-form arxiv).

---

## Tokenizer — LLaMA-3 BPE

| Field | Value |
|---|---|
| Family | Meta LLaMA-3 BPE |
| `vocab_size` | **128,000** |
| `eos_token_id` | **128009** (`<\|eot_id\|>`) |
| `pad_token_id` | 128002 |
| Config `model.vocab_size` | 128000 (must match) |

GPT-OSS-Lite and **LLaMA-3-Lite** share this tokenizer — their token shards are
**bit-identical** when prepared with the universal pipeline.

Token IDs are stored as **uint32** even though vocab fits in uint16 — uint32
is the safe universal dtype up to 4.29 B tokens per shard file.

---

## Shard Format

### Raw uint32 layout (primary format)

Each `shard_NNNNN.bin`:

```
[token_0, token_1, ..., token_{N-1}]   as little-endian uint32
```

- Continuous token stream across documents.
- Documents separated by **EOS token** (128009) in the stream.
- No per-document length table in the shard — boundaries are EOS markers.

### Alternative: torch_save format

`PretrainDataset._detect_format` checks magic bytes:

- `PK` prefix → `torch.save` tensor (legacy / debug)
- Size divisible by 4 → raw uint32
- Else → assume torch_save

Production shards use **raw uint32**.

### Shard metadata in manifest

Per-shard: `index`, `path`, `n_tokens`, `sha256`, `n_eos`.

---

## Manifest Schema

`data/manifest.json` (or the packed corpus under `data/pretrain_chinchilla` including its manifest):

```json
{
  "version": "1.0.0",
  "vocab_size": 128000,
  "eos_token_id": 128009,
  "pad_token_id": 128002,
  "tokenizer_name": "llama3",
  "dtype": "uint32",
  "shard_size_tokens": 50000000,
  "total_tokens": 8000000000,
  "shard_count": 160,
  "shards_dir": "data/shards",
  "shards": [ { "index": 0, "path": "...", "n_tokens": ..., "sha256": "...", "n_eos": ... } ],
  "sources": { ... }
}
```

[`PretrainDataset._load_manifest`](../training/pretrain.py) reads optional
fields: `eos_token_id`, `vocab_size`, `total_tokens`, `shard_count`, `dtype`.

---

## Output Layout — `data/pretrain_chinchilla`

Training config:

```yaml
data:
  train_data_path: "data/pretrain_chinchilla"
```

Expected structure:

```
data/pretrain_chinchilla/
├── manifest.json
├── shard_00000.bin
├── shard_00001.bin
└── ...                          # ~160 shards for 8B tokens
```

This path is a **project-local packed subset** or symlink/copy of universal
`data/pretrain_chinchilla/` after pipeline completion. The name `pretrain_chinchilla`
reflects Chinchilla-optimal 8B token budget for the 502M model.

Intermediate pipeline dirs (under `data/` or `$LLM_DATA_ROOT`):

```
data/
├── raw/<source_id>/data.jsonl
├── clean/<source_id>/data.jsonl
├── tokens/<source_id>/tokens.bin
├── shards/shard_*.bin          # universal output
├── state/                      # resume state
└── pretrain_chinchilla/        # training consumption path
```

---

## Training Loader — `PretrainDataset`

Defined in [`training/pretrain.py`](../training/pretrain.py).

### Constructor

```python
PretrainDataset(data_path: str, max_seq_len: int)
```

- `data_path`: file or directory.
- `max_seq_len`: 4096 for default config.

Raises `FileNotFoundError` with hint to run `python data/prepare_data.py` if
missing.

### Layout modes

| Mode | Trigger | Storage |
|---|---|---|
| `single` | `data_path` is a file | One mmap'd `torch.load` tensor |
| `sharded` | `data_path` is directory with `shard_*.bin` | Multiple mmap shards |

### Sharded initialisation

```python
shard_paths = sorted(Path(data_dir).glob("shard_*.bin"))
shard_formats = [_detect_format(p) for p in shard_paths]
raw_dtype = uint32 → torch.int32 for mmap
shard_sizes, shard_offsets  # cumulative token offsets
_n_samples = (total_tokens - 1) // max_seq_len
```

### `__getitem__` — sliding windows

Returns `(input_ids, target_ids)` each shape `(max_seq_len,)`:

```python
chunk = tokens[start : start + max_seq_len + 1]
return chunk[:-1], chunk[1:]   # next-token prediction
```

**Sample index `idx`:** window starts at token `idx × max_seq_len`.

### Shard loading cache

```python
def _load_shard(shard_idx):
    # mmap torch.load or torch.from_file(shared=True)
    # caches last-loaded shard in self._cache_shard
```

Only one shard cached — sufficient for sequential-ish access; random shuffle
across windows may reload shards frequently (acceptable with mmap).

### Cross-shard windows

`_get_window_sharded` handles windows spanning shard boundaries:

1. Fast path: if window fits in one shard, slice locally.
2. Slow path: concatenate slices from multiple shards, then split input/target.

Windows may cross **shard** boundaries but individual **documents** never cross
shards (packing invariant) — EOS alignment preserved within shard interior.

---

## DataLoader Configuration

From [`pretrain.py`](../training/pretrain.py) + [`AGENTS.md`](../AGENTS.md):

```python
DataLoader(
    ds,
    batch_size=8,              # micro_batch_size
    shuffle=True,
    num_workers=4,
    pin_memory=True,
    persistent_workers=True,
    drop_last=True,
)
```

| Knob | Value | Purpose |
|---|---|---|
| `batch_size` | 8 | Micro-batch |
| `shuffle` | True | Random window order |
| `num_workers` | 4 | Parallel decode |
| `pin_memory` | True | Faster CPU→GPU transfer |
| `persistent_workers` | True | Avoid worker respawn |
| `drop_last` | True | Consistent batch shapes |

Each batch: `input_ids` `(8, 4096)`, `target_ids` `(8, 4096)`.

---

## Cross-Project Sharing

The universal pipeline at `LLM/shared_data/` is shared by five LLM projects.
**Prepare once** if `$LLM_DATA_ROOT` points to a common directory:

```bash
export LLM_DATA_ROOT=/path/to/shared/llm_corpus
python data/prepare_data.py --stage pretrain
```

All projects mmap the same `shard_*.bin` files.

**GPT-OSS + LLaMA-3:** identical tokenizer → **bit-identical shards**.

**HyMo / DeepSeek:** different tokenizers → re-tokenise only (skip download/clean).

---

## Idempotency and Resume

| Stage | Resume mechanism |
|---|---|
| download | Per-source state files |
| clean | `data/state/dedup_<source>.json` every 100k docs |
| tokenize | Per-source progress state |
| pack | `shard_writer_state.json` + partial shard tmp |

Re-running `prepare_data.py` picks up incomplete work. Manifest rebuilt from
on-disk shards after pack — never references missing files.

Orchestrator seeds RNG: `seed=42` from `data_config.yaml` for reproducible
sampling during pack.

---

## Disk and Time Budgets

| Item | Estimate |
|---|---|
| Raw JSONL (all sources) | ~1–2 TB (before dedup) |
| Token shards (uint32) | ~32 GB for 8B tokens (8B × 4 bytes) |
| Per-shard size | ~190 MB (50M × 4 bytes) |
| Shard count | ~160 |
| Full pipeline (1× A100) | ~27–40 h (download-bound) |

`shard_size_tokens: 50000000` in config matches pipeline default.

---

## Validation Checklist

Before launching 61k-step pretrain:

- [ ] the packed corpus under `data/pretrain_chinchilla` including its manifest exists
- [ ] `manifest.total_tokens` ≈ 8.0e9
- [ ] `manifest.vocab_size == 128000`
- [ ] `manifest.eos_token_id == 128009`
- [ ] `manifest.dtype == "uint32"`
- [ ] All `shard_*.bin` files present (count matches `shard_count`)
- [ ] `python -c "from training.pretrain import PretrainDataset; d=PretrainDataset('data/pretrain_chinchilla', 4096); print(len(d), d[0][0].shape)"`
- [ ] No token id ≥ vocab_size in random windows

---

## Appendix A — Token arithmetic

```
Corpus:     8,000,000,000 tokens
Seq len:    4,096
Samples:    floor((8e9 - 1) / 4096) ≈ 1,953,125 windows
Micro-bs:   8
Accum:      4
Tokens/step: 8 × 4 × 4096 = 131,072
Steps:      61,000
Train tokens: 61,000 × 131,072 = 7,995,392,000 ≈ 8.0B
```

---

## Appendix B — Window crossing shards

```
Shard 0: [ ... docA ... EOS ... docB ... EOS ... partial_docC ]
Shard 1: [ ... rest_docC ... EOS ... docD ... ]

Window at idx=k may start in shard 0 and extend into shard 1.
PretrainDataset concatenates mmap slices — no extra RAM for full corpus.

Document never split across shards:
  docC entirely in shard 0 OR entirely in shard 1 — never both.
```

---

## Appendix C — Directory tree

```
GPT-OSS-Lite/
├── data/
│   ├── prepare_data.py          ← shim
│   ├── pretrain_chinchilla/     ← train_data_path
│   │   ├── manifest.json
│   │   └── shard_*.bin
│   └── (pipeline intermediates under data/ or $LLM_DATA_ROOT)
├── configs/
│   └── pretrain_a100_502m.yaml
└── training/
    └── pretrain.py              ← PretrainDataset

LLM/
└── shared_data/                 ← universal pipeline (authoritative)
    ├── prepare_data.py
    ├── config/
    │   ├── mixture.yaml
    │   └── data_config.yaml
    ├── scripts/
    │   ├── download_raw.py
    │   ├── clean.py
    │   ├── tokenize.py
    │   └── pack_shards.py
    ├── shard_writer.py
    ├── manifest.py
    └── dedup.py
```

---

## Appendix D — Source dataset reference

| Mix id | HuggingFace | Text field | Notes |
|---|---|---|---|
| fineweb-edu | `HuggingFaceFW/fineweb-edu` | `text` | `sample-10BT` config |
| fineweb | `HuggingFaceFW/fineweb` | `text` | Raw web diversity |
| the-stack-python | `bigcode/the-stack-python` | `content` | Python only |
| openmath | `nvidia/OpenMathInstruct-2` | `problem` + solution | Concatenated fields |
| arxiv | `cdv/arxiv-classification` | `text` | Long papers |

Quality bounds from mixture YAML per source (`min_chars`, `max_chars`).

---

## Load-Bearing Invariants

1. **EOS after every document** — `add_eos: true`.
2. **No cross-shard documents** — `cross_document_boundary_ok: false`.
3. **uint32 shard dtype** for vocab 128000.
4. **mmap consumption** — `weights_only=True` or `from_file(shared=True)`.
5. **train_data_path** must exist before `pretrain.py` starts.
6. **Vocab/EOS** in manifest must match `ModelConfig.vocab_size`.

---

## References

- [`data/prepare_data.py`](../data/prepare_data.py) — project shim
- `LLM/shared_data/README.md` — workspace canonical pipeline docs
- `LLM/shared_data/documentation/prepare_data.md` — orchestrator detail
- [`training/pretrain.py`](../training/pretrain.py) — `PretrainDataset`
- [`configs/pretrain_a100_502m.yaml`](../configs/pretrain_a100_502m.yaml)
- [training.md](training.md) — DataLoader and batch arithmetic

<!-- docs:verified 2026-07-31 · fa6f918 -->
