# Data Pipeline — GPT-OSS-Lite

> **The full pipeline that turns 8 B tokens of public web text into the
> training corpus for the 502 M-param GPT-OSS-Lite model.**

---

## 1. Overview

The pipeline has **four stages**, each independently resumable and
independently testable. Outputs flow left-to-right:

```
┌────────────┐    ┌──────────────┐    ┌────────────┐    ┌────────────┐
│  download  │ →  │ clean + dedup│ →  │  tokenize  │ →  │ pack shards│
│ raw jsonl  │    │ clean jsonl  │    │ tokens.bin │    │ shard_N.bin│
└────────────┘    └──────────────┘    └────────────┘    └────────────┘
   HF datasets       quality + SHA-256    EOS-separated     round-robin
   streaming         hash-sharded         uint32 stream     atomic write
                                                          + manifest.json
```

| Stage | Module(s) | Input → Output | Wall time (8 B tokens) |
|-------|-----------|---------------|------------------------|
| 1. download | `data/scripts/download_raw.py` | HF datasets → `data/raw/<src>/data.jsonl` | 4–8 h |
| 2. clean + dedup | `data/scripts/tokenize.py::tokenize_source` (`_filter_and_dedup`) | raw JSONL → `data/clean/<src>/data.jsonl` | 30–60 min |
| 3. tokenize | `data/shard_writer.py::TokenStream` | clean JSONL → `data/tokens/<src>/data.bin` | 1–2 h |
| 4. pack + manifest | `data/scripts/pack_shards.py` | per-source tokens → `data/shards/shard_NNNNN.bin` + `data/manifest.json` | 30–60 min |

The top-level orchestrator (`data/prepare_data.py`) runs the four
stages in order, but each can be invoked standalone for re-runs or
partial rebuilds.

---

## 2. Why this design

### 2.1 Per-source independence

Each source has its own download / clean / tokenize directory. This
gives us:

- **Resumability** — if FineWeb-Edu's 4 TB download stalls, the other
  4 sources keep progressing.
- **Per-source accounting** — the manifest records exactly how many
  tokens each source contributed. Useful for ablations and for
  diagnosing "where did my training data come from" questions.
- **Parallelisation** — each source is a self-contained directory;
  the four stages can run in parallel across machines (not
  implemented here, but trivial to add).

### 2.2 SHA-256 hash-sharded dedup

The naive approach (one global `set` of SHA-256 hashes) uses ~12 GB
of RAM for 200 M documents. Our implementation:

1. Hashes every document (SHA-256 of normalised text — strips
   whitespace to defeat trivial evasion).
2. Buckets by `hash → bucket` (256 buckets, modulo).
3. Pass 1 writes per-bucket hash files; pass 2 dedups each bucket
   independently with an in-memory set per bucket (~50 MB RAM).
4. Bloom filters per bucket provide a constant-memory fallback when
   bucket size exceeds `bloom_capacity_per_bucket` (200k by default).

Constant memory, deterministic, resumable.

### 2.3 EOS-separated token streams

Every document boundary is marked with the EOS token id (128009 for
LLaMA-3 BPE). The `TokenStream` writer:

- Strips any trailing EOS that the tokenizer may have inserted.
- Appends exactly one EOS after every document.
- Validates every token id against `[0, vocab_size + 256)` to catch
  silent token-id corruption.

The training script's `PretrainDataset` reads the manifest to know
the EOS id. When a training window straddles two documents, the
EOS acts as a regular token (the model attends to it normally) — no
silent cross-document context leakage.

### 2.4 Atomic shard writes

Each shard is written to `<shard>.bin.tmp` then atomically renamed.
A crash mid-write leaves either the old shard or the new one — never
a half-written one. This is the same pattern used by
`utils/checkpoint.py::CheckpointManager` and is critical because the
training script's NaN-guard relies on being able to roll back to a
known-good shard boundary.

### 2.5 Round-robin packing

Sources are interleaved one-document-at-a-time into the shard buffer.
Without this, the first N shards would be only FineWeb-Edu (the 50%
source) and the last only arxiv (5%) — causing the loss curve to
"drift" as the source mix shifts during training.

---

## 3. The five data sources

From `data/config/mixture.yaml`:

| Source | Weight | Target tokens | Domain |
|--------|-------:|--------------:|--------|
| fineweb-edu | 0.50 | 4.00 B | Web, edu-filtered |
| fineweb | 0.20 | 1.60 B | Web, unfiltered |
| the-stack-python | 0.15 | 1.20 B | Python code |
| openmath | 0.10 | 0.80 B | Math problems + solutions |
| arxiv | 0.05 | 0.40 B | Scientific prose |
| **Total** | **1.00** | **8.00 B** | |

This is the Chinchilla-optimal mix for a 502 M-param model. Each
source has its own quality filter thresholds (e.g. min/max chars)
because code (the-stack-python) tolerates very different doc sizes
than web text (fineweb-edu).

---

## 4. Manifest schema

The manifest (`data/manifest.json`) is the contract between the
pipeline and the training script. Its top-level keys:

```jsonc
{
  "version": "1",
  "created_utc": "2026-06-29T01:23:45Z",
  "vocab_size": 128000,            // regular vocab (0..127999)
  "eos_token_id": 128009,          // LLaMA-3 <|eot_id|>
  "pad_token_id": 128002,
  "tokenizer_name": "llama3",
  "dtype": "uint32",               // shard element dtype
  "shard_size_tokens": 50000000,   // 50M tokens / shard
  "total_tokens": 8000000000,      // sum across all shards
  "shard_count": 161,              // 8B / 50M = 160 + 1 remainder
  "shards_dir": "data/shards",
  "shards": [                      // per-shard metadata
    {"index": 0, "path": "data/shards/shard_00000.bin",
     "n_tokens": 50000000, "sha256": "...", "n_eos": 1234567},
    ...
  ],
  "sources": {                     // per-source attribution
    "fineweb-edu": {
      "target_tokens": 4000000000, "actual_tokens": 3998234567,
      "n_docs": 12345678, "n_dedup_dropped": 23456, "shard_count": 80
    },
    ...
  },
  "config_hash": "sha256:...",
  "mixture_hash": "sha256:..."
}
```

`tools/data_pipeline_checker.py --project gpt-oss-lite --data-dir ...`
reads this file (or scans shards directly) and reports issues:
shard count, total tokens, vocab coverage, EOS presence, mmap cache
size, dedup ratio.

---

## 5. Validation rules

The pipeline enforces correctness at every stage. Failures abort
the stage with a clear error; the per-stage state file is preserved
so the user can fix the input and resume.

| Stage | What we check | Failure mode |
|-------|---------------|--------------|
| download | Streaming iterator returns | raises → state saved |
| clean | Per-doc filter chain | doc dropped + reason counted |
| clean | SHA-256 dedup | duplicate dropped + reason counted |
| tokenize | Token id ≤ vocab_size + 256 | raises immediately |
| tokenize | EOS appended after every doc | enforced by writer |
| pack | No doc split across shards (default) | raises if doc > shard size |
| pack | Atomic write | tmp + rename → no partial files |
| verify | Re-read shard, confirm count + EOS + max | raises on mismatch |

---

## 6. Resumability

Every stage persists a JSON state file in `data/state/`:

- `state/download_<src>.json`   — last row index + char count
- `state/clean_<src>.json`      — last doc index + keep/drop counts + reasons
- `state/tokenize_<src>.json`   — last doc index + token count + `complete` flag
- `state/pack_shards.json`      — last shard index + per-source accumulators

Re-running the pipeline after a crash picks up exactly where it left
off. The shard files are immutable once written (their content
depends only on the immutable input tokens), so we never re-write
existing shards — we only continue from the next shard index.

---

## 7. Usage

### Full pipeline (one command)

```bash
python data/prepare_data.py --stage pretrain
```

This downloads → cleans → tokenises → packs. Expect 6–12 hours on
a single A100 for the full 8 B tokens.

### Just one source

```bash
python data/scripts/download_raw.py --source fineweb-edu
python data/scripts/tokenize.py --source fineweb-edu
python data/scripts/pack_shards.py
```

### Validate an existing pipeline

```bash
python ../tools/data_pipeline_checker.py \
    --project gpt-oss-lite \
    --data-dir LLM/GPT-OSS-Lite/data
```

Expected output:
```
Project:    gpt-oss-lite
Data dir:   LLM/GPT-OSS-Lite/data
Shards:     161
Tokens:     8,000,000,000
Vocab exp:  128,000
EOS in vocab: True
mmap active:  True
OK:         True
```

### Force a fresh rebuild

```bash
rm -rf data/state data/raw data/clean data/tokens data/shards data/manifest.json
python data/prepare_data.py --stage pretrain
```

---

## 8. Implementation details

### 8.1 Why `uint32` and not `torch.long` (int64)?

`torch.long` is int64 (8 bytes/token). For 8 B tokens, that's 64 GB
of shards. `uint32` (4 bytes/token) is 32 GB — same precision for
tokens ≤ 4.29 B (we have ≤ 132 k). The training script reads via
`torch.from_file(..., dtype=torch.int32, shared=True)` which mmaps
the raw bytes without any conversion.

For vocab ≤ 65 535 we could use `uint16` (16 GB), but GPT-OSS-Lite
uses the LLaMA-3 128 k vocab, so `uint32` is the safe default.
`select_token_dtype()` picks the smallest dtype automatically.

### 8.2 Why per-document EOS?

Without EOS, the training windows (max_seq_len = 4096) would
silently concatenate unrelated documents, teaching the model that
the last paragraph of FineWeb-Edu doc #1234 is followed by the
first paragraph of FineWeb-Edu doc #1235. This is a form of data
leakage that *does* hurt long-context modelling.

With EOS after every doc, the windows can safely cross document
boundaries — the EOS just acts as a regular token the model learns
to predict at the end of "real" sequences.

### 8.3 Why mmap the shards?

`torch.from_file(..., shared=True)` mmaps the shard file into the
process's virtual address space. Subsequent `tensor[a:b]` accesses
are zero-copy — the OS pages in the requested range on demand and
the page cache warms for repeat accesses. This is critical for the
8 B-token corpus: we cannot afford to load 32 GB into RAM just to
slice it.

### 8.4 Why not NaN-safe arithmetic in the writer?

The shard writer does arithmetic on Python ints (token counts,
offsets), not on tensors. There is no NaN/Inf failure mode here.
The NaN-guard in `training/pretrain.py` operates on the loss
tensor, not on the data pipeline.

---

## 9. Limitations / future work

- **No incremental manifest updates.** Re-running `pack_shards`
  recomputes the manifest from scratch. For very large rebuilds
  this is fine (the verify step is fast), but it's a candidate
  for optimisation.
- **Single-process dedup.** The hash-sharded dedup is parallelisable
  across processes (each process owns a bucket range), but we don't
  ship a launcher. The current implementation is fast enough on a
  single A100.
- **No language identification model.** The language filter is a
  cheap ASCII-ratio + bigram heuristic. For higher precision,
  swap in `fasttext.langdetect` (~1 MB model, <1 ms/doc).
- **No streaming tokenisation.** Tokenisation currently loads each
  source's clean JSONL fully before tokenising. For 8 B tokens this
  is fine (~200 GB peak RAM at 5 chars/token), but a streaming
  path would let it run on a 64 GB machine. Tracked in the
  backlog below.

### Backlog (not yet implemented)

- Streaming tokenisation (constant-RAM, single source at a time).
- HF datasets cache eviction (free 5 TB after `download_raw`).
- Multiprocess pack with one process per source.

---

## 10. Tests

60 unit + integration tests in `tests/test_data_pipeline.py`:
- 12 IO / hashing / state tests
- 8 quality-filter tests (incl. language hint)
- 6 BloomFilter + Deduper tests (two-pass, resume)
- 11 ShardWriter / TokenStream tests (atomicity, EOS, OOV, splits)
- 6 Manifest tests (round-trip + 4 validation rules)
- 5 PretrainDataset tests (new format + backward compat)
- 1 end-to-end mini pipeline test (synthetic corpus)

CPU-only, all <0.3 s. Run with:

```bash
python3 -m pytest tests/test_data_pipeline.py -v
```
---

## Implementation notes (extracted from code review)

- **mmap zero-copy slices**: `PretrainDataset._load_shard` mmaps each shard
  via `torch.from_file(path, dtype=raw_dtype, shared=True, size=...)` (raw
  bytes layout) or `torch.load(path, mmap=True)` (legacy torch.save layout).
  Each `__getitem__` then returns a zero-copy slice of the mmap'd tensor —
  the OS pages in the requested range on demand and the page cache warms
  for repeat accesses. This is critical for the 8B-token corpus: we cannot
  afford to load 32 GB into RAM just to slice it.
- **DataLoader prefetch knobs**: `training/pretrain.py` constructs the
  DataLoader with `num_workers=4`, `pin_memory=True` (on CUDA), and
  `persistent_workers=(num_workers > 0)`. This enables async H2D transfer
  and keeps the worker processes alive across epochs (avoids the per-epoch
  re-import cost).
- **`uint32` storage**: the manifest's `dtype` field records the shard
  element dtype. GPT-OSS-Lite uses `uint32` (4 bytes/token) because the
  LLaMA-3 vocab is 128K (fits in uint16 numerically, but uint32 is the
  safe default for multi-trillion-token corpora with reserved special
  tokens up to vocab+256). `select_token_dtype(vocab_size)` picks the
  smallest dtype automatically.
