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
5. [Shared Package — `LLM/shared_data/`](#shared-package--llmshared_data)
6. [Pipeline Stages Overview](#pipeline-stages-overview)
7. [Stage 1 — Download (`download_raw`)](#stage-1--download-download_raw)
8. [Stage 2 — Clean (`clean`)](#stage-2--clean-clean)
9. [Stage 3 — Tokenize (`tokenize`)](#stage-3--tokenize-tokenize)
10. [Stage 4 — Pack Shards (`pack_shards`)](#stage-4--pack-shards-pack_shards)
11. [Corpus Mix — `gptoss-default`](#corpus-mix--gptoss-default)
12. [Tokenizer — LLaMA-3 BPE](#tokenizer--llama-3-bpe)
13. [Shard Format](#shard-format)
14. [Manifest Schema](#manifest-schema)
15. [Output Layout — `data/pretrain_chinchilla`](#output-layout--datapretrain_chinchilla)
16. [Training Loader — `PretrainDataset`](#training-loader--pretraindataset)
17. [DataLoader Configuration](#dataloader-configuration)
18. [Cross-Project Sharing](#cross-project-sharing)
19. [Idempotency and Resume](#idempotency-and-resume)
20. [Disk and Time Budgets](#disk-and-time-budgets)
21. [Validation Checklist](#validation-checklist)
22. [Appendix A — Token arithmetic](#appendix-a--token-arithmetic)
23. [Appendix B — Window crossing shards](#appendix-b--window-crossing-shards)
24. [Appendix C — Directory tree](#appendix-c--directory-tree)
25. [Appendix D — Source dataset reference](#appendix-d--source-dataset-reference)
26. [Load-Bearing Invariants](#load-bearing-invariants)
27. [References](#references)

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

1. Prepends `GPT-OSS-Lite/` (`_PROJECT_ROOT`) and `LLM/` (`_LLM_ROOT`) to
   `sys.path`.
2. Prints an info banner (corpus size, tokenizer, shard size).
3. Calls `data/prepare_data.py:main`, which delegates to
   `shared_data.prepare_data.main()`.

```python
_PROJECT_ROOT = Path(__file__).resolve().parents[1]   # GPT-OSS-Lite/
_LLM_ROOT = Path(__file__).resolve().parents[2]       # .../LLM/
for _p in (_PROJECT_ROOT, _LLM_ROOT):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from shared_data.config import UNIVERSAL_TOTAL_TOKENS, load_universal_data_config
# ... info banner ...
from shared_data.prepare_data import main as shared_main
return shared_main()
```

### Path resolution

| Path | Role |
|---|---|
| `GPT-OSS-Lite/data/prepare_data.py` | Project shim |
| `GPT-OSS-Lite/` | On `sys.path` as `_PROJECT_ROOT` |
| `LLM/` | On `sys.path` as `_LLM_ROOT`; `import shared_data` resolves here |
| `LLM/shared_data/` | Workspace universal pipeline (required) |
| `LLM/shared_data/config/mixture.yaml` | Source weights |
| `LLM/shared_data/config/data_config.yaml` | Tokenizer, shard, dedup knobs |

There is **no** `data/shared_data/` vendored copy in this repo. The shim uses
**universal defaults** — no project-local `data_config.yaml` override is
required for GPT-OSS-Lite.

---

## Shared Package — `LLM/shared_data/`

The four-stage pipeline lives in the **workspace-level** package at
`LLM/shared_data/` (sibling of `GPT-OSS-Lite/` under `LLM/`). Run
`data/prepare_data.py` from a CoreProjects-style layout where that package
exists, or ensure `LLM/` is on `PYTHONPATH`.

```
LLM/
├── shared_data/                 ← universal pipeline (required)
│   ├── config/
│   │   ├── mixture.yaml
│   │   └── data_config.yaml
│   ├── scripts/                 ← download_raw, clean, tokenize, pack_shards
│   ├── prepare_data.py          ← orchestrator
│   ├── shard_writer.py
│   ├── manifest.py
│   └── ...
└── GPT-OSS-Lite/
    └── data/
        ├── prepare_data.py      ← project shim
        └── pretrain_chinchilla/ ← training consumption path
```

Other LLM projects in the portfolio may vendor `data/shared_data/` for
standalone clones; GPT-OSS-Lite does not in this repository.

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

### BPE merge algorithm

The LLaMA-3 tokenizer is a **byte-level byte-pair encoding** (BPE): a greedy
algorithm that grows a fixed vocabulary of $V = 128000$ entries by fusing
the most frequent adjacent pair in the corpus, one merge at a time. Training
starts from the 256 raw byte values (one entry per byte value, which guarantees
every possible string is spellable — nothing is out-of-vocabulary) and
iterates:

1. **Count** — for every adjacent pair $(u, v)$ in the current token stream
   $x_1, x_2, \ldots, x_L$, tally how often $u$ is immediately followed by $v$.
2. **Fuse** — pick the most frequent pair, add the fused token $ab$ to the
   vocabulary, and replace every occurrence of the adjacent pair $a\,b$ with
   the single token $ab$.
3. **Repeat** until $|\mathcal{V}| = 128000$.

With $\mathcal{V}_t$ the vocabulary after $t$ merges and $f_t(u,v)$ the corpus
pair count under the current segmentation, one merge step is

$$\mathcal{V}_{t+1} = \mathcal{V}_t \cup \{ab\}, \qquad (a,b) = \arg\max_{(u,v)} f_t(u,v), \qquad f_t(u,v) = \sum_{k=1}^{L-1} \mathbb{1}[x_k = u \;\wedge\; x_{k+1} = v]
\tag{1}$$

Each applied merge shortens the stream by exactly one token per occurrence, so
argmax-frequency is the greedy maximiser of immediate compression per merge —
the same criterion tiktoken-style BPE uses. Encoding a new string replays the
learned merges greedily left to right; decoding expands each token back to the
byte string it was built from. Merges are order-dependent and
overlap-ambiguous (in a run $a\,a\,a$ the pair $(a,a)$ occurs twice but a
left-to-right sweep fuses only once), so a pretrained tokenizer is a fixed
codec: re-training the BPE changes the ids, which is why GPT-OSS-Lite and
LLaMA-3-Lite pin the same `name: llama3` tokenizer and can share bit-identical
shards. The full treatment — merge economics, the 128K coverage-versus-parameter
trade-off, and the ambiguity bounds — lives in
[tokenization_bpe.md](theory/tokenization_bpe.md); this section only states
what the pipeline relies on.

The vocabulary is also a parameter budget: at $V = 128000$ and $d = 768$,
the tied embedding/output head costs $V \cdot d = 98304000$ parameters —
about 19.6% of the 501.8M total — before any transformer block, so the
tokenizer choice is load-bearing for the model's size, not just its data
([tokenization_bpe.md §3.6](theory/tokenization_bpe.md)).

| Field | Value |
|---|---|
| Family | Meta LLaMA-3 BPE |
| `vocab_size` | **128,000** |
| `eos_token_id` | **128009** (`<\|eot_id\|>`) |
| `pad_token_id` | 128002 |
| Config `model.vocab_size` | 128000 (must match) |

GPT-OSS-Lite and **LLaMA-3-Lite** share this tokenizer — the shards produced by
both projects are **bit-identical** and can be shared verbatim (same BPE, same
EOS, same uint32 pack format).

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

### Byte math

The format decision is a byte budget. Each token id is a little-endian uint32,
so every token costs exactly four bytes on disk:

$$B_{\text{token}} = 4 \text{ B/token}
\tag{2}$$

At $S = 50000000$ tokens per shard (`shard_size_tokens` in the shared
`data_config.yaml`), one shard file is

$$B_{\text{shard}} = S \cdot B_{\text{token}} = 5 \times 10^7 \times 4 = 2.0 \times 10^8 \text{ B} = 200 \text{ MB} \;(\approx 190.7 \text{ MiB})
\tag{3}$$

— the "~190 MB" figure quoted in the config, depending on whether the unit is
decimal (MB) or binary (MiB). The 8.0 B-token corpus packs into

$$K = \left\lceil \frac{N_{\text{tok}}}{S} \right\rceil = \left\lceil \frac{8 \times 10^9}{5 \times 10^7} \right\rceil = 160 \text{ shards}
\tag{4}$$

and occupies

$$B_{\text{total}} = K \cdot B_{\text{shard}} = N_{\text{tok}} \cdot B_{\text{token}} = 8 \times 10^9 \times 4 = 3.2 \times 10^{10} \text{ B} = 32 \text{ GB}
\tag{5}$$

on disk. The ceiling in (4) is real: packing targets 50M tokens but stops at a
document boundary (`cross_document_boundary_ok: false`), so the final shard may
hold fewer tokens, and `shard_count` in the manifest records what was actually
written.

**Why uint32 and not uint16 or uint8.** The dtype must represent every token
id the stream can contain: ordinary tokens run to $V - 1 = 127999$, and the
EOS special is $\text{eos\_token\_id} = 128009$. The candidate widths
satisfy

$$2^{8} = 256 < 2^{16} = 65536 < 128009 \leq 2^{32}
\tag{6}$$

so neither uint8 nor uint16 can encode the vocabulary at all — the minimum
width is $\lceil \log_2 128009 \rceil = 17$ bits, and the next native type
is uint32. Its ceiling is far above any realistic id:

$$2^{32} - 1 = 4294967295 \gg 128009
\tag{7}$$

the headroom behind the config comment that uint32 is safe "up to 4.29 B
tokens". The width is a storage cost, not a compute cost: at read time
`training/pretrain.py:PretrainDataset._init_sharded` maps the manifest dtype to
a torch int type (`uint32` → `torch.int32`) and mmaps the file, so the
4 B/token layout never inflates the active working set — only the disk
footprint.

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

### How `PretrainDataset` consumes each field

`training/pretrain.py:PretrainDataset._load_manifest` reads exactly five
optional fields and caches them on the dataset; it never validates them against
the filesystem. Consumption per field:

| Field | Loader behaviour | Role |
|---|---|---|
| `eos_token_id` | Stored as `self.eos_token_id` (`None` when absent) | Records the boundary id the packer used (128009). The loader never scans for it — the packing invariant `cross_document_boundary_ok: false` guarantees no document spans shards — so this is the contract value the validation checklist re-checks. |
| `vocab_size` | Stored as `self.vocab_size` (`None` when absent) | Declares the token space; must equal `ModelConfig.vocab_size` (128000) or embedding lookups shift. |
| `total_tokens` | Stored as `self.total_tokens` (default `0`) | Declared corpus size; informational — `_init_sharded` re-derives the authoritative total from on-disk file sizes. |
| `shard_count` | Stored as `self.shard_count` (default `0`) | Declared shard count; the actual list comes from globbing `shard_*.bin`, so a mismatch surfaces only via the checklist. |
| `dtype` | Stored as `self.dtype` (default `"uint32"` when the key is missing; `None` if no manifest), then mapped in `training/pretrain.py:PretrainDataset._init_sharded` via `{"uint32": torch.int32, "uint16": torch.int16, "uint8": torch.int8}` | **Load-bearing**: sets the mmap element type in `training/pretrain.py:PretrainDataset._load_shard` (`torch.from_file`) and the byte-per-token divisor in `training/pretrain.py:PretrainDataset._size_in_tokens`. A wrong `dtype` silently halves or quadruples every token count. |

The last row is the only field that changes loader *arithmetic*.
`training/pretrain.py:PretrainDataset._init_sharded` computes each shard's
token count from its file size and the dtype width,

$$n_i = \frac{\text{bytes}(i)}{b}, \qquad b = \text{itemsize}(\texttt{dtype})
\tag{8}$$

with $b = 4$ for uint32, then builds cumulative `shard_offsets` from those
$n_i$. `total_tokens` and `shard_count` are therefore advisory — the bytes on
disk and `dtype` are authoritative — which is why a missing manifest degrades
silently (all fields fall back to `None`/`0`/`"uint32"` inside
`training/pretrain.py:PretrainDataset._load_manifest`) and the
[validation checklist](#validation-checklist) re-verifies the manifest before a
61k-step run.

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
LLM/
├── shared_data/                 ← universal pipeline (required)
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
└── GPT-OSS-Lite/
    ├── data/
    │   ├── prepare_data.py      ← shim
    │   ├── pretrain_chinchilla/ ← train_data_path
    │   │   ├── manifest.json
    │   │   └── shard_*.bin
    │   └── (pipeline intermediates under data/ or $LLM_DATA_ROOT)
    ├── configs/
    │   └── pretrain_a100_502m.yaml
    └── training/
        └── pretrain.py          ← PretrainDataset
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

## How to verify

Shard writer, manifest, and `PretrainDataset` integration:

```bash
python3 -m pytest tests/test_data_pipeline.py -v
python3 -m pytest tests/test_training.py -v -k PretrainDataset
```

After packing, smoke-load the corpus (requires `data/pretrain_chinchilla`):

```bash
python3 -c "from training.pretrain import PretrainDataset; d=PretrainDataset('data/pretrain_chinchilla', 4096); print(len(d), d[0][0].shape)"
```

`test_data_pipeline.py` is skipped when the sibling `shared_data` package is not
importable (CoreProjects layout).

---

## References

- [`data/prepare_data.py`](../data/prepare_data.py) — project shim
- `LLM/shared_data/README.md` — workspace canonical pipeline docs
- `LLM/shared_data/documentation/prepare_data.md` — orchestrator detail
- [`training/pretrain.py`](../training/pretrain.py) — `PretrainDataset`
- [`configs/pretrain_a100_502m.yaml`](../configs/pretrain_a100_502m.yaml)
- [training.md](training.md) — DataLoader and batch arithmetic

<!-- docs:verified 2026-08-04 · 5da1a80 -->
