# GPT-OSS-Lite — Tokenization

> **Chapter on `data/prepare_data.py` and the shared corpus pipeline.** What a tokenizer does, why the vocabulary is the interface between data and model, the byte-pair encoding (BPE) merge algorithm and its ambiguity, the 128,000-token LLaMA-3 vocabulary economics, EOS-delimited packing, and how `training/pretrain.py:PretrainDataset` consumes the shards. Pipeline mechanics: [data pipeline](../training.md); batch arithmetic: [training](../training.md); the softmax over this vocabulary: [sampling](optimizers-and-numerics.md).

---

## 60-second summary

A language model has no notion of letters. Its first layer is an integer-keyed embedding lookup, so before training, every byte of the corpus must be mapped to an id from a fixed **vocabulary** — that mapping is a tokenizer's whole job. GPT-OSS-Lite uses the **LLaMA-3 byte-level BPE** tokenizer with a vocabulary of **128,000** entries: 256 single bytes, ~127.5K learned merge tokens, and a reserved block of special/control ids. Documents are encoded to integer streams, an EOS token (`<|eot_id|>`, id 128009) is appended after every document, and the streams are packed into 50M-token shards stored as little-endian uint32. Training reads those shards back through `training/pretrain.py:PretrainDataset`, which consumes the manifest and mmaps the shards. Vocab size is a two-faced design decision: it sets how many bytes each token position covers (sequence economics) and how many parameters the embedding/head occupy (98.3M of the 501.8M total, tied). Corpus preparation lives in the sibling `LLM/shared_data/` package, invoked through the thin shim `data/prepare_data.py:main`.

## 1. Why it matters here

- **The vocabulary is the interface between data and model.** The tokenizer decides the sequence length $T$ every document becomes, and $T$ multiplies into attention FLOPs, KV-cache memory, and the number of training windows. At fixed bytes-per-document, a better tokenizer means a shorter sequence.
- **128,000 is not a free choice.** The embedding matrix is $V \times d$; with $V = 128000$ and $d = 768$ that is 98.3M parameters — about 19.6% of the 501.8M budget — before a single transformer block. Weight tying (`models/transformer.py:GPTOSS.__init__`) reuses that matrix as the output head, so the cost is paid once instead of twice.
- **The model and the data must agree on $V$.** `models/transformer.py:ModelConfig.vocab_size` defaults to 128000, and the manifest written by the pipeline carries the same 128000. If the two drift, every embedding row shifts and training silently corrupts. This chapter exists because the tokenizer is the one component that is *not* in this repo — it is loaded from the shared pipeline, and its constants must be re-verified wherever they cross the boundary.
- **Honesty boundary.** No corpus has been tokenized here yet — `.benchmarks/` is empty. "Typical bytes-per-token" figures are `[INFERENCE]` from public LLaMA-3 tokenizer statistics; what is verifiable in-repo is the config, the shard format, and the loader arithmetic.

## 2. Intuition

A tokenizer is a compressor with a fixed, finite dictionary. Think of the vocabulary as a phone book of spelling units: the 256 bytes cover every possible string (nothing is out-of-vocabulary — anything can be spelled), and each learned **merge** glues two frequently-adjacent units into one entry ("low" + "est" → "lowest"). Byte-pair encoding grows this dictionary greedily: at every step, find the pair of adjacent units that occurs most often in the corpus, fuse it, add it to the book, repeat until the book holds 128,000 entries. Encoding is then "look up the longest entry that matches the next bytes"; decoding expands each entry back to the bytes it was built from.

Why subword at all? Character-level spelling is unambiguous but long: every word becomes several positions and the model must re-learn morphology from scratch. Word-level spelling is short but brittle: a fixed list of common words cannot spell anything unseen, and realistic text has a fat tail of rare words. Subwords sit between: frequent words become single tokens (short sequences), rare words decompose into a handful of known pieces (no OOV), and morphology is shared — "un", "likely", "hood" are reused across "unlikely", "likely", "likelihood".

## 3. Theory and derivation

### 3.1 What a tokenizer does

A tokenizer is a pair of functions between text and integer sequences, anchored to a fixed vocabulary $\mathcal{V}$ of size $V$ — the set of token ids a model can emit or consume:

$$\text{encode}: \text{str} \to \{0, \ldots, V-1\}^{T}, \qquad \text{decode}: \{0, \ldots, V-1\}^{T} \to \text{str}
\tag{1}$$

The vocabulary is built from three disjoint blocks — base alphabet, learned merges, reserved specials:

$$V = V_{\text{base}} + M + S
\tag{10}$$

with $V_{\text{base}}$ the size of the base alphabet, $M$ the number of learned merges, and $S$ the number of reserved special/control tokens. For byte-level BPE, $V_{\text{base}} = 256$ (one entry per byte value). For LLaMA-3, $V = 128000$ with a reserved block of $S = 256$ special ids in the public tokenizer layout, so the learned merges number $M = 128000 - 256 - 256 = 127488$. Every token is either a single byte (which is what makes coverage universal), a merged multi-byte unit, or a special id the raw text never produces.

### 3.2 Why not word-level, why not character-level

**Compression argument.** A document of $B$ bytes becomes $T = B / \rho$ tokens, where $\rho$ is the average number of bytes per token. Byte-level spelling gives $\rho = 1$ and $T = B$ — the longest possible sequences. Word-level maximizes $\rho$ but cannot represent the corpus at all: any word outside the fixed list is unspellable. Subword BPE takes the middle — $\rho \approx 4$ on LLaMA-3-class multilingual text `[INFERENCE]` — with zero OOV because every token ultimately decomposes to bytes.

**Information-capacity view.** A position drawn from a $V$-way distribution carries at most

$$\log_2 V = \log_2 128000 \approx 16.97 \text{ bits/token}, \qquad \log_2 256 = 8 \text{ bits/byte}
\tag{2}$$

A token position can therefore carry roughly 17 bits of information versus 8 for a byte — the same information in fewer positions, and every position costs a full forward step. English text carries on the order of 0.6–1.3 bits per character of true information (the classical Shannon estimate; `[INFERENCE]` for this corpus), so even at 4 bytes/token a token stream is still redundant — but the model pays per-position compute, not per-bit, so fewer positions wins.

### 3.3 Byte-level BPE, step by step

BPE training operates on the corpus as one long token sequence $x_1, x_2, \ldots, x_{L_t}$ under the current segmentation:

1. **Initialize.** $\mathcal{V} \leftarrow$ the 256 byte values (plus any specials). The corpus is the raw byte stream; $L_0 = B$ total bytes.
2. **Count.** For every adjacent pair $(x_k, x_{k+1})$, count occurrences across the whole corpus.
3. **Merge.** Pick the highest-frequency pair, add the fused token $ab$ to $\mathcal{V}$, and replace every occurrence of the adjacent pair $a\,b$ with the single token $ab$.
4. **Repeat** steps 2–3 until $|\mathcal{V}| = V$, i.e. until $M$ merges by (2), or until the target compression is reached.

Encoding a new string runs the *same* merges greedily (longest match, left to right); decoding maps each token back to the byte string it was built from. This is the tiktoken-style lineage (LLaMA-3) as opposed to SentencePiece (LLaMA-1/2): SentencePiece trains merges over Unicode codepoints with a normalizer and a word-boundary marker, while byte-level BPE operates on raw bytes, needs no normalizer, and structurally cannot produce an out-of-vocabulary token. The HF LLaMA-3 tokenizer used here is already trained — the repo's pipeline loads it (`name: llama3`); the algorithm above describes how that vocabulary was produced, and `--train-tokenizer` exists only for the non-default HyMo path.

### 3.4 The merge criterion is frequency maximization

Define the pair frequency at step $t$ of training as

$$f_t(a, b) = \sum_{k=1}^{L_t - 1} \mathbb{1}\!\left[x_k = a \;\wedge\; x_{k+1} = b\right]
\tag{3}$$

BPE's rule is to merge the argmax pair:

$$p_t^* = \arg\max_{(a,b)} f_t(a, b)
\tag{4}$$

Why is raw frequency the right objective? Each applied merge replaces two tokens with one, shortening the sequence by exactly one token per occurrence. If no occurrences overlap, merging pair $p$ reduces the encoded length by

$$\Delta L_t(p) = f_t(p)
\tag{5}$$

so argmax-frequency maximizes the immediate compression per merge. This is the greedy heuristic for minimizing encoded length: every step takes the locally largest reduction. It is *not* globally optimal — a slightly less frequent pair could unlock a cascade of new high-frequency pairs — but it costs one counting pass per merge and is empirically strong; it is exactly what tiktoken-style BPE does.

### 3.5 Ambiguity of merges

Three sources of ambiguity mean "replace every occurrence" needs a deterministic tie-break:

- **Overlaps.** Occurrences can share tokens. In the run $a\,a\,a$, the pair $(a,a)$ occurs at positions (1,2) and (2,3), but merging left to right yields $aa\,a$ — one merge, not two. A maximal run $a^k$ has $f = k-1$ occurrences but yields only $\lfloor k/2 \rfloor$ merges, so (5) is an upper bound; the realized reduction satisfies

$$\left\lceil f_t(p)/2 \right\rceil \;\le\; \Delta L_t(p) \;\le\; f_t(p)
\tag{6}$$

- **Ties.** Two pairs with equal frequency: whichever merges first changes the downstream vocabulary, because a merge creates brand-new pairs (once $ab$ exists, $(ab, c)$ becomes count-able). Reproducibility requires a fixed order — pretrained tokenizers fix one.
- **Order dependence.** Merging $A$ before $B$ can produce a different final vocabulary than $B$ before $A$, because the pair counts at step $t+1$ depend on the segmentation produced at step $t$. BPE's greedy is a fixed schedule, which is exactly why the *same trained tokenizer* must be used for tokenization and inference: a re-trained tokenizer is a different codec.

### 3.6 Why 128,000 — coverage versus parameter cost

Vocab size trades sequence compression against parameters and softmax cost.

**Parameter cost.** The embedding matrix $E \in \mathbb{R}^{V \times d}$ and the output head $W_{\text{head}} \in \mathbb{R}^{d \times V}$ together cost

$$P_{\text{emb}} = \begin{cases} 2 V d & \text{untied} \\[4pt] V d & \text{tied} \end{cases}
\tag{7}$$

With $V = 128000$ and $d = 768$: $V d = 98304000 \approx 98.3$M parameters. Untied, that is 196.6M (two matrices); GPT-OSS-Lite ties them — `models/transformer.py:GPTOSS.__init__` sets `self.head.weight = self.embed.weight` under `models/transformer.py:ModelConfig.weight_tying` — so the cost is 98.3M, and `models/transformer.py:GPTOSS.num_parameters` deliberately excludes the duplicate. That is $98.3 / 501.8 \approx 19.6\%$ of the total parameter budget for the vocabulary alone. Doubling to $V = 256000$ would cost 196.6M at the same $d$ — roughly 39% of this model's budget — which is why a 502M-parameter model stays at 128K.

**Sequence economics.** Let $\rho$ be average bytes per token. A document of $B$ bytes runs $T = B/\rho$ positions, and attention cost per layer per sequence is $O(T^2 d)$ on full layers and $O(T W d)$ on the windowed layers ($W = 128$; [attention math](attention-and-positional.md)). Halving $T$ via a denser tokenizer quarters the full-attention cost; doubling $V$ typically buys only a modest compression gain `[INFERENCE]`, which does not pay for doubling the embedding. Decode also pays $O(V)$ per step at the head softmax ([sampling](optimizers-and-numerics.md)) — another term linear in $V$.

**Coverage.** The LLaMA-3 tokenizer was trained on roughly 30 languages plus code (public model card), which matters for this corpus mix — 70% web, 15% Python, 10% math, 5% arxiv ([data pipeline](../training.md) §Corpus Mix). Code alone needs single-token spellings of common identifiers and operators to keep $\rho$ high on the `the-stack-python` share; a small English-centric vocabulary fragments code and math into long token runs.

### 3.7 Special tokens and EOS

Special ids occupy the top of the vocabulary ($S = 256$ in (2)) and never appear as text: `<|begin_of_text|>` (128000), `<|end_of_text|>` (128001), `<|eot_id|>` (128009), and reserved slots. The pipeline uses exactly one special token: **EOS** = 128009 (`<|eot_id|>`), appended after every document. EOS plays three roles: it tells the model a document boundary — the only possible next token after a document's final word is the boundary token; it lets the packer and loader locate document boundaries with no side table; and at generation time the model can emit it to stop ([sampling](optimizers-and-numerics.md)). BOS is not used: pretraining windows are packed token streams, and the tokenize config sets `add_special_tokens: false` — the encoder adds nothing, EOS is appended manually and only after whole documents.

## 4. Code walkthrough

### 4.1 The shim — `data/prepare_data.py:main`

GPT-OSS-Lite does not implement a tokenizer or a pipeline. `data/prepare_data.py:main` resolves the workspace roots onto `sys.path`, prints an info banner from the universal config, and delegates:

```python
from shared_data.config import UNIVERSAL_TOTAL_TOKENS, load_universal_data_config
cfg = load_universal_data_config()
tok = cfg["pipeline"]["tokenizer"]
print(f"[data/gptoss] universal corpus: {UNIVERSAL_TOTAL_TOKENS:,} tokens")
print(f"[data/gptoss] tokenizer: {tok['name']} "
      f"(vocab={tok['vocab_size']:,}, EOS={tok['eos_token_id']})")

from shared_data.prepare_data import main as shared_main
return shared_main()
```

`data/prepare_data.py:main` is the single entry point; every stage flag (`--stage`, `--skip-tokenize`, `--train-tokenizer`, …) is parsed by the delegated `shared_data.prepare_data.main`. The banner is the repo's own assertion of the §3 constants: `UNIVERSAL_TOTAL_TOKENS = 8_000_000_000` and the tokenizer block are loaded from `LLM/shared_data/config.py` and `LLM/shared_data/config/data_config.yaml` (prose paths — outside this repo, not anchor-checked).

### 4.2 Tokenizer configuration

`LLM/shared_data/config/data_config.yaml` holds the pipeline defaults:

```yaml
tokenizer:
  name: llama3
  path: null                   # optional local tokenizer dir
  vocab_size: 128000
  eos_token_id: 128009         #  in LLaMA-3
  pad_token_id: 128002
  add_eos: true                # append EOS after every document
```

- `name: llama3` — the HF LLaMA-3 BPE, loaded by the sibling `shared_data/scripts/tokenize.py`; when `path` is null the HF tokenizer is used, with a tiktoken `cl100k_base` fallback that *warns* that ids and EOS will not match LLaMA-3 (test/smoke use only).
- `add_eos: true` — the tokenize stage appends id 128009 after every document; `add_special_tokens: false` in the `tokenize:` block means the HF encoder itself adds nothing.
- `vocab_size: 128000` — this must equal `models/transformer.py:ModelConfig.vocab_size` (default 128000). The model never validates the match at runtime; the failure mode is silent (§5).

The tokenize stage writes per-source streams via `TokenStream` (an 8-byte header — version uint32, eos uint32 — then the body), and `pack_shards` concatenates those streams through `ShardWriter` into the final shards.

### 4.3 Shard byte layout and manifest

Final `shard_NNNNN.bin` files are a **headerless flat stream of little-endian uint32 token ids**: document tokens with id 128009 after every document, no length table — boundaries are the EOS markers, and file size is exactly 4 bytes × token count. `ShardWriter` enforces two invariants from the `pack:` config: `cross_document_boundary_ok: false` (a document never spans two shards; writing a doc larger than the shard budget raises) and a 50M-token target per shard, flushed atomically (`.tmp` + `os.replace`) with a resume state file. Contrast with the per-source intermediate `tokens.bin`, which *does* carry the 8-byte header.

`manifest.json` records the vocabulary contract:

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
  "shards": [ { "index": 0, "path": "...", "n_tokens": ..., "sha256": "...", "n_eos": ... } ]
}
```

The pack arithmetic is closed form. With $N = 8 \times 10^9$ total tokens and $S = 5 \times 10^7$ tokens per shard,

$$K = \left\lceil \frac{N}{S} \right\rceil = \left\lceil \frac{8 \times 10^9}{5 \times 10^7} \right\rceil = 160
\tag{8}$$

Each shard is $4 \times 5 \times 10^7 = 2 \times 10^8$ bytes ≈ 190.7 MiB; the corpus is $4 \times 8 \times 10^9 = 32$ GB; and the EOS overhead is one token per document. Training windows: `training/pretrain.py:PretrainDataset._init_sharded` computes $n_{\text{win}} = \lfloor (N - 1)/L \rfloor$ with $L = 4096$:

$$n_{\text{win}} = \left\lfloor \frac{8 \times 10^9 - 1}{4096} \right\rfloor = 1953124
\tag{9}$$

Cross-check against the schedule: 61,000 steps × 131,072 tokens/step ≈ 7.995B ≈ 8.0B consumed ([training](../training.md)). The EOS overhead fraction with $D$ documents is $D/N$: at a typical ~1,500 tokens per web document `[INFERENCE]`, $D \approx 5.3$M and the overhead is ≈ 0.07% — negligible in tokens but load-bearing as the boundary marker.

### 4.4 Consumption — `training/pretrain.py:PretrainDataset`

`training/pretrain.py:PretrainDataset` turns shards into `(input_ids, target_ids)` windows of length `max_seq_len`. The constructor runs three steps: an existence check (with the "run `data/prepare_data.py` first" hint), `training/pretrain.py:PretrainDataset._load_manifest`, and layout selection — `training/pretrain.py:PretrainDataset._init_sharded` for a directory of `shard_*.bin`:

```python
m = json.loads(manifest_path.read_text())
self.eos_token_id = m.get("eos_token_id")
self.vocab_size = m.get("vocab_size")
self.total_tokens = m.get("total_tokens", 0)
self.shard_count = m.get("shard_count", 0)
self.dtype = m.get("dtype", "uint32")
```

- `training/pretrain.py:PretrainDataset._detect_format` classifies each shard by magic bytes: `PK` prefix → `torch_save`; size divisible by 4 → `raw_bytes` (the production case); otherwise → `torch_save`.
- `training/pretrain.py:PretrainDataset._init_sharded` maps `dtype` to a torch int type (`uint32` → `torch.int32`), builds cumulative `shard_offsets`, and sets `_n_samples = (total − 1) // max_seq_len` — the window count (10).
- `training/pretrain.py:PretrainDataset._get_window_sharded` bisects `shard_offsets` for the window start; if the window fits in one shard it slices the mmap'd tensor directly, otherwise it concatenates slices across shards (`torch.from_file(shared=True)` — zero-copy, no full-corpus load).
- `training/pretrain.py:PretrainDataset.__getitem__` returns `chunk[:-1], chunk[1:]` — the next-token pair consumed by the chunked cross-entropy in `training/pretrain.py:chunked_cross_entropy`.

Note the asymmetry: windows **may** cross shard boundaries (the loader stitches), but documents **never** do (the packer's invariant) — EOS alignment is preserved within every shard interior, and no window contains a doc fragment that began in another shard. That invariant is what makes (9)–(10) safe: shards are interchangeable units, and `total_tokens` in the manifest is the only number the loader trusts for the window count.

## 5. Pitfalls + verify

| Failure mode | Symptom | Guard |
|---|---|---|
| **Vocab mismatch.** Tokenizer emits ids ≥ `ModelConfig.vocab_size` (e.g. a larger vocab's ids fed to a 128K model). | Embedding lookup out of range or rows shifted; garbage loss curve. No runtime error. | Manifest `vocab_size == 128000` must equal `models/transformer.py:ModelConfig.vocab_size`. The pipeline's `validate_tokens` rejects ids ≥ vocab + 256 at write time, but the *loader* does not re-validate — check `manifest.json` before training ([data pipeline](../training.md) validation checklist). |
| **EOS mismatch.** `eos_token_id` differs between tokenize and pack (e.g. the tiktoken fallback, whose EOT is 100257). | Document boundaries mis-detected; `n_eos` counts wrong; windows stitch unrelated docs. | `DEFAULT_EOS_TOKEN_ID = 128_009` is one constant in the shared package; the fallback path warns loudly. |
| **Dtype drift.** `dtype: uint16` shards (vocab ≤ 65535 projects) read as uint32. | Token counts halve; ids corrupted by byte pairing. | Manifest `dtype` drives `raw_dtype` in `training/pretrain.py:PretrainDataset._init_sharded`; verify `manifest.dtype == "uint32"`. |
| **Re-tokenizer drift.** `--train-tokenizer` produces a different codec. | Same text → different ids; a model trained on the old ids misreads new shards. | Default pipeline never trains a tokenizer; `--train-tokenizer` is the explicit non-default HyMo path. |
| **Docs spanning shards.** `cross_document_boundary_ok: true` (config change). | Loader windows can start mid-document at a boundary; EOS alignment breaks. | Config is `false`; `ShardWriter.add` raises on oversized documents. |

**Verification.** The end-to-end guards for this chapter are the repo's data-pipeline tests:

```bash
python3 -m pytest tests/test_data_pipeline.py -v
```

This module covers token-dtype selection, `validate_tokens` bounds, `TokenStream`/`ShardWriter` round-trips (EOS placement, atomic write), manifest round-trip and validation, `PretrainDataset` reading raw-bytes shards, and a 100-document end-to-end mini pipeline. It self-skips when the sibling `shared_data` package is not importable. The loader-side smoke check after packing:

```bash
python3 -c "from training.pretrain import PretrainDataset; d=PretrainDataset('data/pretrain_chinchilla', 4096); print(len(d), d[0][0].shape)"
```

`len(d)` should read ≈ 1,953,124 — the window count derived in (10).

## References

- [`training/pretrain.py:PretrainDataset`](../../training/pretrain.py) — shard consumer
- [`data/prepare_data.py`](../../data/prepare_data.py) — pipeline shim
- [training.md](../training.md) — shard format, manifest, DataLoader configuration
- [optimizers-and-numerics.md](optimizers-and-numerics.md) — softmax over the 128K vocabulary

<!-- docs:verified 2026-08-05 · 6491066 -->
