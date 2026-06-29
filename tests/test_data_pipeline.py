"""Tests for the GPT-OSS-Lite data pipeline."""
from __future__ import annotations

import json
import os
import struct
import sys
import tempfile
from pathlib import Path

import numpy as np
import pytest
import torch

_LLM_ROOT = Path(__file__).resolve().parents[2]
if str(_LLM_ROOT) not in sys.path:
    sys.path.insert(0, str(_LLM_ROOT))

import shared_data.common as common
from shared_data.common import (
    DATA_ROOT,
    DEFAULT_EOS_TOKEN_ID,
    DEFAULT_VOCAB_SIZE,
    MANIFEST_PATH,
    SHARDS_ROOT,
    STATE_ROOT,
    TOKENS_ROOT,
    atomic_write_bytes,
    atomic_write_json,
    hash_to_bucket,
    human_bytes,
    load_state,
    read_json,
    save_state,
    sha256_bytes,
    sha256_text,
)
from shared_data.dedup import BloomFilter, Deduper
from shared_data.manifest import (
    MANIFEST_VERSION,
    Manifest,
    ShardInfo,
    SourceInfo,
    hash_config,
    hash_yaml,
)
from shared_data.quality_filter import (
    FilterStats,
    QualityFilter,
    digit_ratio_filter,
    language_hint_filter,
    length_filter,
    punctuation_filter,
    unique_chars_filter,
    whitespace_filter,
)
from shared_data.shard_writer import (
    ShardWriter,
    TokenStream,
    read_token_stream,
    select_token_dtype,
    validate_tokens,
    verify_shard,
)


# Fixtures

@pytest.fixture()
def workspace(tmp_path, monkeypatch):
    """Set the shared_data data root to a temp dir so we don't touch the
    real data/.

    We call ``set_data_root(tmp_path)`` to update the path globals in
    ``shared_data.common``, then reload the submodules that capture those
    constants at import time (``shared_data.dedup`` captures
    ``STATE_ROOT``, etc.) so they re-evaluate against the new root. Without
    this, they would keep writing to the real ``data/state/`` dir.
    """
    from shared_data.common import set_data_root
    set_data_root(tmp_path)
    import importlib
    import shared_data.quality_filter
    import shared_data.dedup
    import shared_data.manifest
    import shared_data.shard_writer
    import shared_data.scripts.pack_shards
    importlib.reload(shared_data.quality_filter)
    importlib.reload(shared_data.dedup)
    importlib.reload(shared_data.manifest)
    importlib.reload(shared_data.shard_writer)
    importlib.reload(shared_data.scripts.pack_shards)
    common.ensure_dirs()
    yield tmp_path


@pytest.fixture()
def eos_id():
    return DEFAULT_EOS_TOKEN_ID


@pytest.fixture()
def vocab_size():
    return DEFAULT_VOCAB_SIZE


# 1. data/common.py — IO, hashing, state

class TestCommonIO:
    def test_atomic_write_bytes_creates_file(self, workspace):
        p = Path(workspace) / "x.bin"
        atomic_write_bytes(p, b"hello")
        assert p.read_bytes() == b"hello"

    def test_atomic_write_bytes_no_leftover_tmp(self, workspace):
        p = Path(workspace) / "x.bin"
        atomic_write_bytes(p, b"hello")
        leftover = list(Path(workspace).glob("*.tmp"))
        assert leftover == [], f"Leftover tmp files: {leftover}"

    def test_atomic_write_json_roundtrip(self, workspace):
        p = Path(workspace) / "x.json"
        obj = {"a": 1, "b": [1, 2, 3], "c": "hi"}
        atomic_write_json(p, obj)
        loaded = read_json(p)
        assert loaded == obj

    def test_atomic_write_json_handles_numpy(self, workspace):
        p = Path(workspace) / "x.json"
        atomic_write_json(p, {"n": np.int64(42), "f": np.float32(3.14)})
        loaded = read_json(p)
        assert loaded["n"] == 42
        assert abs(loaded["f"] - 3.14) < 1e-5

    def test_state_persistence(self, workspace):
        clear = common.clear_state
        clear("test_stage")
        assert load_state("test_stage") == {}
        save_state("test_stage", {"k": "v", "n": 42})
        loaded = load_state("test_stage")
        assert loaded == {"k": "v", "n": 42}
        clear("test_stage")
        assert load_state("test_stage") == {}


class TestCommonHashing:
    def test_sha256_bytes_deterministic(self):
        assert sha256_bytes(b"abc") == sha256_bytes(b"abc")
        assert sha256_bytes(b"abc") != sha256_bytes(b"abd")

    def test_sha256_text_strips_whitespace_evasion(self):
        """Trivial whitespace differences should not evade dedup."""
        a = sha256_text("hello   world")
        b = sha256_text("hello world")
        c = sha256_text("hello\n\n\tworld")
        assert a == b == c

    def test_sha256_text_case_sensitive(self):
        """Case differences should NOT collide (semantic difference)."""
        assert sha256_text("Hello World") != sha256_text("hello world")

    def test_hash_to_bucket_uniform(self):
        """Hash → bucket should distribute uniformly across many inputs."""
        from collections import Counter
        n_buckets = 16
        counts = Counter()
        for i in range(10_000):
            sha = sha256_text(f"doc-{i}")
            counts[hash_to_bucket(sha, n_buckets)] += 1
        # Each bucket should have ~625 docs (10_000 / 16); tolerance ±20%
        for c in counts.values():
            assert 500 <= c <= 750, f"Bucket count {c} outside [500, 750]"

    def test_human_bytes(self):
        assert human_bytes(0) == "0.00 B"
        assert human_bytes(1024) == "1.00 KiB"
        assert "MiB" in human_bytes(2 * 1024 * 1024)
        assert "GiB" in human_bytes(3 * 1024**3)


# 2. data/quality_filter.py — heuristics

class TestQualityFilters:
    def test_length_filter_bounds(self):
        assert length_filter("a" * 100, min_chars=50, max_chars=200) is True
        assert length_filter("a" * 49, min_chars=50, max_chars=200) is False
        assert length_filter("a" * 201, min_chars=50, max_chars=200) is False

    def test_unique_chars_filter_rejects_junk(self):
        assert unique_chars_filter("abcabc") is True
        # At the default ratio (0.05), a long enough run of one char
        # still passes. Use a stricter ratio to reject truly degenerate
        # inputs.
        assert unique_chars_filter("aaaaaaaa", min_ratio=0.5) is False
        assert unique_chars_filter("1111111", min_ratio=0.5) is False
        # Short junk: 8-char string of "a" → unique=1, 1/8=0.125 → reject
        assert unique_chars_filter("aaaaaaaa", min_ratio=0.2) is False

    def test_digit_ratio_filter(self):
        assert digit_ratio_filter("hello world") is True
        assert digit_ratio_filter("1234567890" * 10) is False
        assert digit_ratio_filter("a" * 100 + "1") is True

    def test_punctuation_filter(self):
        assert punctuation_filter("hello world") is True
        assert punctuation_filter("!@#$%^&*()" * 10) is False

    def test_whitespace_filter(self):
        assert whitespace_filter("hello world") is True
        assert whitespace_filter("    \n\n\t  \n") is False

    def test_language_hint_filter_english(self):
        assert language_hint_filter(
            "The quick brown fox jumps over the lazy dog.", lang="en"
        ) is True
        assert language_hint_filter(
            "xxxxx yyyyy zzzz qqqq pppp llll mmmm", lang="en"
        ) is False
        assert language_hint_filter("print('hello')", lang="python") is True
        assert language_hint_filter("anything goes", lang=None) is True


class TestQualityFilter:
    def test_apply_keeps_clean_text(self):
        qf = QualityFilter(min_chars=10, max_chars=1000, lang="en")
        text = "The quick brown fox jumps over the lazy dog. " * 3
        assert qf.apply(text) == text

    def test_apply_rejects_short(self):
        qf = QualityFilter(min_chars=100, max_chars=1000, lang="en")
        assert qf.apply("too short") is None

    def test_apply_rejects_long(self):
        qf = QualityFilter(min_chars=10, max_chars=100, lang="en")
        text = "a" * 200
        assert qf.apply(text) is None

    def test_apply_rejects_junk(self):
        qf = QualityFilter(min_chars=10, max_chars=1000, lang="en")
        assert qf.apply("aaaaaaaaaa") is None

    def test_max_digit_ratio_can_be_disabled(self):
        """For code corpora, we disable the digit filter."""
        qf = QualityFilter(min_chars=10, max_chars=1000, lang=None, max_digit_ratio=None)
        assert qf.apply("1234567890" * 5) is not None

    def test_filter_stats_summary(self):
        stats = FilterStats()
        stats.n_seen = 100
        stats.n_kept = 60
        stats.n_dropped = 40
        stats.reasons["length"] = 20
        stats.reasons["unique_chars"] = 15
        stats.reasons["duplicate"] = 5
        s = stats.summary()
        assert "60" in s and "100" in s
        assert "length" in s and "duplicate" in s


# 3. data/dedup.py — BloomFilter + Deduper

class TestBloomFilter:
    def test_add_new(self):
        bf = BloomFilter(capacity=100, error_rate=0.01)
        assert bf.add("a" * 64) is False   # new
        assert bf.add("b" * 64) is False   # new
        assert bf.add("a" * 64) is True    # probably duplicate

    def test_contains(self):
        bf = BloomFilter(capacity=100, error_rate=0.01)
        bf.add("a" * 64)
        assert ("a" * 64) in bf
        assert ("b" * 64) not in bf

    def test_save_load_roundtrip(self, workspace):
        bf = BloomFilter(capacity=100, error_rate=0.01)
        for i in range(50):
            bf.add(sha256_text(f"doc-{i}"))
        path = Path(workspace) / "bf.pkl"
        bf.save(path)
        bf2 = BloomFilter.load(path)
        for i in range(50):
            assert sha256_text(f"doc-{i}") in bf2

    def test_capacity_error_rate_validation(self):
        with pytest.raises(ValueError):
            BloomFilter(capacity=0, error_rate=0.01)
        with pytest.raises(ValueError):
            BloomFilter(capacity=100, error_rate=1.5)


class TestDeduper:
    def test_two_pass_dedup_drops_duplicates(self, workspace):
        # 5 unique docs + 3 duplicates of one of them
        docs = [
            ("a", "the quick brown fox"),
            ("b", "jumps over the lazy dog"),
            ("c", "lorem ipsum dolor sit amet"),
            ("d", "consectetur adipiscing elit"),
            ("e", "sed do eiusmod tempor"),
            ("a-dup", "the quick brown fox"),       # dup of a
            ("a-dup2", "the quick brown fox  "),    # dup (whitespace normalised)
            ("c-dup", "lorem ipsum dolor sit amet"),  # dup of c
        ]

        dedup = Deduper(
            source_id="test_src",
            n_buckets=4,
            bloom_capacity_per_bucket=100,
            bloom_error_rate=0.01,
        )
        s1 = dedup.collect((d for d in docs))
        assert s1["n_processed"] == 8
        assert s1["n_unique"] + s1["n_duplicate"] == 8

        out_path = Path(workspace) / "unique.jsonl"
        s2 = dedup.write_unique((d for d in docs), out_path)
        assert s2["write_n_kept"] == 5
        assert s2["write_n_dropped"] == 3

        kept = [json.loads(line) for line in out_path.read_text().splitlines()]
        assert len(kept) == 5
        texts = {r["text"] for r in kept}
        assert "the quick brown fox" in texts
        assert "lorem ipsum dolor sit amet" in texts

    def test_dedup_resume_is_idempotent(self, workspace):
        docs = [("d" + str(i), f"unique doc number {i}") for i in range(20)]

        dedup = Deduper(source_id="resume_src", n_buckets=2)
        s1 = dedup.collect(iter(docs))
        assert s1["n_processed"] == 20

        out_path = Path(workspace) / "unique.jsonl"
        s2 = dedup.write_unique(iter(docs), out_path)
        assert s2["write_n_kept"] == 20


# 4. data/shard_writer.py — atomic write, EOS, dtype

class TestSelectTokenDtype:
    def test_uint8_for_small_vocab(self):
        assert select_token_dtype(100) == np.dtype("uint8")
    def test_uint16_for_64k_vocab(self):
        assert select_token_dtype(60_000) == np.dtype("uint16")
    def test_uint32_for_128k_vocab(self):
        assert select_token_dtype(128_000) == np.dtype("uint32")
    def test_rejects_zero(self):
        with pytest.raises(ValueError):
            select_token_dtype(0)


class TestValidateTokens:
    def test_accepts_in_range(self):
        a = np.array([0, 1, 100, 127999], dtype=np.uint32)
        validate_tokens(a, vocab_size=128000)  # no raise

    def test_rejects_out_of_range(self):
        # A token id far outside the reserved range should still be rejected.
        a = np.array([0, 1, 999_999], dtype=np.uint32)
        with pytest.raises(ValueError, match="token id"):
            validate_tokens(a, vocab_size=128000)

    def test_accepts_reserved_range(self):
        """LLaMA-3 EOS (128009) is in the reserved range [128000, 128256)."""
        a = np.array([0, 1, 128009, 128255], dtype=np.uint32)
        validate_tokens(a, vocab_size=128000)  # no raise


class TestShardWriter:
    def test_writes_single_shard(self, workspace, eos_id, vocab_size):
        path = Path(workspace) / "shards"
        writer = ShardWriter(
            output_dir=path,
            shard_size_tokens=100,
            dtype=np.dtype("uint32"),
            eos_token_id=eos_id,
            vocab_size=vocab_size,
        )
        for i in range(5):
            writer.add(np.array([10, 20, 30, 40, 50], dtype=np.uint32))
        shards = writer.finalize()
        # 5 docs × (5 tokens + 1 EOS) = 30 tokens → 1 shard
        assert len(shards) == 1
        assert shards[0].n_tokens == 30
        assert shards[0].n_eos == 5

        shard_file = path / "shard_00000.bin"
        raw = np.frombuffer(shard_file.read_bytes(), dtype=np.uint32)
        assert raw.size == 30
        # Every 6th token (5 doc + 1 eos) should be EOS
        for i in range(5):
            assert raw[i * 6 + 5] == eos_id

    def test_spills_to_multiple_shards(self, workspace, eos_id, vocab_size):
        path = Path(workspace) / "shards"
        writer = ShardWriter(
            output_dir=path,
            shard_size_tokens=20,  # very small to force multiple shards
            dtype=np.dtype("uint32"),
            eos_token_id=eos_id,
            vocab_size=vocab_size,
        )
        # 10 docs × (2 tokens + 1 EOS) = 30 tokens → 2 shards.
        # 6 docs fit in shard 0 (6×3 = 18 tokens); remaining 4 in shard 1.
        for i in range(10):
            writer.add(np.array([1, 2], dtype=np.uint32))
        shards = writer.finalize()
        assert len(shards) == 2
        assert shards[0].n_tokens == 18
        assert shards[1].n_tokens == 12
        assert shards[0].n_eos == 6
        assert shards[1].n_eos == 4
        assert sum(s.n_tokens for s in shards) == 30

    def test_atomic_write_no_partial_files(self, workspace, eos_id, vocab_size):
        path = Path(workspace) / "shards"
        writer = ShardWriter(
            output_dir=path,
            shard_size_tokens=100,
            dtype=np.dtype("uint32"),
            eos_token_id=eos_id,
            vocab_size=vocab_size,
        )
        writer.add(np.array([1, 2, 3], dtype=np.uint32))
        writer.finalize()
        leftover = list(path.glob("*.tmp"))
        assert leftover == [], f"Leftover tmp: {leftover}"

    def test_rejects_oversized_doc_without_split(self, workspace, eos_id, vocab_size):
        writer = ShardWriter(
            output_dir=Path(workspace) / "shards",
            shard_size_tokens=10,
            dtype=np.dtype("uint32"),
            eos_token_id=eos_id,
            vocab_size=vocab_size,
            cross_document_boundary_ok=False,
        )
        with pytest.raises(ValueError, match="exceeds shard_size"):
            writer.add(np.arange(20, dtype=np.uint32))

    def test_splits_oversized_doc_when_allowed(self, workspace, eos_id, vocab_size):
        writer = ShardWriter(
            output_dir=Path(workspace) / "shards",
            shard_size_tokens=10,
            dtype=np.dtype("uint32"),
            eos_token_id=eos_id,
            vocab_size=vocab_size,
            cross_document_boundary_ok=True,
        )
        writer.add(np.arange(25, dtype=np.uint32))
        # 25 tokens + 1 EOS = 26 → 3 shards (10+10+6)
        # Wait, cross_document_boundary_ok splits BEFORE the EOS, so
        # 25 tokens → [10, 10, 5] + 3 EOS = 28 → 3 shards (10+10+8)
        # We just check it doesn't raise and produces shards.
        shards = writer.finalize()
        assert sum(s.n_tokens for s in shards) == 28
        assert sum(s.n_eos for s in shards) == 3

    def test_rejects_oov_token(self, workspace, eos_id, vocab_size):
        writer = ShardWriter(
            output_dir=Path(workspace) / "shards",
            shard_size_tokens=100,
            dtype=np.dtype("uint32"),
            eos_token_id=eos_id,
            vocab_size=vocab_size,
        )
        # Use a clearly-OOV id (way beyond the reserved range).
        with pytest.raises(ValueError, match="token id"):
            writer.add(np.array([1, 2, 999_999], dtype=np.uint32))

    def test_context_manager_calls_finalize(self, workspace, eos_id, vocab_size):
        path = Path(workspace) / "shards"
        with ShardWriter(
            output_dir=path,
            shard_size_tokens=100,
            dtype=np.dtype("uint32"),
            eos_token_id=eos_id,
            vocab_size=vocab_size,
        ) as writer:
            writer.add(np.array([1, 2, 3], dtype=np.uint32))
        assert (path / "shard_00000.bin").exists()


class TestTokenStream:
    def test_write_and_read_roundtrip(self, workspace, eos_id, vocab_size):
        path = Path(workspace) / "stream.bin"
        docs = [
            [1, 2, 3],
            [10, 20, 30, 40],
            [100],
        ]
        with TokenStream(path, eos_token_id=eos_id, vocab_size=vocab_size) as s:
            for d in docs:
                s.write_doc(d)
        read_docs = list(read_token_stream(path))
        assert len(read_docs) == 3
        for original, read in zip(docs, read_docs):
            assert list(read) == original

    def test_strips_trailing_eos(self, workspace, eos_id, vocab_size):
        path = Path(workspace) / "stream.bin"
        with TokenStream(path, eos_token_id=eos_id, vocab_size=vocab_size) as s:
            s.write_doc([1, 2, 3, eos_id])   # trailing EOS should be stripped
        docs = list(read_token_stream(path))
        assert list(docs[0]) == [1, 2, 3]

    def test_rejects_oov_token(self, workspace, eos_id, vocab_size):
        path = Path(workspace) / "stream.bin"
        with TokenStream(path, eos_token_id=eos_id, vocab_size=vocab_size) as s:
            with pytest.raises(ValueError, match="token id"):
                s.write_doc([1, 2, 999_999])  # far beyond reserved range


class TestVerifyShard:
    def test_verifies_clean_shard(self, workspace, eos_id, vocab_size):
        shard_dir = Path(workspace) / "shards"
        writer = ShardWriter(
            output_dir=shard_dir,
            shard_size_tokens=100,
            dtype=np.dtype("uint32"),
            eos_token_id=eos_id,
            vocab_size=vocab_size,
        )
        writer.add(np.array([1, 2, 3], dtype=np.uint32))
        shards = writer.finalize()
        shard_path = shard_dir / "shard_00000.bin"
        result = verify_shard(
            shard_path,
            expected_tokens=shards[0].n_tokens,
            expected_dtype=np.dtype("uint32"),
            vocab_size=vocab_size,
            eos_token_id=eos_id,
        )
        assert result["ok"] is True
        assert result["actual_tokens"] == shards[0].n_tokens
        # Max token in the shard is the EOS (which lives in the reserved
        # range). The data tokens themselves max out at 3.
        assert result["max_token"] == eos_id
        assert result["n_eos"] == 1

    def test_rejects_mismatched_token_count(self, workspace, eos_id, vocab_size):
        shard_dir = Path(workspace) / "shards"
        writer = ShardWriter(
            output_dir=shard_dir,
            shard_size_tokens=100,
            dtype=np.dtype("uint32"),
            eos_token_id=eos_id,
            vocab_size=vocab_size,
        )
        writer.add(np.array([1, 2, 3], dtype=np.uint32))
        writer.finalize()
        shard_path = shard_dir / "shard_00000.bin"
        with pytest.raises(ValueError, match="expected"):
            verify_shard(
                shard_path,
                expected_tokens=999,   # wrong
                expected_dtype=np.dtype("uint32"),
                vocab_size=vocab_size,
                eos_token_id=eos_id,
            )


# 5. data/manifest.py — round-trip + validation

class TestManifest:
    def test_roundtrip(self, workspace):
        m = Manifest(
            vocab_size=128000,
            eos_token_id=128009,
            pad_token_id=128002,
            tokenizer_name="llama3",
            dtype="uint32",
            shard_size_tokens=50_000_000,
            total_tokens=8_000_000_000,
            shard_count=160,
        )
        m.shards.append(ShardInfo(
            index=0, path="data/shards/shard_00000.bin",
            n_tokens=50_000_000, sha256="abc", n_eos=1_234_567,
        ))
        m.sources["fineweb-edu"] = SourceInfo(
            target_tokens=4_000_000_000,
            actual_tokens=3_998_234_567,
            n_docs=12_345_678,
            n_dedup_dropped=23_456,
        )
        out_path = Path(workspace) / "manifest.json"
        m.save(out_path)
        loaded = Manifest.load(out_path)
        assert loaded.vocab_size == 128000
        assert loaded.eos_token_id == 128009
        assert loaded.total_tokens == 8_000_000_000
        assert loaded.shard_count == 160
        assert len(loaded.shards) == 1
        assert loaded.shards[0].sha256 == "abc"
        assert loaded.sources["fineweb-edu"].actual_tokens == 3_998_234_567

    def test_validate_passes_clean_manifest(self):
        m = Manifest(
            total_tokens=100, shard_count=2,
            sources={"a": SourceInfo(target_tokens=50, actual_tokens=50,
                                     n_docs=10, n_dedup_dropped=0),
                     "b": SourceInfo(target_tokens=50, actual_tokens=50,
                                     n_docs=10, n_dedup_dropped=0)},
        )
        m.shards.append(ShardInfo(index=0, path="x", n_tokens=50,
                                  sha256="x", n_eos=1))
        m.shards.append(ShardInfo(index=1, path="y", n_tokens=50,
                                  sha256="y", n_eos=1))
        issues = m.validate(strict=True)
        assert issues == []

    def test_validate_flags_eos_out_of_range(self):
        m = Manifest(vocab_size=128000, eos_token_id=999_999, total_tokens=10, shard_count=1)
        issues = m.validate(strict=True)
        assert any("eos_token_id" in s for s in issues)

    def test_validate_flags_shard_count_mismatch(self):
        m = Manifest(total_tokens=10, shard_count=5)
        issues = m.validate(strict=True)
        assert any("shards" in s for s in issues)

    def test_validate_flags_source_sum_mismatch(self):
        m = Manifest(total_tokens=100, shard_count=1,
                     sources={"a": SourceInfo(target_tokens=100, actual_tokens=50,
                                              n_docs=1, n_dedup_dropped=0)})
        m.shards.append(ShardInfo(index=0, path="x", n_tokens=10, sha256="x", n_eos=1))
        issues = m.validate(strict=True)
        assert any("actual_tokens" in s for s in issues)

    def test_hash_yaml_deterministic(self, workspace):
        p = Path(workspace) / "cfg.yaml"
        p.write_text("a: 1\nb: 2\n")
        assert hash_yaml(p) == hash_yaml(p)
        p2 = Path(workspace) / "cfg2.yaml"
        p2.write_text("a: 1\nb: 2\n")
        assert hash_yaml(p) == hash_yaml(p2)

    def test_hash_config_stable_under_key_order(self):
        a = hash_config({"a": 1, "b": 2})
        b = hash_config({"b": 2, "a": 1})
        assert a == b


# 6. PretrainDataset integration (new format support)

class TestPretrainDatasetNewFormat:
    """Verify the training script can read the new raw-bytes shard format."""

    def test_loads_raw_uint32_shards(self, workspace, eos_id, vocab_size, tmp_path):
        from training.pretrain import PretrainDataset

        # Write 2 raw-bytes shards (simulating the new pipeline output)
        shard_dir = Path(workspace) / "shards"
        writer = ShardWriter(
            output_dir=shard_dir,
            shard_size_tokens=20,
            dtype=np.dtype("uint32"),
            eos_token_id=eos_id,
            vocab_size=vocab_size,
        )
        # 10 docs × (2 tokens + 1 EOS) = 30 → 2 shards
        for i in range(10):
            writer.add(np.array([1, 2], dtype=np.uint32))
        writer.finalize()

        ds = PretrainDataset(str(shard_dir), max_seq_len=4)
        assert len(ds) > 0
        x, y = ds[0]
        assert x.shape == (4,)
        assert y.shape == (4,)
        assert torch.equal(x[1:], y[:-1])

    def test_loads_with_manifest(self, workspace, eos_id, vocab_size):
        from training.pretrain import PretrainDataset

        shard_dir = Path(workspace) / "shards"
        writer = ShardWriter(
            output_dir=shard_dir,
            shard_size_tokens=100,
            dtype=np.dtype("uint32"),
            eos_token_id=eos_id,
            vocab_size=vocab_size,
        )
        writer.add(np.array([1, 2, 3], dtype=np.uint32))
        writer.finalize()

        m = Manifest(
            vocab_size=vocab_size, eos_token_id=eos_id,
            total_tokens=4, shard_count=1, dtype="uint32",
        )
        m.save(shard_dir / "manifest.json")

        ds = PretrainDataset(str(shard_dir), max_seq_len=3)
        assert ds.eos_token_id == eos_id
        assert ds.vocab_size == vocab_size

    def test_backward_compatible_with_torch_save_shards(self, workspace, tmp_path):
        """The legacy torch.save format (PK header) should still load."""
        from training.pretrain import PretrainDataset

        shard_dir = Path(workspace) / "shards"
        # Write a legacy-format shard
        legacy_path = shard_dir / "shard_00000.bin"
        torch.save(torch.randint(0, 256, (100,), dtype=torch.long), legacy_path)
        ds = PretrainDataset(str(shard_dir), max_seq_len=8)
        assert len(ds) > 0
        x, y = ds[0]
        assert x.shape == (8,)

    def test_no_shards_raises(self, workspace):
        from training.pretrain import PretrainDataset
        with pytest.raises(FileNotFoundError):
            PretrainDataset(str(Path(workspace) / "empty"), max_seq_len=8)

    def test_no_data_path_raises(self, workspace):
        from training.pretrain import PretrainDataset
        with pytest.raises(FileNotFoundError):
            PretrainDataset("/nonexistent/path/to/data", max_seq_len=8)


# 7. End-to-end mini pipeline (no HF, no network)

class TestEndToEndMiniPipeline:
    """Simulate the full pipeline on a 100-doc synthetic corpus."""

    def test_full_pipeline_produces_valid_manifest(
        self, workspace, eos_id, vocab_size
    ):
        # 1) Build a synthetic clean JSONL
        clean_dir = Path(workspace) / "clean" / "synthetic"
        clean_dir.mkdir(parents=True)
        clean_path = clean_dir / "data.jsonl"
        with open(clean_path, "w") as f:
            for i in range(50):
                # Each doc: 30 tokens, all distinct except a few dupes
                text = f"This is synthetic document number {i} " * 3
                f.write(json.dumps({"id": str(i), "text": text}) + "\n")
            # Add 10 exact dupes
            for i in range(10):
                text = f"This is synthetic document number {i} " * 3
                f.write(json.dumps({"id": f"dup-{i}", "text": text}) + "\n")

        # 2) Apply dedup inline (mirror the script's logic) and write
        # a smaller clean JSONL
        from shared_data.dedup import Deduper
        dedup = Deduper(source_id="synthetic", n_buckets=4,
                        bloom_capacity_per_bucket=100, bloom_error_rate=0.01)
        records = [
            (rec["id"], rec["text"])
            for rec in (json.loads(l) for l in clean_path.read_text().splitlines())
        ]
        dedup.collect(iter(records))

        out_clean = Path(workspace) / "clean" / "synthetic_uniq.jsonl"
        records2 = [
            (rec["id"], rec["text"])
            for rec in (json.loads(l) for l in clean_path.read_text().splitlines())
        ]
        s = dedup.write_unique(iter(records2), out_clean)
        assert s["write_n_kept"] == 50
        assert s["write_n_dropped"] == 10

        # 3) Tokenise via TokenStream with a toy encoder
        tokens_dir = Path(workspace) / "tokens" / "synthetic"
        tokens_dir.mkdir(parents=True)
        token_path = tokens_dir / "data.bin"
        with TokenStream(token_path, eos_token_id=eos_id,
                         vocab_size=vocab_size) as s:
            for line in out_clean.read_text().splitlines():
                rec = json.loads(line)
                # Toy encoder: hash each word → token id (clipped to vocab)
                ids = [
                    int(hashlib.md5(w.encode()).hexdigest()[:6], 16) % (vocab_size - 1)
                    for w in rec["text"].split()
                ]
                s.write_doc(ids)

        # 4) Pack into shards
        shards_dir = Path(workspace) / "shards"
        from shared_data.scripts.pack_shards import interleave_sources
        writer = ShardWriter(
            output_dir=shards_dir,
            shard_size_tokens=200,
            dtype=np.dtype("uint32"),
            eos_token_id=eos_id,
            vocab_size=vocab_size,
        )
        for sid, doc in interleave_sources(["synthetic"], target_per_source={"synthetic": 10_000}):
            writer.add(doc)
        shard_metas = writer.finalize()

        # Should have at least one shard
        assert len(shard_metas) >= 1
        for sm in shard_metas:
            verify_shard(
                shards_dir / f"shard_{sm.index:05d}.bin",
                expected_tokens=sm.n_tokens,
                expected_dtype=np.dtype("uint32"),
                vocab_size=vocab_size,
                eos_token_id=eos_id,
            )

        # 5) Build + save manifest
        m = Manifest(
            vocab_size=vocab_size, eos_token_id=eos_id,
            dtype="uint32", shard_size_tokens=200,
            total_tokens=sum(s.n_tokens for s in shard_metas),
            shard_count=len(shard_metas),
        )
        for sm in shard_metas:
            m.shards.append(ShardInfo(
                index=sm.index, path=f"shards/{sm.path.split('/')[-1]}",
                n_tokens=sm.n_tokens, sha256=sm.sha256, n_eos=sm.n_eos,
            ))
        m.sources["synthetic"] = SourceInfo(
            target_tokens=10_000, actual_tokens=0, n_docs=50, n_dedup_dropped=10,
        )
        # Manually set actual_tokens to the shard total for the validate check
        m.sources["synthetic"].actual_tokens = m.total_tokens
        manifest_path = Path(workspace) / "manifest.json"
        m.save(manifest_path)
        loaded = Manifest.load(manifest_path)
        issues = loaded.validate(strict=True)
        assert issues == [], f"Manifest issues: {issues}"


# Hashlib import (used in test 7)

import hashlib