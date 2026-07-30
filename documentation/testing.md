# Testing — Test Corpus as a Learning Tool

> **Covers:** The `tests/` suite — how to use it to verify correctness and learn system invariants.

---

## Table of Contents

1. [Philosophy](#philosophy)
2. [Running Tests](#running-tests)
3. [Fixtures (conftest.py)](#fixtures-conftestpy)
4. [Suite Overview](#suite-overview)
5. [Load-Bearing Tests](#load-bearing-tests)
6. [Adding New Tests](#adding-new-tests)

---

## Philosophy

Every test runs on **CPU without Triton** unless marked `@pytest.mark.gpu`. GPU tests auto-skip on CPU-only machines.

**Tests are documentation.** When in doubt about an invariant, search `tests/`.

| Category | What it proves | Example |
|---|---|---|
| Shape | Graph wiring | `test_forward_shape_small` |
| Equivalence | Two paths agree | `test_sliding_window_matches_full_small` |
| Invariant | Architectural rule | `test_sink_bias_clamped_at_forward` |
| Anchor | Headline numbers | `test_anchor_metric_502m_total` |
| Guard | Safety mechanism | NaN guard tests in `test_training.py` |

---

## Running Tests

```bash
# Full suite
python3 -m pytest tests/ -q

# After attention changes
python3 -m pytest tests/test_attention.py -v

# Anchor metrics only
python3 -m pytest tests/test_validation.py -k anchor -v

# MoE + Triton
python3 -m pytest tests/test_moe.py tests/test_moe_triton.py -v
```

Expected: **187 passed** (~40s CPU).

---

## Fixtures (conftest.py)

| Fixture | Purpose |
|---|---|
| `attn_small`, `attn_tiny`, `attn_large` | Attention dim bundles |
| `yarn_cfg`, `yarn_cfg_small` | YaRN parameters |
| `small_cfg` | Tiny model for fast tests |
| `model_cfg` | Production 502M config |
| `tmp_shard_dir` | Temp data shards |

---

## Suite Overview

| File | Tests | Primary coverage |
|---|---:|---|
| `test_attention.py` | 18 | SWA equivalence, sink bias, clamp |
| `test_data_pipeline.py` | 53 | Shard format, manifest, mmap |
| `test_inference.py` | 14 | KV cache, generation, passkey prompts |
| `test_models.py` | 14 | Forward shape, param count, grad flow |
| `test_moe_triton.py` | 9 | Triton reference + opt-in guards |
| `test_moe.py` | 16 | Routing, aux loss, dispatch |
| `test_smoke.py` | 4 | Import + minimal forward |
| `test_training.py` | 12 | LR schedule, NaN guard, checkpoints |
| `test_utils.py` | 9 | Checkpoint, memory, logging |
| `test_validation.py` | 23 | ModelConfig validation, anchor metrics |
| `test_yarn.py` | 15 | YaRN freqs, RoPE magnitude |
| **Total** | **187** | |

---

## Load-Bearing Tests

| Test | Invariant | File |
|---|---|---|
| `test_sliding_window_matches_full_small` | SWA SDPA = reference | `test_attention.py` |
| `test_sink_bias_clamped_at_forward` | BF16 overflow guard | `test_attention.py` |
| `test_anchor_metric_502m_total` | ~502M params | `test_validation.py` |
| `test_anchor_metric_247m_active` | ~247M active | `test_validation.py` |
| `test_MoELayer_default_moe_dispatch_is_stacked` | No silent Triton | `test_moe_triton.py` |
| `test_kv_cache_windowed_preserves_order_after_rollover` | Ring buffer correctness | `test_inference.py` |
| `test_mixed_kv_cache_correct` | Memory estimator KV term | `test_utils.py` |
| `test_compute_yarn_freqs_warns_on_degenerate_ramp` | YaRN degenerate ramp | `test_yarn.py` |

**After any change to `models/attention.py`:** run `pytest tests/test_attention.py -v` (AGENTS.md rule 4).

---

## Adding New Tests

1. Follow naming: `test_<component>_<behaviour>`.
2. CPU-runnable by default; gate GPU with `@pytest.mark.gpu`.
3. For new Triton paths: pure-PyTorch reference test required (AGENTS.md rule 9).
4. Update this file's suite table if adding a new test file.

<!-- docs:verified 2026-07-31 · fd4fe36 -->
