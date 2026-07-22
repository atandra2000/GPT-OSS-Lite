# Design: Implement Ponytail-Audit Findings (GPT-OSS-Lite)

## 1. Problem

The ponytail-audit surfaced ~25 over-engineering findings across the
GPT-OSS-Lite repo. Net: -600 lines, -3 deps possible, but high risk
because the edits span the model package, scripts, tests, and config.

## 2. Goals

- Apply every audit finding except any new test framework or external API
  (no `pyproject.toml` package install — use `sys.path` cleanup only).
- Keep `pytest tests/` passing at every checkpoint (no broken tests at
  the end, no broken `import` graph mid-flight).
- Preserve all public symbols the existing test suite imports.
- Preserve all behavior — this is a structural cleanup, not a refactor.

## 3. Non-goals

- No new features, no correctness fixes, no perf wins.
- No `pyproject.toml` (out of scope; user can `pip install -e .` later
  if they want). The sys.path appends in scripts will be replaced with
  a single shared `scripts/_bootstrap.py` instead.
- No rewrite of the model — only surface-area cleanup.

## 4. Findings grouped by file (with line counts from audit)

### 4.1 models/rotary.py (-25 lines)
- **delete** `prune_rope` function (lines 81-90). No production caller.
- **shrink** `compute_yarn_freqs`: drop `original_max` / `target_max`
  aliasing params. Keep only `original_max_seq_len` / `target_seq_len`.

### 4.2 models/yarn.py (-10 lines)
- **shrink** `YaRNRoPE.__init__`: drop `original_max` / `target_max`
  aliasing. Keep only `original_max_seq_len` / `target_seq_len`.

### 4.3 models/transformer.py (-50 lines)
- **delete** `as_dict` method (line 120-122).
- **delete** `field` from dataclasses import (line 3, never used).
- **delete** `Optional` from typing import (line 4, only used in
  forward's `positions: Optional[torch.Tensor] = None` — can use
  `torch.Tensor | None`).
- **delete** `n_pruned_dims` attribute on `GPTOSSAttention` — compute
  on the fly from `is_windowed` and the global cfg flag.
- **change** `GPTOSSBlock.__init__`: pass `cfg` (the dataclass itself)
  to `GPTOSSAttention` and `MoELayer` instead of `cfg.as_dict()`.

### 4.4 models/attention.py (-50 lines)
- **shrink** collapse `manual_causal_attention`,
  `sliding_window_attention`, `full_causal_attention` into ONE
  `causal_attention(q, k, v, *, window=None, sink=None)` function.
  Three `functools.lru_cache` mask builders collapse into one.
- **change** `GPTOSSAttention.__init__` to take a `ModelConfig`
  directly (not a dict).
- **delete** `repeat_kv` standalone function — inline as
  `x.repeat_interleave(n_rep, dim=1)` (2 lines, stdlib).
- **delete** `_sink_clamp_min` / `_sink_clamp_max` class attrs — make
  them module-level constants in `attention.py`. Read by both
  forward paths in attention.py AND inference/generate.py.
- **delete** `n_pruned_dims` attribute; compute from
  `is_windowed` + cfg in `forward()`.
- **change** `extra_repr` to compute `n_pruned` on the fly.

### 4.5 models/moe.py (-15 lines)
- **delete** `SwiGLUExpert` class — replace with three `nn.Linear`
  lists on `MoELayer` (w1_list, w2_list, w3_list). Construction
  becomes `nn.ModuleList([nn.Linear(...) for _ in range(n)])`.
  Forward in dispatch becomes:
  ```
  out = w2_list[e](F.silu(w1_list[e](x)) * w3_list[e](x))
  ```
- **change** `MoELayer.__init__` to take `ModelConfig` directly.
- **note** Triton path (`models/moe_triton.py`) does
  `torch.stack([e.w1.weight for e in self.experts])` — must be
  rewritten to `torch.stack(self.w1_list, dim=0)`. Otherwise the
  Triton autograd Function stays the same.

### 4.6 models/__init__.py (-3 lines)
- **delete** `prune_rope` from imports and `__all__`.
- **delete** `manual_causal_attention` from imports and `__all__`.
- **delete** `full_causal_attention` / `sliding_window_attention` —
  replaced by single `causal_attention`.

### 4.7 inference/generate.py (-25 lines)
- **change** import `causal_attention` instead of the three.
- **change** inline `repeat_kv` as `repeat_interleave`.
- **change** read sink-clamp constants from `models.attention` instead
  of `attn._sink_clamp_min` / `attn._sink_clamp_max` (deleted attrs).
- **delete** `n_pruned_dims` reference — pass the value via
  `attn.is_windowed` + cfg instead.

### 4.8 inference/__init__.py (delete)
- **delete** empty placeholder.

### 4.9 utils/checkpoint.py (-3 lines)
- **delete** `state_dict: Optional[dict] = None` param from `save`.
- **delete** the `if k != "step"` filter in extra_meta — no caller
  passes a `step` key.

### 4.10 utils/memory.py (-50 lines)
- **delete** private helpers (`_parameter_bytes`, `_optimiser_bytes`,
  `_mixed_kv_cache_bytes`, `_activation_bytes`,
  `_infer_dim_n_layers`, `_detect_overhead_gb`).
- **inline** the bodies into `estimate_model_memory_gb`.

### 4.11 utils/logging.py (no change)
- `json` import is unused; remove for tidiness only.

### 4.12 utils/__init__.py (delete)
- **delete** empty placeholder.

### 4.13 scripts/_bootstrap.py (new, 12 lines)
- New shared module. Path-mutates `sys.path` and re-exports the
  common ModelConfig micro setup.
- All 7 scripts import this and call `from _bootstrap import cfg`
  or similar.

### 4.14 scripts/profile_*.py, step_time_a100.py, microbench_a100.py, passkey_eval.py, kv_cache_benchmark.py
- **delete** the 7-line `sys.path.append` + `import torch, yaml` +
  `import time` boilerplate at the top of each.
- **delete** the 25-line `ModelConfig(...)` micro-block; replace
  with `from _bootstrap import micro_cfg; cfg = micro_cfg()`.
- **delete** the duplicate `time_fn` definitions in profile_*.py and
  step_time_a100.py — move one to `_bootstrap.py`.
- **merge** `profile_inference.py` + `profile_longctx.py` into
  `profile_inference.py` (one test sweep over prompt lengths).
- **delete** `profile_step.py` (already covered by step_time_a100).
- **delete** `launch_a100.sh` (yaml + make handles orchestration).
- **delete** `kv_cache_benchmark.py` and `passkey_eval.py` — pure
  verification scripts; their numbers live in README. Keep
  `microbench_a100.py` (the actual VRAM benchmark with threshold).

### 4.15 tests/conftest.py (-40 lines)
- **shrink** replace 30-line `model_cfg` / `small_cfg` dict literals
  with `ModelConfig(**{...overrides...})` blocks.
- **add** `micro_cfg` fixture for the 4-layer / 64-dim test model
  (used by 4 test files).

### 4.16 tests/test_yarn.py (-25 lines)
- **delete** `test_prune_rope_*` (3 tests). Production caller is
  dead.
- **update** `test_yarn_module_pruned_dims` — passes a
  `cfg.yarn_prune_rope_global` flag now, not a kwarg.

### 4.17 tests/test_attention.py
- **update** imports: drop `manual_causal_attention`,
  `sliding_window_attention`, `full_causal_attention`. Use the new
  `causal_attention(q, k, v, window=..., sink_bias=...)`.
- **update** `n_pruned_dims` test → check via the property or
  call site.
- **delete** `test_repeat_kv_identity` / `test_repeat_kv_doubles`
  (inlined as `repeat_interleave`; covered by stdlib).

### 4.18 tests/test_validation.py
- **delete** `test_modelconfig_as_dict_isolated_from_dataclass` —
  `as_dict` is gone.
- **update** the test that uses `cfg.as_dict()` for tied-weights
  diff — use `ModelConfig(weight_tying=False)` instead.

### 4.19 tests/test_models.py
- **update** `n_pruned_dims` assertion → check via forward path or
  the property.

### 4.20 tests/test_inference.py
- **update** import (drop the 3 deleted attention fns).

## 5. Phased execution (one phase per commit-ready chunk)

Each phase ends with `pytest tests/ -x` passing.

| # | Phase | Files | Risk |
|---|-------|-------|------|
| 1 | Add `scripts/_bootstrap.py` (non-breaking) | 1 new | low |
| 2 | Migrate scripts to bootstrap (delete sys.path + cfg + time_fn) | 8 scripts | low |
| 3 | Delete `prune_rope` + tests | rotary, yarn, init, test_yarn | low |
| 4 | Delete dual-name yarn params | rotary, yarn, init, tests | low |
| 5 | Delete `field` + `Optional` + `as_dict` in transformer.py | transformer, tests | low |
| 6 | Delete `n_pruned_dims` attr + `_sink_clamp_*` attrs + `repeat_kv` | attention, transformer, generate, test | medium |
| 7 | Collapse 3 attn fns → 1 | attention, generate, test_attention, test_inference | medium |
| 8 | Inline `MoELayer` SwiGLUExpert (delete class) | moe, moe_triton, test_moe | medium |
| 9 | Change `MoELayer` / `GPTOSSAttention` to take `ModelConfig` directly | moe, attention, transformer, tests | medium |
| 10 | Inline `utils/memory.py` private helpers | memory | low |
| 11 | `CheckpointManager`: drop `state_dict=None` + `step` filter | checkpoint | low |
| 12 | Delete empty `__init__.py` files + `launch_a100.sh` + `profile_step.py` | init, scripts | low |

Each phase = one TaskCreate + a few Edit/Write tool calls.

## 6. Verification gate

After every phase:
```
pytest tests/ -x --tb=short -q
```
must pass. If it doesn't, fix before moving to the next phase.

After all 12 phases, run the full sweep one more time:
```
pytest tests/ --tb=short -q
```

## 7. Out of scope (deliberately)

- No new `pyproject.toml` (the user can do this later; the
  `sys.path` cleanup is the only path-coercion we do).
- No model behavior changes (e.g. don't change `manual_causal_attention`'s
  FP32 accumulation policy).
- No new tests (the audit is cleanup, not coverage).
- No new docs (the existing `documentation/` is unaffected; doc
  files referencing the deleted symbols — e.g.
  `documentation/ATTENTION_SINKS.md`, `OPTIMIZATIONS.md` — are NOT
  edited; they are descriptive prose about the architecture, not
  user-facing API references).
