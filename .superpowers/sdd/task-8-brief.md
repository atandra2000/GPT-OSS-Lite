### Task 8: Create `operations.md`; delete scripts/utils/OPTIMIZATIONS

**Files:**
- Create: `documentation/operations.md`
- Delete: `documentation/scripts.md`, `documentation/utils.md`, `documentation/OPTIMIZATIONS.md`

**Interfaces:**
- Consumes: three ops chapters
- Produces: one operations handbook

- [ ] **Step 1: Write `operations.md` with this structure**

```markdown
# Operations — Scripts, Utilities, and Optimizations

> Purpose: runbooks for benchmarks, checkpoints/logging/memory helpers, and the
> OPT-1…24 performance catalog.

## Part A — Scripts (`scripts/`)
### A.1 Selection guide (table)
### A.2 `_bootstrap.py`
### A.3 `check_docs.py`
### A.4 `kv_cache_benchmark.py` (headline ≥1.8×)
### A.5 `passkey_eval.py`
### A.6 `e2e_gpu_smoke.py`
### A.7 Profilers (`profile_components`, `profile_moe`, `profile_inference`)
### A.8 `microbench_a100.py` / `step_time_a100.py`

## Part B — Utilities (`utils/`)
### B.1 `CheckpointManager` (atomic safetensors protocol)
### B.2 `TrainingLogger` + WandB
### B.3 `estimate_model_memory_gb` / Mixed KV term / `assert_fits_in_available_gpu`
### B.4 Worked VRAM examples

## Part C — Optimization catalog (OPT-1 … OPT-24)
### C.1 How to read the catalog
### C.2 Attention and masks
### C.3 Tensor layout and dtype
### C.4 MoE dispatch
### C.5 Training loop
### C.6 Inference
### C.7 Numerical stability
### C.8 Compilation
### C.9 Quick reference table

## How to verify
```

Port content from the three sources. Keep OPT numbers stable (do not renumber). One quick-reference table only (drop triplicate indexes).

Verify commands:

```bash
python3 scripts/check_docs.py
python3 scripts/kv_cache_benchmark.py
python3 -m pytest tests/test_utils.py -v
```

- [ ] **Step 2: Delete absorbed files**

```bash
rm documentation/scripts.md documentation/utils.md documentation/OPTIMIZATIONS.md
```

- [ ] **Step 3: Retarget links**

```bash
rg -n 'scripts\.md|utils\.md|OPTIMIZATIONS\.md' documentation/ README.md AGENTS.md SKILLS.md
```

Replace with `operations.md`. In `SKILLS.md`, change OPTIMIZATIONS cross-refs to `documentation/operations.md` Part C.

---

