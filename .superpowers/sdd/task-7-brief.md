### Task 7: Deepen `inference.md`

**Files:**
- Modify: `documentation/inference.md`

**Interfaces:**
- Consumes: existing inference chapter + MixedKVCache details from architecture/ATTENTION_SINKS if duplicated thinly
- Produces: sharper decode complexity + verify section

- [ ] **Step 1: Ensure these sections exist with real detail**

```markdown
## MixedKVCache
### Windowed layers — ring buffer (capacity = window_size)
### Global layers — exponential growth
### Why decode is O(1) per step vs O(T) naive cache rebuild

## generate()
### Prefill vs decode
### Sink clamp cache
### YaRN T=1 fast path

## Passkey retrieval (`inference/long_context.py` / `scripts/passkey_eval.py`)
### Stub behavior on untrained checkpoints
### Target ≥85% at 128K after training

## How to verify
```

Verification block:

```bash
python3 -m pytest tests/test_inference.py -v
python3 scripts/passkey_eval.py --help
```

- [ ] **Step 2: Remove stale “when published” language**

```bash
rg -n 'when published|TODO|TBD' documentation/inference.md
```

Expected: no matches.

- [ ] **Step 3: Cross-link only to `ATTENTION_SINKS.md`, `architecture.md`, `operations.md`**

---

