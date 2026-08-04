### Task 12: Update `check_docs.py`; lint clean; vault sync

**Files:**
- Modify: `scripts/check_docs.py`

- [ ] **Step 1: Update `STALE_PATTERNS`**

Replace/extend so old filenames are flagged if reintroduced:

```python
STALE_PATTERNS: list[tuple[str, str]] = [
    (r"\{,\}", "LaTeX thousand separator `{,}`"),
    (r"\b190 tests?\b", "stale test count (use 187)"),
    (r"\b185 tests?\b", "stale test count (use 187)"),
    (r"\b130 tests?\b", "stale test count (use 187)"),
    (r"\b600-line\b", "stale ATTENTION_SINKS line count"),
    (r"moe_triton\.md", "use moe.md (Triton section)"),
    (r"triton_kernels\.md", "merged into moe.md"),
    (r"transformer\.md", "merged into architecture.md"),
    (r"\battention\.md\b", "merged into ATTENTION_SINKS.md"),
    (r"rotary\.md", "merged into rope_yarn.md"),
    (r"yarn\.md", "merged into rope_yarn.md"),
    (r"configs\.md", "merged into training.md"),
    (r"scripts\.md", "merged into operations.md"),
    (r"utils\.md", "merged into operations.md"),
    (r"OPTIMIZATIONS\.md", "merged into operations.md"),
    (r"ENABLE_TRITON_KERNELS", "removed env-var gate; use moe_dispatch config"),
    (r"when published", "stale placeholder link language"),
]
```

Note: `\battention\.md\b` must not match `ATTENTION_SINKS.md` — use a pattern that requires exact `attention.md` (as above).

- [ ] **Step 2: Run linter and refresh tables**

```bash
python3 scripts/check_docs.py
python3 scripts/check_docs.py --update-sizes
python3 scripts/check_docs.py --stamp-footers
python3 scripts/check_docs.py
```

Expected: final command exits 0 with no issues.

- [ ] **Step 3: Headline metric still green (docs-only change, sanity)**

```bash
python3 scripts/kv_cache_benchmark.py
```

Expected: `HEADLINE METRIC PASSED` (or existing ≥1.8× pass string).

- [ ] **Step 4: Vault sync**

```bash
bash ~/Desktop/CoreProjects/scripts/sync_to_vault.sh
```

Expected: mirrors updated without error.

- [ ] **Step 5: Success criteria checklist**

Re-check spec §7:

- [ ] Only target files remain under `documentation/`  
- [ ] `data/DATA_PIPELINE.md` is stub; full guide in documentation  
- [ ] `CONTEXT.md` removed  
- [ ] `ATTENTION_SINKS.md` still authoritative path  
- [ ] Chapters follow content contract  
- [ ] `check_docs.py` exits 0  
- [ ] Root README matches reality  
- [ ] Vault sync done  

---

