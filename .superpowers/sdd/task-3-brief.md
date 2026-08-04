### Task 3: Create `rope_yarn.md`; delete `rotary.md` and `yarn.md`

**Files:**
- Create: `documentation/rope_yarn.md`
- Delete: `documentation/rotary.md`, `documentation/yarn.md`

**Interfaces:**
- Consumes: full rotary + yarn chapters
- Produces: single position-encoding chapter in the order locked by the spec

- [ ] **Step 1: Write new file with this exact top-level TOC**

```markdown
# RoPE and YaRN — Position Encoding for 128K

> Purpose: end-to-end position encoding from pairwise RoPE geometry through
> YaRN extrapolation and pruned RoPE on global layers.
> Sources: `models/rotary.py`, `models/yarn.py`.

## Table of contents

1. Purpose and mental model
2. RoPE geometry and `apply_rope`
3. Frequency bases
4. YaRN theory (ramp, blend, mscale)
5. Production parameters (θ=100K, scale=32, target=131072)
6. `compute_yarn_freqs` / `compute_yarn_mscale`
7. `YaRNRoPE` module
8. Pruned RoPE on global layers (25% of dims)
9. Dtype / SDPA contract
10. Worked numerical examples
11. Degenerate ramp warning
12. Debugging long-context issues
13. Invariants and failure modes
14. How to verify
```

- [ ] **Step 2: Port content in order**

1. From `rotary.md`: §§1–4, 8–12 (`apply_rope`, geometry, freqs helpers, dtype, broadcasting, worked example).  
2. From `yarn.md`: §§1, 3–14 (skip yarn’s “RoPE Recap” if redundant with step 1).  
3. Merge “Interaction with YaRN” sections into one dataflow subsection.  
4. Keep degenerate-ramp `UserWarning` behavior explicit (AGENTS numerical-stability rule).

Verification commands for §14:

```bash
python3 -m pytest tests/test_yarn.py -v
```

- [ ] **Step 3: Delete old files and fix links**

```bash
rm documentation/rotary.md documentation/yarn.md
rg -n 'rotary\.md|yarn\.md' documentation/ README.md AGENTS.md SKILLS.md
```

Replace with `rope_yarn.md`.

- [ ] **Step 4: Size sanity**

```bash
wc -l documentation/rope_yarn.md
test ! -f documentation/rotary.md && test ! -f documentation/yarn.md
```

Expected: roughly 700–900 lines (not a naive concat of 396+474 with double TOCs).

---

