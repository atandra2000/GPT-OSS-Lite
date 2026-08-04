### Task 4: Merge `triton_kernels.md` into `moe.md`

**Files:**
- Modify: `documentation/moe.md`
- Delete: `documentation/triton_kernels.md`

**Interfaces:**
- Consumes: moe theory + triton kernel contract
- Produces: one MoE chapter; Triton is an explicit opt-in section (no silent fallback)

- [ ] **Step 1: Keep MoE theory front matter**

Retain: Abstract, Why MoE, SwiGLU, routing, aux loss (α=0.01, FP32 softmax), topology, class references, stacked dispatch.

- [ ] **Step 2: Replace thin Triton subsection with full contract**

Where moe.md currently has a short “Triton grouped dispatch” section, expand into:

```markdown
## Sanctioned Triton path (`moe_dispatch="triton_grouped"`)

### Why fuse W1/W3+silu (not W2)
### Public API `triton_moe_w1w3_silu`
### `HAS_TRITON` import policy and hard failure (no silent fallback)
### Kernel tiling `(BLOCK_T=16)×(BLOCK_M=32)×(BLOCK_N=32)`, `num_stages=1`
### Autograd: forward Triton, backward PyTorch reference
### Integration: `MoELayer._dispatch_triton`
### sm_75 vs sm_80 notes
### When to enable
### How to verify

Run: `python3 -m pytest tests/test_moe.py tests/test_moe_triton.py -v`
# GPU-only cases: pytest -m gpu
```

Port the load-bearing content from `triton_kernels.md` (API contracts, error policy, tiling, caps). Drop duplicate abstracts and glossary padding.

- [ ] **Step 3: Delete and retarget**

```bash
rm documentation/triton_kernels.md
rg -n 'triton_kernels\.md|moe_triton\.md' documentation/ README.md AGENTS.md SKILLS.md scripts/
```

Replace with `moe.md`. Update AGENTS “add documentation/\<name\>.md” examples if they cite `triton_kernels.md`.

- [ ] **Step 4: Confirm opt-in language**

```bash
rg -n 'silent fallback|moe_dispatch|triton_grouped' documentation/moe.md
```

Expected: explicit opt-in; hard error if Triton unavailable when selected.

---

