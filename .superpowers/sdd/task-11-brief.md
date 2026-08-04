### Task 11: Update root `README.md`, `AGENTS.md`, `SKILLS.md`

**Files:**
- Modify: `README.md`, `AGENTS.md`, `SKILLS.md`

- [ ] **Step 1: Root README Documentation table**

Replace the docs table with the surviving set. Fix project structure tree:

- `documentation/…` new names  
- `data/DATA_PIPELINE.md` described as stub → documentation  
- Remove `CONTEXT.md`  
- Remove duplicate script lines if still present  

- [ ] **Step 2: AGENTS.md path-only edits**

Keep hard rules intact. Ensure architecture / ATTENTION_SINKS / moe (Triton) paths match. If it mentions `OPTIMIZATIONS.md` or `triton_kernels.md`, retarget.

- [ ] **Step 3: SKILLS.md**

Retarget:

- `data/DATA_PIPELINE.md` full-guide mentions → `documentation/data_pipeline.md`  
- `documentation/OPTIMIZATIONS.md` → `documentation/operations.md`

- [ ] **Step 4: Repo-wide stale path sweep**

```bash
rg -n 'transformer\.md|attention\.md(?!_sink)|rotary\.md|yarn\.md|triton_kernels\.md|configs\.md|scripts\.md|utils\.md|OPTIMIZATIONS\.md|CONTEXT\.md|moe_triton\.md' \
  --glob '*.md' --glob '*.py'
```

Fix every hit in-repo (except historical notes inside `docs/superpowers/` specs/plans, which may name old files as sources).

---

