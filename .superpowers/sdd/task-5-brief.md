### Task 5: Merge `configs.md` into `training.md`

**Files:**
- Modify: `documentation/training.md`
- Delete: `documentation/configs.md`

**Interfaces:**
- Consumes: training loop chapter + YAML encyclopedia
- Produces: loop narrative first; YAML reference as a marked Part B

- [ ] **Step 1: Slim existing “Configuration Walkthrough”**

Keep a short selected-fields preview in the loop narrative. Remove any claim that configs live only in a separate chapter.

- [ ] **Step 2: Append Part B — YAML encyclopedia**

```markdown
## Part B — Configuration reference

### B.1 How configs are loaded (`yaml.safe_load` → `ModelConfig` / dicts)
### B.2 File comparison: `pretrain_a100_502m.yaml` vs `pretrain_gpu_smoke.yaml`
### B.3 Derived quantities (tokens/step, Chinchilla steps, active params)
### B.4 `model` block — every field
### B.5 `training` block — every field
### B.6 `data` block — every field
### B.7 Cross-field interactions
### B.8 Override patterns (CLI)
### B.9 How to verify

Run: `python3 -m pytest tests/test_training.py tests/test_validation.py -v`
```

Port field tables verbatim from `configs.md` (do not drop keys).

- [ ] **Step 3: Delete and retarget**

```bash
rm documentation/configs.md
rg -n 'configs\.md' documentation/ README.md AGENTS.md SKILLS.md CONTEXT.md
```

Replace with `training.md` (optionally `#part-b-configuration-reference`).

- [ ] **Step 4: Content-contract close**

Ensure training.md ends with invariants (NaN guard never disabled without consent) + verify commands + one next link (`data_pipeline.md` / `operations.md`).

---

