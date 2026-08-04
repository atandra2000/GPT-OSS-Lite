### Task 10: Rewrite `documentation/README.md`; delete `CONTEXT.md`

**Files:**
- Rewrite: `documentation/README.md`
- Delete: `CONTEXT.md`
- Modify: root files that mention CONTEXT

- [ ] **Step 1: Write the new index**

Include:

1. One-paragraph project blurb + headline metrics table  
2. Learning path table with **only** the surviving files  
3. Agent routing table (from `CONTEXT.md`):

```markdown
| Question type | Read first |
|---|---|
| How does this repo implement X? | `models/*.py` + matching chapter |
| Sink bias / SWA / YaRN theory + impl | `ATTENTION_SINKS.md` |
| RoPE / YaRN | `rope_yarn.md` |
| MoE / Triton | `moe.md` |
| YAML / train loop | `training.md` |
| Data | `data_pipeline.md` |
| Inference / KV cache | `inference.md` |
| Scripts / utils / OPT-* | `operations.md` |
| What must not break? | `../AGENTS.md` + `architecture.md` invariants |
| Onboarding | `getting_started.md` |
```

4. Maintaining docs (`check_docs.py` commands)  
5. Doc size reference placeholder table (regenerated in Task 12)

- [ ] **Step 2: Delete CONTEXT and fix refs**

```bash
rm CONTEXT.md
rg -n 'CONTEXT\.md' README.md AGENTS.md SKILLS.md documentation/
```

Remove from project trees / lists.

- [ ] **Step 3: Confirm surviving set**

```bash
ls documentation/*.md | sort
```

Expected exactly:

```
ATTENTION_SINKS.md
README.md
architecture.md
data_pipeline.md
foundations.md
getting_started.md
inference.md
moe.md
operations.md
rope_yarn.md
training.md
```

---

