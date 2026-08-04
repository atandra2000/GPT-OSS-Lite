### Task 1: Merge `transformer.md` into `architecture.md`

**Files:**
- Modify: `documentation/architecture.md`
- Delete (end of task): `documentation/transformer.md`
- Verify: `python3 scripts/check_docs.py` (expect failures until later tasks update links — local grep is the gate for this task)

**Interfaces:**
- Consumes: existing architecture sections 1–13; full `transformer.md` sections 1–12
- Produces: single `architecture.md` that owns system map **and** `ModelConfig` / `RMSNorm` / `GPTOSSBlock` / `GPTOSS` / init / weight tying / checkpointing schedule

- [ ] **Step 1: Inventory overlap**

Run:

```bash
rg -n '^## ' documentation/architecture.md documentation/transformer.md
```

Note sections already covered in architecture (parameter accounting, ModelConfig wiring, MoE dispatch mention, MixedKVCache mention). Those must be **deduplicated**, not pasted twice.

- [ ] **Step 2: Append a Part B — Transformer stack**

After the existing system/invariants content (before the final “Where to go next”), insert a clearly marked part with this TOC (renumber as needed to fit the file):

```markdown
## Part B — Transformer stack (`models/transformer.py`)

### B.1 Module overview
### B.2 `ModelConfig` fields and `__post_init__` validation
### B.3 `moe_dispatch` values (`stacked` | `triton_grouped`)
### B.4 `RMSNorm`
### B.5 `GPTOSSBlock` construction and forward
### B.6 `GPTOSS` construction and submodule roles
### B.7 Weight initialization policy
### B.8 Forward pass, `positions`, return contract `(logits, aux_loss)`
### B.9 Gradient checkpointing schedule (`grad_ckpt_every`)
### B.10 `num_parameters` / `num_active_parameters` + 502M breakdown
### B.11 Weight tying
### B.12 Config validation edge cases
### B.13 How to verify

Run: `python3 -m pytest tests/test_models.py tests/test_smoke.py -v`
```

Copy concrete prose, tables, and code blocks from `transformer.md` into these subsections. Delete portfolio-comparison / “where to go next” fluff that duplicates architecture’s existing close.

- [ ] **Step 3: Apply content contract header**

Ensure the top of `architecture.md` has Purpose (≤5 lines), then mental model / file map early. End with one next-step pointer: `ATTENTION_SINKS.md` and `moe.md`.

- [ ] **Step 4: Delete `transformer.md` and fix in-file links**

```bash
rm documentation/transformer.md
rg -n 'transformer\.md' documentation/ README.md AGENTS.md SKILLS.md CONTEXT.md
```

Replace remaining hits that belong to this merge with `architecture.md` (leave hits inside yet-unmerged files if they will be rewritten in later tasks — but fix any you touch now).

- [ ] **Step 5: Spot-check**

```bash
wc -l documentation/architecture.md
test ! -f documentation/transformer.md
```

Expected: `architecture.md` grows (~650 → ~900–1100 lines); `transformer.md` gone.

---

