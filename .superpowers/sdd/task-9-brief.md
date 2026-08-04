### Task 9: Tighten `getting_started.md` and `foundations.md`

**Files:**
- Modify: `documentation/getting_started.md`
- Modify: `documentation/foundations.md`

- [ ] **Step 1: getting_started — cut README echo**

Remove long architecture/config tables that duplicate root README. Keep: install, layout, first commands, smoke train, headline metrics, pitfalls, next chapter link.

Update internal links: `attention.md`→`ATTENTION_SINKS.md`, `rotary.md`/`yarn.md`→`rope_yarn.md`, `configs.md`→`training.md`, etc.

- [ ] **Step 2: foundations — deepen thin spots**

For any section that is mostly a bullet list without “why”, add 1–2 paragraphs of motivation tied to GPT-OSS-Lite numbers (window=128, 6+6 layers, α=0.01). Do not paste implementation code (that belongs in later chapters).

Update “Where to go next” to: `architecture.md` → `ATTENTION_SINKS.md`.

- [ ] **Step 3: Grep stale paths**

```bash
rg -n 'transformer\.md|attention\.md|rotary\.md|yarn\.md|triton_kernels\.md|configs\.md|scripts\.md|utils\.md|OPTIMIZATIONS\.md' \
  documentation/getting_started.md documentation/foundations.md
```

Expected: no matches.

---

