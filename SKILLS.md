# SKILLS.md — GPT-OSS-Lite

> **Project-local skill workflows** for the GPT-OSS-Lite codebase.
> These are the "things to do" patterns that an agent should follow when
> working on this repo.

---

## Skill 1: Run the smoke test suite on the architecture

**When to use:** After any change to `models/`, `inference/`, or `utils/`.
After pulling new code.

**Steps:**

1. **Run the full CPU-friendly test suite (130 tests, ~40s).**
   ```bash
   python3 -m pytest tests/ -v
   ```
   Expected: 130 passed (127 original + 3 ring-buffer ordering tests).

2. **Verify the headline metric is still measured correctly.**
   ```bash
   python3 scripts/kv_cache_benchmark.py
   ```
   Expected: `✅ HEADLINE METRIC PASSED: ... ≥ 1.8× KV-cache reduction`.

3. **If you changed `models/attention.py`, additionally verify the
   sliding-window equivalence tests:**
   ```bash
   python3 -m pytest tests/test_attention.py -v
   ```
   Critical tests:
   - `test_sliding_window_matches_full_small` (atol=1e-5)
   - `test_sliding_window_matches_full_large` (slow, atol=1e-5)
   - `test_sink_bias_absorbs_attention`
   - `test_sink_bias_clamped_at_forward` (verifies BF16-overflow guard)

4. **If you changed the model architecture, verify anchor metrics:**
   ```bash
   python3 -m pytest tests/test_models.py tests/test_validation.py -v
   ```
   Critical tests:
   - `test_anchor_metric_502m_total` (param budget)
   - `test_anchor_metric_247m_active` (active-param anchor)
   - `test_active_params_correct_with_tied_weights` (tie dedup)

**Failure modes:**
- Param count drift outside [500M, 504M] → adjust `ffn_dim` or `n_experts`.
- KV reduction < 1.8× → verify `n_layers=12` with 6 SWA + 6 global.
- NaN in YaRN at 128K → check `compute_yarn_freqs` clamping.
- Sink bias gradient explodes → the forward-time clamp should hold (verify
  with `test_sink_bias_clamped_at_forward`).

---

## Skill 2: Benchmark KV-cache reduction at 128K

**When to use:** When claiming or revising the headline KV-cache metric.
When changing the alternation pattern (e.g. window size, SWA/full split).

**Steps:**

1. **Run the analytical benchmark (CPU-friendly, no GPU needed).**
   ```bash
   python3 scripts/kv_cache_benchmark.py
   ```
   This computes both pure GQA and alternating SWA/full cache sizes at
   4K, 8K, 32K, 64K, 128K contexts.

2. **Interpret the output.** The headline claim is ≥ 1.8× reduction at
   128K. Anything < 1.8× means the README must be revised to match
   reality — do NOT claim a metric the benchmark doesn't support.

3. **If the reduction is < 1.8×, check:**
   - `n_layers = 12` and 6 layers are windowed + 6 are full.
   - `window_size = 128` (smaller window = more savings).
   - The model is using `attn_impl = "sdpa"` (not "manual" which is O(T²)).

---

## Skill 3: Debug YaRN extrapolation at long context

**When to use:** NaN in attention scores at 128K, poor long-context
retrieval, or unstable training at longer sequences.

**Steps:**

1. **Check YaRN frequency computation.**
   ```python
   from models.rotary import compute_yarn_freqs
   freqs = compute_yarn_freqs(
       head_dim=96, theta=100000, scale_factor=32,
       original_max=4096, target_max=131072,
       beta_fast=32, beta_slow=1,
   )
   assert torch.isfinite(freqs).all()
   assert (freqs > 0).all()
   ```

2. **Verify YaRN module produces finite output at 128K.**
   ```python
   from models.yarn import YaRNRoPE
   rope = YaRNRoPE(head_dim=96, theta=100000, scale_factor=32,
                   original_max_seq_len=4096, target_seq_len=131072)
   cos, sin = rope(torch.tensor([131072]))
   assert torch.isfinite(cos).all()
   assert torch.isfinite(sin).all()
   ```

3. **Common causes of NaN at long context:**
   - **Wrong `scale_factor`**: should be `target_max / original_max`. For
     128K from 4K, that's exactly 32.
   - **`beta_fast` / `beta_slow` misconfigured**: ramp bounds degenerate
     when `high <= low`. Use defaults (32, 1).
   - **Missing mscale**: long-context attention is poorly scaled without
     `mscale = True`. Symptom: very small or very large attention scores.

4. **If NaN persists, fall back to standard RoPE** (set
   `yarn_scale_factor=1.0`) to verify it's specifically a YaRN issue.

---

## Skill 4: Run the full pretraining pipeline

**When to use:** Starting a full training run on a GPU machine.

**Steps:**

1. **Pre-flight checks.**
   ```bash
   python3 -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
   python3 -m pytest tests/ -v  # smoke test
   python3 scripts/kv_cache_benchmark.py  # verify headline
   ```

2. **Prepare data** (downloads + tokenizes the universal 8.0B-token corpus;
   can take hours). See `data/DATA_PIPELINE.md` for the full guide.
   ```bash
   python3 data/prepare_data.py --stage pretrain
   # Or skip the download if you've already run the pipeline once:
   python3 data/prepare_data.py --stage pretrain --skip-download
   ```

3. **Run microbench + step-time to estimate wall time.**
   ```bash
   python3 scripts/microbench_a100.py
   python3 scripts/step_time_a100.py --steps 20 --warmup 5
   ```

4. **Start training.**
   ```bash
   bash scripts/launch_a100.sh
   # or directly:
   python3 training/pretrain.py --config configs/pretrain_a100_502m.yaml
   ```

5. **Monitor** via the wandb dashboard (set `WANDB_PROJECT` env var).

---

## Skill 5: Evaluate passkey retrieval at long context

**When to use:** After training completes, to verify the long-context
extrapolation headline (≥ 85% retrieval at 128K).

**Steps:**

1. **Run passkey eval with a trained checkpoint.**
   ```bash
   python3 scripts/passkey_eval.py \
       --checkpoint checkpoints/pretrain_a100/model_step_60000.safetensors \
       --n-trials 100 \
       --context-lengths 4096 8192 32768 65536 131072 \
       --position middle
   ```

2. **Interpret results.** Expected: accuracy stays ≥ 85% from 4K to 128K.
   Each context length uses a separate seeded RNG (`base_seed + ctx_len`),
   so different lengths are statistically independent.

3. **If accuracy degrades at 128K:**
   - Verify YaRN is active (not standard RoPE).
   - Verify `yarn_prune_rope_global = True` (helps global layers at 128K).
   - Verify the model was trained with `yarn_mscale = True`.
   - Increase training data (more tokens = better extrapolation).

4. **If accuracy degrades at 32K or 64K, your model has not learned the
   YaRN ramp properly. Likely cause: insufficient YaRN-active steps.
   Mitigation: increase training tokens or decrease scale_factor.**

## Skill 6: Reproducible training run

**When to use:** When you need bit-exact reproducibility across runs (paper
experiments, ablations, debugging).

**Steps:**

1. **Always pass `--seed N` to the pretraining script:**
   ```bash
   python3 training/pretrain.py \
       --config configs/pretrain_a100_502m.yaml \
       --seed 42
   ```
   Without `--seed`, runs are NOT reproducible.

2. **Resume from a checkpoint (preserves RNG state):**
   ```bash
   python3 training/pretrain.py \
       --config configs/pretrain_a100_502m.yaml \
       --seed 42 \
       --resume-from 40000
   ```
   RNG state is saved as `checkpoints/pretrain_a100/rng_step_N.pt` alongside
   weights and restored on resume.

3. **Verify reproducibility by re-running:**
   ```bash
   # First run
   python3 training/pretrain.py --config ... --seed 42 --max-steps 100
   # Second run
   python3 training/pretrain.py --config ... --seed 42 --max-steps 100
   # The final loss must match within ~1e-4 (BF16 non-determinism floor).
   ```

**Failure modes:**
- Loss differs by > 1e-4 between runs → check that
  `CUBLAS_WORKSPACE_CONFIG=:4096:8` is set (see `pretrain.py:seed_everything`).
- Argsort order changes between runs → the MoE dispatch uses
  `stable=True`; verify with `test_moe_dispatch_is_deterministic`.
- Data shuffle differs → verify the DataLoader uses the seeded generator
  (PyTorch's DataLoader inherits the global RNG by default).

## Skill 7: Adding a new architectural component

**When to use:** Before adding new components that could break the headline metric.

**Steps:**

1. **Identify which existing component it overlaps with:**
   - Attention: avoid MLA (DeepSeek), GDN (FusionLLM), SSM (Mamba-2).
   - MoE: avoid aux-loss-free bias (DeepSeek), top-4-of-20 routing (DeepSeek).
   - Long context: avoid θ=500K (LLaMA-3), constant-state SSM (Mamba-2).

2. **Write the implementation in the appropriate `models/` file.**
   Validate the config in `ModelConfig.__post_init__` if it adds new fields.

3. **Add tests** following the pattern in `tests/test_*.py`.
   - At minimum: shape, forward+backward, gradient flow, determinism.
   - For new attention variants: add an `test_*_compose_*` test that
     verifies interaction with sink bias + sliding window.

4. **Verify the smoke test suite passes:**
   ```bash
   python3 -m pytest tests/ -v
   ```

5. **If the new component changes anchor metrics, update README.md.**

---

## Skill 8: Profile and optimise a hotspot

**When to use:** When a script is too slow and you don't know why. Before
adding more complex optimisations (e.g. CUDA graphs, custom Triton kernels).

**Steps:**

1. **Identify the slow operation with a per-component microbench.**
   The repo has dedicated scripts under `scripts/`:
   ```bash
   python3 scripts/profile_components.py   # per-component ms/op
   python3 scripts/profile_step.py         # end-to-end training step
   python3 scripts/profile_inference.py    # generation throughput
   python3 scripts/profile_longctx.py      # decode-time vs context length
   python3 scripts/profile_moe.py          # MoE dispatch only
   ```

2. **Cross-reference with `documentation/OPTIMIZATIONS.md`.** The doc lists
   every optimisation applied, what it does, and how much it bought. If your
   hotspot is one of the listed items, you're done; if not, you're in
   uncharted territory and should design a new optimisation.

3. **Apply the optimisation, then re-run the microbench AND the full test
   suite to verify bit-exactness:**
   ```bash
   python3 -m pytest tests/ -q
   ```

4. **Document the new optimisation in `documentation/OPTIMIZATIONS.md`**
   following the same format (problem, fix, impact, risk, test coverage).

**Optimisation ideas NOT yet applied (backlog):**

- **Fused QKV**: combine `q_proj` and `kv_proj` into one matmul of
  `d_model → (n_heads + 2 * n_kv_heads) * head_dim`. Saves one kernel
  launch per layer (small, ~1-2% at production scale).
- **Channels-last memory format**: per the workspace AGENTS.md rule 13,
  `model.to(memory_format=torch.channels_last)` is mandatory on RTX 5090
  (Blackwell). For A100/Ampere, the speedup is small (~3%) but memory
  layout is more friendly to cuDNN.
- **CUDA Graphs**: capture the entire prefill + decode step as a CUDA
  graph for fixed-shape batches. Saves 5-10% on small batch/seq where
  kernel launch overhead dominates. Skipped here because production batch
  sizes are large enough that launch overhead is negligible.
- **Triton fused cross-entropy**: the chunked CE is already optimised but
  a single-pass Triton kernel can fuse the log-softmax + nll_loss + grad.
  Save ~2-3% of the training step.
- **Speculative decoding for inference**: not yet implemented; would
  give 2-3× decode speedup on greedy sampling.