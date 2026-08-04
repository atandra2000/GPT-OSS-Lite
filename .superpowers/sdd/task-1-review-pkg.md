# Review package Task 1
BASE=addb6e293221125edf83773c3bdf9ad96494926a HEAD=0f7b619bf18d80df4d7a71074038c5dbb6b2738e

0f7b619 docs: merge transformer.md into architecture.md

 documentation/architecture.md | 437 +++++++++++++++++++++++++++++++++-
 documentation/transformer.md  | 532 ------------------------------------------
 2 files changed, 431 insertions(+), 538 deletions(-)

diff --git a/documentation/architecture.md b/documentation/architecture.md
index 8ce5685..d2cd124 100644
--- a/documentation/architecture.md
+++ b/documentation/architecture.md
@@ -1,30 +1,66 @@
 # Architecture — GPT-OSS-Lite System Design
 
-> **Chapter 2 of the GPT-OSS-Lite documentation.** This chapter maps the production 502M-total / 247M-active model to concrete modules, dataflow, and configuration. For the mathematical motivation behind each primitive, read [foundations.md](foundations.md) first.
+## Purpose
+
+This chapter is the single system map for GPT-OSS-Lite: the 502M-total /
+247M-active decoder, its module boundaries, config wiring, and the transformer
+stack in `models/transformer.py`. Read [foundations.md](foundations.md) first
+for the math behind each primitive; Part B below is the implementation guide
+for `ModelConfig`, `RMSNorm`, `GPTOSSBlock`, and `GPTOSS`.
+
+## Mental model
+
+`models/transformer.py` is the **composition root** — it wires attention, MoE,
+and norms into a 12-layer pre-norm decoder. `ModelConfig` (YAML → dataclass) is
+the single source of truth for shapes and invariants. Even layers run
+sliding-window attention (cheap KV); odd layers run full attention (global
+context). Training returns `(logits, aux_loss)`; inference reuses block
+submodules with `MixedKVCache` (see §9).
+
+> **Chapter 2 of the GPT-OSS-Lite documentation.**
 
 ---
 
 ## Table of contents
 
+**Part A — System design**
+
 1. [System overview](#1-system-overview)
 2. [Layer stack and residual dataflow](#2-layer-stack-and-residual-dataflow)
 3. [`GPTOSS.forward` dataflow](#3-gptossforward-dataflow)
 4. [Alternating attention pattern (layers 0–11)](#4-alternating-attention-pattern-layers-0-11)
 5. [Parameter accounting](#5-parameter-accounting)
 6. [File map and module responsibilities](#6-file-map-and-module-responsibilities)
 7. [`ModelConfig` — config to code wiring](#7-modelconfig--config-to-code-wiring)
 8. [MoE dispatch and Triton opt-in](#8-moe-dispatch-and-triton-opt-in)
 9. [Inference: `MixedKVCache` and generation](#9-inference-mixedkvcache-and-generation)
 10. [Training pipeline integration](#10-training-pipeline-integration)
 11. [Invariants and failure modes](#11-invariants-and-failure-modes)
 12. [Comparison with sibling portfolio models](#12-comparison-with-sibling-portfolio-models)
+
+**Part B — Transformer stack (`models/transformer.py`)**
+
+- [B.1 Module overview](#b1-module-overview)
+- [B.2 `ModelConfig` fields and `__post_init__` validation](#b2-modelconfig-fields-and-__post_init__-validation)
+- [B.3 `moe_dispatch` values (`stacked` \| `triton_grouped`)](#b3-moe_dispatch-values-stacked--triton_grouped)
+- [B.4 `RMSNorm`](#b4-rmsnorm)
+- [B.5 `GPTOSSBlock` construction and forward](#b5-gptossblock-construction-and-forward)
+- [B.6 `GPTOSS` construction and submodule roles](#b6-gptoss-construction-and-submodule-roles)
+- [B.7 Weight initialization policy](#b7-weight-initialization-policy)
+- [B.8 Forward pass, `positions`, return contract `(logits, aux_loss)`](#b8-forward-pass-positions-return-contract-logits-aux_loss)
+- [B.9 Gradient checkpointing schedule (`grad_ckpt_every`)](#b9-gradient-checkpointing-schedule-grad_ckpt_every)
+- [B.10 `num_parameters` / `num_active_parameters` + 502M breakdown](#b10-num_parameters--num_active_parameters--502m-breakdown)
+- [B.11 Weight tying](#b11-weight-tying)
+- [B.12 Config validation edge cases](#b12-config-validation-edge-cases)
+- [B.13 How to verify](#b13-how-to-verify)
+
 13. [Where to go next](#13-where-to-go-next)
 
 ---
 
 ## 1. System overview
 
 GPT-OSS-Lite is a **12-layer decoder-only transformer** with:
 
 - **GQA** attention (8 query heads, 4 KV heads, `head_dim=96`)
 - **Alternating** sliding-window ($W=128$) and full attention
@@ -626,25 +662,414 @@ Specifically `test_sliding_window_matches_full` must pass.
 
 1. **vs DeepSeek-v3-Lite:** Standard aux loss instead of aux-loss-free routing; GQA+SWA instead of MLA; learned sinks instead of none.
 2. **vs LLaMA-3-Lite:** MoE instead of dense FFN; sliding/full alternation; YaRN with train-time 128K alignment.
 3. **vs HyMo:** Pure attention stack (no GDN); standard MoE routing; long-context via YaRN not hybrid recurrence.
 4. **vs Mamba-3-Lite:** Attention-based long context with KV cache (mitigated by SWA) vs constant-size SSM state; MoE vs dense.
 
 GPT-OSS-Lite is the portfolio's **long-context MoE + attention sink** reference implementation.
 
 ---
 
+## Part B — Transformer stack (`models/transformer.py`)
+
+Implementation-level detail for `models/transformer.py`. Part A above covers
+system dataflow and invariants; this part owns the composition root.
+
+### B.1 Module overview
+
+`models/transformer.py` defines:
+
+| Symbol | Role |
+|--------|------|
+| `ModelConfig` | Dataclass mirror of YAML `model:` keys with validation |
+| `RMSNorm` | Pre-norm normalization (no bias, no mean centering) |
+| `GPTOSSBlock` | One transformer block: attention + MoE with residuals |
+| `GPTOSS` | Full model: embed → blocks → final norm → LM head |
+
+Lower-level primitives live in sibling modules:
+
+- `models/attention.py` — `GPTOSSAttention` (SWA/full, sink, YaRN)
+- `models/moe.py` — `MoELayer` (top-2 routed + shared, aux loss)
+- `models/yarn.py` — YaRN frequency scaling for RoPE
+
+The transformer file stays thin: it wires modules together and implements
+cross-cutting concerns (init, checkpointing, param math).
+
+### B.2 `ModelConfig` fields and `__post_init__` validation
+
+`ModelConfig` is a `@dataclass` whose fields map 1:1 to the `model:` block in
+YAML configs. Defaults match `configs/pretrain_a100_502m.yaml`.
+
+```python
+@dataclass
+class ModelConfig:
+    vocab_size: int = 128000
+    d_model: int = 768
+    n_layers: int = 12
+    n_heads: int = 8
+    n_kv_heads: int = 4
+    head_dim: int = 96
+    ffn_dim: int = 1536
+    n_routed_experts: int = 8
+    n_activated_experts: int = 2
+    n_shared_experts: int = 1
+    window_size: int = 128
+    sink_bias: bool = True
+    rope_theta: int = 100000
+    yarn_scale_factor: int = 32
+    yarn_original_max_seq_len: int = 4096
+    yarn_target_seq_len: int = 131072
+    # ... dtype, weight_tying, init_std, moe_dispatch, etc.
+```
+
+**Full field encyclopedia:** see [§7](#7-modelconfig--config-to-code-wiring) and
+[configs.md](configs.md).
+
+Construction **fails fast** on inconsistent configs:
+
+| Rule | Rationale |
+|------|-----------|
+| `n_heads % n_kv_heads == 0` | GQA requires integer repeat factor |
+| `n_heads * head_dim == d_model` | Projection shapes must align |
+| `0 < n_activated_experts <= n_routed_experts` | Valid top-k routing |
+| `yarn_scale_factor >= 1` | 1 means plain RoPE |
+| If `yarn_scale_factor > 1`, `yarn_original_max_seq_len < yarn_target_seq_len` | YaRN needs extrapolation headroom |
+
+Warnings (non-fatal):
+
+- `yarn_prune_rope_global=True` with odd `n_layers` — final layer may be
+  windowed (no pruning applied)
+- `eval_max_seq_len < max_seq_len` — eval context shorter than training
+
+### B.3 `moe_dispatch` values (`stacked` | `triton_grouped`)
+
+Default `"stacked"` — pure PyTorch MoE loop. Opt-in `"triton_grouped"` enables
+the fused kernel in `models/moe_triton.py`.
+
+**Dispatch semantics and Triton contract:** see [§8](#8-moe-dispatch-and-triton-opt-in)
+and [triton_kernels.md](triton_kernels.md).
+
+### B.4 `RMSNorm`
+
+Root Mean Square Layer Normalization replaces LayerNorm in GPT-OSS-Lite. Unlike
+LayerNorm, RMSNorm **does not subtract the mean** — only scales by the RMS of
+activations.
+
+For input vector $x \in \mathbb{R}^d$ and learned scale $\gamma$:
+
+$$
+\mathrm{RMSNorm}(x) = \frac{x}{\sqrt{\frac{1}{d}\sum_i x_i^2 + \epsilon}} \cdot \gamma
+$$
+
+```python
+class RMSNorm(nn.Module):
+    def __init__(self, dim: int, eps: float = 1e-5):
+        super().__init__()
+        self.eps = eps
+        self.weight = nn.Parameter(torch.ones(dim))
+
+    def forward(self, x: torch.Tensor) -> torch.Tensor:
+        rms = x.detach().float().pow(2).mean(dim=-1, keepdim=True).add(self.eps).rsqrt()
+        return (x * (rms * self.weight.to(rms.dtype)).to(x.dtype))
+```
+
+Design notes:
+
+1. **RMS computed in FP32** via `x.detach().float()` — stabilizes BF16 forward
+   without keeping a persistent FP32 copy of activations.
+2. **`detach()` on RMS** — norm statistics do not receive gradients (standard
+   pre-norm practice).
+3. **Learnable `weight`** initialized to ones in `_init_weights`.
+4. **`rms_norm_eps`** from config (default `1e-5`) matches LLaMA-family recipes.
+
+Each `GPTOSSBlock` has **two** RMSNorm layers (`norm1`, `norm2`) — one before
+attention, one before MoE. A third RMSNorm (`GPTOSS.norm`) sits after all
+blocks, before the LM head.
+
+### B.5 `GPTOSSBlock` construction and forward
+
+One block implements the GPT-OSS **pre-norm residual** pattern:
+
+```
+x ← x + Attention(RMSNorm(x))
+x ← x + MoE(RMSNorm(x))
+```
+
+```python
+class GPTOSSBlock(nn.Module):
+    def __init__(self, cfg: ModelConfig, layer_idx: int):
+        super().__init__()
+        self.layer_idx = layer_idx
+        self.attn = GPTOSSAttention(cfg, layer_idx)
+        self.moe = MoELayer(cfg)
+        self.norm1 = RMSNorm(cfg.d_model, cfg.rms_norm_eps)
+        self.norm2 = RMSNorm(cfg.d_model, cfg.rms_norm_eps)
+```
+
+`layer_idx` drives attention alternation inside `GPTOSSAttention`:
+
+- **Even** `layer_idx` → sliding-window attention (`is_windowed=True`)
+- **Odd** `layer_idx` → full attention (`is_windowed=False`)
+
+See [attention.md](attention.md) for mask construction and sink bias.
+
+```python
+def forward(self, x, positions) -> tuple[torch.Tensor, torch.Tensor]:
+    x = x + self.attn(self.norm1(x), positions)
+    moe_out, aux_loss = self.moe(self.norm2(x))
+    x = x + moe_out
+    return x, aux_loss
+```
+
+Returns updated hidden states `(B, T, d_model)` and a per-layer `aux_loss`
+scalar. `GPTOSS.forward` takes the mean across layers.
+
+### B.6 `GPTOSS` construction and submodule roles
+
+```python
+class GPTOSS(nn.Module):
+    def __init__(self, cfg: ModelConfig):
+        super().__init__()
+        self.cfg = cfg
+        self.embed = nn.Embedding(cfg.vocab_size, cfg.d_model)
+        self.blocks = nn.ModuleList([
+            GPTOSSBlock(cfg, i) for i in range(cfg.n_layers)
+        ])
+        self.norm = RMSNorm(cfg.d_model, cfg.rms_norm_eps)
+        self.head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
+        if cfg.weight_tying:
+            self.head.weight = self.embed.weight
+        self._init_weights()
+```
+
+| Submodule | Shape / count | Purpose |
+|-----------|---------------|---------|
+| `embed` | `(vocab_size, d_model)` | Token → vector |
+| `blocks` | `n_layers` × `GPTOSSBlock` | Transformer stack |
+| `norm` | `d_model` | Final RMSNorm before logits |
+| `head` | `(d_model, vocab_size)` | LM projection (tied to embed) |
+
+`extra_repr()` prints a one-line summary including `num_parameters()` for
+debugging in notebooks.
+
+### B.7 Weight initialization policy
+
+`_init_weights()` runs after module construction:
+
+```python
+def _init_weights(self) -> None:
+    std = self.cfg.init_std  # default 0.02
+    for module in self.modules():
+        if isinstance(module, nn.Linear):
+            nn.init.normal_(module.weight, mean=0.0, std=std)
+        elif isinstance(module, nn.Embedding):
+            nn.init.normal_(module.weight, mean=0.0, std=std)
+        elif isinstance(module, RMSNorm):
+            nn.init.ones_(module.weight)
+    for block in self.blocks:
+        if hasattr(block.attn, "sink_bias") and block.attn.sink_bias is not None:
+            nn.init.zeros_(block.attn.sink_bias)
+```
+
+| Parameter group | Init | Notes |
+|-----------------|------|-------|
+| `Linear.weight` | $\mathcal{N}(0, 0.02^2)$ | `init_std=0.02` from config |
+| `Embedding.weight` | $\mathcal{N}(0, 0.02^2)$ | Same std as linear |
+| `RMSNorm.weight` | ones | Standard |
+| `sink_bias` | **zeros** | Model learns sink mass from scratch |
+
+Sink zero-init means early training behaves like standard causal attention;
+sink mass emerges during optimization. Theory:
+[ATTENTION_SINKS.md](ATTENTION_SINKS.md).
+
+Attention and MoE linear layers use `bias=False` — no separate bias init rules.
+
+### B.8 Forward pass, `positions`, return contract `(logits, aux_loss)`
+
+High-level dataflow: [§3](#3-gptossforward-dataflow). Source implementation:
+
+```python
+def forward(self, idx, positions=None) -> tuple[torch.Tensor, torch.Tensor]:
+    B, T = idx.shape
+    if positions is None:
+        positions = torch.arange(T, device=idx.device)
+    x = self.embed(idx)
+
+    aux_losses = []
+    use_grad_ckpt = (
+        getattr(self, "gradient_checkpointing", False)
+        and torch.is_grad_enabled()
+    )
+    grad_ckpt_every = max(1, getattr(self, "grad_ckpt_every", 3))
+
+    for layer_idx, block in enumerate(self.blocks):
+        if use_grad_ckpt and (layer_idx % grad_ckpt_every == 0):
+            x, aux = torch.utils.checkpoint.checkpoint(
+                block, x, positions, use_reentrant=False,
+            )
+        else:
+            x, aux = block(x, positions)
+        aux_losses.append(aux)
+
+    aux_loss = torch.stack(aux_losses).mean() if aux_losses else torch.zeros(...)
+    x = self.norm(x)
+    logits = self.head(x)
+    return logits, aux_loss
+```
+
+```
+input_ids (B, T)
+    │
+    ▼
+Embedding ──► x (B, T, 768)
+    │
+    ├──► Block 0 ──► aux_0
+    ├──► Block 1 ──► aux_1
+    │       ...
+    └──► Block 11 ──► aux_11
+    │
+    ▼
+RMSNorm ──► head ──► logits (B, T, vocab_size)
+    │
+aux_loss = mean(aux_0, ..., aux_11)
+```
+
+**`positions`:** shape `(T,)` or broadcastable; passed to YaRN RoPE inside each
+attention layer. Default `torch.arange(T)` assumes contiguous positions starting
+at 0. For inference with KV cache, `inference/generate.py` passes per-step
+position indices — see [inference.md](inference.md).
+
+**Return contract:** `logits` is `(B, T, vocab_size)` in model dtype (BF16 under
+autocast); `aux_loss` is a scalar mean MoE load-balancing loss. Loss
+composition lives in `training/pretrain.py` — not in the model class.
+
+### B.9 Gradient checkpointing schedule (`grad_ckpt_every`)
+
+Memory-heavy activations are traded for extra backward recomputation:
+
+```python
+def enable_gradient_checkpointing(self, every: int = 3) -> None:
+    self.gradient_checkpointing = True
+    self.grad_ckpt_every = every
+```
+
+When enabled, blocks where `layer_idx % every == 0` are wrapped in
+`torch.utils.checkpoint.checkpoint(..., use_reentrant=False)`.
+
+With `every=3` (A100 config) and 12 layers, blocks **0, 3, 6, 9** are
+checkpointed — four of twelve blocks, ~33% recomputation overhead for
+materially lower activation memory.
+
+`pretrain.py` calls `model.enable_gradient_checkpointing(every=...)` when
+`grad_checkpoint: true`. Checkpointing is **disabled** when
+`torch.is_grad_enabled()` is false (eval / `@torch.no_grad()` inference).
+
+Training integration: [§10](#10-training-pipeline-integration).
+
+### B.10 `num_parameters` / `num_active_parameters` + 502M breakdown
+
+**502M / 247M breakdown and KV math:** [§5](#5-parameter-accounting).
+
+`num_parameters(only_trainable=False)` deduplicates weight tying:
+
+```python
+def num_parameters(self, only_trainable: bool = False) -> int:
+    seen_ids = set()
+    total = 0
+    for p in self.parameters():
+        if only_trainable and not p.requires_grad:
+            continue
+        if id(p) in seen_ids:
+            continue
+        seen_ids.add(id(p))
+        total += p.numel()
+    return total
+```
+
+Production config yields **501,836,640** total (~502M). The tied
+embedding/head weight is counted once.
+
+`num_active_parameters()` estimates parameters used per forward under top-k MoE
+sparsity:
+
+```python
+def num_active_parameters(self) -> int:
+    non_moe = ...  # all params except names containing "experts"
+    expert_params = 3 * d_model * ffn_dim   # W1, W3, W2 per expert
+    moe_active = (n_activated_experts + n_shared_experts) * expert_params
+    router_params = d_model * n_routed_experts
+    return non_moe + (moe_active + router_params) * n_layers
+```
+
+Production: **247,106,400** active (~247M), ~50.8% sparsity. This is an
+**analytical estimate** aligned with Chinchilla active-param reporting — not a
+runtime profiler. Always prefer `model.num_parameters()` over hand sums.
+
+### B.11 Weight tying
+
+When `weight_tying: true`:
+
+```python
+self.head.weight = self.embed.weight
+```
+
+The LM head and token embedding share one `(vocab_size, d_model)` matrix.
+
+1. **~98M parameter savings** at vocab 128K × d_model 768 (see [§5](#5-parameter-accounting))
+2. **Consistent input/output token geometry** — standard in GPT-2/LLaMA families
+
+Implications:
+
+- Optimizer updates apply once to the shared tensor
+- Checkpoint `state_dict` contains one key for the shared weight
+- `num_parameters()` must deduplicate — counting both `embed` and `head` would
+  double-count
+
+Set `weight_tying: false` only for ablation experiments.
+
+### B.12 Config validation edge cases
+
+**`head_dim` must be even** — RoPE rotates pairs of dimensions. Odd `head_dim`
+raises `ValueError`.
+
+**YaRN with `scale_factor=1`** — degenerate case, plain RoPE without length
+extrapolation. Valid for smoke configs with small `eval_max_seq_len`.
+
+**`sink_bias: false`** — attention layers omit learnable sink parameters. Forward
+path skips clamp and sink-augmented softmax. Use only for ablations; production
+GPT-OSS uses sinks.
+
+**Layer count and alternation** — with `n_layers=12`, you get exactly 6 windowed
+and 6 global layers. Changing `n_layers` without updating benchmarks invalidates
+the 2× KV-cache headline unless you re-derive `N_WINDOWED = n_layers // 2`.
+
+### B.13 How to verify
+
+```bash
+python3 -m pytest tests/test_models.py tests/test_smoke.py -v
+```
+
+Spot-check parameter counts on a fresh model:
+
+```python
+from models.transformer import ModelConfig, GPTOSS
+cfg = ModelConfig()
+m = GPTOSS(cfg)
+assert m.num_parameters() == 501_836_640
+assert m.num_active_parameters() == 247_106_400
+```
+
+---
+
 ## 13. Where to go next
 
 | Goal | Resource |
 |------|----------|
+| Sink bias deep-dive | [ATTENTION_SINKS.md](ATTENTION_SINKS.md) |
+| MoE routing and aux loss | [moe.md](moe.md) |
 | Mathematical foundations | [foundations.md](foundations.md) |
-| Sink bias deep-dive | `documentation/ATTENTION_SINKS.md` |
 | Run KV benchmark | `python3 scripts/kv_cache_benchmark.py` |
-| Run passkey eval | `python3 scripts/passkey_eval.py` |
 | Start training | `python3 training/pretrain.py --config configs/pretrain_a100_502m.yaml` |
-| Portfolio rules | `AGENTS.md` |
-| Cross-architecture skill | `.agents/skills/llm-architecture/SKILL.md` |
 
 ---
 
-<!-- docs:verified 2026-07-31 · fa6f918 -->
+<!-- docs:verified 2026-07-31 · task-1-merge -->
diff --git a/documentation/transformer.md b/documentation/transformer.md
deleted file mode 100644
index b4f561e..0000000
--- a/documentation/transformer.md
+++ /dev/null
@@ -1,532 +0,0 @@
-# The GPT-OSS Transformer Stack
-
-> **Chapter on `models/transformer.py`.** This chapter walks through RMSNorm,
-> `GPTOSSBlock`, the `GPTOSS` forward pass, weight initialization, parameter
-> accounting, and gradient-checkpointing hooks. For attention internals see
-> [attention.md](attention.md) and [ATTENTION_SINKS.md](ATTENTION_SINKS.md). For
-> MoE see [moe.md](moe.md). For system-level context see
-> [architecture.md](architecture.md).
-
----
-
-## Table of contents
-
-1. [Module overview](#1-module-overview)
-2. [`ModelConfig`](#2-modelconfig)
-3. [`RMSNorm`](#3-rmsnorm)
-4. [`GPTOSSBlock`](#4-gptossblock)
-5. [`GPTOSS` construction](#5-gptoss-construction)
-6. [Weight initialization](#6-weight-initialization)
-7. [Forward pass](#7-forward-pass)
-8. [Gradient checkpointing](#8-gradient-checkpointing)
-9. [Parameter counting](#9-parameter-counting)
-10. [Weight tying](#10-weight-tying)
-11. [Integration with training and inference](#11-integration-with-training-and-inference)
-12. [Config validation edge cases](#12-config-validation-edge-cases)
-13. [Where to go next](#13-where-to-go-next)
-
----
-
-## 1. Module overview
-
-`models/transformer.py` is the **composition root** of GPT-OSS-Lite. It defines:
-
-| Symbol | Role |
-|--------|------|
-| `ModelConfig` | Dataclass mirror of YAML `model:` keys with validation |
-| `RMSNorm` | Pre-norm normalization (no bias, no mean centering) |
-| `GPTOSSBlock` | One transformer block: attention + MoE with residuals |
-| `GPTOSS` | Full model: embed → blocks → final norm → LM head |
-
-Lower-level primitives live in sibling modules:
-
-- `models/attention.py` — `GPTOSSAttention` (SWA/full, sink, YaRN)
-- `models/moe.py` — `MoELayer` (top-2 routed + shared, aux loss)
-- `models/yarn.py` — YaRN frequency scaling for RoPE
-
-The transformer file deliberately stays thin: it wires modules together and
-implements cross-cutting concerns (init, checkpointing, param math).
-
----
-
-## 2. `ModelConfig`
-
-`ModelConfig` is a `@dataclass` whose fields map 1:1 to the `model:` block in
-YAML configs. Defaults match `configs/pretrain_a100_502m.yaml`.
-
-```python
-@dataclass
-class ModelConfig:
-    vocab_size: int = 128000
-    d_model: int = 768
-    n_layers: int = 12
-    n_heads: int = 8
-    n_kv_heads: int = 4
-    head_dim: int = 96
-    ffn_dim: int = 1536
-    n_routed_experts: int = 8
-    n_activated_experts: int = 2
-    n_shared_experts: int = 1
-    window_size: int = 128
-    sink_bias: bool = True
-    rope_theta: int = 100000
-    yarn_scale_factor: int = 32
-    yarn_original_max_seq_len: int = 4096
-    yarn_target_seq_len: int = 131072
-    # ... dtype, weight_tying, init_std, moe_dispatch, etc.
-```
-
-### `__post_init__` validation
-
-Construction **fails fast** on inconsistent configs. Important invariants:
-
-| Rule | Rationale |
-|------|-----------|
-| `n_heads % n_kv_heads == 0` | GQA requires integer repeat factor |
-| `n_heads * head_dim == d_model` | Projection shapes must align |
-| `0 < n_activated_experts <= n_routed_experts` | Valid top-k routing |
-| `yarn_scale_factor >= 1` | 1 means plain RoPE |
-| If `yarn_scale_factor > 1`, `yarn_original_max_seq_len < yarn_target_seq_len` | YaRN needs extrapolation headroom |
-
-Warnings (non-fatal):
-
-- `yarn_prune_rope_global=True` with odd `n_layers` — final layer may be
-  windowed (no pruning applied)
-- `eval_max_seq_len < max_seq_len` — eval context shorter than training
-
-Full field encyclopedia: [configs.md](configs.md).
-
-### `moe_dispatch`
-
-Default `"stacked"` — pure PyTorch MoE loop. Opt-in `"triton_grouped"` enables
-the fused kernel in `models/moe_triton.py`. See
-[triton_kernels.md](triton_kernels.md).
-
----
-
-## 3. `RMSNorm`
-
-Root Mean Square Layer Normalization replaces LayerNorm in GPT-OSS-Lite. Unlike
-LayerNorm, RMSNorm **does not subtract the mean** — only scales by the RMS of
-activations.
-
-### Definition
-
-For input vector $x \in \mathbb{R}^d$ and learned scale $\gamma$:
-
-$$
-\mathrm{RMSNorm}(x) = \frac{x}{\sqrt{\frac{1}{d}\sum_i x_i^2 + \epsilon}} \cdot \gamma
-$$
-
-### Implementation
-
-```python
-class RMSNorm(nn.Module):
-    def __init__(self, dim: int, eps: float = 1e-5):
-        super().__init__()
-        self.eps = eps
-        self.weight = nn.Parameter(torch.ones(dim))
-
-    def forward(self, x: torch.Tensor) -> torch.Tensor:
-        rms = x.detach().float().pow(2).mean(dim=-1, keepdim=True).add(self.eps).rsqrt()
-        return (x * (rms * self.weight.to(rms.dtype)).to(x.dtype))
-```
-
-Design notes:
-
-1. **RMS computed in FP32** via `x.detach().float()` — stabilizes BF16 forward
-   without keeping a persistent FP32 copy of activations.
-2. **`detach()` on RMS** — norm statistics do not receive gradients (standard
-   pre-norm practice).
-3. **Learnable `weight`** initialized to ones in `_init_weights`.
-4. **`rms_norm_eps`** from config (default `1e-5`) matches LLaMA-family recipes.
-
-Each `GPTOSSBlock` has **two** RMSNorm layers (`norm1`, `norm2`) — one before
-attention, one before MoE. A third RMSNorm (`GPTOSS.norm`) sits after all
-blocks, before the LM head.
-
----
-
-## 4. `GPTOSSBlock`
-
-One block implements the GPT-OSS **pre-norm residual** pattern:
-
-```
-x ← x + Attention(RMSNorm(x))
-x ← x + MoE(RMSNorm(x))
-```
-
-### Construction
-
-```python
-class GPTOSSBlock(nn.Module):
-    def __init__(self, cfg: ModelConfig, layer_idx: int):
-        super().__init__()
-        self.layer_idx = layer_idx
-        self.attn = GPTOSSAttention(cfg, layer_idx)
-        self.moe = MoELayer(cfg)
-        self.norm1 = RMSNorm(cfg.d_model, cfg.rms_norm_eps)
-        self.norm2 = RMSNorm(cfg.d_model, cfg.rms_norm_eps)
-```
-
-`layer_idx` drives attention alternation inside `GPTOSSAttention`:
-
-- **Even** `layer_idx` → sliding-window attention (`is_windowed=True`)
-- **Odd** `layer_idx` → full attention (`is_windowed=False`)
-
-See [attention.md](attention.md) for mask construction and sink bias.
-
-### Forward signature
-
-```python
-def forward(self, x, positions) -> tuple[torch.Tensor, torch.Tensor]:
-    x = x + self.attn(self.norm1(x), positions)
-    moe_out, aux_loss = self.moe(self.norm2(x))
-    x = x + moe_out
-    return x, aux_loss
-```
-
-Returns:
-
-- Updated hidden states `x` with shape `(B, T, d_model)`
-- Per-layer `aux_loss` scalar from MoE load balancing
-
-The block does **not** aggregate aux loss — `GPTOSS.forward` takes the mean
-across layers.
-
----
-
-## 5. `GPTOSS` construction
-
-```python
-class GPTOSS(nn.Module):
-    def __init__(self, cfg: ModelConfig):
-        super().__init__()
-        self.cfg = cfg
-        self.embed = nn.Embedding(cfg.vocab_size, cfg.d_model)
-        self.blocks = nn.ModuleList([
-            GPTOSSBlock(cfg, i) for i in range(cfg.n_layers)
-        ])
-        self.norm = RMSNorm(cfg.d_model, cfg.rms_norm_eps)
-        self.head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
-        if cfg.weight_tying:
-            self.head.weight = self.embed.weight
-        self._init_weights()
-```
-
-### Submodule roles
-
-| Submodule | Shape / count | Purpose |
-|-----------|---------------|---------|
-| `embed` | `(vocab_size, d_model)` | Token → vector |
-| `blocks` | `n_layers` × `GPTOSSBlock` | Transformer stack |
-| `norm` | `d_model` | Final RMSNorm before logits |
-| `head` | `(d_model, vocab_size)` | LM projection (tied to embed) |
-
-`extra_repr()` prints a one-line summary including `num_parameters()` for
-debugging in notebooks.
-
----
-
-## 6. Weight initialization
-
-`_init_weights()` runs after module construction:
-
-```python
-def _init_weights(self) -> None:
-    std = self.cfg.init_std  # default 0.02
-    for module in self.modules():
-        if isinstance(module, nn.Linear):
-            nn.init.normal_(module.weight, mean=0.0, std=std)
-        elif isinstance(module, nn.Embedding):
-            nn.init.normal_(module.weight, mean=0.0, std=std)
-        elif isinstance(module, RMSNorm):
-            nn.init.ones_(module.weight)
-    for block in self.blocks:
-        if hasattr(block.attn, "sink_bias") and block.attn.sink_bias is not None:
-            nn.init.zeros_(block.attn.sink_bias)
-```
-
-### Init policy summary
-
-| Parameter group | Init | Notes |
-|-----------------|------|-------|
-| `Linear.weight` | $\mathcal{N}(0, 0.02^2)$ | `init_std=0.02` from config |
-| `Embedding.weight` | $\mathcal{N}(0, 0.02^2)$ | Same std as linear |
-| `RMSNorm.weight` | ones | Standard |
-| `sink_bias` | **zeros** | Model learns sink mass from scratch |
-
-Sink zero-init means early training behaves like standard causal attention;
-sink mass emerges during optimization. Theory:
-[ATTENTION_SINKS.md](ATTENTION_SINKS.md).
-
-Biases: attention and MoE linear layers use `bias=False` where applicable —
-there are no separate bias init rules.
-
----
-
-## 7. Forward pass
-
-```python
-def forward(self, idx, positions=None) -> tuple[torch.Tensor, torch.Tensor]:
-    B, T = idx.shape
-    if positions is None:
-        positions = torch.arange(T, device=idx.device)
-    x = self.embed(idx)
-
-    aux_losses = []
-    use_grad_ckpt = (
-        getattr(self, "gradient_checkpointing", False)
-        and torch.is_grad_enabled()
-    )
-    grad_ckpt_every = max(1, getattr(self, "grad_ckpt_every", 3))
-
-    for layer_idx, block in enumerate(self.blocks):
-        if use_grad_ckpt and (layer_idx % grad_ckpt_every == 0):
-            x, aux = torch.utils.checkpoint.checkpoint(
-                block, x, positions, use_reentrant=False,
-            )
-        else:
-            x, aux = block(x, positions)
-        aux_losses.append(aux)
-
-    aux_loss = torch.stack(aux_losses).mean() if aux_losses else torch.zeros(...)
-    x = self.norm(x)
-    logits = self.head(x)
-    return logits, aux_loss
-```
-
-### Dataflow diagram
-
-```
-input_ids (B, T)
-    │
-    ▼
-Embedding ──► x (B, T, 768)
-    │
-    ├──► Block 0 ──► aux_0
-    ├──► Block 1 ──► aux_1
-    │       ...
-    └──► Block 11 ──► aux_11
-    │
-    ▼
-RMSNorm ──► head ──► logits (B, T, vocab_size)
-    │
-aux_loss = mean(aux_0, ..., aux_11)
-```
-
-### `positions` tensor
-
-Shape `(T,)` or broadcastable. Passed to YaRN RoPE inside each attention
-layer. Default `torch.arange(T)` assumes contiguous positions starting at 0.
-
-For inference with KV cache, `inference/generate.py` passes per-step position
-indices — see [inference.md](inference.md).
-
-### Return contract
-
-Training code expects:
-
-- `logits`: `(B, T, vocab_size)` in model dtype (BF16 under autocast)
-- `aux_loss`: scalar tensor, mean MoE load-balancing loss across layers
-
-Loss computation (`chunked_cross_entropy` + `aux_alpha * aux_loss`) lives in
-`training/pretrain.py` — not in the model class.
-
----
-
-## 8. Gradient checkpointing
-
-Memory-heavy activations are traded for extra backward recomputation:
-
-```python
-def enable_gradient_checkpointing(self, every: int = 3) -> None:
-    self.gradient_checkpointing = True
-    self.grad_ckpt_every = every
-```
-
-### Checkpoint schedule
-
-When enabled, blocks where `layer_idx % every == 0` are wrapped in
-`torch.utils.checkpoint.checkpoint(..., use_reentrant=False)`.
-
-With `every=3` (A100 config) and 12 layers, blocks **0, 3, 6, 9** are
-checkpointed — four of twelve blocks, ~33% recomputation overhead for
-materially lower activation memory.
-
-`pretrain.py` calls:
-
-```python
-model.enable_gradient_checkpointing(every=train_cfg.get("grad_checkpoint_every", 3))
-```
-
-when `grad_checkpoint: true`.
-
-Checkpointing is **disabled** when `torch.is_grad_enabled()` is false (eval /
-`@torch.no_grad()` inference).
-
----
-
-## 9. Parameter counting
-
-### `num_parameters(only_trainable=False)`
-
-Counts unique parameter tensors, **deduplicating weight tying**:
-
-```python
-def num_parameters(self, only_trainable: bool = False) -> int:
-    seen_ids = set()
-    total = 0
-    for p in self.parameters():
-        if only_trainable and not p.requires_grad:
-            continue
-        if id(p) in seen_ids:
-            continue
-        seen_ids.add(id(p))
-        total += p.numel()
-    return total
-```
-
-Production config yields **~502M** total. The tied embedding/head weight is
-counted once.
-
-### `num_active_parameters()`
-
-Estimates parameters **used per forward pass** under top-k MoE sparsity:
-
-```python
-def num_active_parameters(self) -> int:
-    # Sum all non-expert params (with tying dedup)
-    non_moe = ...
-    expert_params = 3 * d_model * ffn_dim   # W1, W3, W2 per expert
-    moe_active = (n_activated_experts + n_shared_experts) * expert_params
-    router_params = d_model * n_routed_experts
-    return non_moe + (moe_active + router_params) * n_layers
-```
-
-Interpretation:
-
-- **Non-MoE:** embeddings, attention, norms, router gates (router counted per
-  layer in the MoE term), output head
-- **MoE active:** only `n_activated_experts + n_shared_experts` expert FFNs per
-  layer, not all `n_routed_experts`
-
-Production: **~247M active**, ~50.8% sparsity.
-
-This is an **analytical estimate** aligned with Chinchilla "active param" reporting
-— not a runtime profiler.
-
-### Manual breakdown (502M config)
-
-| Component | Approx. share |
-|-----------|---------------|
-| Embedding + tied head | ~98M (single storage) |
-| Attention (Q/KV/O per layer) | ~85M |
-| MoE experts (all 8+1 per layer, stored) | ~280M |
-| Routers + norms + sink bias | ~remainder |
-
-Exact counts depend on `sink_bias` and bias-free linear layers. Always prefer
-`model.num_parameters()` over hand sums.
-
----
-
-## 10. Weight tying
-
-When `weight_tying: true`:
-
-```python
-self.head.weight = self.embed.weight
-```
-
-The LM head and token embedding share one `(vocab_size, d_model)` matrix.
-Benefits:
-
-1. **~98M parameter savings** at vocab 128K × d_model 768
-2. **Consistent input/output token geometry** — standard in GPT-2/LLaMA families
-
-Implications:
-
-- Optimizer updates apply once to the shared tensor
-- Checkpoint `state_dict` contains one key for the shared weight
-- `num_parameters()` must deduplicate — counting both `embed` and `head` would
-  double-count
-
-Set `weight_tying: false` only for ablation experiments.
-
----
-
-## 11. Integration with training and inference
-
-### Training (`training/pretrain.py`)
-
-```python
-model_cfg = ModelConfig(**cfg["model"])
-model = GPTOSS(model_cfg).to(device)
-# ...
-logits, aux_loss = model(input_ids)
-ce = chunked_cross_entropy(logits, target_ids)
-loss = (ce + aux_alpha * aux_loss) / accum
-```
-
-Positions default inside `forward` — training always uses full sequences from
-`PretrainDataset`, so `arange(T)` is correct.
-
-### Inference (`inference/generate.py`)
-
-Generation **does not** call `GPTOSS.forward` directly for cached decode. It
-reimplements the per-layer path in `_attn_forward_layer` to append K/V into
-`MixedKVCache`. The model's `embed`, `norm`, `head`, and block submodules are
-still used.
-
-Rationale: training forward processes full sequences; inference must support
-incremental single-token steps with heterogeneous per-layer cache policies.
-
-See [inference.md](inference.md).
-
-### Checkpoint format
-
-`utils/checkpoint.py` saves `model.state_dict()` to safetensors. Because of
-weight tying, expect keys like `embed.weight` and `head.weight` pointing to
-the same storage after load.
-
----
-
-## 12. Config validation edge cases
-
-### `head_dim` must be even
-
-RoPE rotates pairs of dimensions. Odd `head_dim` raises `ValueError`.
-
-### YaRN with `scale_factor=1`
-
-Degenerate case — plain RoPE without length extrapolation. Valid for smoke
-configs with small `eval_max_seq_len`.
-
-### `sink_bias: false`
-
-Attention layers omit learnable sink parameters. Forward path skips clamp and
-sink-augmented softmax. Use only for ablations — production GPT-OSS uses sinks.
-
-### Layer count and alternation
-
-With `n_layers=12`, you get exactly 6 windowed and 6 global layers. Changing
-`n_layers` without updating benchmarks invalidates the 2× KV-cache headline
-unless you re-derive `N_WINDOWED = n_layers // 2`.
-
----
-
-## 13. Where to go next
-
-| Topic | Document |
-|-------|----------|
-| Attention masks, sink clamp | [attention.md](attention.md), [ATTENTION_SINKS.md](ATTENTION_SINKS.md) |
-| YaRN scaling | [yarn.md](yarn.md), [rotary.md](rotary.md) |
-| MoE routing, aux loss | [moe.md](moe.md) |
-| Mixed KV cache, generation | [inference.md](inference.md) |
-| YAML keys → `ModelConfig` | [configs.md](configs.md) |
-| Training loop | [training.md](training.md) |
-| System file map | [architecture.md](architecture.md) |
-
----
-
-<!-- docs:verified 2026-07-31 · fa6f918 -->
