# Mixture-of-Experts Theory

> **Chapter 3 of the GPT-OSS-Lite theory series.** A dense FFN runs every parameter on every token; a mixture of experts runs a *subset*. This chapter derives the theory of conditional computation, top-k routing, and the Switch/GShard load-balancing loss from first principles, then maps each result onto the implementation in `models/moe.py`. The implementation reference (class-by-class, parameters, FLOP accounting, dispatch layouts) is [moe.md](../moe.md); the fused Triton grouped-GEMM is covered in [triton programming](triton_programming.md); the FP32 stability story is [numerics](numerics.md).

---

## Table of contents

1. [60-second summary](#1-60-second-summary)
2. [Why it matters here](#2-why-it-matters-here)
3. [Intuition](#3-intuition)
4. [Conditional computation: sparse vs dense FFN economics](#4-conditional-computation-sparse-vs-dense-ffn-economics)
5. [Routing as learned categorical selection](#5-routing-as-learned-categorical-selection)
6. [Load-balancing theory: the Switch/GShard aux loss](#6-load-balancing-theory-the-switchgshard-aux-loss)
7. [Expert collapse](#7-expert-collapse)
8. [The shared expert](#8-the-shared-expert)
9. [Grouped and stacked dispatch](#9-grouped-and-stacked-dispatch)
10. [Code walkthrough](#10-code-walkthrough)
11. [Pitfalls + verify](#11-pitfalls--verify)

---

## 1. 60-second summary

A Mixture-of-Experts (MoE) layer replaces one large dense FFN with a *pool* of smaller experts plus a learned *router* that picks which experts process each token. GPT-OSS-Lite stores 9 SwiGLU experts per layer (8 routed + 1 shared) but activates only 3 per token: the router's top-2 choices plus the always-on shared expert. Storing 9× the expert capacity while computing 3× the expert FLOPs is the whole point — the model keeps dense-scale *capacity* (501.8M parameters) at roughly half the *compute* of a same-size dense model (247.0M active, 50.8% sparsity).

The router is a bias-free linear layer producing 8 logits per token, softmaxed in FP32, truncated to the top-2, and renormalised. Because unconstrained routing collapses onto a few experts (rich-get-richer), training adds the standard Switch/GShard auxiliary loss — the product of each expert's *routing frequency* and *mean gate probability*, scaled by the expert count — weighted by α = 0.01. The code is `models/moe.py` (router, aux loss, layer, dispatch), with an opt-in Triton grouped-GEMM for the W1/W3+SiLU stage (`models/moe_triton.py:triton_moe_w1w3_silu`).

## 2. Why it matters here

GPT-OSS-Lite is budgeted as a ~502M-parameter model that must train on a single A100 80GB (`configs/pretrain_a100_502m.yaml`). Every transformer block is MoE — there is no dense FFN anywhere in `models/transformer.py:GPTOSSBlock` — so the MoE design *is* the model's parameter/FLOP identity, not an add-on. The specific choices matter jointly:

- **Top-2 of 8 + 1 shared** (k=2, E=8, s=1) fixes both the 50.8% sparsity ratio and the per-token FFN FLOP count (section 4).
- **Softmax gating with FP32 statistics** keeps the router trainable and the aux loss honest under BF16 autocast ([numerics](numerics.md) §8.3).
- **α = 0.01 aux loss** is a deliberate, documented deviation from DeepSeek-V3-Lite's aux-loss-free gate (config comment, AGENTS.md rule 5); it is the mechanism that prevents expert collapse over 61,000 steps.
- The aux loss is returned from every layer and averaged in `models/transformer.py:GPTOSS.forward`, then folded into the objective in `training/pretrain.py:main` as `loss = (ce + aux_alpha * aux_loss) / accum`.

Quality claims are targets, not measurements: no pretraining run has happened, so the ≥85% passkey @128K and the 16–20 h / 35–40% MFU A100 figures are `[INFERENCE]`. What is derived here — parameter counts, FLOP ratios, the aux-loss lower bound, the dispatch layout — is exact and checked against the source.

## 3. Intuition

Think of a consulting firm with eight specialists and one generalist. Every client (token) is handed to the generalist plus the two specialists whose names appear on the client's intake form (the router's top-2). The firm pays salaries (parameters) for all nine, but only three people do billable work on any given client. If the intake desk always routes to the same two specialists, the other six atrophy — the firm is paying for capacity it never uses. The load-balancing loss is a management rule that penalises the intake desk whenever a specialist is both over-booked and over-preferred, keeping the whole pool exercised.

Geometrically: expert outputs live in $\mathbb{R}^{d_{\text{model}}}$; the router partitions the input manifold into regions, each region sending its tokens to a different pair of experts. A dense FFN is the degenerate case with one region and one expert. MoE trades away the smoothness of a single learned function for a *piecewise* function assembled from specialists — more capacity, same per-token compute, at the price of a discrete selection problem that must itself be learned and kept balanced.

## 4. Conditional computation: sparse vs dense FFN economics

### 4.1 FLOPs of one dense SwiGLU FFN

A dense SwiGLU FFN with input width $d$ and intermediate width $d_{\text{ff}}$ computes $W_2(\text{silu}(W_1 x) \odot (W_3 x))$ — three matrix-vector products. Each product of a $d_{\text{ff}} \times d$ matrix with a $d$-vector costs $2 d\, d_{\text{ff}}$ FLOPs (multiply-accumulate counts twice), and $W_2$ costs $2 d_{\text{ff}} d$:

$$
C_{\text{dense}} = \underbrace{2\,d\,d_{\text{ff}}}_{W_1} + \underbrace{2\,d\,d_{\text{ff}}}_{W_3} + \underbrace{2\,d_{\text{ff}}\,d}_{W_2} = 6\,d\,d_{\text{ff}} \quad \text{FLOPs/token/layer}. \tag{1}
$$

For GPT-OSS-Lite's dense-equivalent width ($d = 768$, $d_{\text{ff}} = 1536$): $C_{\text{dense}} = 6 \cdot 768 \cdot 1536 = 7077888 \approx 7.08$M FLOPs per token per layer. The same factor 6 counts parameters: three matrices of $d \times d_{\text{ff}}$ elements (bias-free, [moe.md](../moe.md) §3), so each expert stores $3 d\, d_{\text{ff}} = 3538944 \approx 3.54$M parameters.

### 4.2 The MoE budget: stored vs active

The layer stores $E$ routed experts of width $d_{\text{ff}}$ plus $s$ shared experts plus the router's gate matrix $W_g \in \mathbb{R}^{E \times d}$:

$$
P_{\text{stored}} = (E + s) \cdot 3\,d\,d_{\text{ff}} + d\,E, \qquad
P_{\text{active}} = (k + s) \cdot 3\,d\,d_{\text{ff}} + d\,E, \tag{2}
$$

where $k$ is the number of routed experts selected per token. Only the $k$ selected routed experts and the $s$ shared experts execute; the other $E - k$ routed experts contribute zero FLOPs. With $E = 8$, $k = 2$, $s = 1$: per layer, 9 stored experts ($31856640 \approx 31.9$M params including the router) versus 3 active ($10622976 \approx 10.6$M). Across 12 layers plus the always-active attention/embedding machinery, the verified totals are 501,836,640 stored and 247,032,672 active parameters — sparsity $1 - 247032672/501836640 = 0.508$. Note that attention, norms, and embeddings are *always* active; the sparsity lives entirely in the FFN blocks.

### 4.3 The honest comparison: equal capacity, fewer FLOPs

The correct baseline is not the MoE layer against one expert — it is the MoE layer against a *dense FFN of equal stored capacity*. Set the dense width $d_{\text{ff}}' = (E+s)\,d_{\text{ff}}$ so both hold the same number of expert parameters. Then by (1),

$$
\frac{C_{\text{moe}}}{C_{\text{dense}}^{\text{eq}}} = \frac{(k+s) \cdot 6\,d\,d_{\text{ff}}}{6\,d\,(E+s)\,d_{\text{ff}}} = \frac{k+s}{E+s} = \frac{3}{9} = \frac{1}{3}. \tag{3}
$$

The MoE layer performs one-third of the FFN FLOPs of a same-capacity dense FFN: 21,233,664 ≈ 21.2M versus 63,700,992 ≈ 63.7M FLOPs per token per layer. The router adds only $2 d E = 12288$ FLOPs/token — 0.06% of the layer's cost. (If the shared expert were counted as part of the pool, the ratio would be $(k+s)/(E+s)$ with $E$ the routed pool — identical expression, and the point stands: idle experts are free at inference, not at storage.)

At the model level, forward FLOPs per token scale as roughly $2N$ for a transformer with $N$ parameters (two FLOPs per parameter per token, forward; attention is a small correction at these sizes). So the MoE model runs $\approx 2 \times 247$M $= 494$M FLOPs/token forward versus $\approx 2 \times 502$M $= 1004$M for a dense model with the same total parameter count — the active parameter ratio (49.2%) is also the FLOP ratio. The dense-equivalent-quality claim is conditional: capacity is what buys quality, and MoE buys it at 49–67% of the FLOPs, at the cost of the balancing machinery that keeps all 8 routed experts usable (sections 6–7).

## 5. Routing as learned categorical selection

### 5.1 The gate

The router is a single bias-free linear map from the token's residual-stream state to $E$ logits (`models/moe.py:MoERouter`). For token $t$ with hidden state $x_t \in \mathbb{R}^d$ and gate weight $W_g \in \mathbb{R}^{E \times d}$:

$$
z_t = x_t W_g^{\top} \in \mathbb{R}^{E}, \qquad
p_{t,i} = \frac{e^{z_{t,i}}}{\sum_{j=1}^{E} e^{z_{t,j}}}, \tag{4}
$$

so $p_t$ is a categorical distribution over experts. The selection is a learned *soft* preference: $W_g$ is trained by gradient descent, and every token votes for every expert with weight $p_{t,i}$ — but only the top-$k$ experts actually compute.

### 5.2 Top-k truncation and renormalisation

The forward pass keeps the $k$ largest probabilities, then renormalises them to a probability distribution over the selected set $\mathcal{I}_t = \{i : p_{t,i} \in \text{top-}k(p_t)\}$:

$$
w_{t,i} = \frac{p_{t,i}}{\sum_{j \in \mathcal{I}_t} p_{t,j}}, \quad i \in \mathcal{I}_t, \qquad
y_t^{\text{routed}} = \sum_{i \in \mathcal{I}_t} w_{t,i}\, E_i(x_t), \tag{5}
$$

with $E_i$ the $i$-th expert's SwiGLU map. Renormalisation is not cosmetic: $\sum_{j \in \mathcal{I}_t} p_{t,j} < 1$ in general, so without it the routed output's scale would wander with how peaked $p_t$ is. Normalising fixes $\sum_i w_{t,i} = 1$ and keeps the routed contribution at unit mass (guarded by `tests/test_moe.py:test_router_weights_sum_to_one`).

### 5.3 Why top-k (k=2), not top-1

Two distinct failure modes motivate $k \geq 2$.

**Vanishing gate gradient.** The selection $\mathcal{I}_t = \text{argmax}_k$ is a discrete op — no gradient flows through the *index*. All gate learning must flow through the *weights*. With $k = 1$ and renormalisation, $w_{t,i} = p_{t,i}/p_{t,i} = 1$ is a constant, so $\partial y_t^{\text{routed}} / \partial z_t = 0$: the router would receive **zero** gradient from the task loss and could learn only from the aux loss. With $k = 2$, $w_{t,i} = p_{t,i}/(p_{t,i} + p_{t,j})$ genuinely depends on the logits, so $\partial w_{t,i}/\partial z_{t,i} \neq 0$ and the gate is trained by the language-modeling objective itself (`tests/test_moe.py:test_router_grad_flow` asserts exactly this). More selected experts = smoother, lower-variance gate gradients: each token's assignment is spread over $k$ experts, so the per-step gradient of the router is an average over $k$ independent draws instead of one hard bet — the categorical-selection analogue of reducing Monte-Carlo gradient noise by stratified sampling.

**Winner-take-all collapse.** With top-1, the runner-up expert receives no forward gradient; only the single argmax winner updates on each token. Experts that lose the argmax never improve, so they keep losing — the selection degenerates to one or two experts (section 7). Top-2 keeps the runner-up inside the gradient path, so every token trains two experts and no expert needs to *win* to improve.

GPT-OSS-Lite fixes $k = 2$ of $E = 8$ (config fields `n_activated_experts`, `n_routed_experts` in `models/transformer.py:ModelConfig`).

## 6. Load-balancing theory: the Switch/GShard aux loss

### 6.1 The two statistics

Over a batch of $N$ tokens (flattened across batch and sequence), define the **routing frequency** — the fraction of all $Nk$ top-k slots assigned to expert $i$ — and the **mean gate probability**:

$$
f_i = \frac{1}{Nk} \sum_{t=1}^{N} \mathbb{1}[i \in \mathcal{I}_t], \qquad
P_i = \frac{1}{N} \sum_{t=1}^{N} p_{t,i}. \tag{6}
$$

Both are probability vectors: $\sum_i f_i = 1$ (each of the $Nk$ slots has one owner) and $\sum_i P_i = 1$ (each $p_t$ sums to 1, so the batch mean does too). $f$ measures what the router *did* (hard assignment); $P$ measures what the router *prefers* (soft assignment). They are different objects: with saturated logits the router can strongly prefer expert 1 ($P_1 \approx 1$) while top-2 ties force slots onto expert 2 ($f_2 = 1/2$).

The auxiliary loss (Switch Transformer, Fedus et al. 2021; GShard, Lepikhin et al. 2020) is

$$
\mathcal{L}_{\text{aux}} = E \sum_{i=1}^{E} f_i\, P_i, \tag{7}
$$

exactly as implemented in `models/moe.py:aux_load_balancing_loss` (where the code's `N` variable is the token count and `n_experts` is $E$).

### 6.2 Why $E \sum_i f_i P_i$ measures imbalance

**Product structure.** $f_i P_i$ is large exactly when expert $i$ is both *frequently selected* and *strongly preferred*. Two experts that both carry mass contribute; the coupling matters: by the rearrangement inequality, $\sum_i f_i P_i$ is maximised over permutations when $f$ and $P$ are sorted in the same order — i.e. when the same experts lead on both axes, which is precisely the collapse regime. It is minimised when the router prefers experts it does not select, a transient the gradient (below) actively discourages.

**The lower bound (Cauchy–Schwarz).** In the coupled equilibrium the router is trained toward $f \approx P$: hard assignment tracks soft preference (section 6.3 shows the aux gradient drives $f$ and $P$ together). On that slice, $\sum_i f_i P_i = \sum_i f_i^2 = \lVert f \rVert_2^2$, and Cauchy–Schwarz on $f$ against the all-ones vector gives

$$
1 = \Big(\sum_i f_i\Big)^2 \le E \sum_i f_i^2 \quad\Longrightarrow\quad
\mathcal{L}_{\text{aux}} = E \sum_i f_i^2 \ge 1, \tag{8}
$$

with equality iff $f_i = 1/E$ for all $i$. Hence: **the aux loss is ≥ 1, and its minimum, exactly 1, sits at uniform routing.** The scaling by $E$ is what normalises the floor to 1, making the loss's magnitude comparable across expert counts. At the opposite extreme — every token routed to the same two experts with $P$ concentrated on one — $f = (\tfrac12, \tfrac12, 0, \dots)$, $P = (1, 0, \dots)$, and $\mathcal{L}_{\text{aux}} = E \cdot \tfrac12 = 4$ for $E = 8$; in general the collapsed value is $E/k \cdot \max_i P_i$'s dominant term, i.e. the loss spans $[1, E/2]$ here. The test `tests/test_moe.py:test_aux_loss_low_for_uniform` pins exactly this ordering: uniform logits score $\mathcal{L}_{\text{aux}} = 1.0$, collapsed logits score $\approx 4.0$.

**Gradient: negative feedback.** The hard indices make $f$ non-differentiable, so the loss's gradient flows only through $P$ (the `topk().indices` in `models/moe.py:aux_load_balancing_loss` is detached by construction). With $\partial P_j / \partial z_{t,i} = \tfrac{1}{N} p_{t,i}(\delta_{ij} - p_{t,j})$ (the softmax Jacobian, averaged over the batch),

$$
\frac{\partial \mathcal{L}_{\text{aux}}}{\partial z_{t,i}} = \frac{E}{N}\, p_{t,i}\Big(f_i - \sum_j f_j\, p_{t,j}\Big). \tag{9}
$$

The bracketed term is the deviation of expert $i$'s frequency from the probability-weighted average frequency $\langle f \rangle_p$. If $i$ is over-loaded ($f_i > \langle f \rangle_p$), gradient descent pushes $z_{t,i}$ *down* — less soft mass on the busy expert; if under-loaded, $z_{t,i}$ rises. Equation (9) is a per-token, per-step negative-feedback loop on imbalance: it does not wait for collapse to develop, and it pulls $f$ toward the shape of $P$ while $P$ is pulled toward uniformity. (The $\mathcal{L}_{\text{aux}} \ge 1$ floor of (8) is exactly the fixed point of (9): at $f = P = $ uniform, the bracket vanishes.)

### 6.3 Why α = 0.01

The total objective is $\mathcal{L} = \mathcal{L}_{\text{CE}} + \alpha \mathcal{L}_{\text{aux}}$ with $\alpha = 0.01$ (`training/pretrain.py:main`, `aux_loss_alpha` in `configs/pretrain_a100_502m.yaml`). Two quantitative facts justify the scale:

1. **Magnitude budget.** $\mathcal{L}_{\text{aux}} \in [1, 4]$ over the whole collapse spectrum, so the aux term contributes $\alpha \mathcal{L}_{\text{aux}} \in [0.01, 0.04]$. Against the initial cross-entropy — a uniform model over the 128,000-token vocabulary scores $\ln 128000 = 11.76$ per token — the aux term is 0.085% of the CE at balance and 0.34% at full collapse. It is a regulariser, not a competing objective.
2. **Counterforce sizing.** The collapse force is *cumulative*: it grows as experts diverge (section 7). The aux gradient (9) is $\mathcal{O}(\alpha E p \Delta f) \approx 0.08 \cdot p \cdot \Delta f$ per token versus an $\mathcal{O}(1)$ token-level CE gradient — a nudge that acts on *every* token, every step, in a direction the task loss is blind to. α = 0.01 is the Switch Transformer default for top-k MoE, large enough to arrest the drift, small enough not to fight legitimate specialisation (an expert that is genuinely best for a region of input space should keep receiving its tokens — the CE gradient says so; the aux gradient only opposes *imbalance*, not *specialisation*).

The alternative design — DeepSeek-V3's auxiliary-loss-free gate with per-expert bias buffers — is deliberately not used here (AGENTS.md rule 5, noted in the config): it is a more complex mechanism, and the aux-loss route is the documented, test-guarded choice.

## 7. Expert collapse

Without balancing, routing is a rich-get-richer feedback loop. Let $n_i = Nk f_i$ be the number of slots expert $i$ receives in a step. Each slot contributes a full update to $W_1^{(i)}, W_2^{(i)}, W_3^{(i)}$, so expert $i$'s gradient norm scales with $n_i$:

$$
\lVert \nabla_{W^{(i)}} \mathcal{L} \rVert \approx n_i \cdot \lVert g_i \rVert, \tag{10}
$$

where $g_i$ is the per-token gradient of one expert. Experts with high $n_i$ improve faster; improved experts produce lower loss for their regions; the gate, trained to minimise loss, raises $p_{t,i}$ for them; higher $P_i$ and better top-k odds raise $n_i$ further. The loop closes: $\lVert \nabla W^{(i)} \rVert \propto n_i$ is the multiplicative engine, and (10) has no counterterm.

The fixed points are degenerate: a small subset of experts absorbs all tokens while the rest starve toward zero routing probability — *dead experts*. In the worst case the effective model is 2 of 8 experts per layer, i.e. the stored capacity collapses to $k/(E+s) = 2/9 \approx 22\%$ of the FFN parameters actually used, yet **all** 9 experts' parameters still occupy memory and are updated (with ~zero gradient) — wasted capacity and wasted optimizer state. Worse, dead experts never recover: with $f_i = 0$ and $P_i \approx 0$, both the task gradient and the aux gradient (9) vanish for expert $i$, a stable absorbing state.

The aux loss is the counterterm to (10): its gradient (9) is proportional to $p_{t,i}(f_i - \langle f\rangle_p)$ and is *largest exactly when the loop is strongest* — an over-loaded expert gets pushed down every step, not just when collapse is complete. The balancing force also has the right symmetry: it opposes imbalance regardless of which experts are good, so it cannot accidentally shield a genuinely poor expert. The interplay is a saddle: the CE gradient wants concentration on the best experts (specialisation), the aux gradient wants uniformity; α sets where the saddle sits, and the warmup (3,000 of 61,000 steps, 4.9%, deliberately above the 2–5% MoE-standard band per the config comment) lets the router find stable structure before the balancing pressure matters. The guard for the *observable* claim — routing must not collapse over a large batch — is `tests/test_moe.py:test_moe_layer_routes_to_all_experts_over_batch`.

## 8. The shared expert

One expert ($s = 1$) is outside the routing pool entirely: `models/moe.py:MoELayer` adds its output to every token unconditionally, and `tests/test_moe.py:test_moe_layer_shared_expert_active` proves it by zeroing its weights and observing a change. Three reasons:

1. **Guaranteed capacity.** Every token, no matter what the gate does, receives a full $3.54$M-parameter transformation. The routed contribution can be thin (near-uniform $p_t$, small top-2 mass) without the FFN output vanishing; the shared expert is the floor under the layer's representational power. It also bounds the worst-case effect of a bad router: a mistrusted gate can starve routed experts, but not the shared one.
2. **Routing-noise reduction.** The shared output is deterministic in $x_t$ — no gate variance. Writing $y_t = S(x_t) + \sum_{i \in \mathcal{I}_t} w_{t,i} E_i(x_t)$, the stochastic part (the routed sum, whose weights and membership vary across tokens) is superimposed on a stable baseline rather than being the whole signal. This damps the per-token gradient variance that discrete routing injects into the FFN path.
3. **Universal-feature offload.** Features that nearly every token needs (syntactic scaffolding, scale calibration) would otherwise be replicated across all 8 routed experts — 8 redundant copies consuming capacity that could hold distinct specialisations. Routing them through the gate would also waste top-2 slots on low-information choices. One always-on expert absorbs the "average" transform; the routed pool specialises in the residual.

The shared expert costs FLOPs on every token — that is exactly why it is counted in the active budget of section 4 and why the sparsity ratio is 49.2% active rather than 25% (which would be 2-of-8 with no shared expert). DeepSeek-V3-style "1 shared + top-k" is the lineage.

## 9. Grouped and stacked dispatch

### 9.1 The gather layout

Routing produces, per token, $k$ (expert, weight) pairs. Execution needs each expert's input tokens *contiguous* so expert $e$'s weight matrix multiplies one block $X_e$ rather than $k$ scattered rows. The standard layout (both dispatch paths in `models/moe.py:MoELayer`):

1. Flatten the $Nk$ slots: `flat_idx = indices.reshape(-1)`, with `token_ids = arange(N).repeat_interleave(k)` remembering each slot's token.
2. Stable-sort slots by expert id: `order = torch.argsort(flat_idx, stable=True)`. The `stable=True` is load-bearing — equal expert ids keep their token order, so the layout is a deterministic function of the routing (guarded by `tests/test_moe.py:test_moe_dispatch_is_deterministic`).
3. Count and offset: `expert_counts = bincount(flat_idx, minlength=E)`; `expert_offsets` from the cumulative sum. Expert $e$'s chunk is `x_sorted[off_e : off_e + cnt_e]` where `x_sorted = flat[sorted_token_ids]`.
4. Compute per expert, scale by the sorted weights, and scatter back: `out.index_add(0, sorted_token_ids, weighted)` — an exact scatter (each token receives exactly $k$ contributions).

### 9.2 Why not a single bmm

The chunked form is a *grouped GEMM*: $E$ matrix products $X_e W_e^{\top}$ with variable row counts $n_e$. A single `torch.bmm` requires equal batch dimensions, so the stacked-tensor formulation would have to pad every chunk to $\max_e n_e$ rows. The FLOP cost is then

$$
C_{\text{bmm}} = 6\,d\,d_{\text{ff}} \sum_{e=1}^{E} \max_e n_e
\;\ge\; 6\,d\,d_{\text{ff}} \sum_e n_e = 6\,d\,d_{\text{ff}}\, Nk = C_{\text{exact}}, \tag{11}
$$

with equality only when routing is perfectly balanced. Padding waste scales with imbalance — precisely the regime the aux loss fights. The per-expert loop (`models/moe.py:MoELayer._dispatch_vectorized`) is exact: $\sum_e n_e = Nk$ always, because every slot has an owner; the price is $E$ small kernel launches per layer.

### 9.3 Launch-bound economics and the Triton path

The small-chunk regime is GPU-hostile: a GEMM's compute time is $t_c = 6\,d\,d_{\text{ff}}\,n_e / P$ (achieved rate $P$), and with per-launch latency $t_l$ the overhead fraction is

$$
\frac{t_l}{t_c} = \frac{t_l\, P}{6\,d\,d_{\text{ff}}\, n_e} \propto \frac{1}{n_e}. \tag{12}
$$

Halving the chunk size doubles the relative overhead; small chunks are the MoE norm (4096 tokens over 8 experts ≈ 512-token chunks before top-2 splitting). The W1/W3+SiLU stage is two of the expert's three GEMMs, so `models/moe.py:MoELayer._dispatch_triton` hands that stage to a single fused grouped-GEMM launch, `models/moe_triton.py:triton_moe_w1w3_silu`, which consumes the same counts/offsets layout natively: per-expert masked tiles of 16 tokens (no padding, no per-expert launch), W2 staying in PyTorch. The numeric launch-vs-compute ratios, the activation-traffic savings ($\approx 96$ MiB/layer at 4096 tokens), and the tile-shape derivation are in [triton programming](triton_programming.md) §6–7, with A100 rates marked `[INFERENCE]` there (`.benchmarks/` is empty).

The two paths are numerically interchangeable — the Triton path is opt-in via `ModelConfig.moe_dispatch` (default `"stacked"`) and its parity is pinned by the GPU-gated tests in `tests/test_moe_triton.py`; on CPU-only machines those tests skip (repo-wide: 190 passed / 2 skipped).

## 10. Code walkthrough

### 10.1 The expert — `models/moe.py:SwiGLUExpert` / `models/moe.py:SwiGLUExpert.forward`

Three bias-free `nn.Linear`s: `w1` ($d \to d_{\text{ff}}$), `w3` ($d \to d_{\text{ff}}$), `w2` ($d_{\text{ff}} \to d$). The forward is the equation of section 4.1 verbatim:

```python
def forward(self, x: torch.Tensor) -> torch.Tensor:
    return self.w2(F.silu(self.w1(x)) * self.w3(x))
```

`models/moe.py:SwiGLUExpert.forward` is the only place the gated activation is computed in the stacked path; `tests/test_moe.py:test_swiglu_expert_shape` and `test_swiglu_expert_grad_flow` pin its shape and its three weight gradients.

### 10.2 The router — `models/moe.py:MoERouter` / `models/moe.py:MoERouter.forward`

```python
def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    logits = self.gate(x)
    all_probs_f32 = F.softmax(logits.float(), dim=-1)
    topk_weights, topk_indices = all_probs_f32.topk(self.n_activated, dim=-1)
    topk_weights = topk_weights / topk_weights.sum(dim=-1, keepdim=True).clamp(min=1e-6)
    return topk_indices, topk_weights.to(x.dtype), logits
```

Line by line against section 5: `self.gate` is the bias-free $\mathbb{R}^{E \times d}$ map of (4); `.float()` before `softmax` is the FP32-island rule ([numerics](numerics.md) §8.3); `topk` implements $\mathcal{I}_t$; the division implements the renormalisation of (5) with `clamp(min=1e-6)` guarding the (theoretical) zero-mass case; `topk_weights.to(x.dtype)` casts the FP32 weights back to the activation dtype (BF16 under autocast) while the *indices* stay integer. The raw `logits` are returned so `models/moe.py:aux_load_balancing_loss` recomputes the same distribution from the un-softmaxed values — the aux path never depends on which weights the forward consumed. `tests/test_moe.py:test_router_topk_indices`, `test_router_weights_sum_to_one`, `test_router_indices_in_range`, `test_router_grad_flow` guard each contract.

### 10.3 The aux loss — `models/moe.py:aux_load_balancing_loss`

```python
probs_f32 = F.softmax(all_logits.float(), dim=-1)
N = probs_f32.size(0)
topk_idx = probs_f32.topk(n_activated, dim=-1).indices.flatten()
f = torch.bincount(topk_idx, minlength=n_experts).to(torch.float32) / float(N * n_activated)
P = probs_f32.mean(dim=0)
return (n_experts * (f * P).sum()).to(all_logits.dtype)
```

This is equation (6)–(7) line for line: `bincount` over the flattened top-2 indices counts the $Nk$ slots per expert, divided by $Nk$ → $f_i$; the mean over the batch dimension → $P_i$; the product-sum scaled by `n_experts` ($E$) → $\mathcal{L}_{\text{aux}}$. Two details matter. First, `topk_idx` is produced from `probs_f32.topk(...).indices` — indices are non-differentiable, so the loss's gradient flows only through $P$, exactly the mechanism of (9). Second, everything is computed in FP32 and cast back at the end, so saturated BF16 logits cannot zero out the $P_i$ statistics (`tests/test_moe.py:test_aux_loss_robust_to_bf16_saturation` builds logits of $\pm 100$ and requires a nonzero loss; `test_aux_loss_finite_and_nonneg` and `test_aux_loss_grad_flow` pin finiteness and differentiability).

### 10.4 The layer — `models/moe.py:MoELayer` / `models/moe.py:MoELayer.forward`

```python
def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    B, T, D = x.shape
    flat = x.view(-1, D)
    N = flat.size(0)

    indices, weights, all_logits = self.router(flat)
    if self.moe_dispatch == "triton_grouped":
        out = self._dispatch_triton(flat, indices, weights)
    else:
        out = self._dispatch_vectorized(flat, indices, weights)
    aux_loss = aux_load_balancing_loss(all_logits, self.n_routed, self.n_activated)
    if self.shared_experts is not None:
        shared_out = sum(e(flat) for e in self.shared_experts)
        out = out + shared_out

    return out.view(B, T, D), aux_loss
```

The sequence is exactly the theory: flatten tokens (the aux loss and router operate on the *whole* batch, as (6) requires — batch-level statistics, not per-sequence); route; dispatch (section 9); compute the aux loss from the same logits; add the shared expert's output to every token (section 8); return the aux loss as a second output. The layer is constructed from the config in `models/moe.py:MoELayer.__init__` (`n_routed`, `n_activated`, `n_shared` read off `ModelConfig`). Integration: `models/transformer.py:GPTOSSBlock.forward` returns `(x + moe_out, aux_loss)` and `models/transformer.py:GPTOSS.forward` stacks the 12 per-layer aux losses and takes the mean, so the scalar that reaches `training/pretrain.py:main` is the layer average; `models/transformer.py:GPTOSS.num_parameters` reproduces the section-4 accounting analytically (it computes active expert params as $(k + s) \cdot 3 d d_{\text{ff}}$ per layer, skipping stored-but-idle experts).

### 10.5 Dispatch — `models/moe.py:MoELayer._dispatch_vectorized` and `_dispatch_triton`

Both implement the layout of section 9.1: flatten slots, `torch.argsort(flat_idx, stable=True)`, `bincount`/`cumsum` for `expert_counts`/`expert_offsets`, per-expert contiguous chunks. The vectorized path runs `self.experts[e](expert_in)` per chunk and scatters with `out.index_add(0, chunk_tokens, expert_out * chunk_weights)`. The Triton path replaces the per-expert W1/W3+SiLU with one `triton_moe_w1w3_silu` call over stacked weights and the same counts/offsets, then finishes W2, weight-scaling, and `index_add_` in PyTorch. `tests/test_moe.py:test_moe_layer_dispatch_correct` recomputes the forward by hand per token and requires `allclose(atol=1e-4)` — the ground truth that both dispatch paths and the Triton kernel must match.

## 11. Pitfalls + verify

| Pitfall | Symptom | Guard |
|---|---|---|
| BF16 router softmax | Small $p_{t,i}$ underflow to 0; renormalised weights and $P_i$ statistics distorted; aux loss goes blind to imbalance | FP32 softmax in `models/moe.py:MoERouter.forward` and `models/moe.py:aux_load_balancing_loss`; `pytest tests/test_moe.py -v` → `test_aux_loss_robust_to_bf16_saturation` |
| Renormalisation divide-by-zero | NaN routed output if top-k mass is 0 | `clamp(min=1e-6)` in `MoERouter.forward`; `test_router_weights_sum_to_one` |
| Top-1 gate | Router gets zero task gradient (weights are constant 1); winner-take-all collapse | $k = 2$ keeps logit dependence in the weights; `test_router_grad_flow`, `test_moe_layer_grad_flow` |
| No aux loss (α = 0) | Rich-get-richer: routing concentrates, dead experts, effective capacity $\to k/(E{+}s) = 22\%$ | α = 0.01 objective in `training/pretrain.py:main`; `test_aux_loss_low_for_uniform` (1.0 vs ~4.0), `test_moe_layer_routes_to_all_experts_over_batch` (mass fraction < 0.9) |
| Non-deterministic dispatch | Same input, different output across runs (unordered gather of tied expert ids) | `torch.argsort(flat_idx, stable=True)` in both dispatch paths; `test_moe_dispatch_is_deterministic` (`torch.equal`) |
| Non-finite aux loss | NaN statistics poison the total loss | FP32 `bincount`/`mean`; `test_aux_loss_finite_and_nonneg`, `test_aux_loss_grad_flow` |
| Triton missing on CPU | Kernel call fails | No silent fallback: `triton_grouped` raises `ImportError` (`models/moe_triton.py:triton_moe_w1w3_silu`); CPU suite uses `"stacked"` and skips only the 2 GPU-gated Triton tests |
| Shared expert accidentally routed | Every token loses its guaranteed baseline | Shared output added unconditionally in `MoELayer.forward`; `test_moe_layer_shared_expert_active` (zeroing shared weights must change output) |

**Verification** (CPU-runnable, no GPU required):

```
pytest tests/test_moe.py -v
```

covers every row above — 190 tests pass repo-wide / 2 skipped (the skips are the GPU-gated Triton parity tests, expected on CPU). Numerical-equality with the Triton path is additionally pinned by `pytest tests/test_moe_triton.py -v` on an sm_75+ GPU (GPU tests auto-skip on CPU). Everything derived in this chapter — (1)–(12) — is exact arithmetic on the code and config, except where marked `[INFERENCE]`: no pretraining run exists yet, so the ≥85% passkey @128K and the A100 time/MFU figures are targets and estimates, not measurements.

---

Related: implementation reference [moe.md](../moe.md); fused kernel [triton programming](triton_programming.md); FP32 islands [numerics](numerics.md); training objective and warmup [training.md](../training.md); model layout [architecture.md](../architecture.md); hardware budget [foundations.md](../foundations.md).

<!-- docs:verified 2026-08-04 · 5da1a80 -->
