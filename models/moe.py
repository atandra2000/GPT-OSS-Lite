"""MoE FFN for GPT-OSS-Lite: top-2 of 8 routed experts + 1 shared expert."""
import torch
import torch.nn as nn
import torch.nn.functional as F


class SwiGLUExpert(nn.Module):
    """Single SwiGLU expert: W2(silu(W1(x)) * W3(x))."""

    def __init__(self, dim: int, inter_dim: int):
        super().__init__()
        self.w1 = nn.Linear(dim, inter_dim, bias=False)
        self.w2 = nn.Linear(inter_dim, dim, bias=False)
        self.w3 = nn.Linear(dim, inter_dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class MoERouter(nn.Module):
    """Top-k gating network."""

    def __init__(self, d_model: int, n_experts: int, n_activated: int):
        super().__init__()
        self.d_model = d_model
        self.n_experts = n_experts
        self.n_activated = n_activated
        self.gate = nn.Linear(d_model, n_experts, bias=False)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        logits = self.gate(x)
        all_probs_f32 = F.softmax(logits.float(), dim=-1)
        topk_weights, topk_indices = all_probs_f32.topk(self.n_activated, dim=-1)
        topk_weights = topk_weights / topk_weights.sum(dim=-1, keepdim=True).clamp(min=1e-6)
        return topk_indices, topk_weights.to(x.dtype), logits


def aux_load_balancing_loss(
    all_logits: torch.Tensor,
    n_experts: int,
    n_activated: int,
) -> torch.Tensor:
    """Standard MoE load-balancing loss (Switch Transformer / GShard style)."""
    probs_f32 = F.softmax(all_logits.float(), dim=-1)
    N = probs_f32.size(0)
    topk_idx = probs_f32.topk(n_activated, dim=-1).indices.flatten()
    f = torch.bincount(topk_idx, minlength=n_experts).to(torch.float32) / float(N * n_activated)
    P = probs_f32.mean(dim=0)
    return (n_experts * (f * P).sum()).to(all_logits.dtype)


class MoELayer(nn.Module):
    """GPT-OSS-Lite MoE layer: top-2 routed + 1 shared expert."""

    def __init__(self, cfg):
        super().__init__()
        self.d_model = cfg.d_model
        self.ffn_dim = cfg.ffn_dim
        self.n_routed = cfg.n_routed_experts
        self.n_activated = cfg.n_activated_experts
        self.n_shared = cfg.n_shared_experts
        self.moe_dispatch = getattr(cfg, "moe_dispatch", "stacked")
        self.router = MoERouter(self.d_model, self.n_routed, self.n_activated)
        self.experts = nn.ModuleList([
            SwiGLUExpert(self.d_model, self.ffn_dim) for _ in range(self.n_routed)
        ])
        if self.n_shared > 0:
            self.shared_experts = nn.ModuleList([
                SwiGLUExpert(self.d_model, self.ffn_dim) for _ in range(self.n_shared)
            ])
        else:
            self.shared_experts = None
    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward returning ``(output (B,T,D), aux_loss scalar)``."""
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

    def _dispatch_triton(
        self,
        flat: torch.Tensor,
        indices: torch.Tensor,
        weights: torch.Tensor,
    ) -> torch.Tensor:
        """Triton-grouped MoE dispatch: one kernel for W1+W3+silu, then W2 in PyTorch."""
        from .moe_triton import triton_moe_w1w3_silu

        N = flat.size(0)
        flat_idx = indices.reshape(-1)
        flat_w = weights.reshape(-1)
        token_ids = torch.arange(N, device=flat.device).repeat_interleave(indices.size(1))

        order = torch.argsort(flat_idx, stable=True)
        sorted_token_ids = token_ids[order]
        sorted_weights = flat_w[order]
        sorted_expert_ids = flat_idx[order]
        x_sorted = flat[sorted_token_ids]

        expert_counts = torch.bincount(flat_idx, minlength=self.n_routed)
        expert_offsets = torch.cat([
            torch.zeros(1, dtype=expert_counts.dtype, device=flat.device),
            expert_counts.cumsum(0)[:-1],
        ])

        W1_stack = torch.stack([e.w1.weight for e in self.experts], dim=0)
        W3_stack = torch.stack([e.w3.weight for e in self.experts], dim=0)
        W2_stack = torch.stack([e.w2.weight for e in self.experts], dim=0)

        gated_sorted = triton_moe_w1w3_silu(
            x_sorted, sorted_expert_ids, expert_counts, expert_offsets,
            W1_stack, W3_stack,
        )

        out_sorted = torch.bmm(
            gated_sorted.unsqueeze(0),
            W2_stack[sorted_expert_ids],
        ).squeeze(0)

        out_sorted = out_sorted * sorted_weights.unsqueeze(-1)
        out = torch.zeros_like(flat)
        out.index_add_(0, sorted_token_ids, out_sorted)
        return out

    def _dispatch_vectorized(
        self,
        flat: torch.Tensor,
        indices: torch.Tensor,
        weights: torch.Tensor,
    ) -> torch.Tensor:
        """Vectorized expert dispatch using stacked-expert bmm."""
        N = flat.size(0)
        flat_idx = indices.reshape(-1)
        flat_w = weights.reshape(-1)
        token_ids = torch.arange(N, device=flat.device).repeat_interleave(indices.size(1))

        order = torch.argsort(flat_idx, stable=True)
        sorted_token_ids = token_ids[order]
        sorted_weights = flat_w[order]
        sorted_expert_ids = flat_idx[order]

        expert_counts = torch.bincount(flat_idx, minlength=self.n_routed)
        expert_offsets = torch.cat([
            torch.zeros(1, dtype=expert_counts.dtype, device=flat.device),
            expert_counts.cumsum(0)[:-1],
        ])

        out = torch.zeros_like(flat)
        counts_cpu = expert_counts.tolist()
        offsets_cpu = expert_offsets.tolist()
        for e in range(self.n_routed):
            cnt = counts_cpu[e]
            if cnt == 0:
                continue
            start = offsets_cpu[e]
            end = start + cnt
            chunk_tokens = sorted_token_ids[start:end]
            chunk_weights = sorted_weights[start:end].unsqueeze(-1)
            expert_in = flat[chunk_tokens]
            expert_out = self.experts[e](expert_in)
            out = out.index_add(0, chunk_tokens, expert_out * chunk_weights)
        return out