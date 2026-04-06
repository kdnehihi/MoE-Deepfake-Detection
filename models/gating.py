"""Top-k gating module used by the LoRA and adapter expert mixtures."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from utils.config import GatingConfig


class TopKGating(nn.Module):
    """
    Routes each sample to a sparse set of experts.

    In the target method, gating consumes pooled transformer features and
    selects the most relevant expert with top-k routing where k=1.
    """

    def __init__(self, input_dim: int, num_experts: int, config: GatingConfig) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.num_experts = num_experts
        self.config = config
        self.pool = nn.Identity()
        self.mlp = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, config.hidden_dim),
            nn.GELU(),
            nn.Linear(config.hidden_dim, num_experts),
        )

    def _pool_tokens(self, tokens: Tensor) -> Tensor:
        if tokens.ndim == 2:
            return tokens
        if self.config.use_cls_token:
            return tokens[:, 0]
        return tokens.mean(dim=1)

    def forward(self, tokens: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        pooled = self._pool_tokens(tokens)
        router_logits = self.mlp(self.pool(pooled))

        top_k = min(self.config.top_k, self.num_experts)
        topk_logits, selected_experts = torch.topk(router_logits, k=top_k, dim=-1)
        topk_weights = F.softmax(topk_logits, dim=-1)

        expert_weights = torch.zeros_like(router_logits)
        expert_weights.scatter_(dim=-1, index=selected_experts, src=topk_weights)
        return router_logits, selected_experts, expert_weights
