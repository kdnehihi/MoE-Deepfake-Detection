"""Top-k gating module used by the LoRA and adapter expert mixtures."""

from __future__ import annotations

import torch
from torch.distributions.normal import Normal
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
        self.w_gate = nn.Parameter(torch.zeros(input_dim, num_experts))
        self.w_noise = nn.Parameter(torch.zeros(input_dim, num_experts))
        self.softplus = nn.Softplus()
        self.softmax = nn.Softmax(dim=1)
        self.normal = Normal(0.0, 1.0)

    def _pool_tokens(self, tokens: Tensor) -> Tensor:
        if tokens.ndim == 2:
            return tokens
        if self.config.use_cls_token:
            return tokens[:, 0]
        return tokens.mean(dim=1)

    @staticmethod
    def _gates_to_load(gates: Tensor) -> Tensor:
        return (gates > 0).sum(0)

    def _prob_in_top_k(
        self,
        clean_values: Tensor,
        noisy_values: Tensor,
        noise_stddev: Tensor,
        noisy_top_values: Tensor,
    ) -> Tensor:
        batch_size = clean_values.size(0)
        m = noisy_top_values.size(1)
        top_values_flat = noisy_top_values.flatten()

        threshold_positions_if_in = torch.arange(batch_size, device=clean_values.device) * m + self.config.top_k
        threshold_if_in = torch.gather(top_values_flat, 0, threshold_positions_if_in).unsqueeze(1)
        is_in = noisy_values > threshold_if_in

        threshold_positions_if_out = threshold_positions_if_in - 1
        threshold_if_out = torch.gather(top_values_flat, 0, threshold_positions_if_out).unsqueeze(1)

        prob_if_in = self.normal.cdf((clean_values - threshold_if_in) / noise_stddev)
        prob_if_out = self.normal.cdf((clean_values - threshold_if_out) / noise_stddev)
        return torch.where(is_in, prob_if_in, prob_if_out)

    def forward(self, tokens: Tensor) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        pooled = self._pool_tokens(tokens)
        clean_logits = pooled @ self.w_gate
        if self.config.noisy_gating and self.training:
            raw_noise_stddev = pooled @ self.w_noise
            noise_stddev = self.softplus(raw_noise_stddev) + self.config.noise_epsilon
            router_logits = clean_logits + (torch.randn_like(clean_logits) * noise_stddev)
        else:
            noise_stddev = None
            router_logits = clean_logits

        top_k = min(self.config.top_k, self.num_experts)
        topk_logits, selected_experts = torch.topk(router_logits, k=top_k, dim=-1)
        topk_weights = self.softmax(topk_logits)

        expert_weights = torch.zeros_like(router_logits)
        expert_weights.scatter_(dim=-1, index=selected_experts, src=topk_weights)

        if self.config.noisy_gating and self.training and top_k < self.num_experts and noise_stddev is not None:
            top_logits, _ = torch.topk(router_logits, k=min(top_k + 1, self.num_experts), dim=-1)
            load = self._prob_in_top_k(clean_logits, router_logits, noise_stddev, top_logits).sum(0)
        else:
            load = self._gates_to_load(expert_weights).to(router_logits.dtype)

        return router_logits, selected_experts, expert_weights, load
