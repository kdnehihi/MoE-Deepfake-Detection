"""Mixture-of-Experts adapter interface for local feature refinement."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn

from models.adapter_experts import ADAPTER_EXPERT_REGISTRY, BaseAdapterExpert
from models.gating import TopKGating
from utils.config import AdapterExpertConfig, GatingConfig


@dataclass(slots=True)
class AdapterAuxiliaryOutput:
    router_logits: Tensor
    selected_experts: Tensor
    expert_weights: Tensor
    importance: Tensor
    load: Tensor


class MoEAdapterLayer(nn.Module):
    """
    Applies sparse local adaptation after the transformer MLP branch.

    The adapter experts focus on complementary local texture priors, while the
    gating network decides which expert contributes for a given input sample.
    """

    def __init__(
        self,
        input_dim: int,
        experts: list[AdapterExpertConfig],
        gating_config: GatingConfig,
    ) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.expert_configs = experts
        self.gate = TopKGating(input_dim=input_dim, num_experts=len(experts), config=gating_config)
        self.experts = nn.ModuleList(
            self._build_expert(input_dim=input_dim, config=expert_config) for expert_config in experts
        )

    @staticmethod
    def _build_expert(input_dim: int, config: AdapterExpertConfig) -> BaseAdapterExpert:
        expert_cls = ADAPTER_EXPERT_REGISTRY[config.name]
        return expert_cls(input_dim=input_dim, config=config)

    def forward(self, tokens: Tensor, spatial_shape: tuple[int, int]) -> tuple[Tensor, AdapterAuxiliaryOutput]:
        router_logits, _, _, load = self.gate(tokens)
        routing_probs = torch.softmax(router_logits, dim=-1)
        top1_values, selected_experts = torch.topk(routing_probs, k=1, dim=-1)

        batch_size, num_tokens, hidden_dim = tokens.shape
        weighted_output = torch.zeros_like(tokens)
        expert_weights = torch.zeros_like(router_logits)
        expert_weights.scatter_(dim=-1, index=selected_experts, src=top1_values)

        # Sparse Top-1 routing: only execute the selected expert for each sample.
        for expert_index, expert in enumerate(self.experts):
            sample_mask = selected_experts.squeeze(-1) == expert_index
            if not sample_mask.any():
                continue

            expert_input = tokens[sample_mask]
            expert_output = expert(expert_input, spatial_shape)
            expert_output = top1_values[sample_mask].view(-1, 1, 1) * expert_output
            weighted_output[sample_mask] = expert_output

        aux = AdapterAuxiliaryOutput(
            router_logits=router_logits,
            selected_experts=selected_experts,
            expert_weights=expert_weights,
            importance=expert_weights.sum(0),
            load=load,
        )
        return weighted_output, aux
