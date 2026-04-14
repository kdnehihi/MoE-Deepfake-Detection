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
        router_logits, selected_experts, expert_weights, load = self.gate(tokens)

        expert_outputs = [expert(tokens, spatial_shape) for expert in self.experts]
        stacked_outputs = torch.stack(expert_outputs, dim=1)
        weighted_output = torch.einsum("be,benc->bnc", expert_weights, stacked_outputs)

        aux = AdapterAuxiliaryOutput(
            router_logits=router_logits,
            selected_experts=selected_experts,
            expert_weights=expert_weights,
            importance=expert_weights.sum(0),
            load=load,
        )
        return weighted_output, aux
