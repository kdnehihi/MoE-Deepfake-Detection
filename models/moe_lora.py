"""Mixture-of-Experts LoRA interface for attention adaptation."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn

from models.gating import TopKGating
from utils.config import GatingConfig, LoRAExpertConfig


@dataclass(slots=True)
class LoRAAuxiliaryOutput:
    router_logits: Tensor
    selected_experts: Tensor
    expert_weights: Tensor
    importance: Tensor
    load: Tensor


class MoELoRALayer(nn.Module):
    """
    Wraps attention projections with sparse LoRA experts.

    Each expert owns its own low-rank update. The gating network decides which
    expert is active for each sample, and only the selected update is combined
    with the frozen attention projection.
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        experts: list[LoRAExpertConfig],
        gating_config: GatingConfig,
    ) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.expert_configs = experts
        self.gate = TopKGating(input_dim=input_dim, num_experts=len(experts), config=gating_config)
        self.lora_a = nn.ModuleList(
            nn.Linear(input_dim, expert.rank, bias=False) for expert in experts
        )
        self.lora_b = nn.ModuleList(
            nn.Linear(expert.rank, output_dim * 3, bias=False) for expert in experts
        )
        self.scaling = [expert.alpha / float(expert.rank) for expert in experts]
        self.dropout = nn.ModuleList(nn.Dropout(expert.dropout) for expert in experts)

        for lora_b in self.lora_b:
            nn.init.zeros_(lora_b.weight)

    def forward(self, tokens: Tensor) -> tuple[Tensor, LoRAAuxiliaryOutput]:
        router_logits, selected_experts, expert_weights, load = self.gate(tokens)

        expert_outputs = []
        for lora_a, lora_b, scale, dropout in zip(self.lora_a, self.lora_b, self.scaling, self.dropout):
            update = lora_b(lora_a(dropout(tokens))) * scale
            expert_outputs.append(update)

        stacked_updates = torch.stack(expert_outputs, dim=1)
        weighted_update = torch.einsum("be,benc->bnc", expert_weights, stacked_updates)

        aux = LoRAAuxiliaryOutput(
            router_logits=router_logits,
            selected_experts=selected_experts,
            expert_weights=expert_weights,
            importance=expert_weights.sum(0),
            load=load,
        )
        return weighted_update, aux
