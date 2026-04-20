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
        router_logits, _, _, load = self.gate(tokens)
        routing_probs = torch.softmax(router_logits, dim=-1)
        top1_values, selected_experts = torch.topk(routing_probs, k=1, dim=-1)

        batch_size, num_tokens, _ = tokens.shape
        weighted_update = torch.zeros(
            batch_size,
            num_tokens,
            self.output_dim * 3,
            device=tokens.device,
            dtype=tokens.dtype,
        )
        expert_weights = torch.zeros_like(router_logits)
        expert_weights.scatter_(dim=-1, index=selected_experts, src=top1_values)

        # Sparse Top-1 routing: only execute the selected LoRA expert for each sample.
        for expert_index, (lora_a, lora_b, scale, dropout) in enumerate(
            zip(self.lora_a, self.lora_b, self.scaling, self.dropout)
        ):
            sample_mask = selected_experts.squeeze(-1) == expert_index
            if not sample_mask.any():
                continue

            expert_input = tokens[sample_mask]
            update = lora_b(lora_a(dropout(expert_input))) * scale
            update = top1_values[sample_mask].view(-1, 1, 1) * update
            weighted_update[sample_mask] = update

        aux = LoRAAuxiliaryOutput(
            router_logits=router_logits,
            selected_experts=selected_experts,
            expert_weights=expert_weights,
            importance=expert_weights.sum(0),
            load=load,
        )
        return weighted_update, aux
