"""Mixture-of-Experts LoRA interface for attention adaptation."""

from __future__ import annotations

from dataclasses import dataclass

from torch import Tensor, nn

from models.gating import TopKGating
from utils.config import GatingConfig, LoRAExpertConfig


@dataclass(slots=True)
class LoRAAuxiliaryOutput:
    router_logits: Tensor
    selected_experts: Tensor
    expert_weights: Tensor


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
        self.experts = experts
        self.gate = TopKGating(input_dim=input_dim, num_experts=len(experts), config=gating_config)

    def forward(self, tokens: Tensor) -> tuple[Tensor, LoRAAuxiliaryOutput]:
        raise NotImplementedError("MoELoRALayer.forward is implemented in Step 3.")

