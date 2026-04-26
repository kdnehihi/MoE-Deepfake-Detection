"""Loss definitions for binary classification and MoE regularization."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from models.model import ModelAuxiliaryOutput


@dataclass(slots=True)
class LossOutput:
    total: Tensor
    classification: Tensor
    load_balance: Tensor


class MoEFFDLoss(nn.Module):
    """Combines cross-entropy with the MoE load-balancing objective."""

    def __init__(
        self,
        load_balance_weight: float,
        lora_balance_scale: float = 200.0,
        adapter_balance_scale: float = 1.0,
    ) -> None:
        super().__init__()
        self.load_balance_weight = load_balance_weight
        self.lora_balance_scale = lora_balance_scale
        self.adapter_balance_scale = adapter_balance_scale

    @staticmethod
    def _cv_squared(values: Tensor) -> Tensor:
        eps = 1e-10
        if values.numel() <= 1:
            return torch.zeros(1, device=values.device, dtype=values.dtype).squeeze(0)
        return values.float().var() / (values.float().mean().pow(2) + eps)

    def forward(self, logits: Tensor, labels: Tensor, aux: ModelAuxiliaryOutput) -> LossOutput:
        classification = F.cross_entropy(logits, labels)

        lora_terms: list[Tensor] = []
        adapter_terms: list[Tensor] = []
        for block_aux in aux.blocks:
            lora_importance = block_aux.lora.qkv.importance
            lora_load = block_aux.lora.qkv.load
            adapter_importance = block_aux.adapter.importance
            adapter_load = block_aux.adapter.load

            lora_terms.append(self._cv_squared(lora_importance))
            lora_terms.append(self._cv_squared(lora_load))
            adapter_terms.append(self._cv_squared(adapter_importance))
            adapter_terms.append(self._cv_squared(adapter_load))

        zero = torch.zeros(1, device=logits.device, dtype=logits.dtype).squeeze(0)
        lora_balance = torch.stack(lora_terms).mean() if lora_terms else zero
        adapter_balance = torch.stack(adapter_terms).mean() if adapter_terms else zero
        # FIX START
        load_balance = lora_balance + adapter_balance
        # FIX END

        total = classification + (self.load_balance_weight * load_balance)
        return LossOutput(total=total, classification=classification, load_balance=load_balance)
