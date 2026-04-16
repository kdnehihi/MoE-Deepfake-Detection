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

    def __init__(self, load_balance_weight: float) -> None:
        super().__init__()
        self.load_balance_weight = load_balance_weight

    @staticmethod
    def _cv_squared(values: Tensor) -> Tensor:
        eps = 1e-10
        if values.numel() <= 1:
            return torch.zeros(1, device=values.device, dtype=values.dtype).squeeze(0)
        return values.float().var(unbiased=False) / (values.float().mean().pow(2) + eps)

    def forward(self, logits: Tensor, labels: Tensor, aux: ModelAuxiliaryOutput) -> LossOutput:
        classification = F.cross_entropy(logits, labels)

        balance_terms: list[Tensor] = []
        for block_aux in aux.blocks:
            lora_importance = block_aux.lora.qkv.importance
            lora_load = block_aux.lora.qkv.load
            adapter_importance = block_aux.adapter.importance
            adapter_load = block_aux.adapter.load

            balance_terms.append(self._cv_squared(lora_importance))
            balance_terms.append(self._cv_squared(lora_load))
            balance_terms.append(self._cv_squared(adapter_importance))
            balance_terms.append(self._cv_squared(adapter_load))

        if balance_terms:
            load_balance = torch.stack(balance_terms).mean()
        else:
            load_balance = torch.zeros(1, device=logits.device, dtype=logits.dtype).squeeze(0)

        total = classification + (self.load_balance_weight * load_balance)
        return LossOutput(total=total, classification=classification, load_balance=load_balance)
