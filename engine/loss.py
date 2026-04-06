"""Loss definitions for binary classification and MoE regularization."""

from __future__ import annotations

from dataclasses import dataclass

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

    def forward(self, logits: Tensor, labels: Tensor, aux: ModelAuxiliaryOutput) -> LossOutput:
        raise NotImplementedError("MoEFFDLoss.forward is implemented in Step 5.")

