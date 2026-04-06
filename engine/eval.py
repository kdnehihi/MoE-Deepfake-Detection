"""Evaluation loop scaffolding."""

from __future__ import annotations

from torch import nn
from torch.utils.data import DataLoader


class Evaluator:
    """Runs validation or test passes and aggregates metrics."""

    def __init__(self, model: nn.Module, data_loader: DataLoader) -> None:
        self.model = model
        self.data_loader = data_loader

    def evaluate(self):
        raise NotImplementedError("Evaluator.evaluate is implemented in Step 5.")

