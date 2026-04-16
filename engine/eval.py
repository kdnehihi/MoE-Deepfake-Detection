"""Evaluation loop scaffolding."""

from __future__ import annotations

import torch
from torch import nn
from torch.utils.data import DataLoader

from engine.loss import MoEFFDLoss
from utils.metrics import BinaryClassificationMetrics, binary_accuracy, binary_auc

class Evaluator:
    """Runs validation or test passes and aggregates metrics."""

    def __init__(self, model: nn.Module, data_loader: DataLoader, criterion: MoEFFDLoss, device: str) -> None:
        self.model = model
        self.data_loader = data_loader
        self.criterion = criterion
        self.device = device

    def evaluate(self):
        self.model.eval()
        total_loss = 0.0
        total_classification = 0.0
        total_load_balance = 0.0
        num_batches = 0

        all_logits: list[torch.Tensor] = []
        all_labels: list[torch.Tensor] = []

        with torch.no_grad():
            for images, labels in self.data_loader:
                images = images.to(self.device)
                labels = labels.to(self.device)

                logits, aux = self.model(images)
                loss_output = self.criterion(logits, labels, aux)

                total_loss += loss_output.total.item()
                total_classification += loss_output.classification.item()
                total_load_balance += loss_output.load_balance.item()
                num_batches += 1

                all_logits.append(logits.detach().cpu())
                all_labels.append(labels.detach().cpu())

        if num_batches == 0:
            return {
                "loss": 0.0,
                "classification_loss": 0.0,
                "load_balance_loss": 0.0,
                "metrics": BinaryClassificationMetrics(),
            }

        logits = torch.cat(all_logits, dim=0)
        labels = torch.cat(all_labels, dim=0)
        probabilities = torch.softmax(logits, dim=1)[:, 1]

        metrics = BinaryClassificationMetrics(
            accuracy=binary_accuracy(logits, labels),
            auc=binary_auc(probabilities, labels),
            ap=0.0,
            eer=0.0,
        )

        return {
            "loss": total_loss / num_batches,
            "classification_loss": total_classification / num_batches,
            "load_balance_loss": total_load_balance / num_batches,
            "metrics": metrics,
        }
