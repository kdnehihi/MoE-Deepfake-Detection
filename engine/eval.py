"""Evaluation loop scaffolding."""

from __future__ import annotations

import torch
from torch import nn
from torch.utils.data import ConcatDataset, DataLoader, Subset

from engine.loss import MoEFFDLoss
from utils.metrics import (
    BinaryClassificationMetrics,
    aggregate_video_scores,
    binary_accuracy,
    binary_auc,
    binary_average_precision,
    binary_eer,
)


def _dataset_samples(dataset) -> list:
    if hasattr(dataset, "samples"):
        return list(dataset.samples)
    if isinstance(dataset, Subset):
        parent_samples = _dataset_samples(dataset.dataset)
        return [parent_samples[index] for index in dataset.indices]
    if isinstance(dataset, ConcatDataset):
        samples: list = []
        for child_dataset in dataset.datasets:
            samples.extend(_dataset_samples(child_dataset))
        return samples
    return []


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
        samples = _dataset_samples(self.data_loader.dataset)

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
        video_ids = [sample.video_id for sample in samples[: len(labels)]]

        metrics = BinaryClassificationMetrics(
            accuracy=binary_accuracy(logits, labels),
            auc=binary_auc(probabilities, labels),
            ap=binary_average_precision(probabilities, labels),
            eer=binary_eer(probabilities, labels),
        )

        video_metrics = BinaryClassificationMetrics()
        if video_ids and len(video_ids) == len(labels):
            video_scores, video_labels = aggregate_video_scores(probabilities, labels, video_ids, topk=5)
            video_predictions = (video_scores >= 0.5).long()
            video_metrics = BinaryClassificationMetrics(
                accuracy=(video_predictions == video_labels).float().mean().item(),
                auc=binary_auc(video_scores, video_labels),
                ap=binary_average_precision(video_scores, video_labels),
                eer=binary_eer(video_scores, video_labels),
            )

        return {
            "loss": total_loss / num_batches,
            "classification_loss": total_classification / num_batches,
            "load_balance_loss": total_load_balance / num_batches,
            "metrics": metrics,
            "video_metrics": video_metrics,
            "num_frames": int(labels.numel()),
            "num_videos": len(set(video_ids)) if video_ids else 0,
        }
