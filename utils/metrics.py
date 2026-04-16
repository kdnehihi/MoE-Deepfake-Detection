"""Metrics interfaces used during evaluation."""

from __future__ import annotations

from dataclasses import dataclass, field

import torch
from torch import Tensor


@dataclass(slots=True)
class BinaryClassificationMetrics:
    accuracy: float = 0.0
    auc: float = 0.0
    ap: float = 0.0
    eer: float = 0.0


@dataclass(slots=True)
class ExpertUsageSummary:
    counts: dict[str, int] = field(default_factory=dict)


def topk_pooling(frame_scores: torch.Tensor, k: int = 5) -> torch.Tensor:
    if frame_scores.ndim > 1:
        frame_scores = frame_scores.reshape(-1)
    if frame_scores.numel() == 0:
        raise ValueError("frame_scores is empty")
    k = min(k, frame_scores.numel())
    topk_values, _ = torch.topk(frame_scores, k=k)
    return topk_values.mean()


def binary_accuracy(logits: Tensor, labels: Tensor) -> float:
    predictions = logits.argmax(dim=1)
    return (predictions == labels).float().mean().item()


def binary_auc(probabilities: Tensor, labels: Tensor) -> float:
    try:
        from sklearn.metrics import roc_auc_score
    except ImportError:
        return 0.0

    labels_np = labels.detach().cpu().numpy()
    probs_np = probabilities.detach().cpu().numpy()

    if len(set(labels_np.tolist())) < 2:
        return 0.0
    return float(roc_auc_score(labels_np, probs_np))
