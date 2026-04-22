"""Metrics interfaces used during evaluation."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable

import numpy as np
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


def binary_average_precision(probabilities: Tensor, labels: Tensor) -> float:
    try:
        from sklearn.metrics import average_precision_score
    except ImportError:
        return 0.0

    labels_np = labels.detach().cpu().numpy()
    probs_np = probabilities.detach().cpu().numpy()

    if len(set(labels_np.tolist())) < 2:
        return 0.0
    return float(average_precision_score(labels_np, probs_np))


def binary_eer(probabilities: Tensor, labels: Tensor) -> float:
    try:
        from sklearn.metrics import roc_curve
    except ImportError:
        return 0.0

    labels_np = labels.detach().cpu().numpy()
    probs_np = probabilities.detach().cpu().numpy()

    if len(set(labels_np.tolist())) < 2:
        return 0.0

    fpr, tpr, _ = roc_curve(labels_np, probs_np)
    fnr = 1.0 - tpr
    differences = np.abs(fpr - fnr)
    min_index = int(np.argmin(differences))

    if min_index == 0 or differences[min_index] == 0.0:
        eer = (fpr[min_index] + fnr[min_index]) / 2.0
        return float(eer)

    previous_index = min_index - 1
    x0 = fpr[previous_index] - fnr[previous_index]
    x1 = fpr[min_index] - fnr[min_index]
    y0 = fpr[previous_index]
    y1 = fpr[min_index]

    if x1 == x0:
        eer = (fpr[min_index] + fnr[min_index]) / 2.0
        return float(eer)

    weight = -x0 / (x1 - x0)
    eer = y0 + weight * (y1 - y0)
    return float(eer)


def aggregate_video_scores(
    probabilities: Tensor,
    labels: Tensor,
    video_ids: Iterable[str],
    topk: int = 5,
) -> tuple[Tensor, Tensor]:
    grouped_scores: dict[str, list[float]] = {}
    grouped_labels: dict[str, int] = {}

    for probability, label, video_id in zip(probabilities.tolist(), labels.tolist(), video_ids):
        grouped_scores.setdefault(str(video_id), []).append(float(probability))
        grouped_labels.setdefault(str(video_id), int(label))

    video_scores: list[float] = []
    video_labels: list[int] = []
    for video_id, scores in grouped_scores.items():
        pooled = topk_pooling(torch.tensor(scores, dtype=torch.float32), k=topk)
        video_scores.append(float(pooled.item()))
        video_labels.append(grouped_labels[video_id])

    return (
        torch.tensor(video_scores, dtype=torch.float32),
        torch.tensor(video_labels, dtype=torch.long),
    )
