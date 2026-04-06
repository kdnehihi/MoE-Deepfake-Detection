"""Metrics interfaces used during evaluation."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(slots=True)
class BinaryClassificationMetrics:
    accuracy: float = 0.0
    auc: float = 0.0
    ap: float = 0.0
    eer: float = 0.0


@dataclass(slots=True)
class ExpertUsageSummary:
    counts: dict[str, int] = field(default_factory=dict)

