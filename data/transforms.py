"""Image transform builders for training and evaluation."""

from __future__ import annotations

from utils.config import DatasetSpec


def build_train_transforms(spec: DatasetSpec):
    raise NotImplementedError("Training transforms are implemented in Step 4.")


def build_eval_transforms(spec: DatasetSpec):
    raise NotImplementedError("Evaluation transforms are implemented in Step 4.")

