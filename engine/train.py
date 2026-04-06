"""Training loop scaffolding for the MoE-FFD detector."""

from __future__ import annotations

from dataclasses import dataclass

from torch import nn
from torch.utils.data import DataLoader

from engine.loss import MoEFFDLoss
from utils.config import OptimizerConfig, TrainConfig


@dataclass(slots=True)
class TrainerState:
    epoch: int = 0
    global_step: int = 0


class Trainer:
    """Encapsulates optimizer, AMP, and epoch orchestration."""

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        criterion: MoEFFDLoss,
        train_config: TrainConfig,
        optimizer_config: OptimizerConfig,
    ) -> None:
        self.model = model
        self.train_loader = train_loader
        self.criterion = criterion
        self.train_config = train_config
        self.optimizer_config = optimizer_config
        self.state = TrainerState()

    def build_optimizer(self):
        raise NotImplementedError("Trainer.build_optimizer is implemented in Step 5.")

    def train_epoch(self):
        raise NotImplementedError("Trainer.train_epoch is implemented in Step 5.")

