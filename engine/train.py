"""Training loop scaffolding for the MoE-FFD detector."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
import time

import torch
from torch import nn
from torch.utils.data import DataLoader

from engine.eval import Evaluator
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
        self.device = str(next(model.parameters()).device)
        self.optimizer = self.build_optimizer()
        self.use_amp = train_config.amp and self.device.startswith("cuda")
        self.autocast_device = "cuda" if self.device.startswith("cuda") else "cpu"
        self.scaler = torch.cuda.amp.GradScaler(enabled=self.use_amp)

    def build_optimizer(self):
        gating_parameters = []
        base_parameters = []

        for name, parameter in self.model.named_parameters():
            if not parameter.requires_grad:
                continue
            if "gate" in name or "w_gate" in name or "w_noise" in name:
                gating_parameters.append(parameter)
            else:
                base_parameters.append(parameter)

        param_groups = []
        if gating_parameters:
            param_groups.append(
                {
                    "params": gating_parameters,
                    "lr": self.optimizer_config.lr_gating,
                    "weight_decay": self.optimizer_config.weight_decay,
                }
            )
        if base_parameters:
            param_groups.append(
                {
                    "params": base_parameters,
                    "lr": self.optimizer_config.lr_base,
                    "weight_decay": self.optimizer_config.weight_decay,
                }
            )

        return torch.optim.Adam(param_groups)

    def train_epoch(self):
        self.model.train()
        total_loss = 0.0
        total_classification = 0.0
        total_load_balance = 0.0
        total_correct = 0
        total_examples = 0
        num_batches = 0
        start_time = time.time()
        total_batches = len(self.train_loader)

        for batch_index, (images, labels) in enumerate(self.train_loader, start=1):
            images = images.to(self.device)
            labels = labels.to(self.device)

            self.optimizer.zero_grad(set_to_none=True)

            with torch.autocast(device_type=self.autocast_device, enabled=self.use_amp):
                logits, aux = self.model(images)
                loss_output = self.criterion(logits, labels, aux)

            if self.use_amp:
                self.scaler.scale(loss_output.total).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                loss_output.total.backward()
                self.optimizer.step()

            predictions = logits.argmax(dim=1)
            total_correct += (predictions == labels).sum().item()
            total_examples += labels.size(0)

            total_loss += loss_output.total.item()
            total_classification += loss_output.classification.item()
            total_load_balance += loss_output.load_balance.item()
            num_batches += 1
            self.state.global_step += 1

            if batch_index == 1 or batch_index % 10 == 0 or batch_index == total_batches:
                elapsed = time.time() - start_time
                running_acc = total_correct / max(total_examples, 1)
                print(
                    f"Epoch {self.state.epoch}/{self.train_config.epochs} | "
                    f"batch {batch_index}/{total_batches} | "
                    f"loss={loss_output.total.item():.4f} | "
                    f"cls={loss_output.classification.item():.4f} | "
                    f"lb={loss_output.load_balance.item():.4f} | "
                    f"acc={running_acc:.4f} | "
                    f"time={elapsed:.1f}s"
                )

        if num_batches == 0:
            return {
                "loss": 0.0,
                "classification_loss": 0.0,
                "load_balance_loss": 0.0,
                "accuracy": 0.0,
            }

        return {
            "loss": total_loss / num_batches,
            "classification_loss": total_classification / num_batches,
            "load_balance_loss": total_load_balance / num_batches,
            "accuracy": total_correct / max(total_examples, 1),
        }

    def fit(self, val_loader: DataLoader | None = None):
        history = []
        evaluator = None
        if val_loader is not None:
            evaluator = Evaluator(self.model, val_loader, self.criterion, self.device)
        on_epoch_end: Callable[[int, dict], None] | None = getattr(self, "on_epoch_end", None)

        for epoch in range(self.train_config.epochs):
            self.state.epoch = epoch + 1
            train_stats = self.train_epoch()
            record = {"epoch": self.state.epoch, "train": train_stats}

            print(
                f"Epoch {self.state.epoch}/{self.train_config.epochs} | "
                f"train_loss={train_stats['loss']:.4f} | train_acc={train_stats['accuracy']:.4f}"
            )

            if evaluator is not None:
                val_stats = evaluator.evaluate()
                record["val"] = val_stats
                video_metrics = val_stats.get("video_metrics")
                if video_metrics is None:
                    print(
                        f"Epoch {self.state.epoch}/{self.train_config.epochs} | "
                        f"val_loss={val_stats['loss']:.4f} | "
                        f"val_acc={val_stats['metrics'].accuracy:.4f} | "
                        f"val_auc={val_stats['metrics'].auc:.4f}"
                    )
                else:
                    print(
                        f"Epoch {self.state.epoch}/{self.train_config.epochs} | "
                        f"val_loss={val_stats['loss']:.4f} | "
                        f"val_acc={val_stats['metrics'].accuracy:.4f} | "
                        f"val_auc={val_stats['metrics'].auc:.4f} | "
                        f"val_eer={val_stats['metrics'].eer:.4f} | "
                        f"val_video_auc={video_metrics.auc:.4f} | "
                        f"val_video_eer={video_metrics.eer:.4f}"
                    )

            if on_epoch_end is not None:
                on_epoch_end(self.state.epoch, record)

            history.append(record)

        return history
