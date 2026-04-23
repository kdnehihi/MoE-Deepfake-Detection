"""Shared training helpers for stage-wise MoE experiments."""

from __future__ import annotations

from pathlib import Path

import torch
from torch.utils.data import DataLoader

from data.dataset import build_dataset
from engine.eval import Evaluator
from engine.loss import MoEFFDLoss
from engine.train import Trainer
from models.model import MoEFFDDetector
from utils.config import DatasetSpec, ModelConfig, OptimizerConfig, TrainConfig


def build_loader(dataset_root: Path, manifest_name: str, split_name: str, image_size: int, batch_size: int, num_workers: int, shuffle: bool) -> DataLoader:
    spec = DatasetSpec(
        name="StageDataset",
        root=str(dataset_root),
        split=split_name,
        image_size=image_size,
        processed_root=str(dataset_root / split_name),
        manifest_path=str(dataset_root / manifest_name),
    )
    dataset = build_dataset(spec)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)


def maybe_load_checkpoint(model: torch.nn.Module, checkpoint_path: str | None) -> None:
    if not checkpoint_path:
        return
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    print(f"Loaded checkpoint: {checkpoint_path}")


def run_stage_training(
    dataset_root: str,
    output_name: str,
    model_config: ModelConfig,
    batch_size: int,
    epochs: int,
    num_workers: int,
    image_size: int,
    device: str,
    init_checkpoint: str | None = None,
    output_dir: str = "outputs",
) -> None:
    dataset_path = Path(dataset_root)
    train_loader = build_loader(dataset_path, "train_manifest.jsonl", "train", image_size, batch_size, num_workers, True)
    val_loader = build_loader(dataset_path, "val_manifest.jsonl", "val", image_size, batch_size, num_workers, False)
    ffpp_test_loader = build_loader(dataset_path, "test_ffpp_manifest.jsonl", "test_ffpp", image_size, batch_size, num_workers, False)
    celebdf_test_loader = build_loader(dataset_path, "test_celebdf_manifest.jsonl", "test_celebdf", image_size, batch_size, num_workers, False)

    optimizer_config = OptimizerConfig()
    train_config = TrainConfig(
        batch_size=batch_size,
        epochs=epochs,
        num_workers=num_workers,
        amp=True,
    )

    model = MoEFFDDetector(model_config).to(device)
    maybe_load_checkpoint(model, init_checkpoint)

    criterion = MoEFFDLoss(
        load_balance_weight=model_config.moe.load_balance_weight,
        lora_balance_scale=model_config.moe.lora_balance_scale,
        adapter_balance_scale=model_config.moe.adapter_balance_scale,
    )
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        criterion=criterion,
        train_config=train_config,
        optimizer_config=optimizer_config,
    )
    history = trainer.fit(val_loader=val_loader)

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    checkpoint_path = output_path / output_name
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "history": history,
            "model_config": model_config,
            "init_checkpoint": init_checkpoint,
        },
        checkpoint_path,
    )
    print(f"Saved checkpoint after train/val to: {checkpoint_path}")

    ffpp_eval = Evaluator(model, ffpp_test_loader, criterion, device).evaluate()
    celebdf_eval = Evaluator(model, celebdf_test_loader, criterion, device).evaluate()

    print(
        f"FF++ test | loss={ffpp_eval['loss']:.4f} | "
        f"acc={ffpp_eval['metrics'].accuracy:.4f} | auc={ffpp_eval['metrics'].auc:.4f}"
    )
    print(
        f"Celeb-DF test | loss={celebdf_eval['loss']:.4f} | "
        f"acc={celebdf_eval['metrics'].accuracy:.4f} | auc={celebdf_eval['metrics'].auc:.4f}"
    )

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "history": history,
            "ffpp_eval": ffpp_eval,
            "celebdf_eval": celebdf_eval,
            "model_config": model_config,
            "init_checkpoint": init_checkpoint,
        },
        checkpoint_path,
    )
    print(f"Saved checkpoint to: {checkpoint_path}")
