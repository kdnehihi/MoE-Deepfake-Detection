"""Train the current model on the clean baseline dataset and evaluate on FF++ and Celeb-DF tests."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from data.dataset import build_dataset
from engine.eval import Evaluator
from engine.loss import MoEFFDLoss
from engine.train import Trainer
from models.model import MoEFFDDetector
from utils.config import DatasetSpec, ModelConfig, OptimizerConfig, TrainConfig


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train baseline clean dataset.")
    parser.add_argument("--dataset-root", type=str, default="data/baseline")
    parser.add_argument("--output-dir", type=str, default="outputs")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--device", type=str, default=None)
    return parser.parse_args()


def build_loader(dataset_root: Path, manifest_name: str, split_name: str, image_size: int, batch_size: int, num_workers: int, shuffle: bool) -> DataLoader:
    spec = DatasetSpec(
        name="Baseline",
        root=str(dataset_root),
        split=split_name,
        image_size=image_size,
        processed_root=str(dataset_root / split_name),
        manifest_path=str(dataset_root / manifest_name),
    )
    dataset = build_dataset(spec)
    print(f"{split_name}: {len(dataset)} samples from {dataset_root / manifest_name}")
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)


def main() -> None:
    args = parse_args()
    dataset_root = Path(args.dataset_root)
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

    train_loader = build_loader(dataset_root, "train_manifest.jsonl", "train", args.image_size, args.batch_size, args.num_workers, True)
    ffpp_test_loader = build_loader(dataset_root, "test_ffpp_manifest.jsonl", "test_ffpp", args.image_size, args.batch_size, args.num_workers, False)
    celebdf_test_loader = build_loader(dataset_root, "test_celebdf_manifest.jsonl", "test_celebdf", args.image_size, args.batch_size, args.num_workers, False)

    model_config = ModelConfig()
    optimizer_config = OptimizerConfig()
    train_config = TrainConfig(
        batch_size=args.batch_size,
        epochs=args.epochs,
        num_workers=args.num_workers,
        amp=True,
    )

    model = MoEFFDDetector(model_config).to(device)
    criterion = MoEFFDLoss(load_balance_weight=model_config.moe.load_balance_weight)
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        criterion=criterion,
        train_config=train_config,
        optimizer_config=optimizer_config,
    )
    history = trainer.fit()

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

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = output_dir / "baseline_clean_last.pt"
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "history": history,
            "ffpp_eval": ffpp_eval,
            "celebdf_eval": celebdf_eval,
            "model_config": model_config,
        },
        checkpoint_path,
    )
    print(f"Saved checkpoint to: {checkpoint_path}")


if __name__ == "__main__":
    main()
