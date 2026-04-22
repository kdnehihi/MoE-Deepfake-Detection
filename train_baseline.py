"""Train the current model on the clean baseline dataset and evaluate on FF++ and Celeb-DF tests."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Subset

from data.dataset import build_dataset
from engine.eval import Evaluator
from engine.loss import MoEFFDLoss
from engine.train import Trainer
from models.model import MoEFFDDetector
from utils.config import DatasetSpec, ModelConfig, OptimizerConfig, TrainConfig


def _average_metric(evaluations: list[dict], field: str, video_level: bool = False) -> float:
    metric_name = "video_metrics" if video_level else "metrics"
    values = [float(getattr(evaluation[metric_name], field)) for evaluation in evaluations]
    return sum(values) / max(len(values), 1)


def _print_eval_summary(prefix: str, ffpp_eval: dict, celebdf_eval: dict) -> None:
    all_evals = [ffpp_eval, celebdf_eval]
    print(
        f"{prefix} FF++ test | loss={ffpp_eval['loss']:.4f} | "
        f"frame_acc={ffpp_eval['metrics'].accuracy:.4f} | "
        f"frame_auc={ffpp_eval['metrics'].auc:.4f} | "
        f"frame_ap={ffpp_eval['metrics'].ap:.4f} | "
        f"frame_eer={ffpp_eval['metrics'].eer:.4f} | "
        f"video_auc={ffpp_eval['video_metrics'].auc:.4f} | "
        f"video_ap={ffpp_eval['video_metrics'].ap:.4f} | "
        f"video_eer={ffpp_eval['video_metrics'].eer:.4f} | "
        f"frames={ffpp_eval['num_frames']} | videos={ffpp_eval['num_videos']}"
    )
    print(
        f"{prefix} Celeb-DF test | loss={celebdf_eval['loss']:.4f} | "
        f"frame_acc={celebdf_eval['metrics'].accuracy:.4f} | "
        f"frame_auc={celebdf_eval['metrics'].auc:.4f} | "
        f"frame_ap={celebdf_eval['metrics'].ap:.4f} | "
        f"frame_eer={celebdf_eval['metrics'].eer:.4f} | "
        f"video_auc={celebdf_eval['video_metrics'].auc:.4f} | "
        f"video_ap={celebdf_eval['video_metrics'].ap:.4f} | "
        f"video_eer={celebdf_eval['video_metrics'].eer:.4f} | "
        f"frames={celebdf_eval['num_frames']} | videos={celebdf_eval['num_videos']}"
    )
    print(
        f"{prefix} AVG test | "
        f"frame_auc={_average_metric(all_evals, 'auc'):.4f} | "
        f"frame_ap={_average_metric(all_evals, 'ap'):.4f} | "
        f"frame_eer={_average_metric(all_evals, 'eer'):.4f} | "
        f"video_auc={_average_metric(all_evals, 'auc', video_level=True):.4f} | "
        f"video_ap={_average_metric(all_evals, 'ap', video_level=True):.4f} | "
        f"video_eer={_average_metric(all_evals, 'eer', video_level=True):.4f}"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train baseline clean dataset.")
    parser.add_argument("--dataset-root", type=str, default="data/baseline")
    parser.add_argument("--output-dir", type=str, default="outputs")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--max-train-samples", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def build_loader(
    dataset_root: Path,
    manifest_name: str,
    split_name: str,
    image_size: int,
    batch_size: int,
    num_workers: int,
    shuffle: bool,
    max_samples: int | None = None,
    seed: int = 42,
) -> DataLoader:
    spec = DatasetSpec(
        name="Baseline",
        root=str(dataset_root),
        split=split_name,
        image_size=image_size,
        processed_root=str(dataset_root / split_name),
        manifest_path=str(dataset_root / manifest_name),
    )
    dataset = build_dataset(spec)
    original_count = len(dataset)
    if max_samples is not None and max_samples < original_count:
        generator = torch.Generator().manual_seed(seed)
        indices = torch.randperm(original_count, generator=generator)[:max_samples].tolist()
        dataset = Subset(dataset, indices)
        print(
            f"{split_name}: {len(dataset)} / {original_count} samples "
            f"from {dataset_root / manifest_name} (seed={seed})"
        )
    else:
        print(f"{split_name}: {original_count} samples from {dataset_root / manifest_name}")
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)


def main() -> None:
    args = parse_args()
    dataset_root = Path(args.dataset_root)
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

    train_loader = build_loader(
        dataset_root,
        "train_manifest.jsonl",
        "train",
        args.image_size,
        args.batch_size,
        args.num_workers,
        True,
        max_samples=args.max_train_samples,
        seed=args.seed,
    )
    ffpp_test_loader = build_loader(dataset_root, "test_ffpp_manifest.jsonl", "test_ffpp", args.image_size, args.batch_size, args.num_workers, False)
    celebdf_test_loader = build_loader(dataset_root, "test_celebdf_manifest.jsonl", "test_celebdf", args.image_size, args.batch_size, args.num_workers, False)

    model_config = ModelConfig()
    optimizer_config = OptimizerConfig()
    train_config = TrainConfig(
        batch_size=args.batch_size,
        epochs=args.epochs,
        num_workers=args.num_workers,
        amp=True,
        seed=args.seed,
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
    ffpp_evaluator = Evaluator(model, ffpp_test_loader, criterion, device)
    celebdf_evaluator = Evaluator(model, celebdf_test_loader, criterion, device)

    def on_epoch_end(epoch: int, record: dict) -> None:
        ffpp_eval = ffpp_evaluator.evaluate()
        celebdf_eval = celebdf_evaluator.evaluate()
        record["ffpp_eval"] = ffpp_eval
        record["celebdf_eval"] = celebdf_eval
        _print_eval_summary(f"Epoch {epoch}/{args.epochs} |", ffpp_eval, celebdf_eval)

    trainer.on_epoch_end = on_epoch_end
    history = trainer.fit()

    ffpp_eval = ffpp_evaluator.evaluate()
    celebdf_eval = celebdf_evaluator.evaluate()
    _print_eval_summary("Final |", ffpp_eval, celebdf_eval)

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
