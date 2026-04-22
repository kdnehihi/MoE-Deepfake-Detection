"""Evaluate a saved baseline checkpoint with frame-level and video-level metrics."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from engine.eval import Evaluator
from engine.loss import MoEFFDLoss
from models.model import MoEFFDDetector
from train_baseline import _average_metric, build_loader
from utils.config import ModelConfig


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate baseline clean checkpoint.")
    parser.add_argument("--dataset-root", type=str, default="data/baseline")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--device", type=str, default=None)
    return parser.parse_args()


def _print_split(name: str, evaluation: dict) -> None:
    print(
        f"{name} | loss={evaluation['loss']:.4f} | "
        f"frame_acc={evaluation['metrics'].accuracy:.4f} | "
        f"frame_auc={evaluation['metrics'].auc:.4f} | "
        f"frame_ap={evaluation['metrics'].ap:.4f} | "
        f"frame_eer={evaluation['metrics'].eer:.4f} | "
        f"video_auc={evaluation['video_metrics'].auc:.4f} | "
        f"video_ap={evaluation['video_metrics'].ap:.4f} | "
        f"video_eer={evaluation['video_metrics'].eer:.4f} | "
        f"frames={evaluation['num_frames']} | videos={evaluation['num_videos']}"
    )


def main() -> None:
    args = parse_args()
    dataset_root = Path(args.dataset_root)
    checkpoint_path = Path(args.checkpoint)
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

    payload = torch.load(checkpoint_path, map_location=device)
    model_config = payload.get("model_config", ModelConfig())

    model = MoEFFDDetector(model_config).to(device)
    model.load_state_dict(payload["model_state_dict"])

    criterion = MoEFFDLoss(load_balance_weight=model_config.moe.load_balance_weight)
    ffpp_loader = build_loader(
        dataset_root,
        "test_ffpp_manifest.jsonl",
        "test_ffpp",
        args.image_size,
        args.batch_size,
        args.num_workers,
        False,
    )
    celebdf_loader = build_loader(
        dataset_root,
        "test_celebdf_manifest.jsonl",
        "test_celebdf",
        args.image_size,
        args.batch_size,
        args.num_workers,
        False,
    )

    ffpp_eval = Evaluator(model, ffpp_loader, criterion, device).evaluate()
    celebdf_eval = Evaluator(model, celebdf_loader, criterion, device).evaluate()
    all_evals = [ffpp_eval, celebdf_eval]

    _print_split("FF++ test", ffpp_eval)
    _print_split("Celeb-DF test", celebdf_eval)
    print(
        "AVG test | "
        f"frame_auc={_average_metric(all_evals, 'auc'):.4f} | "
        f"frame_ap={_average_metric(all_evals, 'ap'):.4f} | "
        f"frame_eer={_average_metric(all_evals, 'eer'):.4f} | "
        f"video_auc={_average_metric(all_evals, 'auc', video_level=True):.4f} | "
        f"video_ap={_average_metric(all_evals, 'ap', video_level=True):.4f} | "
        f"video_eer={_average_metric(all_evals, 'eer', video_level=True):.4f}"
    )


if __name__ == "__main__":
    main()
