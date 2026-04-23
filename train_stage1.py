"""Stage 1 training: SBI pretraining with the full MoE detector."""

from __future__ import annotations

import argparse

import torch

from train_stage_common import run_stage_training
from utils.stage_presets import build_stage1_model_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train Stage 1 SBI pretraining.")
    parser.add_argument("--dataset-root", type=str, default="data/stages/stage1_sbi")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--output-dir", type=str, default="outputs")
    parser.add_argument("--device", type=str, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    run_stage_training(
        dataset_root=args.dataset_root,
        output_name="stage1_last.pt",
        model_config=build_stage1_model_config(),
        batch_size=args.batch_size,
        epochs=args.epochs,
        num_workers=args.num_workers,
        image_size=args.image_size,
        device=device,
        init_checkpoint=None,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()
