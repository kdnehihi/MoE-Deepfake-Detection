"""Stage 3 training: full MoE specialization with adapters and router enabled."""

from __future__ import annotations

import argparse

import torch

from train_stage_common import run_stage_training
from utils.stage_presets import build_stage3_model_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train Stage 3 full MoE specialization.")
    parser.add_argument("--dataset-root", type=str, default="data/stages/stage3_full")
    parser.add_argument("--init-checkpoint", type=str, default="outputs/stage2_last.pt")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=10)
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
        output_name="stage3_last.pt",
        model_config=build_stage3_model_config(),
        batch_size=args.batch_size,
        epochs=args.epochs,
        num_workers=args.num_workers,
        image_size=args.image_size,
        device=device,
        init_checkpoint=args.init_checkpoint,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()
