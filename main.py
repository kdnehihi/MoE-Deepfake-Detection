"""Project entrypoint for staged MoE-FFD reproduction."""

from __future__ import annotations

import argparse

from utils.config import DatasetSpec, load_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="MoE-FFD reproduction project")
    subparsers = parser.add_subparsers(dest="command", required=True)

    prepare_parser = subparsers.add_parser("prepare-data", help="Extract face frames from raw dataset videos.")
    prepare_parser.add_argument("--dataset", type=str, required=True, help="Dataset name, e.g. CelebDF.")
    prepare_parser.add_argument("--root", type=str, required=True, help="Root directory containing raw videos.")
    prepare_parser.add_argument(
        "--split",
        type=str,
        default="train",
        help="Logical split name used for output organization.",
    )
    prepare_parser.add_argument("--frames-per-video", type=int, default=8, help="Number of sampled frames per video.")
    prepare_parser.add_argument("--image-size", type=int, default=224, help="Face crop resolution.")
    prepare_parser.add_argument(
        "--processed-root",
        type=str,
        default=None,
        help="Directory where processed face crops and manifest will be stored.",
    )
    prepare_parser.add_argument("--manifest-path", type=str, default=None, help="Optional custom manifest path.")
    prepare_parser.add_argument("--max-videos", type=int, default=None, help="Optional cap for smoke testing.")
    prepare_parser.add_argument("--detector-margin", type=int, default=24, help="Extra crop margin around faces.")
    prepare_parser.add_argument("--device", type=str, default=None, help="Torch device for MTCNN, e.g. cpu or cuda:0.")
    prepare_parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite already extracted face crops if they exist.",
    )

    split_parser = subparsers.add_parser(
        "prepare-celebdf",
        help="Create balanced train/val/test splits for CelebDF and preprocess all splits.",
    )
    split_parser.add_argument("--root", type=str, required=True, help="Root directory containing CelebDF videos.")
    split_parser.add_argument(
        "--processed-root",
        type=str,
        default="data/processed/celebdf",
        help="Directory where processed face crops and manifests will be stored.",
    )
    split_parser.add_argument("--frames-per-video", type=int, default=8, help="Number of sampled frames per video.")
    split_parser.add_argument("--image-size", type=int, default=224, help="Face crop resolution.")
    split_parser.add_argument("--detector-margin", type=int, default=24, help="Extra crop margin around faces.")
    split_parser.add_argument("--train-ratio", type=float, default=0.7, help="Train split ratio.")
    split_parser.add_argument("--val-ratio", type=float, default=0.15, help="Validation split ratio.")
    split_parser.add_argument("--seed", type=int, default=42, help="Random seed for the split.")
    split_parser.add_argument("--device", type=str, default=None, help="Torch device for MTCNN, e.g. cpu or cuda:0.")
    split_parser.add_argument("--overwrite", action="store_true", help="Overwrite already extracted face crops.")

    config_parser = subparsers.add_parser("show-config", help="Load and validate a YAML experiment config.")
    config_parser.add_argument("--config", type=str, required=True, help="Path to the YAML experiment config.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.command == "prepare-data":
        from data.video_to_frames import prepare_dataset_frames

        spec = DatasetSpec(
            name=args.dataset,
            root=args.root,
            split=args.split,
            frames_per_video=args.frames_per_video,
            image_size=args.image_size,
            processed_root=args.processed_root,
            manifest_path=args.manifest_path,
            max_videos=args.max_videos,
            detector_margin=args.detector_margin,
            overwrite_processed=args.overwrite,
        )
        prepare_dataset_frames(spec=spec, device=args.device)
        return

    if args.command == "prepare-celebdf":
        from data.video_to_frames import prepare_balanced_celebdf

        prepare_balanced_celebdf(
            root=args.root,
            processed_root=args.processed_root,
            frames_per_video=args.frames_per_video,
            image_size=args.image_size,
            detector_margin=args.detector_margin,
            train_ratio=args.train_ratio,
            val_ratio=args.val_ratio,
            seed=args.seed,
            device=args.device,
            overwrite=args.overwrite,
        )
        return

    if args.command == "show-config":
        load_config(args.config)
        print(f"Config loaded successfully: {args.config}")
        return

    raise ValueError(f"Unsupported command: {args.command}")


if __name__ == "__main__":
    main()
