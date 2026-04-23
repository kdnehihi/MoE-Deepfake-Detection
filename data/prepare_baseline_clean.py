"""Prepare a paper-aligned baseline protocol using FF++ for training and Celeb-DF for OOD test."""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data.sampler import filter_by_label, load_manifest, save_manifest, split_by_group


FFPP_SUBSETS = ("original", "Deepfakes", "Face2Face", "FaceSwap", "NeuralTextures")
FFPP_FAKE_TYPES = ("Deepfakes", "Face2Face", "FaceSwap", "NeuralTextures")


def resolve_image_path(sample: dict, dataset_root: Path) -> dict:
    resolved = dict(sample)
    image_path = Path(str(sample["image_path"]))
    if image_path.is_absolute():
        resolved["image_path"] = str(image_path)
        return resolved

    parts = image_path.parts
    anchor = dataset_root.name
    if anchor in parts:
        anchor_index = parts.index(anchor)
        suffix = parts[anchor_index + 1 :]
        resolved["image_path"] = str(dataset_root.joinpath(*suffix))
    else:
        resolved["image_path"] = str((dataset_root / image_path).resolve())
    return resolved


def _split_ffpp_train_valid(samples: list[dict], val_ratio: float, seed: int) -> tuple[list[dict], list[dict]]:
    train_samples: list[dict] = []
    valid_samples: list[dict] = []

    for offset, manipulation_type in enumerate(FFPP_SUBSETS):
        subset = [sample for sample in samples if sample.get("manipulation_type") == manipulation_type]
        subset_train, subset_valid = split_by_group(
            subset,
            val_ratio,
            seed=seed + offset,
            key="source_video",
        )
        train_samples.extend(subset_train)
        valid_samples.extend(subset_valid)
    return train_samples, valid_samples


def _balance_ffpp_train_like_paper(samples: list[dict], seed: int) -> list[dict]:
    original_samples = [sample for sample in samples if sample.get("manipulation_type") == "original"]
    fake_samples = [sample for sample in samples if sample.get("manipulation_type") in FFPP_FAKE_TYPES]
    if not original_samples or not fake_samples:
        return samples

    rng = random.Random(seed)
    repeated_originals = original_samples[:]
    while len(repeated_originals) < len(fake_samples):
        shuffled = original_samples[:]
        rng.shuffle(shuffled)
        repeated_originals.extend(shuffled)

    balanced_originals = repeated_originals[: len(fake_samples)]
    balanced_samples = balanced_originals + fake_samples
    rng.shuffle(balanced_samples)
    return balanced_samples


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare paper-like baseline dataset.")
    parser.add_argument("--celebdf-root", type=str, default="data/processed/celebdf")
    parser.add_argument("--ffpp-root", type=str, default="data/processed/ffpp_generalization")
    parser.add_argument("--output-root", type=str, default="data/baseline")
    parser.add_argument(
        "--val-ratio",
        type=float,
        default=0.125,
        help="Validation ratio carved from FF++ train, per manipulation type, at video level, when no FF++ valid manifest exists.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    celebdf_root = Path(args.celebdf_root)
    ffpp_root = Path(args.ffpp_root)
    output_root = Path(args.output_root)

    ffpp_train = load_manifest(ffpp_root / "ffpp_generalization_train_manifest.jsonl")
    ffpp_valid_path = ffpp_root / "ffpp_generalization_valid_manifest.jsonl"
    ffpp_test = load_manifest(ffpp_root / "ffpp_generalization_test_manifest.jsonl")
    celebdf_test = load_manifest(celebdf_root / "celebdf_test_manifest.jsonl")

    ffpp_train_subset = [
        sample
        for sample in ffpp_train
        if sample.get("manipulation_type") in FFPP_SUBSETS
    ]
    ffpp_test_subset = [
        sample
        for sample in ffpp_test
        if sample.get("manipulation_type") in FFPP_SUBSETS
    ]

    if ffpp_valid_path.exists():
        ffpp_valid = load_manifest(ffpp_valid_path)
        valid_samples = [
            sample
            for sample in ffpp_valid
            if sample.get("manipulation_type") in FFPP_SUBSETS
        ]
        train_samples = ffpp_train_subset
        valid_source = "ffpp_generalization_valid_manifest.jsonl"
    else:
        train_samples, valid_samples = _split_ffpp_train_valid(ffpp_train_subset, args.val_ratio, args.seed)
        valid_source = f"split from train with val_ratio={args.val_ratio}"

    train_samples = _balance_ffpp_train_like_paper(train_samples, args.seed)
    celebdf_test_samples = filter_by_label(celebdf_test, 0) + filter_by_label(celebdf_test, 1)

    output_root.mkdir(parents=True, exist_ok=True)

    train_manifest_path = output_root / "train_manifest.jsonl"
    val_manifest_path = output_root / "val_manifest.jsonl"
    ffpp_manifest_path = output_root / "test_ffpp_manifest.jsonl"
    celebdf_manifest_path = output_root / "test_celebdf_manifest.jsonl"

    if not args.overwrite:
        existing = [train_manifest_path, val_manifest_path, ffpp_manifest_path, celebdf_manifest_path]
        if all(path.exists() for path in existing):
            print("Baseline manifests already exist. Use --overwrite to rebuild them.")
            print("Train manifest:", train_manifest_path)
            print("Val manifest:", val_manifest_path)
            print("FF++ test manifest:", ffpp_manifest_path)
            print("Celeb-DF test manifest:", celebdf_manifest_path)
            return

    train_manifest_samples = [resolve_image_path(sample, ffpp_root) for sample in train_samples]
    val_manifest_samples = [resolve_image_path(sample, ffpp_root) for sample in valid_samples]
    ffpp_manifest_samples = [resolve_image_path(sample, ffpp_root) for sample in ffpp_test_subset]
    celebdf_manifest_samples = [resolve_image_path(sample, celebdf_root) for sample in celebdf_test_samples]

    save_manifest(train_manifest_samples, train_manifest_path)
    save_manifest(val_manifest_samples, val_manifest_path)
    save_manifest(ffpp_manifest_samples, ffpp_manifest_path)
    save_manifest(celebdf_manifest_samples, celebdf_manifest_path)

    print("Paper-aligned baseline dataset prepared.")
    print("Train manifest:", train_manifest_path)
    print("Val manifest:", val_manifest_path)
    print("FF++ test manifest:", ffpp_manifest_path)
    print("Celeb-DF test manifest:", celebdf_manifest_path)
    print("FF++ valid source:", valid_source)
    print("Train frames:", len(train_manifest_samples))
    print("Val frames:", len(val_manifest_samples))
    print("FF++ test frames:", len(ffpp_manifest_samples))
    print("Celeb-DF test frames:", len(celebdf_manifest_samples))
    print(
        "Train effective ratio:",
        f"real={sum(1 for sample in train_manifest_samples if int(sample['label']) == 0)}",
        f"fake={sum(1 for sample in train_manifest_samples if int(sample['label']) == 1)}",
    )


if __name__ == "__main__":
    main()
