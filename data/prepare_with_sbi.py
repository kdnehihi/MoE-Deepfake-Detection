"""Prepare the generalization dataset protocol with SBI augmentation."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data.dataset_builder import materialize_split
from data.sampler import (
    compute_counts,
    compute_total_target,
    filter_by_label,
    filter_ffpp_fake_types,
    load_manifest,
    sample_without_replacement,
)
from data.sbi_generator import generate_sbi_samples


FFPP_FAKE_TYPES = {"Deepfakes", "FaceSwap", "Face2Face", "NeuralTextures"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare dataset with SBI augmentation.")
    parser.add_argument("--celebdf-root", type=str, default="data/processed/celebdf")
    parser.add_argument("--ffpp-root", type=str, default="data/processed/ffpp_generalization")
    parser.add_argument("--output-root", type=str, default="data/with_sbi")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    celebdf_root = Path(args.celebdf_root)
    ffpp_root = Path(args.ffpp_root)
    output_root = Path(args.output_root)

    celebdf_train = load_manifest(celebdf_root / "celebdf_train_manifest.jsonl")
    celebdf_test = load_manifest(celebdf_root / "celebdf_test_manifest.jsonl")
    ffpp_train = load_manifest(ffpp_root / "ffpp_generalization_train_manifest.jsonl")
    ffpp_test = load_manifest(ffpp_root / "ffpp_generalization_test_manifest.jsonl")

    real_pool = filter_by_label(celebdf_train, 0) + filter_by_label(ffpp_train, 0)
    ffpp_fake_pool = filter_ffpp_fake_types(ffpp_train, FFPP_FAKE_TYPES)
    celebdf_fake_pool = filter_by_label(celebdf_train, 1)

    total_target = compute_total_target(
        {
            "real": len(real_pool),
            "ffpp_fake": len(ffpp_fake_pool),
            "celebdf_fake": len(celebdf_fake_pool),
            "sbi_fake": len(real_pool),
        },
        {
            "real": 0.4,
            "ffpp_fake": 0.36,
            "celebdf_fake": 0.15,
            "sbi_fake": 0.09,
        },
    )
    counts = compute_counts(
        total_target,
        {
            "real": 0.4,
            "ffpp_fake": 0.36,
            "celebdf_fake": 0.15,
            "sbi_fake": 0.09,
        },
    )

    train_real = sample_without_replacement(real_pool, counts["real"], seed=args.seed)
    train_ffpp_fake = sample_without_replacement(ffpp_fake_pool, counts["ffpp_fake"], seed=args.seed + 1)
    train_celebdf_fake = sample_without_replacement(celebdf_fake_pool, counts["celebdf_fake"], seed=args.seed + 2)

    sbi_source_real = sample_without_replacement(real_pool, counts["sbi_fake"], seed=args.seed + 3)
    train_sbi_fake, sbi_manifest = generate_sbi_samples(
        real_samples=sbi_source_real,
        count=counts["sbi_fake"],
        output_root=output_root / "_generated_sbi",
        split_name="train",
        seed=args.seed + 4,
        overwrite=args.overwrite,
    )

    train_samples = train_real + train_ffpp_fake + train_celebdf_fake + train_sbi_fake

    ffpp_test_samples = filter_by_label(ffpp_test, 0) + filter_ffpp_fake_types(ffpp_test, FFPP_FAKE_TYPES)
    celebdf_test_samples = filter_by_label(celebdf_test, 0) + filter_by_label(celebdf_test, 1)

    train_manifest = materialize_split(train_samples, output_root / "train", "train_manifest.jsonl", overwrite=args.overwrite)
    ffpp_manifest = materialize_split(ffpp_test_samples, output_root / "test_ffpp", "test_ffpp_manifest.jsonl", overwrite=args.overwrite)
    celebdf_manifest = materialize_split(celebdf_test_samples, output_root / "test_celebdf", "test_celebdf_manifest.jsonl", overwrite=args.overwrite)

    print("Dataset with SBI prepared.")
    print("Train manifest:", train_manifest)
    print("FF++ test manifest:", ffpp_manifest)
    print("Celeb-DF test manifest:", celebdf_manifest)
    print("SBI manifest:", sbi_manifest)
    print(
        "Train counts:",
        {
            "real": len(train_real),
            "ffpp_fake": len(train_ffpp_fake),
            "celebdf_fake": len(train_celebdf_fake),
            "sbi_fake": len(train_sbi_fake),
            "total": len(train_samples),
        },
    )


if __name__ == "__main__":
    main()
