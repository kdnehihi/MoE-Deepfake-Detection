"""Prepare the clean baseline dataset protocol for generalization experiments."""

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


FFPP_FAKE_TYPES = {"Deepfakes", "FaceSwap", "Face2Face", "NeuralTextures"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare baseline clean dataset.")
    parser.add_argument("--celebdf-root", type=str, default="data/processed/celebdf")
    parser.add_argument("--ffpp-root", type=str, default="data/processed/ffpp_generalization")
    parser.add_argument("--output-root", type=str, default="data/baseline")
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

    train_real_pool = filter_by_label(celebdf_train, 0) + filter_by_label(ffpp_train, 0)
    train_fake_pool = filter_ffpp_fake_types(ffpp_train, FFPP_FAKE_TYPES)

    ratios = {"real": 0.4, "fake": 0.6}
    total_target = compute_total_target(
        {"real": len(train_real_pool), "fake": len(train_fake_pool)},
        ratios,
    )
    counts = compute_counts(total_target, ratios)

    train_real = sample_without_replacement(train_real_pool, counts["real"], seed=args.seed)
    train_fake = sample_without_replacement(train_fake_pool, counts["fake"], seed=args.seed + 1)
    train_samples = train_real + train_fake

    ffpp_test_samples = filter_by_label(ffpp_test, 0) + filter_ffpp_fake_types(ffpp_test, FFPP_FAKE_TYPES)
    celebdf_test_samples = filter_by_label(celebdf_test, 0) + filter_by_label(celebdf_test, 1)

    train_manifest = materialize_split(train_samples, output_root / "train", "train_manifest.jsonl", overwrite=args.overwrite)
    ffpp_manifest = materialize_split(ffpp_test_samples, output_root / "test_ffpp", "test_ffpp_manifest.jsonl", overwrite=args.overwrite)
    celebdf_manifest = materialize_split(celebdf_test_samples, output_root / "test_celebdf", "test_celebdf_manifest.jsonl", overwrite=args.overwrite)

    print("Baseline clean dataset prepared.")
    print("Train manifest:", train_manifest)
    print("FF++ test manifest:", ffpp_manifest)
    print("Celeb-DF test manifest:", celebdf_manifest)
    print("Train counts:", {"real": len(train_real), "fake": len(train_fake), "total": len(train_samples)})


if __name__ == "__main__":
    main()
