"""Prepare stage-wise datasets for staged MoE deepfake training."""

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
    split_by_group,
)
from data.sbi_generator import generate_sbi_samples


FFPP_FAKE_TYPES = {"Deepfakes", "FaceSwap", "Face2Face", "NeuralTextures"}


def _load_sources(celebdf_root: Path, ffpp_root: Path) -> dict[str, list[dict]]:
    return {
        "celebdf_train": load_manifest(celebdf_root / "celebdf_train_manifest.jsonl"),
        "celebdf_test": load_manifest(celebdf_root / "celebdf_test_manifest.jsonl"),
        "ffpp_train": load_manifest(ffpp_root / "ffpp_generalization_train_manifest.jsonl"),
        "ffpp_test": load_manifest(ffpp_root / "ffpp_generalization_test_manifest.jsonl"),
    }


def _materialize_tests(output_root: Path, celebdf_test: list[dict], ffpp_test: list[dict], overwrite: bool) -> None:
    ffpp_test_samples = filter_by_label(ffpp_test, 0) + filter_ffpp_fake_types(ffpp_test, FFPP_FAKE_TYPES)
    celebdf_test_samples = filter_by_label(celebdf_test, 0) + filter_by_label(celebdf_test, 1)
    materialize_split(ffpp_test_samples, output_root / "test_ffpp", "test_ffpp_manifest.jsonl", overwrite=overwrite)
    materialize_split(celebdf_test_samples, output_root / "test_celebdf", "test_celebdf_manifest.jsonl", overwrite=overwrite)


def _split_real_pool(celebdf_train: list[dict], ffpp_train: list[dict], val_ratio: float, seed: int) -> tuple[list[dict], list[dict]]:
    real_pool = filter_by_label(celebdf_train, 0) + filter_by_label(ffpp_train, 0)
    return split_by_group(real_pool, val_ratio=val_ratio, seed=seed)


def _split_ffpp_original_pool(ffpp_train: list[dict], val_ratio: float, seed: int) -> tuple[list[dict], list[dict]]:
    ffpp_original = [
        sample
        for sample in ffpp_train
        if int(sample["label"]) == 0 and sample.get("manipulation_type") == "original"
    ]
    return split_by_group(ffpp_original, val_ratio=val_ratio, seed=seed, key="source_video")


def prepare_stage1(
    celebdf_root: Path,
    ffpp_root: Path,
    output_root: Path,
    val_ratio: float,
    seed: int,
    overwrite: bool,
) -> None:
    sources = _load_sources(celebdf_root, ffpp_root)
    train_real_pool, val_real_pool = _split_ffpp_original_pool(sources["ffpp_train"], val_ratio, seed)

    train_real = train_real_pool
    val_real = val_real_pool

    train_sbi, train_sbi_manifest = generate_sbi_samples(
        real_samples=train_real,
        count=len(train_real),
        output_root=output_root / "_generated_sbi",
        split_name="train",
        seed=seed + 1,
        overwrite=overwrite,
    )
    val_sbi, val_sbi_manifest = generate_sbi_samples(
        real_samples=val_real,
        count=len(val_real),
        output_root=output_root / "_generated_sbi",
        split_name="val",
        seed=seed + 2,
        overwrite=overwrite,
    )

    materialize_split(train_real + train_sbi, output_root / "train", "train_manifest.jsonl", overwrite=overwrite)
    materialize_split(val_real + val_sbi, output_root / "val", "val_manifest.jsonl", overwrite=overwrite)
    _materialize_tests(output_root, sources["celebdf_test"], sources["ffpp_test"], overwrite)

    print("Prepared stage1 dataset")
    print("stage1 real source: FF++ original only")
    print("train real:", len(train_real), "train sbi:", len(train_sbi))
    print("val real:", len(val_real), "val sbi:", len(val_sbi))
    print("train SBI manifest:", train_sbi_manifest)
    print("val SBI manifest:", val_sbi_manifest)


def prepare_stage2(
    celebdf_root: Path,
    ffpp_root: Path,
    output_root: Path,
    val_ratio: float,
    seed: int,
    overwrite: bool,
    ffpp_fake_ratio: float = 0.42,
    celebdf_fake_ratio: float = 0.18,
) -> None:
    sources = _load_sources(celebdf_root, ffpp_root)
    train_real_pool, val_real_pool = _split_real_pool(sources["celebdf_train"], sources["ffpp_train"], val_ratio, seed)

    ffpp_fake_train, ffpp_fake_val = split_by_group(
        filter_ffpp_fake_types(sources["ffpp_train"], FFPP_FAKE_TYPES),
        val_ratio=val_ratio,
        seed=seed + 1,
    )
    celebdf_fake_train, celebdf_fake_val = split_by_group(
        filter_by_label(sources["celebdf_train"], 1),
        val_ratio=val_ratio,
        seed=seed + 2,
    )

    ratios = {"real": 0.4, "ffpp_fake": ffpp_fake_ratio, "celebdf_fake": celebdf_fake_ratio}

    train_total = compute_total_target(
        {"real": len(train_real_pool), "ffpp_fake": len(ffpp_fake_train), "celebdf_fake": len(celebdf_fake_train)},
        ratios,
    )
    val_total = compute_total_target(
        {"real": len(val_real_pool), "ffpp_fake": len(ffpp_fake_val), "celebdf_fake": len(celebdf_fake_val)},
        ratios,
    )

    train_counts = compute_counts(train_total, ratios)
    val_counts = compute_counts(val_total, ratios)

    train_samples = (
        sample_without_replacement(train_real_pool, train_counts["real"], seed)
        + sample_without_replacement(ffpp_fake_train, train_counts["ffpp_fake"], seed + 3)
        + sample_without_replacement(celebdf_fake_train, train_counts["celebdf_fake"], seed + 4)
    )
    val_samples = (
        sample_without_replacement(val_real_pool, val_counts["real"], seed + 5)
        + sample_without_replacement(ffpp_fake_val, val_counts["ffpp_fake"], seed + 6)
        + sample_without_replacement(celebdf_fake_val, val_counts["celebdf_fake"], seed + 7)
    )

    materialize_split(train_samples, output_root / "train", "train_manifest.jsonl", overwrite=overwrite)
    materialize_split(val_samples, output_root / "val", "val_manifest.jsonl", overwrite=overwrite)
    _materialize_tests(output_root, sources["celebdf_test"], sources["ffpp_test"], overwrite)

    print("Prepared stage2 dataset")
    print("train counts:", train_counts)
    print("val counts:", val_counts)


def prepare_stage3(
    celebdf_root: Path,
    ffpp_root: Path,
    output_root: Path,
    val_ratio: float,
    seed: int,
    overwrite: bool,
    ffpp_fake_ratio: float = 0.36,
    celebdf_fake_ratio: float = 0.15,
    sbi_fake_ratio: float = 0.09,
) -> None:
    sources = _load_sources(celebdf_root, ffpp_root)
    train_real_pool, val_real_pool = _split_real_pool(sources["celebdf_train"], sources["ffpp_train"], val_ratio, seed)

    ffpp_fake_train, ffpp_fake_val = split_by_group(
        filter_ffpp_fake_types(sources["ffpp_train"], FFPP_FAKE_TYPES),
        val_ratio=val_ratio,
        seed=seed + 1,
    )
    celebdf_fake_train, celebdf_fake_val = split_by_group(
        filter_by_label(sources["celebdf_train"], 1),
        val_ratio=val_ratio,
        seed=seed + 2,
    )

    ratios = {"real": 0.4, "ffpp_fake": ffpp_fake_ratio, "celebdf_fake": celebdf_fake_ratio, "sbi_fake": sbi_fake_ratio}

    train_total = compute_total_target(
        {
            "real": len(train_real_pool),
            "ffpp_fake": len(ffpp_fake_train),
            "celebdf_fake": len(celebdf_fake_train),
            "sbi_fake": len(train_real_pool),
        },
        ratios,
    )
    val_total = compute_total_target(
        {
            "real": len(val_real_pool),
            "ffpp_fake": len(ffpp_fake_val),
            "celebdf_fake": len(celebdf_fake_val),
            "sbi_fake": len(val_real_pool),
        },
        ratios,
    )

    train_counts = compute_counts(train_total, ratios)
    val_counts = compute_counts(val_total, ratios)

    train_real = sample_without_replacement(train_real_pool, train_counts["real"], seed)
    train_ffpp_fake = sample_without_replacement(ffpp_fake_train, train_counts["ffpp_fake"], seed + 3)
    train_celebdf_fake = sample_without_replacement(celebdf_fake_train, train_counts["celebdf_fake"], seed + 4)
    train_sbi: list[dict] = []
    if train_counts["sbi_fake"] > 0:
        train_sbi_sources = sample_without_replacement(train_real_pool, train_counts["sbi_fake"], seed + 5)
        train_sbi, _ = generate_sbi_samples(
            real_samples=train_sbi_sources,
            count=train_counts["sbi_fake"],
            output_root=output_root / "_generated_sbi",
            split_name="train",
            seed=seed + 6,
            overwrite=overwrite,
        )

    val_real = sample_without_replacement(val_real_pool, val_counts["real"], seed + 7)
    val_ffpp_fake = sample_without_replacement(ffpp_fake_val, val_counts["ffpp_fake"], seed + 8)
    val_celebdf_fake = sample_without_replacement(celebdf_fake_val, val_counts["celebdf_fake"], seed + 9)
    val_sbi: list[dict] = []
    if val_counts["sbi_fake"] > 0:
        val_sbi_sources = sample_without_replacement(val_real_pool, val_counts["sbi_fake"], seed + 10)
        val_sbi, _ = generate_sbi_samples(
            real_samples=val_sbi_sources,
            count=val_counts["sbi_fake"],
            output_root=output_root / "_generated_sbi",
            split_name="val",
            seed=seed + 11,
            overwrite=overwrite,
        )

    materialize_split(
        train_real + train_ffpp_fake + train_celebdf_fake + train_sbi,
        output_root / "train",
        "train_manifest.jsonl",
        overwrite=overwrite,
    )
    materialize_split(
        val_real + val_ffpp_fake + val_celebdf_fake + val_sbi,
        output_root / "val",
        "val_manifest.jsonl",
        overwrite=overwrite,
    )
    _materialize_tests(output_root, sources["celebdf_test"], sources["ffpp_test"], overwrite)

    print("Prepared stage3 dataset")
    print("train counts:", train_counts)
    print("val counts:", val_counts)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare staged datasets for MoE deepfake training.")
    parser.add_argument("--stage", choices=["stage1", "stage2", "stage3"], required=True)
    parser.add_argument("--celebdf-root", type=str, default="data/processed/celebdf")
    parser.add_argument("--ffpp-root", type=str, default="data/processed/ffpp_generalization")
    parser.add_argument("--output-root", type=str, required=True)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--stage2-ffpp-fake-ratio", type=float, default=0.42)
    parser.add_argument("--stage2-celebdf-fake-ratio", type=float, default=0.18)
    parser.add_argument("--stage3-ffpp-fake-ratio", type=float, default=0.36)
    parser.add_argument("--stage3-celebdf-fake-ratio", type=float, default=0.15)
    parser.add_argument("--stage3-sbi-fake-ratio", type=float, default=0.09)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    celebdf_root = Path(args.celebdf_root)
    ffpp_root = Path(args.ffpp_root)
    output_root = Path(args.output_root)

    if args.stage == "stage1":
        prepare_stage1(celebdf_root, ffpp_root, output_root, args.val_ratio, args.seed, args.overwrite)
    elif args.stage == "stage2":
        prepare_stage2(
            celebdf_root,
            ffpp_root,
            output_root,
            args.val_ratio,
            args.seed,
            args.overwrite,
            ffpp_fake_ratio=args.stage2_ffpp_fake_ratio,
            celebdf_fake_ratio=args.stage2_celebdf_fake_ratio,
        )
    else:
        prepare_stage3(
            celebdf_root,
            ffpp_root,
            output_root,
            args.val_ratio,
            args.seed,
            args.overwrite,
            ffpp_fake_ratio=args.stage3_ffpp_fake_ratio,
            celebdf_fake_ratio=args.stage3_celebdf_fake_ratio,
            sbi_fake_ratio=args.stage3_sbi_fake_ratio,
        )


if __name__ == "__main__":
    main()
