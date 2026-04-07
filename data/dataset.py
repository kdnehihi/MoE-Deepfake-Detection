"""Dataset definitions for frame-based face forgery detection."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path

from PIL import Image
from torch import Tensor
from torch.utils.data import ConcatDataset, Dataset

from data.transforms import build_eval_transforms, build_train_transforms
from utils.config import DatasetSpec


@dataclass(slots=True)
class FaceSample:
    image_path: Path
    label: int
    dataset_name: str
    video_id: str
    frame_index: int
    source_video: str = ""
    split: str = ""


def _processed_root(spec: DatasetSpec) -> Path:
    if spec.processed_root is not None:
        return Path(spec.processed_root)
    return Path(spec.root) / "processed_faces"


def _manifest_path(spec: DatasetSpec) -> Path:
    if spec.manifest_path is not None:
        return Path(spec.manifest_path)
    return _processed_root(spec) / f"{spec.name.lower()}_{spec.split}_manifest.jsonl"


def _load_manifest(manifest_path: Path) -> list[FaceSample]:
    if not manifest_path.exists():
        return []
    samples: list[FaceSample] = []
    with manifest_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            payload = json.loads(line)
            samples.append(
                FaceSample(
                    image_path=Path(payload["image_path"]),
                    label=int(payload["label"]),
                    dataset_name=payload["dataset_name"],
                    video_id=payload["video_id"],
                    frame_index=int(payload["frame_index"]),
                    source_video=payload.get("source_video", ""),
                    split=payload.get("split", ""),
                )
            )
    return samples


def _infer_samples_from_processed_dir(spec: DatasetSpec) -> list[FaceSample]:
    processed_split_root = _processed_root(spec) / spec.split
    samples: list[FaceSample] = []
    if not processed_split_root.exists():
        return samples

    for label_name, label_value in (("real", 0), ("fake", 1)):
        label_root = processed_split_root / label_name
        if not label_root.exists():
            continue
        for image_path in sorted(label_root.rglob("*.png")):
            name_parts = image_path.stem.split("_frame_")
            video_id = name_parts[0]
            frame_index = int(name_parts[1]) if len(name_parts) == 2 and name_parts[1].isdigit() else 0
            samples.append(
                FaceSample(
                    image_path=image_path,
                    label=label_value,
                    dataset_name=spec.name,
                    video_id=video_id,
                    frame_index=frame_index,
                    split=spec.split,
                )
            )
    return samples


class FaceForgeryDataset(Dataset):
    """
    Unified dataset interface across Celeb-DF and additional sources.

    The full indexing and preprocessing logic is added in the dataset step once
    frame extraction and metadata normalization are implemented.
    """

    def __init__(self, spec: DatasetSpec, transform=None) -> None:
        self.spec = spec
        self.transform = transform or (
            build_train_transforms(spec) if spec.split.lower() == "train" else build_eval_transforms(spec)
        )
        self.samples = self._index_samples()

    def _index_samples(self) -> list[FaceSample]:
        manifest_samples = _load_manifest(_manifest_path(self.spec))
        if manifest_samples:
            return manifest_samples
        return _infer_samples_from_processed_dir(self.spec)

    def __len__(self) -> int:
        return len(self.samples)

    def load_image(self, path: Path) -> Image.Image:
        return Image.open(path).convert("RGB")

    def __getitem__(self, index: int) -> tuple[Tensor, int]:
        sample = self.samples[index]
        image = self.load_image(sample.image_path)
        tensor = self.transform(image) if self.transform is not None else image
        return tensor, sample.label


def build_dataset(spec: DatasetSpec, transform=None) -> FaceForgeryDataset:
    return FaceForgeryDataset(spec=spec, transform=transform)


def build_combined_dataset(specs: list[DatasetSpec]) -> Dataset:
    datasets = [build_dataset(spec) for spec in specs]
    if len(datasets) == 1:
        return datasets[0]
    return ConcatDataset(datasets)
