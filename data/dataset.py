"""Dataset definitions for frame-based and video-based face forgery detection."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path

from PIL import Image
import torch
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


@dataclass(slots=True)
class VideoSample:
    video_id: str
    label: int
    dataset_name: str
    frames: list[FaceSample]
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


def _resolve_manifest_image_path(image_path: Path, spec: DatasetSpec) -> Path:
    if image_path.is_absolute():
        return image_path

    dataset_root = Path(spec.root).resolve()
    processed_root = _processed_root(spec).resolve()
    data_root = dataset_root.parent if dataset_root.name in {"baseline", "stages"} else dataset_root
    path_parts = image_path.parts

    if path_parts and path_parts[0] == "data":
        return data_root.joinpath(*path_parts[1:]).resolve()

    if path_parts and path_parts[0] in {"processed", "stages", "baseline"}:
        return data_root.joinpath(*path_parts).resolve()

    if dataset_root.name in path_parts:
        suffix = path_parts[path_parts.index(dataset_root.name) + 1 :]
        return dataset_root.joinpath(*suffix).resolve()

    if processed_root.name in path_parts:
        suffix = path_parts[path_parts.index(processed_root.name) + 1 :]
        return processed_root.joinpath(*suffix).resolve()

    # First try the most direct interpretation: path relative to the dataset root.
    direct_candidate = (dataset_root / image_path).resolve()
    if direct_candidate.exists():
        return direct_candidate

    # Stage manifests often store paths like `data/stages/stage1_sbi/...`; anchor them back to
    # the actual dataset root mounted on Drive instead of the repo checkout.
    root_parts = dataset_root.parts
    for anchor_index in range(len(root_parts) - 1, -1, -1):
        anchor = root_parts[anchor_index]
        if anchor in path_parts:
            suffix = path_parts[path_parts.index(anchor) + 1 :]
            return dataset_root.joinpath(*suffix).resolve()

    processed_parts = processed_root.parts
    for anchor_index in range(len(processed_parts) - 1, -1, -1):
        anchor = processed_parts[anchor_index]
        if anchor in path_parts:
            suffix = path_parts[path_parts.index(anchor) + 1 :]
            return processed_root.joinpath(*suffix).resolve()

    # Fall back to preserving the relative structure under the dataset root so callers still get
    # a deterministic path even if the file is missing.
    return (dataset_root / image_path).resolve()


def _load_manifest(manifest_path: Path, spec: DatasetSpec) -> list[FaceSample]:
    if not manifest_path.exists():
        return []
    samples: list[FaceSample] = []
    with manifest_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            payload = json.loads(line)
            samples.append(
                FaceSample(
                    image_path=_resolve_manifest_image_path(Path(payload["image_path"]), spec),
                    label=int(payload["label"]),
                    dataset_name=payload["dataset_name"],
                    video_id=payload["video_id"],
                    frame_index=int(payload["frame_index"]),
                    source_video=payload.get("source_video", ""),
                    split=payload.get("split", ""),
                )
            )
    return samples


def _fallback_stage1_path(path: Path, spec: DatasetSpec) -> Path:
    dataset_root = Path(spec.root).resolve()
    if not dataset_root.name.startswith("stage"):
        return path

    split = spec.split
    data_root = dataset_root.parents[1]
    stem_parts = path.stem.split("__")
    label_dir = path.parent.name
    if label_dir == "FaceToFace":
        label_dir = "Face2Face"

    # Stage 1 real files are materialized under `train/real` or `val/real`, but on Drive those
    # symlinks are often missing. Recover the original processed FF++ frame directly.
    if len(stem_parts) == 4 and stem_parts[0] == "FF++" and stem_parts[1] == "original":
        video_id = stem_parts[2]
        frame_index = stem_parts[3]
        ffpp_split = "train" if split in {"train", "val"} else "test"
        return (
            data_root
            / "processed"
            / "ffpp_generalization"
            / ffpp_split
            / "original"
            / video_id
            / f"{frame_index}.png"
        )

    # Stage 1 SBI fakes are generated under `_generated_sbi/<split>/fake`, but manifests point to
    # materialized `train/fake` entries such as `SBI__SBI__sbi_846_01528__01528.png`.
    if len(stem_parts) == 4 and stem_parts[0] == "SBI" and stem_parts[1] == "SBI":
        video_id = stem_parts[2]
        frame_index = stem_parts[3]
        prefix = "sbi_"
        if video_id.startswith(prefix):
            original = video_id[len(prefix) :]
            original_video = original.rsplit("_", 1)[0]
            return (
                dataset_root
                / "_generated_sbi"
                / split
                / "fake"
                / f"sbi__FF++__{original_video}__{frame_index}.png"
            )

    # Stage test splits are often materialized as symlinks too. Recover the underlying processed
    # FF++ or Celeb-DF frame directly when those symlinks are not present on Drive.
    if len(stem_parts) == 4 and stem_parts[0] == "FF++":
        manipulation = stem_parts[1]
        video_id = stem_parts[2]
        frame_index = stem_parts[3]
        return (
            data_root
            / "processed"
            / "ffpp_generalization"
            / "test"
            / manipulation
            / video_id
            / f"{frame_index}.png"
        )

    if len(stem_parts) == 4 and stem_parts[0] == "CelebDF":
        video_id = stem_parts[2]
        frame_index = stem_parts[3]
        return (
            data_root
            / "processed"
            / "celebdf"
            / "test"
            / label_dir
            / f"{video_id}_frame_{frame_index}.png"
        )

    return path


def _fallback_baseline_path(path: Path, spec: DatasetSpec) -> Path:
    dataset_root = Path(spec.root).resolve()
    if dataset_root.name != "baseline":
        return path

    data_root = dataset_root.parent
    label_dir = path.parent.parent.name if len(path.parents) >= 2 else ""
    if label_dir == "FaceToFace":
        label_dir = "Face2Face"
    video_id = path.parent.name
    frame_name = path.name
    split_name = path.parts[-4] if len(path.parts) >= 4 else spec.split

    if split_name in {"train", "val"}:
        return (
            data_root
            / "processed"
            / "ffpp_generalization"
            / "train"
            / label_dir
            / video_id
            / frame_name
        )

    if split_name == "test_ffpp":
        return (
            data_root
            / "processed"
            / "ffpp_generalization"
            / "test"
            / label_dir
            / video_id
            / frame_name
        )

    if split_name == "test_celebdf":
        frame_stem = Path(frame_name).stem
        return data_root / "processed" / "celebdf" / "test" / label_dir / f"{video_id}_frame_{frame_stem}.png"

    return path


def _resolve_existing_image_path(path: Path, spec: DatasetSpec) -> Path:
    if path.exists():
        return path

    fallback = _fallback_stage1_path(path, spec)
    if fallback != path:
        return fallback
    fallback = _fallback_baseline_path(path, spec)
    if fallback != path:
        return fallback
    return path


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
        manifest_samples = _load_manifest(_manifest_path(self.spec), self.spec)
        if manifest_samples:
            return manifest_samples
        return _infer_samples_from_processed_dir(self.spec)

    def __len__(self) -> int:
        return len(self.samples)

    def load_image(self, path: Path) -> Image.Image:
        path = _resolve_existing_image_path(path, self.spec)
        return Image.open(path).convert("RGB")

    def __getitem__(self, index: int) -> tuple[Tensor, int]:
        sample = self.samples[index]
        image = self.load_image(sample.image_path)
        tensor = self.transform(image) if self.transform is not None else image
        return tensor, sample.label


class VideoFaceForgeryDataset(Dataset):
    """Video-style dataset that returns stacked frame tensors per sample."""

    def __init__(self, spec: DatasetSpec, transform=None) -> None:
        self.spec = spec
        self.transform = transform or build_eval_transforms(spec)
        self.samples = self._index_samples()

    def _index_samples(self) -> list[VideoSample]:
        manifest_samples = _load_manifest(_manifest_path(self.spec), self.spec)
        if not manifest_samples:
            manifest_samples = _infer_samples_from_processed_dir(self.spec)

        grouped: dict[tuple[str, str], list[FaceSample]] = {}
        for sample in manifest_samples:
            key = (sample.dataset_name, sample.video_id)
            grouped.setdefault(key, []).append(sample)

        video_samples: list[VideoSample] = []
        for (_, video_id), frames in grouped.items():
            ordered_frames = sorted(frames, key=lambda sample: sample.frame_index)
            first = ordered_frames[0]
            video_samples.append(
                VideoSample(
                    video_id=video_id,
                    label=first.label,
                    dataset_name=first.dataset_name,
                    frames=ordered_frames,
                    source_video=first.source_video,
                    split=first.split,
                )
            )
        video_samples.sort(key=lambda sample: (sample.dataset_name, sample.video_id))
        return video_samples

    def __len__(self) -> int:
        return len(self.samples)

    def load_image(self, path: Path) -> Image.Image:
        path = _resolve_existing_image_path(path, self.spec)
        return Image.open(path).convert("RGB")

    def _select_frames(self, frames: list[FaceSample]) -> list[FaceSample]:
        target_frames = max(int(self.spec.frames_per_video), 1)
        if len(frames) <= target_frames:
            return frames
        if target_frames == 1:
            return [frames[len(frames) // 2]]

        step = (len(frames) - 1) / float(target_frames - 1)
        indices = [min(int(round(step * index)), len(frames) - 1) for index in range(target_frames)]
        deduplicated_indices: list[int] = []
        for index in indices:
            if not deduplicated_indices or deduplicated_indices[-1] != index:
                deduplicated_indices.append(index)
        return [frames[index] for index in deduplicated_indices]

    def __getitem__(self, index: int) -> tuple[Tensor, int]:
        video = self.samples[index]
        selected_frames = self._select_frames(video.frames)
        frame_tensors = []
        for frame in selected_frames:
            image = self.load_image(frame.image_path)
            tensor = self.transform(image) if self.transform is not None else image
            frame_tensors.append(tensor)
        if not frame_tensors:
            raise ValueError(f"Video sample {video.video_id} does not contain any frames.")
        return torch.stack(frame_tensors, dim=0), video.label


def build_dataset(spec: DatasetSpec, transform=None) -> Dataset:
    if spec.group_by_video:
        return VideoFaceForgeryDataset(spec=spec, transform=transform)
    return FaceForgeryDataset(spec=spec, transform=transform)


def build_combined_dataset(specs: list[DatasetSpec]) -> Dataset:
    datasets = [build_dataset(spec) for spec in specs]
    if len(datasets) == 1:
        return datasets[0]
    return ConcatDataset(datasets)
