"""Simple video to face-frame preprocessing."""

from __future__ import annotations

import json
import random
from pathlib import Path

import cv2
import numpy as np
from facenet_pytorch import MTCNN
from PIL import Image

from utils.config import DatasetSpec


def get_processed_root(spec: DatasetSpec) -> Path:
    if spec.processed_root:
        return Path(spec.processed_root)
    return Path("data/processed") / spec.name.lower()


def get_manifest_path(spec: DatasetSpec) -> Path:
    if spec.manifest_path:
        return Path(spec.manifest_path)
    return get_processed_root(spec) / f"{spec.name.lower()}_{spec.split}_manifest.jsonl"


def get_video_paths(spec: DatasetSpec) -> list[Path]:
    root = Path(spec.root)
    video_paths = sorted(root.rglob("*.mp4")) + sorted(root.rglob("*.avi"))
    if spec.max_videos is not None:
        video_paths = video_paths[: spec.max_videos]
    return video_paths


def get_label(dataset_name: str, video_path: Path) -> int:
    path_text = str(video_path).lower()
    if dataset_name.lower() in {"celebdf", "celebdf-v2", "celebdfv2"}:
        if "celeb-synthesis" in path_text:
            return 1
        return 0
    if dataset_name.lower() in {"faceforensics++", "faceforensicspp", "ff++"}:
        if "manipulated_sequences" in path_text:
            return 1
        return 0
    return 0


def sample_frame_indices(total_frames: int, frames_per_video: int) -> list[int]:
    if total_frames <= frames_per_video:
        return list(range(total_frames))
    return np.linspace(0, total_frames - 1, frames_per_video, dtype=int).tolist()


def crop_face(image: Image.Image, detector: MTCNN, margin: int) -> Image.Image | None:
    boxes, _ = detector.detect(image)
    if boxes is None:
        return None

    x1, y1, x2, y2 = boxes[0]
    width, height = image.size
    x1 = max(0, int(x1) - margin)
    y1 = max(0, int(y1) - margin)
    x2 = min(width, int(x2) + margin)
    y2 = min(height, int(y2) + margin)
    return image.crop((x1, y1, x2, y2))


def extract_faces_from_video(video_path: Path, spec: DatasetSpec, detector: MTCNN) -> list[dict]:
    capture = cv2.VideoCapture(str(video_path))
    total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_indices = set(sample_frame_indices(total_frames, spec.frames_per_video))
    label = get_label(spec.name, video_path)
    video_id = video_path.stem
    save_dir = get_processed_root(spec) / spec.split / ("fake" if label == 1 else "real")
    save_dir.mkdir(parents=True, exist_ok=True)

    samples = []
    frame_idx = 0

    while True:
        ok, frame = capture.read()
        if not ok:
            break

        if frame_idx not in frame_indices:
            frame_idx += 1
            continue

        image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        face = crop_face(image, detector, spec.detector_margin)

        if face is not None:
            output_path = save_dir / f"{video_id}_frame_{frame_idx:05d}.png"
            if spec.overwrite_processed or not output_path.exists():
                face.resize((spec.image_size, spec.image_size)).save(output_path)

            samples.append(
                {
                    "image_path": str(output_path),
                    "label": label,
                    "dataset_name": spec.name,
                    "split": spec.split,
                    "video_id": video_id,
                    "frame_index": frame_idx,
                    "source_video": str(video_path),
                }
            )

        frame_idx += 1

    capture.release()
    return samples


def write_manifest(samples: list[dict], manifest_path: Path) -> None:
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with open(manifest_path, "w", encoding="utf-8") as handle:
        for sample in samples:
            handle.write(json.dumps(sample) + "\n")


def prepare_dataset_frames(spec: DatasetSpec, device: str | None = None) -> Path:
    detector = MTCNN(keep_all=False, device=device)
    video_paths = get_video_paths(spec)
    all_samples = []

    print(f"Found {len(video_paths)} videos")

    for index, video_path in enumerate(video_paths, start=1):
        samples = extract_faces_from_video(video_path, spec, detector)
        all_samples.extend(samples)

        if index == 1 or index % 25 == 0 or index == len(video_paths):
            print(f"[{index}/{len(video_paths)}] {video_path.name} -> {len(samples)} faces")

    manifest_path = get_manifest_path(spec)
    write_manifest(all_samples, manifest_path)

    print(f"Saved {len(all_samples)} face images")
    print(f"Manifest: {manifest_path}")
    return manifest_path


def split_videos(video_paths: list[Path], train_ratio: float, val_ratio: float, seed: int) -> dict[str, list[Path]]:
    video_paths = video_paths[:]
    random.Random(seed).shuffle(video_paths)

    train_size = int(len(video_paths) * train_ratio)
    val_size = int(len(video_paths) * val_ratio)

    train_videos = video_paths[:train_size]
    val_videos = video_paths[train_size : train_size + val_size]
    test_videos = video_paths[train_size + val_size :]

    return {
        "train": train_videos,
        "val": val_videos,
        "test": test_videos,
    }


def prepare_balanced_celebdf(
    root: str,
    processed_root: str,
    frames_per_video: int = 8,
    image_size: int = 224,
    detector_margin: int = 24,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    seed: int = 42,
    device: str | None = None,
    overwrite: bool = False,
) -> dict[str, Path]:
    all_videos = sorted(Path(root).rglob("*.mp4")) + sorted(Path(root).rglob("*.avi"))

    real_videos = [video for video in all_videos if get_label("CelebDF", video) == 0]
    fake_videos = [video for video in all_videos if get_label("CelebDF", video) == 1]

    random.Random(seed).shuffle(real_videos)
    random.Random(seed).shuffle(fake_videos)

    target_count = min(len(real_videos), len(fake_videos))
    real_videos = real_videos[:target_count]
    fake_videos = fake_videos[:target_count]

    real_splits = split_videos(real_videos, train_ratio=train_ratio, val_ratio=val_ratio, seed=seed)
    fake_splits = split_videos(fake_videos, train_ratio=train_ratio, val_ratio=val_ratio, seed=seed)

    detector = MTCNN(keep_all=False, device=device)
    manifest_paths = {}

    for split_name in ["train", "val", "test"]:
        spec = DatasetSpec(
            name="CelebDF",
            root=root,
            split=split_name,
            frames_per_video=frames_per_video,
            image_size=image_size,
            processed_root=processed_root,
            detector_margin=detector_margin,
            overwrite_processed=overwrite,
        )

        split_videos_list = real_splits[split_name] + fake_splits[split_name]
        all_samples = []

        print(
            f"{split_name}: {len(real_splits[split_name])} real videos, "
            f"{len(fake_splits[split_name])} fake videos"
        )

        for index, video_path in enumerate(split_videos_list, start=1):
            samples = extract_faces_from_video(video_path, spec, detector)
            all_samples.extend(samples)
            if index == 1 or index % 25 == 0 or index == len(split_videos_list):
                print(f"[{split_name}] {index}/{len(split_videos_list)} {video_path.name} -> {len(samples)} faces")

        manifest_path = get_manifest_path(spec)
        write_manifest(all_samples, manifest_path)
        manifest_paths[split_name] = manifest_path

        print(f"{split_name} done: {len(all_samples)} face images")
        print(f"{split_name} manifest: {manifest_path}")

    return manifest_paths
