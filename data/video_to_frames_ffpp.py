"""Simple FaceForensics++ video-to-face-frame preprocessing."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import cv2
import numpy as np
from facenet_pytorch import MTCNN
from PIL import Image


ALLOWED_SUBSETS = ["original", "Deepfakes", "Face2Face", "FaceSwap"]


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


def collect_videos(root: Path) -> dict[str, list[Path]]:
    videos_by_subset: dict[str, list[Path]] = {}
    for subset in ALLOWED_SUBSETS:
        subset_dir = root / subset
        if not subset_dir.exists():
            videos_by_subset[subset] = []
            continue
        videos = sorted(subset_dir.glob("*.mp4")) + sorted(subset_dir.glob("*.avi"))
        videos_by_subset[subset] = videos
    return videos_by_subset


def split_videos(video_paths: list[Path], train_ratio: float, val_ratio: float, seed: int) -> dict[str, list[Path]]:
    video_paths = video_paths[:]
    random.Random(seed).shuffle(video_paths)

    train_size = int(len(video_paths) * train_ratio)
    val_size = int(len(video_paths) * val_ratio)

    return {
        "train": video_paths[:train_size],
        "val": video_paths[train_size : train_size + val_size],
        "test": video_paths[train_size + val_size :],
    }


def get_label(subset: str) -> int:
    return 0 if subset == "original" else 1


def extract_faces_from_video(
    video_path: Path,
    subset: str,
    split: str,
    detector: MTCNN,
    processed_root: Path,
    frames_per_video: int,
    image_size: int,
    margin: int,
    overwrite: bool,
) -> list[dict]:
    capture = cv2.VideoCapture(str(video_path))
    total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_indices = set(sample_frame_indices(total_frames, frames_per_video))

    label = get_label(subset)
    class_name = "fake" if label == 1 else "real"
    video_id = video_path.stem
    save_dir = processed_root / split / class_name
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
        face = crop_face(image, detector, margin)

        if face is not None:
            output_path = save_dir / f"{subset.lower()}_{video_id}_frame_{frame_idx:05d}.png"
            if overwrite or not output_path.exists():
                face.resize((image_size, image_size)).save(output_path)

            samples.append(
                {
                    "image_path": str(output_path),
                    "label": label,
                    "dataset_name": "FaceForensics++_C23",
                    "split": split,
                    "video_id": video_id,
                    "frame_index": frame_idx,
                    "source_video": str(video_path),
                    "manipulation_type": subset,
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


def prepare_faceforensicspp(
    root: str = "data/raw/FaceForensics++_C23",
    processed_root: str = "data/processed/faceforensicspp_c23",
    frames_per_video: int = 8,
    image_size: int = 224,
    margin: int = 24,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    seed: int = 42,
    device: str | None = None,
    overwrite: bool = False,
) -> dict[str, Path]:
    root_path = Path(root)
    output_root = Path(processed_root)
    detector_device = device
    if detector_device == "mps":
        print("MPS is unstable for MTCNN face detection on arbitrary frame sizes. Falling back to CPU detector.")
        detector_device = "cpu"
    detector = MTCNN(keep_all=False, device=detector_device)

    videos_by_subset = collect_videos(root_path)
    split_map = {"train": [], "val": [], "test": []}

    for subset in ALLOWED_SUBSETS:
        subset_videos = videos_by_subset[subset]
        subset_splits = split_videos(subset_videos, train_ratio=train_ratio, val_ratio=val_ratio, seed=seed)
        print(f"{subset}: {len(subset_videos)} videos")
        for split_name in split_map:
            split_map[split_name].extend((subset, path) for path in subset_splits[split_name])

    manifest_paths: dict[str, Path] = {}

    for split_name, entries in split_map.items():
        all_samples = []
        print(f"{split_name}: {len(entries)} videos")

        for index, (subset, video_path) in enumerate(entries, start=1):
            samples = extract_faces_from_video(
                video_path=video_path,
                subset=subset,
                split=split_name,
                detector=detector,
                processed_root=output_root,
                frames_per_video=frames_per_video,
                image_size=image_size,
                margin=margin,
                overwrite=overwrite,
            )
            all_samples.extend(samples)

            if index == 1 or index % 25 == 0 or index == len(entries):
                print(f"[{split_name}] {index}/{len(entries)} {subset}/{video_path.name} -> {len(samples)} faces")

        manifest_path = output_root / f"faceforensicspp_c23_{split_name}_manifest.jsonl"
        write_manifest(all_samples, manifest_path)
        manifest_paths[split_name] = manifest_path

        print(f"{split_name} done: {len(all_samples)} face images")
        print(f"{split_name} manifest: {manifest_path}")

    return manifest_paths


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Frame FaceForensics++ C23 videos into face crops.")
    parser.add_argument("--root", type=str, default="data/raw/FaceForensics++_C23")
    parser.add_argument("--processed-root", type=str, default="data/processed/faceforensicspp_c23")
    parser.add_argument("--frames-per-video", type=int, default=8)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--margin", type=int, default=24)
    parser.add_argument("--train-ratio", type=float, default=0.7)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    prepare_faceforensicspp(
        root=args.root,
        processed_root=args.processed_root,
        frames_per_video=args.frames_per_video,
        image_size=args.image_size,
        margin=args.margin,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        seed=args.seed,
        device=args.device,
        overwrite=args.overwrite,
    )


if __name__ == "__main__":
    main()
