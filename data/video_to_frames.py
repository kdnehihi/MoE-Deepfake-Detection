"""Video-to-frame extraction and face crop preparation."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Iterable

import numpy as np
from PIL import Image

from utils.config import DatasetSpec

@dataclass(slots=True)
class FrameExtractionJob:
    dataset_name: str
    video_path: Path
    output_dir: Path
    label: int
    frames_per_video: int
    split: str
    image_size: int = 224
    detector_name: str = "mtcnn"
    detector_margin: int = 24
    overwrite: bool = False


@dataclass(slots=True)
class ExtractedFaceSample:
    image_path: str
    label: int
    dataset_name: str
    split: str
    video_id: str
    frame_index: int
    source_video: str


def _ensure_mtcnn(device: str | None = None):
    try:
        from facenet_pytorch import MTCNN
    except ImportError as error:
        raise ImportError(
            "facenet-pytorch is required for face extraction. Install it before running frame extraction."
        ) from error

    return MTCNN(
        image_size=None,
        margin=0,
        keep_all=False,
        post_process=False,
        device=device,
    )


def _open_video(video_path: Path):
    try:
        import cv2
    except ImportError as error:
        raise ImportError("opencv-python is required for video decoding.") from error

    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"Unable to open video: {video_path}")
    return capture, cv2


def _sample_frame_indices(total_frames: int, frames_per_video: int) -> list[int]:
    if total_frames <= 0:
        return []
    if total_frames <= frames_per_video:
        return list(range(total_frames))
    indices = np.linspace(0, total_frames - 1, num=frames_per_video, dtype=int)
    return sorted(set(indices.tolist()))


def _infer_video_id(video_path: Path) -> str:
    return video_path.stem


def _crop_face(image: Image.Image, detector, margin: int) -> Image.Image | None:
    boxes, _ = detector.detect(image)
    if boxes is None or len(boxes) == 0:
        return None

    x1, y1, x2, y2 = boxes[0]
    width, height = image.size
    x1 = max(int(x1) - margin, 0)
    y1 = max(int(y1) - margin, 0)
    x2 = min(int(x2) + margin, width)
    y2 = min(int(y2) + margin, height)
    if x2 <= x1 or y2 <= y1:
        return None
    return image.crop((x1, y1, x2, y2))


def extract_faces_from_video(job: FrameExtractionJob, detector=None) -> list[ExtractedFaceSample]:
    detector = detector or _ensure_mtcnn()
    capture, cv2 = _open_video(job.video_path)
    total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    sampled_indices = set(_sample_frame_indices(total_frames, job.frames_per_video))
    video_id = _infer_video_id(job.video_path)

    job.output_dir.mkdir(parents=True, exist_ok=True)
    samples: list[ExtractedFaceSample] = []
    frame_index = 0

    while True:
        success, frame = capture.read()
        if not success:
            break
        if frame_index not in sampled_indices:
            frame_index += 1
            continue

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(rgb_frame)
        face = _crop_face(image=image, detector=detector, margin=job.detector_margin)
        if face is None:
            frame_index += 1
            continue

        output_name = f"{video_id}_frame_{frame_index:05d}.png"
        output_path = job.output_dir / output_name
        if job.overwrite or not output_path.exists():
            face.resize((job.image_size, job.image_size), Image.Resampling.BILINEAR).save(output_path)

        samples.append(
            ExtractedFaceSample(
                image_path=str(output_path),
                label=job.label,
                dataset_name=job.dataset_name,
                split=job.split,
                video_id=video_id,
                frame_index=frame_index,
                source_video=str(job.video_path),
            )
        )
        frame_index += 1

    capture.release()
    return samples


def _processed_root(spec: DatasetSpec) -> Path:
    if spec.processed_root is not None:
        return Path(spec.processed_root)
    return Path(spec.root) / "processed_faces"


def _manifest_path(spec: DatasetSpec) -> Path:
    if spec.manifest_path is not None:
        return Path(spec.manifest_path)
    return _processed_root(spec) / f"{spec.name.lower()}_{spec.split}_manifest.jsonl"


def _detect_label_from_path(dataset_name: str, video_path: Path) -> int:
    normalized_name = dataset_name.lower()
    normalized_path = str(video_path).lower()
    if normalized_name in {"celebdf", "celebdf-v2", "celebdfv2"}:
        return 0 if any(tag in normalized_path for tag in ("celeb-real", "youtube-real", "real")) else 1
    if normalized_name in {"faceforensics++", "faceforensicspp", "ff++"}:
        return 0 if "original_sequences" in normalized_path else 1
    raise ValueError(f"Unsupported dataset name for label inference: {dataset_name}")


def _discover_celebdf_videos(spec: DatasetSpec) -> list[Path]:
    root = Path(spec.root)
    candidates = [
        root / spec.split,
        root,
    ]
    video_files: list[Path] = []
    for candidate in candidates:
        if candidate.exists():
            video_files.extend(sorted(candidate.rglob("*.mp4")))
            video_files.extend(sorted(candidate.rglob("*.avi")))
    return sorted(set(video_files))


def _discover_faceforensics_videos(spec: DatasetSpec) -> list[Path]:
    root = Path(spec.root)
    split_root = root / spec.split if (root / spec.split).exists() else root
    video_files = sorted(split_root.rglob("*.mp4"))
    return video_files


def discover_extraction_jobs(spec: DatasetSpec) -> list[FrameExtractionJob]:
    normalized_name = spec.name.lower()
    if normalized_name in {"celebdf", "celebdf-v2", "celebdfv2"}:
        video_files = _discover_celebdf_videos(spec)
    elif normalized_name in {"faceforensics++", "faceforensicspp", "ff++"}:
        video_files = _discover_faceforensics_videos(spec)
    else:
        raise ValueError(f"Unsupported dataset: {spec.name}")

    if spec.max_videos is not None:
        video_files = video_files[: spec.max_videos]

    output_root = _processed_root(spec) / spec.split
    jobs = [
        FrameExtractionJob(
            dataset_name=spec.name,
            video_path=video_path,
            output_dir=output_root / ("fake" if _detect_label_from_path(spec.name, video_path) == 1 else "real"),
            label=_detect_label_from_path(spec.name, video_path),
            frames_per_video=spec.frames_per_video,
            split=spec.split,
            image_size=spec.image_size,
            detector_name=spec.face_detector,
            detector_margin=spec.detector_margin,
            overwrite=spec.overwrite_processed,
        )
        for video_path in video_files
    ]
    return jobs


def write_manifest(samples: Iterable[ExtractedFaceSample], manifest_path: Path) -> None:
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with manifest_path.open("w", encoding="utf-8") as handle:
        for sample in samples:
            handle.write(json.dumps(asdict(sample), ensure_ascii=True) + "\n")


def prepare_dataset_frames(spec: DatasetSpec, device: str | None = None) -> Path:
    detector = _ensure_mtcnn(device=device)
    manifest_samples: list[ExtractedFaceSample] = []
    for job in discover_extraction_jobs(spec):
        manifest_samples.extend(extract_faces_from_video(job=job, detector=detector))

    manifest_path = _manifest_path(spec)
    write_manifest(manifest_samples, manifest_path)
    return manifest_path
