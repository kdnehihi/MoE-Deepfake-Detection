"""Video-to-frame extraction and face crop preparation."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(slots=True)
class FrameExtractionJob:
    dataset_name: str
    video_path: Path
    output_dir: Path
    label: int
    frames_per_video: int


def extract_faces_from_video(job: FrameExtractionJob) -> list[Path]:
    raise NotImplementedError("Video frame extraction is implemented in Step 4.")

