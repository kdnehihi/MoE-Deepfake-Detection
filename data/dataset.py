"""Dataset definitions for frame-based face forgery detection."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from PIL import Image
from torch import Tensor
from torch.utils.data import Dataset

from utils.config import DatasetSpec


@dataclass(slots=True)
class FaceSample:
    image_path: Path
    label: int
    dataset_name: str
    video_id: str
    frame_index: int


class FaceForgeryDataset(Dataset):
    """
    Unified dataset interface across Celeb-DF and additional sources.

    The full indexing and preprocessing logic is added in the dataset step once
    frame extraction and metadata normalization are implemented.
    """

    def __init__(self, spec: DatasetSpec, transform=None) -> None:
        self.spec = spec
        self.transform = transform
        self.samples: list[FaceSample] = []

    def __len__(self) -> int:
        return len(self.samples)

    def load_image(self, path: Path) -> Image.Image:
        return Image.open(path).convert("RGB")

    def __getitem__(self, index: int) -> tuple[Tensor, int]:
        raise NotImplementedError("FaceForgeryDataset.__getitem__ is implemented in Step 4.")

