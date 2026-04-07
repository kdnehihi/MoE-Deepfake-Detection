"""Image transform builders for training and evaluation."""

from __future__ import annotations

import random

import numpy as np
import torch
from PIL import Image, ImageEnhance

from utils.config import DatasetSpec


class Compose:
    def __init__(self, transforms_list):
        self.transforms_list = transforms_list

    def __call__(self, image):
        for transform in self.transforms_list:
            image = transform(image)
        return image


class Resize:
    def __init__(self, size: tuple[int, int]) -> None:
        self.size = size

    def __call__(self, image: Image.Image) -> Image.Image:
        return image.resize(self.size, Image.Resampling.BILINEAR)


class RandomHorizontalFlip:
    def __init__(self, p: float = 0.5) -> None:
        self.p = p

    def __call__(self, image: Image.Image) -> Image.Image:
        if random.random() < self.p:
            return image.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
        return image


class ColorJitter:
    def __init__(self, brightness: float, contrast: float, saturation: float, hue: float) -> None:
        self.brightness = brightness
        self.contrast = contrast
        self.saturation = saturation
        self.hue = hue

    def __call__(self, image: Image.Image) -> Image.Image:
        image = ImageEnhance.Brightness(image).enhance(1.0 + random.uniform(-self.brightness, self.brightness))
        image = ImageEnhance.Contrast(image).enhance(1.0 + random.uniform(-self.contrast, self.contrast))
        image = ImageEnhance.Color(image).enhance(1.0 + random.uniform(-self.saturation, self.saturation))
        if self.hue > 0:
            hsv = np.array(image.convert("HSV"), dtype=np.uint8)
            hue_shift = int(255 * random.uniform(-self.hue, self.hue))
            hsv[..., 0] = (hsv[..., 0].astype(int) + hue_shift) % 255
            image = Image.fromarray(hsv, mode="HSV").convert("RGB")
        return image


class ToTensor:
    def __call__(self, image: Image.Image) -> torch.Tensor:
        array = np.asarray(image, dtype=np.float32) / 255.0
        return torch.from_numpy(array).permute(2, 0, 1).contiguous()


class Normalize:
    def __init__(self, mean: tuple[float, float, float], std: tuple[float, float, float]) -> None:
        self.mean = torch.tensor(mean).view(3, 1, 1)
        self.std = torch.tensor(std).view(3, 1, 1)

    def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
        return (tensor - self.mean) / self.std


def _build_compose(transforms_list):
    try:
        from torchvision import transforms
    except ImportError:
        return Compose(transforms_list)
    return transforms.Compose(transforms_list)


def build_train_transforms(spec: DatasetSpec):
    return _build_compose(
        [
            Resize((spec.image_size, spec.image_size)),
            RandomHorizontalFlip(p=0.5),
            ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.02),
            ToTensor(),
            Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ]
    )


def build_eval_transforms(spec: DatasetSpec):
    return _build_compose(
        [
            Resize((spec.image_size, spec.image_size)),
            ToTensor(),
            Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ]
    )
