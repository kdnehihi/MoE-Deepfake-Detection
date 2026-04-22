"""Structured configuration objects for the MoE-FFD project."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass(slots=True)
class DatasetSpec:
    name: str
    root: str
    split: str
    frames_per_video: int = 8
    group_by_video: bool = False
    image_size: int = 224
    face_detector: str = "mtcnn"
    processed_root: str | None = None
    manifest_path: str | None = None
    max_videos: int | None = None
    detector_margin: int = 24
    overwrite_processed: bool = False


@dataclass(slots=True)
class BackboneConfig:
    model_name: str = "vit_base_patch16_224_in21k"
    pretrained: bool = True
    image_size: int = 224
    freeze: bool = True
    embed_dim: int = 768


@dataclass(slots=True)
class GatingConfig:
    hidden_dim: int = 256
    top_k: int = 1
    use_cls_token: bool = True
    noisy_gating: bool = True
    noise_epsilon: float = 1e-2


@dataclass(slots=True)
class LoRAExpertConfig:
    rank: int
    alpha: float
    dropout: float = 0.0


@dataclass(slots=True)
class AdapterExpertConfig:
    name: str
    bottleneck_dim: int
    kernel_size: int = 3


@dataclass(slots=True)
class MoEConfig:
    lora_experts: list[LoRAExpertConfig] = field(
        default_factory=lambda: [
            LoRAExpertConfig(rank=8, alpha=8.0),
            LoRAExpertConfig(rank=16, alpha=16.0),
            LoRAExpertConfig(rank=32, alpha=32.0),
            LoRAExpertConfig(rank=48, alpha=48.0),
            LoRAExpertConfig(rank=64, alpha=64.0),
            LoRAExpertConfig(rank=96, alpha=96.0),
            LoRAExpertConfig(rank=128, alpha=128.0),
        ]
    )
    adapter_experts: list[AdapterExpertConfig] = field(
        default_factory=lambda: [
            AdapterExpertConfig(name="vanilla_conv", bottleneck_dim=8),
            AdapterExpertConfig(name="adc", bottleneck_dim=8),
            AdapterExpertConfig(name="cdc", bottleneck_dim=8),
            AdapterExpertConfig(name="rdc", bottleneck_dim=8),
            AdapterExpertConfig(name="soc", bottleneck_dim=8),
        ]
    )
    top_k: int = 1
    load_balance_weight: float = 1.0
    lora_balance_scale: float = 200.0
    adapter_balance_scale: float = 1.0


@dataclass(slots=True)
class ClassifierConfig:
    num_classes: int = 2
    dropout: float = 0.0


@dataclass(slots=True)
class StageConfig:
    name: str = "stage3"
    enable_lora: bool = True
    enable_adapter: bool = True
    enable_moe_router: bool = True
    enable_classifier: bool = True


@dataclass(slots=True)
class ModelConfig:
    backbone: BackboneConfig = field(default_factory=BackboneConfig)
    gating: GatingConfig = field(default_factory=GatingConfig)
    moe: MoEConfig = field(default_factory=MoEConfig)
    classifier: ClassifierConfig = field(default_factory=ClassifierConfig)
    stage: StageConfig = field(default_factory=StageConfig)


@dataclass(slots=True)
class OptimizerConfig:
    optimizer: str = "adam"
    lr_gating: float = 1e-4
    lr_base: float = 3e-5
    weight_decay: float = 1e-5
    scheduler_step_size: int = 5
    scheduler_gamma: float = 0.5


@dataclass(slots=True)
class TrainConfig:
    batch_size: int = 32
    num_workers: int = 4
    epochs: int = 20
    amp: bool = True
    seed: int = 42


@dataclass(slots=True)
class ProjectConfig:
    model: ModelConfig = field(default_factory=ModelConfig)
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    datasets: list[DatasetSpec] = field(default_factory=list)
    output_dir: str = "outputs"


def load_config(path: str | Path) -> ProjectConfig:
    """Load project configuration from a YAML file."""
    config_path = Path(path)
    with config_path.open("r", encoding="utf-8") as handle:
        payload: dict[str, Any] = yaml.safe_load(handle) or {}
    model_payload = payload.get("model", {})
    moe_payload = model_payload.get("moe", {})

    config = ProjectConfig(
        model=ModelConfig(
            backbone=BackboneConfig(**model_payload.get("backbone", {})),
            gating=GatingConfig(**model_payload.get("gating", {})),
            moe=MoEConfig(
                lora_experts=[
                    LoRAExpertConfig(**expert_payload) for expert_payload in moe_payload.get("lora_experts", [])
                ]
                or MoEConfig().lora_experts,
                adapter_experts=[
                    AdapterExpertConfig(**expert_payload)
                    for expert_payload in moe_payload.get("adapter_experts", [])
                ]
                or MoEConfig().adapter_experts,
                top_k=moe_payload.get("top_k", MoEConfig().top_k),
                load_balance_weight=moe_payload.get(
                    "load_balance_weight",
                    MoEConfig().load_balance_weight,
                ),
                lora_balance_scale=moe_payload.get(
                    "lora_balance_scale",
                    MoEConfig().lora_balance_scale,
                ),
                adapter_balance_scale=moe_payload.get(
                    "adapter_balance_scale",
                    MoEConfig().adapter_balance_scale,
                ),
            ),
            classifier=ClassifierConfig(**model_payload.get("classifier", {})),
            stage=StageConfig(**model_payload.get("stage", {})),
        ),
        optimizer=OptimizerConfig(**payload.get("optimizer", {})),
        train=TrainConfig(**payload.get("train", {})),
        datasets=[DatasetSpec(**dataset_payload) for dataset_payload in payload.get("datasets", [])],
        output_dir=payload.get("output_dir", "outputs"),
    )
    return config
