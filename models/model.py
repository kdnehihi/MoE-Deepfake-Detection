"""Top-level detector model for the MoE-FFD reproduction."""

from __future__ import annotations

from dataclasses import dataclass

from torch import Tensor, nn

from models.transformer_block import BlockAuxiliaryOutput, MoETransformerBlock
from models.vit_backbone import FrozenViTBackbone
from utils.config import ModelConfig


@dataclass(slots=True)
class ModelAuxiliaryOutput:
    blocks: list[BlockAuxiliaryOutput]


class MoEFFDDetector(nn.Module):
    """
    Vision Transformer detector with sparse expert adaptations.

    The backbone stays frozen, while LoRA and adapter experts provide
    parameter-efficient adaptation for face forgery detection.
    """

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.config = config
        self.backbone = FrozenViTBackbone(config.backbone)
        self.blocks = nn.ModuleList(
            MoETransformerBlock(
                block_index=block_index,
                embed_dim=config.backbone.embed_dim,
                config=config,
            )
            for block_index in range(12)
        )
        self.classifier = nn.Linear(config.backbone.embed_dim, config.classifier.num_classes)

    def forward(self, images: Tensor) -> tuple[Tensor, ModelAuxiliaryOutput]:
        raise NotImplementedError("MoEFFDDetector.forward is implemented in Step 3.")

