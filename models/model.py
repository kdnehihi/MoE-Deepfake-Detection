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
                embed_dim=self.backbone.embed_dim,
                config=config,
                frozen_block=frozen_block,
            )
            for block_index, frozen_block in enumerate(self.backbone.model.blocks)
        )
        self.classifier_dropout = nn.Dropout(config.classifier.dropout)
        self.classifier = nn.Linear(self.backbone.embed_dim, config.classifier.num_classes)

    def forward(self, images: Tensor) -> tuple[Tensor, ModelAuxiliaryOutput]:
        tokens = self.backbone.embed_patches(images)
        auxiliary_outputs: list[BlockAuxiliaryOutput] = []

        for block in self.blocks:
            tokens, block_aux = block(tokens)
            auxiliary_outputs.append(block_aux)

        final_norm = getattr(self.backbone.model, "norm", None)
        if final_norm is not None:
            tokens = final_norm(tokens)

        cls_token = tokens[:, 0]
        logits = self.classifier(self.classifier_dropout(cls_token))
        aux = ModelAuxiliaryOutput(blocks=auxiliary_outputs)
        return logits, aux
