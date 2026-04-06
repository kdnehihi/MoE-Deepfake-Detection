"""Transformer block wrapper that hosts MoE-LoRA and MoE-Adapter modules."""

from __future__ import annotations

from dataclasses import dataclass

from torch import Tensor, nn

from models.moe_adapter import AdapterAuxiliaryOutput, MoEAdapterLayer
from models.moe_lora import LoRAAuxiliaryOutput, MoELoRALayer
from utils.config import ModelConfig


@dataclass(slots=True)
class BlockAuxiliaryOutput:
    lora: LoRAAuxiliaryOutput
    adapter: AdapterAuxiliaryOutput


class MoETransformerBlock(nn.Module):
    """
    Wrapper around a ViT block.

    The eventual implementation attaches LoRA experts to the attention
    projections and attaches adapter experts after the MLP branch while keeping
    the frozen backbone weights intact.
    """

    def __init__(self, block_index: int, embed_dim: int, config: ModelConfig) -> None:
        super().__init__()
        self.block_index = block_index
        self.embed_dim = embed_dim
        self.config = config
        self.attn_lora_q = MoELoRALayer(
            input_dim=embed_dim,
            output_dim=embed_dim,
            experts=config.moe.lora_experts,
            gating_config=config.gating,
        )
        self.attn_lora_k = MoELoRALayer(
            input_dim=embed_dim,
            output_dim=embed_dim,
            experts=config.moe.lora_experts,
            gating_config=config.gating,
        )
        self.attn_lora_v = MoELoRALayer(
            input_dim=embed_dim,
            output_dim=embed_dim,
            experts=config.moe.lora_experts,
            gating_config=config.gating,
        )
        self.adapter = MoEAdapterLayer(
            input_dim=embed_dim,
            experts=config.moe.adapter_experts,
            gating_config=config.gating,
        )

    def forward(self, tokens: Tensor) -> tuple[Tensor, BlockAuxiliaryOutput]:
        raise NotImplementedError("MoETransformerBlock.forward is implemented in Step 3.")

