"""Transformer block wrapper that hosts MoE-LoRA and MoE-Adapter modules."""

from __future__ import annotations

from dataclasses import dataclass

import math

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from models.moe_adapter import AdapterAuxiliaryOutput, MoEAdapterLayer
from models.moe_lora import LoRAAuxiliaryOutput, MoELoRALayer
from utils.config import ModelConfig


@dataclass(slots=True)
class AttentionLoRAAuxiliaryOutput:
    qkv: LoRAAuxiliaryOutput


@dataclass(slots=True)
class BlockAuxiliaryOutput:
    lora: AttentionLoRAAuxiliaryOutput
    adapter: AdapterAuxiliaryOutput


class MoETransformerBlock(nn.Module):
    """
    Wrapper around a ViT block.

    The eventual implementation attaches LoRA experts to the attention
    projections and attaches adapter experts after the MLP branch while keeping
    the frozen backbone weights intact.
    """

    def __init__(self, block_index: int, embed_dim: int, config: ModelConfig, frozen_block: nn.Module) -> None:
        super().__init__()
        self.block_index = block_index
        self.embed_dim = embed_dim
        self.config = config
        self.frozen_block = frozen_block
        for parameter in self.frozen_block.parameters():
            parameter.requires_grad = False
        self.attn_lora = MoELoRALayer(
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

    @staticmethod
    def _drop_path(module: nn.Module | None, tokens: Tensor) -> Tensor:
        if module is None:
            return tokens
        return module(tokens)

    @staticmethod
    def _infer_spatial_shape(num_tokens: int) -> tuple[int, int]:
        patch_tokens = num_tokens - 1
        if patch_tokens <= 0:
            raise ValueError("Transformer tokens must include at least one patch token.")
        side = int(math.sqrt(patch_tokens))
        if side * side != patch_tokens:
            raise ValueError(f"Token count {num_tokens} does not map to a square patch grid.")
        return side, side

    def _attention_with_moe(self, tokens: Tensor) -> tuple[Tensor, AttentionLoRAAuxiliaryOutput]:
        normed_tokens = self.frozen_block.norm1(tokens)
        attention = self.frozen_block.attn
        batch_size, num_tokens, _ = normed_tokens.shape
        num_heads = attention.num_heads
        head_dim = self.embed_dim // num_heads

        qkv = F.linear(normed_tokens, attention.qkv.weight, attention.qkv.bias)
        qkv = qkv.view(batch_size, num_tokens, 3, num_heads, head_dim).permute(2, 0, 3, 1, 4)
        query, key, value = qkv.unbind(dim=0)

        qkv_delta, lora_aux = self.attn_lora(normed_tokens)
        qkv_delta = qkv_delta.view(batch_size, num_tokens, 3, num_heads, head_dim).permute(2, 0, 3, 1, 4)
        q_delta, k_delta, v_delta = qkv_delta.unbind(dim=0)

        query = query + q_delta
        key = key + k_delta
        value = value + v_delta

        scores = (query @ key.transpose(-2, -1)) * attention.scale
        attention_probs = scores.softmax(dim=-1)
        attention_drop = getattr(attention, "attn_drop", None)
        if attention_drop is not None:
            attention_probs = attention_drop(attention_probs)

        context = attention_probs @ value
        context = context.transpose(1, 2).reshape(batch_size, num_tokens, self.embed_dim)
        context = attention.proj(context)
        proj_drop = getattr(attention, "proj_drop", None)
        if proj_drop is not None:
            context = proj_drop(context)

        lora_aux = AttentionLoRAAuxiliaryOutput(qkv=lora_aux)
        return context, lora_aux

    def forward(self, tokens: Tensor) -> tuple[Tensor, BlockAuxiliaryOutput]:
        attn_output, lora_aux = self._attention_with_moe(tokens)
        ls1 = getattr(self.frozen_block, "ls1", None)
        if ls1 is not None:
            attn_output = ls1(attn_output)
        drop_path1 = getattr(self.frozen_block, "drop_path1", None)
        tokens = tokens + self._drop_path(drop_path1, attn_output)

        mlp_input = self.frozen_block.norm2(tokens)
        mlp_output = self.frozen_block.mlp(mlp_input)
        spatial_shape = self._infer_spatial_shape(tokens.size(1))
        adapter_output, adapter_aux = self.adapter(mlp_input, spatial_shape)
        residual_output = mlp_output + adapter_output

        ls2 = getattr(self.frozen_block, "ls2", None)
        if ls2 is not None:
            residual_output = ls2(residual_output)
        drop_path2 = getattr(self.frozen_block, "drop_path2", None)
        tokens = tokens + self._drop_path(drop_path2, residual_output)

        aux = BlockAuxiliaryOutput(lora=lora_aux, adapter=adapter_aux)
        return tokens, aux
