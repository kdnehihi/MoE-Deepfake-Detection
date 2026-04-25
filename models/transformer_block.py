"""Transformer block wrapper that hosts MoE-LoRA and MoE-Adapter modules."""

from __future__ import annotations

from dataclasses import dataclass
from copy import deepcopy

import math

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


class PaperAlignedAttention(nn.Module):
    """
    Attention module matching the paper's forward logic while reusing the
    pretrained qkv/proj weights from the frozen timm block.
    """

    def __init__(self, dim: int, config: ModelConfig, frozen_attention: nn.Module) -> None:
        super().__init__()
        self.dim = dim
        self.num_heads = frozen_attention.num_heads
        head_dim = dim // self.num_heads
        self.scale = head_dim ** -0.5

        qkv_bias = frozen_attention.qkv.bias is not None
        proj_bias = frozen_attention.proj.bias is not None

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim, bias=proj_bias)
        self.qkv.load_state_dict(frozen_attention.qkv.state_dict())
        self.proj.load_state_dict(frozen_attention.proj.state_dict())

        self.attn_drop = deepcopy(getattr(frozen_attention, "attn_drop", nn.Identity()))
        self.proj_drop = deepcopy(getattr(frozen_attention, "proj_drop", nn.Identity()))

        self.lora_moe = MoELoRALayer(
            input_dim=dim,
            output_dim=dim,
            experts=config.moe.lora_experts,
            gating_config=config.gating,
        )

        for parameter in self.qkv.parameters():
            parameter.requires_grad = False
        for parameter in self.proj.parameters():
            parameter.requires_grad = False

    def forward(
        self,
        tokens: Tensor,
        *,
        enable_lora: bool,
        router_enabled: bool,
    ) -> tuple[Tensor, AttentionLoRAAuxiliaryOutput]:
        batch_size, num_tokens, hidden_dim = tokens.shape
        qkv = self.qkv(tokens).reshape(batch_size, num_tokens, 3, self.num_heads, hidden_dim // self.num_heads)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        query, key, value = qkv.unbind(dim=0)

        if enable_lora:
            qkv_delta, lora_aux = self.lora_moe(tokens, router_enabled=router_enabled)
            qkv_delta = qkv_delta.reshape(
                batch_size,
                num_tokens,
                3,
                self.num_heads,
                hidden_dim // self.num_heads,
            ).permute(2, 0, 3, 1, 4)
            q_delta, k_delta, v_delta = qkv_delta.unbind(dim=0)
            query = query + q_delta
            key = key + k_delta
            value = value + v_delta
        else:
            lora_aux = self.lora_moe._empty_aux(tokens)

        attention = (query @ key.transpose(-2, -1)) * self.scale
        attention = attention.softmax(dim=-1)
        attention = self.attn_drop(attention)

        outputs = (attention @ value).transpose(1, 2).reshape(batch_size, num_tokens, hidden_dim)
        outputs = self.proj(outputs)
        outputs = self.proj_drop(outputs)
        return outputs, AttentionLoRAAuxiliaryOutput(qkv=lora_aux)


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

        self.norm1 = deepcopy(frozen_block.norm1)
        self.norm2 = deepcopy(frozen_block.norm2)
        self.attn = PaperAlignedAttention(embed_dim, config, frozen_block.attn)
        self.mlp = deepcopy(frozen_block.mlp)
        self.adapter = MoEAdapterLayer(
            input_dim=embed_dim,
            experts=config.moe.adapter_experts,
            gating_config=config.gating,
        )
        self.ls1 = deepcopy(getattr(frozen_block, "ls1", nn.Identity()))
        self.ls2 = deepcopy(getattr(frozen_block, "ls2", nn.Identity()))
        self.drop_path1 = deepcopy(getattr(frozen_block, "drop_path1", nn.Identity()))
        self.drop_path2 = deepcopy(getattr(frozen_block, "drop_path2", nn.Identity()))
        self.enable_lora = config.stage.enable_lora
        self.enable_adapter = config.stage.enable_adapter
        self.enable_moe_router = config.stage.enable_moe_router
        self._configure_trainability()

    @staticmethod
    def _set_requires_grad(module: nn.Module, enabled: bool) -> None:
        for parameter in module.parameters():
            parameter.requires_grad = enabled

    def _configure_trainability(self) -> None:
        self._set_requires_grad(self.attn.lora_moe, self.enable_lora)
        self._set_requires_grad(self.adapter, self.enable_adapter)
        self._set_requires_grad(self.norm1, True)
        self._set_requires_grad(self.norm2, False)
        self._set_requires_grad(self.mlp, False)
        if self.enable_lora and not self.enable_moe_router:
            if hasattr(self.attn.lora_moe, "w_gate"):
                self.attn.lora_moe.w_gate.requires_grad = False
            if hasattr(self.attn.lora_moe, "w_noise"):
                self.attn.lora_moe.w_noise.requires_grad = False
        if self.enable_adapter and not self.enable_moe_router:
            if hasattr(self.adapter, "w_gate"):
                self.adapter.w_gate.requires_grad = False
            if hasattr(self.adapter, "w_noise"):
                self.adapter.w_noise.requires_grad = False

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

    def forward(self, tokens: Tensor) -> tuple[Tensor, BlockAuxiliaryOutput]:
        normed_tokens = self.norm1(tokens)
        attn_output, lora_aux = self.attn(
            normed_tokens,
            enable_lora=self.enable_lora,
            router_enabled=self.enable_moe_router,
        )
        attn_output = self.ls1(attn_output)
        tokens = tokens + self._drop_path(self.drop_path1, attn_output)

        mlp_input = self.norm2(tokens)
        mlp_output = self.mlp(mlp_input)
        mlp_output = self.ls2(mlp_output)
        spatial_shape = self._infer_spatial_shape(tokens.size(1))
        if self.enable_adapter:
            adapter_output, adapter_aux = self.adapter(
                mlp_input,
                spatial_shape,
                router_enabled=self.enable_moe_router,
            )
            tokens = tokens + self._drop_path(self.drop_path2, adapter_output)
        else:
            adapter_aux = self.adapter._empty_aux(mlp_input)
        tokens = tokens + self._drop_path(self.drop_path2, mlp_output)

        aux = BlockAuxiliaryOutput(lora=lora_aux, adapter=adapter_aux)
        return tokens, aux
