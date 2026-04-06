"""Adapter expert definitions used by the MoE adapter branch."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from utils.config import AdapterExpertConfig


class BaseAdapterExpert(nn.Module):
    """Base class shared by the local feature adapter experts."""

    expert_name = "base"

    def __init__(self, input_dim: int, config: AdapterExpertConfig) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.config = config
        self.down_proj = nn.Linear(input_dim, config.bottleneck_dim)
        self.activation = nn.GELU()
        self.up_proj = nn.Linear(config.bottleneck_dim, input_dim)
        self.pre_norm = nn.LayerNorm(input_dim)
        self.post_norm = nn.LayerNorm(input_dim)

    @staticmethod
    def _split_cls_token(tokens: Tensor, spatial_shape: tuple[int, int]) -> tuple[Tensor | None, Tensor]:
        expected_patches = spatial_shape[0] * spatial_shape[1]
        if tokens.size(1) == expected_patches + 1:
            return tokens[:, :1], tokens[:, 1:]
        if tokens.size(1) == expected_patches:
            return None, tokens
        raise ValueError(
            f"Token count {tokens.size(1)} does not match spatial shape {spatial_shape}."
        )

    def _tokens_to_feature_map(self, patch_tokens: Tensor, spatial_shape: tuple[int, int]) -> Tensor:
        batch_size, _, channels = patch_tokens.shape
        height, width = spatial_shape
        features = patch_tokens.view(batch_size, height, width, channels).permute(0, 3, 1, 2).contiguous()
        return features

    def _feature_map_to_tokens(self, features: Tensor) -> Tensor:
        batch_size, channels, height, width = features.shape
        return features.permute(0, 2, 3, 1).contiguous().view(batch_size, height * width, channels)

    def _apply_local_operator(self, features: Tensor) -> Tensor:
        raise NotImplementedError(f"{self.__class__.__name__} must implement _apply_local_operator.")

    def forward(self, tokens: Tensor, spatial_shape: tuple[int, int]) -> Tensor:
        cls_token, patch_tokens = self._split_cls_token(tokens, spatial_shape)
        patch_tokens = self.pre_norm(patch_tokens)
        patch_tokens = self.activation(self.down_proj(patch_tokens))
        features = self._tokens_to_feature_map(patch_tokens, spatial_shape)
        features = self._apply_local_operator(features)
        patch_tokens = self._feature_map_to_tokens(features)
        patch_tokens = self.up_proj(self.activation(patch_tokens))
        patch_tokens = self.post_norm(patch_tokens)
        if cls_token is None:
            return patch_tokens
        cls_delta = torch.zeros_like(cls_token)
        return torch.cat([cls_delta, patch_tokens], dim=1)


class DepthwiseConvAdapterExpert(BaseAdapterExpert):
    """Shared convolutional bottleneck used by local adapter experts."""

    def __init__(self, input_dim: int, config: AdapterExpertConfig) -> None:
        super().__init__(input_dim=input_dim, config=config)
        padding = config.kernel_size // 2
        self.depthwise = nn.Conv2d(
            config.bottleneck_dim,
            config.bottleneck_dim,
            kernel_size=config.kernel_size,
            padding=padding,
            groups=config.bottleneck_dim,
            bias=False,
        )
        self.pointwise = nn.Conv2d(config.bottleneck_dim, config.bottleneck_dim, kernel_size=1, bias=False)
        self.bn = nn.BatchNorm2d(config.bottleneck_dim)

    def _post_process(self, features: Tensor) -> Tensor:
        features = self.pointwise(features)
        features = self.bn(features)
        return self.activation(features)

    def _apply_local_operator(self, features: Tensor) -> Tensor:
        raise NotImplementedError

    @staticmethod
    def _shift(features: Tensor, shift_y: int, shift_x: int) -> Tensor:
        pad_top = max(shift_y, 0)
        pad_bottom = max(-shift_y, 0)
        pad_left = max(shift_x, 0)
        pad_right = max(-shift_x, 0)
        padded = F.pad(features, (pad_left, pad_right, pad_top, pad_bottom))
        height, width = features.shape[-2:]
        start_y = pad_bottom
        start_x = pad_right
        end_y = start_y + height
        end_x = start_x + width
        return padded[:, :, start_y:end_y, start_x:end_x]


class VanillaConvExpert(DepthwiseConvAdapterExpert):
    expert_name = "vanilla_conv"

    def _apply_local_operator(self, features: Tensor) -> Tensor:
        features = self.depthwise(features)
        return self._post_process(features)


class ADCExpert(DepthwiseConvAdapterExpert):
    expert_name = "adc"

    def _apply_local_operator(self, features: Tensor) -> Tensor:
        base = self.depthwise(features)
        horizontal = self._shift(base, 0, 1) - self._shift(base, 0, -1)
        vertical = self._shift(base, 1, 0) - self._shift(base, -1, 0)
        angular = 0.5 * (horizontal + vertical)
        return self._post_process(base + angular)


class CDCExpert(DepthwiseConvAdapterExpert):
    expert_name = "cdc"

    def _apply_local_operator(self, features: Tensor) -> Tensor:
        base = self.depthwise(features)
        kernel_sum = self.depthwise.weight.sum(dim=(2, 3), keepdim=True).view(1, -1, 1, 1)
        center_response = features * kernel_sum
        return self._post_process(base - center_response)


class RDCExpert(DepthwiseConvAdapterExpert):
    expert_name = "rdc"

    def _apply_local_operator(self, features: Tensor) -> Tensor:
        base = self.depthwise(features)
        radial_context = F.avg_pool2d(features, kernel_size=3, stride=1, padding=1)
        ring_response = base - radial_context
        return self._post_process(base + ring_response)


class SOCExpert(DepthwiseConvAdapterExpert):
    expert_name = "soc"

    def _apply_local_operator(self, features: Tensor) -> Tensor:
        base = self.depthwise(features)
        left = self._shift(base, 0, -1)
        right = self._shift(base, 0, 1)
        up = self._shift(base, -1, 0)
        down = self._shift(base, 1, 0)
        second_order = (left + right + up + down) - (4.0 * base)
        return self._post_process(base + second_order)


ADAPTER_EXPERT_REGISTRY = {
    VanillaConvExpert.expert_name: VanillaConvExpert,
    ADCExpert.expert_name: ADCExpert,
    CDCExpert.expert_name: CDCExpert,
    RDCExpert.expert_name: RDCExpert,
    SOCExpert.expert_name: SOCExpert,
}
