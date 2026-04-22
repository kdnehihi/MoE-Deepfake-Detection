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
        cls_delta = self.pre_norm(cls_token)
        cls_delta = self.activation(self.down_proj(cls_delta))
        cls_delta = self.up_proj(self.activation(cls_delta))
        cls_delta = self.post_norm(cls_delta)
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


class Conv3x3Expert(DepthwiseConvAdapterExpert):
    expert_name = "conv3x3"

    def _apply_local_operator(self, features: Tensor) -> Tensor:
        features = self.depthwise(features)
        return self._post_process(features)


class Conv5x5Expert(BaseAdapterExpert):
    expert_name = "conv5x5"

    def __init__(self, input_dim: int, config: AdapterExpertConfig) -> None:
        super().__init__(input_dim=input_dim, config=config)
        self.conv = nn.Conv2d(
            config.bottleneck_dim,
            config.bottleneck_dim,
            kernel_size=5,
            padding=2,
            groups=config.bottleneck_dim,
            bias=False,
        )
        self.pointwise = nn.Conv2d(config.bottleneck_dim, config.bottleneck_dim, kernel_size=1, bias=False)
        self.bn = nn.BatchNorm2d(config.bottleneck_dim)

    def _apply_local_operator(self, features: Tensor) -> Tensor:
        out = self.conv(features)
        out = self.pointwise(out)
        out = self.bn(out)
        return self.activation(out)


class DilatedConvExpert(BaseAdapterExpert):
    expert_name = "dilated_conv"

    def __init__(self, input_dim: int, config: AdapterExpertConfig) -> None:
        super().__init__(input_dim=input_dim, config=config)
        self.conv = nn.Conv2d(
            config.bottleneck_dim,
            config.bottleneck_dim,
            kernel_size=3,
            padding=2,
            dilation=2,
            groups=config.bottleneck_dim,
            bias=False,
        )
        self.pointwise = nn.Conv2d(config.bottleneck_dim, config.bottleneck_dim, kernel_size=1, bias=False)
        self.bn = nn.BatchNorm2d(config.bottleneck_dim)

    def _apply_local_operator(self, features: Tensor) -> Tensor:
        out = self.conv(features)
        out = self.pointwise(out)
        out = self.bn(out)
        return self.activation(out)


class DepthwiseOnlyExpert(DepthwiseConvAdapterExpert):
    expert_name = "depthwise_conv"

    def _apply_local_operator(self, features: Tensor) -> Tensor:
        return self._post_process(self.depthwise(features))


class HighPassExpert(BaseAdapterExpert):
    expert_name = "high_pass"

    def __init__(self, input_dim: int, config: AdapterExpertConfig) -> None:
        super().__init__(input_dim=input_dim, config=config)
        kernel = torch.tensor(
            [[-1.0, -1.0, -1.0], [-1.0, 8.0, -1.0], [-1.0, -1.0, -1.0]],
            dtype=torch.float32,
        )
        kernel = kernel.view(1, 1, 3, 3).repeat(config.bottleneck_dim, 1, 1, 1)
        self.register_buffer("kernel", kernel)
        self.pointwise = nn.Conv2d(config.bottleneck_dim, config.bottleneck_dim, kernel_size=1, bias=False)
        self.bn = nn.BatchNorm2d(config.bottleneck_dim)

    def _apply_local_operator(self, features: Tensor) -> Tensor:
        out = F.conv2d(features, self.kernel, padding=1, groups=features.size(1))
        out = self.pointwise(out)
        out = self.bn(out)
        return self.activation(out)


class LowPassExpert(BaseAdapterExpert):
    expert_name = "low_pass"

    def __init__(self, input_dim: int, config: AdapterExpertConfig) -> None:
        super().__init__(input_dim=input_dim, config=config)
        kernel = torch.ones((3, 3), dtype=torch.float32) / 9.0
        kernel = kernel.view(1, 1, 3, 3).repeat(config.bottleneck_dim, 1, 1, 1)
        self.register_buffer("kernel", kernel)
        self.pointwise = nn.Conv2d(config.bottleneck_dim, config.bottleneck_dim, kernel_size=1, bias=False)
        self.bn = nn.BatchNorm2d(config.bottleneck_dim)

    def _apply_local_operator(self, features: Tensor) -> Tensor:
        out = F.conv2d(features, self.kernel, padding=1, groups=features.size(1))
        out = self.pointwise(out)
        out = self.bn(out)
        return self.activation(out)


class FFTExpert(BaseAdapterExpert):
    expert_name = "fft"

    def __init__(self, input_dim: int, config: AdapterExpertConfig) -> None:
        super().__init__(input_dim=input_dim, config=config)
        self.pointwise = nn.Conv2d(config.bottleneck_dim, config.bottleneck_dim, kernel_size=1, bias=False)
        self.bn = nn.BatchNorm2d(config.bottleneck_dim)

    def _apply_local_operator(self, features: Tensor) -> Tensor:
        original_dtype = features.dtype
        if features.device.type == "mps":
            freq = torch.fft.fft2(features.float().cpu(), norm="ortho")
            mag = torch.abs(freq).to(features.device, dtype=original_dtype)
        else:
            freq = torch.fft.fft2(features.float(), norm="ortho")
            mag = torch.abs(freq).to(features.device, dtype=original_dtype)
        out = self.pointwise(mag)
        out = self.bn(out)
        return self.activation(out)


class CDCExpert(DepthwiseConvAdapterExpert):
    expert_name = "cdc"

    def _apply_local_operator(self, features: Tensor) -> Tensor:
        base = self.depthwise(features)
        kernel_sum = self.depthwise.weight.sum(dim=(2, 3), keepdim=True).view(1, -1, 1, 1)
        center_response = features * kernel_sum
        return self._post_process(base - center_response)


ADAPTER_EXPERT_REGISTRY = {
    Conv3x3Expert.expert_name: Conv3x3Expert,
    Conv5x5Expert.expert_name: Conv5x5Expert,
    DilatedConvExpert.expert_name: DilatedConvExpert,
    DepthwiseOnlyExpert.expert_name: DepthwiseOnlyExpert,
    HighPassExpert.expert_name: HighPassExpert,
    LowPassExpert.expert_name: LowPassExpert,
    FFTExpert.expert_name: FFTExpert,
    CDCExpert.expert_name: CDCExpert,
}
