"""Adapter expert definitions used by the MoE adapter branch."""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from utils.config import AdapterExpertConfig


def create_conv_func(op_type: str):
    if op_type not in {"cv", "cd", "ad", "rd", "scd"}:
        raise ValueError(f"unknown op type: {op_type}")

    if op_type == "cv":
        return F.conv2d

    if op_type == "cd":
        def func(x, weights, bias=None, stride=1, padding=0, dilation=1, groups=1):
            weights_c = weights.sum(dim=[2, 3]) - weights[:, :, 1, 1]
            weights_c = weights_c[:, :, None, None]
            yc = F.conv2d(x, weights_c, stride=stride, padding=0, groups=groups)
            y = F.conv2d(x, weights, bias, stride=stride, padding=padding, dilation=dilation, groups=groups)
            return y - yc

        return func

    if op_type == "ad":
        def func(x, weights, bias=None, stride=1, padding=0, dilation=1, groups=1):
            shape = weights.shape
            flat = weights.view(shape[0], shape[1], -1)
            rotated = flat[:, :, [3, 0, 1, 6, 4, 2, 7, 8, 5]].clone()
            rotated[:, :, 4] = 0.0
            weights_conv = (flat - rotated).view(shape)
            return F.conv2d(x, weights_conv, bias, stride=stride, padding=padding, dilation=dilation, groups=groups)

        return func

    if op_type == "rd":
        def func(x, weights, bias=None, stride=1, padding=0, dilation=1, groups=1):
            shape = weights.shape
            buffer = weights.new_zeros(shape[0], shape[1], 25)
            flat = weights.view(shape[0], shape[1], -1)
            buffer[:, :, [0, 2, 4, 10, 14, 20, 22, 24]] = flat[:, :, 1:]
            buffer[:, :, [6, 7, 8, 11, 13, 16, 17, 18]] = -flat[:, :, 1:]
            buffer[:, :, 12] = flat[:, :, 0]
            buffer = buffer.view(shape[0], shape[1], 5, 5)
            return F.conv2d(x, buffer, bias, stride=stride, padding=2 * dilation, dilation=dilation, groups=groups)

        return func

    def func(x, weights, bias=None, stride=1, padding=0, dilation=1, groups=1):
        shape = weights.shape
        buffer = weights.new_zeros(shape[0], shape[1], 25)
        flat = weights.view(shape[0], shape[1], -1)
        buffer[:, :, [0, 2, 4, 10, 14, 20, 22, 24]] = flat[:, :, 1:]
        buffer[:, :, [6, 7, 8, 11, 13, 16, 17, 18]] = -flat[:, :, 1:] * 2.0
        buffer[:, :, 12] = flat.sum(dim=2)
        buffer = buffer.view(shape[0], shape[1], 5, 5)
        return F.conv2d(x, buffer, bias, stride=stride, padding=2 * dilation, dilation=dilation, groups=groups)

    return func


class Conv2dDiff(nn.Module):
    """Difference convolution used in the original adapter experts."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: int = 0,
        dilation: int = 1,
        groups: int = 1,
        bias: bool = False,
        op_type: str = "cv",
    ) -> None:
        super().__init__()
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.weight = nn.Parameter(
            torch.empty(out_channels, in_channels // groups, kernel_size, kernel_size)
        )
        if bias:
            self.bias = nn.Parameter(torch.empty(out_channels))
        else:
            self.register_parameter("bias", None)
        self.func = create_conv_func(op_type)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1.0 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x: Tensor) -> Tensor:
        return self.func(x, self.weight, self.bias, self.stride, self.padding, self.dilation, self.groups)


class BaseAdapterExpert(nn.Module):
    """Base class shared by the local feature adapter experts."""

    expert_name = "base"
    op_type = "cv"

    def __init__(self, input_dim: int, config: AdapterExpertConfig) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.config = config
        self.adapter_dim = config.bottleneck_dim
        self.adapter_down = nn.Linear(input_dim, self.adapter_dim)
        self.adapter_up = nn.Linear(self.adapter_dim, input_dim)
        nn.init.xavier_uniform_(self.adapter_down.weight)
        nn.init.zeros_(self.adapter_down.bias)
        nn.init.zeros_(self.adapter_up.weight)
        nn.init.zeros_(self.adapter_up.bias)

        self.adapter_conv = Conv2dDiff(
            self.adapter_dim,
            self.adapter_dim,
            kernel_size=config.kernel_size,
            stride=1,
            padding=config.kernel_size // 2,
            bias=True,
            op_type=self.op_type,
        )
        nn.init.zeros_(self.adapter_conv.weight)
        eye = torch.eye(self.adapter_dim, dtype=self.adapter_conv.weight.dtype)
        center = config.kernel_size // 2
        self.adapter_conv.weight.data[:, :, center, center] += eye
        if self.adapter_conv.bias is not None:
            nn.init.zeros_(self.adapter_conv.bias)

    @staticmethod
    def _split_cls_token(tokens: Tensor, spatial_shape: tuple[int, int]) -> tuple[Tensor, Tensor]:
        expected_patches = spatial_shape[0] * spatial_shape[1]
        if tokens.size(1) != expected_patches + 1:
            raise ValueError(f"Token count {tokens.size(1)} does not match spatial shape {spatial_shape}.")
        return tokens[:, :1], tokens[:, 1:]

    def forward(self, tokens: Tensor, spatial_shape: tuple[int, int]) -> Tensor:
        batch_size, _, _ = tokens.shape
        height, width = spatial_shape
        cls_token, patch_tokens = self._split_cls_token(tokens, spatial_shape)

        down_tokens = self.adapter_down(tokens)
        patch_map = down_tokens[:, 1:].reshape(batch_size, height, width, self.adapter_dim).permute(0, 3, 1, 2)
        patch_map = self.adapter_conv(patch_map)
        patch_tokens = patch_map.permute(0, 2, 3, 1).reshape(batch_size, height * width, self.adapter_dim)

        cls_map = down_tokens[:, :1].reshape(batch_size, 1, 1, self.adapter_dim).permute(0, 3, 1, 2)
        cls_map = self.adapter_conv(cls_map)
        cls_token = cls_map.permute(0, 2, 3, 1).reshape(batch_size, 1, self.adapter_dim)

        adapted = torch.cat([cls_token, patch_tokens], dim=1)
        return self.adapter_up(adapted)


class VanillaConvExpert(BaseAdapterExpert):
    expert_name = "vanilla_conv"
    op_type = "cv"


class ADCExpert(BaseAdapterExpert):
    expert_name = "adc"
    op_type = "ad"


class CDCExpert(BaseAdapterExpert):
    expert_name = "cdc"
    op_type = "cd"


class RDCExpert(BaseAdapterExpert):
    expert_name = "rdc"
    op_type = "rd"


class SOCExpert(BaseAdapterExpert):
    expert_name = "soc"
    op_type = "scd"


ADAPTER_EXPERT_REGISTRY = {
    "cv": VanillaConvExpert,
    VanillaConvExpert.expert_name: VanillaConvExpert,
    "ad": ADCExpert,
    ADCExpert.expert_name: ADCExpert,
    "cd": CDCExpert,
    CDCExpert.expert_name: CDCExpert,
    "rd": RDCExpert,
    RDCExpert.expert_name: RDCExpert,
    "scd": SOCExpert,
    SOCExpert.expert_name: SOCExpert,
}
