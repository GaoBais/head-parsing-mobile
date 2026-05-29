"""MobileNetV2 encoder for mobile semantic segmentation."""

from __future__ import annotations

import torch
from torch import nn


def _make_divisible(value: float, divisor: int = 8, min_value: int | None = None) -> int:
    if min_value is None:
        min_value = divisor
    new_value = max(min_value, int(value + divisor / 2) // divisor * divisor)
    if new_value < 0.9 * value:
        new_value += divisor
    return int(new_value)


class ConvBNActivation(nn.Sequential):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        groups: int = 1,
        activation: bool = True,
    ) -> None:
        padding = (kernel_size - 1) // 2
        layers: list[nn.Module] = [
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size,
                stride,
                padding,
                groups=groups,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
        ]
        if activation:
            layers.append(nn.ReLU6(inplace=True))
        super().__init__(*layers)


class InvertedResidual(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride: int, expand_ratio: int) -> None:
        super().__init__()
        if stride not in (1, 2):
            raise ValueError(f"Expected stride 1 or 2, got {stride}")

        hidden_dim = int(round(in_channels * expand_ratio))
        self.use_res_connect = stride == 1 and in_channels == out_channels

        layers: list[nn.Module] = []
        if expand_ratio != 1:
            layers.append(ConvBNActivation(in_channels, hidden_dim, kernel_size=1))
        layers.extend(
            [
                ConvBNActivation(hidden_dim, hidden_dim, stride=stride, groups=hidden_dim),
                nn.Conv2d(hidden_dim, out_channels, kernel_size=1, stride=1, padding=0, bias=False),
                nn.BatchNorm2d(out_channels),
            ]
        )
        self.conv = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.use_res_connect:
            return x + self.conv(x)
        return self.conv(x)


def _make_stage(
    in_channels: int,
    out_channels: int,
    repeats: int,
    stride: int,
    expand_ratio: int,
) -> tuple[nn.Sequential, int]:
    layers: list[nn.Module] = []
    for block_index in range(repeats):
        block_stride = stride if block_index == 0 else 1
        layers.append(InvertedResidual(in_channels, out_channels, block_stride, expand_ratio))
        in_channels = out_channels
    return nn.Sequential(*layers), in_channels


class MobileNetV2Encoder(nn.Module):
    """MobileNetV2 feature extractor.

    Returns a low-level feature at stride 4 and a high-level feature at stride
    16 by default. The stride-32 stage is disabled for the recommended mobile
    segmentation setting.
    """

    def __init__(self, width_mult: float = 1.0, output_stride: int = 16) -> None:
        super().__init__()
        if output_stride not in (16, 32):
            raise ValueError("MobileNetV2Encoder supports output_stride 16 or 32.")

        input_channel = _make_divisible(32 * width_mult, 8)
        last_channel = _make_divisible(1280 * max(1.0, width_mult), 8)

        self.stem = ConvBNActivation(3, input_channel, stride=2)

        def c(channels: int) -> int:
            return _make_divisible(channels * width_mult, 8)

        self.stage1, input_channel = _make_stage(input_channel, c(16), repeats=1, stride=1, expand_ratio=1)
        self.stage2, input_channel = _make_stage(input_channel, c(24), repeats=2, stride=2, expand_ratio=6)
        self.stage3, input_channel = _make_stage(input_channel, c(32), repeats=3, stride=2, expand_ratio=6)
        self.stage4, input_channel = _make_stage(input_channel, c(64), repeats=4, stride=2, expand_ratio=6)
        self.stage5, input_channel = _make_stage(input_channel, c(96), repeats=3, stride=1, expand_ratio=6)

        stage6_stride = 2 if output_stride == 32 else 1
        self.stage6, input_channel = _make_stage(input_channel, c(160), repeats=3, stride=stage6_stride, expand_ratio=6)
        self.stage7, input_channel = _make_stage(input_channel, c(320), repeats=1, stride=1, expand_ratio=6)
        self.final_conv = ConvBNActivation(input_channel, last_channel, kernel_size=1)

        self.low_level_channels = c(24)
        self.out_channels = last_channel
        self.output_stride = output_stride

        self._init_weights()

    def _init_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode="fan_out")
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x = self.stem(x)
        x = self.stage1(x)
        low = self.stage2(x)
        x = self.stage3(low)
        x = self.stage4(x)
        x = self.stage5(x)
        x = self.stage6(x)
        x = self.stage7(x)
        high = self.final_conv(x)
        return low, high
