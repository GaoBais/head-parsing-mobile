"""Lightweight LR-ASPP style decoder."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from src.models.mobilenetv2 import ConvBNActivation


class SeparableConvBNReLU(nn.Sequential):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__(
            ConvBNActivation(in_channels, in_channels, kernel_size=3, groups=in_channels),
            ConvBNActivation(in_channels, out_channels, kernel_size=1),
        )


class LRASPPHead(nn.Module):
    """LR-ASPP-lite decoder with one low-level skip connection."""

    def __init__(
        self,
        low_channels: int,
        high_channels: int,
        num_classes: int,
        decoder_channels: int = 128,
        low_projection_channels: int = 48,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.high_project = ConvBNActivation(high_channels, decoder_channels, kernel_size=1)
        self.scale = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(high_channels, decoder_channels, kernel_size=1, bias=True),
            nn.Sigmoid(),
        )
        self.low_project = ConvBNActivation(low_channels, low_projection_channels, kernel_size=1)
        fused_channels = decoder_channels + low_projection_channels
        layers: list[nn.Module] = [SeparableConvBNReLU(fused_channels, decoder_channels)]
        if dropout > 0:
            layers.append(nn.Dropout2d(p=dropout))
        layers.append(nn.Conv2d(decoder_channels, num_classes, kernel_size=1))
        self.classifier = nn.Sequential(*layers)

    def forward(
        self,
        low: torch.Tensor,
        high: torch.Tensor,
        output_size: tuple[int, int],
    ) -> torch.Tensor:
        high_features = self.high_project(high)
        high_features = high_features * self.scale(high)
        high_features = F.interpolate(
            high_features,
            size=low.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )

        low_features = self.low_project(low)
        logits = self.classifier(torch.cat([low_features, high_features], dim=1))
        logits = F.interpolate(logits, size=output_size, mode="bilinear", align_corners=False)
        return logits
