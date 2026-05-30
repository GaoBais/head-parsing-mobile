"""Mobile head parsing segmentation model."""

from __future__ import annotations

import torch
from torch import nn

from src.models.lraspp import LRASPPHead
from src.models.mobilenetv2 import MobileNetV2Encoder


class HeadParsingMobile(nn.Module):
    """MobileNetV2 + LR-ASPP-lite segmentation network."""

    def __init__(
        self,
        num_classes: int = 20,
        width_mult: float = 1.0,
        output_stride: int = 16,
        decoder_channels: int = 128,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.encoder = MobileNetV2Encoder(width_mult=width_mult, output_stride=output_stride)
        self.decoder = LRASPPHead(
            low_channels=self.encoder.low_level_channels,
            high_channels=self.encoder.out_channels,
            num_classes=num_classes,
            decoder_channels=decoder_channels,
            dropout=dropout,
        )
        self.num_classes = num_classes

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output_size = x.shape[-2:]
        low, high = self.encoder(x)
        return self.decoder(low, high, output_size=output_size)


def build_head_parsing_mobile(
    num_classes: int = 20,
    width_mult: float = 1.0,
    output_stride: int = 16,
    decoder_channels: int = 128,
    dropout: float = 0.0,
) -> HeadParsingMobile:
    return HeadParsingMobile(
        num_classes=num_classes,
        width_mult=width_mult,
        output_stride=output_stride,
        decoder_channels=decoder_channels,
        dropout=dropout,
    )
