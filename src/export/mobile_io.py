"""Mobile v3 I/O graph adaptation: uint8 input, in-graph normalization, in-graph argmax.

v3 export contract (see docs/mobile_v3_export.md):

- Input is raw uint8 RGB (0..255). The ImageNet standard deviation is folded into
  the stem convolution weights (exact algebra, no accuracy cost beyond float
  rounding), so the graph only needs ``x * (1/255) - mean`` in front of the network.
- TFLite: the wrapper performs the cast + affine itself (``Uint8NormalizedInput``);
  channel-last I/O conversion happens later in the converter.
- Core ML: ``ct.ImageType(scale=1/255, bias=-mean)`` applies the affine in-framework,
  so the traced module expects the pre-scaled float input and has no input adapter.
- Output is either raw logits or an in-graph argmax label map (``ArgmaxHead``).
"""

from __future__ import annotations

import copy
from typing import Sequence

import torch
from torch import nn

from src.datasets.transforms import IMAGENET_MEAN, IMAGENET_STD


def fold_std_into_stem(model: nn.Module, std: Sequence[float] = IMAGENET_STD) -> nn.Module:
    """Return a copy of the model whose stem conv absorbs the per-channel 1/std factor.

    ``conv(W, y / s) == conv(W / s, y)`` for a per-input-channel scale ``s``, so the
    folded model accepts ``y = rgb/255 - mean`` directly. The input model is not
    mutated.
    """

    folded = copy.deepcopy(model)
    try:
        stem_conv = folded.encoder.stem[0]
    except (AttributeError, IndexError, TypeError) as exc:
        raise ValueError("Expected model.encoder.stem[0] to be the RGB input Conv2d.") from exc
    if not isinstance(stem_conv, nn.Conv2d) or stem_conv.in_channels != len(tuple(std)):
        raise ValueError("Expected model.encoder.stem[0] to be a Conv2d over the RGB input.")
    with torch.no_grad():
        divisor = torch.tensor(tuple(std), dtype=stem_conv.weight.dtype).view(1, -1, 1, 1)
        stem_conv.weight.div_(divisor)
    return folded


class Uint8NormalizedInput(nn.Module):
    """Cast uint8 RGB (0..255) to float and apply ``x * (1/255) - mean``.

    The std division is expected to be folded into the downstream model
    (``fold_std_into_stem``), keeping this adapter a single MUL + SUB.
    """

    def __init__(self, mean: Sequence[float] = IMAGENET_MEAN) -> None:
        super().__init__()
        self.register_buffer("mean", torch.tensor(tuple(mean), dtype=torch.float32).view(1, -1, 1, 1))
        self.scale = 1.0 / 255.0

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        return image.to(torch.float32) * self.scale - self.mean


class ArgmaxHead(nn.Module):
    """Reduce logits ``[N, C, H, W]`` to a label map ``[N, H, W]`` of ``dtype``."""

    def __init__(self, dtype: torch.dtype = torch.uint8) -> None:
        super().__init__()
        self.dtype = dtype

    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        return torch.argmax(logits, dim=1).to(self.dtype)


class MobileV3ExportWrapper(nn.Module):
    """Composable v3 export graph: [input adapter] -> model -> [argmax head]."""

    def __init__(
        self,
        model: nn.Module,
        input_adapter: nn.Module | None = None,
        argmax_head: nn.Module | None = None,
    ) -> None:
        super().__init__()
        self.input_adapter = input_adapter
        self.model = model
        self.argmax_head = argmax_head

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        x = image if self.input_adapter is None else self.input_adapter(image)
        logits = self.model(x)
        if self.argmax_head is None:
            return logits
        return self.argmax_head(logits)


def build_tflite_v3_module(
    model: nn.Module,
    include_argmax: bool,
    argmax_dtype: torch.dtype = torch.uint8,
    mean: Sequence[float] = IMAGENET_MEAN,
    std: Sequence[float] = IMAGENET_STD,
) -> MobileV3ExportWrapper:
    """TFLite variant: uint8 input adapter in-graph; label map dtype defaults to uint8."""

    return MobileV3ExportWrapper(
        fold_std_into_stem(model, std),
        input_adapter=Uint8NormalizedInput(mean),
        argmax_head=ArgmaxHead(argmax_dtype) if include_argmax else None,
    )


def build_coreml_v3_module(
    model: nn.Module,
    include_argmax: bool,
    argmax_dtype: torch.dtype = torch.int32,
    std: Sequence[float] = IMAGENET_STD,
) -> MobileV3ExportWrapper:
    """Core ML variant: scale/bias handled by ``ct.ImageType``; label map is int32
    (native ``reduce_argmax`` output dtype)."""

    return MobileV3ExportWrapper(
        fold_std_into_stem(model, std),
        input_adapter=None,
        argmax_head=ArgmaxHead(argmax_dtype) if include_argmax else None,
    )
