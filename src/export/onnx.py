"""ONNX export helpers for HeadParsingMobile."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch import nn

from src.models import HeadParsingMobile


@dataclass
class OnnxExportConfig:
    checkpoint: Path
    output: Path
    input_size: tuple[int, int] = (320, 320)
    num_classes: int = 20
    width_mult: float = 1.0
    output_stride: int = 16
    decoder_channels: int = 128
    dropout: float = 0.0
    opset: int = 17
    device: str = "cpu"
    dynamic_batch: bool = False
    include_softmax: bool = False


class ExportWrapper(nn.Module):
    """Optional export-time wrapper for post-logit probabilities."""

    def __init__(self, model: nn.Module, include_softmax: bool = False) -> None:
        super().__init__()
        self.model = model
        self.include_softmax = include_softmax

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        logits = self.model(image)
        if self.include_softmax:
            return torch.softmax(logits, dim=1)
        return logits


def strip_module_prefix(state_dict: dict[str, Any]) -> dict[str, Any]:
    if not any(key.startswith("module.") for key in state_dict):
        return state_dict
    return {key.removeprefix("module."): value for key, value in state_dict.items()}


def resolve_state_dict(checkpoint: Any) -> dict[str, Any]:
    if isinstance(checkpoint, dict):
        for key in ("model", "state_dict", "model_state_dict"):
            value = checkpoint.get(key)
            if isinstance(value, dict):
                return strip_module_prefix(value)
        if checkpoint and all(hasattr(value, "shape") for value in checkpoint.values()):
            return strip_module_prefix(checkpoint)
    raise ValueError("Could not resolve model state_dict from checkpoint.")


def build_model_from_export_config(config: OnnxExportConfig) -> HeadParsingMobile:
    model = HeadParsingMobile(
        num_classes=config.num_classes,
        width_mult=config.width_mult,
        output_stride=config.output_stride,
        decoder_channels=config.decoder_channels,
        dropout=config.dropout,
    )
    checkpoint = torch.load(config.checkpoint, map_location="cpu")
    model.load_state_dict(resolve_state_dict(checkpoint))
    model.eval()
    return model


def export_onnx(config: OnnxExportConfig) -> Path:
    device = torch.device(config.device)
    model = build_model_from_export_config(config).to(device)
    export_model = ExportWrapper(model, include_softmax=config.include_softmax).to(device)
    export_model.eval()

    config.output.parent.mkdir(parents=True, exist_ok=True)
    dummy_input = torch.randn(1, 3, config.input_size[0], config.input_size[1], device=device)
    dynamic_axes = None
    if config.dynamic_batch:
        dynamic_axes = {"image": {0: "batch"}, "logits": {0: "batch"}}

    with torch.no_grad():
        torch.onnx.export(
            export_model,
            dummy_input,
            str(config.output),
            export_params=True,
            opset_version=config.opset,
            do_constant_folding=True,
            input_names=["image"],
            output_names=["logits"],
            dynamic_axes=dynamic_axes,
        )
    return config.output
