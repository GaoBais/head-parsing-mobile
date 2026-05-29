"""LiteRT/TFLite export helpers using LiteRT Torch.

The current preferred conversion path is the Google AI Edge LiteRT Torch
converter (`litert_torch.convert`). Older environments may still expose the
same converter as `ai_edge_torch.convert`, so this module supports both names.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch

from src.export.onnx import ExportWrapper, build_model_from_export_config


@dataclass
class TFLiteExportConfig:
    checkpoint: Path
    output: Path
    input_size: tuple[int, int] = (320, 320)
    num_classes: int = 19
    width_mult: float = 1.0
    output_stride: int = 16
    decoder_channels: int = 128
    dropout: float = 0.0
    device: str = "cpu"
    precision: str = "fp16"
    include_softmax: bool = False


def import_litert_torch_converter():
    """Import the PyTorch to LiteRT converter package."""

    try:
        import litert_torch

        return litert_torch
    except ImportError:
        pass

    try:
        import ai_edge_torch

        return ai_edge_torch
    except ImportError as exc:
        raise RuntimeError(
            "LiteRT Torch converter is required. Install `litert-torch` on the Linux server; "
            "older environments may use `ai-edge-torch`."
        ) from exc


def export_tflite(config: TFLiteExportConfig) -> Path:
    converter = import_litert_torch_converter()
    device = torch.device(config.device)
    model_config = _to_onnx_like_config(config)
    model = build_model_from_export_config(model_config).to(device)
    export_model = ExportWrapper(model, include_softmax=config.include_softmax).to(device)
    export_model.eval()

    precision = config.precision.lower()
    if precision not in {"fp16", "float16", "fp32", "float32"}:
        raise ValueError(f"Unsupported TFLite precision: {config.precision}")

    # Keep the traced PyTorch graph in FP32. LiteRT/TFLite lowering or follow-up
    # quantization should handle FP16 artifacts; CPU PyTorch half tracing is fragile.
    sample_inputs = (torch.randn(1, 3, config.input_size[0], config.input_size[1], device=device),)
    edge_model = converter.convert(export_model, sample_inputs)

    config.output.parent.mkdir(parents=True, exist_ok=True)
    edge_model.export(str(config.output))
    return config.output


def _to_onnx_like_config(config: TFLiteExportConfig):
    from src.export.onnx import OnnxExportConfig

    return OnnxExportConfig(
        checkpoint=config.checkpoint,
        output=config.output.with_suffix(".onnx"),
        input_size=config.input_size,
        num_classes=config.num_classes,
        width_mult=config.width_mult,
        output_stride=config.output_stride,
        decoder_channels=config.decoder_channels,
        dropout=config.dropout,
        device=config.device,
        include_softmax=config.include_softmax,
    )
