"""LiteRT/TFLite export helpers using LiteRT Torch.

The current preferred conversion path is the Google AI Edge LiteRT Torch
converter (`litert_torch.convert`). Older environments may still expose the
same converter as `ai_edge_torch.convert`, so this module supports both names.
"""

from __future__ import annotations

import importlib
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType

import torch

from src.export.onnx import ExportWrapper, build_model_from_export_config


@dataclass
class TFLiteExportConfig:
    checkpoint: Path
    output: Path
    input_size: tuple[int, int] = (320, 320)
    num_classes: int = 20
    width_mult: float = 1.0
    output_stride: int = 16
    decoder_channels: int = 128
    dropout: float = 0.0
    device: str = "cpu"
    precision: str = "fp16"
    include_softmax: bool = False
    converter: str = "auto"


def import_litert_torch_converter(preferred: str = "auto") -> ModuleType:
    """Import the PyTorch to LiteRT converter package."""

    normalized = preferred.lower().replace("-", "_")
    if normalized not in {"auto", "litert_torch", "ai_edge_torch"}:
        raise ValueError(f"Unsupported TFLite converter: {preferred}")

    candidates = ["litert_torch", "ai_edge_torch"] if normalized == "auto" else [normalized]
    errors: list[tuple[str, Exception]] = []

    for module_name in candidates:
        try:
            module = importlib.import_module(module_name)
        except Exception as exc:  # Import can fail on torch/converter compatibility before module load completes.
            errors.append((module_name, exc))
            continue

        if not hasattr(module, "convert"):
            errors.append((module_name, RuntimeError(f"{module_name} does not expose convert().")))
            continue

        return module

    details = "\n".join(f"- {name}: {type(exc).__name__}: {exc}" for name, exc in errors)
    raise RuntimeError(
        "Could not import a usable PyTorch to LiteRT/TFLite converter.\n"
        f"Tried: {', '.join(candidates)}\n"
        f"{details}\n"
        "Use `--converter ai_edge_torch` if the legacy converter is installed, or use a separate CPU export "
        "environment with converter-compatible PyTorch."
    ) from (errors[-1][1] if errors else None)


def export_tflite(config: TFLiteExportConfig, converter_module: ModuleType | None = None) -> Path:
    converter = converter_module or import_litert_torch_converter(config.converter)
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
