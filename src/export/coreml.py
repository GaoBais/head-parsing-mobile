"""Core ML export helpers for HeadParsingMobile."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch

from src.export.onnx import ExportWrapper, build_model_from_export_config


@dataclass
class CoreMLExportConfig:
    checkpoint: Path
    output: Path
    input_size: tuple[int, int] = (320, 320)
    num_classes: int = 20
    width_mult: float = 1.0
    output_stride: int = 16
    decoder_channels: int = 128
    dropout: float = 0.0
    precision: str = "fp16"
    minimum_deployment_target: str = "ios15"
    include_softmax: bool = False


def import_coremltools():
    try:
        import coremltools as ct
    except ImportError as exc:
        raise RuntimeError(
            "coremltools is required for Core ML export. Install it in the server or macOS export environment."
        ) from exc
    return ct


def coreml_target(ct, target: str):
    normalized = target.lower().replace("_", "").replace(".", "")
    mapping = {
        "ios15": ct.target.iOS15,
        "ios16": ct.target.iOS16,
        "ios17": ct.target.iOS17,
        "ios18": getattr(ct.target, "iOS18", ct.target.iOS17),
    }
    if normalized not in mapping:
        raise ValueError(f"Unsupported Core ML deployment target: {target}")
    return mapping[normalized]


def coreml_precision(ct, precision: str):
    normalized = precision.lower()
    if normalized in {"fp16", "float16"}:
        return ct.precision.FLOAT16
    if normalized in {"fp32", "float32"}:
        return ct.precision.FLOAT32
    raise ValueError(f"Unsupported Core ML precision: {precision}")


def export_coreml(config: CoreMLExportConfig) -> Path:
    ct = import_coremltools()
    model_config = _to_onnx_like_config(config)
    model = build_model_from_export_config(model_config)
    export_model = ExportWrapper(model, include_softmax=config.include_softmax)
    export_model.eval()

    dummy_input = torch.randn(1, 3, config.input_size[0], config.input_size[1])
    with torch.no_grad():
        traced = torch.jit.trace(export_model, dummy_input)

    mlmodel = ct.convert(
        traced,
        inputs=[ct.TensorType(name="image", shape=dummy_input.shape)],
        outputs=[ct.TensorType(name="logits")],
        convert_to="mlprogram",
        minimum_deployment_target=coreml_target(ct, config.minimum_deployment_target),
        compute_precision=coreml_precision(ct, config.precision),
    )

    config.output.parent.mkdir(parents=True, exist_ok=True)
    mlmodel.save(str(config.output))
    return config.output


def _to_onnx_like_config(config: CoreMLExportConfig):
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
        device="cpu",
        include_softmax=config.include_softmax,
    )
