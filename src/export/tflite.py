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
from typing import Any

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


@dataclass
class TFLiteV3ExportConfig:
    """v3 I/O contract: uint8 NHWC input, in-graph normalization, optional in-graph argmax.

    ``output_kind``:
    - ``label_map``: in-graph argmax, uint8 label map ``[1, H, W]`` (primary artifact).
    - ``logits``: raw fp32 logits in channel-last ``[1, H, W, C]`` (Android A/B
      fallback; contiguous per-pixel classes make the on-device argmax sequential).
    """

    checkpoint: Path
    output: Path
    input_size: tuple[int, int] = (256, 256)
    num_classes: int = 9
    width_mult: float = 1.0
    output_stride: int = 16
    decoder_channels: int = 128
    dropout: float = 0.0
    device: str = "cpu"
    precision: str = "fp16"
    converter: str = "auto"
    output_kind: str = "label_map"


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


def _patch_litert_uint8_dtype_support(converter: Any) -> None:
    """Patch LiteRT Torch 0.9.x dtype metadata mapping for uint8 inputs.

    The converter can lower uint8 tensors through StableHLO, but 0.9.1 omits
    ``torch.uint8`` from the PyTorch-dtype-to-MLIR-type helper used while building
    flat input metadata. Keep the workaround local to the export process instead
    of asking users to edit site-packages on the server.
    """

    module_name = getattr(converter, "__name__", "") or converter.__class__.__module__
    root = module_name.split(".", 1)[0]
    if root not in {"litert_torch", "ai_edge_torch"}:
        return

    try:
        from litert_converter.mlir import ir
    except Exception:
        return

    module_names = [
        f"{root}.backend.lowerings.utils",
        f"{root}.backend.export_utils",
    ]
    patched = None
    for name in module_names:
        try:
            module = importlib.import_module(name)
        except Exception:
            continue

        original = getattr(module, "torch_dtype_to_ir_element_type", None)
        if original is None:
            continue
        if patched is None:

            def patched(dtype, _original=original):
                if dtype is torch.uint8:
                    return ir.IntegerType.get_unsigned(8)
                return _original(dtype)

        module.torch_dtype_to_ir_element_type = patched


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


def export_tflite_v3(config: TFLiteV3ExportConfig, converter_module: ModuleType | None = None) -> Path:
    """Export the v3 uint8/NHWC artifact (see TFLiteV3ExportConfig for the contract).

    Like the v2 path, the traced PyTorch graph stays FP32; fp16 weights come from
    LiteRT lowering / follow-up quantization outside this function.
    """

    if config.output_kind not in {"label_map", "logits"}:
        raise ValueError(f"Unsupported v3 output_kind: {config.output_kind}")

    converter = converter_module or import_litert_torch_converter(config.converter)
    if not hasattr(converter, "to_channel_last_io"):
        raise RuntimeError(
            "The TFLite converter does not expose to_channel_last_io(); "
            "upgrade litert_torch/ai_edge_torch for v3 NHWC export."
        )
    _patch_litert_uint8_dtype_support(converter)

    from src.export.mobile_io import build_tflite_v3_module

    device = torch.device(config.device)
    model = build_model_from_export_config(_to_onnx_like_config(config)).to(device)
    include_argmax = config.output_kind == "label_map"
    export_model = build_tflite_v3_module(model, include_argmax=include_argmax).to(device)
    export_model.eval()

    # The label map [1, H, W] has no channel axis, so only the input is converted;
    # the logits variant also converts the [1, C, H, W] output to channel-last.
    if include_argmax:
        channel_last_model = converter.to_channel_last_io(export_model, args=[0])
    else:
        channel_last_model = converter.to_channel_last_io(export_model, args=[0], outputs=[0])

    sample_inputs = (
        torch.randint(
            0,
            256,
            (1, config.input_size[0], config.input_size[1], 3),
            dtype=torch.uint8,
            device=device,
        ),
    )
    edge_model = converter.convert(channel_last_model, sample_inputs)

    config.output.parent.mkdir(parents=True, exist_ok=True)
    edge_model.export(str(config.output))
    return config.output


def _to_onnx_like_config(config: TFLiteExportConfig | TFLiteV3ExportConfig):
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
        include_softmax=getattr(config, "include_softmax", False),
    )
