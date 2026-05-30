"""Export HeadParsingMobile checkpoints to LiteRT/TFLite."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export HeadParsingMobile to LiteRT/TFLite.")
    parser.add_argument("--model-config", type=Path, default=Path("configs/model_mobilev2_lraspp_320.yaml"))
    parser.add_argument("--export-config", type=Path, default=Path("configs/export_mobile.yaml"))
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--precision", choices=["fp32", "fp16"], default=None)
    parser.add_argument("--converter", choices=["auto", "litert_torch", "ai_edge_torch"], default="auto")
    parser.add_argument("--include-softmax", action="store_true")
    parser.add_argument("--report", type=Path, default=Path("outputs/export/tflite_report.json"))
    return parser.parse_args()


def require_torch():
    try:
        import torch
    except ImportError as exc:
        raise SystemExit("PyTorch is required for TFLite export. Run this on the Linux server environment.") from exc
    return torch


def resolve_project_path(path_value: str | Path | None) -> Path | None:
    if path_value is None:
        return None
    path = Path(path_value)
    if path.is_absolute():
        return path
    return (PROJECT_ROOT / path).resolve()


def main() -> None:
    args = parse_args()
    require_torch()

    from src.export.tflite import TFLiteExportConfig, export_tflite, import_litert_torch_converter
    from src.utils.config import get_nested, load_yaml

    model_cfg = load_yaml(args.model_config)
    export_cfg = load_yaml(args.export_config)

    checkpoint = resolve_project_path(args.checkpoint or get_nested(export_cfg, "export.checkpoint"))
    output = resolve_project_path(args.output or get_nested(export_cfg, "export.formats.tflite.output"))
    input_size_raw = get_nested(export_cfg, "export.input_size", get_nested(model_cfg, "model.input_size", [320, 320]))
    input_size = (int(input_size_raw[0]), int(input_size_raw[1]))
    precision = args.precision or str(get_nested(export_cfg, "export.formats.tflite.precision", "fp16"))
    include_softmax = bool(args.include_softmax or get_nested(model_cfg, "model.export.include_softmax", False))

    converter_module = import_litert_torch_converter(args.converter)
    config = TFLiteExportConfig(
        checkpoint=checkpoint,
        output=output,
        input_size=input_size,
        num_classes=int(get_nested(model_cfg, "model.num_classes", 20)),
        width_mult=float(get_nested(model_cfg, "model.encoder.width_mult", 1.0)),
        output_stride=int(get_nested(model_cfg, "model.encoder.output_stride", 16)),
        decoder_channels=int(get_nested(model_cfg, "model.decoder.channels", 128)),
        dropout=float(get_nested(model_cfg, "model.decoder.dropout", 0.0)),
        device=args.device,
        precision=precision,
        include_softmax=include_softmax,
        converter=args.converter,
    )
    exported_path = export_tflite(config, converter_module=converter_module)
    report = {
        "tflite_path": str(exported_path),
        "checkpoint": str(checkpoint),
        "input_size": list(input_size),
        "precision": precision,
        "include_softmax": include_softmax,
        "converter": converter_module.__name__,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
