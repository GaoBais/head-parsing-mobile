"""Export HeadParsingMobile checkpoints to ONNX and inspect mobile ops."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export HeadParsingMobile to ONNX.")
    parser.add_argument("--model-config", type=Path, default=Path("configs/model_mobilev2_lraspp_320.yaml"))
    parser.add_argument("--export-config", type=Path, default=Path("configs/export_mobile.yaml"))
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--opset", type=int, default=None)
    parser.add_argument("--dynamic-batch", action="store_true")
    parser.add_argument("--include-softmax", action="store_true")
    parser.add_argument("--skip-graph-check", action="store_true")
    parser.add_argument("--report", type=Path, default=Path("outputs/export/onnx_report.json"))
    return parser.parse_args()


def require_torch():
    try:
        import torch
    except ImportError as exc:
        raise SystemExit("PyTorch is required for ONNX export. Run this on the server environment.") from exc
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

    from src.export.graph_check import inspect_onnx_ops
    from src.export.onnx import OnnxExportConfig, export_onnx
    from src.utils.config import get_nested, load_yaml

    model_cfg = load_yaml(args.model_config)
    export_cfg = load_yaml(args.export_config)

    checkpoint = resolve_project_path(args.checkpoint or get_nested(export_cfg, "export.checkpoint"))
    output = resolve_project_path(args.output or get_nested(export_cfg, "export.formats.onnx.output"))
    input_size_raw = get_nested(export_cfg, "export.input_size", get_nested(model_cfg, "model.input_size", [320, 320]))
    input_size = (int(input_size_raw[0]), int(input_size_raw[1]))
    opset = args.opset or int(get_nested(export_cfg, "export.formats.onnx.opset", 17))

    include_softmax = bool(args.include_softmax or get_nested(model_cfg, "model.export.include_softmax", False))
    config = OnnxExportConfig(
        checkpoint=checkpoint,
        output=output,
        input_size=input_size,
        num_classes=int(get_nested(model_cfg, "model.num_classes", 20)),
        width_mult=float(get_nested(model_cfg, "model.encoder.width_mult", 1.0)),
        output_stride=int(get_nested(model_cfg, "model.encoder.output_stride", 16)),
        decoder_channels=int(get_nested(model_cfg, "model.decoder.channels", 128)),
        dropout=float(get_nested(model_cfg, "model.decoder.dropout", 0.0)),
        opset=opset,
        device=args.device,
        dynamic_batch=args.dynamic_batch,
        include_softmax=include_softmax,
    )

    exported_path = export_onnx(config)
    report = {
        "onnx_path": str(exported_path),
        "checkpoint": str(checkpoint),
        "input_size": list(input_size),
        "opset": opset,
        "dynamic_batch": args.dynamic_batch,
        "include_softmax": include_softmax,
    }

    if not args.skip_graph_check and bool(get_nested(export_cfg, "export.validation.check_mobile_ops", True)):
        graph_report = inspect_onnx_ops(exported_path)
        report["graph"] = graph_report.to_dict()
        if not graph_report.ok:
            print("Warning: ONNX graph contains ops outside the mobile-friendly allowlist:")
            print(json.dumps(graph_report.unsupported_ops, indent=2))

    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
