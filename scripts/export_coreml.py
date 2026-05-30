"""Export HeadParsingMobile checkpoints to Core ML mlpackage."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export HeadParsingMobile to Core ML.")
    parser.add_argument("--model-config", type=Path, default=Path("configs/model_mobilev2_lraspp_320.yaml"))
    parser.add_argument("--export-config", type=Path, default=Path("configs/export_mobile.yaml"))
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--precision", choices=["fp32", "fp16"], default=None)
    parser.add_argument("--minimum-deployment-target", type=str, default=None, help="Example: ios15, ios16, ios17.")
    parser.add_argument("--include-softmax", action="store_true")
    parser.add_argument("--report", type=Path, default=Path("outputs/export/coreml_report.json"))
    return parser.parse_args()


def require_torch():
    try:
        import torch
    except ImportError as exc:
        raise SystemExit("PyTorch is required for Core ML export. Run this in the export environment.") from exc
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

    from src.export.coreml import CoreMLExportConfig, export_coreml, import_coremltools
    from src.utils.config import get_nested, load_yaml

    model_cfg = load_yaml(args.model_config)
    export_cfg = load_yaml(args.export_config)

    checkpoint = resolve_project_path(args.checkpoint or get_nested(export_cfg, "export.checkpoint"))
    output = resolve_project_path(args.output or get_nested(export_cfg, "export.formats.coreml.output"))
    input_size_raw = get_nested(export_cfg, "export.input_size", get_nested(model_cfg, "model.input_size", [320, 320]))
    input_size = (int(input_size_raw[0]), int(input_size_raw[1]))
    precision = args.precision or str(get_nested(export_cfg, "export.formats.coreml.precision", "fp16"))
    target = args.minimum_deployment_target or str(
        get_nested(export_cfg, "export.formats.coreml.minimum_deployment_target", "ios15")
    )
    include_softmax = bool(args.include_softmax or get_nested(model_cfg, "model.export.include_softmax", False))

    ct = import_coremltools()
    config = CoreMLExportConfig(
        checkpoint=checkpoint,
        output=output,
        input_size=input_size,
        num_classes=int(get_nested(model_cfg, "model.num_classes", 20)),
        width_mult=float(get_nested(model_cfg, "model.encoder.width_mult", 1.0)),
        output_stride=int(get_nested(model_cfg, "model.encoder.output_stride", 16)),
        decoder_channels=int(get_nested(model_cfg, "model.decoder.channels", 128)),
        dropout=float(get_nested(model_cfg, "model.decoder.dropout", 0.0)),
        precision=precision,
        minimum_deployment_target=target,
        include_softmax=include_softmax,
    )
    exported_path = export_coreml(config)
    report = {
        "coreml_path": str(exported_path),
        "checkpoint": str(checkpoint),
        "input_size": list(input_size),
        "precision": precision,
        "minimum_deployment_target": target,
        "include_softmax": include_softmax,
        "coremltools_version": getattr(ct, "__version__", "unknown"),
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
