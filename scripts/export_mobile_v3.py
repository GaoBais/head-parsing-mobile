"""Export the four v3 mobile artifacts (TFLite/CoreML x label_map/logits) + metadata.

Run on the export environment (Linux server for TFLite, macOS for CoreML):

    python scripts/export_mobile_v3.py --formats tflite
    python scripts/export_mobile_v3.py --formats coreml
    python scripts/export_mobile_v3.py                    # both

Each artifact gets a sibling ``<stem>_metadata.json`` describing the v3 I/O contract
(input_size / input_layout / input_dtype / output_kind; class table unchanged).
After exporting, run tools/compare_v3_artifacts.py for the >= 99% label-map
agreement gate and scripts/evaluate.py with configs/model_mobilev2_lraspp_256_9cls.yaml
for the 256 accuracy gate.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

_FORMATS = {"tflite", "coreml"}
_OUTPUT_KINDS = {"label_map", "logits"}


@dataclass(frozen=True)
class V3ArtifactSpec:
    format: str
    output_kind: str
    output: Path


def load_v3_artifact_specs(export_cfg: dict) -> list[V3ArtifactSpec]:
    """Parse and validate export_v3.artifacts entries."""

    entries = (export_cfg.get("export_v3") or {}).get("artifacts") or []
    if not entries:
        raise ValueError("export_v3.artifacts must list at least one artifact.")

    specs: list[V3ArtifactSpec] = []
    seen_outputs: set[str] = set()
    for entry in entries:
        fmt = str(entry.get("format", ""))
        kind = str(entry.get("output_kind", ""))
        output = str(entry.get("output", ""))
        if fmt not in _FORMATS:
            raise ValueError(f"Unsupported artifact format: {fmt!r}")
        if kind not in _OUTPUT_KINDS:
            raise ValueError(f"Unsupported artifact output_kind: {kind!r}")
        if not output:
            raise ValueError("Artifact entry is missing output path.")
        if output in seen_outputs:
            raise ValueError(f"Duplicate artifact output path: {output}")
        seen_outputs.add(output)
        specs.append(V3ArtifactSpec(format=fmt, output_kind=kind, output=Path(output)))
    return specs


def filter_specs(specs: list[V3ArtifactSpec], formats: set[str]) -> list[V3ArtifactSpec]:
    return [spec for spec in specs if spec.format in formats]


def metadata_path_for(output: Path) -> Path:
    return output.parent / f"{output.stem}_metadata.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export the v3 mobile artifacts.")
    parser.add_argument("--model-config", type=Path, default=Path("configs/model_mobilev2_lraspp_256_9cls.yaml"))
    parser.add_argument("--export-config", type=Path, default=Path("configs/export_mobile_v3.yaml"))
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--formats", type=str, default="tflite,coreml", help="Comma list: tflite,coreml.")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--converter", choices=["auto", "litert_torch", "ai_edge_torch"], default="auto")
    parser.add_argument("--report", type=Path, default=Path("outputs/export/v3_export_report.json"))
    return parser.parse_args()


def resolve_project_path(path_value: str | Path | None) -> Path | None:
    if path_value is None:
        return None
    path = Path(path_value)
    return path if path.is_absolute() else (PROJECT_ROOT / path).resolve()


def main() -> None:
    args = parse_args()

    import yaml

    from src.export.metadata import build_v3_metadata
    from src.utils.config import get_nested, load_yaml

    model_cfg = load_yaml(args.model_config)
    export_cfg = load_yaml(args.export_config)

    requested_formats = {token.strip() for token in args.formats.split(",") if token.strip()}
    unknown = requested_formats - _FORMATS
    if unknown:
        raise SystemExit(f"Unknown --formats entries: {sorted(unknown)}")

    specs = filter_specs(load_v3_artifact_specs(export_cfg), requested_formats)
    if not specs:
        raise SystemExit("No artifacts left after --formats filtering.")

    checkpoint = resolve_project_path(args.checkpoint or get_nested(export_cfg, "export_v3.checkpoint"))
    input_size_raw = get_nested(export_cfg, "export_v3.input_size", get_nested(model_cfg, "model.input_size", [256, 256]))
    input_size = (int(input_size_raw[0]), int(input_size_raw[1]))
    model_name = str(get_nested(export_cfg, "export_v3.model_name", "HeadParsingMobileV3"))
    color_layout = str(get_nested(export_cfg, "export_v3.color_layout", "RGB"))
    base_metadata_path = resolve_project_path(get_nested(export_cfg, "export_v3.base_metadata"))
    base_metadata = json.loads(Path(base_metadata_path).read_text(encoding="utf-8"))

    common = {
        "num_classes": int(get_nested(model_cfg, "model.num_classes", 9)),
        "width_mult": float(get_nested(model_cfg, "model.encoder.width_mult", 1.0)),
        "output_stride": int(get_nested(model_cfg, "model.encoder.output_stride", 16)),
        "decoder_channels": int(get_nested(model_cfg, "model.decoder.channels", 128)),
        "dropout": float(get_nested(model_cfg, "model.decoder.dropout", 0.0)),
    }

    report_entries = []
    for spec in specs:
        output = resolve_project_path(spec.output)
        if spec.format == "tflite":
            from src.export.tflite import TFLiteV3ExportConfig, export_tflite_v3

            exported = export_tflite_v3(
                TFLiteV3ExportConfig(
                    checkpoint=checkpoint,
                    output=output,
                    input_size=input_size,
                    device=args.device,
                    converter=args.converter,
                    output_kind=spec.output_kind,
                    **common,
                )
            )
        else:
            from src.export.coreml import CoreMLV3ExportConfig, export_coreml_v3

            exported = export_coreml_v3(
                CoreMLV3ExportConfig(
                    checkpoint=checkpoint,
                    output=output,
                    input_size=input_size,
                    output_kind=spec.output_kind,
                    color_layout=color_layout,
                    **common,
                )
            )

        metadata = build_v3_metadata(
            base_metadata,
            model_name=model_name,
            input_size=input_size,
            framework=spec.format,
            output_kind=spec.output_kind,
            color_layout=color_layout,
        )
        metadata_path = metadata_path_for(exported)
        metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")

        print(f"exported {spec.format}/{spec.output_kind}: {exported}")
        report_entries.append(
            {
                "format": spec.format,
                "output_kind": spec.output_kind,
                "path": str(exported),
                "metadata": str(metadata_path),
            }
        )

    args.report.parent.mkdir(parents=True, exist_ok=True)
    report = {
        "checkpoint": str(checkpoint),
        "input_size": list(input_size),
        "model_name": model_name,
        "artifacts": report_entries,
    }
    args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"report: {args.report}")
    print(yaml.safe_dump({"exported": len(report_entries)}, default_flow_style=True).strip())


if __name__ == "__main__":
    main()
