"""v3 export gate: compare converted artifacts against the fp32 reference on one image.

Reads the report produced by scripts/export_mobile_v3.py, runs every listed artifact
that exists locally, and enforces >= 99% pixelwise label-map agreement with the fp32
PyTorch reference (guards channel-order / normalization-folding mistakes end to end).

TFLite artifacts need `ai-edge-litert` (or tensorflow) in the environment; CoreML
artifacts need `coremltools` on macOS. Run per platform with --formats.

    python tools/compare_v3_artifacts.py --checkpoint outputs/train/best.pt \
        --image data/samples/face.jpg --formats tflite
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def artifact_label_map(raw: np.ndarray, output_kind: str, framework: str) -> np.ndarray:
    """Normalize an artifact's raw output to a [H, W] label map."""

    from src.export.compare import logits_to_label_map

    arr = np.asarray(raw)
    if output_kind == "label_map":
        return arr[0] if arr.ndim == 3 and arr.shape[0] == 1 else arr
    if output_kind == "logits":
        # TFLite v3 logits are channel-last, CoreML logits stay channel-first.
        return logits_to_label_map(arr, class_axis=-1 if framework == "tflite" else 1)
    raise ValueError(f"Unsupported output_kind: {output_kind}")


def build_gate_report(reference: np.ndarray, entries, threshold: float = 0.99) -> dict:
    """Aggregate per-artifact agreement into one gate report."""

    from src.export.compare import label_map_agreement

    artifacts = []
    for name, label_map in entries:
        result = label_map_agreement(reference, label_map, threshold=threshold)
        artifacts.append({"artifact": name, **result.to_dict()})
    return {"ok": all(item["ok"] for item in artifacts), "threshold": threshold, "artifacts": artifacts}


def load_input_image(path: Path, input_size: tuple[int, int]) -> np.ndarray:
    from PIL import Image

    image = Image.open(path).convert("RGB").resize((input_size[1], input_size[0]), Image.BILINEAR)
    return np.asarray(image, dtype=np.uint8)


def run_tflite(path: Path, image_hwc_uint8: np.ndarray) -> np.ndarray:
    try:
        from ai_edge_litert.interpreter import Interpreter
    except ImportError:
        try:
            from tensorflow.lite import Interpreter  # type: ignore[no-redef]
        except ImportError as exc:
            raise RuntimeError("Install ai-edge-litert (or tensorflow) to run TFLite artifacts.") from exc

    interpreter = Interpreter(model_path=str(path))
    interpreter.allocate_tensors()
    input_detail = interpreter.get_input_details()[0]
    output_detail = interpreter.get_output_details()[0]
    interpreter.set_tensor(input_detail["index"], image_hwc_uint8[None, ...])
    interpreter.invoke()
    return interpreter.get_tensor(output_detail["index"])


def run_coreml(path: Path, image_hwc_uint8: np.ndarray, output_kind: str) -> np.ndarray:
    try:
        import coremltools as ct
    except ImportError as exc:
        raise RuntimeError("Install coremltools (macOS) to run CoreML artifacts.") from exc
    from PIL import Image

    model = ct.models.MLModel(str(path))
    prediction = model.predict({"image": Image.fromarray(image_hwc_uint8)})
    key = "label_map" if output_kind == "label_map" else "logits"
    return np.asarray(prediction[key])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Gate v3 artifacts against the fp32 reference.")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--image", type=Path, required=True, help="Fixed RGB test image (a val face crop).")
    parser.add_argument("--model-config", type=Path, default=Path("configs/model_mobilev2_lraspp_256_9cls.yaml"))
    parser.add_argument("--export-report", type=Path, default=Path("outputs/export/v3_export_report.json"))
    parser.add_argument("--formats", type=str, default="tflite,coreml")
    parser.add_argument("--threshold", type=float, default=0.99)
    parser.add_argument("--report", type=Path, default=Path("outputs/export/v3_gate_report.json"))
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    from src.export.compare import compute_reference_label_map
    from src.export.onnx import OnnxExportConfig, build_model_from_export_config
    from src.utils.config import get_nested, load_yaml

    model_cfg = load_yaml(args.model_config)
    export_report = json.loads(args.export_report.read_text(encoding="utf-8"))
    input_size = tuple(int(v) for v in export_report.get("input_size", get_nested(model_cfg, "model.input_size")))
    formats = {token.strip() for token in args.formats.split(",") if token.strip()}

    model = build_model_from_export_config(
        OnnxExportConfig(
            checkpoint=args.checkpoint,
            output=Path("unused.onnx"),
            input_size=(input_size[0], input_size[1]),
            num_classes=int(get_nested(model_cfg, "model.num_classes", 9)),
            width_mult=float(get_nested(model_cfg, "model.encoder.width_mult", 1.0)),
            output_stride=int(get_nested(model_cfg, "model.encoder.output_stride", 16)),
            decoder_channels=int(get_nested(model_cfg, "model.decoder.channels", 128)),
        )
    )
    image = load_input_image(args.image, (input_size[0], input_size[1]))
    reference = compute_reference_label_map(model, image)

    entries = []
    skipped = []
    for artifact in export_report.get("artifacts", []):
        if artifact["format"] not in formats:
            continue
        path = Path(artifact["path"])
        if not path.exists():
            skipped.append(artifact["path"])
            continue
        raw = run_tflite(path, image) if artifact["format"] == "tflite" else run_coreml(
            path, image, artifact["output_kind"]
        )
        name = f"{artifact['format']}/{artifact['output_kind']}"
        entries.append((name, artifact_label_map(raw, artifact["output_kind"], artifact["format"])))

    if not entries:
        print("No artifacts found to compare (check --formats and export report paths).")
        return 1

    report = build_gate_report(reference, entries, threshold=args.threshold)
    report["image"] = str(args.image)
    report["skipped"] = skipped
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    for item in report["artifacts"]:
        marker = "PASS" if item["ok"] else "FAIL"
        print(f"[{marker}] {item['artifact']}: agreement={item['agreement']:.4f} (threshold {args.threshold})")
    if skipped:
        print(f"skipped (missing files): {skipped}")
    print(f"report: {args.report}")
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
