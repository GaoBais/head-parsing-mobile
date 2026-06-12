"""Run a smoke test for exported TFLite/LiteRT segmentation models."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.datasets.transforms import IMAGENET_MEAN, IMAGENET_STD
from src.deploy.palette import CLASS_NAMES, PALETTE
from src.utils.visualization import blend_image_mask, colorize_mask


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inspect and optionally run a TFLite segmentation model.")
    parser.add_argument("--model", type=Path, required=True, help="Path to .tflite model.")
    parser.add_argument("--image", type=Path, default=None, help="Optional RGB image for one inference.")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/tflite_smoke"))
    parser.add_argument("--num-threads", type=int, default=1)
    parser.add_argument("--metadata", type=Path, default=Path("configs/mobile_metadata_mouth2teeth.json"))
    return parser.parse_args()


def import_tflite_interpreter():
    errors: list[str] = []

    try:
        import tensorflow as tf

        return tf.lite.Interpreter, "tensorflow"
    except Exception as exc:
        errors.append(f"tensorflow: {type(exc).__name__}: {exc}")

    try:
        from tflite_runtime.interpreter import Interpreter

        return Interpreter, "tflite_runtime"
    except Exception as exc:
        errors.append(f"tflite_runtime: {type(exc).__name__}: {exc}")

    for module_name in ("ai_edge_litert.interpreter", "litert.interpreter"):
        try:
            module = __import__(module_name, fromlist=["Interpreter"])
            return module.Interpreter, module_name
        except Exception as exc:
            errors.append(f"{module_name}: {type(exc).__name__}: {exc}")

    raise RuntimeError(
        "No usable TFLite/LiteRT Python interpreter was found. Install one of: "
        "`tensorflow`, `tflite-runtime`, or an AI Edge LiteRT interpreter package.\n"
        + "\n".join(f"- {error}" for error in errors)
    )


def create_interpreter(interpreter_cls, model_path: Path, num_threads: int):
    try:
        return interpreter_cls(model_path=str(model_path), num_threads=num_threads)
    except TypeError:
        return interpreter_cls(model_path=str(model_path))


def tensor_shape(detail: dict[str, Any]) -> list[int]:
    shape = detail.get("shape_signature")
    if shape is None or any(int(value) < 0 for value in shape):
        shape = detail.get("shape")
    return [int(value) for value in shape]


def infer_image_layout(shape: list[int]) -> tuple[str, int, int]:
    if len(shape) != 4:
        raise ValueError(f"Expected 4D input tensor, got shape={shape}")
    if shape[1] == 3:
        return "NCHW", int(shape[2]), int(shape[3])
    if shape[3] == 3:
        return "NHWC", int(shape[1]), int(shape[2])
    raise ValueError(f"Could not infer RGB layout from input shape={shape}")


def preprocess_image(image_path: Path, input_detail: dict[str, Any]) -> tuple[np.ndarray, Image.Image, str]:
    shape = tensor_shape(input_detail)
    layout, height, width = infer_image_layout(shape)
    image = Image.open(image_path).convert("RGB")
    resized = image.resize((width, height), Image.Resampling.BILINEAR)
    arr = np.asarray(resized, dtype=np.float32) / 255.0
    arr = (arr - np.asarray(IMAGENET_MEAN, dtype=np.float32)) / np.asarray(IMAGENET_STD, dtype=np.float32)

    if layout == "NCHW":
        arr = np.transpose(arr, (2, 0, 1))
    arr = np.expand_dims(arr, axis=0)

    dtype = input_detail.get("dtype", np.float32)
    if np.issubdtype(dtype, np.floating):
        arr = arr.astype(dtype)
    elif np.issubdtype(dtype, np.integer):
        scale, zero_point = input_detail.get("quantization", (0.0, 0))
        if scale <= 0:
            raise ValueError(f"Integer input tensor has invalid quantization={input_detail.get('quantization')}")
        info = np.iinfo(dtype)
        arr = np.clip(np.round(arr / scale + zero_point), info.min, info.max).astype(dtype)
    else:
        raise ValueError(f"Unsupported input dtype: {dtype}")

    return arr, image, layout


def load_mobile_metadata(metadata_path: Path) -> tuple[list[str], list[list[int]], dict[str, Any] | None]:
    if not metadata_path.exists():
        return list(CLASS_NAMES), [list(color) for color in PALETTE], None

    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    classes = metadata.get("classes", [])
    if not classes:
        return list(CLASS_NAMES), [list(color) for color in PALETTE], metadata

    max_id = max(int(item["id"]) for item in classes)
    class_names = [""] * (max_id + 1)
    palette = [[128, 128, 128] for _ in range(max_id + 1)]
    for item in classes:
        index = int(item["id"])
        class_names[index] = str(item["name"])
        palette[index] = [int(value) for value in item.get("color", [128, 128, 128])]
    if any(not name for name in class_names):
        raise ValueError(f"Metadata class ids must be contiguous from 0 to {max_id}: {metadata_path}")
    return class_names, palette, metadata


def decode_prediction(output: np.ndarray, num_classes: int = len(CLASS_NAMES)) -> tuple[np.ndarray, str]:
    if output.ndim == 4:
        if output.shape[1] == num_classes:
            return np.argmax(output[0], axis=0).astype(np.uint8), "NCHW"
        if output.shape[-1] == num_classes:
            return np.argmax(output[0], axis=-1).astype(np.uint8), "NHWC"
    if output.ndim == 3:
        if output.shape[0] == num_classes:
            return np.argmax(output, axis=0).astype(np.uint8), "CHW"
        if output.shape[-1] == num_classes:
            return np.argmax(output, axis=-1).astype(np.uint8), "HWC"
    if output.ndim == 2:
        return output.astype(np.uint8), "HW"
    raise ValueError(f"Could not decode segmentation output with shape={list(output.shape)}")


def class_histogram(mask: np.ndarray, class_names: list[str]) -> dict[str, int]:
    counts = np.bincount(mask.reshape(-1), minlength=len(class_names))
    return {name: int(counts[index]) for index, name in enumerate(class_names)}


def json_safe(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.dtype):
        return str(value)
    if isinstance(value, type) and issubclass(value, np.generic):
        return str(np.dtype(value))
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    return value


def main() -> None:
    args = parse_args()
    class_names, palette, metadata = load_mobile_metadata(args.metadata)
    interpreter_cls, runtime_name = import_tflite_interpreter()
    interpreter = create_interpreter(interpreter_cls, args.model, args.num_threads)
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    report: dict[str, Any] = {
        "model": str(args.model),
        "runtime": runtime_name,
        "input_details": json_safe(input_details),
        "output_details": json_safe(output_details),
        "metadata": str(args.metadata) if args.metadata.exists() else None,
        "num_classes": len(class_names),
        "class_names": class_names,
    }
    if metadata is not None:
        report["metadata_name"] = metadata.get("model_name")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.image is not None:
        if len(input_details) != 1:
            raise ValueError(f"Expected one input tensor, got {len(input_details)}")
        if len(output_details) != 1:
            raise ValueError(f"Expected one output tensor, got {len(output_details)}")

        input_tensor, image, input_layout = preprocess_image(args.image, input_details[0])
        interpreter.set_tensor(input_details[0]["index"], input_tensor)
        interpreter.invoke()
        output = interpreter.get_tensor(output_details[0]["index"])
        pred_mask, output_layout = decode_prediction(output, num_classes=len(class_names))

        pred_path = args.output_dir / f"{args.image.stem}_pred.png"
        color_path = args.output_dir / f"{args.image.stem}_color.png"
        overlay_path = args.output_dir / f"{args.image.stem}_overlay.jpg"
        Image.fromarray(pred_mask).save(pred_path)
        colorize_mask(pred_mask, palette=palette).save(color_path)
        blend_image_mask(image, pred_mask, palette=palette).save(overlay_path, quality=95)

        report.update(
            {
                "image": str(args.image),
                "input_layout": input_layout,
                "input_shape": list(input_tensor.shape),
                "output_shape": list(output.shape),
                "output_layout": output_layout,
                "prediction_mask": str(pred_path),
                "color_mask": str(color_path),
                "overlay": str(overlay_path),
                "predicted_classes": class_histogram(pred_mask, class_names),
            }
        )

    report_path = args.output_dir / "tflite_smoke_report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
