"""Mobile metadata generation for the v3 I/O contract.

The v3 metadata keeps the v2 structure (nested ``input``/``output`` + ``classes``)
and adds flat top-level keys (``input_size`` / ``input_layout`` / ``input_dtype`` /
``output_kind``) that the SDK reads to drive its pre/post-processing, replacing
hard-coded constants. Classes are carried over untouched: the class table is a
by-name mapping contract with the SDK.
"""

from __future__ import annotations

import copy
from typing import Sequence

from src.datasets.transforms import IMAGENET_MEAN, IMAGENET_STD

_FRAMEWORKS = {"tflite", "coreml"}
_OUTPUT_KINDS = {"label_map", "logits"}


def build_v3_metadata(
    base: dict,
    *,
    model_name: str,
    input_size: Sequence[int],
    framework: str,
    output_kind: str,
    color_layout: str = "RGB",
) -> dict:
    """Derive a v3 metadata dict from the v2 base metadata (classes preserved)."""

    if framework not in _FRAMEWORKS:
        raise ValueError(f"Unsupported v3 framework: {framework}")
    if output_kind not in _OUTPUT_KINDS:
        raise ValueError(f"Unsupported v3 output_kind: {output_kind}")
    classes = base.get("classes")
    if not classes:
        raise ValueError("Base metadata must carry the class table (classes).")

    height, width = int(input_size[0]), int(input_size[1])
    num_classes = len(classes)

    normalization = {
        "in_graph": True,
        "mean": list(IMAGENET_MEAN),
        "std": list(IMAGENET_STD),
        "formula": "((rgb / 255.0) - mean) / std, std folded into the stem conv weights",
    }

    if framework == "tflite":
        input_layout = "NHWC"
        input_section = {
            "name": "image",
            "shape": [1, height, width, 3],
            "layout": "NHWC",
            "color": color_layout,
            "dtype": "uint8",
            "resize": f"bilinear_to_{width}x{height}",
            "normalization": normalization,
        }
        label_map_dtype = "uint8"
    else:
        input_layout = "image"
        input_section = {
            "name": "image",
            "shape": [1, 3, height, width],
            "layout": "image",
            "color": color_layout,
            "dtype": "uint8",
            "resize": f"bilinear_to_{width}x{height}",
            "normalization": normalization,
        }
        label_map_dtype = "int32"

    if output_kind == "label_map":
        output_section = {
            "name": "label_map",
            "shape": [1, height, width],
            "dtype": label_map_dtype,
            "activation": "none",
            "postprocess": "none (argmax in-graph); nearest-neighbor resize mask to display size if needed",
        }
    elif framework == "tflite":
        output_section = {
            "name": "logits",
            "shape": [1, height, width, num_classes],
            "layout": "NHWC",
            "dtype": "float32",
            "activation": "none",
            "postprocess": "argmax over the last (class) axis, then nearest-neighbor resize mask if needed",
        }
    else:
        output_section = {
            "name": "logits",
            "shape": [1, num_classes, height, width],
            "layout": "NCHW",
            "dtype": "float32",
            "activation": "none",
            "postprocess": "argmax over the class/channel axis, then nearest-neighbor resize mask if needed",
        }

    return {
        "model_name": model_name,
        "run_id": base.get("run_id"),
        "task": base.get("task", "head_semantic_segmentation"),
        "io_schema": "v3",
        "input_size": [height, width],
        "input_layout": input_layout,
        "input_dtype": "uint8",
        "output_kind": output_kind,
        "input": input_section,
        "output": output_section,
        "classes": copy.deepcopy(classes),
    }
