"""ONNX graph inspection for mobile deployment readiness."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from pathlib import Path


MOBILE_FRIENDLY_ONNX_OPS = {
    "Abs",
    "Add",
    "AveragePool",
    "BatchNormalization",
    "Cast",
    "Clip",
    "Concat",
    "Constant",
    "Conv",
    "Div",
    "Exp",
    "Flatten",
    "Gather",
    "GlobalAveragePool",
    "Identity",
    "MaxPool",
    "Mul",
    "Pad",
    "Relu",
    "Reshape",
    "Resize",
    "Shape",
    "Sigmoid",
    "Slice",
    "Softmax",
    "Squeeze",
    "Sub",
    "Transpose",
    "Unsqueeze",
}


@dataclass
class OnnxGraphReport:
    path: str
    ops: dict[str, int]
    unsupported_ops: dict[str, int]
    num_nodes: int

    @property
    def ok(self) -> bool:
        return not self.unsupported_ops

    def to_dict(self) -> dict:
        return {
            "path": self.path,
            "num_nodes": self.num_nodes,
            "ok": self.ok,
            "ops": self.ops,
            "unsupported_ops": self.unsupported_ops,
        }


def inspect_onnx_ops(path: str | Path, allowed_ops: set[str] | None = None) -> OnnxGraphReport:
    try:
        import onnx
    except ImportError as exc:
        raise RuntimeError("onnx is required for graph inspection. Install requirements.txt first.") from exc

    model_path = Path(path)
    if not model_path.exists():
        raise FileNotFoundError(f"ONNX model does not exist: {model_path}")

    model = onnx.load(str(model_path))
    onnx.checker.check_model(model)

    ops = Counter(node.op_type for node in model.graph.node)
    allowed = allowed_ops or MOBILE_FRIENDLY_ONNX_OPS
    unsupported = {op: count for op, count in sorted(ops.items()) if op not in allowed}
    return OnnxGraphReport(
        path=str(model_path),
        ops=dict(sorted(ops.items())),
        unsupported_ops=unsupported,
        num_nodes=sum(ops.values()),
    )
