"""v3 export gate: label-map agreement between the fp32 reference and converted artifacts.

Gate contract: on a fixed input, the converted artifact's label map must agree with
the fp32 server reference on >= 99% of pixels (guards NHWC/uint8 channel-order and
normalization-folding mistakes in the export chain).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class AgreementReport:
    agreement: float
    threshold: float
    total_pixels: int
    mismatched_pixels: int

    @property
    def ok(self) -> bool:
        return self.agreement >= self.threshold

    def to_dict(self) -> dict:
        return {
            "agreement": self.agreement,
            "threshold": self.threshold,
            "total_pixels": self.total_pixels,
            "mismatched_pixels": self.mismatched_pixels,
            "ok": self.ok,
        }


def _squeeze_batch(label_map: np.ndarray) -> np.ndarray:
    arr = np.asarray(label_map)
    if arr.ndim == 3 and arr.shape[0] == 1:
        arr = arr[0]
    if arr.ndim != 2:
        raise ValueError(f"Expected a [H, W] (or [1, H, W]) label map, got shape {arr.shape}.")
    return arr


def label_map_agreement(
    reference: np.ndarray,
    candidate: np.ndarray,
    threshold: float = 0.99,
) -> AgreementReport:
    """Pixelwise agreement ratio between two label maps (dtype-insensitive)."""

    ref = _squeeze_batch(reference)
    got = _squeeze_batch(candidate)
    if ref.shape != got.shape:
        raise ValueError(f"Label map shapes differ: {ref.shape} vs {got.shape}.")
    total = int(ref.size)
    mismatched = int(np.count_nonzero(ref.astype(np.int64) != got.astype(np.int64)))
    return AgreementReport(
        agreement=1.0 - mismatched / total,
        threshold=threshold,
        total_pixels=total,
        mismatched_pixels=mismatched,
    )


def logits_to_label_map(logits: np.ndarray, class_axis: int) -> np.ndarray:
    """Argmax a logits tensor (optionally batched with leading 1) to a [H, W] map."""

    arr = np.asarray(logits)
    if arr.ndim == 4 and arr.shape[0] == 1:
        arr = arr[0]
        if class_axis > 0:
            class_axis -= 1
    if arr.ndim != 3:
        raise ValueError(f"Expected a 3D (or batch-1 4D) logits tensor, got shape {arr.shape}.")
    return np.argmax(arr, axis=class_axis)


def compute_reference_label_map(model, image_hwc_uint8: np.ndarray) -> np.ndarray:
    """fp32 server reference: v2 preprocessing ((rgb/255 - mean) / std) -> model -> argmax.

    ``image_hwc_uint8`` must already be at the model input size; resizing/ROI policy
    belongs to the caller so the reference and the artifact consume identical pixels.
    """

    import torch

    from src.datasets.transforms import IMAGENET_MEAN, IMAGENET_STD

    image = np.asarray(image_hwc_uint8)
    if image.ndim != 3 or image.shape[2] != 3 or image.dtype != np.uint8:
        raise ValueError(f"Expected an [H, W, 3] uint8 image, got {image.shape} {image.dtype}.")

    x = image.astype(np.float32) / 255.0
    x = (x - np.asarray(IMAGENET_MEAN, dtype=np.float32)) / np.asarray(IMAGENET_STD, dtype=np.float32)
    tensor = torch.from_numpy(np.transpose(x, (2, 0, 1))).unsqueeze(0)
    with torch.no_grad():
        logits = model(tensor)
    return logits.argmax(dim=1)[0].cpu().numpy()
