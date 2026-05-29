"""Segmentation metrics."""

from __future__ import annotations

import torch


class SegmentationMetrics:
    """Streaming confusion-matrix based segmentation metrics."""

    def __init__(self, num_classes: int, ignore_index: int = 255, device: str | torch.device = "cpu") -> None:
        self.num_classes = int(num_classes)
        self.ignore_index = int(ignore_index)
        self.confusion_matrix = torch.zeros((self.num_classes, self.num_classes), dtype=torch.int64, device=device)

    @torch.no_grad()
    def update(self, logits_or_pred: torch.Tensor, target: torch.Tensor) -> None:
        if logits_or_pred.ndim == 4:
            pred = torch.argmax(logits_or_pred, dim=1)
        else:
            pred = logits_or_pred

        pred = pred.reshape(-1)
        target = target.reshape(-1)
        valid = (target != self.ignore_index) & (target >= 0) & (target < self.num_classes)
        target = target[valid]
        pred = pred[valid].clamp(0, self.num_classes - 1)

        indices = target * self.num_classes + pred
        bins = torch.bincount(indices, minlength=self.num_classes * self.num_classes)
        self.confusion_matrix += bins.reshape(self.num_classes, self.num_classes)

    def reset(self) -> None:
        self.confusion_matrix.zero_()

    def compute(self) -> dict[str, torch.Tensor]:
        matrix = self.confusion_matrix.to(torch.float32)
        true_positive = torch.diag(matrix)
        gt = matrix.sum(dim=1)
        pred = matrix.sum(dim=0)
        union = gt + pred - true_positive

        iou = true_positive / union.clamp_min(1.0)
        pixel_acc = true_positive.sum() / matrix.sum().clamp_min(1.0)
        class_acc = true_positive / gt.clamp_min(1.0)

        valid_iou = union > 0
        valid_acc = gt > 0
        mean_iou = iou[valid_iou].mean() if torch.any(valid_iou) else torch.tensor(0.0, device=matrix.device)
        mean_acc = class_acc[valid_acc].mean() if torch.any(valid_acc) else torch.tensor(0.0, device=matrix.device)

        return {
            "pixel_acc": pixel_acc,
            "mean_acc": mean_acc,
            "mean_iou": mean_iou,
            "per_class_iou": iou,
            "confusion_matrix": self.confusion_matrix.clone(),
        }
