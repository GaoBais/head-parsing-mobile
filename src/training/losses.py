"""Segmentation losses for head parsing training."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn


class OhemCrossEntropyLoss(nn.Module):
    """Online hard example mining cross entropy for segmentation."""

    def __init__(self, threshold: float = 0.7, min_kept: int = 100000, ignore_index: int = 255) -> None:
        super().__init__()
        self.threshold = float(threshold)
        self.min_kept = int(min_kept)
        self.ignore_index = int(ignore_index)

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        losses = F.cross_entropy(logits, target, ignore_index=self.ignore_index, reduction="none")
        valid = target != self.ignore_index
        losses = losses[valid]

        if losses.numel() == 0:
            return logits.sum() * 0.0

        kept = min(self.min_kept, losses.numel())
        sorted_losses, _ = torch.sort(losses, descending=True)
        threshold_loss = -torch.log(torch.tensor(self.threshold, device=logits.device, dtype=logits.dtype))

        if sorted_losses[kept - 1] > threshold_loss:
            selected = sorted_losses[sorted_losses > threshold_loss]
        else:
            selected = sorted_losses[:kept]
        return selected.mean()


class DiceLoss(nn.Module):
    """Multi-class soft Dice loss."""

    def __init__(self, num_classes: int, ignore_index: int = 255, eps: float = 1e-6) -> None:
        super().__init__()
        self.num_classes = int(num_classes)
        self.ignore_index = int(ignore_index)
        self.eps = float(eps)

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        valid = target != self.ignore_index
        safe_target = target.masked_fill(~valid, 0)
        probs = torch.softmax(logits, dim=1)
        one_hot = F.one_hot(safe_target, num_classes=self.num_classes).permute(0, 3, 1, 2).to(probs.dtype)
        valid = valid.unsqueeze(1).to(probs.dtype)

        probs = probs * valid
        one_hot = one_hot * valid

        dims = (0, 2, 3)
        intersection = torch.sum(probs * one_hot, dim=dims)
        denominator = torch.sum(probs + one_hot, dim=dims)
        dice = (2.0 * intersection + self.eps) / (denominator + self.eps)
        return 1.0 - dice.mean()


class BoundaryCrossEntropyLoss(nn.Module):
    """Cross entropy focused on label boundary pixels."""

    def __init__(self, ignore_index: int = 255) -> None:
        super().__init__()
        self.ignore_index = int(ignore_index)

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        ce = F.cross_entropy(logits, target, ignore_index=self.ignore_index, reduction="none")
        boundary = label_boundary_mask(target, ignore_index=self.ignore_index)
        valid = boundary & (target != self.ignore_index)

        if not torch.any(valid):
            return ce[target != self.ignore_index].mean()
        return ce[valid].mean()


def label_boundary_mask(target: torch.Tensor, ignore_index: int = 255) -> torch.Tensor:
    """Return a boolean mask for pixels adjacent to a different class."""

    boundary = torch.zeros_like(target, dtype=torch.bool)
    valid = target != ignore_index

    diff_h = (target[:, 1:, :] != target[:, :-1, :]) & valid[:, 1:, :] & valid[:, :-1, :]
    diff_w = (target[:, :, 1:] != target[:, :, :-1]) & valid[:, :, 1:] & valid[:, :, :-1]

    boundary[:, 1:, :] |= diff_h
    boundary[:, :-1, :] |= diff_h
    boundary[:, :, 1:] |= diff_w
    boundary[:, :, :-1] |= diff_w
    return boundary


@dataclass
class SegmentationLossConfig:
    num_classes: int = 19
    ce_weight: float = 1.0
    dice_weight: float = 0.5
    boundary_weight: float = 0.2
    ohem: bool = True
    ohem_threshold: float = 0.7
    min_kept: int = 100000
    ignore_index: int = 255


class CombinedSegmentationLoss(nn.Module):
    """Weighted CE/OHEM + Dice + boundary loss."""

    def __init__(self, config: SegmentationLossConfig) -> None:
        super().__init__()
        self.config = config
        if config.ohem:
            self.ce = OhemCrossEntropyLoss(
                threshold=config.ohem_threshold,
                min_kept=config.min_kept,
                ignore_index=config.ignore_index,
            )
        else:
            self.ce = nn.CrossEntropyLoss(ignore_index=config.ignore_index)
        self.dice = DiceLoss(num_classes=config.num_classes, ignore_index=config.ignore_index)
        self.boundary = BoundaryCrossEntropyLoss(ignore_index=config.ignore_index)

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> dict[str, torch.Tensor]:
        ce_loss = self.ce(logits, target)
        dice_loss = self.dice(logits, target)
        boundary_loss = self.boundary(logits, target)

        total = (
            self.config.ce_weight * ce_loss
            + self.config.dice_weight * dice_loss
            + self.config.boundary_weight * boundary_loss
        )
        return {
            "loss": total,
            "ce": ce_loss.detach(),
            "dice": dice_loss.detach(),
            "boundary": boundary_loss.detach(),
        }
