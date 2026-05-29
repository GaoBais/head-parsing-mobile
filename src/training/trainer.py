"""Training and evaluation loops."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import torch
from torch import nn

from src.training.distillation import SoftTargetDistillationLoss
from src.training.metrics import SegmentationMetrics


def move_sample_to_device(sample: dict[str, Any], device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    image = sample["image"].to(device, non_blocking=True)
    mask = sample["mask"].to(device, non_blocking=True)
    return image, mask


def train_one_epoch(
    model: nn.Module,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    data_loader: Iterable,
    device: torch.device,
    epoch: int,
    scaler: torch.cuda.amp.GradScaler | None = None,
    scheduler: torch.optim.lr_scheduler.LRScheduler | None = None,
    teacher_model: nn.Module | None = None,
    distillation_criterion: SoftTargetDistillationLoss | None = None,
    distillation_weight: float = 0.0,
    log_interval: int = 20,
) -> dict[str, float]:
    model.train()
    if teacher_model is not None:
        teacher_model.eval()
    totals: dict[str, float] = {}
    batches = 0

    for step, sample in enumerate(data_loader, start=1):
        images, masks = move_sample_to_device(sample, device)
        optimizer.zero_grad(set_to_none=True)

        use_amp = scaler is not None
        with torch.cuda.amp.autocast(enabled=use_amp):
            logits = model(images)
            losses = criterion(logits, masks)
            loss = losses["loss"] if isinstance(losses, dict) else losses
            if teacher_model is not None and distillation_criterion is not None and distillation_weight > 0:
                with torch.no_grad():
                    teacher_logits = teacher_model(images)
                distill_loss = distillation_criterion(logits, teacher_logits)
                loss = loss + distillation_weight * distill_loss
                if isinstance(losses, dict):
                    losses = dict(losses)
                    losses["distill"] = distill_loss.detach()
                    losses["loss"] = loss

        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        if scheduler is not None:
            scheduler.step()

        batches += 1
        if isinstance(losses, dict):
            for key, value in losses.items():
                totals[key] = totals.get(key, 0.0) + float(value.detach().cpu())
        else:
            totals["loss"] = totals.get("loss", 0.0) + float(loss.detach().cpu())

        if log_interval > 0 and step % log_interval == 0:
            lr = optimizer.param_groups[0]["lr"]
            avg_loss = totals.get("loss", 0.0) / max(1, batches)
            print(f"epoch={epoch} step={step} loss={avg_loss:.5f} lr={lr:.7f}")

    return {key: value / max(1, batches) for key, value in totals.items()}


@torch.no_grad()
def evaluate(
    model: nn.Module,
    criterion: nn.Module | None,
    data_loader: Iterable,
    device: torch.device,
    num_classes: int,
    ignore_index: int = 255,
    include_per_class: bool = False,
) -> dict[str, Any]:
    model.eval()
    metrics = SegmentationMetrics(num_classes=num_classes, ignore_index=ignore_index, device=device)
    totals: dict[str, float] = {}
    batches = 0

    for sample in data_loader:
        images, masks = move_sample_to_device(sample, device)
        logits = model(images)
        metrics.update(logits, masks)

        if criterion is not None:
            losses = criterion(logits, masks)
            batches += 1
            if isinstance(losses, dict):
                for key, value in losses.items():
                    totals[key] = totals.get(key, 0.0) + float(value.detach().cpu())

    result = {key: value / max(1, batches) for key, value in totals.items()}
    metric_values = metrics.compute()
    result["pixel_acc"] = float(metric_values["pixel_acc"].detach().cpu())
    result["mean_acc"] = float(metric_values["mean_acc"].detach().cpu())
    result["mean_iou"] = float(metric_values["mean_iou"].detach().cpu())
    if include_per_class:
        result["per_class_iou"] = [float(value) for value in metric_values["per_class_iou"].detach().cpu()]
        result["confusion_matrix"] = metric_values["confusion_matrix"].detach().cpu().tolist()
    return result
