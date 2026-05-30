"""Teacher loading and logit distillation helpers."""

from __future__ import annotations

import importlib
import sys
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn


@dataclass
class DistillationConfig:
    enabled: bool = False
    weight: float = 0.5
    temperature: float = 2.0


class SoftTargetDistillationLoss(nn.Module):
    """KL-divergence loss between student and teacher logits."""

    def __init__(self, temperature: float = 2.0) -> None:
        super().__init__()
        if temperature <= 0:
            raise ValueError("temperature must be positive")
        self.temperature = float(temperature)

    def forward(self, student_logits: torch.Tensor, teacher_logits: torch.Tensor) -> torch.Tensor:
        if teacher_logits.shape[-2:] != student_logits.shape[-2:]:
            teacher_logits = F.interpolate(
                teacher_logits,
                size=student_logits.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )

        temperature = self.temperature
        student_log_probs = F.log_softmax(student_logits / temperature, dim=1)
        teacher_probs = F.softmax(teacher_logits / temperature, dim=1)
        per_pixel_kl = F.kl_div(student_log_probs, teacher_probs, reduction="none").sum(dim=1)
        return per_pixel_kl.mean() * (temperature * temperature)


def first_logits(output: torch.Tensor | tuple[torch.Tensor, ...] | list[torch.Tensor]) -> torch.Tensor:
    """Normalize model outputs to the main logits tensor."""

    if isinstance(output, torch.Tensor):
        return output
    if isinstance(output, (tuple, list)) and output:
        first = output[0]
        if isinstance(first, torch.Tensor):
            return first
    raise TypeError(f"Expected tensor or non-empty tensor tuple/list, got {type(output)!r}")


class TeacherAdapter(nn.Module):
    """Wrap a teacher model that may return auxiliary outputs."""

    def __init__(self, teacher: nn.Module) -> None:
        super().__init__()
        self.teacher = teacher

    @torch.no_grad()
    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return first_logits(self.teacher(images))


def _resolve_state_dict(checkpoint: Any) -> dict[str, torch.Tensor]:
    if isinstance(checkpoint, dict):
        for key in ("model", "state_dict", "model_state_dict"):
            value = checkpoint.get(key)
            if isinstance(value, dict):
                return value
        if checkpoint and all(isinstance(value, torch.Tensor) for value in checkpoint.values()):
            return checkpoint
    raise ValueError("Could not resolve a model state_dict from the teacher checkpoint.")


def _strip_module_prefix(state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    if not any(key.startswith("module.") for key in state_dict):
        return state_dict
    return {key.removeprefix("module."): value for key, value in state_dict.items()}


def _force_no_weights(builder):
    def wrapped_builder(*args, **kwargs):
        kwargs["weights"] = None
        return builder(*args, **kwargs)

    return wrapped_builder


@contextmanager
def _disable_teacher_backbone_default_weights(*extra_modules: Any):
    """Temporarily force the upstream face-parsing ResNet builders to avoid downloads."""

    resnet_module = importlib.import_module("models.resnet")
    candidate_modules = [resnet_module, *extra_modules]
    imported_bisenet = sys.modules.get("models.bisenet")
    if imported_bisenet is not None:
        candidate_modules.append(imported_bisenet)

    originals = []
    seen: set[tuple[int, str]] = set()
    for module in candidate_modules:
        if module is None:
            continue
        for attr_name in ("resnet18", "resnet34"):
            if not hasattr(module, attr_name):
                continue
            key = (id(module), attr_name)
            if key in seen:
                continue
            seen.add(key)
            original = getattr(module, attr_name)
            originals.append((module, attr_name, original))
            setattr(module, attr_name, _force_no_weights(original))

    try:
        yield
    finally:
        for module, attr_name, original in originals:
            setattr(module, attr_name, original)


def load_bisenet_teacher(
    repo_path: str | Path,
    checkpoint_path: str | Path,
    backbone: str = "resnet34",
    num_classes: int = 20,
    device: str | torch.device = "cpu",
) -> TeacherAdapter:
    """Load the current `../face-parsing` BiSeNet model as a teacher."""

    repo = Path(repo_path).resolve()
    checkpoint = Path(checkpoint_path)
    if not repo.exists():
        raise FileNotFoundError(f"Teacher repository does not exist: {repo}")
    if not checkpoint.exists():
        raise FileNotFoundError(f"Teacher checkpoint does not exist: {checkpoint}")

    repo_str = str(repo)
    if repo_str not in sys.path:
        sys.path.insert(0, repo_str)

    try:
        bisenet_module = importlib.import_module("models.bisenet")
    except Exception as exc:
        raise RuntimeError(f"Failed to import teacher BiSeNet from {repo}") from exc

    with _disable_teacher_backbone_default_weights(bisenet_module):
        model = bisenet_module.BiSeNet(num_classes=num_classes, backbone_name=backbone)
    loaded = torch.load(checkpoint, map_location=device)
    state_dict = _strip_module_prefix(_resolve_state_dict(loaded))
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    for parameter in model.parameters():
        parameter.requires_grad_(False)

    return TeacherAdapter(model)
