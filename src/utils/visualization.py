"""Visualization helpers for segmentation masks and predictions."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np
from PIL import Image, ImageDraw

from src.deploy.palette import CLASS_NAMES, PALETTE


def load_mask(mask: str | Path | Image.Image | np.ndarray) -> np.ndarray:
    if isinstance(mask, np.ndarray):
        return mask.astype(np.uint8, copy=False)
    if isinstance(mask, Image.Image):
        return np.asarray(mask.convert("L"), dtype=np.uint8)
    return np.asarray(Image.open(mask).convert("L"), dtype=np.uint8)


def colorize_mask(mask: str | Path | Image.Image | np.ndarray, palette: Sequence[Sequence[int]] = PALETTE) -> Image.Image:
    mask_arr = load_mask(mask)
    palette_arr = np.asarray(palette, dtype=np.uint8)
    clipped = np.clip(mask_arr, 0, len(palette_arr) - 1)
    color = palette_arr[clipped]
    return Image.fromarray(color)


def blend_image_mask(
    image: str | Path | Image.Image | np.ndarray,
    mask: str | Path | Image.Image | np.ndarray,
    alpha: float = 0.45,
    palette: Sequence[Sequence[int]] = PALETTE,
) -> Image.Image:
    if isinstance(image, np.ndarray):
        image_pil = Image.fromarray(image.astype(np.uint8, copy=False)).convert("RGB")
    elif isinstance(image, Image.Image):
        image_pil = image.convert("RGB")
    else:
        image_pil = Image.open(image).convert("RGB")

    color_mask = colorize_mask(mask, palette=palette).resize(image_pil.size, Image.Resampling.NEAREST)
    return Image.blend(image_pil, color_mask, alpha=alpha)


def make_prediction_grid(
    image: str | Path | Image.Image | np.ndarray,
    pred_mask: str | Path | Image.Image | np.ndarray,
    target_mask: str | Path | Image.Image | np.ndarray | None = None,
    alpha: float = 0.45,
    palette: Sequence[Sequence[int]] = PALETTE,
) -> Image.Image:
    if isinstance(image, np.ndarray):
        image_pil = Image.fromarray(image.astype(np.uint8, copy=False)).convert("RGB")
    elif isinstance(image, Image.Image):
        image_pil = image.convert("RGB")
    else:
        image_pil = Image.open(image).convert("RGB")

    pred_overlay = blend_image_mask(image_pil, pred_mask, alpha=alpha, palette=palette)
    panels = [image_pil, colorize_mask(pred_mask, palette=palette).resize(image_pil.size, Image.Resampling.NEAREST), pred_overlay]
    labels = ["image", "prediction", "overlay"]

    if target_mask is not None:
        target_color = colorize_mask(target_mask, palette=palette).resize(image_pil.size, Image.Resampling.NEAREST)
        target_overlay = blend_image_mask(image_pil, target_mask, alpha=alpha, palette=palette)
        panels.extend([target_color, target_overlay])
        labels.extend(["target", "target overlay"])

    width, height = image_pil.size
    label_h = 24
    canvas = Image.new("RGB", (width * len(panels), height + label_h), (255, 255, 255))
    draw = ImageDraw.Draw(canvas)

    for index, (panel, label) in enumerate(zip(panels, labels, strict=True)):
        x = index * width
        canvas.paste(panel.resize((width, height), Image.Resampling.BILINEAR), (x, label_h))
        draw.text((x + 6, 5), label, fill=(0, 0, 0))

    return canvas


def mask_class_histogram(
    mask: str | Path | Image.Image | np.ndarray,
    num_classes: int = len(CLASS_NAMES),
    class_names: Sequence[str] = CLASS_NAMES,
) -> dict[str, int]:
    mask_arr = load_mask(mask)
    counts = np.bincount(mask_arr.reshape(-1), minlength=num_classes)
    return {class_names[index]: int(counts[index]) for index in range(num_classes)}
