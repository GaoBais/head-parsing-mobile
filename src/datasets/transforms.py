"""Image and mask transforms for mobile head parsing.

The training transform intentionally simulates direct camera-frame inference
without landmark-assisted cropping: scale variation, translation, rotation,
letterbox-style placement, quality loss, and occlusion.
"""

from __future__ import annotations

import io
import random
from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np
from PIL import Image, ImageEnhance, ImageFilter

try:
    import torch
except ImportError:
    torch = None


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

DEFAULT_LABEL_SWAPS = {
    2: 3,
    3: 2,
    4: 5,
    5: 4,
    7: 8,
    8: 7,
}


def _as_size(size: int | Sequence[int]) -> tuple[int, int]:
    if isinstance(size, int):
        return size, size
    if len(size) != 2:
        raise ValueError(f"Expected size with 2 elements, got {size}")
    return int(size[0]), int(size[1])


def _image_fill(image: Image.Image) -> tuple[int, int, int]:
    arr = np.asarray(image.convert("RGB"), dtype=np.uint8)
    color = arr.reshape(-1, 3).mean(axis=0)
    return tuple(int(x) for x in color)


def _resize_pair(image: Image.Image, mask: Image.Image, size: tuple[int, int]) -> tuple[Image.Image, Image.Image]:
    return (
        image.resize(size, Image.Resampling.BILINEAR),
        mask.resize(size, Image.Resampling.NEAREST),
    )


def _crop_or_pad(
    image: Image.Image,
    mask: Image.Image,
    output_size: tuple[int, int],
    image_fill: tuple[int, int, int],
    translate_ratio: float = 0.0,
    random_letterbox: bool = True,
) -> tuple[Image.Image, Image.Image]:
    out_w, out_h = output_size
    in_w, in_h = image.size

    if in_w >= out_w:
        max_left = in_w - out_w
        left = _translated_offset(max_left, translate_ratio, randomize=random_letterbox)
        right = left + out_w
        image = image.crop((left, 0, right, in_h))
        mask = mask.crop((left, 0, right, in_h))
    else:
        canvas = Image.new("RGB", (out_w, in_h), image_fill)
        mask_canvas = Image.new("L", (out_w, in_h), 0)
        left = _translated_offset(out_w - in_w, translate_ratio, randomize=random_letterbox)
        canvas.paste(image, (left, 0))
        mask_canvas.paste(mask, (left, 0))
        image, mask = canvas, mask_canvas

    in_w, in_h = image.size
    if in_h >= out_h:
        max_top = in_h - out_h
        top = _translated_offset(max_top, translate_ratio, randomize=random_letterbox)
        bottom = top + out_h
        image = image.crop((0, top, in_w, bottom))
        mask = mask.crop((0, top, in_w, bottom))
    else:
        canvas = Image.new("RGB", (in_w, out_h), image_fill)
        mask_canvas = Image.new("L", (in_w, out_h), 0)
        top = _translated_offset(out_h - in_h, translate_ratio, randomize=random_letterbox)
        canvas.paste(image, (0, top))
        mask_canvas.paste(mask, (0, top))
        image, mask = canvas, mask_canvas

    return image, mask


def _translated_offset(max_offset: int, translate_ratio: float, randomize: bool = True) -> int:
    if max_offset <= 0:
        return 0
    if not randomize or translate_ratio <= 0:
        return int(round(max_offset / 2.0))
    center = max_offset / 2.0
    radius = max_offset * min(1.0, translate_ratio)
    return int(round(max(0.0, min(float(max_offset), random.uniform(center - radius, center + radius)))))


def _swap_left_right_labels(mask: Image.Image, label_swaps: dict[int, int] | None = None) -> Image.Image:
    label_swaps = DEFAULT_LABEL_SWAPS if label_swaps is None else label_swaps
    if not label_swaps:
        return mask
    arr = np.asarray(mask, dtype=np.uint8)
    swapped = arr.copy()
    for src, dst in label_swaps.items():
        swapped[arr == int(src)] = int(dst)
    return Image.fromarray(swapped)


def _to_tensor_or_numpy(
    image: Image.Image,
    mask: Image.Image,
    mean: Sequence[float],
    std: Sequence[float],
):
    image_arr = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0
    image_arr = (image_arr - np.asarray(mean, dtype=np.float32)) / np.asarray(std, dtype=np.float32)
    image_arr = np.transpose(image_arr, (2, 0, 1)).astype(np.float32)
    mask_arr = np.asarray(mask, dtype=np.int64)

    if torch is None:
        return image_arr, mask_arr

    return torch.from_numpy(image_arr), torch.from_numpy(mask_arr)


class Compose:
    """Compose pair transforms that operate on `(image, mask)`."""

    def __init__(self, transforms: Iterable):
        self.transforms = list(transforms)

    def __call__(self, image: Image.Image, mask: Image.Image):
        for transform in self.transforms:
            image, mask = transform(image, mask)
        return image, mask


@dataclass
class MobileTrainTransform:
    """Training transform for direct mobile camera-frame inference."""

    output_size: int | Sequence[int] = 320
    scale_range: tuple[float, float] = (0.5, 2.0)
    rotate_degrees: float = 30.0
    translate_ratio: float = 0.25
    random_letterbox: bool = True
    horizontal_flip_prob: float = 0.5
    color_jitter_prob: float = 0.8
    blur_prob: float = 0.2
    jpeg_prob: float = 0.2
    occlusion_prob: float = 0.25
    label_swaps: dict[int, int] | None = None
    mean: Sequence[float] = IMAGENET_MEAN
    std: Sequence[float] = IMAGENET_STD

    def __call__(self, image: Image.Image, mask: Image.Image):
        image = image.convert("RGB")
        mask = mask.convert("L")
        output_size = _as_size(self.output_size)
        fill = _image_fill(image)

        if self.horizontal_flip_prob > 0 and random.random() < self.horizontal_flip_prob:
            image = image.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
            mask = _swap_left_right_labels(mask, self.label_swaps).transpose(Image.Transpose.FLIP_LEFT_RIGHT)

        if self.rotate_degrees > 0:
            angle = random.uniform(-self.rotate_degrees, self.rotate_degrees)
            image = image.rotate(angle, resample=Image.Resampling.BILINEAR, fillcolor=fill)
            mask = mask.rotate(angle, resample=Image.Resampling.NEAREST, fillcolor=0)

        scale = random.uniform(*self.scale_range)
        scaled_w = max(1, int(round(output_size[0] * scale)))
        scaled_h = max(1, int(round(output_size[1] * scale)))
        image, mask = _resize_pair(image, mask, (scaled_w, scaled_h))
        image, mask = _crop_or_pad(
            image,
            mask,
            output_size,
            fill,
            translate_ratio=self.translate_ratio,
            random_letterbox=self.random_letterbox,
        )

        if self.color_jitter_prob > 0 and random.random() < self.color_jitter_prob:
            image = self._color_jitter(image)

        if self.blur_prob > 0 and random.random() < self.blur_prob:
            radius = random.uniform(0.3, 1.4)
            image = image.filter(ImageFilter.GaussianBlur(radius=radius))

        if self.jpeg_prob > 0 and random.random() < self.jpeg_prob:
            image = self._jpeg_compress(image)

        if self.occlusion_prob > 0 and random.random() < self.occlusion_prob:
            image, mask = self._random_occlusion(image, mask)

        return _to_tensor_or_numpy(image, mask, self.mean, self.std)

    @staticmethod
    def _color_jitter(image: Image.Image) -> Image.Image:
        brightness = random.uniform(0.65, 1.35)
        contrast = random.uniform(0.65, 1.35)
        saturation = random.uniform(0.65, 1.35)

        image = ImageEnhance.Brightness(image).enhance(brightness)
        image = ImageEnhance.Contrast(image).enhance(contrast)
        image = ImageEnhance.Color(image).enhance(saturation)
        return image

    @staticmethod
    def _jpeg_compress(image: Image.Image) -> Image.Image:
        buffer = io.BytesIO()
        quality = random.randint(35, 90)
        image.save(buffer, format="JPEG", quality=quality)
        buffer.seek(0)
        return Image.open(buffer).convert("RGB")

    @staticmethod
    def _random_occlusion(image: Image.Image, mask: Image.Image) -> tuple[Image.Image, Image.Image]:
        width, height = image.size
        occ_w = random.randint(max(1, width // 12), max(2, width // 3))
        occ_h = random.randint(max(1, height // 12), max(2, height // 3))
        left = random.randint(0, max(0, width - occ_w))
        top = random.randint(0, max(0, height - occ_h))
        color = tuple(random.randint(0, 255) for _ in range(3))

        image_arr = np.asarray(image).copy()
        mask_arr = np.asarray(mask).copy()
        image_arr[top : top + occ_h, left : left + occ_w, :] = color
        mask_arr[top : top + occ_h, left : left + occ_w] = 0
        return Image.fromarray(image_arr), Image.fromarray(mask_arr)


@dataclass
class EvalTransform:
    """Deterministic validation/test transform."""

    output_size: int | Sequence[int] = 320
    mean: Sequence[float] = IMAGENET_MEAN
    std: Sequence[float] = IMAGENET_STD

    def __call__(self, image: Image.Image, mask: Image.Image):
        image = image.convert("RGB")
        mask = mask.convert("L")
        image, mask = _resize_pair(image, mask, _as_size(self.output_size))
        return _to_tensor_or_numpy(image, mask, self.mean, self.std)
