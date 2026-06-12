"""Class names and visualization palettes for head parsing datasets."""

from __future__ import annotations

from typing import Sequence


CLASS_NAMES_20 = [
    "background",
    "skin",
    "l_brow",
    "r_brow",
    "l_eye",
    "r_eye",
    "eye_g",
    "l_ear",
    "r_ear",
    "ear_r",
    "nose",
    "mouth",
    "u_lip",
    "l_lip",
    "neck",
    "neck_l",
    "cloth",
    "hair",
    "hat",
    "teeth",
]

PALETTE_20 = [
    [0, 0, 0],
    [255, 85, 0],
    [255, 170, 0],
    [255, 0, 85],
    [255, 0, 170],
    [0, 255, 0],
    [85, 255, 0],
    [170, 255, 0],
    [0, 255, 85],
    [0, 255, 170],
    [0, 0, 255],
    [85, 0, 255],
    [170, 0, 255],
    [0, 85, 255],
    [0, 170, 255],
    [255, 255, 0],
    [255, 255, 85],
    [255, 255, 170],
    [255, 0, 255],
    [0, 255, 255],
]

CLASS_NAMES_9 = [
    "background",
    "skin",
    "l_brow",
    "r_brow",
    "mouth",
    "u_lip",
    "l_lip",
    "teeth",
    "hair",
]

PALETTE_9 = [
    PALETTE_20[CLASS_NAMES_20.index(name)]
    for name in CLASS_NAMES_9
]

CLASS_SETS = {
    "mouth2teeth_20cls": (CLASS_NAMES_20, PALETTE_20),
    "20cls": (CLASS_NAMES_20, PALETTE_20),
    "mouth2teeth_9cls": (CLASS_NAMES_9, PALETTE_9),
    "9cls": (CLASS_NAMES_9, PALETTE_9),
}

# Backward-compatible defaults for the existing 20-class path.
CLASS_NAMES = CLASS_NAMES_20
PALETTE = PALETTE_20


def get_class_set(name: str | None = None) -> tuple[list[str], list[list[int]]]:
    if name is None:
        return list(CLASS_NAMES), [list(color) for color in PALETTE]
    key = str(name)
    if key not in CLASS_SETS:
        raise KeyError(f"Unknown class set: {name}")
    class_names, palette = CLASS_SETS[key]
    return list(class_names), [list(color) for color in palette]


def palette_for_class_names(class_names: Sequence[str]) -> list[list[int]]:
    color_by_name = {name: PALETTE_20[index] for index, name in enumerate(CLASS_NAMES_20)}
    fallback = [128, 128, 128]
    return [list(color_by_name.get(name, fallback)) for name in class_names]
