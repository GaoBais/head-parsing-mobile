"""Prepare CelebAMask-HQ for mobile head parsing training.

This script merges the original per-attribute CelebAMask-HQ masks into one
class-id PNG per image, copies source images into the processed directory, and
creates deterministic train/val/test split files.
"""

from __future__ import annotations

import argparse
import json
import random
import shutil
import sys
from pathlib import Path

import numpy as np
from PIL import Image

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable, **kwargs):
        return iterable

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.deploy.palette import CLASS_NAMES


ATTRIBUTES = CLASS_NAMES[1:]
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare CelebAMask-HQ class-id masks and splits.")
    parser.add_argument("--image-dir", type=Path, required=True, help="Directory containing CelebA-HQ images.")
    parser.add_argument(
        "--mask-anno-dir",
        type=Path,
        required=True,
        help="Directory containing original CelebAMask-HQ-mask-anno folders.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("data/processed/celebamask_hq"),
        help="Processed dataset output root.",
    )
    parser.add_argument("--split-dir", type=Path, default=Path("data/splits"), help="Split file output directory.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for split generation.")
    parser.add_argument("--train-ratio", type=float, default=0.9, help="Train split ratio.")
    parser.add_argument("--val-ratio", type=float, default=0.05, help="Validation split ratio.")
    parser.add_argument("--max-samples", type=int, default=None, help="Optional cap for debugging.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing processed masks/images.")
    return parser.parse_args()


def list_images(image_dir: Path) -> list[Path]:
    if not image_dir.exists():
        raise FileNotFoundError(f"Image directory does not exist: {image_dir}")

    image_paths = [p for p in image_dir.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS]
    return sorted(image_paths, key=lambda path: int(path.stem) if path.stem.isdigit() else path.stem)


def source_mask_path(mask_anno_dir: Path, image_id: int, attribute: str) -> Path:
    folder = str(image_id // 2000)
    return mask_anno_dir / folder / f"{image_id:05d}_{attribute}.png"


def merge_attribute_masks(mask_anno_dir: Path, image_id: int, output_path: Path) -> int:
    mask = np.zeros((512, 512), dtype=np.uint8)
    found = 0

    for class_id, attribute in enumerate(ATTRIBUTES, start=1):
        attr_path = source_mask_path(mask_anno_dir, image_id, attribute)
        if not attr_path.exists():
            continue

        attr_mask = np.array(Image.open(attr_path).convert("L"))
        positive = attr_mask > 0
        if not positive.any():
            continue

        mask[positive] = class_id
        found += 1

    Image.fromarray(mask).save(output_path)
    return found


def write_split(path: Path, stems: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(stems) + "\n", encoding="utf-8")


def create_splits(stems: list[str], split_dir: Path, train_ratio: float, val_ratio: float, seed: int) -> None:
    if train_ratio <= 0 or val_ratio < 0 or train_ratio + val_ratio >= 1:
        raise ValueError("Expected train_ratio > 0, val_ratio >= 0, and train_ratio + val_ratio < 1.")

    shuffled = list(stems)
    random.Random(seed).shuffle(shuffled)

    train_end = int(len(shuffled) * train_ratio)
    val_end = train_end + int(len(shuffled) * val_ratio)

    write_split(split_dir / "train.txt", shuffled[:train_end])
    write_split(split_dir / "val.txt", shuffled[train_end:val_end])
    write_split(split_dir / "test.txt", shuffled[val_end:])


def prepare_dataset(args: argparse.Namespace) -> None:
    if not args.mask_anno_dir.exists():
        raise FileNotFoundError(f"Mask annotation directory does not exist: {args.mask_anno_dir}")

    images_out = args.output_root / "images"
    masks_out = args.output_root / "masks"
    images_out.mkdir(parents=True, exist_ok=True)
    masks_out.mkdir(parents=True, exist_ok=True)
    args.split_dir.mkdir(parents=True, exist_ok=True)

    image_paths = list_images(args.image_dir)
    if args.max_samples is not None:
        image_paths = image_paths[: args.max_samples]

    processed_stems: list[str] = []
    missing_id_count = 0
    empty_mask_count = 0

    for image_path in tqdm(image_paths, desc="Preparing CelebAMask-HQ"):
        if not image_path.stem.isdigit():
            missing_id_count += 1
            continue

        image_id = int(image_path.stem)
        stem = image_path.stem
        image_dst = images_out / image_path.name
        mask_dst = masks_out / f"{stem}.png"

        if args.overwrite or not image_dst.exists():
            shutil.copy2(image_path, image_dst)

        if args.overwrite or not mask_dst.exists():
            found = merge_attribute_masks(args.mask_anno_dir, image_id, mask_dst)
        else:
            found = 1

        if found == 0:
            empty_mask_count += 1

        processed_stems.append(stem)

    create_splits(processed_stems, args.split_dir, args.train_ratio, args.val_ratio, args.seed)

    metadata = {
        "classes": CLASS_NAMES,
        "num_classes": len(CLASS_NAMES),
        "num_samples": len(processed_stems),
        "empty_mask_count": empty_mask_count,
        "skipped_non_numeric_images": missing_id_count,
        "image_dir": str(images_out),
        "mask_dir": str(masks_out),
        "split_dir": str(args.split_dir),
    }
    (args.output_root / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    print(f"Prepared {len(processed_stems)} samples under {args.output_root}")
    print(f"Empty masks: {empty_mask_count}; skipped non-numeric images: {missing_id_count}")


def main() -> None:
    prepare_dataset(parse_args())


if __name__ == "__main__":
    main()
