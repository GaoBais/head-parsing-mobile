"""Remap processed class-id segmentation masks to a smaller class set."""

from __future__ import annotations

import argparse
import json
import random
import shutil
import sys
from pathlib import Path
from typing import Iterable

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

from src.deploy.palette import CLASS_NAMES_9


IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png")
MOUTH2TEETH_9CLS_REMAP = {
    0: 0,
    1: 1,
    2: 2,
    3: 3,
    11: 4,
    12: 5,
    13: 6,
    19: 7,
    17: 8,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Remap 20-class mouth2teeth masks to a 9-class dataset.")
    parser.add_argument("--input-root", type=Path, required=True, help="Processed 20-class dataset root.")
    parser.add_argument("--output-root", type=Path, required=True, help="Output root for remapped dataset.")
    parser.add_argument("--split-dir", type=Path, required=True, help="Output split directory.")
    parser.add_argument(
        "--source-split-dir",
        type=Path,
        default=None,
        help="Optional split directory to copy train/val/test stems from.",
    )
    parser.add_argument("--mapping", choices=["mouth2teeth_9cls"], default="mouth2teeth_9cls")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-ratio", type=float, default=0.9)
    parser.add_argument("--val-ratio", type=float, default=0.05)
    parser.add_argument("--max-samples", type=int, default=None, help="Optional cap for smoke tests.")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def build_lookup(mapping: dict[int, int]) -> np.ndarray:
    lookup = np.zeros(256, dtype=np.uint8)
    for old_id, new_id in mapping.items():
        lookup[int(old_id)] = int(new_id)
    return lookup


def remap_mask_array(mask: np.ndarray, lookup: np.ndarray) -> np.ndarray:
    if mask.dtype != np.uint8:
        mask = mask.astype(np.uint8)
    return lookup[mask]


def list_images(image_dir: Path) -> list[Path]:
    if not image_dir.exists():
        raise FileNotFoundError(f"Image directory does not exist: {image_dir}")
    return sorted(
        [path for path in image_dir.iterdir() if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS],
        key=lambda path: int(path.stem) if path.stem.isdigit() else path.stem,
    )


def write_split(path: Path, stems: Iterable[str]) -> None:
    values = list(stems)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(values) + ("\n" if values else ""), encoding="utf-8")


def read_split(path: Path) -> list[str]:
    return [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def create_splits(stems: list[str], split_dir: Path, train_ratio: float, val_ratio: float, seed: int) -> dict[str, list[str]]:
    if train_ratio <= 0 or val_ratio < 0 or train_ratio + val_ratio >= 1:
        raise ValueError("Expected train_ratio > 0, val_ratio >= 0, and train_ratio + val_ratio < 1.")

    shuffled = list(stems)
    random.Random(seed).shuffle(shuffled)
    train_end = int(len(shuffled) * train_ratio)
    val_end = train_end + int(len(shuffled) * val_ratio)
    splits = {
        "train": shuffled[:train_end],
        "val": shuffled[train_end:val_end],
        "test": shuffled[val_end:],
    }
    for split, split_stems in splits.items():
        write_split(split_dir / f"{split}.txt", split_stems)
    return splits


def copy_splits(source_split_dir: Path, split_dir: Path, allowed_stems: set[str]) -> dict[str, list[str]]:
    splits: dict[str, list[str]] = {}
    for split in ("train", "val", "test"):
        source = source_split_dir / f"{split}.txt"
        if not source.exists():
            raise FileNotFoundError(f"Source split file does not exist: {source}")
        stems = [Path(value).stem for value in read_split(source)]
        stems = [stem for stem in stems if stem in allowed_stems]
        write_split(split_dir / f"{split}.txt", stems)
        splits[split] = stems
    return splits


def update_stats(mask: np.ndarray, stats: dict[str, list[int]], split: str | None = None) -> None:
    counts = np.bincount(mask.reshape(-1), minlength=len(CLASS_NAMES_9))
    for index, count in enumerate(counts[: len(CLASS_NAMES_9)]):
        if split is None:
            stats["pixels"][index] += int(count)
            if count > 0:
                stats["images"][index] += 1
        else:
            split_stats = stats["splits"][split]
            split_stats["pixels"][index] += int(count)
            if count > 0:
                split_stats["images"][index] += 1


def empty_stats() -> dict:
    return {
        "pixels": [0 for _ in CLASS_NAMES_9],
        "images": [0 for _ in CLASS_NAMES_9],
        "splits": {
            split: {
                "pixels": [0 for _ in CLASS_NAMES_9],
                "images": [0 for _ in CLASS_NAMES_9],
            }
            for split in ("train", "val", "test")
        },
    }


def class_stats_dict(stats: dict) -> dict:
    result = {}
    for index, name in enumerate(CLASS_NAMES_9):
        result[name] = {
            "pixels": int(stats["pixels"][index]),
            "images": int(stats["images"][index]),
        }
    return result


def split_stats_dict(stats: dict) -> dict:
    result = {}
    for split, split_stats in stats["splits"].items():
        result[split] = {}
        for index, name in enumerate(CLASS_NAMES_9):
            result[split][name] = {
                "pixels": int(split_stats["pixels"][index]),
                "images": int(split_stats["images"][index]),
            }
    return result


def prepare_remapped_dataset(args: argparse.Namespace) -> dict:
    input_images = args.input_root / "images"
    input_masks = args.input_root / "masks"
    output_images = args.output_root / "images"
    output_masks = args.output_root / "masks"
    output_images.mkdir(parents=True, exist_ok=True)
    output_masks.mkdir(parents=True, exist_ok=True)
    args.split_dir.mkdir(parents=True, exist_ok=True)

    if not input_masks.exists():
        raise FileNotFoundError(f"Mask directory does not exist: {input_masks}")

    image_paths = list_images(input_images)
    if args.max_samples is not None:
        image_paths = image_paths[: args.max_samples]

    lookup = build_lookup(MOUTH2TEETH_9CLS_REMAP)
    processed_stems: list[str] = []
    skipped_missing_masks = 0
    stats = empty_stats()

    for image_path in tqdm(image_paths, desc="Remapping masks"):
        stem = image_path.stem
        source_mask = input_masks / f"{stem}.png"
        if not source_mask.exists():
            skipped_missing_masks += 1
            continue

        image_dst = output_images / image_path.name
        mask_dst = output_masks / f"{stem}.png"

        if args.overwrite or not image_dst.exists():
            shutil.copy2(image_path, image_dst)

        if args.overwrite or not mask_dst.exists():
            source = np.asarray(Image.open(source_mask).convert("L"), dtype=np.uint8)
            remapped = remap_mask_array(source, lookup)
            Image.fromarray(remapped).save(mask_dst)
        else:
            remapped = np.asarray(Image.open(mask_dst).convert("L"), dtype=np.uint8)

        if int(remapped.max(initial=0)) >= len(CLASS_NAMES_9):
            raise ValueError(f"Remapped mask contains invalid class id: {mask_dst}")

        update_stats(remapped, stats)
        processed_stems.append(stem)

    allowed_stems = set(processed_stems)
    if args.source_split_dir is not None:
        splits = copy_splits(args.source_split_dir, args.split_dir, allowed_stems)
    else:
        splits = create_splits(processed_stems, args.split_dir, args.train_ratio, args.val_ratio, args.seed)

    stem_to_split = {stem: split for split, stems in splits.items() for stem in stems}
    for stem in processed_stems:
        split = stem_to_split.get(stem)
        if split is None:
            continue
        mask = np.asarray(Image.open(output_masks / f"{stem}.png").convert("L"), dtype=np.uint8)
        update_stats(mask, stats, split=split)

    metadata = {
        "mapping": args.mapping,
        "source_root": str(args.input_root),
        "classes": CLASS_NAMES_9,
        "num_classes": len(CLASS_NAMES_9),
        "num_samples": len(processed_stems),
        "skipped_missing_masks": skipped_missing_masks,
        "image_dir": str(output_images),
        "mask_dir": str(output_masks),
        "split_dir": str(args.split_dir),
        "remap": {str(old_id): new_id for old_id, new_id in MOUTH2TEETH_9CLS_REMAP.items()},
        "class_stats": class_stats_dict(stats),
        "split_stats": split_stats_dict(stats),
        "splits": {split: len(stems) for split, stems in splits.items()},
    }
    args.output_root.mkdir(parents=True, exist_ok=True)
    (args.output_root / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return metadata


def main() -> None:
    metadata = prepare_remapped_dataset(parse_args())
    print(json.dumps({key: metadata[key] for key in ("num_samples", "splits", "class_stats")}, indent=2))


if __name__ == "__main__":
    main()
