"""CelebAMask-HQ dataset reader for processed image/mask pairs."""

from __future__ import annotations

from pathlib import Path
from typing import Callable

from PIL import Image

try:
    from torch.utils.data import Dataset
except ImportError:
    class Dataset:  # type: ignore[no-redef]
        pass

from src.datasets.transforms import EvalTransform, MobileTrainTransform


IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png")


def read_split_file(split_path: str | Path) -> list[str]:
    path = Path(split_path)
    if not path.exists():
        raise FileNotFoundError(f"Split file does not exist: {path}")

    stems: list[str] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        value = line.strip()
        if value:
            stems.append(Path(value).stem)
    return stems


def resolve_image_path(image_dir: Path, stem: str) -> Path:
    for extension in IMAGE_EXTENSIONS:
        candidate = image_dir / f"{stem}{extension}"
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"No image found for stem '{stem}' in {image_dir}")


class CelebAMaskHQDataset(Dataset):
    """Read processed CelebAMask-HQ samples from split files.

    Expected processed structure:

    ```text
    processed_root/
      images/
        0.jpg
      masks/
        0.png
    ```
    """

    def __init__(
        self,
        image_dir: str | Path,
        mask_dir: str | Path,
        split_file: str | Path,
        transform: Callable | None = None,
        return_paths: bool = False,
    ) -> None:
        super().__init__()
        self.image_dir = Path(image_dir)
        self.mask_dir = Path(mask_dir)
        self.stems = read_split_file(split_file)
        self.transform = transform
        self.return_paths = return_paths

        if not self.image_dir.exists():
            raise FileNotFoundError(f"Image directory does not exist: {self.image_dir}")
        if not self.mask_dir.exists():
            raise FileNotFoundError(f"Mask directory does not exist: {self.mask_dir}")

    @classmethod
    def train(
        cls,
        image_dir: str | Path,
        mask_dir: str | Path,
        split_file: str | Path,
        output_size: int = 320,
        transform_kwargs: dict | None = None,
        **kwargs,
    ) -> "CelebAMaskHQDataset":
        transform_kwargs = transform_kwargs or {}
        return cls(
            image_dir=image_dir,
            mask_dir=mask_dir,
            split_file=split_file,
            transform=MobileTrainTransform(output_size=output_size, **transform_kwargs),
            **kwargs,
        )

    @classmethod
    def eval(
        cls,
        image_dir: str | Path,
        mask_dir: str | Path,
        split_file: str | Path,
        output_size: int = 320,
        **kwargs,
    ) -> "CelebAMaskHQDataset":
        return cls(
            image_dir=image_dir,
            mask_dir=mask_dir,
            split_file=split_file,
            transform=EvalTransform(output_size=output_size),
            **kwargs,
        )

    def __len__(self) -> int:
        return len(self.stems)

    def __getitem__(self, index: int):
        stem = self.stems[index]
        image_path = resolve_image_path(self.image_dir, stem)
        mask_path = self.mask_dir / f"{stem}.png"
        if not mask_path.exists():
            raise FileNotFoundError(f"Mask does not exist: {mask_path}")

        image = Image.open(image_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")

        if self.transform is not None:
            image, mask = self.transform(image, mask)

        sample = {
            "image": image,
            "mask": mask,
            "id": stem,
        }
        if self.return_paths:
            sample["image_path"] = str(image_path)
            sample["mask_path"] = str(mask_path)
        return sample
