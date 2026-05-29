import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from src.datasets import CelebAMaskHQDataset, EvalTransform


class DatasetTests(unittest.TestCase):
    def test_dataset_reads_split_and_sample(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            image_dir = root / "images"
            mask_dir = root / "masks"
            split_dir = root / "splits"
            image_dir.mkdir()
            mask_dir.mkdir()
            split_dir.mkdir()

            image = np.zeros((64, 64, 3), dtype=np.uint8)
            image[:, :, 0] = 128
            mask = np.zeros((64, 64), dtype=np.uint8)
            mask[16:48, 16:48] = 1

            Image.fromarray(image).save(image_dir / "00000.jpg")
            Image.fromarray(mask).save(mask_dir / "00000.png")
            (split_dir / "train.txt").write_text("00000\n", encoding="utf-8")

            dataset = CelebAMaskHQDataset(
                image_dir=image_dir,
                mask_dir=mask_dir,
                split_file=split_dir / "train.txt",
                transform=EvalTransform(output_size=32),
                return_paths=True,
            )
            sample = dataset[0]

            self.assertEqual(len(dataset), 1)
            self.assertEqual(sample["id"], "00000")
            self.assertEqual(tuple(sample["image"].shape), (3, 32, 32))
            self.assertEqual(tuple(sample["mask"].shape), (32, 32))
            self.assertIn("image_path", sample)

    def test_train_factory_passes_transform_kwargs(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            image_dir = root / "images"
            mask_dir = root / "masks"
            split_dir = root / "splits"
            image_dir.mkdir()
            mask_dir.mkdir()
            split_dir.mkdir()
            Image.fromarray(np.zeros((64, 64, 3), dtype=np.uint8)).save(image_dir / "00000.jpg")
            Image.fromarray(np.zeros((64, 64), dtype=np.uint8)).save(mask_dir / "00000.png")
            (split_dir / "train.txt").write_text("00000\n", encoding="utf-8")

            dataset = CelebAMaskHQDataset.train(
                image_dir=image_dir,
                mask_dir=mask_dir,
                split_file=split_dir / "train.txt",
                output_size=32,
                transform_kwargs={
                    "scale_range": (1.0, 1.0),
                    "rotate_degrees": 0.0,
                    "translate_ratio": 0.0,
                    "random_letterbox": False,
                },
            )

            self.assertEqual(dataset.transform.scale_range, (1.0, 1.0))
            self.assertEqual(dataset.transform.rotate_degrees, 0.0)
            self.assertEqual(dataset.transform.translate_ratio, 0.0)
            self.assertEqual(dataset.transform.random_letterbox, False)


if __name__ == "__main__":
    unittest.main()
