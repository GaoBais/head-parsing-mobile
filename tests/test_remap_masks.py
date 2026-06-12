import argparse
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from scripts.remap_masks import (
    MOUTH2TEETH_9CLS_REMAP,
    build_lookup,
    prepare_remapped_dataset,
    remap_mask_array,
)


class RemapMasksTests(unittest.TestCase):
    def test_remap_mask_array(self):
        source = np.array([[0, 1, 2, 3], [11, 12, 13, 17], [19, 4, 10, 18]], dtype=np.uint8)
        remapped = remap_mask_array(source, build_lookup(MOUTH2TEETH_9CLS_REMAP))

        expected = np.array([[0, 1, 2, 3], [4, 5, 6, 8], [7, 0, 0, 0]], dtype=np.uint8)
        np.testing.assert_array_equal(remapped, expected)

    def test_prepare_remapped_dataset(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            input_root = root / "input"
            output_root = root / "output"
            source_splits = root / "source_splits"
            split_dir = root / "splits"
            (input_root / "images").mkdir(parents=True)
            (input_root / "masks").mkdir(parents=True)
            source_splits.mkdir()

            for stem in ("00000", "00001", "00002"):
                image = np.zeros((4, 4, 3), dtype=np.uint8)
                Image.fromarray(image).save(input_root / "images" / f"{stem}.jpg")

            mask0 = np.array([[1, 2], [3, 19]], dtype=np.uint8)
            mask1 = np.array([[11, 12], [13, 17]], dtype=np.uint8)
            mask2 = np.array([[4, 5], [10, 18]], dtype=np.uint8)
            Image.fromarray(mask0).save(input_root / "masks" / "00000.png")
            Image.fromarray(mask1).save(input_root / "masks" / "00001.png")
            Image.fromarray(mask2).save(input_root / "masks" / "00002.png")

            (source_splits / "train.txt").write_text("00000\n", encoding="utf-8")
            (source_splits / "val.txt").write_text("00001\n", encoding="utf-8")
            (source_splits / "test.txt").write_text("00002\n", encoding="utf-8")

            metadata = prepare_remapped_dataset(
                argparse.Namespace(
                    input_root=input_root,
                    output_root=output_root,
                    split_dir=split_dir,
                    source_split_dir=source_splits,
                    mapping="mouth2teeth_9cls",
                    seed=42,
                    train_ratio=0.9,
                    val_ratio=0.05,
                    max_samples=None,
                    overwrite=True,
                )
            )

            self.assertEqual(metadata["num_classes"], 9)
            self.assertEqual(metadata["num_samples"], 3)
            self.assertEqual(metadata["splits"], {"train": 1, "val": 1, "test": 1})
            remapped = np.asarray(Image.open(output_root / "masks" / "00002.png").convert("L"))
            self.assertEqual(int(remapped.max()), 0)
            self.assertEqual(metadata["class_stats"]["teeth"]["images"], 1)
            self.assertEqual(metadata["class_stats"]["hair"]["images"], 1)


if __name__ == "__main__":
    unittest.main()
