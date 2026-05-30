import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from scripts.prepare_celebamask_hq import merge_attribute_masks
from src.deploy.palette import CLASS_NAMES


class PrepareCelebAMaskHQTests(unittest.TestCase):
    def test_merge_attribute_masks(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            anno_dir = root / "anno"
            folder = anno_dir / "0"
            folder.mkdir(parents=True)
            mask = np.zeros((512, 512), dtype=np.uint8)
            mask[10:20, 10:20] = 255
            Image.fromarray(mask).save(folder / "00000_skin.png")
            Image.fromarray(np.zeros((512, 512), dtype=np.uint8)).save(folder / "00000_teeth.png")

            output_path = root / "0.png"
            found = merge_attribute_masks(anno_dir, 0, output_path)
            merged = np.asarray(Image.open(output_path))

            self.assertEqual(found, 1)
            self.assertEqual(int(merged[15, 15]), 1)
            self.assertEqual(int(merged[0, 0]), 0)

    def test_merge_teeth_as_last_class(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            anno_dir = root / "anno"
            folder = anno_dir / "0"
            folder.mkdir(parents=True)
            teeth = np.zeros((512, 512), dtype=np.uint8)
            teeth[30:40, 30:40] = 255
            Image.fromarray(teeth).save(folder / "00000_teeth.png")

            output_path = root / "0.png"
            found = merge_attribute_masks(anno_dir, 0, output_path)
            merged = np.asarray(Image.open(output_path))

            self.assertEqual(found, 1)
            self.assertEqual(CLASS_NAMES[-1], "teeth")
            self.assertEqual(int(merged[35, 35]), CLASS_NAMES.index("teeth"))


if __name__ == "__main__":
    unittest.main()
