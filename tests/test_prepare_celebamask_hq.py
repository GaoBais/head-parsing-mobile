import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from scripts.prepare_celebamask_hq import merge_attribute_masks


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

            output_path = root / "0.png"
            found = merge_attribute_masks(anno_dir, 0, output_path)
            merged = np.asarray(Image.open(output_path))

            self.assertEqual(found, 1)
            self.assertEqual(int(merged[15, 15]), 1)
            self.assertEqual(int(merged[0, 0]), 0)


if __name__ == "__main__":
    unittest.main()
