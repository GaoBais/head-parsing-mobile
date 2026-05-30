import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from src.utils.visualization import colorize_mask, make_prediction_grid, mask_class_histogram


class VisualizationTests(unittest.TestCase):
    def test_colorize_and_grid(self):
        image = np.zeros((32, 32, 3), dtype=np.uint8)
        image[:, :, 0] = 120
        mask = np.zeros((32, 32), dtype=np.uint8)
        mask[8:24, 8:24] = 1

        color = colorize_mask(mask)
        grid = make_prediction_grid(image, mask, mask)
        histogram = mask_class_histogram(mask)

        self.assertEqual(color.size, (32, 32))
        self.assertEqual(grid.size[1], 56)
        self.assertEqual(histogram["skin"], 16 * 16)

    def test_tool_output_inputs(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            image = np.zeros((16, 16, 3), dtype=np.uint8)
            mask = np.zeros((16, 16), dtype=np.uint8)
            mask[4:12, 4:12] = 19
            image_path = root / "image.png"
            mask_path = root / "mask.png"
            Image.fromarray(image).save(image_path)
            Image.fromarray(mask).save(mask_path)

            grid = make_prediction_grid(image_path, mask_path)
            histogram = mask_class_histogram(mask_path)
            self.assertEqual(grid.size[0], 48)
            self.assertEqual(histogram["teeth"], 8 * 8)


if __name__ == "__main__":
    unittest.main()
