import unittest
import tempfile
from pathlib import Path

import numpy as np

from tools.tflite_smoke_test import decode_prediction, infer_image_layout, load_mobile_metadata


class TFLiteSmokeTests(unittest.TestCase):
    def test_infer_image_layout(self):
        self.assertEqual(infer_image_layout([1, 3, 320, 320]), ("NCHW", 320, 320))
        self.assertEqual(infer_image_layout([1, 320, 320, 3]), ("NHWC", 320, 320))

    def test_decode_prediction_nchw(self):
        output = np.zeros((1, 20, 4, 4), dtype=np.float32)
        output[:, 19, 1:3, 1:3] = 10.0
        mask, layout = decode_prediction(output)

        self.assertEqual(layout, "NCHW")
        self.assertEqual(mask.dtype, np.uint8)
        self.assertEqual(int((mask == 19).sum()), 4)

    def test_decode_prediction_nhwc(self):
        output = np.zeros((1, 4, 4, 20), dtype=np.float32)
        output[:, 1:3, 1:3, 19] = 10.0
        mask, layout = decode_prediction(output)

        self.assertEqual(layout, "NHWC")
        self.assertEqual(int((mask == 19).sum()), 4)

    def test_decode_prediction_9class(self):
        output = np.zeros((1, 9, 4, 4), dtype=np.float32)
        output[:, 7, 1:3, 1:3] = 10.0
        mask, layout = decode_prediction(output, num_classes=9)

        self.assertEqual(layout, "NCHW")
        self.assertEqual(int((mask == 7).sum()), 4)

    def test_load_mobile_metadata_9class(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "metadata.json"
            path.write_text(
                """{
  "classes": [
    {"id": 0, "name": "background", "color": [0, 0, 0]},
    {"id": 1, "name": "skin", "color": [255, 85, 0]},
    {"id": 2, "name": "l_brow", "color": [255, 170, 0]},
    {"id": 3, "name": "r_brow", "color": [255, 0, 85]},
    {"id": 4, "name": "mouth", "color": [85, 0, 255]},
    {"id": 5, "name": "u_lip", "color": [170, 0, 255]},
    {"id": 6, "name": "l_lip", "color": [0, 85, 255]},
    {"id": 7, "name": "teeth", "color": [0, 255, 255]},
    {"id": 8, "name": "hair", "color": [255, 255, 170]}
  ]
}""",
                encoding="utf-8",
            )

            class_names, palette, metadata = load_mobile_metadata(path)

        self.assertIsNotNone(metadata)
        self.assertEqual(len(class_names), 9)
        self.assertEqual(class_names[7], "teeth")
        self.assertEqual(palette[8], [255, 255, 170])


if __name__ == "__main__":
    unittest.main()
