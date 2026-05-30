import unittest

import numpy as np

from tools.tflite_smoke_test import decode_prediction, infer_image_layout


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


if __name__ == "__main__":
    unittest.main()
