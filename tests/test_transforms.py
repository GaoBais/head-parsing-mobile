import unittest

import numpy as np
from PIL import Image

from src.datasets.transforms import EvalTransform, MobileTrainTransform


class TransformTests(unittest.TestCase):
    def setUp(self):
        image = np.zeros((512, 512, 3), dtype=np.uint8)
        image[:, :, 0] = 128
        image[128:384, 128:384, 1] = 220
        mask = np.zeros((512, 512), dtype=np.uint8)
        mask[160:352, 160:352] = 1
        self.image = Image.fromarray(image)
        self.mask = Image.fromarray(mask)

    def test_eval_transform_shapes(self):
        transform = EvalTransform(output_size=320)
        image, mask = transform(self.image, self.mask)
        self.assertEqual(tuple(image.shape), (3, 320, 320))
        self.assertEqual(tuple(mask.shape), (320, 320))

    def test_mobile_train_transform_shapes(self):
        transform = MobileTrainTransform(output_size=256)
        image, mask = transform(self.image, self.mask)
        self.assertEqual(tuple(image.shape), (3, 256, 256))
        self.assertEqual(tuple(mask.shape), (256, 256))
        self.assertTrue(mask.max() <= 18)


if __name__ == "__main__":
    unittest.main()
