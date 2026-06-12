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

    def test_9class_label_swaps_only_swap_brows(self):
        image = Image.fromarray(np.zeros((4, 4, 3), dtype=np.uint8))
        mask = np.array(
            [
                [2, 3, 4, 5],
                [6, 7, 8, 0],
                [2, 3, 7, 8],
                [4, 5, 6, 0],
            ],
            dtype=np.uint8,
        )
        transform = MobileTrainTransform(
            output_size=4,
            scale_range=(1.0, 1.0),
            rotate_degrees=0,
            horizontal_flip_prob=1.0,
            color_jitter_prob=0,
            blur_prob=0,
            jpeg_prob=0,
            occlusion_prob=0,
            random_letterbox=False,
            label_swaps={2: 3, 3: 2},
        )

        _, transformed = transform(image, Image.fromarray(mask))

        expected = np.array(
            [
                [5, 4, 2, 3],
                [0, 8, 7, 6],
                [8, 7, 2, 3],
                [0, 6, 5, 4],
            ],
            dtype=np.int64,
        )
        np.testing.assert_array_equal(transformed, expected)


if __name__ == "__main__":
    unittest.main()
