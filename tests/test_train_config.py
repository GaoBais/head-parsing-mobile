import unittest

from scripts.train import build_train_transform_kwargs


class TrainConfigTests(unittest.TestCase):
    def test_build_train_transform_kwargs(self):
        cfg = {
            "train": {
                "augmentation": {
                    "random_scale": [0.75, 1.25],
                    "random_rotate_deg": 15,
                    "random_translate_ratio": 0.1,
                    "random_letterbox": False,
                    "color_jitter": False,
                    "blur": True,
                    "jpeg": False,
                    "occlusion": True,
                    "label_swaps": {2: 3, 3: 2},
                }
            }
        }

        kwargs = build_train_transform_kwargs(cfg)

        self.assertEqual(kwargs["scale_range"], (0.75, 1.25))
        self.assertEqual(kwargs["rotate_degrees"], 15.0)
        self.assertEqual(kwargs["translate_ratio"], 0.1)
        self.assertEqual(kwargs["random_letterbox"], False)
        self.assertEqual(kwargs["color_jitter_prob"], 0.0)
        self.assertEqual(kwargs["blur_prob"], 0.2)
        self.assertEqual(kwargs["jpeg_prob"], 0.0)
        self.assertEqual(kwargs["occlusion_prob"], 0.25)
        self.assertEqual(kwargs["label_swaps"], {2: 3, 3: 2})


if __name__ == "__main__":
    unittest.main()
