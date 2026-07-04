import unittest
from pathlib import Path

from scripts.train import build_train_transform_kwargs, resolve_model_state_dict
from src.utils.config import get_nested, load_yaml


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

    def test_resolve_model_state_dict_accepts_training_checkpoint(self):
        checkpoint = {
            "model": {
                "module.encoder.stem.0.weight": object(),
                "module.decoder.head.weight": object(),
            },
            "optimizer": {"state": {}},
        }

        state = resolve_model_state_dict(checkpoint)

        self.assertEqual(set(state), {"encoder.stem.0.weight", "decoder.head.weight"})

    def test_resolve_model_state_dict_accepts_raw_state_dict(self):
        class TensorLike:
            shape = (1,)

        state = resolve_model_state_dict({"layer.weight": TensorLike()})

        self.assertEqual(set(state), {"layer.weight"})

    def test_resolve_model_state_dict_rejects_unknown_checkpoint(self):
        with self.assertRaises(ValueError):
            resolve_model_state_dict({"optimizer": {"state": {}}})

    def test_shipped_256_finetune_config(self):
        cfg = load_yaml(Path("configs/train_mouth2teeth_9cls_finetune_256.yaml"))

        self.assertLessEqual(int(get_nested(cfg, "train.epochs")), 30)
        self.assertLess(float(get_nested(cfg, "train.optimizer.lr")), 0.01)
        self.assertEqual(int(get_nested(cfg, "train.scheduler.warmup_epochs")), 1)
        self.assertTrue(bool(get_nested(cfg, "train.losses.ohem")))
        self.assertEqual(get_nested(cfg, "train.augmentation.label_swaps"), {2: 3, 3: 2})


if __name__ == "__main__":
    unittest.main()
