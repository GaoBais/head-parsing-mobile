import tempfile
import unittest
from pathlib import Path

from scripts.evaluate import resolve_class_names, save_metrics


class EvaluateConfigTests(unittest.TestCase):
    def test_resolve_9class_names_from_dataset_config(self):
        dataset_cfg = {
            "dataset": {
                "classes": [
                    "background",
                    "skin",
                    "l_brow",
                    "r_brow",
                    "mouth",
                    "u_lip",
                    "l_lip",
                    "teeth",
                    "hair",
                ]
            }
        }

        class_names = resolve_class_names(dataset_cfg, num_classes=9)

        self.assertEqual(len(class_names), 9)
        self.assertEqual(class_names[7], "teeth")

    def test_save_metrics_uses_configured_class_names(self):
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "metrics.json"
            save_metrics(
                {"per_class_iou": [0.0, 0.1, 0.2]},
                output,
                ["background", "skin", "l_brow"],
            )

            text = output.read_text(encoding="utf-8")

        self.assertIn('"skin"', text)
        self.assertIn('"l_brow"', text)


if __name__ == "__main__":
    unittest.main()
