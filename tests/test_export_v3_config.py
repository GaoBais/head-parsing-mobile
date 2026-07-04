import unittest
from pathlib import Path

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]


class V3ArtifactSpecTests(unittest.TestCase):
    def _cfg(self, artifacts):
        return {"export_v3": {"checkpoint": "outputs/train/best.pt", "artifacts": artifacts}}

    def test_load_specs_from_config(self):
        from scripts.export_mobile_v3 import load_v3_artifact_specs

        specs = load_v3_artifact_specs(
            self._cfg(
                [
                    {"format": "tflite", "output_kind": "label_map", "output": "weights/a.tflite"},
                    {"format": "coreml", "output_kind": "logits", "output": "weights/B.mlpackage"},
                ]
            )
        )
        self.assertEqual(len(specs), 2)
        self.assertEqual(specs[0].format, "tflite")
        self.assertEqual(specs[0].output_kind, "label_map")
        self.assertEqual(specs[1].format, "coreml")
        self.assertEqual(Path(specs[1].output).name, "B.mlpackage")

    def test_rejects_bad_format_kind_duplicates_and_empty(self):
        from scripts.export_mobile_v3 import load_v3_artifact_specs

        with self.assertRaises(ValueError):
            load_v3_artifact_specs(self._cfg([{"format": "onnx", "output_kind": "label_map", "output": "x"}]))
        with self.assertRaises(ValueError):
            load_v3_artifact_specs(self._cfg([{"format": "tflite", "output_kind": "softmax", "output": "x"}]))
        with self.assertRaises(ValueError):
            load_v3_artifact_specs(
                self._cfg(
                    [
                        {"format": "tflite", "output_kind": "label_map", "output": "same.tflite"},
                        {"format": "tflite", "output_kind": "logits", "output": "same.tflite"},
                    ]
                )
            )
        with self.assertRaises(ValueError):
            load_v3_artifact_specs(self._cfg([]))

    def test_filter_specs_by_format(self):
        from scripts.export_mobile_v3 import filter_specs, load_v3_artifact_specs

        specs = load_v3_artifact_specs(
            self._cfg(
                [
                    {"format": "tflite", "output_kind": "label_map", "output": "a.tflite"},
                    {"format": "coreml", "output_kind": "label_map", "output": "b.mlpackage"},
                ]
            )
        )
        only_tflite = filter_specs(specs, {"tflite"})
        self.assertEqual([spec.format for spec in only_tflite], ["tflite"])

    def test_metadata_path_sits_next_to_artifact(self):
        from scripts.export_mobile_v3 import metadata_path_for

        self.assertEqual(
            metadata_path_for(Path("weights/x_fp16.tflite")),
            Path("weights/x_fp16_metadata.json"),
        )
        self.assertEqual(
            metadata_path_for(Path("weights/Model.mlpackage")),
            Path("weights/Model_metadata.json"),
        )


class ShippedV3ConfigTests(unittest.TestCase):
    def test_shipped_export_config_has_four_artifacts(self):
        from scripts.export_mobile_v3 import load_v3_artifact_specs

        cfg = yaml.safe_load((PROJECT_ROOT / "configs" / "export_mobile_v3.yaml").read_text(encoding="utf-8"))
        specs = load_v3_artifact_specs(cfg)
        self.assertEqual(len(specs), 4)
        combos = {(spec.format, spec.output_kind) for spec in specs}
        self.assertEqual(
            combos,
            {
                ("tflite", "label_map"),
                ("tflite", "logits"),
                ("coreml", "label_map"),
                ("coreml", "logits"),
            },
        )
        self.assertEqual(cfg["export_v3"]["input_size"], [256, 256])
        self.assertTrue(cfg["export_v3"]["base_metadata"].endswith("mobile_metadata_mouth2teeth_9cls.json"))

    def test_shipped_256_model_config(self):
        cfg = yaml.safe_load(
            (PROJECT_ROOT / "configs" / "model_mobilev2_lraspp_256_9cls.yaml").read_text(encoding="utf-8")
        )
        self.assertEqual(cfg["model"]["input_size"], [256, 256])
        self.assertEqual(cfg["model"]["num_classes"], 9)
        self.assertEqual(cfg["model"]["encoder"]["width_mult"], 1.0)


if __name__ == "__main__":
    unittest.main()
