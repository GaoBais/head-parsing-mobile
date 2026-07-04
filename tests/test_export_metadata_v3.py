import copy
import json
import unittest
from pathlib import Path


BASE_METADATA_PATH = Path(__file__).resolve().parents[1] / "configs" / "mobile_metadata_mouth2teeth_9cls.json"


class MetadataV3Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.base = json.loads(BASE_METADATA_PATH.read_text(encoding="utf-8"))

    def test_tflite_label_map_metadata(self):
        from src.export.metadata import build_v3_metadata

        meta = build_v3_metadata(
            self.base,
            model_name="HeadParsingMobile256Mouth2Teeth9ClsV3",
            input_size=(256, 256),
            framework="tflite",
            output_kind="label_map",
        )
        # Flat keys the SDK reads to drive pre/post-processing.
        self.assertEqual(meta["io_schema"], "v3")
        self.assertEqual(meta["input_size"], [256, 256])
        self.assertEqual(meta["input_layout"], "NHWC")
        self.assertEqual(meta["input_dtype"], "uint8")
        self.assertEqual(meta["output_kind"], "label_map")

        self.assertEqual(meta["input"]["shape"], [1, 256, 256, 3])
        self.assertTrue(meta["input"]["normalization"]["in_graph"])
        self.assertEqual(meta["input"]["normalization"]["mean"], [0.485, 0.456, 0.406])
        self.assertEqual(meta["input"]["normalization"]["std"], [0.229, 0.224, 0.225])

        self.assertEqual(meta["output"]["name"], "label_map")
        self.assertEqual(meta["output"]["shape"], [1, 256, 256])
        self.assertEqual(meta["output"]["dtype"], "uint8")

        self.assertEqual(meta["classes"], self.base["classes"])
        self.assertEqual(meta["model_name"], "HeadParsingMobile256Mouth2Teeth9ClsV3")

    def test_tflite_logits_metadata_is_channel_last(self):
        from src.export.metadata import build_v3_metadata

        meta = build_v3_metadata(
            self.base,
            model_name="X",
            input_size=(256, 256),
            framework="tflite",
            output_kind="logits",
        )
        self.assertEqual(meta["output_kind"], "logits")
        self.assertEqual(meta["output"]["name"], "logits")
        self.assertEqual(meta["output"]["shape"], [1, 256, 256, 9])
        self.assertEqual(meta["output"]["layout"], "NHWC")
        self.assertEqual(meta["output"]["dtype"], "float32")

    def test_coreml_label_map_metadata(self):
        from src.export.metadata import build_v3_metadata

        meta = build_v3_metadata(
            self.base,
            model_name="X",
            input_size=(256, 256),
            framework="coreml",
            output_kind="label_map",
        )
        self.assertEqual(meta["input_layout"], "image")
        self.assertEqual(meta["input"]["shape"], [1, 3, 256, 256])
        self.assertEqual(meta["output"]["dtype"], "int32")

    def test_base_metadata_is_not_mutated(self):
        from src.export.metadata import build_v3_metadata

        snapshot = copy.deepcopy(self.base)
        build_v3_metadata(self.base, model_name="X", input_size=(256, 256), framework="tflite", output_kind="label_map")
        self.assertEqual(self.base, snapshot)

    def test_rejects_bad_inputs(self):
        from src.export.metadata import build_v3_metadata

        with self.assertRaises(ValueError):
            build_v3_metadata({}, model_name="X", input_size=(256, 256), framework="tflite", output_kind="label_map")
        with self.assertRaises(ValueError):
            build_v3_metadata(self.base, model_name="X", input_size=(256, 256), framework="onnx", output_kind="label_map")
        with self.assertRaises(ValueError):
            build_v3_metadata(self.base, model_name="X", input_size=(256, 256), framework="tflite", output_kind="softmax")


if __name__ == "__main__":
    unittest.main()
