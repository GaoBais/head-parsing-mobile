import importlib.util
import unittest


TORCH_AVAILABLE = importlib.util.find_spec("torch") is not None


class OnnxGraphCheckTests(unittest.TestCase):
    def test_allowed_op_set_contains_expected_mobile_ops(self):
        from src.export.graph_check import MOBILE_FRIENDLY_ONNX_OPS

        self.assertIn("Conv", MOBILE_FRIENDLY_ONNX_OPS)
        self.assertIn("Resize", MOBILE_FRIENDLY_ONNX_OPS)
        self.assertIn("Concat", MOBILE_FRIENDLY_ONNX_OPS)


@unittest.skipUnless(TORCH_AVAILABLE, "PyTorch is not installed in the local Windows dev environment.")
class OnnxExportTests(unittest.TestCase):
    def test_export_config_dataclass(self):
        from pathlib import Path

        from src.export.onnx import OnnxExportConfig

        config = OnnxExportConfig(checkpoint=Path("x.pt"), output=Path("x.onnx"))
        self.assertEqual(config.input_size, (320, 320))
        self.assertEqual(config.opset, 17)


if __name__ == "__main__":
    unittest.main()
