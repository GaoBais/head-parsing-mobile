import importlib.util
import unittest
from pathlib import Path


TORCH_AVAILABLE = importlib.util.find_spec("torch") is not None


@unittest.skipUnless(TORCH_AVAILABLE, "PyTorch is not installed in the local Windows dev environment.")
class TFLiteExportTests(unittest.TestCase):
    def test_tflite_export_config_defaults(self):
        from src.export.tflite import TFLiteExportConfig

        config = TFLiteExportConfig(checkpoint=Path("x.pt"), output=Path("x.tflite"))
        self.assertEqual(config.precision, "fp16")
        self.assertEqual(config.input_size, (320, 320))


if __name__ == "__main__":
    unittest.main()
