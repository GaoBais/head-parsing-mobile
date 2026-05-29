import importlib.util
import unittest
from pathlib import Path


TORCH_AVAILABLE = importlib.util.find_spec("torch") is not None
COREMLTOOLS_AVAILABLE = importlib.util.find_spec("coremltools") is not None


@unittest.skipUnless(TORCH_AVAILABLE, "PyTorch is not installed in the local Windows dev environment.")
class CoreMLExportConfigTests(unittest.TestCase):
    def test_coreml_export_config_defaults(self):
        from src.export.coreml import CoreMLExportConfig

        config = CoreMLExportConfig(checkpoint=Path("x.pt"), output=Path("x.mlpackage"))
        self.assertEqual(config.precision, "fp16")
        self.assertEqual(config.minimum_deployment_target, "ios15")


@unittest.skipUnless(
    TORCH_AVAILABLE and COREMLTOOLS_AVAILABLE,
    "PyTorch/coremltools are not installed in the local Windows dev environment.",
)
class CoreMLTargetTests(unittest.TestCase):
    def test_coreml_target(self):
        from src.export.coreml import coreml_target, import_coremltools

        ct = import_coremltools()
        self.assertEqual(coreml_target(ct, "ios15"), ct.target.iOS15)


if __name__ == "__main__":
    unittest.main()
