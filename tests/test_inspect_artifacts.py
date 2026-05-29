import tempfile
import unittest
from pathlib import Path

from tools.inspect_artifacts import inspect_artifact


class InspectArtifactsTests(unittest.TestCase):
    def test_inspect_file_artifact(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "model.tflite"
            path.write_bytes(b"abc")

            info = inspect_artifact(path)

            self.assertEqual(info["type"], "tflite")
            self.assertEqual(info["size_bytes"], 3)
            self.assertEqual(len(info["sha256"]), 64)

    def test_inspect_directory_artifact(self):
        with tempfile.TemporaryDirectory() as tmp:
            package = Path(tmp) / "Model.mlpackage"
            package.mkdir()
            (package / "manifest.json").write_text("{}", encoding="utf-8")

            info = inspect_artifact(package)

            self.assertEqual(info["type"], "coreml")
            self.assertTrue(info["is_directory"])
            self.assertGreater(info["size_bytes"], 0)


if __name__ == "__main__":
    unittest.main()
