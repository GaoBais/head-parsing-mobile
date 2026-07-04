import importlib.util
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace


TORCH_AVAILABLE = importlib.util.find_spec("torch") is not None


def _make_fake_ct():
    """Fake coremltools mirroring the convert(traced, ...) call surface."""
    import torch

    class FakeImageType:
        def __init__(self, name=None, shape=None, scale=None, bias=None, color_layout=None):
            self.name = name
            self.shape = shape
            self.scale = scale
            self.bias = bias
            self.color_layout = color_layout

    class FakeTensorType:
        def __init__(self, name=None, dtype=None):
            self.name = name
            self.dtype = dtype

    class FakeMLModel:
        def __init__(self, result):
            self.result = result

        def save(self, path):
            Path(path).write_bytes(b"fake-mlpackage")

    class FakeCT:
        ImageType = FakeImageType
        TensorType = FakeTensorType
        colorlayout = SimpleNamespace(RGB="rgb-layout", BGR="bgr-layout")
        target = SimpleNamespace(iOS15="t15", iOS16="t16", iOS17="t17", iOS18="t18")
        precision = SimpleNamespace(FLOAT16="fp16", FLOAT32="fp32")

        def __init__(self):
            self.convert_kwargs = None
            self.result = None

        def convert(self, traced, inputs=None, outputs=None, convert_to=None,
                    minimum_deployment_target=None, compute_precision=None):
            self.convert_kwargs = {
                "inputs": inputs,
                "outputs": outputs,
                "convert_to": convert_to,
                "minimum_deployment_target": minimum_deployment_target,
                "compute_precision": compute_precision,
            }
            shape = inputs[0].shape
            with torch.no_grad():
                self.result = traced(torch.rand(*shape))
            return FakeMLModel(self.result)

    return FakeCT()


@unittest.skipUnless(TORCH_AVAILABLE, "PyTorch is not installed in the local dev environment.")
class CoreMLV3ExportTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import torch

        from src.models import HeadParsingMobile

        torch.manual_seed(7)
        cls.tmp = tempfile.TemporaryDirectory()
        cls.checkpoint = Path(cls.tmp.name) / "ckpt.pt"
        model = HeadParsingMobile(num_classes=9, width_mult=0.5)
        torch.save({"model": model.state_dict()}, cls.checkpoint)

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def _config(self, output_kind, name, color_layout="RGB"):
        from src.export.coreml import CoreMLV3ExportConfig

        return CoreMLV3ExportConfig(
            checkpoint=self.checkpoint,
            output=Path(self.tmp.name) / name,
            input_size=(64, 64),
            num_classes=9,
            width_mult=0.5,
            output_kind=output_kind,
            color_layout=color_layout,
        )

    def test_label_map_variant_image_input_and_int32_output(self):
        import numpy as np
        import torch

        from src.datasets.transforms import IMAGENET_MEAN
        from src.export.coreml import export_coreml_v3

        ct = _make_fake_ct()
        out = export_coreml_v3(self._config("label_map", "v3.mlpackage"), ct=ct)

        self.assertTrue(out.exists())
        image_input = ct.convert_kwargs["inputs"][0]
        self.assertEqual(image_input.name, "image")
        self.assertEqual(tuple(image_input.shape), (1, 3, 64, 64))
        self.assertAlmostEqual(image_input.scale, 1.0 / 255.0)
        for got, mean in zip(image_input.bias, IMAGENET_MEAN):
            self.assertAlmostEqual(got, -mean)
        self.assertEqual(image_input.color_layout, "rgb-layout")

        output = ct.convert_kwargs["outputs"][0]
        self.assertEqual(output.name, "label_map")
        self.assertEqual(output.dtype, np.int32)
        self.assertEqual(ct.convert_kwargs["convert_to"], "mlprogram")
        self.assertEqual(ct.convert_kwargs["minimum_deployment_target"], "t15")
        self.assertEqual(ct.convert_kwargs["compute_precision"], "fp16")

        self.assertEqual(tuple(ct.result.shape), (1, 64, 64))
        self.assertEqual(ct.result.dtype, torch.int32)

    def test_logits_variant_keeps_float_logits(self):
        import torch

        from src.export.coreml import export_coreml_v3

        ct = _make_fake_ct()
        export_coreml_v3(self._config("logits", "v3logits.mlpackage"), ct=ct)

        output = ct.convert_kwargs["outputs"][0]
        self.assertEqual(output.name, "logits")
        self.assertIsNone(output.dtype)
        self.assertEqual(tuple(ct.result.shape), (1, 9, 64, 64))
        self.assertEqual(ct.result.dtype, torch.float32)

    def test_bgr_layout_reverses_bias_order(self):
        from src.datasets.transforms import IMAGENET_MEAN
        from src.export.coreml import export_coreml_v3

        ct = _make_fake_ct()
        export_coreml_v3(self._config("label_map", "bgr.mlpackage", color_layout="BGR"), ct=ct)

        image_input = ct.convert_kwargs["inputs"][0]
        self.assertEqual(image_input.color_layout, "bgr-layout")
        for got, mean in zip(image_input.bias, reversed(IMAGENET_MEAN)):
            self.assertAlmostEqual(got, -mean)

    def test_rejects_unknown_output_kind_and_color_layout(self):
        from src.export.coreml import export_coreml_v3

        with self.assertRaises(ValueError):
            export_coreml_v3(self._config("softmax", "bad.mlpackage"), ct=_make_fake_ct())
        with self.assertRaises(ValueError):
            export_coreml_v3(self._config("label_map", "bad2.mlpackage", color_layout="RGBA"), ct=_make_fake_ct())


if __name__ == "__main__":
    unittest.main()
