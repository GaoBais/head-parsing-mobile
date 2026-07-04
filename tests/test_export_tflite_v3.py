import importlib.util
import sys
import tempfile
import types
import unittest
from pathlib import Path


TORCH_AVAILABLE = importlib.util.find_spec("torch") is not None


def _make_fakes():
    """Fake LiteRT converter mirroring the real to_channel_last_io/convert semantics."""
    import torch
    from torch import nn

    class ChannelLastShim(nn.Module):
        def __init__(self, inner, args, outputs):
            super().__init__()
            self.inner = inner
            self.args = args
            self.outputs = outputs

        def forward(self, x):
            if 0 in self.args:
                x = x.permute(0, 3, 1, 2)
            out = self.inner(x)
            if 0 in self.outputs:
                out = out.permute(0, 2, 3, 1)
            return out

    class FakeEdgeModel:
        def __init__(self, result):
            self.result = result

        def export(self, path):
            Path(path).write_bytes(b"fake-tflite")

    class FakeConverter:
        def __init__(self):
            self.channel_last_calls = []
            self.sample_args = None
            self.result = None

        def to_channel_last_io(self, module, args=None, outputs=None):
            self.channel_last_calls.append({"args": args, "outputs": outputs})
            return ChannelLastShim(module, args or [], outputs or [])

        def convert(self, module, sample_args):
            self.sample_args = sample_args
            with torch.no_grad():
                self.result = module(*sample_args)
            return FakeEdgeModel(self.result)

    return FakeConverter()


@unittest.skipUnless(TORCH_AVAILABLE, "PyTorch is not installed in the local dev environment.")
class TFLiteV3ExportTests(unittest.TestCase):
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

    def _config(self, output_kind, name):
        from src.export.tflite import TFLiteV3ExportConfig

        return TFLiteV3ExportConfig(
            checkpoint=self.checkpoint,
            output=Path(self.tmp.name) / name,
            input_size=(64, 64),
            num_classes=9,
            width_mult=0.5,
            output_kind=output_kind,
        )

    def test_label_map_variant_uint8_nhwc_input_and_hw_output(self):
        import torch

        from src.export.tflite import export_tflite_v3

        converter = _make_fakes()
        config = self._config("label_map", "labelmap.tflite")
        out = export_tflite_v3(config, converter_module=converter)

        self.assertTrue(out.exists())
        sample = converter.sample_args[0]
        self.assertEqual(sample.dtype, torch.uint8)
        self.assertEqual(tuple(sample.shape), (1, 64, 64, 3))
        # Label map has no channel axis: only the input is converted to channel-last.
        self.assertEqual(converter.channel_last_calls, [{"args": [0], "outputs": None}])
        self.assertEqual(tuple(converter.result.shape), (1, 64, 64))
        self.assertEqual(converter.result.dtype, torch.uint8)

    def test_logits_variant_converts_output_to_channel_last(self):
        import torch

        from src.export.tflite import export_tflite_v3

        converter = _make_fakes()
        config = self._config("logits", "logits.tflite")
        export_tflite_v3(config, converter_module=converter)

        self.assertEqual(converter.channel_last_calls, [{"args": [0], "outputs": [0]}])
        self.assertEqual(tuple(converter.result.shape), (1, 64, 64, 9))
        self.assertEqual(converter.result.dtype, torch.float32)

    def test_rejects_unknown_output_kind(self):
        from src.export.tflite import export_tflite_v3

        with self.assertRaises(ValueError):
            export_tflite_v3(self._config("softmax", "bad.tflite"), converter_module=_make_fakes())

    def test_requires_channel_last_capable_converter(self):
        from src.export.tflite import export_tflite_v3

        class LegacyConverter:
            def convert(self, module, sample_args):  # pragma: no cover - should not be reached
                raise AssertionError("convert must not be called")

        with self.assertRaises(RuntimeError):
            export_tflite_v3(self._config("label_map", "x.tflite"), converter_module=LegacyConverter())

    def test_patches_litert_uint8_dtype_mapping(self):
        import torch

        from src.export.tflite import _patch_litert_uint8_dtype_support

        module_names = [
            "litert_converter",
            "litert_converter.mlir",
            "litert_converter.mlir.ir",
            "litert_torch",
            "litert_torch.backend",
            "litert_torch.backend.lowerings",
            "litert_torch.backend.lowerings.utils",
            "litert_torch.backend.export_utils",
        ]
        original_modules = {name: sys.modules.get(name) for name in module_names}

        try:
            litert_converter = types.ModuleType("litert_converter")
            mlir = types.ModuleType("litert_converter.mlir")
            ir = types.ModuleType("litert_converter.mlir.ir")

            class IntegerType:
                def __init__(self, bits, kind):
                    self.width = bits
                    self.kind = kind
                    self.is_unsigned = kind == "unsigned"
                    self.is_signless = kind == "signless"

                @staticmethod
                def get_unsigned(bits):
                    return IntegerType(bits, "unsigned")

                @staticmethod
                def get_signless(bits):
                    return IntegerType(bits, "signless")

                def __eq__(self, other):
                    return isinstance(other, IntegerType) and (self.width, self.kind) == (other.width, other.kind)

                def __repr__(self):
                    return f"{self.kind}{self.width}"

            ir.IntegerType = IntegerType
            mlir.ir = ir

            litert_torch = types.ModuleType("litert_torch")
            backend = types.ModuleType("litert_torch.backend")
            lowerings = types.ModuleType("litert_torch.backend.lowerings")
            utils = types.ModuleType("litert_torch.backend.lowerings.utils")
            export_utils = types.ModuleType("litert_torch.backend.export_utils")

            def missing_uint8(dtype):
                raise KeyError(dtype)

            def unsupported_ir(ty):
                raise RuntimeError(f"Unsupported ir element type: {ty}")

            utils.torch_dtype_to_ir_element_type = missing_uint8
            export_utils.torch_dtype_to_ir_element_type = missing_uint8
            export_utils.ir_element_type_to_torch_dtype = unsupported_ir

            sys.modules.update(
                {
                    "litert_converter": litert_converter,
                    "litert_converter.mlir": mlir,
                    "litert_converter.mlir.ir": ir,
                    "litert_torch": litert_torch,
                    "litert_torch.backend": backend,
                    "litert_torch.backend.lowerings": lowerings,
                    "litert_torch.backend.lowerings.utils": utils,
                    "litert_torch.backend.export_utils": export_utils,
                }
            )

            converter = types.SimpleNamespace(__name__="litert_torch")
            _patch_litert_uint8_dtype_support(converter)

            self.assertEqual(utils.torch_dtype_to_ir_element_type(torch.uint8), IntegerType.get_unsigned(8))
            self.assertEqual(export_utils.torch_dtype_to_ir_element_type(torch.uint8), IntegerType.get_unsigned(8))
            self.assertEqual(
                export_utils.ir_element_type_to_torch_dtype(IntegerType.get_unsigned(8)),
                torch.uint8,
            )
            self.assertEqual(
                export_utils.ir_element_type_to_torch_dtype(IntegerType.get_signless(8)),
                torch.int8,
            )
        finally:
            for name, module in original_modules.items():
                if module is None:
                    sys.modules.pop(name, None)
                else:
                    sys.modules[name] = module


if __name__ == "__main__":
    unittest.main()
