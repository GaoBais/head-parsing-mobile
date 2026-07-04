import importlib.util
import unittest


TORCH_AVAILABLE = importlib.util.find_spec("torch") is not None


@unittest.skipUnless(TORCH_AVAILABLE, "PyTorch is not installed in the local dev environment.")
class MobileIoV3Tests(unittest.TestCase):
    """v3 mobile I/O graph adaptation: uint8 input, in-graph normalization, in-graph argmax."""

    @classmethod
    def setUpClass(cls):
        import torch

        from src.models import HeadParsingMobile

        torch.manual_seed(7)
        cls.model = HeadParsingMobile(num_classes=9, width_mult=0.5).eval()
        cls.image_u8 = torch.randint(0, 256, (1, 3, 64, 64), dtype=torch.uint8)

    def _v2_reference_logits(self):
        """Reference: the v2 on-device contract ((rgb/255 - mean) / std) -> model."""
        import torch

        from src.datasets.transforms import IMAGENET_MEAN, IMAGENET_STD

        mean = torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1)
        std = torch.tensor(IMAGENET_STD).view(1, 3, 1, 1)
        x = (self.image_u8.float() / 255.0 - mean) / std
        with torch.no_grad():
            return self.model(x)

    def test_tflite_module_matches_v2_normalization_end_to_end(self):
        import torch

        from src.export.mobile_io import build_tflite_v3_module

        wrapper = build_tflite_v3_module(self.model, include_argmax=False).eval()
        with torch.no_grad():
            got = wrapper(self.image_u8)
        ref = self._v2_reference_logits()
        self.assertEqual(got.shape, ref.shape)
        self.assertLess((got - ref).abs().max().item(), 1e-3)
        self.assertTrue(torch.equal(got.argmax(dim=1), ref.argmax(dim=1)))

    def test_fold_std_does_not_mutate_original_model(self):
        import torch

        from src.export.mobile_io import fold_std_into_stem

        before = self.model.encoder.stem[0].weight.detach().clone()
        folded = fold_std_into_stem(self.model)
        self.assertIsNot(folded, self.model)
        self.assertTrue(torch.equal(self.model.encoder.stem[0].weight, before))
        self.assertFalse(torch.equal(folded.encoder.stem[0].weight, before))

    def test_fold_std_rejects_model_without_rgb_stem_conv(self):
        from torch import nn

        from src.export.mobile_io import fold_std_into_stem

        class Odd(nn.Module):
            def __init__(self):
                super().__init__()
                self.encoder = nn.Module()
                self.encoder.stem = nn.Sequential(nn.Conv2d(4, 8, 3))

        with self.assertRaises(ValueError):
            fold_std_into_stem(Odd())

    def test_argmax_head_matches_torch_argmax_with_dtype(self):
        import torch

        from src.export.mobile_io import ArgmaxHead

        logits = torch.randn(1, 9, 16, 16)
        out = ArgmaxHead(dtype=torch.uint8)(logits)
        self.assertEqual(out.dtype, torch.uint8)
        self.assertTrue(torch.equal(out.long(), torch.argmax(logits, dim=1)))
        out32 = ArgmaxHead(dtype=torch.int32)(logits)
        self.assertEqual(out32.dtype, torch.int32)

    def test_tflite_label_map_variant_outputs_uint8_hw(self):
        import torch

        from src.export.mobile_io import build_tflite_v3_module

        wrapper = build_tflite_v3_module(self.model, include_argmax=True).eval()
        with torch.no_grad():
            out = wrapper(self.image_u8)
        self.assertEqual(out.shape, (1, 64, 64))
        self.assertEqual(out.dtype, torch.uint8)
        ref = self._v2_reference_logits().argmax(dim=1)
        self.assertTrue(torch.equal(out.long(), ref))

    def test_coreml_module_expects_prescaled_input_and_matches_reference(self):
        """CoreML ImageType applies scale/bias in-framework; the traced module must
        expect `rgb/255 - mean` (std folded into the stem conv) and no uint8 adapter."""
        import torch

        from src.datasets.transforms import IMAGENET_MEAN
        from src.export.mobile_io import build_coreml_v3_module

        mean = torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1)
        prescaled = self.image_u8.float() / 255.0 - mean
        wrapper = build_coreml_v3_module(self.model, include_argmax=False).eval()
        with torch.no_grad():
            got = wrapper(prescaled)
        ref = self._v2_reference_logits()
        self.assertLess((got - ref).abs().max().item(), 1e-3)

    def test_coreml_label_map_variant_outputs_int32(self):
        import torch

        from src.datasets.transforms import IMAGENET_MEAN
        from src.export.mobile_io import build_coreml_v3_module

        mean = torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1)
        prescaled = self.image_u8.float() / 255.0 - mean
        wrapper = build_coreml_v3_module(self.model, include_argmax=True).eval()
        with torch.no_grad():
            out = wrapper(prescaled)
        self.assertEqual(out.shape, (1, 64, 64))
        self.assertEqual(out.dtype, torch.int32)


if __name__ == "__main__":
    unittest.main()
