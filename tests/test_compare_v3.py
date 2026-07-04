import importlib.util
import unittest

import numpy as np


TORCH_AVAILABLE = importlib.util.find_spec("torch") is not None


class LabelMapAgreementTests(unittest.TestCase):
    def test_identical_maps_pass_gate(self):
        from src.export.compare import label_map_agreement

        ref = np.random.default_rng(3).integers(0, 9, size=(64, 64), dtype=np.uint8)
        report = label_map_agreement(ref, ref.copy())
        self.assertEqual(report.agreement, 1.0)
        self.assertEqual(report.mismatched_pixels, 0)
        self.assertTrue(report.ok)

    def test_agreement_ratio_and_gate_threshold(self):
        from src.export.compare import label_map_agreement

        ref = np.zeros((10, 10), dtype=np.uint8)
        got = ref.copy()
        got[0, :2] = 1  # 2/100 pixels differ -> 0.98
        report = label_map_agreement(ref, got, threshold=0.99)
        self.assertAlmostEqual(report.agreement, 0.98)
        self.assertEqual(report.mismatched_pixels, 2)
        self.assertFalse(report.ok)
        self.assertTrue(label_map_agreement(ref, got, threshold=0.95).ok)

    def test_shape_mismatch_raises(self):
        from src.export.compare import label_map_agreement

        with self.assertRaises(ValueError):
            label_map_agreement(np.zeros((4, 4), dtype=np.uint8), np.zeros((4, 5), dtype=np.uint8))

    def test_batch_dim_is_squeezed(self):
        from src.export.compare import label_map_agreement

        ref = np.zeros((1, 8, 8), dtype=np.int32)
        got = np.zeros((8, 8), dtype=np.uint8)
        self.assertTrue(label_map_agreement(ref, got).ok)

    def test_logits_to_label_map_channel_last_and_first(self):
        from src.export.compare import logits_to_label_map

        rng = np.random.default_rng(5)
        logits_hwc = rng.normal(size=(1, 6, 6, 9)).astype(np.float32)
        expected = np.argmax(logits_hwc[0], axis=-1)
        np.testing.assert_array_equal(logits_to_label_map(logits_hwc, class_axis=-1), expected)

        logits_chw = np.moveaxis(logits_hwc[0], -1, 0)[None]
        np.testing.assert_array_equal(logits_to_label_map(logits_chw, class_axis=1), expected)

    def test_report_to_dict_is_json_friendly(self):
        import json

        from src.export.compare import label_map_agreement

        report = label_map_agreement(np.zeros((4, 4), dtype=np.uint8), np.zeros((4, 4), dtype=np.uint8))
        payload = json.dumps(report.to_dict())
        self.assertIn('"agreement"', payload)
        self.assertIn('"ok"', payload)


@unittest.skipUnless(TORCH_AVAILABLE, "PyTorch is not installed in the local dev environment.")
class ReferenceLabelMapTests(unittest.TestCase):
    def test_reference_matches_v3_tflite_wrapper_exactly(self):
        """The fp32 server reference and the v3 uint8 wrapper must agree ~100%
        on the same input (the on-device gate then compares converted artifacts
        against this reference)."""
        import torch

        from src.export.compare import compute_reference_label_map, label_map_agreement
        from src.export.mobile_io import build_tflite_v3_module
        from src.models import HeadParsingMobile

        torch.manual_seed(11)
        model = HeadParsingMobile(num_classes=9, width_mult=0.5).eval()
        rng = np.random.default_rng(11)
        image = rng.integers(0, 256, size=(64, 64, 3), dtype=np.uint8)

        ref = compute_reference_label_map(model, image)
        self.assertEqual(ref.shape, (64, 64))

        wrapper = build_tflite_v3_module(model, include_argmax=True).eval()
        with torch.no_grad():
            got = wrapper(torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0)).numpy()
        self.assertTrue(label_map_agreement(ref, got, threshold=0.99).ok)


if __name__ == "__main__":
    unittest.main()
