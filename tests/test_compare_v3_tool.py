import unittest

import numpy as np


class CompareToolDispatchTests(unittest.TestCase):
    def test_label_map_outputs_are_squeezed_for_both_frameworks(self):
        from tools.compare_v3_artifacts import artifact_label_map

        raw = np.arange(16, dtype=np.uint8).reshape(1, 4, 4)
        for framework in ("tflite", "coreml"):
            got = artifact_label_map(raw, output_kind="label_map", framework=framework)
            np.testing.assert_array_equal(got, raw[0])

    def test_tflite_logits_argmax_over_last_axis(self):
        from tools.compare_v3_artifacts import artifact_label_map

        rng = np.random.default_rng(9)
        logits = rng.normal(size=(1, 4, 4, 9)).astype(np.float32)
        got = artifact_label_map(logits, output_kind="logits", framework="tflite")
        np.testing.assert_array_equal(got, np.argmax(logits[0], axis=-1))

    def test_coreml_logits_argmax_over_channel_axis(self):
        from tools.compare_v3_artifacts import artifact_label_map

        rng = np.random.default_rng(9)
        logits = rng.normal(size=(1, 9, 4, 4)).astype(np.float32)
        got = artifact_label_map(logits, output_kind="logits", framework="coreml")
        np.testing.assert_array_equal(got, np.argmax(logits[0], axis=0))

    def test_rejects_unknown_kind(self):
        from tools.compare_v3_artifacts import artifact_label_map

        with self.assertRaises(ValueError):
            artifact_label_map(np.zeros((1, 4, 4)), output_kind="softmax", framework="tflite")


class GateReportTests(unittest.TestCase):
    def test_gate_report_aggregates_and_flags_failures(self):
        from tools.compare_v3_artifacts import build_gate_report

        ref = np.zeros((10, 10), dtype=np.uint8)
        bad = ref.copy()
        bad[0, :3] = 1  # 0.97 < 0.99
        entries = [
            ("tflite/label_map", ref.copy()),
            ("tflite/logits", bad),
        ]
        report = build_gate_report(ref, entries, threshold=0.99)
        self.assertFalse(report["ok"])
        by_name = {item["artifact"]: item for item in report["artifacts"]}
        self.assertTrue(by_name["tflite/label_map"]["ok"])
        self.assertFalse(by_name["tflite/logits"]["ok"])
        self.assertAlmostEqual(by_name["tflite/logits"]["agreement"], 0.97)

    def test_gate_report_all_green(self):
        from tools.compare_v3_artifacts import build_gate_report

        ref = np.ones((6, 6), dtype=np.uint8)
        report = build_gate_report(ref, [("a", ref.copy()), ("b", ref.copy())], threshold=0.99)
        self.assertTrue(report["ok"])
        self.assertEqual(len(report["artifacts"]), 2)


if __name__ == "__main__":
    unittest.main()
