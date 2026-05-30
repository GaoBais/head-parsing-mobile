import importlib.util
import unittest


TORCH_AVAILABLE = importlib.util.find_spec("torch") is not None


@unittest.skipUnless(TORCH_AVAILABLE, "PyTorch is not installed in the local Windows dev environment.")
class LossMetricTests(unittest.TestCase):
    def test_combined_loss_and_metrics(self):
        import torch

        from src.training import CombinedSegmentationLoss, SegmentationLossConfig, SegmentationMetrics

        logits = torch.randn(2, 20, 32, 32)
        target = torch.randint(0, 20, (2, 32, 32))
        criterion = CombinedSegmentationLoss(SegmentationLossConfig(num_classes=20, min_kept=128))
        losses = criterion(logits, target)
        self.assertIn("loss", losses)
        self.assertTrue(torch.isfinite(losses["loss"]))

        metrics = SegmentationMetrics(num_classes=20)
        metrics.update(logits, target)
        result = metrics.compute()
        self.assertIn("mean_iou", result)


if __name__ == "__main__":
    unittest.main()
