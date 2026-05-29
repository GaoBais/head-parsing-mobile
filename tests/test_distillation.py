import importlib.util
import unittest


TORCH_AVAILABLE = importlib.util.find_spec("torch") is not None


@unittest.skipUnless(TORCH_AVAILABLE, "PyTorch is not installed in the local Windows dev environment.")
class DistillationTests(unittest.TestCase):
    def test_soft_target_distillation_loss(self):
        import torch

        from src.training.distillation import SoftTargetDistillationLoss

        student = torch.randn(2, 19, 32, 32)
        teacher = torch.randn(2, 19, 16, 16)
        loss = SoftTargetDistillationLoss(temperature=2.0)(student, teacher)
        self.assertTrue(torch.isfinite(loss))

    def test_disable_teacher_backbone_default_weights(self):
        import sys
        import types

        from src.training.distillation import _disable_teacher_backbone_default_weights

        resnet_module = types.ModuleType("models.resnet")
        bisenet_module = types.ModuleType("models.bisenet")
        calls = []

        def resnet18(**kwargs):
            calls.append(("resnet18", kwargs.get("weights", "missing")))

        def resnet34(**kwargs):
            calls.append(("resnet34", kwargs.get("weights", "missing")))

        resnet_module.resnet18 = resnet18
        resnet_module.resnet34 = resnet34
        bisenet_module.resnet18 = resnet18
        bisenet_module.resnet34 = resnet34
        original_resnet = sys.modules.get("models.resnet")
        original_bisenet = sys.modules.get("models.bisenet")
        sys.modules["models.resnet"] = resnet_module
        sys.modules["models.bisenet"] = bisenet_module
        try:
            with _disable_teacher_backbone_default_weights(bisenet_module):
                resnet_module.resnet18(weights="default")
                resnet_module.resnet34(weights="default")
                bisenet_module.resnet18(weights="default")
                bisenet_module.resnet34(weights="default")
        finally:
            if original_resnet is None:
                sys.modules.pop("models.resnet", None)
            else:
                sys.modules["models.resnet"] = original_resnet
            if original_bisenet is None:
                sys.modules.pop("models.bisenet", None)
            else:
                sys.modules["models.bisenet"] = original_bisenet

        self.assertEqual(
            calls,
            [("resnet18", None), ("resnet34", None), ("resnet18", None), ("resnet34", None)],
        )


if __name__ == "__main__":
    unittest.main()
