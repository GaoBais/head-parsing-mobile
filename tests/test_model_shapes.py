import importlib.util
import unittest


TORCH_AVAILABLE = importlib.util.find_spec("torch") is not None


@unittest.skipUnless(TORCH_AVAILABLE, "PyTorch is not installed in the local Windows dev environment.")
class ModelShapeTests(unittest.TestCase):
    def test_head_parsing_mobile_shape(self):
        import torch

        from src.models import HeadParsingMobile

        model = HeadParsingMobile(num_classes=20, width_mult=0.5, decoder_channels=64)
        model.eval()
        with torch.no_grad():
            output = model(torch.randn(2, 3, 320, 320))

        self.assertEqual(tuple(output.shape), (2, 20, 320, 320))


if __name__ == "__main__":
    unittest.main()
