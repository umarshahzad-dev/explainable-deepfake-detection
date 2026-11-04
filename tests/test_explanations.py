import unittest
import numpy as np
from PIL import Image
from tests.support import module


class ExplanationTests(unittest.TestCase):
    def test_binary_class_direction_and_hook_cleanup(self):
        api = module(self, "explanations")
        import torch
        class Tiny(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.conv = torch.nn.Conv2d(3, 1, 1, bias=False)
                self.conv.weight.data.fill_(1 / 3)
            def forward(self, x): return self.conv(x).mean(dim=(2, 3))
        model = Tiny()
        x = torch.zeros(1, 3, 4, 4)
        x[:, :, :, 2:] = 1
        real = api.grad_cam(model, x, target_class=1, layer=model.conv)
        fake = api.grad_cam(model, x, target_class=0, layer=model.conv)
        self.assertGreater(float(real[:, 2:].mean()), 0.9)
        self.assertEqual(float(real[:, :2].sum()), 0)
        self.assertEqual(float(fake.sum()), 0)
        self.assertEqual(len(model.conv._forward_hooks), 0)

    def test_overlay_only_changes_unpadded_crop_region(self):
        api = module(self, "preprocessing")
        geometry = api.Geometry(-2, 0, 4, 6, 6)
        original = Image.new("RGB", (6, 6), (0, 0, 255))
        overlay = api.overlay_heatmap(original, geometry, np.ones((4, 4)))
        self.assertEqual(overlay.getpixel((5, 5)), (0, 0, 255))
        self.assertNotEqual(overlay.getpixel((0, 1)), (0, 0, 255))


if __name__ == "__main__":
    unittest.main()
