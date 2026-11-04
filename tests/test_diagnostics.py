import copy
import unittest
from unittest.mock import patch
from tests.support import module, workspace


class DiagnosticTests(unittest.TestCase):
    def test_short_diagnostic_retries_scaler_skip_with_lower_scale(self):
        api = module(self, "diagnostics")
        import torch
        cfg = module(self, "config").load_config(overrides={"training": {"batch_size": 2, "accumulation": 8}})
        cfg["model"]["input_size"] = 4
        before = copy.deepcopy(cfg)
        class Tiny(torch.nn.Module):
            def __init__(self): super().__init__(); self.head = torch.nn.Linear(3, 1)
            def forward(self, x): return self.head(x.mean((2, 3)))
        scales = []
        class SimulatedScaler:
            def __init__(self, *args, init_scale=65536, **kwargs): self.value = init_scale; scales.append(init_scale)
            def scale(self, loss): return loss
            def unscale_(self, optimizer): pass
            def get_scale(self): return self.value
            def step(self, optimizer):
                if self.value < 1024: optimizer.step()
            def update(self):
                if self.value >= 1024: self.value /= 2
        ws = module(self, "paths").Workspace(workspace(self))
        with patch("deepfake_detection.common.diagnostics.initialize_model", return_value=Tiny()), patch("torch.amp.GradScaler", SimulatedScaler):
            result = api.diagnostic(cfg, ws, synthetic=True, steps=2, warmup=0, device="cpu")
        self.assertEqual(result["optimizer_updates"], 1)
        self.assertGreater(result["amp_retries"], 0)
        self.assertLess(scales[-1], scales[0])
        self.assertEqual(cfg, before)

    def test_no_amp_flag_is_diagnostic_only(self):
        from tools.laptop_workflow import build_parser
        parser = build_parser()
        self.assertTrue(parser.parse_args(["profile", "--synthetic", "--no-amp"]).no_amp)
        import io
        with patch("sys.stderr", io.StringIO()), self.assertRaises(SystemExit):
            parser.parse_args(["train", "--run", "r", "--manifests", "m", "--no-amp"])

    def test_missing_gradient_has_distinct_error(self):
        api = module(self, "training")
        import torch
        class Detached(torch.nn.Module):
            def __init__(self): super().__init__(); self.weight = torch.nn.Parameter(torch.ones(1))
            def forward(self, x): return torch.zeros_like(x)
        model = Detached()
        with self.assertRaisesRegex(ValueError, "No gradient flowed"):
            api.train_epoch(model, [(torch.ones(1, 1), torch.ones(1, 1))], torch.optim.SGD(model.parameters(), lr=.1), amp=False)
