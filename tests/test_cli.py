import importlib
from io import StringIO
from contextlib import redirect_stdout
import unittest
from unittest.mock import patch
from tests.support import module, workspace


class CLITests(unittest.TestCase):
    def test_failed_extraction_returns_nonzero(self):
        api = self.cli()
        ws = module(self, "paths").Workspace(workspace(self))
        with patch("deepfake_detection.data.extraction.extract", return_value={"failed_videos": 1}), patch("deepfake_detection.data.face_detector.FaceDetector"), patch("sys.stderr", StringIO()):
            self.assertEqual(api.main(["extract", "--manifest", "fixture", "--device", "cpu"], workspace=ws), 2)
    def cli(self):
        try:
            return importlib.import_module("tools.laptop_workflow")
        except ModuleNotFoundError:
            self.fail("workflow CLI is not implemented")

    def test_pilot_flags_are_bounded_and_names_match_requested_interface(self):
        api = self.cli()
        args = api.build_parser().parse_args(["extract", "--manifest", "ffpp-pilot", "--limit-videos", "2", "--max-frames", "2"])
        self.assertEqual((args.limit_videos, args.max_frames, args.manifest), (2, 2, "ffpp-pilot"))
        with redirect_stdout(StringIO()), patch("sys.stderr", StringIO()), self.assertRaises(SystemExit):
            api.build_parser().parse_args(["profile", "--synthetic", "--steps", "0"])

    def test_metric_self_test_runs_without_data_or_network(self):
        api = self.cli()
        paths = module(self, "paths")
        ws = paths.Workspace(workspace(self))
        with redirect_stdout(StringIO()) as output:
            result = api.main(["metric-self-test"], workspace=ws)
        self.assertEqual(result, 0)
        self.assertIn('"fake_recall": 0.75', output.getvalue())

    def test_diagnostic_uses_toy_model_and_saves_no_checkpoint(self):
        api = module(self, "diagnostics")
        paths = module(self, "paths")
        cfg = module(self, "config").load_config(overrides={"training": {"batch_size": 2, "accumulation": 2, "amp": False}})
        cfg["model"]["input_size"] = 4
        import torch
        class Tiny(torch.nn.Module):
            def __init__(self): super().__init__(); self.linear = torch.nn.Linear(3, 1)
            def forward(self, x): return self.linear(x.mean(dim=(2, 3)))
        ws = paths.Workspace(workspace(self))
        with patch("deepfake_detection.common.diagnostics.initialize_model", return_value=Tiny()):
            result = api.diagnostic(cfg, ws, synthetic=True, steps=2, warmup=0, device="cpu")
        self.assertEqual(result["microbatches"], 2)
        self.assertEqual(result["optimizer_updates"], 1)
        self.assertTrue(result["synthetic"])
        self.assertFalse(list(ws.runs.rglob("*.pt")))


if __name__ == "__main__":
    unittest.main()
