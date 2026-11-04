import unittest
from pathlib import Path
from tests.support import module, workspace


class ModelTests(unittest.TestCase):
    def test_export_rejects_changed_preprocessing_before_loading(self):
        api = module(self, "model")
        paths = module(self, "paths")
        ws = paths.Workspace(workspace(self))
        cfg = module(self, "config").load_config()
        paths.write_json(ws.run("r") / "config.json", cfg, ws)
        paths.write_json(ws.run("r") / "best_metadata.json", {"preprocessing_hash": "wrong"}, ws)
        with self.assertRaisesRegex(ValueError, "contract"):
            api.load_run_model(ws.run("r"), workspace=ws)

    def test_failed_best_publication_preserves_previous_generation(self):
        api = module(self, "model")
        paths = module(self, "paths")
        from unittest.mock import patch
        import torch
        ws = paths.Workspace(workspace(self))
        run = ws.run("r")
        cfg = module(self, "config").load_config()
        model = torch.nn.Linear(1, 1)
        api.publish_best(model, run, cfg, {"validation_macro_auc": 0.9}, ws)
        before = paths.read_json(run / "best_metadata.json")
        model.weight.data.add_(1)
        with patch("deepfake_detection.models.model_loader.write_json", side_effect=OSError("interrupted")):
            with self.assertRaises(OSError):
                api.publish_best(model, run, cfg, {"validation_macro_auc": 0.95}, ws)
        self.assertEqual(paths.read_json(run / "best_metadata.json"), before)
        self.assertEqual(paths.digest_file(run / before["weights_file"]), before["weights_sha256"])
    def test_bad_weight_checksum_is_rejected(self):
        api = module(self, "model")
        path = workspace(self) / "wrong.safetensors"
        path.write_bytes(b"abc")
        with self.assertRaises(ValueError):
            api.verify_weights(path, expected_size=3, expected_sha="0" * 64)

    def test_verified_local_b0_backbone_and_single_logit_head(self):
        api = module(self, "model")
        cfg = module(self, "config").load_config()
        path = Path(cfg["weights"])
        if not path.exists(): self.skipTest("Optional local ImageNet B0 weights are not installed")
        model = api.initialize_model(cfg, device="cpu")
        self.assertEqual((model.get_classifier().in_features, model.get_classifier().out_features), (1280, 1))
        from safetensors.torch import load_file
        import torch
        source = load_file(str(path))
        self.assertTrue(torch.equal(model.state_dict()["conv_stem.weight"], source["conv_stem.weight"]))

    def test_old_project_checkpoint_cannot_be_initialization(self):
        api = module(self, "model")
        from unittest.mock import patch
        cfg = module(self, "config").load_config(overrides={"weights": str(Path(__file__).absolute().parents[1] / ".old-project/models/checkpoints/model_new_focal_93.pth")})
        # Verify the legacy-path gate without requiring an ignored local archive.
        with patch("deepfake_detection.models.pretraining.source_weights", return_value=("imagenet", Path(cfg["weights"]))), self.assertRaises(ValueError):
            api.initialize_model(cfg, device="cpu")


if __name__ == "__main__":
    unittest.main()
