import copy
import unittest
from tests.support import module, workspace


class TrainingTests(unittest.TestCase):
    def test_two_phase_controller_saves_isolated_checkpoints_and_rejects_completed_resume(self):
        api = module(self, "training")
        cfg_api = module(self, "config")
        preprocessing = module(self, "preprocessing")
        paths = module(self, "paths")
        from PIL import Image
        from unittest.mock import patch
        import torch
        root = workspace(self)
        ws = paths.Workspace(root)
        cfg = cfg_api.load_config(root=root, overrides={"training": {"head_epochs": 1, "finetune_epochs": 1, "workers": 0, "epoch_samples": 4, "batch_size": 2, "accumulation": 2, "augmentation": "none", "amp": False}})
        class Tiny(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.backbone = torch.nn.Linear(3, 3)
                self.classifier = torch.nn.Linear(3, 1)
            def get_classifier(self): return self.classifier
            def forward(self, x): return self.classifier(self.backbone(x.mean(dim=(2, 3))))
        rows = []
        for split in ("train", "val"):
            for label in (0, 1):
                name = f"{split}-{label}"
                path = root / "crops" / (name + ".png")
                Image.new("RGB", (299, 299), (20 + label * 100, 30, 50 + (split == "val"))).save(path)
                rows.append({"sample_id": name, "video_id": name, "dataset": "ffpp", "method": "fake" if label == 0 else "original", "label": label,
                             "parents": [name], "known_person_ids": [], "identity_verified": False, "split": split,
                             "crop_path": str(path), "crop_sha256": paths.digest_file(path), "sha256": name, "pilot": True, "subject_verified": True, "preprocessing_hash": preprocessing.preprocessing_hash(cfg), "extraction_status": "completed"})
        ws.run("fixture-run").mkdir(parents=True)
        (ws.run("fixture-run") / "run.log").write_text("preflight log\n")
        with patch("deepfake_detection.training.trainer.initialize_model", return_value=Tiny()):
            result = api.run_training("fixture-run", cfg, rows, ws, device="cpu")
        self.assertEqual(result["status"], "completed")
        self.assertTrue(result["pilot"])
        self.assertTrue(result["best_checkpoint_path"])
        self.assertGreater(result["elapsed_seconds"],0)
        self.assertTrue(paths.read_json(ws.run("fixture-run") / "config.json")["pilot"])
        self.assertTrue((ws.run("fixture-run") / "head_latest.pt").exists())
        self.assertTrue((ws.run("fixture-run") / "finetune_latest.pt").exists())
        self.assertTrue((ws.run("fixture-run") / "best.safetensors").exists())
        with self.assertRaisesRegex(ValueError, "Pilot"):
            api.run_training("fixture-run", cfg, rows, ws, device="cpu", resume="finetune", max_batches=1)
        with self.assertRaises(ValueError):
            api.run_training("fixture-run", cfg, rows, ws, device="cpu", resume="finetune")
        self.assertEqual(paths.read_json(ws.run("fixture-run") / "status.json")["status"], "completed")

    def test_remainder_and_unequal_microbatches_match_direct_sample_means(self):
        api = module(self, "training")
        import torch
        torch.set_num_threads(2)
        model = torch.nn.Linear(1, 1, bias=False)
        model.weight.data.zero_()
        reference = copy.deepcopy(model)
        x = torch.tensor([[1.], [2.], [3.], [4.], [5.]])
        y = torch.ones((5, 1))
        batches = [(x[:2], y[:2]), (x[2:3], y[2:3]), (x[3:], y[3:])]
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
        result = api.train_epoch(model, batches, optimizer, device="cpu", accumulation=2, amp=False, clip_norm=None)
        ref_optimizer = torch.optim.SGD(reference.parameters(), lr=0.1)
        for xx, yy in [(x[:3], y[:3]), (x[3:], y[3:])]:
            ref_optimizer.zero_grad()
            torch.nn.functional.binary_cross_entropy_with_logits(reference(xx), yy).backward()
            ref_optimizer.step()
        self.assertEqual(result["optimizer_updates"], 2)
        self.assertEqual(result["samples"], 5)
        self.assertTrue(torch.allclose(model.weight, reference.weight, atol=1e-7))

    def test_nonfinite_loss_is_a_failure_not_success(self):
        api = module(self, "training")
        import torch
        model = torch.nn.Linear(1, 1)
        with self.assertRaises(FloatingPointError):
            api.train_epoch(model, [(torch.tensor([[float("nan")]]), torch.ones(1, 1))], torch.optim.SGD(model.parameters(), lr=0.1), device="cpu", amp=False)

    def test_frozen_bn_does_not_update_running_statistics(self):
        api = module(self, "training")
        import torch
        model = torch.nn.Sequential(torch.nn.BatchNorm1d(2), torch.nn.Linear(2, 1))
        model.train()
        api.freeze_bn(model)
        before = model[0].running_mean.clone()
        model(torch.full((4, 2), 5.0))
        self.assertTrue(torch.equal(before, model[0].running_mean))

    def test_resume_rejects_each_important_incompatibility(self):
        api = module(self, "checkpoints")
        cfg = module(self, "config").load_config()
        expected = api.resume_signature(cfg, "manifest-sha", "finetune", "cpu")
        for key, value in [("phase", "head"), ("manifest_sha256", "different"), ("loss", "focal"), ("preprocessing_hash", "different"), ("class_map", {"fake": 1, "real": 0})]:
            changed = copy.deepcopy(expected)
            changed[key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                api.assert_compatible(changed, expected)

    def test_checkpoint_round_trip_restores_rng_and_updated_best(self):
        api = module(self, "checkpoints")
        paths = module(self, "paths")
        repro = module(self, "repro")
        import torch, random, numpy as np
        root = workspace(self)
        ws = paths.Workspace(root)
        repro.seed_all(42)
        rng = repro.rng_state()
        expected = (random.random(), float(np.random.random()), float(torch.rand(())))
        checkpoint = {"signature": {"phase": "head"}, "rng": rng, "best_val_auc": 0.88, "model_state_dict": {"weight": torch.tensor([1.0])}}
        path = root / "runs/checkpoint.pt"
        api.save_checkpoint(path, checkpoint, ws)
        loaded = api.load_checkpoint(path, {"phase": "head"}, workspace=ws)
        repro.restore_rng(loaded["rng"])
        actual = (random.random(), float(np.random.random()), float(torch.rand(())))
        self.assertEqual(expected, actual)
        self.assertEqual(loaded["best_val_auc"], 0.88)

    def test_sampler_is_epoch_seeded_and_cannot_include_external_rows(self):
        api = module(self, "data")
        rows = [{"label": c, "dataset": "ffpp", "method": "original" if c else "Deepfakes", "video_id": f"v{c}", "split": "train"} for c in (0, 1)]
        sampler = api.HierarchicalSampler(rows, samples=20, seed=42)
        first = list(sampler)
        self.assertEqual(first, list(sampler))
        sampler.set_epoch(1)
        self.assertNotEqual(first, list(sampler))
        self.assertEqual(set(first), {0, 1})
        rows[0]["split"] = "external"
        with self.assertRaises(ValueError):
            api.HierarchicalSampler(rows, samples=20, seed=42)


if __name__ == "__main__":
    unittest.main()
