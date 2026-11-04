import unittest
from tests.support import module, workspace


class EvaluationTests(unittest.TestCase):
    def test_heldout_audit_rejects_relabelled_training_parent(self):
        api = module(self, "evaluation")
        paths = module(self, "paths")
        ws = paths.Workspace(workspace(self))
        original = {"sample_id": "old", "label": 0, "split": "train", "parents": ["person-source"], "dataset": "ffpp"}
        paths.write_json(ws.run("r") / "input_manifest.json", {"rows": [original]}, ws)
        with self.assertRaises(ValueError):
            api.audit_against_run(ws.run("r"), [{**original, "sample_id": "renamed", "split": "test"}])

    def test_raw_evaluation_reports_detector_abstention_separately(self):
        api = module(self, "evaluation")
        self.assertTrue(hasattr(api, "evaluate_raw"), "Raw pipeline evaluation not implemented")
        paths = module(self, "paths")
        cfg = module(self, "config").load_config()
        root = workspace(self)
        ws = paths.Workspace(root)
        raw = root / "fixture.mp4"
        raw.write_bytes(b"raw-frame-fixture")
        row = {"sample_id": "dfd:fake:fixture", "dataset": "dfd", "method": "DFD", "label": 0,
               "split": "external", "parents": ["dfd:1", "dfd:2"], "known_person_ids": [], "identity_verified": False, "subject_verified": True, "verification_source":"manual_registry",
               "raw_path": str(raw), "sha256": paths.digest_file(raw), "cluster": "dfd:1"}
        paths.write_jsonl(ws.manifest("fixture") / "videos.jsonl", [row], ws)
        paths.write_json(ws.run("fixture-run") / "input_manifest.json", {"rows": []}, ws)
        from PIL import Image
        class Reader:
            count = 2
            fps = 1
            def __init__(self, path): pass
            def __enter__(self): return self
            def __exit__(self, *args): pass
            def frames(self, targets):
                for i in targets: yield i, Image.new("RGB", (100, 100))
        class Engine:
            metadata = {"weights_sha256": "fixture", "preprocessing_hash": "contract"}
            calibration = {"threshold": 0.5}
            def __init__(self): self.count = 0; self.cfg = cfg
            def analyze(self, image, explain=False):
                self.count += 1
                return {"faces": [] if self.count == 1 else [{"status": "accepted", "real_score": 0.1, "fake_score": 0.9}]}
        report = api.evaluate_raw("fixture-run", ["fixture"], ws, engine=Engine(), reader_factory=Reader, max_frames=2)
        self.assertEqual(report["frames_attempted"], 2)
        self.assertEqual(report["classified_frames"], 1)
        self.assertEqual(report["classification_coverage"], 0.5)
        self.assertEqual(report["abstentions"]["no_faces"], 1)
        self.assertEqual(report["overall"]["accuracy"], 1)

    def test_report_plots_are_written_inside_runs(self):
        api = module(self, "evaluation")
        paths = module(self, "paths")
        metrics = module(self, "metrics")
        root = workspace(self)
        ws = paths.Workspace(root)
        output = ws.report("figures")
        api.save_plots(metrics.metric_fixture(), output, ws)
        from PIL import Image
        for filename in ("roc.png", "pr.png", "calibration.png", "confusion_matrix.png"):
            with Image.open(output / filename) as image:
                self.assertGreater(image.width, 100)

    def test_external_data_cannot_be_used_to_fit_calibration(self):
        api = module(self, "evaluation")
        rows = [{"split": "calibration", "dataset": "dfd", "label": 0}, {"split": "calibration", "dataset": "ffpp", "label": 1}]
        with self.assertRaises(ValueError):
            api.calibration_rows(rows)

    def test_evaluation_exports_correct_metrics_and_sample_scores(self):
        api = module(self, "evaluation")
        paths = module(self, "paths")
        preprocessing = module(self, "preprocessing")
        cfg = module(self, "config").load_config(overrides={"training": {"workers": 0, "batch_size": 2, "amp": False}})
        from PIL import Image
        from unittest.mock import patch
        import torch
        root = workspace(self)
        ws = paths.Workspace(root)
        rows = []
        for label in (0, 1):
            path = root / "crops" / f"{label}.png"
            Image.new("RGB", (299, 299), (255 * label, 0, 0)).save(path)
            rows.append({"sample_id": str(label), "video_id": str(label), "dataset": "ffpp", "method": "Deepfakes" if label == 0 else "original",
                         "label": label, "split": "test", "parents": [str(label)], "sha256": str(label), "known_person_ids": [],
                         "identity_verified": False, "crop_path": str(path), "cluster": str(label),
                         "preprocessing_hash": preprocessing.preprocessing_hash(cfg), "subject_verified": True, "extraction_status": "completed"})
        paths.write_jsonl(ws.manifest("fixture") / "crops.jsonl", rows, ws)
        paths.write_json(ws.run("fixture-run") / "input_manifest.json", {"rows": []}, ws)
        class Tiny(torch.nn.Module):
            def forward(self, x): return x[:, 0].mean(dim=(1, 2)).unsqueeze(1)
        metadata = {"weights_sha256": "fixture", "run_id": "fixture-run", "preprocessing_hash": preprocessing.preprocessing_hash(cfg), "pilot": True}
        with patch("deepfake_detection.evaluation.evaluator.load_run_model", return_value=(Tiny(), cfg, metadata)):
            result = api.evaluate("fixture-run", ["fixture"], ws, device="cpu", plots=False, bootstrap=10)
        self.assertEqual(result["overall"]["confusion_matrix"], [[1, 0], [0, 1]])
        self.assertEqual(result["overall"]["accuracy"], 1)
        self.assertTrue(result["pilot"])
        self.assertTrue((ws.report("fixture-run-evaluation") / "predictions.csv").exists())
        self.assertEqual(paths.read_json(ws.report("fixture-run-evaluation") / "metrics.json")["weights_sha256"], "fixture")


if __name__ == "__main__":
    unittest.main()
