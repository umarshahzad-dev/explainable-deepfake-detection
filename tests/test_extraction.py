import unittest
from PIL import Image
from tests.support import module, workspace


class ExtractionTests(unittest.TestCase):
    def test_unreviewed_fake_is_not_trainable(self):
        api = module(self, "data")
        cfg = module(self, "config").load_config()
        row = {"label": 0, "preprocessing_hash": module(self, "preprocessing").preprocessing_hash(cfg),
               "crop_path": str(workspace(self) / "crops/x.png"), "extraction_status": "completed"}
        with self.assertRaisesRegex(ValueError, "subject"):
            api.CropDataset([row], cfg)
    def test_inventory_limits_checks_without_truncating_collection_counts(self):
        api = module(self, "inventory")
        paths = module(self, "paths")
        root = workspace(self)
        raw = root / "raw/original_sequences/youtube/c23/videos"
        raw.mkdir(parents=True)
        (raw / "001.mp4").write_bytes(b"one")
        (raw / "002.mp4").write_bytes(b"two")
        before = (raw / "001.mp4").stat().st_mtime_ns
        cfg = module(self, "config").load_config(root=root, overrides={"raw_root": str(root / "raw")})
        report = api.inventory(cfg, paths.Workspace(root), limit_files=1, decode=False)
        self.assertEqual(report["total_videos"], 2)
        self.assertEqual(report["checked_files"], 1)
        self.assertEqual((raw / "001.mp4").stat().st_mtime_ns, before)

    def test_sample_indices_are_spaced_unique_and_in_range(self):
        api = module(self, "extraction")
        self.assertEqual(api.sample_indices(100, 4), [12, 37, 62, 87])
        self.assertEqual(api.sample_indices(3, 20), [0, 1, 2])
        self.assertEqual(api.sample_indices(0, 4), [])

    def test_generated_frames_save_rgb_masters_and_resume_without_duplicates(self):
        api = module(self, "extraction")
        paths = module(self, "paths")
        preprocessing = module(self, "preprocessing")
        detector_api = module(self, "detector")
        cfg = module(self, "config").load_config()
        root = workspace(self)
        ws = paths.Workspace(root)
        raw = root / "fixture.mp4"
        raw.write_bytes(b"generated-frame-decoder-fixture")
        row = {"sample_id": "ffpp:original:004", "dataset": "ffpp", "method": "original", "raw_path": str(raw),
               "sha256": paths.digest_file(raw), "label": 1, "parents": ["ffpp:004"], "known_person_ids": [],
               "identity_verified": False, "cluster": "ffpp:004", "split": "train"}
        paths.write_jsonl(ws.manifest("fixture") / "videos.jsonl", [row], ws)
        class Reader:
            count = 3
            fps = 1
            def __init__(self, path): pass
            def __enter__(self): return self
            def __exit__(self, *args): pass
            def frames(self, targets):
                for i in targets: yield i, Image.new("RGB", (100, 100), (255, 0, 0))
        class Detector:
            def detect(self, image):
                crop = preprocessing.square_crop(image, [10, 10, 90, 90])
                return [detector_api.Face([10, 10, 90, 90], 0.99, [], "accepted", crop)]
        result = api.extract("fixture", cfg, ws, detector=Detector(), reader_factory=Reader)
        self.assertEqual(result["saved_crops"], 3)
        records = paths.read_jsonl(ws.manifest("fixture") / "crops.jsonl")
        self.assertEqual([r["frame_index"] for r in records], [0, 1, 2])
        for record in records:
            self.assertTrue(str(record["crop_path"]).startswith(str(ws.crops)))
            with Image.open(record["crop_path"]) as image:
                self.assertEqual((image.size, image.mode), ((299, 299), "RGB"))
        result = api.extract("fixture", cfg, ws, detector=Detector(), reader_factory=Reader)
        self.assertEqual(result["skipped_completed_videos"], 1)
        self.assertEqual(len(paths.read_jsonl(ws.manifest("fixture") / "crops.jsonl")), 3)
        cfg["extraction"]["train_caps"]["ffpp"] = 2
        paths.write_jsonl(ws.manifest("capped") / "videos.jsonl", [row], ws)
        result = api.extract("capped", cfg, ws, detector=Detector(), reader_factory=Reader)
        self.assertEqual(result["saved_crops"], 1)
        self.assertEqual(len(list((ws.crops / "capped").rglob("*.png"))), 1)

    def test_unreadable_video_does_not_become_completed(self):
        api = module(self, "extraction")
        paths = module(self, "paths")
        root = workspace(self)
        ws = paths.Workspace(root)
        raw = root / "bad.mp4"
        raw.write_bytes(b"bad")
        row = {"sample_id": "x", "dataset": "ffpp", "method": "original", "raw_path": str(raw), "sha256": paths.digest_file(raw),
               "label": 1, "parents": ["x"], "known_person_ids": [], "identity_verified": False, "cluster": "x", "split": "train"}
        paths.write_jsonl(ws.manifest("fixture") / "videos.jsonl", [row], ws)
        class BrokenReader:
            def __init__(self, path): raise ValueError("unreadable")
        result = api.extract("fixture", module(self, "config").load_config(), ws, detector=object(), reader_factory=BrokenReader)
        self.assertEqual(result["failed_videos"], 1)
        progress = paths.read_json(ws.manifest("fixture") / "extraction_progress.json")
        self.assertEqual(progress["videos"]["x"]["status"], "failed")
        self.assertEqual(result["saved_crops"], 0)


if __name__ == "__main__":
    unittest.main()
