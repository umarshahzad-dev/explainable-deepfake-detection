import unittest
from PIL import Image
from tests.support import module, workspace


class RetryTests(unittest.TestCase):
    def assert_segments(self, count, quota):
        api = module(self, "extraction")
        groups = api.temporal_candidates(count, quota, seed=42, video_id="short", attempts=5)
        self.assertEqual(groups, api.temporal_candidates(count, quota, seed=42, video_id="short", attempts=5))
        self.assertEqual(len(groups), min(count, quota))
        for segment, indices in enumerate(groups):
            low, high = segment*count//len(groups), (segment+1)*count//len(groups)
            self.assertTrue(all(low <= index < high for index in indices), (segment,indices,low,high))
            self.assertEqual(len(indices), len(set(indices)))
        flat = [i for group in groups for i in group]
        self.assertEqual(len(flat), len(set(flat)))

    def test_short_5_into_4(self): self.assert_segments(5,4)
    def test_short_2_into_4(self): self.assert_segments(2,4)
    def test_short_1_into_4(self): self.assert_segments(1,4)
    def test_control_100_into_4(self): self.assert_segments(100,4)
    def setup_video(self):
        paths = module(self, "paths")
        ws = paths.Workspace(workspace(self))
        raw = ws.root / "fixture.mp4"
        raw.write_bytes(b"fixture")
        row = {"sample_id": "video", "dataset": "ffpp", "method": "original", "label": 1, "split": "train",
               "raw_path": str(raw), "sha256": paths.digest_file(raw), "parents": ["ffpp:1"]}
        paths.write_jsonl(ws.manifest("fixture") / "videos.jsonl", [row], ws)
        cfg = module(self, "config").load_config(overrides={"extraction": {"train_real": 1}})
        class Reader:
            count = 20
            def __init__(self, path): pass
            def __enter__(self): return self
            def __exit__(self, *args): pass
            def frames(self, targets):
                for index in sorted(targets): yield index, Image.new("RGB", (100, 100), (index, 0, 0))
        return ws, cfg, Reader

    def test_retry_succeeds_after_primary_has_no_face(self):
        api = module(self, "extraction")
        ws, cfg, reader = self.setup_video()
        preprocessing = module(self, "preprocessing")
        face = module(self, "detector").Face
        class Detector:
            calls = 0
            def detect(self, image):
                self.calls += 1
                if self.calls == 1: return []
                return [face([10,10,90,90], .99, [], "accepted", preprocessing.square_crop(image, [10,10,90,90]))]
        detector = Detector()
        result = api.extract("fixture", cfg, ws, detector=detector, reader_factory=reader)
        self.assertEqual(result["saved_crops"], 1)
        self.assertEqual(detector.calls, 2)
        self.assertEqual(result["quality_counts"]["retry_attempts"], 1)
        self.assertEqual(result["quality_counts"]["retry_successes"], 1)

    def test_retry_budget_is_respected_for_no_face_video(self):
        api = module(self, "extraction")
        ws, cfg, reader = self.setup_video()
        class Detector:
            calls = 0
            def detect(self, image): self.calls += 1; return []
        detector = Detector()
        result = api.extract("fixture", cfg, ws, detector=detector, reader_factory=reader)
        self.assertEqual(detector.calls, 5)
        self.assertEqual(result["saved_crops"], 0)
        self.assertEqual(result["quality_counts"]["segments_exhausted"], 1)

    def test_candidates_are_deterministic_and_stay_in_segment(self):
        api = module(self, "extraction")
        first = api.temporal_candidates(100, 4, seed=42, video_id="video", attempts=5)
        self.assertEqual(first, api.temporal_candidates(100, 4, seed=42, video_id="video", attempts=5))
        for segment, candidates in enumerate(first):
            self.assertEqual(len(candidates), 5)
            self.assertEqual(len(set(candidates)), 5)
            self.assertTrue(all(segment*25 <= i < (segment+1)*25 for i in candidates))
