import unittest
from tests.support import module
from tests import test_extraction_retries as retry_support


class AutoVerifyTests(unittest.TestCase):
    def test_flag_marks_fake_and_default_still_requires_manual_review(self):
        api = module(self, "extraction")
        ws, cfg, reader = retry_support.RetryTests.setup_video(self)
        paths = module(self, "paths")
        rows = paths.read_jsonl(ws.manifest("fixture") / "videos.jsonl")
        rows[0]["label"] = 0
        paths.write_jsonl(ws.manifest("fixture") / "videos.jsonl", rows, ws)
        cfg["extraction"]["train_fake"] = 1
        preprocessing = module(self, "preprocessing")
        face = module(self, "detector").Face
        class Detector:
            def detect(self, image): return [face([10,10,90,90], .99, [], "accepted", preprocessing.square_crop(image, [10,10,90,90]))]
        result = api.extract("fixture", cfg, ws, detector=Detector(), reader_factory=reader)
        manual = paths.read_jsonl(ws.manifest("fixture") / "crops.jsonl")[0]
        self.assertEqual(manual["extraction_status"], "subject_unverified")
        paths.write_jsonl(ws.manifest("automatic") / "videos.jsonl", rows, ws)
        result = api.extract("automatic", cfg, ws, detector=Detector(), reader_factory=reader, auto_verify_single_face=True)
        auto = paths.read_jsonl(ws.manifest("automatic") / "crops.jsonl")[0]
        self.assertTrue(auto["subject_verified"])
        self.assertTrue(auto["auto_verified"])
        self.assertEqual(auto["verification_source"], "auto_single_face_heuristic")
        self.assertEqual(auto["review_note"], "single-face heuristic")
        self.assertEqual(auto["extraction_status"], "completed")
        self.assertEqual(result["auto_verified_crops"], 1)

    def test_multiple_detected_faces_never_auto_verify(self):
        api = module(self, "extraction")
        preprocessing = module(self, "preprocessing")
        face = module(self, "detector").Face
        from PIL import Image
        image = Image.new("RGB", (200,200))
        first = face([20,20,100,100], .99, [], "accepted", preprocessing.square_crop(image,[20,20,100,100]))
        near = face([105,20,120,50], .4, [], "low_quality", None)
        self.assertFalse(api.single_face_heuristic([first,near], .25))
        self.assertTrue(api.single_face_heuristic([first], .25))
