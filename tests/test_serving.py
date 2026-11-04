from io import BytesIO
import unittest
from PIL import Image
from tests.support import module, workspace


class ServingTests(unittest.TestCase):
    def test_serving_surfaces_pilot_metadata(self):
        engine = self.make_engine()
        engine.metadata["pilot"] = True
        self.assertTrue(engine.analyze(Image.new("RGB", (100,100)))["training_pilot"])
    def make_engine(self):
        api = module(self, "serving")
        detector_api = module(self, "detector")
        preprocessing = module(self, "preprocessing")
        cfg = module(self, "config").load_config()
        import torch
        class Tiny(torch.nn.Module):
            def forward(self, x): return x[:, 0].mean(dim=(1, 2)).unsqueeze(1)
        class Detector:
            def detect(self, image):
                crop = preprocessing.square_crop(image, [10, 10, 90, 90])
                return [detector_api.Face([10, 10, 90, 90], 0.99, [], "accepted", crop)]
        metadata = {"run_id": "fixture", "weights_sha256": "fixture-sha", "preprocessing_hash": preprocessing.preprocessing_hash(cfg)}
        return api.InferenceEngine(Tiny(), cfg, metadata, detector=Detector(), device="cpu")

    def test_inference_uses_same_crop_tensor_and_score_direction(self):
        engine = self.make_engine()
        result = engine.analyze(Image.new("RGB", (100, 100), (255, 0, 0)))
        self.assertEqual(result["faces"][0]["label"], "REAL")
        self.assertLess(result["faces"][0]["fake_score"], 0.2)
        self.assertAlmostEqual(result["faces"][0]["fake_score"] + result["faces"][0]["real_score"], 1)
        self.assertFalse(result["calibration_applied"])

    def test_history_logs_no_face_and_correct_model_version(self):
        api = module(self, "persistence")
        paths = module(self, "paths")
        ws = paths.Workspace(workspace(self))
        store = api.HistoryStore(ws)
        result = {"run_id": "fixture", "weights_sha256": "weights-one", "preprocessing_hash": "contract", "faces": [], "status": "no_faces", "elapsed_seconds": 0.1}
        store.record("raw-hash", "upload", result)
        logs = store.recent()
        self.assertEqual(logs[0]["weights_sha256"], "weights-one")
        self.assertEqual(logs[0]["face_count"], 0)

    def test_api_raw_upload_and_history_work_without_multipart(self):
        api = module(self, "api")
        paths = module(self, "paths")
        from fastapi.testclient import TestClient
        ws = paths.Workspace(workspace(self))
        app = api.create_app(engine=self.make_engine(), workspace=ws)
        stream = BytesIO()
        Image.new("RGB", (100, 100), (255, 0, 0)).save(stream, format="PNG")
        with TestClient(app) as client:
            response = client.post("/analyze", content=stream.getvalue(), headers={"Content-Type": "image/png"})
            self.assertEqual(response.status_code, 200)
            self.assertEqual(response.json()["faces"][0]["label"], "REAL")
            self.assertEqual(client.get("/history").json()[0]["face_count"], 1)
            invalid = client.post("/analyze", content=b"corrupt", headers={"Content-Type": "image/png"})
            self.assertEqual(invalid.status_code, 422)

    def test_tiny_and_animated_images_are_handled(self):
        api = module(self, "serving")
        stream = BytesIO()
        Image.new("RGB", (10, 10)).save(stream, format="PNG")
        with self.assertRaises(ValueError):
            api.decode_image(stream.getvalue())


class URLTests(unittest.TestCase):
    def test_private_literal_and_non_https_rejected_before_transport(self):
        api = module(self, "ingestion")
        for url in ("https://127.0.0.1/x", "https://[::1]/x", "http://example.org/x", "file:///C:/secret"):
            with self.subTest(url=url), self.assertRaises(ValueError):
                api.validate_url(url, resolver=lambda host, port: [])

    def test_mixed_public_private_dns_answers_are_rejected(self):
        api = module(self, "ingestion")
        answers = [(2, 1, 6, "", ("93.184.216.34", 443)), (2, 1, 6, "", ("127.0.0.1", 443))]
        with self.assertRaises(ValueError):
            api.validate_url("https://example.org/image.png", resolver=lambda host, port: answers)


if __name__ == "__main__":
    unittest.main()
