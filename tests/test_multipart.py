import io
import unittest
from unittest.mock import patch
from PIL import Image
from tests.support import module, workspace
from tests import test_serving as serving_support


def parser_importable():
    try:
        try:
            from python_multipart.multipart import parse_options_header
        except ImportError:
            from multipart.multipart import parse_options_header
        return True
    except ImportError:
        return False


class MultipartTests(unittest.TestCase):
    def make_app(self):
        api = module(self, "api")
        return api.create_app(engine=serving_support.ServingTests.make_engine(self), workspace=module(self, "paths").Workspace(workspace(self)))

    def test_unavailable_parser_warns_and_raw_upload_still_works(self):
        api = module(self, "api")
        from fastapi.testclient import TestClient
        with patch("app.backend.api.multipart_available", return_value=False), self.assertLogs("app.backend.api", level="WARNING") as logs:
            app = self.make_app()
        self.assertIn("raw-body", " ".join(logs.output))
        stream = io.BytesIO()
        Image.new("RGB", (100,100), "red").save(stream, format="PNG")
        with TestClient(app) as client:
            self.assertEqual(client.post("/analyze", content=stream.getvalue(), headers={"Content-Type":"image/png"}).status_code, 200)
            self.assertEqual(client.post("/analyze", files={"file": ("image.png", stream.getvalue(), "image/png")}).status_code, 200)

    @unittest.skipUnless(parser_importable(), "Optional python-multipart is not installed")
    def test_optional_multipart_image_upload(self):
        from fastapi.testclient import TestClient
        stream = io.BytesIO()
        Image.new("RGB", (100,100), "red").save(stream, format="PNG")
        with TestClient(self.make_app()) as client:
            response = client.post("/analyze", files={"file": ("image.png", stream.getvalue(), "image/png")}, data={"explain": "false"})
            self.assertEqual(response.status_code, 200)
            self.assertEqual(response.json()["faces"][0]["label"], "REAL")
