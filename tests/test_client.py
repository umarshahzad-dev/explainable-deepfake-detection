import unittest
from tests.support import module


class ClientTests(unittest.TestCase):
    def test_client_sends_raw_image_and_propagates_api_result(self):
        api = module(self, "client")
        class Transport:
            def post(self, url, **kwargs):
                valid = kwargs.get("data") == b"image" and kwargs["headers"]["Content-Type"] == "image/png" and "files" not in kwargs
                class Response:
                    status_code = 200 if valid else 422
                    def json(self): return {"faces": [], "status": "no_faces"} if valid else {"detail": "bad transport"}
                return Response()
        result = api.analyze_upload("http://local.invalid", b"image", "image/png", transport=Transport())
        self.assertEqual(result["status"], "no_faces")


if __name__ == "__main__":
    unittest.main()
