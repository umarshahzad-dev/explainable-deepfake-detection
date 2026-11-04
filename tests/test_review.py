import unittest
from PIL import Image
from tests.support import module, workspace


class ReviewTests(unittest.TestCase):
    def test_review_template_never_auto_verifies_people(self):
        api = module(self, "review")
        paths = module(self, "paths")
        root = workspace(self)
        path = root / "crops/real.png"
        Image.new("RGB", (299, 299), "white").save(path)
        rows = [{"label": 1, "parents": ["ffpp:004"], "crop_path": str(path), "sample_id": "real"},
                {"label": 0, "parents": ["ffpp:fake"], "crop_path": str(path), "sample_id": "fake"}]
        result = api.contact_review(rows, paths.Workspace(root), limit=5)
        registry = paths.read_json(result["template_path"])
        self.assertEqual(list(registry["identities"]), ["ffpp:004"])
        self.assertFalse(registry["identities"]["ffpp:004"]["verified"])
        self.assertIsNone(registry["identities"]["ffpp:004"]["person_id"])
        self.assertFalse(registry["subjects"]["fake"]["verified"])
        self.assertTrue(result["subject_sheets"])
        self.assertTrue(result["sheets"])


if __name__ == "__main__":
    unittest.main()
