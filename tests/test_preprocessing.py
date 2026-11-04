import unittest
import numpy as np
from PIL import Image
from tests.support import module


class PreprocessingTests(unittest.TestCase):
    def test_master_contract_is_independent_of_classifier_transform(self):
        api = module(self, "preprocessing")
        import copy
        cfg = module(self, "config").load_config()
        other = copy.deepcopy(cfg)
        other["model"].update(input_size=299, mean=[0.5]*3, std=[0.5]*3)
        self.assertEqual(api.preprocessing_hash(cfg), api.preprocessing_hash(other))
    def test_rectangle_becomes_centered_integer_square(self):
        api = module(self, "preprocessing")
        crop = api.square_crop(Image.new("RGB", (20, 20), "white"), [2, 2, 6, 10], margin=1.3, size=299)
        self.assertEqual((crop.geometry.x, crop.geometry.y, crop.geometry.side), (-2, 0, 11))
        self.assertEqual(crop.image.size, (299, 299))

    def test_boundary_padding_is_black_and_does_not_stretch(self):
        api = module(self, "preprocessing")
        crop = api.square_crop(Image.new("RGB", (6, 6), "white"), [-2, 0, 2, 4], margin=1, size=4)
        pixels = np.array(crop.image)
        np.testing.assert_array_equal(pixels[:, :2], np.zeros((4, 2, 3), dtype=np.uint8))
        np.testing.assert_array_equal(pixels[:, 2:], np.full((4, 2, 3), 255, dtype=np.uint8))

    def test_red_rgb_crop_has_correct_channels_and_normalization(self):
        api = module(self, "preprocessing")
        tensor = api.model_tensor(Image.new("RGB", (299, 299), (255, 0, 0)))
        self.assertEqual(tuple(tensor.shape), (3, 224, 224))
        np.testing.assert_allclose(tensor[:, 50, 50].numpy(), [2.2489083, -2.0357143, -1.8044444], atol=1e-6)

    def test_exif_orientation_applied_before_detection(self):
        api = module(self, "preprocessing")
        image = Image.new("RGB", (2, 3))
        image.getexif()[274] = 6
        self.assertEqual(api.to_rgb(image).size, (3, 2))

    def test_invalid_box_rejected(self):
        api = module(self, "preprocessing")
        for box in ([0, 0, 0, 2], [0, 0, float("nan"), 2]):
            with self.subTest(box=box), self.assertRaises(ValueError):
                api.square_crop(Image.new("RGB", (20, 20)), box)


class DetectorTests(unittest.TestCase):
    def test_tiny_input_does_not_reach_mtcnn(self):
        api = module(self, "detector")
        class Backend:
            def detect(self, *args, **kwargs):
                raise AssertionError("Tiny images must be rejected before MTCNN")
        self.assertEqual(api.FaceDetector(backend=Backend()).detect(Image.new("RGB", (10, 10))), [])

    def test_quality_gate_preserves_correct_face_and_landmarks(self):
        api = module(self, "detector")
        class Backend:
            def detect(self, image, landmarks):
                return np.array([[0, 0, 80, 80], [20, 20, 100, 100]]), np.array([0.8, 0.99]), np.array([[[1, 1]] * 5, [[40, 40]] * 5])
        faces = api.FaceDetector(backend=Backend()).detect(Image.new("RGB", (150, 150), "white"))
        self.assertEqual([f.status for f in faces], ["low_quality", "accepted"])
        self.assertIsNone(faces[0].crop)
        self.assertEqual(faces[1].landmarks[0], [40.0, 40.0])
        self.assertEqual(faces[1].crop.image.size, (299, 299))


if __name__ == "__main__":
    unittest.main()
