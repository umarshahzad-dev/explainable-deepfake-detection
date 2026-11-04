import math
import unittest
import numpy as np
from PIL import Image
from tests.support import module


class LandmarkAlignmentTests(unittest.TestCase):
    def test_aligned_cam_is_reprojected_to_original_frame(self):
        api=module(self,'preprocessing')
        geometry=api.Geometry(10,10,20,100,100,90)
        output=np.asarray(api.overlay_heatmap(Image.new('RGB',(100,100)),geometry,np.ones((20,20)),alpha=1))
        self.assertGreater(output[20,80,0],200)
        self.assertEqual(output[20,20,0],0)
    def points(self): return np.array([[55,65],[105,65],[80,85],[60,110],[100,110]],dtype=float)

    def test_thirty_degree_roll_produces_horizontal_eyes(self):
        api=module(self,'preprocessing')
        angle=math.radians(30)
        rotation=np.array([[math.cos(angle),-math.sin(angle)],[math.sin(angle),math.cos(angle)]])
        points=(self.points()-80)@rotation.T+80
        crop=api.align_by_landmarks(Image.new('RGB',(160,160),'white'),points)
        self.assertAlmostEqual(crop.geometry.rotation_degrees,30,places=5)
        self.assertAlmostEqual(crop.aligned_landmarks[0][1],crop.aligned_landmarks[1][1],places=5)
        self.assertEqual(crop.image.size,(299,299))

    def test_already_horizontal_matches_unrotated_landmark_crop(self):
        api=module(self,'preprocessing')
        image=Image.fromarray(np.tile(np.arange(160,dtype=np.uint8)[None,:,None],(160,1,3)))
        crop=api.align_by_landmarks(image,self.points(),output_size=299,margin=1.5)
        expected=image.crop((42,50,117,125)).resize((299,299),Image.Resampling.BICUBIC)
        self.assertAlmostEqual(crop.geometry.rotation_degrees,0)
        np.testing.assert_allclose(np.asarray(crop.image),np.asarray(expected),atol=1)

    def test_missing_or_degenerate_landmarks_have_clear_errors(self):
        api=module(self,'preprocessing')
        for points in (None,[],[[1,1]]*5):
            with self.subTest(points=points),self.assertRaisesRegex(ValueError,'landmark|eye'):
                api.align_by_landmarks(Image.new('RGB',(160,160)),points)

    def test_detector_selects_alignment_only_when_enabled(self):
        api=module(self,'detector')
        cfg=module(self,'config').load_config()['preprocessing']
        cfg['alignment']='landmark'
        points=self.points()
        class Backend:
            def detect(self,image,landmarks): return np.array([[20,20,140,140]]),np.array([.99]),np.array([points])
        face=api.FaceDetector(cfg,backend=Backend()).detect(Image.new('RGB',(160,160)))[0]
        self.assertIsNotNone(face.crop.aligned_landmarks)

    def test_canonical_pose_is_finite_and_scaled(self):
        api=module(self,'preprocessing')
        points=api.canonical_pose_landmarks(299)
        self.assertEqual(points.shape,(5,2))
        self.assertTrue(np.isfinite(points).all())
        np.testing.assert_allclose(points*2,api.canonical_pose_landmarks(598))
