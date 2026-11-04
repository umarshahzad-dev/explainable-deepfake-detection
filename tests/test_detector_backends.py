import copy
import unittest
from tests.support import module


class DetectorBackendTests(unittest.TestCase):
    def check_backend(self,name,package):
        cfg=copy.deepcopy(module(self,'config').load_config()['preprocessing'])
        cfg['detector_backend']=name
        with self.assertRaisesRegex(NotImplementedError,package): module(self,'detector').FaceDetector(cfg,device='cpu')
    def test_retinaface_requirement(self): self.check_backend('retinaface','retina-face')
    def test_scrfd_requirement(self): self.check_backend('scrfd','insightface')
    def test_yoloface_requirement(self): self.check_backend('yoloface','ultralytics')
    def test_unknown_backend_rejected_before_detection(self):
        cfg=copy.deepcopy(module(self,'config').load_config()['preprocessing']);cfg['detector_backend']='wrong'
        with self.assertRaisesRegex(ValueError,'backend'): module(self,'detector').FaceDetector(cfg,device='cpu')
