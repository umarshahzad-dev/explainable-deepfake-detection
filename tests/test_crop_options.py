import copy
import unittest
import numpy as np
from PIL import Image
from tests.support import module


class CropOptionTests(unittest.TestCase):
    def test_three_margins_create_three_frame_crops_and_unique_names(self):
        from tests import test_extraction_retries as support
        from unittest.mock import patch
        ws,cfg,reader=support.RetryTests.setup_video(self)
        cfg['preprocessing']['multi_margin']=[1.3,1.5,1.7]
        preprocessing=module(self,'preprocessing')
        Face=module(self,'detector').Face
        class Detector:
            def detect(self,image): return [Face([10,10,90,90],.99,[],'accepted',preprocessing.square_crop(image,[10,10,90,90]))]
        result=module(self,'extraction').extract('fixture',cfg,ws,detector=Detector(),reader_factory=reader)
        rows=module(self,'paths').read_jsonl(ws.manifest('fixture')/'crops.jsonl')
        self.assertEqual(result['saved_crops'],3)
        self.assertEqual({r['crop_margin'] for r in rows},{1.3,1.5,1.7})
        self.assertEqual(len({r['sample_id'] for r in rows}),3)
        self.assertTrue(all('_m' in r['crop_path'] for r in rows))
    def test_image_center_and_box_center_are_valid_distinct_crops(self):
        api=module(self,'preprocessing')
        image=Image.new('RGB',(200,200),'blue')
        image.paste(Image.new('RGB',(60,60),'red'),(10,10))
        center=api.square_crop(image,[10,10,70,70],anchor='center')
        box=api.square_crop(image,[10,10,70,70],anchor='box_center')
        self.assertEqual(center.image.size,box.image.size)
        self.assertEqual(center.geometry.x,61)
        self.assertEqual(box.geometry.x,1)
        self.assertNotEqual(center.image.getpixel((149,149)),box.image.getpixel((149,149)))

    def test_new_master_defaults_to_center_anchor(self):
        cfg=module(self,'config').load_config('configs/archive/xception-final-original.json')
        self.assertEqual(cfg['preprocessing']['crop_anchor'],'center')

    def test_detector_honors_anchor_and_rejects_unknown_anchor(self):
        cfg=copy.deepcopy(module(self,'config').load_config()['preprocessing'])
        cfg['crop_anchor']='center'
        class Backend:
            def detect(self,image,landmarks): return np.array([[10,10,90,90]]),np.array([.99]),None
        detector=module(self,'detector').FaceDetector(cfg,backend=Backend())
        self.assertEqual(detector.detect(Image.new('RGB',(200,200)))[0].crop.geometry.x,48)
        cfg['crop_anchor']='unknown'
        with self.assertRaisesRegex(ValueError,'anchor'): detector.detect(Image.new('RGB',(200,200)))
    def test_configured_margin_13_vs_15_changes_geometry(self):
        cfg=module(self,'config').load_config('configs/archive/xception-final-original.json')['preprocessing']
        cfg=copy.deepcopy(cfg);cfg['alignment']='none'
        cfg.pop('multi_margin',None)  # Isolate margin_factor rather than an explicit margin-list override.
        class Backend:
            def detect(self,image,landmarks): return np.array([[20,20,100,100]]),np.array([.99]),None
        detector=module(self,'detector')
        image=Image.new('RGB',(150,150),'white')
        larger=detector.FaceDetector(cfg,backend=Backend()).detect(image)[0].crop
        cfg['margin_factor']=1.3
        smaller=detector.FaceDetector(cfg,backend=Backend()).detect(image)[0].crop
        self.assertEqual(larger.geometry.side,120)
        self.assertEqual(smaller.geometry.side,104)
        self.assertNotEqual(larger.geometry,smaller.geometry)
