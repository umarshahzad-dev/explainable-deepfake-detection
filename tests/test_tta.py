import unittest
import numpy as np
from PIL import Image


class TTATests(unittest.TestCase):
    def test_153_mc_dropout_uncertainty_bn_rng_and_modes(self):
        import torch
        from deepfake_detection.evaluation.test_time_augmentation import mc_dropout_predict
        model=torch.nn.Sequential(torch.nn.BatchNorm1d(3),torch.nn.Dropout(.5),torch.nn.Linear(3,1))
        model[-1].weight.data.fill_(1); model.eval(); before=model[0].running_mean.clone()
        state=torch.get_rng_state().clone()
        result=mc_dropout_predict(model,torch.ones(1,3),samples=30)
        repeat=mc_dropout_predict(model,torch.ones(1,3),samples=30)
        self.assertEqual(result,repeat); self.assertGreater(result['real_std'][0],0)
        self.assertTrue(torch.equal(state,torch.get_rng_state()))
        self.assertTrue(torch.equal(before,model[0].running_mean)); self.assertFalse(model[1].training)
        with self.assertRaisesRegex(ValueError,'dropout'): mc_dropout_predict(torch.nn.Linear(3,1),torch.ones(1,3))
    def test_148_evaluation_dispatch_preserves_corruption_and_labels(self):
        from deepfake_detection.evaluation.evaluator import inference_predictions
        from deepfake_detection.data.datasets import CropDataset
        from tests.fixtures import crop_fixture,TinyClassifier
        from unittest.mock import Mock
        ws,cfg,rows=crop_fixture(self); cfg['inference']={'tta_hflip':True}
        dataset=CropDataset(rows,cfg,workspace=ws); dataset.augment=Mock(side_effect=lambda image:image)
        output=inference_predictions(TinyClassifier(),dataset,cfg,device='cpu')
        self.assertEqual(output['labels'],[row['label'] for row in rows])
        self.assertEqual(dataset.augment.call_count,len(rows)); self.assertTrue(output['tta_applied'])
    def test_148_probability_average_calibration_and_model_modes(self):
        import torch
        from deepfake_detection.evaluation.test_time_augmentation import combined_tta,tta_views
        from deepfake_detection.data.preprocessing import model_tensor
        class Pixel(torch.nn.Module):
            def forward(self,x): return (x[:,0,0,0]*20-3).reshape(-1,1)
        model=Pixel().train(); image,cfg=self.fixture(); cfg['inference']['tta_hflip']=True
        probabilities=[]
        for view in tta_views(image,cfg): probabilities.append(float(model(model_tensor(view['image'],cfg['model']).unsqueeze(0)).sigmoid()))
        mean=sum(probabilities)/len(probabilities)
        result=combined_tta(model,image,cfg,device='cpu')
        self.assertAlmostEqual(result['real_probability'],mean,places=6); self.assertTrue(model.training)
        scaled=combined_tta(model,image,cfg,device='cpu',temperature=2)
        self.assertAlmostEqual(scaled['real_probability'],float(torch.sigmoid(torch.tensor(result['logit']/2))),places=6)

    def test_148_full_master_cartesian_bounded_batches(self):
        import torch
        from deepfake_detection.evaluation.test_time_augmentation import combined_tta
        class Constant(torch.nn.Module):
            def forward(self,x):
                self.asserted_batch=len(x); return torch.zeros(len(x),1)
        model=Constant(); image,cfg=self.fixture()
        cfg['inference'].update(tta_hflip=True,tta_vflip=True,tta_multi_crop=True,tta_multi_scale=[8,9,10],tta_rotation=[0,-5,5])
        result=combined_tta(model,image,cfg,device='cpu')
        self.assertEqual(result['views'],180); self.assertEqual(model.asserted_batch,1)
        self.assertEqual(result['real_probability'],.5)
    def test_147_color_jitter_deterministic_identity_plus_view(self):
        from deepfake_detection.evaluation.test_time_augmentation import tta_views
        image,cfg=self.fixture(); cfg['inference']['tta_color_jitter']=True
        first=list(tta_views(image,cfg)); second=list(tta_views(image,cfg))
        self.assertEqual(len(first),2)
        self.assertTrue(np.array_equal(np.asarray(first[0]['image']),np.asarray(image)))
        self.assertFalse(np.array_equal(np.asarray(first[1]['image']),np.asarray(image)))
        self.assertTrue(all(np.array_equal(np.asarray(a['image']),np.asarray(b['image'])) for a,b in zip(first,second)))
    def test_146_rotation_views_and_zero_identity(self):
        from deepfake_detection.evaluation.test_time_augmentation import tta_views
        image,cfg=self.fixture(); cfg['inference']['tta_rotation']=[-5,0,5]
        views=list(tta_views(image,cfg)); self.assertEqual([view['rotation'] for view in views],[-5,0,5])
        self.assertTrue(np.array_equal(np.asarray(views[1]['image']),np.asarray(image)))
        self.assertTrue(all(view['image'].size==image.size for view in views))
        cfg['inference']['tta_rotation']=[float('nan')]
        with self.assertRaises(ValueError): list(tta_views(image,cfg))
    def test_145_scales_preserve_source_pixels_until_transform(self):
        from deepfake_detection.evaluation.test_time_augmentation import tta_views
        image,cfg=self.fixture(); cfg['inference']['tta_multi_scale']=[224,256,299]
        views=list(tta_views(image,cfg)); self.assertEqual([view['input_size'] for view in views],[224,256,299])
        self.assertTrue(all(np.array_equal(np.asarray(view['image']),np.asarray(image)) for view in views))
        cfg['inference']['tta_multi_scale']=[0]
        with self.assertRaises(ValueError): list(tta_views(image,cfg))
    def test_144_five_crops_geometry(self):
        from deepfake_detection.evaluation.test_time_augmentation import tta_views
        image,cfg=self.fixture(); cfg['inference']['tta_multi_crop']=True
        views=list(tta_views(image,cfg)); self.assertEqual(len(views),5)
        self.assertTrue(all(view['image'].size==(7,7) for view in views))
        self.assertEqual(views[1]['crop_box'],(0,0,7,7)); self.assertEqual(views[-1]['crop_box'],(1,1,8,8))
        self.assertTrue(np.array_equal(np.asarray(views[-1]['image']),np.asarray(image)[1:,1:]))
    def test_143_vertical_cartesian_views(self):
        from deepfake_detection.evaluation.test_time_augmentation import tta_views
        image,cfg=self.fixture(); cfg['inference'].update(tta_hflip=True,tta_vflip=True)
        views=list(tta_views(image,cfg)); self.assertEqual(len(views),4)
        self.assertTrue(np.array_equal(np.asarray(views[-1]['image']),np.asarray(image)[::-1,::-1]))
    def fixture(self):
        return Image.fromarray(np.arange(8*8*3,dtype=np.uint8).reshape(8,8,3)),{'model':{'input_size':8,'mean':[0,0,0],'std':[1,1,1]},'inference':{}}

    def test_142_horizontal_views_and_disabled_identity(self):
        from deepfake_detection.evaluation.test_time_augmentation import tta_views
        image,cfg=self.fixture(); self.assertEqual(len(list(tta_views(image,cfg))),1)
        cfg['inference']['tta_hflip']=True; views=list(tta_views(image,cfg))
        self.assertEqual(len(views),2)
        self.assertTrue(np.array_equal(np.asarray(views[1]['image']),np.asarray(image)[:,::-1]))
