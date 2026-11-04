import unittest
import torch
from unittest.mock import patch


class PrecisionTests(unittest.TestCase):
    def test_197_gradient_clip_bounds_remainder_update(self):
        from deepfake_detection.training.trainer import train_epoch
        model=torch.nn.Linear(1,1,bias=False); model.weight.data.zero_()
        optimizer=torch.optim.SGD(model.parameters(),lr=1)
        train_epoch(model,[(torch.full((2,1),100.),torch.ones(2))],optimizer,device='cpu',amp=False,clip_norm=.2,accumulation=8)
        self.assertAlmostEqual(float(model.weight.detach()),.2,places=5)
    def test_194_unsupported_rejected_before_model_construction(self):
        from tests.fixtures import crop_fixture
        from deepfake_detection.training.trainer import run_training
        ws,cfg,rows=crop_fixture(self); cfg['training'].update(amp=True,amp_dtype='bf16')
        with patch('torch.cuda.get_device_capability',return_value=(7,5)),patch('deepfake_detection.training.trainer.initialize_model') as factory:
            with self.assertRaisesRegex(ValueError,'BF16'): run_training('unsupported',cfg,rows,ws,device='cuda')
        factory.assert_not_called()

    def test_194_nonfinite_bf16_gradients_do_not_update_weights(self):
        from deepfake_detection.training.trainer import train_epoch
        model=torch.nn.Linear(3,1); before=model.weight.detach().clone()
        model.weight.register_hook(lambda gradient:torch.full_like(gradient,float('inf')))
        optimizer=torch.optim.SGD(model.parameters(),lr=.1)
        with patch('deepfake_detection.training.trainer.amp_settings',return_value=(True,torch.bfloat16,False)):
            with self.assertRaisesRegex(FloatingPointError,'BF16'): train_epoch(model,[(torch.ones(2,3),torch.tensor([0,1]))],optimizer,device='cpu',amp=True,recipe={'amp_dtype':'bf16'})
        self.assertTrue(torch.equal(before,model.weight))
    def test_194_bf16_native_support_and_scaler_selection(self):
        from deepfake_detection.training.precision import amp_settings
        with patch('torch.cuda.get_device_capability',return_value=(7,5)):
            with self.assertRaisesRegex(ValueError,'BF16'): amp_settings(True,'cuda',{'amp_dtype':'bfloat16'})
        with patch('torch.cuda.get_device_capability',return_value=(8,0)):
            self.assertEqual(amp_settings(True,'cuda',{'amp_dtype':'bfloat16'}),(True,torch.bfloat16,False))
        self.assertEqual(amp_settings(True,'cuda',{}),(True,torch.float16,True))
        self.assertEqual(amp_settings(True,'cpu',{}),(False,torch.float16,False))
        with self.assertRaises(ValueError): amp_settings(True,'cpu',{'amp_dtype':'bf16'})
        with self.assertRaises(ValueError): amp_settings(False,'cpu',{'amp_dtype':'invalid'})
