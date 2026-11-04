import unittest
import torch
from tests.test_attention import ToyBackbone


class FrequencyTests(unittest.TestCase):
    def test_61_haar_constant_highbands_and_odd_size(self):
        from deepfake_detection.models.frequency import WaveletBranch
        from deepfake_detection.models.architecture import enhance
        block=WaveletBranch(12); bands=block.transform(torch.ones(2,3,9,11))
        self.assertEqual(bands.shape,(2,12,5,6)); self.assertEqual(float(bands[:,3:].abs().max()),0)
        x=torch.randn(2,3,9,11,requires_grad=True); y=block(x)
        self.assertEqual(y.shape,(2,12)); y.sum().backward(); self.assertGreater(float(x.grad.abs().sum()),0)
        model=enhance(ToyBackbone(),{'use_wavelet':True,'frequency_features':12})
        self.assertIn('wavelet',model.streams); self.assertEqual(model(x.detach()).shape,(2,1))
    def test_60_residual_constant_and_gradient(self):
        from deepfake_detection.models.frequency import NoiseResidualBranch
        from deepfake_detection.models.architecture import enhance
        block=NoiseResidualBranch(12)
        self.assertLess(float(block.residual(torch.ones(2,3,9,11)).abs().max()),1e-6)
        x=torch.randn(2,3,9,11,requires_grad=True); y=block(x)
        self.assertEqual(y.shape,(2,12)); y.sum().backward(); self.assertGreater(float(x.grad.abs().sum()),0)
        model=enhance(ToyBackbone(),{'use_prnu':True,'frequency_features':12})
        self.assertIn('residual',model.streams); self.assertEqual(model(x.detach()).shape,(2,1))
    def test_59_dct_branch_gradient_and_model(self):
        from deepfake_detection.models.frequency import DCTBranch
        from deepfake_detection.models.architecture import enhance
        x=torch.randn(2,3,19,17,requires_grad=True); y=DCTBranch(12)(x)
        self.assertEqual(y.shape,(2,12)); y.sum().backward(); self.assertGreater(float(x.grad.abs().sum()),0)
        model=enhance(ToyBackbone(),{'use_dct':True,'frequency_features':12})
        self.assertIn('dct',model.streams); self.assertEqual(model(x.detach()).shape,(2,1))
    def test_58_fft_branch_odd_size_gradient_and_model(self):
        from deepfake_detection.models.frequency import FFTBranch
        from deepfake_detection.models.architecture import enhance
        x=torch.randn(2,3,19,17,requires_grad=True); block=FFTBranch(12)
        y=block(x); self.assertEqual(y.shape,(2,12)); self.assertTrue(torch.isfinite(y).all())
        y.sum().backward(); self.assertGreater(float(x.grad.abs().sum()),0)
        model=enhance(ToyBackbone(),{'use_frequency':True,'frequency_features':12})
        self.assertEqual(model(torch.randn(2,3,19,17)).shape,(2,1))
        self.assertIn('fft',model.streams)
    def test_57_gabor_fixed_bank_and_model(self):
        from deepfake_detection.models.frequency import GaborBank
        from deepfake_detection.models.architecture import enhance
        block=GaborBank(features=12); x=torch.randn(2,3,17,19,requires_grad=True)
        y=block(x); self.assertEqual(y.shape,(2,12)); y.sum().backward()
        self.assertGreater(float(x.grad.abs().sum()),0)
        self.assertFalse(block.kernels.requires_grad)
        self.assertLess(float(block.kernels.sum((1,2,3)).abs().max()),1e-5)
        enabled=enhance(ToyBackbone(),{'use_gabor':True,'frequency_features':12})
        self.assertEqual(enabled(torch.randn(2,3,17,19)).shape,(2,1))
        base=ToyBackbone(); self.assertIs(enhance(base,{}),base)
