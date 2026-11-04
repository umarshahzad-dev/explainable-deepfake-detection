import unittest
import torch
from deepfake_detection.common.configuration import load_config


class RegularizationTests(unittest.TestCase):
    def test_89_weight_noise_seeded_frozen_and_disabled(self):
        from deepfake_detection.training.trainer import add_weight_noise,train_epoch
        from unittest.mock import patch
        model=torch.nn.Linear(2,1); model.weight.data.zero_(); model.bias.requires_grad_(False); before=model.bias.clone()
        torch.manual_seed(42); add_weight_noise(model,.01); first=model.weight.detach().clone()
        self.assertGreater(float(first.abs().sum()),0); self.assertTrue(torch.equal(model.bias,before))
        model.weight.data.zero_(); torch.manual_seed(42); add_weight_noise(model,.01); self.assertTrue(torch.equal(model.weight,first))
        add_weight_noise(model,0); self.assertTrue(torch.equal(model.weight,first))
        batches=[(torch.ones(2,2),torch.ones(2,1))]
        with patch('deepfake_detection.training.trainer.add_weight_noise') as noise:
            train_epoch(model,batches,torch.optim.SGD(model.parameters(),lr=.1),amp=False,recipe={'use_weight_noise':True,'weight_noise_std':.02})
            noise.assert_called_once_with(model,.02)
    def test_82_elastic_penalty_and_train_epoch_routing(self):
        from deepfake_detection.training.trainer import train_epoch,regularization_loss
        model=torch.nn.Linear(1,1,bias=False); model.weight.data.fill_(2)
        expected=torch.nn.functional.binary_cross_entropy_with_logits(torch.tensor([[2.]]),torch.ones(1,1))+.1*2+.2*4
        self.assertAlmostEqual(float(regularization_loss(model,.1,.2)),1.,places=6)
        result=train_epoch(model,[(torch.ones(1,1),torch.ones(1,1))],torch.optim.SGD(model.parameters(),lr=.01),amp=False,clip_norm=None,recipe={'regularization_l1':.1,'regularization_l2':.2})
        self.assertAlmostEqual(result['loss'],float(expected),places=6)
        with self.assertRaises(ValueError): regularization_loss(model,-1,0)
    def test_82_frozen_parameters_excluded(self):
        from deepfake_detection.training.trainer import regularization_loss
        model=torch.nn.Linear(1,1); model.weight.requires_grad_(False); model.bias.data.fill_(2)
        self.assertEqual(float(regularization_loss(model,1,0)),2)
    def test_79_master_dropout_tuning(self):
        cfg=load_config('configs/archive/xception-final-original.json')
        self.assertEqual(cfg['model']['dropout'],.35)
        self.assertEqual(cfg['model']['classifier_dropout'],.35)
    def test_80_master_drop_path_tuning(self):
        cfg=load_config('configs/archive/xception-final-original.json')
        self.assertEqual(cfg['model']['drop_path'],.1)
        self.assertEqual(cfg['model']['stochastic_depth_schedule'],'linear')
    def test_81_weight_decay_reaches_optimizer(self):
        from tests.fixtures import TinyClassifier
        from deepfake_detection.training.trainer import _optimizer
        cfg=load_config('configs/archive/xception-final-original.json')
        optimizer=_optimizer(TinyClassifier(),cfg,'finetune')
        self.assertTrue(all(group['weight_decay']==.0005 for group in optimizer.param_groups))
