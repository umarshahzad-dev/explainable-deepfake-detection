import unittest
import torch
from tests.support import module


class LossRegistryTests(unittest.TestCase):
    def test_configured_loss_is_used_in_train_epoch(self):
        from unittest.mock import patch
        train=module(self,'training')
        model=torch.nn.Linear(1,1);model.weight.data.zero_();model.bias.data.fill_(1)
        expected=module(self,'losses').LabelSmoothingBCE(.05)(model(torch.ones(2,1)),torch.tensor([[0.],[0.]]))
        result=train.train_epoch(model,[(torch.ones(2,1),torch.tensor([0.,0.]))],torch.optim.SGD(model.parameters(),lr=.01),amp=False,recipe={'loss':'label_smoothing_bce','label_smoothing':.05})
        self.assertAlmostEqual(result['loss'],float(expected.detach()),places=6)

    def test_run_controller_accepts_label_smoothing_loss(self):
        from tests.fixtures import crop_fixture,TinyClassifier
        from unittest.mock import patch
        ws,cfg,rows=crop_fixture(self)
        cfg['training'].update(loss='label_smoothing_bce',label_smoothing=.05)
        with patch('deepfake_detection.training.trainer.initialize_model',return_value=TinyClassifier()):
            result=module(self,'training').run_training('smoothed',cfg,rows,ws,device='cpu')
        self.assertEqual(result['status'],'completed')

    def test_embedding_objective_keeps_bce_and_excludes_soft_samples(self):
        api=module(self,'losses')
        objective=api.TrainingObjective({'loss':'center','auxiliary_loss_weight':.1},embedding_dim=8)
        logits=torch.randn(4,1,requires_grad=True);targets=torch.tensor([[0.],[1.],[.3],[.7]])
        features=torch.randn(4,8,requires_grad=True)
        result=objective(logits,targets,{'features':features})
        result.backward();self.assertGreater(float(logits.grad.abs().sum()),0)
        self.assertGreater(float(features.grad[:2].abs().sum()),0)
        self.assertEqual(float(features.grad[2:].abs().sum()),0)
    def test_binary_losses_are_finite_and_differentiable(self):
        api=module(self,'losses')
        for name in ('bce','label_smoothing_bce','focal','weighted_bce','lovasz','dice','tversky','mse','hinge','adversarial'):
            with self.subTest(name=name):
                logits=torch.tensor([[-1.],[.5],[1.],[-.2]],requires_grad=True)
                labels=torch.tensor([[0.],[1.],[1.],[0.]])
                loss=api.make_loss(name,embedding_dim=8)(logits,labels)
                self.assertTrue(torch.isfinite(loss));loss.backward()
                self.assertTrue(torch.isfinite(logits.grad).all());self.assertGreater(float(logits.grad.abs().sum()),0)

    def test_embedding_losses_are_finite_and_differentiable(self):
        api=module(self,'losses')
        for name in ('contrastive','triplet','arcface','cosface','center','supcon'):
            with self.subTest(name=name):
                features=torch.randn(6,8,requires_grad=True);labels=torch.tensor([0,0,1,1,0,1])
                loss=api.make_loss(name,embedding_dim=8)(features,labels)
                self.assertTrue(torch.isfinite(loss));loss.backward()
                self.assertTrue(torch.isfinite(features.grad).all());self.assertGreater(float(features.grad.abs().sum()),0)

    def test_paired_losses_are_finite_and_differentiable(self):
        api=module(self,'losses')
        for name in ('kl','consistency','frequency_penalty'):
            with self.subTest(name=name):
                inputs=torch.rand(4,3,8,8,requires_grad=True) if name=='frequency_penalty' else torch.randn(4,1,requires_grad=True)
                targets=torch.rand_like(inputs)
                loss=api.make_loss(name,embedding_dim=8)(inputs,targets)
                self.assertTrue(torch.isfinite(loss));loss.backward()
                self.assertTrue(torch.isfinite(inputs.grad).all());self.assertGreater(float(inputs.grad.abs().sum()),0)

    def test_combined_loss_respects_weights(self):
        api=module(self,'losses');logits=torch.tensor([[.3],[-.2]],requires_grad=True);targets=torch.tensor([[1.],[0.]])
        bce=api.make_loss('bce');mse=api.make_loss('mse')
        combined=api.CombinedLoss([bce,mse],[.7,.3])
        self.assertTrue(torch.allclose(combined(logits,targets),.7*bce(logits,targets)+.3*mse(logits,targets)))

    def test_smoothed_bce_handles_soft_labels(self):
        api=module(self,'losses');logits=torch.tensor([[.3],[-.2]],requires_grad=True);targets=torch.tensor([[.7],[.3]])
        expected=torch.nn.functional.binary_cross_entropy_with_logits(logits,.95*targets+.025)
        self.assertTrue(torch.allclose(api.LabelSmoothingBCE(.05)(logits,targets),expected))
