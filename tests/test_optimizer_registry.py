import unittest
import torch
from tests.support import module


class OptimizerRegistryTests(unittest.TestCase):
    def test_asam_adaptive_wrapper_and_state(self):
        api=module(self,'optimizers');parameter=torch.nn.Parameter(torch.tensor([1.]))
        optimizer=api.ASAMWrapper(torch.optim.SGD([parameter],lr=.1),rho=.05)
        def closure():
            optimizer.zero_grad();loss=parameter.square().sum();loss.backward();return loss
        optimizer.step(closure)
        self.assertTrue(optimizer.adaptive)
        self.assertTrue(optimizer.state_dict()['adaptive'])
        self.assertLess(float(parameter.detach()),1)

    def test_sam_exception_restores_exact_original_weights(self):
        api=module(self,'optimizers');parameter=torch.nn.Parameter(torch.tensor([1.]))
        optimizer=api.SAMWrapper(torch.optim.SGD([parameter],lr=.1));calls=[]
        def closure():
            calls.append(1)
            if len(calls)==2: raise RuntimeError('fixture second pass')
            optimizer.zero_grad();loss=parameter.square().sum();loss.backward();return loss
        with self.assertRaisesRegex(RuntimeError,'second pass'): optimizer.step(closure)
        self.assertEqual(float(parameter.detach()),1)
        self.assertFalse(optimizer.perturbations)

    def test_sam_second_pass_does_not_double_update_bn_statistics(self):
        api=module(self,'optimizers');train=module(self,'training')
        model=torch.nn.Sequential(torch.nn.BatchNorm1d(2),torch.nn.Linear(2,1))
        optimizer=api.SAMWrapper(torch.optim.SGD(model.parameters(),lr=.01))
        train.train_epoch(model,[(torch.rand(4,2),torch.tensor([0.,1.,0.,1.]))],optimizer,amp=False,freeze_statistics=False)
        self.assertEqual(int(model[0].num_batches_tracked),1)
    def test_each_available_optimizer_steps(self):
        api=module(self,'optimizers')
        for name in ('adamw','adam','sgd','sgd_nesterov','rmsprop','adagrad'):
            with self.subTest(name=name):
                parameter=torch.nn.Parameter(torch.tensor([1.]))
                optimizer=api.make_optimizer([parameter],{'optimizer':name,'head_lr':.1,'weight_decay':0,'momentum':.9})
                parameter.square().sum().backward();optimizer.step()
                self.assertLess(float(parameter.detach()),1)

    def test_unavailable_optimizer_errors_are_actionable(self):
        from unittest.mock import patch
        api=module(self,'optimizers')
        for name,package in (('lamb','torch_optimizer'),('lion','lion_pytorch'),('sophia','sophia_optimizer')):
            with self.subTest(name=name),patch('deepfake_detection.training.optimizers.importlib.import_module',side_effect=ImportError('missing')),self.assertRaisesRegex(ImportError,package):
                api.make_optimizer([torch.nn.Parameter(torch.ones(1))],{'optimizer':name})

    def test_lookahead_tracks_slow_weights_and_roundtrips(self):
        api=module(self,'optimizers');parameter=torch.nn.Parameter(torch.tensor([1.]))
        optimizer=api.LookaheadWrapper(torch.optim.SGD([parameter],lr=.1),k=2,alpha=.5)
        for _ in range(2):
            optimizer.zero_grad();parameter.sum().backward();optimizer.step()
        self.assertAlmostEqual(float(parameter.detach()),.9,places=6)
        state=optimizer.state_dict()
        other=torch.nn.Parameter(torch.tensor([.9]));restored=api.LookaheadWrapper(torch.optim.SGD([other],lr=.1),k=2,alpha=.5)
        restored.load_state_dict(state)
        self.assertEqual(restored.steps,2)

    def test_sam_requires_two_forward_backward_passes(self):
        api=module(self,'optimizers');parameter=torch.nn.Parameter(torch.tensor([1.]))
        optimizer=api.SAMWrapper(torch.optim.SGD([parameter],lr=.1),rho=.05)
        calls=[]
        def closure():
            calls.append(1);optimizer.zero_grad();loss=parameter.square().sum();loss.backward();return loss
        optimizer.step(closure)
        self.assertEqual(len(calls),2)
        self.assertAlmostEqual(float(parameter.detach()),.79,places=5)
        with self.assertRaisesRegex(RuntimeError,'closure'): optimizer.step()

    def test_train_epoch_sam_replays_accumulation_and_restores_weights(self):
        api=module(self,'optimizers');train=module(self,'training')
        class CountModel(torch.nn.Linear):
            calls=0
            def forward(self,inputs): self.calls+=1;return super().forward(inputs)
        model=CountModel(1,1)
        optimizer=api.SAMWrapper(torch.optim.SGD(model.parameters(),lr=.1),rho=.05)
        result=train.train_epoch(model,[(torch.ones(2,1),torch.ones(2,1)),(torch.ones(1,1),torch.zeros(1,1))],optimizer,amp=False,accumulation=4)
        self.assertEqual(model.calls,4)
        self.assertEqual(result['optimizer_updates'],1)
        self.assertFalse(optimizer.perturbations)
