import unittest
import torch
from tests.test_attention import ToyBackbone


class ArchitectureTests(unittest.TestCase):
    def test_master_xception_forward_and_head_phase_parameters(self):
        import json
        from pathlib import Path
        from deepfake_detection.models.model_loader import architecture
        from deepfake_detection.training.trainer import _optimizer
        torch.set_num_threads(2)
        cfg=json.loads((Path(__file__).parents[1]/'configs/archive/xception-final-original.json').read_text())
        model=architecture(cfg['model']).eval()
        with torch.no_grad(): logits,context=model.forward_with_features(torch.randn(1,3,299,299))
        self.assertEqual(logits.shape,(1,1)); self.assertEqual(context['features'].shape,(1,512))
        optimizer=_optimizer(model,cfg,'head')
        optimized={id(p) for group in optimizer.param_groups for p in group['params']}
        self.assertTrue(all(id(p) in optimized and p.requires_grad for p in model.head_parameters()))
        self.assertTrue(all(not p.requires_grad for p in model.base.parameters()))
    def test_75_activation_replacement_and_disabled(self):
        from deepfake_detection.models.architecture import enhance
        class Activated(ToyBackbone):
            def __init__(self): super().__init__(); self.relu=torch.nn.ReLU()
            def forward_features(self,x): return self.relu(super().forward_features(x))
        for name,kind in [('gelu',torch.nn.GELU),('silu',torch.nn.SiLU)]:
            model=enhance(Activated(),{'activation':name})
            self.assertIsInstance(model.relu,kind); self.assertEqual(model(torch.randn(2,3,11,13)).shape,(2,1))
        self.assertIsInstance(enhance(Activated(),{}).relu,torch.nn.ReLU)
        with self.assertRaises(ValueError): enhance(Activated(),{'activation':'unknown'})
    def test_74_multibranch_resource_error_only_when_enabled(self):
        from deepfake_detection.models.architecture import enhance
        with self.assertRaisesRegex(NotImplementedError,'Multi-branch Xception'):
            enhance(ToyBackbone(),{'use_multi_branch':True})
        base=ToyBackbone(); self.assertIs(enhance(base,{'use_multi_branch':False}),base)
    def test_73_weight_standardization_preserves_original_and_gradient(self):
        from deepfake_detection.models.architecture import enhance
        from torch.nn.utils import parametrize
        base=ToyBackbone(); original=base.conv.weight.detach().clone()
        model=enhance(base,{'use_weight_standardization':True})
        self.assertTrue(parametrize.is_parametrized(model.conv,'weight'))
        self.assertTrue(torch.equal(model.conv.parametrizations.weight.original,original))
        self.assertLess(float(model.conv.weight.mean((1,2,3)).abs().max()),1e-5)
        output=model(torch.randn(2,3,11,13)); self.assertEqual(output.shape,(2,1)); output.sum().backward()
        self.assertGreater(float(model.conv.parametrizations.weight.original.grad.abs().sum()),0)
        self.assertIs(enhance(ToyBackbone(),{'use_weight_standardization':False}).conv.__class__,torch.nn.Conv2d)
    def test_72_stochastic_depth_linear_schedule(self):
        from deepfake_detection.models.architecture import DropPath,inject_drop_path
        class Residual(torch.nn.Module):
            def __init__(self): super().__init__(); self.rep=torch.nn.Identity()
        root=torch.nn.Sequential(*[Residual() for _ in range(4)])
        inject_drop_path(root,.2,schedule='linear')
        rates=[m.probability for m in root.modules() if isinstance(m,DropPath)]
        self.assertEqual(len(rates),4)
        for a,b in zip(rates,[.05,.1,.15,.2]): self.assertAlmostEqual(a,b)
        with self.assertRaises(ValueError): inject_drop_path(root,.2,schedule='unknown')
    def test_71_classifier_dropout_train_eval_and_disable(self):
        from deepfake_detection.models.architecture import enhance
        model=enhance(ToyBackbone(),{'head_hidden':32,'classifier_dropout':.5})
        self.assertIsInstance(model.classifier_dropout,torch.nn.Dropout)
        x=torch.randn(10,3,11,13); model.train(); torch.manual_seed(42); a=model(x); torch.manual_seed(43); b=model(x)
        self.assertFalse(torch.equal(a,b)); model.eval(); self.assertTrue(torch.equal(model(x),model(x)))
        disabled=enhance(ToyBackbone(),{'use_cbam':True,'classifier_dropout':0})
        self.assertEqual(disabled.classifier_dropout.p,0)
    def test_70_deeper_head_embedding_and_gradients(self):
        from deepfake_detection.models.architecture import enhance
        model=enhance(ToyBackbone(),{'head_hidden':512,'classifier_activation':'gelu'})
        logits,context=model.forward_with_features(torch.randn(2,3,11,13))
        self.assertEqual(logits.shape,(2,1)); self.assertEqual(context['features'].shape,(2,512))
        self.assertEqual(model.embedding_dim,512); self.assertEqual(model.get_classifier().in_features,512)
        logits.sum().backward(); self.assertGreater(float(model.feature_projection[0].weight.grad.abs().sum()),0)
        base=ToyBackbone(); self.assertIs(enhance(base,{'head_hidden':0}),base)
    def test_69_drop_path_training_eval_and_residual_only(self):
        from deepfake_detection.models.architecture import DropPath,inject_drop_path,enhance
        x=torch.ones(40,8,3,3); block=DropPath(.5); block.train()
        torch.manual_seed(42); first=block(x); torch.manual_seed(42); self.assertTrue(torch.equal(first,block(x)))
        self.assertEqual(set(first.flatten().tolist()),{0.,2.}); block.eval(); self.assertTrue(torch.equal(block(x),x))
        class Residual(torch.nn.Module):
            def __init__(self): super().__init__(); self.rep=torch.nn.Sequential(torch.nn.Conv2d(8,8,1))
            def forward(self,x): return x+self.rep(x)
        root=torch.nn.Sequential(Residual()); original=root[0].rep
        self.assertEqual(inject_drop_path(root,.1),1); self.assertIs(root[0].rep[0],original)
        root.eval(); self.assertEqual(root(x).shape,x.shape)
        base=ToyBackbone(); self.assertIs(enhance(base,{'drop_path':0}),base)
    def test_68_layernorm_option_with_single_sample(self):
        from deepfake_detection.models.architecture import enhance
        model=enhance(ToyBackbone(),{'use_layernorm':True})
        self.assertIsInstance(model.feature_norm,torch.nn.LayerNorm)
        output=model(torch.randn(1,3,11,13)); self.assertEqual(output.shape,(1,1)); self.assertTrue(torch.isfinite(output).all())
        output.sum().backward(); self.assertGreater(float(model.feature_norm.weight.grad.abs().sum()),0)
    def test_66_multi_head_average_and_all_gradients(self):
        from deepfake_detection.models.architecture import MultiHeadClassifier,enhance
        block=MultiHeadClassifier(8,heads=3); x=torch.randn(2,8)
        expected=torch.stack([head(x) for head in block.heads]).mean(0)
        y=block(x); self.assertTrue(torch.allclose(y,expected)); y.sum().backward()
        self.assertTrue(all(head.weight.grad.abs().sum()>0 for head in block.heads))
        model=enhance(ToyBackbone(),{'use_multi_head':True,'classification_heads':3})
        self.assertIsInstance(model.classifier,MultiHeadClassifier)
        self.assertEqual(model(torch.randn(2,3,11,13)).shape,(2,1))
    def test_65_auxiliary_classifier_supervised_and_off(self):
        from deepfake_detection.models.architecture import enhance
        from deepfake_detection.models.losses import TrainingObjective,objective_forward
        model=enhance(ToyBackbone(),{'use_auxiliary_classifier':True,'fpn_layers':['conv']})
        x=torch.randn(2,3,11,13); targets=torch.tensor([[0.],[1.]])
        logits,context=model.forward_with_features(x)
        self.assertEqual(context['auxiliary_logits'].shape,(2,1))
        expected=torch.nn.functional.binary_cross_entropy_with_logits(logits,targets)+.3*torch.nn.functional.binary_cross_entropy_with_logits(context['auxiliary_logits'],targets)
        _,loss=objective_forward(model,x,targets,TrainingObjective({'auxiliary_loss_weight':.3},embedding_dim=8))
        self.assertTrue(torch.allclose(loss,expected)); loss.backward()
        self.assertGreater(float(model.auxiliary_classifier.weight.grad.abs().sum()),0)
        self.assertNotIn('auxiliary_logits',enhance(ToyBackbone(),{'use_cbam':True}).forward_with_features(x)[1])
    def test_64_fpn_real_taps_shapes_and_gradient(self):
        from deepfake_detection.models.architecture import FPNHead,enhance
        block=FPNHead([4,8],output_features=8,hidden=4)
        a=torch.randn(2,4,11,13,requires_grad=True); b=torch.randn(2,8,5,6,requires_grad=True)
        y=block([a,b]); self.assertEqual(y.shape,(2,8)); y.sum().backward()
        self.assertGreater(float(a.grad.abs().sum()),0); self.assertGreater(float(b.grad.abs().sum()),0)
        model=enhance(ToyBackbone(),{'use_fpn':True,'fpn_layers':['conv'],'fpn_features':4})
        self.assertIsInstance(model.fpn,FPNHead); self.assertEqual(model(torch.randn(2,3,11,13)).shape,(2,1))
        self.assertEqual(len(model.base.conv._forward_hooks),0)
    def test_62_dual_stream_changes_model_and_fuses_gradients(self):
        from deepfake_detection.models.architecture import enhance,DualStreamFusion
        model=enhance(ToyBackbone(),{'use_dual_stream':True,'frequency_features':12})
        self.assertIsInstance(model.fusion,DualStreamFusion)
        self.assertIn('residual',model.streams)
        output=model(torch.randn(2,3,13,15)); self.assertEqual(output.shape,(2,1)); output.sum().backward()
        self.assertGreater(float(model.base.conv.weight.grad.abs().sum()),0)
        self.assertGreater(float(model.streams['residual'].encoder[0].weight.grad.abs().sum()),0)
        base=ToyBackbone(); self.assertIs(enhance(base,{}),base)
