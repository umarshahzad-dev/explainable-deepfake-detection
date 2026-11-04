import unittest
import torch
from torch import nn
from unittest.mock import patch


class ToyBackbone(nn.Module):
    num_features=8
    def __init__(self):
        super().__init__(); self.conv=nn.Conv2d(3,8,3,padding=1); self.fc=nn.Linear(8,1)
    def forward_features(self,x): return self.conv(x)
    def forward(self,x): return self.fc(self.forward_features(x).mean((2,3)))
    def get_classifier(self): return self.fc


class AttentionTests(unittest.TestCase):
    def test_78_global_context_identity_init_and_gradient(self):
        from deepfake_detection.models.attention import GlobalContext
        from deepfake_detection.models.architecture import enhance
        block=GlobalContext(8); x=torch.randn(2,8,7,9,requires_grad=True)
        y=block(x); self.assertTrue(torch.equal(y,x)); y.sum().backward()
        self.assertGreater(float(block.transform[-1].weight.grad.abs().sum()),0)
        model=enhance(ToyBackbone(),{'use_gcnet':True})
        self.assertIsInstance(model.global_context,GlobalContext); self.assertEqual(model(x.detach()[:,:3]).shape,(2,1))
    def test_77_layer_attention_weights_normalized_and_taps(self):
        from deepfake_detection.models.attention import LayerAttentionWeights
        from deepfake_detection.models.architecture import enhance
        block=LayerAttentionWeights([4,8],8)
        y=block([torch.randn(2,4,11,13),torch.randn(2,8,5,6)])
        self.assertEqual(y.shape,(2,8)); self.assertAlmostEqual(float(block.weights.sum()),1)
        y.sum().backward(); self.assertGreater(float(block.log_weights.grad.abs().sum()),0)
        model=enhance(ToyBackbone(),{'use_layer_attention':True,'fpn_layers':['conv']})
        self.assertIsInstance(model.layer_attention,LayerAttentionWeights)
        self.assertEqual(model(torch.randn(2,3,11,13)).shape,(2,1)); self.assertEqual(len(model.base.conv._forward_hooks),0)
    def test_76_head_se_feature_gate(self):
        from deepfake_detection.models.attention import SqueezeExcite
        from deepfake_detection.models.architecture import enhance
        model=enhance(ToyBackbone(),{'use_se_before_head':True,'head_hidden':32})
        self.assertIsInstance(model.head_se,SqueezeExcite)
        # Keep the fixture's ReLU gate active instead of depending on random init.
        model.head_se.gate[0].weight.data.zero_(); model.head_se.gate[0].bias.data.fill_(1)
        model.head_se.gate[2].weight.data.fill_(.2); model.classifier.weight.data.fill_(.1)
        logits,context=model.forward_with_features(torch.randn(2,3,11,13))
        self.assertEqual(context['features'].shape,(2,32)); logits.sum().backward()
        self.assertGreater(float(model.head_se.gate[0].weight.grad.abs().sum()),0)
        base=ToyBackbone(); self.assertIs(enhance(base,{'use_se_before_head':False}),base)
    def test_67_attention_pool_normalized_and_model(self):
        from deepfake_detection.models.attention import AttentionPooling
        from deepfake_detection.models.architecture import enhance
        block=AttentionPooling(8); block.score.weight.data.zero_(); block.score.bias.data.zero_()
        x=torch.randn(2,8,7,9,requires_grad=True); y=block(x)
        self.assertTrue(torch.allclose(y,x.mean((2,3)),atol=1e-6)); y.sum().backward()
        self.assertGreater(float(x.grad.abs().sum()),0)
        model=enhance(ToyBackbone(),{'use_attention_pooling':True})
        self.assertIsInstance(model.pool,AttentionPooling); self.assertEqual(model(x.detach()[:,:3]).shape,(2,1))
    def test_63_cross_attention_shapes_and_both_stream_gradients(self):
        from deepfake_detection.models.attention import CrossAttentionFusion
        from deepfake_detection.models.architecture import enhance
        block=CrossAttentionFusion(8,12,heads=2)
        x=torch.randn(2,8,7,9,requires_grad=True); f=torch.randn(2,12,requires_grad=True)
        y=block(x,f); self.assertEqual(y.shape,(2,8)); y.sum().backward()
        self.assertGreater(float(x.grad.abs().sum()),0); self.assertGreater(float(f.grad.abs().sum()),0)
        model=enhance(ToyBackbone(),{'use_cross_attention':True,'frequency_features':12})
        self.assertIsInstance(model.cross_attention,CrossAttentionFusion)
        self.assertEqual(model(torch.randn(2,3,11,13)).shape,(2,1))
    def test_56_eca_channels_and_option(self):
        from deepfake_detection.models.attention import ECA
        from deepfake_detection.models.architecture import enhance
        x=torch.randn(2,8,7,7,requires_grad=True); y=ECA(8)(x)
        self.assertEqual(y.shape,x.shape); y.sum().backward(); self.assertGreater(float(x.grad.abs().sum()),0)
        model=enhance(ToyBackbone(),{'use_eca':True})
        self.assertIsInstance(model.eca,ECA); self.assertEqual(model(torch.randn(2,3,9,9)).shape,(2,1))
    def test_55_se_shape_gradient_and_option(self):
        from deepfake_detection.models.attention import SqueezeExcite
        from deepfake_detection.models.architecture import enhance
        x=torch.randn(2,8,7,7,requires_grad=True)
        y=SqueezeExcite(8)(x); self.assertEqual(y.shape,x.shape); y.sum().backward()
        self.assertGreater(float(x.grad.abs().sum()),0)
        model=enhance(ToyBackbone(),{'use_se':True})
        self.assertIsInstance(model.se,SqueezeExcite)
        self.assertEqual(model(torch.randn(2,3,9,9)).shape,(2,1))
    def test_54_cbam_shape_and_gradient(self):
        from deepfake_detection.models.attention import CBAM
        x=torch.randn(2,8,7,7,requires_grad=True); block=CBAM(8,reduction=16)
        y=block(x); self.assertEqual(y.shape,x.shape); y.sum().backward()
        self.assertGreater(float(x.grad.abs().sum()),0)
    def test_54_model_option_on_and_off(self):
        from deepfake_detection.models.model_loader import architecture
        base=ToyBackbone(); cfg={'name':'legacy_xception.tf_in1k','dropout':0}
        with patch('timm.create_model',return_value=base):
            self.assertIs(architecture(cfg),base)
            enabled=architecture({**cfg,'use_cbam':True})
        self.assertIsNot(enabled,base)
        self.assertEqual(enabled(torch.randn(2,3,11,11)).shape,(2,1))
        self.assertEqual(enabled.forward_with_features(torch.randn(2,3,11,11))[1]['features'].shape,(2,8))
