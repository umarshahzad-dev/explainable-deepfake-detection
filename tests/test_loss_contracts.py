"""Independent mathematical contracts for Tasks 36–53, beyond smoke gradients."""
import unittest
import torch
from deepfake_detection.models import losses as L


class LossContracts(unittest.TestCase):
    def test_36_focal_gamma_zero(self):
        x=torch.tensor([[.3],[-.2]]); y=torch.tensor([[0.],[1.]])
        self.assertTrue(torch.allclose(L.FocalLoss(.5,0)(x,y),.5*torch.nn.functional.binary_cross_entropy_with_logits(x,y)))
    def test_37_weighted_fake_semantics(self):
        x=torch.zeros(2,1); y=torch.tensor([[0.],[1.]])
        self.assertAlmostEqual(float(L.WeightedBCE(3,1)(x,y)),2*float(torch.log(torch.tensor(2.))),places=6)
    def test_38_lovasz_perfect_margin(self):
        self.assertEqual(float(L.LovaszHingeLoss()(torch.tensor([[-3.],[3.]]),torch.tensor([[0.],[1.]]))),0)
    def test_39_dice_fake_positive(self):
        self.assertLess(float(L.DiceLoss()(torch.tensor([[-20.],[20.]]),torch.tensor([[0.],[1.]]))),1e-5)
    def test_40_tversky_false_negative_cost(self):
        y=torch.tensor([[0.],[1.]])
        fn=L.TverskyLoss(.3,.7)(torch.tensor([[20.],[20.]]),y)
        fp=L.TverskyLoss(.3,.7)(torch.tensor([[-20.],[-20.]]),y)
        self.assertGreater(float(fn),float(fp))
    def test_41_contrastive_separated_identical_classes(self):
        self.assertEqual(float(L.ContrastiveLoss()(torch.tensor([[0.,0.],[0.,0.],[3.,3.]]),torch.tensor([0,0,1]))),0)
    def test_42_triplet_separated(self):
        self.assertEqual(float(L.TripletLoss()(torch.tensor([[0.,0.],[0.,0.],[3.,3.],[3.,3.]]),torch.tensor([0,0,1,1]))),0)
    def test_43_arcface_zero_margin(self):
        loss=L.ArcFaceLoss(2,margin=0,scale=1); loss.weight.data.copy_(torch.eye(2))
        self.assertTrue(torch.allclose(loss(torch.eye(2),torch.tensor([0,1])),torch.nn.functional.cross_entropy(torch.eye(2).clamp(-1+1e-6,1-1e-6),torch.tensor([0,1]))))
    def test_44_cosface_target_margin(self):
        a=L.CosFaceLoss(2,margin=0,scale=1); b=L.CosFaceLoss(2,margin=.35,scale=1)
        a.weight.data.copy_(torch.eye(2)); b.weight.data.copy_(a.weight)
        self.assertGreater(float(b(torch.eye(2),torch.tensor([0,1]))),float(a(torch.eye(2),torch.tensor([0,1]))))
    def test_45_center_exact_distance(self):
        self.assertEqual(float(L.CenterLoss(2)(torch.tensor([[3.,4.]]),torch.tensor([0]))),12.5)
    def test_46_supcon_no_positive_is_zero(self):
        self.assertEqual(float(L.SupConLoss()(torch.eye(2),torch.tensor([0,1]))),0)
    def test_47_kl_equal_distribution(self):
        self.assertLess(abs(float(L.KLDivLoss()(torch.zeros(2,1),torch.full((2,1),.5)))),1e-6)
    def test_48_frequency_constant_has_no_high_frequency(self):
        self.assertLess(float(L.FrequencyPenaltyLoss()(torch.ones(2,3,8,8))),1e-8)
    def test_49_consistency_detaches_reference(self):
        a=torch.ones(2,1,requires_grad=True); b=torch.zeros(2,1,requires_grad=True)
        L.ConsistencyLoss()(a,b).backward(); self.assertIsNone(b.grad); self.assertGreater(float(a.grad.abs().sum()),0)
    def test_50_adversarial_is_discriminator_bce(self):
        x=torch.ones(2,1); y=torch.zeros(2,1)
        self.assertTrue(torch.allclose(L.AdversarialLoss()(x,y),torch.nn.functional.binary_cross_entropy_with_logits(x,y)))
    def test_51_combined_rejects_all_zero(self):
        with self.assertRaises(ValueError): L.CombinedLoss([L.MSELoss()],[0])
    def test_52_mse_probability(self):
        self.assertEqual(float(L.MSELoss()(torch.zeros(2,1),torch.ones(2,1))),.25)
    def test_53_hinge_margin(self):
        self.assertEqual(float(L.HingeLoss()(torch.tensor([[-2.],[2.]]),torch.tensor([[0.],[1.]]))),0)
