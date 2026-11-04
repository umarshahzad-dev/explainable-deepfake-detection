import unittest
import torch
from unittest.mock import patch


class AdversarialTests(unittest.TestCase):
    def model(self):
        return torch.nn.Sequential(torch.nn.Conv2d(3,4,1),torch.nn.ReLU(),torch.nn.AdaptiveAvgPool2d(1),torch.nn.Flatten(),torch.nn.Linear(4,1))
    def test_88_fgsm_pgd_bounds_and_no_parameter_grad(self):
        from deepfake_detection.training.adversarial import fgsm_attack,pgd_attack
        torch.manual_seed(42); model=self.model().train(); x=torch.rand(4,3,9,11)*1.6-.8; labels=torch.tensor([0.,1.,0.,1.])
        for attack in (fgsm_attack,pgd_attack):
            result=attack(model,x,labels,eps=.01,mean=[.5]*3,std=[.5]*3)
            self.assertEqual(result.shape,x.shape); self.assertLessEqual(float((result-x).abs().max()),.020001)
            self.assertTrue((result>=-1).all() and (result<=1).all()); self.assertFalse(result.requires_grad)
            self.assertTrue(all(p.grad is None for p in model.parameters())); self.assertTrue(model.training)
        torch.manual_seed(42); a=pgd_attack(model,x,labels)
        torch.manual_seed(42); self.assertTrue(torch.equal(a,pgd_attack(model,x,labels)))
    def test_88_training_flag_routes_attack_and_disabled_does_not(self):
        from deepfake_detection.training.trainer import train_epoch
        model=self.model(); batch=[(torch.full((2,3,7,7),.5),torch.tensor([0.,1.]))]
        with patch('deepfake_detection.training.trainer.fgsm_attack',side_effect=lambda m,x,y,**kw:x+.01) as attack:
            train_epoch(model,batch,torch.optim.SGD(model.parameters(),lr=.01),amp=False,recipe={'use_adversarial':True})
            self.assertEqual(attack.call_count,1)
            train_epoch(model,batch,torch.optim.SGD(model.parameters(),lr=.01),amp=False)
            self.assertEqual(attack.call_count,1)
