import unittest
import torch
from torch import nn
from deepfake_detection.common.configuration import load_config


class BlockModel(nn.Module):
    def __init__(self):
        super().__init__(); self.blocks=nn.Sequential(*[nn.Linear(3,3) for _ in range(5)]); self.classifier=nn.Linear(3,1)
    def get_classifier(self): return self.classifier
    def forward(self,x): return self.classifier(self.blocks(x))


class StrategyTests(unittest.TestCase):
    def test_208_public_parameter_groups_depth_and_unique_coverage(self):
        from deepfake_detection.training.trainer import build_param_groups
        model=BlockModel(); groups=build_param_groups(model,1e-3,.85)
        ids=[id(parameter) for group in groups for parameter in group['params']]
        self.assertEqual(len(ids),len(set(ids))); self.assertEqual(set(ids),{id(p) for p in model.parameters()})
        for depth,group in enumerate(groups[:-1]): self.assertAlmostEqual(group['lr'],1e-3*.85**depth)
        self.assertEqual({id(p) for p in groups[-1]['params']},{id(p) for p in model.get_classifier().parameters()})
        with self.assertRaises(ValueError): build_param_groups(model,1e-3,0)
    def test_118_cascade_unfreezes_additional_blocks(self):
        from deepfake_detection.training.trainer import apply_unfreeze_schedule
        cfg=load_config(overrides={'training':{'unfreeze_schedule':'cascade','cascade_epochs_per_stage':1}}); model=BlockModel()
        for epoch in range(5):
            apply_unfreeze_schedule(model,cfg,'finetune',epoch)
            self.assertEqual(sum(any(p.requires_grad for p in block.parameters()) for block in model.blocks),epoch+1)
        self.assertTrue(all(p.requires_grad for p in model.parameters()))
        cfg['training']['cascade_epochs_per_stage']=0
        with self.assertRaises(ValueError): apply_unfreeze_schedule(model,cfg,'finetune',0)
    def test_117_progressive_sizes_and_controller_final_validation(self):
        from deepfake_detection.training.trainer import progressive_size,run_training
        from tests.fixtures import crop_fixture,TinyClassifier
        from unittest.mock import patch
        cfg=load_config('configs/archive/xception-final-original.json',overrides={'training':{'progressive_resizing':True,'progressive_interval':2}})
        self.assertEqual([progressive_size(cfg,epoch) for epoch in range(7)],[224,224,256,256,299,299,299])
        cfg['training']['progressive_resizing']=False; self.assertEqual(progressive_size(cfg,0),299)
        ws,cfg,rows=crop_fixture(self); cfg['training'].update(progressive_resizing=True,progressive_sizes=[32,64,224],progressive_interval=1)
        observed=[]
        class Sized(TinyClassifier):
            def forward(self,x): observed.append((self.training,x.shape[-1])); return super().forward(x)
        with patch('deepfake_detection.training.trainer.initialize_model',return_value=Sized()): run_training('progressive',cfg,rows,ws,device='cpu')
        self.assertIn((True,32),observed); self.assertIn((True,64),observed)
        self.assertTrue(all(size==224 for training,size in observed if not training))
    def test_104_master_microbatch(self): self.assertEqual(load_config('configs/archive/xception-final-original.json')['training']['batch_size'],4)
    def test_105_master_accumulation(self): self.assertEqual(load_config('configs/archive/xception-final-original.json')['training']['accumulation'],8)
    def test_106_master_effective_batch(self):
        cfg=load_config('configs/archive/xception-final-original.json')['training']; self.assertEqual(cfg['batch_size']*cfg['accumulation'],32)
    def test_103_layer_warmup_not_compounded_by_step_scheduler(self):
        from deepfake_detection.training.trainer import _optimizer,apply_unfreeze_schedule
        from deepfake_detection.training.schedulers import make_scheduler
        cfg=load_config(overrides={'training':{'layer_warmup_epochs':3,'unfreeze_schedule':'gradual','layer_lr_decay':1.,'scheduler':'step','scheduler_cycle_length':1,'scheduler_step_gamma':.5,'warmup_epochs':0,'backbone_lr':.1,'min_lr':1e-8}})
        model=BlockModel(); optimizer=_optimizer(model,cfg,'finetune'); self.assertEqual(len(optimizer.param_groups),6)
        scheduler=make_scheduler(optimizer,cfg['training'],1)
        apply_unfreeze_schedule(model,cfg,'finetune',0); scheduler.set_layer_warmup(0,3)
        active=optimizer.param_groups[2]; self.assertAlmostEqual(active['lr'],.1/3)
        optimizer.step(); scheduler.step(); self.assertAlmostEqual(active['lr'],.05/3)
        optimizer.step(); scheduler.step(); self.assertAlmostEqual(active['lr'],.025/3)
        apply_unfreeze_schedule(model,cfg,'finetune',1); scheduler.set_layer_warmup(1,3); self.assertAlmostEqual(active['lr'],.025*2/3)
        apply_unfreeze_schedule(model,cfg,'finetune',3); scheduler.set_layer_warmup(3,3)
        self.assertEqual(optimizer.param_groups[0]['unfrozen_at_epoch'],3)
        self.assertAlmostEqual(optimizer.param_groups[0]['lr'],.025/3)
        state=scheduler.state_dict(); self.assertIn('unscaled_lrs',state)
    def test_94_master_warmup(self): self.assertEqual(load_config('configs/archive/xception-final-original.json')['training']['warmup_epochs'],3)
    def test_95_master_epochs(self): self.assertEqual(load_config('configs/archive/xception-final-original.json')['training']['finetune_epochs'],18)
    def test_96_master_patience(self): self.assertEqual(load_config('configs/archive/xception-final-original.json')['training']['patience'],8)
    def test_99_mixup_aware_bce_uses_soft_targets(self):
        from deepfake_detection.training.trainer import train_epoch
        from unittest.mock import patch
        model=nn.Linear(1,1); model.weight.data.zero_(); model.bias.data.fill_(1)
        batch=[(torch.ones(2,1),torch.tensor([0.,1.]))]; mixed=torch.tensor([[.2],[.2]])
        expected=torch.nn.functional.binary_cross_entropy_with_logits(torch.ones(2,1),mixed)
        with patch('deepfake_detection.training.trainer.mixup_batch',return_value=(batch[0][0],mixed)):
            result=train_epoch(model,batch,torch.optim.SGD(model.parameters(),lr=.01),amp=False,recipe={'mixup_alpha':.2,'mixup_aware_loss':True})
        self.assertAlmostEqual(result['loss'],float(expected),places=6)
    def test_93_layer_lr_depth_and_unique_parameter_coverage(self):
        from deepfake_detection.training.trainer import _optimizer
        cfg=load_config(overrides={'training':{'layer_lr_decay':.8}}); model=BlockModel()
        optimizer=_optimizer(model,cfg,'finetune')
        self.assertEqual(len(optimizer.param_groups),6)
        for depth,group in enumerate(optimizer.param_groups[:-1]): self.assertAlmostEqual(group['lr'],cfg['training']['backbone_lr']*.8**depth)
        ids=[id(p) for group in optimizer.param_groups for p in group['params']]
        self.assertEqual(len(ids),len(set(ids))); self.assertEqual(set(ids),{id(p) for p in model.parameters()})
        cfg['training']['layer_lr_decay']=1; self.assertEqual(len(_optimizer(model,cfg,'finetune').param_groups),2)
    def test_91_discriminative_learning_rates(self):
        from deepfake_detection.training.trainer import _optimizer
        cfg=load_config('configs/archive/xception-final-original.json'); model=BlockModel(); optimizer=_optimizer(model,cfg,'finetune')
        self.assertEqual(optimizer.param_groups[0]['lr'],.00002)
        self.assertEqual(optimizer.param_groups[-1]['lr'],.0003)
    def test_92_gradual_unfreeze_epoch_boundaries(self):
        from deepfake_detection.training.trainer import apply_unfreeze_schedule
        model=BlockModel(); cfg=load_config(overrides={'training':{'unfreeze_schedule':'gradual'}})
        apply_unfreeze_schedule(model,cfg,'head',0)
        self.assertFalse(any(p.requires_grad for p in model.blocks.parameters())); self.assertTrue(all(p.requires_grad for p in model.classifier.parameters()))
        for epoch in (0,1,2):
            apply_unfreeze_schedule(model,cfg,'finetune',epoch)
            self.assertEqual([any(p.requires_grad for p in block.parameters()) for block in model.blocks],[False,False,True,True,True])
        apply_unfreeze_schedule(model,cfg,'finetune',3); self.assertTrue(all(p.requires_grad for p in model.parameters()))
    def test_92_controller_calls_schedule_before_epoch(self):
        from tests.fixtures import crop_fixture,TinyClassifier
        from deepfake_detection.training.trainer import run_training
        from unittest.mock import patch
        ws,cfg,rows=crop_fixture(self)
        with patch('deepfake_detection.training.trainer.initialize_model',return_value=TinyClassifier()),patch('deepfake_detection.training.trainer.apply_unfreeze_schedule') as schedule:
            run_training('schedule',cfg,rows,ws,device='cpu')
        self.assertEqual([call.args[2:] for call in schedule.call_args_list],[('head',0),('finetune',0)])
