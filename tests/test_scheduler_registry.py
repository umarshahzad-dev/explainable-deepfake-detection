import unittest
import torch
from tests.support import module


class SchedulerRegistryTests(unittest.TestCase):
    def test_restart_returns_to_peak_after_cycle(self):
        optimizer,scheduler=self.schedule('cosine_warm_restarts')
        values=[]
        for _ in range(4): optimizer.step();scheduler.step();values.append(optimizer.param_groups[0]['lr'])
        self.assertLess(values[2],values[0]);self.assertAlmostEqual(values[3],.1)

    def test_legacy_cosine_state_loads_without_changing_lr(self):
        optimizer,scheduler=self.schedule('cosine',warmup=1)
        for _ in range(3): optimizer.step();scheduler.step()
        legacy=scheduler.raw.state_dict()
        _,restored=self.schedule('cosine',warmup=1)
        restored.load_state_dict(legacy)
        self.assertEqual(restored.raw.state_dict(),legacy)
    def schedule(self,name,warmup=0):
        parameter=torch.nn.Parameter(torch.ones(1));optimizer=torch.optim.SGD([parameter],lr=.1)
        cfg=module(self,'config').load_config(overrides={'training':{'scheduler':name,'head_lr':.1,'finetune_epochs':4,'warmup_epochs':warmup,'min_lr':.001,'scheduler_cycle_length':1,'scheduler_step_gamma':.5,'scheduler_patience':0}})
        return optimizer,module(self,'schedulers').make_scheduler(optimizer,cfg['training'],steps_per_epoch=4)

    def test_available_schedules_step_with_finite_lrs(self):
        for name in ('cosine','cosine_warm_restarts','onecycle','linear','polynomial','step','reduce_on_plateau','noam','warm_restarts'):
            with self.subTest(name=name):
                optimizer,scheduler=self.schedule(name)
                for _ in range(8): optimizer.step();scheduler.step();scheduler.step_metric(.5)
                self.assertTrue(.001<=optimizer.param_groups[0]['lr']<=.1+1e-8)

    def test_linear_polynomial_and_step_decay(self):
        for name in ('linear','polynomial','step'):
            optimizer,scheduler=self.schedule(name)
            first=optimizer.param_groups[0]['lr']
            for _ in range(8): optimizer.step();scheduler.step()
            self.assertLess(optimizer.param_groups[0]['lr'],first)

    def test_warmup_and_noam_increase_then_decay(self):
        optimizer,scheduler=self.schedule('cosine',warmup=1)
        first=optimizer.param_groups[0]['lr']
        for _ in range(4): optimizer.step();scheduler.step()
        self.assertGreater(optimizer.param_groups[0]['lr'],first)
        optimizer,scheduler=self.schedule('noam',warmup=1)
        values=[]
        for _ in range(12): optimizer.step();scheduler.step();values.append(optimizer.param_groups[0]['lr'])
        self.assertGreater(max(values),values[0]);self.assertLess(values[-1],max(values))

    def test_plateau_steps_only_on_validation_and_roundtrips(self):
        optimizer,scheduler=self.schedule('reduce_on_plateau')
        scheduler.step_metric(.8);scheduler.step_metric(.7)
        self.assertLess(optimizer.param_groups[0]['lr'],.1)
        state=scheduler.state_dict()
        _,restored=self.schedule('reduce_on_plateau');restored.load_state_dict(state)
        self.assertEqual(restored.state_dict(),state)
