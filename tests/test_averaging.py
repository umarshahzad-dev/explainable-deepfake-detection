import unittest
import torch
from unittest.mock import patch


class AveragingTests(unittest.TestCase):
    def test_90_epoch_window_evicts_and_resumes(self):
        from deepfake_detection.training.averaging import WeightAveraging
        model=torch.nn.Linear(1,1,bias=False); average=WeightAveraging(model,window=2)
        for epoch,value in enumerate([1.,3.,8.],1): model.weight.data.fill_(value); average.update(model,epoch)
        self.assertEqual(float(average.shadow['weight']),5.5); self.assertEqual(len(average.snapshots),2)
        other=WeightAveraging(model,2); other.load_state_dict(average.state_dict()); self.assertEqual(float(other.shadow['weight']),5.5)
        with self.assertRaises(ValueError): average.update(model,3)
    def test_90_run_checkpoint_and_candidate(self):
        from tests.fixtures import crop_fixture,TinyClassifier
        from deepfake_detection.training.trainer import run_training
        from deepfake_detection.common.paths import read_json
        ws,cfg,rows=crop_fixture(self); cfg['training'].update(use_weight_averaging=True,weight_average_window=2)
        with patch('deepfake_detection.training.trainer.initialize_model',return_value=TinyClassifier()):
            run_training('window',cfg,rows,ws,device='cpu')
        self.assertIn('window',read_json(ws.run('window')/'history.json')[-1]['candidate_validation_auc'])
        self.assertIn('weight_average_state_dict',torch.load(ws.run('window')/'finetune_latest.pt',weights_only=True))
    def test_87_bounded_bn_update_uses_source_loader(self):
        from tests.fixtures import crop_fixture,TinyClassifier
        from deepfake_detection.training.trainer import run_training
        from deepfake_detection.common.paths import read_json
        ws,cfg,rows=crop_fixture(self); cfg['training'].update(use_swa=True,swa_start_epoch=2,use_swa_final_bn_update=True)
        class Normalized(TinyClassifier):
            def __init__(self): super().__init__(); self.norm=torch.nn.BatchNorm2d(3)
            def forward(self,x): return super().forward(self.norm(x))
        with patch('deepfake_detection.training.trainer.initialize_model',return_value=Normalized()):
            result=run_training('bounded-swa',cfg,rows,ws,device='cpu',max_batches=1)
        self.assertEqual(result['status'],'completed')
        self.assertIn('swa',read_json(ws.run('bounded-swa')/'history.json')[-1]['candidate_validation_auc'])
    def test_87_swa_start_mean_state_and_bn_train_only(self):
        from deepfake_detection.training.averaging import StochasticWeightAveraging
        model=torch.nn.Sequential(torch.nn.BatchNorm1d(2),torch.nn.Linear(2,1)); model[1].weight.data.fill_(1)
        swa=StochasticWeightAveraging(model,start_epoch=2); swa.update(model,epoch=1); self.assertEqual(swa.updates,0)
        swa.update(model,epoch=2); model[1].weight.data.fill_(3); swa.update(model,epoch=3)
        self.assertTrue(torch.equal(swa.shadow['1.weight'],torch.full_like(model[1].weight,2)))
        other=StochasticWeightAveraging(model,2); other.load_state_dict(swa.state_dict()); self.assertEqual(other.updates,2)
        class Dataset(torch.utils.data.Dataset):
            rows=[{'split':'train'}]*4
            def __len__(self): return 4
            def __getitem__(self,i): return torch.full((2,),4.),torch.tensor(1.)
        loader=torch.utils.data.DataLoader(Dataset(),batch_size=2)
        model.eval(); before=model[0].running_mean.clone()
        with swa.average_parameters(model): swa.bn_update(loader,'cpu',model=model)
        self.assertTrue(torch.equal(model[0].running_mean,before))
        self.assertTrue(torch.allclose(swa.shadow['0.running_mean'],torch.full((2,),4.)))
        loader.dataset.rows[0]['split']='val'
        with self.assertRaisesRegex(ValueError,'train'): swa.bn_update(loader,'cpu',model=model)
    def test_87_run_chooses_swa_over_raw_and_ema(self):
        from tests.fixtures import crop_fixture,TinyClassifier
        from deepfake_detection.training.trainer import run_training
        from deepfake_detection.common.paths import read_json
        ws,cfg,rows=crop_fixture(self); cfg['training'].update(use_ema=True,use_swa=True,swa_start_epoch=2,final_bn_update=False)
        calls=[]
        def metrics(labels,p,*args,**kwargs):
            calls.append(1); return {'roc_auc':[.25,.5,.25,.5,.9][len(calls)-1]}
        with patch('deepfake_detection.training.trainer.initialize_model',return_value=TinyClassifier()),patch('deepfake_detection.training.trainer.compute_metrics',side_effect=metrics):
            run_training('swa',cfg,rows,ws,device='cpu')
        self.assertEqual(read_json(ws.run('swa')/'best_metadata.json')['weight_variant'],'swa')
        state=torch.load(ws.run('swa')/'finetune_latest.pt',weights_only=True)
        self.assertEqual(state['swa_state_dict']['updates'],1)
    def test_86_ema_arithmetic_buffers_and_roundtrip(self):
        from deepfake_detection.training.averaging import WeightEMA
        model=torch.nn.Linear(1,1,bias=False); model.weight.data.fill_(1)
        ema=WeightEMA(model,decay=.5); model.weight.data.fill_(3); ema.update(model)
        self.assertEqual(float(ema.shadow['weight']),2)
        with ema.average_parameters(model): self.assertEqual(float(model.weight),2)
        self.assertEqual(float(model.weight),3)
        other=WeightEMA(model,.5); other.load_state_dict(ema.state_dict()); self.assertEqual(float(other.shadow['weight']),2)
        model.weight.data.fill_(9)
        with self.assertRaises(RuntimeError):
            with ema.average_parameters(model): raise RuntimeError('validation failed')
        self.assertEqual(float(model.weight),9)
    def test_86_ema_updates_only_successful_steps(self):
        from deepfake_detection.training.trainer import train_epoch
        from deepfake_detection.training.averaging import WeightEMA
        model=torch.nn.Linear(1,1); ema=WeightEMA(model,.5)
        batches=[(torch.ones(1,1),torch.ones(1,1)) for _ in range(3)]
        result=train_epoch(model,batches,torch.optim.SGD(model.parameters(),lr=.1),accumulation=2,amp=False,update_observers=[ema])
        self.assertEqual(result['optimizer_updates'],2); self.assertEqual(ema.updates,2)
    def test_86_run_selects_best_auc_and_checkpoints_ema(self):
        from tests.fixtures import crop_fixture,TinyClassifier
        from deepfake_detection.training.trainer import run_training
        from deepfake_detection.common.paths import read_json
        ws,cfg,rows=crop_fixture(self); cfg['training']['use_ema']=True
        # raw=.25, EMA=.75 each epoch; criterion/probabilities are deliberately controlled.
        def scores(labels,p,*args,**kwargs): return {'roc_auc': .75 if scores.calls%2 else .25}
        scores.calls=0
        def metrics(labels,p,*args,**kwargs):
            out=scores(labels,p); scores.calls+=1; return out
        with patch('deepfake_detection.training.trainer.initialize_model',return_value=TinyClassifier()),patch('deepfake_detection.training.trainer.compute_metrics',side_effect=metrics):
            run_training('ema',cfg,rows,ws,device='cpu')
        meta=read_json(ws.run('ema')/'best_metadata.json'); self.assertEqual(meta['weight_variant'],'ema')
        state=torch.load(ws.run('ema')/'finetune_latest.pt',weights_only=True)
        self.assertIn('ema_state_dict',state); self.assertGreater(state['ema_state_dict']['updates'],0)
