import unittest
import torch


class DomainTests(unittest.TestCase):
    def test_129_sampler_mixes_training_domains_deterministically(self):
        from deepfake_detection.data.datasets import HierarchicalSampler
        rows=[{'split':'train','dataset':domain,'label':label,'method':'real' if label else 'fake','video_id':f'{domain}-{label}'} for domain in ('ffpp','celebdf') for label in (0,1)]
        sampler=HierarchicalSampler(rows,samples=1000,seed=42)
        first=list(sampler); self.assertEqual(first,list(sampler))
        self.assertEqual({rows[i]['dataset'] for i in first},{'ffpp','celebdf'})
        counts={domain:sum(rows[i]['dataset']==domain for i in first) for domain in ('ffpp','celebdf')}
        self.assertTrue(all(400<n<600 for n in counts.values()))
    def test_128_macro_validation_keeps_domain_scores(self):
        from tests.fixtures import crop_fixture,TinyClassifier
        from deepfake_detection.training.trainer import run_training
        from deepfake_detection.common.paths import read_json
        from unittest.mock import patch
        ws,cfg,rows=crop_fixture(self,count=2)
        for row in rows: row['dataset']='celebdf' if row['sample_id'].endswith('-1') else 'ffpp'
        with patch('deepfake_detection.training.trainer.initialize_model',return_value=TinyClassifier()): run_training('macro',cfg,rows,ws,device='cpu')
        for entry in read_json(ws.run('macro')/'history.json'):
            scores=entry['validation_domain_auc']
            self.assertEqual(set(scores),{'celebdf','ffpp'})
            self.assertAlmostEqual(entry['validation_macro_auc'],sum(scores.values())/2)
    def test_127_maml_resource_stub(self):
        from deepfake_detection.training.domain_adaptation import maml_adaptation
        with self.assertRaisesRegex(NotImplementedError,'episodic'): maml_adaptation(None,None)
    def test_126_self_training_confidence_both_classes_and_no_updates(self):
        from deepfake_detection.training.domain_adaptation import self_train
        class Toy(torch.nn.Module):
            def __init__(self): super().__init__(); self.classifier=torch.nn.Linear(3,1,bias=False); self.classifier.weight.data.copy_(torch.tensor([[4.,0.,0.]]))
            def get_classifier(self): return self.classifier
            def forward(self,x): return self.classifier(x)
        model=Toy(); loader=torch.utils.data.DataLoader(self.adaptation_dataset(),batch_size=4)
        result=self_train(model,loader,.9)
        self.assertEqual(result['accepted'],2); self.assertEqual(result['pseudo_class_counts'],{'fake':1,'real':1})
        model.classifier.weight.data.zero_(); before=model.classifier.weight.clone()
        self.assertEqual(self_train(model,loader,.9)['updates'],0)
        self.assertTrue(torch.equal(before,model.classifier.weight))
        with self.assertRaises(ValueError): self_train(model,loader,.4)
    def test_125_tent_updates_affine_only_and_restores_flags(self):
        from deepfake_detection.training.domain_adaptation import tent_adaptation
        model=torch.nn.Sequential(torch.nn.BatchNorm1d(3),torch.nn.Linear(3,1))
        model[1].weight.data.fill_(1.); model[1].bias.data.fill_(.4)
        before={k:v.clone() for k,v in model.state_dict().items()}
        loader=torch.utils.data.DataLoader(self.adaptation_dataset(),batch_size=4)
        result=tent_adaptation(model,loader,steps=2,lr=.01)
        self.assertEqual(result['updates'],2)
        self.assertFalse(torch.equal(model[0].weight,before['0.weight']))
        self.assertTrue(torch.equal(model[1].weight,before['1.weight']))
        self.assertTrue(torch.equal(model[0].running_mean,before['0.running_mean']))
        self.assertTrue(model[0].track_running_stats); self.assertTrue(model.training)
        with self.assertRaisesRegex(ValueError,'BatchNorm'): tent_adaptation(torch.nn.Linear(3,1),loader)
    def adaptation_dataset(self):
        dataset=torch.utils.data.TensorDataset(torch.tensor([[1.,0.,0.],[0.,1.,0.],[-1.,0.,0.],[0.,-1.,0.]]),torch.tensor([1,1,0,0]))
        dataset.rows=[{'split':'adaptation','dataset':'target-dev','sample_id':str(i)} for i in range(4)]
        return dataset

    def test_124_anchor_updates_only_head_and_gates_provenance(self):
        from deepfake_detection.training.domain_adaptation import few_shot_anchor
        class Toy(torch.nn.Module):
            def __init__(self):
                super().__init__(); self.backbone=torch.nn.Linear(3,3); self.classifier=torch.nn.Linear(3,1)
            def get_classifier(self): return self.classifier
            def forward(self,x): return self.classifier(self.backbone(x))
        model=Toy(); before={k:v.clone() for k,v in model.state_dict().items()}
        report=few_shot_anchor(model,self.adaptation_dataset(),num_anchor=3)
        self.assertEqual(report['samples'],3)
        self.assertTrue(torch.equal(before['backbone.weight'],model.backbone.weight))
        self.assertFalse(torch.equal(before['classifier.weight'],model.classifier.weight))
        bad=self.adaptation_dataset(); bad.rows[0]['split']='test'
        with self.assertRaisesRegex(ValueError,'development'): few_shot_anchor(model,bad)
        with self.assertRaisesRegex(ValueError,'provenance'): few_shot_anchor(model,torch.utils.data.TensorDataset(torch.randn(2,3),torch.zeros(2)))
    def test_alignment_objectives_and_single_domain_batch(self):
        from deepfake_detection.models.losses import TrainingObjective
        for method in ('mmd','coral'):
            features=torch.randn(6,3,requires_grad=True)
            objective=TrainingObjective({'loss':'bce','domain_loss_weight':.2,'domain_method':method,'domain_map':{'a':0,'b':1}},3)
            logits=torch.zeros(6,1,requires_grad=True); targets=torch.ones_like(logits)
            context={'features':features,'domain_ids':torch.tensor([0,0,0,1,1,1])}
            objective(logits,targets,context).backward()
            self.assertGreater(float(features.grad.abs().sum()),0)
            context['domain_ids']=torch.zeros(6,dtype=torch.long)
            self.assertAlmostEqual(float(objective(logits,targets,context)),float(torch.nn.functional.binary_cross_entropy_with_logits(logits,targets)),places=6)
    def test_123_coral_covariance_and_small_batch(self):
        from deepfake_detection.training.domain_adaptation import coral_loss,domain_alignment
        x=torch.tensor([[1.,0.],[-1.,0.],[0.,1.],[0.,-1.]],requires_grad=True)
        self.assertAlmostEqual(float(coral_loss(x,x)),0,places=6)
        loss=coral_loss(x,x.detach()*2); self.assertGreater(float(loss),0)
        loss.backward(); self.assertGreater(float(x.grad.abs().sum()),0)
        self.assertEqual(float(coral_loss(x[:1],x)),0)
        self.assertAlmostEqual(float(domain_alignment(torch.cat([x,x*2]),torch.tensor([0]*4+[1]*4),'coral')),float(loss),places=6)
    def test_122_mmd_identity_shift_and_gradients(self):
        from deepfake_detection.training.domain_adaptation import mmd_loss
        x=torch.randn(5,4,requires_grad=True)
        self.assertAlmostEqual(float(mmd_loss(x,x)),0,places=6)
        loss=mmd_loss(x,x.detach()+2); self.assertGreater(float(loss),0)
        loss.backward(); self.assertGreater(float(x.grad.abs().sum()),0)
        self.assertAlmostEqual(float(mmd_loss(x,x,'linear')),0,places=6)
        with self.assertRaises(ValueError): mmd_loss(x,x,'invalid')
    def test_121_dann_model_binary_and_domain_gradients(self):
        from deepfake_detection.training.domain_adaptation import DANNModel
        model=DANNModel(torch.nn.Linear(3,8),torch.nn.Linear(8,1),torch.nn.Linear(8,2))
        logits,domains=model(torch.randn(4,3),alpha=.1)
        self.assertEqual(logits.shape,(4,1)); self.assertEqual(domains.shape,(4,2))
        (logits.square().mean()+domains.square().mean()).backward()
        self.assertGreater(float(model.domain_classifier.weight.grad.abs().sum()),0)
    def test_121_domain_dataset_and_training_objective(self):
        from tests.fixtures import crop_fixture,TinyClassifier
        from deepfake_detection.data.datasets import CropDataset
        from deepfake_detection.training.trainer import run_training
        from deepfake_detection.common.paths import read_json
        from unittest.mock import patch
        ws,cfg,rows=crop_fixture(self,count=2)
        for row in rows: row['dataset']='celebdf' if row['sample_id'].endswith('-1') else 'ffpp'
        dataset=CropDataset(rows,cfg,training=True,workspace=ws,domain_map={'ffpp':0,'celebdf':1})
        self.assertEqual(len(dataset[0]),3); self.assertEqual(len(CropDataset(rows,cfg,workspace=ws)[0]),2)
        cfg['domain_adaptation']={'method':'dann','grl_lambda':.1}; cfg['training'].update(domain_adaptation_weight=.2,mixup_alpha=.2)
        with patch('deepfake_detection.training.trainer.initialize_model',return_value=TinyClassifier()): result=run_training('dann',cfg,rows,ws,device='cpu')
        self.assertEqual(result['status'],'completed')
        state=torch.load(ws.run('dann')/'finetune_latest.pt',weights_only=True)
        self.assertTrue(any('domain' in key for key in state['loss_state_dict']))
        self.assertEqual(read_json(ws.run('dann')/'config.json')['training']['domain_map'],{'celebdf':0,'ffpp':1})
    def test_120_grl_identity_and_reversed_gradient(self):
        from deepfake_detection.training.domain_adaptation import grad_reverse
        x=torch.tensor([[1.,2.]],requires_grad=True); y=grad_reverse(x,.4)
        self.assertTrue(torch.equal(x,y)); y.square().sum().backward(); self.assertTrue(torch.allclose(x.grad,-.8*x.detach()))
        with self.assertRaises(ValueError): grad_reverse(x,-1)
