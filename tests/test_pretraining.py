import unittest
import torch
from unittest.mock import patch


class PretrainingTests(unittest.TestCase):
    def test_119_synthetic_pretraining_explicit_resource_gate(self):
        from deepfake_detection.models.pretraining import synthetic_face_pretraining
        from deepfake_detection.models.model_loader import initialize_model
        from deepfake_detection.common.configuration import load_config
        with self.assertRaisesRegex(NotImplementedError,'generator'): synthetic_face_pretraining()
        with self.assertRaises(NotImplementedError): initialize_model(load_config(overrides={'pretraining':{'pretrain_synthetic':True}}))
    def test_116_teacher_overlap_and_missing_fit_rejected_before_student(self):
        from deepfake_detection.training.trainer import run_training
        from deepfake_detection.common.paths import write_json
        from tests.fixtures import crop_fixture
        ws,cfg,rows=crop_fixture(self); cfg['training'].update(knowledge_distillation=True,teacher_run='teacher')
        write_json(ws.run('teacher')/'config.json',cfg,ws)
        train=dict(next(row for row in rows if row['split']=='train'))
        train['parents']=next(row['parents'] for row in rows if row['split']=='val')
        for data,reason in [([train],'held-out'),([row for row in rows if row['split']=='val'],'training provenance')]:
            write_json(ws.run('teacher')/'input_manifest.json',{'rows':data},ws)
            with patch('deepfake_detection.models.pretraining.artifact_metadata',return_value={'weights_sha256':'teacher-sha'}),patch('deepfake_detection.training.trainer.initialize_model') as initialize:
                with self.assertRaisesRegex(ValueError,reason): run_training('blocked',cfg,rows,ws,device='cpu')
                initialize.assert_not_called()
    def test_116_distillation_loss_and_teacher_frozen(self):
        from deepfake_detection.models.pretraining import DistillationLoss
        teacher=torch.nn.Linear(2,1); loss=DistillationLoss(teacher,temperature=4,alpha=.5)
        student=torch.tensor([[.2],[-.3]],requires_grad=True); reference=torch.tensor([[1.],[-1.]],requires_grad=True); labels=torch.tensor([[1.],[0.]])
        result=loss(student,reference,labels); result.backward()
        self.assertGreater(float(student.grad.abs().sum()),0); self.assertIsNone(reference.grad)
        self.assertFalse(any(p.requires_grad for p in teacher.parameters()))
        self.assertFalse(teacher.training)
        equal=student.detach(); self.assertLess(abs(float(loss.distill(equal,equal))),1e-5)
    def test_116_training_dispatch_and_teacher_provenance(self):
        from deepfake_detection.training.trainer import run_training
        from deepfake_detection.common.paths import write_json,read_json
        from tests.fixtures import crop_fixture,TinyClassifier
        ws,cfg,rows=crop_fixture(self); cfg['training'].update(knowledge_distillation=True,teacher_run='teacher')
        teacher_cfg=dict(cfg); teacher_cfg['training']=dict(cfg['training'])
        write_json(ws.run('teacher')/'config.json',teacher_cfg,ws)
        write_json(ws.run('teacher')/'input_manifest.json',{'rows':rows},ws)
        with patch('deepfake_detection.models.pretraining.artifact_metadata',return_value={'weights_sha256':'teacher-sha'}),patch('deepfake_detection.models.pretraining.load_run_model',return_value=(TinyClassifier(),teacher_cfg,{'weights_sha256':'teacher-sha'})),patch('deepfake_detection.training.trainer.initialize_model',return_value=TinyClassifier()):
            result=run_training('distilled',cfg,rows,ws,device='cpu')
        self.assertEqual(result['status'],'completed')
        self.assertEqual(read_json(ws.run('distilled')/'config.json')['training']['teacher_weights_sha256'],'teacher-sha')
    def test_115_self_supervised_explicit_stub(self):
        from deepfake_detection.models.pretraining import self_supervised_pretraining
        from deepfake_detection.models.model_loader import initialize_model
        from deepfake_detection.common.configuration import load_config
        with self.assertRaisesRegex(NotImplementedError,'1.?2 weeks'): self_supervised_pretraining()
        cfg=load_config(overrides={'pretraining':{'self_supervised':True}})
        with self.assertRaises(NotImplementedError): initialize_model(cfg)
    def initialize_source(self,source):
        from tests.fixtures import crop_fixture,TinyClassifier
        from deepfake_detection.models.model_loader import initialize_model
        from safetensors.torch import save_file
        ws,cfg,_=crop_fixture(self); cfg['model']['pretrained_source']=source
        with self.assertRaisesRegex(FileNotFoundError,source): initialize_model(cfg)
        path=ws.runs/(source+'.safetensors'); model=TinyClassifier(); state=model.state_dict()
        state['classifier.weight']=torch.randn(7,3); state['classifier.bias']=torch.randn(7)
        save_file(state,str(path)); cfg['pretraining']={source+'_weights_path':str(path)}
        with patch('deepfake_detection.models.model_loader.architecture',return_value=TinyClassifier()): result=initialize_model(cfg)
        self.assertTrue(torch.equal(result.backbone.weight,state['backbone.weight']))
        self.assertEqual(result.get_classifier().out_features,1)
    def test_112_vggface2_explicit_local_backbone(self): self.initialize_source('vggface2')
    def test_113_ms1m_explicit_local_backbone(self): self.initialize_source('ms1m')
    def test_114_webface_explicit_local_backbone(self): self.initialize_source('webface')
