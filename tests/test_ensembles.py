import unittest
from copy import deepcopy


class FakeEngine:
    def __init__(self,score,name='toy'):
        self.score=score; self.metadata={'run_id':name,'preprocessing_hash':'shared','weights_sha256':name}
    def analyze(self,image,explain=False):
        return {'faces':[{'face_index':0,'box':[1,2,10,12],'status':'accepted','fake_score':self.score,'real_score':1-self.score,'label':'FAKE'}], 'training_pilot':False,'status':'completed'}


class EnsembleTests(unittest.TestCase):
    def test_200_ensemble_config_dispatch_without_parent_model_load(self):
        from tests.support import workspace
        from deepfake_detection.common.paths import Workspace,write_json
        from deepfake_detection.inference.engine import InferenceEngine
        from unittest.mock import patch
        ws=Workspace(workspace(self)); cfg={'ensemble':{'enabled':True,'runs':['x','b'],'fusion':'mean','weights':[.7,.3]}}
        write_json(ws.run('ensemble-entry')/'config.json',cfg,ws)
        with patch('deepfake_detection.models.ensembles.EnsembleScorer.from_runs',return_value='scorer') as factory,patch('deepfake_detection.inference.engine.load_run_model') as loader:
            result=InferenceEngine.from_run('ensemble-entry',ws,device='cpu')
        self.assertEqual(result,'scorer'); loader.assert_not_called()
        self.assertEqual(factory.call_args.kwargs['fusion'],'mean')
    def test_140_training_emits_only_opted_in_finetune_snapshots(self):
        from tests.fixtures import crop_fixture,TinyClassifier
        from deepfake_detection.training.trainer import run_training
        from deepfake_detection.common.paths import read_json
        from unittest.mock import patch
        ws,cfg,rows=crop_fixture(self); cfg['training']['snapshot_interval_epochs']=1
        with patch('deepfake_detection.training.trainer.initialize_model',return_value=TinyClassifier()): run_training('snapshot-training',cfg,rows,ws,device='cpu')
        registry=read_json(ws.run('snapshot-training')/'snapshots.json')
        self.assertEqual(len(registry['snapshots']),1); self.assertEqual(registry['snapshots'][0]['epoch'],1)
    def test_140_snapshot_registry_hashes_and_loading(self):
        from deepfake_detection.models.ensembles import SnapshotEnsemble,publish_snapshot
        from tests.fixtures import crop_fixture,TinyClassifier
        from deepfake_detection.common.paths import write_json,read_json
        from unittest.mock import patch
        ws,cfg,rows=crop_fixture(self); run=ws.run('snapshots'); run.mkdir()
        write_json(run/'config.json',cfg,ws)
        model=TinyClassifier()
        publish_snapshot(model,run,cfg,epoch=1,workspace=ws)
        model.classifier.bias.data.add_(.1)
        publish_snapshot(model,run,cfg,epoch=2,workspace=ws)
        with patch('deepfake_detection.models.model_loader.architecture',side_effect=lambda cfg:TinyClassifier()),patch('deepfake_detection.inference.engine.FaceDetector',return_value=object()):
            ensemble=SnapshotEnsemble(run,workspace=ws)
            self.assertEqual(len(ensemble.engines),2)
            registry=read_json(run/'snapshots.json'); registry['snapshots'][0]['weights_sha256']='0'*64
            write_json(run/'snapshots.json',registry,ws)
            with self.assertRaisesRegex(ValueError,'hash'): SnapshotEnsemble(run,workspace=ws)
    def test_139_bayesian_member_variance(self):
        from deepfake_detection.models.ensembles import bayesian_ensemble
        result=bayesian_ensemble([[.9,.8],[.1,.2]],[.7,.3])
        self.assertAlmostEqual(result['mean'][0],.66)
        self.assertAlmostEqual(result['epistemic_variance'][0],.1344)
        self.assertAlmostEqual(result['total_variance'][0],.66*.34)
        self.assertEqual(bayesian_ensemble([.5,.5],[1,1])['epistemic_variance'],0)
        with self.assertRaises(ValueError): bayesian_ensemble([float('nan'),.2],[1,1])
    def test_138_fusion_modes(self):
        from deepfake_detection.models.ensembles import EnsembleScorer
        expected={'weighted_average':.69,'mean':.55,'max':.9,'min':.2,'vote':.5}
        for mode,value in expected.items():
            scorer=EnsembleScorer([FakeEngine(.9),FakeEngine(.2)],[.7,.3],fusion=mode)
            self.assertAlmostEqual(scorer.analyze(None)['faces'][0]['fake_score'],value)
        with self.assertRaises(ValueError): EnsembleScorer([FakeEngine(.5)],fusion='invalid')
    def test_137_stacking_validation_fit_and_overlap_gate(self):
        from deepfake_detection.models.ensembles import StackingEnsemble
        engines=[FakeEngine(.9,'a'),FakeEngine(.8,'b')]
        for engine in engines:
            engine.fit_rows=[{'split':'train','parents':['training-parent'],'known_person_ids':[]}]
            engine.score_crop=lambda row: row['scores'][0] if row['label']==0 else row['scores'][1]
        rows=[{'split':'val','dataset':'ffpp','sample_id':str(i),'parents':[f'validation-{i}'],'known_person_ids':[],'label':i%2,'scores':[.9,.1]} for i in range(12)]
        stack=StackingEnsemble(['a','b'],engines=engines)
        with self.assertRaisesRegex(ValueError,'fit'): stack.predict(None)
        report=stack.fit(rows); self.assertEqual(report['samples'],12)
        self.assertEqual(stack.predict(None)['faces'][0]['label'],'FAKE')
        rows[0]['parents']=['training-parent']
        with self.assertRaisesRegex(ValueError,'overlap'): stack.fit(rows)
        rows[0]['parents']=['validation-0']; rows[0]['split']='external'
        with self.assertRaisesRegex(ValueError,'validation'): stack.fit(rows)
    def test_136_kfold_connected_parents_persons_and_determinism(self):
        from deepfake_detection.models.ensembles import kfold_manifests
        rows=[{'sample_id':str(i),'split':'train','dataset':'ffpp','label':i%2,'parents':[f'p{i}'],'known_person_ids':[]} for i in range(8)]
        rows[1]['parents']=['p0','p1']; rows[2]['known_person_ids']=['same']; rows[3]['known_person_ids']=['same']
        folds=kfold_manifests(rows,n_splits=3,seed=42)
        self.assertEqual(folds,kfold_manifests(rows,n_splits=3,seed=42))
        held=[]
        for fold in folds:
            train,val=fold['train_rows'],fold['validation_rows']
            self.assertTrue(train and val)
            self.assertFalse({p for row in train for p in row['parents']} & {p for row in val for p in row['parents']})
            self.assertFalse({p for row in train for p in row['known_person_ids']} & {p for row in val for p in row['known_person_ids']})
            held.extend(row['sample_id'] for row in val)
        self.assertEqual(sorted(held),sorted(row['sample_id'] for row in rows))
        self.assertTrue(all(row['split']=='train' for row in rows))
        with self.assertRaisesRegex(ValueError,'training'): kfold_manifests([{**rows[0],'split':'test'}])
    def test_135_multi_augmentation_gate(self):
        from deepfake_detection.models.ensembles import MultiAugmentationEnsemble
        from unittest.mock import patch
        engines=[FakeEngine(.7,'a'),FakeEngine(.3,'b')]
        for index,engine in enumerate(engines): engine.cfg={'model':{'name':'xception'},'training':{'augmentation':('robust','robust_v2')[index]}}
        with patch('deepfake_detection.inference.engine.InferenceEngine.from_run',side_effect=engines):
            self.assertAlmostEqual(MultiAugmentationEnsemble(['a','b']).analyze(None)['faces'][0]['fake_score'],.5)
        engines[1].cfg['training']=dict(engines[0].cfg['training'])
        with patch('deepfake_detection.inference.engine.InferenceEngine.from_run',side_effect=engines):
            with self.assertRaisesRegex(ValueError,'augmentation'): MultiAugmentationEnsemble(['a','b'])
    def test_134_multiseed_local_runs_and_seed_gate(self):
        from deepfake_detection.models.ensembles import MultiSeedEnsemble
        from unittest.mock import patch
        engines=[FakeEngine(.7,'a'),FakeEngine(.3,'b')]
        for index,engine in enumerate(engines): engine.cfg={'seed':42+index,'model':{'name':'xception'},'training':{'augmentation':'robust'}}
        with patch('deepfake_detection.inference.engine.InferenceEngine.from_run',side_effect=engines) as loader:
            ensemble=MultiSeedEnsemble(['a','b'])
        self.assertEqual(loader.call_count,2); self.assertAlmostEqual(ensemble.analyze(None)['faces'][0]['fake_score'],.5)
        engines[1].cfg['seed']=42
        with patch('deepfake_detection.inference.engine.InferenceEngine.from_run',side_effect=engines):
            with self.assertRaisesRegex(ValueError,'seeds'): MultiSeedEnsemble(['a','b'])
    def test_131_b4_generic_engine(self):
        from deepfake_detection.models.ensembles import EnsembleScorer
        self.assertEqual(EnsembleScorer([FakeEngine(.6,'b4')]).analyze(None)['ensemble_members'],['b4'])
    def test_132_vit_generic_engine(self):
        from deepfake_detection.models.ensembles import EnsembleScorer
        self.assertEqual(EnsembleScorer([FakeEngine(.6,'vit')]).analyze(None)['ensemble_members'],['vit'])
    def test_133_swin_generic_engine(self):
        from deepfake_detection.models.ensembles import EnsembleScorer
        self.assertEqual(EnsembleScorer([FakeEngine(.6,'swin')]).analyze(None)['ensemble_members'],['swin'])
    def test_130_weighted_probabilities_and_member_failures(self):
        from deepfake_detection.models.ensembles import EnsembleScorer
        first,second=FakeEngine(.8,'xception'),FakeEngine(.2,'b0')
        scorer=EnsembleScorer([first,second],weights=[.7,.3])
        result=scorer.analyze(None)
        self.assertAlmostEqual(result['faces'][0]['fake_score'],.62)
        self.assertEqual(result['faces'][0]['label'],'FAKE')
        self.assertFalse(result['calibration_applied'])
        self.assertEqual(result['ensemble_members'],['xception','b0'])
        second.metadata['preprocessing_hash']='different'
        with self.assertRaisesRegex(ValueError,'preprocessing'): EnsembleScorer([first,second])
        with self.assertRaises(ValueError): EnsembleScorer([first],[-1])

    def test_130_face_correspondence_and_abstention(self):
        from deepfake_detection.models.ensembles import EnsembleScorer
        from unittest.mock import patch
        a,b=FakeEngine(.8),FakeEngine(.2)
        bad=b.analyze(None); bad['faces'][0]['box']=[5,6,9,10]
        with patch.object(b,'analyze',return_value=bad):
            with self.assertRaisesRegex(ValueError,'correspondence'): EnsembleScorer([a,b]).analyze(None)
        missing=b.analyze(None); missing['faces'][0].update(status='rejected',fake_score=None)
        with patch.object(b,'analyze',return_value=missing):
            result=EnsembleScorer([a,b]).analyze(None)
        self.assertIsNone(result['faces'][0]['label']); self.assertIsNone(result['faces'][0]['fake_score'])
