import unittest
import numpy as np
from deepfake_detection.evaluation.metrics import compute_metrics


class EvaluationExtensionTests(unittest.TestCase):
    def test_170_learning_curves_saved_from_history(self):
        from deepfake_detection.evaluation.analysis import save_learning_curves
        from tests.support import workspace
        from deepfake_detection.common.paths import Workspace,write_json
        ws=Workspace(workspace(self)); run=ws.run('history')
        history=[{'phase':'head','epoch':1,'train_loss':.8,'validation_loss':.9,'validation_macro_auc':.6},{'phase':'finetune','epoch':1,'train_loss':.5,'validation_loss':.7,'validation_macro_auc':.8}]
        write_json(run/'history.json',history,ws)
        result=save_learning_curves(run/'history.json',run,workspace=ws)
        from pathlib import Path
        self.assertTrue(Path(result['path']).read_bytes().startswith(b'\x89PNG'))
        self.assertEqual(result['epochs'],2)
    def test_169_explicit_comparator_scores_same_heldout_rows(self):
        from tests.fixtures import crop_fixture,TinyClassifier
        from deepfake_detection.evaluation.evaluator import evaluate
        from deepfake_detection.common.paths import write_json
        from deepfake_detection.data.preprocessing import preprocessing_hash
        from unittest.mock import patch
        ws,cfg,rows=crop_fixture(self,count=2); cfg['evaluation']={'significance_resamples':40}
        for name in ('candidate','reference'): write_json(ws.run(name)/'input_manifest.json',{'rows':[]},ws)
        metadata=lambda name:{'run_id':name,'weights_sha256':name,'preprocessing_hash':preprocessing_hash(cfg)}
        with patch('deepfake_detection.evaluation.evaluator.load_run_model',side_effect=[(TinyClassifier(),cfg,metadata('candidate')),(TinyClassifier(),cfg,metadata('reference'))]):
            report=evaluate('candidate',['fixture'],ws,plots=False,bootstrap=10,compare_run='reference')
        self.assertEqual(report['significance']['reference_run'],'reference')
        self.assertEqual(report['significance']['paired_bootstrap']['sampling_unit'],'dependency_family')
    def test_169_paired_bootstrap_and_delong_identical_and_reversed(self):
        from deepfake_detection.evaluation.analysis import paired_bootstrap_test,delong_test
        labels=[0,0,0,1,1,1]; good=[.1,.2,.3,.7,.8,.9]; reversed_scores=good[::-1]
        same=paired_bootstrap_test(labels,good,good,n=100,seed=42,clusters=['a','b','c','a','b','c'])
        self.assertEqual(same['p_value'],1); self.assertEqual(same['auc_difference'],0)
        self.assertEqual(same['interval'],[0,0])
        improved=paired_bootstrap_test(labels,good,reversed_scores,n=100,seed=42)
        self.assertEqual(improved['auc_difference'],1); self.assertLess(improved['p_value'],.05)
        self.assertEqual(delong_test(labels,good,good)['p_value'],1)
        self.assertEqual(delong_test(labels,good,reversed_scores)['p_value'],0)
        with self.assertRaises(ValueError): delong_test([0,1],[.1,.9],[.2,.8])
    def test_168_failure_exports_rank_wrong_predictions_and_preserve_source(self):
        from deepfake_detection.evaluation.analysis import save_failure_cases
        from tests.fixtures import crop_fixture
        from deepfake_detection.common.paths import digest_file,read_jsonl
        ws,cfg,rows=crop_fixture(self); selected=rows[:4]
        labels=[row['label'] for row in selected]
        scores=[.95 if label==0 else .02 for label in labels]
        original=[digest_file(row['crop_path']) for row in selected]
        output=ws.report('failures')
        result=save_failure_cases(selected,scores,labels,output,top_k=2,workspace=ws)
        self.assertEqual(result['saved'],2); records=read_jsonl(output/'failures.jsonl')
        self.assertGreaterEqual(records[0]['confidence'],records[1]['confidence'])
        self.assertTrue(all(record['true_label']!=record['prediction'] for record in records))
        self.assertEqual([digest_file(row['crop_path']) for row in selected],original)
        self.assertTrue(all(digest_file(record['saved_path'])==record['crop_sha256'] for record in records))
    def test_167_fairness_fake_positive_rates_and_undefined(self):
        from deepfake_detection.evaluation.analysis import fairness_metrics
        labels=[0,0,1,1]*2; decisions=[0,0,0,1,0,1,1,1]
        predictions=[{'label':label,'prediction':decision} for label,decision in zip(labels,decisions)]
        result=fairness_metrics(predictions,['a']*4+['b']*4)['group']
        self.assertAlmostEqual(result['demographic_parity_gap'],.5)
        self.assertAlmostEqual(result['equalized_odds_gap'],.5)
        self.assertAlmostEqual(result['predictive_parity_gap'],1/3)
        undefined=fairness_metrics([{'label':1,'prediction':1}],['a'])['group']
        self.assertIsNone(undefined['groups']['a']['fake_tpr']); self.assertIsNone(undefined['equalized_odds_gap'])
    def test_166_adversarial_robustness_no_weight_or_bn_changes(self):
        import torch
        from deepfake_detection.evaluation.analysis import adversarial_robustness
        class Toy(torch.nn.Module):
            def __init__(self): super().__init__(); self.weight=torch.nn.Parameter(torch.tensor(10.))
            def forward(self,x): return ((x.mean((1,2,3))-.5)*self.weight).reshape(-1,1)
        model=Toy().train(); before=model.weight.clone()
        images=torch.stack([torch.full((3,8,8),.2),torch.full((3,8,8),.8)])
        loader=torch.utils.data.DataLoader(torch.utils.data.TensorDataset(images,torch.tensor([0,1])),batch_size=2)
        result=adversarial_robustness(model,loader,eps_list=[0,.01,.05])
        self.assertEqual(result['clean_auc'],1); self.assertEqual(len(result['attacks']),3)
        self.assertEqual(result['attacks'][0]['auc_degradation'],0)
        self.assertTrue(torch.equal(before,model.weight)); self.assertIsNone(model.weight.grad); self.assertTrue(model.training)
        with self.assertRaises(ValueError): adversarial_robustness(model,loader,eps_list=[-1])
    def test_164_cross_dataset_scores_remain_separate(self):
        from deepfake_detection.evaluation.evaluator import _summaries
        rows=[{'dataset':domain,'method':'fake' if label==0 else 'real','label':label} for domain in ('ffpp','dfd') for label in (0,1)]
        datasets,methods=_summaries(rows,[0,1,0,1],[.1,.9,.9,.1],.5)
        self.assertEqual(datasets['ffpp']['roc_auc'],1); self.assertEqual(datasets['dfd']['roc_auc'],0)

    def test_165_compression_variants_preserve_source_and_dimensions(self):
        from deepfake_detection.evaluation.evaluator import Corruption
        from PIL import Image
        image=Image.fromarray(np.random.default_rng(42).integers(0,256,(32,32,3),dtype=np.uint8))
        before=np.asarray(image).copy()
        for variant in ('jpeg60','resize_half'):
            output=Corruption(variant)(image)
            self.assertEqual(output.size,image.size)
            self.assertTrue(np.array_equal(before,np.asarray(image)))
            self.assertFalse(np.array_equal(before,np.asarray(output)))
    def test_163_evaluation_dispatches_only_when_enabled(self):
        from tests.fixtures import crop_fixture,TinyClassifier
        from deepfake_detection.evaluation.evaluator import evaluate
        from deepfake_detection.common.paths import write_json,write_jsonl
        from deepfake_detection.data.preprocessing import preprocessing_hash
        from unittest.mock import patch
        ws,cfg,rows=crop_fixture(self)
        for row in rows: row['demographics']={'gender':'annotated'}
        write_jsonl(ws.manifest('fixture')/'crops.jsonl',rows,ws)
        write_json(ws.run('demographic')/'input_manifest.json',{'rows':[]},ws)
        cfg['evaluation']={'compute_per_demographic':True}
        metadata={'run_id':'demographic','weights_sha256':'toy','preprocessing_hash':preprocessing_hash(cfg)}
        with patch('deepfake_detection.evaluation.evaluator.load_run_model',return_value=(TinyClassifier(),cfg,metadata)):
            result=evaluate('demographic',['fixture'],ws,plots=False,bootstrap=10)
        self.assertEqual(result['per_demographic']['gender']['annotated']['samples'],2)
    def test_163_demographic_annotations_required_and_grouped(self):
        from deepfake_detection.evaluation.analysis import demographic_summaries
        rows=[{'demographics':{'gender':group}} for group in ('a','a','b','b')]
        report=demographic_summaries(rows,[0,1,0,1],[.1,.9,.2,.8],threshold=.5)
        self.assertEqual(set(report['gender']),{'a','b'})
        self.assertTrue(all(group['roc_auc']==1 for group in report['gender'].values()))
        with self.assertRaisesRegex(ValueError,'annotations'): demographic_summaries([{}],[0],[.1])
    def test_162_method_breakdown_keeps_real_reference(self):
        from deepfake_detection.evaluation.evaluator import _summaries
        rows=[{'dataset':'ffpp','method':method,'label':label} for method,label in [('real',1),('Deepfakes',0),('FaceSwap',0)]]
        datasets,methods=_summaries(rows,[1,0,0],[.9,.1,.2],.5)
        self.assertEqual(set(methods),{'ffpp/Deepfakes','ffpp/FaceSwap'})
        self.assertTrue(all(value['samples']==2 and value['roc_auc']==1 for value in methods.values()))
    def metric_fixture(self): return compute_metrics([0,0,1,1],[.1,.1,.9,.9])
    def test_156_auc(self): self.assertEqual(self.metric_fixture()['roc_auc'],1)
    def test_157_pr_auc(self): self.assertEqual(self.metric_fixture()['fake_average_precision'],1)
    def test_158_ece(self): self.assertAlmostEqual(self.metric_fixture()['ece_binary_fake_probability'],.1)
    def test_159_brier(self): self.assertAlmostEqual(self.metric_fixture()['brier_score'],.01)
    def test_160_recall_at_fpr(self):
        for key in ('fake_recall_at_real_fpr_1pct','fake_recall_at_real_fpr_5pct'): self.assertEqual(self.metric_fixture()[key],1)
