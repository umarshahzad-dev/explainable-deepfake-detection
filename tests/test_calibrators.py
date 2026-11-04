import unittest
import numpy as np


class CalibratorTests(unittest.TestCase):
    def test_155_conformal_finite_sample_quantile_and_sets(self):
        from deepfake_detection.evaluation.calibration import conformal_threshold,conformal_predict
        threshold=conformal_threshold([0,0,1,1],[.1,.2,.8,.9],alpha=.4)
        self.assertAlmostEqual(threshold,.2)
        self.assertEqual(conformal_predict([.1,.5,.9],threshold),[[0],[],[1]])
        self.assertEqual(conformal_threshold([0,1],[.1,.9],alpha=.05),1)
        self.assertEqual(conformal_predict([.5],.6),[[0,1]])
        with self.assertRaises(ValueError): conformal_threshold([0,1],[.1,.9],alpha=1)
    def test_154_median_calibrator_and_json_roundtrip(self):
        from deepfake_detection.evaluation.calibration import CalibratorEnsemble,apply_calibration
        import json
        ensemble=CalibratorEnsemble()
        with self.assertRaises(ValueError): ensemble.transform([.2,.8])
        fitted=ensemble.fit([0,0,1,1],[.1,.2,.8,.9])
        scores=np.asarray([.15,.5,.85])
        expected=np.median([apply_calibration(scores,member) for member in fitted['calibrators']],axis=0)
        self.assertTrue(np.allclose(ensemble.transform(scores),expected))
        self.assertTrue(np.allclose(apply_calibration(scores,json.loads(json.dumps(fitted))),expected))
    def test_152_isotonic_json_knots_and_clipped_bounds(self):
        from deepfake_detection.evaluation.calibration import fit_isotonic,apply_calibration
        import json
        fitted=fit_isotonic([0,0,1,1],[.1,.2,.8,.9])
        restored=json.loads(json.dumps(fitted))
        self.assertTrue(np.allclose(apply_calibration([0,.1,.2,.8,.9,1],restored),[0,0,0,1,1,1]))
        self.assertTrue(np.all(np.diff(apply_calibration(np.linspace(0,1,100),fitted))>=0))
        with self.assertRaises(ValueError): fit_isotonic([1,1],[.5,.6])
        with self.assertRaises(ValueError): apply_calibration([.2],{'method':'isotonic','x':[.3,.2],'y':[0,1]})
    def test_151_platt_serializable_and_real_score_semantics(self):
        from deepfake_detection.evaluation.calibration import fit_platt,apply_calibration
        import json
        labels=[0,0,1,1]; scores=[.1,.2,.8,.9]
        fitted=fit_platt(labels,scores); restored=json.loads(json.dumps(fitted))
        output=apply_calibration(scores,restored)
        self.assertTrue(np.all(output[:2]<.5)); self.assertTrue(np.all(output[2:]>.5))
        self.assertTrue(np.allclose(output,apply_calibration(scores,fitted)))
        with self.assertRaises(ValueError): fit_platt([0,0],[.1,.2])
        with self.assertRaises(ValueError): fit_platt([0,1],[float('nan'),.5])

    def test_151_calibrate_selects_platt_and_persists(self):
        from tests.fixtures import crop_fixture,TinyClassifier
        from deepfake_detection.evaluation.evaluator import calibrate
        from deepfake_detection.common.paths import write_json,read_json
        from deepfake_detection.data.preprocessing import preprocessing_hash
        from unittest.mock import patch
        ws,cfg,rows=crop_fixture(self); cfg['inference']={'calibration_method':'platt'}
        run=ws.run('platt'); write_json(run/'input_manifest.json',{'rows':[]},ws)
        metadata={'run_id':'platt','weights_sha256':'fixture','preprocessing_hash':preprocessing_hash(cfg)}
        with patch('deepfake_detection.evaluation.evaluator.load_run_model',return_value=(TinyClassifier(),cfg,metadata)):
            report=calibrate('platt',['fixture'],ws)
        self.assertEqual(report['method'],'platt'); self.assertIn('coef',report)
        self.assertEqual(read_json(run/report['calibration_version'])['method'],'platt')
