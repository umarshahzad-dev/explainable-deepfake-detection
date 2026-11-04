import unittest
from unittest.mock import patch
from tests.support import module
from tests.fixtures import crop_fixture,TinyClassifier


class CalibrationVersionTests(unittest.TestCase):
    def test_after_metrics_use_selected_threshold_and_match_evaluation(self):
        ws,cfg,meta,model=self.setup_case()
        api=module(self,"evaluation")
        # Distinguishable predictions at threshold0.3 versus default0.5.
        from deepfake_detection.evaluation.metrics import compute_metrics, real_probability
        predictions={"labels":[0,1],"logits":[0.4,1.2],"loss":0.5}
        with patch("deepfake_detection.evaluation.evaluator.load_run_model",return_value=(model,cfg,meta)), \
             patch("deepfake_detection.evaluation.evaluator.predict_loader",return_value=predictions), \
             patch("deepfake_detection.evaluation.evaluator.fit_temperature",return_value=1.0), \
             patch("deepfake_detection.evaluation.evaluator.choose_threshold",return_value={"threshold":.3,"calibration_fake_recall":1.0,"calibration_real_fpr":0.0}):
            result=api.calibrate("r",["fixture"],ws)
        expected=compute_metrics(predictions["labels"],real_probability(predictions["logits"]),threshold=.3)
        self.assertEqual(result["after"]["threshold"],result["threshold"])
        self.assertEqual(result["after"]["confusion_matrix"],expected["confusion_matrix"])
        self.assertEqual(result["after"]["fake_recall"],result["calibration_fake_recall"])
        persisted=module(self,"paths").read_json(ws.run("r")/result["calibration_version"])
        self.assertEqual(persisted["after"],expected)
    def setup_case(self):
        ws,cfg,rows=crop_fixture(self)
        paths=module(self,"paths")
        paths.write_json(ws.run("r")/"input_manifest.json",{"rows":[]},ws)
        metadata={"run_id":"r","weights_sha256":"fixture-sha","preprocessing_hash":module(self,"preprocessing").preprocessing_hash(cfg)}
        return ws,cfg,metadata,TinyClassifier()

    def test_first_calibration_writes_version_and_pointer(self):
        ws,cfg,meta,model=self.setup_case()
        with patch("deepfake_detection.evaluation.evaluator.load_run_model",return_value=(model,cfg,meta)):
            result=module(self,"evaluation").calibrate("r",["fixture"],ws)
        pointer=module(self,"paths").read_json(ws.run("r")/"calibration.json")
        self.assertTrue((ws.run("r")/pointer["version"]).exists())
        self.assertEqual(pointer["version"],result["calibration_version"])

    def test_second_without_force_refused(self):
        ws,cfg,meta,model=self.setup_case()
        with patch("deepfake_detection.evaluation.evaluator.load_run_model",return_value=(model,cfg,meta)):
            api=module(self,"evaluation")
            api.calibrate("r",["fixture"],ws)
            with self.assertRaisesRegex(ValueError,"force"): api.calibrate("r",["fixture"],ws)

    def test_force_preserves_history_and_updates_pointer(self):
        ws,cfg,meta,model=self.setup_case()
        with patch("deepfake_detection.evaluation.evaluator.load_run_model",return_value=(model,cfg,meta)):
            api=module(self,"evaluation")
            first=api.calibrate("r",["fixture"],ws)
            second=api.calibrate("r",["fixture"],ws,force=True)
        self.assertNotEqual(first["calibration_version"],second["calibration_version"])
        self.assertTrue((ws.run("r")/first["calibration_version"]).exists())

    def test_evaluation_records_frozen_calibration_version(self):
        ws,cfg,meta,model=self.setup_case()
        with patch("deepfake_detection.evaluation.evaluator.load_run_model",return_value=(model,cfg,meta)):
            api=module(self,"evaluation")
            calibration=api.calibrate("r",["fixture"],ws)
            result=api.evaluate("r",["fixture"],ws,plots=False,bootstrap=5)
        self.assertEqual(result["calibration_version"],calibration["calibration_version"])
        self.assertEqual(result["calibration_weights_sha256"],meta["weights_sha256"])
        self.assertTrue(result["calibration_frozen_at"])

    def test_evaluation_refuses_wrong_weight_calibration(self):
        ws,cfg,meta,model=self.setup_case()
        with patch("deepfake_detection.evaluation.evaluator.load_run_model",return_value=(model,cfg,meta)):
            module(self,"evaluation").calibrate("r",["fixture"],ws)
        with patch("deepfake_detection.evaluation.evaluator.load_run_model",return_value=(model,cfg,{**meta,"weights_sha256":"different"})),self.assertRaisesRegex(ValueError,"different model weights"):
            module(self,"evaluation").evaluate("r",["fixture"],ws,plots=False)
