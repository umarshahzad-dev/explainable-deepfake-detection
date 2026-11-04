import unittest
from unittest.mock import patch
from tests.support import module
from tests.fixtures import crop_fixture, TinyClassifier


class IntegrityTests(unittest.TestCase):
    def test_failed_pilot_run_keeps_pilot_status(self):
        ws,cfg,rows=crop_fixture(self)
        for row in rows: row["pilot"]=True
        with patch("deepfake_detection.training.trainer.initialize_model",return_value=TinyClassifier()),patch("deepfake_detection.training.trainer.train_epoch",side_effect=FloatingPointError("fixture loss")):
            with self.assertRaises(FloatingPointError): module(self,"training").run_training("r",cfg,rows,ws,device="cpu")
        self.assertTrue(module(self,"paths").read_json(ws.run("r")/"status.json")["pilot"])
    def test_matching_hashes_pass(self):
        ws,cfg,rows=crop_fixture(self)
        result=module(self,"integrity").preflight_crops(rows,cfg,ws)
        self.assertEqual(result["verified"],len(rows))
        self.assertEqual(result["mismatches"],0)

    def test_mismatch_aborts_before_model_construction(self):
        ws,cfg,rows=crop_fixture(self)
        rows[0]["crop_sha256"]="0"*64
        with patch("deepfake_detection.training.trainer.initialize_model") as build, self.assertRaisesRegex(ValueError,"expected=.*actual="):
            module(self,"training").run_training("r",cfg,rows,ws,device="cpu")
        build.assert_not_called()

    def test_explicit_skip_bypasses_integrity_only(self):
        ws,cfg,rows=crop_fixture(self)
        rows[0]["crop_sha256"]="0"*64
        with patch("deepfake_detection.training.trainer.initialize_model",return_value=TinyClassifier()):
            result=module(self,"training").run_training("r",cfg,rows,ws,device="cpu",skip_integrity_check=True)
        self.assertEqual(result["status"],"completed")
        self.assertTrue(module(self,"paths").read_json(ws.run("r")/"config.json")["integrity_check_skipped"])

    def test_per_partition_limit_is_stratified(self):
        ws,cfg,rows=crop_fixture(self,count=3)
        result=module(self,"integrity").preflight_crops(rows,cfg,ws,limit_per_partition=2)
        self.assertEqual(result["verified"],8)
        self.assertTrue(all(set(item["labels"])=={0,1} for item in result["partitions"]))
