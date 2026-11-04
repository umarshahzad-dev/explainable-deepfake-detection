import unittest
from unittest.mock import patch
from tests.support import module
from tests.fixtures import crop_fixture,TinyClassifier


class ArtifactCommitTests(unittest.TestCase):
    def test_interruption_after_weights_keeps_previous_pointer(self):
        ws,cfg,_=crop_fixture(self)
        api=module(self,"model")
        run=ws.run("r")
        model=TinyClassifier()
        api.publish_best(model,run,cfg,{"phase":"head","validation_macro_auc":.5},ws,phase="head")
        before=module(self,"paths").read_json(run/"head_best_metadata.json")
        model.classifier.weight.data.add_(1)
        with patch("deepfake_detection.models.model_loader.write_json",side_effect=OSError("interrupt")),self.assertRaises(OSError):
            api.publish_best(model,run,cfg,{"phase":"head","validation_macro_auc":.7},ws,phase="head")
        self.assertEqual(module(self,"paths").read_json(run/"head_best_metadata.json"),before)

    def test_alias_failure_after_metadata_is_nonfatal(self):
        ws,cfg,_=crop_fixture(self)
        api=module(self,"model")
        real=api.atomic_bytes
        def fail_alias(path,data,workspace):
            if path.name=="best.safetensors": raise OSError("alias interrupted")
            return real(path,data,workspace)
        with patch("deepfake_detection.models.model_loader.atomic_bytes",side_effect=fail_alias),self.assertLogs("deepfake_detection.models.model_loader",level="WARNING"):
            meta=api.publish_best(TinyClassifier(),ws.run("r"),cfg,{"phase":"head","validation_macro_auc":.7},ws)
        self.assertEqual(api.artifact_metadata(ws.run("r"),cfg,ws)["weights_sha256"],meta["weights_sha256"])

    def resume_corruption(self,missing):
        ws,cfg,rows=crop_fixture(self)
        train=module(self,"training")
        paths=module(self,"paths")
        with patch("deepfake_detection.training.trainer.initialize_model",return_value=TinyClassifier()): train.run_training("r",cfg,rows,ws,device="cpu")
        meta=paths.read_json(ws.run("r")/"best_metadata.json")
        target=ws.output(ws.run("r")/meta["weights_file"])
        if missing: target.unlink()
        else: target.write_bytes(b"mismatch")
        with self.assertRaisesRegex(ValueError,"artifact|Artifact"):
            train.run_training("r",cfg,rows,ws,device="cpu",resume="finetune")

    def test_missing_weights_on_resume_refused(self): self.resume_corruption(True)
    def test_mismatched_hash_on_resume_refused(self): self.resume_corruption(False)
