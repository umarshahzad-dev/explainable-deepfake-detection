import unittest
import torch
from unittest.mock import patch


class ToyXception(torch.nn.Module):
    def __init__(self,classes): super().__init__(); self.body=torch.nn.Linear(3,4); self.fc=torch.nn.Linear(4,classes)
    def get_classifier(self): return self.fc
    def forward(self,x): return self.fc(self.body(x))


class XceptionRegistrationTests(unittest.TestCase):
    def test_190_pretrained_fc_replaced_backbone_preserved(self):
        from deepfake_detection.models.model_loader import initialize_model
        ws,cfg,source=self.source_fixture()
        with patch('deepfake_detection.models.model_loader.architecture',side_effect=lambda *args,**kwargs:ToyXception(1)): model=initialize_model(cfg)
        self.assertEqual(model.fc.weight.shape,(1,4)); self.assertEqual(model.fc.bias.shape,(1,))
        self.assertTrue(torch.equal(model.body.weight,source.body.weight))

    def test_191_bad_transform_rejected_before_construction(self):
        from deepfake_detection.models.model_loader import initialize_model
        ws,cfg,source=self.source_fixture(); cfg['model']['mean']=[.485,.456,.406]
        with patch('deepfake_detection.models.model_loader.architecture') as factory:
            with self.assertRaisesRegex(ValueError,'Xception requires'): initialize_model(cfg)
        factory.assert_not_called()

    def test_192_binary_export_exact_roundtrip(self):
        from deepfake_detection.models.model_loader import publish_best,load_run_model
        from deepfake_detection.common.paths import write_json
        ws,cfg,source=self.source_fixture(); model=ToyXception(1).eval(); run=ws.run('roundtrip')
        write_json(run/'config.json',cfg,ws); publish_best(model,run,cfg,{'run_id':'roundtrip'},ws)
        with patch('deepfake_detection.models.model_loader.architecture',return_value=ToyXception(1)): restored,_,_=load_run_model(run,workspace=ws)
        inputs=torch.randn(3,3)
        self.assertTrue(torch.equal(model(inputs),restored(inputs)))
    def source_fixture(self):
        from tests.support import workspace
        from deepfake_detection.common.paths import Workspace
        from deepfake_detection.common.configuration import load_config
        from deepfake_detection.models.model_loader import export_weights
        ws=Workspace(workspace(self)); path=ws.runs/'xception-source.safetensors'
        source=ToyXception(1000); export_weights(source,path,ws)
        cfg=load_config(overrides={'weights':str(path),'model':{'name':'legacy_xception.tf_in1k','input_size':299,'mean':[.5]*3,'std':[.5]*3}})
        return ws,cfg,source

    def test_189_xception_supplied_checksum_enforced(self):
        from deepfake_detection.models.model_loader import initialize_model
        ws,cfg,source=self.source_fixture(); cfg['pretraining']={'imagenet_sha256':'0'*64}
        with patch('deepfake_detection.models.model_loader.architecture',return_value=ToyXception(1)):
            with self.assertRaisesRegex(ValueError,'checksum'): initialize_model(cfg)

    def test_189_installed_xception_registration_and_binary_fc(self):
        from deepfake_detection.models.model_loader import architecture
        model=architecture({'name':'legacy_xception.tf_in1k','dropout':.35})
        self.assertEqual(model.get_classifier().out_features,1)
        self.assertIn('fc.weight',model.state_dict()); self.assertIn('fc.bias',model.state_dict())
