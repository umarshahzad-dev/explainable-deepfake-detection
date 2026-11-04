import unittest


class ResearchTests(unittest.TestCase):
    def scope_documented(self,task):
        from pathlib import Path
        document=Path(__file__).parents[1]/'docs/IMPROVEMENTS.md'
        if not document.is_file(): self.skipTest('Internal documentation is intentionally excluded from GitHub releases')
        text=document.read_text(encoding='utf-8')
        import re
        self.assertTrue(re.search(rf'Task {task}[^\n]*Not applicable to still-image detection pipeline\.',text),f'Missing scope statement for Task{task}')
    def test_185_temporal_scope_documented(self): self.scope_documented(185)
    def test_186_physiological_scope_documented(self): self.scope_documented(186)
    def test_187_lipsync_scope_documented(self): self.scope_documented(187)
    def test_188_audiovisual_scope_documented(self): self.scope_documented(188)
    def test_184_face_xray_resource_stub(self):
        from deepfake_detection.models.research import face_xray_detection
        with self.assertRaisesRegex(NotImplementedError,'boundary'): face_xray_detection()
    def test_183_self_blending_landmarks_determinism_and_label_dispatch(self):
        from deepfake_detection.models.research import self_blended_images,master_landmarks
        from PIL import Image
        import numpy as np,random
        image=Image.fromarray(np.random.default_rng(42).integers(0,256,(32,32,3),dtype=np.uint8)); points=[[10,10],[22,10],[16,16],[12,22],[20,22]]
        first=self_blended_images(image,1,landmarks=points,rng=random.Random(42))
        second=self_blended_images(image,1,landmarks=points,rng=random.Random(42))
        self.assertTrue(np.array_equal(np.asarray(first),np.asarray(second))); self.assertFalse(np.array_equal(np.asarray(first),np.asarray(image)))
        with self.assertRaisesRegex(ValueError,'landmarks'): self_blended_images(image,1)
        geometry={'x':10,'y':20,'side':100,'source_width':200,'source_height':200}
        mapped=master_landmarks({'landmarks':[[35,45],[85,45],[60,70],[45,95],[75,95]],'geometry':geometry},100)
        self.assertTrue(np.allclose(mapped[0],[25,25]))
        from tests.fixtures import crop_fixture
        from deepfake_detection.data.datasets import CropDataset
        ws,cfg,rows=crop_fixture(self); cfg['training']['sbi_prob']=1
        for row in rows: row['crop_landmarks']=[[100,100],[200,100],[150,155],[115,200],[185,200]]
        dataset=CropDataset(rows,cfg,training=True,workspace=ws)
        self.assertTrue(all(dataset[i][1]==0 for i in range(len(dataset))))
    def test_182_recce_resource_stub_reference(self):
        from deepfake_detection.models.research import recce_detection
        with self.assertRaisesRegex(NotImplementedError,'Reconstruction-Classification'): recce_detection()
    def test_181_spsl_resource_stub_reference(self):
        from deepfake_detection.models.research import spsl_detection
        with self.assertRaisesRegex(NotImplementedError,'Spatial-Phase'): spsl_detection()
    def test_178_pbt_resource_stub(self):
        from deepfake_detection.models.research import population_based_training
        with self.assertRaisesRegex(NotImplementedError,'population'): population_based_training()
    def test_177_nas_resource_stub(self):
        from deepfake_detection.models.research import neural_architecture_search
        with self.assertRaisesRegex(NotImplementedError,'search'): neural_architecture_search()
    def test_176_registered_timm_models_random_cpu_forward(self):
        import torch,gc
        from deepfake_detection.models.architecture import create_alternative_model,ALTERNATIVE_MODELS
        for name in ALTERNATIVE_MODELS:
            with self.subTest(name=name):
                model=create_alternative_model({'name':name,'input_size':224,'dropout':.2}).eval()
                with torch.no_grad(): output=model(torch.zeros(1,3,224,224))
                self.assertEqual(output.shape,(1,1)); self.assertTrue(torch.isfinite(output).all())
                del model; gc.collect()

    def test_176_local_initialization_discovers_head_and_keeps_native_drop_path(self):
        import torch,timm
        from tests.support import workspace
        from deepfake_detection.common.paths import Workspace,digest_file
        from deepfake_detection.common.configuration import load_config
        from deepfake_detection.models.model_loader import initialize_model,export_weights
        from unittest.mock import patch
        class Toy(torch.nn.Module):
            def __init__(self,classes): super().__init__(); self.body=torch.nn.Linear(3,4); self.head=torch.nn.Linear(4,classes)
            def get_classifier(self): return self.head
        ws=Workspace(workspace(self)); path=ws.runs/'source.safetensors'; source=Toy(1000)
        export_weights(source,path,ws); defaults=timm.get_pretrained_cfg('vit_tiny_patch16_224')
        cfg=load_config(overrides={'weights':str(path),'model':{'name':'vit','input_size':224,'mean':list(defaults.mean),'std':list(defaults.std),'drop_path':.1,'pretrained_sha256':digest_file(path)}})
        with patch('timm.create_model',return_value=Toy(1)): result=initialize_model(cfg)
        self.assertEqual(result.head.out_features,1); self.assertTrue(torch.equal(result.body.weight,source.body.weight))
    def test_176_hybrid_offline_dispatch_and_enhancement_gate(self):
        from deepfake_detection.models.architecture import create_alternative_model
        from unittest.mock import patch
        with patch('timm.create_model',return_value=object()) as factory:
            create_alternative_model({'name':'hybrid_cnn_vit','dropout':.2,'input_size':224})
        self.assertEqual(factory.call_args.args[0],'vit_tiny_r_s16_p8_224'); self.assertFalse(factory.call_args.kwargs['pretrained'])
        with self.assertRaisesRegex(ValueError,'enhancements'):
            create_alternative_model({'name':'hybrid_cnn_vit','use_cbam':True})
    def test_175_convnext_offline_dispatch(self):
        from deepfake_detection.models.architecture import create_alternative_model
        from unittest.mock import patch
        with patch('timm.create_model',return_value=object()) as factory:
            create_alternative_model({'name':'convnext','dropout':.2,'input_size':224})
        self.assertEqual(factory.call_args.args[0],'convnext_tiny'); self.assertFalse(factory.call_args.kwargs['pretrained'])
    def test_174_swin_offline_dispatch(self):
        from deepfake_detection.models.architecture import create_alternative_model
        from unittest.mock import patch
        with patch('timm.create_model',return_value=object()) as factory:
            create_alternative_model({'name':'swin','dropout':.2,'input_size':224})
        self.assertEqual(factory.call_args.args[0],'swin_tiny_patch4_window7_224'); self.assertFalse(factory.call_args.kwargs['pretrained'])
    def test_173_vit_offline_dispatch(self):
        from deepfake_detection.models.architecture import create_alternative_model
        from unittest.mock import patch
        cfg={'name':'vit','dropout':.2,'input_size':224}
        with patch('timm.create_model',return_value=object()) as factory:
            create_alternative_model(cfg)
        self.assertEqual(factory.call_args.args[0],'vit_tiny_patch16_224'); self.assertFalse(factory.call_args.kwargs['pretrained'])
        self.assertEqual(factory.call_args.kwargs['num_classes'],1)
        with patch('timm.is_model',return_value=False):
            with self.assertRaisesRegex(ValueError,'timm'): create_alternative_model(cfg)
    def test_172_lora_resource_stub(self):
        from deepfake_detection.models.research import lora_adaptation
        with self.assertRaisesRegex(NotImplementedError,'adapter'): lora_adaptation()
    def test_171_clip_resource_stub(self):
        from deepfake_detection.models.research import clip_detection
        with self.assertRaisesRegex(NotImplementedError,'open_clip'): clip_detection()
