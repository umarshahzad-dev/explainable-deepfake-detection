import unittest
from copy import deepcopy
from unittest.mock import patch


class FinalIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import torch
        torch.set_num_threads(2)

    def test_combined_context_uses_component_weights_not_standalone_weight(self):
        import torch
        from deepfake_detection.models.losses import objective_for_model, objective_forward

        class Toy(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.conv_head = torch.nn.Conv2d(3, 4, 1)
                self.classifier = torch.nn.Linear(4, 1)

            def get_classifier(self):
                return self.classifier

            def forward(self, x):
                return self.classifier(self.conv_head(x).mean((2, 3)))

        for component, requirement in [('consistency', 'needs_alternative'),
                                        ('center', 'needs_features'),
                                        ('frequency_penalty', 'needs_spatial')]:
            with self.subTest(component=component):
                model = Toy()
                config = {'loss': 'combined', 'auxiliary_loss_weight': 0,
                          'combined_losses': [{'name': 'bce', 'weight': 1},
                                              {'name': component, 'weight': .2}]}
                objective = objective_for_model(model, config)
                self.assertTrue(getattr(objective, requirement))
                logits, loss = objective_forward(model, torch.rand(4, 3, 8, 8),
                                                 torch.tensor([[0.], [1.], [0.], [1.]]), objective)
                loss.backward()
                self.assertTrue(torch.isfinite(loss))
                self.assertIsNotNone(model.classifier.weight.grad)
                disabled = objective_for_model(model, {**config, 'combined_losses': [
                    {'name': 'bce', 'weight': 1}, {'name': component, 'weight': 0}]})
                self.assertFalse(getattr(disabled, requirement))

    def test_distilled_run_persists_teacher_fit_and_selection_lineage(self):
        from tests.fixtures import crop_fixture, TinyClassifier
        from deepfake_detection.common.paths import write_json, read_json
        from deepfake_detection.training.trainer import run_training
        from deepfake_detection.evaluation.evaluator import audit_against_run
        ws, cfg, rows = crop_fixture(self)
        cfg['training'].update(knowledge_distillation=True, teacher_run='teacher')
        teacher_cfg = deepcopy(cfg)
        teacher_rows = []
        for kind in ('train', 'val'):
            row = deepcopy(next(r for r in rows if r['split'] == kind))
            row.update(sample_id='teacher-' + kind, video_id='teacher-' + kind,
                       parents=['teacher-' + kind], sha256='teacher-raw-' + kind,
                       crop_sha256='teacher-crop-' + kind)
            teacher_rows.append(row)
        write_json(ws.run('teacher') / 'config.json', teacher_cfg, ws)
        write_json(ws.run('teacher') / 'input_manifest.json', {'rows': teacher_rows}, ws)
        with patch('deepfake_detection.models.pretraining.artifact_metadata', return_value={'weights_sha256': 'teacher-sha'}), \
             patch('deepfake_detection.models.pretraining.load_run_model', return_value=(TinyClassifier(), teacher_cfg, {'weights_sha256': 'teacher-sha'})), \
             patch('deepfake_detection.training.trainer.initialize_model', return_value=TinyClassifier()):
            run_training('distilled-lineage', cfg, rows, ws, device='cpu')
        saved = read_json(ws.run('distilled-lineage') / 'input_manifest.json')
        self.assertIn('teacher_exposure', saved)
        for partition in ('test', 'calibration'):
            for parent in ('teacher-train', 'teacher-val'):
                candidate = deepcopy(next(r for r in rows if r['split'] == partition))
                candidate.update(sample_id='new-' + parent + partition, parents=[parent])
                with self.subTest(partition=partition, parent=parent), self.assertRaisesRegex(ValueError, 'Teacher exposure'):
                    audit_against_run(ws.run('distilled-lineage'), [candidate])
        audit_against_run(ws.run('distilled-lineage'), [r for r in rows if r['split'] == 'test'])
        saved['teacher_exposure'][0]['parents'] = ['edited-ancestor']
        write_json(ws.run('distilled-lineage') / 'input_manifest.json', saved, ws)
        with self.assertRaisesRegex(ValueError, 'lineage'):
            audit_against_run(ws.run('distilled-lineage'), [r for r in rows if r['split'] == 'test'])

    def test_transitive_teacher_fit_lineage_is_admitted_and_carried_forward(self):
        from tests.fixtures import crop_fixture
        from deepfake_detection.common.paths import write_json, fingerprint
        from deepfake_detection.models.pretraining import distillation_spec
        ws, cfg, rows = crop_fixture(self)
        cfg['training'].update(knowledge_distillation=True, teacher_run='teacher')
        inherited = [{**deepcopy(rows[0]), 'sample_id': 'ancestor-train',
                      'video_id': 'ancestor-train', 'parents': ['ancestor-parent'],
                      'sha256': 'ancestor-sha', 'crop_sha256': 'ancestor-crop',
                      'exposure_kind': 'train', 'exposure_run': 'ancestor',
                      'exposure_weights_sha256': 'ancestor-weights'}]
        teacher_cfg = deepcopy(cfg)
        teacher_cfg['training'].update(teacher_weights_sha256='ancestor-weights',
                                      teacher_lineage_sha256=fingerprint(inherited))
        write_json(ws.run('teacher') / 'config.json', teacher_cfg, ws)
        write_json(ws.run('teacher') / 'input_manifest.json', {
            'rows': [r for r in rows if r['split'] in ('train', 'val')],
            'teacher_exposure': inherited}, ws)
        with patch('deepfake_detection.models.pretraining.artifact_metadata', return_value={'weights_sha256': 'teacher-sha'}):
            admitted = distillation_spec(cfg, rows, ws)
            self.assertIn('ancestor-parent', {p for r in admitted['exposure_rows'] for p in r['parents']})
            held = deepcopy(next(r for r in rows if r['split'] == 'test'))
            held.update(sample_id='new-held', parents=['ancestor-parent'])
            with self.assertRaisesRegex(ValueError, 'Teacher training'):
                distillation_spec(cfg, [*rows, held], ws)

    def test_teacher_selection_cannot_overlap_student_calibration_at_admission(self):
        from tests.fixtures import crop_fixture
        from deepfake_detection.common.paths import write_json
        from deepfake_detection.models.pretraining import distillation_spec
        ws, cfg, rows = crop_fixture(self)
        cfg['training'].update(knowledge_distillation=True, teacher_run='teacher')
        teacher_train = deepcopy(next(r for r in rows if r['split'] == 'train'))
        teacher_val = deepcopy(next(r for r in rows if r['split'] == 'calibration'))
        teacher_val['split'] = 'val'
        write_json(ws.run('teacher') / 'config.json', cfg, ws)
        write_json(ws.run('teacher') / 'input_manifest.json', {'rows': [teacher_train, teacher_val]}, ws)
        with patch('deepfake_detection.models.pretraining.artifact_metadata', return_value={'weights_sha256': 'teacher-sha'}):
            with self.assertRaisesRegex(ValueError, 'Teacher selection'):
                distillation_spec(cfg, rows, ws)

    def test_enabled_unused_adaptation_selectors_reject_before_model_construction(self):
        from tests.fixtures import crop_fixture
        from deepfake_detection.training.trainer import run_training
        for setting in ({'test_time_adaptation': True}, {'coral_weight': .1}):
            with self.subTest(setting=setting):
                ws, cfg, rows = crop_fixture(self)
                cfg['domain_adaptation'] = setting
                with patch('deepfake_detection.training.trainer.initialize_model') as initialize:
                    with self.assertRaisesRegex(ValueError, 'explicit|domain_adaptation_weight'):
                        run_training('invalid-adaptation', cfg, rows, ws, device='cpu')
                    initialize.assert_not_called()

    def test_alternative_initialization_seeds_before_constructing_binary_head(self):
        import torch
        import timm
        from tests.support import workspace
        from deepfake_detection.common.paths import Workspace
        from deepfake_detection.common.configuration import load_config
        from deepfake_detection.models.model_loader import initialize_model, export_weights

        class Toy(torch.nn.Module):
            def __init__(self, classes):
                super().__init__()
                self.body = torch.nn.Linear(3, 4)
                self.head = torch.nn.Linear(4, classes)

            def get_classifier(self):
                return self.head

        ws = Workspace(workspace(self))
        source = ws.runs / 'source.safetensors'
        export_weights(Toy(1000), source, ws)
        defaults = timm.get_pretrained_cfg('vit_tiny_patch16_224')
        cfg = load_config(overrides={'weights': str(source), 'model': {
            'name': 'vit', 'input_size': 224,
            'mean': list(defaults.mean), 'std': list(defaults.std)}})
        with patch('timm.create_model', side_effect=lambda *a, **k: Toy(1)):
            torch.manual_seed(7)
            first = initialize_model(cfg)
            torch.manual_seed(900)
            second = initialize_model(cfg)
        self.assertTrue(torch.equal(first.head.weight, second.head.weight))
        self.assertTrue(torch.equal(first.head.bias, second.head.bias))

    def test_random_erasing_is_consumed_and_independently_disabled(self):
        from PIL import Image
        import numpy as np
        from deepfake_detection.data.datasets import Augment
        image = Image.new('RGB', (32, 32), (230, 150, 70))
        cfg = {'jpeg_prob': 0, 'rescale_prob': 0, 'blur_prob': 0,
               'random_erase_prob': 1}
        with patch('random.random', return_value=.9):
            output = Augment('robust_v2', cfg)(image)
        self.assertFalse(np.array_equal(np.asarray(image), np.asarray(output)))
        self.assertEqual(output.size, image.size)
        self.assertEqual(image.getpixel((0, 0)), (230, 150, 70))
        with patch('random.random', return_value=.9):
            disabled = Augment('robust_v2', {**cfg, 'random_erase_prob': 0})(image)
        self.assertTrue(np.array_equal(np.asarray(disabled), np.asarray(image)))
        with patch('random.random', return_value=.9):
            legacy = Augment('robust', cfg)(image)
        self.assertTrue(np.array_equal(np.asarray(legacy), np.asarray(image)))

    def test_multi_augmentation_distinguishes_actual_policy_parameter_keys(self):
        from tests.test_ensembles import FakeEngine
        from deepfake_detection.models.ensembles import MultiAugmentationEnsemble
        for key in ('perspective_scale', 'randaugment_magnitude',
                    'randaugment_num_ops', 'trivialaugment_bins',
                    'augmix_width', 'augmix_depth', 'augmix_alpha'):
            with self.subTest(key=key):
                engines = [FakeEngine(.7, 'a'), FakeEngine(.3, 'b')]
                for index, engine in enumerate(engines):
                    engine.cfg = {'model': {'name': 'xception'},
                                  'training': {'augmentation': 'robust_v2', key: index + 1}}
                with patch('deepfake_detection.inference.engine.InferenceEngine.from_run', side_effect=engines):
                    scorer = MultiAugmentationEnsemble(['a', 'b'])
                self.assertEqual(len(scorer.engines), 2)
