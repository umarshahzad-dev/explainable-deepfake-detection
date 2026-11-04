import unittest
import io
import torch
from unittest.mock import patch


class LRFinderTests(unittest.TestCase):
    def test_107_cli_persists_log_and_summary(self):
        from tools.laptop_workflow import main
        from tests.fixtures import crop_fixture,TinyClassifier
        from deepfake_detection.common.paths import write_jsonl
        ws,cfg,rows=crop_fixture(self)
        write_jsonl(ws.manifest('fixture')/'crops.jsonl',rows,ws)
        with patch('deepfake_detection.training.lr_finder.initialize_model',return_value=TinyClassifier()),patch('sys.stdout',io.StringIO()):
            code=main(['find-lr','--run','cli-lr','--manifests','fixture','--steps','3','--device','cpu'],workspace=ws)
        self.assertEqual(code,0)
        log=(ws.run('cli-lr')/'run.log').read_text(); self.assertIn('START',log); self.assertIn('END exit_status=0',log); self.assertIn('SUMMARY',log)
    def test_107_synthetic_range_writes_plot_and_restores_model(self):
        from deepfake_detection.training.lr_finder import find_lr
        from tests.fixtures import crop_fixture,TinyClassifier
        from deepfake_detection.common.paths import read_json
        ws,cfg,rows=crop_fixture(self); model=TinyClassifier(); original={k:v.clone() for k,v in model.state_dict().items()}
        with patch('deepfake_detection.training.lr_finder.initialize_model',return_value=model):
            result=find_lr('lr',cfg,rows,ws,device='cpu',steps=4,start_lr=1e-5,end_lr=1e-2)
        self.assertEqual(result['optimizer_updates'],4); self.assertEqual(len(result['curve']),4)
        self.assertAlmostEqual(result['curve'][0]['lr'],1e-5); self.assertAlmostEqual(result['curve'][-1]['lr'],1e-2)
        self.assertTrue((ws.run('lr')/'learning_rate_finder.png').read_bytes().startswith(b'\x89PNG'))
        self.assertEqual(read_json(ws.run('lr')/'lr_finder.json')['pilot'],True)
        self.assertTrue(all(torch.equal(model.state_dict()[k],v) for k,v in original.items()))
    def test_107_cli_and_heldout_rejection(self):
        from tools.laptop_workflow import build_parser
        from deepfake_detection.training.lr_finder import find_lr
        from tests.fixtures import crop_fixture
        args=build_parser().parse_args(['find-lr','--run','range','--manifests','ffpp-clean','--steps','4'])
        self.assertEqual(args.steps,4)
        ws,cfg,rows=crop_fixture(self); rows[0]['dataset']='dfd'
        with self.assertRaisesRegex(ValueError,'OOD'): find_lr('bad',cfg,rows,ws,device='cpu',steps=4)
