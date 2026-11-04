"""Contract against the exact master configuration supplied in the request."""
import json
from pathlib import Path
import unittest
from deepfake_detection.common.configuration import load_config

ROOT=Path(__file__).absolute().parents[1]


class XceptionFinalConfigTests(unittest.TestCase):
    def test_every_specified_master_value_matches(self):
        expected=json.loads((ROOT/'tests/fixtures/xception_final_spec.json').read_text(encoding='utf-8'))
        actual=load_config('configs/archive/xception-final-original.json')
        def compare(spec,loaded,trail=()):
            for key,value in spec.items():
                path=(*trail,key)
                with self.subTest(path='.'.join(path)):
                    self.assertIn(key,loaded)
                    if isinstance(value,dict): compare(value,loaded[key],path)
                    elif not trail and key in {'raw_root','weights','official_splits'}:
                        reference=Path(value)
                        reference=reference if reference.is_absolute() else ROOT/reference
                        self.assertEqual(Path(loaded[key]).resolve(),reference.resolve())
                    else: self.assertEqual(loaded[key],value)
        compare(expected,actual)

    def test_master_targets_xception_299_and_correct_labels(self):
        cfg=load_config('configs/archive/xception-final-original.json')
        self.assertEqual(cfg['model']['name'],'legacy_xception.tf_in1k')
        self.assertEqual(cfg['model']['input_size'],299)
        self.assertEqual(cfg['model']['class_map'],{'fake':0,'real':1})
        self.assertEqual(cfg['model']['mean'],[.5]*3)
        self.assertEqual(cfg['model']['std'],[.5]*3)


if __name__=='__main__': unittest.main()
