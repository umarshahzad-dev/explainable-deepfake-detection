import json
import unittest
from pathlib import Path


class PortabilityTests(unittest.TestCase):
    def test_project_root_is_derived_from_package_location(self):
        from deepfake_detection.common.paths import ROOT
        self.assertEqual(ROOT, Path(__file__).resolve().parents[1])

    def test_checked_in_config_paths_are_relative(self):
        root = Path(__file__).resolve().parents[1]
        for path in (root / 'configs').rglob('*.json'):
            config = json.loads(path.read_text(encoding='utf-8-sig'))
            for key in ('raw_root', 'weights', 'official_splits', 'identity_registry'):
                value = config.get(key)
                if value:
                    with self.subTest(config=path.name, key=key):
                        self.assertFalse(Path(value).is_absolute(), value)
