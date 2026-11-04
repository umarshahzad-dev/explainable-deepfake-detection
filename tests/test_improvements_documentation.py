import json
import re
import unittest
from pathlib import Path


INTERNAL_DOCUMENTATION_AVAILABLE = (
    (Path(__file__).parents[1] / 'IMPROVEMENTS_CATALOG.json').is_file()
    and (Path(__file__).parents[1] / 'docs/IMPROVEMENTS.md').is_file()
)


@unittest.skipUnless(INTERNAL_DOCUMENTATION_AVAILABLE,
                     'Internal documentation is intentionally excluded from GitHub releases')
class ImprovementsDocumentationTests(unittest.TestCase):
    root = Path(__file__).parents[1]

    def test_210_catalog_covers_authoritative_scope_with_actionable_controls(self):
        catalog = json.loads((self.root / 'IMPROVEMENTS_CATALOG.json').read_text(encoding='utf-8'))
        self.assertEqual([row['id'] for row in catalog], list(range(1, 211)))
        for row in catalog:
            with self.subTest(task=row['id']):
                self.assertIn(row['status'], {'IMPLEMENTED', 'STUBBED', 'RESOURCE-PENDING', 'NOT APPLICABLE'})
                for key in ('name', 'section', 'control', 'impact', 'cost'):
                    self.assertTrue(row[key].strip(), key)
                self.assertTrue((self.root / row['file']).is_file())

    def test_210_document_embeds_exact_master_and_all_catalog_rows(self):
        document = (self.root / 'docs/IMPROVEMENTS.md').read_text(encoding='utf-8')
        block = re.search(r'```json\n(.*?)\n```', document, re.S)
        self.assertIsNotNone(block)
        self.assertEqual(json.loads(block.group(1)), json.loads((self.root / 'configs/archive/xception-final-original.json').read_text()))
        self.assertEqual([int(x) for x in re.findall(r'^\| (\d+) \|', document, re.M)], list(range(1, 211)))

    def test_210_document_does_not_claim_measured_improvement_or_missing_scope(self):
        document = (self.root / 'docs/IMPROVEMENTS.md').read_text(encoding='utf-8')
        for phrase in ('210 tasks', '243', 'not an empirical winner', '180 views',
                       'Techniques considered but not implemented', 'No real training',
                       'independent conformal', 'artifact_label_policy'):
            self.assertIn(phrase, document)
