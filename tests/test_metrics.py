import unittest
import numpy as np
from tests.support import module


class MetricTests(unittest.TestCase):
    def test_fractional_labels_are_not_silently_truncated(self):
        api = module(self, "metrics")
        with self.assertRaises(ValueError):
            api.compute_metrics([0.9, 1], [0.1, 0.9])
    def test_mandatory_fixture_has_fake_positive_semantics(self):
        api = module(self, "metrics")
        labels = [0] * 60 + [1] * 40
        real_scores = [0.1] * 45 + [0.9] * 15 + [0.1] * 4 + [0.9] * 36
        result = api.compute_metrics(labels, real_scores)
        self.assertAlmostEqual(result["accuracy"], 0.81)
        self.assertAlmostEqual(result["fake_precision"], 45 / 49)
        self.assertAlmostEqual(result["fake_recall"], 0.75)
        self.assertAlmostEqual(result["real_recall"], 0.90)
        self.assertAlmostEqual(result["roc_auc"], 0.825)
        self.assertAlmostEqual(result["brier_score"], 0.162)
        self.assertEqual(result["confusion_matrix"], [[45, 15], [4, 36]])
        self.assertEqual(result["axis_labels"], ["FAKE", "REAL"])

    def test_nonfinite_scores_are_rejected(self):
        api = module(self, "metrics")
        with self.assertRaises(ValueError):
            api.compute_metrics([0, 1], [float("nan"), 0.7])

    def test_single_class_returns_explicit_auc_limitation(self):
        api = module(self, "metrics")
        result = api.compute_metrics([0, 0], [0.1, 0.9])
        self.assertIsNone(result["roc_auc"])
        self.assertIn("single_class", result["limitations"])

    def test_temperature_fit_reduces_overconfident_wrong_nll(self):
        api = module(self, "metrics")
        logits = np.array([-6, 6, -6, 6], dtype=float)
        labels = np.array([0, 1, 1, 0])
        temperature = api.fit_temperature(labels, logits)
        before = np.mean(np.logaddexp(0, logits) - labels * logits)
        after = np.mean(np.logaddexp(0, logits / temperature) - labels * logits / temperature)
        self.assertGreater(temperature, 1)
        self.assertLess(after, before)

    def test_calibration_threshold_respects_real_false_alarm_budget(self):
        api = module(self, "metrics")
        result = api.choose_threshold([0, 0, 1, 1, 1], [0.1, 0.7, 0.6, 0.9, 0.8], max_fpr=0)
        self.assertEqual(result["calibration_real_fpr"], 0)
        self.assertEqual(result["calibration_fake_recall"], 0.5)

    def test_bootstrap_does_not_pretend_one_cluster_is_many_samples(self):
        api = module(self, "metrics")
        result = api.cluster_auc_interval([0, 1], [0.1, 0.9], ["same", "same"], repetitions=10)
        self.assertIsNone(result["interval"])
        self.assertEqual(result["clusters"], 1)
        self.assertIsNone(result["fake_recall_at_fpr_1pct_interval"])


if __name__ == "__main__":
    unittest.main()
