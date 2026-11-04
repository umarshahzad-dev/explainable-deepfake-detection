import unittest
from unittest.mock import patch
from tests.support import module


class TelemetryTests(unittest.TestCase):
    def test_cpu_monitor_explicitly_reports_no_cuda_measurement(self):
        api = module(self, "telemetry")
        with api.MemoryMonitor("cpu") as monitor:
            pass
        self.assertIsNone(monitor.report()["peak_process_mib"])

    def test_process_query_counts_only_current_pid_and_rejects_wddm_na(self):
        api = module(self, "telemetry")
        self.assertEqual(api.parse_process_memory("123, 80\n456, 300\n123, 20", 123), 100)
        self.assertIsNone(api.parse_process_memory("123, N/A", 123))

    def test_environment_snapshot_is_offline_and_in_resume_signature(self):
        cfg = module(self, "config").load_config()
        signature = module(self, "checkpoints").resume_signature(cfg, "sha", "head", "cpu")
        self.assertIn("timm", signature["environment"])
        self.assertIn("python", signature["environment"])
