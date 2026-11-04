import copy
import unittest
from unittest.mock import patch
from tests.support import module
from tests.fixtures import crop_fixture,TinyClassifier


class Clock:
    def __init__(self): self.value=0
    def __call__(self): self.value+=.01;return self.value


class BenchmarkTests(unittest.TestCase):
    def case(self): return crop_fixture(self,count=2)
    def run_benchmark(self,ws,cfg,clock=None):
        with patch("deepfake_detection.common.benchmark.initialize_model",return_value=TinyClassifier()):
            return module(self,"benchmark").benchmark(cfg,["fixture"],ws,crops=8,device="cpu",clock=clock)

    def test_benchmark_completes_without_steps_or_file_changes(self):
        ws,cfg,rows=self.case();before=copy.deepcopy(cfg)
        hashes=[module(self,"paths").digest_file(r["crop_path"]) for r in rows]
        result=self.run_benchmark(ws,cfg)
        self.assertEqual(result["samples"],8)
        self.assertEqual(result["optimizer_updates"],0)
        self.assertEqual(cfg,before)
        self.assertEqual(hashes,[module(self,"paths").digest_file(r["crop_path"]) for r in rows])

    def test_all_measurement_fields_are_present(self):
        ws,cfg,_=self.case();result=self.run_benchmark(ws,cfg)
        for key in ("images_per_second","dataset_load_seconds","augmentation_seconds","forward_backward_seconds","peak_allocated_mib","peak_reserved_mib","vram_sampled_mib"):
            self.assertIn(key,result)
        self.assertTrue((ws.report("benchmark")/"benchmark.json").exists())

    def test_loading_and_augmentation_are_actually_timed(self):
        ws,cfg,_=self.case();result=self.run_benchmark(ws,cfg)
        self.assertGreater(result["dataset_load_seconds"],0)
        self.assertGreater(result["augmentation_seconds"],0)
        self.assertGreater(result["forward_backward_seconds"],0)

    def test_seeded_benchmark_timing_harness_is_comparable(self):
        ws,cfg,_=self.case()
        first=self.run_benchmark(ws,cfg,Clock())
        second=self.run_benchmark(ws,cfg,Clock())
        self.assertLessEqual(abs(first["images_per_second"]-second["images_per_second"])/first["images_per_second"],.2)
        self.assertEqual(first["selected_sample_ids"],second["selected_sample_ids"])
