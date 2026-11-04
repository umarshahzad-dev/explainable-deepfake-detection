import unittest
from PIL import Image
from tests.support import module
from tests import test_extraction_retries as retries


class OrphanTests(unittest.TestCase):
    def test_partial_write_records_and_disk_accounting_survive_failure(self):
        from unittest.mock import patch
        ws,cfg,reader=retries.RetryTests.setup_video(self)
        cfg["extraction"]["train_real"]=2
        api=module(self,"extraction")
        preprocessing=module(self,"preprocessing")
        Face=module(self,"detector").Face
        class Detector:
            def detect(self,image): return [Face([10,10,90,90],.99,[],"accepted",preprocessing.square_crop(image,[10,10,90,90]))]
        real=api.atomic_bytes
        calls=[]
        def fail_second(path,data,workspace):
            calls.append(path)
            if len(calls)==2: raise OSError("simulated write failure")
            return real(path,data,workspace)
        with patch("deepfake_detection.data.extraction.atomic_bytes",side_effect=fail_second):
            result=api.extract("fixture",cfg,ws,detector=Detector(),reader_factory=reader)
        records=module(self,"paths").read_jsonl(ws.manifest("fixture")/"crops.jsonl")
        self.assertEqual(len(records),1)
        self.assertEqual(records[0]["extraction_status"],"failed")
        self.assertGreater(result["stored_png_bytes"],0)
    def setup_case(self):
        ws,cfg,reader=retries.RetryTests.setup_video(self)
        api=module(self,"extraction")
        root=ws.crops/"fixture"
        root.mkdir()
        tracked=root/"tracked.png"
        Image.new("RGB",(299,299),"red").save(tracked)
        paths=module(self,"paths")
        paths.write_jsonl(ws.manifest("fixture")/"crops.jsonl",[{"crop_path":str(tracked),"crop_sha256":paths.digest_file(tracked)}],ws)
        return api,ws,root

    def test_no_orphans(self):
        api,ws,root=self.setup_case()
        self.assertEqual(api.scan_orphans("fixture",ws)["orphan_png_count"],0)

    def test_one_orphan_detected_without_deletion(self):
        api,ws,root=self.setup_case()
        orphan=root/"orphan.png"
        Image.new("RGB",(299,299),"blue").save(orphan)
        report=api.scan_orphans("fixture",ws)
        self.assertEqual(report["orphan_png_count"],1)
        self.assertEqual(report["orphan_bytes"],orphan.stat().st_size)
        self.assertTrue(orphan.exists())

    def test_listing_contains_hash_size_mtime(self):
        api,ws,root=self.setup_case()
        Image.new("RGB",(299,299),"blue").save(root/"orphan.png")
        api.scan_orphans("fixture",ws)
        records=module(self,"paths").read_jsonl(ws.manifest("fixture")/"orphans.jsonl")
        self.assertEqual(set(records[0]),{"path","sha256","size","mtime"})

    def test_orphans_not_inserted_into_crop_manifest(self):
        api,ws,root=self.setup_case()
        Image.new("RGB",(299,299),"blue").save(root/"orphan.png")
        api.scan_orphans("fixture",ws)
        records=module(self,"paths").read_jsonl(ws.manifest("fixture")/"crops.jsonl")
        self.assertEqual(len(records),1)
        self.assertTrue(records[0]["crop_path"].endswith("tracked.png"))
