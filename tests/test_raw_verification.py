import unittest
from PIL import Image
from tests.support import module, workspace


class RawVerificationTests(unittest.TestCase):
    def evaluate_case(self, source=None, edited=False):
        paths = module(self, "paths")
        ws = paths.Workspace(workspace(self))
        cfg = module(self, "config").load_config()
        raw = ws.root / "raw.mp4"
        raw.write_bytes(b"generated")
        row = {"sample_id":"v", "dataset":"dfd", "method":"DFD", "label":0, "split":"external", "parents":["p"],
               "raw_path":str(raw), "sha256":paths.digest_file(raw), "subject_verified":True}
        if source == "manual_registry": row["verification_source"] = source
        paths.write_jsonl(ws.manifest("m") / "videos.jsonl", [row], ws)
        if source == "auto_single_face_heuristic" or edited:
            crop = {**row, "sample_id":"v:f0", "video_id":"v", "frame_index":0, "verification_source":source}
            if edited: crop.pop("verification_source")
            paths.write_jsonl(ws.manifest("m") / "crops.jsonl", [crop], ws)
        paths.write_json(ws.run("r") / "input_manifest.json", {"rows":[]}, ws)
        class Reader:
            count=1
            def __init__(self,path): pass
            def __enter__(self): return self
            def __exit__(self,*args): pass
            def frames(self,targets):
                for index in targets: yield index, Image.new("RGB",(100,100))
        class Engine:
            metadata={"weights_sha256":"fixture", "preprocessing_hash":"fixture"}
            calibration={"threshold":.5}
            def __init__(self): self.cfg=cfg
            def analyze(self,image,explain=False): return {"faces":[{"status":"accepted","real_score":.1}]}
        return module(self,"evaluation").evaluate_raw("r",["m"],ws,engine=Engine(),reader_factory=Reader,plots=False)

    def test_auto_verified_fake_passes_raw(self): self.assertEqual(self.evaluate_case("auto_single_face_heuristic")["classified_frames"],1)
    def test_unverified_fake_abstains(self): self.assertEqual(self.evaluate_case()["abstentions"]["subject_unverified"],1)
    def test_manual_verified_fake_passes_raw(self): self.assertEqual(self.evaluate_case("manual_registry")["classified_frames"],1)
    def test_edited_crop_missing_source_abstains(self): self.assertEqual(self.evaluate_case("auto_single_face_heuristic",edited=True)["classified_frames"],0)
