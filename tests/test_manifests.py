import unittest
import json
from tests.support import module, workspace


def record(name, split, parents, *, people=None, sha=None):
    return {"sample_id": name, "split": split, "parents": parents,
            "known_person_ids": people or [], "sha256": sha or name,
            "identity_verified": False, "label": 0, "dataset": "ffpp"}


class ManifestTests(unittest.TestCase):
    def test_bootstrap_families_join_both_parents_and_originals(self):
        api = module(self, "manifests")
        rows = [record("a", "test", ["x"]), record("b", "test", ["y"]), record("c", "test", ["x", "y"])]
        self.assertEqual(len(set(api.dependency_families(rows))), 1)
    def test_duplicate_crops_across_partitions_are_rejected(self):
        api = module(self, "manifests")
        rows = [record("a", "train", ["x"]), record("b", "test", ["y"])]
        for row in rows:
            row["crop_sha256"] = "same-crop"
        self.assertFalse(api.check_splits(rows)["ok"])

    def test_prepare_writes_usable_manifest_and_cross_pair_quarantine(self):
        api = module(self, "manifests")
        cfg_api = module(self, "config")
        paths_api = module(self, "paths")
        root = workspace(self)
        raw = root / "raw"
        originals = raw / "original_sequences/youtube/c23/videos"
        fake = raw / "manipulated_sequences/Deepfakes/c23/videos"
        originals.mkdir(parents=True)
        fake.mkdir(parents=True)
        for name in ("004", "982", "110", "111", "800", "801"):
            (originals / (name + ".mp4")).write_bytes(name.encode())
        for name in ("004_982", "004_111"):
            (fake / (name + ".mp4")).write_bytes(name.encode())
        official = root / "official"
        official.mkdir()
        for split, pairs in {"train": [["004", "982"]], "val": [["110", "111"]], "test": [["800", "801"]]}.items():
            (official / (split + ".json")).write_text(json.dumps(pairs))
        cfg = cfg_api.load_config(root=root, overrides={"raw_root": str(raw), "official_splits": str(official)})
        report = api.prepare("ffpp", cfg, paths_api.Workspace(root))
        rows = paths_api.read_jsonl(root / "runs/manifests/ffpp/videos.jsonl")
        self.assertTrue(report["ok"])
        self.assertFalse(report["pilot"])
        self.assertTrue(all(not row["pilot"] for row in rows))
        matching = next(r for r in rows if r["raw_path"].endswith("004_982.mp4"))
        self.assertEqual(matching["split"], "train")
        self.assertEqual(matching["label"], 0)
        rejected = paths_api.read_jsonl(root / "runs/manifests/ffpp/quarantine.jsonl")
        self.assertEqual(rejected[0]["reason"], "cross_partition_parents")
        limited = api.prepare("ffpp", cfg, paths_api.Workspace(root), name="limited", limit=2)
        self.assertTrue(limited["pilot"])
        self.assertTrue(all(row["pilot"] and row["pilot_limit"] == 2 for row in paths_api.read_jsonl(root / "runs/manifests/limited/videos.jsonl")))

    def test_both_official_parents_use_same_split_not_numeric_range(self):
        api = module(self, "manifests")
        assignments = api.ffpp_assignments({"train": [["004", "982"]], "val": [["010", "011"], ["020", "021"]], "test": [["800", "801"]]}, 42)
        self.assertEqual(assignments["004"], "train")
        self.assertEqual(assignments["982"], "train")
        self.assertEqual(assignments["010"], assignments["011"])
        self.assertEqual({assignments["010"], assignments["020"]}, {"val", "calibration"})

    def test_source_overlap_is_rejected(self):
        api = module(self, "manifests")
        result = api.check_splits([record("a", "train", ["ffpp:004"]), record("b", "test", ["ffpp:004"])])
        self.assertFalse(result["ok"])
        self.assertEqual(result["parent_overlap"], ["ffpp:004"])

    def test_known_identity_overlap_and_duplicate_content_are_rejected(self):
        api = module(self, "manifests")
        result = api.check_splits([record("a", "train", ["x"], people=["p"], sha="same"), record("b", "external", ["y"], people=["p"], sha="same")])
        self.assertFalse(result["ok"])
        self.assertEqual(result["known_person_overlap"], ["p"])
        self.assertEqual(result["duplicate_content_overlap"], ["same"])

    def test_unknown_people_are_not_claimed_identity_disjoint(self):
        api = module(self, "manifests")
        result = api.check_splits([record("a", "train", ["x"]), record("b", "test", ["y"])])
        self.assertTrue(result["ok"])
        self.assertFalse(result["person_disjoint_verified"])

    def test_cross_boundary_pair_is_quarantined(self):
        api = module(self, "manifests")
        admitted, rejected = api.admit([record("a", "train", ["ffpp:004", "ffpp:982"])], {"ffpp:004": "train", "ffpp:982": "test"})
        self.assertEqual(admitted, [])
        self.assertEqual(rejected[0]["reason"], "cross_partition_parents")

    def test_strict_person_policy_rejects_unknown_identity(self):
        api = module(self, "manifests")
        admitted, rejected = api.admit([record("a", "train", ["x"])], {"x": "train"}, policy="strict-person")
        self.assertFalse(admitted)
        self.assertEqual(rejected[0]["reason"], "unverified_person")

    def test_missing_parent_provenance_is_quarantined(self):
        api = module(self, "manifests")
        admitted, rejected = api.admit([record("a", "train", ["missing"])], {})
        self.assertEqual(admitted, [])
        self.assertEqual(rejected[0]["reason"], "unresolved_parent")

    def test_pilot_sampling_includes_each_class_partition(self):
        api = module(self, "manifests")
        rows = []
        for split in ("train", "val", "calibration", "test"):
            for label in (0, 1):
                for i in range(3):
                    row = record(f"{split}-{label}-{i}", split, [f"{split}-{label}-{i}"])
                    row["label"] = label
                    rows.append(row)
        selected = api.stratified_limit(rows, 8, 42)
        self.assertEqual({(r["split"], r["label"]) for r in selected}, {(s, c) for s in ("train", "val", "calibration", "test") for c in (0, 1)})


if __name__ == "__main__":
    unittest.main()
