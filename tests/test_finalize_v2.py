import csv
from pathlib import Path

from deepfake_data.finalize_v2 import (
    TARGET_DF40_METHODS,
    UnionFind,
    _cache_hash,
    build_release,
    stable_partition,
)
import sqlite3


def _write_csv(path: Path, fields, rows):
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _group(partition, prefix):
    i = 0
    while stable_partition(f"{prefix}-{i}", 123) != partition:
        i += 1
    return f"{prefix}-{i}"


def test_stable_group_split_is_deterministic_and_row_order_independent():
    assert stable_partition("same-component", 7) == stable_partition("same-component", 7)
    assert stable_partition("same-component", 7) != stable_partition("same-component", 8)
    groups = UnionFind(["original", "manipulated"])
    groups.union("original", "manipulated")
    assert groups.find("original") == groups.find("manipulated")


def test_hash_cache_reuses_only_matching_file_state(tmp_path):
    path = tmp_path / "sample.bin"
    path.write_bytes(b"first")
    stat = path.stat()
    db = sqlite3.connect(":memory:")
    db.execute("CREATE TABLE hashes(path TEXT PRIMARY KEY,sha256 TEXT,size INTEGER,mtime_ns INTEGER,status TEXT)")
    first, source = _cache_hash(db, str(path), stat.st_size, stat.st_mtime_ns, {})
    assert source == "new_sha256"
    assert _cache_hash(db, str(path), stat.st_size, stat.st_mtime_ns, {}) == (first, "resume_cache")
    path.write_bytes(b"changed-content")
    changed = path.stat()
    second, source = _cache_hash(db, str(path), changed.st_size, changed.st_mtime_ns, {})
    assert source == "new_sha256"
    assert second != first


def test_build_release_keeps_verified_groups_and_exact_duplicates_together(tmp_path):
    media = tmp_path / "media"
    media.mkdir()
    inventory = tmp_path / "inventory.csv"
    old_groups = tmp_path / "old-groups.csv"
    components = tmp_path / "components.csv"
    hashes = tmp_path / "completed-hashes.csv"
    baseline = tmp_path / "baseline.csv"
    fields = ["dataset", "release", "method", "source_domain", "relative_path", "absolute_path",
              "media_type", "label", "original_json_label", "original_partition", "project_partition",
              "sample_id", "group_id", "group_status", "relationship_status", "validation_status",
              "metadata_sources", "sha256", "metadata_status", "experiment_eligibility", "eligibility_reason"]
    rows, group_rows, component_rows = [], [], []

    def add(dataset, method, label, group, *, official="", content=None):
        rel = f"{dataset}/{method}/{group}.png"
        path = media / f"{len(rows):06d}.png"
        path.write_bytes(content if content is not None else f"{dataset}:{method}:{group}:{label}".encode())
        rows.append({"dataset": dataset, "release": dataset, "method": method,
                     "source_domain": "test", "relative_path": rel, "absolute_path": str(path),
                     "media_type": "image", "label": str(label), "original_json_label": str(label),
                     "original_partition": official, "project_partition": "", "sample_id": group,
                     "group_id": group, "group_status": "sample_verified", "relationship_status": "unknown",
                     "validation_status": "not_decoded", "metadata_sources": "fixture", "sha256": "",
                     "metadata_status": "matched", "experiment_eligibility": "eligible_labeled_record",
                     "eligibility_reason": "fixture"})
        group_rows.append({"dataset": dataset, "relative_path": rel, "group_id": group})
        component_rows.append({"group_id": group, "component_id": group,
                               "evidence_status": "unlinked_or_within_dataset_only"})
        return rel, path

    # Both labels occur in every project partition for the three contributing datasets.
    for dataset in ("ffpp", "celebdf"):
        for partition in ("train", "val", "calibration", "test"):
            for label in (0, 1):
                group = _group(partition, f"{dataset}-{label}")
                official = "official_test" if dataset == "celebdf" and partition == "test" else ""
                add(dataset, "original" if label == 1 else "fake", label, group, official=official)

    # Every target DF40 method has a train fake group, with multiple examples in one
    # component proving that related media remain together.
    for method in sorted(TARGET_DF40_METHODS):
        group = _group("train", f"df40-{method}")
        add("df40", method, 0, group)
    add("df40", "MidJourney", 1, _group("train", "df40-midjourney-real"))
    linked_group = _group("val", "linked-original")
    linked_other = _group("test", "linked-derivative")
    official_group = _group("test", "celebdf-1")
    _, shared = add("df40", "MidJourney", 0, linked_group, content=b"exact-duplicate")
    add("ffpp", "DeepFakeDetection", 0, linked_other, content=b"exact-duplicate")
    exact_a = _group("val", "exact-copy-a")
    exact_b = _group("train", "exact-copy-b")
    exact_a_path, _ = add("df40", "MidJourney", 0, exact_a, content=b"byte-identical-independent-groups")
    exact_b_path, _ = add("ffpp", "DeepFakeDetection", 0, exact_b, content=b"byte-identical-independent-groups")
    suspected_train = _group("train", "suspected-link-train")
    suspected_val = _group("val", "suspected-link-val")
    add("df40", "MRAA", 0, suspected_train)
    add("ffpp", "DeepFakeDetection", 0, suspected_val)
    # The source-component mapping links them even before exact-byte union.
    for entry in component_rows:
        if entry["group_id"] in {linked_group, linked_other, official_group}:
            entry["component_id"] = "verified-source-link"
        if entry["group_id"] in {suspected_train, suspected_val}:
            entry["component_id"] = "suspected-cross-dataset-link"
            entry["evidence_status"] = "contains_suspected_links"

    baseline_group = _group("train", "baseline-reserved")
    rel, _ = add("df40", "DiT", 0, baseline_group)
    empty_group = _group("train", "empty-fixture")
    empty_rel, empty_path = add("df40", "MidJourney", 0, empty_group)
    empty_path.write_bytes(b"")
    _write_csv(inventory, fields, rows)
    _write_csv(old_groups, ["dataset", "relative_path", "group_id"], group_rows)
    _write_csv(components, ["group_id", "component_id", "evidence_status"], component_rows)
    _write_csv(hashes, ["absolute_path", "sha256", "status", "size", "mtime_ns"], [])
    _write_csv(baseline, ["dataset", "relative_path", "group_id", "source_component_id"],
               [{"dataset": "df40", "relative_path": rel, "group_id": baseline_group,
                 "source_component_id": baseline_group}])

    report = build_release(inventory=inventory, old_groups=old_groups, components=components,
                           completed_hashes=hashes, baseline_selection=baseline,
                           output_dir=tmp_path / "release", seed=123, progress_every=10000)
    assert report["status"] == "PASS_FROZEN", report["blockers"]
    assert report["empty_source_files_quarantined"] == 1
    assert report["suspected_relationship_component_cross_partition_count"] >= 1
    assert len(report["target_df40_methods_with_fake_training"]) == 26
    with (tmp_path / "release/df40-focused-v2.csv").open(newline="", encoding="utf-8") as stream:
        manifest = list(csv.DictReader(stream))
    by_group = {}
    for row in manifest:
        if row["project_partition"] in {"train", "val", "calibration", "test"}:
            by_group.setdefault(row["source_component_id"], set()).add(row["project_partition"])
    assert by_group["verified-source-link"] == {"test"}  # one duplicate links to official test anchor
    assert next(r for r in manifest if r["relative_path"] == rel)["project_partition"] == "test"
    empty_row = next(r for r in manifest if r["relative_path"] == empty_rel)
    assert empty_row["project_partition"] == "quarantine"
    assert empty_row["quarantine_reason"] == "empty_source_file"
    by_path = {r["relative_path"]: r for r in manifest}
    assert by_path[exact_a_path]["project_partition"] == by_path[exact_b_path]["project_partition"]
    suspected_rows = [r for r in manifest if r["source_group_id"] in {suspected_train, suspected_val}]
    assert {r["project_partition"] for r in suspected_rows} == {"train", "val"}
    assert all(r["relationship_component_id"] == "suspected-cross-dataset-link" for r in suspected_rows)
    assert all(r["source_component_id"] == r["source_group_id"] for r in suspected_rows)
    assert all(r["sha256"] for r in manifest if r["project_partition"] in {"train", "val", "calibration", "test"})
    assert all(r["label"] in {"0", "1"} for r in manifest if r["project_partition"] in {"train", "val", "calibration", "test"})

