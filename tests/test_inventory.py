import csv
import json
from pathlib import Path
from deepfake_data.inventory import generate, metadata_files, metadata_records, disk_key
from deepfake_data.split import split_rows, validate
from deepfake_data.extract import sampled_indices, choose_face


def test_inventory_reconciles_metadata(tmp_path):
    config=tmp_path / "configs"; config.mkdir()
    root=tmp_path / "data/raw/df40/extracted/e4s/cdf/frames/a"; root.mkdir(parents=True)
    for name in ("000.png", "unknown.gif", "points.npy", "archive.zip"): (root/name).touch()
    meta=tmp_path / "data/raw/df40/metadata/dataset_json"; meta.mkdir(parents=True)
    record={"e4s_cdf":{"e4s_Fake":{"test":{"a":{"label":"e4s_Fake","frames":["deepfakes_detection_datasets/DF40/e4s/cdf/frames/a/000.png", "deepfakes_detection_datasets/DF40/e4s/cdf/frames/a/missing.png"]}}}}}
    (meta/"e4s_cdf.json").write_text(json.dumps(record))
    (meta/"DF40_all.json").write_text(json.dumps(record))
    (config/"datasets.json").write_text(json.dumps({"raw_root":"data/raw","output_dir":"artifacts/inventory","datasets":{"df40":{"path":"df40/extracted"}}}))
    manifest, report=generate(config/"datasets.json")
    with manifest.open(newline="") as f: rows=list(csv.DictReader(f))
    assert len(rows)==2
    assert rows[0]["method"] == "e4s" and rows[0]["label"] == "0" and rows[0]["original_partition"] == "test"
    assert rows[1]["label"] == "" and rows[1]["media_type"] == "gif"
    assert report["unmatched_metadata_paths"] == 1
    assert report["discovered_extensions"]["df40"][".npy"] == 1
    assert report["metadata_selected"] == ["e4s_cdf.json"]


def test_metadata_conflict_and_duplicates(tmp_path):
    meta=tmp_path/"x.json"
    meta.write_text(json.dumps({"x":{"x_Real":{"test":{"v":{"label":"x_Real","frames":["a", "a"]}}}}}))
    assert [x["label"] for x in metadata_records(meta)] == [1,1]
    assert disk_key(Path("x/x/real/a.jpg")) == "x/real/a.jpg"


def test_splits_quarantine_and_holdout():
    rows=[dict(dataset="ffpp",method="a",label="0",group_id="g1",group_status="verified",verified_independent_of="celebdf"),dict(dataset="ffpp",method="b",label="1",group_id="g1",group_status="verified",verified_independent_of="celebdf"),dict(dataset="df40",method="c",label="0",group_id="",group_status="unresolved"),dict(dataset="celebdf",method="r",label="1",group_id="c1",group_status="verified")]
    assert split_rows(rows,42)==split_rows(rows,42)
    split=split_rows(rows,42,holdout="celebdf")
    assert next(r for r in split if r["dataset"]=="celebdf")["project_partition"] == "test"
    assert next(r for r in split if r["dataset"]=="df40")["project_partition"] == "quarantine"
    assert len({r["project_partition"] for r in split if r["group_id"]=="g1"})==1
    assert not validate(split)["group_overlap"]


def test_detect_overlap():
    report=validate([{"project_partition":"train","group_id":"g","sha256":"a"},{"project_partition":"test","group_id":"g","sha256":"a"}])
    assert "g" in report["group_overlap"] and "a" in report["hash_overlap"]


def test_frame_sampling_and_face_selection():
    assert sampled_indices(100,3)==[0,49,99]
    assert sampled_indices(0,3)==[]
    assert choose_face([[0,0,3,3],[0,0,5,5]],[.99,.95],["a","b"],.9)[1] == "b"
    assert choose_face([[0,0,3,3]],[.2],["a"],.9) is None


def test_experimental_training_never_evaluates_unverified():
    row={"dataset":"df40","method":"e4s","source_domain":"cdf","label":"0","group_id":"df40:e4s:cdf:a","group_status":"sample_verified"}
    assert split_rows([row],protocol="experimental")[0]["project_partition"] == "train"
    assert split_rows([row],protocol="strict")[0]["project_partition"] == "quarantine"
    assert split_rows([row],holdout="celebdf",protocol="experimental")[0]["project_partition"] == "quarantine"


def test_unknown_labels_remain_quarantined():
    row={"dataset":"df40","label":"","group_id":"g","group_status":"sample_verified"}
    assert split_rows([row],protocol="experimental")[0]["project_partition"] == "quarantine"


def test_invalid_ratios_rejected():
    import pytest
    with pytest.raises(ValueError): split_rows([],ratios=(.1,.1,.1))


def test_training_manifest_includes_images_and_reports_unprocessed_video(tmp_path):
    from deepfake_data.training_manifest import build_rows
    rows=[{"dataset":"df40","media_type":"image","absolute_path":"face.png","label":"0","project_partition":"train"}, {"dataset":"ffpp","media_type":"video","absolute_path":"video.mp4","relative_path":"video.mp4","label":"1","project_partition":"train"}, {"dataset":"df40","media_type":"gif","absolute_path":"a.gif","label":"0","project_partition":"train"}]
    included, skipped=build_rows(rows,tmp_path)
    assert included[0]["input_path"] == "face.png"
    assert {r["reason"] for r in skipped} == {"not_extracted","gif_requires_conversion"}
