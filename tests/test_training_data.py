import hashlib
import json
from pathlib import Path

import pytest
from PIL import Image

from deepfake_data.training_data import (
    GroupBalancedBatchSampler,
    ImageVerificationCache,
    XceptionTrainingImageDataset,
    resolve_legacy_crop_path,
    audit_prior_sources,
    verify_legacy_large_train,
    verify_image_file,
)
from deepfake_data.training_smoke import select_smoke_rows
import deepfake_data.training_data as training_data_module
from deepfake_data.prepare_training import (
    V1_RELEASE_SHA256, classify_df40_duplicate, check_df40_method_coverage,
    is_excluded_df40_method, seed_verified_v1_cache,
)


def _png(path: Path, color=(100, 120, 140)) -> str:
    Image.new("RGB", (299, 299), color).save(path)
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _rows():
    rows = []
    fake_datasets = {"df40": ["method-a", "method-b"], "ffpp": ["FaceSwap"], "celebdf": ["celeb-synthesis"]}
    real_datasets = {"df40": ["CollabDiff"], "ffpp": ["original"], "celebdf": ["original"], "ffhq": ["original"]}
    for label, mapping in (("0", fake_datasets), ("1", real_datasets)):
        for dataset, methods in mapping.items():
            for method in methods:
                for n in range(20):
                    rows.append({"label": label, "dataset": dataset, "method": method,
                                 "source_group_id": f"{dataset}:{method}:{n}", "partition": "train",
                                 "content_group_id": f"sha256:{dataset}:{method}:{n}",
                                 "input_path": "unused.png"})
    return rows


def test_sampler_is_balanced_deterministic_and_group_distinct():
    rows = _rows()
    sampler = GroupBalancedBatchSampler(rows, batch_size=8, seed=17)
    first = list(sampler)
    assert first == list(GroupBalancedBatchSampler(rows, batch_size=8, seed=17))
    for batch in first[:10]:
        assert len(batch) == 8
        assert sum(rows[i]["label"] == "0" for i in batch) == 4
        assert len({rows[i]["source_group_id"] for i in batch}) == 8
    sampler.set_epoch(1)
    assert first != list(sampler)


@pytest.mark.parametrize("bad", [
    [{"label": "0", "dataset": "df40", "source_group_id": "g", "partition": "test"},
     {"label": "1", "dataset": "ffhq", "source_group_id": "r", "partition": "train"}],
    [{"label": "?", "dataset": "df40", "source_group_id": "g", "partition": "train"},
     {"label": "1", "dataset": "ffhq", "source_group_id": "r", "partition": "train"}],
])
def test_sampler_rejects_nontrain_or_unknown_labels(bad):
    with pytest.raises(ValueError):
        GroupBalancedBatchSampler(bad, batch_size=2)


def test_sampler_does_not_draw_duplicate_content_aliases_in_one_batch():
    rows = _rows()
    rows[0]["content_group_id"] = "sha256:alias"
    rows[1]["content_group_id"] = "sha256:alias"
    sampler = GroupBalancedBatchSampler(rows, batch_size=8, seed=2)
    for batch in list(sampler)[:20]:
        values = [rows[i]["content_group_id"] for i in batch]
        assert len(values) == len(set(values))


def test_dataset_reads_rgb_and_uses_clean_model_normalization(tmp_path):
    path = tmp_path / "sample.png"
    _png(path, (255, 0, 127))
    dataset = XceptionTrainingImageDataset([{
        "input_path": str(path), "partition": "train", "label": "1",
        "dataset": "ffhq", "source_group_id": "ffhq:sample",
    }])
    tensor, label = dataset[0]
    assert tuple(tensor.shape) == (3, 299, 299)
    assert label.item() == 1.0
    assert tensor[0, 0, 0].item() == pytest.approx(1.0)
    assert tensor[1, 0, 0].item() == pytest.approx(-1.0)


def test_legacy_large_crop_requires_matching_train_source_hash_and_integrity(tmp_path):
    crop = tmp_path / "crop.png"
    crop_hash = _png(crop)
    source_hash = "a" * 64
    source_path = r"D:\legacy\data\raw\manipulated_sequences\FaceSwap\c23\videos\000_001.mp4"
    old = {
        "split": "train", "dataset": "ffpp", "label": 0,
        "sha256": source_hash, "raw_path": source_path, "crop_path": str(crop),
        "crop_sha256": crop_hash, "frame_index": 12,
    }
    input_manifest = tmp_path / "input.json"
    input_manifest.write_text(json.dumps({"rows": [old]}), encoding="utf-8")
    release = [{
        "dataset": "ffpp", "sha256": source_hash,
        "relative_path": "manipulated_sequences/FaceSwap/c23/videos/000_001.mp4",
        "project_partition": "train", "label": "0", "absolute_path": "source.mp4",
        "source_group_id": "ffpp:000", "method": "FaceSwap",
    }]
    usable, report = verify_legacy_large_train(release, input_manifest)
    assert len(usable) == 1
    assert usable[0]["source_group_id"] == "ffpp:000"
    assert report["reusable_verified_unique_crops"] == 1

    release[0]["project_partition"] = "test"
    usable, _ = verify_legacy_large_train(release, input_manifest)
    assert usable == []


def test_legacy_nontrain_crop_may_be_reused_only_for_frozen_train_source(tmp_path):
    crop = tmp_path / "crop.png"
    crop_hash = _png(crop)
    old = {"split": "test", "dataset": "ffpp", "label": 0, "sha256": "c" * 64,
           "raw_path": r"D:\legacy\data\raw\x.mp4", "crop_path": str(crop),
           "crop_sha256": crop_hash, "frame_index": 2}
    manifest = tmp_path / "input.json"
    manifest.write_text(json.dumps({"rows": [old]}), encoding="utf-8")
    release = [{"dataset": "ffpp", "sha256": "c" * 64, "relative_path": "x.mp4",
                "project_partition": "train", "label": "0", "absolute_path": "x.mp4",
                "source_group_id": "ffpp:x", "method": "FaceSwap"}]
    usable, report = verify_legacy_large_train(release, manifest)
    assert len(usable) == 1
    assert usable[0]["partition"] == "train"
    assert report["prior_large_rows_by_split"]["test"] == 1


def test_legacy_crop_hash_mismatch_is_rejected(tmp_path):
    crop = tmp_path / "crop.png"
    _png(crop)
    old = {"split": "train", "dataset": "ffpp", "label": 0,
           "sha256": "a" * 64, "raw_path": r"D:\legacy\data\raw\x.mp4",
           "crop_path": str(crop), "crop_sha256": "b" * 64, "frame_index": 1}
    manifest = tmp_path / "input.json"
    manifest.write_text(json.dumps({"rows": [old]}), encoding="utf-8")
    release = [{"dataset": "ffpp", "sha256": "a" * 64, "relative_path": "x.mp4",
                "project_partition": "train", "label": "0", "absolute_path": "x.mp4",
                "source_group_id": "ffpp:x", "method": "FaceSwap"}]
    usable, report = verify_legacy_large_train(release, manifest)
    assert not usable
    assert report["invalid_or_duplicate"]["crop_hash_mismatch"] == 1


def test_legacy_crop_path_resolves_previous_project_crops_junction(tmp_path):
    project = tmp_path / "old-project"
    crop = project / "crops" / "ffpp-large" / "b6d7493c1193" / "train" / "fake" / "a.png"
    crop.parent.mkdir(parents=True)
    crop.write_bytes(b"existing crop")
    manifest = project / "runs" / "archive" / "run" / "input_manifest.json"
    manifest.parent.mkdir(parents=True)
    legacy_path = project / "experiment" / "crops" / "ffpp-large" / "b6d7493c1193" / "train" / "fake" / "a.png"
    resolved = resolve_legacy_crop_path(str(legacy_path), manifest)
    assert resolved == crop


def test_verified_image_reports_format_before_exif_transpose(tmp_path):
    path = tmp_path / "sample.jpg"
    Image.new("RGB", (256, 256)).save(path, format="JPEG")
    _, width, height, fmt = verify_image_file(path)
    assert (width, height, fmt) == (256, 256, "JPEG")


def test_smoke_selector_returns_verified_train_rows_with_both_labels(tmp_path):
    import csv
    import gzip
    paths = []
    for name in ("fake", "real"):
        path = tmp_path / f"{name}.png"
        _png(path)
        paths.append(path)
    manifest = tmp_path / "images.csv.gz"
    fields = ["input_path", "partition", "label", "dataset", "source_group_id"]
    with gzip.open(manifest, "wt", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerow({"input_path": paths[0], "partition": "train", "label": "0",
                         "dataset": "df40", "source_group_id": "fake-source"})
        writer.writerow({"input_path": paths[1], "partition": "train", "label": "1",
                         "dataset": "ffhq", "source_group_id": "real-source"})
    rows = select_smoke_rows(manifest)
    assert {row["label"] for row in rows} == {"0", "1"}


def test_image_verification_cache_reuses_unchanged_file_and_invalidates_changed_stat(tmp_path, monkeypatch):
    path = tmp_path / "cached.png"
    _png(path)
    actual = training_data_module.verify_image_file
    calls = 0

    def counted(image_path):
        nonlocal calls
        calls += 1
        return actual(image_path)

    monkeypatch.setattr(training_data_module, "verify_image_file", counted)
    cache_path = tmp_path / "cache.sqlite3"
    cache = ImageVerificationCache(cache_path)
    first = cache.verify(path)
    assert cache.verify(path) == first
    assert calls == 1
    cache.close()

    resumed = ImageVerificationCache(cache_path)
    assert resumed.verify(path) == first
    assert calls == 1
    stat = path.stat()
    import os
    os.utime(path, ns=(stat.st_atime_ns, stat.st_mtime_ns + 2_000_000))
    resumed.verify(path)
    assert calls == 2
    resumed.close()


def test_previous_verified_v1_manifest_seeds_cache_only_when_manifest_hash_matches(tmp_path, monkeypatch):
    import csv
    import gzip
    from deepfake_data.training_data import sha256_file

    processed = tmp_path / "processed"
    image = processed / "train" / "fake" / "one.png"
    image.parent.mkdir(parents=True)
    _png(image)
    legacy = tmp_path / "xception-v2-ready-v1"
    legacy.mkdir()
    manifest = legacy / "images.csv.gz"
    fields = ["input_path", "input_sha256", "media_type", "preprocessing_profile", "partition"]
    with gzip.open(manifest, "wt", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerow({"input_path": str(image), "input_sha256": "a" * 64,
                         "media_type": "video_crop", "preprocessing_profile": "v2-mtcnn-square299",
                         "partition": "train"})
    report = legacy / "preparation-report.json"
    report.write_text(json.dumps({"release_sha256": V1_RELEASE_SHA256,
                                  "manifest_sha256": sha256_file(manifest)}), encoding="utf-8")
    cache = ImageVerificationCache(tmp_path / "cache.sqlite3")
    seeded = seed_verified_v1_cache(cache, manifest=manifest, report_path=report, processed_root=processed)
    assert seeded == 1
    monkeypatch.setattr(training_data_module, "verify_image_file",
                        lambda _path: (_ for _ in ()).throw(AssertionError("unexpected rehash")))
    assert cache.verify(image) == ("a" * 64, 299, 299, "PNG")
    cache.close()

    stat = image.stat()
    import os
    os.utime(image, ns=(stat.st_atime_ns, stat.st_mtime_ns + 2_000_000_000))
    other = ImageVerificationCache(tmp_path / "other-cache.sqlite3")
    assert seed_verified_v1_cache(other, manifest=manifest, report_path=report,
                                  processed_root=processed) == 0  # touched after verified manifest
    report.write_text(json.dumps({"release_sha256": V1_RELEASE_SHA256,
                                  "manifest_sha256": "0" * 64}), encoding="utf-8")
    assert seed_verified_v1_cache(other, manifest=manifest, report_path=report,
                                  processed_root=processed) == 0  # report digest mismatch
    other.close()


def test_df40_duplicate_policy_preserves_method_aliases_and_rejects_label_conflicts():
    digest = "a" * 64
    seen = {digest: ("StyleGAN2", "0")}
    assert classify_df40_duplicate(digest, ("StyleGAN2", "0"), seen, set()) == "same_method_duplicate"
    assert classify_df40_duplicate(digest, ("StyleGAN3", "0"), seen, set()) == "cross_method_alias"
    assert classify_df40_duplicate(digest, ("RealReference", "1"), seen, set()) == "conflicting_label"
    assert classify_df40_duplicate("b" * 64, ("StyleGAN3", "0"), {}, set()) == "new"


def test_df40_coverage_requires_exactly_26_included_fake_methods():
    expected = {f"method-{n}" for n in range(26)}
    assert check_df40_method_coverage(expected, expected)["ready"] is True
    missing = check_df40_method_coverage(expected, expected - {"method-25"})
    assert missing["ready"] is False
    assert missing["missing"] == ["method-25"]


def test_sadtalker_and_wav2lip_are_excluded_without_label_changes():
    assert is_excluded_df40_method("SadTalker")
    assert is_excluded_df40_method("wav2lip")
    assert not is_excluded_df40_method("HeyGen")


def test_prior_clean_source_groups_are_flagged_in_evaluation_partitions(tmp_path):
    digest = "f" * 64
    prior = tmp_path / "prior.json"
    prior.write_text(json.dumps({"rows": [{"split": "train", "sha256": digest}]}), encoding="utf-8")
    release = [
        {"dataset": "ffpp", "sha256": digest, "source_group_id": "ffpp:linked", "project_partition": "train"},
        {"dataset": "ffpp", "sha256": "e" * 64, "source_group_id": "ffpp:linked", "project_partition": "test"},
    ]
    result = audit_prior_sources(release, prior)
    assert result["matched_v2_source_groups"] == 1
    assert result["evaluation_rows_excluded_from_independence_claim"]["ffpp"]["test"] == 1
    assert result["identity_independence_claim"].startswith("not established")

