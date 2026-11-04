import csv
import hashlib
import json
from pathlib import Path

import pytest

from deepfake_data.baseline_eval import (
    average_binary_probabilities,
    build_common_support_comparison,
    calibrate_real_probabilities,
    calculate_metrics,
    load_prior_training,
    legacy_face_indices,
    normalize_saved_calibration,
    prior_source_overlap,
    real_to_fake_probabilities,
    select_evaluation_records,
    write_selection,
)


def _write_image(path: Path, color=(120, 80, 40)) -> None:
    from PIL import Image
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (96, 96), color).save(path, format="PNG")


def _row(path: Path, *, method="HeyGen", label="0", group="g0", relative=None):
    return {
        "dataset": "df40",
        "release": "DF40 official testing release",
        "method": method,
        "source_domain": "test",
        "relative_path": relative or path.name,
        "absolute_path": str(path),
        "media_type": "image",
        "label": label,
        "original_json_label": "fake" if label == "0" else "real",
        "original_partition": "test|val",
        "project_partition": "quarantine",
        "sample_id": f"sample-{group}",
        "group_id": f"df40:{method}:{group}",
        "group_status": "sample_verified",
        "relationship_status": "unknown",
        "validation_status": "not_decoded",
        "metadata_sources": "fixture",
        "sha256": "",
        "metadata_status": "matched",
        "experiment_eligibility": "eligible_labeled_record",
        "eligibility_reason": "synthetic test fixture",
    }


def _write_inventory(path: Path, rows):
    fields = list(rows[0])
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def test_source_video_link_to_prior_training_is_excluded(tmp_path):
    prior = load_prior_training([_old_manifest(tmp_path)])
    assert prior_source_overlap({"group_id": "df40:StyleGAN2:cdf:id0_id16_0003"}, prior) == "exact_celebdf_training_video_id"
    assert prior_source_overlap({"group_id": "df40:DiT:ff:001"}, prior) == "exact_ffpp_source_id_in_training_groups"
    assert prior_source_overlap({"group_id": "df40:DiT:ff:002"}, prior) == ""


def _old_manifest(tmp_path: Path) -> Path:
    path = tmp_path / "baseline-eval-prior.json"
    path.write_text(json.dumps({"hash": "fixture-hash", "rows": [
        {"split": "train", "dataset": "celebdf", "video_id": "celebdf:Celeb-DF-v2:id0_id16_0003", "crop_sha256": "a" * 64},
        {"split": "train", "dataset": "ffpp", "cluster": "ffpp:001|ffpp:870", "crop_sha256": "b" * 64},
        {"split": "test", "dataset": "celebdf", "video_id": "celebdf:Celeb-DF-v2:id0_id16_0004"},
    ]}), encoding="utf-8")
    return path


def test_selection_is_deterministic_verified_and_group_separated(tmp_path):
    prior = _old_manifest(tmp_path)
    files = []
    rows = []
    for label in ("0", "1"):
        for i in range(4):
            image = tmp_path / f"{label}-{i}.png"
            _write_image(image, (30 + i, 70 + i, 110 + i))
            rows.append(_row(image, method="HeyGen", label=label, group=f"{label}-{i}"))
    inventory = tmp_path / "inventory.csv"
    _write_inventory(inventory, rows)
    first, report = select_evaluation_records(inventory, None, [prior], seed=7, fake_groups_per_method=2, real_groups_per_method=2)
    second, _ = select_evaluation_records(inventory, None, [prior], seed=7, fake_groups_per_method=2, real_groups_per_method=2)
    assert [row["relative_path"] for row in first] == [row["relative_path"] for row in second]
    assert len(first) == 4
    assert {row["label"] for row in first} == {"0", "1"}
    assert len({row["group_id"] for row in first}) == len(first)
    assert all(row["reservation_partition"] == "baseline_eval_test" for row in first)
    assert report["selected_labels"] == {"fake_0": 2, "real_1": 2}
    manifest, report_path = write_selection(first, report, tmp_path / "release")
    assert manifest.exists() and report_path.exists()
    with manifest.open(newline="", encoding="utf-8") as stream:
        saved = list(csv.DictReader(stream))
    assert len(saved) == 4


def test_exact_duplicates_do_not_become_independent_evaluation_images(tmp_path):
    prior = _old_manifest(tmp_path)
    image = tmp_path / "copy.png"
    _write_image(image)
    copy = tmp_path / "copy2.png"
    copy.write_bytes(image.read_bytes())
    rows = [
        _row(image, method="HeyGen", label="0", group="fake-a"),
        _row(copy, method="HeyGen", label="0", group="fake-b"),
    ]
    # One group is a direct match to the prior crop hash only when its bytes match;
    # this fixture uses distinct raw bytes, so duplicate selection is the expected gate.
    real = tmp_path / "real.png"
    _write_image(real, (1, 2, 3))
    rows.append(_row(real, method="HeyGen", label="1", group="real-a"))
    inventory = tmp_path / "inventory.csv"
    _write_inventory(inventory, rows)
    selected, report = select_evaluation_records(inventory, None, [prior], seed=2, fake_groups_per_method=2, real_groups_per_method=1)
    assert len([row for row in selected if row["label"] == "0"]) == 1
    assert report["excluded_counts"].get("exact_duplicate_within_evaluation", 0) == 1


def test_unverified_labels_are_not_selected(tmp_path):
    prior = _old_manifest(tmp_path)
    image = tmp_path / "unknown.png"
    _write_image(image)
    row = _row(image, label="unknown")
    inventory = tmp_path / "inventory.csv"
    _write_inventory(inventory, [row])
    selected, report = select_evaluation_records(inventory, None, [prior])
    assert selected == []
    assert report["excluded_counts"]["unknown_or_unverified_label"] == 1


def test_df40_group_linked_to_official_celeb_test_component_is_excluded(tmp_path):
    prior = _old_manifest(tmp_path)
    official = _row(tmp_path / "official.mp4", method="original", label="1", group="official")
    official.update({"dataset": "celebdf", "group_id": "celebdf:original:official", "original_partition": "official_test", "experiment_eligibility": "eligible_labeled_record", "media_type": "video"})
    blocked = tmp_path / "linked.png"
    allowed = tmp_path / "unlinked.png"
    _write_image(blocked, (1, 4, 7))
    _write_image(allowed, (8, 4, 1))
    fake_linked = _row(blocked, method="HeyGen", label="0", group="linked")
    fake_unlinked = _row(allowed, method="HeyGen", label="0", group="unlinked")
    inventory = tmp_path / "inventory.csv"
    _write_inventory(inventory, [official, fake_linked, fake_unlinked])
    components = tmp_path / "components.csv"
    components.write_text(
        "group_id,component_id,evidence_status\n"
        "celebdf:original:official,protected-test-component,contains_suspected_links\n"
        "df40:HeyGen:linked,protected-test-component,contains_suspected_links\n"
        "df40:HeyGen:unlinked,separate-component,unlinked_or_within_dataset_only\n",
        encoding="utf-8",
    )
    selected, report = select_evaluation_records(inventory, components, [prior], fake_groups_per_method=1, real_groups_per_method=0)
    assert [row["group_id"] for row in selected] == ["df40:HeyGen:unlinked"]
    assert report["excluded_counts"]["protected_celebdf_official_test_component"] == 1


def test_conflicting_exact_duplicate_is_quarantined(tmp_path):
    prior = _old_manifest(tmp_path)
    same = tmp_path / "same.png"
    _write_image(same)
    fake = _row(same, method="HeyGen", label="0", group="fake")
    real = _row(same, method="HeyGen", label="1", group="real")
    inventory = tmp_path / "inventory.csv"
    _write_inventory(inventory, [fake, real])
    selected, report = select_evaluation_records(inventory, None, [prior], fake_groups_per_method=1, real_groups_per_method=1)
    assert selected == []
    assert len(report["conflicting_selected_hashes"]) == 1
    assert report["excluded_counts"]["conflicting_label_duplicate"] == 1


def test_completed_project_hash_audit_excludes_exact_source_digest(tmp_path):
    prior = _old_manifest(tmp_path)
    image = tmp_path / "audited.png"
    _write_image(image)
    digest = hashlib.sha256(image.read_bytes()).hexdigest()
    inventory = tmp_path / "inventory.csv"
    _write_inventory(inventory, [_row(image, method="HeyGen", label="0", group="fake-a")])
    hash_audit = tmp_path / "hash-audit.csv"
    with hash_audit.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=["sha256", "status", "partition"])
        writer.writeheader()
        writer.writerow({"sha256": digest, "status": "ok", "partition": "train"})
    selected, report = select_evaluation_records(
        inventory, None, [prior], fake_groups_per_method=1, real_groups_per_method=0,
        hash_audit_path=hash_audit,
    )
    assert selected == []
    assert report["excluded_counts"]["exact_hash_matches_completed_source_audit"] == 1


def test_metrics_use_fake_zero_as_positive_class():
    result = calculate_metrics([0, 0, 1, 1], [0.9, 0.1, 0.7, 0.2], threshold=0.5)
    assert result["accuracy"] == pytest.approx(0.5)
    assert result["balanced_accuracy"] == pytest.approx(0.5)
    assert result["precision_fake"] == pytest.approx(0.5)
    assert result["recall_fake"] == pytest.approx(0.5)
    assert result["false_positive_rate"] == pytest.approx(0.5)
    assert result["false_negative_rate"] == pytest.approx(0.5)
    assert result["confusion_matrix_true_fake_real_pred_fake_real"] == [[1, 1], [1, 1]]
    assert result["roc_auc_fake"] == pytest.approx(0.5)


def test_roc_auc_uses_average_ranks_for_tied_scores():
    result = calculate_metrics([0, 1], [0.5, 0.5], threshold=0.5)
    assert result["roc_auc_fake"] == pytest.approx(0.5)


def test_legacy_horizontal_flip_averaging_returns_probability_not_sum():
    assert average_binary_probabilities(0.8, 0.6) == pytest.approx(0.7)
    with pytest.raises(ValueError):
        average_binary_probabilities(0.8, 1.2)


def test_focal_legacy_crop_keeps_best_low_confidence_face_but_rejects_too_narrow_box():
    boxes = [[0, 0, 60, 18], [0, 0, 49, 100], [0, 0, 80, 80]]
    probabilities = [0.12, 0.99, 0.0]
    assert legacy_face_indices(boxes, probabilities) == [0, 2]


def test_saved_calibration_without_method_uses_original_temperature_default():
    calibration = normalize_saved_calibration({"temperature": 2.3, "threshold": 0.75})
    assert calibration["method"] == "temperature"
    assert calibration["temperature"] == 2.3


def test_none_calibration_is_identity_on_real_probabilities():
    import numpy as np

    scores = np.asarray([0.0, 0.2, 0.7, 1.0])
    result = calibrate_real_probabilities(scores, {"method": "none"}, lambda *_: pytest.fail("identity must not call legacy calibrator"))
    assert np.array_equal(result, scores)
    assert result is not scores


@pytest.mark.parametrize("method", ["temperature", "platt", "isotonic", "median"])
def test_supported_calibration_methods_delegate_to_previous_implementation(method):
    import numpy as np

    seen = {}

    def legacy_apply(probabilities, calibration):
        seen["method"] = calibration["method"]
        seen["probabilities"] = probabilities.copy()
        return probabilities * 0 + 0.375

    result = calibrate_real_probabilities([0.2, 0.8], {"method": method}, legacy_apply)
    assert seen["method"] == method
    assert np.array_equal(seen["probabilities"], np.asarray([0.2, 0.8]))
    assert np.array_equal(result, np.asarray([0.375, 0.375]))


def test_unknown_calibration_method_is_not_silently_identity():
    def legacy_apply(_probabilities, calibration):
        raise ValueError(f"Unknown calibration method: {calibration.get('method')}")

    with pytest.raises(ValueError, match="Unknown calibration method: mystery"):
        calibrate_real_probabilities([0.2, 0.8], {"method": "mystery"}, legacy_apply)


def test_model_real_scores_are_inverted_for_fake_zero_class_metrics():
    fake_scores = real_to_fake_probabilities([0.1, 0.9])
    assert fake_scores == pytest.approx([0.9, 0.1])
    metrics = calculate_metrics([0, 1], fake_scores, threshold=0.5)
    assert metrics["accuracy"] == 1.0
    assert metrics["recall_fake"] == 1.0


@pytest.mark.parametrize("scores", [[-0.1, 0.4], [0.2, 1.1], [float("nan")], []])
def test_identity_calibration_rejects_non_probabilities(scores):
    with pytest.raises(ValueError, match="finite one-dimensional probabilities"):
        calibrate_real_probabilities(scores, {"method": "none"}, lambda *_: None)


def test_common_support_recalculates_metrics_on_intersection_and_keeps_failures_separate():
    selection = [
        {"evaluation_id": "a", "label": "0", "method": "DiT"},
        {"evaluation_id": "b", "label": "1", "method": "HeyGen"},
        {"evaluation_id": "c", "label": "0", "method": "MRAA"},
        {"evaluation_id": "d", "label": "1", "method": "HeyGen"},
        {"evaluation_id": "e", "label": "0", "method": "DiT"},
    ]

    def pred(evaluation_id, label, real_probability, fake_probability):
        return {
            "evaluation_id": evaluation_id,
            "label": label,
            "real_probability_raw": real_probability,
            "fake_probability_scored": fake_probability,
        }

    report = {
        "status": "failed_common_image_support",
        "models": [
            {
                "name": "clean", "checkpoint_sha256": "clean-hash",
                "preprocessing": {"contract": "clean"},
                "saved_calibration_operating_point": {"threshold": 0.8, "calibration_method": "temperature"},
                "predictions": [pred("a", 0, .1, .9), pred("b", 1, .9, .1), pred("c", 0, .7, .3), pred("d", 1, .3, .7)],
                "failures": [{"evaluation_id": "e", "reason": "no face"}],
            },
            {
                "name": "mixed", "checkpoint_sha256": "mixed-hash",
                "preprocessing": {"contract": "mixed"},
                "saved_calibration_operating_point": {"threshold": 0.7, "calibration_method": "platt"},
                "predictions": [pred("a", 0, .2, .8), pred("b", 1, .8, .2), pred("c", 0, .4, .6), pred("e", 0, .3, .7)],
                "failures": [{"evaluation_id": "d", "reason": "decode failure"}],
            },
            {
                "name": "focal", "checkpoint_sha256": "focal-hash",
                "preprocessing": {"contract": "focal"},
                "saved_calibration_operating_point": {"threshold": 0.5, "calibration_method": "none"},
                "predictions": [pred("a", 0, .15, .85), pred("b", 1, .85, .15), pred("c", 0, .45, .55), pred("d", 1, .3, .7), pred("e", 0, .2, .8)],
                "failures": [],
            },
        ],
    }
    result = build_common_support_comparison(report, selection)

    assert result["common_successful_images"] == 3
    assert result["common_evaluation_ids"] == ["a", "b", "c"]
    assert result["common_class_counts"] == {"fake_0": 2, "real_1": 1}
    assert [model["common_support_images"] for model in result["models"]] == [3, 3, 3]
    assert all(model["fixed_raw_fake_score_threshold_0_5"]["samples"] == 3 for model in result["models"])
    assert all(model["saved_calibration_operating_point"]["metrics"]["samples"] == 3 for model in result["models"])
    assert result["models"][0]["fixed_raw_fake_score_threshold_0_5"]["confusion_matrix_true_fake_real_pred_fake_real"] == [[1, 1], [0, 1]]
    assert result["models"][1]["failed_images_in_original_report"] == [{"evaluation_id": "d", "reason": "decode failure"}]
    assert {row["evaluation_id"] for row in result["excluded_from_common_support"]} == {"d", "e"}
    assert result["source_comparison_status"] == "failed_common_image_support"
    assert report["status"] == "failed_common_image_support"  # source report was not mutated

