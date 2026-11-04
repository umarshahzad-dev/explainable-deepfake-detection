"""Read-only, group-reserved evaluation of legacy Xception checkpoints on DF40.

The selector writes references and hashes only; it never copies or edits source media.
The generated source groups and content digests must be excluded from any later
training release. The model runtime is imported lazily from the previous project.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
import sys
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator, Mapping, Sequence


TARGET_METHODS = (
    "CollabDiff", "DiT", "MRAA", "MidJourney", "RDDM", "SiT", "StyleGAN2",
    "StyleGAN3", "StyleGANXL", "e4s", "facedancer", "fsgan", "heygen",
    "hyperreenact", "inswap", "lia", "mcnet", "mobileswap", "one_shot_free",
    "pirender", "pixart", "simswap", "starganv2", "tpsm", "uniface",
    "whichfaceisreal",
)
REAL_METHODS = ("CollabDiff", "HeyGen", "MidJourney", "StarGANv2", "WhichFaceIsReal")
SELECTION_FIELDS = (
    "evaluation_id", "protocol", "reservation_partition", "dataset", "release",
    "method", "source_domain", "relative_path", "absolute_path", "media_type",
    "label", "original_json_label", "original_partition", "sample_id", "group_id",
    "source_component_id", "group_status", "relationship_status", "validation_status",
    "source_sha256", "prior_source_overlap", "prior_training_exact_hash_overlap",
    "selection_seed", "selection_reason",
)


def sha256_file(path: str | Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _commit_generated_temp(temp: Path, target: Path) -> None:
    """Promote a same-directory report temp, with a Windows sandbox fallback."""
    try:
        temp.replace(target)
    except PermissionError:
        # Some Windows managed temp locations deny rename despite allowing file
        # creation. Outputs are small audit artifacts; copy only in this fallback.
        target.write_bytes(temp.read_bytes())
        temp.unlink(missing_ok=True)


def stable_rank(seed: int, *parts: str) -> str:
    payload = "\0".join((str(seed), *map(str, parts))).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _read_csv(path: str | Path) -> Iterator[dict[str, str]]:
    with Path(path).open("r", encoding="utf-8-sig", newline="") as stream:
        yield from csv.DictReader(stream)


def _json(path: str | Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def normalize_saved_calibration(record: Mapping[str, Any]) -> dict[str, Any]:
    """The old calibrator treats an omitted method as temperature scaling."""
    normalized = dict(record)
    normalized.setdefault("method", "temperature")
    return normalized


def calibrate_real_probabilities(
    real_probabilities: Sequence[float],
    calibration: Mapping[str, Any],
    apply_calibration: Any,
):
    """Apply a saved real-probability calibrator, including explicit identity.

    Model logits are converted with sigmoid (and, for focal, averaged in
    probability space across horizontal flips) before reaching this function.
    The previous project's calibrator validates and handles temperature,
    Platt, isotonic, and median methods. Its explicit ``none`` case is absent,
    so identity is handled here after validating the probability contract.
    Unknown methods are still delegated and rejected by that implementation.
    """
    import numpy as np

    probabilities = np.asarray(real_probabilities, dtype=float)
    method = calibration.get("method", "temperature")
    if method == "none":
        if (
            probabilities.ndim != 1
            or not probabilities.size
            or not np.isfinite(probabilities).all()
            or ((probabilities < 0) | (probabilities > 1)).any()
        ):
            raise ValueError("Identity calibration requires finite one-dimensional probabilities")
        return probabilities.copy()
    return apply_calibration(probabilities, calibration)


def real_to_fake_probabilities(real_probabilities: Sequence[float]) -> list[float]:
    """Convert real-class probabilities to fake scores (labels: fake=0, real=1)."""
    return [1.0 - float(value) for value in real_probabilities]


def _tail_video_id(value: str) -> str:
    value = (value or "").replace("\\", "/").rsplit("/", 1)[-1].rsplit(":", 1)[-1]
    return Path(value).stem.casefold()


@dataclass
class PriorTraining:
    celeb_video_stems: set[str]
    ff_source_ids: set[str]
    crop_hashes: set[str]
    run_manifest_hashes: dict[str, str]


def load_prior_training(manifest_paths: Sequence[str | Path]) -> PriorTraining:
    """Read old Xception input manifests and identify only explicitly train rows."""
    celeb: set[str] = set()
    ff: set[str] = set()
    crop_hashes: set[str] = set()
    manifest_hashes: dict[str, str] = {}
    for path in manifest_paths:
        data = _json(path)
        if not isinstance(data, dict) or not isinstance(data.get("rows"), list):
            raise ValueError(f"Invalid previous-project input manifest: {path}")
        run_hash = str(data.get("hash", ""))
        manifest_hashes[str(Path(path).resolve())] = run_hash
        for row in data["rows"]:
            if row.get("split") != "train":
                continue
            if row.get("dataset") == "celebdf":
                stem = _tail_video_id(str(row.get("video_id") or row.get("raw_path") or ""))
                if stem:
                    celeb.add(stem)
            elif row.get("dataset") == "ffpp":
                cluster = str(row.get("cluster") or "")
                ff.update(token.casefold().zfill(3) for token in re.findall(r"ffpp:(\d+)", cluster, re.I))
            crop_hash = str(row.get("crop_sha256") or row.get("sha256") or "").casefold()
            if re.fullmatch(r"[0-9a-f]{64}", crop_hash):
                crop_hashes.add(crop_hash)
    nonempty = {value for value in manifest_hashes.values() if value}
    if len(nonempty) > 1:
        raise ValueError("Previous model manifests describe different training inputs; review overlap evidence")
    return PriorTraining(celeb, ff, crop_hashes, manifest_hashes)


def prior_source_overlap(row: Mapping[str, str], prior: PriorTraining) -> str:
    """Return a reason only for exact source-video/source-id matches in prior train."""
    group = str(row.get("group_id") or "").casefold()
    if ":cdf:" in group:
        source_clip = group.split(":cdf:", 1)[1].split(":", 1)[0]
        if source_clip in prior.celeb_video_stems:
            return "exact_celebdf_training_video_id"
    if ":ff:" in group:
        source_id = group.split(":ff:", 1)[1].split(":", 1)[0].zfill(3)
        if source_id in prior.ff_source_ids:
            return "exact_ffpp_source_id_in_training_groups"
    return ""


def load_component_map(path: str | Path | None) -> dict[str, str]:
    if path is None or not Path(path).is_file():
        return {}
    result: dict[str, str] = {}
    for row in _read_csv(path):
        group = row.get("group_id", "")
        component = row.get("component_id", "")
        if group and component:
            result[group] = component
    return result


def load_completed_hash_audit(path: str | Path | None) -> tuple[dict[str, set[str]], int]:
    """Load exact source digests and their audited project partitions."""
    if path is None or not Path(path).is_file():
        return {}, 0
    partitions: dict[str, set[str]] = defaultdict(set)
    count = 0
    for row in _read_csv(path):
        digest = str(row.get("sha256", "")).casefold()
        if row.get("status") != "ok" or not re.fullmatch(r"[0-9a-f]{64}", digest):
            continue
        partitions[digest].add(str(row.get("partition") or "unknown"))
        count += 1
    return dict(partitions), count


def _verified_image_row(row: Mapping[str, str]) -> bool:
    return (
        row.get("dataset") == "df40"
        and row.get("method", "").casefold() in {name.casefold() for name in TARGET_METHODS}
        and row.get("label") in {"0", "1"}
        and row.get("experiment_eligibility") == "eligible_labeled_record"
        and row.get("media_type") in {"image", "gif"}
        and Path(row.get("absolute_path", "")).suffix.casefold() in {".png", ".jpg", ".jpeg", ".gif"}
        and bool(row.get("group_id"))
    )


def collect_group_candidates(
    inventory_path: str | Path,
    component_path: str | Path | None,
    prior: PriorTraining,
    *,
    seed: int,
    candidates_per_group: int = 4,
) -> tuple[dict[tuple[str, str, str], list[dict[str, str]]], Counter[str]]:
    """Stream inventory; retain bounded, deterministic representatives per group."""
    if candidates_per_group < 1:
        raise ValueError("candidates_per_group must be positive")
    components = load_component_map(component_path)
    grouped: dict[tuple[str, str, str], list[dict[str, str]]] = defaultdict(list)
    exclusions: Counter[str] = Counter()
    method_names = {name.casefold(): name for name in TARGET_METHODS}
    official_test_components: set[str] = set()
    for source in _read_csv(inventory_path):
        if source.get("dataset") == "celebdf" and "official_test" in source.get("original_partition", "").casefold():
            component = components.get(source.get("group_id", ""), "")
            if component:
                official_test_components.add(component)
    for source in _read_csv(inventory_path):
        method = method_names.get(source.get("method", "").casefold())
        if source.get("dataset") != "df40" or method is None:
            continue
        if source.get("label") not in {"0", "1"} or source.get("experiment_eligibility") != "eligible_labeled_record":
            exclusions["unknown_or_unverified_label"] += 1
            continue
        if source.get("media_type") not in {"image", "gif"} or Path(source.get("absolute_path", "")).suffix.casefold() not in {".png", ".jpg", ".jpeg", ".gif"}:
            exclusions["unsupported_media"] += 1
            continue
        if not source.get("group_id"):
            exclusions["missing_source_group"] += 1
            continue
        component = components.get(source["group_id"], "")
        if component and component in official_test_components:
            exclusions["protected_celebdf_official_test_component"] += 1
            continue
        overlap = prior_source_overlap(source, prior)
        if overlap:
            exclusions[overlap] += 1
            continue
        item = dict(source)
        item["method"] = method
        item["source_component_id"] = component
        item["prior_source_overlap"] = ""
        item["prior_training_exact_hash_overlap"] = ""
        key = (method, source["label"], source["group_id"])
        bucket = grouped[key]
        rank = stable_rank(seed, method, source["label"], source["group_id"], source.get("relative_path", ""))
        item["_rank"] = rank
        bucket.append(item)
        bucket.sort(key=lambda candidate: candidate["_rank"])
        del bucket[candidates_per_group:]
    return grouped, exclusions


def select_evaluation_records(
    inventory_path: str | Path,
    component_path: str | Path | None,
    prior_manifest_paths: Sequence[str | Path],
    *,
    seed: int = 20261104,
    fake_groups_per_method: int = 10,
    real_groups_per_method: int = 52,
    hash_audit_path: str | Path | None = "artifacts/combined/source-hashes-v2.csv",
) -> tuple[list[dict[str, str]], dict[str, Any]]:
    """Reserve one verified still per source group for a future DF40-only test."""
    if fake_groups_per_method < 1 or real_groups_per_method < 0:
        raise ValueError("Fake quota must be positive; real quota may be zero to disable real selection")
    prior = load_prior_training(prior_manifest_paths)
    audited_hash_partitions, audited_hash_rows = load_completed_hash_audit(hash_audit_path)
    candidates, excluded = collect_group_candidates(
        inventory_path, component_path, prior, seed=seed
    )
    # A missing focal training manifest is reported explicitly; the user's checkpoint
    # has no trustworthy run manifest in the prior project.
    selected: list[dict[str, str]] = []
    hashes_seen: dict[str, dict[str, str]] = {}
    method_counts: dict[str, dict[str, int]] = {m: {"fake": 0, "real": 0, "fake_groups": 0, "real_groups": 0} for m in TARGET_METHODS}
    conflicts: list[dict[str, str]] = []
    for method in TARGET_METHODS:
        for label, quota in (("0", fake_groups_per_method), ("1", real_groups_per_method if method.casefold() in {name.casefold() for name in REAL_METHODS} else 0)):
            if quota == 0:
                continue
            groups = [key for key in candidates if key[0] == method and key[1] == label]
            groups.sort(key=lambda key: stable_rank(seed, "group", *key))
            accepted_groups = 0
            for key in groups:
                if accepted_groups >= quota:
                    break
                group_candidates = sorted(candidates[key], key=lambda x: x["_rank"])
                chosen = None
                for candidate in group_candidates:
                    path = Path(candidate["absolute_path"])
                    if not path.is_file():
                        excluded["selected_candidate_missing_on_disk"] += 1
                        continue
                    if path.stat().st_size <= 0:
                        excluded["zero_byte_media"] += 1
                        continue
                    try:
                        from PIL import Image
                        with Image.open(path) as image:
                            image.verify()
                    except Exception:
                        excluded["selected_candidate_unreadable"] += 1
                        continue
                    digest = sha256_file(path)
                    if digest in prior.crop_hashes:
                        excluded["exact_hash_matches_previous_training_crop"] += 1
                        continue
                    if digest in audited_hash_partitions:
                        excluded["exact_hash_matches_completed_source_audit"] += 1
                        continue
                    previous = hashes_seen.get(digest)
                    if previous:
                        if previous["label"] != label:
                            conflicts.append({"sha256": digest, "first_path": previous["relative_path"], "conflicting_path": candidate["relative_path"]})
                            excluded["conflicting_label_duplicate"] += 1
                            selected.remove(previous)
                            hashes_seen.pop(digest, None)
                            previous_method = previous["method"]
                            previous_label_key = "fake" if previous["label"] == "0" else "real"
                            previous_group_key = "fake_groups" if previous["label"] == "0" else "real_groups"
                            method_counts[previous_method][previous_label_key] -= 1
                            method_counts[previous_method][previous_group_key] -= 1
                        else:
                            excluded["exact_duplicate_within_evaluation"] += 1
                        continue
                    candidate["source_sha256"] = digest
                    chosen = candidate
                    break
                if chosen is None:
                    continue
                clean = {k: v for k, v in chosen.items() if not k.startswith("_")}
                clean.update({
                    "evaluation_id": f"df40-baseline-v1-{len(selected)+1:05d}",
                    "protocol": "df40-baseline-reserved-group-holdout-v1",
                    "reservation_partition": "baseline_eval_test",
                    "selection_seed": str(seed),
                    "selection_reason": "verified label; deterministic source-group reservation; exact prior training source-ID and crop-byte exclusions applied",
                })
                selected.append(clean)
                hashes_seen[chosen["source_sha256"]] = clean
                method_counts[method]["fake" if label == "0" else "real"] += 1
                method_counts[method]["fake_groups" if label == "0" else "real_groups"] += 1
                accepted_groups += 1
    labels = Counter(row["label"] for row in selected)
    report = {
        "status": "smoke_ready_best_effort_not_independence_certified",
        "protocol": "df40-baseline-reserved-group-holdout-v1",
        "seed": seed,
        "requested_per_method": {"fake_groups": fake_groups_per_method, "real_groups_on_real_methods": real_groups_per_method},
        "selected_records": len(selected),
        "selected_labels": {"fake_0": labels["0"], "real_1": labels["1"]},
        "target_methods": list(TARGET_METHODS),
        "real_reference_methods": list(REAL_METHODS),
        "method_counts": method_counts,
        "selected_group_status": dict(Counter(row.get("group_status", "") for row in selected)),
        "selected_relationship_status": dict(Counter(row.get("relationship_status", "") for row in selected)),
        "excluded_counts": dict(excluded),
        "conflicting_selected_hashes": conflicts,
        "distinct_selected_sha256": len(hashes_seen),
        "distinct_selected_source_groups": len({row["group_id"] for row in selected}),
        "distinct_selected_source_components": len({row["source_component_id"] for row in selected if row.get("source_component_id")}),
        "completed_source_hash_audit": {"path": str(Path(hash_audit_path).resolve()) if hash_audit_path and Path(hash_audit_path).is_file() else None, "rows": audited_hash_rows, "distinct_sha256": len(audited_hash_partitions)},
        "prior_training": {
            "training_manifest_hashes": prior.run_manifest_hashes,
            "known_celebdf_training_source_videos": len(prior.celeb_video_stems),
            "known_ffpp_training_source_ids": len(prior.ff_source_ids),
            "known_training_crop_hashes": len(prior.crop_hashes),
            "focal_checkpoint_training_manifest": "unavailable; source overlap cannot be checked against its training set",
        },
        "limitations": [
            "DF40 source groups are reserved by this evaluation selection; any later training release must exclude these group IDs, source components where applicable, and source SHA-256 values.",
            "The clean and mixed models share one known prior training manifest. The focal checkpoint has no recoverable input manifest, so its training-source overlap cannot be independently verified.",
            "Suspected name/domain cross-dataset links are retained as uncertainty evidence; exact filename/ID matches to known prior train sources are excluded, but this does not establish person-identity independence.",
            "The selection is a custom DF40 evaluation subset and is not an untouched official DF40 benchmark.",
            "DF40 original val/test strings are provenance only; project reservation is separate.",
        ],
    }
    report["input_sha256"] = {
        "phase1_inventory_csv": sha256_file(inventory_path),
        "source_components_csv": sha256_file(component_path) if component_path and Path(component_path).is_file() else None,
        "completed_source_hash_audit_csv": sha256_file(hash_audit_path) if hash_audit_path and Path(hash_audit_path).is_file() else None,
        "prior_training_manifests": {str(Path(path).resolve()): sha256_file(path) for path in prior_manifest_paths},
    }
    return selected, report


def write_selection(records: Sequence[Mapping[str, Any]], report: Mapping[str, Any], out_dir: str | Path) -> tuple[Path, Path]:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    manifest_path, report_path = out / "selection.csv", out / "selection-report.json"
    temp = manifest_path.with_suffix(".csv.tmp")
    with temp.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=SELECTION_FIELDS, extrasaction="ignore")
        writer.writeheader()
        for row in records:
            writer.writerow(row)
    _commit_generated_temp(temp, manifest_path)
    payload = dict(report)
    payload["selection_csv_sha256"] = sha256_file(manifest_path)
    report_temp = report_path.with_suffix(".json.tmp")
    report_temp.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    _commit_generated_temp(report_temp, report_path)
    return manifest_path, report_path


def _load_selected(path: str | Path, *, limit: int | None = None) -> list[dict[str, str]]:
    rows = list(_read_csv(path))
    if not rows:
        raise ValueError(f"Evaluation selection is empty: {path}")
    ids = [row["evaluation_id"] for row in rows]
    if len(ids) != len(set(ids)):
        raise ValueError("Selection contains duplicate evaluation IDs")
    if any(row.get("label") not in {"0", "1"} or row.get("reservation_partition") != "baseline_eval_test" for row in rows):
        raise ValueError("Selection contains unresolved labels or invalid reservation partition")
    if len({row["source_sha256"] for row in rows}) != len(rows):
        raise ValueError("Selection contains exact-byte duplicate images")
    if limit is not None:
        rows = rows[:limit]
    return rows


def calculate_metrics(labels: Sequence[int], fake_scores: Sequence[float], threshold: float = 0.5) -> dict[str, Any]:
    """Metrics with fake as the positive class (dataset labels remain fake=0, real=1)."""
    if not math.isfinite(threshold) or not 0 <= threshold <= 1 or len(labels) != len(fake_scores) or not labels:
        raise ValueError("Invalid metric inputs")
    if any(label not in (0, 1) for label in labels) or any(not math.isfinite(s) or not 0 <= s <= 1 for s in fake_scores):
        raise ValueError("Labels must be 0/1 and fake scores finite probabilities")
    y_fake = [int(label == 0) for label in labels]
    predictions_fake = [int(score >= threshold) for score in fake_scores]
    tp = sum(y and p for y, p in zip(y_fake, predictions_fake))
    tn = sum((not y) and (not p) for y, p in zip(y_fake, predictions_fake))
    fp = sum((not y) and p for y, p in zip(y_fake, predictions_fake))
    fn = sum(y and (not p) for y, p in zip(y_fake, predictions_fake))
    fake_recall = tp / (tp + fn) if tp + fn else 0.0
    real_recall = tn / (tn + fp) if tn + fp else 0.0
    precision = tp / (tp + fp) if tp + fp else 0.0
    try:
        from sklearn.metrics import roc_auc_score
        auc = float(roc_auc_score(y_fake, fake_scores)) if len(set(y_fake)) == 2 else None
    except ImportError:
        auc = None
    return {
        "threshold_fake_score": threshold,
        "samples": len(labels),
        "accuracy": (tp + tn) / len(labels),
        "balanced_accuracy": (fake_recall + real_recall) / 2,
        "precision_fake": precision,
        "recall_fake": fake_recall,
        "f1_fake": 2 * precision * fake_recall / (precision + fake_recall) if precision + fake_recall else 0.0,
        "roc_auc_fake": auc,
        "false_positive_rate": fp / (fp + tn) if fp + tn else 0.0,
        "false_negative_rate": fn / (fn + tp) if fn + tp else 0.0,
        "confusion_matrix_true_fake_real_pred_fake_real": [[tp, fn], [fp, tn]],
    }


def _model_specs(previous_root: Path) -> list[dict[str, Any]]:
    return [
        {"name": "clean", "checkpoint": previous_root / "runs/best_xception/xcep-clean-s42/best.safetensors",
         "config": previous_root / "runs/best_xception/xcep-clean-s42/config.json",
         "run_dir": previous_root / "runs/best_xception/xcep-clean-s42", "format": "safetensors", "legacy_hflip_tta": False},
        {"name": "mixed_robust", "checkpoint": previous_root / "runs/archive/xcep-mixed-robust-s42/best.safetensors",
         "config": previous_root / "runs/archive/xcep-mixed-robust-s42/config.json",
         "run_dir": previous_root / "runs/archive/xcep-mixed-robust-s42", "format": "safetensors", "legacy_hflip_tta": False},
        {"name": "focal", "checkpoint": previous_root / ".old-project/models/checkpoints/model_new_focal_93.pth",
         "config": previous_root / ".old-project/before-refactor/configs/xception-clean.json",
         "run_dir": None, "format": "pytorch_state_dict", "legacy_hflip_tta": True},
    ]


def _load_legacy_runtime(previous_root: Path, device: str) -> tuple[Any, Any, Any, Any]:
    if str(previous_root) not in sys.path:
        sys.path.insert(0, str(previous_root))
    # The previous code is imported only for its architecture, preprocessing and
    # calibration contracts. It is never imported or written into.
    from core.data.face_detector import FaceDetector
    from core.data.preprocessing import model_tensor
    from core.evaluation.calibration import apply_calibration
    from core.models.model_loader import architecture
    import torch
    return torch, FaceDetector, model_tensor, (apply_calibration, architecture)


def _load_checkpoint_model(spec: Mapping[str, Any], previous_root: Path, runtime: tuple[Any, Any, Any, Any], device: str) -> tuple[Any, dict[str, Any], dict[str, Any]]:
    torch, _, _, helpers = runtime
    apply_calibration, architecture = helpers
    cfg = _json(spec["config"])
    model_cfg = cfg["model"]
    model = architecture(model_cfg, enhancements=False)
    checkpoint = Path(spec["checkpoint"])
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")
    digest = sha256_file(checkpoint)
    state = None
    if spec["format"] == "safetensors":
        from safetensors.torch import load_file
        state = load_file(str(checkpoint), device="cpu")
        metadata = _json(Path(spec["run_dir"]) / "best_metadata.json")
        if metadata.get("weights_sha256") != digest:
            raise ValueError(f"Checkpoint SHA-256 does not match saved metadata: {checkpoint}")
    else:
        state = torch.load(checkpoint, map_location="cpu", weights_only=True)
        if isinstance(state, dict) and "state_dict" in state:
            state = state["state_dict"]
        metadata = {"weights_sha256": digest, "calibration": "unavailable"}
    if not isinstance(state, Mapping):
        raise ValueError(f"Checkpoint is not a state dictionary: {checkpoint}")
    model.load_state_dict(state, strict=True)
    model.to(device).eval()
    calibration = {"method": "none", "temperature": 1.0, "threshold": 0.5}
    if spec.get("run_dir"):
        run_dir = Path(spec["run_dir"])
        ref = _json(run_dir / "calibration.json")
        cal_path = run_dir / str(ref.get("version", ""))
        if not cal_path.is_file():
            raise FileNotFoundError(f"Saved calibration artifact is missing: {cal_path}")
        calibration = normalize_saved_calibration(_json(cal_path))
        if calibration.get("weights_sha256") != digest:
            raise ValueError(f"Calibration weights do not match checkpoint: {checkpoint}")
    metadata.update({"checkpoint_sha256": digest, "strict_state_load": True})
    return model, cfg, {"metadata": metadata, "calibration": calibration, "model_cfg": model_cfg}


def _read_rgb(path: str | Path):
    from PIL import Image, ImageOps
    with Image.open(path) as image:
        if getattr(image, "is_animated", False) and getattr(image, "n_frames", 1) != 1:
            raise ValueError("Animated GIF is not accepted in this DF40 baseline evaluation")
        image.seek(0)
        image = ImageOps.exif_transpose(image).convert("RGB")
        image.load()
        return image


def legacy_face_indices(boxes: Any, probabilities: Any, *, minimum_width: float = 50.0) -> list[int]:
    """Match the old extraction wrapper: confidence >= 0 and width >= 50 px.

    The 0 confidence floor is intentional: the historical pipeline searched for
    a best crop below 0.95 when no gold crop existed, then stored it as "best".
    """
    if boxes is None or probabilities is None:
        return []
    accepted = []
    for index, (box, probability) in enumerate(zip(boxes, probabilities)):
        try:
            values = [float(value) for value in box]
            confidence = float(probability)
        except (TypeError, ValueError):
            continue
        if len(values) == 4 and all(math.isfinite(value) for value in values) and math.isfinite(confidence):
            if values[2] - values[0] >= minimum_width and confidence >= 0.0:
                accepted.append(index)
    return accepted


def _make_preprocessor(spec: Mapping[str, Any], cfg: Mapping[str, Any], runtime: tuple[Any, Any, Any, Any], device: str):
    torch, FaceDetector, _, _ = runtime
    if cfg.get("preprocessing"):
        return FaceDetector(cfg["preprocessing"], device=device)
    from facenet_pytorch import MTCNN
    return MTCNN(keep_all=True, device=device, post_process=False, min_face_size=50)


def _inference_tensor(row: Mapping[str, str], spec: Mapping[str, Any], cfg: Mapping[str, Any], runtime: tuple[Any, Any, Any, Any], device: str, preprocessor: Any):
    """Apply the old MTCNN box crop and Xception normalization; no source writes."""
    torch, FaceDetector, model_tensor, helpers = runtime
    image = _read_rgb(row["absolute_path"])
    pre_cfg = cfg.get("preprocessing")
    if pre_cfg:
        faces = preprocessor.detect(image)
        accepted = [face for face in faces if face.status == "accepted" and face.crop is not None]
        if not accepted:
            raise ValueError(f"No accepted MTCNN face in {row['relative_path']}")
        # The legacy old-project serving code uses a face detector; choose the
        # largest accepted face deterministically when it returns multiple faces.
        face = max(accepted, key=lambda item: ((item.box[2]-item.box[0])*(item.box[3]-item.box[1]), tuple(-v for v in item.box)))
        master = face.crop.image
        box, confidence = face.box, face.probability
        tensor = model_tensor(master, cfg["model"])
    else:
        # Legacy focal config predates the refactored crop contract. Its audited
        # detector uses MTCNN >=0.95, a 1.3x square crop, black boundary padding,
        # OpenCV linear resize, then ToTensor/Normalize(0.5,0.5).
        import numpy as np
        import cv2
        boxes, probs, _ = preprocessor.detect(image, landmarks=True)
        accepted = legacy_face_indices(boxes, probs, minimum_width=50.0)
        if not accepted:
            raise ValueError(f"No accepted legacy MTCNN face in {row['relative_path']}")
        i = max(accepted, key=lambda j: ((boxes[j][2]-boxes[j][0])*(boxes[j][3]-boxes[j][1]), tuple(-float(v) for v in boxes[j])))
        box = np.asarray(boxes[i], dtype=float)
        arr = np.asarray(image.convert("RGB"))
        side = max(box[2]-box[0], box[3]-box[1]) * 1.3
        cx, cy = (box[0]+box[2])/2, (box[1]+box[3])/2
        x0, y0 = int(cx-side/2), int(cy-side/2)
        x1, y1 = int(cx+side/2), int(cy+side/2)
        tile = np.zeros((y1-y0, x1-x0, 3), dtype=np.uint8)
        sx0, sy0, sx1, sy1 = max(0,x0), max(0,y0), min(image.width,x1), min(image.height,y1)
        if sx1 <= sx0 or sy1 <= sy0:
            raise ValueError("Legacy crop falls outside source image")
        tile[sy0-y0:sy1-y0, sx0-x0:sx1-x0] = arr[sy0:sy1, sx0:sx1]
        crop = cv2.resize(tile, (299,299), interpolation=cv2.INTER_LINEAR)
        # Historical extraction wrote JPEG quality 95 and ImageFolder later
        # decoded it before the 299px tensor transform. Reproduce that codec step.
        from io import BytesIO
        from PIL import Image
        buffer = BytesIO()
        Image.fromarray(crop).save(buffer, format="JPEG", quality=95)
        buffer.seek(0)
        with Image.open(buffer) as saved_crop:
            crop_rgb = np.asarray(saved_crop.convert("RGB"))
        tensor = torch.from_numpy(crop_rgb.transpose(2,0,1).copy()).float().div_(255)
        tensor = (tensor - 0.5) / 0.5
        confidence = float(probs[i])
        box = box.tolist()
    return tensor, {"box": [float(v) for v in box], "confidence": float(confidence), "source_size": list(image.size)}


def run_smoke(selection_path: str | Path, previous_root: str | Path, output_path: str | Path, *, device: str = "auto", batch_size: int = 1) -> dict[str, Any]:
    if batch_size != 1:
        raise ValueError("The bounded checkpoint smoke test requires batch_size=1")
    rows = _load_selected(selection_path)
    # Pick one fixed fake and one fixed real record. Every checkpoint must score
    # these exact image identities; model-specific substitution would invalidate
    # the smoke comparison.
    picked: list[dict[str, str]] = []
    for label in ("0", "1"):
        candidate = next((r for r in rows if r["label"] == label and r not in picked), None)
        if candidate:
            picked.append(candidate)
    if len(picked) < 2:
        raise ValueError("Smoke selection needs at least one fake and one real image")
    root = Path(previous_root).resolve()
    torch, FaceDetector, model_tensor, helpers = _load_legacy_runtime(root, device)
    device_name = "cuda" if device == "auto" and torch.cuda.is_available() else ("cpu" if device == "auto" else device)
    if device_name.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable in the selected environment")
    results: list[dict[str, Any]] = []
    for spec in _model_specs(root):
        model, cfg, info = _load_checkpoint_model(spec, root, (torch, FaceDetector, model_tensor, helpers), device_name)
        preprocessor = _make_preprocessor(spec, cfg, (torch, FaceDetector, model_tensor, helpers), device_name)
        outputs = []
        for row in picked:
            try:
                tensor, face = _inference_tensor(row, spec, cfg, (torch, FaceDetector, model_tensor, helpers), device_name, preprocessor)
            except Exception as exc:
                raise RuntimeError(f"Smoke image preprocessing failed for {spec['name']} on {row['evaluation_id']}: {type(exc).__name__}: {exc}") from exc
            x = tensor.unsqueeze(0).to(device_name)
            with torch.inference_mode():
                logit = model(x).float().reshape(-1)
                if spec.get("legacy_hflip_tta"):
                    flipped = model(torch.flip(x, dims=(-1,))).float().reshape(-1)
                    # Original focal test script averages sigmoid probabilities.
                    real_score = average_binary_probabilities(
                        float(torch.sigmoid(logit).mean().cpu()),
                        float(torch.sigmoid(flipped).mean().cpu()),
                    )
                else:
                    real_score = float(torch.sigmoid(logit).mean().cpu())
            outputs.append({"evaluation_id": row["evaluation_id"], "label": int(row["label"]), "real_probability": real_score, "fake_probability": 1-real_score, "face": face})
        preprocessing = cfg.get("preprocessing") or {
            "version": "legacy_mtcNN_square_299_linear_v1",
            "margin_factor": 1.3,
            "confidence_rejection_floor": 0.0,
            "historical_gold_target": 0.95,
            "minimum_face_width": 50,
            "interpolation": "OpenCV INTER_LINEAR",
            "crop_encoding": "JPEG quality 95 before ImageFolder decode",
        }
        results.append({"model": spec["name"], "checkpoint_sha256": info["metadata"]["checkpoint_sha256"], "strict_load": True, "device": device_name, "class_map": {"fake": 0, "real": 1}, "preprocessing": preprocessing, "horizontal_flip_tta": bool(spec.get("legacy_hflip_tta")), "calibration": {"method":info["calibration"].get("method","none"),"threshold":info["calibration"].get("threshold",0.5)}, "preprocessing_failures": [], "smoke_predictions": outputs})
        del model
        if device_name.startswith("cuda"):
            torch.cuda.empty_cache()
    report = {
        "status": "smoke_passed_not_full_evaluation",
        "selection_sha256": sha256_file(selection_path),
        "sample_count": len(picked),
        "same_images_for_all_models": True,
        "cuda_available": bool(torch.cuda.is_available()),
        "cuda_device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "models": results,
        "note": "This smoke output is not a performance comparison and must not be cited as evaluation results.",
    }
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(out.suffix + ".tmp")
    tmp.write_text(json.dumps(report, indent=2), encoding="utf-8")
    _commit_generated_temp(tmp, out)
    return report


def average_binary_probabilities(*views: float) -> float:
    if not views or any(not math.isfinite(value) or not 0 <= value <= 1 for value in views):
        raise ValueError("Probability views must be finite values in [0,1]")
    return sum(views) / len(views)


def evaluate(selection_path: str | Path, previous_root: str | Path, output_dir: str | Path, *, device: str = "auto", batch_size: int = 4) -> dict[str, Any]:
    """Run the full comparison. This command is deliberately separate from smoke."""
    if batch_size < 1 or batch_size > 4:
        raise ValueError("Use inference batches 1-4 for the GTX 1650 4 GB target")
    rows = _load_selected(selection_path)
    labels = [int(row["label"]) for row in rows]
    if len(set(labels)) != 2:
        raise ValueError("Full evaluation selection must include both verified classes")
    root = Path(previous_root).resolve()
    torch, FaceDetector, model_tensor, helpers = _load_legacy_runtime(root, device)
    device_name = "cuda" if device == "auto" and torch.cuda.is_available() else ("cpu" if device == "auto" else device)
    if device_name.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    output: dict[str, Any] = {"status": "completed", "selection_sha256": sha256_file(selection_path), "images": len(rows), "label_map": {"0": "fake", "1": "real"}, "device": device_name, "batch_size": batch_size, "models": []}
    for spec in _model_specs(root):
        model, cfg, info = _load_checkpoint_model(spec, root, (torch, FaceDetector, model_tensor, helpers), device_name)
        preprocessor = _make_preprocessor(spec, cfg, (torch, FaceDetector, model_tensor, helpers), device_name)
        batch_tensors: list[Any] = []
        accepted_rows: list[dict[str, str]] = []
        raw_real: list[float] = []
        failures: list[dict[str, str]] = []

        def score_pending() -> None:
            if not batch_tensors:
                return
            batch = torch.stack(batch_tensors).to(device_name)
            with torch.inference_mode():
                logits = model(batch).float().reshape(-1)
                if spec.get("legacy_hflip_tta"):
                    logits_flip = model(torch.flip(batch, dims=(-1,))).float().reshape(-1)
                    probabilities = (torch.sigmoid(logits) + torch.sigmoid(logits_flip)) / 2
                else:
                    probabilities = torch.sigmoid(logits)
            raw_real.extend(probabilities.cpu().tolist())
            batch_tensors.clear()

        for row in rows:
            try:
                tensor, face = _inference_tensor(row, spec, cfg, (torch, FaceDetector, model_tensor, helpers), device_name, preprocessor)
                accepted_rows.append(row)
                batch_tensors.append(tensor)
                if len(batch_tensors) >= batch_size:
                    score_pending()
            except Exception as exc:
                failures.append({"evaluation_id": row["evaluation_id"], "reason": f"{type(exc).__name__}: {exc}"})
        model.eval()
        score_pending()
        if len(accepted_rows) < 2 or {int(row["label"]) for row in accepted_rows} != {0,1}:
            raise RuntimeError(f"{spec['name']} preprocessing left fewer than two classes; refusing partial comparison")
        calibration = info["calibration"]
        apply_calibration, _ = helpers
        # Saved refactored calibrators operate on real probabilities. Focal has no
        # saved calibration and remains raw at its documented 0.5 threshold.
        real_scores = calibrate_real_probabilities(raw_real, calibration, apply_calibration)
        native_threshold = float(calibration.get("threshold", 0.5))
        fake_scores = real_to_fake_probabilities(real_scores)
        accepted_labels = [int(row["label"]) for row in accepted_rows]
        universal_fake = real_to_fake_probabilities(raw_real)
        fixed_metrics = calculate_metrics(accepted_labels, universal_fake, 0.5)
        native_metrics = calculate_metrics(accepted_labels, fake_scores, native_threshold)
        per_method: dict[str, Any] = {}
        for method in TARGET_METHODS:
            indices = [i for i,row in enumerate(accepted_rows) if row["method"].casefold()==method.casefold() and int(row["label"])==0]
            if indices:
                per_method[method] = {"fake_count":len(indices), "fake_recall":sum(fake_scores[i]>=native_threshold for i in indices)/len(indices)}
        per_dataset = {"df40": {"fake": 0, "real": 0}}
        for row in accepted_rows:
            per_dataset["df40"]["fake" if int(row["label"]) == 0 else "real"] += 1
        predictions = [{"evaluation_id": row["evaluation_id"], "label": int(row["label"]), "real_probability_raw": raw_real[i], "fake_probability_scored": fake_scores[i]} for i,row in enumerate(accepted_rows)]
        item = {
            "name": spec["name"], "checkpoint_sha256": info["metadata"]["checkpoint_sha256"], "strict_load": True,
            "preprocessing": {"contract": "model-specific original MTCNN/crop/resize/Normalize(0.5) behavior", "legacy_horizontal_flip_tta": bool(spec.get("legacy_hflip_tta")), "accepted_faces":len(accepted_rows), "failed_faces":len(failures)},
            "fixed_raw_threshold_0_5": fixed_metrics,
            "saved_calibration_operating_point": {"threshold":native_threshold,"calibration_method":calibration.get("method","none"),"metrics":native_metrics},
            "per_method_fake_recall":per_method,
            "per_dataset_class_counts":per_dataset,
            "failures":failures,
            "predictions": predictions,
        }
        output["models"].append(item)
        del model, batch_tensors
        if device_name.startswith("cuda"):
            torch.cuda.empty_cache()
    # Compare only rows scored by all checkpoints; stricter model-specific face
    # detection failures invalidate the aggregate comparison.
    accepted_sets = [{p["evaluation_id"] for p in model["predictions"]} for model in output["models"]]
    common_ids = set.intersection(*accepted_sets) if accepted_sets else set()
    output["common_scored_images"] = len(common_ids)
    output["all_models_scored_identical_selection"] = (
        all(ids == accepted_sets[0] for ids in accepted_sets) and len(common_ids) == len(rows)
    ) if accepted_sets else False
    output["limitation"] = "Metrics are comparable only when all_models_scored_identical_selection is true. The comparison refuses to silently treat model-specific face-detection failures as scored examples."
    if not output["all_models_scored_identical_selection"]:
        output["status"] = "failed_common_image_support"
        output["blocking_reason"] = "At least one model failed preprocessing on a different set of selected images; common-support metrics must be recomputed before ranking."
    out_dir = Path(output_dir); out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / "comparison.json"; temp = out.with_suffix(".json.tmp")
    temp.write_text(json.dumps(output, indent=2), encoding="utf-8"); _commit_generated_temp(temp, out)
    return output


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    select = sub.add_parser("select", help="create a read-only, reserved DF40 evaluation manifest")
    select.add_argument("--inventory", type=Path, default=Path("artifacts/inventory/df40-phase1-final/manifest.csv"))
    select.add_argument("--components", type=Path, default=Path("artifacts/combined/relationships/source-components.csv"))
    select.add_argument("--hash-audit", type=Path, default=Path("artifacts/combined/source-hashes-v2.csv"))
    select.add_argument("--prior-manifest", type=Path, action="append", required=True)
    select.add_argument("--out-dir", type=Path, default=Path("artifacts/evaluations/df40-baseline-v1"))
    select.add_argument("--seed", type=int, default=20261104)
    select.add_argument("--fake-groups-per-method", type=int, default=10)
    select.add_argument("--real-groups-per-method", type=int, default=52)
    run = sub.add_parser("smoke", help="strict-load three checkpoints and score only two reserved images")
    run.add_argument("--selection", type=Path, default=Path("artifacts/evaluations/df40-baseline-v1/selection.csv"))
    run.add_argument("--previous-root", type=Path, default=Path(r"D:\University\explainable-deepfake-detection-system"))
    run.add_argument("--output", type=Path, default=Path("artifacts/evaluations/df40-baseline-v1/smoke.json"))
    run.add_argument("--device", default="auto")
    full = sub.add_parser("evaluate", help="run the full comparison; requires user approval before execution")
    full.add_argument("--selection", type=Path, default=Path("artifacts/evaluations/df40-baseline-v1/selection.csv"))
    full.add_argument("--previous-root", type=Path, default=Path(r"D:\University\explainable-deepfake-detection-system"))
    full.add_argument("--out-dir", type=Path, default=Path("artifacts/evaluations/df40-baseline-v1"))
    full.add_argument("--device", default="auto")
    full.add_argument("--batch-size", type=int, default=2)
    args = parser.parse_args(argv)
    if args.command == "select":
        records, report = select_evaluation_records(args.inventory, args.components, args.prior_manifest, seed=args.seed, fake_groups_per_method=args.fake_groups_per_method, real_groups_per_method=args.real_groups_per_method, hash_audit_path=args.hash_audit)
        manifest, report_path = write_selection(records, report, args.out_dir)
        print(json.dumps({"selection":str(manifest),"report":str(report_path),"records":len(records),"labels":report["selected_labels"],"ready_for_smoke":report["status"]},indent=2))
    elif args.command == "smoke":
        print(json.dumps(run_smoke(args.selection,args.previous_root,args.output,device=args.device),indent=2))
    else:
        print(json.dumps(evaluate(args.selection,args.previous_root,args.out_dir,device=args.device,batch_size=args.batch_size),indent=2))


if __name__ == "__main__":
    main()
