"""Audit and assemble a no-copy Xception v2 training manifest.

This command never extracts video or copies images. It verifies existing legacy
crops, recovers completed v1 per-video checkpoints, adds train-only FFHQ, and
indexes frozen DF40 stills. The large compressed manifest contains file paths,
not image payloads.
"""
from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import os
import shutil
import sys
from concurrent.futures import ThreadPoolExecutor
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Mapping

from PIL import Image

from .training_data import (
    PREVIOUS_CLEAN_MANIFEST, PREVIOUS_LARGE_MANIFEST, RELEASE_SHA256,
    _image_info, iter_release, read_json_rows, sha256_file,
    ImageVerificationCache, verify_image_file, verify_legacy_large_train,
)

V1_RELEASE_SHA256 = "5aa0ce1f9c5f9db2b430f492ec672536dd98f63d375c17f0bb5bb644d4952aeb"
EXCLUDED_DF40_METHODS = {"sadtalker", "wav2lip"}
FIELDS = ["sample_id", "input_path", "input_sha256", "dataset", "method", "source_domain",
          "source_path", "source_sha256", "source_group_id", "label", "partition",
          "frame_index", "media_type", "preprocessing_profile", "prior_model_seen",
          "content_group_id", "duplicate_content_alias"]


def classify_df40_duplicate(digest: str, method_label: tuple[str, str],
                            direct_seen: Mapping[str, tuple[str, str]],
                            seen_output_hashes: set[str]) -> str:
    """Keep same-label cross-method aliases, but reject contradictory labels."""
    previous = direct_seen.get(digest)
    if previous is not None:
        if previous[1] != method_label[1]:
            return "conflicting_label"
        return "cross_method_alias" if previous[0] != method_label[0] else "same_method_duplicate"
    if digest in seen_output_hashes:
        return "duplicate_existing_output"
    return "new"


def check_df40_method_coverage(expected: set[str], observed: set[str]) -> dict[str, Any]:
    missing = sorted(expected - observed)
    unexpected = sorted(observed - expected)
    return {"ready": len(expected) == 26 and not missing and not unexpected,
            "expected_count": len(expected), "observed_count": len(observed),
            "missing": missing, "unexpected": unexpected}


def is_excluded_df40_method(method: str) -> bool:
    return method.strip().casefold() in EXCLUDED_DF40_METHODS


def _read_rows(path: Path):
    with Path(path).open(newline="", encoding="utf-8") as stream:
        yield from csv.DictReader(stream)


def seed_verified_v1_cache(cache: ImageVerificationCache, *, manifest: Path,
                           report_path: Path, processed_root: Path) -> int:
    """Seed cached v1 crops from the prior hashed manifest without reading PNG bytes.

    The compressed manifest's SHA must match its prior report, and candidate
    paths must resolve under the preserved processed root. The stat signature
    captured now is valid under the project's immutable-output policy.
    """
    if not Path(manifest).is_file() or not Path(report_path).is_file():
        return 0
    report = json.loads(Path(report_path).read_text(encoding="utf-8"))
    if report.get("release_sha256") != V1_RELEASE_SHA256:
        return 0
    expected = str(report.get("manifest_sha256", "")).lower()
    if not expected or sha256_file(manifest).lower() != expected:
        return 0
    allowed_root = Path(processed_root).resolve()
    verified_at_ns = Path(manifest).stat().st_mtime_ns
    seeded = 0
    with gzip.open(manifest, "rt", encoding="utf-8", newline="") as stream:
        for row in csv.DictReader(stream):
            if (row.get("media_type") != "video_crop"
                    or not str(row.get("preprocessing_profile", "")).startswith("v2-mtcnn")
                    or row.get("partition") != "train"):
                continue
            path = Path(row.get("input_path", ""))
            try:
                resolved = path.resolve(strict=True)
                if not resolved.is_relative_to(allowed_root) or not resolved.is_file():
                    continue
                stat = resolved.stat()
                # The manifest was emitted after its crop hashes were checked.
                # Reject crops whose filesystem change/modify time is newer.
                if stat.st_mtime_ns > verified_at_ns or stat.st_ctime_ns > verified_at_ns:
                    continue
                digest = str(row.get("input_sha256", "")).lower()
                if len(digest) != 64 or any(ch not in "0123456789abcdef" for ch in digest):
                    continue
                cache.seed_verified(resolved, (digest, 299, 299, "PNG"))
                seeded += 1
            except OSError:
                continue
    cache._connection.commit()
    return seeded


def _write_evaluation_exclusions(output_dir: Path, group_seen_ids: Mapping[str, Mapping[str, set[str]]]) -> Path:
    path = Path(output_dir) / "prior-training-source-evaluation-exclusions.csv"
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=["dataset", "source_group_id", "frozen_partition", "reason"],
                                lineterminator="\n")
        writer.writeheader()
        for dataset in sorted(group_seen_ids):
            for partition in sorted(group_seen_ids[dataset]):
                if partition == "train":
                    continue
                for group in sorted(group_seen_ids[dataset][partition]):
                    writer.writerow({"dataset": dataset, "source_group_id": group,
                                     "frozen_partition": partition,
                                     "reason": "source bytes were present in previous Xception Clean training manifest"})
    return path


def _release_indexes(release: Path, prior_train_hashes: set[str]):
    legacy_rows: list[dict[str, str]] = []
    by_path: dict[tuple[str, str], dict[str, str]] = {}
    by_sha: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
    prior_groups: set[str] = set()
    all_train_hashes: set[str] = set()
    train_counts: Counter[tuple[str, str, str]] = Counter()
    train_groups: dict[str, set[str]] = defaultdict(set)
    media_counts: Counter[tuple[str, str]] = Counter()
    group_counts: Counter[tuple[str, str, str]] = Counter()
    for row in iter_release(release):
        ds, part = row["dataset"], row["project_partition"]
        digest = row.get("sha256", "").lower()
        if ds in {"ffpp", "celebdf"}:
            legacy_rows.append(row)
            by_path[(ds, row["relative_path"].replace("\\", "/").casefold())] = row
            by_sha[(ds, digest)].append(row)
        if digest in prior_train_hashes:
            prior_groups.add(row.get("source_group_id", ""))
        group_counts[(row["dataset"], row["project_partition"], row.get("source_group_id", ""))] += 1
        if part == "train":
            train_counts[(ds, row.get("method", ""), row["label"])] += 1
            train_groups[ds].add(row.get("source_group_id", ""))
            media_counts[(ds, row.get("media_type", ""))] += 1
            all_train_hashes.add(digest)
    prior_rows_by_dataset_partition: Counter[tuple[str, str]] = Counter()
    prior_group_ids: dict[str, dict[str, set[str]]] = defaultdict(lambda: defaultdict(set))
    for (dataset, partition, group), count in group_counts.items():
        if group in prior_groups:
            prior_rows_by_dataset_partition[(dataset, partition)] += count
            prior_group_ids[dataset][partition].add(group)
    return (legacy_rows, by_path, by_sha, prior_groups, all_train_hashes,
            train_counts, train_groups, media_counts, prior_rows_by_dataset_partition, prior_group_ids)


def _group_counts_by_dataset(counts: Counter[tuple[str, str]]) -> dict[str, dict[str, int]]:
    result: dict[str, dict[str, int]] = defaultdict(dict)
    for (dataset, partition), count in counts.items():
        result[dataset][partition] = count
    return result


def _report_existing_manifest(*, manifest: Path, report_path: Path, release: Path,
                              release_report: Mapping[str, Any], processed_root: Path,
                              prior_groups: set[str], group_seen_counts: Counter,
                              group_seen_ids: Mapping[str, Mapping[str, set[str]]]) -> dict[str, Any]:
    counts: Counter[tuple[str, str, str, str]] = Counter()
    methods: Counter[tuple[str, str, str]] = Counter()
    source_groups: dict[str, set[str]] = defaultdict(set)
    profile_counts: Counter[tuple[str, str, str]] = Counter()
    df40_methods = set()
    image_count = 0
    with gzip.open(manifest, "rt", encoding="utf-8", newline="") as stream:
        for row in csv.DictReader(stream):
            image_count += 1
            counts[(row["partition"], row["dataset"], row["label"], row["media_type"])] += 1
            methods[(row["dataset"], row["method"], row["label"])] += 1
            source_groups[row["dataset"]].add(row["source_group_id"])
            profile_counts[(row["preprocessing_profile"], row["dataset"], row["label"])] += 1
            if row["dataset"] == "df40" and row["label"] == "0":
                df40_methods.add(row["method"])
    expected_groups = release_report.get("source_group_counts", {}).get("train", {})
    video_groups = {ds: int(expected_groups.get(ds, 0)) for ds in ("ffpp", "celebdf")}
    crop_groups: dict[str, set[str]] = defaultdict(set)
    with gzip.open(manifest, "rt", encoding="utf-8", newline="") as stream:
        for row in csv.DictReader(stream):
            if row["media_type"] == "video_crop":
                crop_groups[row["dataset"]].add(row["source_group_id"])
    pending = {ds: max(0, total - len(crop_groups.get(ds, set()))) for ds, total in video_groups.items()}
    prior_summary = {ds: {part: len(groups) for part, groups in parts.items()}
                     for ds, parts in group_seen_ids.items()}
    prior_rows = defaultdict(dict)
    for (ds, part), count in group_seen_counts.items():
        prior_rows[ds][part] = count
    exclusion_path = _write_evaluation_exclusions(manifest.parent, group_seen_ids)
    png_count = sum(1 for part in ("train", "val", "calibration", "test")
                    for cls in ("fake", "real")
                    for _root, _dirs, files in os.walk(Path(processed_root) / part / cls)
                    for name in files if name.lower().endswith(".png"))
    v1_crops = sum(n for (profile, _ds, _label), n in profile_counts.items() if profile.startswith("v2-mtcnn"))
    legacy_crops = sum(n for (profile, _ds, _label), n in profile_counts.items() if profile.startswith("legacy-large"))
    ffhq_count = sum(n for (profile, ds, label), n in profile_counts.items() if ds == "ffhq" and label == "1")
    df40_counts = {f"{m}|{label}": n for (ds, m, label), n in methods.items() if ds == "df40"}
    report = {
        "status": "PARTIAL_READY" if any(pending.values()) else "READY",
        "release_path": str(release), "release_sha256": RELEASE_SHA256,
        "manifest_path": str(manifest), "manifest_sha256": sha256_file(manifest),
        "source_image_copy_count": 0,
        "legacy_large": {"reusable_verified_unique_crops": legacy_crops,
                         "reusable_by_dataset_label": {f"{ds}|{label}": n for (profile, ds, label), n in profile_counts.items() if profile.startswith("legacy-large")},
                         "verification": "per-image crop digest matched previous input_manifest.json; all included crops decoded at 299x299"},
        "existing_v2_pngs": {"complete_video_checkpoints": int(json.loads((Path(processed_root)/"preparation-report.json").read_text(encoding="utf-8")).get("processed_sources", 0)),
                              "png_files_on_disk": png_count, "reused_verified_unique_crops": v1_crops,
                              "note": "The prior preparation report ended with KeyboardInterrupt and images.csv was not committed; valid completed outputs were reconstructed from per-video checkpoints."},
        "prior_clean_training_source_audit": {
            "matched_groups_by_dataset_partition": prior_summary,
            "v2_rows_in_prior_seen_groups": {ds: dict(parts) for ds, parts in prior_rows.items()},
            "evaluation_exclusions_path": str(exclusion_path),
            "evaluation_exclusions_sha256": sha256_file(exclusion_path),
            "evaluation_independence": "Exclude all records in these source groups from independent evaluation; frozen v2.1 partitions remain unchanged.",
            "person_identity_independence": "not established",
        },
        "ffhq": {"verified_unique_train_images": ffhq_count, "partition": "train only", "label": "1=real"},
        "df40": {"train_images_by_method_label": df40_counts,
                 "target_methods_with_fake_train": sorted(df40_methods),
                 "target_fake_method_coverage": f"{len(df40_methods)}/26"},
        "ready_image_count": image_count,
        "ready_counts_by_partition_dataset_label_type": {"|".join(k): v for k, v in counts.items()},
        "ready_counts_by_dataset_method_label": {"|".join(k): v for k, v in methods.items()},
        "training_video_source_groups": video_groups,
        "training_video_groups_without_verified_crops": pending,
        "sampler_policy": {
            "classes": "50/50 fake/real per even-sized batch; target 0=fake and 1=real",
            "fake_dataset_weights": {"df40": .50, "ffpp": .25, "celebdf": .25},
            "real_dataset_weights": {"df40": .40, "ffpp": .20, "celebdf": .20, "ffhq": .20},
            "within_dataset": "uniform source groups; DF40 fake method first, then source group; no repeated source group within a batch",
        },
        "xception_contract": {"architecture": "legacy_xception.tf_in1k", "input_size": 299,
                              "mean": [.5, .5, .5], "std": [.5, .5, .5], "class_map": {"fake": 0, "real": 1}},
        "limitations": [
            "Previous Clean training source groups occur in v2.1 evaluation partitions; those groups are excluded from independent evaluation without changing v2.1.",
            "FFHQ is training-only; cross-dataset identity independence is unverified.",
            "Legacy large crops use a 1.5 face margin versus 1.3 in the Clean model's training configuration.",
            "The interrupted v1 preparation has only 5,240 completed video checkpoints; remaining v2.1 FF++/Celeb-DF sources still require extraction.",
        ],
    }
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def prepare_training_manifest(
    *, release: Path, release_report: Path, previous_root: Path,
    processed_root: Path, ffhq_root: Path, output_dir: Path,
    verify_legacy_hashes: bool = True,
    workers: int = 8,
) -> dict[str, Any]:
    release = Path(release).resolve(); release_report = Path(release_report).resolve()
    output_dir = Path(output_dir).resolve()
    report_data = json.loads(release_report.read_text(encoding="utf-8"))
    if report_data.get("status") != "PASS_FROZEN" or report_data.get("release_sha256") != RELEASE_SHA256:
        raise ValueError("DF40-focused v2.1 release gate or checksum failed")
    clean_manifest = Path(previous_root) / "runs/best_xception/xcep-clean-s42/input_manifest.json"
    large_manifest = Path(previous_root) / "runs/archive/xcep-final-v1-s42-STACKED-REGULARIZATION-FAILED/input_manifest.json"
    clean_rows = read_json_rows(clean_manifest)
    clean_train = [row for row in clean_rows if row.get("split") == "train"]
    prior_train_hashes = {str(row.get("sha256", "")).lower() for row in clean_train if row.get("sha256")}
    prior_train_frames = {
        (str(row.get("dataset", "")), str(row.get("sha256", "")).lower(), int(row.get("frame_index", -1)))
        for row in clean_train if row.get("sha256") and row.get("frame_index") is not None
    }
    del clean_rows
    (legacy_rows, by_path, by_sha, prior_groups, train_hashes, train_counts,
     train_groups, media_counts, group_seen_counts, group_seen_ids) = _release_indexes(release, prior_train_hashes)
    v1_config_path = Path(processed_root) / ".state/config.json"
    v1_config = json.loads(v1_config_path.read_text(encoding="utf-8"))
    if v1_config.get("manifest_sha256") != V1_RELEASE_SHA256:
        raise ValueError("Existing v1 crop checkpoints do not match the frozen v1 release")

    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "images.csv.gz"
    report_path = output_dir / "preparation-report.json"
    if report_path.exists():
        raise FileExistsError("Refusing to overwrite an existing training preparation report")
    if manifest_path.exists():
        report = _report_existing_manifest(manifest=manifest_path, report_path=report_path,
            release=release, release_report=report_data, processed_root=processed_root,
            prior_groups=prior_groups, group_seen_counts=group_seen_counts, group_seen_ids=group_seen_ids)
        print(json.dumps({"status": report["status"], "ready_image_count": report["ready_image_count"],
                          "manifest_path": report["manifest_path"],
                          "pending_video_groups": report["training_video_groups_without_verified_crops"]}, indent=2))
        return report
    free_bytes = shutil.disk_usage(output_dir).free
    if free_bytes < 2 * 1024**3:
        raise OSError("At least 2 GiB free space is required for the compressed manifest and temporary file")
    verification_cache = ImageVerificationCache(output_dir / "image-verification-cache.sqlite3")
    v1_cached_seed_count = seed_verified_v1_cache(
        verification_cache,
        manifest=output_dir.parent / "xception-v2-ready-v1" / "images.csv.gz",
        report_path=output_dir.parent / "xception-v2-ready-v1" / "preparation-report.json",
        processed_root=processed_root,
    )
    partial = manifest_path.with_suffix(".csv.gz.partial")
    counts: Counter[tuple[str, str, str, str]] = Counter()
    method_counts: Counter[tuple[str, str, str]] = Counter()
    seen_source_frames: set[tuple[str, str, int]] = set()
    seen_output_hashes: set[str] = set()
    reused_v1: Counter[tuple[str, str]] = Counter()
    skipped_v1: Counter[str] = Counter()
    image_count = 0
    video_groups_with_crops: set[str] = set()
    prior_frame_skips = 0
    duplicate_skips = 0

    def write_example(writer, row: dict[str, Any], *, source_frame_key: tuple[str, str, int] | None = None,
                      allow_content_alias: bool = False):
        nonlocal image_count, prior_frame_skips, duplicate_skips
        if source_frame_key is not None:
            if source_frame_key in prior_train_frames:
                prior_frame_skips += 1
                return
            if source_frame_key in seen_source_frames:
                duplicate_skips += 1
                return
        digest = str(row.get("input_sha256", ""))
        if digest and digest in seen_output_hashes and not allow_content_alias:
            duplicate_skips += 1
            return
        if source_frame_key is not None:
            seen_source_frames.add(source_frame_key)
        if digest:
            seen_output_hashes.add(digest)
        row["sample_id"] = hashlib.sha256(
            f"{row['dataset']}|{row['source_path']}|{row.get('frame_index','')}|{digest}".encode("utf-8")
        ).hexdigest()[:24]
        row["source_domain"] = row.get("source_domain") or row["dataset"]
        row["prior_model_seen"] = str(row.get("source_group_id") in prior_groups).lower()
        row["content_group_id"] = f"sha256:{digest}" if digest else ""
        row["duplicate_content_alias"] = str(bool(allow_content_alias)).lower()
        writer.writerow({field: row.get(field, "") for field in FIELDS})
        image_count += 1
        counts[(row["partition"], row["dataset"], row["label"], row["media_type"])] += 1
        method_counts[(row["dataset"], row["method"], row["label"])] += 1
        if row["media_type"] == "video_crop":
            video_groups_with_crops.add(str(row["source_group_id"]))

    with partial.open("wb") as raw:
        import io
        gz = gzip.GzipFile(fileobj=raw, mode="wb", mtime=0)
        text = io.TextIOWrapper(gz, encoding="utf-8", newline="")
        writer = csv.DictWriter(text, fieldnames=FIELDS, lineterminator="\n")
        writer.writeheader()

        # Reuse only intact v1 checkpoint outputs that retain the same frozen partition.
        state_root = Path(processed_root) / ".state/videos"
        checkpoint_count = 0
        v1_candidates = []
        for directory in state_root.iterdir():
            if directory.is_symlink() or not directory.is_dir():
                continue
            complete = directory / "complete.json"
            if not complete.is_file():
                continue
            checkpoint_count += 1
            try:
                meta = json.loads(complete.read_text(encoding="utf-8"))
                config = meta["config"]; prov = config["provenance"]
                key = (prov["dataset"], prov["relative_path"].replace("\\", "/").casefold())
                current = by_path.get(key)
                if current is None or current["project_partition"] != "train" or prov.get("project_partition") != "train":
                    skipped_v1["source_missing_or_partition_changed"] += len(meta.get("frames", [])); continue
                if current["label"] != str(prov.get("label")):
                    skipped_v1["label_mismatch"] += len(meta.get("frames", [])); continue
                for frame in meta.get("frames", []):
                    path = (directory / frame["file"]).resolve()
                    v1_candidates.append((current, frame, path))
            except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError):
                skipped_v1["invalid_checkpoint_or_image"] += 1

        with ThreadPoolExecutor(max_workers=max(1, workers)) as executor:
            for index, ((current, frame, path), verification) in enumerate(
                zip(v1_candidates, executor.map(lambda item: verification_cache.verify(item[2]), v1_candidates)), 1):
                try:
                    digest, width, height, _ = verification
                    if (width, height) != (299, 299):
                        skipped_v1["dimension_mismatch"] += 1; continue
                    source_frame = (current["dataset"], current["sha256"], int(frame["frame_index"]))
                    row = {"input_path": str(path), "input_sha256": digest, "dataset": current["dataset"],
                           "method": current["method"], "label": current["label"], "partition": "train",
                           "source_path": current["absolute_path"], "source_sha256": current["sha256"],
                           "source_group_id": current["source_group_id"], "frame_index": frame["frame_index"],
                           "media_type": "video_crop", "preprocessing_profile": "v2-mtcnn-quality-square299-margin1.5"}
                    before = image_count
                    write_example(writer, row, source_frame_key=source_frame)
                    if image_count > before:
                        reused_v1[(current["dataset"], current["label"])] += 1
                except (OSError, ValueError) as exc:
                    skipped_v1[str(exc)] += 1
                if index % 5000 == 0:
                    print(f"verified {index:,}/{len(v1_candidates):,} existing v1 crops", flush=True)

        legacy_usable, legacy_report = verify_legacy_large_train(
            legacy_rows, large_manifest, verify_hashes=verify_legacy_hashes, workers=workers,
            verification_cache=verification_cache,
        )
        legacy_added = Counter()
        for row in legacy_usable:
            source_key = (row["dataset"], row["source_sha256"], int(row["frame_index"]))
            before = image_count
            write_example(writer, row, source_frame_key=source_key)
            if image_count > before:
                legacy_added[(row["dataset"], row["label"])] += 1

        # FFHQ is verified real-only training material. Each image forms its own group.
        ffhq_count = 0; ffhq_duplicate_count = 0; ffhq_bad: Counter[str] = Counter()
        ffhq_files = sorted(Path(ffhq_root).glob("*.jpg"))
        ffhq_sha_against_sources = train_hashes
        def verify_ffhq(path):
            try:
                return verification_cache.verify(path), None
            except (OSError, ValueError) as exc:
                return None, str(exc)
        with ThreadPoolExecutor(max_workers=max(1, workers)) as executor:
          for index, (path, result) in enumerate(zip(ffhq_files, executor.map(verify_ffhq, ffhq_files)), 1):
            try:
                verification, error = result
                if error:
                    ffhq_bad["unreadable"] += 1; continue
                digest, width, height, fmt = verification
                if fmt not in {"JPEG", "MPO"} or min(width, height) < 64:
                    ffhq_bad["unsupported_or_small"] += 1; continue
                if digest in seen_output_hashes or digest in ffhq_sha_against_sources:
                    ffhq_duplicate_count += 1; continue
                source_group = f"ffhq:{digest}"
                row = {"input_path": str(path.resolve()), "input_sha256": digest, "dataset": "ffhq",
                       "method": "original", "label": "1", "partition": "train",
                       "source_path": str(path.resolve()), "source_sha256": digest,
                       "source_group_id": source_group, "frame_index": "", "media_type": "still_image",
                       "preprocessing_profile": f"ffhq-square-resize299-bicubic-source-{width}x{height}"}
                before = image_count
                write_example(writer, row)
                if image_count > before:
                    ffhq_count += 1
                else:
                    ffhq_duplicate_count += 1
            except (OSError, ValueError):
                ffhq_bad["unreadable"] += 1
            if index % 10000 == 0:
                print(f"verified {index:,}/{len(ffhq_files):,} FFHQ files", flush=True)

        # DF40 is already image-based. Keep its files in place and decode at training time.
        df40_direct = Counter(); df40_missing = Counter(); df40_invalid = Counter()
        df40_excluded = Counter()
        direct_seen: dict[str, tuple[str, str]] = {}
        df40_aliases = 0
        df40_conflicts = 0
        df40_expected_methods: set[str] = set()
        train_video_sources = Counter()
        for row in iter_release(release):
            if row["project_partition"] != "train":
                continue
            if row["media_type"] == "video":
                train_video_sources[(row["dataset"], row["source_group_id"])] += 1
                continue
            if row["dataset"] != "df40":
                continue
            if is_excluded_df40_method(row["method"]):
                df40_excluded[row["method"]] += 1
                continue
            if row["media_type"] not in {"image", "gif"}:
                continue
            if row["label"] == "0":
                df40_expected_methods.add(row["method"])
            path = Path(row["absolute_path"])
            if not path.is_file():
                df40_missing[row["method"]] += 1; continue
            try:
                digest, width, height, _fmt = verification_cache.verify(path)
            except (OSError, ValueError) as exc:
                df40_invalid[f"{row['method']}:{type(exc).__name__}"] += 1
                continue
            if digest.lower() != row["sha256"].lower():
                df40_invalid[f"{row['method']}:sha256_mismatch"] += 1
                continue
            if min(width, height) < 64:
                df40_invalid[f"{row['method']}:dimensions_below_64"] += 1
                continue
            method_label = (row["method"], row["label"])
            duplicate_action = classify_df40_duplicate(digest, method_label, direct_seen, seen_output_hashes)
            allow_alias = duplicate_action == "cross_method_alias"
            if duplicate_action == "conflicting_label":
                df40_conflicts += 1
                df40_invalid[f"{row['method']}:conflicting_duplicate_label"] += 1
                continue
            if duplicate_action in {"same_method_duplicate", "duplicate_existing_output"}:
                df40_direct["exact_duplicate_dropped"] += 1; continue
            if allow_alias:
                df40_aliases += 1
            else:
                direct_seen[digest] = method_label
            output_row = {"input_path": str(path), "input_sha256": digest, "dataset": "df40",
                          "method": row["method"], "label": row["label"], "partition": "train",
                          "source_path": row["absolute_path"], "source_sha256": digest,
                          "source_group_id": row["source_group_id"], "frame_index": 0 if row["media_type"] == "gif" else "",
                          "media_type": "gif" if row["media_type"] == "gif" else "still_image",
                          "preprocessing_profile": "df40-preprocessed-no-recrop-resize299-bicubic"}
            before = image_count
            write_example(writer, output_row, allow_content_alias=allow_alias)
            if image_count > before:
                df40_direct[(row["method"], row["label"])] += 1
        # Include method exclusions and pending video processing explicitly in the report.
        text.flush(); text.detach(); gz.close()
        raw.flush(); os.fsync(raw.fileno())
    os.replace(partial, manifest_path)

    train_groups_total = {dataset: len(groups) for dataset, groups in train_groups.items()}
    video_groups = {dataset: len({g for (ds, g) in train_video_sources if ds == dataset}) for dataset in {d for d, _ in train_video_sources}}
    video_crop_groups = defaultdict(set)
    # Re-read the compressed manifest to compute precisely covered source groups.
    import io
    with gzip.open(manifest_path, "rt", encoding="utf-8", newline="") as stream:
        for row in csv.DictReader(stream):
            if row["media_type"] == "video_crop":
                video_crop_groups[row["dataset"]].add(row["source_group_id"])
    pending_video_groups = {}
    for dataset in video_groups:
        active = {group for ds, group in train_video_sources if ds == dataset}
        pending_video_groups[dataset] = len(active - video_crop_groups.get(dataset, set()))

    exclusion_path = _write_evaluation_exclusions(output_dir, group_seen_ids)

    covered_df40_methods = {method for (method, label), count in df40_direct.items()
                            if isinstance(method, str) and label == "0" and count > 0}
    df40_coverage = check_df40_method_coverage(df40_expected_methods, covered_df40_methods)
    report = {
        "status": ("BLOCKED_DF40_METHOD_COVERAGE" if not df40_coverage["ready"]
                   else ("PARTIAL_READY" if any(pending_video_groups.values()) else "READY")),
        "release_path": str(release), "release_sha256": RELEASE_SHA256,
        "manifest_path": str(manifest_path), "manifest_sha256": sha256_file(manifest_path),
        "source_image_copy_count": 0,
        "legacy_large": {**legacy_report, "added_after_frame_and_byte_dedup": dict(legacy_added)},
        "existing_v2_pngs": {"complete_video_checkpoints": checkpoint_count,
                              "png_files_on_disk": sum(1 for part in ("train", "val", "calibration", "test")
                                  for cls in ("fake", "real")
                                  for _root, _dirs, files in os.walk(Path(processed_root) / part / cls)
                                  for name in files if name.lower().endswith(".png")),
                              "reused_after_hash_and_partition_checks": dict(reused_v1),
                              "rejected_frames": dict(skipped_v1),
                              "note": "The prior preparation report ended with KeyboardInterrupt and images.csv was not committed; verified per-video completions were reconstructed without changing the old output."},
        "prior_clean_training_source_audit": {
            "prior_train_crop_rows": len(prior_train_frames),
            "prior_train_source_hashes": len(prior_train_hashes),
            "matched_source_groups": {ds: {part: len(groups) for part, groups in parts.items()}
                                      for ds, parts in group_seen_ids.items()},
            "v2_rows_in_prior_seen_groups": {ds: dict(parts) for ds, parts in _group_counts_by_dataset(group_seen_counts).items()},
            "evaluation_exclusions_path": str(exclusion_path),
            "evaluation_exclusions_sha256": sha256_file(exclusion_path),
            "policy": "Do not use prior-Clean-trained source groups as independent v2 evaluation; retain frozen v2.1 partitions and mask those groups in derived evaluation reports.",
            "person_identity_independence": "not established",
        },
        "ffhq": {"jpg_candidates": len(ffhq_files), "verified_unique_train_images": ffhq_count,
                 "duplicates_against_training_sources_or_output": ffhq_duplicate_count, "invalid": dict(ffhq_bad),
                 "partition": "train only", "label": "1 (real) by validated FFHQ real-image source convention"},
        "df40": {"deduplicated_train_images_by_method_label": {f"{m}|{y}": n for (m, y), n in df40_direct.items() if isinstance(m, str)},
                 "missing_by_method": dict(df40_missing), "invalid_by_method_reason": dict(df40_invalid),
                 "expected_fake_training_methods": sorted(df40_expected_methods), "target_methods_with_train_fake": sorted(
                     method for (method, label), n in df40_direct.items() if isinstance(method, str) and label == "0" and n > 0),
                 "target_method_coverage": df40_coverage,
                 "cross_method_exact_content_alias_rows": df40_aliases,
                 "conflicting_label_duplicates_rejected": df40_conflicts,
                 "excluded_experiment_methods": ["SadTalker", "Wav2Lip"],
                 "excluded_records_by_method": dict(df40_excluded)},
        "ready_image_count": image_count,
        "verification_cache": {"path": str(verification_cache.database),
                               "sha256_cache_hits": verification_cache.hits,
                               "full_hash_and_decode_misses": verification_cache.misses,
                               "seeded_from_sha_verified_v1_manifest": v1_cached_seed_count,
                               "cache_key": "resolved path + size + mtime_ns + ctime_ns + device + file id"},
        "ready_counts_by_partition_dataset_label_type": {"|".join(k): v for k, v in counts.items()},
        "ready_counts_by_dataset_method_label": {"|".join(k): v for k, v in method_counts.items()},
        "train_source_groups_by_dataset": train_groups_total,
        "training_video_source_groups": video_groups,
        "training_video_groups_without_verified_crops": pending_video_groups,
        "deduplicated_prior_clean_frames": prior_frame_skips,
        "deduplicated_frame_or_exact_output_bytes": duplicate_skips,
        "sampler_policy": {
            "classes": "50/50 fake/real per even-sized batch; target 0=fake and 1=real",
            "fake_dataset_weights": {"df40": .50, "ffpp": .25, "celebdf": .25},
            "real_dataset_weights": {"df40": .40, "ffpp": .20, "celebdf": .20, "ffhq": .20},
            "within_dataset": "uniform source groups; for DF40 fake, uniform method then source group; no repeated group in a batch",
        },
        "xception_contract": {"architecture": "legacy_xception.tf_in1k", "input_size": 299,
                              "mean": [0.5, 0.5, 0.5], "std": [0.5, 0.5, 0.5],
                              "class_map": {"fake": 0, "real": 1}},
        "limitations": [
            "FFHQ and DF40 cross-dataset person identity independence is unknown; FFHQ is training-only.",
            "The v2.1 test partition includes exact source videos used in the prior Clean training; those source groups must be excluded from independent evaluation.",
            "Legacy large crops use a 1.5 crop margin versus 1.3 for the Clean checkpoint; both are 299 RGB inputs with identical normalization but crop distributions differ.",
            "The current v1 preparation run was interrupted; only complete per-video checkpoints with valid 299px outputs are reused.",
        ],
    }
    verification_cache.close()
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def main(argv: list[str] | None = None) -> int:
    root = Path(__file__).resolve().parents[2]
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--release", type=Path, default=root / "artifacts/allocation/df40-focused-v2.1-final/df40-focused-v2.1.csv")
    p.add_argument("--release-report", type=Path, default=root / "artifacts/allocation/df40-focused-v2.1-final/readiness.json")
    p.add_argument("--previous-root", type=Path, default=Path(r"D:\University\explainable-deepfake-detection-system"))
    p.add_argument("--processed-root", type=Path, default=root / "data/processed/xception-v1")
    p.add_argument("--ffhq-root", type=Path, default=Path(r"D:\University\datasets\deepfake\ffhq\resized"))
    p.add_argument("--output-dir", type=Path, default=root / "artifacts/training/xception-v2-ready-v2")
    p.add_argument("--workers", type=int, default=min(8, os.cpu_count() or 1))
    a = p.parse_args(argv)
    report = prepare_training_manifest(release=a.release, release_report=a.release_report,
        previous_root=a.previous_root, processed_root=a.processed_root, ffhq_root=a.ffhq_root,
        output_dir=a.output_dir, workers=max(1, a.workers))
    print(json.dumps({"status": report["status"], "ready_image_count": report["ready_image_count"],
                      "manifest_path": report["manifest_path"],
                      "pending_video_groups": report["training_video_groups_without_verified_crops"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
