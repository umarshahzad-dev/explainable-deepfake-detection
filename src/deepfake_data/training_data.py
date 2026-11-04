"""Verified, no-copy training inputs and group-aware Xception batch sampling.

The module treats the frozen allocation as the source of labels and partitions.
Legacy crop folders and interrupted v1 outputs are merely candidate image
sources; a crop is reusable only when its source resolves to a train row and
its recorded content digest and dimensions validate.
"""
from __future__ import annotations

import csv
import hashlib
import io
import json
import os
import random
import sqlite3
import threading
from concurrent.futures import ThreadPoolExecutor
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping, Sequence

from PIL import Image, ImageOps

RELEASE_SHA256 = "ba181d1085fb82fb6d58d14f9cca869d4d105236e9e4d5402c65c32b91d1117f"
PREVIOUS_CLEAN_MANIFEST = Path(
    r"D:\University\explainable-deepfake-detection-system\runs\best_xception\xcep-clean-s42\input_manifest.json"
)
PREVIOUS_LARGE_MANIFEST = Path(
    r"D:\University\explainable-deepfake-detection-system\runs\archive\xcep-final-v1-s42-STACKED-REGULARIZATION-FAILED\input_manifest.json"
)


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        while chunk := stream.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def iter_release(path: Path, expected_sha256: str = RELEASE_SHA256) -> Iterator[dict[str, str]]:
    path = Path(path)
    actual = sha256_file(path)
    if actual != expected_sha256:
        raise ValueError(f"Frozen v2.1 release SHA mismatch: {actual}")
    with path.open(newline="", encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            if row.get("project_partition") not in {"train", "val", "calibration", "test"}:
                continue
            if row.get("label") not in {"0", "1"}:
                raise ValueError("Active release row has an unresolved label")
            if not row.get("source_group_id") or not row.get("sha256"):
                raise ValueError("Active release row lacks source group or content hash")
            yield row


def read_release(path: Path, expected_sha256: str = RELEASE_SHA256) -> list[dict[str, str]]:
    return list(iter_release(path, expected_sha256))


def read_json_rows(path: Path) -> list[dict[str, Any]]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict) or not isinstance(value.get("rows"), list):
        raise ValueError(f"Expected an input_manifest.json rows array: {path}")
    return value["rows"]


def _image_info(path: Path) -> tuple[int, int, str]:
    data = Path(path).read_bytes()
    with Image.open(io.BytesIO(data)) as image:
        image.verify()
    with Image.open(io.BytesIO(data)) as image:
        image = ImageOps.exif_transpose(image)
        return image.width, image.height, image.format or "unknown"


def verify_image_file(path: Path) -> tuple[str, int, int, str]:
    """Read each file once, then verify bytes and decode from a memory buffer."""
    data = Path(path).read_bytes()
    digest = hashlib.sha256(data).hexdigest()
    with Image.open(io.BytesIO(data)) as image:
        image_format = image.format or "unknown"
        image.verify()
    with Image.open(io.BytesIO(data)) as image:
        image = ImageOps.exif_transpose(image)
        return digest, image.width, image.height, image_format


class ImageVerificationCache:
    """Resumable image integrity cache keyed by path and stable file metadata.

    The SHA and decode are computed once. Subsequent runs reuse them only while
    size, mtime, ctime, device and file id remain unchanged. The cache is for
    immutable source/crop trees; a changed stat fingerprint forces revalidation.
    """

    def __init__(self, database: Path):
        self.database = Path(database)
        self.database.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._local = threading.local()
        self._connection = sqlite3.connect(self.database, check_same_thread=False)
        self._connection.execute("PRAGMA journal_mode=WAL")
        self._connection.execute("PRAGMA synchronous=NORMAL")
        self._connection.execute("""CREATE TABLE IF NOT EXISTS verified_images (
            path TEXT PRIMARY KEY, size INTEGER NOT NULL, mtime_ns INTEGER NOT NULL,
            ctime_ns INTEGER NOT NULL, device INTEGER NOT NULL, file_id INTEGER NOT NULL,
            sha256 TEXT NOT NULL, width INTEGER NOT NULL, height INTEGER NOT NULL,
            image_format TEXT NOT NULL, verified_utc TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
        )""")
        self._connection.commit()
        self.hits = 0
        self.misses = 0
        self.seeded = 0

    def verify(self, path: Path) -> tuple[str, int, int, str]:
        path = Path(path).resolve(strict=True)
        stat = path.stat()
        signature = (stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns,
                     int(getattr(stat, "st_dev", 0)), int(getattr(stat, "st_ino", 0)))
        key = os.path.normcase(str(path))
        with self._lock:
            cached = self._connection.execute(
                "SELECT size,mtime_ns,ctime_ns,device,file_id,sha256,width,height,image_format "
                "FROM verified_images WHERE path=?", (key,)
            ).fetchone()
            if cached and tuple(cached[:5]) == signature:
                self.hits += 1
                return str(cached[5]), int(cached[6]), int(cached[7]), str(cached[8])

        # Hash and decode outside the lock so independent worker threads can
        # verify files in parallel.
        result = verify_image_file(path)
        with self._lock:
            self._connection.execute(
                "INSERT INTO verified_images(path,size,mtime_ns,ctime_ns,device,file_id,sha256,width,height,image_format) "
                "VALUES(?,?,?,?,?,?,?,?,?,?) ON CONFLICT(path) DO UPDATE SET "
                "size=excluded.size,mtime_ns=excluded.mtime_ns,ctime_ns=excluded.ctime_ns,"
                "device=excluded.device,file_id=excluded.file_id,sha256=excluded.sha256,"
                "width=excluded.width,height=excluded.height,image_format=excluded.image_format,"
                "verified_utc=CURRENT_TIMESTAMP",
                (key, *signature, *result),
            )
            self.misses += 1
            if self.misses % 128 == 0:
                self._connection.commit()
        return result

    def seed_verified(self, path: Path, result: tuple[str, int, int, str]) -> None:
        """Import a per-file digest from a separately SHA-verified manifest."""
        path = Path(path).resolve(strict=True)
        stat = path.stat()
        signature = (stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns,
                     int(getattr(stat, "st_dev", 0)), int(getattr(stat, "st_ino", 0)))
        key = os.path.normcase(str(path))
        with self._lock:
            self._connection.execute(
                "INSERT INTO verified_images(path,size,mtime_ns,ctime_ns,device,file_id,sha256,width,height,image_format) "
                "VALUES(?,?,?,?,?,?,?,?,?,?) ON CONFLICT(path) DO UPDATE SET "
                "size=excluded.size,mtime_ns=excluded.mtime_ns,ctime_ns=excluded.ctime_ns,"
                "device=excluded.device,file_id=excluded.file_id,sha256=excluded.sha256,"
                "width=excluded.width,height=excluded.height,image_format=excluded.image_format,"
                "verified_utc=CURRENT_TIMESTAMP",
                (key, *signature, *result),
            )
            self.seeded += 1
            if self.seeded % 128 == 0:
                self._connection.commit()

    def close(self) -> None:
        with self._lock:
            self._connection.commit()
            self._connection.close()


def resolve_legacy_crop_path(value: str, input_manifest: Path) -> Path:
    """Resolve old experiment/crops paths through the previous project's crops junction."""
    direct = Path(value)
    if direct.is_file():
        return direct
    parts = Path(value).parts
    indexes = [i for i, part in enumerate(parts) if part.casefold() == "crops"]
    if indexes:
        project_root = Path(input_manifest).resolve().parents[3]
        candidate = project_root / "crops" / Path(*parts[indexes[-1] + 1:])
        return candidate
    return direct


def _legacy_relative_path(row: Mapping[str, Any]) -> str:
    value = str(row.get("raw_path") or "").replace("\\", "/").casefold()
    for marker in ("/data/raw/", "/celeb-df-v2/", "/ffpp/"):
        if marker in value:
            return value.split(marker, 1)[1]
    return value.rsplit("/", 1)[-1]


def _legacy_source_matches(row: Mapping[str, Any], by_sha: Mapping[tuple[str, str], list[dict[str, str]]], by_path: Mapping[tuple[str, str], dict[str, str]]) -> dict[str, str] | None:
    digest = str(row.get("sha256") or "").lower()
    dataset = str(row.get("dataset") or "")
    path_match = by_path.get((dataset, _legacy_relative_path(row)))
    if path_match is not None and path_match.get("sha256", "").lower() == digest:
        return path_match
    matches = by_sha.get((dataset, digest), [])
    if len(matches) == 1:
        return matches[0]
    return None


def audit_prior_sources(release_rows: Sequence[Mapping[str, str]], prior_manifest: Path) -> dict[str, Any]:
    """Mark v2.1 groups seen in the previous Clean checkpoint's train rows.

    Source byte digests are the linkage evidence. Person identity is not inferred.
    """
    rows = read_json_rows(prior_manifest)
    trained_source_hashes = {
        str(row.get("sha256", "")).lower()
        for row in rows
        if row.get("split") == "train" and row.get("sha256")
    }
    matched_rows = [row for row in release_rows if row.get("sha256", "").lower() in trained_source_hashes]
    prior_groups = {str(row["source_group_id"]) for row in matched_rows}
    group_rows = [row for row in release_rows if str(row.get("source_group_id", "")) in prior_groups]
    counts: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    group_counts: dict[str, dict[str, set[str]]] = defaultdict(lambda: defaultdict(set))
    eval_excluded: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for row in group_rows:
        ds, part = str(row["dataset"]), str(row["project_partition"])
        counts[ds][part] += 1
        group_counts[ds][part].add(str(row["source_group_id"]))
        if part in {"val", "calibration", "test"}:
            eval_excluded[ds][part] += 1
    return {
        "previous_manifest": str(prior_manifest),
        "prior_train_rows": sum(row.get("split") == "train" for row in rows),
        "matched_v2_source_rows": len(matched_rows),
        "matched_v2_source_groups": len(prior_groups),
        "matched_groups_by_dataset_partition": {
            ds: {part: len(groups) for part, groups in parts.items()}
            for ds, parts in group_counts.items()
        },
        "all_v2_rows_in_prior_seen_groups": {ds: dict(parts) for ds, parts in counts.items()},
        "evaluation_rows_excluded_from_independence_claim": {
            ds: dict(parts) for ds, parts in eval_excluded.items()
        },
        "identity_independence_claim": "not established; byte-linked source files and release source groups only",
    }


def verify_legacy_large_train(
    release_rows: Sequence[Mapping[str, str]],
    input_manifest: Path = PREVIOUS_LARGE_MANIFEST,
    *,
    verify_hashes: bool = True,
    workers: int = 8,
    verification_cache: ImageVerificationCache | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Return valid old large-crop rows mapping to frozen v2.1 train sources.

    Old train, validation, calibration and test crops may all be used only when
    their source is assigned to v2.1 train. This never promotes an old train
    source into a new evaluation partition.
    """
    by_sha: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
    by_path: dict[tuple[str, str], dict[str, str]] = {}
    for row in release_rows:
        by_sha[(row["dataset"], row["sha256"].lower())].append(dict(row))
        by_path[(row["dataset"], row["relative_path"].replace("\\", "/").casefold())] = dict(row)
    legacy_rows = read_json_rows(input_manifest)
    usable: list[dict[str, Any]] = []
    counts: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    excluded: dict[str, int] = defaultdict(int)
    invalid: dict[str, int] = defaultdict(int)
    seen_frame: set[tuple[str, str, int]] = set()
    seen_hash: set[str] = set()
    candidates = []
    legacy_split_counts: dict[str, int] = defaultdict(int)
    for old in legacy_rows:
        if old.get("split") not in {"train", "val", "calibration", "test"}:
            continue
        legacy_split_counts[str(old.get("split"))] += 1
        current = _legacy_source_matches(old, by_sha, by_path)
        if current is None or current.get("project_partition") != "train":
            excluded[str(old.get("dataset", "unknown"))] += 1
            continue
        if str(old.get("label")) != str(current.get("label")):
            invalid["label_conflict"] += 1
            continue
        path = resolve_legacy_crop_path(str(old.get("crop_path", "")), input_manifest)
        candidates.append((old, current, path))

    def verify(item):
        old, current, path = item
        if not path.is_file():
            return item, None, "missing_crop"
        try:
            if verify_hashes:
                digest, width, height, fmt = (verification_cache.verify(path) if verification_cache
                                               else verify_image_file(path))
            else:
                digest = str(old.get("crop_sha256", ""))
                width, height, fmt = _image_info(path)
            if not digest or (old.get("crop_sha256") and digest.lower() != str(old["crop_sha256"]).lower()):
                return item, None, "crop_hash_mismatch"
            if (width, height) != (299, 299):
                return item, None, "crop_dimensions_not_299"
            return item, (digest, fmt), None
        except (OSError, ValueError) as exc:
            return item, None, str(exc)

    with ThreadPoolExecutor(max_workers=max(1, workers)) as executor:
      for checked, (item, verified_value, error) in enumerate(executor.map(verify, candidates), 1):
        old, current, path = item
        if error:
            invalid[error] += 1
            continue
        digest, fmt = verified_value
        frame_key = (str(old["dataset"]), str(old.get("sha256", "")), int(old.get("frame_index", -1)))
        if frame_key in seen_frame or digest in seen_hash:
            invalid["duplicate_frame_or_bytes"] += 1
            continue
        seen_frame.add(frame_key)
        seen_hash.add(digest)
        row = {
            "input_path": str(path.resolve()), "input_sha256": digest,
            "dataset": current["dataset"], "method": current["method"],
            "label": current["label"], "partition": "train",
            "source_path": current["absolute_path"], "source_sha256": current["sha256"],
            "source_group_id": current["source_group_id"],
            "frame_index": int(old.get("frame_index", -1)),
            "media_type": "video_crop", "preprocessing_profile": "legacy-large-square299-margin1.5-bicubic",
            "preprocessing_compatible": True,
        }
        usable.append(row)
        counts[current["dataset"]][current["label"]] += 1
        if checked % 5000 == 0:
            print(f"verified {checked:,}/{len(candidates):,} legacy crop candidates", flush=True)
    report = {
        "legacy_manifest": str(input_manifest),
        "manifest_rows": len(legacy_rows),
        "prior_large_rows_by_split": dict(Counter(str(row.get("split")) for row in legacy_rows)),
        "reusable_verified_unique_crops": len(usable),
        "reusable_by_dataset_label": {ds: dict(labels) for ds, labels in counts.items()},
        "excluded_source_not_v2_train": dict(excluded),
        "invalid_or_duplicate": dict(invalid),
        "hash_verification": "full crop SHA-256 compared with legacy input manifest" if verify_hashes else "not requested",
        "verification_workers": max(1, workers),
    }
    return usable, report


def iter_completed_v1_crops(
    state_root: Path, release_rows: Sequence[Mapping[str, str]], *, verify_hashes: bool = True
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Recover intact per-video outputs without changing the interrupted v1 tree."""
    lookup = {(r["dataset"], r["relative_path"].replace("\\", "/").casefold()): dict(r) for r in release_rows}
    state_root = Path(state_root)
    video_state = state_root / ".state" / "videos"
    result: list[dict[str, Any]] = []
    counts: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    rejected: dict[str, int] = defaultdict(int)
    seen: set[tuple[str, str, int]] = set()
    for directory in video_state.iterdir():
        if directory.is_symlink() or not directory.is_dir():
            continue
        complete = directory / "complete.json"
        if not complete.is_file():
            continue
        try:
            meta = json.loads(complete.read_text(encoding="utf-8"))
            config = meta["config"]
            provenance = config["provenance"]
            key = (provenance["dataset"], provenance["relative_path"].replace("\\", "/").casefold())
            current = lookup.get(key)
            if current is None:
                rejected["source_not_in_release"] += len(meta.get("frames", []))
                continue
            if current["project_partition"] != "train" or provenance.get("project_partition") != "train":
                rejected["partition_changed_or_not_train"] += len(meta.get("frames", []))
                continue
            if current["label"] != str(provenance.get("label")):
                rejected["label_mismatch"] += len(meta.get("frames", []))
                continue
            for frame in meta.get("frames", []):
                path = (directory / frame["file"]).resolve()
                try:
                    if not path.is_file():
                        raise OSError("missing_frame")
                    digest = sha256_file(path) if verify_hashes else ""
                    width, height, _ = _image_info(path)
                    if (width, height) != (299, 299):
                        raise ValueError("frame_dimensions_not_299")
                    frame_key = (key[0], current["sha256"], int(frame["frame_index"]))
                    if frame_key in seen:
                        rejected["duplicate_source_frame"] += 1
                        continue
                    seen.add(frame_key)
                    result.append({
                        "input_path": str(path), "input_sha256": digest,
                        "dataset": current["dataset"], "method": current["method"],
                        "label": current["label"], "partition": "train",
                        "source_path": current["absolute_path"], "source_sha256": current["sha256"],
                        "source_group_id": current["source_group_id"],
                        "frame_index": int(frame["frame_index"]), "media_type": "video_crop",
                        "preprocessing_profile": "v2-mtcnn-quality-square299-margin1.5",
                        "preprocessing_compatible": True,
                    })
                    counts[current["dataset"]][current["label"]] += 1
                except (OSError, ValueError) as exc:
                    rejected[str(exc)] += 1
        except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError):
            rejected["invalid_checkpoint"] += 1
    return result, {
        "complete_video_checkpoints_seen": sum(1 for p in video_state.iterdir() if p.is_dir() and (p / "complete.json").is_file()),
        "reusable_verified_unique_crops": len(result),
        "reusable_by_dataset_label": {ds: dict(labels) for ds, labels in counts.items()},
        "rejected_frames": dict(rejected),
        "hash_verification": "full output SHA-256 computed" if verify_hashes else "not requested",
    }


class XceptionTrainingImageDataset:
    """Lazy RGB loader using the previous Clean model's 299/.5 normalization."""
    def __init__(self, rows: Sequence[Mapping[str, Any]], *, size: int = 299, return_metadata: bool = False):
        self.rows = list(rows)
        self.size = size
        self.return_metadata = return_metadata
        if size != 299:
            raise ValueError("Xception Clean requires 299x299 input")
        for row in self.rows:
            if row.get("partition") != "train" or str(row.get("label")) not in {"0", "1"}:
                raise ValueError("Training dataset accepts only verified train rows")

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int):
        import numpy as np
        import torch
        row = self.rows[index]
        with Image.open(row["input_path"]) as image:
            frame_index = int(row.get("gif_frame_index", 0))
            if getattr(image, "is_animated", False):
                image.seek(frame_index)
            rgb = ImageOps.exif_transpose(image).convert("RGB")
            rgb = rgb.resize((self.size, self.size), Image.Resampling.BICUBIC)
            array = np.asarray(rgb, dtype=np.float32) / 255.0
        tensor = torch.from_numpy(array.transpose(2, 0, 1).copy())
        tensor = (tensor - 0.5) / 0.5
        target = torch.tensor(float(row["label"]), dtype=torch.float32)
        return (tensor, target, row) if self.return_metadata else (tensor, target)


class GroupBalancedBatchSampler:
    """Deterministic class/dataset/method/source-group aware batch sampler."""
    def __init__(self, rows: Sequence[Mapping[str, Any]], batch_size: int = 8, seed: int = 2026,
                 dataset_weights: Mapping[str, Mapping[str, float]] | None = None):
        if batch_size < 2 or batch_size % 2:
            raise ValueError("batch_size must be even and at least 2")
        self.rows = list(rows)
        self.batch_size = batch_size
        self.seed = seed
        self.epoch = 0
        self.weights = dataset_weights or {
            "0": {"df40": .50, "ffpp": .25, "celebdf": .25},
            "1": {"df40": .40, "ffpp": .20, "celebdf": .20, "ffhq": .20},
        }
        self.index: dict[str, dict[str, dict[str, dict[str, list[int]]]]] = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(list))))
        for i, row in enumerate(self.rows):
            label = str(row.get("label")); dataset = str(row.get("dataset")); group = str(row.get("source_group_id") or "")
            if label not in {"0", "1"} or row.get("partition") != "train" or not group:
                raise ValueError("Sampler requires verified labels, train partition and source groups")
            self.index[label][dataset][str(row.get("method") or "original")][group].append(i)
        for label in ("0", "1"):
            if label not in self.index:
                raise ValueError(f"No examples for class label {label}")
        self._groups = {
            label: sum(len(groups) for methods in datasets.values() for groups in methods.values())
            for label, datasets in self.index.items()
        }
        if min(self._groups.values()) < batch_size // 2:
            raise ValueError("Not enough distinct source groups for a class-balanced batch")

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __len__(self) -> int:
        return max(1, len(self.rows) // self.batch_size)

    def __iter__(self) -> Iterator[list[int]]:
        rng = random.Random(self.seed + self.epoch)
        per_class = self.batch_size // 2
        for _ in range(len(self)):
            batch: list[int] = []
            chosen_groups: set[tuple[str, str]] = set()
            chosen_content: set[str] = set()
            for label in ("0", "1"):
                for dataset in self._weighted_choices(self.weights.get(label, {}), per_class, rng):
                    def usable_indices(indices):
                        return [i for i in indices if not self.rows[i].get("content_group_id")
                                or self.rows[i]["content_group_id"] not in chosen_content]

                    methods = self.index[label].get(dataset, {})
                    available = {
                        method: [(group, usable_indices(indices)) for group, indices in groups.items()
                                 if (dataset, group) not in chosen_groups and usable_indices(indices)]
                        for method, groups in methods.items()
                    }
                    available = {method: groups for method, groups in available.items() if groups}
                    if not available:
                        # A configured source may be absent in a small/custom
                        # manifest; fall back to a present dataset, then sample
                        # its method uniformly before choosing a source group.
                        alternatives = []
                        for alt_dataset, alt_methods in self.index[label].items():
                            eligible_methods = {}
                            for alt_method, groups in alt_methods.items():
                                eligible_groups = [(g, usable_indices(ix)) for g, ix in groups.items()
                                                   if (alt_dataset, g) not in chosen_groups and usable_indices(ix)]
                                if eligible_groups:
                                    eligible_methods[alt_method] = eligible_groups
                            if eligible_methods:
                                alternatives.append((alt_dataset, eligible_methods))
                        if not alternatives:
                            raise RuntimeError("Unable to draw distinct source groups for batch")
                        dataset, available = rng.choice(alternatives)
                    selected_method = rng.choice(list(available))
                    group, sample_indices = rng.choice(available[selected_method])
                    chosen_groups.add((dataset, group))
                    sample_index = rng.choice(sample_indices)
                    batch.append(sample_index)
                    content_id = str(self.rows[sample_index].get("content_group_id", ""))
                    if content_id:
                        chosen_content.add(content_id)
            rng.shuffle(batch)
            yield batch

    @staticmethod
    def _weighted_choices(weights: Mapping[str, float], count: int, rng: random.Random) -> list[str]:
        options = [(k, float(v)) for k, v in weights.items() if v > 0]
        if not options:
            raise ValueError("No positive dataset sampling weights")
        names, values = zip(*options)
        return rng.choices(names, weights=values, k=count)


def load_clean_xception(checkpoint: Path, metadata_path: Path, *, device: str = "cpu"):
    """Load the existing Clean model strictly; never writes checkpoint files."""
    import torch
    import timm
    from safetensors.torch import load_file
    metadata = json.loads(Path(metadata_path).read_text(encoding="utf-8"))
    actual = sha256_file(checkpoint)
    if actual.lower() != str(metadata.get("weights_sha256", "")).lower():
        raise ValueError("Clean checkpoint digest does not match its immutable metadata")
    if metadata.get("class_map") != {"fake": 0, "real": 1}:
        raise ValueError("Clean checkpoint label mapping is incompatible")
    model = timm.create_model("legacy_xception.tf_in1k", pretrained=False, num_classes=1, drop_rate=0.2)
    state = load_file(str(checkpoint), device="cpu")
    model.load_state_dict(state, strict=True)
    return model.to(device), metadata

