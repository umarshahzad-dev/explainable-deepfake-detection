"""Read-only dataset inventory and conservative DF40 metadata reconciliation."""
from __future__ import annotations
import argparse
import csv
import json
import os
import sqlite3
from collections import Counter, defaultdict
from pathlib import Path

VIDEO_EXT = {".mp4", ".avi", ".mov", ".mkv", ".webm"}
IMAGE_EXT = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}
MEDIA_EXT = VIDEO_EXT | IMAGE_EXT | {".gif"}
FIELDS = ["dataset", "release", "method", "source_domain", "relative_path", "absolute_path", "media_type", "label", "original_json_label", "original_partition", "project_partition", "sample_id", "group_id", "group_status", "relationship_status", "validation_status", "metadata_sources", "sha256", "metadata_status", "experiment_eligibility", "eligibility_reason"]
EXCLUDED_DF40_METHODS = {"sadtalker", "wav2lip"}
DF40_METHOD_ALIASES = {"whichfaceisreal": ("whichisreal",)}
APPLEDOUBLE_PREFIX = "._"
APPLE_METADATA_NAMES = {".ds_store", "thumbs.db"}


def iter_all_files(root: Path, include_macosx: bool = False):
    """Walk the explicitly selected root, not nested directory junctions."""
    def onerror(error):
        raise error
    for folder, dirs, names in os.walk(root, followlinks=False, onerror=onerror):
        dirs[:] = sorted(d for d in dirs if (include_macosx or d.casefold() != "__macosx") and not (Path(folder) / d).is_symlink() and not getattr((Path(folder) / d).lstat(), "st_file_attributes", 0) & 0x400)
        for name in sorted(names):
            # os.walk does not recurse through file entries; reparse-point
            # directories are filtered above. Avoid an extra lstat per file.
            yield Path(folder) / name


def identify(dataset: str, relative: Path):
    parts = relative.parts
    low = [p.casefold() for p in parts]
    if dataset == "ffpp" and len(parts) >= 4:
        if low[0] == "original_sequences":
            return "original", parts[1], 1, None
        if low[0] == "manipulated_sequences":
            return parts[1], parts[2], 0, None
    if dataset == "celebdf" and len(parts) >= 2 and low[0] in {"celeb-real", "youtube-real", "celeb-synthesis"}:
        return ("celeb-synthesis" if low[0] == "celeb-synthesis" else "original", parts[0], 0 if low[0] == "celeb-synthesis" else 1, None)
    if dataset == "df40" and len(parts) >= 2:
        domain = next((x for x in low if x in {"cdf", "ff"}), "unknown")
        if domain == "unknown" and any("celeb-real" in x or "celeb-df" in x for x in low): domain = "cdf"
        if domain == "unknown" and any("ff++" in x or "faceforensics" in x for x in low): domain = "ff"
        return parts[0], domain, None, None
    return None


def metadata_files(root: Path, methods: set[str]):
    """Choose domain-specific files over same-method combined and aggregate files."""
    files = {p.stem.casefold(): p for p in root.glob("*.json")}
    selected = []
    for method in sorted(methods, key=str.casefold):
        key = method.casefold()
        aliases = (key, *DF40_METHOD_ALIASES.get(key, ()))
        variants = [files[k] for alias in aliases for k in (alias + "_cdf", alias + "_ff") if k in files]
        direct = [files[alias] for alias in aliases if alias in files]
        selected.extend(variants or direct[:1])
    return selected


def metadata_records(path: Path):
    data = json.loads(path.read_text(encoding="utf-8-sig"))
    for collection, classes in data.items():
        for class_name, partitions in classes.items():
            for partition, samples in partitions.items():
                for sample, record in samples.items():
                    for frame in record.get("frames", []):
                        label = record.get("label", "")
                        suffix = label.casefold().rsplit("_", 1)[-1]
                        yield {"path": frame.replace("\\", "/"), "label": 1 if suffix == "real" else 0 if suffix == "fake" else None, "original_json_label": label, "original_partition": partition, "sample_id": sample, "collection": collection, "class_name": class_name, "metadata_source": path.name}


def canonical_df40(path: str):
    parts = path.replace("\\", "/").split("/")
    for i, part in enumerate(parts):
        if part.casefold() == "df40" and i + 1 < len(parts):
            return _normalize_df40_key(parts[i + 1:])
    return None


def _normalize_df40_key(parts):
    """Normalize official and local method path spellings to one relative key."""
    lowered = [str(part).strip().replace("\\", "/").casefold() for part in parts if str(part)]
    if len(lowered) > 1 and lowered[0] == lowered[1]:
        lowered.pop(1)
    elif len(lowered) > 1 and lowered[0] == "heygen" and lowered[1] == "heygen_new":
        lowered.pop(1)
    if lowered and lowered[0] == "whichisreal":
        lowered[0] = "whichfaceisreal"
    return "/".join(lowered)


def disk_key(relative: Path):
    parts = list(relative.parts)
    lowered = [p.casefold() for p in parts]
    # Some archives contain an embedded original absolute-path tree.
    for index in range(1, len(parts) - 1):
        if lowered[index] == "df40" and lowered[index + 1] == lowered[0] and "deepfakes_detection_datasets" in lowered[:index]:
            parts = parts[index + 1:]
            break
    return _normalize_df40_key(parts)


def _safe_directory_exists(directory: Path, base: Path, cache: dict[Path, bool]) -> bool:
    """Check candidate alias paths without traversing Windows reparse points."""
    try:
        relative = directory.relative_to(base)
    except ValueError:
        return False
    current = base
    for part in relative.parts:
        current = current / part
        if current not in cache:
            try:
                info = current.lstat()
                cache[current] = not current.is_symlink() and not (getattr(info, "st_file_attributes", 0) & 0x400) and current.is_dir()
            except OSError:
                cache[current] = False
        if not cache[current]:
            return False
    return True


def _df40_alias_paths(relative: Path, base: Path, key: str) -> set[Path]:
    """Return alternate on-disk layouts which could map to the same JSON path."""
    parts = relative.parts
    candidates = set()
    if len(parts) > 1 and (parts[0].casefold() == parts[1].casefold() or (parts[0].casefold() == "heygen" and parts[1].casefold() == "heygen_new")):
        candidates.add(base.joinpath(parts[0], *parts[2:]))
    if parts and parts[0].casefold() == "whichfaceisreal":
        rest = parts[2:] if len(parts) > 1 and parts[0].casefold() == parts[1].casefold() else parts[1:]
        candidates.add(base.joinpath("whichisreal", *rest))
    candidates.add(base.joinpath(*key.split("/")))
    candidates.discard(base / relative)
    return candidates


def _row_eligibility(dataset: str, method: str, label: str, metadata_status: str):
    """Report label-level eligibility only; this does not certify split safety."""
    if dataset == "df40" and method.casefold() in EXCLUDED_DF40_METHODS:
        return "excluded_by_experiment_choice", "method explicitly excluded from planned v2 experiment"
    if label not in {"0", "1"}:
        return "unresolved_label", "verified binary label is unavailable or conflicting"
    if dataset == "df40" and metadata_status not in {"matched", "partition_overlap"}:
        return "needs_metadata_review", f"DF40 metadata status is {metadata_status or 'missing'}"
    return "eligible_labeled_record", "binary label is supported by source metadata or dataset layout convention"


def validation_markdown(report: dict) -> str:
    """Render the concise human-readable companion to the machine report."""
    lines = [
        "# Dataset Inventory Validation",
        "",
        f"- Complete scan: **{'yes' if not report.get('scan_limited') and not report.get('errors') else 'no'}**",
        f"- Manifest SHA-256: `{report.get('manifest_sha256', '')}`",
        "- Labels: `0 = fake`, `1 = real`; unknown labels remain unresolved.",
        "- Eligibility means label/method eligibility only. It does not certify source independence or split safety.",
        f"- AppleDouble/OS metadata files excluded from media rows: {report.get('apple_metadata_files_excluded', 0):,}",
        f"- Files under `__MACOSX` excluded from media rows: {report.get('apple_metadata_tree_files_excluded', 0):,}; their extension counts are retained separately.",
        f"- Official metadata keys appearing in multiple partitions: {report.get('metadata_partition_overlap_keys', 0):,}; these are preserved in `original_partition` as provenance.",
        f"- Unmatched local metadata paths: {report.get('unmatched_metadata_paths', 0):,}",
        "",
        "## Method, label, media type, eligibility",
        "",
        "| Dataset | Method | Label | Media type | Eligibility | Records |",
        "|---|---|---:|---|---|---:|",
    ]
    for composite, count in sorted(report.get("method_label_media_eligibility", {}).items()):
        dataset, method, label, media_type, eligibility = composite.split("|", 4)
        lines.append(f"| {dataset} | {method} | {label} | {media_type} | {eligibility} | {count:,} |")
    lines.extend(["", "## Auxiliary files by extension", "", "| Dataset | Extension | Files |", "|---|---|---:|"])
    for dataset, extensions in sorted(report.get("auxiliary_extensions", {}).items()):
        for extension, count in sorted(extensions.items()):
            lines.append(f"| {dataset} | `{extension}` | {count:,} |")
    lines.extend(["", "## Issues and scope", ""])
    for error in report.get("errors", []):
        lines.append(f"- ERROR: {error}")
    for limitation in report.get("limitations", []):
        lines.append(f"- {limitation}")
    return "\n".join(lines) + "\n"


def ffpp_source_groups(base: Path):
    """Connected FF++ YouTube source pairs, per the official filename convention."""
    import re
    original = base / "original_sequences" / "youtube"
    ids = {p.stem for p in iter_all_files(original) if p.suffix.casefold() == ".mp4" and re.fullmatch(r"\d{3}", p.stem)} if original.exists() else set()
    parent = {x: x for x in ids}
    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]; x = parent[x]
        return x
    for method in ("Deepfakes", "Face2Face", "FaceSwap", "FaceShifter", "NeuralTextures"):
        folder = base / "manipulated_sequences" / method
        if not folder.exists(): continue
        for p in iter_all_files(folder):
            if p.suffix.casefold() != ".mp4":
                continue
            match = re.fullmatch(r"(\d{3})_(\d{3})", p.stem)
            if match and all(x in ids for x in match.groups()):
                a,b=map(find,match.groups()); parent[max(a,b)] = min(a,b)
    return {x:find(x) for x in ids}


def ffpp_group(relative: Path, groups):
    import re
    parts=relative.parts
    if len(parts)<4: return None
    if parts[0].casefold()=="original_sequences" and parts[1].casefold()=="youtube" and relative.stem in groups:
        return "ffpp:youtube:"+groups[relative.stem]
    if parts[0].casefold()=="manipulated_sequences" and parts[1].casefold() in {"deepfakes","face2face","faceswap","faceshifter","neuraltextures"}:
        match=re.fullmatch(r"(\d{3})_(\d{3})",relative.stem)
        if match and all(x in groups for x in match.groups()) and groups[match[1]]==groups[match[2]]:
            return "ffpp:youtube:"+groups[match[1]]
    return None

def generate(config_path: Path, limit: int | None = None, output_dir: Path | str | None = None):
    if limit is not None and limit < 1:
        raise ValueError("limit must be positive")
    config_path = config_path.resolve()
    config = json.loads(config_path.read_text(encoding="utf-8-sig"))
    project = config_path.parent.parent
    raw = project / config["raw_root"]
    out = Path(output_dir) if output_dir is not None else project / config["output_dir"]
    if not out.is_absolute():
        out = project / out
    out.mkdir(parents=True, exist_ok=True)
    settings = config["datasets"]
    dfroot = raw / settings["df40"]["path"]
    metadata_root = raw / settings["df40"].get("metadata_path", "df40/metadata/dataset_json")
    methods = {p.name for p in dfroot.iterdir() if p.is_dir() and not p.is_symlink() and not (getattr(p.lstat(), "st_file_attributes", 0) & 0x400) and p.name.casefold() != "__macosx"} if dfroot.is_dir() else set()
    selected = metadata_files(metadata_root, methods)
    # SQLite keeps the multi-million-reference index on disk instead of retaining
    # a Python object graph for every source record. Input JSON is loaded one file at a time.
    database = sqlite3.connect("")
    database.execute("PRAGMA cache_size=-16384")
    database.execute("CREATE TABLE records (key TEXT NOT NULL, item TEXT NOT NULL, partition_name TEXT, label INTEGER, sample_id TEXT)")
    database.execute("CREATE TABLE matched (key TEXT PRIMARY KEY)")
    external = Counter(); batch = []
    for source in selected:
        for item in metadata_records(source):
            key = canonical_df40(item["path"])
            if key is None:
                external[item["path"].split("/")[1] if "/" in item["path"] else "unknown"] += 1
            else:
                batch.append((key, json.dumps(item, separators=(",", ":")), item["original_partition"], item["label"], item["sample_id"]))
                if len(batch) >= 10000:
                    database.executemany("INSERT INTO records VALUES (?, ?, ?, ?, ?)", batch); batch.clear()
    if batch: database.executemany("INSERT INTO records VALUES (?, ?, ?, ?, ?)", batch)
    database.execute("CREATE INDEX records_key ON records(key)")
    database.commit()
    counts, extensions, method_counts = Counter(), defaultdict(Counter), defaultdict(Counter)
    errors, examples = [], defaultdict(list)
    method_label_media_eligibility = Counter()
    auxiliary_extensions = defaultdict(Counter)
    apple_metadata_tree_extensions = defaultdict(Counter)
    apple_metadata_files = 0
    apple_metadata_tree_files = 0
    safe_directory_cache: dict[Path, bool] = {}
    ff_groups = ffpp_source_groups(raw / settings.get("ffpp", {}).get("path", "__missing_ffpp__")) if "ffpp" in settings else {}
    celeb_test = {}
    celeb_root = raw / settings.get("celebdf", {}).get("path", "")
    listing = celeb_root / "List_of_testing_videos.txt"
    if listing.is_file():
        for line in listing.read_text(encoding="utf-8-sig").splitlines():
            fields = line.split(maxsplit=1)
            if len(fields) == 2: celeb_test[fields[1].replace("\\", "/").casefold()] = fields[0]
    manifest = out / "manifest.csv"
    with manifest.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS); writer.writeheader()
        for dataset, opts in settings.items():
            if not opts.get("enabled", True): continue
            base = raw / opts["path"]
            if not base.is_dir():
                errors.append("Missing root: " + str(base)); continue
            indexed = 0
            try:
                for path in iter_all_files(base, include_macosx=True):
                    ext = path.suffix.casefold() or "<none>"
                    extensions[dataset][ext] += 1
                    relative = path.relative_to(base)
                    in_apple_tree = any(part.casefold() == "__macosx" for part in relative.parts)
                    if in_apple_tree:
                        apple_metadata_tree_files += 1
                        apple_metadata_tree_extensions[dataset][ext] += 1
                        auxiliary_extensions[dataset][ext] += 1
                        continue
                    if path.name.casefold() in APPLE_METADATA_NAMES or path.name.startswith(APPLEDOUBLE_PREFIX):
                        apple_metadata_files += 1
                        auxiliary_extensions[dataset][ext] += 1
                        continue
                    if ext == ".npy" or ext in {".zip", ".7z", ".rar", ".tar", ".gz"}:
                        auxiliary_extensions[dataset][ext] += 1
                    if ext not in MEDIA_EXT: continue
                    if limit is not None and indexed >= limit:
                        counts["limited:" + dataset] += 1; break
                    info = identify(dataset, relative)
                    if info is None:
                        counts["unrecognized_media:" + dataset] += 1
                        if len(examples["unrecognized"]) < 20: examples["unrecognized"].append(str(relative))
                        continue
                    method, domain, label, _ = info
                    row = dict.fromkeys(FIELDS, "")
                    row.update(metadata_status="path_rule" if dataset != "df40" else "unlisted", dataset=dataset, release={"ffpp":"FaceForensics++", "celebdf":"Celeb-DF-v2", "df40":"DF40 official testing-data release"}.get(dataset, dataset), method=method, source_domain=domain, relative_path=relative.as_posix(), absolute_path=str(path.absolute()), media_type="video" if ext in VIDEO_EXT else "gif" if ext == ".gif" else "image", label="" if label is None else str(label), sample_id=relative.stem, group_status="unresolved", validation_status="not_decoded")
                    if dataset in {"ffpp", "celebdf"}:
                        row["group_id"] = f"{dataset}:{relative.stem}"
                        row["group_status"] = "sample_verified"
                        row["relationship_status"] = "suspected"
                    if dataset == "ffpp":
                        group = ffpp_group(relative, ff_groups)
                        if group:
                            row["group_id"] = group; row["group_status"] = "verified"; row["relationship_status"] = "source_link_verified"
                    if dataset == "celebdf" and relative.as_posix().casefold() in celeb_test:
                        row["original_partition"] = "official_test"
                        if celeb_test[relative.as_posix().casefold()] != str(label):
                            row["label"] = ""; counts["label_conflicts"] += 1
                    if dataset == "df40":
                        key = disk_key(relative); entries = [json.loads(item) for (item,) in database.execute("SELECT item FROM records WHERE key=?", (key,))]
                        aliases = _df40_alias_paths(relative, base, key)
                        if any(_safe_directory_exists(alias.parent, base, safe_directory_cache) and alias.is_file() for alias in aliases):
                            entries = []
                            counts["ambiguous_disk_paths"] += 1
                            row["metadata_status"] = "ambiguous"
                        if entries:
                            database.execute("INSERT OR IGNORE INTO matched VALUES (?)", (key,))
                            row["metadata_status"] = "matched"
                            labels = {e["label"] for e in entries}; parts = {e["original_partition"] for e in entries}; samples = {e["sample_id"] for e in entries}
                            resolved_label = next(iter(labels)) if len(labels) == 1 and None not in labels else ""
                            row.update(label="" if resolved_label == "" else str(resolved_label), original_json_label="|".join(sorted({str(e["original_json_label"]) for e in entries})), original_partition="|".join(sorted(parts)), sample_id=next(iter(samples)) if len(samples) == 1 else "", metadata_sources="|".join(sorted({e["metadata_source"] for e in entries})))
                            if len(labels) != 1 or None in labels or len(samples) != 1:
                                counts["metadata_conflicts"] += 1; row["metadata_status"] = "conflict"; row["label"] = ""
                            elif row["sample_id"]:
                                row["group_id"] = f"df40:{method}:{domain}:{row['sample_id']}"
                                row["group_status"] = "sample_verified"
                                row["relationship_status"] = "suspected" if domain in {"cdf", "ff"} else "unknown"
                            if len(parts) > 1:
                                counts["metadata_partition_overlaps"] += 1
                        elif row["metadata_status"] != "ambiguous":
                            counts["df40_unlisted_media"] += 1
                            if len(examples["unlisted"]) < 20: examples["unlisted"].append(relative.as_posix())
                    eligibility, eligibility_reason = _row_eligibility(dataset, method, row["label"], row["metadata_status"])
                    row["experiment_eligibility"] = eligibility
                    row["eligibility_reason"] = eligibility_reason
                    method_label_media_eligibility[(dataset, method, row["label"] if row["label"] in {"0", "1"} else "unknown", row["media_type"], eligibility)] += 1
                    writer.writerow(row); indexed += 1; counts["indexed:" + dataset] += 1
                    counts[f'label:{dataset}:{row["label"] if row["label"] != "" else "unknown"}'] += 1
                    method_counts[dataset][method] += 1
            except OSError as exc:
                errors.append(f"{dataset}: {exc}")
    unmatched_count = database.execute("SELECT COUNT(DISTINCT r.key) FROM records r LEFT JOIN matched m ON r.key=m.key WHERE m.key IS NULL").fetchone()[0]
    unmatched_examples = [r[0] for r in database.execute("SELECT DISTINCT r.key FROM records r LEFT JOIN matched m ON r.key=m.key WHERE m.key IS NULL ORDER BY r.key LIMIT 20")]
    duplicate_rows = database.execute("SELECT COALESCE(SUM(n - 1), 0) FROM (SELECT COUNT(*) AS n FROM records GROUP BY key, item HAVING COUNT(*) > 1)").fetchone()[0]
    duplicate_path_keys = database.execute("SELECT COUNT(*) FROM (SELECT key FROM records GROUP BY key HAVING COUNT(*) > 1)").fetchone()[0]
    partition_overlap_count = database.execute("SELECT COUNT(*) FROM (SELECT key FROM records GROUP BY key HAVING COUNT(DISTINCT partition_name) > 1)").fetchone()[0]
    archive = raw / settings["df40"].get("archives_path", "df40/archives")
    archive_counts = Counter(p.suffix.casefold() or "<none>" for p in iter_all_files(archive)) if archive.is_dir() else Counter()
    report = {"counts": dict(sorted(counts.items())), "discovered_extensions": {k: dict(sorted(v.items())) for k,v in extensions.items()}, "auxiliary_extensions": {k: dict(sorted(v.items())) for k,v in auxiliary_extensions.items()}, "apple_metadata_files_excluded": apple_metadata_files, "apple_metadata_tree_files_excluded": apple_metadata_tree_files, "apple_metadata_tree_extensions": {k: dict(sorted(v.items())) for k,v in apple_metadata_tree_extensions.items()}, "archive_extensions": dict(archive_counts), "method_counts": {k:dict(v) for k,v in method_counts.items()}, "method_label_media_eligibility": {"|".join(k): v for k,v in sorted(method_label_media_eligibility.items())}, "metadata_selected": [p.name for p in selected], "methods_without_named_metadata": sorted(m for m in methods if not metadata_files(metadata_root, {m})), "metadata_duplicate_rows": duplicate_rows, "metadata_duplicate_path_keys": duplicate_path_keys, "metadata_partition_overlap_keys": partition_overlap_count, "unmatched_metadata_paths": unmatched_count, "unmatched_metadata_examples": unmatched_examples, "external_reference_records": dict(external), "examples": dict(examples), "errors": errors, "scan_limited": limit is not None, "limitations": ["No media decoded; media integrity/decoding is not asserted", "Group identities and cross-dataset independence are unverified", "Unmatched counts are incomplete for a limited scan", "Metadata outside DF40 is counted as external reference, not a missing local DF40 file", "experiment_eligibility is label/method eligibility only and does not certify split or leakage safety"]}
    database.close()
    from .split import file_sha256
    report["manifest_sha256"] = file_sha256(manifest)
    (out / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    (out / "report.md").write_text(validation_markdown(report), encoding="utf-8")
    return manifest, report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/datasets.json")
    parser.add_argument("--limit-per-dataset", type=int)
    parser.add_argument("--output-dir", type=Path, help="Write this inventory to a new output directory instead of the configured historical inventory directory")
    args = parser.parse_args()
    manifest, report = generate(Path(args.config), args.limit_per_dataset, args.output_dir)
    print("Manifest:", manifest); print(json.dumps(report, indent=2))

if __name__ == "__main__": main()
