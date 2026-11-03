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
FIELDS = ["dataset", "release", "method", "source_domain", "relative_path", "absolute_path", "media_type", "label", "original_json_label", "original_partition", "project_partition", "sample_id", "group_id", "group_status", "relationship_status", "validation_status", "metadata_sources", "sha256", "metadata_status"]


def iter_all_files(root: Path):
    """Walk the explicitly selected root, not nested directory junctions."""
    def onerror(error):
        raise error
    for folder, dirs, names in os.walk(root, followlinks=False, onerror=onerror):
        dirs[:] = sorted(d for d in dirs if d.casefold() != "__macosx" and not (Path(folder) / d).is_symlink() and not getattr((Path(folder) / d).lstat(), "st_file_attributes", 0) & 0x400)
        for name in sorted(names):
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
        variants = [files[k] for k in (key + "_cdf", key + "_ff") if k in files]
        selected.extend(variants or ([files[key]] if key in files else []))
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
            return "/".join(parts[i + 1:]).casefold()
    return None


def disk_key(relative: Path):
    parts = list(relative.parts)
    lowered = [p.casefold() for p in parts]
    # Some archives contain an embedded original absolute-path tree.
    for index in range(1, len(parts) - 1):
        if lowered[index] == "df40" and lowered[index + 1] == lowered[0] and "deepfakes_detection_datasets" in lowered[:index]:
            parts = parts[index + 1:]
            break
    if len(parts) > 1 and parts[0].casefold() == parts[1].casefold():
        parts.pop(1)
    return "/".join(parts).casefold()


def ffpp_source_groups(base: Path):
    """Connected FF++ YouTube source pairs, per the official filename convention."""
    import re
    original = base / "original_sequences" / "youtube"
    ids = {p.stem for p in original.rglob("*.mp4") if re.fullmatch(r"\d{3}", p.stem)} if original.exists() else set()
    parent = {x: x for x in ids}
    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]; x = parent[x]
        return x
    for method in ("Deepfakes", "Face2Face", "FaceSwap", "FaceShifter", "NeuralTextures"):
        folder = base / "manipulated_sequences" / method
        if not folder.exists(): continue
        for p in folder.rglob("*.mp4"):
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

def generate(config_path: Path, limit: int | None = None):
    if limit is not None and limit < 1:
        raise ValueError("limit must be positive")
    config_path = config_path.resolve()
    config = json.loads(config_path.read_text(encoding="utf-8-sig"))
    project = config_path.parent.parent
    raw = project / config["raw_root"]
    out = project / config["output_dir"]
    out.mkdir(parents=True, exist_ok=True)
    settings = config["datasets"]
    dfroot = raw / settings["df40"]["path"]
    metadata_root = raw / settings["df40"].get("metadata_path", "df40/metadata/dataset_json")
    methods = {p.name for p in dfroot.iterdir() if p.is_dir()} if dfroot.is_dir() else set()
    selected = metadata_files(metadata_root, methods)
    # SQLite keeps the multi-million-reference index on disk instead of retaining
    # a Python object graph for every source record. Input JSON is loaded one file at a time.
    database = sqlite3.connect("")
    database.execute("PRAGMA cache_size=-16384")
    database.execute("CREATE TABLE records (key TEXT NOT NULL, item TEXT NOT NULL)")
    external = Counter(); batch = []
    for source in selected:
        for item in metadata_records(source):
            key = canonical_df40(item["path"])
            if key is None:
                external[item["path"].split("/")[1] if "/" in item["path"] else "unknown"] += 1
            else:
                batch.append((key, json.dumps(item, separators=(",", ":"))))
                if len(batch) >= 10000:
                    database.executemany("INSERT INTO records VALUES (?, ?)", batch); batch.clear()
    if batch: database.executemany("INSERT INTO records VALUES (?, ?)", batch)
    database.execute("CREATE INDEX records_key ON records(key)")
    database.commit()
    counts, extensions, method_counts = Counter(), defaultdict(Counter), defaultdict(Counter)
    matched, errors, examples = set(), [], defaultdict(list)
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
                for path in iter_all_files(base):
                    ext = path.suffix.casefold() or "<none>"
                    extensions[dataset][ext] += 1
                    if ext not in MEDIA_EXT: continue
                    if limit is not None and indexed >= limit:
                        counts["limited:" + dataset] += 1; break
                    relative = path.relative_to(base); info = identify(dataset, relative)
                    if info is None:
                        counts["unrecognized_media:" + dataset] += 1
                        if len(examples["unrecognized"]) < 20: examples["unrecognized"].append(str(relative))
                        continue
                    method, domain, label, _ = info
                    row = dict.fromkeys(FIELDS, "")
                    row.update(metadata_status="path_rule" if dataset != "df40" else "unlisted", dataset=dataset, release={"ffpp":"FaceForensics++", "celebdf":"Celeb-DF-v2", "df40":"DF40 official testing-data release"}.get(dataset, dataset), method=method, source_domain=domain, relative_path=relative.as_posix(), absolute_path=str(path.absolute()), media_type="video" if ext in VIDEO_EXT else "gif" if ext == ".gif" else "image", label="" if label is None else label, sample_id=relative.stem, group_status="unresolved", validation_status="not_decoded")
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
                        parts = relative.parts
                        alternate = base.joinpath(parts[0], *parts[2:]) if len(parts) > 1 and parts[0].casefold() == parts[1].casefold() else base.joinpath(parts[0], parts[0], *parts[1:])
                        canonical = base.joinpath(*key.split("/"))
                        if alternate.is_file() or (canonical != path and canonical.is_file()):
                            entries = []
                            counts["ambiguous_disk_paths"] += 1
                            row["metadata_status"] = "ambiguous"
                        if entries:
                            matched.add(key)
                            row["metadata_status"] = "matched"
                            labels = {e["label"] for e in entries}; parts = {e["original_partition"] for e in entries}; samples = {e["sample_id"] for e in entries}
                            row.update(label=next(iter(labels)) if len(labels) == 1 and None not in labels else "", original_json_label="|".join(sorted({str(e["original_json_label"]) for e in entries})), original_partition="|".join(sorted(parts)), sample_id=next(iter(samples)) if len(samples) == 1 else "", metadata_sources="|".join(sorted({e["metadata_source"] for e in entries})))
                            if len(labels) != 1 or None in labels or len(samples) != 1:
                                counts["metadata_conflicts"] += 1; row["metadata_status"] = "conflict"; row["label"] = ""
                            elif row["sample_id"]:
                                row["group_id"] = f"df40:{method}:{domain}:{row['sample_id']}"
                                row["group_status"] = "sample_verified"
                                row["relationship_status"] = "suspected" if domain in {"cdf", "ff"} else "unknown"
                            if len(parts) > 1:
                                counts["metadata_partition_conflicts"] += 1
                                if row["metadata_status"] == "matched": row["metadata_status"] = "partition_conflict"
                        elif row["metadata_status"] != "ambiguous":
                            counts["df40_unlisted_media"] += 1
                            if len(examples["unlisted"]) < 20: examples["unlisted"].append(relative.as_posix())
                    writer.writerow(row); indexed += 1; counts["indexed:" + dataset] += 1
                    counts[f'label:{dataset}:{row["label"] if row["label"] != "" else "unknown"}'] += 1
                    method_counts[dataset][method] += 1
            except OSError as exc:
                errors.append(f"{dataset}: {exc}")
    # Match keys are held as strings only, rather than full JSON records.
    database.execute("CREATE TABLE matched (key TEXT PRIMARY KEY)")
    database.executemany("INSERT OR IGNORE INTO matched VALUES (?)", ((key,) for key in matched))
    unmatched_count = database.execute("SELECT COUNT(DISTINCT r.key) FROM records r LEFT JOIN matched m ON r.key=m.key WHERE m.key IS NULL").fetchone()[0]
    unmatched_examples = [r[0] for r in database.execute("SELECT DISTINCT r.key FROM records r LEFT JOIN matched m ON r.key=m.key WHERE m.key IS NULL ORDER BY r.key LIMIT 20")]
    duplicate_count = database.execute("SELECT COUNT(*)-COUNT(DISTINCT key) FROM records").fetchone()[0]
    archive = raw / settings["df40"].get("archives_path", "df40/archives")
    archive_counts = Counter(p.suffix.casefold() or "<none>" for p in iter_all_files(archive)) if archive.is_dir() else Counter()
    report = {"counts": dict(sorted(counts.items())), "discovered_extensions": {k: dict(sorted(v.items())) for k,v in extensions.items()}, "archive_extensions": dict(archive_counts), "method_counts": {k:dict(v) for k,v in method_counts.items()}, "metadata_selected": [p.name for p in selected], "methods_without_named_metadata": sorted(m for m in methods if not metadata_files(metadata_root, {m})), "metadata_duplicate_keys": duplicate_count, "unmatched_metadata_paths": unmatched_count, "unmatched_metadata_examples": unmatched_examples, "external_reference_records": dict(external), "examples": dict(examples), "errors": errors, "scan_limited": limit is not None, "limitations": ["No media decoded", "Group identities and cross-dataset independence are unverified", "Unmatched counts are incomplete for a limited scan", "Metadata outside DF40 is counted as external reference, not a missing local DF40 file"]}
    database.close()
    from .split import file_sha256
    report["manifest_sha256"] = file_sha256(manifest)
    (out / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return manifest, report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/datasets.json")
    parser.add_argument("--limit-per-dataset", type=int)
    args = parser.parse_args()
    manifest, report = generate(Path(args.config), args.limit_per_dataset)
    print("Manifest:", manifest); print(json.dumps(report, indent=2))

if __name__ == "__main__": main()
