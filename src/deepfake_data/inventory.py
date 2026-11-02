"""Read-only inventory of heterogeneous deepfake datasets.

Labels are assigned only from explicitly recognized source paths.
This inventory does not establish leakage-free splits or validate image contents.
"""
import argparse
import csv
import json
import os
from collections import Counter
from pathlib import Path

VIDEO_EXT = {".mp4", ".avi", ".mov", ".mkv", ".webm"}
IMAGE_EXT = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
FIELDS = ["dataset", "method", "source_subset", "sample_id", "path", "media_type", "label", "group_id", "partition"]


def identify(dataset: str, relative: Path):
    """Return (method, subset, label, group_id) or None for unsupported paths."""
    parts = relative.parts
    low = [x.lower() for x in parts]
    if dataset == "ffpp":
        if len(parts) >= 3 and low[0] == "original_sequences":
            return "original", parts[1], 1, f"ffpp:{relative.stem}"
        if len(parts) >= 4 and low[0] == "manipulated_sequences":
            return parts[1], parts[2], 0, f"ffpp:{relative.stem}"
        return None
    if dataset == "celebdf":
        if len(parts) >= 2 and low[0] in {"celeb-real", "youtube-real", "celeb-synthesis"}:
            return ("original" if low[0] != "celeb-synthesis" else "celeb-synthesis"), parts[0], (0 if low[0] == "celeb-synthesis" else 1), f"celebdf:{parts[0]}:{relative.stem}"
        return None
    if dataset == "df40":
        if len(parts) < 2 or "__macosx" in low:
            return None
        method = parts[0]
        subset = "cdf" if "cdf" in low else ("ff" if "ff" in low else "unknown")
        # A path explicitly under fake/real is labelable; method-specific
        # cdf/ff frames and Fake_from_* are fake by their documented layout.
        if "fake" in low:
            label = 0
        elif "real" in low:
            label = 1
        elif any(x.startswith("fake_from_") for x in low):
            label = 0
        elif "frames" in low or method.lower() in {"midjourney", "starganv2"}:
            label = None  # Requires method-specific provenance validation.
        else:
            label = None
        # Group by containing directory; never treat adjacent frames as independent groups.
        group = relative.parent.as_posix()
        return method, subset, label, f"df40:{group}"
    return None


def iter_files(root: Path):
    # os.walk does not follow directory symlinks by default; root itself can be a junction.
    for folder, dirs, filenames in os.walk(root):
        dirs[:] = sorted(d for d in dirs if d.lower() != "__macosx")
        for filename in sorted(filenames):
            path = Path(folder) / filename
            if path.suffix.lower() in VIDEO_EXT | IMAGE_EXT:
                yield path


def generate(config_path: Path, limit: int | None = None):
    config_path = config_path.resolve()
    config = json.loads(config_path.read_text(encoding="utf-8-sig"))
    project_root = config_path.parent.parent
    raw_root = project_root / config["raw_root"]
    out = project_root / config["output_dir"]
    out.mkdir(parents=True, exist_ok=True)
    counts = Counter()
    examples = []
    manifest = out / "manifest.csv"
    with manifest.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS)
        writer.writeheader()
        for dataset, settings in config["datasets"].items():
            if not settings.get("enabled", True):
                continue
            base = raw_root / settings["path"]
            if not base.is_dir():
                counts[f"missing_root:{dataset}"] += 1
                continue
            seen = 0
            for path in iter_files(base):
                if limit is not None and seen >= limit:
                    break
                seen += 1
                relative = path.relative_to(base)
                result = identify(dataset, relative)
                if result is None:
                    counts[f"unsupported:{dataset}"] += 1
                    if len(examples) < 20:
                        examples.append(f"{dataset}: {relative.as_posix()}")
                    continue
                method, subset, label, group = result
                media = "video" if path.suffix.lower() in VIDEO_EXT else "image"
                writer.writerow(dict(dataset=dataset, method=method, source_subset=subset,
                                     sample_id=relative.as_posix(), path=str(path.absolute()),
                                     media_type=media, label="" if label is None else label,
                                     group_id=group, partition=settings.get("partition", "unassigned")))
                counts[f"indexed:{dataset}"] += 1
                counts[f"label:{dataset}:{label if label is not None else 'unknown'}"] += 1
    report = {"counts": dict(sorted(counts.items())), "unsupported_examples": examples,
              "limitations": ["No media decoding or corruption checks", "No verified identity linkage across datasets", "No train/val/test splits generated", "DF40 provenance and partition not verified"]}
    (out / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return manifest, report


def main():
    parser = argparse.ArgumentParser(description="Generate a read-only dataset inventory")
    parser.add_argument("--config", default="configs/datasets.json")
    parser.add_argument("--limit-per-dataset", type=int, default=None)
    args = parser.parse_args()
    if args.limit_per_dataset is not None and args.limit_per_dataset < 1:
        parser.error("--limit-per-dataset must be positive")
    path, report = generate(Path(args.config), args.limit_per_dataset)
    print(f"Manifest: {path}")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
