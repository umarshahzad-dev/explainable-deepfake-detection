"""Read-only inventory of FF++, Celeb-DF and DF40 media.

Inventory counts paths, not unique identities, valid decodes, or independent samples.
Never use partition='unverified' rows for training without provenance review.
"""
import argparse
import csv
import json
import os
from collections import Counter, defaultdict
from pathlib import Path

VIDEO_EXT = {'.mp4', '.avi', '.mov', '.mkv', '.webm'}
IMAGE_EXT = {'.jpg', '.jpeg', '.png', '.webp', '.bmp', '.tif', '.tiff'}
ANIMATED_EXT = {'.gif'}
MEDIA_EXT = VIDEO_EXT | IMAGE_EXT | ANIMATED_EXT
FIELDS = ['dataset', 'method', 'source_subset', 'sample_id', 'path', 'media_type',
          'label', 'group_id', 'partition']


def identify(dataset: str, relative: Path):
    """Conservative path-derived metadata; None means not identifiable."""
    parts = relative.parts
    low = [part.casefold() for part in parts]
    if dataset == 'ffpp':
        if len(parts) >= 3 and low[0] == 'original_sequences':
            return 'original', parts[1], 1, f'ffpp:{relative.stem}'
        if len(parts) >= 4 and low[0] == 'manipulated_sequences':
            return parts[1], parts[2], 0, f'ffpp:{relative.stem}'
        return None
    if dataset == 'celebdf':
        if len(parts) >= 2 and low[0] in {'celeb-real', 'youtube-real', 'celeb-synthesis'}:
            return ('original' if low[0] != 'celeb-synthesis' else 'celeb-synthesis',
                    parts[0], 0 if low[0] == 'celeb-synthesis' else 1,
                    f'celebdf:{parts[0]}:{relative.stem}')
        return None
    if dataset == 'df40':
        if len(parts) < 2 or '__macosx' in low:
            return None
        method = parts[0]
        subset = 'cdf' if 'cdf' in low else ('ff' if 'ff' in low else 'unknown')
        # Only explicit class markers are treated as labels. No implied labels
        # from method names or the cdf/ff folder structure.
        if 'fake' in low or any(part.startswith('fake_from_') for part in low):
            label = 0
        elif 'real' in low:
            label = 1
        else:
            label = None
        # Adjacent frames, including GIFs, belong to a common source group.
        group = relative.parent.as_posix()
        return method, subset, label, f'df40:{group}'
    return None


def iter_all_files(root: Path):
    """Walk the junction root, never follow nested directory links."""
    for folder, dirs, filenames in os.walk(root, followlinks=False):
        dirs[:] = sorted(d for d in dirs if d.casefold() != '__macosx')
        for filename in sorted(filenames):
            yield Path(folder) / filename


def generate(config_path: Path, limit: int | None = None):
    if limit is not None and limit < 1:
        raise ValueError('limit must be positive')
    config_path = config_path.resolve()
    config = json.loads(config_path.read_text(encoding='utf-8-sig'))
    project_root = config_path.parent.parent
    raw_root = project_root / config['raw_root']
    out = project_root / config['output_dir']
    out.mkdir(parents=True, exist_ok=True)
    counts = Counter()
    ext_counts = defaultdict(Counter)
    excluded = defaultdict(Counter)
    missing_examples = []
    unknown_examples = []
    method_counts = defaultdict(Counter)
    manifest = out / 'manifest.csv'
    with manifest.open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS)
        writer.writeheader()
        for dataset, settings in config['datasets'].items():
            if not settings.get('enabled', True):
                continue
            base = raw_root / settings['path']
            if not base.is_dir():
                counts[f'missing_root:{dataset}'] += 1
                missing_examples.append(str(base))
                continue
            indexed_in_dataset = 0
            for path in iter_all_files(base):
                ext = path.suffix.casefold() or '<none>'
                ext_counts[dataset][ext] += 1
                if ext not in MEDIA_EXT:
                    excluded[dataset][ext] += 1
                    continue
                if limit is not None and indexed_in_dataset >= limit:
                    counts[f'limited:{dataset}'] += 1
                    break
                relative = path.relative_to(base)
                result = identify(dataset, relative)
                if result is None:
                    counts[f'unrecognized_media:{dataset}'] += 1
                    if len(unknown_examples) < 30:
                        unknown_examples.append(f'{dataset}: {relative.as_posix()}')
                    continue
                method, subset, label, group = result
                media = ('video' if ext in VIDEO_EXT else
                         'animated_image' if ext in ANIMATED_EXT else 'image')
                writer.writerow(dict(dataset=dataset, method=method,
                                     source_subset=subset, sample_id=relative.as_posix(),
                                     path=str(path.absolute()), media_type=media,
                                     label='' if label is None else label,
                                     group_id=group,
                                     partition=settings.get('partition', 'unassigned')))
                indexed_in_dataset += 1
                counts[f'indexed:{dataset}'] += 1
                counts[f'label:{dataset}:{label if label is not None else "unknown"}'] += 1
                counts[f'media:{dataset}:{media}'] += 1
                method_counts[dataset][method] += 1
    report = {
        'counts': dict(sorted(counts.items())),
        'discovered_extensions': {k: dict(sorted(v.items())) for k, v in sorted(ext_counts.items())},
        'excluded_nonmedia_extensions': {k: dict(sorted(v.items())) for k, v in sorted(excluded.items())},
        'method_counts': {k: dict(sorted(v.items())) for k, v in sorted(method_counts.items())},
        'missing_roots': missing_examples,
        'unrecognized_media_examples': unknown_examples,
        'scan_limited': limit is not None,
        'limitations': [
            'Inventory does not decode media or validate corruption',
            'File paths are not independent samples or verified identities',
            'FF++ and Celeb-DF official splits are not assigned',
            'DF40 training/testing provenance remains unverified',
            'GIFs are inventoried as files; animated frames are not decoded',
            'Non-media auxiliary files (including .npy) are excluded from the manifest',
            'Nested archives are not automatically extracted or inventoried internally',
        ],
    }
    (out / 'report.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    return manifest, report


def main():
    parser = argparse.ArgumentParser(description='Audit and inventory dataset media')
    parser.add_argument('--config', default='configs/datasets.json')
    parser.add_argument('--limit-per-dataset', type=int)
    args = parser.parse_args()
    if args.limit_per_dataset is not None and args.limit_per_dataset < 1:
        parser.error('--limit-per-dataset must be positive')
    manifest, report = generate(Path(args.config), args.limit_per_dataset)
    print('Manifest:', manifest)
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
