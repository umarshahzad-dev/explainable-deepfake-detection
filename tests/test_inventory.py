import csv
import json
from pathlib import Path
from deepfake_data.inventory import generate, identify


def test_ffpp_labels():
    assert identify('ffpp', Path('original_sequences/youtube/c23/videos/001.mp4'))[2] == 1
    assert identify('ffpp', Path('manipulated_sequences/Deepfakes/c23/videos/001_002.mp4'))[2] == 0


def test_celebdf_labels():
    assert identify('celebdf', Path('Celeb-synthesis/id0_id1.mp4'))[2] == 0
    assert identify('celebdf', Path('YouTube-real/0001.mp4'))[2] == 1


def test_df40_explicit_labels_and_uncertain_frames():
    assert identify('df40', Path('MidJourney/MidJourney/fake/example.gif'))[2] == 0
    assert identify('df40', Path('MidJourney/MidJourney/real/example.jpg'))[2] == 1
    assert identify('df40', Path('e4s/cdf/frames/id0_id1/000.png'))[2] is None


def test_df40_cdf_ff_never_implies_partition():
    assert identify('df40', Path('simswap/ff/frames/video/000.png'))[2] is None


def test_gif_coverage_and_auxiliary_exclusions(tmp_path):
    config = tmp_path / 'configs'
    config.mkdir()
    df40 = tmp_path / 'data/raw/df40'
    (df40 / 'MidJourney/MidJourney/fake').mkdir(parents=True)
    (df40 / 'MidJourney/MidJourney/real').mkdir(parents=True)
    (df40 / 'e4s/cdf/frames/clip').mkdir(parents=True)
    (df40 / 'unknown_root').mkdir(parents=True)
    for relative in ['MidJourney/MidJourney/fake/001.gif',
                     'MidJourney/MidJourney/fake/002.png',
                     'MidJourney/MidJourney/real/003.jpg',
                     'e4s/cdf/frames/clip/000.png',
                     'e4s/cdf/frames/clip/001.npy',
                     'unknown_root/data.png']:
        (df40 / relative).touch()
    (config / 'datasets.json').write_text(json.dumps({
        'raw_root': 'data/raw', 'output_dir': 'artifacts/inventory',
        'datasets': {'df40': {'path': 'df40', 'partition': 'unverified'}}}), encoding='utf-8')
    manifest, report = generate(config / 'datasets.json')
    with manifest.open(newline='', encoding='utf-8') as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 5
    assert sum(row['media_type'] == 'animated_image' for row in rows) == 1
    assert report['discovered_extensions']['df40']['.gif'] == 1
    assert report['excluded_nonmedia_extensions']['df40']['.npy'] == 1
    assert all(row['partition'] == 'unverified' for row in rows)
    assert report['counts']['label:df40:unknown'] == 2


def test_missing_root_is_reported(tmp_path):
    config = tmp_path / 'configs'
    config.mkdir()
    (config / 'datasets.json').write_text(json.dumps({
        'raw_root': 'data/raw', 'output_dir': 'artifacts/inventory',
        'datasets': {'df40': {'path': 'missing'}}}), encoding='utf-8')
    _, report = generate(config / 'datasets.json')
    assert report['counts']['missing_root:df40'] == 1
