from pathlib import Path
from deepfake_data.inventory import identify


def test_ffpp_original():
    assert identify("ffpp", Path("original_sequences/youtube/c23/videos/001.mp4"))[2] == 1


def test_ffpp_fake():
    assert identify("ffpp", Path("manipulated_sequences/Deepfakes/c23/videos/001_002.mp4"))[2] == 0


def test_celebdf():
    assert identify("celebdf", Path("Celeb-synthesis/id0_id1.mp4"))[2] == 0
    assert identify("celebdf", Path("YouTube-real/0001.mp4"))[2] == 1


def test_df40_uncertain_frames():
    assert identify("df40", Path("e4s/cdf/frames/id0_id1/000.png"))[2] is None


def test_df40_explicit_fake():
    assert identify("df40", Path("CollabDiff/CollabDiff/fake/0/000.png"))[2] == 0
