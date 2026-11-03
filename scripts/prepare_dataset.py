"""Prepare the frozen Xception release; --smoke uses an isolated bounded output."""
import argparse
from pathlib import Path
from deepfake_data.prepare import run, CONFIG
from deepfake_data.extract import ExtractionConfig


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--smoke',action='store_true',help='Process only one source from each dataset into artifacts/diagnostics')
    args=parser.parse_args()
    root=Path(__file__).resolve().parents[1]
    output=root/('artifacts/diagnostics/prepare-smoke-v6' if args.smoke else 'data/processed/xception-v1')
    config=ExtractionConfig(max_frames=2,candidates_per_segment=2,min_sharpness=15) if args.smoke else CONFIG
    run(root/'artifacts/combined/releases/best-effort-v1.csv',output,smoke=args.smoke,config=config)


if __name__=='__main__':main()
