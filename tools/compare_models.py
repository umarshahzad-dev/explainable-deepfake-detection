"""Compare all trained models' verdicts on the same image or image folder."""
import sys
from pathlib import Path
from PIL import Image
import torch

EXPERIMENT = Path(__file__).absolute().parents[1]
for import_root in (EXPERIMENT / "src", EXPERIMENT):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

from deepfake_detection.common.paths import Workspace
from deepfake_detection.inference.engine import InferenceEngine
from deepfake_detection.data.preprocessing import to_rgb


RUNS = [
    ("b0-ffpp-basic", "B0 · FF++ only, basic aug"),
    ("b0-ffpp-robust", "B0 · FF++ only, robust aug"),
    ("b0-mixed-robust", "B0 · FF++ + Celeb-DF"),
    ("xception-clean", "Xception · FF++ + Celeb-DF"),
]


def compare(image_path, runs=RUNS):
    ws = Workspace()
    image = to_rgb(Image.open(image_path))
    rows = []
    for model_id, label in runs:
        try:
            engine = InferenceEngine.from_run(model_id, ws, device="cuda" if torch.cuda.is_available() else "cpu")
            result = engine.analyze(image, explain=False)
            if not result["faces"]:
                rows.append((model_id, label, "no faces detected", None, None, 0))
            else:
                face = result["faces"][0]
                verdict = face["label"] or face["status"]
                fake_score = face["fake_score"]
                threshold = result.get("threshold", 0.5)
                rows.append((model_id, label, verdict, fake_score, threshold, len(result["faces"])))
            del engine
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception as exc:
            rows.append((model_id, label, f"ERROR: {type(exc).__name__}: {exc}", None, None, 0))
    return rows


def print_table(image_path, rows):
    print(f"\nImage: {image_path}")
    print("=" * 100)
    print(f"{'Run':<22} {'Description':<34} {'Verdict':<14} {'Fake score':<11} {'Threshold':<10} {'Faces'}")
    print("-" * 100)
    for run, label, verdict, score, threshold, faces in rows:
        score_str = f"{score:.4f}" if score is not None else "—"
        threshold_str = f"{threshold:.4f}" if threshold is not None else "—"
        print(f"{run:<22} {label:<34} {str(verdict):<14} {score_str:<11} {threshold_str:<10} {faces}")
    print("=" * 100)

    # Consensus check
    verdicts = [row[2] for row in rows if row[2] in {"FAKE", "REAL"}]
    if verdicts:
        fake_count = verdicts.count("FAKE")
        real_count = verdicts.count("REAL")
        if fake_count == len(verdicts):
            print(f"\n→ All {len(verdicts)} models agree: FAKE")
        elif real_count == len(verdicts):
            print(f"\n→ All {len(verdicts)} models agree: REAL")
        else:
            print(f"\n→ Models disagree: {fake_count} say FAKE, {real_count} say REAL")


def main():
    if len(sys.argv) < 2:
        print("Usage: python tools/compare_models.py <image.png> [image2.png ...]")
        print("   or: python tools/compare_models.py <folder>")
        sys.exit(1)
    target = Path(sys.argv[1])
    images = sorted(target.glob("*")) if target.is_dir() else [target]
    for image_path in images:
        if image_path.suffix.lower() not in {".png", ".jpg", ".jpeg"}:
            continue
        print_table(image_path, compare(image_path))


if __name__ == "__main__":
    main()
