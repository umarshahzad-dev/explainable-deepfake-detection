"""Bounded decode inspection for representative images and GIFs."""
import argparse
import csv
import json
from pathlib import Path


def inspect_image(path):
    from PIL import Image
    try:
        with Image.open(path) as image:
            info={"path": str(path), "status":"ok", "format":image.format, "size":list(image.size), "frames":getattr(image,"n_frames",1), "mode":image.mode}
            image.seek(0); image.load()
            return info
    except (OSError, ValueError) as exc:
        return {"path":str(path),"status":"error","error":str(exc)}


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("manifest",type=Path);p.add_argument("--limit",type=int,default=20);p.add_argument("--output",type=Path,required=True);a=p.parse_args()
    if a.limit < 1: p.error("limit must be positive")
    try: import PIL  # noqa: F401
    except ImportError: raise SystemExit("Install the optional inspection dependencies: pip install -e '.[inspection]'")
    with a.manifest.open(newline="",encoding="utf-8") as f:
        selected=[]
        for row in csv.DictReader(f):
            if row.get("media_type") in {"image","gif"}: selected.append(row)
            if len(selected) >= a.limit: break
    report=[inspect_image(Path(row["absolute_path"])) for row in selected]
    a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(report,indent=2),encoding="utf-8")
    print(f"Inspected {len(report)} images; errors: {sum(r['status'] != 'ok' for r in report)}")

if __name__ == "__main__":main()
