"""Extract a bounded selection of GIF frames while preserving GIF provenance."""
import argparse
import hashlib
import json
from pathlib import Path
from .extract import sampled_indices


def extract_gif(source, output, max_frames=10):
    from PIL import Image
    source = Path(source).resolve(); output = Path(output).resolve()
    if output == source.parent or source.is_relative_to(output) or output.is_relative_to(source.parent):
        raise ValueError("Choose an output directory outside the source GIF directory")
    raw_root = Path("data/raw").resolve(); datasets = Path(r"D:\University\datasets").resolve()
    if output.is_relative_to(raw_root) or output.is_relative_to(datasets): raise ValueError("Output must be outside raw datasets")
    key=hashlib.sha256(str(source).encode()).hexdigest()[:20]; target=output/key; target.mkdir(parents=True,exist_ok=True)
    with Image.open(source) as image:
        count=getattr(image,"n_frames",1)
        signature={"source":str(source),"bytes":source.stat().st_size,"mtime_ns":source.stat().st_mtime_ns,"max_frames":max_frames}
        metadata=target/"complete.json"
        if metadata.exists():
            result=json.loads(metadata.read_text(encoding="utf-8"))
            if result["config"] != signature: raise ValueError("Different source or settings; choose a new output directory")
            if all((target/r["file"]).is_file() for r in result["frames"]): return result
        rows=[]
        for index in sampled_indices(count,max_frames):
            image.seek(index); name=f"frame_{index:09d}.png"; image.convert("RGB").save(target/name)
            rows.append({"source_gif":str(source),"frame_index":index,"file":name})
        result={"config":signature,"source_frames":count,"animated":count>1,"frames":rows}
        metadata.write_text(json.dumps(result,indent=2),encoding="utf-8");return result


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("gif",type=Path);p.add_argument("--output",type=Path,required=True);p.add_argument("--max-frames",type=int,default=10);a=p.parse_args()
    if a.max_frames < 1: p.error("max-frames must be positive")
    try: import PIL  # noqa: F401
    except ImportError: raise SystemExit("Install optional inspection dependencies: pip install -e '.[inspection]'")
    print(json.dumps(extract_gif(a.gif,a.output,a.max_frames),indent=2))

if __name__ == "__main__":main()
