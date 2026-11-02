"""Bounded video face extraction. Requires optional extraction dependencies."""
from __future__ import annotations
import argparse
import csv
import json
import os
from pathlib import Path


def sampled_indices(frame_count: int, max_frames: int):
    if frame_count <= 0 or max_frames <= 0: return []
    count = min(frame_count, max_frames)
    return sorted({(i * (frame_count - 1)) // max(1, count - 1) for i in range(count)})


def choose_face(boxes, probabilities, landmarks, threshold):
    candidates = [(float((b[2]-b[0])*(b[3]-b[1])), i) for i,b in enumerate(boxes) if probabilities[i] is not None and probabilities[i] >= threshold and b[2] > b[0] and b[3] > b[1]]
    if not candidates: return None
    _, i = max(candidates)
    return boxes[i], landmarks[i]


def extract_video(video: Path, output: Path, detector, max_frames=20, size=256, threshold=.9, min_difference=3.0):
    import cv2
    import numpy as np
    output.mkdir(parents=True, exist_ok=True)
    completion = output / "complete.json"
    signature = {"video": str(video.resolve()), "bytes": video.stat().st_size, "mtime_ns": video.stat().st_mtime_ns, "max_frames": max_frames, "size": size, "threshold": threshold, "min_difference": min_difference}
    state = output / "state.json"
    if state.exists() and json.loads(state.read_text(encoding="utf-8")) != signature:
        raise ValueError("Partial output has different input or settings; choose another output directory")
    if completion.exists():
        old=json.loads(completion.read_text(encoding="utf-8"))
        if old.get("config") == signature and all((output / r["file"]).is_file() for r in old.get("frames", [])): return old
        raise ValueError("Existing completed output uses different input or settings; choose another output directory")
    if not state.exists():
        state.write_text(json.dumps(signature, indent=2), encoding="utf-8")
    capture = cv2.VideoCapture(str(video))
    if not capture.isOpened(): raise ValueError(f"Cannot open video: {video}")
    rows, errors, previous = [], [], None
    try:
        total = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        indices = sampled_indices(total, max_frames)
        if not indices: raise ValueError("Video has no readable frame count")
        for index in indices:
            capture.set(cv2.CAP_PROP_POS_FRAMES, index); ok, frame = capture.read()
            if not ok: errors.append({"frame":index,"error":"read_failed"}); continue
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            boxes, probs, landmarks = detector.detect(rgb, landmarks=True)
            if boxes is None or landmarks is None:
                errors.append({"frame":index,"error":"no_face"}); continue
            face = choose_face(boxes, probs, landmarks, threshold)
            if face is None: errors.append({"frame":index,"error":"no_face_above_threshold"}); continue
            _, points = face
            # Five-point affine alignment, preserving a neutral output size.
            source = np.asarray(points, dtype=np.float32)
            dest = np.asarray([[.315,.38],[.685,.38],[.5,.56],[.37,.75],[.63,.75]], dtype=np.float32) * size
            matrix, _ = cv2.estimateAffinePartial2D(source, dest, method=cv2.LMEDS)
            if matrix is None: errors.append({"frame":index,"error":"alignment_failed"}); continue
            crop=cv2.warpAffine(frame, matrix, (size,size), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)
            gray=cv2.resize(cv2.cvtColor(crop,cv2.COLOR_BGR2GRAY),(32,32)).astype(np.float32)
            if previous is not None and np.mean(np.abs(gray-previous)) < min_difference:
                errors.append({"frame":index,"error":"near_duplicate"}); continue
            name=f"frame_{index:09d}.png"; temporary=output / (name + ".tmp.png")
            if not cv2.imwrite(str(temporary), crop): raise OSError(f"Failed to write {temporary}")
            os.replace(temporary, output / name); previous=gray
            rows.append({"frame_index":index,"source_video":str(video.resolve()),"file":name})
            print(f"{video.name}: frame {index}/{total}", flush=True)
    finally: capture.release()
    result={"config":signature,"frames":rows,"errors":errors}
    temp=output / "complete.json.tmp"; temp.write_text(json.dumps(result,indent=2),encoding="utf-8"); os.replace(temp,completion)
    return result


def main():
    import hashlib
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("video", nargs="?", type=Path)
    p.add_argument("--manifest", type=Path, help="Split manifest; process train videos only")
    p.add_argument("--output",required=True,type=Path)
    p.add_argument("--max-videos", type=int, default=1, help="Bound processing; use --all-videos only after approval")
    p.add_argument("--all-videos", action="store_true")
    p.add_argument("--max-frames",type=int,default=20)
    p.add_argument("--size",type=int,default=256)
    p.add_argument("--threshold",type=float,default=.9)
    p.add_argument("--device",choices=("cpu","cuda"),default="cpu")
    a=p.parse_args()
    if bool(a.video) == bool(a.manifest): p.error("Choose exactly one video or --manifest")
    if a.max_frames < 1 or a.max_videos < 1 or a.size < 32 or not 0 <= a.threshold <= 1: p.error("Invalid extraction settings")
    raw_root = Path("data/raw").resolve()
    dataset_root = Path(r"D:\University\datasets").resolve()
    target = a.output.resolve()
    if target.is_relative_to(raw_root) or target.is_relative_to(dataset_root): p.error("Output must be outside raw dataset directories")
    if a.video:
        rows=[{"absolute_path":str(a.video), "dataset":"single", "relative_path":a.video.name}]
    else:
        with a.manifest.open(newline="",encoding="utf-8") as source:
            rows=[r for r in csv.DictReader(source) if r.get("project_partition") == "train" and r.get("media_type") == "video"]
        if not a.all_videos: rows=rows[:a.max_videos]
    if not rows: raise SystemExit("No training videos selected")
    try:
        from facenet_pytorch import MTCNN
        import torch
        import cv2  # noqa: F401
    except ImportError as exc: raise SystemExit("Install optional extraction dependencies first: " + str(exc))
    if a.device == "cuda" and not torch.cuda.is_available(): raise SystemExit("CUDA is not available in the installed PyTorch environment")
    detector=MTCNN(keep_all=True, device=a.device)
    errors=[]
    for number,row in enumerate(rows,1):
        video=Path(row["absolute_path"])
        if video.resolve().is_relative_to(target): raise SystemExit("Output must not contain a source video")
        key=hashlib.sha256((row.get("dataset", "")+":"+row.get("relative_path",str(video))).encode()).hexdigest()[:20]
        print(f"Video {number}/{len(rows)}: {video}", flush=True)
        try:
            result=extract_video(video,target/key,detector,a.max_frames,a.size,a.threshold)
            print(f"Saved {len(result['frames'])} frames; {len(result['errors'])} skipped",flush=True)
        except (OSError,ValueError,RuntimeError) as exc:
            errors.append({"video":str(video),"error":str(exc)})
            print(f"Failed: {exc}",flush=True)
    target.mkdir(parents=True,exist_ok=True)
    (target / "errors.json").write_text(json.dumps(errors,indent=2),encoding="utf-8")
    if errors: raise SystemExit("Some videos failed; see errors.json")

if __name__ == "__main__": main()
