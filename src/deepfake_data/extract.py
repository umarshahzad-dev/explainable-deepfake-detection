"""Deterministic, bounded face extraction with resumable per-video checkpoints."""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
import math
import os
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass(frozen=True)
class ExtractionConfig:
    max_frames: int = 20
    candidates_per_segment: int = 3
    size: int = 299
    margin: float = 1.5
    threshold: float = 0.95
    min_face_size: int = 80
    min_sharpness: float = 15.0
    max_padding: float = 0.35
    crop_mode: str = "square"
    face_strategy: str = "largest"

    def validate(self):
        if not all(math.isfinite(v) for v in (self.margin, self.threshold, self.min_sharpness, self.max_padding)): raise ValueError("Settings must be finite")
        if self.max_frames < 1 or self.candidates_per_segment < 1 or self.size < 32 or self.margin < 1 or not 0 < self.threshold <= 1 or self.min_face_size < 1 or self.min_sharpness < 0 or not 0 <= self.max_padding < 1:
            raise ValueError("Invalid extraction settings")
        if self.crop_mode not in {"square", "aligned"} or self.face_strategy not in {"largest", "highest_confidence"}:
            raise ValueError("Invalid crop or face strategy")


def sampled_indices(frame_count: int, max_frames: int):
    if frame_count <= 0 or max_frames <= 0: return []
    count = min(frame_count, max_frames)
    return sorted({(i * (frame_count - 1)) // max(1, count - 1) for i in range(count)})


def candidate_segments(frame_count, max_frames, candidates_per_segment):
    if frame_count <= 0: return []
    segments = min(frame_count, max_frames)
    result = []
    for segment in range(segments):
        start = segment * frame_count // segments
        end = (segment + 1) * frame_count // segments
        count = min(candidates_per_segment, end - start)
        result.append([start + ((2 * i + 1) * (end - start)) // (2 * count) for i in range(count)])
    return result


def choose_face(boxes, probabilities, landmarks, threshold, strategy="largest"):
    if strategy not in {"largest", "highest_confidence"}: raise ValueError("Unknown face strategy")
    candidates = [(float((b[2]-b[0])*(b[3]-b[1])) if strategy == "largest" else float(probabilities[i]), -i, i) for i,b in enumerate(boxes) if probabilities[i] is not None and math.isfinite(float(probabilities[i])) and float(probabilities[i]) > 0 and probabilities[i] >= threshold and all(math.isfinite(float(x)) for x in b) and b[2] > b[0] and b[3] > b[1]]
    if not candidates: return None
    i = max(candidates)[2]
    return boxes[i], landmarks[i] if landmarks is not None else None


def square_geometry(box, width, height, margin):
    x1,y1,x2,y2 = map(float,box)
    side = max(1, math.ceil(max(x2-x1,y2-y1) * margin))
    left = math.floor((x1+x2-side)/2); top = math.floor((y1+y2-side)/2)
    right, bottom = left+side, top+side
    ix1,iy1,ix2,iy2 = max(0,left), max(0,top), min(width,right), min(height,bottom)
    area = max(0,ix2-ix1)*max(0,iy2-iy1)
    return (left,top,right,bottom), 1.0-area/(side*side)


def square_crop(frame, box, config):
    import cv2
    bounds, padding = square_geometry(box, frame.shape[1], frame.shape[0], config.margin)
    x1,y1,x2,y2=bounds; height,width=frame.shape[:2]
    if x2 <= 0 or y2 <= 0 or x1 >= width or y1 >= height: return None, padding
    part=frame[max(0,y1):min(height,y2),max(0,x1):min(width,x2)]
    part=cv2.copyMakeBorder(part,max(0,-y1),max(0,y2-height),max(0,-x1),max(0,x2-width),cv2.BORDER_CONSTANT,value=(0,0,0))
    interpolation=cv2.INTER_AREA if part.shape[0] > config.size else cv2.INTER_CUBIC
    return cv2.resize(part,(config.size,config.size),interpolation=interpolation), padding


def evaluate_candidate(frame, boxes, probs, landmarks, index, config):
    import cv2
    import numpy as np
    result={"frame_index":index}
    if boxes is None or probs is None or len(boxes)==0:
        return result | {"reason":"no_face"}, None
    chosen=choose_face(boxes,probs,landmarks,config.threshold,config.face_strategy)
    if chosen is None: return result | {"reason":"low_confidence_or_invalid_box"}, None
    box, points=chosen
    match=next(i for i,b in enumerate(boxes) if b is box or np.array_equal(b,box))
    confidence=float(probs[match]); box=[float(v) for v in box]
    width,height=box[2]-box[0],box[3]-box[1]
    result.update(box=box,confidence=confidence,face_width=width,face_height=height)
    if min(width,height) < config.min_face_size: return result | {"reason":"face_too_small"}, None
    crop,padding=square_crop(frame,box,config); result["padding_fraction"]=padding
    if crop is None or padding > config.max_padding: return result | {"reason":"excessive_padding"}, None
    x1,y1,x2,y2=map(int,(max(0,box[0]),max(0,box[1]),min(frame.shape[1],box[2]),min(frame.shape[0],box[3])))
    face=frame[y1:y2,x1:x2]
    if face.size == 0: return result | {"reason":"empty_face"}, None
    sharpness=float(cv2.Laplacian(cv2.cvtColor(face,cv2.COLOR_BGR2GRAY),cv2.CV_64F).var())
    result["sharpness"]=sharpness
    if not math.isfinite(sharpness) or sharpness < config.min_sharpness: return result | {"reason":"blurred"}, None
    if config.crop_mode == "aligned":
        if points is None: return result | {"reason":"missing_landmarks"}, None
        src=np.asarray(points,dtype=np.float32); dst=np.asarray([[.315,.38],[.685,.38],[.5,.56],[.37,.75],[.63,.75]],dtype=np.float32)*config.size
        matrix,_=cv2.estimateAffinePartial2D(src,dst,method=cv2.LMEDS)
        if matrix is None: return result | {"reason":"alignment_failed"}, None
        crop=cv2.warpAffine(frame,matrix,(config.size,config.size),flags=cv2.INTER_LINEAR,borderMode=cv2.BORDER_CONSTANT)
    # Balanced score saturates sharpness and size; no single metric dominates indefinitely.
    result["score"] = round(confidence + min(sharpness/500.0,1.0) + min(min(width,height)/300.0,1.0) + (1.0-padding), 8)
    return result | {"reason":"eligible"}, crop


def best_candidate(candidates):
    eligible=[item for item in candidates if item[0]["reason"]=="eligible"]
    return max(eligible,key=lambda item:(item[0]["score"],-item[0]["frame_index"])) if eligible else None


def output_key(row):
    return hashlib.sha256((row.get("dataset", "")+":"+row.get("relative_path",row["absolute_path"])).encode()).hexdigest()[:20]


def _write_image(path, image):
    import cv2
    ok, encoded=cv2.imencode(".png",image)
    if not ok: raise OSError(f"Cannot encode {path}")
    temporary=path.with_name(path.name+".partial")
    with temporary.open("wb") as f:
        f.write(encoded.tobytes()); f.flush(); os.fsync(f.fileno())
    os.replace(temporary,path)


def extract_video(video: Path, output: Path, detector, max_frames=20, size=299, threshold=.95, min_difference=None, face_strategy="largest", provenance=None, *, config=None, save_originals=False, image_output=None):
    import cv2
    if min_difference is not None: raise ValueError("min_difference is no longer supported; use segment selection settings")
    config=config or ExtractionConfig(max_frames=max_frames,size=size,threshold=threshold,face_strategy=face_strategy)
    config.validate(); video=Path(video); output=Path(output); output.mkdir(parents=True,exist_ok=True)
    from importlib.metadata import PackageNotFoundError, version
    try: detector_version=version("facenet-pytorch")
    except PackageNotFoundError: detector_version="not_installed"
    runtime={"detector":f"{type(detector).__module__}.{type(detector).__qualname__}","device":str(getattr(detector,"device","unspecified")),"facenet_pytorch":detector_version,"opencv":getattr(cv2,"__version__","unknown")}
    signature={"version":2,"video":str(video.resolve()),"bytes":video.stat().st_size,"mtime_ns":video.stat().st_mtime_ns,"parameters":asdict(config),"runtime":runtime,"provenance":provenance or {},"save_originals":save_originals}
    image_dir=Path(image_output).resolve() if image_output is not None else output.resolve()
    if image_output is not None:
        signature['image_output']=str(image_dir)
        image_dir.mkdir(parents=True,exist_ok=True)
    state=output/"state.json"; completion=output/"complete.json"; progress=output/"progress.jsonl"
    if state.exists() and json.loads(state.read_text(encoding="utf-8")) != signature: raise ValueError("Incompatible extraction checkpoint; choose a new output directory")
    if completion.exists():
        try: old=json.loads(completion.read_text(encoding="utf-8"))
        except json.JSONDecodeError: old=None
        if old is not None:
            if old.get("config") != signature: raise ValueError("Incompatible completed extraction; choose a new output directory")
            if all((output/r["file"]).is_file() and (not save_originals or (output/r["original_file"]).is_file()) for r in old["frames"]): return old
    if not state.exists(): state.write_text(json.dumps(signature,indent=2),encoding="utf-8")
    events=[]; valid_bytes=0
    if progress.exists():
        with progress.open("rb") as f:
            for line in f:
                try:
                    if not line.endswith(b"\n"): break
                    event=json.loads(line); selected=event.get("selected")
                    if selected and (not (output/selected["file"]).is_file() or (save_originals and not (output/selected["original_file"]).is_file())): break
                except (ValueError,KeyError): break
                events.append(event); valid_bytes+=len(line)
        with progress.open("r+b") as f: f.truncate(valid_bytes)
    done={e["segment"] for e in events}
    capture=cv2.VideoCapture(str(video))
    if not capture.isOpened(): raise ValueError(f"Cannot open video: {video}")
    try:
        total=int(capture.get(cv2.CAP_PROP_FRAME_COUNT)); segments=candidate_segments(total,config.max_frames,config.candidates_per_segment)
        if not segments: raise ValueError("Video has no readable frame count")
        for segment,indices in enumerate(segments):
            if segment in done: continue
            candidates=[]
            for index in indices:
                capture.set(cv2.CAP_PROP_POS_FRAMES,index); ok,frame=capture.read()
                if not ok or frame is None:
                    candidates.append(({"frame_index":index,"reason":"read_failed"},None,None)); continue
                boxes,probs,landmarks=detector.detect(cv2.cvtColor(frame,cv2.COLOR_BGR2RGB),landmarks=True)
                metadata,crop=evaluate_candidate(frame,boxes,probs,landmarks,index,config)
                candidates.append((metadata,crop,frame if save_originals else None))
            winner=best_candidate(candidates); selected=None
            decisions=[]
            for metadata,crop,frame in candidates:
                if metadata["reason"]=="eligible" and (winner is None or metadata is not winner[0]): metadata={**metadata,"reason":"lower_rank"}
                decisions.append(metadata)
            if winner is not None:
                metadata,crop,original=winner; name=f"frame_{metadata['frame_index']:09d}.png"
                _write_image(image_dir/name,crop)
                selected={**metadata,"source_video":str(video.resolve()),"file":os.path.relpath(image_dir/name,output),"segment":segment}
                if save_originals:
                    original_name=f"original_{metadata['frame_index']:09d}.png"; _write_image(output/original_name,original); selected["original_file"]=original_name
            event={"segment":segment,"candidates":decisions,"selected":selected}
            with progress.open("a",encoding="utf-8") as log:
                log.write(json.dumps(event)+"\n"); log.flush(); os.fsync(log.fileno())
            events.append(event)
            print(f"{video.name}: segment {segment+1}/{len(segments)} selected={selected is not None}",flush=True)
    finally: capture.release()
    frames=[e["selected"] for e in events if e["selected"] is not None]
    rejected=[{**c,"segment":e["segment"]} for e in events for c in e["candidates"] if c["reason"]!="eligible"]
    result={"config":signature,"frames":frames,"errors":rejected,"rejection_counts":dict(Counter(r["reason"] for r in rejected)),"zero_yield":not frames}
    temp=completion.with_name("complete.json.partial");temp.write_text(json.dumps(result,indent=2),encoding="utf-8");os.replace(temp,completion)
    return result


def select_videos(rows, partition):
    selected=[]
    for row in rows:
        if row.get("project_partition")!=partition or row.get("media_type")!="video":continue
        if row.get("proposal_status")=="DRAFT_NOT_APPROVED":raise ValueError("Draft proposals cannot be extracted")
        selected.append(row)
    return selected


def add_options(p):
    p.add_argument("--max-frames",type=int,default=20);p.add_argument("--candidates-per-segment",type=int,default=3)
    p.add_argument("--size",type=int,default=299);p.add_argument("--margin",type=float,default=1.5)
    p.add_argument("--threshold",type=float,default=.95);p.add_argument("--min-face-size",type=int,default=80)
    p.add_argument("--min-sharpness",type=float,default=15);p.add_argument("--max-padding",type=float,default=.35)
    p.add_argument("--crop-mode",choices=("square","aligned"),default="square");p.add_argument("--face-strategy",choices=("largest","highest_confidence"),default="largest")
    p.add_argument("--device",choices=("cpu","cuda"),default="cpu")


def config_from_args(a):
    return ExtractionConfig(**{key:getattr(a,key) for key in ExtractionConfig.__dataclass_fields__})


def validate_output(target):
    target=Path(target).resolve()
    for root in (Path("data/raw").resolve(),Path(r"D:\University\datasets").resolve()):
        if target.is_relative_to(root): raise ValueError("Output must be outside raw dataset directories")
    return target


def make_detector(device):
    try:
        from facenet_pytorch import MTCNN
        import torch
    except ImportError as exc: raise SystemExit("Use the existing extraction environment: " + str(exc))
    if device=="cuda" and not torch.cuda.is_available(): raise SystemExit("CUDA is unavailable")
    return MTCNN(keep_all=True,device=device)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("video",nargs="?",type=Path);p.add_argument("--manifest",type=Path);p.add_argument("--output",required=True,type=Path)
    p.add_argument("--max-videos",type=int,default=1);p.add_argument("--all-videos",action="store_true");p.add_argument("--allow-empty",action="store_true");p.add_argument("--partition",choices=("train","val","calibration","test"),default="train");add_options(p);a=p.parse_args()
    if bool(a.video)==bool(a.manifest): p.error("Choose exactly one video or --manifest")
    try: config=config_from_args(a); config.validate(); target=validate_output(a.output)
    except ValueError as exc: p.error(str(exc))
    if a.max_videos<1:p.error("max-videos must be positive")
    if a.video: rows=[{"absolute_path":str(a.video.resolve()),"dataset":"single","relative_path":str(a.video.resolve())}]
    else:
        try:
            with a.manifest.open(newline="",encoding="utf-8") as source: rows=select_videos(csv.DictReader(source),a.partition)
        except ValueError as exc:raise SystemExit(str(exc))
        if not a.all_videos: rows=rows[:a.max_videos]
    if not rows:
        if not a.allow_empty:raise SystemExit("No videos selected for partition")
        target.mkdir(parents=True,exist_ok=True)
        (target / f"errors-{a.partition}.json").write_text("[]",encoding="utf-8")
        print(f"No videos selected for {a.partition}; skipped")
        return
    detector=make_detector(a.device); errors=[]
    for number,row in enumerate(rows,1):
        video=Path(row["absolute_path"])
        if video.resolve().is_relative_to(target): raise SystemExit("Output must not contain a source video")
        print(f"Video {number}/{len(rows)}: {video}",flush=True)
        try:
            result=extract_video(video,target/output_key(row),detector,config=config,provenance={k:row.get(k,"") for k in ("dataset","method","relative_path","group_id","project_partition")})
            print(f"Saved {len(result['frames'])}; rejected {len(result['errors'])}; zero_yield={result['zero_yield']}",flush=True)
        except (OSError,ValueError,RuntimeError) as exc:
            errors.append({"video":str(video),"error":str(exc)});print(f"Failed: {exc}",flush=True)
    target.mkdir(parents=True,exist_ok=True);(target/"errors.json").write_text(json.dumps(errors,indent=2),encoding="utf-8")
    if errors: raise SystemExit("Some videos failed; see errors.json")

if __name__ == "__main__":main()
