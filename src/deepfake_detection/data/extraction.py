"""Explicit extraction execution only. Importing this module never reads a video."""
from collections import Counter
from io import BytesIO
from pathlib import Path
import time
import random
import math
from PIL import Image
from deepfake_detection.data.face_detector import FaceDetector
from deepfake_detection.data.manifests import stable_key
from deepfake_detection.common.paths import Workspace, atomic_bytes, digest_file, fingerprint, read_json, read_jsonl, write_json, write_jsonl
from deepfake_detection.data.preprocessing import preprocessing_hash,face_crop_variants
from deepfake_detection.common.progress import Progress,console
from deepfake_detection.common.presentation import extraction_detail


def sample_indices(count, quota):
    if count <= 0 or quota <= 0:
        return []
    quota = min(count, quota)
    return [min(count - 1, int((i + 0.5) * count / quota)) for i in range(quota)]


def temporal_candidates(count, quota, *, seed, video_id, attempts=5):
    if not 1 <= attempts <= 20:
        raise ValueError("Attempts per segment must be 1..20")
    primary = sample_indices(count, quota)
    groups = []
    for segment, center in enumerate(primary):
        low, high = segment * count // len(primary), (segment + 1) * count // len(primary)
        center = min(high - 1, max(low, center))
        rng = random.Random(stable_key(f"{video_id}:segment:{segment}", seed))
        # Sample without materializing an entire long-video frame range.
        choices = rng.sample(range(low, high - 1), min(attempts - 1, high - low - 1))
        alternatives = [i if i < center else i + 1 for i in choices]
        groups.append([center, *alternatives])
    return groups


def segment_frames(reader, groups,*,limits=None,stats=None,verbose=False):
    """Decode sequentially once; retain at most one segment's candidate images."""
    defaults={"retry_max_frames_in_memory":8,"retry_max_pixels_per_frame":4000000,"retry_frame_memory_budget_mb":512}
    limits={**defaults,**(limits or {})}
    stats=stats if stats is not None else {}
    stats.setdefault("peak_retry_memory_mb",0)
    stats.setdefault("retry_budget_exhaustions",0)
    stats.setdefault("segment_decisions",[])
    budget=limits["retry_frame_memory_budget_mb"]*2**20
    for segment, indices in enumerate(groups):
        images = {}
        memory=0
        exhausted=False
        reason=None
        stream=iter(reader.frames(sorted(set(indices))))
        try:
            for index,image in stream:
                pixels=image.width*image.height
                size=pixels*(4 if image.mode in {"I","F"} else 2 if image.mode.startswith("I;16") else len(image.getbands()))
                if pixels>limits["retry_max_pixels_per_frame"] or len(images)>=limits["retry_max_frames_in_memory"] or memory+size>budget:
                    exhausted=True
                    reason="frame_pixels" if pixels>limits["retry_max_pixels_per_frame"] else "frame_count" if len(images)>=limits["retry_max_frames_in_memory"] else "memory_budget"
                    image=None
                    break
                images[index]=image
                memory+=size
                stats["peak_retry_memory_mb"]=max(stats["peak_retry_memory_mb"],memory/2**20)
        finally:
            close=getattr(stream,"close",None)
            if close: close()
        decision={"segment":segment,"candidate_retries":max(0,len(indices)-1),"frames_buffered":len(images),
                  "memory_mb":memory/2**20,"exhausted":exhausted,"reason":reason}
        stats["segment_decisions"].append(decision)
        if verbose: console(f"segment {segment}: {decision['candidate_retries']} candidate retries, {len(images)} frames buffered, {memory/2**20:.2f} MB; {reason or 'within budget'}")
        if exhausted:
            stats["retry_budget_exhaustions"]+=1
            images.clear()
            continue
        for attempt, index in enumerate(indices):
            yield segment, attempt, index, images.get(index)


def single_face_heuristic(faces, proximity_threshold=0.25):
    """Require one detection, including low-quality detections; never prove identity."""
    if not math.isfinite(proximity_threshold) or not 0 <= proximity_threshold <= 2:
        raise ValueError("Subject proximity must be a finite crop-side fraction in [0,2]")
    accepted = [face for face in faces if face.status == "accepted"]
    if len(accepted) != 1:
        return False
    chosen = accepted[0]
    geometry = chosen.crop.geometry
    padding = geometry.side * proximity_threshold
    x1, y1 = geometry.x - padding, geometry.y - padding
    x2, y2 = geometry.x + geometry.side + padding, geometry.y + geometry.side + padding
    for other in faces:
        if other is chosen:
            continue
        a, b, c, d = other.box
        if a <= x2 and c >= x1 and b <= y2 and d >= y1:
            return False
    # Even a distant second detection fails the user-requested exactly-one rule.
    return len(faces) == 1


class VideoReader:
    def __init__(self, path):
        import cv2
        self.cv2 = cv2
        self.cap = cv2.VideoCapture(str(path))
        self.count = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.fps = float(self.cap.get(cv2.CAP_PROP_FPS))
        self.position=0
        self.width=int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height=int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        if not self.cap.isOpened() or self.count < 1:
            self.cap.release()
            raise ValueError(f"Cannot decode video: {path}")

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.cap.release()

    def frames(self, targets):
        targets = set(targets)
        if not targets:
            return
        last = max(targets)
        index = self.position
        while index <= last and self.cap.grab():
            self.position=index+1
            if index in targets:
                ok, bgr = self.cap.retrieve()
                if not ok or bgr is None:
                    raise ValueError(f"Frame decode failed at {index}")
                yield index, Image.fromarray(self.cv2.cvtColor(bgr, self.cv2.COLOR_BGR2RGB))
            index=self.position


def _valid_saved(records):
    for row in records:
        path = Path(row["crop_path"])
        if not path.is_file() or digest_file(path) != row["crop_sha256"]:
            return False
        try:
            with Image.open(path) as image:
                image.load()
                if image.mode != "RGB" or image.size != (row["master_size"], row["master_size"]):
                    return False
        except (OSError, ValueError):
            return False
    return True


def scan_orphans(name,workspace=None):
    ws=workspace or Workspace()
    manifest=ws.manifest(name)/"crops.jsonl"
    references={str(ws.output(row["crop_path"]).resolve()) for row in read_jsonl(manifest)} if manifest.exists() else set()
    files=sorted(ws.output(ws.crops/name).rglob("*.png"))
    records=[]
    for file in files:
        file=ws.output(file)
        if str(file.resolve()) not in references:
            stat=file.stat()
            records.append({"path":str(file),"sha256":digest_file(file),"size":stat.st_size,"mtime":stat.st_mtime})
    write_jsonl(ws.manifest(name)/"orphans.jsonl",records,ws)
    console(f"Orphan scan: {len(records)} orphan PNGs of {len(files)} total")
    return {"orphan_png_count":len(records),"orphan_bytes":sum(r["size"] for r in records),
            "physical_png_bytes":sum(p.stat().st_size for p in files),"physical_png_count":len(files)}


def extract(name, cfg, workspace=None, *, limit_videos=None, max_frames=None, detector=None, reader_factory=VideoReader,
            attempts_per_segment=5, auto_verify_single_face=False, subject_proximity=0.25,verbose=False):
    ws = workspace or Workspace()
    target = ws.manifest(name)
    source_path = target / "videos.jsonl"
    rows = read_jsonl(source_path)
    if not rows:
        raise ValueError("No admitted videos: review the quarantine/identity policy first")
    scan_orphans(name,ws)
    if limit_videos is not None and limit_videos < 1 or max_frames is not None and max_frames < 1:
        raise ValueError("Pilot limits must be positive")
    contract = preprocessing_hash(cfg)
    if not 1 <= attempts_per_segment <= 20:
        raise ValueError("Attempts per segment must be 1..20")
    if not math.isfinite(subject_proximity) or not 0 <= subject_proximity <= 2:
        raise ValueError("Subject proximity must be in [0,2]")
    job = fingerprint({"source_manifest": digest_file(source_path), "preprocessing": contract, "quotas": cfg["extraction"],
                       "selection": "temporal-retries-v1", "attempts_per_segment": attempts_per_segment, "seed": cfg["seed"],
                       "auto_verify_single_face": auto_verify_single_face, "subject_proximity": subject_proximity})
    progress_path = target / "extraction_progress.json"
    progress = read_json(progress_path) if progress_path.exists() else {"job": job, "videos": {}}
    if progress["job"] != job:
        raise ValueError("Extraction contract/manifest changed; use a new manifest name rather than mixing crops")
    cache_dir = target / "progress"
    cached = {}
    for sample_id, entry in progress["videos"].items():
        path = ws.output(entry["detail_path"])
        if path.exists():
            cached[sample_id] = read_json(path)
    selected = sorted(rows, key=lambda r: stable_key(r["sample_id"], cfg["seed"]))
    if limit_videos:
        selected = selected[:limit_videos]
    detector = detector or FaceDetector(cfg["preprocessing"])
    attempted = 0
    skipped = 0
    start = time.perf_counter()
    stored_counts = Counter()
    by_stem={fingerprint(r["sample_id"])[:16]:r for r in rows}
    for file in ws.output(ws.crops/name/contract[:12]).rglob("*.png"):
        source=by_stem.get(file.name.split("_f")[0])
        if source: stored_counts[(source["dataset"],source["split"],source["label"])]+=1
    targets = Counter()
    for row in selected:
        key = (row["dataset"], row["split"], row["label"])
        quota_key = ("train" if row["split"] == "train" else "eval") + ("_real" if row["label"] else "_fake")
        targets[key] += (cfg["extraction"]["dfd_frames"] if row["dataset"] == "dfd" else cfg["extraction"][quota_key])*len(cfg['preprocessing'].get('multi_margin',[cfg['preprocessing']['margin_factor']]))
    crop_target = sum(min(n, cfg["extraction"]["train_caps"].get(d, n*2)//2) if s == "train" else n for (d,s,c), n in targets.items())
    feedback = Progress("Video", len(selected))
    verification=Counter()
    for entry in cached.values():
        for r in entry["records"]:
            verification["auto"]+=bool(r.get("auto_verified"))
            verification["review"]+=r["label"]==0 and not r.get("subject_verified")
    def progress_detail(): return extraction_detail(sum(stored_counts.values()),crop_target,verification["auto"],verification["review"])
    for video_number, source in enumerate(selected, 1):
        sample_id = source["sample_id"]
        previous = cached.get(sample_id)
        if previous and previous["status"] == "completed" and _valid_saved(previous["records"]):
            # Source bytes are still checked; replacing a video must not silently reuse old crops.
            if digest_file(source["raw_path"]) == source["sha256"]:
                skipped += 1
                feedback.update(video_number,progress_detail())
                continue
        if max_frames is not None and attempted >= max_frames:
            break
        records = []
        if previous:
            for r in previous["records"]:
                verification["auto"]-=bool(r.get("auto_verified"))
                verification["review"]-=r["label"]==0 and not r.get("subject_verified")
        storage_key = (source["dataset"], source["split"], source["label"])
        quality = Counter()
        retry_stats={}
        status, error = "completed", None
        try:
            if digest_file(source["raw_path"]) != source["sha256"]:
                raise ValueError("Raw source hash changed; prepare a new manifest")
            with reader_factory(source["raw_path"]) as reader:
                extraction_cfg = cfg["extraction"]
                if source["dataset"] == "dfd":
                    quota = extraction_cfg["dfd_frames"]
                else:
                    prefix = "train" if source["split"] == "train" else "eval"
                    quota = extraction_cfg[prefix + ("_real" if source["label"] == 1 else "_fake")]
                groups = temporal_candidates(reader.count, quota, seed=cfg["seed"], video_id=sample_id, attempts=attempts_per_segment)
                succeeded = set()
                decoded_count = 0
                for segment, retry, index, image in segment_frames(reader, groups,limits=cfg["extraction"],stats=retry_stats,verbose=verbose):
                    if segment in succeeded:
                        continue
                    if max_frames is not None and attempted >= max_frames:
                        status = "pilot_partial"
                        break
                    attempted += 1
                    feedback.update(video_number-1,progress_detail())
                    quality["frame_attempts"] += 1
                    if retry:
                        quality["retry_attempts"] += 1
                    if image is None:
                        quality["decode_missing"] += 1
                        continue
                    decoded_count += 1
                    faces = detector.detect(image)
                    if len(faces) != 1:
                        quality["no_face" if not faces else "ambiguous_multiple_faces"] += 1
                        continue
                    face = faces[0]
                    if face.status != "accepted":
                        quality["low_quality"] += 1
                        continue
                    variants=getattr(face,'variant_crops',None)
                    if variants is None:
                        variants=face_crop_variants(image,face.box,face.landmarks,cfg['preprocessing']) if len(cfg['preprocessing'].get('multi_margin',[]))>1 else [(cfg['preprocessing']['margin_factor'],face.crop)]
                    for margin,variant in variants:
                        suffix=('_m'+f'{margin:g}'.replace('.','p')) if len(variants)>1 else ''
                        id_suffix=(':m'+f'{margin:g}') if len(variants)>1 else ''
                        stem = fingerprint(sample_id)[:16]
                        path = ws.output(ws.crops / name / contract[:12] / source["split"] / ("real" if source["label"] else "fake") / f"{stem}_f{index:07d}{suffix}.png")
                        new_file=not path.exists()
                        cap = cfg["extraction"]["train_caps"].get(source["dataset"])
                        if new_file and source["split"] == "train" and cap is not None and stored_counts[storage_key] >= cap // 2:
                            quality["storage_cap"] += 1
                            continue
                        buffer = BytesIO()
                        variant.image.save(buffer, format="PNG", compress_level=6)
                        atomic_bytes(path, buffer.getvalue(), ws)
                        record = {**source, "sample_id": f"{sample_id}:f{index}{id_suffix}", "video_id": sample_id,
                                  "frame_index": index, "crop_path": str(path), "crop_sha256": digest_file(path),
                                  "temporal_segment": segment, "selection_attempt": retry + 1,
                                  "box": face.box, "geometry": variant.geometry.as_dict(), "landmarks": face.landmarks,
                                  "detection_probability": face.probability, "preprocessing_hash": contract,
                                  "master_size": cfg["preprocessing"]["master_size"], "crop_margin":margin}
                        record["auto_verified"] = bool(source["label"] == 0 and auto_verify_single_face and single_face_heuristic(faces, subject_proximity))
                        record["detected_face_count"] = len(faces)
                        record["pilot"] = bool(source.get("pilot") or limit_videos is not None or max_frames is not None)
                        if record["auto_verified"]:
                            record.update(subject_verified=True, verification_source="auto_single_face_heuristic", review_note="single-face heuristic", subject_proximity=subject_proximity)
                        records.append(record)
                        verification["auto"]+=record["auto_verified"]
                        verification["review"]+=record["label"]==0 and not record.get("subject_verified")
                        stored_counts[storage_key] += int(new_file)
                    succeeded.add(segment)
                    if retry:
                        quality["retry_successes"] += 1
                quality["segments_exhausted"] += len(groups) - len(succeeded) if status == "completed" else 0
                if status == "completed" and not decoded_count and not retry_stats.get("retry_budget_exhaustions"):
                    raise ValueError("Video yielded no decodable candidate frames")
        except Exception as exc:
            status, error = "failed", f"{type(exc).__name__}: {exc}"
        for record in records:
            record["extraction_status"] = "subject_unverified" if status == "completed" and record["label"] == 0 and not record.get("subject_verified") else status
        detail = {"status": status, "source_sha256": source["sha256"], "records": records,
                  "retry_stats":retry_stats,
                  "quality_counts": dict(quality), "error": error}
        detail_path = ws.output(cache_dir / (fingerprint(sample_id) + ".json"))
        write_json(detail_path, detail, ws)
        cached[sample_id] = detail
        progress["videos"][sample_id] = {"status": status, "saved": len(records), "detail_path": str(detail_path), "error": error}
        write_json(progress_path, progress, ws)
        feedback.update(video_number,progress_detail())
    full_records = [r for entry in cached.values() for r in entry["records"]]
    # Cap stored training examples without duplicating images. Sampling later balances method/domain/video.
    capped = []
    counts = Counter()
    for row in sorted(full_records, key=lambda r: stable_key(r["sample_id"], cfg["seed"])):
        key = (row["dataset"], row["split"], row["label"])
        cap = cfg["extraction"]["train_caps"].get(row["dataset"])
        if row["split"] != "train" or cap is None or counts[key] < cap // 2:
            capped.append(row)
            counts[key] += 1
    # Restore temporal order for readability, not for sampling.
    capped.sort(key=lambda r: (r["video_id"], r["frame_index"]))
    write_jsonl(target / "crops.jsonl", capped, ws)
    orphan_report=scan_orphans(name,ws)
    quality_total = Counter()
    for entry in cached.values():
        quality_total.update(entry["quality_counts"])
    report = {"saved_crops": len(capped), "failed_videos": sum(e["status"] == "failed" for e in cached.values()),
              "peak_retry_memory_mb":max((e.get("retry_stats",{}).get("peak_retry_memory_mb",0) for e in cached.values()),default=0),
              "retry_budget_exhaustions":sum(e.get("retry_stats",{}).get("retry_budget_exhaustions",0) for e in cached.values()),
              **orphan_report,
              "auto_verified_crops": sum(bool(r.get("auto_verified")) for r in capped),
              "review_required_crops": sum(r["label"] == 0 and not r.get("subject_verified") for r in capped),
              "stored_png_bytes": sum(Path(r["crop_path"]).stat().st_size for r in full_records),
              "pilot_partial_videos": sum(e["status"] == "pilot_partial" for e in cached.values()),
              "skipped_completed_videos": skipped, "frames_attempted_this_invocation": attempted,
              "quality_counts": dict(quality_total), "elapsed_seconds": time.perf_counter() - start,
              "attempts_per_segment": attempts_per_segment, "selection_version": "temporal-retries-v1",
              "preprocessing_hash": contract, "manifest_path": str(target / "crops.jsonl"),
              "counts": [{"dataset": d, "split": s, "label": label, "count": n} for (d, s, label), n in sorted(counts.items())]}
    write_json(target / "extraction_report.json", report, ws)
    feedback.finish(sum(s["sample_id"] in cached for s in selected),progress_detail())
    return report
