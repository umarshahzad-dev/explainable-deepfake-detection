"""Deterministic, stratified sampled PNG integrity gate before model construction."""
from collections import defaultdict
import time
from PIL import Image
from deepfake_detection.common.paths import Workspace, digest_file, fingerprint
from deepfake_detection.data.manifests import stratified_limit
from deepfake_detection.common.progress import console


def preflight_crops(rows,cfg,workspace=None,*,limit_per_partition=200):
    ws=workspace or Workspace()
    started=time.perf_counter()
    groups=defaultdict(list)
    for row in rows: groups[row["split"]].append(row)
    verified=0
    partitions=[]
    samples=[]
    for split,group in sorted(groups.items()):
        selected=stratified_limit(group,min(limit_per_partition,len(group)),cfg["seed"])
        partitions.append({"split":split,"sampled":len(selected),"labels":sorted({r["label"] for r in selected})})
        for row in selected:
            path=ws.output(row["crop_path"])
            expected=row.get("crop_sha256")
            actual="unreadable"
            try:
                actual=digest_file(path)
                if not expected or actual != expected:
                    raise ValueError(f"PNG integrity mismatch: {path}; expected={expected}; actual={actual}")
                with Image.open(path) as image:
                    image.load()
                    if image.mode != "RGB" or image.size != (cfg["preprocessing"]["master_size"],)*2:
                        raise ValueError(f"PNG decode/geometry mismatch: {path}; expected={expected}; actual={actual}")
            except (OSError,ValueError) as exc:
                console(f"Integrity preflight: {verified}/{len(rows)} crops verified, 1 mismatches, elapsed {time.perf_counter()-started:.2f}s")
                raise ValueError(f"PNG integrity failed: {path}; expected={expected}; actual={actual}; {exc}") from exc
            verified+=1
            samples.append((str(path),actual))
    elapsed=time.perf_counter()-started
    console(f"Integrity preflight: {verified}/{len(rows)} crops verified, 0 mismatches, elapsed {elapsed:.2f}s")
    return {"verified":verified,"total":len(rows),"mismatches":0,"elapsed_seconds":elapsed,
            "partitions":partitions,"sample_content_hash":fingerprint(samples),"limit_per_partition":limit_per_partition}
