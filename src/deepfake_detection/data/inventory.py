from pathlib import Path
from deepfake_detection.data.manifests import METHODS, videos
from deepfake_detection.common.paths import Workspace, digest_file, write_json


def inventory(cfg, workspace=None, *, limit_files=2, decode=True, hashes=False):
    ws = workspace or Workspace()
    root = Path(cfg["raw_root"])
    locations = {"ffpp-real": "original_sequences/youtube/c23/videos",
                 "dfd-real": "original_sequences/actors/c23/videos",
                 "dfd-fake": "manipulated_sequences/DeepFakeDetection/c23/videos",
                 "celebdf-real": "Celeb-DF-v2/Celeb-real", "celebdf-fake": "Celeb-DF-v2/Celeb-synthesis"}
    locations.update({f"ffpp-{method}": f"manipulated_sequences/{method}/c23/videos" for method in METHODS})
    counts, all_files = {}, []
    for name, relative in locations.items():
        paths = videos(root / relative)
        counts[name] = {"videos": len(paths), "bytes": sum(p.stat().st_size for p in paths)}
        all_files.extend(paths)
    checked = all_files if limit_files is None else all_files[:max(0, limit_files)]
    results = []
    for path in checked:
        item = {"path": str(path.absolute()), "bytes": path.stat().st_size}
        if hashes:
            item["sha256"] = digest_file(path)
        if decode:
            import cv2
            cap = cv2.VideoCapture(str(path))
            try:
                ok, frame = cap.read()
                item.update({"decodable": bool(cap.isOpened() and ok and frame is not None),
                             "frames_reported": int(cap.get(cv2.CAP_PROP_FRAME_COUNT)), "fps": float(cap.get(cv2.CAP_PROP_FPS))})
            finally:
                cap.release()
        results.append(item)
    report = {"collections": counts, "total_videos": len(all_files), "checked_files": len(checked),
              "checked": results, "full_decode_audit": False,
              "excluded_youtube_real": len(videos(root / "Celeb-DF-v2/YouTube-real")),
              "total_bytes": sum(c["bytes"] for c in counts.values())}
    write_json(ws.report("inventory") / "inventory.json", report, ws)
    return report
