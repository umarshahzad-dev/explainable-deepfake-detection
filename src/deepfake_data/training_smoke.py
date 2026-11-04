"""Bounded no-weight-update compatibility check for the Clean Xception setup."""
from __future__ import annotations

import argparse
import csv
import gzip
import json
from pathlib import Path
from typing import Any

from .training_data import XceptionTrainingImageDataset, load_clean_xception


def select_smoke_rows(manifest: Path, limit: int = 4) -> list[dict[str, Any]]:
    """Select a deterministic, source-distinct sample with both verified labels."""
    selected: list[dict[str, Any]] = []
    seen = set()
    opener = gzip.open if str(manifest).endswith(".gz") else Path.open
    with opener(manifest, "rt", encoding="utf-8", newline="") as stream:
        for row in csv.DictReader(stream):
            if row.get("partition") != "train" or row.get("label") not in {"0", "1"}:
                continue
            group = (row.get("label"), row.get("dataset"), row.get("source_group_id"))
            if group in seen:
                continue
            path = Path(row["input_path"])
            if not path.is_file():
                raise FileNotFoundError(path)
            selected.append(row)
            seen.add(group)
            labels = {sample["label"] for sample in selected}
            if len(selected) >= limit and labels == {"0", "1"}:
                break
    if not selected or {row["label"] for row in selected} != {"0", "1"}:
        raise ValueError("Smoke sample must contain both verified labels")
    return selected


def run_smoke(manifest: Path, checkpoint: Path, metadata: Path, *, batch_size: int = 2,
              device: str = "cuda") -> dict[str, Any]:
    import torch
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    actual_device = device
    rows = select_smoke_rows(manifest, max(batch_size, 2))
    model, model_metadata = load_clean_xception(checkpoint, metadata, device=actual_device)
    model.eval()
    dataset = XceptionTrainingImageDataset(rows)
    images, labels = zip(*(dataset[i] for i in range(len(dataset))))
    x = torch.stack(images).to(actual_device)
    y = torch.stack(labels).to(actual_device).view(-1, 1)
    if actual_device.startswith("cuda"):
        torch.cuda.reset_peak_memory_stats()
    with torch.no_grad():
        logits = model(x)
        if tuple(logits.shape) != (len(rows), 1):
            raise ValueError(f"Unexpected Xception output shape: {tuple(logits.shape)}")
        loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, y)
    peak = torch.cuda.max_memory_allocated() if actual_device.startswith("cuda") else None
    return {
        "status": "PASS_NO_WEIGHT_UPDATE",
        "device": actual_device,
        "cuda_name": torch.cuda.get_device_name() if actual_device.startswith("cuda") else None,
        "batch_images": len(rows),
        "input_shape": list(x.shape),
        "labels": [int(row["label"]) for row in rows],
        "output_shape": list(logits.shape),
        "loss_finite": bool(torch.isfinite(loss).item()),
        "logit_range": [float(logits.min().cpu()), float(logits.max().cpu())],
        "peak_cuda_allocated_bytes": int(peak) if peak is not None else None,
        "checkpoint_metadata": {
            "architecture": model_metadata.get("architecture") or model_metadata.get("model_name"),
            "class_map": model_metadata.get("class_map"),
            "weights_sha256": model_metadata.get("weights_sha256"),
        },
        "optimizer_steps": 0,
    }


def main(argv: list[str] | None = None) -> int:
    root = Path(__file__).resolve().parents[2]
    previous = Path(r"D:\University\explainable-deepfake-detection-system")
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--manifest", type=Path, default=root / "artifacts/training/xception-v2-ready-v2/images.csv.gz")
    p.add_argument("--checkpoint", type=Path, default=previous / "runs/best_xception/xcep-clean-s42/best.safetensors")
    p.add_argument("--metadata", type=Path, default=previous / "runs/best_xception/xcep-clean-s42/best_metadata.json")
    p.add_argument("--batch-size", type=int, default=2)
    p.add_argument("--device", default="cuda")
    p.add_argument("--report", type=Path, default=root / "artifacts/training/xception-v2-ready-v2/gpu-dryrun.json")
    a = p.parse_args(argv)
    result = run_smoke(a.manifest, a.checkpoint, a.metadata, batch_size=a.batch_size, device=a.device)
    a.report.parent.mkdir(parents=True, exist_ok=True)
    a.report.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
