"""Bounded diagnostic actions; never export their temporary updated weights."""
import itertools
import time
import torch
from torch.utils.data import DataLoader
from deepfake_detection.data.datasets import CropDataset, crop_rows
from deepfake_detection.data.manifests import stratified_limit
from deepfake_detection.models.model_loader import initialize_model
from deepfake_detection.common.paths import Workspace, write_json
from deepfake_detection.common.reproducibility import seed_all
from deepfake_detection.training.trainer import train_epoch, NoOptimizerUpdates


def diagnostic(cfg, workspace=None, *, synthetic=False, manifests=None, steps=2, warmup=1,
               device=None, kind="profile", limit_images=16, no_amp=False):
    if not 1 <= steps <= 500 or not 0 <= warmup <= 20:
        raise ValueError("Diagnostic steps must be 1..500 and warmup 0..20")
    ws = workspace or Workspace()
    device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    seed_all(cfg["seed"])
    model = initialize_model(cfg, device=device)
    recipe = cfg["training"]
    optimizer = torch.optim.AdamW(model.parameters(), lr=recipe["backbone_lr"], weight_decay=recipe["weight_decay"])
    if synthetic:
        size = cfg["model"]["input_size"]
        inputs = torch.randn(recipe["batch_size"], 3, size, size)
        labels = torch.arange(recipe["batch_size"]) % 2
        factory = lambda count: ((inputs, labels) for _ in range(count))
    else:
        if not manifests:
            raise ValueError("Choose --synthetic or an explicit crop manifest")
        rows = [r for r in crop_rows(manifests, ws) if r["split"] == "train"]
        rows = stratified_limit(rows, limit_images, cfg["seed"])
        if not rows:
            raise ValueError("No completed training crops available; use --synthetic for the first pilot")
        dataset = CropDataset(rows, cfg, training=True, workspace=ws)
        def factory(count):
            def batches():
                remaining = count
                while remaining:
                    for batch in DataLoader(dataset, batch_size=recipe["batch_size"], shuffle=False, num_workers=0):
                        yield batch
                        remaining -= 1
                        if not remaining:
                            return
            return batches()
    from deepfake_detection.training.precision import amp_settings
    amp = bool(recipe["amp"] and not no_amp)
    _,_,scale_enabled=amp_settings(amp,device,recipe)
    options = {"device": device, "accumulation": recipe["accumulation"], "amp": amp, "clip_norm": recipe["clip_norm"], "recipe":{"amp_dtype":recipe.get("amp_dtype","float16")}}
    # Diagnostics only: a short pilot cannot amortize the default scale's overflow warmup.
    scaler = torch.amp.GradScaler("cuda", enabled=scale_enabled, init_scale=1024)
    retries = 0
    def run_segment(count):
        nonlocal scaler, retries
        for attempt in range(4):
            try:
                return train_epoch(model, factory(count), optimizer, scaler=scaler, **options)
            except NoOptimizerUpdates as exc:
                if exc.reason != "scaler_skipped" or attempt == 3:
                    raise
                retries += 1
                scaler = torch.amp.GradScaler("cuda", enabled=scale_enabled, init_scale=max(1, scaler.get_scale() / 8))
    if warmup:
        run_segment(warmup)
    if device.type == "cuda":
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats(device)
    started = time.perf_counter()
    from deepfake_detection.common.telemetry import MemoryMonitor
    with MemoryMonitor(device) as memory:
        result = run_segment(steps)
    if device.type == "cuda":
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - started
    report = {"kind": kind, "synthetic": synthetic, "microbatches": steps,
              "memory": memory.report(),
              "optimizer_updates": result["optimizer_updates"], "samples": result["samples"],
              "loss": result["loss"], "elapsed_seconds": elapsed, "images_per_second": result["samples"] / elapsed,
              "device": str(device), "amp_used": amp and device.type == "cuda", "amp_retries": retries,
              "diagnostic_no_amp": no_amp, "final_amp_scale": scaler.get_scale(),
              "batch_size": recipe["batch_size"], "accumulation": recipe["accumulation"],
              "peak_allocated_mib": torch.cuda.max_memory_allocated(device) / 2**20 if device.type == "cuda" else None,
              "peak_reserved_mib": torch.cuda.max_memory_reserved(device) / 2**20 if device.type == "cuda" else None,
              "note": "Diagnostic temporary updates only. No weights saved. Synthetic throughput excludes real image I/O and augmentation."}
    write_json(ws.report(kind) / "diagnostic.json", report, ws)
    return report
