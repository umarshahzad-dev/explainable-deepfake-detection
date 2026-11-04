"""Offline defaults from the authoritative Branch B plan, without parent config imports."""
import copy
import json
from pathlib import Path
from deepfake_detection.common.paths import ROOT

DEFAULTS = {
    "raw_root": "data/raw",
    "weights": "weights/efficientnet_b0.safetensors",
    "official_splits": "data/splits/ffpp_official",
    "identity_registry": None,
    "identity_policy": "source-parent",
    "seed": 42,
    "preprocessing": {
        "version": "square299_bicubic224_mtcnn50_v1", "master_size": 299,
        "margin_factor": 1.3, "confidence": 0.95, "min_face_size": 50,
        "interpolation": "bicubic", "max_pixels": 8000000, "max_faces": 10,
        "detector": {"image_size": 299, "margin": 0, "min_face_size": 50,
                     "thresholds": [0.6, 0.7, 0.7], "factor": 0.709,
                     "post_process": False, "select_largest": True,
                     "selection_method": "largest", "keep_all": True},
    },
    "model": {"name": "efficientnet_b0.ra_in1k", "input_size": 224,
              "mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225],
              "class_map": {"fake": 0, "real": 1}, "dropout": 0.2},
    "training": {"batch_size": 8, "accumulation": 4, "workers": 2, "pin_memory": True,
                 "amp": True, "head_epochs": 2, "finetune_epochs": 10,
                 "head_lr": 0.001, "backbone_lr": 0.0001, "finetune_head_lr": 0.0003,
                 "weight_decay": 0.0001, "min_lr": 0.000001, "warmup_epochs": 1,
                 "patience": 4, "clip_norm": 1.0, "loss": "bce", "freeze_bn": True,
                 "augmentation": "robust", "epoch_samples": 40000},
    "extraction": {"train_real": 20, "train_fake": 4, "eval_real": 10, "eval_fake": 2,
                   "retry_max_frames_in_memory":8,"retry_max_pixels_per_frame":4000000,"retry_frame_memory_budget_mb":512,
                   "dfd_frames": 20, "dfd_real_clips": 200, "dfd_fake_clips": 200,
                   "train_caps": {"ffpp": 24000, "celebdf": 16000}},
}


def merge(base, values):
    for key, value in values.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            merge(base[key], value)
        else:
            base[key] = value
    return base


def load_config(path=None, *, root=ROOT, overrides=None):
    cfg = copy.deepcopy(DEFAULTS)
    if path:
        source = Path(path)
        source = source if source.is_absolute() else Path(root) / source
        merge(cfg, json.loads(source.read_text(encoding="utf-8-sig")))
    if overrides:
        merge(cfg, overrides)
    for key in ("raw_root", "weights", "official_splits", "identity_registry"):
        if cfg.get(key):
            value = Path(cfg[key])
            cfg[key] = str(value.absolute() if value.is_absolute() else (Path(root) / value).absolute())
    if cfg["identity_policy"] not in {"source-parent", "strict-person"}:
        raise ValueError("identity_policy must be source-parent or strict-person")
    if cfg["model"]["class_map"] != {"fake": 0, "real": 1}:
        raise ValueError("This experiment requires fake=0, real=1")
    for key in ("batch_size", "accumulation", "workers", "head_epochs", "finetune_epochs", "epoch_samples"):
        minimum = 0 if key == "workers" else 1
        if cfg["training"][key] < minimum:
            raise ValueError(f"Invalid training value: {key}")
    for key in ("retry_max_frames_in_memory","retry_max_pixels_per_frame","retry_frame_memory_budget_mb"):
        if cfg["extraction"][key]<=0: raise ValueError(f"Invalid extraction guard: {key}")
    return cfg
