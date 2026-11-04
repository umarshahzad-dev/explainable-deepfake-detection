import copy
from io import BytesIO
import torch
from deepfake_detection.common.paths import Workspace, atomic_bytes, fingerprint
from deepfake_detection.data.preprocessing import preprocessing_hash


def resume_signature(cfg, manifest_sha, phase, device):
    from deepfake_detection.common.telemetry import environment_snapshot
    return {"schema_version": 1, "phase": phase, "architecture": cfg["model"]["name"],
            "model_configuration": copy.deepcopy(cfg["model"]), "class_map": copy.deepcopy(cfg["model"]["class_map"]),
            "loss": cfg["training"]["loss"], "manifest_sha256": manifest_sha,
            "preprocessing_hash": preprocessing_hash(cfg), "recipe": copy.deepcopy(cfg["training"]),
            "seed": cfg["seed"], "device_type": torch.device(device).type, "torch_version": str(torch.__version__), "environment": environment_snapshot()}


def assert_compatible(saved, expected):
    if saved != expected:
        keys = sorted(k for k in set(saved) | set(expected) if saved.get(k) != expected.get(k))
        raise ValueError("Resume compatibility mismatch: " + ", ".join(keys))


def save_checkpoint(path, state, workspace=None):
    buffer = BytesIO()
    torch.save(state, buffer)
    return atomic_bytes(path, buffer.getvalue(), workspace)


def load_checkpoint(path, expected, *, workspace=None, device="cpu"):
    path = (workspace or Workspace()).output(path)
    # RNG states contain only safe primitives/tensors, so unrestricted pickle is unnecessary.
    state = torch.load(path, map_location=device, weights_only=True)
    assert_compatible(state["signature"], expected)
    return state
