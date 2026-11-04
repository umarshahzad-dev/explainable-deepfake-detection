"""Stable model IDs and resolution to local, verified model artifacts.

Historical experiment run paths are retained in each specification for
provenance. New training outputs continue to live under ``runs/``; registered
restored checkpoints live under ``models/``.
"""
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType

from deepfake_detection.common.paths import Workspace


@dataclass(frozen=True)
class ModelSpec:
    """Model registry entry with project-relative artifact and history paths."""

    identifier: str
    artifact_path: str
    category: str
    inference_enabled: bool
    description: str
    historical_run_path: str | None = None


_SPECS = {
    "xception-clean": ModelSpec(
        "xception-clean", "models/primary/xception_clean", "primary", True,
        "Verified Xception Clean detector; default demonstration model.",
        "runs/best_xception/xcep-clean-s42",
    ),
    "xception-mixed-robust": ModelSpec(
        "xception-mixed-robust", "models/archive/xception_mixed_robust", "archive", True,
        "Verified run-managed model; optional comparison model.",
        "runs/archive/xcep-mixed-robust-s42",
    ),
    "xception-final-v1": ModelSpec(
        "xception-final-v1", "models/archive/xception_final_v1", "archive", True,
        "Loadable archived model with weak/incomplete experiment evidence; not recommended for demonstration.",
        "runs/archive/xcep-final-v1-s42-STACKED-REGULARIZATION-FAILED",
    ),
    "b0-ffpp-basic": ModelSpec(
        "b0-ffpp-basic", "models/archive/efficientnet_b0_ffpp_basic", "archive", True,
        "Verified FF++-only EfficientNet-B0 run.", "runs/archive/b0-ffpp-basic",
    ),
    "b0-ffpp-robust": ModelSpec(
        "b0-ffpp-robust", "models/archive/efficientnet_b0_ffpp_robust", "archive", True,
        "Verified FF++-only EfficientNet-B0 run with robust augmentation.",
        "runs/archive/b0-ffpp-robust",
    ),
    "b0-mixed-robust": ModelSpec(
        "b0-mixed-robust", "models/archive/efficientnet_b0_mixed_robust", "archive", True,
        "Verified mixed-dataset EfficientNet-B0 run.", "runs/trained_models/b0-mixed-robust",
    ),
    "e2e-pilot-bounded": ModelSpec(
        "e2e-pilot-bounded", "models/archive/e2e_pilot_bounded", "archive", True,
        "Loadable pilot artifact; tiny pilot results are not performance evidence.",
        "runs/archive/e2e-pilot-bounded",
    ),
    "xception-focal-legacy": ModelSpec(
        "xception-focal-legacy", "models/archive/xception_focal_legacy/model_new_focal_93.pth",
        "legacy-unverified", False,
        "Architecture-loadable, but class direction, preprocessing, and calibration are unknown.",
        "old_models_archive/models/checkpoints/model_new_focal_93.pth",
    ),
}

MODEL_REGISTRY = MappingProxyType(_SPECS)

# Stable model IDs are preferred; these aliases preserve earlier CLI/API inputs.
MODEL_ALIASES = MappingProxyType({
    "xcep-clean-s42": "xception-clean",
    "best_xception/xcep-clean-s42": "xception-clean",
    "xcep-mixed-robust-s42": "xception-mixed-robust",
    "archive/xcep-mixed-robust-s42": "xception-mixed-robust",
    "xcep-final-v1-s42-STACKED-REGULARIZATION-FAILED": "xception-final-v1",
    "archive/xcep-final-v1-s42-STACKED-REGULARIZATION-FAILED": "xception-final-v1",
    "archive/b0-ffpp-basic": "b0-ffpp-basic",
    "archive/b0-ffpp-robust": "b0-ffpp-robust",
    "trained_models/b0-mixed-robust": "b0-mixed-robust",
    "archive/e2e-pilot-bounded": "e2e-pilot-bounded",
})


def model_spec(identifier: str) -> ModelSpec | None:
    """Return registered metadata, accepting documented legacy aliases."""
    canonical = MODEL_ALIASES.get(identifier, identifier)
    return MODEL_REGISTRY.get(canonical)


def resolve_model_path(identifier: str, workspace: Workspace | None = None) -> Path:
    """Resolve a registered artifact or a custom run path inside ``runs/``."""
    ws = workspace or Workspace()
    spec = model_spec(identifier)
    if spec is not None:
        if not spec.inference_enabled:
            raise ValueError(
                f"Model '{spec.identifier}' is not enabled for inference: {spec.description}"
            )
        return ws.output(ws.root / spec.artifact_path)
    return ws.run(identifier)


# Compatibility name for existing API and tools; new code should say model path.
resolve_model_run = resolve_model_path


def registered_models() -> tuple[ModelSpec, ...]:
    """Return model specifications in registry order."""
    return tuple(MODEL_REGISTRY.values())
