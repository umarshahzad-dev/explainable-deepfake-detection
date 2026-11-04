"""Strictly local pretrained initialization and exported-model loading."""
from pathlib import Path
import logging
from deepfake_detection.common.paths import ROOT, Workspace, atomic_bytes, digest_file, read_json, write_json, fingerprint
from deepfake_detection.common.reproducibility import seed_all

B0_SHA = "d569899762ea9b1384ee07f4af64805cf8caa1c55f9253ebb1080dc40e87a2cd"
B0_SIZE = 21355344


def verify_weights(path, *, expected_size=B0_SIZE, expected_sha=B0_SHA):
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Offline weight file missing: {path}; downloads are disabled")
    size = path.stat().st_size
    if expected_size is not None and size != expected_size:
        raise ValueError(f"Weight size mismatch: {size} != {expected_size}")
    sha = digest_file(path)
    if expected_sha is not None and sha.lower() != expected_sha.lower():
        raise ValueError("Pretrained weight checksum mismatch")
    return {"path": str(path.absolute()), "bytes": size, "sha256": sha}


def architecture(model_cfg, *, enhancements=True):
    import timm
    from deepfake_detection.models.architecture import alternative_name,create_alternative_model
    if alternative_name(model_cfg['name']): return create_alternative_model(model_cfg)
    if model_cfg["name"] not in {"efficientnet_b0.ra_in1k", "legacy_xception.tf_in1k"}:
        raise ValueError("Only the plan's B0 and explicit offline Xception baseline are supported")
    base=timm.create_model(model_cfg["name"], pretrained=False, num_classes=1, drop_rate=model_cfg["dropout"])
    if not enhancements: return base
    from deepfake_detection.models.architecture import enhance
    return enhance(base,model_cfg)


def _local_state(path):
    import torch
    if Path(path).suffix == ".safetensors":
        from safetensors.torch import load_file
        return load_file(str(path), device="cpu")
    return torch.load(path, map_location="cpu", weights_only=True)


def initialize_model(cfg, *, device="cpu"):
    import torch
    from deepfake_detection.models.pretraining import source_weights
    source,path=source_weights(cfg)
    legacy = (ROOT / ".old-project" / "models" / "checkpoints").resolve()
    if path.resolve().is_relative_to(legacy):
        raise ValueError("Legacy fine-tuned project checkpoints cannot initialize a new experiment")
    name = cfg["model"]["name"]
    seed_all(cfg["seed"])
    from deepfake_detection.models.architecture import alternative_name
    bare_model=None
    if name == "efficientnet_b0.ra_in1k":
        if source=='imagenet': verify_weights(path)
        else: verify_weights(path,expected_size=None,expected_sha=cfg.get('pretraining',{}).get(source+'_sha256'))
        classifier = "classifier"
    elif name == "legacy_xception.tf_in1k":
        verify_weights(path, expected_size=None, expected_sha=cfg.get('pretraining',{}).get(source+'_sha256') or cfg['model'].get('pretrained_sha256'))
        if cfg["model"]["input_size"] != 299 or cfg["model"]["mean"] != [0.5] * 3 or cfg["model"]["std"] != [0.5] * 3:
            raise ValueError("Xception requires the plan's 299 input and mean/std 0.5")
        classifier = "fc"
    elif alternative_name(name):
        verify_weights(path,expected_size=None,expected_sha=cfg.get('pretraining',{}).get(source+'_sha256',cfg['model'].get('pretrained_sha256')))
        import timm
        defaults=timm.get_pretrained_cfg(alternative_name(name))
        if tuple(cfg['model']['mean'])!=defaults.mean or tuple(cfg['model']['std'])!=defaults.std:
            raise ValueError('Alternative normalization must match its locally registered timm pretrained config')
        bare_model=architecture(cfg['model'],enhancements=False)
        selected=bare_model.get_classifier()
        classifier=next((key for key,module in bare_model.named_modules() if module is selected),None)
        if not classifier or not hasattr(selected,'weight'): raise ValueError('Alternative must expose its actual linear classifier module')
    else:
        raise ValueError("Unsupported model")
    state = _local_state(path)
    if not isinstance(state, dict) or "model_state_dict" in state:
        raise ValueError("Initialization requires raw ImageNet weights, not a training-resume checkpoint")
    head_key = classifier + ".weight"
    if head_key not in state or state[head_key].ndim!=2 or (state[head_key].shape[0] != 1000 if source=='imagenet' else state[head_key].shape[0]<2):
        raise ValueError("Initialization must have the source pretraining classification head (ImageNet:1000); binary deepfake exports cannot initialize it")
    if not all(bool(torch.isfinite(v).all()) for v in state.values()):
        raise ValueError("Nonfinite initialization weights")
    model = bare_model if bare_model is not None else architecture(cfg["model"],enhancements=False)
    backbone = {k: v for k, v in state.items() if k not in {head_key, classifier + ".bias"}}
    result = model.load_state_dict(backbone, strict=False)
    if set(result.missing_keys) != {head_key, classifier + ".bias"} or result.unexpected_keys:
        raise ValueError(f"Backbone mismatch: {result}")
    from deepfake_detection.models.architecture import enhance
    if bare_model is not None: return model.to(device)
    return enhance(model,cfg['model']).to(device)


def export_weights(model, path, workspace=None):
    from safetensors.torch import save
    state = {k: t.detach().cpu().contiguous() for k, t in model.state_dict().items()}
    atomic_bytes(path, save(state), workspace)
    return digest_file(path)


def load_run_model(run_path, *, device="cpu", workspace=None):
    import torch
    ws = workspace or Workspace()
    run_path = ws.output(run_path)
    cfg = read_json(run_path / "config.json")
    metadata = artifact_metadata(run_path,cfg,ws)
    path = ws.output(run_path / metadata["weights_file"])
    state = _local_state(path)
    if not all(bool(torch.isfinite(v).all()) for v in state.values()):
        raise ValueError("Export contains nonfinite tensors")
    model = architecture(cfg["model"])
    model.load_state_dict(state, strict=True)
    return model.to(device).eval(), cfg, metadata


def artifact_metadata(run_path,cfg,workspace=None,*,phase=None):
    ws=workspace or Workspace()
    run_path=ws.output(run_path)
    prefix=f"{phase}_best" if phase else "best"
    try:
        metadata=read_json(run_path/(prefix+"_metadata.json"))
    except OSError as exc:
        raise ValueError(f"Artifact metadata missing: {run_path}/{prefix}_metadata.json") from exc
    from deepfake_detection.data.preprocessing import preprocessing_hash
    if metadata.get("preprocessing_hash") != preprocessing_hash(cfg) or metadata.get("model_transform_hash") != fingerprint(cfg["model"]):
        raise ValueError("Exported preprocessing/model contract does not match saved configuration")
    if cfg["model"]["class_map"] != {"fake": 0, "real": 1}:
        raise ValueError("Exported class map is incompatible")
    filename=metadata.get("weights_file", "")
    if not filename or Path(filename).name != filename:
        raise ValueError("Artifact generation must be a filename inside its run")
    path=ws.output(run_path/filename)
    try:
        verify_weights(path,expected_size=None,expected_sha=metadata["weights_sha256"])
    except (OSError,ValueError) as exc:
        raise ValueError(f"Artifact generation invalid: {path}; {exc}") from exc
    return metadata


def publish_best(model, run, cfg, metadata, workspace=None,*,phase=None):
    """Immutable weights plus a single atomic metadata commit; alias is convenience only."""
    from safetensors.torch import save
    from deepfake_detection.data.preprocessing import preprocessing_hash
    ws = workspace or Workspace()
    payload = save({k: t.detach().cpu().contiguous() for k, t in model.state_dict().items()})
    import hashlib
    sha = hashlib.sha256(payload).hexdigest()
    prefix=f"{phase}_best" if phase else "best"
    filename = prefix + "-" + sha + ".safetensors"
    atomic_bytes(run / filename, payload, ws)
    result = {**metadata, "weights_sha256": sha, "weights_file": filename,
              "preprocessing_hash": preprocessing_hash(cfg), "model_transform_hash": fingerprint(cfg["model"])}
    write_json(run / (prefix+"_metadata.json"), result, ws)
    try:
        atomic_bytes(run / (prefix+".safetensors"), payload, ws)
    except OSError as exc:
        logging.getLogger(__name__).warning("Artifact committed; convenience alias update failed: %s",exc)
    return result
