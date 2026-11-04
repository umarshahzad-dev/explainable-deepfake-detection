from deepfake_detection.models.registry import resolve_model_path
import csv
from io import StringIO
import os
import time
from pathlib import Path
from datetime import datetime, timezone
import uuid
import numpy as np
from torch.utils.data import DataLoader
from deepfake_detection.data.datasets import CropDataset, crop_rows
from deepfake_detection.data.manifests import check_splits, stratified_limit, dependency_families
from deepfake_detection.evaluation.metrics import compute_metrics, real_probability, fit_temperature, choose_threshold, cluster_auc_interval, metric_fixture
from deepfake_detection.models.model_loader import load_run_model
from deepfake_detection.common.paths import Workspace, atomic_bytes, digest_file, fingerprint, read_json, read_jsonl, write_json, write_jsonl
from deepfake_detection.training.trainer import predict_loader
from deepfake_detection.common.telemetry import MemoryMonitor
from deepfake_detection.common.progress import Progress
from deepfake_detection.evaluation.test_time_augmentation import tta_enabled,inference_augmented,predict_tta_dataset
from deepfake_detection.evaluation.calibration import fit_platt,fit_isotonic,apply_calibration,CalibratorEnsemble


def inference_predictions(model,dataset,cfg,*,device='cpu',progress=None):
    if inference_augmented(cfg): return predict_tta_dataset(model,dataset,cfg,device=device,progress=progress)
    return predict_loader(model,DataLoader(dataset,batch_size=cfg['training']['batch_size'],shuffle=False,num_workers=0),device=device,amp=cfg['training']['amp'],progress=progress,amp_dtype=cfg['training'].get('amp_dtype','float16'))


def audit_against_run(run_path, rows):
    """Include saved exposure so a new manifest cannot rename a trained parent."""
    document=read_json(run_path / "input_manifest.json")
    prior = document["rows"]
    cfg=read_json(run_path/'config.json') if (run_path/'config.json').is_file() else {}
    exposure=document.get('teacher_exposure',[])
    if cfg.get('training',{}).get('teacher_weights_sha256'):
        if not exposure or fingerprint(exposure)!=cfg['training'].get('teacher_lineage_sha256'):
            raise ValueError('Teacher exposure lineage missing or changed; held-out evaluation refused')
    if exposure:
        from deepfake_detection.models.pretraining import assert_teacher_separation
        assert_teacher_separation(exposure,[r for r in rows if r.get('split') in {'calibration','test','external'}])
    combined = {}
    for row in [*prior, *rows]:
        key = row["sample_id"]
        if key in combined and combined[key]["split"] != row["split"]:
            raise ValueError("Held-out sample was repartitioned after training")
        if key in combined:
            for field in ("parents", "known_person_ids", "sha256", "crop_sha256"):
                if combined[key].get(field) != row.get(field):
                    raise ValueError("Saved exposure provenance changed: " + field)
        combined[key] = row
    audit = check_splits(list(combined.values()))
    if not audit["ok"]:
        raise ValueError(f"Held-out data conflicts with saved exposure: {audit}")
    return audit


def calibration_rows(rows):
    selected = [r for r in rows if r.get("split") == "calibration"]
    if any(r.get("dataset") == "dfd" for r in selected):
        raise ValueError("DFD is external and cannot calibrate a model")
    if not selected or {r["label"] for r in selected} != {0, 1}:
        raise ValueError("Calibration requires independent calibration groups with both classes")
    return selected


def calibration_for(run_path, metadata,cfg=None):
    path = run_path / "calibration.json"
    if not path.exists():
        return {"temperature": 1.0, "threshold": 0.5, "applied": False, "limitation": "uncalibrated_model_scores",
                "calibration_version":None,"calibration_weights_sha256":None,"calibration_frozen_at":None}
    pointer=read_json(path)
    version=pointer.get("version")
    if not version or Path(version).name != version:
        raise ValueError("Calibration pointer invalid or legacy format; refit with --force")
    result = read_json(run_path/version)
    if result["weights_sha256"] != metadata["weights_sha256"]:
        raise ValueError("Calibration belongs to different model weights; refit on calibration groups")
    if cfg is not None:
        scoring_hash=fingerprint(cfg.get('inference',{}))
        if (result.get('inference_hash') is not None or inference_augmented(cfg)) and result.get('inference_hash')!=scoring_hash:
            raise ValueError('Calibration inference/TTA contract differs; refit calibration with the current views')
    return {**result, "applied": True}


def calibrate(name, manifests, workspace=None, *, device="cpu", limit=None,force=False):
    ws = workspace or Workspace()
    model, cfg, metadata = load_run_model(resolve_model_path(name, ws), device=device, workspace=ws)
    started=time.perf_counter()
    pointer_path=resolve_model_path(name, ws)/"calibration.json"
    if pointer_path.exists() and not force:
        pointer=read_json(pointer_path)
        if pointer.get("weights_sha256")==metadata["weights_sha256"]:
            raise ValueError("Calibration already exists for these weights; use --force to create a new version")
    rows = calibration_rows(crop_rows(manifests, ws))
    audit_against_run(resolve_model_path(name, ws), rows)
    if limit:
        rows = stratified_limit(rows, limit, cfg["seed"])
        rows = calibration_rows(rows)
    dataset = CropDataset(rows, cfg, workspace=ws)
    predictions = inference_predictions(model,dataset,cfg,device=device)
    raw_scores=real_probability(predictions['logits'])
    method=cfg.get('inference',{}).get('calibration_method','temperature')
    if method=='temperature': fitted={'method':method,'temperature':fit_temperature(predictions['labels'],predictions['logits'])}
    elif method=='platt': fitted=fit_platt(predictions['labels'],raw_scores)
    elif method=='isotonic': fitted=fit_isotonic(predictions['labels'],raw_scores)
    elif method=='median': fitted=CalibratorEnsemble().fit(predictions['labels'],raw_scores)
    else: raise ValueError(f'Unknown calibration method: {method}')
    temperature=fitted.get('temperature',1.)
    scores = apply_calibration(raw_scores,fitted)
    selected = choose_threshold(predictions["labels"], scores)
    result = {**fitted,"temperature": temperature, **selected, "inference_hash":fingerprint(cfg.get('inference',{})),
              "weights_sha256": metadata["weights_sha256"], "calibration_manifest_hash": fingerprint(rows),
              "samples": len(rows), "split": "calibration", "pilot": limit is not None or metadata.get("pilot", False) or cfg.get("pilot", False) or any(r.get("pilot") for r in rows),
              "before": compute_metrics(predictions["labels"], real_probability(predictions["logits"])),
              "after": compute_metrics(predictions["labels"], scores, threshold=selected["threshold"])}
    timestamp=datetime.now(timezone.utc)
    version="calibration-"+timestamp.strftime("%Y%m%dT%H%M%S.%fZ")+"-"+uuid.uuid4().hex[:8]+".json"
    result.update(calibration_version=version,calibration_weights_sha256=metadata["weights_sha256"],
                  calibration_frozen_at=timestamp.isoformat(),elapsed_seconds=time.perf_counter()-started)
    if pointer_path.exists():
        old=read_json(pointer_path)
        if "version" not in old:
            write_json(resolve_model_path(name, ws)/("calibration-legacy-"+fingerprint(old)[:16]+".json"),old,ws)
    write_json(resolve_model_path(name, ws)/version,result,ws)
    write_json(pointer_path,{"version":version,"weights_sha256":metadata["weights_sha256"],"frozen_at":result["calibration_frozen_at"]},ws)
    return result


def save_plots(metrics, output, workspace=None):
    ws = workspace or Workspace()
    output = ws.output(output)
    output.mkdir(parents=True, exist_ok=True)
    cache = ws.output(ws.runs / ".cache" / "matplotlib")
    cache.mkdir(parents=True, exist_ok=True)
    os.environ["MPLCONFIGDIR"] = str(cache)
    import matplotlib
    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots()
    curve = metrics["roc_curve"]
    if curve:
        ax.plot(curve["fpr"], curve["tpr"], label=f"AUC={metrics['roc_auc']:.3f}")
        ax.legend()
    ax.plot([0, 1], [0, 1], "--", color="grey")
    ax.set(xlabel="Real false-alarm rate", ylabel="Fake recall", title="FAKE-positive ROC")
    fig.savefig(output / "roc.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    fig, ax = plt.subplots()
    curve = metrics["pr_curve"]
    if curve:
        ax.plot(curve["recall"], curve["precision"])
    ax.axhline(metrics["fake_prevalence"], linestyle="--", color="grey")
    ax.set(xlabel="Fake recall", ylabel="Fake precision", title="Precision–recall")
    fig.savefig(output / "pr.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    fig, ax = plt.subplots()
    points = metrics["reliability"]
    ax.plot([p["mean_fake_score"] for p in points], [p["observed_fake_rate"] for p in points], "o-")
    ax.plot([0, 1], [0, 1], "--", color="grey")
    ax.set(xlabel="Mean FAKE score", ylabel="Observed fake rate", title="Reliability")
    fig.savefig(output / "calibration.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    fig, ax = plt.subplots()
    cm = np.asarray(metrics["confusion_matrix"])
    ax.imshow(cm, cmap="Blues")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center")
    ax.set(xticks=[0, 1], yticks=[0, 1], xticklabels=["FAKE", "REAL"], yticklabels=["FAKE", "REAL"], xlabel="Predicted", ylabel="True")
    fig.savefig(output / "confusion_matrix.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def _summaries(rows, labels, scores, threshold):
    datasets, methods = {}, {}
    labels, scores = np.asarray(labels), np.asarray(scores)
    for dataset in sorted({r["dataset"] for r in rows}):
        mask = np.array([r["dataset"] == dataset for r in rows])
        datasets[dataset] = compute_metrics(labels[mask], scores[mask], threshold=threshold)
        for method in sorted({r["method"] for r in rows if r["dataset"] == dataset and r["label"] == 0}):
            subset = np.array([r["dataset"] == dataset and (r["label"] == 1 or r["method"] == method) for r in rows])
            methods[dataset + "/" + method] = compute_metrics(labels[subset], scores[subset], threshold=threshold)
    return datasets, methods


def evaluate(name, manifests, workspace=None, *, device="cpu", limit=None, plots=True, bootstrap=200, report_name=None, variant="clean",compare_run=None):
    ws = workspace or Workspace()
    metric_fixture()
    model, cfg, metadata = load_run_model(resolve_model_path(name, ws), device=device, workspace=ws)
    all_rows = crop_rows(manifests, ws)
    audit = audit_against_run(resolve_model_path(name, ws), all_rows)
    if not audit["ok"]:
        raise ValueError(f"Evaluation split check failed: {audit}")
    rows = [r for r in all_rows if r["split"] in {"test", "external"}]
    if limit:
        rows = stratified_limit(rows, limit, cfg["seed"])
    if not rows:
        raise ValueError("No held-out test/external crops found")
    if variant not in {"clean", "jpeg60", "resize_half"}:
        raise ValueError("Variant must be clean, jpeg60 or resize_half")
    dataset = CropDataset(rows, cfg, workspace=ws)
    if variant != "clean":
        dataset.augment = Corruption(variant)
    calibration = calibration_for(resolve_model_path(name, ws), metadata,cfg)
    started = time.perf_counter()
    with MemoryMonitor(device) as memory:
        predicted = inference_predictions(model,dataset,cfg,device=device,progress=Progress("Evaluated", len(rows), unit="faces"))
    scores = apply_calibration(real_probability(predicted['logits']),calibration)
    threshold = calibration["threshold"]
    datasets, methods = _summaries(rows, predicted["labels"], scores, threshold)
    report = {"run_id": name, "weights_sha256": metadata["weights_sha256"], "manifest_hash": fingerprint(rows),
              "memory": memory.report(),
              "preprocessing_hash": metadata["preprocessing_hash"], "mode": "extracted_crops", "variant": variant,
              "calibration": calibration, "pilot": limit is not None or metadata.get("pilot", False) or calibration.get("pilot", False) or cfg.get("pilot", False) or any(r.get("pilot") for r in rows),
              **{key:calibration.get(key) for key in ("calibration_version","calibration_weights_sha256","calibration_frozen_at")},
              "overall": compute_metrics(predicted["labels"], scores, threshold=threshold), "per_dataset": datasets, "per_method": methods,
              "auc_cluster_interval": cluster_auc_interval(predicted["labels"], scores, dependency_families(rows), repetitions=bootstrap, seed=cfg["seed"]),
              "elapsed_seconds": time.perf_counter() - started, "split_audit": audit,
              "per_demographic": None, "limitations": ["No demographic annotations supplied", "Crop accuracy does not include detector coverage"]}
    report["seconds_per_face"] = report["elapsed_seconds"] / len(rows)
    if cfg.get('evaluation',{}).get('compute_per_demographic'):
        from deepfake_detection.evaluation.analysis import demographic_summaries
        report['per_demographic']=demographic_summaries(rows,predicted['labels'],scores,threshold=threshold)
        report['limitations'].remove('No demographic annotations supplied')
        report['limitations'].append('Demographic strata pool evaluation domains and may be small/confounded; annotations are supplied, not inferred')
    if cfg.get('evaluation',{}).get('compute_adversarial'):
        from deepfake_detection.evaluation.analysis import adversarial_robustness
        report['adversarial_robustness']=adversarial_robustness(model,DataLoader(dataset,batch_size=cfg['training']['batch_size'],shuffle=False,num_workers=0),cfg.get('evaluation',{}).get('adversarial_eps_list',[.01,.02,.05]),mean=cfg['model']['mean'],std=cfg['model']['std'])
    if cfg.get('evaluation',{}).get('compute_fairness'):
        from deepfake_detection.evaluation.analysis import fairness_metrics
        decisions=[{'label':label,'prediction':0 if 1-score>=threshold else 1} for label,score in zip(predicted['labels'],scores)]
        report['fairness']=fairness_metrics(decisions,[row.get('demographics',{}) for row in rows])
    if compare_run is not None or cfg.get('evaluation',{}).get('compute_significance'):
        if compare_run is None: report['significance']={'status':'not_performed','reason':'An explicit independent reference run is required; no comparator selected from test performance'}
        else:
            from deepfake_detection.evaluation.analysis import paired_bootstrap_test,delong_test
            other_model,other_cfg,other_metadata=load_run_model(resolve_model_path(compare_run, ws),device=device,workspace=ws)
            if other_metadata['preprocessing_hash']!=metadata['preprocessing_hash']: raise ValueError('Paired comparison requires a shared crop preprocessing contract')
            audit_against_run(resolve_model_path(compare_run, ws),all_rows)
            other_dataset=CropDataset(rows,other_cfg,workspace=ws)
            if variant!='clean': other_dataset.augment=Corruption(variant)
            other_calibration=calibration_for(resolve_model_path(compare_run, ws),other_metadata,other_cfg)
            other_prediction=inference_predictions(other_model,other_dataset,other_cfg,device=device)
            if other_prediction['labels']!=predicted['labels']: raise ValueError('Paired comparison row order differs')
            other_scores=apply_calibration(real_probability(other_prediction['logits']),other_calibration)
            paired=paired_bootstrap_test(predicted['labels'],scores,other_scores,n=cfg.get('evaluation',{}).get('significance_resamples',10000),seed=cfg['seed'],clusters=dependency_families(rows))
            enough=min(predicted['labels'].count(0),predicted['labels'].count(1))>=2
            report['significance']={'reference_run':compare_run,'reference_weights_sha256':other_metadata['weights_sha256'],'paired_bootstrap':paired,'delong_exploratory':delong_test(predicted['labels'],scores,other_scores) if enough else {'p_value':None,'limitation':'fewer_than_two_examples_per_class'},'limitations':['Reference must be prespecified; repeated comparisons need multiplicity control.']}
    output = ws.report(report_name or name + "-evaluation")
    report["report_path"]=str(output)
    if cfg.get('evaluation',{}).get('save_failure_cases'):
        from deepfake_detection.evaluation.analysis import save_failure_cases
        report['failure_cases']=save_failure_cases(rows,scores,predicted['labels'],output/'failure_cases',cfg.get('evaluation',{}).get('failure_top_k',50),threshold=threshold,workspace=ws)
    write_json(output / "metrics.json", report, ws)
    stream = StringIO(newline="")
    fields = ["sample_id", "dataset", "method", "label", "prediction", "score_real", "score_fake", "cluster", "crop_path"]
    writer = csv.DictWriter(stream, fieldnames=fields)
    writer.writeheader()
    for row, score in zip(rows, scores):
        item = {k: row[k] for k in ("sample_id", "dataset", "method", "label", "cluster", "crop_path")}
        item.update({"prediction": 0 if 1 - score >= threshold else 1, "score_real": float(score), "score_fake": float(1 - score)})
        writer.writerow(item)
    atomic_bytes(output / "predictions.csv", stream.getvalue().encode("utf-8"), ws)
    if plots:
        save_plots(report["overall"], output, ws)
    return report


class Corruption:
    """Deterministic test variants generated only after partitions are frozen."""
    def __init__(self, variant):
        self.variant = variant

    def __call__(self, image):
        from io import BytesIO
        from PIL import Image
        if self.variant == "jpeg60":
            stream = BytesIO()
            image.save(stream, format="JPEG", quality=60)
            with Image.open(BytesIO(stream.getvalue())) as decoded:
                return decoded.convert("RGB")
        return image.resize((image.width // 2, image.height // 2), Image.Resampling.BICUBIC).resize(image.size, Image.Resampling.BICUBIC)


def evaluate_raw(name, manifests, workspace=None, *, engine=None, reader_factory=None, device=None,
                 limit_videos=None, max_frames=None, report_name=None, plots=True):
    """Evaluate still frames through the serving contract; abstention is never hidden."""
    from collections import Counter
    from deepfake_detection.data.extraction import VideoReader, sample_indices
    from deepfake_detection.inference.engine import InferenceEngine
    ws = workspace or Workspace()
    engine = engine or InferenceEngine.from_run(name, ws, device=device)
    reader_factory = reader_factory or VideoReader
    all_rows = [row for source in manifests for row in read_jsonl(ws.manifest(source) / "videos.jsonl")]
    verified_frames = {}
    for source in manifests:
        path = ws.manifest(source) / "crops.jsonl"
        if path.exists():
            for crop in read_jsonl(path):
                verified_frames[(crop["video_id"], crop["frame_index"])] = crop
    audit = audit_against_run(resolve_model_path(name, ws), all_rows)
    if not audit["ok"]:
        raise ValueError(f"Raw evaluation split check failed: {audit}")
    rows = [r for r in all_rows if r["split"] in {"test", "external"}]
    if limit_videos:
        rows = stratified_limit(rows, limit_videos, engine.cfg["seed"])
    if not rows:
        raise ValueError("No held-out raw videos found")
    abstentions = Counter()
    records, classified, labels, real_scores, errors = [], [], [], [], []
    attempted, with_faces = 0, 0
    started = time.perf_counter()
    memory = MemoryMonitor(getattr(engine, "device", "cpu")).start()
    frame_target = sum(engine.cfg["extraction"]["dfd_frames"] if r["dataset"] == "dfd" else engine.cfg["extraction"]["eval_real" if r["label"] else "eval_fake"] for r in rows)
    feedback = Progress("Evaluated", min(frame_target, max_frames) if max_frames else frame_target, unit="candidate frames")
    for row in rows:
        if max_frames is not None and attempted >= max_frames:
            break
        try:
            if digest_file(row["raw_path"]) != row["sha256"]:
                raise ValueError("Source video bytes changed")
            with reader_factory(row["raw_path"]) as reader:
                quota = engine.cfg["extraction"]["dfd_frames"] if row["dataset"] == "dfd" else engine.cfg["extraction"]["eval_real" if row["label"] else "eval_fake"]
                targets = set(sample_indices(reader.count, quota))
                targets.update(index for (video, index), crop in verified_frames.items()
                               if video == row["sample_id"] and crop.get("verification_source") == "auto_single_face_heuristic")
                for index, image in reader.frames(sorted(targets)):
                    if max_frames is not None and attempted >= max_frames:
                        break
                    attempted += 1
                    feedback.update(attempted)
                    result = engine.analyze(image, explain=False)
                    faces = result["faces"]
                    with_faces += bool(faces)
                    record = {"video_id": row["sample_id"], "frame_index": index, "dataset": row["dataset"], "label": row["label"], "faces": faces}
                    records.append(record)
                    # No all-faces-fake assumption for an image with a video-level label.
                    if len(faces) != 1:
                        abstentions["no_faces" if not faces else "ambiguous_multiple_faces"] += 1
                    elif faces[0]["status"] != "accepted":
                        abstentions["low_quality"] += 1
                    elif row["label"] == 0 and not raw_frame_verified(row, index, verified_frames):
                        abstentions["subject_unverified"] += 1
                    else:
                        labels.append(row["label"])
                        real_scores.append(faces[0]["real_score"])
                        classified.append({**row, "sample_id": f"{row['sample_id']}:f{index}", "video_id": row["sample_id"], "frame_index": index})
        except (OSError, ValueError, RuntimeError) as exc:
            errors.append({"sample_id": row["sample_id"], "error": f"{type(exc).__name__}: {exc}"})
    threshold = engine.calibration["threshold"]
    memory.stop()
    feedback.finish(attempted)
    overall = compute_metrics(labels, real_scores, threshold=threshold) if labels else None
    datasets, methods = _summaries(classified, labels, real_scores, threshold) if labels else ({}, {})
    report = {"run_id": name, "weights_sha256": engine.metadata["weights_sha256"],
              **{key:engine.calibration.get(key) for key in ("calibration_version","calibration_weights_sha256","calibration_frozen_at")},
              "memory": memory.report(),
              "auc_cluster_interval": cluster_auc_interval(labels, real_scores, dependency_families(classified), seed=engine.cfg["seed"]) if labels else None,
              "preprocessing_hash": engine.metadata["preprocessing_hash"], "manifest_hash": fingerprint(rows),
              "mode": "raw_still_frames", "frames_attempted": attempted, "frames_with_detected_faces": with_faces,
              "classified_frames": len(labels), "detection_coverage": with_faces / attempted if attempted else 0,
              "classification_coverage": len(labels) / attempted if attempted else 0, "abstentions": dict(abstentions),
              "errors": errors, "overall": overall, "per_dataset": datasets, "per_method": methods,
              "split_audit": audit, "elapsed_seconds": time.perf_counter() - started,
              "pilot": limit_videos is not None or max_frames is not None or engine.metadata.get("pilot", False) or engine.calibration.get("pilot", False) or any(r.get("pilot") for r in [*rows, *verified_frames.values()]),
              "limitations": ["Fake labels require reviewed manipulated-subject provenance and a single accepted face", "This is still-image detection, not temporal video inference"]}
    output = ws.report(report_name or name + "-raw-evaluation")
    report["report_path"]=str(output)
    write_json(output / "metrics.json", report, ws)
    write_jsonl(output / "raw_results.jsonl", records, ws)
    if plots and overall:
        save_plots(overall, output, ws)
    return report


def raw_frame_verified(row, index, verified_frames):
    crop = verified_frames.get((row["sample_id"], index))
    evidence = crop if crop is not None else row
    source = evidence.get("verification_source")
    if crop is None and source == "auto_single_face_heuristic":
        return False  # A frame-level heuristic cannot certify an entire video.
    return bool(evidence.get("subject_verified") and source in {"auto_single_face_heuristic", "manual_registry"}
                and evidence.get("sha256") == row.get("sha256"))
