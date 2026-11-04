"""FAKE-positive metrics; model logits retain ImageFolder fake=0, real=1."""
import math
import numpy as np
from sklearn.metrics import (accuracy_score, auc, average_precision_score, brier_score_loss,
                             classification_report, confusion_matrix, precision_recall_curve,
                             precision_recall_fscore_support, roc_auc_score, roc_curve)


def real_probability(logits, temperature=1):
    if not math.isfinite(float(temperature)) or temperature <= 0:
        raise ValueError("Temperature must be finite and positive")
    logits = np.asarray(logits, dtype=float) / temperature
    if not np.isfinite(logits).all():
        raise ValueError("Nonfinite logits")
    # Stable sigmoid without overflow warnings.
    return np.exp(-np.logaddexp(0, -logits))


def _arrays(labels, real_scores):
    labels, scores = np.asarray(labels), np.asarray(real_scores, dtype=float)
    if labels.ndim != 1 or scores.shape != labels.shape or not len(labels):
        raise ValueError("Labels/scores must be nonempty matching 1-D arrays")
    if not np.isin(labels, [0, 1]).all() or not np.isfinite(scores).all() or np.any((scores < 0) | (scores > 1)):
        raise ValueError("Invalid labels or nonfinite/out-of-range scores")
    return labels, scores


def compute_metrics(labels, real_scores, *, threshold=0.5, bins=10):
    labels, scores = _arrays(labels, real_scores)
    if not 0 <= threshold <= 1:
        raise ValueError("Fake decision threshold must be in [0,1]")
    fake_scores = 1 - scores
    predictions = np.where(fake_scores >= threshold, 0, 1)
    true_fake = (labels == 0).astype(int)
    precision, recall, f1, support = precision_recall_fscore_support(labels, predictions, labels=[0, 1], zero_division=0)
    cm = confusion_matrix(labels, predictions, labels=[0, 1])
    reliability = []
    ece = 0.0
    bin_ids = np.minimum((fake_scores * bins).astype(int), bins - 1)
    for index in range(bins):
        mask = bin_ids == index
        if np.any(mask):
            predicted, actual = float(fake_scores[mask].mean()), float(true_fake[mask].mean())
            ece += float(mask.mean()) * abs(predicted - actual)
            reliability.append({"bin": index, "count": int(mask.sum()), "mean_fake_score": predicted, "observed_fake_rate": actual})
    result = {"axis_labels": ["FAKE", "REAL"], "confusion_matrix": cm.tolist(), "threshold": float(threshold),
              "samples": len(labels), "fake_prevalence": float(true_fake.mean()), "accuracy": float(accuracy_score(labels, predictions)),
              "balanced_accuracy": float(np.mean(recall[support > 0])), "fake_precision": float(precision[0]),
              "fake_recall": float(recall[0]), "fake_f1": float(f1[0]), "real_precision": float(precision[1]),
              "real_recall": float(recall[1]), "real_f1": float(f1[1]), "brier_score": float(brier_score_loss(true_fake, fake_scores)),
              "ece_binary_fake_probability": ece, "reliability": reliability,
              "classification_report": classification_report(labels, predictions, labels=[0, 1], target_names=["FAKE", "REAL"], output_dict=True, zero_division=0),
              "limitations": [], "roc_auc": None, "fake_average_precision": None, "fake_pr_auc_trapezoid": None,
              "fake_recall_at_real_fpr_1pct": None, "fake_recall_at_real_fpr_5pct": None, "roc_curve": None, "pr_curve": None}
    if len(np.unique(labels)) < 2:
        result["limitations"].append("single_class")
        return result
    fpr, tpr, roc_thresholds = roc_curve(true_fake, fake_scores)
    pr_precision, pr_recall, pr_thresholds = precision_recall_curve(true_fake, fake_scores)
    result.update({"roc_auc": float(roc_auc_score(true_fake, fake_scores)),
                   "fake_average_precision": float(average_precision_score(true_fake, fake_scores)),
                   "fake_pr_auc_trapezoid": float(auc(pr_recall[::-1], pr_precision[::-1])),
                   "fake_recall_at_real_fpr_1pct": float(tpr[fpr <= 0.01].max()),
                   "fake_recall_at_real_fpr_5pct": float(tpr[fpr <= 0.05].max()),
                   "roc_curve": {"fpr": fpr.tolist(), "tpr": tpr.tolist(), "thresholds": [float(x) if np.isfinite(x) else None for x in roc_thresholds]},
                   "pr_curve": {"precision": pr_precision.tolist(), "recall": pr_recall.tolist(), "thresholds": pr_thresholds.tolist()}})
    return result


def metric_fixture():
    labels = [0] * 60 + [1] * 40
    scores = [0.1] * 45 + [0.9] * 15 + [0.1] * 4 + [0.9] * 36
    report = compute_metrics(labels, scores)
    expected = {"accuracy": 0.81, "fake_precision": 45 / 49, "fake_recall": 0.75, "real_recall": 0.9, "roc_auc": 0.825}
    for key, value in expected.items():
        if not math.isclose(report[key], value, abs_tol=1e-10):
            raise AssertionError(f"Metric fixture failed: {key}")
    if report["confusion_matrix"] != [[45, 15], [4, 36]]:
        raise AssertionError("Metric fixture axes are reversed")
    return report


def fit_temperature(labels, logits):
    import torch
    labels = np.asarray(labels)
    logits = np.asarray(logits, dtype=float)
    _arrays(labels, real_probability(logits))
    if len(np.unique(labels)) < 2:
        raise ValueError("Calibration requires both classes")
    values = torch.tensor(logits, dtype=torch.float64)
    targets = torch.tensor(labels, dtype=torch.float64)
    log_temperature = torch.zeros((), dtype=torch.float64, requires_grad=True)
    optimizer = torch.optim.LBFGS([log_temperature], lr=0.2, max_iter=50, line_search_fn="strong_wolfe")
    def closure():
        optimizer.zero_grad()
        loss = torch.nn.functional.binary_cross_entropy_with_logits(values / log_temperature.exp().clamp(0.05, 20), targets)
        loss.backward()
        return loss
    optimizer.step(closure)
    temperature = float(log_temperature.detach().exp().clamp(0.05, 20))
    if not math.isfinite(temperature):
        raise ValueError("Calibration produced a nonfinite temperature")
    return temperature


def choose_threshold(labels, real_scores, *, max_fpr=0.05):
    labels, scores = _arrays(labels, real_scores)
    if not 0 <= max_fpr <= 1 or len(np.unique(labels)) != 2:
        raise ValueError("Threshold calibration requires both classes and a valid false-alarm budget")
    fake = 1 - scores
    candidates = np.unique(np.r_[fake, 1.0])
    best = None
    for threshold in candidates:
        predictions = fake >= threshold
        fpr = float(predictions[labels == 1].mean())
        recall = float(predictions[labels == 0].mean())
        if fpr <= max_fpr:
            candidate = (recall, float(threshold), fpr)
            if best is None or candidate[:2] > best[:2]:
                best = candidate
    if best is None:
        raise ValueError("No threshold in [0,1] meets the calibration false-alarm budget")
    return {"threshold": best[1], "calibration_fake_recall": best[0], "calibration_real_fpr": best[2], "target_real_fpr": max_fpr}


def cluster_auc_interval(labels, real_scores, clusters, *, repetitions=200, seed=42):
    labels, scores = _arrays(labels, real_scores)
    if len(clusters) != len(labels):
        raise ValueError("Cluster IDs must match observations")
    unique = sorted(set(clusters))
    if len(unique) < 2:
        return {"clusters": len(unique), "interval": None, "fake_recall_at_fpr_1pct_interval": None,
                "fake_recall_at_fpr_5pct_interval": None, "limitation": "insufficient_independent_clusters"}
    indices = {group: np.where(np.asarray(clusters) == group)[0] for group in unique}
    generator = np.random.default_rng(seed)
    results = []
    tails = {"fake_recall_at_fpr_1pct": [], "fake_recall_at_fpr_5pct": []}
    for _ in range(repetitions):
        picked = generator.choice(unique, size=len(unique), replace=True)
        sample = np.concatenate([indices[group] for group in picked])
        if len(np.unique(labels[sample])) == 2:
            results.append(roc_auc_score((labels[sample] == 0).astype(int), 1 - scores[sample]))
            fpr, tpr, _ = roc_curve((labels[sample] == 0).astype(int), 1 - scores[sample])
            for key, budget in (("fake_recall_at_fpr_1pct", 0.01), ("fake_recall_at_fpr_5pct", 0.05)):
                tails[key].append(float(tpr[fpr <= budget].max()))
    return {"clusters": len(unique), "valid_repetitions": len(results), "seed": seed,
            "interval": np.quantile(results, [0.025, 0.975]).tolist() if results else None,
            **{key + "_interval": np.quantile(values, [0.025, 0.975]).tolist() if values else None for key, values in tails.items()},
            "limitation": "Connected known parent/person families are resampled together. Unknown person aliases may remain; small tail samples do not support precise population claims."}
