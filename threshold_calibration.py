"""Calibrate on a fixed internal subset, then score on disjoint validation data."""
from __future__ import annotations

import hashlib
import math
import numpy as np

from metrics import binary_metrics


def _arrays(labels, probabilities=None):
    y = np.asarray(labels)
    if y.ndim != 1 or len(y) == 0 or not np.isin(y, [0, 1]).all():
        raise ValueError("Expected nonempty one-dimensional binary labels")
    y = y.astype(np.int64)
    if set(y.tolist()) != {0, 1}:
        raise ValueError("Both real and AI labels are required for calibration")
    if probabilities is None:
        return y
    p = np.asarray(probabilities, dtype=np.float64)
    if p.shape != y.shape or not np.isfinite(p).all() or ((p < 0) | (p > 1)).any():
        raise ValueError("Expected one finite probability in [0, 1] per label")
    return y, p


def fixed_calibration_split(labels, sample_ids, fraction=0.25, seed=42):
    """Stable by sample ID, independent of row order, epoch and predictions.

    With 4,000 balanced images, returns exactly 1,000 calibration / 3,000
    evaluation images; each partition is balanced. Smoke sets scale down.
    """
    y = _arrays(labels)
    ids = [str(s) for s in sample_ids]
    if len(ids) != len(y) or len(set(ids)) != len(ids):
        raise ValueError("Calibration needs one UNIQUE sample ID per image")
    fraction = float(fraction)
    if not math.isfinite(fraction) or not 0 < fraction < 1:
        raise ValueError("calibration.fraction must be strictly between 0 and 1")
    calibration, validation = [], []
    for label in (0, 1):
        indices = np.flatnonzero(y == label).tolist()
        if len(indices) < 2:
            raise ValueError("Need at least two images of each class for a disjoint split")
        indices.sort(key=lambda i: (hashlib.sha256(f"{seed}:{ids[i]}".encode()).hexdigest(), ids[i]))
        count = min(len(indices) - 1, max(1, round(len(indices) * fraction)))
        calibration.extend(indices[:count])
        validation.extend(indices[count:])
    return np.asarray(calibration, dtype=np.int64), np.asarray(validation, dtype=np.int64)


def split_fingerprint(sample_ids, indices):
    selected = sorted(str(sample_ids[i]) for i in indices)
    return hashlib.sha256("\n".join(selected).encode()).hexdigest()


def select_threshold(labels, probabilities, default_threshold=0.5):
    """Maximize balanced accuracy using calibration labels ONLY.

    Search decision boundaries in O(n log n), keeping >= semantics consistent
    with metrics.py. Equal optima prefer the threshold closest to 0.5.
    """
    y, p = _arrays(labels, probabilities)
    default_threshold = float(default_threshold)
    if not math.isfinite(default_threshold) or not 0 <= default_threshold <= 1:
        raise ValueError("Default threshold must be in [0, 1]")
    unique = np.unique(p)
    mids = unique[:-1] + (unique[1:] - unique[:-1]) / 2
    candidates = np.unique(np.concatenate(([0.0, default_threshold, 1.0], mids)))
    order = np.argsort(p, kind="stable")
    positives_before = np.concatenate(([0], np.cumsum(y[order])))
    cutoffs = np.searchsorted(p[order], candidates, side="left")
    false_negatives = positives_before[cutoffs]
    true_negatives = cutoffs - false_negatives
    n_positive, n_negative = int(y.sum()), int((1 - y).sum())
    scores = 0.5 * ((n_positive - false_negatives) / n_positive + true_negatives / n_negative)
    best = np.flatnonzero(np.isclose(scores, scores.max(), atol=1e-12, rtol=0))
    winner = best[np.argmin(np.abs(candidates[best] - default_threshold))]
    return float(candidates[winner]), float(scores[winner])


def calibrated_validation(labels, probabilities, sample_ids, settings, default_threshold=0.5):
    y, p = _arrays(labels, probabilities)
    calibration, validation = fixed_calibration_split(
        y, sample_ids, fraction=settings.get("fraction", 0.25), seed=settings.get("seed", 42)
    )
    threshold, calibration_score = select_threshold(y[calibration], p[calibration], default_threshold)
    result = binary_metrics(y[validation], p[validation], threshold)
    baseline = binary_metrics(y[validation], p[validation], 0.5)
    pred = p[validation] >= threshold
    val_y = y[validation]
    result.update({
        "threshold": threshold,
        "accuracy_at_0_5": baseline["accuracy"],
        "f1_at_0_5": baseline["f1"],
        "balanced_accuracy": float(0.5 * (pred[val_y == 1].mean() + (~pred[val_y == 0]).mean())),
        "calibration": {
            "metric": "balanced_accuracy", "count": len(calibration),
            "fraction": float(settings.get("fraction", 0.25)),
            "seed": int(settings.get("seed", 42)),
            "balanced_accuracy": calibration_score,
            "calibration_ids_sha256": split_fingerprint(sample_ids, calibration),
            "validation_ids_sha256": split_fingerprint(sample_ids, validation),
        },
    })
    return result, validation


def resolve_evaluation_threshold(evaluation, checkpoint):
    source = evaluation.get("threshold_source", "config")
    if source == "checkpoint":
        if "decision_threshold" not in checkpoint:
            raise ValueError("This checkpoint has no calibrated threshold. For an OLD model, use its "
                             "old config or set evaluation.threshold_source: config explicitly.")
        threshold = float(checkpoint["decision_threshold"])
    elif source == "config":
        threshold = float(evaluation.get("threshold", 0.5))
    else:
        raise ValueError("evaluation.threshold_source must be config or checkpoint")
    if not math.isfinite(threshold) or not 0 <= threshold <= 1:
        raise ValueError("Evaluation threshold must be finite and in [0, 1]")
    return threshold
