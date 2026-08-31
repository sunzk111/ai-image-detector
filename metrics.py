from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score


def binary_metrics(
    labels: list[float] | np.ndarray,
    probabilities: list[float] | np.ndarray,
    threshold: float = 0.5,
) -> dict[str, Any]:
    labels_array = np.asarray(labels, dtype=np.int64)
    probabilities_array = np.asarray(probabilities, dtype=np.float64)
    predictions = (probabilities_array >= threshold).astype(np.int64)
    metrics: dict[str, Any] = {
        "accuracy": float(accuracy_score(labels_array, predictions)),
        "f1": float(f1_score(labels_array, predictions, zero_division=0)),
        "auroc": None,
        "count": int(labels_array.size),
    }
    if np.unique(labels_array).size == 2:
        metrics["auroc"] = float(roc_auc_score(labels_array, probabilities_array))
    return metrics

