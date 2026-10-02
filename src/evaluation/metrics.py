"""Comprehensive evaluation metrics and stratified reporting suite."""

import numpy as np
from typing import Dict, Any, List, Optional
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    log_loss,
    confusion_matrix
)
from src.models.calibration import calculate_expected_calibration_error


def evaluate_predictions(
    y_true: np.ndarray,
    probs: np.ndarray,
    classes: np.ndarray,
    confidence_bins: bool = True
) -> Dict[str, Any]:
    """Calculate all required statistical and probabilistic metrics for a set of predictions."""
    y_true = np.asarray(y_true)
    probs = np.asarray(probs)
    classes = np.asarray(classes)

    preds_idx = np.argmax(probs, axis=1)
    y_pred = classes[preds_idx]

    acc = float(accuracy_score(y_true, y_pred))
    bal_acc = float(balanced_accuracy_score(y_true, y_pred))
    prec = float(precision_score(y_true, y_pred, average="macro", zero_division=0))
    rec = float(recall_score(y_true, y_pred, average="macro", zero_division=0))
    f1 = float(f1_score(y_true, y_pred, average="macro", zero_division=0))

    # Multi-class Log Loss
    ll = float(log_loss(y_true, probs, labels=classes))

    # Multi-class Brier Score: mean sum squared difference
    y_one_hot = np.zeros_like(probs)
    for i, val in enumerate(y_true):
        idx = np.where(classes == val)[0]
        if len(idx) > 0:
            y_one_hot[i, idx[0]] = 1.0

    brier = float(np.mean(np.sum((probs - y_one_hot) ** 2, axis=1)))
    ece = calculate_expected_calibration_error(probs, y_true, classes)
    conf_mat = confusion_matrix(y_true, y_pred, labels=classes).tolist()

    report: Dict[str, Any] = {
        "accuracy": round(acc, 4),
        "balanced_accuracy": round(bal_acc, 4),
        "precision_macro": round(prec, 4),
        "recall_macro": round(rec, 4),
        "f1_macro": round(f1, 4),
        "log_loss": round(ll, 4),
        "brier_score": round(brier, 4),
        "expected_calibration_error": round(ece, 4),
        "confusion_matrix": conf_mat,
        "sample_count": len(y_true)
    }

    if confidence_bins:
        # Evaluate performance by confidence bin
        confidences = np.max(probs, axis=1)
        bin_edges = [0.50, 0.55, 0.60, 0.65, 0.70, 0.80, 1.01]
        bin_labels = ["0.50-0.55", "0.55-0.60", "0.60-0.65", "0.65-0.70", "0.70-0.80", "0.80+"]
        bin_stats = {}

        for b_i in range(len(bin_edges) - 1):
            low, high = bin_edges[b_i], bin_edges[b_i + 1]
            mask = (confidences >= low) & (confidences < high)
            cnt = int(np.sum(mask))
            if cnt > 0:
                bin_acc = float(np.mean(y_pred[mask] == y_true[mask]))
                bin_conf = float(np.mean(confidences[mask]))
                bin_stats[bin_labels[b_i]] = {
                    "count": cnt,
                    "accuracy": round(bin_acc, 4),
                    "mean_confidence": round(bin_conf, 4)
                }
            else:
                bin_stats[bin_labels[b_i]] = {"count": 0, "accuracy": None, "mean_confidence": None}

        report["performance_by_confidence_bin"] = bin_stats

    return report
