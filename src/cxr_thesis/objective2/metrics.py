"""Reproducible dependency-light multilabel classification metrics."""

from __future__ import annotations

import numpy as np
from scipy.stats import rankdata


def _binary_auroc(target: np.ndarray, probability: np.ndarray) -> float:
    target = np.asarray(target, dtype=np.int8)
    probability = np.asarray(probability, dtype=np.float64)
    positives = int(target.sum())
    negatives = int(target.size - positives)
    if positives == 0 or negatives == 0:
        return float("nan")
    ranks = rankdata(probability, method="average")
    positive_rank_sum = float(ranks[target == 1].sum())
    return (positive_rank_sum - positives * (positives + 1) / 2) / (
        positives * negatives
    )


def _binary_average_precision(target: np.ndarray, probability: np.ndarray) -> float:
    target = np.asarray(target, dtype=np.int8)
    probability = np.asarray(probability, dtype=np.float64)
    positives = int(target.sum())
    if positives == 0:
        return float("nan")
    order = np.argsort(-probability, kind="mergesort")
    ordered = target[order]
    cumulative = np.cumsum(ordered)
    precision = cumulative / (np.arange(target.size) + 1)
    return float(precision[ordered == 1].sum() / positives)


def select_f1_thresholds(
    probabilities: np.ndarray,
    targets: np.ndarray,
    grid: np.ndarray | None = None,
) -> np.ndarray:
    probabilities = np.asarray(probabilities, dtype=np.float64)
    targets = np.asarray(targets, dtype=np.int8)
    if probabilities.shape != targets.shape or probabilities.ndim != 2:
        raise ValueError("probabilities and targets must have matching [cases, labels] shapes")
    candidates = (
        np.asarray(grid, dtype=np.float64)
        if grid is not None
        else np.arange(0.05, 0.951, 0.05)
    )
    thresholds = np.empty(probabilities.shape[1], dtype=np.float64)
    for label in range(probabilities.shape[1]):
        best_threshold = 0.5
        best_f1 = -1.0
        truth = targets[:, label]
        for threshold in candidates:
            prediction = probabilities[:, label] >= threshold
            true_positive = int(np.sum(prediction & (truth == 1)))
            false_positive = int(np.sum(prediction & (truth == 0)))
            false_negative = int(np.sum(~prediction & (truth == 1)))
            denominator = 2 * true_positive + false_positive + false_negative
            f1 = 0.0 if denominator == 0 else 2 * true_positive / denominator
            if f1 > best_f1:
                best_f1 = f1
                best_threshold = float(threshold)
        thresholds[label] = best_threshold
    return thresholds


def multilabel_metrics(
    probabilities: np.ndarray,
    targets: np.ndarray,
    *,
    thresholds: np.ndarray | float = 0.5,
) -> dict[str, object]:
    probabilities = np.asarray(probabilities, dtype=np.float64)
    targets = np.asarray(targets, dtype=np.int8)
    if probabilities.shape != targets.shape or probabilities.ndim != 2:
        raise ValueError("probabilities and targets must have matching [cases, labels] shapes")
    threshold_values = np.broadcast_to(
        np.asarray(thresholds, dtype=np.float64), (probabilities.shape[1],)
    )
    per_label = []
    predictions = probabilities >= threshold_values[None, :]
    for label in range(probabilities.shape[1]):
        truth = targets[:, label]
        prediction = predictions[:, label]
        true_positive = int(np.sum(prediction & (truth == 1)))
        true_negative = int(np.sum(~prediction & (truth == 0)))
        false_positive = int(np.sum(prediction & (truth == 0)))
        false_negative = int(np.sum(~prediction & (truth == 1)))
        f1_denominator = 2 * true_positive + false_positive + false_negative
        per_label.append(
            {
                "auroc": _binary_auroc(truth, probabilities[:, label]),
                "auprc": _binary_average_precision(truth, probabilities[:, label]),
                "f1": 0.0 if f1_denominator == 0 else 2 * true_positive / f1_denominator,
                "sensitivity": 0.0 if true_positive + false_negative == 0 else true_positive / (true_positive + false_negative),
                "specificity": 0.0 if true_negative + false_positive == 0 else true_negative / (true_negative + false_positive),
                "threshold": float(threshold_values[label]),
            }
        )
    macro = {
        key: float(np.nanmean([entry[key] for entry in per_label]))
        for key in ("auroc", "auprc", "f1", "sensitivity", "specificity")
    }
    flat_truth = targets.reshape(-1)
    flat_probability = probabilities.reshape(-1)
    return {
        "macro": macro,
        "micro_auroc": _binary_auroc(flat_truth, flat_probability),
        "micro_auprc": _binary_average_precision(flat_truth, flat_probability),
        "per_label": per_label,
    }
