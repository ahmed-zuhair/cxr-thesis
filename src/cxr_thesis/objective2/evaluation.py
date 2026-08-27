"""Locked-test evaluation helpers for frozen Objective 2 candidates."""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np

from .metrics import multilabel_metrics


def percentile_interval(values: np.ndarray, confidence: float = 0.95) -> list[float]:
    array = np.asarray(values, dtype=np.float64)
    array = array[np.isfinite(array)]
    if array.size == 0:
        return [float("nan"), float("nan")]
    tail = (1.0 - confidence) / 2.0
    return [
        float(np.quantile(array, tail)),
        float(np.quantile(array, 1.0 - tail)),
    ]


def paired_bootstrap_comparison(
    probabilities: Mapping[str, np.ndarray],
    targets: np.ndarray,
    thresholds: Mapping[str, np.ndarray],
    *,
    reference_model: str,
    replicates: int = 500,
    seed: int = 42,
) -> dict[str, object]:
    """Bootstrap aggregate metrics using identical case resamples for all models."""

    if replicates <= 0:
        raise ValueError("Bootstrap replicates must be positive")
    if reference_model not in probabilities:
        raise ValueError("Reference model is missing from probabilities")
    target_array = np.asarray(targets, dtype=np.int8)
    if target_array.ndim != 2 or target_array.shape[0] < 2:
        raise ValueError("Targets must have shape [cases, labels]")
    model_names = list(probabilities)
    for model in model_names:
        values = np.asarray(probabilities[model], dtype=np.float64)
        if values.shape != target_array.shape:
            raise ValueError(f"Probability shape mismatch for {model}")
        threshold = np.asarray(thresholds[model], dtype=np.float64)
        if threshold.shape != (target_array.shape[1],):
            raise ValueError(f"Threshold shape mismatch for {model}")

    metric_names = ("auroc", "auprc", "f1")
    distributions = {
        model: {metric: np.empty(replicates, dtype=np.float64) for metric in metric_names}
        for model in model_names
    }
    rng = np.random.default_rng(seed)
    for replicate in range(replicates):
        indices = rng.integers(0, target_array.shape[0], size=target_array.shape[0])
        sampled_targets = target_array[indices]
        for model in model_names:
            metrics = multilabel_metrics(
                np.asarray(probabilities[model])[indices],
                sampled_targets,
                thresholds=np.asarray(thresholds[model]),
            )["macro"]
            for metric in metric_names:
                distributions[model][metric][replicate] = float(metrics[metric])

    intervals: dict[str, object] = {}
    paired_differences: dict[str, object] = {}
    for model in model_names:
        intervals[model] = {
            metric: percentile_interval(distributions[model][metric])
            for metric in metric_names
        }
        if model == reference_model:
            continue
        paired_differences[model] = {}
        for metric in metric_names:
            difference = (
                distributions[model][metric]
                - distributions[reference_model][metric]
            )
            finite = difference[np.isfinite(difference)]
            probability_nonpositive = float(np.mean(finite <= 0.0))
            probability_nonnegative = float(np.mean(finite >= 0.0))
            paired_differences[model][metric] = {
                "model_minus_reference_mean": float(np.mean(finite)),
                "bootstrap_95_ci": percentile_interval(finite),
                "two_sided_bootstrap_p": float(
                    min(1.0, 2.0 * min(probability_nonpositive, probability_nonnegative))
                ),
            }
    return {
        "method": "paired case bootstrap",
        "replicates": int(replicates),
        "seed": int(seed),
        "reference_model": reference_model,
        "model_metric_95_ci": intervals,
        "paired_model_minus_reference": paired_differences,
    }
