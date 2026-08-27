"""Losses and imbalance controls for enhanced multi-label classification."""

from __future__ import annotations

import numpy as np
import torch
from torch import nn


def transform_positive_weights(
    positive_counts: np.ndarray,
    cases: int,
    *,
    transform: str = "raw",
    maximum: float | None = None,
) -> np.ndarray:
    """Calculate reproducible BCE weights without allowing rare labels to dominate."""

    positives = np.asarray(positive_counts, dtype=np.float32)
    if positives.ndim != 1 or cases <= 0:
        raise ValueError(
            "One-dimensional positive counts and a positive case count are required"
        )
    if np.any(positives < 0) or np.any(positives > cases):
        raise ValueError("Positive counts fall outside the cohort")
    raw = (float(cases) - positives) / np.maximum(positives, 1.0)
    if transform == "raw":
        weights = raw
    elif transform == "sqrt":
        weights = np.sqrt(raw)
    elif transform == "log1p":
        weights = np.log1p(raw)
    elif transform == "none":
        weights = np.ones_like(raw)
    else:
        raise ValueError("Unknown positive-weight transform")
    if maximum is not None:
        if maximum < 1.0:
            raise ValueError("maximum positive weight must be at least one")
        weights = np.minimum(weights, float(maximum))
    return weights.astype(np.float32)


class AsymmetricLoss(nn.Module):
    """Asymmetric focal-style loss for imbalanced multi-label targets."""

    def __init__(
        self,
        *,
        gamma_negative: float = 4.0,
        gamma_positive: float = 1.0,
        negative_clip: float = 0.05,
        epsilon: float = 1e-8,
    ) -> None:
        super().__init__()
        if min(gamma_negative, gamma_positive, negative_clip) < 0.0:
            raise ValueError("Asymmetric-loss parameters must be non-negative")
        self.gamma_negative = float(gamma_negative)
        self.gamma_positive = float(gamma_positive)
        self.negative_clip = float(negative_clip)
        self.epsilon = float(epsilon)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        if logits.shape != targets.shape:
            raise ValueError("Logits and targets must have the same shape")
        positive_probability = torch.sigmoid(logits)
        negative_probability = 1.0 - positive_probability
        if self.negative_clip > 0.0:
            negative_probability = (negative_probability + self.negative_clip).clamp(
                max=1.0
            )
        log_likelihood = targets * torch.log(
            positive_probability.clamp_min(self.epsilon)
        ) + (1.0 - targets) * torch.log(negative_probability.clamp_min(self.epsilon))
        gamma = targets * self.gamma_positive + (1.0 - targets) * self.gamma_negative
        probability = (
            targets * positive_probability + (1.0 - targets) * negative_probability
        )
        weight = torch.pow((1.0 - probability).clamp_min(0.0), gamma)
        return -(weight * log_likelihood).mean()
