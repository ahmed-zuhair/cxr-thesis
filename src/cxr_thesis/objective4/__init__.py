"""Objective 4 explainable-AI utilities."""

from .explainability import GradCAM, integrated_gradients, normalise_saliency
from .metrics import (
    deletion_insertion_auc,
    imagenet_gamma_perturbation,
    saliency_concentration,
    saliency_spearman,
)

__all__ = [
    "GradCAM",
    "integrated_gradients",
    "normalise_saliency",
    "deletion_insertion_auc",
    "imagenet_gamma_perturbation",
    "saliency_concentration",
    "saliency_spearman",
]
