"""Quantitative, model-faithful saliency metrics for Objective 4."""

from __future__ import annotations

import numpy as np
import torch
from scipy.stats import spearmanr


def saliency_spearman(first: torch.Tensor, second: torch.Tensor) -> float:
    left = first.detach().float().cpu().numpy().ravel()
    right = second.detach().float().cpu().numpy().ravel()
    value = float(spearmanr(left, right).statistic)
    return value if np.isfinite(value) else 0.0


def saliency_concentration(saliency: torch.Tensor, mask: torch.Tensor) -> float:
    values = saliency.detach().float()
    if values.ndim == 4:
        values = values[0, 0]
    region = mask.detach().to(device=values.device, dtype=torch.bool)
    if region.ndim == 4:
        region = region[0, 0]
    if values.shape != region.shape:
        raise ValueError("Saliency and ROI mask shapes do not match")
    total = values.clamp_min(0).sum()
    if float(total) <= 1e-12:
        return 0.0
    return float(values.clamp_min(0)[region].sum() / total)


def imagenet_gamma_perturbation(image: torch.Tensor, gamma: float = 0.95) -> torch.Tensor:
    if image.ndim != 4 or image.shape[1] != 3:
        raise ValueError("ImageNet perturbation expects [batch, 3, H, W]")
    if gamma <= 0:
        raise ValueError("Gamma must be positive")
    mean = torch.tensor(
        [0.485, 0.456, 0.406], device=image.device, dtype=image.dtype
    )[None, :, None, None]
    std = torch.tensor(
        [0.229, 0.224, 0.225], device=image.device, dtype=image.dtype
    )[None, :, None, None]
    unit = (image * std + mean).clamp(0.0, 1.0)
    return (unit.pow(gamma) - mean) / std


@torch.no_grad()
def deletion_insertion_auc(
    model,
    image: torch.Tensor,
    clinical: torch.Tensor,
    label_index: int,
    saliency: torch.Tensor,
    *,
    steps: int = 11,
) -> dict[str, object]:
    if steps < 2:
        raise ValueError("Faithfulness curves require at least two steps")
    if len(image) != 1 or saliency.shape[:2] != (1, 1):
        raise ValueError("Faithfulness metric currently accepts one case")
    fractions = torch.linspace(0.0, 1.0, steps, device=image.device)
    pixels = int(image.shape[-2] * image.shape[-1])
    order = torch.argsort(saliency.flatten(), descending=True)
    spatial_masks = torch.zeros((steps, pixels), device=image.device, dtype=torch.bool)
    for index, fraction in enumerate(fractions):
        count = min(pixels, int(round(float(fraction) * pixels)))
        if count:
            spatial_masks[index, order[:count]] = True
    original = image.expand(steps, -1, -1, -1)
    baseline = torch.zeros_like(original)
    mask = spatial_masks[:, None, :].expand(-1, image.shape[1], -1)
    original_flat = original.reshape(steps, image.shape[1], pixels)
    baseline_flat = baseline.reshape(steps, image.shape[1], pixels)
    deletion = torch.where(mask, baseline_flat, original_flat).reshape_as(original)
    insertion = torch.where(mask, original_flat, baseline_flat).reshape_as(original)
    repeated_clinical = clinical.expand(steps, -1)
    deletion_probability = torch.sigmoid(model(deletion, repeated_clinical))[
        :, label_index
    ]
    insertion_probability = torch.sigmoid(model(insertion, repeated_clinical))[
        :, label_index
    ]
    x = fractions.detach().cpu().numpy()
    deletion_values = deletion_probability.detach().cpu().numpy()
    insertion_values = insertion_probability.detach().cpu().numpy()
    return {
        "fractions": x.tolist(),
        "deletion_curve": deletion_values.tolist(),
        "insertion_curve": insertion_values.tolist(),
        "deletion_auc": float(np.trapezoid(deletion_values, x)),
        "insertion_auc": float(np.trapezoid(insertion_values, x)),
    }
