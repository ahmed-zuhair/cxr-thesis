"""ROI segmentation model, post-processing, and quantitative validation."""

from __future__ import annotations

from collections.abc import Callable

import cv2
import numpy as np
import torch
from scipy.ndimage import binary_erosion, distance_transform_edt
from torch import nn
from torch.nn import functional as F

from .config import SegmentationConfig


class DoubleConv(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.block(inputs)


class UNet2D(nn.Module):
    """Compact U-Net for lung/heart ROI experiments.

    ``classes=1`` produces a union mask. Use ``classes=3`` for left lung,
    right lung, and heart channels with a multi-label segmentation loss.
    """

    def __init__(
        self,
        in_channels: int = 1,
        classes: int = 1,
        channels: tuple[int, int, int, int] = (32, 64, 128, 256),
    ) -> None:
        super().__init__()
        c1, c2, c3, c4 = channels
        self.enc1 = DoubleConv(in_channels, c1)
        self.enc2 = DoubleConv(c1, c2)
        self.enc3 = DoubleConv(c2, c3)
        self.bottleneck = DoubleConv(c3, c4)
        self.pool = nn.MaxPool2d(2)
        self.up3 = nn.ConvTranspose2d(c4, c3, 2, stride=2)
        self.dec3 = DoubleConv(c3 + c3, c3)
        self.up2 = nn.ConvTranspose2d(c3, c2, 2, stride=2)
        self.dec2 = DoubleConv(c2 + c2, c2)
        self.up1 = nn.ConvTranspose2d(c2, c1, 2, stride=2)
        self.dec1 = DoubleConv(c1 + c1, c1)
        self.output = nn.Conv2d(c1, classes, 1)

    @staticmethod
    def _match(skip: torch.Tensor, value: torch.Tensor) -> torch.Tensor:
        if value.shape[-2:] != skip.shape[-2:]:
            value = F.interpolate(value, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        return value

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        x1 = self.enc1(inputs)
        x2 = self.enc2(self.pool(x1))
        x3 = self.enc3(self.pool(x2))
        bottleneck = self.bottleneck(self.pool(x3))
        x = self._match(x3, self.up3(bottleneck))
        x = self.dec3(torch.cat([x3, x], dim=1))
        x = self._match(x2, self.up2(x))
        x = self.dec2(torch.cat([x2, x], dim=1))
        x = self._match(x1, self.up1(x))
        return self.output(self.dec1(torch.cat([x1, x], dim=1)))


def soft_dice_loss(logits: torch.Tensor, target: torch.Tensor, epsilon: float = 1e-6) -> torch.Tensor:
    probabilities = torch.sigmoid(logits)
    target = target.to(probabilities.dtype)
    dimensions = tuple(range(2, probabilities.ndim))
    intersection = (probabilities * target).sum(dim=dimensions)
    denominator = probabilities.sum(dim=dimensions) + target.sum(dim=dimensions)
    return 1.0 - ((2.0 * intersection + epsilon) / (denominator + epsilon)).mean()


def segmentation_loss(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Equal-weight BCE and soft Dice loss."""
    return 0.5 * F.binary_cross_entropy_with_logits(logits, target.float()) + 0.5 * soft_dice_loss(
        logits, target
    )


def postprocess_binary_mask(
    probability: np.ndarray,
    config: SegmentationConfig,
) -> np.ndarray:
    """Threshold and retain the largest plausible connected components."""
    array = np.asarray(probability, dtype=np.float32)
    if array.ndim != 2:
        raise ValueError("A single 2D probability map is required")
    binary = (array >= config.threshold).astype(np.uint8)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    if count <= 1:
        return binary
    component_ids = list(range(1, count))
    component_ids.sort(key=lambda item: int(stats[item, cv2.CC_STAT_AREA]), reverse=True)
    kept = component_ids[: config.keep_largest_components]
    return np.isin(labels, kept).astype(np.uint8)


def remove_small_components(
    mask: np.ndarray,
    *,
    min_component_fraction: float = 0.001,
    min_component_pixels: int = 0,
) -> tuple[np.ndarray, dict[str, float | int]]:
    """Remove only components that are negligible relative to the ROI.

    Unlike :func:`postprocess_binary_mask`, this function does not impose a
    fixed component count. That matters for abnormal or cropped radiographs
    where a valid lung region can be disconnected. A component is retained
    when its area is at least the larger of ``min_component_pixels`` and
    ``min_component_fraction`` times the total foreground area.
    """
    binary = (np.asarray(mask) > 0).astype(np.uint8)
    if binary.ndim != 2:
        raise ValueError("A single 2D binary mask is required")
    if not 0.0 <= min_component_fraction <= 1.0:
        raise ValueError("min_component_fraction must be between 0 and 1")
    if min_component_pixels < 0:
        raise ValueError("min_component_pixels cannot be negative")

    foreground_pixels = int(binary.sum())
    if foreground_pixels == 0:
        return binary, {
            "components_before": 0,
            "components_after": 0,
            "foreground_pixels_before": 0,
            "foreground_pixels_after": 0,
            "removed_pixels": 0,
            "removed_fraction": 0.0,
            "minimum_component_pixels": int(min_component_pixels),
        }

    count, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    component_ids = list(range(1, count))
    threshold_pixels = max(
        int(min_component_pixels),
        int(np.ceil(foreground_pixels * min_component_fraction)),
    )
    kept = [
        component_id
        for component_id in component_ids
        if int(stats[component_id, cv2.CC_STAT_AREA]) >= threshold_pixels
    ]
    if not kept and component_ids:
        kept = [
            max(
                component_ids,
                key=lambda item: int(stats[item, cv2.CC_STAT_AREA]),
            )
        ]
    cleaned = np.isin(labels, kept).astype(np.uint8)
    retained_pixels = int(cleaned.sum())
    removed_pixels = foreground_pixels - retained_pixels
    return cleaned, {
        "components_before": int(len(component_ids)),
        "components_after": int(len(kept)),
        "foreground_pixels_before": foreground_pixels,
        "foreground_pixels_after": retained_pixels,
        "removed_pixels": int(removed_pixels),
        "removed_fraction": float(removed_pixels / foreground_pixels),
        "minimum_component_pixels": int(threshold_pixels),
    }


def validate_roi_mask(mask: np.ndarray, config: SegmentationConfig) -> dict[str, float | bool]:
    binary = np.asarray(mask) > 0
    if binary.ndim != 2:
        raise ValueError("ROI mask must be two-dimensional")
    fraction = float(binary.mean())
    touches_border = bool(
        binary[0].any() or binary[-1].any() or binary[:, 0].any() or binary[:, -1].any()
    )
    plausible = config.min_roi_fraction <= fraction <= config.max_roi_fraction
    return {
        "roi_fraction": fraction,
        "touches_border": touches_border,
        "is_nonempty": bool(binary.any()),
        "is_plausible": bool(plausible),
    }


def dice_score(prediction: np.ndarray, target: np.ndarray, epsilon: float = 1e-8) -> float:
    pred = np.asarray(prediction, dtype=bool)
    true = np.asarray(target, dtype=bool)
    if pred.shape != true.shape:
        raise ValueError("Prediction and target masks must have equal shapes")
    if not pred.any() and not true.any():
        return 1.0
    return float((2.0 * np.logical_and(pred, true).sum() + epsilon) / (pred.sum() + true.sum() + epsilon))


def iou_score(prediction: np.ndarray, target: np.ndarray, epsilon: float = 1e-8) -> float:
    pred = np.asarray(prediction, dtype=bool)
    true = np.asarray(target, dtype=bool)
    if pred.shape != true.shape:
        raise ValueError("Prediction and target masks must have equal shapes")
    union = np.logical_or(pred, true).sum()
    if union == 0:
        return 1.0
    return float((np.logical_and(pred, true).sum() + epsilon) / (union + epsilon))


def hausdorff95(prediction: np.ndarray, target: np.ndarray) -> float:
    """Symmetric 95th-percentile surface distance in pixels."""
    pred = np.asarray(prediction, dtype=bool)
    true = np.asarray(target, dtype=bool)
    if pred.shape != true.shape:
        raise ValueError("Prediction and target masks must have equal shapes")
    if not pred.any() and not true.any():
        return 0.0
    if not pred.any() or not true.any():
        return float("inf")
    pred_surface = pred ^ binary_erosion(pred)
    true_surface = true ^ binary_erosion(true)
    distance_to_true = distance_transform_edt(~true_surface)[pred_surface]
    distance_to_pred = distance_transform_edt(~pred_surface)[true_surface]
    distances = np.concatenate([distance_to_true, distance_to_pred])
    return float(np.percentile(distances, 95))


def evaluate_segmentation(predictions: list[np.ndarray], targets: list[np.ndarray]) -> dict[str, float]:
    if len(predictions) != len(targets) or not predictions:
        raise ValueError("Equal non-empty prediction and target lists are required")
    dice = np.asarray([dice_score(p, t) for p, t in zip(predictions, targets)], dtype=float)
    iou = np.asarray([iou_score(p, t) for p, t in zip(predictions, targets)], dtype=float)
    hd95 = np.asarray([hausdorff95(p, t) for p, t in zip(predictions, targets)], dtype=float)
    finite_hd95 = hd95[np.isfinite(hd95)]
    return {
        "cases": float(len(predictions)),
        "dice_mean": float(dice.mean()),
        "dice_std": float(dice.std()),
        "iou_mean": float(iou.mean()),
        "hd95_mean": float(finite_hd95.mean()) if finite_hd95.size else float("inf"),
        "empty_failure_rate": float(np.isinf(hd95).mean()),
    }


@torch.inference_mode()
def predict_mask(
    model: nn.Module,
    image: np.ndarray,
    config: SegmentationConfig,
    *,
    device: str | torch.device = "cpu",
) -> np.ndarray:
    """Run a segmentation model on a preprocessed uint8/float CXR."""
    array = np.asarray(image, dtype=np.float32)
    if array.ndim != 2:
        raise ValueError("predict_mask expects a 2D image")
    if array.max() > 1.0:
        array = array / 255.0
    tensor = torch.from_numpy(array)[None, None].to(device)
    model = model.to(device).eval()
    probability = torch.sigmoid(model(tensor))[0, 0].cpu().numpy()
    return postprocess_binary_mask(probability, config)
