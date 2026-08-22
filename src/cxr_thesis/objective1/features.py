"""ROI-conditioned radiomic, handcrafted, and clinical features."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Mapping

import cv2
import numpy as np

from .config import FeatureConfig


def _masked_values(image: np.ndarray, mask: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    array = np.asarray(image, dtype=np.float32)
    binary = np.asarray(mask) > 0
    if array.shape != binary.shape:
        raise ValueError(f"Image shape {array.shape} differs from mask shape {binary.shape}")
    if array.ndim not in (2, 3):
        raise ValueError("Only 2D and 3D features are supported")
    values = array[binary & np.isfinite(array)]
    if values.size == 0:
        raise ValueError("Cannot extract ROI features from an empty mask")
    return array, binary, values


def _distribution_features(values: np.ndarray, prefix: str) -> dict[str, float]:
    quantiles = np.percentile(values, [10, 25, 50, 75, 90])
    mean = float(values.mean())
    std = float(values.std())
    centered = values - mean
    skew = float(np.mean(centered**3) / (std**3 + 1e-8))
    kurtosis = float(np.mean(centered**4) / (std**4 + 1e-8) - 3.0)
    return {
        f"{prefix}_mean": mean,
        f"{prefix}_std": std,
        f"{prefix}_min": float(values.min()),
        f"{prefix}_max": float(values.max()),
        f"{prefix}_p10": float(quantiles[0]),
        f"{prefix}_p25": float(quantiles[1]),
        f"{prefix}_median": float(quantiles[2]),
        f"{prefix}_p75": float(quantiles[3]),
        f"{prefix}_p90": float(quantiles[4]),
        f"{prefix}_iqr": float(quantiles[3] - quantiles[1]),
        f"{prefix}_skew": skew,
        f"{prefix}_excess_kurtosis": kurtosis,
    }


def _normalise_roi(image: np.ndarray, mask: np.ndarray) -> np.ndarray:
    values = image[mask]
    low, high = np.percentile(values, [1, 99])
    if high <= low:
        return np.zeros(image.shape, dtype=np.uint8)
    scaled = (np.clip(image, low, high) - low) / (high - low)
    return np.rint(scaled * 255).astype(np.uint8)


def _lbp_histogram(image: np.ndarray, mask: np.ndarray, bins: int = 16) -> np.ndarray:
    center = image.astype(np.int16)
    code = np.zeros(image.shape, dtype=np.uint8)
    offsets = [(-1, -1), (-1, 0), (-1, 1), (0, 1), (1, 1), (1, 0), (1, -1), (0, -1)]
    for bit, (dy, dx) in enumerate(offsets):
        shifted = np.roll(center, shift=(dy, dx), axis=(0, 1))
        code |= ((shifted >= center).astype(np.uint8) << bit)
    valid = mask.copy()
    valid[[0, -1], :] = False
    valid[:, [0, -1]] = False
    hist, _ = np.histogram(code[valid], bins=bins, range=(0, 256))
    return hist.astype(np.float64) / max(1, hist.sum())


def _gradient_histogram(image: np.ndarray, mask: np.ndarray, bins: int) -> tuple[np.ndarray, float]:
    gx = cv2.Sobel(image.astype(np.float32), cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(image.astype(np.float32), cv2.CV_32F, 0, 1, ksize=3)
    magnitude, angle = cv2.cartToPolar(gx, gy, angleInDegrees=True)
    hist, _ = np.histogram(angle[mask] % 180.0, bins=bins, range=(0, 180), weights=magnitude[mask])
    hist = hist.astype(np.float64) / max(1e-8, hist.sum())
    return hist, float(magnitude[mask].mean())


def _left_right_asymmetry(image: np.ndarray, mask: np.ndarray) -> float:
    width = image.shape[1]
    half = width // 2
    if half == 0:
        return 0.0
    left_image = image[:, :half]
    right_image = np.fliplr(image[:, width - half :])
    left_mask = mask[:, :half]
    right_mask = np.fliplr(mask[:, width - half :])
    overlap = left_mask & right_mask
    if not overlap.any():
        return float("nan")
    scale = float(np.std(image[mask])) + 1e-8
    return float(np.mean(np.abs(left_image[overlap] - right_image[overlap])) / scale)


def extract_handcrafted_2d(
    image: np.ndarray,
    mask: np.ndarray,
    config: FeatureConfig | None = None,
    *,
    prefix: str = "roi",
) -> dict[str, float]:
    """Extract auditable intensity, texture, gradient, and shape features."""
    cfg = config or FeatureConfig()
    array, binary, values = _masked_values(image, mask)
    if array.ndim != 2:
        raise ValueError("extract_handcrafted_2d expects a 2D image")
    normalised = _normalise_roi(array, binary)
    features = _distribution_features(values, prefix)

    hist, _ = np.histogram(normalised[binary], bins=cfg.histogram_bins, range=(0, 256))
    hist = hist.astype(np.float64) / max(1, hist.sum())
    features.update({f"{prefix}_intensity_hist_{index:02d}": float(value) for index, value in enumerate(hist)})

    lbp = _lbp_histogram(normalised, binary, bins=cfg.histogram_bins)
    features.update({f"{prefix}_lbp_hist_{index:02d}": float(value) for index, value in enumerate(lbp)})
    hog, gradient_mean = _gradient_histogram(normalised, binary, cfg.hog_bins)
    features.update({f"{prefix}_hog_hist_{index:02d}": float(value) for index, value in enumerate(hog)})
    features[f"{prefix}_gradient_mean"] = gradient_mean

    components = binary.astype(np.uint8)
    contours, _ = cv2.findContours(components, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    perimeter = float(sum(cv2.arcLength(contour, True) for contour in contours))
    y, x = np.where(binary)
    bbox_h = int(y.max() - y.min() + 1)
    bbox_w = int(x.max() - x.min() + 1)
    area = float(binary.sum())
    probability = hist[hist > 0]
    entropy = float(-(probability * np.log2(probability)).sum())
    features.update(
        {
            f"{prefix}_area_pixels": area,
            f"{prefix}_area_fraction": float(binary.mean()),
            f"{prefix}_perimeter_pixels": perimeter,
            f"{prefix}_compactness": float((perimeter**2) / (4 * math.pi * area + 1e-8)),
            f"{prefix}_bbox_aspect": float(bbox_w / max(1, bbox_h)),
            f"{prefix}_entropy": entropy,
            f"{prefix}_left_right_asymmetry": _left_right_asymmetry(array, binary),
        }
    )
    return features


def extract_handcrafted_3d(
    volume: np.ndarray,
    mask: np.ndarray,
    spacing: tuple[float, float, float],
    *,
    prefix: str = "roi3d",
) -> dict[str, float]:
    """Extract basic 3D ROI features; radiomics adds richer texture features."""
    array, binary, values = _masked_values(volume, mask)
    if array.ndim != 3:
        raise ValueError("extract_handcrafted_3d expects a 3D volume")
    features = _distribution_features(values, prefix)
    coordinates = np.argwhere(binary)
    extents = coordinates.max(axis=0) - coordinates.min(axis=0) + 1
    voxel_volume = float(np.prod(np.asarray(spacing, dtype=float)))
    features.update(
        {
            f"{prefix}_voxels": float(binary.sum()),
            f"{prefix}_volume_mm3": float(binary.sum() * voxel_volume),
            f"{prefix}_bbox_z_mm": float(extents[0] * spacing[0]),
            f"{prefix}_bbox_y_mm": float(extents[1] * spacing[1]),
            f"{prefix}_bbox_x_mm": float(extents[2] * spacing[2]),
        }
    )
    return features


def encode_clinical_features(metadata: Mapping[str, object]) -> dict[str, float]:
    """Encode non-target clinical context with explicit missingness flags."""
    age_raw = metadata.get("age")
    try:
        age = float(age_raw) if age_raw not in (None, "") else float("nan")
    except (TypeError, ValueError):
        age = float("nan")
    age_missing = not np.isfinite(age)
    sex = str(metadata.get("sex", "")).strip().upper()
    view = str(metadata.get("view", "")).strip().upper()
    return {
        "clinical_age_scaled": 0.0 if age_missing else float(np.clip(age, 0, 120) / 120.0),
        "clinical_age_missing": float(age_missing),
        "clinical_sex_female": float(sex in {"F", "FEMALE"}),
        "clinical_sex_male": float(sex in {"M", "MALE"}),
        "clinical_sex_missing": float(sex not in {"F", "FEMALE", "M", "MALE"}),
        "clinical_view_pa": float(view == "PA"),
        "clinical_view_ap": float(view == "AP"),
        "clinical_view_lateral": float(view in {"LATERAL", "LL"}),
        "clinical_view_missing": float(view not in {"PA", "AP", "LATERAL", "LL"}),
    }


def extract_pyradiomics(
    image: np.ndarray,
    mask: np.ndarray,
    *,
    spacing: tuple[float, ...],
    parameter_file: str | Path | None = None,
    label: int = 1,
) -> dict[str, float]:
    """Extract PyRadiomics features for a 2D or 3D ROI.

    This dependency is optional. Diagnostics and non-scalar outputs are
    intentionally excluded from the machine-learning feature table.
    """
    try:
        import SimpleITK as sitk
        from radiomics import featureextractor
    except ImportError as exc:
        raise ImportError("Install the 'medical' extra to enable PyRadiomics") from exc

    array, binary, _ = _masked_values(image, mask)
    if len(spacing) != array.ndim:
        raise ValueError("Spacing dimensionality must match the image")
    sitk_image = sitk.GetImageFromArray(array.astype(np.float32))
    sitk_mask = sitk.GetImageFromArray(binary.astype(np.uint8))
    sitk_image.SetSpacing(tuple(float(value) for value in reversed(spacing)))
    sitk_mask.SetSpacing(tuple(float(value) for value in reversed(spacing)))
    extractor = featureextractor.RadiomicsFeatureExtractor(str(parameter_file)) if parameter_file else featureextractor.RadiomicsFeatureExtractor()
    if array.ndim == 2:
        extractor.settings["force2D"] = True
    raw = extractor.execute(sitk_image, sitk_mask, label=label)
    output: dict[str, float] = {}
    for name, value in raw.items():
        if str(name).startswith("diagnostics_"):
            continue
        scalar = np.asarray(value)
        if scalar.size == 1 and np.issubdtype(scalar.dtype, np.number):
            output[f"radiomics_{name}"] = float(scalar.reshape(-1)[0])
    return output

