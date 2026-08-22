"""Reversible 2D/3D medical-image preprocessing primitives."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image
from scipy.ndimage import zoom

from .config import PreprocessingConfig


@dataclass(frozen=True)
class ResizeGeometry:
    original_height: int
    original_width: int
    output_height: int
    output_width: int
    resized_height: int
    resized_width: int
    pad_top: int
    pad_left: int
    scale: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def ensure_grayscale(image: np.ndarray) -> np.ndarray:
    """Return a two-dimensional float32 image."""
    array = np.asarray(image)
    if array.ndim == 2:
        return array.astype(np.float32, copy=False)
    if array.ndim == 3 and array.shape[-1] in (3, 4):
        code = cv2.COLOR_RGBA2GRAY if array.shape[-1] == 4 else cv2.COLOR_RGB2GRAY
        return cv2.cvtColor(array.astype(np.uint8), code).astype(np.float32)
    raise ValueError(f"Expected a 2D grayscale or RGB(A) image, received shape {array.shape}")


def percentile_to_uint8(
    image: np.ndarray,
    lower_percentile: float = 0.5,
    upper_percentile: float = 99.5,
) -> np.ndarray:
    """Robustly map image intensities to uint8 without dataset leakage."""
    array = ensure_grayscale(image)
    finite = array[np.isfinite(array)]
    if finite.size == 0:
        raise ValueError("Image contains no finite pixels")
    low, high = np.percentile(finite, [lower_percentile, upper_percentile])
    if high <= low:
        return np.zeros(array.shape, dtype=np.uint8)
    scaled = (np.clip(array, low, high) - low) / (high - low)
    return np.rint(scaled * 255.0).astype(np.uint8)


def resize_with_padding(
    image: np.ndarray,
    output_shape: tuple[int, int],
    *,
    interpolation: int = cv2.INTER_AREA,
    pad_value: int | float = 0,
) -> tuple[np.ndarray, ResizeGeometry]:
    """Resize while preserving aspect ratio, then symmetrically pad."""
    if image.ndim != 2:
        raise ValueError("resize_with_padding expects a 2D array")
    source_h, source_w = image.shape
    output_h, output_w = output_shape
    if min(source_h, source_w, output_h, output_w) <= 0:
        raise ValueError("Image and output dimensions must be positive")
    scale = min(output_h / source_h, output_w / source_w)
    resized_h = max(1, int(round(source_h * scale)))
    resized_w = max(1, int(round(source_w * scale)))
    resized = cv2.resize(image, (resized_w, resized_h), interpolation=interpolation)
    pad_top = (output_h - resized_h) // 2
    pad_left = (output_w - resized_w) // 2
    output = np.full((output_h, output_w), pad_value, dtype=resized.dtype)
    output[pad_top : pad_top + resized_h, pad_left : pad_left + resized_w] = resized
    geometry = ResizeGeometry(
        original_height=source_h,
        original_width=source_w,
        output_height=output_h,
        output_width=output_w,
        resized_height=resized_h,
        resized_width=resized_w,
        pad_top=pad_top,
        pad_left=pad_left,
        scale=float(scale),
    )
    return output, geometry


def transform_mask(mask: np.ndarray, geometry: ResizeGeometry) -> np.ndarray:
    """Apply an image resize geometry to a binary ROI mask."""
    binary = (np.asarray(mask) > 0).astype(np.uint8)
    if binary.shape != (geometry.original_height, geometry.original_width):
        raise ValueError(
            f"Mask shape {binary.shape} does not match original image shape "
            f"{(geometry.original_height, geometry.original_width)}"
        )
    resized = cv2.resize(
        binary,
        (geometry.resized_width, geometry.resized_height),
        interpolation=cv2.INTER_NEAREST,
    )
    output = np.zeros((geometry.output_height, geometry.output_width), dtype=np.uint8)
    output[
        geometry.pad_top : geometry.pad_top + geometry.resized_height,
        geometry.pad_left : geometry.pad_left + geometry.resized_width,
    ] = resized
    return output


def restore_mask(mask: np.ndarray, geometry: ResizeGeometry) -> np.ndarray:
    """Map a model-space mask back to the original image resolution."""
    binary = (np.asarray(mask) > 0).astype(np.uint8)
    expected = (geometry.output_height, geometry.output_width)
    if binary.shape != expected:
        raise ValueError(f"Model-space mask shape {binary.shape} does not match {expected}")
    cropped = binary[
        geometry.pad_top : geometry.pad_top + geometry.resized_height,
        geometry.pad_left : geometry.pad_left + geometry.resized_width,
    ]
    return cv2.resize(
        cropped,
        (geometry.original_width, geometry.original_height),
        interpolation=cv2.INTER_NEAREST,
    ).astype(np.uint8)


def preprocess_cxr(
    image: np.ndarray,
    config: PreprocessingConfig,
) -> tuple[np.ndarray, ResizeGeometry]:
    """Normalise, optionally CLAHE-enhance, and letterbox a CXR."""
    output = percentile_to_uint8(
        image,
        lower_percentile=config.lower_percentile,
        upper_percentile=config.upper_percentile,
    )
    if config.apply_clahe:
        clahe = cv2.createCLAHE(
            clipLimit=config.clahe_clip_limit,
            tileGridSize=(config.clahe_grid_size, config.clahe_grid_size),
        )
        output = clahe.apply(output)
    return resize_with_padding(
        output,
        (config.image_size, config.image_size),
        pad_value=config.pad_value,
    )


def load_image(path: str | Path) -> np.ndarray:
    """Load PNG/JPEG/TIFF, NumPy, or a single DICOM image.

    DICOM support is optional so the core unit tests remain lightweight.
    """
    source = Path(path)
    suffix = source.suffix.lower()
    if suffix == ".npy":
        return np.load(source)
    if suffix in {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}:
        return np.asarray(Image.open(source))
    if suffix in {".dcm", ".dicom", ""}:
        try:
            import pydicom
        except ImportError as exc:
            raise ImportError("Install the 'medical' extra to read DICOM images") from exc
        dataset = pydicom.dcmread(str(source))
        pixels = dataset.pixel_array.astype(np.float32)
        slope = float(getattr(dataset, "RescaleSlope", 1.0))
        intercept = float(getattr(dataset, "RescaleIntercept", 0.0))
        pixels = pixels * slope + intercept
        if str(getattr(dataset, "PhotometricInterpretation", "")) == "MONOCHROME1":
            pixels = pixels.max() + pixels.min() - pixels
        return pixels
    raise ValueError(f"Unsupported image extension: {suffix}")


def preprocess_ct_volume(
    volume_hu: np.ndarray,
    source_spacing: tuple[float, float, float],
    config: PreprocessingConfig,
) -> tuple[np.ndarray, dict[str, object]]:
    """Window and resample a 3D CT volume in z-y-x array order."""
    volume = np.asarray(volume_hu, dtype=np.float32)
    if volume.ndim != 3:
        raise ValueError(f"Expected a 3D CT volume, received {volume.shape}")
    spacing = np.asarray(source_spacing, dtype=np.float64)
    target = np.asarray(config.ct_target_spacing, dtype=np.float64)
    if np.any(spacing <= 0) or np.any(target <= 0):
        raise ValueError("CT spacings must be positive")
    factors = spacing / target
    clipped = np.clip(volume, config.ct_window_low, config.ct_window_high)
    denominator = config.ct_window_high - config.ct_window_low
    if denominator <= 0:
        raise ValueError("ct_window_high must exceed ct_window_low")
    normalised = (clipped - config.ct_window_low) / denominator
    resampled = zoom(normalised, zoom=factors, order=1, mode="nearest").astype(np.float32)
    metadata = {
        "original_shape": list(volume.shape),
        "resampled_shape": list(resampled.shape),
        "source_spacing": [float(value) for value in spacing],
        "target_spacing": [float(value) for value in target],
        "zoom_factors": [float(value) for value in factors],
        "window": [config.ct_window_low, config.ct_window_high],
    }
    return resampled, metadata
