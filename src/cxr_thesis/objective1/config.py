"""Typed configuration for the Objective 1 pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class PreprocessingConfig:
    image_size: int = 224
    apply_clahe: bool = False
    clahe_clip_limit: float = 2.0
    clahe_grid_size: int = 8
    lower_percentile: float = 0.5
    upper_percentile: float = 99.5
    pad_value: int = 0
    ct_window_low: float = -1000.0
    ct_window_high: float = 400.0
    ct_target_spacing: tuple[float, float, float] = (1.0, 1.0, 1.0)


@dataclass(frozen=True)
class SegmentationConfig:
    threshold: float = 0.5
    min_roi_fraction: float = 0.05
    max_roi_fraction: float = 0.90
    keep_largest_components: int = 2


@dataclass(frozen=True)
class FeatureConfig:
    histogram_bins: int = 16
    hog_bins: int = 9
    enable_radiomics: bool = False
    radiomics_parameter_file: str | None = None


@dataclass(frozen=True)
class GraphConfig:
    patch_grid_2d: tuple[int, int] = (7, 7)
    patch_grid_3d: tuple[int, int, int] = (4, 4, 4)
    connectivity_2d: int = 8
    connectivity_3d: int = 6
    knn_k: int = 3
    include_empty_nodes: bool = False


@dataclass(frozen=True)
class Objective1Config:
    seed: int = 42
    preprocessing: PreprocessingConfig = field(default_factory=PreprocessingConfig)
    segmentation: SegmentationConfig = field(default_factory=SegmentationConfig)
    features: FeatureConfig = field(default_factory=FeatureConfig)
    graph: GraphConfig = field(default_factory=GraphConfig)


def _tuple_value(value: Any, length: int, name: str) -> tuple:
    if not isinstance(value, (list, tuple)) or len(value) != length:
        raise ValueError(f"{name} must contain exactly {length} values")
    return tuple(value)


def load_config(path: str | Path) -> Objective1Config:
    """Load a YAML config and reject unknown top-level sections."""
    with Path(path).open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}
    allowed = {"seed", "preprocessing", "segmentation", "features", "graph"}
    unknown = set(raw) - allowed
    if unknown:
        raise ValueError(f"Unknown Objective 1 config sections: {sorted(unknown)}")

    prep = dict(raw.get("preprocessing", {}))
    if "ct_target_spacing" in prep:
        prep["ct_target_spacing"] = _tuple_value(
            prep["ct_target_spacing"], 3, "ct_target_spacing"
        )
    graph = dict(raw.get("graph", {}))
    if "patch_grid_2d" in graph:
        graph["patch_grid_2d"] = _tuple_value(graph["patch_grid_2d"], 2, "patch_grid_2d")
    if "patch_grid_3d" in graph:
        graph["patch_grid_3d"] = _tuple_value(graph["patch_grid_3d"], 3, "patch_grid_3d")

    config = Objective1Config(
        seed=int(raw.get("seed", 42)),
        preprocessing=PreprocessingConfig(**prep),
        segmentation=SegmentationConfig(**dict(raw.get("segmentation", {}))),
        features=FeatureConfig(**dict(raw.get("features", {}))),
        graph=GraphConfig(**graph),
    )
    if config.preprocessing.image_size <= 0:
        raise ValueError("image_size must be positive")
    if not 0 < config.segmentation.threshold < 1:
        raise ValueError("segmentation threshold must be between 0 and 1")
    return config

