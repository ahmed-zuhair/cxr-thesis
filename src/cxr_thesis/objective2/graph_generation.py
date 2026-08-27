"""Frozen-model ROI graph generation for Objective 2 classifiers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import numpy as np

from cxr_thesis.objective1.config import Objective1Config
from cxr_thesis.objective1.graphs import GraphSample, build_patch_graph_2d
from cxr_thesis.objective1.segmentation import (
    remove_small_components,
    validate_roi_mask,
)


@dataclass(frozen=True)
class GeneratedROIGraph:
    """One graph plus the auditable mask-quality information used to build it."""

    graph: GraphSample
    mask_quality: dict[str, object]
    cleanup: dict[str, object]


def safe_graph_name(value: object) -> str:
    """Return a filesystem-safe graph stem for a manifest image identifier."""

    text = str(value).strip().replace("/", "_").replace("\\", "_")
    if text in {"", ".", ".."}:
        raise ValueError(f"Unsafe image identifier: {value!r}")
    return text


def build_frozen_roi_graph(
    processed_image: np.ndarray,
    probability: np.ndarray,
    *,
    threshold: float,
    config: Objective1Config,
    record: Mapping[str, object],
    checkpoint_sha256: str,
    min_component_fraction: float = 0.001,
    min_component_pixels: int = 0,
) -> GeneratedROIGraph:
    """Convert one frozen U-Net probability map into an ROI-aware patch graph.

    The probability map and image remain in model space. No predicted mask or
    preprocessed medical image is written to disk by this helper.
    """

    image = np.asarray(processed_image)
    values = np.asarray(probability, dtype=np.float32)
    if image.ndim != 2 or values.shape != image.shape:
        raise ValueError("A same-shaped 2D image and probability map are required")
    if not np.isfinite(values).all():
        raise ValueError("The segmentation probability map contains non-finite values")
    if not 0.0 < threshold < 1.0:
        raise ValueError("The frozen segmentation threshold must be between zero and one")

    raw_mask = values >= float(threshold)
    cleaned_mask, cleanup = remove_small_components(
        raw_mask,
        min_component_fraction=min_component_fraction,
        min_component_pixels=min_component_pixels,
    )
    quality = validate_roi_mask(cleaned_mask, config.segmentation)
    if not quality["is_nonempty"]:
        raise ValueError("Frozen segmentation produced an empty ROI mask")

    graph = build_patch_graph_2d(
        image,
        cleaned_mask,
        grid=config.graph.patch_grid_2d,
        connectivity=config.graph.connectivity_2d,
        knn_k=config.graph.knn_k,
        include_empty_nodes=config.graph.include_empty_nodes,
    )
    if graph.x.shape[1] != 7:
        raise ValueError(f"Objective 2 expects seven node features, received {graph.x.shape[1]}")
    graph.metadata.update(
        {
            "image_id": str(record["image_id"]),
            "patient_id": str(record["patient_id"]),
            "dataset": str(record["dataset"]),
            "split": str(record["split"]),
            "mask_source": "frozen_adapted_unet_probability",
            "mask_checkpoint_sha256": str(checkpoint_sha256),
            "mask_threshold": float(threshold),
        }
    )
    graph.validate()
    return GeneratedROIGraph(
        graph=graph,
        mask_quality=dict(quality),
        cleanup=dict(cleanup),
    )
