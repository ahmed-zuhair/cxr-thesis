"""End-to-end Objective 1 extraction from a canonical CXR manifest."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Mapping

import numpy as np
import pandas as pd
from PIL import Image

from .config import Objective1Config
from .features import encode_clinical_features, extract_handcrafted_2d, extract_pyradiomics
from .graphs import build_patch_graph_2d
from .manifest import validate_manifest
from .preprocessing import load_image, preprocess_cxr, transform_mask
from .segmentation import validate_roi_mask


def _resolve(value: object, root: Path) -> Path:
    path = Path(str(value))
    return path if path.is_absolute() else root / path


def _safe_name(value: object) -> str:
    text = str(value).strip().replace("/", "_").replace("\\", "_")
    if text in {"", ".", ".."}:
        raise ValueError(f"Unsafe image_id: {value!r}")
    return text


def _positive_float(value: object, default: float = 1.0) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return default
    return parsed if np.isfinite(parsed) and parsed > 0 else default


def process_cxr_record(
    record: Mapping[str, object],
    config: Objective1Config,
    output_root: str | Path,
    *,
    data_root: str | Path = ".",
    allow_full_image_roi: bool = False,
) -> dict[str, object]:
    """Preprocess one image, extract features, and save an ROI-aware graph.

    A real ROI mask is required by default. ``allow_full_image_roi`` exists
    only for pipeline smoke tests and must not be used for final experiments.
    """
    output_root = Path(output_root)
    data_root = Path(data_root)
    image_id = str(record["image_id"])
    artifact_name = _safe_name(image_id)
    image = load_image(_resolve(record["image_path"], data_root))
    processed, geometry = preprocess_cxr(image, config.preprocessing)

    mask_value = record.get("mask_path", "")
    has_mask = mask_value is not None and str(mask_value).strip() not in {"", "nan", "None"}
    if has_mask:
        raw_mask = load_image(_resolve(mask_value, data_root))
        if raw_mask.ndim == 3:
            raw_mask = raw_mask[..., 0]
        mask = transform_mask(raw_mask, geometry)
        mask_source = "manifest"
    elif allow_full_image_roi:
        mask = np.ones(processed.shape, dtype=np.uint8)
        mask_source = "full_image_smoke_test_only"
    else:
        raise ValueError(
            f"{image_id} has no ROI mask. Run segmentation first or explicitly use "
            "allow_full_image_roi for a non-research smoke test."
        )

    mask_quality = validate_roi_mask(mask, config.segmentation)
    if not mask_quality["is_nonempty"]:
        raise ValueError(f"{image_id} has an empty ROI mask")
    if mask_source == "manifest" and not mask_quality["is_plausible"]:
        raise ValueError(f"{image_id} has an implausible ROI fraction: {mask_quality}")

    handcrafted = extract_handcrafted_2d(processed, mask, config.features)
    clinical = encode_clinical_features(record)
    radiomics: dict[str, float] = {}
    if config.features.enable_radiomics:
        parameter_file = config.features.radiomics_parameter_file
        original_spacing_y = _positive_float(record.get("pixel_spacing_y"))
        original_spacing_x = _positive_float(record.get("pixel_spacing_x"))
        processed_spacing = (
            original_spacing_y / geometry.scale,
            original_spacing_x / geometry.scale,
        )
        radiomics = extract_pyradiomics(
            processed,
            mask,
            spacing=processed_spacing,
            parameter_file=parameter_file,
        )

    graph = build_patch_graph_2d(
        processed,
        mask,
        grid=config.graph.patch_grid_2d,
        connectivity=config.graph.connectivity_2d,
        knn_k=config.graph.knn_k,
        include_empty_nodes=config.graph.include_empty_nodes,
    )
    graph.metadata.update(
        {
            "image_id": image_id,
            "patient_id": str(record["patient_id"]),
            "dataset": str(record["dataset"]),
            "split": str(record["split"]),
            "mask_source": mask_source,
            "mask_model_id": str(record.get("mask_model_id", "")),
            "mask_checkpoint_sha256": str(record.get("mask_checkpoint_sha256", "")),
        }
    )

    image_dir = output_root / "preprocessed"
    mask_dir = output_root / "masks"
    graph_dir = output_root / "graphs"
    metadata_dir = output_root / "metadata"
    for directory in (image_dir, mask_dir, graph_dir, metadata_dir):
        directory.mkdir(parents=True, exist_ok=True)
    Image.fromarray(processed).save(image_dir / f"{artifact_name}.png")
    Image.fromarray(mask.astype(np.uint8) * 255).save(mask_dir / f"{artifact_name}.png")
    graph_path = graph.save(graph_dir / f"{artifact_name}.npz")
    with (metadata_dir / f"{artifact_name}.json").open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "geometry": geometry.to_dict(),
                "mask_quality": mask_quality,
                "mask_source": mask_source,
                "mask_model_id": str(record.get("mask_model_id", "")),
                "mask_checkpoint_sha256": str(record.get("mask_checkpoint_sha256", "")),
            },
            handle,
            indent=2,
            sort_keys=True,
        )
    return {
        "image_id": image_id,
        "patient_id": str(record["patient_id"]),
        "dataset": str(record["dataset"]),
        "split": str(record["split"]),
        "graph_path": str(graph_path),
        "graph_nodes": int(graph.x.shape[0]),
        "graph_edges": int(graph.edge_index.shape[1]),
        **mask_quality,
        **clinical,
        **handcrafted,
        **radiomics,
    }


def run_cxr_manifest(
    manifest: pd.DataFrame,
    config: Objective1Config,
    output_root: str | Path,
    *,
    data_root: str | Path = ".",
    allow_full_image_roi: bool = False,
    limit: int | None = None,
) -> pd.DataFrame:
    """Run Objective 1 for a manifest and write a study-level feature table."""
    validate_manifest(manifest, require_files=True, root=data_root)
    selected = manifest if limit is None else manifest.iloc[:limit]
    rows = [
        process_cxr_record(
            record,
            config,
            output_root,
            data_root=data_root,
            allow_full_image_roi=allow_full_image_roi,
        )
        for record in selected.to_dict(orient="records")
    ]
    table = pd.DataFrame(rows)
    target = Path(output_root) / "features.csv"
    target.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(target, index=False)
    return table
