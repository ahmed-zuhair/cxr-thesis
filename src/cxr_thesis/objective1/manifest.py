"""Canonical study manifest and leakage-safe split utilities."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd


CORE_COLUMNS = (
    "dataset",
    "patient_id",
    "study_id",
    "image_id",
    "image_path",
    "modality",
    "view",
    "split",
)
VALID_SPLITS = frozenset({"train", "val", "test", "external"})
VALID_MODALITIES = frozenset({"CXR", "CT"})


def _normalise_id(value: object) -> str:
    if pd.isna(value):
        return ""
    return str(value).strip()


def validate_manifest(
    manifest: pd.DataFrame,
    *,
    require_files: bool = False,
    root: str | Path | None = None,
) -> dict[str, object]:
    """Validate schema, identifiers, file existence, and patient leakage.

    The function raises ``ValueError`` for invalid research data instead of
    relying on ``assert`` statements, which can be disabled by Python.
    """
    missing = sorted(set(CORE_COLUMNS) - set(manifest.columns))
    if missing:
        raise ValueError(f"Manifest is missing required columns: {missing}")
    if manifest.empty:
        raise ValueError("Manifest is empty")

    frame = manifest.copy()
    for column in ("dataset", "patient_id", "study_id", "image_id", "split"):
        frame[column] = frame[column].map(_normalise_id)
        if (frame[column] == "").any():
            rows = frame.index[frame[column] == ""].tolist()[:10]
            raise ValueError(f"Empty {column} values at rows {rows}")

    invalid_splits = sorted(set(frame["split"]) - VALID_SPLITS)
    if invalid_splits:
        raise ValueError(f"Invalid split names: {invalid_splits}")
    invalid_modalities = sorted(set(frame["modality"].astype(str)) - VALID_MODALITIES)
    if invalid_modalities:
        raise ValueError(f"Invalid modalities: {invalid_modalities}")
    if frame["image_id"].duplicated().any():
        duplicated = frame.loc[frame["image_id"].duplicated(), "image_id"].tolist()[:10]
        raise ValueError(f"Duplicate image_id values: {duplicated}")

    patient_splits = frame.groupby(["dataset", "patient_id"])["split"].nunique()
    leaked = patient_splits[patient_splits > 1]
    if not leaked.empty:
        examples = [f"{dataset}:{patient}" for dataset, patient in leaked.index[:10]]
        raise ValueError(f"Patient leakage across splits: {examples}")

    if require_files:
        base = Path(root) if root is not None else Path.cwd()
        absent: list[str] = []
        for value in frame["image_path"]:
            candidate = Path(str(value))
            if not candidate.is_absolute():
                candidate = base / candidate
            if not candidate.is_file():
                absent.append(str(candidate))
                if len(absent) == 10:
                    break
        if absent:
            raise ValueError(f"Missing image files (first {len(absent)}): {absent}")

    return {
        "rows": int(len(frame)),
        "patients": int(frame[["dataset", "patient_id"]].drop_duplicates().shape[0]),
        "studies": int(frame[["dataset", "study_id"]].drop_duplicates().shape[0]),
        "datasets": sorted(frame["dataset"].unique().tolist()),
        "modalities": frame["modality"].value_counts().to_dict(),
        "splits": frame["split"].value_counts().to_dict(),
    }


def assign_patient_validation_split(
    frame: pd.DataFrame,
    *,
    source_split: str = "train",
    val_fraction: float = 0.10,
    seed: int = 42,
) -> pd.DataFrame:
    """Move a reproducible subset of source patients to validation."""
    if not 0 < val_fraction < 1:
        raise ValueError("val_fraction must be strictly between 0 and 1")
    result = frame.copy()
    eligible = result.loc[result["split"] == source_split, "patient_id"].astype(str).unique()
    if len(eligible) < 2:
        raise ValueError("At least two source patients are required to create validation")
    rng = np.random.default_rng(seed)
    shuffled = rng.permutation(eligible)
    count = max(1, int(round(len(shuffled) * val_fraction)))
    count = min(count, len(shuffled) - 1)
    selected = set(shuffled[:count].tolist())
    mask = (result["split"] == source_split) & result["patient_id"].astype(str).isin(selected)
    result.loc[mask, "split"] = "val"
    validate_manifest(result)
    return result


def build_nih_manifest(
    metadata_csv: str | Path,
    train_val_list: str | Path,
    test_list: str | Path,
    images_root: str | Path,
    *,
    val_fraction: float = 0.10,
    seed: int = 42,
    label_names: Sequence[str] | None = None,
) -> pd.DataFrame:
    """Build a canonical NIH ChestX-ray14 manifest.

    The official test-file list is preserved. Validation patients are sampled
    only from the official training/validation file list.
    """
    metadata = pd.read_csv(metadata_csv).rename(
        columns={
            "Image Index": "filename",
            "Finding Labels": "finding_labels",
            "Patient ID": "patient_id",
            "Patient Age": "age",
            "Patient Gender": "sex",
            "View Position": "view",
            "Follow-up #": "follow_up",
            "OriginalImagePixelSpacing_x": "pixel_spacing_x",
            "OriginalImagePixelSpacing_y": "pixel_spacing_y",
        }
    )
    required = {"filename", "finding_labels", "patient_id"}
    missing = sorted(required - set(metadata.columns))
    if missing:
        raise ValueError(f"NIH metadata is missing columns: {missing}")

    def read_names(path: str | Path) -> set[str]:
        with Path(path).open("r", encoding="utf-8") as handle:
            return {line.strip() for line in handle if line.strip()}

    train_names = read_names(train_val_list)
    test_names = read_names(test_list)
    overlap = train_names & test_names
    if overlap:
        raise ValueError(f"NIH train/test filename overlap: {sorted(overlap)[:10]}")

    known = train_names | test_names
    metadata = metadata[metadata["filename"].isin(known)].copy()
    metadata["split"] = np.where(metadata["filename"].isin(test_names), "test", "train")
    metadata["patient_id"] = metadata["patient_id"].map(_normalise_id)
    metadata["study_id"] = metadata.apply(
        lambda row: f"nih-{row['patient_id']}-{int(row.get('follow_up', 0))}", axis=1
    )
    metadata["image_id"] = "nih-" + metadata["filename"].astype(str)
    metadata["dataset"] = "NIH-ChestXray14"
    metadata["modality"] = "CXR"
    image_root = Path(images_root)
    filenames = set(metadata["filename"].astype(str))
    direct_paths = {name: image_root / name for name in filenames}
    missing_direct = {name for name, path in direct_paths.items() if not path.is_file()}
    if missing_direct:
        # Kaggle's NIH dataset commonly uses images_001/images/<filename>.
        # Scan once rather than performing an expensive recursive lookup for
        # every one of the 112k manifest rows.
        for candidate in image_root.rglob("*.png"):
            if candidate.name in missing_direct:
                direct_paths[candidate.name] = candidate
                missing_direct.remove(candidate.name)
                if not missing_direct:
                    break
    metadata["image_path"] = metadata["filename"].map(
        lambda name: str(direct_paths[str(name)])
    )
    metadata["mask_path"] = ""
    metadata["indication"] = ""

    if label_names is None:
        observed = {
            label
            for value in metadata["finding_labels"].fillna("")
            for label in str(value).split("|")
            if label and label != "No Finding"
        }
        label_names = sorted(observed)
    for label in label_names:
        metadata[f"label_{label}"] = metadata["finding_labels"].fillna("").map(
            lambda value, item=label: int(item in str(value).split("|"))
        )
    metadata["labels_json"] = metadata["finding_labels"].fillna("").map(
        lambda value: json.dumps([] if value == "No Finding" else str(value).split("|"))
    )

    columns = list(CORE_COLUMNS) + [
        "mask_path",
        "age",
        "sex",
        "indication",
        "pixel_spacing_x",
        "pixel_spacing_y",
        "finding_labels",
        "labels_json",
    ] + [f"label_{label}" for label in label_names]
    frame = metadata.reindex(columns=columns)
    frame = assign_patient_validation_split(
        frame, source_split="train", val_fraction=val_fraction, seed=seed
    )
    validate_manifest(frame)
    return frame.reset_index(drop=True)


def write_manifest(frame: pd.DataFrame, path: str | Path) -> Path:
    """Validate and write a manifest as CSV."""
    validate_manifest(frame)
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(target, index=False)
    return target
