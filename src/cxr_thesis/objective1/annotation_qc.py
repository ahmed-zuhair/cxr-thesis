"""Private quality-control metrics for completed lung ROI annotations."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from scipy import ndimage


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_gray(path: Path) -> np.ndarray:
    with Image.open(path) as handle:
        return np.asarray(handle.convert("L"), dtype=np.uint8)


def _quantiles(values: list[float]) -> dict[str, float]:
    if not values:
        return {}
    return {
        str(point): float(np.quantile(values, point))
        for point in (0.0, 0.05, 0.25, 0.5, 0.75, 0.95, 1.0)
    }


def audit_completed_annotations(
    workspace: str | Path,
    role: str,
    output_dir: str | Path,
    *,
    min_foreground_fraction: float = 0.05,
    max_foreground_fraction: float = 0.55,
    max_components: int = 5,
    max_change_fraction: float = 0.20,
    min_foreground_ratio: float = 0.50,
    max_foreground_ratio: float = 1.75,
) -> dict[str, object]:
    """Audit saved masks without changing annotations or cohort membership."""

    workspace_root = Path(workspace).resolve()
    role_root = workspace_root / role
    output = Path(output_dir).resolve()
    if output.exists():
        raise FileExistsError(f"Annotation QC output already exists: {output}")
    worklist_path = role_root / "annotation_worklist.csv"
    progress_path = role_root / "annotation_progress.csv"
    if not worklist_path.is_file():
        raise FileNotFoundError(worklist_path)
    if not progress_path.is_file():
        raise FileNotFoundError(progress_path)
    worklist = pd.read_csv(worklist_path, keep_default_na=False)
    progress = pd.read_csv(progress_path, keep_default_na=False)
    required = {
        "candidate_code",
        "image_filename",
        "preannotation_filename",
        "required_output_mask",
    }
    missing = sorted(required - set(worklist.columns))
    if missing:
        raise ValueError(f"Worklist is missing columns: {missing}")
    if worklist["candidate_code"].duplicated().any():
        raise ValueError("Worklist contains duplicate candidate codes")
    if progress["candidate_code"].duplicated().any():
        raise ValueError("Progress contains duplicate candidate codes")
    progress_status = dict(zip(progress["candidate_code"], progress["status"]))

    audit_rows: list[dict[str, object]] = []
    for _, row in worklist.iterrows():
        code = str(row["candidate_code"])
        image_path = role_root / str(row["image_filename"])
        annotation_path = role_root / str(row["required_output_mask"])
        preannotation_value = str(row["preannotation_filename"])
        preannotation_path = role_root / preannotation_value if preannotation_value else None
        flags: list[str] = []
        if not image_path.is_file():
            flags.append("missing_image")
        if not annotation_path.is_file():
            flags.append("missing_annotation")
        if flags:
            audit_rows.append(
                {
                    "candidate_code": code,
                    "cohort_role": role,
                    "foreground_fraction": np.nan,
                    "component_count": np.nan,
                    "touches_border": False,
                    "change_fraction_from_preannotation": np.nan,
                    "foreground_ratio_to_preannotation": np.nan,
                    "identical_to_preannotation": False,
                    "progress_status": progress_status.get(code, "missing"),
                    "qc_flags": ";".join(flags),
                    "requires_review": True,
                }
            )
            continue

        image = _load_gray(image_path)
        annotation = _load_gray(annotation_path)
        if image.shape != annotation.shape:
            flags.append("shape_mismatch")
        unique_values = set(np.unique(annotation).tolist())
        if not unique_values.issubset({0, 255}):
            flags.append("nonbinary_mask")
        binary = annotation > 0
        foreground = float(binary.mean())
        if foreground == 0.0:
            flags.append("empty_mask")
        if foreground == 1.0:
            flags.append("full_mask")
        if foreground < min_foreground_fraction:
            flags.append("foreground_below_threshold")
        if foreground > max_foreground_fraction:
            flags.append("foreground_above_threshold")
        _, component_count = ndimage.label(binary, structure=np.ones((3, 3)))
        if int(component_count) > max_components:
            flags.append("component_count_above_threshold")
        touches_border = bool(
            binary[0, :].any()
            or binary[-1, :].any()
            or binary[:, 0].any()
            or binary[:, -1].any()
        )
        if touches_border:
            flags.append("touches_image_border")

        change_fraction = np.nan
        foreground_ratio = np.nan
        identical = False
        if preannotation_path is not None:
            if not preannotation_path.is_file():
                flags.append("missing_preannotation")
            else:
                preannotation = _load_gray(preannotation_path)
                if preannotation.shape != annotation.shape:
                    flags.append("preannotation_shape_mismatch")
                else:
                    change_fraction = float(np.mean(preannotation != annotation))
                    identical = bool(change_fraction == 0.0)
                    pre_foreground = float(np.mean(preannotation > 0))
                    if pre_foreground > 0:
                        foreground_ratio = foreground / pre_foreground
                    if change_fraction > max_change_fraction:
                        flags.append("large_change_from_preannotation")
                    if np.isfinite(foreground_ratio) and (
                        foreground_ratio < min_foreground_ratio
                        or foreground_ratio > max_foreground_ratio
                    ):
                        flags.append("foreground_ratio_outlier")

        status = progress_status.get(code, "missing")
        if status == "missing":
            flags.append("missing_progress_record")
        if status == "needs_review":
            flags.append("progress_needs_review")
        audit_rows.append(
            {
                "candidate_code": code,
                "cohort_role": role,
                "foreground_fraction": foreground,
                "component_count": int(component_count),
                "touches_border": touches_border,
                "change_fraction_from_preannotation": change_fraction,
                "foreground_ratio_to_preannotation": foreground_ratio,
                "identical_to_preannotation": identical,
                "progress_status": status,
                "qc_flags": ";".join(sorted(set(flags))),
                "requires_review": bool(flags),
            }
        )

    audit = pd.DataFrame(audit_rows).sort_values("candidate_code", kind="stable")
    output.mkdir(parents=True)
    audit_path = output / f"{role}_annotation_qc_private.csv"
    audit.to_csv(audit_path, index=False)
    flagged = audit[audit["requires_review"]]
    flag_counts: dict[str, int] = {}
    for value in flagged["qc_flags"]:
        for flag in str(value).split(";"):
            if flag:
                flag_counts[flag] = flag_counts.get(flag, 0) + 1
    summary = {
        "artifact": "Private lung ROI annotation quality control",
        "cohort_role": role,
        "worklist_cases": int(len(worklist)),
        "saved_annotations": int(audit["foreground_fraction"].notna().sum()),
        "binary_shape_validated": int(
            len(audit)
            - sum(
                audit["qc_flags"].str.contains(
                    "shape_mismatch|nonbinary_mask|missing_annotation", regex=True
                )
            )
        ),
        "identical_to_preannotation": int(audit["identical_to_preannotation"].sum()),
        "changed_from_preannotation": int((~audit["identical_to_preannotation"]).sum()),
        "requires_focused_review": int(audit["requires_review"].sum()),
        "flag_counts": dict(sorted(flag_counts.items())),
        "foreground_fraction_quantiles": _quantiles(
            audit["foreground_fraction"].dropna().astype(float).tolist()
        ),
        "change_fraction_quantiles": _quantiles(
            audit["change_fraction_from_preannotation"].dropna().astype(float).tolist()
        ),
        "thresholds": {
            "min_foreground_fraction": min_foreground_fraction,
            "max_foreground_fraction": max_foreground_fraction,
            "max_components": max_components,
            "max_change_fraction": max_change_fraction,
            "min_foreground_ratio": min_foreground_ratio,
            "max_foreground_ratio": max_foreground_ratio,
        },
        "audit_csv_sha256": sha256_file(audit_path),
        "candidate_identifiers_publication_allowed": False,
        "annotation_masks_publication_allowed": False,
    }
    summary_path = output / f"{role}_annotation_qc_summary_private.json"
    summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return {"audit": audit, "summary": summary, "summary_path": summary_path}

