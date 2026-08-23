"""Private workspace helpers for manual lung-ROI annotation."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image


ROLE_POLICIES = {
    "adaptation_train": True,
    "target_validation": True,
    "locked_target_test": False,
}

PROJECTION_DECISIONS = {
    "eligible_frontal",
    "ineligible_lateral",
    "ineligible_other",
    "uncertain",
}

REQUIRED_WORKLIST_COLUMNS = {
    "candidate_code",
    "cohort_role",
    "image_filename",
    "preannotation_filename",
    "required_output_mask",
}


@dataclass(frozen=True)
class AnnotationCase:
    candidate_code: str
    role: str
    image_path: Path
    preannotation_path: Path | None
    output_path: Path


@dataclass(frozen=True)
class ProjectionAuditCase:
    candidate_code: str
    role: str
    image_path: Path


def load_projection_audit_worklist(
    workspace: str | Path,
    role: str,
) -> tuple[pd.DataFrame, Path]:
    """Load image-only audit inputs without resolving any prediction path."""
    if role not in ROLE_POLICIES:
        raise ValueError(f"Unsupported annotation role: {role!r}")
    role_root = Path(workspace).resolve() / role
    worklist_path = role_root / "annotation_worklist.csv"
    if not worklist_path.is_file():
        raise FileNotFoundError(f"Annotation worklist not found: {worklist_path}")
    frame = pd.read_csv(worklist_path, keep_default_na=False)
    required = {"candidate_code", "cohort_role", "image_filename"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"Projection worklist is missing columns: {missing}")
    if frame.empty:
        raise ValueError("Projection worklist is empty")
    if frame["candidate_code"].duplicated().any():
        raise ValueError("Projection worklist contains duplicate candidate codes")
    if set(frame["cohort_role"].astype(str)) != {role}:
        raise ValueError("Projection worklist role mismatch")
    return frame.reset_index(drop=True), role_root


def resolve_projection_audit_cases(
    frame: pd.DataFrame,
    role_root: str | Path,
    *,
    role: str,
) -> list[ProjectionAuditCase]:
    """Resolve only source-image paths; mask fields are never read."""
    root = Path(role_root).resolve()
    cases: list[ProjectionAuditCase] = []
    for _, row in frame.iterrows():
        image_path = root / str(row["image_filename"])
        if not image_path.is_file():
            raise FileNotFoundError(f"Projection-audit image not found: {image_path}")
        cases.append(
            ProjectionAuditCase(
                candidate_code=str(row["candidate_code"]),
                role=role,
                image_path=image_path,
            )
        )
    return cases


def load_projection_image(case: ProjectionAuditCase) -> np.ndarray:
    with Image.open(case.image_path) as handle:
        return np.asarray(handle.convert("L"), dtype=np.uint8)


def update_projection_audit(
    audit_path: str | Path,
    *,
    candidate_code: str,
    role: str,
    auditor: str,
    decision: str,
    note: str,
) -> pd.DataFrame:
    """Atomically upsert one prediction-blind projection decision."""
    if decision not in PROJECTION_DECISIONS:
        raise ValueError(f"Unsupported projection decision: {decision!r}")
    path = Path(audit_path)
    columns = [
        "candidate_code",
        "cohort_role",
        "projection_decision",
        "auditor",
        "updated_utc",
        "note",
    ]
    if path.is_file():
        frame = pd.read_csv(path, keep_default_na=False).reindex(columns=columns)
        frame = frame[frame["candidate_code"].astype(str) != str(candidate_code)]
    else:
        frame = pd.DataFrame(columns=columns)
    row = pd.DataFrame(
        [
            {
                "candidate_code": str(candidate_code),
                "cohort_role": role,
                "projection_decision": decision,
                "auditor": auditor,
                "updated_utc": datetime.now(timezone.utc).isoformat(),
                "note": str(note).strip(),
            }
        ]
    )
    frame = pd.concat([frame, row], ignore_index=True).sort_values("candidate_code")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.tmp")
    frame.to_csv(temporary, index=False)
    temporary.replace(path)
    return frame.reset_index(drop=True)


def load_annotation_worklist(
    workspace: str | Path,
    role: str,
) -> tuple[pd.DataFrame, Path]:
    """Load and validate one role without inspecting any other role directory."""
    if role not in ROLE_POLICIES:
        raise ValueError(f"Unsupported annotation role: {role!r}")
    role_root = Path(workspace).resolve() / role
    worklist_path = role_root / "annotation_worklist.csv"
    if not worklist_path.is_file():
        raise FileNotFoundError(f"Annotation worklist not found: {worklist_path}")
    frame = pd.read_csv(worklist_path, keep_default_na=False)
    missing = sorted(REQUIRED_WORKLIST_COLUMNS - set(frame.columns))
    if missing:
        raise ValueError(f"Annotation worklist is missing columns: {missing}")
    if frame.empty:
        raise ValueError("Annotation worklist is empty")
    if frame["candidate_code"].duplicated().any():
        raise ValueError("Annotation worklist contains duplicate candidate codes")
    if set(frame["cohort_role"].astype(str)) != {role}:
        raise ValueError("Worklist cohort_role does not match the selected role")

    preannotations_allowed = ROLE_POLICIES[role]
    declared = frame["preannotation_filename"].astype(str).str.strip()
    preannotation_directory = role_root / "preannotations"
    if not preannotations_allowed:
        if declared.ne("").any():
            raise RuntimeError("Locked target test declares forbidden pre-annotations")
        if preannotation_directory.exists() and any(preannotation_directory.iterdir()):
            raise RuntimeError("Locked target test contains forbidden pre-annotation files")
    elif declared.eq("").any():
        raise ValueError(f"{role} requires a pre-annotation for every case")
    return frame.reset_index(drop=True), role_root


def resolve_annotation_case(
    row: pd.Series,
    role_root: str | Path,
    *,
    role: str,
) -> AnnotationCase:
    """Resolve one case and enforce role-specific prediction visibility."""
    root = Path(role_root).resolve()
    image_path = root / str(row["image_filename"])
    output_path = root / str(row["required_output_mask"])
    preannotation_text = str(row["preannotation_filename"]).strip()
    preannotation_path = root / preannotation_text if preannotation_text else None

    if not image_path.is_file():
        raise FileNotFoundError(f"Source image not found: {image_path}")
    if ROLE_POLICIES[role]:
        if preannotation_path is None or not preannotation_path.is_file():
            raise FileNotFoundError(f"Pre-annotation not found: {preannotation_path}")
    elif preannotation_path is not None:
        raise RuntimeError("Locked target test may not resolve a pre-annotation")
    return AnnotationCase(
        candidate_code=str(row["candidate_code"]),
        role=role,
        image_path=image_path,
        preannotation_path=preannotation_path,
        output_path=output_path,
    )


def load_annotation_case(case: AnnotationCase) -> tuple[np.ndarray, np.ndarray, str]:
    """Load a grayscale image and either a saved mask, pre-mask, or blank mask."""
    with Image.open(case.image_path) as handle:
        image = np.asarray(handle.convert("L"), dtype=np.uint8)

    if case.output_path.is_file():
        mask_path = case.output_path
        source = "saved_annotation"
    elif case.preannotation_path is not None:
        mask_path = case.preannotation_path
        source = "preannotation"
    else:
        mask_path = None
        source = "blank_prediction_blind"

    if mask_path is None:
        mask = np.zeros(image.shape, dtype=np.uint8)
    else:
        with Image.open(mask_path) as handle:
            mask = (np.asarray(handle.convert("L")) > 0).astype(np.uint8)
    if mask.shape != image.shape:
        raise ValueError(
            f"Image/mask shape mismatch for {case.candidate_code}: "
            f"{image.shape} versus {mask.shape}"
        )
    return image, mask, source


def save_binary_annotation(
    labels: np.ndarray,
    output_path: str | Path,
    *,
    expected_shape: tuple[int, int],
) -> dict[str, float | int | str]:
    """Atomically save a same-size binary 0/255 PNG annotation."""
    array = np.asarray(labels)
    if array.ndim != 2 or tuple(array.shape) != tuple(expected_shape):
        raise ValueError(
            f"Annotation shape {array.shape} does not match image {expected_shape}"
        )
    binary = (array > 0).astype(np.uint8)
    target = Path(output_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f"{target.stem}.tmp{target.suffix}")
    Image.fromarray(binary * 255, mode="L").save(temporary)
    temporary.replace(target)
    return {
        "output_path": str(target),
        "foreground_pixels": int(binary.sum()),
        "foreground_fraction": float(binary.mean()),
        "height": int(binary.shape[0]),
        "width": int(binary.shape[1]),
    }


def update_annotation_progress(
    progress_path: str | Path,
    *,
    candidate_code: str,
    role: str,
    annotator: str,
    foreground_fraction: float,
    needs_review: bool,
    note: str,
) -> pd.DataFrame:
    """Atomically upsert one case in the private annotation progress log."""
    path = Path(progress_path)
    columns = [
        "candidate_code",
        "cohort_role",
        "status",
        "annotator",
        "updated_utc",
        "foreground_fraction",
        "needs_review",
        "note",
    ]
    if path.is_file():
        frame = pd.read_csv(path, keep_default_na=False).reindex(columns=columns)
        frame = frame[frame["candidate_code"].astype(str) != str(candidate_code)]
    else:
        frame = pd.DataFrame(columns=columns)
    row = pd.DataFrame(
        [
            {
                "candidate_code": str(candidate_code),
                "cohort_role": role,
                "status": "needs_review" if needs_review else "complete",
                "annotator": annotator,
                "updated_utc": datetime.now(timezone.utc).isoformat(),
                "foreground_fraction": float(foreground_fraction),
                "needs_review": bool(needs_review),
                "note": str(note).strip(),
            }
        ]
    )
    frame = pd.concat([frame, row], ignore_index=True).sort_values("candidate_code")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.tmp")
    frame.to_csv(temporary, index=False)
    temporary.replace(path)
    return frame.reset_index(drop=True)
