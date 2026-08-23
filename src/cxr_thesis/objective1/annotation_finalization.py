"""Freeze a reviewed private ROI annotation set and emit safe aggregates."""

from __future__ import annotations

import hashlib
import json
import shutil
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

from .annotation_workspace import load_annotation_worklist


STRUCTURAL_QC_FLAGS = {
    "missing_image",
    "missing_annotation",
    "shape_mismatch",
    "nonbinary_mask",
    "empty_mask",
    "full_mask",
    "missing_progress_record",
    "progress_needs_review",
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_unique(path: Path, *, required: set[str], label: str) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(path)
    frame = pd.read_csv(path, keep_default_na=False)
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"{label} is missing columns: {missing}")
    if frame["candidate_code"].astype(str).duplicated().any():
        raise ValueError(f"{label} contains duplicate candidate codes")
    return frame


def _truthy(values: pd.Series) -> pd.Series:
    return values.astype(str).str.lower().isin({"true", "1", "yes"})


def finalize_reviewed_annotation_set(
    workspace: str | Path,
    role: str,
    *,
    qc_audit_path: str | Path,
    focused_review_path: str | Path,
    provenance_path: str | Path,
    output_dir: str | Path,
    expected_cases: int,
) -> dict[str, object]:
    """Validate, hash, and freeze a reviewed annotation set without copying masks."""

    output = Path(output_dir).resolve()
    if output.exists():
        raise FileExistsError(f"Annotation lock output already exists: {output}")
    worklist, role_root = load_annotation_worklist(workspace, role)
    worklist = worklist.copy()
    worklist["candidate_code"] = worklist["candidate_code"].astype(str)
    if len(worklist) != expected_cases:
        raise ValueError(
            f"Expected {expected_cases} worklist cases, found {len(worklist)}"
        )
    codes = set(worklist["candidate_code"])

    progress = _read_unique(
        role_root / "annotation_progress.csv",
        required={"candidate_code", "cohort_role", "status"},
        label="Annotation progress",
    )
    progress = progress[progress["cohort_role"].astype(str) == role].copy()
    if set(progress["candidate_code"].astype(str)) != codes:
        raise ValueError("Annotation progress does not match the role worklist")
    if set(progress["status"].astype(str)) != {"complete"}:
        raise ValueError("Annotation progress contains unresolved cases")

    qc_path = Path(qc_audit_path).resolve()
    qc = _read_unique(
        qc_path,
        required={
            "candidate_code",
            "cohort_role",
            "qc_flags",
            "requires_review",
            "foreground_fraction",
        },
        label="Annotation QC audit",
    )
    qc = qc[qc["cohort_role"].astype(str) == role].copy()
    qc["candidate_code"] = qc["candidate_code"].astype(str)
    if set(qc["candidate_code"]) != codes:
        raise ValueError("Annotation QC audit does not match the role worklist")
    structural_violations = 0
    for value in qc["qc_flags"].astype(str):
        structural_violations += int(
            bool({flag for flag in value.split(";") if flag} & STRUCTURAL_QC_FLAGS)
        )
    if structural_violations:
        raise ValueError("Annotation QC contains structural integrity violations")

    review_path = Path(focused_review_path).resolve()
    review = _read_unique(
        review_path,
        required={
            "candidate_code",
            "cohort_role",
            "review_action",
            "review_status",
        },
        label="Focused-QC review log",
    )
    review = review[review["cohort_role"].astype(str) == role].copy()
    review["candidate_code"] = review["candidate_code"].astype(str)
    if not set(review["candidate_code"]).issubset(codes):
        raise ValueError("Focused-QC review log contains cases outside the worklist")
    if set(review["review_status"].astype(str)) != {"resolved"}:
        raise ValueError("Focused-QC review log contains unresolved cases")
    flagged_codes = set(qc.loc[_truthy(qc["requires_review"]), "candidate_code"])
    if not flagged_codes.issubset(set(review["candidate_code"])):
        raise ValueError("A QC-flagged case lacks a resolved focused review")

    provenance_file = Path(provenance_path).resolve()
    if not provenance_file.is_file():
        raise FileNotFoundError(provenance_file)
    provenance = json.loads(provenance_file.read_text(encoding="utf-8"))
    if provenance.get("cohort_role") != role:
        raise ValueError("Annotation provenance role mismatch")
    if int(provenance.get("cases_reviewed_by_radiologist", 0)) != expected_cases:
        raise ValueError("Radiologist-review coverage is incomplete")
    if float(provenance.get("review_coverage_fraction", 0.0)) != 1.0:
        raise ValueError("Radiologist-review coverage is not 100 percent")
    if not bool(provenance.get("anonymous_description_permitted", False)):
        raise ValueError("Anonymous reviewer description was not permitted")

    qc_by_code = qc.set_index("candidate_code")
    manifest_rows: list[dict[str, object]] = []
    combined_digest = hashlib.sha256()
    for _, row in worklist.sort_values("candidate_code", kind="stable").iterrows():
        code = str(row["candidate_code"])
        image_path = role_root / str(row["image_filename"])
        mask_path = role_root / str(row["required_output_mask"])
        if not image_path.is_file() or not mask_path.is_file():
            raise FileNotFoundError("A worklist image or final mask is missing")
        with Image.open(image_path) as handle:
            image_shape = np.asarray(handle.convert("L")).shape
        with Image.open(mask_path) as handle:
            mask = np.asarray(handle.convert("L"), dtype=np.uint8)
        if mask.shape != image_shape:
            raise ValueError("A final mask does not match its source-image shape")
        if not set(np.unique(mask).tolist()).issubset({0, 255}):
            raise ValueError("A final mask is not binary 0/255")
        foreground = float(np.mean(mask > 0))
        if not np.isclose(
            foreground,
            float(qc_by_code.at[code, "foreground_fraction"]),
            rtol=0.0,
            atol=1e-12,
        ):
            raise ValueError("A final mask changed after the supplied QC audit")
        mask_hash = _sha256(mask_path)
        combined_digest.update(f"{code}\t{mask_hash}\n".encode("utf-8"))
        manifest_rows.append(
            {
                "candidate_code": code,
                "cohort_role": role,
                "mask_relative_path": str(mask_path.relative_to(role_root)),
                "mask_sha256": mask_hash,
                "mask_bytes": int(mask_path.stat().st_size),
                "foreground_fraction": foreground,
            }
        )

    parent = output.parent
    parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{output.name}-", dir=parent))
    try:
        private_dir = temporary / "private"
        public_dir = temporary / "public"
        private_dir.mkdir()
        public_dir.mkdir()
        manifest_path = private_dir / f"{role}_final_mask_manifest_private.csv"
        pd.DataFrame(manifest_rows).to_csv(manifest_path, index=False)
        manifest_hash = _sha256(manifest_path)
        mask_set_hash = combined_digest.hexdigest()
        identical_mask = _truthy(qc["identical_to_preannotation"])
        changed = int((~identical_mask).sum())
        identical = int(identical_mask.sum())
        corrected = int((review["review_action"] == "corrected").sum())
        approved = int((review["review_action"] == "approved_as_is").sum())
        common = {
            "artifact": "Final reviewed lung ROI annotation-set lock",
            "cohort_role": role,
            "cases": expected_cases,
            "reviewer_code": str(provenance.get("reviewer_code", "")),
            "reviewer_professional_qualification": str(
                provenance.get("reviewer_professional_qualification", "")
            ),
            "reviewer_years_experience": int(
                provenance.get("reviewer_years_experience", 0)
            ),
            "review_mode": str(provenance.get("review_mode", "")),
            "radiologist_review_coverage_fraction": 1.0,
            "final_masks_changed_from_preannotation": changed,
            "final_masks_identical_to_preannotation": identical,
            "focused_review_cases": int(len(review)),
            "focused_review_corrected": corrected,
            "focused_review_approved_unchanged": approved,
            "remaining_conservative_qc_flags": int(len(flagged_codes)),
            "qc_flags_without_resolved_review": 0,
            "structural_integrity_violations": 0,
            "final_mask_manifest_sha256": manifest_hash,
            "final_mask_set_sha256": mask_set_hash,
            "qc_audit_sha256": _sha256(qc_path),
            "focused_review_log_sha256": _sha256(review_path),
            "provenance_record_sha256": _sha256(provenance_file),
            "locked_target_test_used": False,
        }
        private_record = {
            **common,
            "private_mask_manifest": manifest_path.name,
            "private_record_publication_allowed": False,
        }
        public_summary = {
            **common,
            "patient_or_image_identifiers_included": False,
            "medical_images_included": False,
            "annotation_masks_included": False,
            "private_manifests_included": False,
            "aggregate_summary_publication_allowed": True,
        }
        private_record_path = private_dir / f"{role}_annotation_lock_private.json"
        public_summary_path = public_dir / f"{role}_annotation_summary_public.json"
        private_record_path.write_text(
            json.dumps(private_record, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        public_summary_path.write_text(
            json.dumps(public_summary, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        for path in (manifest_path, private_record_path, public_summary_path):
            path.with_suffix(path.suffix + ".sha256").write_text(
                f"{_sha256(path)}  {path.name}\n", encoding="utf-8"
            )
        temporary.replace(output)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise

    return {
        "summary": public_summary,
        "output_dir": output,
        "private_manifest_sha256": manifest_hash,
        "public_summary_sha256": _sha256(
            output / "public" / f"{role}_annotation_summary_public.json"
        ),
    }
