#!/usr/bin/env python3
"""Read-only PadChest report-generation feasibility and leakage audit."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
from typing import Any

import numpy as np
import pandas as pd

from cxr_thesis.objective6.text import normalise_report, tokenise_report


FRONTAL_PROJECTIONS = ("PA", "AP", "AP_horizontal")
PROJECTION_RANK = {projection: rank for rank, projection in enumerate(FRONTAL_PROJECTIONS)}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def column(frame: pd.DataFrame, *names: str) -> str:
    lookup = {str(name).casefold(): str(name) for name in frame.columns}
    for name in names:
        if name.casefold() in lookup:
            return lookup[name.casefold()]
    raise ValueError(f"None of the required columns exist: {names}")


def optional_column(frame: pd.DataFrame, *names: str) -> str | None:
    lookup = {str(name).casefold(): str(name) for name in frame.columns}
    for name in names:
        if name.casefold() in lookup:
            return lookup[name.casefold()]
    return None


def canonical_identifier(value: object) -> str:
    text = str(value).strip()
    if text == "" or text.casefold() in {"nan", "none", "null"}:
        return ""
    digits = re.findall(r"\d+", text)
    if digits:
        return str(int(digits[-1]))
    return text.casefold()


def quantiles(values: list[int]) -> dict[str, float]:
    if not values:
        return {name: 0.0 for name in ("minimum", "p05", "p25", "median", "p75", "p95", "p99", "maximum")}
    array = np.asarray(values, dtype=np.float64)
    points = np.quantile(array, [0.0, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99, 1.0])
    return {
        name: float(value)
        for name, value in zip(
            ("minimum", "p05", "p25", "median", "p75", "p95", "p99", "maximum"),
            points,
            strict=True,
        )
    }


def discover_objective5_manifests(root: Path) -> list[Path]:
    required = {
        "padchest_adaptation_train_private.csv",
        "padchest_target_validation_private.csv",
        "padchest_locked_test_private.csv",
    }
    matches = {path.name: path for path in root.rglob("*.csv") if path.name in required}
    missing = sorted(required - set(matches))
    if missing:
        raise FileNotFoundError(
            "Objective 5 PadChest manifests are incomplete; missing: " + ", ".join(missing)
        )
    return [matches[name] for name in sorted(required)]


def excluded_patients(paths: list[Path]) -> tuple[set[str], list[dict[str, Any]]]:
    patients: set[str] = set()
    inventory: list[dict[str, Any]] = []
    for path in paths:
        frame = pd.read_csv(path, low_memory=False)
        patient_column = column(frame, "patient_id", "PatientID", "patient")
        role_patients = {
            canonical_identifier(value)
            for value in frame[patient_column]
            if canonical_identifier(value)
        }
        patients.update(role_patients)
        inventory.append(
            {
                "file": path.name,
                "rows": int(len(frame)),
                "patients": int(len(role_patients)),
                "sha256": sha256(path),
            }
        )
    return patients, inventory


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metadata-csv", required=True, type=Path)
    parser.add_argument("--image-root", required=True, type=Path)
    parser.add_argument("--objective5-private-root", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--expected-excluded-patients", type=int, default=40000)
    args = parser.parse_args()

    if not args.metadata_csv.is_file():
        raise FileNotFoundError(args.metadata_csv)
    if not args.image_root.is_dir():
        raise FileNotFoundError(args.image_root)
    if not args.objective5_private_root.is_dir():
        raise FileNotFoundError(args.objective5_private_root)
    if args.output_dir.exists():
        raise FileExistsError(
            f"Audit output already exists and will not be overwritten: {args.output_dir}"
        )

    manifests = discover_objective5_manifests(args.objective5_private_root)
    excluded, manifest_inventory = excluded_patients(manifests)
    if len(excluded) != args.expected_excluded_patients:
        raise RuntimeError(
            "Objective 5 exclusion set is not exact: "
            f"expected {args.expected_excluded_patients}, found {len(excluded)}"
        )

    metadata = pd.read_csv(args.metadata_csv, low_memory=False)
    image_column = column(metadata, "ImageID")
    study_column = column(metadata, "StudyID")
    report_id_column = column(metadata, "ReportID")
    report_column = column(metadata, "Report")
    patient_column = column(metadata, "PatientID")
    projection_column = column(metadata, "Projection")
    age_column = optional_column(metadata, "PatientAge", "Age")
    birth_column = optional_column(metadata, "PatientBirth", "BirthYear")
    study_date_column = optional_column(metadata, "StudyDate_DICOM", "StudyDate")
    if age_column is None and (birth_column is None or study_date_column is None):
        raise ValueError(
            "Age cannot be derived: expected an age column or both "
            "PatientBirth and StudyDate_DICOM"
        )
    pediatric_column = column(metadata, "Pediatric")

    selected_columns = [
        image_column,
        study_column,
        report_id_column,
        report_column,
        patient_column,
        projection_column,
        pediatric_column,
    ]
    if age_column is not None:
        selected_columns.append(age_column)
    else:
        assert birth_column is not None and study_date_column is not None
        selected_columns.extend([birth_column, study_date_column])
    frame = metadata[selected_columns].copy()
    rename = {
        image_column: "image_id",
        study_column: "study_id",
        report_id_column: "report_id",
        report_column: "report",
        patient_column: "patient_id",
        projection_column: "projection",
        pediatric_column: "pediatric",
    }
    if age_column is not None:
        rename[age_column] = "age"
    else:
        assert birth_column is not None and study_date_column is not None
        rename[birth_column] = "birth_year"
        rename[study_date_column] = "study_date"
    frame = frame.rename(columns=rename)
    total_rows = int(len(frame))
    frame["report_normalised"] = frame["report"].map(normalise_report)
    nonempty = frame["report_normalised"].str.len().gt(0)
    frontal = frame["projection"].astype(str).isin(FRONTAL_PROJECTIONS)
    if "age" in frame:
        age = pd.to_numeric(frame["age"], errors="coerce")
        age_source = "provided age column"
    else:
        birth_year = pd.to_numeric(frame["birth_year"], errors="coerce")
        study_date = pd.to_numeric(frame["study_date"], errors="coerce")
        study_year = np.floor(study_date / 10000.0)
        age = study_year - birth_year
        age_source = "StudyDate_DICOM year minus PatientBirth year"
    age = age.where(age.between(0, 120))
    pediatric = frame["pediatric"].astype(str).str.strip().str.casefold()
    explicit_pediatric = pediatric.isin({"yes", "true", "1", "si", "sí", "y"})
    adult = age.ge(18) & ~explicit_pediatric

    eligible = frame.loc[nonempty & frontal & adult].copy()
    eligible["patient_key"] = eligible["patient_id"].map(canonical_identifier)
    eligible = eligible[eligible["patient_key"].str.len().gt(0)]
    eligible["image_exists"] = eligible["image_id"].map(
        lambda value: (args.image_root / str(value)).is_file()
    )
    missing_images = int((~eligible["image_exists"]).sum())
    eligible = eligible[eligible["image_exists"]].copy()

    # Select one representative image for each study/report without reading its text.
    eligible["projection_rank"] = eligible["projection"].map(PROJECTION_RANK)
    eligible["selection_group"] = eligible["study_id"].astype(str)
    missing_study = eligible["selection_group"].isin({"", "nan", "None"})
    eligible.loc[missing_study, "selection_group"] = eligible.loc[
        missing_study, "report_id"
    ].astype(str)
    eligible = eligible.sort_values(
        ["selection_group", "projection_rank", "image_id"], kind="stable"
    ).drop_duplicates("selection_group", keep="first")

    reports = eligible["report_normalised"].tolist()
    token_lengths = [len(tokenise_report(report)) for report in reports]
    character_lengths = [len(report) for report in reports]
    report_counts = pd.Series(reports).value_counts()
    exact_duplicate_rows = int(report_counts[report_counts.gt(1)].sum())
    unique_tokens = {token for report in reports for token in tokenise_report(report)}

    before_exclusion_cases = int(len(eligible))
    before_exclusion_patients = int(eligible["patient_key"].nunique())
    remaining = eligible[~eligible["patient_key"].isin(excluded)].copy()
    excluded_cases = before_exclusion_cases - int(len(remaining))

    summary: dict[str, Any] = {
        "artifact": "Objective 6 PadChest report-data feasibility audit",
        "version": "v1.0.0",
        "source_metadata_sha256": sha256(args.metadata_csv),
        "source_rows": total_rows,
        "nonempty_report_rows": int(nonempty.sum()),
        "frontal_rows": int(frontal.sum()),
        "age_source": age_source,
        "rows_with_valid_age": int(age.notna().sum()),
        "adult_nonempty_frontal_rows_before_path_check": int(
            (nonempty & frontal & adult).sum()
        ),
        "missing_image_rows_after_eligibility": missing_images,
        "representative_studies_before_objective5_exclusion": before_exclusion_cases,
        "patients_before_objective5_exclusion": before_exclusion_patients,
        "objective5_exclusion": {
            "expected_patients": int(args.expected_excluded_patients),
            "verified_unique_patients": int(len(excluded)),
            "eligible_studies_removed": excluded_cases,
            "manifests": manifest_inventory,
        },
        "candidate_pool_after_objective5_exclusion": {
            "studies": int(len(remaining)),
            "patients": int(remaining["patient_key"].nunique()),
        },
        "report_statistics_before_objective5_exclusion": {
            "unique_normalised_reports": int(report_counts.size),
            "rows_belonging_to_repeated_exact_reports": exact_duplicate_rows,
            "unique_tokens": int(len(unique_tokens)),
            "character_length": quantiles(character_lengths),
            "token_length": quantiles(token_lengths),
        },
        "selection_read_report_content": False,
        "selection_read_labels": False,
        "selection_read_predictions": False,
        "objective5_locked_test_labels_accessed": False,
        "raw_reports_written": False,
        "patient_or_image_identifiers_written": False,
        "medical_images_copied": False,
        "model_inference_performed": False,
        "training_performed": False,
    }

    args.output_dir.mkdir(parents=True)
    output = args.output_dir / "objective6_report_data_audit_public.json"
    output.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    digest = sha256(output)
    checksum = output.with_suffix(output.suffix + ".sha256")
    checksum.write_text(f"{digest}  {output.name}\n", encoding="utf-8")

    print(json.dumps(summary, indent=2, sort_keys=True))
    print("\n--- OBJECTIVE 6 REPORT-DATA AUDIT STATUS ---")
    print("Public aggregate audit:", output)
    print("Audit SHA-256:", digest)
    print("Raw reports displayed:", False)
    print("Identifiers displayed:", False)
    print("Model training performed:", False)
    print("OBJECTIVE 6 PADCHEST REPORT-DATA AUDIT SUCCESSFUL")


if __name__ == "__main__":
    main()
