#!/usr/bin/env python3
"""Lock patient-disjoint PadChest report-generation cohorts and protocol."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from cxr_thesis.objective6.cohorts import (
    FRONTAL_PROJECTIONS,
    PROJECTION_RANK,
    canonical_patient_id,
    derive_padchest_age,
    patient_partition,
    private_case_code,
)
from cxr_thesis.objective6.text import normalise_report, tokenise_report


AUDIT_SHA256 = "f69eab8e5b9d8a9608d71ecd1c0dfef3dde83791e86c1b4bb36e4ce410ed4be0"
EXPECTED_CANDIDATE_STUDIES = 42066
EXPECTED_CANDIDATE_PATIENTS = 25342
EXPECTED_OBJECTIVE5_PATIENTS = 40000
OBJECTIVE5_NAMES = (
    "padchest_adaptation_train_private.csv",
    "padchest_target_validation_private.csv",
    "padchest_locked_test_private.csv",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(payload: dict[str, Any], path: Path) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def write_json_with_checksum(payload: dict[str, Any], path: Path) -> str:
    atomic_json(payload, path)
    digest = sha256(path)
    path.with_suffix(path.suffix + ".sha256").write_text(
        f"{digest}  {path.name}\n", encoding="utf-8"
    )
    return digest


def find_column(frame: pd.DataFrame, name: str) -> str:
    matches = [column for column in frame.columns if str(column).casefold() == name.casefold()]
    if len(matches) != 1:
        raise ValueError(f"Expected exactly one {name} column")
    return str(matches[0])


def objective5_patients(root: Path) -> tuple[set[str], dict[str, str]]:
    patients: set[str] = set()
    hashes: dict[str, str] = {}
    for name in OBJECTIVE5_NAMES:
        path = root / name
        if not path.is_file():
            raise FileNotFoundError(path)
        frame = pd.read_csv(path, low_memory=False)
        patient_column = find_column(frame, "patient_id")
        role = {canonical_patient_id(value) for value in frame[patient_column]}
        role.discard("")
        patients.update(role)
        hashes[name] = sha256(path)
    if len(patients) != EXPECTED_OBJECTIVE5_PATIENTS:
        raise RuntimeError(
            f"Expected {EXPECTED_OBJECTIVE5_PATIENTS} Objective 5 patients, "
            f"found {len(patients)}"
        )
    return patients, hashes


def length_summary(reports: pd.Series) -> dict[str, float]:
    lengths = np.asarray([len(tokenise_report(value)) for value in reports], dtype=float)
    points = np.quantile(lengths, [0.0, 0.5, 0.95, 0.99, 1.0])
    return {
        name: float(value)
        for name, value in zip(
            ("minimum", "median", "p95", "p99", "maximum"), points, strict=True
        )
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metadata-csv", type=Path, required=True)
    parser.add_argument("--image-root", type=Path, required=True)
    parser.add_argument("--objective5-private-root", type=Path, required=True)
    parser.add_argument("--audit-summary", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--repository-commit", required=True)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    for path in (args.metadata_csv, args.audit_summary):
        if not path.is_file():
            raise FileNotFoundError(path)
    if not args.image_root.is_dir() or not args.objective5_private_root.is_dir():
        raise FileNotFoundError("PadChest image or Objective 5 private root is missing")
    if args.output_dir.exists():
        raise FileExistsError(f"Output exists and will not be overwritten: {args.output_dir}")
    if sha256(args.audit_summary) != AUDIT_SHA256:
        raise RuntimeError("Objective 6 feasibility audit hash changed")

    public = args.output_dir / "public"
    private = args.output_dir / "private"
    public.mkdir(parents=True)
    private.mkdir(parents=True)

    protocol: dict[str, Any] = {
        "artifact": "Objective 6 clinical report-generation protocol",
        "version": "v1.0.0",
        "status": "locked before cohort materialisation, vocabulary fitting, training, and evaluation",
        "dataset": "PadChest-small",
        "report_language": "Spanish",
        "source_metadata_sha256": sha256(args.metadata_csv),
        "feasibility_audit_sha256": AUDIT_SHA256,
        "seed": args.seed,
        "eligibility": {
            "adult_only": True,
            "frontal_projections": list(FRONTAL_PROJECTIONS),
            "nonempty_report": True,
            "available_image": True,
            "one_representative_image_per_study": True,
            "exclude_all_objective5_padchest_patients": True,
        },
        "split": {
            "unit": "patient",
            "algorithm": "SHA-256 deterministic patient partition",
            "training_fraction": 0.70,
            "validation_fraction": 0.15,
            "locked_test_fraction": 0.15,
            "uses_labels": False,
            "uses_report_content_after_nonempty_eligibility": False,
            "uses_predictions": False,
            "patient_disjoint": True,
        },
        "model_comparisons": [
            "nearest-training-image report retrieval",
            "image-only DenseNet-Transformer",
            "image-plus-clinical DenseNet-Transformer",
        ],
        "primary_model": {
            "visual_encoder": "Objective 5 PadChest-adapted DenseNet-121",
            "visual_encoder_sha256": "109db89a723c6e2f24442cb5866bfcf4084e85083936cda91bce3b8ae4365d9d",
            "clinical_inputs": ["age", "sex", "projection"],
            "decoder": "four-layer autoregressive Transformer",
            "ground_truth_labels_as_decoder_input": False,
        },
        "training": {
            "vocabulary_fit": "training reports only",
            "validation_only_model_selection": True,
            "private_epoch_recovery": True,
        },
        "evaluation": {
            "locked_test_evaluations": 1,
            "primary_population": "all locked-test studies",
            "sensitivity_population": "locked-test references not exactly present in training",
            "lexical_metrics": [
                "BLEU-1", "BLEU-2", "BLEU-3", "BLEU-4", "ROUGE-L", "METEOR", "CIDEr"
            ],
            "clinical_metrics": [
                "concept precision", "concept recall", "concept F1", "negation error rate"
            ],
            "safety_metrics": [
                "empty-report rate", "repetition rate", "training-report exact-match rate"
            ],
            "confidence_intervals": "1000-replicate patient-cluster bootstrap",
        },
        "privacy": {
            "raw_reports_public": False,
            "identifiers_public": False,
            "private_manifests_public": False,
            "case_level_predictions_public": False,
            "medical_images_public": False,
        },
    }
    protocol_path = public / "objective6_report_generation_protocol_public.json"
    protocol_hash = write_json_with_checksum(protocol, protocol_path)

    exclusions, objective5_hashes = objective5_patients(args.objective5_private_root)
    metadata = pd.read_csv(args.metadata_csv, low_memory=False)
    columns = {
        name: find_column(metadata, name)
        for name in (
            "ImageID", "StudyID", "ReportID", "Report", "PatientID",
            "PatientBirth", "StudyDate_DICOM", "PatientSex_DICOM",
            "Projection", "Pediatric", "Labels", "LabelsLocalizationsBySentence",
        )
    }
    frame = metadata[list(columns.values())].rename(
        columns={value: key for key, value in columns.items()}
    )
    frame["report_normalised"] = frame["Report"].map(normalise_report)
    frame["age"] = derive_padchest_age(frame["StudyDate_DICOM"], frame["PatientBirth"])
    pediatric = frame["Pediatric"].astype(str).str.strip().str.casefold()
    frame["patient_key"] = frame["PatientID"].map(canonical_patient_id)
    frame["image_path"] = frame["ImageID"].map(lambda value: str(args.image_root / str(value)))
    frame["image_exists"] = frame["image_path"].map(lambda value: Path(value).is_file())
    eligible = frame[
        frame["Projection"].astype(str).isin(FRONTAL_PROJECTIONS)
        & frame["report_normalised"].str.len().gt(0)
        & frame["age"].ge(18)
        & ~pediatric.isin({"yes", "true", "1", "si", "sí", "y"})
        & frame["patient_key"].str.len().gt(0)
        & frame["image_exists"]
        & ~frame["patient_key"].isin(exclusions)
    ].copy()
    eligible["projection_rank"] = eligible["Projection"].map(PROJECTION_RANK)
    eligible = eligible.sort_values(
        ["StudyID", "projection_rank", "ImageID"], kind="stable"
    ).drop_duplicates("StudyID", keep="first")
    if len(eligible) != EXPECTED_CANDIDATE_STUDIES:
        raise RuntimeError(f"Candidate studies changed: {len(eligible)}")
    if eligible["patient_key"].nunique() != EXPECTED_CANDIDATE_PATIENTS:
        raise RuntimeError("Candidate patient count changed")

    eligible["split"] = eligible["patient_key"].map(
        lambda value: patient_partition(value, seed=args.seed)
    )
    eligible["case_code"] = [
        private_case_code(patient, study, seed=args.seed)
        for patient, study in zip(eligible["patient_key"], eligible["StudyID"], strict=True)
    ]
    eligible["sex"] = eligible["PatientSex_DICOM"]
    eligible["view"] = eligible["Projection"].map(
        lambda value: "AP" if str(value).startswith("AP") else str(value)
    )

    output_columns = [
        "case_code", "image_path", "ImageID", "PatientID", "StudyID", "ReportID",
        "report_normalised", "age", "sex", "view", "Projection", "Labels",
        "LabelsLocalizationsBySentence", "split",
    ]
    manifest_hashes: dict[str, str] = {}
    split_frames: dict[str, pd.DataFrame] = {}
    for split in ("train", "val", "test"):
        selected = eligible[eligible["split"].eq(split)][output_columns].copy()
        selected = selected.rename(
            columns={
                "ImageID": "image_id", "PatientID": "patient_id",
                "StudyID": "study_id", "ReportID": "report_id",
                "Projection": "projection", "Labels": "labels",
                "LabelsLocalizationsBySentence": "labels_by_sentence",
                "report_normalised": "report",
            }
        ).sort_values(["patient_id", "study_id", "image_id"], kind="stable")
        path = private / f"{split}_report_cohort_private.csv"
        selected.to_csv(path, index=False, lineterminator="\n")
        manifest_hashes[path.name] = sha256(path)
        path.with_suffix(".csv.sha256").write_text(
            f"{manifest_hashes[path.name]}  {path.name}\n", encoding="utf-8"
        )
        split_frames[split] = selected

    patient_sets = {
        split: {canonical_patient_id(value) for value in frame["patient_id"]}
        for split, frame in split_frames.items()
    }
    overlaps = {
        "train_vs_val": len(patient_sets["train"] & patient_sets["val"]),
        "train_vs_test": len(patient_sets["train"] & patient_sets["test"]),
        "val_vs_test": len(patient_sets["val"] & patient_sets["test"]),
    }
    if any(overlaps.values()):
        raise RuntimeError(f"Patient leakage detected: {overlaps}")

    train_reports = set(split_frames["train"]["report"])
    summary: dict[str, Any] = {
        "artifact": "Objective 6 locked report-generation cohort summary",
        "version": "v1.0.0",
        "protocol_sha256": protocol_hash,
        "repository_commit": args.repository_commit,
        "source_metadata_sha256": sha256(args.metadata_csv),
        "objective5_manifest_sha256": objective5_hashes,
        "objective5_patients_excluded": len(exclusions),
        "candidate_studies": len(eligible),
        "candidate_patients": int(eligible["patient_key"].nunique()),
        "split_cases": {split: len(frame) for split, frame in split_frames.items()},
        "split_patients": {split: len(patient_sets[split]) for split in split_frames},
        "patient_overlap": overlaps,
        "report_token_length": {
            split: length_summary(frame["report"]) for split, frame in split_frames.items()
        },
        "validation_reference_seen_exactly_in_training_fraction": float(
            split_frames["val"]["report"].isin(train_reports).mean()
        ),
        "locked_test_reference_seen_exactly_in_training_fraction": float(
            split_frames["test"]["report"].isin(train_reports).mean()
        ),
        "locked_test_novel_reference_cases": int(
            (~split_frames["test"]["report"].isin(train_reports)).sum()
        ),
        "private_manifest_sha256": manifest_hashes,
        "split_membership_used_labels": False,
        "split_membership_used_report_content": False,
        "split_membership_used_predictions": False,
        "locked_test_reference_content_used_for_selection": False,
        "locked_test_reference_aggregated_only_after_membership_lock": True,
        "model_training_performed": False,
        "model_inference_performed": False,
        "locked_test_evaluated": False,
        "raw_reports_public": False,
        "identifiers_public": False,
    }
    summary_path = public / "objective6_report_cohort_summary_public.json"
    summary_hash = write_json_with_checksum(summary, summary_path)
    lock: dict[str, Any] = {
        "artifact": "Final Objective 6 pre-training protocol lock",
        "immutable": True,
        "protocol_sha256": protocol_hash,
        "cohort_summary_sha256": summary_hash,
        "private_manifest_sha256": manifest_hashes,
        "training_started": False,
        "locked_test_evaluated": False,
        "locked_test_evaluation_count": 0,
    }
    lock_path = public / "FINAL_OBJECTIVE6_PRETRAINING_LOCK.json"
    lock_hash = write_json_with_checksum(lock, lock_path)

    print(json.dumps(summary, indent=2, sort_keys=True))
    print("\n--- FINAL OBJECTIVE 6 PRE-TRAINING LOCK ---")
    print("Protocol SHA-256:", protocol_hash)
    print("Cohort summary SHA-256:", summary_hash)
    print("Final lock SHA-256:", lock_hash)
    print("Training performed:", False)
    print("Locked test evaluated:", False)
    print("Private reports allowed for public upload:", False)
    print("OBJECTIVE 6 REPORT-GENERATION PROTOCOL AND COHORTS LOCKED SUCCESSFULLY")


if __name__ == "__main__":
    main()
