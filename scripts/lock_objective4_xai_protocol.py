#!/usr/bin/env python3
"""Lock a deterministic, validation-only cohort for Objective 4 XAI."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd
import yaml


LABELS = (
    "Infiltration", "Effusion", "Atelectasis", "Nodule", "Mass",
    "Consolidation", "Pneumothorax", "Pleural_Thickening",
    "Cardiomegaly", "Emphysema", "Edema", "Fibrosis",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def stable_rank(seed: int, label: str, patient: object, image: object) -> str:
    value = f"{seed}|{label}|{patient}|{image}".encode("utf-8")
    return hashlib.sha256(value).hexdigest()


def select_cohort(
    manifest: pd.DataFrame, *, seed: int, cases_per_label: int
) -> pd.DataFrame:
    required = {"patient_id", "image_id", "split", *LABELS}
    missing = sorted(required - set(manifest.columns))
    if missing:
        raise ValueError(f"Validation manifest columns are missing: {missing}")
    if set(manifest["split"].astype(str).str.lower()) != {"val"}:
        raise ValueError("Objective 4 selection accepts validation rows only")
    if manifest["image_id"].duplicated().any():
        raise ValueError("Validation manifest contains duplicate image identifiers")

    selected: list[pd.Series] = []
    used_patients: set[str] = set()
    used_images: set[str] = set()
    positive_counts = {
        label: int(pd.to_numeric(manifest[label], errors="raise").sum())
        for label in LABELS
    }
    # Allocate rare labels first so common multi-label cases cannot exhaust them.
    allocation_order = sorted(LABELS, key=lambda label: (positive_counts[label], label))
    target_index = {label: index for index, label in enumerate(LABELS)}

    for label in allocation_order:
        candidates = manifest.loc[pd.to_numeric(manifest[label]) == 1].copy()
        candidates["_rank"] = [
            stable_rank(seed, label, patient, image)
            for patient, image in zip(
                candidates["patient_id"], candidates["image_id"], strict=True
            )
        ]
        candidates = candidates.sort_values(
            ["_rank", "patient_id", "image_id"], kind="stable"
        )
        chosen = 0
        for _, row in candidates.iterrows():
            patient = str(row["patient_id"])
            image = str(row["image_id"])
            if patient in used_patients or image in used_images:
                continue
            output = row.drop(labels=["_rank"]).copy()
            output["xai_target_label"] = label
            output["xai_target_index"] = target_index[label]
            output["xai_selection_rank"] = chosen + 1
            selected.append(output)
            used_patients.add(patient)
            used_images.add(image)
            chosen += 1
            if chosen == cases_per_label:
                break
        if chosen != cases_per_label:
            raise RuntimeError(
                f"Could not select {cases_per_label} unique patients for {label}; "
                f"selected {chosen}"
            )

    return pd.DataFrame(selected).sort_values(
        ["xai_target_index", "xai_selection_rank"], kind="stable"
    ).reset_index(drop=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--validation-manifest", type=Path, required=True)
    parser.add_argument("--expected-validation-sha256", required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--expected-checkpoint-sha256", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.output_root.exists():
        raise FileExistsError(
            f"Objective 4 protocol output already exists: {args.output_root}"
        )
    actual_validation_hash = sha256(args.validation_manifest)
    if actual_validation_hash != args.expected_validation_sha256:
        raise RuntimeError("Validation manifest SHA-256 does not match the frozen cohort")

    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    seed = int(config["seed"])
    total_cases = int(config["cohort"]["cases"])
    if total_cases % len(LABELS):
        raise ValueError("Configured cohort size must be divisible by 12 labels")
    cases_per_label = total_cases // len(LABELS)
    manifest = pd.read_csv(args.validation_manifest)
    cohort = select_cohort(manifest, seed=seed, cases_per_label=cases_per_label)

    checks = {
        "case_count": len(cohort) == total_cases,
        "patient_unique": cohort["patient_id"].astype(str).nunique() == total_cases,
        "image_unique": cohort["image_id"].astype(str).nunique() == total_cases,
        "validation_only": set(cohort["split"].astype(str).str.lower()) == {"val"},
        "balanced": set(cohort["xai_target_label"].value_counts()) == {cases_per_label},
    }
    if not all(checks.values()):
        raise RuntimeError(f"Objective 4 cohort validation failed: {checks}")

    private_root = args.output_root / "private"
    public_root = args.output_root / "public"
    private_root.mkdir(parents=True)
    public_root.mkdir(parents=True)
    private_manifest = private_root / "xai_validation_cohort_private.csv"
    cohort.to_csv(private_manifest, index=False, lineterminator="\n")
    private_hash = sha256(private_manifest)
    (private_root / "xai_validation_cohort_private.sha256").write_text(
        f"{private_hash}  {private_manifest.name}\n", encoding="utf-8"
    )
    label_counts = cohort["xai_target_label"].value_counts().sort_index().to_dict()
    protocol = {
        "artifact": "Objective 4 quantitative XAI protocol lock",
        "status": "locked_before_explanation_generation",
        "objective": 4,
        "seed": seed,
        "model": config["model"],
        "model_selection_basis": config["checkpoint_source"],
        "expected_checkpoint_sha256": args.expected_checkpoint_sha256,
        "source_validation_manifest_sha256": actual_validation_hash,
        "private_xai_cohort_sha256": private_hash,
        "cohort": {
            "split": "val", "cases": total_cases,
            "unique_patients": total_cases, "unique_images": total_cases,
            "cases_per_target_label": cases_per_label,
            "target_label_counts": label_counts,
            "selection": "deterministic_label_positive_patient_unique",
            "labels_used_for_selection": True,
            "predictions_used_for_selection": False,
            "risk_scores_used_for_selection": False,
        },
        "methods": config["methods"],
        "metrics": config["metrics"],
        "reporting": config["reporting"],
        "protections": {
            "test_manifest_opened": False, "test_labels_accessed": False,
            "test_evaluated": False, "threshold_tuning": False,
            "manual_masking_required": False,
            "patient_identifiers_public": False,
            "image_identifiers_public": False,
            "medical_images_public": False,
            "case_level_explanations_public": False,
            "private_manifest_allowed_for_public_upload": False,
        },
    }
    protocol_path = public_root / "objective4_xai_protocol_lock_public.json"
    protocol_path.write_text(
        json.dumps(protocol, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    protocol_hash = sha256(protocol_path)
    (public_root / "objective4_xai_protocol_lock_public.sha256").write_text(
        f"{protocol_hash}  {protocol_path.name}\n", encoding="utf-8"
    )

    print("--- OBJECTIVE 4 XAI PROTOCOL LOCK ---")
    print("Validation cases available:", len(manifest))
    print("XAI cases selected:", len(cohort))
    print("Unique patients:", cohort["patient_id"].astype(str).nunique())
    print("Cases per target label:", cases_per_label)
    print("Target-label counts:", json.dumps(label_counts, sort_keys=True))
    print("Validation manifest SHA-256:", actual_validation_hash)
    print("Private cohort SHA-256:", private_hash)
    print("Public protocol SHA-256:", protocol_hash)
    print("Predictions used for selection:", False)
    print("Risk scores used for selection:", False)
    print("Test manifest opened:", False)
    print("Test labels accessed:", False)
    print("Test evaluated:", False)
    print("Manual masking required:", False)
    print("Private cohort allowed for public upload:", False)
    print("OBJECTIVE 4 VALIDATION-ONLY XAI PROTOCOL LOCK SUCCESSFUL")


if __name__ == "__main__":
    main()
