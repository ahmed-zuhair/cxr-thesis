#!/usr/bin/env python3
"""Create a new label-blind Objective 2 confirmation cohort."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

import pandas as pd

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = REPOSITORY_ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from cxr_thesis.objective2.cohort_recovery import (
    select_disjoint_confirmation_patients,
    serialize_cohort,
)

IDENTITY_COLUMNS = ("patient_id", "split")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--original-locked-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--expected-manifest-sha256", required=True)
    parser.add_argument("--expected-original-locked-sha256", required=True)
    parser.add_argument("--seed", type=int, default=3042)
    parser.add_argument("--target-images", type=int, default=5_000)
    parser.add_argument("--private-hf-repo")
    parser.add_argument("--private-hf-path")
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(payload: dict[str, object], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def main() -> None:
    args = parse_args()
    if args.output_dir.exists():
        raise FileExistsError(
            f"Confirmation output already exists: {args.output_dir}"
        )
    if not args.manifest.is_file():
        raise FileNotFoundError(args.manifest)
    if not args.original_locked_manifest.is_file():
        raise FileNotFoundError(args.original_locked_manifest)
    if bool(args.private_hf_repo) != bool(args.private_hf_path):
        raise ValueError(
            "--private-hf-repo and --private-hf-path must be supplied together"
        )
    manifest_hash = sha256_file(args.manifest)
    original_hash = sha256_file(args.original_locked_manifest)
    if manifest_hash != args.expected_manifest_sha256:
        raise RuntimeError("Full NIH manifest SHA-256 does not match")
    if original_hash != args.expected_original_locked_sha256:
        raise RuntimeError("Original locked-test SHA-256 does not match")

    # Both dataframes deliberately exclude all disease-label columns.
    identity = pd.read_csv(
        args.manifest,
        usecols=list(IDENTITY_COLUMNS),
        dtype={"patient_id": str, "split": str},
    )
    original_identity = pd.read_csv(
        args.original_locked_manifest,
        usecols=list(IDENTITY_COLUMNS),
        dtype={"patient_id": str, "split": str},
    )
    if set(original_identity["split"].str.lower()) != {"test"}:
        raise RuntimeError("Original locked cohort contains a non-test split")
    excluded_patients = set(original_identity["patient_id"])
    selected = select_disjoint_confirmation_patients(
        identity,
        excluded_patient_ids=excluded_patients,
        split="test",
        seed=args.seed,
        target_images=args.target_images,
    )

    # Full rows are loaded only after the identities are irreversibly selected.
    # No label statistic is calculated or displayed in this program.
    full = pd.read_csv(
        args.manifest,
        dtype={"patient_id": str, "study_id": str, "image_id": str},
    )
    payload = serialize_cohort(
        full,
        selected_patients=selected,
        row_order="manifest",
        selection_order=selected,
    )
    confirmation_hash = hashlib.sha256(payload).hexdigest()
    private_root = args.output_dir / "private"
    public_root = args.output_dir / "public"
    private_root.mkdir(parents=True)
    public_root.mkdir(parents=True)
    confirmation_manifest = private_root / "confirmation_cohort_private.csv"
    confirmation_manifest.write_bytes(payload)
    confirmation_checksum = confirmation_manifest.with_suffix(".sha256")
    confirmation_checksum.write_text(
        f"{confirmation_hash}  {confirmation_manifest.name}\n",
        encoding="utf-8",
    )
    confirmation = pd.read_csv(
        confirmation_manifest,
        usecols=list(IDENTITY_COLUMNS),
        dtype={"patient_id": str, "split": str},
    )
    confirmation_patients = set(confirmation["patient_id"])
    overlap = confirmation_patients & excluded_patients
    if len(confirmation) != args.target_images or overlap:
        raise RuntimeError("Final confirmation cohort validation failed")

    private_record = {
        "artifact": "Objective 2 private independent confirmation cohort lock",
        "source_manifest_sha256": manifest_hash,
        "original_locked_test_sha256": original_hash,
        "confirmation_manifest_sha256": confirmation_hash,
        "confirmation_cases": len(confirmation),
        "confirmation_patients": len(confirmation_patients),
        "excluded_original_locked_patients": len(excluded_patients),
        "patient_overlap_with_original_locked_test": len(overlap),
        "role_seed": args.seed,
        "selection_patient_order": "numeric",
        "selection_randomizer": "numpy_default_rng",
        "selection_complete_patients_only": True,
        "selection_used_labels": False,
        "selection_used_predictions": False,
        "selection_used_risk_scores": False,
        "confirmation_label_statistics_calculated": False,
        "confirmation_label_statistics_displayed": False,
        "enhancement_developed_after_original_locked_test": True,
        "allowed_for_public_upload": False,
    }
    private_record_path = private_root / "confirmation_cohort_lock_private.json"
    atomic_json(private_record, private_record_path)
    public_record = {
        key: value
        for key, value in private_record.items()
        if key not in {"allowed_for_public_upload"}
    }
    public_record.update(
        {
            "artifact": "Objective 2 independent confirmation protocol lock",
            "patient_identifiers_included": False,
            "image_identifiers_included": False,
            "medical_images_included": False,
            "private_manifest_included": False,
            "status": "locked before confirmation-label evaluation",
        }
    )
    public_record_path = public_root / "confirmation_protocol_lock_public.json"
    atomic_json(public_record, public_record_path)
    public_checksum = public_record_path.with_suffix(".json.sha256")
    public_checksum.write_text(
        f"{sha256_file(public_record_path)}  {public_record_path.name}\n",
        encoding="utf-8",
    )

    if args.private_hf_repo:
        token = os.environ.get("HF_TOKEN", "").strip()
        if not token:
            raise RuntimeError("HF_TOKEN is required for private cohort recovery")
        from huggingface_hub import CommitOperationAdd, HfApi

        api = HfApi(token=token)
        info = api.model_info(args.private_hf_repo, token=token)
        if not bool(info.private):
            raise RuntimeError("Confirmation recovery repository must be private")
        remote_root = args.private_hf_path.strip("/")
        api.create_commit(
            repo_id=args.private_hf_repo,
            repo_type="model",
            token=token,
            operations=[
                CommitOperationAdd(
                    path_in_repo=f"{remote_root}/{path.name}",
                    path_or_fileobj=str(path),
                )
                for path in (
                    confirmation_manifest,
                    confirmation_checksum,
                    private_record_path,
                    public_record_path,
                    public_checksum,
                )
            ],
            commit_message="protocol: lock independent Objective 2 confirmation cohort",
        )

    print("--- OBJECTIVE 2 INDEPENDENT CONFIRMATION COHORT ---")
    print("Confirmation cases:", len(confirmation))
    print("Confirmation patients:", len(confirmation_patients))
    print("Original locked-test patients excluded:", len(excluded_patients))
    print("Patient overlap with original locked test:", len(overlap))
    print("Selection seed:", args.seed)
    print("Confirmation manifest SHA-256:", confirmation_hash)
    print("Public protocol SHA-256:", sha256_file(public_record_path))
    print("Labels used during selection:", False)
    print("Predictions used during selection:", False)
    print("Risk scores used during selection:", False)
    print("Confirmation label statistics calculated:", False)
    print("Confirmation label statistics displayed:", False)
    print("Private recovery enabled:", bool(args.private_hf_repo))
    print("Allowed for public upload (private manifest):", False)
    print("OBJECTIVE 2 INDEPENDENT CONFIRMATION COHORT LOCK SUCCESSFUL")


if __name__ == "__main__":
    main()
