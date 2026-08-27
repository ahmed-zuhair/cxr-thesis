#!/usr/bin/env python3
"""Lock the label-blind, previously unused Objective 3 final cohort."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

import pandas as pd
import yaml

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
    parser.add_argument(
        "--excluded-manifest",
        type=Path,
        action="append",
        required=True,
        help="Repeat for every earlier evaluation cohort.",
    )
    parser.add_argument("--validation-summary", type=Path, required=True)
    parser.add_argument("--protocol-amendment", type=Path, required=True)
    parser.add_argument("--frozen-config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--expected-manifest-sha256", required=True)
    parser.add_argument(
        "--expected-excluded-sha256",
        action="append",
        required=True,
    )
    parser.add_argument("--expected-validation-summary-sha256", required=True)
    parser.add_argument("--expected-protocol-sha256", required=True)
    parser.add_argument("--expected-config-sha256", required=True)
    parser.add_argument("--seed", type=int, default=4042)
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
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    os.replace(temporary, path)


def validate_advance(summary: dict[str, object]) -> None:
    checks = {
        "artifact": summary.get("artifact")
        == "Objective 3 v1.1 enhanced paired validation result",
        "architecture": summary.get("architecture_version")
        == "v1_1_reupload_gated",
        "seeds": summary.get("seeds") == [42, 43, 44],
        "advance": summary.get("advance_to_single_final_evaluation") is True,
        "mean_gain": float(
            summary.get("mean_quantum_minus_classical_validation_macro_auroc", 0.0)
        )
        > 0.0,
        "seed_wins": int(summary.get("quantum_seed_wins", 0)) >= 2,
        "test_manifest": summary.get("test_manifest_opened") is False,
        "test_labels": summary.get("test_labels_accessed") is False,
        "test": summary.get("test_evaluated") is False,
    }
    failed = [name for name, passed in checks.items() if not passed]
    if failed:
        raise RuntimeError(f"Objective 3 validation advance checks failed: {failed}")


def main() -> None:
    args = parse_args()
    if args.output_dir.exists():
        raise FileExistsError(f"Final-cohort output already exists: {args.output_dir}")
    if len(args.excluded_manifest) != len(args.expected_excluded_sha256):
        raise ValueError("Every excluded manifest needs one expected SHA-256")
    required = [
        args.manifest,
        args.validation_summary,
        args.protocol_amendment,
        args.frozen_config,
        *args.excluded_manifest,
    ]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError("Required files are missing:\n" + "\n".join(missing))
    if bool(args.private_hf_repo) != bool(args.private_hf_path):
        raise ValueError("Private HF repository and path must be supplied together")
    if sha256_file(args.manifest) != args.expected_manifest_sha256:
        raise RuntimeError("Full NIH manifest SHA-256 does not match")
    if sha256_file(args.validation_summary) != args.expected_validation_summary_sha256:
        raise RuntimeError("Objective 3 validation summary SHA-256 does not match")
    if sha256_file(args.protocol_amendment) != args.expected_protocol_sha256:
        raise RuntimeError("Objective 3 protocol amendment SHA-256 does not match")
    if sha256_file(args.frozen_config) != args.expected_config_sha256:
        raise RuntimeError("Objective 3 frozen configuration SHA-256 does not match")
    excluded_hashes = []
    for path, expected in zip(
        args.excluded_manifest, args.expected_excluded_sha256, strict=True
    ):
        actual = sha256_file(path)
        if actual != expected:
            raise RuntimeError(f"Excluded manifest SHA-256 does not match: {path}")
        excluded_hashes.append(actual)

    validation = json.loads(args.validation_summary.read_text(encoding="utf-8"))
    validate_advance(validation)
    protocol = json.loads(args.protocol_amendment.read_text(encoding="utf-8"))
    frozen_config = yaml.safe_load(args.frozen_config.read_text(encoding="utf-8"))
    final = frozen_config.get("final_evaluation", {})
    checks = {
        "architecture": protocol.get("frozen_enhancement", {}).get("architecture")
        == "v1_1_reupload_gated",
        "target": final.get("target_images") == args.target_images,
        "seed": final.get("selection_seed") == args.seed,
        "label_blind": final.get("selection_uses_labels") is False,
        "one_evaluation": final.get("evaluations") == 1,
        "test_blind": protocol.get("test_evaluated") is False,
    }
    failed = [name for name, passed in checks.items() if not passed]
    if failed:
        raise RuntimeError(f"Objective 3 protocol checks failed: {failed}")

    # Only identity/split columns are read during irreversible patient selection.
    identity = pd.read_csv(
        args.manifest,
        usecols=list(IDENTITY_COLUMNS),
        dtype={"patient_id": str, "split": str},
    )
    excluded_patients: set[str] = set()
    excluded_counts: list[dict[str, object]] = []
    for path, digest in zip(args.excluded_manifest, excluded_hashes, strict=True):
        earlier = pd.read_csv(
            path,
            usecols=list(IDENTITY_COLUMNS),
            dtype={"patient_id": str, "split": str},
        )
        if set(earlier["split"].astype(str).str.lower()) != {"test"}:
            raise RuntimeError(f"Excluded cohort contains a non-test split: {path}")
        patients = set(earlier["patient_id"].astype(str))
        excluded_patients.update(patients)
        excluded_counts.append(
            {
                "manifest_sha256": digest,
                "cases": len(earlier),
                "patients": len(patients),
            }
        )
    selected = select_disjoint_confirmation_patients(
        identity,
        excluded_patient_ids=excluded_patients,
        split="test",
        seed=args.seed,
        target_images=args.target_images,
    )

    # Full rows (including labels) are serialized only after identities are frozen.
    # No label statistic is calculated or displayed here.
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
    cohort_hash = hashlib.sha256(payload).hexdigest()
    private_root = args.output_dir / "private"
    public_root = args.output_dir / "public"
    private_root.mkdir(parents=True)
    public_root.mkdir(parents=True)
    cohort_path = private_root / "objective3_final_cohort_private.csv"
    cohort_path.write_bytes(payload)
    cohort_checksum = cohort_path.with_suffix(".sha256")
    cohort_checksum.write_text(
        f"{cohort_hash}  {cohort_path.name}\n", encoding="utf-8"
    )
    locked_identity = pd.read_csv(
        cohort_path,
        usecols=list(IDENTITY_COLUMNS),
        dtype={"patient_id": str, "split": str},
    )
    final_patients = set(locked_identity["patient_id"].astype(str))
    overlap = final_patients & excluded_patients
    if len(locked_identity) != args.target_images or overlap:
        raise RuntimeError("Final Objective 3 cohort validation failed")

    private_record = {
        "artifact": "Objective 3 private independent final cohort lock",
        "objective": 3,
        "source_manifest_sha256": args.expected_manifest_sha256,
        "excluded_evaluation_cohorts": excluded_counts,
        "excluded_unique_patients": len(excluded_patients),
        "final_cohort_manifest_sha256": cohort_hash,
        "final_cohort_cases": len(locked_identity),
        "final_cohort_patients": len(final_patients),
        "patient_overlap_with_prior_evaluation_cohorts": len(overlap),
        "selection_seed": args.seed,
        "selection_patient_order": "numeric",
        "selection_randomizer": "numpy_default_rng",
        "selection_complete_patients_only": True,
        "selection_used_labels": False,
        "selection_used_predictions": False,
        "selection_used_risk_scores": False,
        "label_statistics_calculated": False,
        "label_statistics_displayed": False,
        "validation_summary_sha256": args.expected_validation_summary_sha256,
        "protocol_amendment_sha256": args.expected_protocol_sha256,
        "frozen_configuration_sha256": args.expected_config_sha256,
        "advance_rule_verified": True,
        "final_evaluation_count_allowed": 1,
        "final_evaluated": False,
        "allowed_for_public_upload": False,
    }
    private_record_path = private_root / "objective3_final_cohort_lock_private.json"
    atomic_json(private_record, private_record_path)
    public_record = {
        key: value
        for key, value in private_record.items()
        if key != "allowed_for_public_upload"
    }
    public_record.update(
        {
            "artifact": "Objective 3 independent final evaluation protocol lock",
            "status": "locked before final-cohort label evaluation",
            "patient_identifiers_included": False,
            "image_identifiers_included": False,
            "medical_images_included": False,
            "private_manifest_included": False,
            "case_level_predictions_included": False,
        }
    )
    public_record_path = public_root / "objective3_final_protocol_lock_public.json"
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
            raise RuntimeError("Objective 3 recovery repository must remain private")
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
                    cohort_path,
                    cohort_checksum,
                    private_record_path,
                    public_record_path,
                    public_checksum,
                )
            ],
            commit_message="protocol: lock Objective 3 independent final cohort",
        )

    print("--- OBJECTIVE 3 INDEPENDENT FINAL COHORT ---")
    print("Final cases:", len(locked_identity))
    print("Final patients:", len(final_patients))
    print("Prior evaluation patients excluded:", len(excluded_patients))
    print("Patient overlap:", len(overlap))
    print("Selection seed:", args.seed)
    print("Final cohort SHA-256:", cohort_hash)
    print("Public protocol SHA-256:", sha256_file(public_record_path))
    print("Validation advance rule verified:", True)
    print("Labels used during selection:", False)
    print("Label statistics calculated:", False)
    print("Final evaluation performed:", False)
    print("Private recovery enabled:", bool(args.private_hf_repo))
    print("Private manifest allowed for public upload:", False)
    print("OBJECTIVE 3 INDEPENDENT FINAL COHORT LOCK SUCCESSFUL")


if __name__ == "__main__":
    main()
