"""Lock a private, balanced NIH ROI annotation cohort."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import pandas as pd

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = REPOSITORY_ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from cxr_thesis.objective1.cohort_selection import select_roi_annotation_cohort


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Select a prediction-blind 40-case test and active-QC 160-case development cohort"
    )
    parser.add_argument("--candidate-manifest", required=True)
    parser.add_argument("--private-mapping", required=True)
    parser.add_argument("--ranked-audit", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--expected-candidate-manifest-sha256")
    parser.add_argument("--expected-checkpoint-sha256")
    parser.add_argument("--repository-commit")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    candidate_path = Path(args.candidate_manifest).resolve()
    mapping_path = Path(args.private_mapping).resolve()
    ranked_path = Path(args.ranked_audit).resolve()
    output = Path(args.output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)

    candidate_hash = sha256_file(candidate_path)
    if (
        args.expected_candidate_manifest_sha256
        and candidate_hash.lower() != args.expected_candidate_manifest_sha256.lower()
    ):
        raise RuntimeError(
            "Candidate manifest SHA-256 mismatch: "
            f"expected {args.expected_candidate_manifest_sha256}, received {candidate_hash}"
        )

    candidate = pd.read_csv(candidate_path)
    mapping = pd.read_csv(mapping_path)
    ranked = pd.read_csv(ranked_path)
    if len(candidate) != len(mapping) or set(candidate["image_id"].astype(str)) != set(
        mapping["image_id"].astype(str)
    ):
        raise RuntimeError("Candidate manifest and private mapping describe different images")

    checkpoint_values = sorted(
        set(ranked.get("checkpoint_sha256", pd.Series(dtype=str)).dropna().astype(str))
    )
    if args.expected_checkpoint_sha256 and checkpoint_values != [
        args.expected_checkpoint_sha256
    ]:
        raise RuntimeError(
            "Ranked-audit checkpoint mismatch: "
            f"expected {args.expected_checkpoint_sha256}, received {checkpoint_values}"
        )

    roles = select_roi_annotation_cohort(mapping, ranked, seed=args.seed)
    filenames = {
        "adaptation_train": "adaptation_train_120_private.csv",
        "target_validation": "target_validation_40_private.csv",
        "locked_target_test": "locked_target_test_40_private.csv",
        "master": "annotation_cohort_200_private.csv",
    }
    private_hashes: dict[str, str] = {}
    for role, filename in filenames.items():
        target = output / filename
        roles[role].to_csv(target, index=False)
        digest = sha256_file(target)
        private_hashes[filename] = digest
        target.with_suffix(".sha256").write_text(
            f"{digest}  {filename}\n", encoding="utf-8"
        )

    master = roles["master"]
    public_strata = (
        master.groupby(
            ["cohort_role", "cohort_stratum", "selection_basis"], dropna=False
        )
        .size()
        .rename("cases")
        .reset_index()
        .sort_values(["cohort_role", "cohort_stratum", "selection_basis"])
    )
    public_strata_path = output / "annotation_cohort_public_strata.csv"
    public_strata.to_csv(public_strata_path, index=False)

    role_counts = {
        str(key): int(value)
        for key, value in master["cohort_role"].value_counts().items()
    }
    summary = {
        "objective": "Objective 1 NIH ROI target-domain manual annotation cohort",
        "selection_seed": int(args.seed),
        "total_cases": int(len(master)),
        "unique_patients": int(master["patient_id"].nunique()),
        "role_counts": role_counts,
        "cases_per_view_sex_finding_stratum": {
            "adaptation_train": 15,
            "target_validation": 5,
            "locked_target_test": 5,
        },
        "adaptation_high_risk_per_stratum": 7,
        "validation_high_risk_per_stratum": 2,
        "locked_test_selection": "deterministic SHA-256 ordering from identifiers only",
        "locked_test_selected_before_ranked_audit_join": True,
        "locked_test_selection_uses_prediction_metrics": False,
        "locked_test_contains_prediction_columns": False,
        "patient_overlap_between_roles": 0,
        "image_overlap_between_roles": 0,
        "official_nih_test_used": False,
        "candidate_manifest_sha256": candidate_hash,
        "private_mapping_sha256": sha256_file(mapping_path),
        "private_ranked_audit_sha256": sha256_file(ranked_path),
        "checkpoint_sha256": checkpoint_values[0] if checkpoint_values else None,
        "repository_commit": args.repository_commit,
        "private_cohort_file_sha256": private_hashes,
        "privacy": {
            "patient_identifiers_intended_for_publication": False,
            "image_identifiers_intended_for_publication": False,
            "source_images_intended_for_publication": False,
            "predicted_masks_intended_for_publication": False,
        },
        "annotation_policy": {
            "adaptation_train": "predictions may be used only as pre-annotations requiring manual correction",
            "target_validation": "predictions may be used only as pre-annotations requiring manual correction",
            "locked_target_test": "annotate from scratch without displaying model predictions",
        },
    }
    summary_path = output / "annotation_cohort_public_summary.json"
    summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    print("--- LOCKED ANNOTATION COHORT ---")
    print(json.dumps(summary, indent=2))
    print("\n--- PUBLIC STRATA ---")
    print(public_strata.to_string(index=False))
    print("\nPRIVATE 200-CASE ANNOTATION COHORT LOCKED SUCCESSFULLY")


if __name__ == "__main__":
    main()
