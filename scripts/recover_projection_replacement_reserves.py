"""Recover private replacement reserves after ephemeral Kaggle storage loss."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

import pandas as pd
from PIL import Image

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = REPOSITORY_ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from cxr_thesis.objective1.cohort_selection import (
    match_cohort_fingerprints_to_manifest,
    select_blind_projection_recovery_reserves,
)


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Match a private cohort by image hash and build blind reserves"
    )
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--fingerprints", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--bundle", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--reserves-per-slot", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=10)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--expected-checkpoint-sha256", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest_path = Path(args.manifest).resolve()
    fingerprint_path = Path(args.fingerprints).resolve()
    checkpoint_path = Path(args.checkpoint).resolve()
    config_path = Path(args.config).resolve()
    output = Path(args.output_dir).resolve()
    bundle_path = Path(args.bundle).resolve()
    if output.exists():
        raise FileExistsError(f"Recovery output already exists: {output}")
    if bundle_path.exists() or bundle_path.with_suffix(".sha256").exists():
        raise FileExistsError(f"Recovery bundle already exists: {bundle_path}")
    for required in (
        manifest_path,
        fingerprint_path,
        checkpoint_path,
        config_path,
    ):
        if not required.is_file():
            raise FileNotFoundError(required)
    checkpoint_hash = sha256_file(checkpoint_path)
    if checkpoint_hash != args.expected_checkpoint_sha256.lower():
        raise RuntimeError(
            f"Checkpoint SHA-256 mismatch: expected {args.expected_checkpoint_sha256}, "
            f"received {checkpoint_hash}"
        )

    output.mkdir(parents=True)
    manifest = pd.read_csv(
        manifest_path,
        dtype={"patient_id": str, "study_id": str, "image_id": str},
    )
    fingerprints = pd.read_csv(fingerprint_path, keep_default_na=False)
    fingerprint_role_counts = {
        str(key): int(value)
        for key, value in fingerprints["cohort_role"].value_counts().items()
    }
    expected_role_counts = {
        "adaptation_train": 120,
        "target_validation": 40,
        "locked_target_test": 40,
    }
    if fingerprint_role_counts != expected_role_counts:
        raise ValueError(
            "Fingerprint role counts differ from the locked cohort: "
            f"{fingerprint_role_counts}"
        )
    rejected_fingerprints = fingerprints[
        fingerprints["projection_decision"] != "eligible_frontal"
    ]
    rejected_counts = {
        str(key): int(value)
        for key, value in rejected_fingerprints["cohort_role"].value_counts().items()
    }
    if rejected_counts != {"adaptation_train": 1, "target_validation": 1}:
        raise ValueError(f"Fingerprint replacement counts are unexpected: {rejected_counts}")
    recovered = match_cohort_fingerprints_to_manifest(manifest, fingerprints)
    recovered_path = output / "recovered_cohort_identity_private.csv"
    recovered.to_csv(recovered_path, index=False)
    reserves = select_blind_projection_recovery_reserves(
        manifest,
        recovered,
        seed=args.seed,
        reserves_per_slot=args.reserves_per_slot,
    )
    reserve_mapping_path = output / "replacement_reserve_mapping_private.csv"
    reserves.to_csv(reserve_mapping_path, index=False)

    candidate_manifest_path = output / "replacement_reserve_manifest_private.csv"
    reserves.to_csv(candidate_manifest_path, index=False)
    mask_dir = output / "generated_masks"
    mask_manifest_path = output / "replacement_reserve_manifest_with_masks_private.csv"
    audit_path = output / "replacement_reserve_mask_audit_private.csv"
    generation_summary_path = output / "replacement_reserve_generation_private.json"
    generation_command = [
        sys.executable,
        str(REPOSITORY_ROOT / "scripts" / "generate_roi_masks.py"),
        "--manifest",
        str(candidate_manifest_path),
        "--checkpoint",
        str(checkpoint_path),
        "--data-root",
        "/",
        "--config",
        str(config_path),
        "--mask-dir",
        str(mask_dir),
        "--output-manifest",
        str(mask_manifest_path),
        "--audit-csv",
        str(audit_path),
        "--summary-json",
        str(generation_summary_path),
        "--batch-size",
        str(args.batch_size),
        "--save-every",
        str(args.reserves_per_slot),
        "--device",
        args.device,
        "--min-component-fraction",
        "0.001",
        "--min-component-pixels",
        "0",
        "--uncertainty-margin",
        "0.10",
        "--expected-checkpoint-sha256",
        args.expected_checkpoint_sha256,
    ]
    subprocess.run(generation_command, check=True)
    generated = pd.read_csv(
        mask_manifest_path,
        dtype={"patient_id": str, "study_id": str, "image_id": str},
    )
    if len(generated) != len(reserves):
        raise RuntimeError("Generated reserve manifest row count changed")
    if set(generated["mask_generation_status"]) != {"complete"}:
        raise RuntimeError("One or more replacement reserve masks failed")
    if generated["mask_path"].fillna("").eq("").any():
        raise RuntimeError("One or more replacement reserve masks are missing")
    generated_lookup = generated.set_index("image_id")

    bundle_root = output / "private_replacement_review_workspace"
    bundle_root.mkdir()
    private_rows: list[dict[str, object]] = []
    for role, role_reserves in reserves.groupby("cohort_role", sort=True):
        role_root = bundle_root / str(role)
        image_dir = role_root / "images"
        preannotation_dir = role_root / "preannotations"
        annotation_dir = role_root / "annotations"
        image_dir.mkdir(parents=True)
        preannotation_dir.mkdir()
        annotation_dir.mkdir()
        worklist_rows: list[dict[str, object]] = []
        for _, row in role_reserves.sort_values(
            ["replacement_slot", "reserve_rank"], kind="stable"
        ).iterrows():
            replacement_code = str(row["replacement_code"])
            image_source = Path(str(row["image_path"]))
            mask_source = Path(str(generated_lookup.loc[str(row["image_id"]), "mask_path"]))
            image_target = image_dir / f"{replacement_code}.png"
            mask_target = preannotation_dir / f"{replacement_code}.png"
            shutil.copy2(image_source, image_target)
            shutil.copy2(mask_source, mask_target)
            with Image.open(image_target) as image_handle, Image.open(mask_target) as mask_handle:
                if image_handle.size != mask_handle.size:
                    raise RuntimeError(
                        f"Replacement reserve image/mask mismatch: {replacement_code}"
                    )
            worklist_rows.append(
                {
                    "candidate_code": replacement_code,
                    "cohort_role": str(role),
                    "view": str(row["view_group"]),
                    "sex": str(row["sex_group"]),
                    "finding_group": str(row["finding_group"]),
                    "selection_basis": str(row["original_selection_basis"]),
                    "replacement_selection_basis": str(
                        row["replacement_selection_basis"]
                    ),
                    "replacement_slot": str(row["replacement_slot"]),
                    "reserve_rank": int(row["reserve_rank"]),
                    "image_filename": f"images/{replacement_code}.png",
                    "preannotation_filename": f"preannotations/{replacement_code}.png",
                    "required_output_mask": f"annotations/{replacement_code}.png",
                }
            )
            private_rows.append(
                {
                    "replacement_code": replacement_code,
                    "candidate_code": str(row["replaces_candidate_code"]),
                    "cohort_role": str(role),
                    "patient_id": str(row["patient_id"]),
                    "image_id": str(row["image_id"]),
                    "replacement_slot": str(row["replacement_slot"]),
                    "reserve_rank": int(row["reserve_rank"]),
                }
            )
        pd.DataFrame(worklist_rows).to_csv(
            role_root / "annotation_worklist.csv", index=False
        )

    private_bundle_mapping = bundle_root / "replacement_identity_private.csv"
    pd.DataFrame(private_rows).to_csv(private_bundle_mapping, index=False)
    role_counts = {
        str(key): int(value)
        for key, value in reserves["cohort_role"].value_counts().items()
    }
    summary = {
        "artifact": "Private Objective 1 projection replacement reserves",
        "recovery_method": "exact SHA-256 image matching plus prediction-blind same-stratum hashing",
        "selection_seed": int(args.seed),
        "matched_original_cohort_cases": int(len(recovered)),
        "matched_original_cohort_patients": int(recovered["patient_id"].nunique()),
        "replacement_slots": int(reserves["replacement_slot"].nunique()),
        "reserves_per_slot": int(args.reserves_per_slot),
        "total_reserves": int(len(reserves)),
        "unique_reserve_patients": int(reserves["patient_id"].nunique()),
        "role_counts": role_counts,
        "patient_overlap_with_original_cohort": int(
            len(
                set(reserves["patient_id"].astype(str)).intersection(
                    recovered["patient_id"].astype(str)
                )
            )
        ),
        "official_nih_test_used": bool((reserves["split"] == "test").any()),
        "locked_target_test_modified": False,
        "replacement_selection_uses_predictions": False,
        "preannotations_generated_after_reserve_selection": True,
        "checkpoint_sha256": checkpoint_hash,
        "nih_manifest_sha256": sha256_file(manifest_path),
        "fingerprint_request_sha256": sha256_file(fingerprint_path),
        "private_identity_publication_allowed": False,
        "source_images_or_masks_publication_allowed": False,
    }
    summary_path = bundle_root / "replacement_recovery_summary_private.json"
    summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    bundle_path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(bundle_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(bundle_root.rglob("*")):
            if path.is_file():
                archive.write(path, path.relative_to(bundle_root).as_posix())
    bundle_hash = sha256_file(bundle_path)
    checksum_path = bundle_path.with_suffix(".sha256")
    checksum_path.write_text(
        f"{bundle_hash}  {bundle_path.name}\n", encoding="utf-8"
    )

    print("--- PRIVATE PROJECTION RECOVERY SUMMARY ---")
    print(json.dumps(summary, indent=2, sort_keys=True))
    print(f"Private bundle: {bundle_path}")
    print(f"Private bundle size MB: {bundle_path.stat().st_size / (1024 * 1024):.2f}")
    print(f"Private bundle SHA-256: {bundle_hash}")
    print("Candidate/patient/image identifiers displayed: False")
    print("Allowed for public upload: False")
    print("PRIVATE PROJECTION REPLACEMENT RECOVERY COMPLETE")


if __name__ == "__main__":
    main()
